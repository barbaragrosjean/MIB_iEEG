'''
submit_to_cluster -n 2 -q short.q "python -u LB11_ripples.py" --project MINDLAB2021_MEG-TempSeqAges
'''

import os
import pickle
import numpy as np
import pandas as pd
import json 
import mne
import matplotlib.pyplot as plt
from mne_bids import BIDSPath, read_raw_bids

from src.setting import ExcludSubj, GetInfo
from src.config import OUT_PATH, PROJECT_PATH, TASK
from src.preproc import get_bads, set_reference, LB_event_fun

FREQ_BAND = [80, 120]
FS=600

def load_and_filter(subj, out_path=OUT_PATH + '/ripples', save=False, freq_band=FREQ_BAND, fs=FS):

    from scipy.signal import butter, sosfiltfilt, hilbert

    if not os.path.exists(out_path) :
        os.makedirs(out_path)

    # Load raw iEEG data and bad channels
    bids_root = os.path.join(PROJECT_PATH, 'data/BIDS/')
    bids_path = BIDSPath(subject=subj, task=TASK, root=bids_root)
    elec_path = os.path.join(PROJECT_PATH, 'misc/electrodes/', subj + '_electrodes.csv')
    bads_path = os.path.join(PROJECT_PATH, 'misc/bad_channels/', subj + '_bad_channels.csv')
    bad_channels, white_channels, _ = get_bads(subj, bads_path, elec_path)

    raw = read_raw_bids(bids_path)
    raw.pick_types(seeg=True, ecog=True)

    # Keep the continuous task recording with margins around the events
    onsets = []
    for ano in raw.annotations :
        if 'bach' in ano['description'] :
            onsets.append(ano['onset'] - raw.first_time)
    if len(onsets) == 0 :
        raise ValueError(f'No bach events found for {subj}')
    tmin_ = max(0, min(onsets) - 2)
    tmax_ = min(raw.times[-1], max(onsets) + 50)
    raw.crop(tmin=tmin_, tmax=tmax_)
    raw.load_data()

    # Bipolar reference and remove bad channels
    raw = set_reference(raw, bads=bad_channels, white=white_channels)
    raw = raw.drop_channels(raw.info['bads'], on_missing='warn')

    # Resample and remove slow drift and line-frequency harmonics before epoching
    raw.resample(sfreq=fs)
    raw.filter(l_freq=0.1, h_freq=None)
    nfreqs = np.arange(1, np.ceil(raw.info['lowpass']/raw.info['line_freq'])) * raw.info['line_freq']
    if nfreqs.size > 0 :
        raw.notch_filter(freqs=nfreqs)

    events, event_id = mne.events_from_annotations(raw)
    events, event_id = LB_event_fun(events, event_id, sfreq=raw.info['sfreq'])

    # Filter the continuous signal (order 2 prototype, order 4 bandpass, forward-backward)
    sos = butter(2, freq_band, btype='bandpass', output='sos', fs=raw.info['sfreq'])
    raw.apply_function(lambda x : sosfiltfilt(sos, x), picks='all')

    # Extract the continuous amplitude envelope, keeping voltage units
    raw.apply_function(lambda x : np.abs(hilbert(x)), picks='all')

    # Epoch the envelope only after filtering and the Hilbert transform
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=-1.5, tmax=5,
                       preload=True, baseline=None, on_missing='warn', reject=None)
    env = epochs.get_data()
    del raw

    # SAVE OUTPUT: matrix (trials, channels, time)
    if save:
        if not os.path.exists(out_path) : 
            os.makedirs(out_path)

        with open(out_path + f'/{subj}_env{str(freq_band[0])}_{str(freq_band[1])}.p', "wb") as f:
            pickle.dump(env, f)
        info = {'chnames': epochs.ch_names, 'time_epoch': epochs.times.tolist(),
                'n_trials': len(epochs), 'sfreq': float(epochs.info['sfreq'])}
        with open(out_path + f'/{subj}_info.json', 'w') as f:
            json.dump(info, f)
        ch_coord_df = pd.DataFrame({ch['ch_name']: ch['loc'][:3] * 1000 for ch in epochs.info['chs']}).T
        ch_coord_df.columns = ['x', 'y', 'z']
        ch_coord_df['subj'] = subj
        ch_coord_df.to_csv(out_path + f'/{subj}_coords.csv')
        del env
    else :    
        return env


def identify_ripples(subj,env=None, env_path=OUT_PATH + '/ripples', out_path=OUT_PATH + '/ripples', freq_band=FREQ_BAND, fs=FS, save=False) :
    # Load envelopes with shape (trials, channels, time)
    if env is None :
        file = env_path + f'/{subj}_env{str(freq_band[0])}_{str(freq_band[1])}.p'
        with open(file, "rb") as f:
            env = pickle.load(f)

    env = np.asarray(env, dtype=float)
    
    # One threshold per channel, estimated across all supplied trials and times
    threshold = np.mean(env, axis=(0, 2)) + 3 * np.std(env, axis=(0, 2))
    candidates = env > threshold[None, :, None]
    detection = np.zeros(env.shape, dtype=bool)
    point15ms = int(np.floor(fs * 0.015))
    point25ms = int(np.ceil(fs * 0.025))

    # Parallel lists: each entry describes one event; offsets are exclusive
    meta_data = {}
    meta_data['subj'] = subj
    meta_data['sfreq'] = float(fs)
    meta_data['threshold'] = threshold.tolist()
    meta_data['threshold_method'] = 'channel mean + 3 SD across supplied trials and times'
    meta_data['time_units'] = 'samples from epoch start; offset exclusive'
    meta_data['index_base'] = 0
    for key in ['trial', 'channel', 'onset', 'offset', 'durations', 'duration_seconds', 'peak_amplitude', 'touches_boundary'] :
        meta_data[key] = []

    for tr in range(env.shape[0]) :
        for ch in range(env.shape[1]) :
            # Find contiguous intervals above the threshold along the time axis
            padded = np.concatenate(([False], candidates[tr, ch], [False]))
            diff = np.diff(padded.astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]

            # Merge intervals separated by at most 15 ms of below-threshold samples
            merged = []
            for start, end in zip(starts, ends) :
                if len(merged) > 0 and start - merged[-1][1] <= point15ms :
                    merged[-1][1] = int(end)
                else :
                    merged.append([int(start), int(end)])

            # Keep merged intervals lasting at least 25 ms, including filled gaps
            for start, end in merged :
                duration = end - start
                if duration < point25ms :
                    continue
                detection[tr, ch, start:end] = True
                meta_data['trial'].append(tr)
                meta_data['channel'].append(ch)
                meta_data['onset'].append(start)
                meta_data['offset'].append(end)
                meta_data['durations'].append(duration)
                meta_data['duration_seconds'].append(float(duration / fs))
                meta_data['peak_amplitude'].append(float(np.max(env[tr, ch, start:end])))
                meta_data['touches_boundary'].append(start == 0 or end == env.shape[2])

    # Apply the final mask only after merging and duration rejection
    rip = np.where(detection, env, np.nan)

    if save:
        if not os.path.exists(out_path) : 
            os.makedirs(out_path)

        with open(out_path + f'/{subj}_env_rip{str(freq_band[0])}_{str(freq_band[1])}.p', "wb") as f:
            pickle.dump(rip, f)
        del rip

        with open(out_path + f'/{subj}_env_rip{str(freq_band[0])}_{str(freq_band[1])}.json', 'w') as f:
            json.dump(meta_data, f)

    else :    
        return rip, meta_data


def test_adjacent(subj, env_path=OUT_PATH + '/ripples', out_path=OUT_PATH + '/ripples', freq_band=FREQ_BAND, fs=FS, save=False) :

    # Get the channels region of belonging 
    _, _, _, _, regions = GetInfo([subj], project_path = PROJECT_PATH, data_path = OUT_PATH + '/Data', save=False) # TODO UPDATE THE DATAPATH
    
    HPC_ch_num = regions.count("HPC")
    HPC_ch_idx = np.where(np.array(regions) == 'HPC')

    # For patients who had more than one anterior hippocampal electrode
    if HPC_ch_num >=2 :
        # we examined whether the same ripple was picked up by the different electrodes (within 20 ms).
        file = env_path + f'/{subj}_env_rip{str(freq_band[0])}_{str(freq_band[1])}.p'
        with open(file, "rb") as f:
            env_rip = pickle.load(f)  # array channel x trials
        
        # select the HPC 
        hpc_rip = env_rip[HPC_ch_idx, :] # array channel x trials
        hpc_rip_bool = np.where(hpc_rip.isnan(), False)  # array channel x trials

        # Test of there are mutilpe electordes capturing the same within 20ms
        point20ms = fs*0.02 
        for tr in hpc_rip_bool.shape[1] :
            pass

        # save as before 
                    

def subj_ripples_trials(subj, env_path=OUT_PATH + '/ripples', out_path=OUT_PATH + '/ripples/figures', freq_band=FREQ_BAND):
    # Load events and the channel order used for the envelopes
    with open(env_path + f'/{subj}_env_rip{freq_band[0]}_{freq_band[1]}.json') as f:
        meta_data = json.load(f)
    with open(env_path + f'/{subj}_info.json') as f:
        info = json.load(f)
    _, _, elect_list, _, regions = GetInfo([subj], project_path=PROJECT_PATH, data_path=env_path, save=False)
    region_by_ch = dict(zip(elect_list, regions))
    labels = [ch + ' (' + region_by_ch[ch] + ')' for ch in info['chnames']]

    # Count events over the full epoch for each electrode and trial
    counts = np.zeros((len(labels), info['n_trials']), dtype=int)
    for tr, ch in zip(meta_data['trial'], meta_data['channel']) :
        counts[ch, tr] += 1

    fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.25)))
    im = ax.imshow(counts, aspect='auto', interpolation='nearest', cmap='viridis', vmin=0,
                   extent=[0.5, info['n_trials'] + 0.5, len(labels) - 0.5, -0.5])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel('Trial')
    ax.set_ylabel('Electrode (region)')
    ax.set_title(f'{subj}: ripple counts per trial')
    cbar = fig.colorbar(im, ax=ax, label='Ripple count')
    cbar.locator = plt.MaxNLocator(integer=True)
    cbar.update_ticks()
    fig.tight_layout()

    if not os.path.exists(out_path) :
        os.makedirs(out_path)
    fig.savefig(out_path + f'/{subj}_trials.png', dpi=300)
    plt.close(fig)
    return counts


def subj_ripples_in_time(subj, env_path=OUT_PATH + '/ripples', out_path=OUT_PATH + '/ripples/figures', freq_band=FREQ_BAND, bin_width=0.25):
    # Load events and electrode regions
    with open(env_path + f'/{subj}_env_rip{freq_band[0]}_{freq_band[1]}.json') as f:
        meta_data = json.load(f)
    with open(env_path + f'/{subj}_info.json') as f:
        info = json.load(f)
    _, _, elect_list, _, regions = GetInfo([subj], project_path=PROJECT_PATH, data_path=env_path, save=False)
    region_by_ch = dict(zip(elect_list, regions))
    labels = [ch + ' (' + region_by_ch[ch] + ')' for ch in info['chnames']]

    # Count each event once, using its onset relative to stimulus onset
    if not np.isfinite(bin_width) or bin_width <= 0 :
        raise ValueError('bin_width must be positive and expressed in seconds')
    times = np.array(info['time_epoch'])
    bins = np.arange(times[0], times[-1], bin_width)
    bins = np.append(bins, times[-1])
    onsets = times[0] + np.array(meta_data['onset']) / meta_data['sfreq']
    channels = np.array(meta_data['channel'])
    counts = np.zeros((len(labels), len(bins) - 1), dtype=int)
    for ch in range(len(labels)) :
        counts[ch], _ = np.histogram(onsets[channels == ch], bins=bins)

    fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.25)))
    im = ax.pcolormesh(bins, np.arange(len(labels) + 1) - 0.5, counts,
                       cmap='viridis', vmin=0, shading='flat')
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color='white', linestyle='--', linewidth=1)
    ax.set_xlabel('Time from stimulus onset (s)')
    ax.set_ylabel('Electrode (region)')
    ax.set_title(f'{subj}: ripple counts across trials ({bin_width:g} s bins)')
    cbar = fig.colorbar(im, ax=ax, label='Ripple count across trials')
    cbar.locator = plt.MaxNLocator(integer=True)
    cbar.update_ticks()
    fig.tight_layout()

    if not os.path.exists(out_path) :
        os.makedirs(out_path)
    fig.savefig(out_path + f'/{subj}_time.png', dpi=300)
    plt.close(fig)
    return counts, bins


def group_ripples_in_time(subj_included):
    pass
    # Same than above but group all participant, you should have for each subject the average count over trials:
    # (condition, electrodes, time_bins ). you should group the participant and divid by regions of interest. I want one plot for each RIOs
    # then an figure for the average taking the average count per region of interest so a 2d plot of ROIs x time at group level. 
    # save at the end in OUT_PATH + 'ripples/figures' and the name of the file should be group 


if __name__ == "__main__":
    subj_included = [file.replace('sub-', '') for file in os.listdir(PROJECT_PATH + '/data/BIDS') if file[:4] == 'sub-']
    #subj_included = ExcludSubj(subj_included=subj_included, data_path=OUT_PATH + '/Data_longWOBS')

    failled =['COG023', 'LL10', 'CP41', 'LL23', 'CP40', 'COG022', 'LL30', 'LL15']

    for s in failled : 
        if s in subj_included :
            subj_included.remove(s)

    print("Subject imcluded: ", len(subj_included))

    for subj in subj_included[:1]:
        print("Start processing:", subj)
        load_and_filter(subj, save=True)
        print("Ripples indentification")
        identify_ripples(subj, save=True)
        #test_adjacent(subj)
        subj_ripples_trials(subj)
        subj_ripples_in_time(subj)
