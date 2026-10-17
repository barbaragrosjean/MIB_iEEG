import os
import pickle
import numpy as np
import pandas as pd
import json 
import mne
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

        # save as before 
                    

def subj_ripples_trials(subj, ):
    # OUTPUT a figure of a 2d histogram trial vs electrodes, of the ripples count over time for each trials and electrodes
    # I want that in the annotation ylabel we see the area in which the electrodes is implented. 
    # For each subject.
    # save at the end in OUT_PATH + 'ripples/figures' and the name of the file should be subj + _trials

def subj_ripples_in_time(subj, ):
    # OUTPUT a figure of a 2d histogram time vs electrodes, of the ripples count over trials I want that in the annotation ylabel 
    # we see the area in which the electrodes is implented. For each subject. you should define time bins to count over the trials.
    # save at the end in OUT_PATH + 'ripples/figures' and the name of the file should be subj + _time


def group_ripples_in_time(subj_included):
    # Same than above but group all participant, you should have for each subject the average count over trials:
    # (condition, electrodes, time_bins ). you should group the participant and divid by regions of interest. I want one plot for each RIOs
    # then an figure for the average taking the average count per region of interest so a 2d plot of ROIs x time at group level. 
    # save at the end in OUT_PATH + 'ripples/figures' and the name of the file should be group 


if __name__ == "__main__":
    subj_included = [file.replace('sub-', '') for file in os.listdir(PROJECT_PATH + '/data/BIDS') if file[:4] == 'sub-']
    subj_included = ExcludSubj(subj_included=subj_included, data_path=OUT_PATH + '/Data_longWOBS')

    failled =['COG023', 'LL10', 'CP41', 'LL23', 'CP40', 'COG022', 'LL30', 'LL15']

    for s in failled : 
        if s in subj_included :
            subj_included.remove(s)

    for subj in subj_included[0]:
        load_and_filter(subj)
        identify_ripples(subj)
        test_adjacent(subj)
