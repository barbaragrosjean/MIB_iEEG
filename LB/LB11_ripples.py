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

subj_included = [file.replace('sub-', '') for file in os.listdir(PROJECT_PATH + '/data/BIDS') if file[:4] == 'sub-']
subj_included = ExcludSubj(subj_included=subj_included, data_path=OUT_PATH + '/Data_longWOBS')

failled =['COG023', 'LL10', 'CP41', 'LL23', 'CP40', 'COG022', 'LL30', 'LL15']

for s in failled : 
    if s in subj_included :
        subj_included.remove(s)

FREQ_BAND = [80, 120]
FS=600

def load_and_filter(subj, out_path=OUT_PATH + '/ripples', save=False, freq_band=FREQ_BAND, fs=FS):

    from scipy.signal import butter, sosfiltfilt, hilbert

    info = {}
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

    ch_coord_df = pd.DataFrame({ch['ch_name']: ch['loc'][:3] for ch in raw.info['chs']}).T
    ch_coord_df = ch_coord_df.rename(columns={0: 'x', 1: 'y', 2: 'z'})
    for c in ['x', 'y', 'z'] :
        ch_coord_df[c] = ch_coord_df[c].apply(lambda x : x * 1000)
    ch_coord_df['subj'] = subj

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

    # Save channel coordinates and trial information as in preproc
    info['chnames'] = epochs.info['ch_names']
    info['time_epoch'] = list(epochs.times)
    info['event_count'] = [[list(event_id.keys())[list(event_id.values()).index(event)], str(count)] for event, count in zip(np.unique(epochs.events[:, 2], return_counts=True)[0], np.unique(epochs.events[:, 2], return_counts=True)[1])]
    info['event_id'] = [str(e) for e in epochs.events[:, 2]]
    with open(out_path + f'/{subj}_info.json', 'w') as f:
        json.dump(info, f)
    ch_coord_df.to_csv(out_path + f'/{subj}_coords.csv')

    # SAVE OUTPUT: matrix (trials, channels, time)
    if save:
        if not os.path.exists(out_path) : 
            os.makedirs(out_path)

        with open(out_path + f'/{subj}_env{str(freq_band[0])}_{str(freq_band[1])}.p', "wb") as f:
            pickle.dump(env, f)
        del env
    else :    
        return env


def identify_ripples(subj,env='none', env_path=OUT_PATH + '/ripples', out_path=OUT_PATH + '/ripples', freq_band=FREQ_BAND, fs=FS, save=False) : 
    # Take the envolopes
    if env == 'none' :
        file = env_path + f'/{subj}_env{str(freq_band[0])}_{str(freq_band[1])}.p'
        with open(file, "rb") as f:
            env = pickle.load(f)  
        
    # Ripples were identified as events with a maximum amplitude three s.d. above the mean, and with a duration of at least 25 ms
    detection = np.where(env > np.mean(env) + 3 * np.std(env), True, False)
    true_indices = [i for i, val in enumerate(detection) if val]

    # Ripples that were detected within 15 ms of each other were merged. --turn True
    point15ms = fs*0.015 
    for i in range(len(true_indices) - 1):
        start = true_indices[i]
        end = true_indices[i+1]
        if end - start <= point15ms:
            for j in range(start + 1, end):
                detection[j] = True

    rip = np.where(detection ==True, env, np.nan)

    # Test the 25ms duration --turn False if below
    point20ms = fs*0.025 
    onset_bool = np.where(detection & ~np.concatenate(([False], detection[:-1])), True, False)
    for i in range(len(true_indices) - 1):
        start = onset_bool[i]
        c=0
        while detection[true_indices[i] +c] == True :
            c+=1
        end = true_indices[i] + c # duration of the event i 
        if end - start <= point20ms:
            for j in range(start + 1, end):
                detection[j] = False

    # json fils with the subject, duration and onset for each trials and each channels
    meta_data = {}
    meta_data['subj']=subj
    meta_data['onset'] = np.where(detection & ~np.concatenate(([False], detection[:-1])))[0] # 2D array chxtrials
    padded = np.empty(len(detection) + 2, dtype=bool)
    padded[0], padded[-1] = False, False
    padded[1:-1] = detection
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    meta_data['durations'] = ends - starts # 2D array chxtrials

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


def ripples_pipline(subject_included) :
    for subj in subject_included:
        load_and_filter(subj)
        identify_ripples(subj)
        test_adjacent(subj)
