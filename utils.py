import os.path as op
import os
import pickle
import json

import mne
from mne_bids import BIDSPath, read_raw_bids

import numpy as np
import pandas as pd
import re
import random

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from nilearn import plotting

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tslearn.neighbors import KNeighborsTimeSeriesClassifier 
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.decomposition import NMF
from sklearn.decomposition import PCA

from PIL import Image

plt.style.use('seaborn-v0_8-dark')

# Relative paths
PROJECT_PATH = '../MINDLAB2021_MEG-TempSeqAges/scratch/learning_bach_iEEG'
OUT_PATH = 'outs'


scripts_path = op.join(PROJECT_PATH, 'scripts')
from sys import path; path.append(scripts_path)
    
#from src.preprocessing import get_bads, set_reference, normalize_epochs, smooth, LB_event_fun
#from src.TFR import MM_compute_TFR

################################### GLOBAL VARIABLES ###################################

EVENT_ID = {'old/correct': 1,
 'new/correct': 2,
 'old/incorrect': 101,
 'new/incorrect': 102,
 'old/null': 201,
 'new/null': 202}

FREQS = [0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,15,17,19,21,24,27,30,35,40,45,50,55,60,70,80,90,100,110,120,130,140,150,160,180]
BWIDTH = np.array([0.5,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,5,5,5,5,5,5,10,10,10,10,10,10,10,10,10,10,20])

# freq band to distinguish 
FREQ_BAND_DICT = {
    'Delta': [0.5, 1, 2, 3, 4],
    'Theta': [4, 5, 6, 7, 8],
    'Alpha': [8, 9, 10, 11, 12, 13],
    'Low_Beta': [13, 15, 17, 19],
    'High_Beta': [21, 24, 27, 30],
    'Low_Gamma': [30, 35, 40, 45, 50, 55],
    'High_Gamma': [55, 60, 70, 80, 90, 100]
}

FREQ_BAND = ['delta', 'theta','alpha', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma']

REGION = {'parietal': ['IPS','IP','SP','SPL','AG','SMG','TPJ'],
        'premotor': ['SFG','SFS','MFG','FEF','SMA'],
        'DLPFC': ['MFG','FEF','SFS','IFS'],
        'M1': ['preCG','M1','PreCG'],
        'S1': ['postCG','PostCG'],
        'INS': ['INS'],
        'VLPFC': ['IFG','FOP','IFS'],
        'MTL': ['HPC','EC','MEC','PRH','PHG','PHC','LEC'],
        'A1': ['A1'],
        'MTG': ['MTG'],
        'AMY': ['AMY'],
        'PCC': ['PCC'],
        'ACC': ['ACC','MCC'],
        'HPC': ['HPC'],
        'PHC': ['PHC','PHG'],
        'STG': ['STG'],
        'STS': ['STS'],
        'TP': ['TP'],
        'OFC': ['OFC'],
        'VS': ['LG','FUG','ITG','ITS'],
        'THAL': ['THAL']}

TASK = 'MusicMemory'

################################### FUNCTIONS PREPROC ###################################

def get_bads(SBJ, bads_path, elec_path, segments_path=None):

    # Load bads channels
    bads_df = pd.read_csv(bads_path)
    bad_channels = list(bads_df['name'])

    # Add bad segments
    bad_segments = {'onsets': [], 'duration': [], 'description': []}
    if segments_path:
        segs_df = pd.read_csv(segments_path)
        bad_segments['onsets'] = list(segs_df['onset'])
        bad_segments['duration'] = list(segs_df['duration'])
        bad_segments['description'] = ['bad']*len(segs_df['onset']) #[so for so in str(list(bads_info['bad_segment_description'].iloc[subix])[0]).split(';') if so != 'nan']

    #Exclude out of brain, lesion and fetch white matter
    elecs = pd.read_csv(elec_path)
    bad_channels += [elecs.loc[e]['electrode'] for e in range(len(elecs['electrode'])) if elecs.loc[e]['label'] in ['OOB','LSN']]
    white_channels = [elecs.loc[e]['electrode'] for e in range(len(elecs['electrode'])) if elecs.loc[e]['label'] in ['WM','VENT']]
    white_channels = [wch for wch in white_channels if wch not in bad_channels]
    
    return bad_channels, white_channels, bad_segments

def LB_event_fun(events, event_id, sfreq):
    wanted_labels = ['learning_bach_10','learning_bach_20'] #,'learning_bach_48','learning_bach_49','learning_bach_50']
    wanted_codes = np.array([event_id[cc] for cc in wanted_labels])
    events2 = events[[e in wanted_codes for e in events[:,2]],:]
    cond_mappings =  {'learning_bach_10': ['old',1], 
                      'learning_bach_20': ['new',2]}
    for cix,ce in enumerate(events2[:,2]):
        wix = np.where(wanted_codes == ce)[0][0]
        events2[cix,2] = cond_mappings[wanted_labels[wix]][1]
    
    ### Classify correct vs incorrect vs null
    # First get key presses
    wanted_keys = ['learning_bach_48','learning_bach_49','learning_bach_50']
    key_codes = np.array([event_id[c] for c in wanted_keys if c in event_id.keys()])
    key_events = events[[e in key_codes for e in events[:,2]],:]

    # Get correct, incorrect or null
    for eix2 in range(len(events2)):
        # Select first key press between 0.1 and 4 seconds after each onset
        rix = np.where([x and y for x,y in zip((np.array(key_events[:,0])/sfreq - np.array(events2[eix2,0])/sfreq) > 0.1,
                                               (np.array(key_events[:,0])/sfreq - np.array(events2[eix2,0])/sfreq) < 5)])[0]
        
        # Calssify key press
        if len(rix) > 0:
            rix = rix[0]        
            if (key_events[rix,2] == event_id['learning_bach_49']) and (events2[eix2,2] == 2):
                events2[eix2,2] = events2[eix2,2] + 100
            elif (key_events[rix,2] == event_id['learning_bach_50']) and (events2[eix2,2] == 1):
                events2[eix2,2] = events2[eix2,2] + 100
            elif ('learning_bach_48' in event_id.keys()) and (key_events[rix,2] == event_id['learning_bach_48']):
                events2[eix2,2] = events2[eix2,2] + 200
                    
    # Update event ids
    event_id2 = {}
    acc_map = {'correct': 0, 'incorrect': 100, 'null': 200}
    for am in acc_map:
        for ei in cond_mappings:
            cm = cond_mappings[ei]
            event_id2[cm[0] + '/' + am] = cm[1] + acc_map[am]
            
    return events2, event_id2

def smooth_data(data, tstart, tstep, twin, Fs, taxis=2):
    
    # get data shape
    old_dims = data.shape
    
    # Arrange dimensions in standard form
    new_dimord = np.array([taxis] + [d for d in range(len(old_dims)) if d != taxis])
    old_dimord = np.argsort(new_dimord)
    data = np.transpose(data,new_dimord)
    new_dims = data.shape
    
    # Calculate old and new time vectors
    tend = tstart + new_dims[0]/Fs
    ctime = np.arange(tstart, tend + 1/Fs, 1/Fs)
    ntime = np.arange(tstart + twin/2, tend-twin/2 + 1/Fs, tstep)
    
    # Initialize output data
    new_data = np.ones((tuple([len(ntime)]) + new_dims[1:])) * np.nan
    
    # Loop over timesteps and smooth
    for ntix, nt in enumerate(ntime):
        lims = np.array([nt - twin / 2, nt + twin / 2]) # Current interval to average
        cix = [np.argmin(np.abs(l - ctime)) for l in lims] # Limit indices
        new_data[ntix] = np.mean(data[cix[0]:(cix[1]+1)],0) # Average interval and store
    
    # Reorder dimesions and return
    new_data = np.transpose(new_data, old_dimord)
    return new_data, ntime

def smooth(indata, tstep, twin):
    
    # Copy data to avoid rewriting
    sdata = indata.copy()
    
    # Identify time axis
    taxis = np.where(np.array(sdata.data.shape) == len(sdata.times))[0][0]
    
    # Define starting time
    tstart = sdata.times[0]
    
    # Get sampling frequency
    Fs = sdata.info['sfreq']
    
    # Smooth data
    cdata, times = smooth_data(sdata.data, tstart=tstart,tstep=tstep,twin=twin,Fs = Fs,taxis=taxis)
    
    # Update instance
    with sdata.info._unlock():
            sdata.info['sfreq'] = 1/tstep

    sdata.data = cdata
    sdata._set_times(np.array(times, dtype=float))
    sdata._raw_times = np.array(times, dtype=float)

    return sdata

def normalize_epochs(epochs):
    data_mean = np.mean(epochs.get_data(), axis=(0,2), keepdims=True)
    data_std = np.std(epochs.get_data(), axis=(0,2), keepdims=True)
    norm_data = (epochs.get_data() - data_mean)/data_std
    norm_data /= 1000
    epochs = mne.EpochsArray(norm_data, epochs.info, events=epochs.events, baseline=epochs.baseline,
                              event_id = epochs.event_id, tmin = epochs.tmin, on_missing='warn')
    return epochs        

def set_reference(raw0, bads=None, white=None, rename=True, summary=False):
    
    if bads is None:
        bads = []
    if white is None:
        white = []
    
    pattern = r'[0-9]'
    shafts = [re.sub(pattern, '', s) for s in raw0.ch_names]

    # Get pairs of adjacent contacts on same shaft
    pairs = [(ch1, ch2) for ch1, ch2, s1, s2 in zip(
        raw0.ch_names[:-1], raw0.ch_names[1:], shafts[:-1], shafts[1:]) if s1 == s2]

    anode = [p[0] for p in pairs]
    cathode = [p[1] for p in pairs]

    # Compute new coordinates as midpoint of each pair
    coords = {ch['ch_name']: ch['loc'][:3] for ch in raw0.info['chs']}
    new_coords = [(coords[a] + coords[c]) / 2 for a, c in pairs]

    # Bipolar reference
    raw_bip = mne.set_bipolar_reference(raw0, anode, cathode, drop_refs=True)

    # Rename channels meaningfully
    if rename:
        new_names = {f"{a}-{c}": f"{a}_{c}" for a, c in pairs}  # optional underscore
        raw_bip.rename_channels(new_names)

    # Build montage dictionary (channel_name → coordinates)
    montage_coords = {raw_bip.ch_names[i]: new_coords[i] for i in range(len(raw_bip.ch_names))}
    montage = mne.channels.make_dig_montage(ch_pos=montage_coords, coord_frame='mni_tal')
    raw_bip.set_montage(montage, on_missing='warn')

    # Restore annotations
    raw_bip.set_annotations(raw0.annotations.copy())

    # Mark bad/white channels
    for idx, name in enumerate(raw_bip.ch_names):
        a, c = pairs[idx]
        if a in bads or c in bads:
            if summary: print(f"Marking bad: {name}")
            raw_bip.info['bads'].append(name)
        elif a in white or c in white:
            if summary: print(f"Marking white: {name}")
            raw_bip.info['bads'].append(name)

    return raw_bip

def ExcludSubj(subj_included, data_path = OUT_PATH + '/Data') : 
    '''
    Exclude the ones that < 50 % performance or number of trials <24.
    '''
    excluded = []
    for subj in subj_included :   
        info_file = data_path + f'/{subj}_info.json'
        with open(info_file) as f:
            info = json.load(f)
            events_index = np.array([int(i) for i in info['event_id']])
        id_ev1 = np.where(events_index == 1)[0]
        id_ev2 = np.where(events_index == 2)[0]

        if len(events_index) < 24 :
            excluded.append(subj)
        elif 2 * len(id_ev1) / len(events_index) < 0.5 :
            excluded.append(subj)
        elif 2* len(id_ev2) / len(events_index) < 0.5 :
            excluded.append(subj)
        
    # update subj_include 
    subj_return = subj_included.copy()

    for e in excluded :
        subj_return.remove(e)

    return subj_return

################################### FUNCTIONS TFR ###################################

def MM_compute_TFR(epochs, freqs, n_cycles, baseline, zscore=True, trial_baseline = True, picks='all',n_jobs=2, summary=False):
    if summary : print('###### Call tfr morlet')
    TFR = mne.time_frequency.tfr_morlet(epochs,freqs,n_cycles,return_itc=False,average=False,n_jobs = n_jobs,picks=picks)
    
    if summary : print('##### Log transforming')
    for r in np.arange(TFR.data.shape[0]):
        if summary : print('trial ', r+1)
        TFR.data[r] = np.log(TFR.data[r])

    if zscore:
        if summary : print('##### z-scoring to baseline')
        bix = [a and b  for a, b in zip(TFR.times >= baseline[0], TFR.times <= baseline[1])]
        bmean = np.nanmean(TFR.data[:,:,:,bix],axis=(0,3),keepdims=True) 
        bstd = np.nanstd(TFR.data[:,:,:,bix],axis=(0,3),keepdims=True)

        TFR.data -= bmean 
        TFR.data /= bstd
        
    if trial_baseline:
        if summary : print('#####subtracting baseline per trial')
        bix = [a and b  for a, b in zip(TFR.times >= baseline[0], TFR.times <= baseline[1])]
        bmean = np.nanmean(TFR.data[:,:,:,bix],axis=(3),keepdims=True)
        bstd = np.nanstd(TFR.data[:,:,:,bix],axis=(3),keepdims=True)
        TFR.data -= bmean 
    return TFR

def TFR_mean(TFR,freq_bands=FREQ_BAND_DICT, freqs=FREQS, event_id= EVENT_ID, trials=True): 
    band_indices = {band: [freqs.index(f) for f in band_freqs if f in freqs] for band, band_freqs in freq_bands.items()}
    TFR_mean_band = np.zeros((TFR.data.shape[0], TFR.data.shape[1], len(freq_bands), TFR.data.shape[3]))
    for i, inde in enumerate(band_indices.values()):
        TFR_mean_band[:, :, i, :] = np.mean(TFR.data[:, :, inde, :], axis=2)

    unique_ev = np.unique(TFR.events[:, 2])
    id_ev ={}

    for val in unique_ev :
        idx = np.where(TFR.events[:, 2] == val)[0]
        id_ev[val] = idx


    if trials : 
        return TFR_mean_band
    else :
        TFR_mean_band_trials = np.zeros((len(event_id.values()), TFR_mean_band.shape[1], TFR_mean_band.shape[2], TFR_mean_band.shape[3]))
        for ev_id, (_, idx) in enumerate(id_ev.items()):
            TFR_mean_band_trials[ev_id, :, :, :] = np.mean(TFR_mean_band[idx, :, :, :], axis=0)

        return TFR_mean_band, TFR_mean_band_trials
    
def preproc(subj, sfreq = 600,new_sfreq = 200, freqs = FREQS, bwidth = BWIDTH, event_id = EVENT_ID, project_path = PROJECT_PATH, out_path = OUT_PATH + '/Data', trials = True, save_epoch=True, compute_TFR=True) :  
    info = {}
    if not os.path.exists(out_path) : 
        os.makedirs(out_path)

    bids_root = op.join(project_path,'data/BIDS/')
    bids_path = BIDSPath(subject=subj,task=TASK,root=bids_root)
    elec_path = op.join(project_path, 'misc/electrodes/', subj + '_electrodes.csv')

    bads_path = op.join(project_path, 'misc/bad_channels/', subj + '_bad_channels.csv')
    bad_channels, white_channels, _ = get_bads(subj, bads_path, elec_path)

    raw = read_raw_bids(bids_path)  
    raw.pick_types(seeg=True, ecog = True)

    onsets = []
    decription = []

    for ano in raw.annotations : 
        if 'bach' in ano['description'] : 
            onsets.append(ano['onset'])
            decription.append(ano['description'])
    if min(onsets) -2 <0 : 
        tmin_  = min(onsets)
    else : 
        tmin_ = min(onsets) -2
    raw.crop(tmin = tmin_ , tmax = max(onsets) + 50)
    raw.load_data() 
    raw = set_reference(raw, bads = bad_channels, white = white_channels)
    raw = raw.drop_channels(raw.info['bads'],on_missing='warn')

    ch_coord_df = pd.DataFrame({ch['ch_name']: ch['loc'][:3] for ch in raw.info['chs']}).T
    ch_coord_df = ch_coord_df.rename(columns = {0 : 'x', 1 : 'y', 2:'z'})
    
    for c in ['x', 'y', 'z'] :
        ch_coord_df[c] = ch_coord_df[c].apply(lambda x : x * 1000) # change the coordinate unit

    ch_coord_df['subj'] = subj

    raw.resample(sfreq=sfreq)
    l_freq=0.1
    h_freq=None
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    nfreqs = np.arange(1,np.ceil(raw.info['lowpass']/raw.info['line_freq']))*raw.info['line_freq']

    if nfreqs.size>0:            
        raw.notch_filter(freqs=nfreqs)

    try : 
        events, event_id = mne.events_from_annotations(raw)      
        events, event_id = LB_event_fun(events, event_id, sfreq=sfreq)
    except Exception as e : 
        print(subj) 
        print(e)
        return subj
    
    epoching_kwargs = {'tmin':-1.5, 'tmax': 5, 'baseline': None, 'resample': 600, 'l_freq': None, 'h_freq': None,
                             'event_fun': LB_event_fun, 'event_fun_kwargs': {'sfreq': 600}}
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=epoching_kwargs['tmin'], tmax=epoching_kwargs['tmax'],
                            preload=True, baseline=epoching_kwargs['baseline'], on_missing='warn',
                            reject=None)
    epochs = normalize_epochs(epochs)

    del raw

    if compute_TFR :
        n_cycles = np.array(freqs) * 2 / np.array(bwidth)
        TFR = MM_compute_TFR(epochs,np.array(freqs), n_cycles, baseline = (-1.5,4), zscore=True, trial_baseline = False, picks='all',n_jobs=1, summary = False)
        TFR = TFR.crop(-1.5,4)

        smooth_kwargs = {'tstep': 0.025, 'twin': .1}
        TFR = smooth(TFR, **smooth_kwargs)

        if trials :
            TFRtrials = TFR_mean(TFR=TFR, trials=trials)
        else : 
            TFRtrials, TFRm = TFR_mean(TFR=TFR) 
            with open(out_path + f'/{subj}_TFRm.p', "wb") as f:
                pickle.dump(TFRm, f)
            del TFRm

        with open(out_path + f'/{subj}_TFRtrials.p', "wb") as f:
            pickle.dump(TFRtrials, f)

        info['time_tfr'] = list(TFR.times)
        del TFRtrials
        del TFR

    epochs = epochs.crop(-1.5,4)
    epochs = epochs.resample(new_sfreq)

    if save_epoch : 
        with open(out_path + f'/{subj}_epochs.p', "wb") as f:
            pickle.dump(epochs.get_data(), f)

    # extracted and save info 
    info['chnames'] = epochs.info['ch_names']
    info['time_epoch'] = list(epochs.times)
    info['event_count'] = [[list(event_id.keys())[list(event_id.values()).index(event)], str(count)] for event, count in zip(np.unique(epochs.events[: , 2], return_counts=True)[0], np.unique(epochs.events[: , 2], return_counts=True)[1])]
    info['event_id'] = [str(e) for e in epochs.events[:, 2]]
    
    with open(out_path + f'/{subj}_info.json', "w") as f:
        json.dump(info, f)

    ch_coord_df.to_csv(out_path + f'/{subj}_coords.csv' )

def TFRmEvents(subj, event_ids = [1, 2], test_id=False, freq_id=False, events_index=[],  baseline_corr=False, data_path = OUT_PATH + '/Data') : 
    
    info_file = f'{data_path}/{subj}_info.json'
    with open(info_file) as f:
        info = json.load(f)
        time = info['time_tfr']
        if len(events_index) == 0 : 
            events_index = np.array([int(i) for i in info['event_id']])

    with open(f'{data_path}/{subj}_TFRtrials.p', "rb") as f:
        TFRtrials = pickle.load(f)

    if freq_id == False :
        TFRm = np.zeros((len(event_ids), TFRtrials.shape[1], TFRtrials.shape[2], TFRtrials.shape[3]))
    else :
        TFRm = np.zeros((len(event_ids), TFRtrials.shape[1], TFRtrials.shape[3]))

    for i, ev_id in enumerate(event_ids) :
        index_condi = list(np.where(events_index == ev_id)[0])

        if baseline_corr == 'mean' : 
            baseline_end = int(time.index([t for t in time if t>=-0.5][0]))
            TFRbmean = TFRtrials[np.array(index_condi), :, :, :baseline_end].mean(axis=-1, keepdims=True)
            TFRbstd = TFRtrials[np.array(index_condi), :, :, :baseline_end].std(axis=-1, keepdims=True)
            tfr = (TFRtrials[np.array(index_condi), :, :, :] - TFRbmean) 
        
        elif baseline_corr == 'z_score' :
            baseline_end = int(time.index([t for t in time if t>=-0.5][0]))
            TFRbmean = TFRtrials[np.array(index_condi), :, :, :baseline_end].mean(axis=-1, keepdims=True)
            TFRbstd = TFRtrials[np.array(index_condi), :, :, :baseline_end].std(axis=-1, keepdims=True)
            tfr = (TFRtrials[np.array(index_condi), :, :, :] - TFRbmean)/TFRbstd

        else :
            tfr = TFRtrials[np.array(index_condi), :, :, :]

        if test_id != False : # remove the excluded trials that we keep for testing
            index_condi.remove(test_id[i])
        
        if freq_id == False:
            TFRm[i, :,:, :] = tfr.mean(0) 

        else : 
            TFRm[i, :, :] = tfr[:, :,freq_id,:].mean(0)

    return TFRm
    
def BbEvents(subj, event_ids = [1, 2], test_id=False, events_index=[], data_path=OUT_PATH + '/Data') : 
    
    if len(events_index) == 0 : 
        info_file = f'{data_path}/{subj}_info.json'
        with open(info_file) as f:
            info = json.load(f)
            events_index = np.array([int(i) for i in info['event_id']])

    with open(f'{data_path}/{subj}_epochs.p', "rb") as f:
        epochs = pickle.load(f)

    epochsm = np.zeros((len(event_ids), epochs.shape[1], epochs.shape[2]))
    
    for i, ev_id in enumerate(event_ids) :
        index_condi1 = list(np.where(events_index == ev_id)[0])
        if test_id != False : # remove the excluded trials that we keep for testing
            index_condi1.remove(test_id[i])

        epochsm[i, :, :] = epochs[np.array(index_condi1), :,:].mean(0)
    return epochsm

################################### FUNCTIONS PCA ###################################
def ConcatPCA(concat_dict, ch_id = False, nb_compo =2, freq_band=FREQ_BAND, method ='pca') : 
    # Try to concat at subject level the event correct 
    df_list = []
    df_Componants = {}

    for subj, concat in concat_dict.items() : 
        df_compo_list = []
        if ch_id == False : 
            concat_ = concat.copy()
        else :
            id_channels = ch_id[subj]
            concat_ = concat[id_channels, :,:]

        for i, freq in enumerate(freq_band) : 
            if len(concat_.shape) == 2: 
                X = concat_ # if already one freq band or broad band
            else :
                X = concat_[:, i, :]

            # PCA
            if method == 'pca':
                pca = PCA(n_components=nb_compo)
                X_transformed = pca.fit_transform(X.T)
                df_compo = pd.DataFrame(pca.components_)

            elif method == 'nmf' :
                nmf = NMF(n_components=nb_compo, max_iter=1000)
                scaler = TimeSeriesScalerMinMax()
                X = scaler.fit_transform(X)[:,:,0]
                X_transformed = nmf.fit_transform(X.T) 
                df_compo =  pd.DataFrame(nmf.components_)
            
            # Save compo
            df_compo.loc[:, 'compo'] = ['compo' + str(i_compo+1) for i_compo in range(nb_compo)]
            df_compo.loc[:, 'freq'] = [freq_band[i]]*nb_compo
            df_compo_list.append(df_compo)

            # save 
            df = pd.DataFrame(X_transformed.T)
            df.loc[:, 'compo'] = ['compo' + str(i_compo+1) for i_compo in range(nb_compo)]
            df.loc[:, 'freq'] = [freq]*(nb_compo)
            df.loc[:, 'subj'] = [subj]*(nb_compo)
            if method == 'pca' :
                df.loc[:, 'expl_var'] = pca.explained_variance_ratio_ 
            df_list.append(df)   

        df_Componants[subj] = pd.concat(df_compo_list, axis=0)
    df_X_transformed = pd.concat(df_list) 
    return df_Componants, df_X_transformed

def PlotCompoIndividual(subj, df_Componants,subj_included=[], nb_compo = 2, save = True, show=False,browser=False, freq_band= FREQ_BAND, data_path = OUT_PATH +'/Data', out_path = OUT_PATH, project_path=PROJECT_PATH, method='pca') : 
    if subj == 'grp' : 
        if subj_included == [] : 
            subj_included = [file.replace('_info.json', '') for file in os.listdir(data_path) if file[-len('info.json'):] == 'info.json']

        coord = []
        for s in subj_included :
            coord_file = pd.read_csv(f'{data_path}/{s}_coords.csv')
            coord.extend(np.vstack([coord_file['x'].values,  coord_file['y'].values, coord_file['z'].values]).T)
        coord = np.array(coord)
    else : 
        coord_file = pd.read_csv(f'{data_path}/{subj}_coords.csv')
        coord = np.vstack([coord_file['x'].values,  coord_file['y'].values, coord_file['z'].values]).T

    combined_images = []
    to_remove = []
    for band in freq_band: 
        row_images = []
        data = df_Componants[subj].query("freq == @band").drop(columns = ['compo', 'freq']).values
        vlim = np.abs(data.flatten()).max()
        thr = data.mean() + abs(data.std())
        if method =='pca' :
            node_map = 'seismic'
            node_vmin=-vlim
            node_vmax=vlim
        elif method == 'nmf' :
            node_map ='Reds'
            node_vmin = 0
            node_vmax = vlim
        
        for compo_id in range(nb_compo) :
            index_thr = np.where(abs(data[compo_id, :]) > thr)
            fig = plotting.plot_markers(node_coords = coord[index_thr, :][0],  node_size=10, node_values=data[compo_id, index_thr], node_cmap=node_map, title=f'{band}:PC n{compo_id+1}',display_mode='ortho', node_vmin=node_vmin, node_vmax=node_vmax)
            
            #signal = df_Componants[subj].query(f'freq == @band and compo == "compo{compo_id +1}"').drop(columns = ['compo', 'freq']).values
            #fig = plotting.plot_markers(node_coords =coord,  node_size=10, node_values=signal, node_cmap='seismic', title=f'PC{compo_id+1} {band}',display_mode='ortho', node_vmin=-vlim, node_vmax=vlim)
            
            if show : plt.show()
            else : plt.close()

            if save :                
                temp_png = f'{out_path}/{subj}_PC{str(compo_id+1)}_{band}.png'
                fig.savefig(temp_png)
                row_images.append(Image.open(temp_png))
                to_remove.append(temp_png)

            if browser : 
                vmin = -np.max(np.abs(data)) 
                vmax = np.max(np.abs(data))
                norm_signal = (data - vmin) / (vmax - vmin)
                cmap = plt.cm.Blues
                colors = [tuple(cmap(val)) for val in norm_signal.flatten()]
                view = plotting.view_markers(coord, marker_color=colors,marker_size=5,title=f'PC{compo_id+1} {subj} {band}', title_fontsize=25 )
                view.open_in_browser()
            
        if save : 
            w, h = row_images[0].size
            row_combined = Image.new('RGB', (w*nb_compo, h))
            for idx, img in enumerate(row_images):
                row_combined.paste(img, (idx*w, 0))
            
            combined_images.append(row_combined)
    if save : 
        pdf_path = f'{out_path}/{subj}_compos.pdf'
        combined_images[0].save(pdf_path, save_all=True, append_images=combined_images[1:])

        #for img in to_remove : 
            #os.remove(img)

def PlotTimeSerie(subj_list, df_X_transformed, out_path, region='', show=False, save=True, tfr=True, data_path = OUT_PATH + '/Data')  :    
    for subj in subj_list :  
        # get the time 
        if subj == 'grp' : 
            json_file = data_path + '/' + [f for f in os.listdir(data_path) if '_info.json' in f][0]
        else : 
            json_file = f'{data_path}/{subj}_info.json'

        with open(json_file) as json_data:
            d = json.load(json_data)
            if tfr:
                time=np.array(d['time_tfr'])

            else : 
                time=np.array(d['time'])
                
        df_subset = df_X_transformed.query('subj == @subj')
        freq_nb = len(np.unique(df_subset.freq.values))
        compo_nb = len(np.unique(df_subset.compo.values))

        if freq_nb == 0 :
            freq_nb =1

        fig, axs = plt.subplots(1 , freq_nb, figsize = (30, 5))
        fig.suptitle('-- ' + subj + ' --')
        color_1 = [plt.cm.Blues(i) for i in np.linspace(0.7, 0.3, compo_nb)] 
        color_2 = [plt.cm.Reds(i) for i in np.linspace(0.7, 0.3, compo_nb)] 

        for band_id, band in enumerate(FREQ_BAND) : 
            for compo_id in range(compo_nb) : 
                df_to_plot = df_subset.query(f'freq == @band')
                the_ax = axs[band_id]

                # event1 
                df_to_plot_1 = df_to_plot.set_index('compo').loc[:, :len(time)-1]
                df_to_plot_1.loc['time', :] = time

                the_ax.plot(df_to_plot_1.loc['time', :], df_to_plot_1.loc['compo' + str(compo_id+1), :], label = f'PC{compo_id+1}', color=color_1[compo_id])
                
                # event2
                df_to_plot_2 = df_to_plot.set_index('compo').loc[:, len(time):len(time)*2 -1]                 
                df_to_plot_2.loc['time', :] = time
                the_ax.plot(df_to_plot_2.loc['time', :], df_to_plot_2.loc['compo' + str(compo_id+1), :], label = f'PC{compo_id+1}', color=color_2[compo_id])
            
                the_ax.set_xlabel('Time (s)')
                the_ax.set_title(band)
                the_ax.grid()

                if band_id == len(FREQ_BAND)-1 :
                    line_handles, line_labels = the_ax.get_legend_handles_labels()

                    cmap_patches = [mpatches.Patch(color=color_1[0], label='Old / Correct (Blues)'), 
                                    mpatches.Patch(color=color_2[0], label='New / Correct (Reds)')]
                    
                    handles = line_handles + cmap_patches
                    labels = line_labels + [p.get_label() for p in cmap_patches]

                    the_ax.legend(handles, labels, loc='upper left', frameon=True, bbox_to_anchor = (1.05, 0.7))

        if save : fig.savefig(f'{out_path}/{subj}_{region}_PCs.png')
        if show : plt.show()
        else  : plt.close()

################################### COMPO ANALYSIS ###################################
def GetInfo(subj_included, project_path = PROJECT_PATH, data_path = OUT_PATH + '/Data') : 
    coord = []
    areas = []
    elect_list = []
    subj_list = []
    for subj in subj_included: 
        df = pd.read_csv(f'{data_path}/{subj}_coords.csv').rename(columns={'Unnamed: 0' :'channels'})
        coord.extend(np.vstack([df['x'].values,  df['y'].values, df['z'].values]).T)

        elect = pd.read_csv(project_path + f'/misc/electrodes/{subj}_electrodes.csv').set_index('electrode')
        df['area1'] = df['channels'].apply(lambda x : elect.loc[x.split('_')[0], 'label'])
        df['area2'] = df['channels'].apply(lambda x : elect.loc[x.split('_')[0], 'label'])

        areas.extend(np.vstack([df['area1'], df['area2']]).T)
        elect_list.extend(df['channels'])

        subj_list.extend([subj]*len(df['channels']))
        
    return coord, areas, elect_list, subj_list

def CompoThr(data, replace=0) : 
    data_thr = data.copy()
    for i in range(data.shape[0]):
        thr = data[i,:].mean() + abs(data[i, :].std())      
        index_thr = np.where(abs(data[i, :]) < thr)
        data_thr[i, index_thr[0].flatten()] = replace

    return data_thr

def FindRegion(x) :
    for key, val in REGION.items() : 
        if x in val : 
            return key
    return None

################################### DECODING ###################################
def CheckTrials(X_train, y_train, event=list(EVENT_ID.keys())[:2] , out_path=OUT_PATH + '/Decoding', label = '', save=False, color_ev = {0 : 'r', 1 : 'b'}, freq='high_gamma', data_path=OUT_PATH + '/Data') : 
    # get time 
    files_info = [file for file in os.listdir(data_path) if file[-len('info.json'):] == 'info.json']
    with open(data_path + f'/{files_info[0]}') as json_data:
        d = json.load(json_data)
        if freq =='broadband' :
            time = d['time_epoch']
        else : 
            time = d['time_tfr']
        json_data.close()

    id_ev1 = np.where(np.array(y_train) == 1)[0]
    id_ev2 = np.where(np.array(y_train) == 2)[0]

    fig, axs = plt.subplots(figsize = (5, 3)) 
    fig.suptitle('Mean PC1 over trials', y=1.005)
    fig.tight_layout()
    for ev_i, ev in enumerate([id_ev1, id_ev2]) : 
        axs.plot(time, X_train[ev, :].mean(0), label = event[ev_i], color = color_ev[ev_i])
    axs.legend()
    if save :
        fig.savefig(out_path + f'/{freq}_{label}CheckTrials.png')
        plt.close()
    else : 
        plt.show()

    fig, axs = plt.subplots(1, 2, figsize = (15, 4))
    fig.suptitle(f'PCs trial exemple on event', y= 1.1)

    for ev_i, ev in enumerate([id_ev1, id_ev2]) :
        ids = random.sample(list(ev), 5) 
        for i in ids :
            axs[ev_i].plot(time, X_train[i, :])
            axs[ev_i].set_title(f'Event {event[ev_i]}')

    if save :
        fig.savefig(out_path + f'/{freq}_{label}CheckTrials.png')
        plt.close()
    else :
        plt.show()

def CheckTrialsMean(X_train,X_test, y_train, freq , event=list(EVENT_ID.keys())[:2], color_ev = {0 : 'r', 1 : 'b'}, save = False, out_path =OUT_PATH + '/Decoding', label='', data_path=OUT_PATH + '/Data') :
    files_info = [file for file in os.listdir(data_path) if file[-len('info.json'):] == 'info.json']
    with open(data_path + f'/{files_info[0]}') as json_data:
        d = json.load(json_data)
        time = d['time_tfr']
        json_data.close()
    id_ev1 = np.where(np.array(y_train) == 1)[0]
    id_ev2 = np.where(np.array(y_train) == 2)[0]
    fig, axs = plt.subplots(figsize = (15, 4))
    fig.suptitle(f'Train and Test samples')

    for ev_i, ev in enumerate([id_ev1, id_ev2]) :
        axs.plot(time, X_train[ev, :].mean(0), color = color_ev[ev_i], label =event[ev_i] + ' - Mean Train')
        axs.fill_between(time, X_train[ev, :].mean(0) - X_train[ev, :].std(0), X_train[ev, :].mean(0) + X_train[ev, :].std(0), color = color_ev[ev_i], alpha = 0.2,  label =event[ev_i] + ' - Std Train')
        axs.plot(time, X_test[ev_i, :], color = color_ev[ev_i], linestyle='dashed',  label =event[ev_i] + ' - Train')
    axs.legend()
    axs.grid()

    if save :
        fig.savefig(out_path + f'/{freq}_{label}CheckTrialsMean.png')
        plt.close()
    else : 
        plt.show()

def DataAugmentation(TFRtr,event_ids, data_aug_method='mean') : 
    id_ev1 = event_ids[0]
    id_ev2 = event_ids[1]
    TFR_trials_filled = np.full((23, 2, TFRtr.shape[1], TFRtr.shape[2]), np.nan)
    TFR_trials_filled[:len(id_ev1), 0, :] =  TFRtr[id_ev1, :, :] 
    TFR_trials_filled[:len(id_ev2), 1, :] =  TFRtr[id_ev2, :, :]
    true_trials = ~np.any(np.isnan(TFR_trials_filled), axis=(2, 3))

    if data_aug_method == 'mean':
        for i in range(2):  
            event_means = np.nanmean(TFR_trials_filled[:, i, :, :], axis=0, keepdims=True)
            nan_mask = np.isnan(TFR_trials_filled[:, i, :, :])
            TFR_trials_filled[:, i, :, :] = np.where(nan_mask, event_means, TFR_trials_filled[:, i, :, :])

    if data_aug_method == 'duplicat' :
        ids_w_fake1 = id_ev1
        ids_w_fake2 = id_ev2
        while len(ids_w_fake1) + len(id_ev1) < 23 : 
            ids_w_fake1 = np.concat([ids_w_fake1,id_ev1], axis=0)
        ids_w_fake1 = np.concat([ids_w_fake1, random.sample(list(id_ev1), 23-ids_w_fake1.shape[0])]).astype(int)

        while len(ids_w_fake2) + len(id_ev2) < 23 : 
            ids_w_fake2 = np.concat([ids_w_fake2,id_ev2], axis=0)
        ids_w_fake2 = np.concat([ids_w_fake2, random.sample(list(id_ev2), 23-ids_w_fake2.shape[0])]).astype(int)

        TFR_trials_filled[:, 0, :, :] =  TFRtr[ids_w_fake1, :, :] 
        TFR_trials_filled[:, 1, :, :] =  TFRtr[ids_w_fake2, :, :]

    return np.concatenate([TFR_trials_filled[:,i, :, :] for i in [0, 1]], axis = 0), true_trials

def DataTransformationM1(freq, freq_band=FREQ_BAND, out_path = OUT_PATH, PC_use=0, subj_included=[], method_pca='mean', data_aug_method='mean', shuffle_index = False, data_path = OUT_PATH + '/Data') : 
    TFRm_list = []

    Train_sample = []
    Test_sample = []
    truth = []

    if subj_included ==[] : 
        subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
 
    for subj in subj_included : 
        info_file = data_path + f'/{subj}_info.json'
        with open(info_file) as f:
            info = json.load(f)
            events_index = np.array([int(i) for i in info['event_id']])

        id_ev1 = np.where(events_index == 1)[0]
        id_ev2 = np.where(events_index == 2)[0]

        # Keep 1 id per condi for testing 
        id_test= [random.sample(list(id_ev1),1), random.sample(list(id_ev2),1)]
        id_ev1 = list(id_ev1)
        id_ev1.remove(id_test[0])
        id_ev1 = np.array(id_ev1)
        id_ev2 = list(id_ev2)
        id_ev2.remove(id_test[1])
        id_ev2 = np.array(id_ev2)

        if shuffle_index : # TO ADJUST TODO
            ev_shuffl= shuffle(np.concat([id_ev1, id_ev2]))
            id_ev1_s = ev_shuffl[:id_ev1.shape[0]]
            id_ev2_s = ev_shuffl[id_ev1.shape[0]:]

        # Compute TFRm 
        if freq == 'broadband' :
            TFRm = BbEvents(subj, test_id = id_test, events_index=events_index, data_path=data_path)
        else : 
            freq_id = freq_band.index(freq)
            TFRm = TFRmEvents(subj, test_id = id_test, freq_id = freq_id, events_index=events_index, data_path=data_path)

        # Save for PCA computation at grp level
        if method_pca == 'concat' :
            TFRm_list.append(np.concatenate([TFRm[i, :,:] for i in [0, 1]], axis = 1))
        if method_pca == 'mean' : 
            TFRm_list.append(np.mean(TFRm[[0, 1], :,:], axis = 0))

        # Get the data
        if freq == 'broadband' :
            file = data_path + f'/{subj}_epochs.p'
            with open(file, "rb") as f:
                TFRtr = pickle.load(f)  

            TFRtr_augmented, true_trials = DataAugmentation(TFRtr[:, :, :], [id_ev1, id_ev2], data_aug_method) # return 48, ch, time
            Train_sample.append(TFRtr_augmented)
            truth.append(true_trials)
            Test_sample.append(TFRtr[id_test,:, :])

        else :
            file = data_path + f'/{subj}_TFRtrials.p'
            with open(file, "rb") as f:
                TFRtr = pickle.load(f)  

            # Augment the data
            TFRtr_augmented, true_trials = DataAugmentation(TFRtr[:, :, freq_id, :], [id_ev1, id_ev2], data_aug_method) # return 48, ch, time
            Train_sample.append(TFRtr_augmented)

            truth.append(true_trials)
            Test_sample.append(TFRtr[id_test,:, freq_id, :])


    concat_all = np.concatenate(TFRm_list, axis = 0)
    del TFRm_list
    df_Componants, _ = ConcatPCA({'grp' : concat_all}, ch_id = False, nb_compo=3, freq_band=[freq])
    weights = df_Componants['grp'].query("freq == @freq").drop(columns = ['freq', 'compo']).values

    Train_all = np.concatenate(Train_sample, axis=1)
    Test_all = np.concatenate(Test_sample, axis =2)

    # Transform the data using the weights
    if type(PC_use) == list :
        Train_transformed = np.zeros([Train_all.shape[0],len(PC_use), Train_all.shape[-1]])
        Test_transformed = np.zeros([Test_all.shape[0], len(PC_use),Test_all.shape[-1]])
        for pc in PC_use : 
            Train_transformed[:, pc, :] = weights[pc, :] @ Train_all
            Test_transformed[:, pc, :] = weights[pc, :] @ Test_all[:,0,:]
            
    else : 
        Train_transformed = weights[PC_use, :] @ Train_all
        Test_transformed = weights[PC_use, :] @ Test_all[:,0,:]

    return Train_transformed, [1]*23 + [2]*23, Test_transformed, [1, 2], np.stack(truth, axis=2)*1, weights # X_train, y_train, X_test, y_test, subj_track_train, proportion of true trail in each supersample
 
def DataTransformationM1Raw(freq, freq_band=FREQ_BAND, out_path = OUT_PATH, subj_included=[], data_aug_method='mean', data_path = OUT_PATH + '/Data') :
    Train_sample = []
    Test_sample = []
    truth = []

    if subj_included ==[] : 
        subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
 
    for subj in subj_included : 
        info_file = data_path + f'/{subj}_info.json'
        with open(info_file) as f:
            info = json.load(f)
            events_index = np.array([int(i) for i in info['event_id']])

        id_ev1 = np.where(events_index == 1)[0]
        id_ev2 = np.where(events_index == 2)[0]

        # Keep 1 id per condi for testing 
        id_test= [random.sample(list(id_ev1),1), random.sample(list(id_ev2),1)]
        id_ev1 = list(id_ev1)
        id_ev1.remove(id_test[0])
        id_ev1 = np.array(id_ev1)
        id_ev2 = list(id_ev2)
        id_ev2.remove(id_test[1])
        id_ev2 = np.array(id_ev2)

        # Get the data
        if freq == 'broadband' :
            file = data_path + f'/{subj}_epochs.p'
            with open(file, "rb") as f:
                TFRtr = pickle.load(f)  

            TFRtr_augmented, true_trials = DataAugmentation(TFRtr[:, :, :], [id_ev1, id_ev2], data_aug_method) # return 48, ch, time
            Train_sample.append(TFRtr_augmented)
            truth.append(true_trials)
            Test_sample.append(TFRtr[id_test,:, :])

        else :
            freq_id = freq_band.index(freq)
            file = data_path + f'/{subj}_TFRtrials.p'
            with open(file, "rb") as f:
                TFRtr = pickle.load(f)  

            # Augment the data
            TFRtr_augmented, true_trials = DataAugmentation(TFRtr[:, :, freq_id, :], [id_ev1, id_ev2], data_aug_method) # return 48, ch, time
            Train_sample.append(TFRtr_augmented)

            truth.append(true_trials)
            Test_sample.append(TFRtr[id_test,:, freq_id, :])

    Train_all = np.concatenate(Train_sample, axis=1)
    Test_all = np.concatenate(Test_sample, axis =2)

    return Train_all, [1]*23 + [2]*23, Test_all[:, 0, :,:], [1, 2], np.stack(truth, axis=2)*1 

def LR(band, method_pca, data_aug_method,subj_included, iteration=100, perm=False, PC_use=0, save=False, out_path=f'{OUT_PATH}/Decoding/', iter_perm=1, data_path = OUT_PATH + '/Data') : 
    Y_PRED = []
    Y_TEST = []
    MODELS_weights = []
    X_TEST = []
    PCA_weights= []
    PVALUES =[]
    PVALUES_ll=[]
    ll_trial = []
    acc_trial =[]
    
    for i in range(iteration) :   
        X_train, y_train, X_test, y_test, True_trials, pca_weights = DataTransformationM1(freq= band, method_pca=method_pca, data_aug_method=data_aug_method, subj_included=subj_included, PC_use=PC_use, data_path=data_path)        
        X_train, y_train = shuffle(X_train, y_train, random_state =0)

        # scale -- -0.5

        if i == 0 :
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs']  # works with l2
            }

            base_model = LogisticRegression(max_iter=1000)
            grid = GridSearchCV(base_model, param_grid, cv=5)
            grid.fit(X_train, y_train)
            best_params = grid.best_params_

        model = LogisticRegression(**best_params, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        MODELS_weights.append(model.coef_)
    
        Y_PRED.extend(y_pred)
        Y_TEST.extend([1, 2])

        PCA_weights.append(pca_weights[PC_use, :])

        current_acc = accuracy_score(y_pred=y_pred, y_true=y_test)

        if perm : 
            acc_perm = []
            ll=[]
            for j in range(iter_perm) :
                # Shuffle the labels
                y_train_s = shuffle(y_train) #, random_state=0)
                model_s = LogisticRegression(**best_params, max_iter=1000)
                model_s.fit(X_train, y_train_s)    
                y_pred = model_s.predict(X_test) 
                acc_perm.append(accuracy_score(y_pred=y_pred, y_true=y_test))

                # LL
                ll.append(log_loss(y_test, model_s.predict_proba(X_test)[:,1]))
            
            # Proportion of permuted mean accuracies equal or higher than the original accuracy.
            #PVALUES.append(len([np.array(acc_perm) >= current_acc])/iter_perm)
            PVALUES.extend(acc_perm)
            PVALUES_ll.extend(ll)

        X_TEST.append(X_test)
        acc_trial.append(current_acc)
        ll_trial.append(log_loss(y_test, model.predict_proba(X_test)[:,1]))


    # save the result
    PCA_weights = np.vstack(PCA_weights)
    MODELS_weights = np.vstack(MODELS_weights)

    # Analysis 
    out_dir = out_path + band
    if not os.path.exists(out_dir) : 
        os.makedirs(out_dir)
    
    # Check trials plot
    #CheckTrialsMean(X_train,X_test, y_train, label = band+'_' + method_pca + '_' + data_aug_method, save=True, freq=band, data_path = data_path)
    #CheckTrials(X_train, y_train, label = band+'_' + method_pca + '_' + data_aug_method, save=True, freq=band,  data_path = data_path)

    #2. summary
    sumsum = pd.DataFrame()
    # info test
    sumsum.loc['band', 0] = band
    sumsum.loc['method_pca', 0] = method_pca
    sumsum.loc['method_data_augm', 0] = data_aug_method
    sumsum.loc['nb_iter', 0] = iteration
    sumsum.loc['trial_truth', 0] = np.round(True_trials.mean(), 2)

    # info model capacities
    sumsum.loc['F1', 0] = np.round(f1_score(Y_PRED, Y_TEST), 3)
    sumsum.loc['accuracy', 0] = np.round(accuracy_score(Y_PRED, Y_TEST), 3)

    count_unique = np.unique(Y_PRED, return_counts=True)
    sumsum.loc[f'count', 0] = count_unique

    if perm :
        sumsum.loc['nb_iter_perm', 0] = iter_perm
        sumsum.loc['for_pvalues_acc', 0] = PVALUES
        sumsum.loc['for_pvalues_ll', 0] = PVALUES_ll
        sumsum.loc['current_acc', 0] = acc_trial
        sumsum.loc['current_ll', 0] = ll_trial

    # info correlation pca stability
    n_runs = PCA_weights.shape[0]
    corr_matrix = np.zeros((n_runs, n_runs))
    for i in range(n_runs):
        for j in range(i, n_runs):
            rho, _ = spearmanr(PCA_weights[i], PCA_weights[j])
            corr_matrix[i, j] = rho
            corr_matrix[j, i] = rho
    corr = pd.DataFrame(np.abs(corr_matrix))
    corr.to_csv(out_dir + f'/{band}_{method_pca}_{data_aug_method}_{PC_use}_correlation.csv')
    
    lower_tri = np.tril(corr.values, k=-1)  
    mean = lower_tri[lower_tri != 0].mean()
    sumsum.loc['mean_corPC', 0] = mean

    #title = f"{sumsum.loc['band', :].values[0]} - {sumsum.loc['method_pca', :].values[0]} - {sumsum.loc['method_data_augm', :].values[0]} - mean:{np.round(mean, 2)}"
    #fig, ax = plt.subplots()
    #fig.suptitle('PCs correlations')
    #sns.heatmap(corr, xticklabels=False, yticklabels=False, cmap= 'Blues', ax=ax)
    #ax.set_title(title)

    if save:
        #fig.savefig(out_dir + f'/{band}_{method_pca}_{data_aug_method}_CorrPCs.png')
        sumsum.to_csv(out_dir + f'/{band}_{method_pca}_{data_aug_method}_{PC_use}_summary.csv')
        #plt.close()
    #else :
        #plt.show()

    # Model temporal importance plot
    fig, axs = plt.subplots(2, 1, figsize = (10, 8), sharex=True)
    fig.suptitle('Model analysis: Temporal Importance')
    fig.tight_layout()

    with open(data_path + f'/{subj_included[0]}_info.json') as json_data:
        d = json.load(json_data)
        if band == 'broadband' :
            time = d['time_epoch']
        else :
            time = d['time_tfr']

    json_data.close()

    color_ev = {0 : 'r', 1 : 'b'}
    event= list(EVENT_ID.keys())[:2]

    weights_clf = MODELS_weights.mean(0).ravel()
    axs[1].plot(time,abs(weights_clf),linewidth=1,color='black', label = 'Mean')
    axs[1].fill_between(time,abs(weights_clf - MODELS_weights.std(0).ravel()),abs(weights_clf + MODELS_weights.std(0).ravel()),alpha=0.3, linewidth=0,color='gray', label='Std')
    axs[1].set_title('LR Weights over iterations', size = 10)
    axs[1].grid()
    axs[1].legend()

    id_ev1 = np.where(np.array(y_train) == 1)[0]
    id_ev2 = np.where(np.array(y_train) == 2)[0]

    for ev_i, ev in enumerate([id_ev1, id_ev2]) : 
        axs[0].plot(time, X_train[ev, :].mean(0), color = color_ev[ev_i], label =event[ev_i] + ' - Mean over Training')
        axs[0].fill_between(time, X_train[ev, :].mean(0) - X_train[ev, :].std(0), X_train[ev, :].mean(0) + X_train[ev, :].std(0), color = color_ev[ev_i], alpha = 0.2,  label =event[ev_i] + ' - Std over Training')
        axs[0].plot(time, X_test[ev_i, :], color = color_ev[ev_i], linestyle='dashed',  label =event[ev_i] + ' - Testing')
    
    axs[0].legend()
    axs[0].grid()
    axs[0].set_title('Last run example', size=10)

    if save: 
        fig.savefig(out_dir + f'/{band}_{method_pca}_{data_aug_method}_{PC_use}_TemporalImportance.png' )
        plt.close()
    else : 
        plt.show()
        return sumsum

def TemporalGeneralization(band,method_pca,data_aug_method, subj_included, PC_use=0, undersampling=False, save=False, data_path = OUT_PATH + '/Data') : 
    out_dir = f'{OUT_PATH}/Decoding/{band}'
    if not os.path.exists(out_dir) : 
        os.makedirs(out_dir)

    X_train, y_train, X_test, y_test, _, _ = DataTransformationM1(freq= band, method_pca=method_pca, data_aug_method=data_aug_method, subj_included=subj_included, PC_use=PC_use, data_path=data_path)      
    
    X = np.concat([X_train, X_test], axis =0)
    y = np.concat([y_train, y_test])

    if undersampling : 
        X = X[:, ::5]

    _, n_time = X.shape
    scores = np.zeros((n_time, n_time))

    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    for t_train in range(n_time):
        X_t = X[:, t_train].reshape(-1, 1)

        for t_test in range(n_time):
            X_te = X[:, t_test].reshape(-1, 1)
            fold_scores = []

            for train_idx, test_idx in kf.split(X_t):
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_t[train_idx], y[train_idx])
                y_pred = clf.predict(X_te[test_idx])
                fold_scores.append(accuracy_score(y[test_idx], y_pred))

            scores[t_train, t_test] = np.mean(fold_scores)

    fig, ax = plt.subplots()
    im = ax.imshow(scores, vmin=0, vmax=1, origin='lower', aspect='auto', cmap='Blues')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Decoding Accuracy")
    ax.set_xlabel("Test Time")
    ax.set_ylabel("Train Time")
    ax.set_title(f"Temporal Generalization - mean score : {np.round(np.mean(scores), 2)}")
    if save :
        fig.savefig(OUT_PATH + '/Decoding/'+ band + f'/{band}_{method_pca}_{data_aug_method}_{PC_use}_TemporalGeneralization.png' )
        plt.close()
    else : 
        plt.show()

def TemporalLR(band, method_pca, data_aug_method,subj_included, iteration=100, PC_use=0, save=False, data_path = OUT_PATH + '/Data'):
    Y_PRED = []
    Y_TEST = []
    MODELS_weights = []

    param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']}
    best_params = {}

    for i in range(iteration) :     
        
        X_train, y_train, X_test, y_test, _, _ = DataTransformationM1(freq= band, method_pca=method_pca, data_aug_method=data_aug_method, subj_included=subj_included, PC_use=PC_use, data_path=data_path )        
        X_train, y_train = shuffle(X_train, y_train, random_state =0)

        weights = []
        y_pred= []
        y_test_all = []
        for t_point in range(X_train.shape[1]) :
            if i == 0 : 
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
                grid = GridSearchCV(LogisticRegression(), param_grid, cv=cv, scoring='accuracy')
                grid.fit(X_train[:, t_point].reshape(-1, 1), y_train)
                best_params[t_point] = grid.best_params_
                best_param = best_params[t_point]
                
            else : 
                best_param = best_params[t_point]

            clf = LogisticRegression(**best_param).fit(X_train[:,t_point].reshape(-1, 1), y_train)
            weights.append(clf.coef_.ravel())
            y_pred.append(clf.predict(X_test[:,t_point].reshape(-1, 1)))
            y_test_all.append(y_test)

        Y_PRED.append(np.vstack(y_pred))
        Y_TEST.append(np.vstack(y_test_all))

        MODELS_weights.append(np.vstack(weights)) 
    MODELS_weights = np.hstack(MODELS_weights)
    Y_PRED = np.concat(Y_PRED, axis =1)
    Y_TEST = np.concat(Y_TEST,  axis =1)

    # analysis 
    fig, axs = plt.subplots(3, 1, figsize = (10, 8),height_ratios=[0.2, 0.4, 0.4], sharex=True)
    fig.suptitle('Model analysis: Temporal Importance')
    fig.tight_layout()

    with open(data_path + f'/{subj_included[0]}_info.json') as json_data:
        d = json.load(json_data)
        if band == 'broadband' :
            time = d['time_epoch']
        else :
            time = d['time_tfr']
        json_data.close()
    color_ev = {0 : 'r', 1 : 'b'}
    event= list(EVENT_ID.keys())[:2]

    weights_clf = MODELS_weights.mean(1).ravel()
    axs[1].plot(time,abs(weights_clf),linewidth=1,color='black', label = 'Mean')
    axs[1].fill_between(time,abs(weights_clf - MODELS_weights.std(1).ravel()),abs(weights_clf + MODELS_weights.std(1).ravel()),alpha=0.3, linewidth=0,color='gray', label='Std')
    axs[1].set_title('LR Weights over iterations', size = 10)
    axs[1].grid()
    axs[1].legend()

    id_ev1 = np.where(np.array(y_train) == 1)[0]
    id_ev2 = np.where(np.array(y_train) == 2)[0]

    for ev_i, ev in enumerate([id_ev1, id_ev2]) : 
        axs[0].plot(time, X_train[ev, :].mean(0), color = color_ev[ev_i], label =event[ev_i] + ' - Mean over Training')
        axs[0].fill_between(time, X_train[ev, :].mean(0) - X_train[ev, :].std(0), X_train[ev, :].mean(0) + X_train[ev, :].std(0), color = color_ev[ev_i], alpha = 0.2,  label =event[ev_i] + ' - Std over Training')
        axs[0].plot(time, X_test[ev_i, :], color = color_ev[ev_i], linestyle='dashed',  label =event[ev_i] + ' - Testing')
    
    axs[0].legend()
    axs[0].grid()
    axs[0].set_title('Last run example', size=10)

    accuracies = (Y_PRED == Y_TEST).mean(axis=1)
    axs[2].plot(time, accuracies, color='black',linewidth=1, label='Mean')
    axs[2].set_title("Accuracy per Time Point", size = 10)
    axs[2].legend()
    axs[2].set_xlabel("Time")

    #2. summary
    sumsum = pd.DataFrame()

    # info test
    sumsum.loc['band', 0] = band
    sumsum.loc['method_pca', 0] = method_pca
    sumsum.loc['method_data_augm', 0] = data_aug_method
    sumsum.loc['nb_iter', 0] = iteration
    sumsum.loc['acc', 0] = np.mean(accuracies)

    # save
    if save:
        out_dir = f'{OUT_PATH}/Decoding/{band}'
        if not os.path.exists(out_dir) : 
            os.makedirs(out_dir)
        sumsum.to_csv(out_dir+ f'/{band}_{method_pca}_{data_aug_method}_{PC_use}_TpointSummary.csv')
        fig.savefig(out_dir+ f'/{band}_{method_pca}_{data_aug_method}_{PC_use}_TpointTemporalImportance.png' )
        plt.close()
    else :
        plt.plot()
        return sumsum

def TemporalLRRaw(band, data_aug_method,subj_included, iteration=100, PC_use=False, method_pca=False, save=False, data_path = OUT_PATH + '/Data'):
    Y_PRED = []
    Y_TEST = []
    MODELS_weights = []
    best_params = {}
    param_grid = {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']}

    for i in range(iteration) :     
        if PC_use == False : 
            X_train, y_train, X_test, y_test, _ = DataTransformationM1Raw(freq= band, data_aug_method=data_aug_method, subj_included=subj_included, data_path=data_path)        
        else : 
            X_train, y_train, X_test, y_test, _, _ = DataTransformationM1(freq= band, method_pca=method_pca, data_aug_method=data_aug_method, subj_included=subj_included, PC_use=PC_use, data_path = data_path )                

        # Shuffle 
        X_train, y_train = shuffle(X_train, y_train, random_state =0)
        # Scale
        #for pc in range(X_train.shape[1]) :    
        #    scaler = StandardScaler()
        #    X_train[:,pc, :] = scaler.fit_transform(X_train[:, pc, :])
        #    X_test[:, pc, :] = scaler.transform(X_test[:, pc, :])
   
        weights = []
        y_pred= []
        y_test_all = []
        for t_point in range(X_train.shape[-1]) :
            if i == 0 : 
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
                grid = GridSearchCV(LogisticRegression(), param_grid, cv=cv, scoring='accuracy')
                grid.fit(X_train[:,:,t_point], y_train)
                best_params[t_point] = grid.best_params_
                best_param = best_params[t_point]
                
            else : 
                best_param = best_params[t_point]

            clf = LogisticRegression(**best_param).fit(X_train[:,:,t_point], y_train)
            weights.append(clf.coef_.ravel())
            y_pred.append(clf.predict(X_test[:,:,t_point]))
            y_test_all.append(y_test)

        Y_PRED.append(np.vstack(y_pred))
        Y_TEST.append(np.vstack(y_test_all))
        MODELS_weights.append(np.vstack(weights)) 

    MODELS_weights = np.concat([i.reshape(i.shape[0], i.shape[1], 1) for i in MODELS_weights], axis=2)
    Y_PRED = np.concat(Y_PRED, axis =1)
    Y_TEST = np.concat(Y_TEST,  axis =1)

    # analysis 
    fig, axs = plt.subplots(3, 1, figsize = (10, 8),height_ratios=[0.4, 0.4, 0.4], sharex=False)
    fig.suptitle('Model analysis: Temporal Importance')
    fig.tight_layout()

    with open(data_path + f'/{subj_included[0]}_info.json') as json_data:
        d = json.load(json_data)
        if band == 'broadband' :
            time = d['time_epoch']
        else :
            time = d['time_tfr']
        json_data.close()

    weights_clf = MODELS_weights.mean(-1)

    color_ev = {0 : 'r', 1 : 'b'}
    event= list(EVENT_ID.keys())[:2]

    id_ev1 = np.where(np.array(y_train) == 1)[0]
    id_ev2 = np.where(np.array(y_train) == 2)[0]

    if PC_use == False : 
        X_train_to_plot = X_train.mean(1)
        X_test_to_plot = X_test.mean(1)
        title_label = 'Mean channels'
    else : 
        X_train_to_plot = X_train[:, 0, :]
        X_test_to_plot = X_test[:, 0, :]
        title_label = 'PC1'

    for ev_i, ev in enumerate([id_ev1, id_ev2]) : 
        axs[0].plot(time, X_train_to_plot[ev, :].mean(0), color = color_ev[ev_i], label =event[ev_i] + ' - Mean over Training')
        axs[0].fill_between(time, X_train_to_plot[ev,:].mean(0) - X_train_to_plot[ev, :].std(0), X_train_to_plot[ev, :].mean(0) + X_train_to_plot[ev, :].std(0), color = color_ev[ev_i], alpha = 0.2,  label =event[ev_i] + ' - Std over Training')
        axs[0].plot(time, X_test_to_plot[ev_i, :], color = color_ev[ev_i], linestyle='dashed',  label =event[ev_i] + ' - Testing')
    
    axs[0].legend()
    axs[0].grid()
    axs[0].set_title(title_label + ' last run example', size=10)

    sns.heatmap(abs(weights_clf.T), cmap='Blues', xticklabels=False, yticklabels=False,center=0, ax=axs[1])
    axs[1].set_title('LR Weights per time point mean over iterations', size = 10)
    if PC_use == False : 
        axs[1].set_ylabel('Channels')
    else : 
        axs[1].set_ylabel('PCs')

    accuracies = (Y_PRED == Y_TEST).mean(axis=1)
    std = (Y_PRED == Y_TEST).std(axis=1)
    axs[2].plot(time, accuracies, color='black',linewidth=1, label='Mean')
    axs[2].fill_between(time,abs(accuracies - std),abs(accuracies + std),alpha=0.3, linewidth=0,color='gray', label='Std')
    
    if PC_use == False : 
        l = 'Raw'
    else : 
        l = method_pca + '_pc' + ''.join([str(item+1) for item in PC_use])
    
    axs[2].set_title(f"Accuracy per Time Point {l} - mean : {np.round(np.mean(accuracies), 2)}", size = 10)
    axs[2].legend()
    axs[2].set_xlabel("Time")

    #2. summary
    sumsum = pd.DataFrame()
    # info test
    sumsum.loc['band', 0] = band
    sumsum.loc['method_data_augm', 0] = data_aug_method
    sumsum.loc['nb_iter', 0] = iteration
    sumsum.loc['accuracy', 0] = np.round(np.mean(accuracies), 2)
    # save
    if save :
        out_dir = f'{OUT_PATH}/Decoding/{band}'
        if not os.path.exists(out_dir) : 
            os.makedirs(out_dir)
        sumsum.to_csv(out_dir + f'/{band}_{l}_{data_aug_method}_TpointSummary.csv')
        fig.savefig(out_dir + f'/{band}_{l}_{data_aug_method}_TpointTemporalImportance.png' )
        plt.close()
    else : 
        plt.show()
        return sumsum

def TemporalGeneralizationRaw(band, data_aug_method, subj_included, save=False,PC_use=False,method_pca=False, undersampling=False, data_path = OUT_PATH + '/Data') : 
    out_dir = f'{OUT_PATH}/Decoding/{band}'
    if not os.path.exists(out_dir) : 
        os.makedirs(out_dir)

    if PC_use == False : 
        X_train, y_train, X_test, y_test, _ = DataTransformationM1Raw(freq= band, data_aug_method=data_aug_method, subj_included=subj_included, data_path=data_path)        
    else : 
        X_train, y_train, X_test, y_test, _, _ = DataTransformationM1(freq= band, method_pca=method_pca, data_aug_method=data_aug_method, subj_included=subj_included, PC_use=PC_use, data_path=data_path)                

    X = np.concat([X_train, X_test], axis =0)
    y = np.concat([y_train, y_test])

    if undersampling : 
        X = X[:, :, ::5]

    #for pc in range(X.shape[1]) :    
        #scaler = StandardScaler()
        #X[:,pc, :] = scaler.fit_transform(X[:, pc, :])

    _, _, n_time = X.shape
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = np.zeros((n_time, n_time))

    for t_train in range(n_time):
        X_t_train = X[:, :, t_train] 
        for t_test in range(n_time):
            X_t_test = X[:, :, t_test]
            fold_scores = []
            for train_idx, test_idx in kf.split(X_t_train):
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_t_train[train_idx], y[train_idx])
                y_pred = clf.predict(X_t_test[test_idx])
                fold_scores.append(accuracy_score(y[test_idx], y_pred))
            scores[t_train, t_test] = np.mean(fold_scores)

    fig, ax = plt.subplots()
    im = ax.imshow(scores, vmin=0, vmax=1, origin='lower', aspect='auto', cmap='Blues')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Decoding Accuracy")
    ax.set_xlabel("Test Time")
    ax.set_ylabel("Train Time")
    if PC_use == False : 
        l = 'Raw'
    else : 
        l = method_pca + '_pc' + ''.join([str(item+1) for item in PC_use])
    
    ax.set_title(f"Temporal Generalization {l.replace('_', ' ')} -- mean accuracy {np.round(np.mean(scores), 2)}")

    if save :
        fig.savefig(out_dir + f'/{band}_{l}_{data_aug_method}_TemporalGeneralization.png' )
        plt.close()
    else : 
        plt.show()

def CompareClassifier(band,method_pca,data_aug_method,subj_included, nb_iter=100, PC_use=0, save=False, perm=False, out_path = OUT_PATH + '/Decoding', data_path=OUT_PATH + '/Data') :

    classifiers = {
        'LR': LogisticRegression(max_iter=1000),
        'SVC_linear': SVC(kernel='linear'),
        'SVC_rbf': SVC(kernel='rbf'),
        'RandomForest': RandomForestClassifier(),
        'kNN_DTW': KNeighborsTimeSeriesClassifier(metric='dtw'),
    }

    param_grids = {
        'LR': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']},
        'SVC_linear': {'C': [0.01, 0.1, 1, 10]},
        'SVC_rbf': {'C': [0.01, 0.1, 1, 10], 'gamma': ['scale', 'auto']},
        'RandomForest': {'n_estimators': [50, 100], 'max_depth': [None, 5, 10], 'max_features': ['sqrt', 'log2']},
        'kNN_DTW': {'n_neighbors': [1, 3, 5]}, 
    }

    best_params = {}
    results = {key: {'accuracy': [], 'f1': [], 'y_pred': [], 'y_test':[]} for key in classifiers.keys()}

    for run in range(nb_iter):
        X_train, y_train, X_test, y_test, _, _ = DataTransformationM1(freq=band,method_pca=method_pca,data_aug_method=data_aug_method,subj_included=subj_included,PC_use=PC_use, data_path=data_path)
        X_train, y_train = shuffle(X_train, y_train, random_state=run)
        if perm : 
            y_train = shuffle(y_train, random_state=0)

        #scaler = StandardScaler()
        #X_train_scaled = scaler.fit_transform(X_train)[:, :, 0]  # (samples, timepoints)
        #X_test_scaled = scaler.transform(X_test)[:, :, 0]

        for name, clf in classifiers.items():
            if run == 0:
                grid = GridSearchCV(clf, param_grids[name], cv=5, scoring='accuracy')
                if name in ['kNN_DTW', 'TimeSeriesSVC']:
                    grid.fit(X_train[:, :, np.newaxis], y_train)
                else:
                    grid.fit(X_train, y_train)
                
                best_params[name] = grid.best_params_
                clf = clf.set_params(**best_params[name])
            else:
                clf = clf.set_params(**best_params[name])
            
            if name in ['kNN_DTW', 'TimeSeriesSVC']:
                clf.fit(X_train[:, :, np.newaxis], y_train)
                y_pred = clf.predict(X_test[:, :, np.newaxis])
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
            
            results[name]['accuracy'].append(accuracy_score(y_test, y_pred))
            results[name]['f1'].append(f1_score(y_test, y_pred, average='macro'))
            results[name]['y_pred'].extend(y_pred)
            results[name]['y_test'].extend(y_test)

    #PlotCompareModels(band, method_pca, data_aug_method, results,pc_used=PC_use, save=save, perm=perm)
    if perm :
        l = '_permuted'
    else : 
        l=''

    # save result 
    if save :
        df_res = pd.DataFrame(results)
        if not os.path.exists(out_path + f'/{band}') :
            os.makedirs(out_path + f'/{band}')

        df_res.to_csv(out_path + f'/{band}/{method_pca}_{data_aug_method}{l}_{PC_use}_CompareModels.csv')
    else : 
        return df_res

def PlotCompareModels(band, method_pca, data_aug_method, results, pc_used, save=False, out_path = OUT_PATH +'/Decoding', perm=False):
    clf_names = [k.replace('_', ' ') for k in results.keys()]
    acc_mean = [np.mean(results[k]['accuracy']) for k in results.keys()]
    acc_std  = [np.std(results[k]['accuracy'])  for k in results.keys()]
    f1_mean  = [np.mean(results[k]['f1'])       for k in results.keys()]
    f1_std   = [np.std(results[k]['f1'])       for k in results.keys()]
    x = np.arange(len(clf_names))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, acc_mean, marker='o', color='tab:blue', label='Accuracy (mean)')
    ax.fill_between(x, 
                    np.array(acc_mean) - np.array(acc_std),
                    np.array(acc_mean) + np.array(acc_std),
                    alpha=0.2, color='tab:blue')
    ax.plot(x, f1_mean, marker='o', color='tab:red', label='F1 Score (mean)')
    ax.fill_between(x, 
                    np.array(f1_mean) - np.array(f1_std),
                    np.array(f1_mean) + np.array(f1_std),
                    alpha=0.2, color='tab:red')

    ax.set_xticks(x)
    ax.set_xticklabels(clf_names, rotation=45, ha='right')
    ax.set_ylabel("Score")
    if perm :
        l = '_permuted'
    else : 
        l=''
        
    ax.set_title(f"Classifier Performance (100 runs) {l.replace('_', '')}: Accuracy & F1 ± 1 Std")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save :
        if not os.path.exists(out_path + f'/{band}') :
            os.makedirs(out_path + f'/{band}')
            
        fig.savefig(out_path + f'/{band}/{method_pca}_{data_aug_method}{l}_{pc_used}_CompareModels.png')
        plt.close()
    else : 
        plt.show()

################################### STATS ###################################

def DataTransformationM2(freq, freq_band=FREQ_BAND, out_path = OUT_PATH, PC_use=0, subj_included=[], method_pca='mean') : 
    subj_list = []
    TFRm_list = []
    ids_test = {}
    events = {}
    events_index = {}

    if subj_included ==[] : 
        subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(out_path + '/Data') if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
 
    for subj in subj_included : 
        coords_file = OUT_PATH + f'/Data/{subj}_coords.csv'
        coords= pd.read_csv(coords_file).subj
        subj_list.extend(coords.values)

        info_file = out_path + f'/Data/{subj}_info.json'
        with open(info_file) as f:
            info = json.load(f)
            events_index[subj] = np.array([int(i) for i in info['event_id']])

        id_ev1 = np.where(events_index[subj] == 1)[0]
        id_ev2 = np.where(events_index[subj] == 2)[0]

        # Keep 1 id per condi for testing 
        id_test= [random.sample(list(id_ev1),1), random.sample(list(id_ev2),1)]
        id_ev1 = list(id_ev1)
        id_ev1.remove(id_test[0])
        id_ev1 = np.array(id_ev1)
        id_ev2 = list(id_ev2)
        id_ev2.remove(id_test[1])
        id_ev2 = np.array(id_ev2)
        events[subj] = np.concat([id_ev1, id_ev2])

        # Compute TFRm 
        TFRm = TFRmEvents(subj, test_id = id_test)

        # Save for PCA computation at grp level
        if method_pca == 'concat' :
            TFRm_list.append(np.concatenate([TFRm[i, :, :,:] for i in [0, 1]], axis = 2))
        if method_pca == 'mean' : 
            TFRm_list.append(np.mean(TFRm[[0, 1], :, :,:], axis = 0))
        
        ids_test[subj] = id_test

    concat_all = np.concatenate(TFRm_list, axis = 0)

    df_Componants, _ = ConcatPCA({'grp' : concat_all}, ch_id = False, nb_compo=3)

    weights = df_Componants['grp'].query("freq == @freq").drop(columns = ['freq', 'compo']).values
    df_weights = pd.DataFrame(weights).T
    df_weights.loc[:, 'subj'] = subj_list
    df_weights = df_weights.set_index('subj')

    Train_sample = []
    Test_sample = []
    event_train = []
    event_test = []
    subj_track = []

    for subj in subj_included : 
        # Get the data
        file = out_path + f'/Data/{subj}_TFRtrials.p'
        freq_id = freq_band.index(freq)

        with open(file, "rb") as f:
            TFRtr = pickle.load(f)  

        # Transform the data using the weights
        subj_weights = df_weights.loc[subj, PC_use].values
        TFRtr_transformed = subj_weights @ TFRtr[:,:,freq_id, :]

        Train_sample.append(TFRtr_transformed[events[subj], :])
        event_train.extend(events_index[subj][events[subj]])

        subj_track.extend([subj]*len(events[subj]))

        Test_sample.append(TFRtr_transformed[ids_test[subj], :])
        event_test.extend([1, 2])


    return np.vstack(Train_sample), event_train, np.vstack(Test_sample)[:, 0,:], event_test, subj_track # X_train, y_train, X_test, y_test, subj_track_train

def PermLR_distrib(band, method_pca, data_aug_method,subj_included, iteration=100, PC_use=0, save=False, out_path=f'{OUT_PATH}/Decoding', iter_perm=1) : 
    acc =[]
    acc_s =[]
    scores=[]
    scores_s=[]
    for j in range(iter_perm) : 
        Y_PRED = []
        Y_TEST = []
        Y_PRED_S= []
        p=[]
        p_s=[]
        
        for i in range(iteration) :   
            X_train, y_train, X_test, y_test, _ , _ = DataTransformationM1(freq= band, method_pca=method_pca, data_aug_method=data_aug_method, subj_included=subj_included, PC_use=PC_use)        
            X_train, y_train = shuffle(X_train, y_train, random_state =0)
            Y_TEST.extend(y_test)

            if i == 0 :
                param_grid = {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2'],
                    'solver': ['lbfgs']  # works with l2
                }

                base_model = LogisticRegression(max_iter=1000)
                grid = GridSearchCV(base_model, param_grid, cv=5)
                grid.fit(X_train, y_train)
                best_params = grid.best_params_
            # model not shuffle
            model = LogisticRegression(**best_params, max_iter=1000)
            model.fit(X_train, y_train)
            Y_PRED.extend(model.predict(X_test))

            # model shuffle 

            y_train_s = shuffle(y_train, random_state=0)
            model_s = LogisticRegression(**best_params, max_iter=1000)
            model_s.fit(X_train, y_train_s)    
            Y_PRED_S.extend(model_s.predict(X_test)) 

            p.extend(model.predict_proba(X_test)[:,1]) # CHECK THAT I M DOING THE RIGHT THINGS WITH THE LABELS
            p_s.extend(model_s.predict_proba(X_test)[:,1])

        acc.append(accuracy_score(y_pred=Y_PRED, y_true=Y_TEST))
        acc_s.append(accuracy_score(y_pred=Y_PRED_S, y_true=Y_TEST))
        scores.append(log_loss(Y_TEST, p))
        scores_s.append(log_loss(Y_TEST, p_s))
    
    #2. summary
    if not os.path.isdir(out_path + '/stats') : 
        os.makedirs(out_path+'/stats')

    sumsum = pd.DataFrame()
    # info test
    sumsum.loc['band', 0] = band
    sumsum.loc['method_pca', 0] = method_pca
    sumsum.loc['method_data_augm', 0] = data_aug_method
    sumsum.loc['nb_iter', 0] = iteration # Iteration to built up the testing 

    # info perm 
    sumsum.loc['nb_iter_perm', 0] = iter_perm # Iteration to built up the pvalue
    sumsum.loc['acc', 0] = acc
    sumsum.loc['acc_p', 0] = acc_s
    sumsum.loc['entropy', 0] = scores
    sumsum.loc['entropy_p', 0] = scores_s

    if save : 
        sumsum.to_csv(out_path + f'/stats/PermDistrub_{band}_{method_pca}_{data_aug_method}.csv')

    else : 
        return sumsum

def PermLR_null(band, method_pca, data_aug_method,subj_included, iteration=100, PC_use=0, save=False, out_path=f'{OUT_PATH}/Decoding', iter_perm=1, accuracy = True, entropy=True) : 
    sumsum = pd.DataFrame()
    for s in range(3):
        Y_PRED = []
        Y_TEST = []
        p=[]

        for i in range(iteration) :   
            X_train, y_train, X_test, y_test, _ , _ = DataTransformationM1(freq= band, method_pca=method_pca, data_aug_method=data_aug_method, subj_included=subj_included, PC_use=PC_use)        
            X_train, y_train = shuffle(X_train, y_train, random_state =0)

            if i == 0 :
                param_grid = {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l2'],
                    'solver': ['lbfgs']  # works with l2
                }

                base_model = LogisticRegression(max_iter=1000)
                grid = GridSearchCV(base_model, param_grid, cv=5)
                grid.fit(X_train, y_train)
                best_params = grid.best_params_

            # model not shuffle
            model = LogisticRegression(**best_params, max_iter=1000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
            Y_PRED.extend(y_pred)
            Y_TEST.extend(y_test)
            p.extend(model.predict_proba(X_test)[:,1]) # CHECK THAT I M DOING THE RIGHT THINGS WITH THE LABELS
        
        # STATS BUILT UP THE NULL 
        acc = accuracy_score(Y_TEST, Y_PRED)
        if accuracy : 
            acc_perm_list = []
            for k in range(iter_perm):
                acc_perm_list.append(accuracy_score(shuffle(Y_TEST), Y_PRED) )
            
            #perm_scores_array = np.array(acc_perm_list)
            #mask = [perm_scores_array >= acc]
            #p_value_acc = len(perm_scores_array[mask])/iter_perm

        score = log_loss(Y_TEST, p)
        if entropy :
            perm_scores =[]
            for k in range(iter_perm):
                y_perm = np.random.permutation(Y_TEST)
                score_k = log_loss(y_perm, p)
                perm_scores.append(score_k)

            #perm_scores_array = np.array(perm_scores)
            #mask = [perm_scores_array <= score]
            #p_value_entropy = len(perm_scores_array[mask])/iter_perm

        #2. summary
        if not os.path.isdir(out_path + 'stats/') : 
            os.makedirs(out_path+'stats/')

        
        # info test
        sumsum.loc['band', s] = band
        sumsum.loc['method_pca', s] = method_pca
        sumsum.loc['method_data_augm', s] = data_aug_method
        sumsum.loc['nb_iter', s] = iteration # Iteration to built up the testing 

        # info model capacities
        sumsum.loc['acc', s] = np.round(acc, 3)
        sumsum.loc['entropy', s] = np.round(score, 3)

        # info perm 
        sumsum.loc['nb_iter_perm', s] = iter_perm # Iteration to built up the pvalue
        if entropy : 
            sumsum.loc['entropy_p', s] = str(perm_scores)
            #sumsum.loc['entropy_pvalue', 0] = p_value_entropy
        if accuracy : 
            sumsum.loc['acc_p', s] = str(acc_perm_list)
            #sumsum.loc['acc_pvalues', 0] = p_value_acc

    if save : 
        sumsum.to_csv(out_path + f'/stats/PermNull_{band}_{method_pca}_{data_aug_method}_pc{PC_use+1}.csv')

    else : 
        return sumsum
    
def PermLR_Final(band, method_pca, data_aug_method,subj_included, iteration=100, PC_use=0, save=False, out_path=f'{OUT_PATH}/Decoding', iter_perm=1, data_path=OUT_PATH + '/Data') :

    TFRm_list = []
    Train_sample = []
    Test_sample = []
    truth = []

    # TO STORE
    Y_TEST = []
    Y_PRED = []
    Y_PRED_SH = []
    p= []
    p_sh  =[]
    weights_model = []
    weights_model_sh = []

    if subj_included ==[] : 
        subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']

    for i in range(iteration):
        for subj in subj_included[:3] : 
            info_file = data_path + f'/{subj}_info.json'
            with open(info_file) as f:
                info = json.load(f)
                events_index = np.array([int(i) for i in info['event_id']])

            id_ev1 = np.where(events_index == 1)[0]
            id_ev2 = np.where(events_index == 2)[0]

            # Keep 1 id per condi for testing 
            id_test= [random.sample(list(id_ev1),1), random.sample(list(id_ev2),1)]
            id_ev1 = list(id_ev1)
            id_ev1.remove(id_test[0])
            id_ev1 = np.array(id_ev1)
            id_ev2 = list(id_ev2)
            id_ev2.remove(id_test[1])
            id_ev2 = np.array(id_ev2)
            
            # Compute TFRm 
            if freq == 'broadband' :
                TFRm = BbEvents(subj, test_id = id_test, events_index=events_index, data_path=data_path)
            else : 
                freq_id = FREQ_BAND.index(freq)
                TFRm = TFRmEvents(subj, test_id = id_test, freq_id = freq_id, events_index=events_index, data_path=data_path)

            # Save for PCA computation at grp level
            if method_pca == 'concat' :
                TFRm_list.append(np.concatenate([TFRm[i, :,:] for i in [0, 1]], axis = 1))
            if method_pca == 'mean' : 
                TFRm_list.append(np.mean(TFRm[[0, 1], :,:], axis = 0))

            # Get the data
            if freq == 'broadband' :
                file = out_path + f'/Data/{subj}_epochs.p'
                with open(file, "rb") as f:
                    TFRtr = pickle.load(f)  

                TFRtr_augmented, true_trials = DataAugmentation(TFRtr[:, :, :], [id_ev1, id_ev2], data_aug_method) # return 48, ch, time
                Train_sample.append(TFRtr_augmented)
                truth.append(true_trials)
                Test_sample.append(TFRtr[id_test,:, :])

            else :
                file = out_path + f'/Data/{subj}_TFRtrials.p'
                with open(file, "rb") as f:
                    TFRtr = pickle.load(f)  

                # Augment the data
                TFRtr_augmented, true_trials = DataAugmentation(TFRtr[:, :, freq_id, :], [id_ev1, id_ev2], data_aug_method) # return 48, ch, time
                Train_sample.append(TFRtr_augmented)

                truth.append(true_trials)
                Test_sample.append(TFRtr[id_test,:, freq_id, :])

            
        concat_all = np.concatenate(TFRm_list, axis = 0) 
        Train_all = np.concatenate(Train_sample, axis=1)
        y_train = [1]*23 + [2]*23
        Test_all = np.concatenate(Test_sample, axis =2)
        Y_TEST.extend([1, 2])

        # NORMAL 
        df_Componants, _ = ConcatPCA({'grp' : concat_all}, ch_id = False, nb_compo=3, freq_band=[freq])
        weights = df_Componants['grp'].query("freq == @freq").drop(columns = ['freq', 'compo']).values

        if type(PC_use) == list :
            Train_transformed = np.zeros([Train_all.shape[0],len(PC_use), Train_all.shape[-1]])
            Test_transformed = np.zeros([Test_all.shape[0], len(PC_use),Test_all.shape[-1]])
            for pc in PC_use : 
                Train_transformed[:, pc, :] = weights[pc, :] @ Train_all
                Test_transformed[:, pc, :] = weights[pc, :] @ Test_all[:,0,:]
                
        else : 
            Train_transformed = weights[PC_use, :] @ Train_all
            Test_transformed = weights[PC_use, :] @ Test_all[:,0,:]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(Train_transformed)
        X_test = scaler.transform(X_test)
        
        if i == 0 :
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs']  # works with l2
            }

            base_model = LogisticRegression(max_iter=1000)
            grid = GridSearchCV(base_model, param_grid, cv=5)
            grid.fit(X_train, y_train)
            best_params = grid.best_params_

        # model not shuffle
        model = LogisticRegression(**best_params, max_iter=1000)
        model.fit(X_train, y_train)
        Y_PRED.extend(model.predict(X_test))
        p.extend(model.predict_proba(X_test)[:,1])
        weights_model.append(model.coef_)

        # SHUFFLED 
        for j in range(iter_perm) : 
            # Shuffleing TFRm to compute PCA
            # split the concat shuffle the of one before one after 
            concat_ev1 = concat_all[:, :int(concat_all.shape[1]/2)]
            concat_ev2 = concat_all[:, int(concat_all.shape[1]/2):]
            concat_ev12=np.concatenate([[concat_ev1], [concat_ev2]]) # got (2, channels, time) 
            concat_ev12_shuffled = concat_ev12.copy()
            np.random.shuffle(concat_ev12_shuffled)  # shuffle axis = 0
            concat_all_sh = np.concat([concat_ev12_shuffled[i, :,:] for i in range(2)], axis = 1) # concat the time
            df_Componants_sh, _ = ConcatPCA({'grp' : concat_all_sh}, ch_id = False, nb_compo=3, freq_band=[freq])
            weights_sh = df_Componants_sh['grp'].query("freq == @freq").drop(columns = ['freq', 'compo']).values

            # Applied on train 
            if type(PC_use) == list : 
                Train_transformed_sh = np.zeros([Train_all.shape[0],len(PC_use), Train_all.shape[-1]])
                Test_transformed_sh = np.zeros([Test_all.shape[0], len(PC_use),Test_all.shape[-1]])
                for pc in PC_use : 
                    Train_transformed_sh[:, pc, :] = weights_sh[pc, :] @ Train_all
                    Test_transformed_sh[:, pc, :] = weights_sh[pc, :] @ Test_all[:,0,:]
                    
            else : 
                Train_transformed_sh = weights_sh[PC_use, :] @ Train_all
                Test_transformed_sh = weights_sh[PC_use, :] @ Test_all[:,0,:]
            
            # Shuffle the labels
            y_train_sh = shuffle(y_train)

            Train_transformed_sh = scaler.fit_transform(Train_transformed_sh)
            Test_transformed_sh = scaler.transform(Test_transformed_sh)
         
            # applied the model
            model_sh = LogisticRegression(**best_params, max_iter=1000)
            model_sh.fit(Train_transformed_sh, y_train_sh)
            Y_PRED_SH.extend(model_sh.predict(Test_transformed_sh))
            p_sh.extend(model_sh.predict_proba(Test_transformed_sh)[:,1])
            weights_model_sh.append(model_sh.coef_)

        sumsum= pd.DataFrame()
        sumsum['band'] =freq
        sumsum['method_pca'] = method_pca
        sumsum['data_aug_method'] = data_aug_method
        sumsum['iter'] = iteration
        sumsum['iter_perm']=iter_perm
        sumsum['y_pred'] = Y_PRED
        sumsum['y_pred_sh'] = Y_PRED_SH
        sumsum['y_test'] = Y_TEST
        sumsum['entropy'] = p
        sumsum['entropy_sh'] = p_sh
        sumsum['weight'] = weights_model
        sumsum['weight_sh'] = weights_model_sh

        if save : 
            if not os.path.isdir(out_path + f'/stats/{freq}') : 
                os.makedirs(out_path + f'/stats/{freq}')
                
            sumsum.to_csv(out_path + f'/stats/{freq}/{method_pca}_{data_aug_method}_{PC_use}_PermDistrub.csv')

        else : 
            return sumsum

################################### VIZ AND INTRO (CHAT) ###################################
    
def PolarChannel(data1, title="Channels", elects=[], subjs = [], cmap_name = 'Blues', to_black = [], data_path = OUT_PATH + '/Data'):
    C, T = data1.shape
    cmap = cm.get_cmap(cmap_name, C)
    theta = np.linspace(0, 2*np.pi, C, endpoint=False)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
    subj_included = ExcludSubj(subj_included, data_path=data_path)
    with open(data_path + f'/{subj_included[0]}_info.json') as json_data:
        d = json.load(json_data)
        time = d['time_tfr']
        json_data.close()
    z = time
    
    for ch in range(C):
        amp = data1[ch]            
        th  = theta[ch]             
        x = amp * np.cos(th)
        y = amp * np.sin(th)

        color = cmap(0.3 + (1 - 0.3) * (ch / (C - 1))) 
        ax.plot(z, y, x, alpha=0.9, label=f"Ch {elects[ch].replace('`', '')} subj {subjs[ch]}", color=color)

    if len(to_black) != 0 :
        for i in to_black : 
            amp = data1[i]            
            th  = theta[i]             
            x = amp * np.cos(th)
            y = amp * np.sin(th)
            ax.scatter(z, y,x,s=2, alpha=0.9, label=f"Ch {elects[i].replace('`', '')} subj {subjs[i]}", color='black')
        
    ax.legend(bbox_to_anchor=(1.6, 1))
    ax.set_title(title)
    ax.set_zlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("Time")
    plt.show()

def PolarChannelWithPCA(data1, title="Channels", elects=[], subjs=[], cmap_name='Blues', data_full=None, idx_picks=None, data_path = OUT_PATH + '/Data'):
    C, T = data1.shape
    theta = np.linspace(0, 2*np.pi, C, endpoint=False)
    cmap = cm.get_cmap(cmap_name, C)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file.endswith('TFRtrials.p')]
    subj_included = ExcludSubj(subj_included, data_path=data_path)
    with open(data_path + f'/{subj_included[0]}_info.json') as json_data:
        d = json.load(json_data)
        time = d['time_tfr']

    z = time 
    for ch in range(C):
        amp = data1[ch]
        th = theta[ch]
        x = amp * np.cos(th)
        y = amp * np.sin(th)
        color = cmap(0.3 + 0.7 * (ch / (C - 1)))
        ax.plot(z, y, x, alpha=0.9, label=f"Ch {elects[ch]} subj {subjs[ch]}", color=color)

    pca = PCA(n_components=2)
    pca.fit(data_full.T)  # transpose: samples x features
    components = pca.components_  # shape (2, C)

    scale = 40
    for i, pc in enumerate(components):
        pc_x = np.sum(pc[idx_picks] * np.cos(theta)) * scale
        pc_y = np.sum(pc[idx_picks] * np.sin(theta)) * scale
        pc_z = 0 # draw arrow at z=0 (bottom of plot)

        ax.quiver(0, 0, pc_z, pc_y, pc_x, 0, color='red' if i==0 else 'blue', 
                  linewidth=2, arrow_length_ratio=0.2, label=f'PC{i+1}')

    ax.set_title(title)
    ax.set_zlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)
    ax.set_xlabel("Time")
    ax.legend(bbox_to_anchor=(1.6, 1))
    plt.show()

def PolarChannelPloty(data, title="Channels", elects='', subjs='', cmap_name = 'Blues', data_path = OUT_PATH + '/Data'):
    C, T = data.shape
    cmap = cm.get_cmap(cmap_name, C)
    theta = np.linspace(0, 2 * np.pi, C, endpoint=False)
    subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
    subj_included = ExcludSubj(subj_included, data_path=data_path)
    with open(data_path + f'/{subj_included[0]}_info.json') as json_data:
        d = json.load(json_data)
        time = d['time_tfr']
        json_data.close()

    z = time
    fig = go.Figure()
    for ch in range(C):
        amp = data[ch]                # radius = amplitude
        th  = theta[ch]               # fixed angle
        
        x = amp * np.cos(th)
        y = amp * np.sin(th)
        import matplotlib.colors as mcolors

        c = cmap(0.1 + (1 - 0.1) * (ch / (C - 1))) 
        color = mcolors.to_hex(c)

        fig.add_trace(go.Scatter3d(
            x=z,
            y=y,
            z=x,
            mode="lines",
            line=dict(width=4),
            name=f"Ch {elects[ch]} subj {subjs[ch]}", 
            line_color=color
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            zaxis_title="",
            yaxis_title=")",
            xaxis_title="Time",
            aspectmode="cube"
        ),
        showlegend=True,
        width=900,
        height=800
    )

    fig.show()

def PolarChannelSequential(data1, data2, title="Channels", elects=[], subjs=[], data_path = OUT_PATH + '/Data'):
    C, T1 = data1.shape
    _, T2 = data2.shape
    theta = np.linspace(0, 2*np.pi, C, endpoint=False)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    subj_included = [file.replace('_TFRtrials.p', '') 
                     for file in os.listdir(data_path) 
                     if file.endswith('TFRtrials.p')]
    
    subj_included = ExcludSubj(subj_included, data_path=data_path)
    with open(data_path + f'/{subj_included[0]}_info.json') as json_data:
        d = json.load(json_data)
        time1 = np.array(d['time_tfr'])
    
    z1 = time1
    z2 = time1[-1] + np.diff(time1).mean() + time1 
    cmap1 = cm.get_cmap("Blues")
    cmap2 = cm.get_cmap("Reds")

    for ch in range(C):
        amp = data1[ch]
        th = theta[ch]
        x = amp * np.cos(th)
        y = amp * np.sin(th)
        color = cmap1(0.3 + 0.7 * (ch / (C - 1)))
        ax.plot(z1, y, x, color=color, alpha=0.9,
                label=f"Ch {elects[ch]} subj {subjs[ch]} (data1)")

    for ch in range(C):
        amp = data2[ch]
        th = theta[ch]
        x = amp * np.cos(th)
        y = amp * np.sin(th)
        color = cmap2(0.3 + 0.7 * (ch / (C - 1)))
        ax.plot(z2, y, x, color=color, alpha=0.9,
                label=f"Ch {elects[ch]} subj {subjs[ch]} (data2)")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.legend(bbox_to_anchor=(1.6, 1))
    plt.show()

