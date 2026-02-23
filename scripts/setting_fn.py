import os.path as op
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

plt.style.use('seaborn-v0_8-dark')
OUT_PATH = 'outs'    
from scripts.preproc_fn import TFRmEvents, BbEvents
################################### GLOBAL VARIABLES ###################################

EVENT_ID = {'old/correct': 1,
 'new/correct': 2,
 'old/incorrect': 101,
 'new/incorrect': 102,
 'old/null': 201,
 'new/null': 202}

FREQS = [0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,15,17,19,21,24,27,30,35,40,45,50,55,60,70,80,90,100,110,120,130,140,150,160,180]
BWIDTH = np.array([0.5,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,5,5,5,5,5,5,10,10,10,10,10,10,10,10,10,10,20])

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
        'VS': ['LG','FUG','ITG','ITS'], #ventral stream
        'THAL': ['THAL']}

TASK = 'MusicMemory'

################################### FUNCTIONS ###################################

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

def get_data_grp(subj_included, type_data='epoch', polarity_cor = (True, 0.7), return_subj=False, data_path=OUT_PATH + '/Data') : 
    data = []
    subj_list = []
    for subj in subj_included: 
        if type_data == 'epoch' :
            data_mean = BbEvents(subj, data_path=data_path)
        else : 
            data_mean = TFRmEvents(subj, data_path=data_path)
        data.append(data_mean)
        subj_list.extend([subj] * data_mean.shape[1])

    data_grp = np.hstack(data)
    if polarity_cor[0]:
        data_grp = PolarityCor(TFRtr=data_grp, data_path=data_path, method_pca='mean', subj_included=subj_included, cor_thr=polarity_cor[1])

    if return_subj :
        return data_grp, subj_list
    else :
        return data_grp
    
def GetInfo(subj_included, project_path = PROJECT_PATH, data_path = OUT_PATH + '/Data', save=False) : 
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
    region = [FindRegion(a[0]) for a in areas]
    if save : 
        d = pd.DataFrame(coord, columns = ['x', 'y', 'z'])
        d.loc[:, ['area1', 'area2']] = areas
        d.loc[:, 'elect'] = elect_list
        d.loc[:, 'subj'] = subj_list
        d.loc[:, 'region'] = region

        d.to_csv(data_path + '/grp_info.csv')        
    else : 
        return coord, areas, elect_list, subj_list, region

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

def PolarityCor(TFRtr, subj_included=[], data_path =OUT_PATH + '/Data', method_pca='mean', cor_thr=0.5) : 
    data_mean_list = []
    if subj_included == [] :    
        subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
 
    for subj in subj_included: 
        data_mean = BbEvents(subj, data_path=data_path)
        data_mean_list.append(data_mean)
        
    data_grp = np.hstack(data_mean_list)
    pca = PCA(1)

    if method_pca == 'mean' : 
        data_grp_mean = data_grp.mean(0)
        ref = pca.fit_transform(data_grp_mean.T)[:, 0]
        ref = ref - ref.mean(0)

    if method_pca == 'concat':
        data_grp_concat = np.concat([data_grp[i, :,:] for i in [0, 1]], axis=1)
        ref = pca.fit_transform(data_grp_concat.T)[:, 0]
        ref = (ref[:int(ref.shape[0]/2)] +  ref[int(ref.shape[0]/2):])/2 # mean of the 2 compo
        ref = ref - ref.mean(0)
    
    signal_ref = smooth_moving_average(ref[100:600],window=80 )

    corr =[]
    if len(TFRtr.shape) == 2 : 
        length = TFRtr.shape[0] # nb of electrode to check and possibly flip
    else :
        length = TFRtr.shape[1]

    for i_signal in range(length) :
        if len(TFRtr.shape) == 2 : 
            signal = TFRtr[i_signal, : ]
            #signal_ = (signal[:int(signal.shape[0]/2)] +  signal[int(signal.shape[0]/2):])/2 # mean accros condition
        else : 
            signal = TFRtr[:,i_signal, : ].mean(0) # mean accros condition
        
        signal = signal - signal.mean(0)
        signal = smooth_moving_average(signal[100:600], window=80)
        try : 
            spe = pearsonr(signal_ref, signal).statistic # out (ch, ch)
            corr.append(spe)
        except :
            print('Failled TFRtr shape', TFRtr.shape, 'signal_', signal.shape )
            corr.append(np.nan)
            

    idx_corr_neg = np.where(np.array(corr)>= cor_thr)

    # flip 
    TFRtr_cor = TFRtr.copy()

    if len(TFRtr.shape) == 2 :
        TFRtr_cor[idx_corr_neg, :] = -1* TFRtr[idx_corr_neg, :]
    else : 
        TFRtr_cor[:, idx_corr_neg, :] = -1* TFRtr[:, idx_corr_neg, :]
    return TFRtr_cor

def smooth_moving_average(x, window):
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')
