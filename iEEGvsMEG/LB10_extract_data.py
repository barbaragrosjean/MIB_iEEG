import os
import mat73
import numpy as np
import sys
import pandas as pd
import pickle

sys.path.append("..")

from src.config import OUT_PATH
from src.preproc import preproc
from src.setting import ExcludSubj


## MEG
meg_datapath = '/projects/MINDLAB2025_MEG-Auditory_Cognitive_Maps/scratch/APR2020_Block3_SingleTrial_BarbaraNikita'
meg_outpath = '/users/barbara/Desktop/MIB_iEEG/iEEGvsMEG/MEG/dataMEG'

meg_subj_list = [file.replace('_norm0_abs_0.mat', '') for file in os.listdir(meg_datapath) if 'SUBJ' in file]
nb_subj_meg = len(meg_subj_list)
if not os.path.exists(meg_outpath) :
    os.makedirs(meg_outpath)
    
for subj in meg_subj_list : 
    subj_mat =  mat73.loadmat(meg_datapath + '/' + subj + '_norm0_abs_0.mat')
    MEG_sensor = np.zeros((2, 102, 1026))*np.nan
    MEG_bss =np.zeros((2, 3559, 1026))*np.nan
    for ic, condi in enumerate([0, 1]) :
        d = subj_mat['OUT']['data_MEG_sensors'][condi]
        MEG_sensor[ic, :,:]= d.mean(-1) #2, 102, 1026, 27 average trials
        d = subj_mat['OUT']['sources_ERFs'][condi][0]
        MEG_bss[ic, :, :] = d.mean(-1)  #2, 3559, 1026, 27 average trials
    pos_source= subj_mat['OUT']['pos_brainsources_MNI8'] #(3559, 3)
    time_meg = subj_mat['OUT']['time'] # 1026

    # save in MEG
    with open(meg_outpath + f'/{subj}_sensor.p', "wb") as f:
        pickle.dump(MEG_sensor, f)

    with open(meg_outpath + f'/{subj}_source.p', "wb") as f:
        pickle.dump(MEG_bss, f)
    
    pd.DataFrame(pos_source).to_csv(meg_outpath + f'/{subj}_pos.csv')

    del subj_mat


# IEEG 
subj_included = [file.replace('_epochs.p', '') for file in os.listdir('../' + OUT_PATH + '/Data_longWOBS') if file[-len('epochs.p'):] == 'epochs.p']
subj_included = ExcludSubj(subj_included=subj_included, data_path= '../' + OUT_PATH + '/Data_longWOBS')

failled =['COG023', 'LL10', 'CP41', 'LL23', 'CP40', 'COG022', 'LL30', 'LL15']

for s in failled : 
    if s in subj_included :
        subj_included.remove(s)
path_to_save =  '/ieeg_shortWOBS_fs250'

if not os.path.exists(path_to_save) :
    os.makedirs(path_to_save)

for subj in subj_included : 
    if not os.path.exists(path_to_save + f'/{subj}_epochs.p'):
        print(subj)
        preproc(subj, new_sfreq=250, trials=True, save_epoch=True, compute_TFR=False, out_path=path_to_save)
    

