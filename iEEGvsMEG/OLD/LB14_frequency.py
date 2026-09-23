import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import json
import sys
sys.path.append("..")
from src.setting import GetInfo, PROJECT_PATH
from utils import get_ieeg_data, gaussian_frequency_filter, match_meg
from sklearn.decomposition import PCA


ieeg_datapath = 'ieeg_shortWOBS_fs400'
meg_datapath = '/projects/MINDLAB2025_MEG-Auditory_Cognitive_Maps/scratch/APR2020_Block3_SingleTrial_BarbaraNikita'
meg_outpath ='MEG/dataMEG'

meg_subj_list = [file.replace('_sensor.p', '') for file in os.listdir(meg_outpath) if '_sensor' in file]
print('MEG Subject number : ', len(meg_subj_list))

ieeg_subj_list = [file.replace('_epochs.p', '') for file in os.listdir(ieeg_datapath) if file[-len('epochs.p'):] == 'epochs.p']
print('iEEG Subject number : ', len(ieeg_subj_list))

compute_ieeg = False
# Get the coordinate
project_path = '../' + PROJECT_PATH
coord, areas, elect_list, subj_list, regions_ieeg = GetInfo(ieeg_subj_list, data_path=ieeg_datapath, project_path=project_path)
coord=np.array(coord)
coord = np.where(abs(coord) >100, coord/1000, coord)
coord = np.where(abs(coord) >100, coord/1000, coord)
coord = coord/1000

# Define frequencies
frequencies_common = np.geomspace(0.5,100,20)
ratio = frequencies_common[1] / frequencies_common[0]
n_ieeg = int(np.floor(np.log(180 / 0.5) / np.log(ratio))) + 1
frequencies = 0.5 * ratio ** np.arange(n_ieeg)
frequencies = frequencies[frequencies <= 180]

# MEG 
type_ = 'source_avg'
out_path='out/freq' 
nb_compo = 1026
eigenvalue_spectrum = np.zeros((len(frequencies_common), nb_compo))
explained_variance_ratio = np.zeros((len(frequencies_common), nb_compo))
X_trans = None 
total_power = None

for i, frequency in enumerate(frequencies_common):
    # meg data 
    meg_freq = []
    for subj in meg_subj_list : 
        with open(meg_outpath +f'/{subj}_source_freq.p', "rb") as f:
            m=pickle.load(f)[:, :, i, :].astype(np.float32, copy=False) # is (2, 3559, 20, 1026) only freq i 
        meg_freq.append(m)
    meg_freq = np.stack(meg_freq, axis=0) # is subj, condi, ch, time
    # Match subject electrod with ieeg
    # should take something like (32, 2, 3559, 1025) and output (2576, 1025)
    if type_ == 'match' : 
        match_data = match_meg(meg_freq[:, :,:, :],ieeg_subj_list,coord,meg_subj_list,subj_list,return_mask=False,average_subject=False).mean(0)
    elif type_ == 'match_avg' : 
        match_data = match_meg(meg_freq[:, :,:, :],ieeg_subj_list,coord,meg_subj_list,subj_list,return_mask=False,average_subject=True).mean(0)
    elif type_ == 'source_avg' : 
        match_data = meg_freq.mean(0).mean(0)

    # power 
    if total_power is None:
        total_power = np.zeros((len(frequencies_common),match_data.shape[1]),dtype=np.float32)
    total_power[i, :] = np.mean(match_data ** 2,axis=0) #(frequency, time)

    pca = PCA(nb_compo)
    if X_trans is None:
        X_trans = np.zeros((len(frequencies_common), match_data.shape[1],5 ),dtype=np.float32) # freq, time, compo
    X_trans[i, :, :5] =  pca.fit_transform(match_data.T)[:, :5]
    eigenvalue_spectrum[i, :] = pca.explained_variance_
    explained_variance_ratio[i, :] = pca.explained_variance_ratio_
    del match_data
    del meg_freq

if not os.path.exists(out_path):
    os.makedirs(out_path)
with open(out_path + f'/{type_}_power_{nb_compo}.p', "wb") as f:
    pickle.dump(total_power, f)
with open(out_path + f'/{type_}_trans_{nb_compo}.p', "wb") as f:
    pickle.dump(X_trans, f)
with open(out_path + f'/{type_}_expl_var_r_{nb_compo}.p', "wb") as f:
    pickle.dump(explained_variance_ratio, f)
with open(out_path + f'/{type_}_expl_var_{nb_compo}.p', "wb") as f:
    pickle.dump(eigenvalue_spectrum, f)

# iEEG 
if compute_ieeg : 
    out_path='out/freq' 
    nb_compo = 1026 # eventually change this 
    eigenvalue_spectrum = np.zeros((len(frequencies), nb_compo))
    explained_variance_ratio = np.zeros((len(frequencies), nb_compo))
    X_trans = None 
    total_power = None


    for i, frequency in enumerate(frequencies):
        ieeg_list = []
        for subj in ieeg_subj_list : 
            with open(ieeg_datapath + f'/{subj}_ieeg_freq.p', "rb") as f:
                ieeg_freq=pickle.load(f)[:, :, i, :].astype(np.float32, copy=False) # (condi, ch, freq, time) only i freq
            ieeg_list.append(ieeg_freq)
        ieeg_freq = np.concat([m for m in ieeg_list], axis=1).mean(0) # ch_all, time

        # power 
        if total_power is None:
            total_power = np.zeros((len(frequencies),ieeg_freq.shape[1]),dtype=np.float32)
        total_power[i, :] = np.mean(ieeg_freq ** 2,axis=0)

        pca = PCA(nb_compo)
        if X_trans is None:
            X_trans = np.zeros((len(frequencies), ieeg_freq.shape[1],5 ),dtype=np.float32) # freq, time, compo
        X_trans[i, :, :5] =  pca.fit_transform(ieeg_freq.T)[:, :5]
        pca.fit(ieeg_freq.T)
        eigenvalue_spectrum[i, :] = pca.explained_variance_
        explained_variance_ratio[i, :] = pca.explained_variance_ratio_

    with open(out_path + f'/ieeg_power_{nb_compo}.p', "wb") as f:
        pickle.dump(total_power, f)
    with open(out_path + f'/ieeg_trans_{nb_compo}.p', "wb") as f:
        pickle.dump(X_trans, f)
    with open(out_path + f'/ieeg_expl_var_r_{nb_compo}.p', "wb") as f:
        pickle.dump(explained_variance_ratio, f)
    with open(out_path + f'/ieeg_expl_var_{nb_compo}.p', "wb") as f:
        pickle.dump(eigenvalue_spectrum, f)


