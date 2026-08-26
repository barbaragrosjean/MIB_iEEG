
import os
import numpy as np
import pandas as pd
import sys
import json
import pickle
sys.path.append("..")
from src.decomposition import CompoThr
from src.setting import GetInfo, PROJECT_PATH
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSSVD , PLSRegression


ieeg_datapath = 'ieeg_shortWOBS_fs250'
meg_datapath = '/projects/MINDLAB2025_MEG-Auditory_Cognitive_Maps/scratch/APR2020_Block3_SingleTrial_BarbaraNikita'
meg_outpath ='MEG/dataMEG'

os.path.isdir(meg_datapath)
meg_subj_list = [file.replace('_sensor.p', '') for file in os.listdir(meg_outpath) if '_sensor' in file]
print('MEG Subject number : ', len(meg_subj_list))

ieeg_subj_list = [file.replace('_epochs.p', '') for file in os.listdir(ieeg_datapath) if file[-len('epochs.p'):] == 'epochs.p']
print('iEEG Subject number : ', len(ieeg_subj_list))

info_file = ieeg_datapath + f'/{ieeg_subj_list[0]}_info.json'
with open(info_file) as json_data:
    d = json.load(json_data)
    time_ieeg = d['time_epoch']

project_path = '../' + PROJECT_PATH
coord, areas, elect_list, subj_list, regions_ieeg = GetInfo(ieeg_subj_list, data_path=ieeg_datapath, project_path=project_path)
coord= np.array(coord)
coord = np.where(abs(coord) >100, coord/1000, coord)
coord = np.where(abs(coord) >100, coord/1000, coord)
coord = coord/1000

from utils import match_meg, mask_meg, get_meg_data, get_ieeg_data
r=5

ieeg_data_mean = get_ieeg_data(ieeg_datapath, ieeg_subj_list)*1000 # Stupid parameter in the preproc TO REMOVE 
ieeg_data= ieeg_data_mean.mean(0)
meg_data_source = get_meg_data(meg_outpath, meg_subj_list)
data_matched = match_meg(meg_data_source, ieeg_subj_list, coord, meg_subj_list, subj_list, average_subject=False).mean(0)
#meg_data = np.concat([meg_data_source[s, :, :,:] for s in range(meg_data_source.shape[0])], axis=1).mean(0)
meg_data = data_matched

# scaler  --> NO
#zero = np.argmin(abs(np.array(time_ieeg)))
#mu = np.mean(ieeg_data[:, :zero], axis=1 , keepdims=True)
#sigma = np.std(ieeg_data[:, :zero], axis=1, keepdims=True)
#ieeg_scaled = (ieeg_data - mu) /sigma
#mu = np.mean(data_matched[:, :zero], axis=1 , keepdims=True)
#sigma = np.std(data_matched[:, :zero], axis=1, keepdims=True)
#meg_scaled = (data_matched-mu)/sigma

ieeg_scaled = ieeg_data 
meg_scaled = meg_data

total_var_X = np.sum(ieeg_scaled.T**2)
total_var_Y = np.sum(meg_scaled.T**2)


# PLS SVD
name = 'PLSSVD'
scale_ = True
cca = PLSSVD(n_components=10, scale=scale_)
X_c, Y_c = cca.fit_transform(ieeg_scaled.T, meg_scaled.T)

T = ieeg_scaled.T @ cca.x_weights_    
U = meg_scaled.T @ cca.y_weights_    

P_X = np.linalg.lstsq(T, ieeg_scaled.T, rcond=None)[0]
P_Y = np.linalg.lstsq(U, meg_scaled.T, rcond=None)[0]

explained_X = []
explained_Y = []

for k in range(1, 11):

    X_hat = T[:, :k] @ P_X[:k, :]
    Y_hat = U[:, :k] @ P_Y[:k, :]

    r2_X = 1 - np.sum((ieeg_scaled.T - X_hat)**2) / total_var_X
    r2_Y = 1 - np.sum((meg_scaled.T - Y_hat)**2) / total_var_Y

    explained_X.append(r2_X)
    explained_Y.append(r2_Y)

df = pd.DataFrame()
df['correlation'] = [np.corrcoef(X_c[:, i],Y_c[:, i])[0, 1] for i in range(10)]
df['explained_X'] = explained_X
df['explained_Y'] = explained_Y

outs=f'out/cross_models_scale_{str(scale_)}_match'
if not os.path.exists(outs):
    os.makedirs(outs)

df.to_csv(f'{outs}/{name}_expl_var.csv')

with open(f'{outs}/{name}_model.p', "wb") as f:
    pickle.dump(cca, f)

with open(f'{outs}/{name}_X.p', "wb") as f:
    pickle.dump(X_c, f)

with open(f'{outs}/{name}_Y.p', "wb") as f:
    pickle.dump(Y_c, f)


# scale to mimic the function 
if scale_ :
    ieeg_scaled = (ieeg_scaled - ieeg_scaled.mean(axis=1, keepdims=True))/ieeg_scaled.std(axis=1, keepdims=True)
    meg_scaled = (meg_scaled - meg_scaled.mean(axis=1, keepdims=True))/meg_scaled.std(axis=1, keepdims=True)

C = ieeg_scaled @ meg_scaled.T
Ux, s, UyT = np.linalg.svd(C, full_matrices=False)

name = 'PLSSVD_handmade'
with open(f'{outs}/{name}_Ux.p', "wb") as f:
    pickle.dump(Ux, f)

with open(f'{outs}/{name}_UyT.p', "wb") as f:
    pickle.dump(UyT, f)

T = ieeg_scaled.T @ Ux    
U = meg_scaled.T @ UyT.T  

P_X = np.linalg.lstsq(T, ieeg_scaled.T, rcond=None)[0]
P_Y = np.linalg.lstsq(U, meg_scaled.T, rcond=None)[0]

explained_X = []
explained_Y = []

for k in range(1, 11):

    X_hat = T[:, :k] @ P_X[:k, :]
    Y_hat = U[:, :k] @ P_Y[:k, :]

    r2_X = 1 - np.sum((ieeg_scaled.T - X_hat)**2) / total_var_X
    r2_Y = 1 - np.sum((meg_scaled.T - Y_hat)**2) / total_var_Y

    explained_X.append(r2_X)
    explained_Y.append(r2_Y)

df = pd.DataFrame()
corr = [np.corrcoef(T[:, i],U[:, i])[0, 1] for i in range(10)]
df['explained_X'] = explained_X
df['explained_Y'] = explained_Y
df['cum_expl_var'] = s[:10]**2 / np.sum(s**2)
df['correlation'] = [np.corrcoef(X_c[:, i],Y_c[:, i])[0, 1] for i in range(10)]

df.to_csv(f'{outs}/{name}_expl_var.csv')