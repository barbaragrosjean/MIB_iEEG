import os
import mat73
import pandas as pd
import numpy as np
import sys
import json
import pickle
sys.path.append("..")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
import scipy
from nilearn import plotting
from src.decomposition import CompoThr
from src.setting import GetInfo, PROJECT_PATH
from src.config import COL_REG, col_pc

from sklearn.cross_decomposition import CCA
from scipy.linalg import subspace_angles

import nibabel as nib
from nilearn import datasets
from nilearn.image import load_img
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE

from utils import get_region_label, mask_meg, match_meg, update_COL_REG, get_meg_data
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
coord=np.array(coord)
coord = np.where(abs(coord) >100, coord/1000, coord)
coord = np.where(abs(coord) >100, coord/1000, coord)
coord = coord/1000

from utils import match_meg, mask_meg, get_meg_data, get_ieeg_data
r=5

ieeg_data_mean = get_ieeg_data(ieeg_datapath, ieeg_subj_list)*1000 # Stupid parameter in the preproc TO REMOVE 
ieeg_data= ieeg_data_mean.mean(0)
meg_data_source = get_meg_data(meg_outpath, meg_subj_list)

# Concat and match
#data_matched = match_meg(meg_data_source, ieeg_subj_list, coord, meg_subj_list, subj_list).mean(0)

# Avg and match
#data_matched = match_meg(meg_data_source, ieeg_subj_list, coord, meg_subj_list, subj_list, average_subject=True).mean(0)

# Concat and source
meg_data = np.concat([meg_data_source[s, :, :,:] for s in range(meg_data_source.shape[0])], axis=1).mean(0)

# Avg and source
#meg_data = meg_data_source.mean(0).mean(0)

X_joint = np.concatenate([ieeg_data, meg_data],axis=0).T

# Joint manifold
n_compo= 3
algorithms = {
    'Isomap': Isomap(n_components=n_compo), 
    'LLE' : LocallyLinearEmbedding(n_components=n_compo),
    'MDS' : MDS(n_components=n_compo),
    'SpectralEmbedding' : SpectralEmbedding(n_components=n_compo), 
    'TSNE' : TSNE(n_components=n_compo)
}

Z_joint = {}
for algo_name, algo in algorithms.items() : 
    Z_joint[algo_name] = algo.fit_transform(X_joint)

Z_joint_space = {}
for algo_name, algo in algorithms.items() : 
    try : 
        Z_joint_space[algo_name] = algo.fit_transform(X_joint.T)
    except:
        continue

out_path = 'out/manifolds/manifolds_source'
if not os.path.exists(out_path):
    os.makedirs(out_path)

with open(out_path + '/Z_joint_time.p', "wb") as f:
    pickle.dump(Z_joint, f)

with open(out_path + '/Z_joint_space.p', "wb") as f:
    pickle.dump(Z_joint_space, f)   