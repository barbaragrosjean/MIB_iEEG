
import os
import mat73
from scipy.spatial import cKDTree
import pandas as pd
import numpy as np
import sys
sys.path.append("..")
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
import scipy
from nilearn import plotting

from src.decomposition import CompoThr
from src.setting import GetInfo
from src.setting import PROJECT_PATH
from src.config import COL_REG
import matplotlib.colors as mcolors

import nibabel as nib
from nilearn import datasets
from nilearn.image import load_img
import pandas as pd
import random 
import pickle 
import json
import matplotlib.pyplot as plt

def get_ieeg_data(ieeg_datapath,ieeg_subj_list, tr_avg=True) : 
    ieeg_data ={}
    for subj in ieeg_subj_list :
        info_file = ieeg_datapath + f'/{subj}_info.json'
        with open(info_file) as json_data:
            d = json.load(json_data)
            events_index = np.array([int(i) for i in d['event_id']])

        id_ev1 = np.where(events_index == 1)[0]
        id_ev2 = np.where(events_index == 2)[0]

        file = ieeg_datapath + f'/{subj}_epochs.p'
        with open(file, "rb") as f:
            epoch = pickle.load(f)
            ieeg_data[subj] = np.zeros((2, 24, epoch.shape[1], epoch.shape[2])) *np.nan
            ieeg_data[subj][0, :len(id_ev1), :, :]=epoch[id_ev1, :,:]
            ieeg_data[subj][1, :len(id_ev2), :, :]=epoch[id_ev2, :,:]

    # mean tr
    subj_data= []
    if tr_avg :

        for subj in ieeg_subj_list :
            subj_data.append(np.nanmean(ieeg_data[subj], axis=1))
            if len(np.unique(np.isnan(np.nanmean(ieeg_data[subj], axis=1)))) >1 :
                print(subj)
        ieeg_data_mean = np.concat(subj_data, axis=1)
        return ieeg_data_mean
    
    else : 
        for subj in ieeg_subj_list :
            subj_data.append(ieeg_data[subj])

        return subj_data # list of 30 subj trial level 

def get_meg_data(meg_outpath,meg_subj_list, type='source' ) :
    meg_data_source=[]
    for subj in meg_subj_list :
        file = meg_outpath + f'/{subj}_{type}.p' #meg_outpath + f'/{subj}_sensor.p'
        with open(file, "rb") as f:
            data = pickle.load(f)
        data = data[:2, :, :-1]

        # Normalize this subject
        mean = data.mean(axis=(0, 2), keepdims=True)
        std = data.std(axis=(0, 2), keepdims=True)
        data = (data - mean) / std

        meg_data_source.append(data)

    meg_data_source = np.stack(meg_data_source)
    
    return meg_data_source  #(32, 2, 3559, 1025)

def match_meg_(meg_data_source, ieeg_subj_list, coord, meg_subj_list, subj_list, return_mask=False) :
    matched_meg=[]
    mask = {}
    # random idx meg
    id_meg_subj=np.arange(0, len(ieeg_subj_list), 1)
    random.shuffle(id_meg_subj)

    for i, subj in enumerate(ieeg_subj_list):
        id_subj = np.where(np.array(subj_list) == subj)
        coord_subj = np.array(coord)[id_subj]

        pos = pd.read_csv(f'MEG/dataMEG/{meg_subj_list[id_meg_subj[i]]}_pos.csv').drop(columns='Unnamed: 0')
        tree = cKDTree(pos)
        _, nearest_idx = tree.query(coord_subj, k=1)
        to_keep = meg_data_source[i,:,nearest_idx,:]
        matched_meg.append(to_keep)
        mask[meg_subj_list[id_meg_subj[i]]] = nearest_idx

    matched_meg = np.concat(matched_meg, axis=0).transpose(1, 0, 2)
    if return_mask : 
        return mask
    
    return matched_meg

def match_meg(meg_data_source,ieeg_subj_list,coord,meg_subj_list,subj_list,return_mask=False,average_subject=False):
    matched_meg = []
    mask = {}
    id_meg_subj = np.arange(len(ieeg_subj_list))
    random.shuffle(id_meg_subj)
    for i, subj in enumerate(ieeg_subj_list):
        id_subj = np.where(np.array(subj_list) == subj)
        coord_subj = np.array(coord)[id_subj]

        if average_subject:
            matched_subj = []
            for j, meg_subj in enumerate(meg_subj_list):
                pos = pd.read_csv(f'MEG/dataMEG/{meg_subj}_pos.csv').drop(columns='Unnamed: 0')
                tree = cKDTree(pos)
                _, nearest_idx = tree.query(coord_subj, k=1)
                to_keep = meg_data_source[j, :, nearest_idx, :]
                matched_subj.append(to_keep)
                mask[meg_subj] = nearest_idx
            to_keep = np.mean(matched_subj, axis=0)

        else:
            meg_subj = meg_subj_list[id_meg_subj[i]]
            pos = pd.read_csv(f'MEG/dataMEG/{meg_subj}_pos.csv').drop(columns='Unnamed: 0')
            tree = cKDTree(pos)
            _, nearest_idx = tree.query(coord_subj, k=1)
            to_keep = meg_data_source[i, :, nearest_idx, :]
            mask[meg_subj] = nearest_idx

        matched_meg.append(to_keep)
    matched_meg = np.concatenate(matched_meg, axis=0).transpose(1, 0, 2)

    if return_mask:
        return mask

    return matched_meg

def mask_meg(data, coord, radius_1=5, radius_2=3, show=False, return_mask=False):

    file_path = 'MEG/MNI152_8mm_coord_dyi.mat'
    MEG_coords = scipy.io.loadmat(file_path)['MNI8']

    tree = cKDTree(np.array(coord)*1000)
    dist, _ = tree.query(MEG_coords)
    mask = dist <= radius_1
    meg_coords_masked = MEG_coords[mask]

    if show :
        plotting.plot_markers([1]*meg_coords_masked.shape[0], meg_coords_masked*1000) 
        plotting.plot_markers([1]*len(coord), np.array(coord)*1000)

    if return_mask : 
        return mask
    else : 
        return data[:,:, mask, :]

def gaussian_frequency_filter(X, fs, center_frequency, bandwidth):
    X = np.asarray(X, dtype=float)
    n_samples = X.shape[0]
    frequencies = np.fft.fftfreq(n_samples,d=1 / fs)
    sigma = bandwidth / (2 * np.sqrt(2 * np.log(2)))
    kernel = np.exp(-0.5 * ((np.abs(frequencies) - center_frequency)/ sigma)**2)
    X_fft = np.fft.fft(X,axis=0)
    X_fft_filtered = (X_fft* kernel[:, None])
    X_filtered = np.fft.ifft(X_fft_filtered,axis=0).real
    return X_filtered

def update_COL_REG(col_reg,all_regions, show=False):
    used_colors = {c.lower() for c in col_reg.values()}
    cmap = plt.get_cmap("tab20", max(20, len(all_regions)))

    color_idx = 0
    for region in all_regions:
        if region not in col_reg:
            while True:
                color = mcolors.to_hex(cmap(color_idx))
                color_idx += 1
                if color.lower() not in used_colors:
                    break

            col_reg[region] = color
            used_colors.add(color.lower())

    if show : 
        fig, ax = plt.subplots(figsize=(4, len(col_reg) * 0.4))

        for i, (region, color) in enumerate(col_reg.items()):
            ax.scatter(0, -i, s=200, color=color)
            ax.text(0.1, -i, region, va='center', fontsize=12)

        ax.set_xlim(-0.1, 1)
        ax.set_ylim(-len(col_reg), 1)
        ax.axis('off')

    return col_reg

def get_region_label(pos_m) : 
    # Load atlas
    atlas = datasets.fetch_atlas_aal()
    atlas_img = load_img(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine
    inv_affine = np.linalg.inv(affine)
    ijk = nib.affines.apply_affine(inv_affine, pos_m.values* 1000)
    ijk = np.round(ijk).astype(int)
    shape = atlas_data.shape
    inside = ((ijk[:,0] >= 0) & (ijk[:,0] < shape[0]) & (ijk[:,1] >= 0) & (ijk[:,1] < shape[1]) & (ijk[:,2] >= 0) & (ijk[:,2] < shape[2]))

    label_values = np.zeros(len(ijk), dtype=int)
    label_values[inside] = atlas_data[
        ijk[inside,0],
        ijk[inside,1],
        ijk[inside,2]
    ].astype(int)
    mapping = dict(zip(map(int, atlas.indices), atlas.labels))
    regions = [mapping.get(v, "Background") for v in label_values]

    aal_to_region = {
        # Background
        "Background": 'N',

        # Frontal
        "Frontal_Sup_L": "DLPFC",
        "Frontal_Sup_R": "DLPFC",
        "Frontal_Sup_Orb_L": "OFC",
        "Frontal_Sup_Orb_R": "OFC",
        "Frontal_Mid_L": "DLPFC",
        "Frontal_Mid_R": "DLPFC",
        "Frontal_Mid_Orb_L": "OFC",
        "Frontal_Mid_Orb_R": "OFC",
        "Frontal_Inf_Oper_L": "VLPFC",
        "Frontal_Inf_Oper_R": "VLPFC",
        "Frontal_Inf_Tri_L": "VLPFC",
        "Frontal_Inf_Tri_R": "VLPFC",
        "Frontal_Inf_Orb_L": "OFC",
        "Frontal_Inf_Orb_R": "OFC",
        "Rolandic_Oper_L": "VLPFC",   # closest to frontal operculum
        "Rolandic_Oper_R": "VLPFC",
        "Supp_Motor_Area_L": "premotor",
        "Supp_Motor_Area_R": "premotor",
        "Olfactory_L": "OFC",
        "Olfactory_R": "OFC",
        "Rectus_L": "OFC",
        "Rectus_R": "OFC",
        "Insula_L": "INS",
        "Insula_R": "INS",

        # Cingulate
        "Cingulum_Ant_L": "ACC",
        "Cingulum_Ant_R": "ACC",
        "Cingulum_Mid_L": "ACC",
        "Cingulum_Mid_R": "ACC",
        "Cingulum_Post_L": "PCC",
        "Cingulum_Post_R": "PCC",

        # Motor / sensory
        "Precentral_L": "M1",
        "Precentral_R": "M1",
        "Postcentral_L": "S1",
        "Postcentral_R": "S1",
        "Paracentral_Lobule_L": "M1",
        "Paracentral_Lobule_R": "M1",

        # Parietal
        "Parietal_Sup_L": "parietal",
        "Parietal_Sup_R": "parietal",
        "Parietal_Inf_L": "parietal",
        "Parietal_Inf_R": "parietal",
        "SupraMarginal_L": "parietal",
        "SupraMarginal_R": "parietal",
        "Angular_L": "parietal",
        "Angular_R": "parietal",
        "Precuneus_L": "PCC",
        "Precuneus_R": "PCC",

        # Temporal
        "Heschl_L": "A1",
        "Heschl_R": "A1",
        "Temporal_Sup_L": "STG",
        "Temporal_Sup_R": "STG",
        "Temporal_Mid_L": "MTG",
        "Temporal_Mid_R": "MTG",
        "Temporal_Inf_L": "VS",      # ventral temporal
        "Temporal_Inf_R": "VS",
        "Temporal_Pole_Sup_L": "TP",
        "Temporal_Pole_Sup_R": "TP",
        "Temporal_Pole_Mid_L": "TP",
        "Temporal_Pole_Mid_R": "TP",
        "Fusiform_L": "VS",
        "Fusiform_R": "VS",

        # Medial temporal
        "Hippocampus_L": "HPC",
        "Hippocampus_R": "HPC",
        "ParaHippocampal_L": "PHC",
        "ParaHippocampal_R": "PHC",
        "Amygdala_L": "AMY",
        "Amygdala_R": "AMY",

        # Occipital (no equivalent in REGION)
        "Calcarine_L": 'Occ',
        "Calcarine_R": 'Occ',
        "Cuneus_L": 'Occ',
        "Cuneus_R": 'Occ',
        "Lingual_L": 'Occ',
        "Lingual_R": 'Occ',
        "Occipital_Sup_L": 'Occ',
        "Occipital_Sup_R": 'Occ',
        "Occipital_Mid_L": 'Occ',
        "Occipital_Mid_R": 'Occ',
        "Occipital_Inf_L": 'Occ',
        "Occipital_Inf_R": 'Occ',

        # Deep nuclei
        "Caudate_L": 'Caud',
        "Caudate_R": 'Caud',
        "Putamen_L": 'Put',
        "Putamen_R": 'Put',
        "Pallidum_L": 'Pal',
        "Pallidum_R": 'Pal',
        "Thalamus_L": "THAL",
        "Thalamus_R": "THAL",

        # Cerebellum / vermis (not represented)
        "Cerebelum_Crus1_L": 'CERB',
        "Cerebelum_Crus1_R": 'CERB',
        "Cerebelum_Crus2_L": 'CERB',
        "Cerebelum_Crus2_R": 'CERB',
        "Cerebelum_3_L": 'CERB',
        "Cerebelum_3_R": 'CERB',
        "Cerebelum_4_5_L": 'CERB',
        "Cerebelum_4_5_R": 'CERB',
        "Cerebelum_6_L": 'CERB',
        "Cerebelum_6_R": 'CERB',
        "Cerebelum_7b_L": 'CERB',
        "Cerebelum_7b_R": 'CERB',
        "Cerebelum_8_L": 'CERB',
        "Cerebelum_8_R": 'CERB',
        "Cerebelum_9_L": 'CERB',
        "Cerebelum_9_R": 'CERB',
        "Cerebelum_10_L": 'CERB',
        "Cerebelum_10_R": 'CERB',
        "Vermis_1_2": 'CERB',
        "Vermis_3": 'CERB',
        "Vermis_4_5": 'CERB',
        "Vermis_6": 'CERB',
        "Vermis_7": 'CERB',
        "Vermis_8": 'CERB',
        "Vermis_9": 'CERB',

        "Frontal_Sup_Medial_L": "DLPFC",
        "Frontal_Sup_Medial_R": "DLPFC",

        "Frontal_Med_Orb_L": "OFC",
        "Frontal_Med_Orb_R": "OFC",
    }

    regions_meg = [aal_to_region.get(v) for v in regions]

    for i, r in enumerate(regions_meg) :
        if r == None : 
            print(regions[i])
    
    return regions_meg