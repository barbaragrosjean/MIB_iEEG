

import os
import numpy as np
import json
import random
import pickle
from scipy.stats import pearsonr

from sklearn.decomposition import PCA

from src.decomposition import DataAugmentation, ConcatPCA
from src.preproc import TFRmEvents, BbEvents
from src.setting import get_data_grp
from src.config import FREQ_BAND, OUT_PATH


dataset_label = 'mf70-160'
data_path = OUT_PATH + '/Data_longWOBS_' + dataset_label

freq ='high_gamma'
method_pca = 'concat'
nb_compo = 5
freq_band = FREQ_BAND
data_aug_method = 'duplicat'

subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
out_path = OUT_PATH + '/grpPCA_' + dataset_label+ '/supsubj_' + method_pca

def Compute_weights_LOOtr(nb_compo, freq, subj_included) :
    TFRm_list = []

    for subj in subj_included : 
        info_file = data_path + f'/{subj}_info.json'
        with open(info_file) as f:
            info = json.load(f)
            events_index = np.array([int(i) for i in info['event_id']])

        id_ev1 = np.where(events_index == 1)[0]
        id_ev2 = np.where(events_index == 2)[0]

        id_test= [random.sample(list(id_ev1),1), random.sample(list(id_ev2),1)]
        id_ev1 = list(id_ev1)
        id_ev1.remove(id_test[0])
        id_ev1 = np.array(id_ev1)
        id_ev2 = list(id_ev2)
        id_ev2.remove(id_test[1])
        id_ev2 = np.array(id_ev2)

        if freq == 'broadband' :
            TFRm = BbEvents(subj, test_id = id_test, events_index=events_index, data_path=data_path)
             
        else : 
            freq_id = freq_band.index(freq)
            TFRm = TFRmEvents(subj, test_id = id_test, freq_id = freq_id, events_index=events_index, data_path=data_path)

        if method_pca == 'concat' :
            TFRm_list.append(np.concatenate([TFRm[i, :,:] for i in [0, 1]], axis = 1))
        if method_pca == 'mean' : 
            TFRm_list.append(np.mean(TFRm[[0, 1], :,:], axis = 0))

    concat_all = np.concatenate(TFRm_list, axis = 0)
    del TFRm_list

    pca = PCA(nb_compo)
    pca.fit(concat_all.T)

    return pca.components_

def Compute_weights(nb_compo, freq, subj_included) :
    TFRm_list = []

    for subj in subj_included : 
        info_file = data_path + f'/{subj}_info.json'
        with open(info_file) as f:
            info = json.load(f)
            events_index = np.array([int(i) for i in info['event_id']])

        id_ev1 = np.where(events_index == 1)[0]
        id_ev2 = np.where(events_index == 2)[0]

        if freq == 'broadband' :
            TFRm = BbEvents(subj, test_id = False, events_index=events_index, data_path=data_path)
             
        else : 
            freq_id = freq_band.index(freq)
            TFRm = TFRmEvents(subj, test_id = False, freq_id = freq_id, events_index=events_index, data_path=data_path)

        if method_pca == 'concat' :
            TFRm_list.append(np.concatenate([TFRm[i, :,:] for i in [0, 1]], axis = 1))
        if method_pca == 'mean' : 
            TFRm_list.append(np.mean(TFRm[[0, 1], :,:], axis = 0))

    concat_all = np.concatenate(TFRm_list, axis = 0)

    del TFRm_list

    pca = PCA(nb_compo)
    pca.fit(concat_all.T)

    return pca.components_

def subject_influence(W_SUBJ, W_ALL):
    """
    W_SUBJ:
        shape = (n_subject_removals, n_iterations, n_components, n_channels)

    W_ALL:
        shape = (n_iterations, n_components, n_channels)

    Returns
    -------
    loo_stability:
        shape = (n_subjects, n_components)
        Mean correlation between iterations within each subject removal.

    loo_vs_all:
        shape = (n_subjects, n_components)
        Correlation between the mean LOO weights and the mean full-subject
        weights.

    """
    n_subjects = len(W_SUBJ)
    n_iter, n_compo, n_channels = W_SUBJ[0].shape

    mean_W_ALL = np.mean(W_ALL, axis=0)
    mean_W_SUBJ = [np.mean(w_subj, axis=0) for w_subj in W_SUBJ]

    loo_stability = np.zeros((n_subjects, n_compo))
    loo_vs_all = np.zeros((n_subjects, n_compo))

    for s in range(n_subjects):
        for c in range(n_compo):
            W_sc = W_SUBJ[s][:, c, :]
            corr = np.corrcoef(W_sc)
            mask = ~np.eye(n_iter, dtype=bool)
            loo_stability[s, c] = np.mean(corr[mask])

    return loo_stability

def mean_component_correlation(W):
    n_iter, n_compo, n_channels = W.shape

    mean_corr = np.zeros(n_compo)
    corr_matrices = []

    for c in range(n_compo):
        W_c = W[:, c, :]
        corr = np.corrcoef(np.abs(W_c))
        corr_matrices.append(corr)
        mask = ~np.eye(n_iter, dtype=bool)
        mean_corr[c] = np.mean(corr[mask])

    return mean_corr, np.array(corr_matrices)

W_ALL = []
W_SUBJ = []
W_TR = []
subj_rm = []

for iter in range(100) : 
    w_all = Compute_weights(nb_compo, freq, subj_included) # variability in terms of the association of the suject together
    W_ALL.append(w_all)
W_ALL = np.concat([w[None, :, :] for w in W_ALL], axis=0)
mean_corr_all, corr_all = mean_component_correlation(W_ALL)
with open(out_path + '/mean_corr_all.p', "wb") as f:
    pickle.dump(mean_corr_all, f)

for iter in range(100) : 
    # Exclude a trial
    w_tr = Compute_weights_LOOtr(nb_compo, freq, subj_included)
    W_TR.append(w_tr)
W_TR = np.concat([w[None, :, :] for w in W_TR], axis=0)
mean_corr_tr, corr_tr = mean_component_correlation(W_TR)
with open(out_path + '/mean_corr_tr.p', "wb") as f:
    pickle.dump(mean_corr_tr, f)

TRANS =[]
data_grp = get_data_grp(subj_included=subj_included, type_data='tfr', data_path=data_path)
data_grp = np.concat([data_grp[i, :, -1, :] for i in [0, 1]], axis=-1)
big_trans=data_grp.T @ W_ALL.mean(0).T #n compo x n channnel

for subj in subj_included:
    w_subj =[]
    transform=[]
    for iter in range(50):
        subj_included_ = subj_included.copy()
        subj_included_.remove(subj)
        w_subj.append(Compute_weights(nb_compo,freq,subj_included_))

        # get the TS
        data_grp = get_data_grp(subj_included=subj_included_, type_data='tfr', data_path=data_path)
        data_grp = np.concat([data_grp[i, :, -1, :] for i in [0, 1]], axis=-1)
        pca = PCA(5)
        transform.append(pca.fit_transform(data_grp.T)) # compo x time

    W_SUBJ.append(np.concat([w[None, :, :] for w in w_subj], axis =0)) 
    subj_rm.append(subj)
    TRANS.append(np.concat([t[None, :, :] for t in transform], axis =0)) 

corr_trans = np.zeros(len(subj_included), 50, 5)
for s in range(len(subj_included)):
    for i in range(50):
        for c in range(5):
            corr_trans[s, i, c] = np.corrcoef(big_trans[:, c],TRANS[s][i, c, :])[0, 1]
corr_trans_mean =corr_trans.mean(1)
loo_stability = subject_influence(W_SUBJ, W_ALL)

with open(out_path + '/W_subj_stability.p', "wb") as f:
    pickle.dump(loo_stability, f)

with open(out_path + '/corr_trans_subj.p', "wb") as f:
    pickle.dump(corr_trans_mean, f)



















    