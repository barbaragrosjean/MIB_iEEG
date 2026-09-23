import os
import numpy as np
import pandas as pd
import sys
import json
import pickle
from sklearn.cluster import KMeans

from utils import match_meg, get_meg_data, match_meg

sys.path.append("..")
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.setting import GetInfo
from src.setting import PROJECT_PATH

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

meg_data_source = get_meg_data(meg_outpath, meg_subj_list)

pca_source = PCA(10)
meg_trans = pca_source.fit_transform(np.concat([meg_data_source.mean(1)[i, :, :] for i in range(meg_data_source.shape[0])]).T) 

with open('out/trans_meg_concat.p', "wb") as f:
    pickle.dump(meg_trans, f)

with open('out/expl_var_meg_concat.p', "wb") as f:
    pickle.dump(pca_source.explained_variance_ratio_[:10], f)

w = pca_source.components_
df = pd.DataFrame()

for c in range(3):
    W = w[c, :].T.reshape(-1, 1)
    cluster_range = range(2, 11)
    silhouette_scores = []

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k,random_state=0,n_init=10)
        labels = kmeans.fit_predict(W)
        score = silhouette_score(W, labels)
        silhouette_scores.append(score)
    optimal_k = cluster_range[np.argmax(silhouette_scores)]

    final_model = KMeans(n_clusters=optimal_k,random_state=0,n_init=10)
    final_labels = final_model.fit_predict(W)
    df[f'meg_pc{c}'] = final_labels

df.to_csv('out/labels_Kmean_MEG_source_pc.csv')

df = pd.DataFrame()
pca_source.fit(meg_data_source.mean(0).mean(0).T) 
w = pca_source.components_

for c in range(3):
    W = w[c, :].T.reshape(-1, 1)
    cluster_range = range(2, 11)
    silhouette_scores = []

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k,random_state=0,n_init=10)
        labels = kmeans.fit_predict(W)
        score = silhouette_score(W, labels)
        silhouette_scores.append(score)
    optimal_k = cluster_range[np.argmax(silhouette_scores)]

    final_model = KMeans(n_clusters=optimal_k,random_state=0,n_init=10)
    final_labels = final_model.fit_predict(W)
    df[f'meg_pc{c}'] = final_labels

df.to_csv('out/labels_Kmean_MEG_source_avg_pc.csv')

df = pd.DataFrame()
data_matched = match_meg(meg_data_source, ieeg_subj_list, coord, meg_subj_list, subj_list)
pca_source.fit(data_matched.mean(0).T) 
w = pca_source.components_

for c in range(3):
    W = w[c, :].T.reshape(-1, 1)
    cluster_range = range(2, 11)
    silhouette_scores = []

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k,random_state=0,n_init=10)
        labels = kmeans.fit_predict(W)
        score = silhouette_score(W, labels)
        silhouette_scores.append(score)
    optimal_k = cluster_range[np.argmax(silhouette_scores)]

    final_model = KMeans(n_clusters=optimal_k,random_state=0,n_init=10)
    final_labels = final_model.fit_predict(W)
    df[f'meg_pc{c}'] = final_labels

df.to_csv('out/labels_Kmean_MEG_match_pc.csv')

df = pd.DataFrame()
data_matched_average = match_meg(meg_data_source, ieeg_subj_list, coord, meg_subj_list, subj_list, average_subject=True)
pca_source.fit(data_matched_average.mean(0).T) 

w = pca_source.components_

for c in range(3):
    W = w[c, :].T.reshape(-1, 1)
    cluster_range = range(2, 11)
    silhouette_scores = []

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k,random_state=0,n_init=10)
        labels = kmeans.fit_predict(W)
        score = silhouette_score(W, labels)
        silhouette_scores.append(score)
    optimal_k = cluster_range[np.argmax(silhouette_scores)]

    final_model = KMeans(n_clusters=optimal_k,random_state=0,n_init=10)
    final_labels = final_model.fit_predict(W)
    df[f'meg_pc{c}'] = final_labels

df.to_csv('out/labels_Kmean_MEG_match_avg_pc.csv')
