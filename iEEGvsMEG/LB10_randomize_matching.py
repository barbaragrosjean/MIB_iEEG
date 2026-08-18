import os
import numpy as np
import sys
import json
import pickle

from utils import match_meg, get_meg_data

sys.path.append("..")
from sklearn.decomposition import PCA
from src.setting import GetInfo
from src.setting import PROJECT_PATH

ieeg_datapath = 'ieeg_shortWOBS_fs250'
meg_datapath = '/projects/MINDLAB2025_MEG-Auditory_Cognitive_Maps/scratch/APR2020_Block3_SingleTrial_BarbaraNikita'
meg_outpath ='MEG/dataMEG'

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

meg_data_source = get_meg_data(meg_outpath, meg_subj_list)
pca_matched = PCA(5)

trans =[]
compo = []
runs=50
for r in range(runs) : 
    data_matched = match_meg(meg_data_source, ieeg_subj_list, coord, meg_subj_list, subj_list)
    meg_match_trans = pca_matched.fit_transform(data_matched.mean(0).T) 

    trans.append(meg_match_trans)
    compo.append(pca_matched.components_)

trans = np.stack(trans) # run, time, compo
compo = np.stack(compo) #run, compo, channels 

out_path =  'out'
if not os.path.exists(out_path) : 
    os.makedirs(out_path)

with open(out_path + f"/meg_matched_trans_{runs}.pkl", "wb") as f:
    pickle.dump(trans, f)

with open(out_path + f"/meg_matched_compo_{runs}.pkl.pkl", "wb") as f:
    pickle.dump(compo, f)