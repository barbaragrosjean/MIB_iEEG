import os
import mat73
import numpy as np
import sys
import pandas as pd
import pickle
import json 
sys.path.append("..")

from src.config import OUT_PATH, PROJECT_PATH
from src.preproc import preproc
from src.setting import ExcludSubj
from utils import gaussian_frequency_filter

def get_meg_from_raw(type_ = 'sensor', freq=False, fs=250) :
    meg_datapath = '/projects/MINDLAB2025_MEG-Auditory_Cognitive_Maps/scratch/APR2020_Block3_SingleTrial_BarbaraNikita'
    meg_outpath = '/users/barbara/Desktop/MIB_iEEG/iEEGvsMEG/MEG/dataMEG'
    meg_subj_list = [file.replace('_norm0_abs_0.mat', '') for file in os.listdir(meg_datapath) if 'SUBJ' in file]
    nb_subj_meg = len(meg_subj_list)

    if not os.path.exists(meg_outpath) :
        os.makedirs(meg_outpath)
    
    f_min=0.5
    f_max= 100
    if f_max >=fs/2:
        print(f'f_max={f_max} > Nyquist freq/2 : {fs/2}')
        return 0
    
    n_frequencies=20
    frequencies = np.geomspace(f_min,f_max,n_frequencies)

    if type_ == 'sensor':
        ch =102
    if type_=='source' : 
        ch =3559 

    if freq:
        l='_freq'
    else : 
        l =''

    for subj in meg_subj_list : 
        if not os.path.exists(meg_outpath + f'/{subj}_{type_}{l}.p'):

            if freq :
                MEG = np.zeros((2, ch, len(frequencies), 1026))*np.nan
            else :
                MEG = np.zeros((2, ch, 1026))*np.nan

            print('subj ', subj)
            subj_mat =  mat73.loadmat(meg_datapath + '/' + subj + '_norm0_abs_0.mat')
            for ic, condi in enumerate([0, 1]) :
                if type_ == 'sensor': d = subj_mat['OUT']['data_MEG_sensors'][condi]
                if type_=='source' : d = subj_mat['OUT']['sources_ERFs'][condi][0]

                if freq :
                    # Frequency decomposition at trial level of each condition 
                    for ifreq, freq in enumerate(frequencies) : 
                        bandwidth = max(0.5,0.20 * freq)
                        print('freq', freq)
                        filtered_d_tr =np.zeros((ch, 1026, d.shape[-1]))
                        for itr in range(d.shape[-1]) : # over the trials
                            filtered_d_tr[:, :, itr] = gaussian_frequency_filter(d[:, :, itr], fs, freq, bandwidth) # ch, time,
                        MEG[ic, :, ifreq, :] = filtered_d_tr.mean(-1)

                else :
                    MEG[ic, :,:]= d.mean(-1) #2, 102, 1026, 27 or #2, 3559, 1026, 27 average trials
                del d
            pos_source= subj_mat['OUT']['pos_brainsources_MNI8'] #(3559, 3)
            time_meg = subj_mat['OUT']['time'] # 1026
            del subj_mat

            # save in MEG
            with open(meg_outpath + f'/{subj}_{type_}{l}.p', "wb") as f:
                pickle.dump(MEG, f)

            pd.DataFrame(pos_source).to_csv(meg_outpath + f'/{subj}_pos.csv')
            

def get_ieeg_from_raw(fs=400, freq=False) :
    subj_included = [file.replace('_epochs.p', '') for file in os.listdir('../' + OUT_PATH + '/Data_longWOBS_mf70-160') if file[-len('epochs.p'):] == 'epochs.p']
    #subj_included = [file.replace('sub-', '') for file in os.listdir(PROJECT_PATH + '/data/BIDS') if file[:4] == 'sub-']
    subj_included = ExcludSubj(subj_included=subj_included, data_path= '../' + OUT_PATH + '/Data_longWOBS_mf70-160')
    failled =['COG023', 'LL10', 'CP41', 'LL23', 'CP40', 'COG022', 'LL30', 'LL15']
   
    f_min=0.5
    f_max= 180
    frequencies_common = np.geomspace(f_min,100,20)

    if f_max >=fs/2:
        print(f'f_max={f_max} > Nyquist freq/2 : {fs/2}')
        return 0
    
    ratio = frequencies_common[1] / frequencies_common[0]
    n_ieeg = int(np.floor(np.log(f_max / f_min) / np.log(ratio))) + 1
    frequencies = f_min * ratio ** np.arange(n_ieeg)
    frequencies = frequencies[frequencies <= f_max]
   
    for s in failled : 
        if s in subj_included :
            subj_included.remove(s)

    path_to_save = f'ieeg_shortWOBS_fs{fs}'
    if not os.path.exists(path_to_save) :
        os.makedirs(path_to_save)

    if freq:l='_freq'
    else : l =''

    for subj in subj_included : 
        # re compute the preproc if missing
        if not os.path.exists(path_to_save + f'/{subj}_epochs.p'):
            print(subj)
            preproc(subj, new_sfreq=fs, trials=True, save_epoch=True, compute_TFR=False, out_path=path_to_save)

        if not os.path.exists(path_to_save + f'/{subj}_ieeg_{l}.p'):
            # get those data and freq decomposition
            info_file = path_to_save + f'/{subj}_info.json'
            with open(info_file) as json_data:
                d = json.load(json_data)
                events_index = np.array([int(i) for i in d['event_id']])

            id_ev1 = np.where(events_index == 1)[0]
            id_ev2 = np.where(events_index == 2)[0]

            file = path_to_save + f'/{subj}_epochs.p'
            with open(file, "rb") as f:
                epoch = pickle.load(f)

            if freq :
                ieeg_data_subj = np.zeros((2, epoch.shape[1], len(frequencies), epoch.shape[2])) *np.nan
                for ic, id_ev in enumerate([id_ev1,id_ev2 ]) : 
                    d = epoch[id_ev, :,:]  # tial, channel, time for one condition
                    filtered_d = np.zeros((len(id_ev), d.shape[1], len(frequencies), d.shape[-1]))*np.nan
                    for ifreq, freq in enumerate(frequencies) : 
                        bandwidth = max(0.5,0.20 * freq)
                        print('freq', freq)
                        for itr in range(len(id_ev)) : # over the trials
                            filtered_d[itr, :, ifreq, :] = gaussian_frequency_filter(d[itr, :,:], fs, freq, bandwidth) 
                    # average across trials            
                    ieeg_data_subj[ic, :, :, :] = np.nanmean(filtered_d, axis=0) #channel freq, time
            else :
                ieeg_data_subj = np.zeros((2, epoch.shape[1], epoch.shape[2])) *np.nan
                for ic, id_ev in enumerate([id_ev1,id_ev2 ]) : 
                    ieeg_data_subj[ic, :, :]=epoch[id_ev, :,:].mean(0)

            # save
            with open(path_to_save + f'/{subj}_ieeg{l}.p', "wb") as f:
                pickle.dump(ieeg_data_subj, f)
            del epoch

        
if __name__ == "__main__":
    #get_meg_from_raw(type_ = 'source', freq=True,fs=250)
    get_ieeg_from_raw(fs=400, freq=True)

       





