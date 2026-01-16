import os.path as op
import os
import pickle
import json
import numpy as np
import pandas as pd
import re
import random

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from nilearn import plotting

from sklearn.utils import shuffle
from scipy.stats import spearmanr, pearsonr
from scipy.signal import find_peaks

from sklearn.decomposition import NMF
from sklearn.decomposition import PCA

from PIL import Image

plt.style.use('seaborn-v0_8-dark')

OUT_PATH = 'outs'
    
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

################################### FUNCTIONS DATA TRANSFORMATION ###################################

def CompoThr(data, replace=0) : 
    data_thr = data.copy()
    for i in range(data.shape[0]):
        thr = data[i,:].mean() + abs(data[i, :].std())      
        index_thr = np.where(abs(data[i, :]) < thr)
        data_thr[i, index_thr[0].flatten()] = replace

    return data_thr

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

def DataAugmentation(TFRtr,event_ids, data_aug_method='mean') : 
    id_ev1 = event_ids[0]
    id_ev2 = event_ids[1]
    TFR_trials_filled = np.full((23, 2, TFRtr.shape[1], TFRtr.shape[2]), np.nan)
    TFR_trials_filled[:len(id_ev1), 0, :] =  TFRtr[id_ev1, :, :] 
    TFR_trials_filled[:len(id_ev2), 1, :] =  TFRtr[id_ev2, :, :]
    true_trials = ~np.any(np.isnan(TFR_trials_filled), axis=(2, 3))

    if data_aug_method == 'mean':
        for i in range(2):  
            event_means = np.nanmean(TFR_trials_filled[:, i, :, :], axis=0, keepdims=True)
            nan_mask = np.isnan(TFR_trials_filled[:, i, :, :])
            TFR_trials_filled[:, i, :, :] = np.where(nan_mask, event_means, TFR_trials_filled[:, i, :, :])

    if data_aug_method == 'duplicat' :
        ids_w_fake1 = id_ev1
        ids_w_fake2 = id_ev2
        while len(ids_w_fake1) + len(id_ev1) < 23 : 
            ids_w_fake1 = np.concat([ids_w_fake1,id_ev1], axis=0)
        ids_w_fake1 = np.concat([ids_w_fake1, random.sample(list(id_ev1), 23-ids_w_fake1.shape[0])]).astype(int)

        while len(ids_w_fake2) + len(id_ev2) < 23 : 
            ids_w_fake2 = np.concat([ids_w_fake2,id_ev2], axis=0)
        ids_w_fake2 = np.concat([ids_w_fake2, random.sample(list(id_ev2), 23-ids_w_fake2.shape[0])]).astype(int)

        TFR_trials_filled[:, 0, :, :] =  TFRtr[ids_w_fake1, :, :] 
        TFR_trials_filled[:, 1, :, :] =  TFRtr[ids_w_fake2, :, :]

    return np.concatenate([TFR_trials_filled[:,i, :, :] for i in [0, 1]], axis = 0), true_trials

def DataTransformationM1(freq, freq_band=FREQ_BAND, PC_use=0, subj_included=[], method_pca='mean', data_aug_method='mean', shuffle_index = False, data_path = OUT_PATH + '/Data', pol_cor=False, method ='pca') : 
    TFRm_list = []
    
    Train_sample = []
    Test_sample = []
    truth = []

    if subj_included ==[] : 
        subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
 
    for subj in subj_included : 
        info_file = data_path + f'/{subj}_info.json'
        with open(info_file) as f:
            info = json.load(f)
            events_index = np.array([int(i) for i in info['event_id']])

        id_ev1 = np.where(events_index == 1)[0]
        id_ev2 = np.where(events_index == 2)[0]

        # Keep 1 id per condi for testing 
        id_test= [random.sample(list(id_ev1),1), random.sample(list(id_ev2),1)]
        id_ev1 = list(id_ev1)
        id_ev1.remove(id_test[0])
        id_ev1 = np.array(id_ev1)
        id_ev2 = list(id_ev2)
        id_ev2.remove(id_test[1])
        id_ev2 = np.array(id_ev2)

        if shuffle_index : # TO ADJUST TODO
            ev_shuffl= shuffle(np.concat([id_ev1, id_ev2]))
            id_ev1_s = ev_shuffl[:id_ev1.shape[0]]
            id_ev2_s = ev_shuffl[id_ev1.shape[0]:]

        # Compute TFRm 
        if freq == 'broadband' :
            TFRm = BbEvents(subj, test_id = id_test, events_index=events_index, data_path=data_path)
            if pol_cor : 
                TFRm = PolarityCor(TFRm, subj_included=subj_included, data_path =data_path, method_pca=method_pca)
            
        else : 
            freq_id = freq_band.index(freq)
            TFRm = TFRmEvents(subj, test_id = id_test, freq_id = freq_id, events_index=events_index, data_path=data_path)

        # Save for PCA computation at grp level
        if method_pca == 'concat' :
            TFRm_list.append(np.concatenate([TFRm[i, :,:] for i in [0, 1]], axis = 1))
        if method_pca == 'mean' : 
            TFRm_list.append(np.mean(TFRm[[0, 1], :,:], axis = 0))

        # Get the data
        if freq == 'broadband' :
            file = data_path + f'/{subj}_epochs.p'
            with open(file, "rb") as f:
                TFRtr = pickle.load(f)  

            if pol_cor : 
                TFRtr = PolarityCor(TFRtr, subj_included=subj_included, data_path =data_path, method_pca=method_pca)

            TFRtr_augmented, true_trials = DataAugmentation(TFRtr, [id_ev1, id_ev2], data_aug_method) # return 48, ch, time
            Train_sample.append(TFRtr_augmented)
            truth.append(true_trials)
            Test_sample.append(TFRtr[id_test,:, :])

        else :
            file = data_path + f'/{subj}_TFRtrials.p'
            with open(file, "rb") as f:
                TFRtr = pickle.load(f)  

            # Augment the data
            TFRtr_augmented, true_trials = DataAugmentation(TFRtr[:, :, freq_id, :], [id_ev1, id_ev2], data_aug_method) # return 48, ch, time
            Train_sample.append(TFRtr_augmented)

            truth.append(true_trials)
            Test_sample.append(TFRtr[id_test,:, freq_id, :])


    concat_all = np.concatenate(TFRm_list, axis = 0)
    if pol_cor : 
        concat_all = PolarityCor(concat_all, method_pca=method_pca, subj_included=subj_included, data_path=data_path)
    
    del TFRm_list
    df_Componants, _, means = ConcatPCA({'grp' : concat_all}, ch_id = False, nb_compo=3, freq_band=[freq],method=method, return_mean=True)
    weights = df_Componants['grp'].query("freq == @freq").drop(columns = ['freq', 'compo']).values
    Train_all = np.concatenate(Train_sample, axis=1)
    Test_all = np.concatenate(Test_sample, axis =2)

    # to center the data before transform
    mean_pca = means[freq]
    m_train= mean_pca[None, :, None]
    m_test = mean_pca[None, :, None]

    # Transform the data using the weights
    if type(PC_use) == list :
        Train_transformed = np.zeros([Train_all.shape[0],len(PC_use), Train_all.shape[-1]])
        Test_transformed = np.zeros([Test_all.shape[0], len(PC_use),Test_all.shape[-1]])
        for pc in PC_use : 
            Train_transformed[:, pc, :] = weights[pc, :] @ (Train_all -m_train)
            Test_transformed[:, pc, :] = weights[pc, :] @ (Test_all[:,0,:] - m_test)
            
    else : 
        Train_transformed = weights[PC_use, :] @ (Train_all -m_train)
        Test_transformed = weights[PC_use, :] @ (Test_all[:,0,:] - m_test)

    return Train_transformed, [1]*23 + [2]*23, Test_transformed, [1, 2], np.stack(truth, axis=2)*1, weights # X_train, y_train, X_test, y_test, subj_track_train, proportion of true trail in each supersample
 
def DataTransformationM1Raw(freq, freq_band=FREQ_BAND, out_path = OUT_PATH, subj_included=[], data_aug_method='mean', data_path = OUT_PATH + '/Data', pol_cor=False) :
    Train_sample = []
    Test_sample = []
    truth = []

    if subj_included ==[] : 
        subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
 
    for subj in subj_included : 
        info_file = data_path + f'/{subj}_info.json'
        with open(info_file) as f:
            info = json.load(f)
            events_index = np.array([int(i) for i in info['event_id']])

        id_ev1 = np.where(events_index == 1)[0]
        id_ev2 = np.where(events_index == 2)[0]

        # Keep 1 id per condi for testing 
        id_test= [random.sample(list(id_ev1),1), random.sample(list(id_ev2),1)]
        id_ev1 = list(id_ev1)
        id_ev1.remove(id_test[0])
        id_ev1 = np.array(id_ev1)
        id_ev2 = list(id_ev2)
        id_ev2.remove(id_test[1])
        id_ev2 = np.array(id_ev2)

        # Get the data
        if freq == 'broadband' :
            file = data_path + f'/{subj}_epochs.p'
            with open(file, "rb") as f:
                TFRtr = pickle.load(f)  
            if pol_cor : 
                TFRtr = PolarityCor(TFRtr, subj_included=subj_included, data_path =data_path, method_pca=method_pca)
            TFRtr_augmented, true_trials = DataAugmentation(TFRtr[:, :, :], [id_ev1, id_ev2], data_aug_method) # return 48, ch, time
            Train_sample.append(TFRtr_augmented)
            truth.append(true_trials)
            Test_sample.append(TFRtr[id_test,:, :])

        else :
            freq_id = freq_band.index(freq)
            file = data_path + f'/{subj}_TFRtrials.p'
            with open(file, "rb") as f:
                TFRtr = pickle.load(f)  

            # Augment the data
            TFRtr_augmented, true_trials = DataAugmentation(TFRtr[:, :, freq_id, :], [id_ev1, id_ev2], data_aug_method) # return 48, ch, time
            Train_sample.append(TFRtr_augmented)

            truth.append(true_trials)
            Test_sample.append(TFRtr[id_test,:, freq_id, :])

    Train_all = np.concatenate(Train_sample, axis=1)
    Test_all = np.concatenate(Test_sample, axis =2)

    return Train_all, [1]*23 + [2]*23, Test_all[:, 0, :,:], [1, 2], np.stack(truth, axis=2)*1 

################################### FUNCTIONS PCA ###################################
def ConcatPCA(concat_dict, ch_id = False, nb_compo =2, freq_band=FREQ_BAND, method ='pca', return_mean = False) : 
    # Try to concat at subject level the event correct 
    df_list = []
    df_Componants = {}
    means= {}
    for subj, concat in concat_dict.items() : 
        df_compo_list = []
        if ch_id == False : 
            concat_ = concat.copy()
        else :
            id_channels = ch_id[subj]
            concat_ = concat[id_channels, :,:]

        for i, freq in enumerate(freq_band) : 
            if len(concat_.shape) == 2: 
                X = concat_ # if already one freq band or broad band
            else :
                X = concat_[:, i, :]

            # PCA
            if method == 'pca':
                pca = PCA(n_components=nb_compo)
                X_transformed = pca.fit_transform(X.T)
                df_compo = pd.DataFrame(pca.components_)
                means[freq] = pca.mean_

            elif method == 'nmf' :
                X_pos = X - X.min() + 1e-6 # shift to make it non negative
                # optimize NMF 
                means[freq] = X.mean(1, keepdims=False)
                best, _ = run_nmf_multistart(X_pos.T, nb_compo, n_runs=10, random_states=None, max_iter=10000)
                X_transformed = best['X_transformed']
                df_compo =  pd.DataFrame(best['Compo'])
            
            # Save compo
            df_compo.loc[:, 'compo'] = ['compo' + str(i_compo+1) for i_compo in range(nb_compo)]
            df_compo.loc[:, 'freq'] = [freq_band[i]]*nb_compo
            df_compo_list.append(df_compo)

            # save 
            df = pd.DataFrame(X_transformed.T)
            df.loc[:, 'compo'] = ['compo' + str(i_compo+1) for i_compo in range(nb_compo)]
            df.loc[:, 'freq'] = [freq]*(nb_compo)
            df.loc[:, 'subj'] = [subj]*(nb_compo)
            if method == 'pca' :
                df.loc[:, 'expl_var'] = pca.explained_variance_ratio_ 
            if method == 'nmf' : 
                df.loc[:, 'error'] = best['error']

            df_list.append(df)   

        df_Componants[subj] = pd.concat(df_compo_list, axis=0)
    df_X_transformed = pd.concat(df_list) 
    if return_mean :
        return df_Componants, df_X_transformed, means
    else :
        return df_Componants, df_X_transformed

def PlotCompoIndividual(subj, df_Componants,subj_included=[], nb_compo = 2, save = True, show=False,browser=False, freq_band= FREQ_BAND, data_path = OUT_PATH +'/Data', out_path = OUT_PATH, project_path=PROJECT_PATH, method='pca') : 
    if not os.path.exists(out_path) and save : 
        os.makedirs(out_path)
    
    if subj == 'grp' : 
        if subj_included == [] : 
            subj_included = [file.replace('_info.json', '') for file in os.listdir(data_path) if file[-len('info.json'):] == 'info.json']

        coord = []
        for s in subj_included :
            coord_file = pd.read_csv(f'{data_path}/{s}_coords.csv')
            coord.extend(np.vstack([coord_file['x'].values,  coord_file['y'].values, coord_file['z'].values]).T)
        coord = np.array(coord)
    else : 
        coord_file = pd.read_csv(f'{data_path}/{subj}_coords.csv')
        coord = np.vstack([coord_file['x'].values,  coord_file['y'].values, coord_file['z'].values]).T

    combined_images = []
    to_remove = []
    for band in freq_band: 
        row_images = []
        data = df_Componants[subj].query("freq == @band").drop(columns = ['compo', 'freq']).values
        vlim = np.abs(data.flatten()).max()
        data_thr = CompoThr(data,  replace=0)

        if method =='pca' :
            node_map = 'seismic'
            node_vmin=-vlim
            node_vmax=vlim
        elif method == 'nmf' :
            node_map ='Reds'
            node_vmin = 0
            node_vmax = vlim
        
        for compo_id in range(nb_compo) :
            index_thr = np.where(data_thr[compo_id, :] != 0)
            fig = plotting.plot_markers(node_coords = coord[index_thr, :][0],  node_size=10, node_values=data[compo_id, index_thr], node_cmap=node_map, title=f'{band}:PC n{compo_id+1}',display_mode='ortho', node_vmin=node_vmin, node_vmax=node_vmax)
            
            if show : plt.show()
            else : plt.close()

            if save :                
                temp_png = f'{out_path}/{subj}_PC{str(compo_id+1)}_{band}.png'
                fig.savefig(temp_png)
                row_images.append(Image.open(temp_png))
                to_remove.append(temp_png)

            if browser : 
                vmin = -np.max(np.abs(data)) 
                vmax = np.max(np.abs(data))
                norm_signal = (data - vmin) / (vmax - vmin)
                cmap = plt.cm.Blues
                colors = [tuple(cmap(val)) for val in norm_signal.flatten()]
                view = plotting.view_markers(coord, marker_color=colors,marker_size=5,title=f'PC{compo_id+1} {subj} {band}', title_fontsize=25 )
                view.open_in_browser()
            
        if save : 
            w, h = row_images[0].size
            row_combined = Image.new('RGB', (w*nb_compo, h))
            for idx, img in enumerate(row_images):
                row_combined.paste(img, (idx*w, 0))
            
            combined_images.append(row_combined)
    if save : 
        pdf_path = f'{out_path}/{subj}_compos.pdf'
        combined_images[0].save(pdf_path, save_all=True, append_images=combined_images[1:])

        for img in to_remove : 
            os.remove(img)

def PlotTimeSerie(subj_list, df_X_transformed, out_path, region='', show=False, save=True, tfr=True, data_path = OUT_PATH + '/Data')  :    
    for subj in subj_list :  
        # get the time 
        if subj == 'grp' : 
            json_file = data_path + '/' + [f for f in os.listdir(data_path) if '_info.json' in f][0]
        else : 
            json_file = f'{data_path}/{subj}_info.json'

        with open(json_file) as json_data:
            d = json.load(json_data)
            if tfr:
                time=np.array(d['time_tfr'])

            else : 
                time=np.array(d['time'])
                
        df_subset = df_X_transformed.query('subj == @subj')
        freq_nb = len(np.unique(df_subset.freq.values))
        compo_nb = len(np.unique(df_subset.compo.values))

        if freq_nb == 0 :
            freq_nb =1

        fig, axs = plt.subplots(1 , freq_nb, figsize = (30, 5))
        fig.suptitle('-- ' + subj + ' --')
        color_1 = [plt.cm.Blues(i) for i in np.linspace(0.7, 0.3, compo_nb)] 
        color_2 = [plt.cm.Reds(i) for i in np.linspace(0.7, 0.3, compo_nb)] 

        for band_id, band in enumerate(FREQ_BAND) : 
            for compo_id in range(compo_nb) : 
                df_to_plot = df_subset.query(f'freq == @band')
                the_ax = axs[band_id]

                # event1 
                df_to_plot_1 = df_to_plot.set_index('compo').loc[:, :len(time)-1]
                df_to_plot_1.loc['time', :] = time

                the_ax.plot(df_to_plot_1.loc['time', :], df_to_plot_1.loc['compo' + str(compo_id+1), :], label = f'PC{compo_id+1}', color=color_1[compo_id])
                
                # event2
                df_to_plot_2 = df_to_plot.set_index('compo').loc[:, len(time):len(time)*2 -1]                 
                df_to_plot_2.loc['time', :] = time
                the_ax.plot(df_to_plot_2.loc['time', :], df_to_plot_2.loc['compo' + str(compo_id+1), :], label = f'PC{compo_id+1}', color=color_2[compo_id])
            
                the_ax.set_xlabel('Time (s)')
                the_ax.set_title(band)
                the_ax.grid()

                if band_id == len(FREQ_BAND)-1 :
                    line_handles, line_labels = the_ax.get_legend_handles_labels()

                    cmap_patches = [mpatches.Patch(color=color_1[0], label='Old / Correct (Blues)'), 
                                    mpatches.Patch(color=color_2[0], label='New / Correct (Reds)')]
                    
                    handles = line_handles + cmap_patches
                    labels = line_labels + [p.get_label() for p in cmap_patches]

                    the_ax.legend(handles, labels, loc='upper left', frameon=True, bbox_to_anchor = (1.05, 0.7))


        if save : 
            if not os.path.exists(out_path) :
                os.makedirs(out_path) 
            fig.savefig(f'{out_path}/{subj}_{region}_PCs.png')
        if show : plt.show()
        else  : plt.close()

def run_nmf_multistart(X, n_components, n_runs=10, random_states=None, max_iter=1000):
    if random_states is None:
        random_states = np.random.randint(0, 10_000, size=n_runs)

    results = []

    for rs in random_states:
        nmf = NMF(
            n_components=n_components,
            init="nndsvda",
            solver="cd",
            max_iter=max_iter,
            random_state=rs,
        )

        W = nmf.fit_transform(X)
        H = nmf.components_
        error = nmf.reconstruction_err_

        results.append({
            "X_transformed": W,
            "Compo": H,
            "error": error,
            "random_state": rs,
        })

    best = min(results, key=lambda x: x["error"])
    return best, results

################################### COMPO ANALYSIS ###################################
def crosscorr(datax, datay, lag=0):
    return datax.corr(datay.shift(lag), method='kendall')
    
def computeLagPeak(data_condi, time, win_high, method_pca, pc1=0, pc2=1, show=True, save=False, out_path =OUT_PATH, dis=60) : 
    peaks_coni1 = {c:[] for c in [pc1, pc2]}
    fig, ax = plt.subplots(3, 1, figsize=(12, 9))
    fig.suptitle(f'Peaks detection on PC1 and 2 time series -- New condition -- {method_pca} PCA')
    lag_peak_time = []
    c=['blue', 'orange']

    for i in [pc1, pc2] : 
        for win, high in win_high[i]:
            coni = data_condi[i]
            peaks, _ = find_peaks(coni[win[0]:win[1]], height=high, distance=dis)
            peaks_coni1[i].extend([p + win[0] for p in peaks])

        ax[i].plot(coni, c=c[i], label=f'PC {i+1} Time series')
        for p in peaks_coni1[i] :
            ax[i].scatter(p, coni[p], color='red', label='Detected Peaks')
        ax[i].set_title(f'PC {i+1}')
        ax[i].grid()

    for j in range(len(peaks_coni1[0])) :
        lag_peak_time.append(time[np.array(peaks_coni1[0][j])] - time[np.array(peaks_coni1[1][j])])
    mean_lag = np.mean(lag_peak_time)
    std_lag = np.std(lag_peak_time)
    ax[2].plot(time, data_condi[0], label='PC1', c=c[0])
    ax[2].plot(time + mean_lag, data_condi[1], label='PC2', c=c[1])
    ax[2].set_title('PC2 shifted by mean lag')
    ax[2].set_xlabel('Time (s)')
    ax[2].legend()
    ax[2].grid()
    ax[2].text(s=f'Mean lag : {np.round(mean_lag, 3)}s ± {np.round(std_lag, 3)}s', x=-0.5, y=-0.015)
    if show :
        plt.show()
    if save :
        fig.savefig(out_path + f'/lagPeak_{method_pca}.png')

    return lag_peak_time
        
def intersubject_correlation(data_grp_, subj_list, W, k): # how much the transform of single subject using PCA weight are the same 
    Z = []

    for subj in np.unique(subj_list):
        idx= np.where(np.array(subj_list) == subj)[0]
        Xs = data_grp_[idx, :].T # take only the signal from subject subj
        ws = W[k, idx].T # take only the wiehgts of subject subj
        z = Xs @ ws # commpute the transform
        Z.append(z)

    S = len(Z)
    R = []

    for i in range(S):
        for j in range(i+1, S):
            r, _ = pearsonr(Z[i], Z[j])
            R.append(r)

    return np.mean(R), np.array(R)

def loso_pca_stability(data_grp_, subj_list, n_components): # LOO one subject and copute PCA to see if the componant are stable enought
    pca_full = PCA(n_components=n_components).fit(data_grp_.T)
    C_full = pca_full.components_     

    stabilities = []

    for subj in np.unique(subj_list):
        idx= np.where(np.array(subj_list) == subj)[0]
        
        mask = np.ones(data_grp_.shape[0], dtype=bool)
        mask[idx] = False

        X_loo = data_grp_[mask, :]
        pca_loo = PCA(n_components=n_components).fit(X_loo.T)
        C_loo = pca_loo.components_

        corr = np.abs(np.corrcoef(C_full[:, mask], C_loo)[:n_components, n_components:])
        stabilities.append(np.diag(corr))

    return np.array(stabilities)  

def subject_variance_explained(data_grp_, subj_list, W): # var explain by subject data in the 3 first componants
    results = []
 
    for subj in np.unique(subj_list):
        idx= np.where(np.array(subj_list) == subj)[0]
        Xs = data_grp_[idx, :]          
        Ws = W[idx, :]     
        Zs = Ws.T @ Xs                    
        Xs_hat = Ws @ Zs         

        r2 = np.linalg.norm(Xs_hat)**2 / np.linalg.norm(Xs)**2
        results.append(r2)

    return np.array(results)

def WeightSpearman(data1, data2, labels=[], out_path=OUT_PATH) : 
    if labels == [] : 
        labels = ['data1', 'data2']
    R = np.zeros((data1.shape[0], data2.shape[0]))
    P = np.zeros((data1.shape[0], data2.shape[0]))

    for i in range(data1.shape[0]):
        for j in range(data2.shape[0]):
            r, pval = spearmanr(data1[i, :], data2[j, :])
            R[i, j] = r
            P[i, j] = pval

    fig, ax = plt.subplots(figsize = (8, 6))
    sns.heatmap(abs(R), vmin=0, vmax=1, cmap='Blues', ax=ax, annot=True , fmt='.3f')
    ax.set_xticklabels(['PC' + str(i+1) for i in range(data2.shape[0])])
    ax.set_xlabel(labels[1])
    ax.set_yticklabels(['PC' + str(i+1) for i in range(data1.shape[0])])
    ax.set_ylabel(labels[0])
    ax.set_title('Spearman correlation of weights')
    fig.savefig(out_path)

