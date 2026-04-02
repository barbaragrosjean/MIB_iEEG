import os
import json
import numpy as np
import pandas as pd
import pickle
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from nilearn import plotting
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA

from PIL import Image

plt.style.use('seaborn-v0_8-dark')

from src.config import PROJECT_PATH, OUT_PATH, FREQ_BAND
from src.preproc import BbEvents, TFRmEvents
from src.setting import PolarityCor

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

# Transform
def CompoThr(data, replace=0) : 
    data_thr = data.copy()
    for i in range(data.shape[0]):
        thr = data[i,:].mean() + abs(data[i, :].std())      
        index_thr = np.where(abs(data[i, :]) < thr)
        data_thr[i, index_thr[0].flatten()] = replace

    return data_thr

def DataAugmentation(TFRtr,event_ids, data_aug_method='mean', nb_trials=23) : 
    id_ev1 = event_ids[0]
    id_ev2 = event_ids[1]
    TFR_trials_filled = np.full((nb_trials, 2, TFRtr.shape[1], TFRtr.shape[2]), np.nan)
    
    if len(id_ev1) > nb_trials:
        TFR_trials_filled[:nb_trials, 0, :] =  TFRtr[random.sample(list(id_ev1), nb_trials), :, :] 
    else : 
        TFR_trials_filled[:len(id_ev1), 0, :] =  TFRtr[id_ev1, :, :] 

    if len(id_ev2) > nb_trials:
        TFR_trials_filled[:nb_trials, 1, :] =  TFRtr[random.sample(list(id_ev2), nb_trials), :, :]
    else :
        TFR_trials_filled[:len(id_ev2), 1, :] =  TFRtr[id_ev2, :, :]

    true_trials = ~np.any(np.isnan(TFR_trials_filled), axis=(2, 3))


    if data_aug_method == 'mean':
        for i in range(2):  
            event_means = np.nanmean(TFR_trials_filled[:, i, :, :], axis=0, keepdims=True)
            nan_mask = np.isnan(TFR_trials_filled[:, i, :, :])
            TFR_trials_filled[:, i, :, :] = np.where(nan_mask, event_means, TFR_trials_filled[:, i, :, :])

    elif data_aug_method == 'duplicat' :
        ids_w_fake1 = id_ev1
        ids_w_fake2 = id_ev2
        while len(ids_w_fake1) + len(id_ev1) < nb_trials : 
            ids_w_fake1 = np.concat([ids_w_fake1,id_ev1], axis=0)
        ids_w_fake1 = np.concat([ids_w_fake1, random.sample(list(id_ev1), nb_trials-ids_w_fake1.shape[0])]).astype(int)

        while len(ids_w_fake2) + len(id_ev2) < nb_trials : 
            ids_w_fake2 = np.concat([ids_w_fake2,id_ev2], axis=0)
        ids_w_fake2 = np.concat([ids_w_fake2, random.sample(list(id_ev2), nb_trials-ids_w_fake2.shape[0])]).astype(int)

        TFR_trials_filled[:, 0, :, :] =  TFRtr[ids_w_fake1, :, :] 
        TFR_trials_filled[:, 1, :, :] =  TFRtr[ids_w_fake2, :, :]  

    return np.concatenate([TFR_trials_filled[:,i, :, :] for i in [0, 1]], axis = 0), true_trials

def DataTransformationM1(freq, freq_band=FREQ_BAND, PC_use=0,nb_trials=23, nb_compo = 3,subj_included=[], method_pca='mean', data_aug_method='mean', data_path = OUT_PATH + '/Data', pol_cor=False, method ='pca') : 
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

        # Compute TFRm 
        if freq == 'broadband' :
            TFRm = BbEvents(subj, test_id = id_test, events_index=events_index, data_path=data_path)
             
        else : 
            freq_id = freq_band.index(freq)
            TFRm = TFRmEvents(subj, test_id = id_test, freq_id = freq_id, events_index=events_index, data_path=data_path)

        # Save for PCA computation at grp level
        if method_pca == 'concat' :
            TFRm_list.append(np.concatenate([TFRm[i, :,:] for i in [0, 1]], axis = 1))
        if method_pca == 'mean' : 
            TFRm_list.append(np.mean(TFRm[[0, 1], :,:], axis = 0))

        if freq == 'broadband' :
            file = data_path + f'/{subj}_epochs.p'
            with open(file, "rb") as f:
                TFRtr = pickle.load(f)  

            TFRtr_augmented, true_trials = DataAugmentation(TFRtr[:, :, :], [id_ev1, id_ev2], data_aug_method, nb_trials=nb_trials) # return 48, ch, time
            Test_sample.append(TFRtr[id_test,:, :])

        else :
            file = data_path + f'/{subj}_TFRtrials.p'
            with open(file, "rb") as f:
                TFRtr = pickle.load(f)  

            # Augment the data
            TFRtr_augmented, true_trials = DataAugmentation(TFRtr[:, :, freq_band.index(freq), :], [id_ev1, id_ev2], data_aug_method, nb_trials=nb_trials) # return 48, ch, time
            Test_sample.append(TFRtr[id_test,:, freq_band.index(freq), :])
        
        a = np.concat([TFRtr_augmented[None, :nb_trials, :, :], TFRtr_augmented[None, nb_trials:, :, :]], axis = 0)
        perm_trials = np.random.permutation(a.shape[1])
        a[0, :, :, :] = a[0, perm_trials, :, :]
        perm_trials = np.random.permutation(a.shape[1])
        a[1, :, :, :] = a[1, perm_trials, :, :]
        TFRtr_augmented = np.concat([a[0, :, :, :], a[1, :, :, :]], axis = 0)

        Train_sample.append(TFRtr_augmented)
        truth.append(true_trials)

    Train_all = np.concatenate(Train_sample, axis=1)
    Test_all = np.concatenate(Test_sample, axis =2)


    concat_all = np.concatenate(TFRm_list, axis = 0)
    if pol_cor : 
        concat_all = PolarityCor(concat_all, method_pca=method_pca, subj_included=subj_included, data_path=data_path)
    
    del TFRm_list
    df_Componants, _, means = ConcatPCA({'grp' : concat_all}, ch_id = False, nb_compo=nb_compo, freq_band=[freq],method=method, return_mean=True)
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

    return Train_transformed, [1]*nb_trials + [2]*nb_trials, Test_transformed, [1, 2], np.stack(truth, axis=2)*1, weights # X_train, y_train, X_test, y_test, subj_track_train, proportion of true trail in each supersample
 
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

        if freq == 'broadband' :
            file = data_path + f'/{subj}_epochs.p'
            with open(file, "rb") as f:
                TFRtr = pickle.load(f)  

            TFRtr_augmented, true_trials = DataAugmentation(TFRtr[:, :, :], [id_ev1, id_ev2], data_aug_method) # return 48, ch, time
            Test_sample.append(TFRtr[id_test,:, :])

        else :
            file = data_path + f'/{subj}_TFRtrials.p'
            with open(file, "rb") as f:
                TFRtr = pickle.load(f)  

            # Augment the data
            TFRtr_augmented, true_trials = DataAugmentation(TFRtr[:, :, freq_band.index(freq), :], [id_ev1, id_ev2], data_aug_method) # return 48, ch, time
            Test_sample.append(TFRtr[id_test,:, freq_band.index(freq), :])
        
        a = np.concat([TFRtr_augmented[None, :23, :, :], TFRtr_augmented[None, 23:, :, :]], axis = 0)
        perm_trials = np.random.permutation(a.shape[1])
        a[0, :, :, :] = a[0, perm_trials, :, :]
        perm_trials = np.random.permutation(a.shape[1])
        a[1, :, :, :] = a[1, perm_trials, :, :]
        TFRtr_augmented = np.concat([a[0, :, :, :], a[1, :, :, :]], axis = 0)

        Train_sample.append(TFRtr_augmented)
        truth.append(true_trials)

    Train_all = np.concatenate(Train_sample, axis=1)
    Test_all = np.concatenate(Test_sample, axis =2)

    return Train_all, [1]*23 + [2]*23, Test_all[:, 0, :,:], [1, 2], np.stack(truth, axis=2)*1 

def prep_data_trial(band, method_pca, data_aug_method, subj_included, PC_use,data_path=OUT_PATH + '/Data', nb_trials=23) : 
    X_train0, y_train0, X_test0, y_test0, true_trials, _ = DataTransformationM1(freq= band, method_pca=method_pca, data_aug_method=data_aug_method, subj_included=subj_included, PC_use=PC_use, data_path=data_path, nb_trials=nb_trials)      

    X_0 = np.concat([X_train0, X_test0], axis=0)
    y_0 = np.concat([y_train0, y_test0], axis=0)
    X_0_old = X_0[np.where(y_0 == 1)]
    X_0_new = X_0[np.where(y_0 == 2)]

    return X_0,  X_0_old, X_0_new, y_0, true_trials