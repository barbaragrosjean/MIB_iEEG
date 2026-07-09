import os 
import numpy as np
import pandas as pd
import json
import numpy as np
import pickle
from scipy.stats import spearmanr, pearsonr

from sklearn.decomposition import PCA
from src.setting import ExcludSubj, get_data_grp
from src.behavior import extract_RT,get_cross_point, get_slop_pc2, area_pc2, get_slop_pc3

from src.config import OUT_PATH, PROJECT_PATH
from sklearn.decomposition import PCA

path_behavior=PROJECT_PATH + '/misc/events'
path_class = PROJECT_PATH + '/misc/concat_beh_data.csv'
epoch_path = OUT_PATH + '/Data_longWOBS'

subj_included = [file.replace('_epochs.p', '') for file in os.listdir(epoch_path) if file[-len('epochs.p'):] == 'epochs.p']
subj_included = ExcludSubj(subj_included=subj_included, data_path=epoch_path)

nb_trials = 16
subj_included_restricted = {16: ['BJH072', 'LL36', 'BJH069', 'SLCH020', 'BJH045', 'LL14', 'BJH050',
       'BJH027', 'OS70', 'BJH052', 'OS61', 'BJH041', 'BJH046', 'BJH056',
       'LL31', 'BJH039', 'BJH042', 'BJH026', 'LL08', 'BJH029',
       'DA037', 'BJH058', 'SLCH024', 'BJH049']}

path = epoch_path + f'/{subj_included[0]}_info.json'
with open(path) as json_data:
    d = json.load(json_data)
    time_tfr=d['time_tfr']

r_run  = 500
nb_trials = 16

df_RT = extract_RT(subj_included=subj_included, path_behavior=path_behavior, path_info=epoch_path)

pca = PCA(3)
subj_list=subj_included_restricted[nb_trials]
data_grp = get_data_grp(subj_list, return_subj=False, data_path=epoch_path, polarity_cor=(False,0), type_data='tfr')[:, :, -1, :]
sorted_='RT'

find_tp = {'0_pc1': lambda x, tr : get_cross_point(tr, x, tstart=90, tend=150), 
           '0_pc2': lambda x, tr :  get_cross_point(tr, x, tstart=50, tend=115), 
           '0_pc3': lambda x, tr : get_cross_point(tr, x, tstart=120, tend=145), 
           'min_pc2': lambda x, tr : np.argmin(x[tr, 100:]) + 100, 
           'bump_pc3' : lambda x, tr : 135 + np.argmax(x[tr, 135:]),
           'desc_slop_pc2': lambda x, tr : get_slop_pc2(tr, x, time=time_tfr)[0] , 
           'asc_slop_pc2' : lambda x, tr : get_slop_pc2(tr, x, time=time_tfr)[1], 
           'slop_pc3': lambda x, tr : get_slop_pc3(tr, x, time=time_tfr),
           'area_under_pc2': lambda x, tr : area_pc2(tr, x)[0], 
           'area_above_pc2': lambda x, tr : area_pc2(tr, x)[1]}

to_try = list(find_tp.keys())
metr_list = []
corr_list=[]
rt_list=[]
trans_save=np.zeros((r_run, 3, 2, nb_trials, data_grp.shape[-1]))
pca.fit(np.concat([data_grp[0, :,:], data_grp[1, :, :]], axis=-1).T)

for r_ in range(r_run):
    rt_mean =  {0 :[], 1: []}
    supsubj= {0 :[], 1: []}
    for ic, condi in enumerate(['1', '2'] ): 
        for subj in subj_list:
            df = df_RT[(df_RT['subject'] == subj)].reset_index(drop=True)
            df= df.dropna()
            if len(df) <4 : 
                print(subj)
            rt_condi=df[df['ev_id'] == condi].reset_index()[['RT', 'index']]

            # Randomness
            if len(rt_condi) > nb_trials : # if the there are more than 16 trials 
                rt_condi = rt_condi.sample(n=nb_trials, random_state=None)
            else : 
                rt_condi = rt_condi.sample(n=nb_trials, replace=True)
            rt_condi_sorted = rt_condi.sort_values(by = sorted_).reset_index(drop=True)

            file = epoch_path + f'/{subj}_TFRtrials.p'
            with open(file, "rb") as f:
                TFRtr = pickle.load(f)[:, :, -1, :]
            
            random_index = rt_condi_sorted.loc[:, 'index'].values
            supsubj[ic].append(TFRtr[random_index,:,:])
            rt_mean[ic].append(df.loc[random_index, 'RT'])
    rt_all=[]

    for condi in range(2):
        rt_all.append(np.concat([r.values[None, :] for r in rt_mean[condi]], axis=0).mean(0))
        dataset=np.concat(supsubj[condi], axis=1)
        for pc_use in range(3) : 
            for tr in range(nb_trials) : 
                trans_save[r_, pc_use, condi, tr, :] = dataset[tr, :, :].T @ pca.components_[pc_use, :]
    rt_list.append(rt_all)

    for to_t in to_try :
        trans = trans_save[r_, int(to_t[-1:])-1, :, :, :]
        print('############# TEST ############# ', to_t)

        for i in range(2):
            if to_t in ['0_pc1','0_pc2', '0_pc3'] : 
                peaks = np.array([find_tp[to_t](trans[i], tr) for tr in range(trans[i].shape[0])])
                metr = np.asarray(time_tfr)[peaks]
                r, p = pearsonr(rt_all[i], metr)
                metr_list.append(metr)
                corr_list.append((r, p, to_t, i, 'delay'))

            elif to_t in ['desc_slop_pc2', 'asc_slop_pc2', 'slop_pc3'] : 
                metr=[]
                for tr in range(trans[i].shape[0]) :
                    try : 
                        m = find_tp[to_t](trans[i], tr)
                    except : 
                        m = np.nan
                    metr.append(m)

                r, p = pearsonr(rt_all[i], metr)
                corr_list.append((r, p, to_t, i, 'slope'))
                metr_list.append(metr)

            elif to_t in ['area_under_pc2', 'area_above_pc2'] :
                metr = np.array([find_tp[to_t](trans[i], tr) for tr in range(trans[i].shape[0])])
                r, p = pearsonr(rt_all[i], metr)
                metr_list.append(metr)
                corr_list.append((r, p, to_t, i, 'auc'))

            elif to_t in ['min_pc2', 'bump_pc3'] : 
                peaks = np.array([find_tp[to_t](trans[i], tr) for tr in range(trans[i].shape[0])])
                ampl = trans[i][np.arange(trans[i].shape[0]), peaks]
                delay = np.asarray(time_tfr)[peaks]
                r_amp, p_amp = pearsonr(rt_all[i], ampl)
                r_del, p_del = pearsonr(rt_all[i], delay)
                metr_list.extend([ampl, delay])
                corr_list.extend([(r_amp, p_amp, to_t, i, 'ampl'), (r_del, p_del, to_t, i, 'delay')])
            
        
corr = np.vstack(corr_list)
metr = np.vstack(metr_list)
rt = np.stack(rt_list, axis=0)

# save 
out_path = OUT_PATH + '/Behavior'
if not os.path.exists(out_path) : 
    os.makedirs(out_path)
    
with open(out_path + f"/tp_corr_{r_run}.pkl", "wb") as f:
    pickle.dump(corr, f)
with open(out_path + f"/tp_metr_{r_run}.pkl", "wb") as f:
    pickle.dump(metr, f)
with open(out_path + f"/tp_rt_{r_run}.pkl", "wb") as f:
    pickle.dump(rt, f)
with open(out_path + f"/tp_trans_{r_run}.pkl", "wb") as f:
    pickle.dump(trans_save, f)
        

def time_to_time_corr(df_RT, pca_transform):
    corr = np.zeros((r_run, 2, len(time_tfr)))
    pval = np.zeros((r_run, 2,  len(time_tfr)))
    rt_mean_all=np.zeros((r_run, 2, len(nb_trials)))

    for r in range(r_run) : 
        dataset0=[]
        dataset1 = []
        rt_mean0 = []
        rt_mean1=[]

        # subset of each 
        supsubj = {0 :[], 1: []}
        full_dataset = {0 :[], 1: []}
        
        for ic, condi in enumerate(['1', '2'] ): 
            rt_subj=[]
            for subj in subj_included:
                df = df_RT[(df_RT['subject'] == subj)].reset_index(drop=True)
                df = df.dropna()
                rt_condi=df[df['ev_id'] == condi].reset_index()[['RT', 'index']]

                if len(rt_condi) > nb_trials : # if the there are more than 16 trials 
                    rt_condi = rt_condi.sample(n=nb_trials, random_state=None)
                else : 
                    rt_condi = rt_condi.sample(n=nb_trials, replace=True)

                rt_condi_sorted = rt_condi.sort_values(by = 'RT').reset_index(drop=True)

                file = epoch_path + f'/{subj}_TFRtrials.p'
                with open(file, "rb") as f:
                    TFRtr = pickle.load(f)[:, :, -1, :]
                random_index = rt_condi_sorted.loc[:, 'index'].values
                supsubj[ic].append(TFRtr[random_index,:,:])
                rt_subj.append(df.loc[random_index, 'RT'])

            rt_mean_all[r, ic, :] = np.concat([r.values[None, :] for r in rt_subj], axis=0).mean(0)

        dataset0.append(np.concat(supsubj[0], axis=1))
        dataset1.append(np.concat(supsubj[1], axis=1))

        dataset0 = np.concat([d[:,:] for d in dataset0], axis=0)
        dataset1 = np.concat([d[:,:] for d in dataset1], axis=0)

        transform0 = np.zeros((nb_trials,  len(time_tfr)))
        transform1 = np.zeros((nb_trials,  len(time_tfr)))
        
        for tr in range(nb_trials) : 
            transform0[tr, :] = dataset0[tr, :, :].T @ pca.components_[pc_use, :]
            transform1[tr, :] = dataset1[tr, :, :].T @ pca.components_[pc_use, :]

        # corr with the RT
        for t in range(len(time_tfr)) : 
            corr[r, 0, t], pval[r, 0, t] = spearmanr(transform0[:, t], rt_mean_all[r, 0, :]) 
            corr[r, 1, t], pval[r, 1, t] = spearmanr(transform1[:, t], rt_mean_all[r, 1, :]) 

        
    # save 
    out_path = OUT_PATH + '/Behavior'
    if not os.path.exists(out_path) : 
        os.makedirs(out_path)
        
    with open(out_path + f"/corr_{pc_use}.pkl", "wb") as f:
        pickle.dump(corr, f)
    with open(out_path + f"/pval_{pc_use}.pkl", "wb") as f:
        pickle.dump(pval, f)
    with open(out_path + f"/pca_transform_{pc_use}.pkl", "wb") as f:
        pickle.dump(pca_transform, f)
    with open(out_path + f"/rt_{pc_use}.pkl", "wb") as f:
        pickle.dump(rt_mean_all, f)