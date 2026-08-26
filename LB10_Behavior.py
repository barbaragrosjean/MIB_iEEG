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

dataset_lab='mf70-160'
path_behavior=PROJECT_PATH + '/misc/events'
path_class = PROJECT_PATH + '/misc/concat_beh_data.csv'
epoch_path = OUT_PATH + '/Data_longWOBS_' + dataset_lab 

subj_included = [file.replace('_epochs.p', '') for file in os.listdir(epoch_path) if file[-len('epochs.p'):] == 'epochs.p']
subj_included = ExcludSubj(subj_included=subj_included, data_path=epoch_path)

subj_included_restricted = {16: ['BJH072', 'LL36', 'BJH069', 'SLCH020', 'BJH045', 'LL14', 'BJH050',
       'BJH027', 'OS70', 'BJH052', 'OS61', 'BJH041', 'BJH046', 'BJH056',
       'LL31', 'BJH039', 'BJH042', 'BJH026', 'LL08', 'BJH029',
       'DA037', 'BJH058', 'SLCH024', 'BJH049']}

path = epoch_path + f'/{subj_included[0]}_info.json'
with open(path) as json_data:
    d = json.load(json_data)
    time_tfr=d['time_tfr']

r_run  = 5000
nb_trials = 16
rng = np.random.default_rng(12345)

df_RT = extract_RT(subj_included=subj_included, path_behavior=path_behavior, path_info=epoch_path)

pca = PCA(3)
subj_list=subj_included_restricted[nb_trials]
data_grp = get_data_grp(subj_list, return_subj=False, data_path=epoch_path, polarity_cor=(False,0), type_data='tfr')[:, :, -1, :]
sorted_='RT'

find_tp = {'0_pc1': lambda x, tr : get_cross_point(tr, x, tstart=90, tend=150), 
           '0_pc2': lambda x, tr :  get_cross_point(tr, x, tstart=60, tend=115), 
           '0_pc3': lambda x, tr : get_cross_point(tr, x, tstart=120, tend=140), 
           'min_pc2': lambda x, tr : np.argmax(x[tr, 100:]) + 100, 
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

corr_list_perm=[]
rt_list_perm=[]

#corr_spearman = np.zeros((r_run, 3, 2, len(time_tfr)))
#p_spearman = np.zeros((r_run, 3, 2, len(time_tfr)))
#trans_save=np.zeros((r_run, 3, 2, nb_trials, len(time_tfr)))
trans_save = np.zeros((3, 2, nb_trials, len(time_tfr)),dtype=np.float32)
pca.fit(np.concat([data_grp[0, :,:], data_grp[1, :, :]], axis=-1).T)

# load subject TFR data
TFR_data = {}
for subj in subj_list:
    file = epoch_path + f'/{subj}_TFRtrials.p'
    with open(file, "rb") as f:
        TFR_data[subj] = pickle.load(f)[:, :, -1, :]


for r_ in range(r_run):
    if r_ % 100 == 0:
        print(f"Permutation {r_}/{r_run}", flush=True)

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

            TFRtr = TFR_data[subj]
            random_index = rt_condi_sorted.loc[:, 'index'].values
            supsubj[ic].append(TFRtr[random_index,:,:])
            rt_mean[ic].append(df.loc[random_index, 'RT'])
    rt_all=[]
    rt_all_perm=[]

    for condi in range(2):
        rt_condi=np.concat([r.values[None, :] for r in rt_mean[condi]], axis=0).mean(0)
        rt_condi_perm = rng.permutation(rt_condi)
        rt_all.append(rt_condi)
        rt_all_perm.append(rt_condi_perm)
        dataset=np.concat(supsubj[condi], axis=1)
        for pc_use in range(3) : 
            for tr in range(nb_trials) : 
                #trans_save[r_, pc_use, condi, tr, :] = dataset[tr, :, :].T @ pca.components_[pc_use, :]
                trans_save[pc_use, condi, tr, :] = dataset[tr, :, :].T @ pca.components_[pc_use, :]

            #for t in range(len(time_tfr)) : 
            #    corr_spearman[r_, pc_use, condi, t], p_spearman[r_, pc_use, condi, t] = spearmanr(trans_save[r_,pc_use,condi, :, t], rt_condi) 

    rt_list.append(rt_all)
    rt_list_perm.append(rt_all_perm)

    for to_t in to_try :
        trans = trans_save[int(to_t[-1:])-1, :, :, :] 
        #trans = trans_save[r_, int(to_t[-1:])-1, :, :, :]

        for i in range(2):
            if to_t in ['0_pc1','0_pc2', '0_pc3'] : 
                peaks = np.array([find_tp[to_t](trans[i], tr) for tr in range(trans[i].shape[0])])
                metr = np.asarray(time_tfr)[peaks]
                r, p = pearsonr(rt_all[i], metr)
                r_perm, p_perm = pearsonr(rt_all_perm[i], metr)
                metr_list.append(metr)
                corr_list.append((r, p, to_t, i, 'delay'))
                corr_list_perm.append((r_perm, p_perm, to_t, i, 'delay'))

            elif to_t in ['desc_slop_pc2', 'asc_slop_pc2', 'slop_pc3'] : 
                metr=[]
                for tr in range(trans[i].shape[0]) :
                    try : 
                        m = find_tp[to_t](trans[i], tr)
                    except : 
                        m = np.nan
                    metr.append(m)

                r, p = pearsonr(rt_all[i], metr)
                r_perm, p_perm = pearsonr(rt_all_perm[i], metr)
                corr_list.append((r, p, to_t, i, 'slope'))
                corr_list_perm.append((r_perm, p_perm, to_t, i, 'slope'))
                metr_list.append(metr)

            elif to_t in ['area_under_pc2', 'area_above_pc2'] :
                metr = np.array([find_tp[to_t](trans[i], tr) for tr in range(trans[i].shape[0])])
                r, p = pearsonr(rt_all[i], metr)
                r_perm, p_perm = pearsonr(rt_all_perm[i], metr)
                metr_list.append(metr)
                corr_list_perm.append((r_perm, p_perm, to_t, i, 'auc'))
                corr_list.append((r, p, to_t, i, 'auc'))

            elif to_t in ['min_pc2', 'bump_pc3'] : 
                peaks = np.array([find_tp[to_t](trans[i], tr) for tr in range(trans[i].shape[0])])
                ampl = trans[i][np.arange(trans[i].shape[0]), peaks]
                delay = np.asarray(time_tfr)[peaks]
                r_amp, p_amp = pearsonr(rt_all[i], ampl)
                r_del, p_del = pearsonr(rt_all[i], delay)
                r_amp_perm, p_amp_perm = pearsonr(rt_all_perm[i], ampl)
                r_del_perm, p_del_perm = pearsonr(rt_all_perm[i], delay)
                metr_list.extend([ampl, delay])

                corr_list.extend([(r_amp, p_amp, to_t, i, 'ampl'), (r_del, p_del, to_t, i, 'delay')])
                corr_list_perm.extend([(r_amp_perm, p_amp_perm, to_t, i, 'ampl'), (r_del_perm, p_del_perm, to_t, i, 'delay')])
        
corr = np.vstack(corr_list)
corr_perm = np.vstack(corr_list_perm)
metr = np.vstack(metr_list)
rt = np.stack(rt_list, axis=0)
rt_list_perm = np.stack(rt_list_perm, axis=0)
# save 
out_path = OUT_PATH + '/Behavior_' + dataset_lab
if not os.path.exists(out_path) : 
    os.makedirs(out_path)
    
with open(out_path + f"/tp_corr_{r_run}.pkl", "wb") as f:
    pickle.dump(corr, f)
with open(out_path + f"/tp_corr_{r_run}_perm.pkl", "wb") as f:
    pickle.dump(corr_perm, f)

with open(out_path + f"/tp_metr_{r_run}.pkl", "wb") as f:
    pickle.dump(metr, f)
with open(out_path + f"/tp_rt_{r_run}.pkl", "wb") as f:
    pickle.dump(rt, f)

with open(out_path + f"/tp_rt_{r_run}_perm.pkl", "wb") as f:
    pickle.dump(rt_list_perm, f)

#with open(out_path + f"/tp_trans_{r_run}{l}.pkl", "wb") as f:
    #pickle.dump(trans_save, f)
#with open(out_path + f"/spear_corr_{r_run}.pkl", "wb") as f:
#    pickle.dump(corr_spearman, f)
#with open(out_path + f"/spear_pval_{r_run}.pkl", "wb") as f:
#    pickle.dump(p_spearman, f)
