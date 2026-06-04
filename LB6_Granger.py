import numpy as np
import pandas as pd
import os
import json

from src.config import OUT_PATH
from src.decomposition import prep_data_trial
from src.analysis import compute_tr_gc_surrogate

tfr_path = OUT_PATH+ '/Data_longWOBS'

nb_trials = 16
subj_included_restricted = {16: ['BJH072', 'LL36', 'BJH069', 'SLCH020', 'BJH045', 'LL14', 'BJH050',
       'BJH027', 'OS70', 'BJH052', 'OS61', 'BJH041', 'BJH046', 'BJH056',
       'LL31', 'BJH039', 'BJH042', 'BJH026', 'LL08', 'SLCH018', 'BJH029',
       'DA037', 'BJH058', 'SLCH024', 'BJH049']}

path = tfr_path + f'/{subj_included_restricted[16][0]}_info.json'
with open(path) as json_data:
    d = json.load(json_data)
    time=d['time_tfr']

band = 'high_gamma'
method_pca = 'concat'

X_0,  X_0_old, X_0_new, y_0, true_trials = prep_data_trial(band, method_pca, None, subj_included_restricted[nb_trials], 0,tfr_path, nb_trials=nb_trials-1)
X_1,  X_1_old, X_1_new, y_1, true_trials = prep_data_trial(band, method_pca, None,  subj_included_restricted[nb_trials], 1,tfr_path, nb_trials=nb_trials-1)
X_2,  X_2_old, X_2_new, y_2, true_trials = prep_data_trial(band, method_pca, None, subj_included_restricted[nb_trials], 2,tfr_path, nb_trials=nb_trials-1)

X_=[X_0, X_1, X_2]
X_old=[X_0_old, X_1_old, X_2_old]
X_new=[X_0_new, X_1_new, X_2_new]

res_across_data = {}
lags = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15] 
window_len=15
method_perm = 'circular'   #circular, shuffle, block
n_perm=100

for X, n in zip([X_, X_old, X_new], ['both', 'old',  'new']):
    Res_all ={}
    for idx0, idx1 in [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]:
        lab=f'{idx0}to{idx1}'
        data=(X[idx0], X[idx1])
        Res_all[lab] = pd.DataFrame(columns = ['gc', 'bic', 'Fval', 'pval','p_emp', 'F_null','gc_s_p', 'gc_s_m', 'gc_s_sd', 'lag (tp)', 'lag (ms)', 'tmean', 'tstart', 'tend'])
        for e in range(len(time)-window_len):
            start = e
            tend = e+ window_len
            res = pd.DataFrame(columns = ['gc', 'bic', 'Fval', 'pval','p_emp', 'F_null', 'gc_s_p', 'gc_s_m', 'gc_s_sd'])
            for l, mlag in enumerate(lags) :
                if ((data[0].shape[0] * ((window_len) - mlag)) / (2*mlag) >= 10) and (mlag <= abs(window_len)):
                    res.loc[l, :] = compute_tr_gc_surrogate(x=data[0], y=data[1], start=start, end=tend, maxlag=mlag, z=None,n_perm=n_perm, perm=method_perm)
                else : 
                    res.loc[l, :] = [np.nan, np.nan,np.nan, np.nan , np.nan, np.nan,np.nan , np.nan, np.nan]

            res.loc[:, 'lag (tp)'] = lags
            res.loc[:, 'lag (ms)'] = [1000*l/40 for l in lags]
            res.loc[:, 'tmean'] = np.mean(time[start:tend])
            res.loc[:, 'tstart'] = time[start]
            res.loc[:, 'tend'] = time[tend]
            Res_all[lab].loc[e, :] = res.loc[np.argmin(res['bic']), :]

        path_to_save = OUT_PATH + f'/Granger/{band}/{lab}_{n}_{method_perm}{n_perm}.csv'
        Res_all[lab].to_csv(path_to_save)