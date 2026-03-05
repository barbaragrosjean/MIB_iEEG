import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from utils import OUT_PATH
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.stats import ttest_1samp
import json
import matplotlib.gridspec as gridspec
from numpy.linalg import lstsq

from utils import ExcludSubj, DataTransformationM1

tfr_path = OUT_PATH+ '/Data_longWOBS'
subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(tfr_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
subj_included = ExcludSubj(subj_included, data_path=tfr_path)


path = tfr_path + f'/{subj_included[0]}_info.json'
with open(path) as json_data:
    d = json.load(json_data)
    time_tfr=d['time_tfr']

col_pc = {0:"#0072B2", 1:"#D55E00", 2:"#009E73", 4:"#009E73"}

    
def prep_data(band, method_pca, data_aug_method, subj_included, PC_use,tfr_path) : 
    X_train0, y_train0, X_test0, y_test0, _, _ = DataTransformationM1(freq= band, method_pca=method_pca, data_aug_method=data_aug_method, subj_included=subj_included, PC_use=PC_use, data_path=tfr_path)      

    X_0 = np.concat([X_train0, X_test0], axis=0)
    y_0 = np.concat([y_train0, y_test0], axis=0)
    X_0_old = X_0[np.where(y_0 == 1)]
    X_0_new = X_0[np.where(y_0 == 2)]
    
    return X_0, X_0_old, X_0_new, y_0

band = 'high_gamma'
method_pca = 'concat'
data_aug_method = 'duplicat'

X_0,  X_0_old, X_0_new, y_0 = prep_data(band, method_pca, data_aug_method, subj_included, 0,tfr_path)
X_1,  X_1_old, X_1_new, y_1 = prep_data(band, method_pca, data_aug_method, subj_included, 1,tfr_path)
X_2,  X_2_old, X_2_new, y_2 = prep_data(band, method_pca, data_aug_method, subj_included, 2,tfr_path)


def compute_tr_gc(x, y, time_tfr, window_len=20, step_len=2, maxlag=10):
    """
    Compute time-resolved Granger causality (X -> Y) for iEEG trials using
    window length and step in number of time points.

    Parameters
    ----------
    x : np.ndarray
        Shape (n_trials, n_times), first signal (X)
    y : np.ndarray
        Shape (n_trials, n_times), second signal (Y)
    time_tfr : np.ndarray
        Real time points for each sample, shape (n_times,)
    window_len : int
        Length of sliding window in **samples** (time points)
    step_len : int
        Step size in **samples**
    maxlag : int
        Maximum VAR order in samples

    Returns
    -------
    gc_time : np.ndarray
        Shape (n_windows,), GC values per time window
    times : np.ndarray
        Center time of each window using `time_tfr`
    """

    n_trials, n_times = x.shape
    win_samples = window_len
    step_samples = step_len
    
    n_windows = (n_times - win_samples) // step_samples + 1
    gc_time = np.zeros(n_windows)
    times = np.zeros(n_windows)

    for w in range(n_windows):
        start = w * step_samples
        end = start + win_samples
        Y_window = []
        
        # Collect data per trial
        for tr in range(n_trials):
            xt = x[tr, start:end]
            yt = y[tr, start:end]
            
            # Demean for stationarity
            xt = xt - np.mean(xt)
            yt = yt - np.mean(yt)
            
            data_tr = np.column_stack([yt, xt])  # order: Y, X
            Y_window.append(data_tr)
        
        # Concatenate trials along time for this window
        data_window = np.vstack(Y_window)
        T_window = data_window.shape[0]
        
        if T_window <= maxlag:
            raise ValueError("Window too short for chosen maxlag")
        
        # Dependent variable
        Y_dep = data_window[maxlag:, 0]  # Y equation only
        
        # Build full design matrix (intercept + Y lags + X lags)
        X_full = np.ones((T_window - maxlag, 1))  # intercept
        for lag in range(1, maxlag + 1):
            X_full = np.column_stack([X_full,
                                      data_window[maxlag - lag:-lag, 0],  # Y lags
                                      data_window[maxlag - lag:-lag, 1]])  # X lags
        
        # Fit full model
        beta_f, _, _, _ = lstsq(X_full, Y_dep, rcond=None)
        resid_f = Y_dep - X_full @ beta_f
        sigma_f = np.var(resid_f)
        
        # Build restricted model (remove X lags)
        cols_keep = [0]  # intercept
        for lag in range(maxlag):
            cols_keep.append(1 + lag * 2)  # Y lag columns only
        X_r = X_full[:, cols_keep]
        
        # Fit restricted model
        beta_r, _, _, _ = lstsq(X_r, Y_dep, rcond=None)
        resid_r = Y_dep - X_r @ beta_r
        sigma_r = np.var(resid_r)
        
        # Compute log-variance GC (Geweke)
        gc_time[w] = np.log(sigma_r / sigma_f)
        
        # Use time_tfr to define window center
        times[w] = time_tfr[start:end].mean()

    return gc_time, times


window_len, step_len, maxlag = 60, 6, 5

fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3, height_ratios=[2/5, 3/5])
ax_TS = fig.add_subplot(gs[0, :])  
ax = fig.add_subplot(gs[1, 0])
ax_old = fig.add_subplot(gs[1, 1])
ax_new = fig.add_subplot(gs[1, 2])

lab = ['PC2', 'PC3']

ax_TS.plot(time_tfr, np.mean(X_1, axis=0), ls='-', c=col_pc[1], label=lab[0])
ax_TS.plot(time_tfr, np.mean(X_2, axis=0), ls='-', c=col_pc[2], label=lab[1])
ax_TS.set_ylabel('PCs')
ax_TS.set_xlabel('Time (s)')

axes = [ax, ax_old, ax_new]
name_lab = ['full', 'old', 'new']
marker = ['x', 'o', '^']

for i, (x1, x2) in enumerate([(X_1, X_2), (X_1_old, X_2_old), (X_1_new, X_2_new)]): 
    
    x1_sh = np.random.permutation(x1.T).T
    x2_sh = np.random.permutation(x2.T).T
    gc_x2y_sh, times = compute_tr_gc(x1_sh, x2_sh, time_tfr=np.array(time_tfr), window_len=window_len, step_len=step_len, maxlag=maxlag)
    gc_y2x_sh, _ = compute_tr_gc(x2_sh, x1_sh, time_tfr=np.array(time_tfr), window_len=window_len, step_len=step_len, maxlag=maxlag)
    axes[i].plot(times, gc_x2y_sh, label=f'{lab[0]} → {lab[1]} shuffled', c='grey', ls=':', linewidth = 0.8)
    axes[i].plot(times, gc_y2x_sh, label=f'{lab[1]} → {lab[0]} shuffled',  c='k', ls=':', linewidth = 0.8)

    gc_x2y, times = compute_tr_gc(x1, x2, time_tfr=np.array(time_tfr), window_len=window_len, step_len=step_len, maxlag=maxlag)
    gc_y2x, _ = compute_tr_gc(x2, x1, time_tfr=np.array(time_tfr), window_len=window_len, step_len=step_len, maxlag=maxlag)

    ax_TS.scatter([times[gc_x2y.argmax()]], [5+i*0.5], c='grey', marker = marker[i], label=name_lab[i])
    ax_TS.scatter([times[gc_y2x.argmax()]], [5+i*0.5], c='k',marker = marker[i],  label=name_lab[i])

    axes[i].plot(times, gc_x2y, label=f'{lab[0]} → {lab[1]}', c='grey', ls='-')
    axes[i].plot(times, gc_y2x, label=f'{lab[1]} → {lab[0]}',  c='k', ls='-')
    axes[i].set_ylim(0, 0.7)
    axes[i].set_ylabel('GC (log-variance)')
    axes[i].set_xlabel('Time (s)')
    axes[i].set_title(name_lab[i])
    axes[i].grid()
axes[i].legend(ncol =4, bbox_to_anchor = (-0.75, -0.25), loc='center')
ax_TS.grid()
ax_TS.legend()

