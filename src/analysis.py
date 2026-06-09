import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.stats import spearmanr, pearsonr
from scipy.signal import find_peaks

from sklearn.decomposition import PCA
from numpy.linalg import lstsq
from scipy.stats import ttest_1samp, f
plt.style.use('seaborn-v0_8-dark')

from src.config import OUT_PATH
from src.setting import ExcludSubj

################################### COMPO ANALYSIS ################################### ok
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

def WeightSpearman(data1, data2, labels=[], figsize=(8, 6), ticks=False) : 
    if labels == [] : 
        labels = ['data1', 'data2']
    R = np.zeros((data1.shape[0], data2.shape[0]))
    P = np.zeros((data1.shape[0], data2.shape[0]))

    for i in range(data1.shape[0]):
        for j in range(data2.shape[0]):
            r, pval = spearmanr(data1[i, :], data2[j, :])
            R[i, j] = r
            P[i, j] = pval


    fig, ax = plt.subplots(figsize =figsize)
    sns.heatmap(abs(R), vmin=0, vmax=1, cmap='Greys', ax=ax, annot=True , fmt='.3f')
    if not ticks :
        x_ticks = ['PC' + str(i+1) for i in range(data2.shape[0])]
        y_ticks = ['PC' + str(i+1) for i in range(data1.shape[0])]
    else : 
        x_ticks = ticks[0]
        y_ticks = ticks[1]
    ax.set_xticklabels(x_ticks, rotation = 90)
    ax.set_xlabel(labels[1])
    ax.set_yticklabels(y_ticks, rotation = 0)
    ax.set_ylabel(labels[0])
    ax.set_title('Spearman correlation of weights')

################################### Granger and Cross Corr Between componants ###################################
def cross_cor(X_0, X_1) : 
    cc = []
    max_lag = X_0.shape[1] // 2
    lags = np.arange(-max_lag, max_lag, step=10)

    for l in lags:
        r = np.zeros(len(X_0))
        for i in range(len(X_0)):
            
            if l < 0:
                x = X_0[i, :l]          # remove last |l| points
                y = X_1[i, -l:]         # remove first |l| points
            elif l > 0:
                x = X_0[i, l:]          # remove first l points
                y = X_1[i, :-l]         # remove last l points
            else:
                x = X_0[i, :]
                y = X_1[i, :]
            
            r[i], _ = spearmanr(x, y)

        z = np.arctanh(r)
        t_stat, p_val = ttest_1samp(z, 0)
        cc.append((t_stat, p_val))
        
    return lags, cc

def compute_tr_gc_win(x, y, time_tfr,window_len,step_len,maxlag,  z=None):
    
    n_trials, n_times = x.shape
    win_samples = window_len
    step_samples = step_len
    
    n_windows = (n_times - win_samples) // step_samples + 1
    gc_time = np.zeros(n_windows)
    times = np.zeros(n_windows)
    bics = np.zeros(n_windows)
    Fval = np.zeros(n_windows)
    pval = np.zeros(n_windows)

    for w in range(n_windows):
        start = w * step_samples
        end = start + win_samples
        Y_window = []
        
        for tr in range(n_trials):
            xt = x[tr, start:end] - np.mean(x[tr, start:end])
            yt = y[tr, start:end] - np.mean(y[tr, start:end])
            
            if z is not None:
                zt = z[tr, start:end] - np.mean(z[tr, start:end])
                data_tr = np.column_stack([yt, xt, zt])
            else:
                data_tr = np.column_stack([yt, xt])
            
            Y_window.append(data_tr)
        
        data_window = np.vstack(Y_window)
        T_window = data_window.shape[0]
        
        Y_dep = data_window[maxlag:, 0]  # current Y values
        
        # Build full design matrix with lags
        X_full = np.ones((T_window - maxlag, 1))  # intercept
        for lag in range(1, maxlag + 1):
            X_full = np.column_stack([X_full, data_window[maxlag - lag:-lag, 0]])  # Y lag
            X_full = np.column_stack([X_full, data_window[maxlag - lag:-lag, 1]])  # X lag
            if z is not None:
                X_full = np.column_stack([X_full, data_window[maxlag - lag:-lag, 2]])  # Z lag
        
        # Full model
        beta_f, _, _, _ = lstsq(X_full, Y_dep, rcond=None)
        resid_f = Y_dep - X_full @ beta_f
        sigma_f = np.var(resid_f)
        
        # Reduced model (remove X and Z lags)
        cols_keep = [0]  
        for lag in range(maxlag):
            cols_keep.append(1 + lag * (2 if z is None else 3))  # Y lag columns only
        X_r = X_full[:, cols_keep]
        beta_r, _, _, _ = lstsq(X_r, Y_dep, rcond=None)
        resid_r = Y_dep - X_r @ beta_r
        sigma_r = np.var(resid_r)
        
        gc_time[w] = np.log(sigma_r / sigma_f)
        times[w] = time_tfr[start:end].mean()

        # Fvalues 
        RSS_f = np.sum(resid_f**2)
        RSS_r = np.sum(resid_r**2)
        q = X_full.shape[1] - X_r.shape[1]
        Fval[w] = ((RSS_r - RSS_f) / q) / (RSS_f / (len(Y_dep) - X_full.shape[1]))
        pval[w] = 1 - f.cdf(Fval[w], q, len(Y_dep) - X_full.shape[1])

        # BIC
        bics[w] = np.log(sigma_f) + (X_full.shape[1] * np.log(len(Y_dep))) / len(Y_dep)
        
    return gc_time, times, bics, Fval, pval

def compute_tr_gc(x, y, start,end, maxlag, z=None, perm = None):
  
    n_trials, _ = x.shape
    Y_window = []
    
    for tr in range(n_trials):
        xt = x[tr, start:end] - np.mean(x[tr, start:end])

        if perm == 'circular' : 
            shift = np.random.randint(1, len(xt))
            xt = np.roll(xt, shift)
        elif perm == 'shuffle':
            xt=np.random.permutation(xt)
        elif perm == 'block' : 
            block_size = 5
            blocks = np.array_split(xt,np.arange(block_size, len(xt), block_size))
            np.random.shuffle(blocks)
            xt = np.concatenate(blocks)

        yt = y[tr, start:end] - np.mean(y[tr, start:end])
        
        if z is not None:
            zt = z[tr, start:end] - np.mean(z[tr, start:end])
            data_tr = np.column_stack([yt, xt, zt])
        else:
            data_tr = np.column_stack([yt, xt])
        
        Y_window.append(data_tr)
    
    data_window = np.vstack(Y_window)
    T_window = data_window.shape[0]
    
    Y_dep = data_window[maxlag:, 0]  # current Y values
    
    # Build full design matrix with lags
    X_full = np.ones((T_window - maxlag, 1))  # intercept
    for lag in range(1, maxlag + 1):
        X_full = np.column_stack([X_full, data_window[maxlag - lag:-lag, 0]])  # Y lag
        X_full = np.column_stack([X_full, data_window[maxlag - lag:-lag, 1]])  # X lag
        if z is not None:
            X_full = np.column_stack([X_full, data_window[maxlag - lag:-lag, 2]])  # Z lag
    
    # Full model
    beta_f, _, _, _ = lstsq(X_full, Y_dep, rcond=None)
    resid_f = Y_dep - X_full @ beta_f
    sigma_f = np.var(resid_f)
    
    # Reduced model (remove X and Z lags)
    cols_keep = [0]  
    for lag in range(maxlag):
        cols_keep.append(1 + lag * (2 if z is None else 3))  # Y lag columns only
    X_r = X_full[:, cols_keep]
    beta_r, _, _, _ = lstsq(X_r, Y_dep, rcond=None)
    resid_r = Y_dep - X_r @ beta_r
    sigma_r = np.var(resid_r)
    
    gc = np.log(sigma_r / sigma_f)

    # Stats
    RSS_f = np.sum(resid_f**2)
    RSS_r = np.sum(resid_r**2)
    q = X_full.shape[1] - X_r.shape[1]
    Fval = ((RSS_r - RSS_f) / q) / (RSS_f / (len(Y_dep) - X_full.shape[1]))
    pval = 1 - f.cdf(Fval, q, len(Y_dep) - X_full.shape[1])

    # BIC
    bic = np.log(sigma_f) + (X_full.shape[1] * np.log(len(Y_dep))) / len(Y_dep)

    return gc, bic, Fval, pval

def compute_tr_gc_surrogate(x, y, start, end, maxlag, z=None,n_perm=2, perm=None):
    gc_obs, bic_obs, F_obs, p_obs = compute_tr_gc(x, y, start, end, maxlag, z)
    F_null = np.zeros(n_perm)
    Gc_null=np.zeros(n_perm)
    for iperm in range(n_perm):
        gc_s, _, F_s, _ = compute_tr_gc(x,y,start,end,maxlag,z, perm=perm)
        F_null[iperm] = F_s
        Gc_null[iperm] = gc_s
    p_emp = (np.sum(F_null >= F_obs) + 1) / (n_perm + 1)

    return gc_obs,bic_obs, F_obs, p_obs, p_emp, np.percentile(F_null, 95),np.mean(F_null),np.std(F_null), np.percentile(Gc_null, 95), np.mean(Gc_null), np.std(Gc_null)

################################### VIZ AND INTRO (CHAT) ###################################
    
def PolarChannel(data1, title="Channels", elects=[], subjs = [], cmap_name = 'Blues', to_black = [], data_path = OUT_PATH + '/Data'):
    C, T = data1.shape
    cmap = cm.get_cmap(cmap_name, C)
    theta = np.linspace(0, 2*np.pi, C, endpoint=False)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
    subj_included = ExcludSubj(subj_included, data_path=data_path)
    with open(data_path + f'/{subj_included[0]}_info.json') as json_data:
        d = json.load(json_data)
        time = d['time_tfr']
        json_data.close()
    z = time
    
    for ch in range(C):
        amp = data1[ch]            
        th  = theta[ch]             
        x = amp * np.cos(th)
        y = amp * np.sin(th)

        color = cmap(0.3 + (1 - 0.3) * (ch / (C - 1))) 
        ax.plot(z, y, x, alpha=0.9, label=f"Ch {elects[ch].replace('`', '')} subj {subjs[ch]}", color=color)

    if len(to_black) != 0 :
        for i in to_black : 
            amp = data1[i]            
            th  = theta[i]             
            x = amp * np.cos(th)
            y = amp * np.sin(th)
            ax.scatter(z, y,x,s=2, alpha=0.9, label=f"Ch {elects[i].replace('`', '')} subj {subjs[i]}", color='black')
        
    ax.legend(bbox_to_anchor=(1.6, 1))
    ax.set_title(title)
    ax.set_zlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("Time")
    plt.show()

def PolarChannelWithPCA(data1, title="Channels", elects=[], subjs=[], cmap_name='Blues', data_full=None, idx_picks=None, data_path = OUT_PATH + '/Data'):
    C, T = data1.shape
    theta = np.linspace(0, 2*np.pi, C, endpoint=False)
    cmap = cm.get_cmap(cmap_name, C)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file.endswith('TFRtrials.p')]
    subj_included = ExcludSubj(subj_included, data_path=data_path)
    with open(data_path + f'/{subj_included[0]}_info.json') as json_data:
        d = json.load(json_data)
        time = d['time_tfr']

    z = time 
    for ch in range(C):
        amp = data1[ch]
        th = theta[ch]
        x = amp * np.cos(th)
        y = amp * np.sin(th)
        color = cmap(0.3 + 0.7 * (ch / (C - 1)))
        ax.plot(z, y, x, alpha=0.9, label=f"Ch {elects[ch]} subj {subjs[ch]}", color=color)

    pca = PCA(n_components=2)
    pca.fit(data_full.T)  # transpose: samples x features
    components = pca.components_  # shape (2, C)

    scale = 40
    for i, pc in enumerate(components):
        pc_x = np.sum(pc[idx_picks] * np.cos(theta)) * scale
        pc_y = np.sum(pc[idx_picks] * np.sin(theta)) * scale
        pc_z = 0 # draw arrow at z=0 (bottom of plot)

        ax.quiver(0, 0, pc_z, pc_y, pc_x, 0, color='red' if i==0 else 'blue', 
                  linewidth=2, arrow_length_ratio=0.2, label=f'PC{i+1}')

    ax.set_title(title)
    ax.set_zlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)
    ax.set_xlabel("Time")
    ax.legend(bbox_to_anchor=(1.6, 1))
    plt.show()

def PolarChannelPloty(data, title="Channels", elects='', subjs='', cmap_name = 'Blues', data_path = OUT_PATH + '/Data'):
    C, T = data.shape
    cmap = cm.get_cmap(cmap_name, C)
    theta = np.linspace(0, 2 * np.pi, C, endpoint=False)
    subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(data_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
    subj_included = ExcludSubj(subj_included, data_path=data_path)
    with open(data_path + f'/{subj_included[0]}_info.json') as json_data:
        d = json.load(json_data)
        time = d['time_tfr']
        json_data.close()

    z = time
    fig = go.Figure()
    for ch in range(C):
        amp = data[ch]                # radius = amplitude
        th  = theta[ch]               # fixed angle
        
        x = amp * np.cos(th)
        y = amp * np.sin(th)
        import matplotlib.colors as mcolors

        c = cmap(0.1 + (1 - 0.1) * (ch / (C - 1))) 
        color = mcolors.to_hex(c)

        fig.add_trace(go.Scatter3d(
            x=z,
            y=y,
            z=x,
            mode="lines",
            line=dict(width=4),
            name=f"Ch {elects[ch]} subj {subjs[ch]}", 
            line_color=color
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            zaxis_title="",
            yaxis_title=")",
            xaxis_title="Time",
            aspectmode="cube"
        ),
        showlegend=True,
        width=900,
        height=800
    )

    fig.show()

def PolarChannelSequential(data1, data2, title="Channels", elects=[], subjs=[], data_path = OUT_PATH + '/Data'):
    C, T1 = data1.shape
    _, T2 = data2.shape
    theta = np.linspace(0, 2*np.pi, C, endpoint=False)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    subj_included = [file.replace('_TFRtrials.p', '') 
                     for file in os.listdir(data_path) 
                     if file.endswith('TFRtrials.p')]
    
    subj_included = ExcludSubj(subj_included, data_path=data_path)
    with open(data_path + f'/{subj_included[0]}_info.json') as json_data:
        d = json.load(json_data)
        time1 = np.array(d['time_tfr'])
    
    z1 = time1
    z2 = time1[-1] + np.diff(time1).mean() + time1 
    cmap1 = cm.get_cmap("Blues")
    cmap2 = cm.get_cmap("Reds")

    for ch in range(C):
        amp = data1[ch]
        th = theta[ch]
        x = amp * np.cos(th)
        y = amp * np.sin(th)
        color = cmap1(0.3 + 0.7 * (ch / (C - 1)))
        ax.plot(z1, y, x, color=color, alpha=0.9,
                label=f"Ch {elects[ch]} subj {subjs[ch]} (data1)")

    for ch in range(C):
        amp = data2[ch]
        th = theta[ch]
        x = amp * np.cos(th)
        y = amp * np.sin(th)
        color = cmap2(0.3 + 0.7 * (ch / (C - 1)))
        ax.plot(z2, y, x, color=color, alpha=0.9,
                label=f"Ch {elects[ch]} subj {subjs[ch]} (data2)")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.zaxis.set_visible(False)
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.legend(bbox_to_anchor=(1.6, 1))
    plt.show()

