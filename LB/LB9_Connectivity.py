import pandas as pd
import numpy as np
import json
import os
import pickle

from scipy.stats import spearmanr, zscore, f, ttest_1samp
from scipy.signal import welch, csd
from scipy.linalg import lstsq
from scipy.signal import hilbert, butter, sosfiltfilt

from src.config import OUT_PATH, FREQ_BAND, EVENT_ID, events
from src.setting import GetInfo, ExcludSubj

def group_TFRm(band, subj_list, data_ch, tfr_path, events) :  

    trials ={ev:[] for ev in events}
    for subj in np.unique(subj_list) :
        data_subj = data_ch.query('subj == @subj').reset_index()
        ch = data_subj.query("ch1 == True").index

        file = tfr_path + f'/{subj}_TFRtrials.p'
        with open(file, "rb") as f:
            tr = pickle.load(f)

        info_file = tfr_path + f'/{subj}_info.json'
        with open(info_file) as f:
            info = json.load(f)
            events_index = np.array([int(i) for i in info['event_id']])

        id_ch = np.array(ch)
        for ev in events : 
            id_ev = np.where(events_index == EVENT_ID[ev])[0]

            if len(id_ch) == 0 : 
                continue
            tr_band = tr[np.ix_(id_ev, id_ch,[FREQ_BAND.index(band)], range(tr.shape[-1]))]
            trials[ev].append(tr_band)
    return trials

def bandpass(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data

def compute_aec(x, y, fisher=True):
    # Pearson product-moment correlation coefficients between envelops.
    n_trials = x.shape[0]
    corrs = []

    for t in range(n_trials):
        xt = zscore(x[t])
        yt = zscore(y[t])
        r = np.corrcoef(xt, yt)[0, 1]
        corrs.append(r)

    corrs = np.array(corrs)

    if fisher: # normalize to run the stat test with 0
        corrs_z = np.arctanh(corrs)
        tval, pval = ttest_1samp(corrs_z, 0)
        mean_z = np.mean(corrs_z)
        mean_r = np.tanh(mean_z)  
    else:
        tval, pval = ttest_1samp(corrs, 0)
        mean_r = np.mean(corrs)

    return mean_r, pval

def compute_plv(x_phase, y_phase):
    phase_diff = x_phase - y_phase  
    complex_phase = np.exp(1j * phase_diff)
    plv_time = np.abs(np.mean(complex_phase, axis=0))  
    return np.mean(plv_time)  

def compute_granger(x, y, maxlag=10): # on raw of filtered before hilbert, F-statistic across trials

    n_trials, T = x.shape
    Fvals = []

    for tr in range(n_trials):
        xt = x[tr]
        yt = y[tr]

        # Build lag matrix
        Y = yt[maxlag:]
        X_full = []
        X_restricted = []

        for lag in range(1, maxlag+1):
            X_full.append(yt[maxlag-lag:-lag])
            X_full.append(xt[maxlag-lag:-lag])
            X_restricted.append(yt[maxlag-lag:-lag])

        X_full = np.column_stack(X_full)
        X_restricted = np.column_stack(X_restricted)

        beta_f, _, _, _ = lstsq(X_full, Y)
        beta_r, _, _, _ = lstsq(X_restricted, Y)

        rss_f = np.sum((Y - X_full @ beta_f)**2)
        rss_r = np.sum((Y - X_restricted @ beta_r)**2)

        df1 = maxlag
        df2 = len(Y) - 2*maxlag

        Fval = ((rss_r - rss_f)/df1) / (rss_f/df2)
        Fvals.append(Fval)

    return np.mean(Fvals)

def compute_psi(x, y, fs, fmin, fmax, nperseg=256): # on raw of filtered before hilbert
    psi_vals = []

    for tr in range(x.shape[0]):
        f, Pxy = csd(x[tr], y[tr], fs=fs, nperseg=nperseg)
        _, Pxx = welch(x[tr], fs=fs, nperseg=nperseg)
        _, Pyy = welch(y[tr], fs=fs, nperseg=nperseg)

        Sxy = Pxy / np.sqrt(Pxx * Pyy)

        freq_mask = (f >= fmin) & (f <= fmax)
        Sxy_band = Sxy[freq_mask]

        psi = np.sum(np.imag(np.conj(Sxy_band[:-1]) * Sxy_band[1:]))
        psi_vals.append(psi)

    return np.mean(psi_vals)

def group_hilbert(band, param_filter,subj_list, data_ch, tfr_path, events,spl_rate=200 ): 
    trials_env ={ev:[] for ev in events}
    trials_phase ={ev:[] for ev in events}

    for subj in np.unique(subj_list):
        data_subj = data_ch.query('subj == @subj').reset_index()
        ch = data_subj.query("ch1 == True").index

        file = tfr_path + f'/{subj}_epochs.p'
        with open(file, "rb") as f:
            tr = pickle.load(f)
            analog_env = np.zeros_like(tr)
            analog_phase = np.zeros_like(tr)
            for c in range(tr.shape[1]) : 
                for e in range(tr.shape[0]) :
                    analog = hilbert(bandpass(tr[e, c,:], param_filter[band], spl_rate))
                    analog_env[e, c, :] = abs(analog)
                    analog_phase[e, c, :] = np.angle(analog) 

        info_file = tfr_path + f'/{subj}_info.json'
        with open(info_file) as f:
            info = json.load(f)
            events_index = np.array([int(i) for i in info['event_id']])

        id_ch = np.array(ch)
        for ev in events : 
            id_ev = np.where(events_index == EVENT_ID[ev])[0]

            if len(id_ch) == 0 : 
                continue
            tr_band_env = analog_env[np.ix_(id_ev, id_ch, range(tr.shape[-1]))]
            tr_band_phase = analog_phase[np.ix_(id_ev, id_ch, range(tr.shape[-1]))]

            trials_env[ev].append(tr_band_env)
            trials_phase[ev].append(tr_band_phase)

    return trials_env, trials_phase

def group_time_domain(band, param_filter,subj_list, data_ch, tfr_path, events,spl_rate=200 ): 
    trials ={ev:[] for ev in events}

    for subj in np.unique(subj_list):
        data_subj = data_ch.query('subj == @subj').reset_index()
        ch = data_subj.query("ch1 == True").index

        file = tfr_path + f'/{subj}_epochs.p'
        with open(file, "rb") as f:
            tr = pickle.load(f)
            filtered = np.zeros_like(tr)
            for c in range(tr.shape[1]) : 
                for e in range(tr.shape[0]) :
                    filtered[e, c,:] = bandpass(tr[e, c,:], param_filter[band], spl_rate)
                    
        info_file = tfr_path + f'/{subj}_info.json'
        with open(info_file) as f:
            info = json.load(f)
            events_index = np.array([int(i) for i in info['event_id']])

        id_ch = np.array(ch)
        for ev in events : 
            id_ev = np.where(events_index == EVENT_ID[ev])[0]

            if len(id_ch) == 0 : 
                continue
            tr_band = filtered[np.ix_(id_ev, id_ch, range(tr.shape[-1]))]

            trials[ev].append(tr_band)

    return trials

def compute_connectivity(band, pc, rois, trials,method, events, param_filter=None, spl_rate=None, out_path=OUT_PATH, tw=['all']):
    pval_bool = False
    for ev in events :
        data =[]
        for tr in trials[ev] : 
            data.extend([tr[:,c,:] for c in range(tr.shape[1])])

        for t_win in tw : 
            if t_win == 'all' : 
                tstart = 0
                tend=-1
                lab = 'all'
            else :
                tstart = t_win[0]
                tend=t_win[1]
                lab = f'{tstart}_{tend}'

            print('Computing connectivity: ', ev, method, pc, lab)
            con = np.zeros([len(data), len(data)])
            pval = np.zeros([len(data), len(data)])
            
            for i,x_ in enumerate(data):
                x = x_[:, tstart:tend]
                for j,y_ in enumerate(data) :
                    min_trial = np.min([x_.shape[0], y_.shape[0]])
                    y = y_[:, tstart:tend]
                    if method == 'spearman' :
                        con[i, j] = np.mean(spearmanr(x[:min_trial, :], y[:min_trial, :]).statistic)  #np.mean(spearmanr(x[:min_trial, :], y[:min_trial, :]).statistic) 

                    if method == 'aec' :
                        pval_bool = True
                        con[i, j], pval[i, j] = compute_aec(x[:min_trial, :], y[:min_trial, :])

                    if method =='plv':
                        con[i, j] = compute_plv(x[:min_trial, :], y[:min_trial, :])

                    if method == 'psi' :
                        con[i, j] = compute_psi(x[:min_trial, :], y[:min_trial, :], fs=spl_rate, fmin=param_filter[band][0], fmax=param_filter[band][1])

                    if method == 'granger' :
                        con[i, j] = compute_granger(x[:min_trial, :], y[:min_trial, :])
                        
            df = pd.DataFrame(con, columns = rois)
            df['rois'] = rois
            df= df.set_index('rois')
            df = df.reset_index().groupby('rois').mean().T.reset_index().groupby('index').mean().T
            df.to_csv(out_path+ f'/{method}_{pc}_{ev[:3]}_{lab}.csv')

            if pval_bool:
                df2 = pd.DataFrame(con, columns = rois)
                df2['rois'] = rois
                df2= df2.set_index('rois')
                #df2 = df2.reset_index().groupby('rois').mean().T.reset_index().groupby('index').mean().T
                df2.to_csv(out_path+ f'/{method}_pval_{pc}_{ev[:3]}_{lab}_ch.csv')


if __name__ == "__main__":
    param_filter={'high_gamma' : [55, 99]}
    spl_rate=200
    tfr_path = OUT_PATH+ '/Data_longWOBS'
    subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(tfr_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
    subj_included = ExcludSubj(subj_included, data_path=tfr_path)
    path = tfr_path + f'/{subj_included[0]}_info.json'
    with open(path) as json_data:
        d = json.load(json_data)
        time_tfr=d['time_tfr']
        
    coord, areas, _, subj_list, regions = GetInfo(subj_included, data_path=tfr_path) 

    band = 'high_gamma'
    pc='compo3'
    method_pca = 'concat'
    tw= [(25, 67), (67, 109)] #[(15, 73), (73, 131)]  #compo2 [(25, 67), (67, 109)] #compo3

    #if not pc :
    weight = pd.read_csv(OUT_PATH + f'/grpPCA/supsubj_{method_pca}/grp_{method_pca}_Compo_PCA5.csv').query('freq == @band & compo == @pc').drop(columns = ['Unnamed: 0', 'freq', 'compo']).values[0, :]
    data_ch = pd.DataFrame()
    data_ch['ch1'] = np.where(abs(weight) > abs(weight).mean() + abs(weight).std(), True, False)
    # else : 
        #data_ch = pd.DataFrame()
        #data_ch['ch1'] = [True]*len(subj_list)

    data_ch['subj'] = subj_list
    data_ch['ROIs'] = [a[0] for a in areas] #regions
    rois = data_ch[data_ch['ch1'] == True]['ROIs'].values

    out_path = OUT_PATH+f'/Connectivity_area/{band}'
    if not os.path.exists(out_path) :
        os.makedirs(out_path)

    trials_env, trials_phase = group_hilbert(band=band, param_filter=param_filter, subj_list=subj_list, data_ch=data_ch,tfr_path=tfr_path, events=events)
    compute_connectivity(band=band, pc=pc, rois=rois, trials=trials_env,method='aec', events=events, out_path=out_path, tw=tw)
    #compute_connectivity(band=band, pc=pc, rois=rois, trials=trials_phase,method='plv', events=events, out_path=out_path, tw=tw)

    trials_time = group_time_domain(band=band, param_filter=param_filter, subj_list=subj_list, data_ch=data_ch, tfr_path=tfr_path, events=events)
    #compute_connectivity(band=band, pc=pc, rois=rois, trials=trials_time, method='granger', events=events, out_path=out_path, tw=tw)
    #compute_connectivity(band=band, pc=pc, rois=rois, trials=trials_time,method='psi', events=events, param_filter=param_filter, spl_rate=spl_rate, out_path=out_path, tw=tw)

