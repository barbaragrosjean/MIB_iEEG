import pandas as pd
import numpy as np
import json
import os
import pickle
from scipy.stats import spearmanr, zscore, f
from scipy.signal import welch, csd
from scipy.linalg import lstsq

from utils import OUT_PATH, GetInfo, ExcludSubj, FREQ_BAND, EVENT_ID
from scipy.signal import hilbert, butter, sosfiltfilt

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

def compute_aec(x, y, fisher=False): # for env
    n_trials = x.shape[0]
    corrs = []

    for t in range(n_trials):
        xt = zscore(x[t])
        yt = zscore(y[t])
        r = np.corrcoef(xt, yt)[0, 1]
        corrs.append(r)

    corrs = np.array(corrs)

    if fisher:
        corrs = np.arctanh(corrs)

    return np.mean(corrs)

def compute_plv(x_phase, y_phase): # on phase
    n_trials = x_phase.shape[0]
    plvs = []

    for t in range(n_trials):
        phase_diff = x_phase[t] - y_phase[t]
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        plvs.append(plv)

    return np.mean(plvs)

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

    for subj in np.unique(subj_list)[:1]:
        data_subj = data_ch.query('subj == @subj').reset_index()
        ch = data_subj.query("ch1 == True").index

        file = tfr_path + f'/{subj}_epochs.p'
        with open(file, "rb") as f:
            tr = pickle.load(f)
            filtered = np.zeros_like(tr)
            for c in range(tr.shape[1]) : 
                for e in range(tr.shape[0]) :
                    filtered = bandpass(tr[e, c,:], param_filter[band], spl_rate)
                    
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

def compute_connectivity(band, pc, rois, trials,method, events, param_filter=None, spl_rate=None):
    for ev in events :
        data =[]
        for tr in trials[ev] : 
            data.extend([tr[:,c,:] for c in range(tr.shape[1])])

        print('Computing static connectivity: ', ev, method)
        con = np.zeros([len(data), len(data)])
        for i,x in enumerate(data):
            for j,y in enumerate(data) :
                min_trial = np.min([x.shape[0], y.shape[0]])

                if method == 'spearman' :
                    con[i, j] = np.mean(spearmanr(x[:min_trial, :], y[:min_trial, :]).statistic)  #np.mean(spearmanr(x[:min_trial, :], y[:min_trial, :]).statistic) 

                if method == 'aec' :
                    con[i, j] = compute_aec(x[:min_trial, :], y[:min_trial, :])

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
        df.to_csv(OUT_PATH + f'/Connectivity/{band}/{method}_{pc}_{ev[:3]}_all.csv')

if __name__ == "__main__":
    events = ['old/correct', 'new/correct']
    param_filter={'high_gamma' : [55, 99]}
    spl_rate=200
    tfr_path = OUT_PATH+ '/Data_longWOBS'
    subj_included = [file.replace('_TFRtrials.p', '') for file in os.listdir(tfr_path) if file[-len('TFRtrials.p'):] == 'TFRtrials.p']
    subj_included = ExcludSubj(subj_included, data_path=tfr_path)
    path = tfr_path + f'/{subj_included[0]}_info.json'
    with open(path) as json_data:
        d = json.load(json_data)
        time_tfr=d['time_tfr']
        
    _, _, _, subj_list, regions = GetInfo(subj_included, data_path=tfr_path)

    band = 'high_gamma'
    pc='compo3'

    weight = pd.read_csv(OUT_PATH + '/grpPCA/supsubj_concat/grp_concat_Compo_PCA5.csv').query('freq == @band & compo == @pc').drop(columns = ['Unnamed: 0', 'freq', 'compo']).values[0, :]
    data_ch = pd.DataFrame()
    data_ch['ch1'] = np.where(abs(weight) > abs(weight).mean() + abs(weight).std(), True, False)
    data_ch['subj'] = subj_list
    data_ch['ROIs'] = regions
    rois = data_ch[data_ch['ch1'] == True]['ROIs'].values

    if not os.path.exists(OUT_PATH+f'/Connectivity/{band}') :
        os.makedirs(OUT_PATH+f'/Connectivity/{band}')

    trials_env, trials_phase = group_hilbert(band=band, param_filter=param_filter, subj_list=subj_list, data_ch=data_ch,tfr_path=tfr_path, events=events)
    compute_connectivity(band=band, pc=pc, rois=rois, trials=trials_env,method='aec', events=events)
    #compute_connectivity(band=band, pc=pc, rois=rois, trials=trials_phase,methods='plv', events=events)

    #trials_time = group_time_domain(band, param_filter, subj_list, data_ch, tfr_path)
    #compute_connectivity(band=band, pc=pc, rois=rois, trials=trials_time,methods='granger', event=events)
    #compute_connectivity(band=band, pc=pc, rois=rois, trials=trials_time,methods='psi', event=events, param_filter=param_filter, spl_rate=spl_rate)

