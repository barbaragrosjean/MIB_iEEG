import pandas as pd
import numpy as np
import json
from pathlib import Path

from src.config import OUT_PATH, PROJECT_PATH
from collections import deque
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, pearsonr, linregress

RT_CORRECTION_PATH = (
    '/users/barbara/Desktop/MINDLAB2021_MEG-TempSeqAges/'
    'scratch/learning_bach_iEEG/misc/RT_correction'
)


def extract_RT(subj_included, path_behavior=PROJECT_PATH+'/misc/events',
               path_info=OUT_PATH + '/Data_longWOBS',
               path_rt_correction=RT_CORRECTION_PATH):
    """Extract RTs in seconds, preferring subject-specific corrected BIDS events.

    Only sub-<subject>_task-MusicMemory_events.tsv files are overrides;
    *_BIDS_events_old.tsv backups are ignored. Subjects without an override
    retain the original CSV extraction. Pass None to disable corrections.
    An explicitly configured correction directory must exist, preventing
    accidental use of uncorrected data when the path is misspelled.
    """
    correction_dir = Path(path_rt_correction) if path_rt_correction is not None else None
    if correction_dir is not None and not correction_dir.is_dir():
        raise FileNotFoundError(f'RT correction directory not found: {correction_dir}')
    df_all = []
    for subj in subj_included:
        subject_id = subj.removeprefix('sub-')
        corrected_file = (
            correction_dir / f'sub-{subject_id}_task-MusicMemory_events.tsv'
            if correction_dir is not None else None
        )
        if corrected_file is not None and corrected_file.is_file():
            df = pd.read_csv(corrected_file, sep='\t')
            required = {'onset', 'trial_type'}
            if not required.issubset(df.columns):
                raise ValueError(f'{corrected_file}: missing columns {required - set(df.columns)}')
            event_types = {
                'learning_bach_10': 'recognition',
                'learning_bach_20': 'recognition',
                'learning_bach_48': 'keypress',
                'learning_bach_49': 'keypress',
                'learning_bach_50': 'keypress',
            }
            df = df.loc[df['trial_type'].isin(event_types)].copy()
            df['task'] = 'learning_bach'
            df['type'] = df['trial_type'].map(event_types)
            df['onset'] = pd.to_numeric(df['onset'], errors='raise')
            if not np.isfinite(df['onset']).all() or not df['type'].eq('recognition').any():
                raise ValueError(f'{corrected_file}: invalid onsets or no recognition events')
            print(f'{subj}: using corrected RT events from {corrected_file}')
        else:
            df = pd.read_csv(str(Path(path_behavior) / f'{subj}_events.csv'))
        tmp = df[
            (df['task'] == 'learning_bach') &
            (df['type'].isin(['recognition', 'KeyPress', 'keypress']))
        ].copy()

        first_rec_idx = tmp.index[tmp["type"] == "recognition"][0]
        tmp = tmp.loc[first_rec_idx:].sort_values("onset")

        pending_rec = deque()
        pending_key = None
        rows = []

        for _, row in tmp.iterrows():

            event = row["type"].lower()

            if event == "recognition":
                if pending_key is not None and len(pending_rec) > 0:
                    while len(pending_rec) > 1:
                        rec = pending_rec.popleft()
                        rows.append({
                            "recognition_onset": rec,
                            "keypress_onset": np.nan,
                            "RT": np.nan
                        })

                    rec = pending_rec.pop()
                    rows.append({
                        "recognition_onset": rec,
                        "keypress_onset": pending_key,
                        "RT": pending_key - rec
                    })

                    pending_key = None

                pending_rec.append(row["onset"])

            elif event == "keypress":
                pending_key = row["onset"]

        if pending_key is not None and len(pending_rec) > 0:

            while len(pending_rec) > 1:
                rec = pending_rec.popleft()
                rows.append({
                    "recognition_onset": rec,
                    "keypress_onset": np.nan,
                    "RT": np.nan
                })

            rec = pending_rec.pop()
            rows.append({
                "recognition_onset": rec,
                "keypress_onset": pending_key,
                "RT": pending_key - rec
            })

        while len(pending_rec) > 0:
            rec = pending_rec.popleft()
            rows.append({
                "recognition_onset": rec,
                "keypress_onset": np.nan,
                "RT": np.nan
            })

        rt = pd.DataFrame(rows)
        rt["subject"] = subj

        info_file = path_info + f'/{subj}_info.json'
        with open(info_file) as f:
            info = json.load(f)
            events = info['event_id']
        rt = rt.dropna()
        rt["ev_id"] = events[:len(rt)]
        df_all.append(rt)

    df_RT = pd.concat(df_all, ignore_index=True)

    return df_RT

def get_cross_point(i, signals, tstart=0, tend=-1, n_confirm=10) : 
    x = gaussian_filter1d(signals[i, tstart:tend], sigma=2)
    candidates = np.argsort(np.abs(x))

    for idx in candidates:
        if idx == 0 or idx + n_confirm >= len(x):
            continue

        s0 = np.sign(x[idx - 1])
        s1 = np.sign(x[idx + 1:idx + 1 + n_confirm])
        if s0 == 0 or np.any(s1 == 0):
            continue

        if np.all(s1 == -s0):
            return idx + tstart

    return candidates[0] + tstart

def get_slop_pc3(i, signals, time) : 
    signal = gaussian_filter1d(signals[i, :], sigma=10)
    crossing = get_cross_point(i, signals, tstart=120, tend=140)           
    maxima = 135 + np.argmax(signal[ 135:])      
    x = np.array(time)[crossing:maxima]
    y = signal[crossing:maxima]
    slop = linregress(x, y).slope

    return slop

def get_slop_pc2(i, signals, time) : 
    signal = gaussian_filter1d(signals[i, :], sigma=10)
    crossing  = get_cross_point(i, signals, tstart=50, tend=115)
    minima = np.argmax(signals[i, 100:]) + 100
    x = np.array(time)[crossing:minima]
    y = signal[crossing:minima]
    slope1 = linregress(x, y).slope

    x = np.array(time)[minima :]
    y = signal[minima:]
    slope2 = linregress(x, y).slope

    return slope1, slope2

def area_pc2(i, signals): 
    signal = gaussian_filter1d(signals[i, :], sigma=10)
    minima = np.argmax(signal)
    signal_neg = np.where(signal < 0, signal, 0)
    signal_pos = np.where(signal> 0, signal, 0)

    return abs(np.trapezoid(signal_neg[ :minima])), abs(np.trapezoid(signal_pos[ :minima]))
