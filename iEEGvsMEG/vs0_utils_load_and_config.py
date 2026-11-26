"""Shared recording loading, metadata and preprocessing corrections."""
from pathlib import Path
import json
import pickle
import warnings
import numpy as np
import pandas as pd


# Local copies from LB/src/setting.py, LB/src/config.py, and LB/src/decoding.py.
# Keep defaults and region membership unchanged for analysis reproducibility.
PROJECT_PATH = '../../MINDLAB2021_MEG-TempSeqAges/scratch/learning_bach_iEEG'
OUT_PATH = 'outs'
REGION = {'parietal': ['IPS','IP','SP','SPL','AG','SMG','TPJ'],
        #'premotor': ['SFG','SFS','MFG','FEF','SMA'],
        #'DLPFC': ['MFG','FEF','SFS','IFS'],
        'premotor' : ['FEF','SMA'],
        'DLPFC': ['MFG','SFS','IFS', 'SFG'],
        'M1': ['preCG','M1','PreCG'],
        'S1': ['postCG','PostCG'],
        'INS': ['INS'],
        'VLPFC': ['IFG','FOP','IFS'],
        #'MTL': ['EC','HPC','MEC','PRH','PHG','PHC','LEC'],
        'EC': ['EC','MEC','LEC'],
        'PRH' : ['PRH'],
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

def FindRegion(x) :
    for key, val in REGION.items() :
        if x in val :
            return key
    return 'N'

def GetInfo(subj_included, project_path = PROJECT_PATH, data_path = OUT_PATH + '/Data', save=False) :
    coord = []
    areas = []
    elect_list = []
    subj_list = []
    for subj in subj_included:
        df = pd.read_csv(f'{data_path}/{subj}_coords.csv').rename(columns={'Unnamed: 0' :'channels'})
        coord.extend(np.vstack([df['x'].values,  df['y'].values, df['z'].values]).T)

        elect = pd.read_csv(project_path + f'/misc/electrodes/{subj}_electrodes.csv').set_index('electrode')
        df['area1'] = df['channels'].apply(lambda x : elect.loc[x.split('_')[0], 'label'])
        df['area2'] = df['channels'].apply(lambda x : elect.loc[x.split('_')[0], 'label'])

        areas.extend(np.vstack([df['area1'], df['area2']]).T)
        elect_list.extend(df['channels'])
        subj_list.extend([subj]*len(df['channels']))
    region = [FindRegion(a[0]) for a in areas]

    if save :
        d = pd.DataFrame(coord, columns = ['x', 'y', 'z'])
        d.loc[:, ['area1', 'area2']] = areas
        d.loc[:, 'elect'] = elect_list
        d.loc[:, 'subj'] = subj_list
        d.loc[:, 'region'] = region

        d.to_csv(data_path + '/grp_info.csv')
    else :
        return coord, areas, elect_list, subj_list, region

def cleandfdecodingTS(df_final, shape_time) :
    import re
    from sklearn.metrics import accuracy_score

    return_dict ={}

    for f in ['acc', 'acc_sh', 'pca_weight', 'pca_weight_sh', 'weight_mean', 'weight_std', 'weight_sh_mean','weight_sh_std', 'entropy', 'entropy_sh_mean', 'entropy_sh_std']:
        return_dict[f] =[]

    y_pred = np.array([int(x.replace('[', '').replace('np.int64(', '').replace(')', '').replace(']', '')) for x in df_final.y_pred.values[0].split(', ')])
    y_pred_sh = np.array([int(x.replace('[', '').replace('np.int64(', '').replace(')', '').replace(']', '')) for x in df_final.y_pred_sh.values[0].split(', ')]) # 100 * 100 iteration
    y_pred_sh = y_pred_sh.reshape(int(df_final.loc[0, 'iter'])*2, int(df_final.loc[0, 'iter_perm']))
    y_test = np.array([int(x.replace('[', '').replace(']', '')) for x in df_final.y_test.values[0].split(', ')])
    entropy= np.array([float(x.replace('[', '').replace('np.float64(', '').replace(')', '').replace(']', '')) for x in df_final.entropy.values[0].split(', ')])
    entropy_sh= np.array([float(x.replace('[', '').replace('np.float64(', '').replace(')', '').replace(']', '')) for x in df_final.entropy_sh.values[0].split(', ')])
    entropy_sh = entropy_sh.reshape(int(df_final.loc[0, 'iter']), int(df_final.loc[0, 'iter_perm']))
    numbers_only = ' '.join(re.findall(r'[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d+\.\d*|[-+]?\d+', df_final.weight.values[0]))
    weight = np.fromstring(numbers_only, sep=' ').reshape(-1, shape_time)
    numbers_only = ' '.join(re.findall(r'[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d+\.\d*|[-+]?\d+', df_final.weight_sh.values[0]))
    try :
        weight_sh = np.fromstring(numbers_only, sep=' ').reshape(int(df_final.loc[0, 'iter']), int(df_final.loc[0, 'iter_perm']), shape_time) # TO check
        return_dict['weight_sh_mean'] = weight_sh.reshape(-1, shape_time).mean(0)
        return_dict['weight_sh_std'] = weight_sh.reshape(-1, shape_time).std(0)
        return_dict['weight_sh'] = weight_sh.reshape(-1, shape_time)
    except :
         return_dict['weight_sh'] = None

    return_dict['weight'] = weight
    return_dict['weight_mean'] = weight.mean(0)
    return_dict['weight_std'] = weight.std(0)
    return_dict['y_pred'] = y_pred
    return_dict['y_pred_sh'] = y_pred_sh
    return_dict['y_test'] = y_test

    # compute accuracy
    acc_along_perm=[]
    for i in range(int(df_final.loc[0, 'iter_perm'])) :
        acc_along_perm.append(accuracy_score(y_pred_sh[:, i], y_test))

    return_dict['acc_sh'] = np.mean(acc_along_perm) # acc global with the stability of perms
    return_dict['acc'] = accuracy_score(y_pred, y_test) # acc global

    # weights pca
    numbers_only = ' '.join(re.findall(r'[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d+\.\d*|[-+]?\d+', df_final.pca_weight.values[0]))
    return_dict['pca_weight'] = np.fromstring(numbers_only, sep=' ')
    numbers_only = ' '.join(re.findall(r'[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d+\.\d*|[-+]?\d+', df_final.pca_weight_sh.values[0]))
    return_dict['pca_weight_sh'] = np.fromstring(numbers_only, sep=' ')

    # entropy
    return_dict['entropy'] = entropy
    return_dict['entropy_sh'] = entropy_sh
    return_dict['entropy_sh_mean'] = np.array(entropy_sh).mean(axis=1)
    return_dict['entropy_sh_std']  = np.array(entropy_sh).std(axis=1)

    return return_dict


def coordinates_mm(values, unit):
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError('Coordinates must be a finite (channels, 3) array.')
    if unit not in ('m', 'mm'):
        raise ValueError("Coordinate unit must explicitly be 'm' or 'mm'.")
    a = a * (1000 if unit == 'm' else 1)
    if np.max(np.abs(a)) > 250:
        raise ValueError('Coordinates exceed 250 mm: check units/upstream metadata; no automatic repair.')
    return a


def ieeg_getinfo_coordinates_m(values):
    """Apply the project's requested GetInfo coordinate correction, returning metres.

    Raw GetInfo values only: externally supplied metadata already declares its
    unit and must not receive this correction a second time.
    """
    coord = np.array(values, dtype=float)
    coord = np.where(abs(coord) > 100, coord / 1000, coord)
    coord = np.where(abs(coord) > 100, coord / 1000, coord)
    return coord / 1000


def _preprocess(a, mode, multiplier=1.):
    a = np.asarray(a, dtype=np.float32) * multiplier
    if a.ndim != 3 or not np.isfinite(a).all():
        raise ValueError('Expected finite condition x channel x time data.')
    if mode == 'channel_zscore':
        mean = a.mean(axis=(0, 2), keepdims=True, dtype=np.float64)
        std = a.std(axis=(0, 2), keepdims=True, dtype=np.float64)
        if np.any(std == 0):
            warnings.warn('Constant channels retained as zero after standardisation.')
        a = ((a - mean) / np.where(std > 0, std, 1)).astype(np.float32)
    elif mode != 'none':
        raise ValueError("Preprocessing must be 'none' or 'channel_zscore'.")
    return a


def load_project_data(meg_dir, ieeg_dir, *, metadata_csv=None,
                      electrode_metadata=None, meg_subjects=None, ieeg_subjects=None,
                      project_path=None, meg_coordinate_unit='m',
                      ieeg_coordinate_unit='mm', meg_scaling='channel_zscore',
                      ieeg_scaling='none', ieeg_multiplier=1000.,
                      meg_times_file=None, meg_tmin=None, sfreq=250.,
                      conditions=(1, 2), average_cache_dir=None):
    """Load the original pickle/JSON layout, without importing legacy utils.

    MEG: <subject>_source.p and <subject>_pos.csv. iEEG:
    <subject>_epochs.p (trial x channel x time), <subject>_info.json.
    CSV metadata: subject, channel_index (zero-based within saved epochs),
    x, y, z; optional channel/region. Pass the same columns directly as an
    electrode_metadata DataFrame to reuse already prepared GetInfo metadata.
    Optional subject lists preserve the supplied participant selection/order.
    Alternatively use original GetInfo. Raw GetInfo coordinates receive the
    project correction (two >100 divisions, then /1000 to metres). Explicit
    metadata is used with its declared unit and is not corrected again.
    Pickles must come from a trusted source.

    MEG time must be supplied as .npy/.csv or explicit tmin and sfreq.
    The legacy extra final MEG sample is removed only when its time vector
    matches the iEEG time vector after removing exactly that sample.
    """
    from vs0_utils_pairing import active_pairing
    selected = active_pairing()
    if selected is not None:
        meg_subjects = selected.meg_subject.tolist()
        ieeg_subjects = selected.ieeg_subject.tolist()
    meg_dir, ieeg_dir = Path(meg_dir), Path(ieeg_dir)
    raw_meg = any(meg_dir.glob('*_norm0_abs_0.mat'))
    if meg_subjects is None:
        meg_subjects = (sorted(p.name.removesuffix('_norm0_abs_0.mat') for p in meg_dir.glob('*_norm0_abs_0.mat'))
                        if raw_meg else sorted(p.name.removesuffix('_source.p') for p in meg_dir.glob('*_source.p')))
    else:
        meg_subjects = list(map(str, meg_subjects))
    ieeg_subjects = (sorted(p.name.removesuffix('_epochs.p') for p in ieeg_dir.glob('*_epochs.p'))
                     if ieeg_subjects is None else list(map(str, ieeg_subjects)))
    if len(set(meg_subjects)) != len(meg_subjects) or len(set(ieeg_subjects)) != len(ieeg_subjects):
        raise ValueError('Participant lists must not contain duplicates.')
    if metadata_csv is not None and electrode_metadata is not None:
        raise ValueError('Provide metadata_csv or electrode_metadata, not both.')
    if not meg_subjects or not ieeg_subjects:
        raise FileNotFoundError(f'No source/epoch pickles found. Check {meg_dir} and {ieeg_dir}.')
    ieeg_arrays, counts, trial_rows, times = [], [], [], None
    for subject in ieeg_subjects:
        info = json.loads((ieeg_dir / f'{subject}_info.json').read_text())
        with (ieeg_dir / f'{subject}_epochs.p').open('rb') as f:
            epochs = np.asarray(pickle.load(f))
        events = np.asarray(info['event_id'], dtype=int).ravel()
        t = np.asarray(info['time_epoch'], dtype=float)
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
            raise ValueError(f'{subject}: time vector must be finite and strictly increasing.')
        if epochs.ndim != 3 or len(events) != epochs.shape[0] or len(t) != epochs.shape[2]:
            raise ValueError(f'{subject}: event/time metadata do not match epochs.')
        if times is None:
            times = t
        elif times.shape != t.shape or not np.allclose(times, t, atol=1e-7, rtol=0):
            raise ValueError('iEEG time axes differ between participants.')
        averages = []
        for condition in conditions:
            selected = events == condition
            if not selected.any():
                raise ValueError(f'{subject}: no trials for condition {condition}.')
            averages.append(epochs[selected].mean(axis=0))
            trial_rows.append(dict(subject=subject, condition=condition, n_trials=int(selected.sum())))
        ieeg_arrays.append(_preprocess(np.stack(averages), ieeg_scaling, ieeg_multiplier))
        counts.append(epochs.shape[1])
    owners = np.repeat(ieeg_subjects, counts)
    if metadata_csv is not None or electrode_metadata is not None:
        meta = (pd.read_csv(metadata_csv, dtype={'subject': str})
                if electrode_metadata is None else electrode_metadata.copy())
        required = {'subject', 'channel_index', 'x', 'y', 'z'}
        if not required <= set(meta.columns):
            raise ValueError(f'Electrode metadata requires {sorted(required)}.')
        meta['subject'] = meta['subject'].astype(str)
        if meta.duplicated(['subject', 'channel_index']).any():
            raise ValueError('Duplicate subject/channel_index metadata.')
        expected = pd.MultiIndex.from_tuples(
            [(s, i) for s, n in zip(ieeg_subjects, counts) for i in range(n)],
            names=['subject', 'channel_index'])
        meta = meta.set_index(['subject', 'channel_index']).reindex(expected).reset_index()
        coord = meta[['x', 'y', 'z']].to_numpy()
    else:
        coord, areas, electrodes, reported_owners, regions = GetInfo(
            ieeg_subjects, data_path=str(ieeg_dir),
            project_path=str(project_path if project_path is not None else PROJECT_PATH))
        if not np.array_equal(np.asarray(reported_owners, dtype=str), owners):
            raise ValueError('GetInfo channel order does not match concatenated epoch participant order.')
        meta = pd.DataFrame({'subject': owners, 'channel': electrodes, 'region': regions,
                             'channel_index': np.concatenate([np.arange(n) for n in counts])})
        coord = ieeg_getinfo_coordinates_m(coord)
        ieeg_coordinate_unit = 'm'
    coord = coordinates_mm(coord, ieeg_coordinate_unit)
    meta[['x', 'y', 'z']] = coord
    meg, positions, meg_times = [], [], None
    for subject in meg_subjects:
        raw = None
        if raw_meg:
            if tuple(conditions) != (1, 2):
                raise ValueError('Raw MEG loading uses the first two conditions, (1, 2).')
            a, raw_pos, current_times = meg_average(meg_dir / f'{subject}_norm0_abs_0.mat', average_cache_dir)
            raw = {'pos_brainsources_MNI8': raw_pos}
            if meg_times is not None and (current_times.shape != meg_times.shape or not np.allclose(current_times, meg_times, atol=1e-7, rtol=0)):
                raise ValueError(f'{subject}: MEG time axes differ between participants.')
            meg_times = current_times
        else:
            with (meg_dir / f'{subject}_source.p').open('rb') as f:
                a = np.asarray(pickle.load(f))
        if a.ndim != 3 or a.shape[0] != len(conditions):
            raise ValueError(f'{subject}: expected {len(conditions)} saved MEG conditions, in configured order.')
        if meg_times is None:
            if meg_times_file is not None:
                p = Path(meg_times_file)
                meg_times = np.load(p) if p.suffix == '.npy' else np.loadtxt(p, delimiter=',')
                meg_times = np.asarray(meg_times).ravel()
            elif meg_tmin is not None:
                meg_times = float(meg_tmin) + np.arange(a.shape[-1]) / sfreq
            else:
                raise ValueError('Set MEG_TIMES_FILE or verified MEG_TMIN; equal length does not establish temporal alignment.')
        if a.shape[-1] != len(meg_times):
            raise ValueError(f'{subject}: MEG time vector length mismatch.')
        if len(meg_times) == len(times) + 1 and np.allclose(meg_times[:-1], times, atol=1e-7, rtol=0):
            a = a[..., :-1]
        elif len(meg_times) != len(times) or not np.allclose(meg_times, times, atol=1e-7, rtol=0):
            raise ValueError('MEG/iEEG times are not aligned. Align upstream; no silent interpolation.')
        if raw is not None:
            pos = coordinates_mm(raw['pos_brainsources_MNI8'], meg_coordinate_unit)
        else:
            pos = pd.read_csv(meg_dir / f'{subject}_pos.csv')
            pos = pos.loc[:, ~pos.columns.astype(str).str.startswith('Unnamed:')]
            pos = coordinates_mm(pos.to_numpy(), meg_coordinate_unit)
        if len(pos) != a.shape[1]:
            raise ValueError(f'{subject}: source coordinate count mismatch.')
        positions.append(pos)
        meg.append(_preprocess(a, meg_scaling))
    return dict(meg=meg, meg_positions=positions, meg_subjects=meg_subjects,
                ieeg=np.concatenate(ieeg_arrays, axis=1), ieeg_subjects=ieeg_subjects,
                electrode_subjects=owners, electrode_positions=coord,
                electrode_metadata=meta, times=times,
                trial_counts=pd.DataFrame(trial_rows),
                load_config=dict(meg_dir=str(meg_dir), ieeg_dir=str(ieeg_dir),
                    metadata_source=('prepared_dataframe' if electrode_metadata is not None else
                                     str(metadata_csv) if metadata_csv is not None else 'GetInfo'),
                    meg_coordinate_unit=meg_coordinate_unit, ieeg_coordinate_unit=ieeg_coordinate_unit,
                    meg_scaling=meg_scaling, ieeg_scaling=ieeg_scaling, ieeg_multiplier=ieeg_multiplier,
                    meg_times_file=str(meg_times_file) if meg_times_file is not None else None,
                    meg_tmin=meg_tmin, sfreq=sfreq, conditions=list(conditions)))


# Cluster inputs; editable paths are provided in out/vs0_pairs/config.json.
MEG_DATA_DIR = Path('/projects/MINDLAB2020_MEG-AuditoryPatternRecognition/scratch/leonardo/after_maxfilter_v2/Source_LBPD/Block_3/Beam_abs_0_sens_1_freq_broadband_invers_5')
BEHAVIOR_DIR = Path(__file__).resolve().parent / 'MEG' / 'Final_xlsx'
MATCHED_PAIRS_CSV = Path(__file__).resolve().parent / 'out/vs0_pairs/match_pairing_dem.csv'


def load_meg_mat(path):
    """Read conventional and v7.3 MATLAB OUT records without changing units."""
    try:
        from scipy.io import loadmat
        return loadmat(str(path), simplify_cells=True)['OUT']
    except (NotImplementedError, ValueError):
        import mat73
        return mat73.loadmat(str(path))['OUT']


def meg_condition_arrays(out, n_conditions=2):
    """Decode the established sources_ERFs condition cells: source x time x trial."""
    arrays = []
    for c in range(n_conditions):
        value = out['sources_ERFs'][c]
        while isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if isinstance(value, (list, tuple)):
            value = value[0]
        if isinstance(value, np.ndarray) and value.dtype == object and value.size > 1:
            value = value.ravel()[0]  # First LBPD cell column, as in the original exporter.
        a = np.asarray(value)
        while a.dtype == object and a.size == 1:
            a = np.asarray(a.item())
        if a.ndim not in (2, 3) or not np.isfinite(a).all():
            raise ValueError('Expected finite source x time [x trial] MEG condition.')
        arrays.append(a)
    return arrays


def read_matched_pairs(path=MATCHED_PAIRS_CSV):
    frame = pd.read_csv(path, dtype={'ieeg_subject': str, 'meg_subject': str})
    if not {'ieeg_subject', 'meg_subject'} <= set(frame):
        raise ValueError(f'{path}: missing participant columns.')
    if frame[['ieeg_subject', 'meg_subject']].isna().any().any() or not len(frame):
        raise ValueError('Pairings must be nonempty and complete.')
    if frame.ieeg_subject.duplicated().any() or frame.meg_subject.duplicated().any():
        raise ValueError('Pairings must be one-to-one.')
    return frame


def select_subject(ieeg_dem_csv, meg_dem_xlsx, ieeg_dir, meg_dir=MEG_DATA_DIR,
                   behavior_dir=BEHAVIOR_DIR, project_path=PROJECT_PATH,
                   meg_sheet='Background questionnaire'):
    """Return eligible iEEG/MEG tables and an explicit exclusion audit.

    iEEG demographics are already behaviorally screened. MEG requires >=.5 in
    each condition and >=24 total old/new trials (omissions remain in denominators).
    """
    from vs0_utils_meg_behavior import meg_behavior_performance, meg_subject_key
    ieeg = pd.read_csv(ieeg_dem_csv, dtype={'subject': str})[['subject', 'age', 'sex']]
    meg = pd.read_excel(meg_dem_xlsx, sheet_name=meg_sheet)[['Subject', 'Age', 'Sex']].rename(
        columns={'Subject': 'subject', 'Age': 'age', 'Sex': 'sex'})
    behavior = meg_behavior_performance(behavior_dir).set_index('subject')
    available = {}
    for path in sorted(Path(meg_dir).glob('*_norm0_abs_0.mat')):
        subject = path.name.removesuffix('_norm0_abs_0.mat')
        key = meg_subject_key(subject)
        if key in available:
            raise ValueError(f'Duplicate MEG recording identity: {key}')
        available[key] = subject
    audit, cohorts = [], []
    for modality, frame in [('ieeg', ieeg), ('meg', meg)]:
        frame = frame.copy()
        frame['subject'] = frame.subject.astype('string').str.strip()
        frame['age'] = pd.to_numeric(frame.age, errors='coerce')
        frame['sex'] = frame.sex.astype('string').str.strip().str.lower().replace(
            {'f': 'female', 'woman': 'female', 'm': 'male', 'man': 'male'})
        seen, kept = set(), []
        for row in frame.to_dict('records'):
            original = row['subject']
            reasons = []
            if pd.isna(original) or not str(original):
                reasons.append('missing_id')
            else:
                key = meg_subject_key(original) if modality == 'meg' else original
                if key in seen:
                    raise ValueError(f'Duplicate {modality} participant {key}')
                seen.add(key)
            if not np.isfinite(row['age']) or row['age'] <= 0:
                reasons.append('missing_or_invalid_age')
            if pd.isna(row['sex']) or not str(row['sex']).strip():
                reasons.append('missing_sex')
            if not pd.isna(original) and str(original):
                if modality == 'ieeg':
                    required = [Path(ieeg_dir)/f'{original}{suffix}' for suffix in ('_epochs.p', '_info.json', '_coords.csv')]
                    required.append(Path(project_path)/'misc/electrodes'/f'{original}_electrodes.csv')
                    if not all(p.is_file() for p in required): reasons.append('missing_recording_or_metadata')
                else:
                    if key not in available: reasons.append('missing_recording')
                    if key not in behavior.index:
                        reasons.append('missing_behavior')
                    else:
                        perf = behavior.loc[key].to_dict()
                        row.update(perf)
                        if not bool(perf['eligible']): reasons.append('behavior_below_threshold_or_missing_condition')
                    row['subject'] = available.get(key, original)
            audit.append(dict(modality=modality, subject=original, recording_subject=row['subject'],
                              eligible=not reasons, reason=';'.join(reasons)))
            if not reasons: kept.append(row)
        if not kept: raise ValueError(f'No eligible {modality} participants; inspect input paths and eligibility.')
        cohorts.append(pd.DataFrame(kept).sort_values('subject').reset_index(drop=True))
    if len(cohorts[1]) < len(cohorts[0]):
        raise ValueError('Insufficient distinct eligible MEG participants.')
    return cohorts[0], cohorts[1], pd.DataFrame(audit)


def match_sources(electrode_positions_mm, meg_positions_mm, *, seed=2026):
    """Nearest source per electrode; reproducibly sample exact-distance ties.

    Returns source indices and distances in mm. Sources may be reused.
    """
    from scipy.spatial import cKDTree
    electrodes = coordinates_mm(electrode_positions_mm, 'mm')
    sources = coordinates_mm(meg_positions_mm, 'mm')
    if not len(sources): raise ValueError('No MEG sources.')
    tree, rng = cKDTree(sources), np.random.default_rng(seed)
    distances, indices = tree.query(electrodes)
    for i, (point, distance) in enumerate(zip(electrodes, distances)):
        candidates = np.asarray(tree.query_ball_point(point, np.nextafter(distance, np.inf)), dtype=int)
        squared = np.sum((sources[candidates] - point)**2, axis=1)
        ties = candidates[squared == squared.min()]
        indices[i] = int(rng.choice(np.sort(ties)))
    return indices.astype(int), np.linalg.norm(sources[indices] - electrodes, axis=1)


def meg_average(path, cache_dir=None):
    """Cache condition averages once per MEG participant, retaining raw provenance."""
    path = Path(path)
    signature = dict(path=str(path.resolve()), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    target = None if cache_dir is None else Path(cache_dir)/f'{path.stem}.npz'
    if target is not None and target.exists():
        with np.load(target, allow_pickle=False) as saved:
            if json.loads(str(saved['signature'])) != signature:
                raise ValueError(f'{path}: source changed; invalidate its average cache explicitly.')
            return saved['data'].copy(), saved['positions'].copy(), saved['times'].copy()
    raw = load_meg_mat(path)
    parts = meg_condition_arrays(raw, 2)
    arrays = np.stack([a.mean(axis=2) if a.ndim == 3 else a for a in parts])
    pos, times = np.asarray(raw['pos_brainsources_MNI8']), np.asarray(raw['time']).ravel()
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_suffix('.npz.tmp')
        with pending.open('wb') as stream:
            np.savez_compressed(stream, data=arrays, positions=pos, times=times, signature=json.dumps(signature))
        pending.replace(target)
    return arrays, pos, times
