"""Shared recording loading, metadata and preprocessing corrections."""
from pathlib import Path
import json
import pickle
import warnings
import numpy as np
import pandas as pd

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
                      conditions=(1, 2)):
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
    meg_dir, ieeg_dir = Path(meg_dir), Path(ieeg_dir)
    meg_subjects = (sorted(p.name.removesuffix('_source.p') for p in meg_dir.glob('*_source.p'))
                    if meg_subjects is None else list(map(str, meg_subjects)))
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
        try:
            from src.setting import GetInfo, PROJECT_PATH
        except ImportError as exc:
            raise ImportError('Provide METADATA_CSV or restore the original src.setting/GetInfo module.') from exc
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

