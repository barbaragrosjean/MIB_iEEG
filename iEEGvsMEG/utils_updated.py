"""Coverage ablation and memory-bounded PCA for the iEEGvsMEG project.

Input arrays: condition x channel x time, one array per MEG participant.
All coordinates passed to the analysis API are MNI millimetres. No implicit
coordinate repair, absolute-value weights, or extra post-aggregation scaling.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


SETUP_NAMES = (
    'full_average', 'full_concatenated', 'coverage_average',
    'paired_coverage', 'random_control',
)


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
    Alternatively use original GetInfo.
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
            project_path=str(project_path if project_path is not None else Path('..') / PROJECT_PATH))
        if not np.array_equal(np.asarray(reported_owners, dtype=str), owners):
            raise ValueError('GetInfo channel order does not match concatenated epoch participant order.')
        meta = pd.DataFrame({'subject': owners, 'channel': electrodes, 'region': regions,
                             'channel_index': np.concatenate([np.arange(n) for n in counts])})
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


def observations(a, condition_mode='average'):
    """Return observations x features; stacked order is condition then time."""
    if condition_mode == 'average':
        return a.mean(axis=0).T
    if condition_mode == 'stack':
        return a.transpose(0, 2, 1).reshape(-1, a.shape[1])
    raise ValueError("condition_mode must be 'average' or 'stack'.")


@dataclass
class Dataset:
    name: str
    arrays: list
    metadata: pd.DataFrame
    electrode_to_feature: np.ndarray
    condition_mode: str = 'average'
    source_data: dict = None  # shared loaded arrays; populated by load_dataset
    matching: pd.DataFrame = None
    pairing: dict = None

    def blocks(self):
        for a in self.arrays:
            yield observations(a, self.condition_mode)

    @property
    def n_features(self):
        return sum(a.shape[1] for a in self.arrays)

    @property
    def n_observations(self):
        a = self.arrays[0]
        return a.shape[-1] * (a.shape[0] if self.condition_mode == 'stack' else 1)


def _metadata(pos, subject, source_index=None):
    out = pd.DataFrame(np.asarray(pos), columns=['x', 'y', 'z'])
    out['meg_subject'] = subject
    out['source_index'] = np.arange(len(pos)) if source_index is None else source_index
    return out


def construct_five_datasets(meg, meg_positions, meg_subjects, electrode_positions,
                            electrode_subjects, *, seed=2026, pairing=None,
                            condition_mode='average', random_preserve_duplicates=True,
                            kinds=None):
    """Build five setups, preserving electrode row identity in all matches.

    Full concatenation is a list of participant blocks, never a giant matrix.
    Group coverage is one nearest-source feature per pooled iEEG electrode,
    selected separately in EVERY MEG participant, then participant-averaged.
    This matches legacy average_subject=True (not a radius-mask definition).
    Pairing uses all available MEG subjects as the sampling pool, without
    replacement. The control keeps that pairing and randomises source locations.
    By default it also preserves matched-source duplication multiplicities.
    """
    requested = set(SETUP_NAMES if kinds is None else kinds)
    if not requested or not requested <= set(SETUP_NAMES):
        raise ValueError(f'Unknown MEG setup: {requested - set(SETUP_NAMES)}')
    meg_subjects = list(map(str, meg_subjects))
    owners = np.asarray(electrode_subjects, dtype=str)
    coords = np.asarray(electrode_positions, dtype=float)
    subjects = list(dict.fromkeys(owners))
    if not meg or len(meg) != len(meg_subjects) or len(meg_positions) != len(meg):
        raise ValueError('MEG arrays, positions and subject lists must have equal nonzero length.')
    if len(set(meg_subjects)) != len(meg_subjects):
        raise ValueError('Duplicate MEG participant IDs.')
    if len(coords) != len(owners) or coords.shape != (len(owners), 3) or not len(coords):
        raise ValueError('Electrode coordinates and owners must align.')
    if not np.isfinite(coords).all():
        raise ValueError('Nonfinite electrode coordinates.')
    if condition_mode not in ('average', 'stack'):
        raise ValueError("condition_mode must be 'average' or 'stack'.")
    shape = meg[0].shape
    for a, pos in zip(meg, meg_positions):
        if a.ndim != 3 or a.shape != shape or np.asarray(pos).shape != (a.shape[1], 3):
            raise ValueError('Full participant averaging requires identical condition/source/time dimensions.')
        if not np.isfinite(a).all() or not np.isfinite(pos).all():
            raise ValueError('Nonfinite MEG data/coordinates.')
        if not np.allclose(pos, meg_positions[0], atol=1e-3, rtol=0):
            raise ValueError('Full source averaging requires the same ordered source grid; morph/register upstream.')
    if len(subjects) > len(meg_subjects):
        raise ValueError('Not enough MEG participants for one-to-one pairing; supply a justified alternative upstream.')
    pair_rng, control_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    if pairing is None:
        pairing = dict(zip(subjects, pair_rng.choice(meg_subjects, len(subjects), replace=False)))
    pairing = {str(k): str(v) for k, v in pairing.items()}
    if set(pairing) != set(subjects) or not set(pairing.values()) <= set(meg_subjects):
        raise ValueError('Pairing must map every iEEG subject to an existing MEG subject.')
    if len(set(pairing.values())) != len(pairing):
        raise ValueError('Pairing must be one-to-one.')
    lookup = {s: i for i, s in enumerate(meg_subjects)}
    n = len(coords)
    nearest, distances = [], []
    for pos in meg_positions:
        d, ix = cKDTree(pos).query(coords)
        nearest.append(ix)
        distances.append(d)
    working_dtype = np.result_type(np.float32, *[a.dtype for a in meg])
    matched_shape = (shape[0], n, shape[2])
    avg = np.zeros(shape, dtype=working_dtype) if 'full_average' in requested else None
    group = np.zeros(matched_shape, dtype=working_dtype) if 'coverage_average' in requested else None
    if avg is not None or group is not None:
        for a, ix in zip(meg, nearest):
            if avg is not None:
                avg += a / len(meg)
            if group is not None:
                group += np.take(a, ix, axis=1) / len(meg)
    paired = np.empty(matched_shape, dtype=working_dtype) if 'paired_coverage' in requested else None
    control = np.empty(matched_shape, dtype=working_dtype) if 'random_control' in requested else None
    paired_pos, random_pos = np.empty((n, 3)), np.empty((n, 3))
    paired_indices, random_indices = np.empty(n, int), np.empty(n, int)
    paired_subjects = np.empty(n, object)
    full_map = np.empty(n, int)
    offsets = np.cumsum([0] + [a.shape[1] for a in meg])
    audit = []
    for subject in subjects:
        rows = np.flatnonzero(owners == subject)
        j = lookup[pairing[subject]]
        ix = nearest[j][rows]
        if random_preserve_duplicates:
            unique, inverse = np.unique(ix, return_inverse=True)
            rx = control_rng.choice(meg[j].shape[1], len(unique), replace=False)[inverse]
        else:
            if len(rows) > meg[j].shape[1]:
                raise ValueError('More electrodes than available random sources.')
            rx = control_rng.choice(meg[j].shape[1], len(rows), replace=False)
        if paired is not None:
            paired[:, rows, :] = np.take(meg[j], ix, axis=1)
        if control is not None:
            control[:, rows, :] = np.take(meg[j], rx, axis=1)
        paired_pos[rows], random_pos[rows] = meg_positions[j][ix], meg_positions[j][rx]
        paired_indices[rows], random_indices[rows] = ix, rx
        paired_subjects[rows] = meg_subjects[j]
        full_map[rows] = offsets[j] + ix
        for row, source, random_source in zip(rows, ix, rx):
            audit.append(dict(electrode_index=int(row), ieeg_subject=subject,
                              meg_subject=meg_subjects[j], source_index=int(source),
                              random_source_index=int(random_source),
                              distance_mm=float(distances[j][row]),
                              random_distance_mm=float(np.linalg.norm(meg_positions[j][random_source] - coords[row]))))
    full_meta = (pd.concat([_metadata(p, s) for p, s in zip(meg_positions, meg_subjects)], ignore_index=True)
                 if 'full_concatenated' in requested else None)
    group_pos = np.mean([p[ix] for p, ix in zip(meg_positions, nearest)], axis=0)
    builders = {
        'full_average': lambda: Dataset('full_average', [avg], _metadata(meg_positions[0], 'participant_average'), nearest[0], condition_mode),
        'full_concatenated': lambda: Dataset('full_concatenated', meg, full_meta, full_map, condition_mode),
        'coverage_average': lambda: Dataset('coverage_average', [group], _metadata(group_pos, 'participant_average', nearest[0]), np.arange(n), condition_mode),
        'paired_coverage': lambda: Dataset('paired_coverage', [paired], _metadata(paired_pos, paired_subjects, paired_indices), np.arange(n), condition_mode),
        'random_control': lambda: Dataset('random_control', [control], _metadata(random_pos, paired_subjects, random_indices), np.arange(n), condition_mode),
    }
    datasets = {name: builders[name]() for name in SETUP_NAMES if name in requested}
    for name in requested.intersection(SETUP_NAMES[2:]):
        datasets[name].metadata['electrode_index'] = np.arange(n)
        datasets[name].metadata['ieeg_subject'] = owners
    for ds in datasets.values():
        ds.metadata.insert(0, 'feature_index', np.arange(ds.n_features))
    return datasets, pd.DataFrame(audit).sort_values('electrode_index').reset_index(drop=True), pairing


def make_ieeg_dataset(ieeg, metadata, condition_mode='average'):
    if len(metadata) != ieeg.shape[1]:
        raise ValueError('iEEG metadata and feature count differ.')
    return Dataset('iEEG', [ieeg], metadata.copy(), np.arange(ieeg.shape[1]), condition_mode)


def variance_summary(datasets):
    """Total variance = trace of sample covariance (ddof=1), before PCA."""
    rows = []
    for name, ds in datasets.items():
        total = 0.
        constants = 0
        for x in ds.blocks():
            v = np.var(x, axis=0, ddof=1, dtype=np.float64)
            total += float(v.sum())
            constants += int((v == 0).sum())
        rows.append(dict(dataset=name, n_observations=ds.n_observations,
                         n_features=ds.n_features, total_variance=total,
                         mean_feature_variance=total / ds.n_features,
                         constant_features=constants,
                         dense_float64_GiB=ds.n_observations * ds.n_features * 8 / 2**30))
    return pd.DataFrame(rows).set_index('dataset')


@dataclass
class PCAResult:
    scores: np.ndarray
    weights: np.ndarray  # features x components
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    total_variance: float
    dataset: Dataset = None  # attached by compute_pca for plotting/comparison


def fit_block_pca(dataset, n_components=10, feature_chunk=1024, max_gram_gib=2.):
    """Exact centred PCA via observation Gram matrix, bounded feature chunks.

    Equivalent to ordinary unscaled PCA, including on full concatenation.
    Does not construct a channel-by-channel covariance matrix or the full
    concatenated observations matrix. Accumulation and eigensolve are float64.
    Budget allows roughly three Gram matrices for construction/eigensolve.
    """
    n, p = dataset.n_observations, dataset.n_features
    if n < 2 or p < 1 or feature_chunk < 1 or n_components < 1:
        raise ValueError('PCA requires >=2 observations, >=1 feature and positive parameters.')
    if 3 * n * n * 8 / 2**30 > max_gram_gib:
        raise MemoryError('Observation Gram allocation exceeds budget; reduce time window or increase MAX_GRAM_GIB.')
    gram = np.zeros((n, n), dtype=np.float64)
    for block in dataset.blocks():
        for start in range(0, block.shape[1], feature_chunk):
            x = np.array(block[:, start:start + feature_chunk], dtype=np.float64, copy=True)
            x -= x.mean(axis=0)
            gram += x @ x.T
    total_ss = float(np.trace(gram))
    if total_ss <= 0:
        raise ValueError(f'{dataset.name}: no temporal variance.')
    k = min(n_components, n - 1, p)
    values, vectors = eigh(gram, subset_by_index=(n - k, n - 1), check_finite=False)
    values, vectors = values[::-1], vectors[:, ::-1]
    keep = values > max(values[0], total_ss / n) * np.finfo(float).eps * max(n, p)
    values, vectors = values[keep], vectors[:, keep]
    if not len(values):
        raise ValueError('No numerically nonzero PCA components.')
    if len(values) < n_components:
        warnings.warn(f'{dataset.name}: only {len(values)} nonzero requested components available.')
    singular = np.sqrt(values)
    scores = vectors * singular
    weights = np.empty((p, len(values)), dtype=np.float64)
    offset = 0
    for block in dataset.blocks():
        for start in range(0, block.shape[1], feature_chunk):
            x = np.array(block[:, start:start + feature_chunk], dtype=np.float64, copy=True)
            x -= x.mean(axis=0)
            width = x.shape[1]
            weights[offset + start:offset + start + width] = x.T @ vectors / singular
        offset += block.shape[1]
    # Deterministic orientation only; PCA signs still have no intrinsic meaning.
    signs = np.sign(weights[np.argmax(np.abs(weights), axis=0), np.arange(len(values))])
    signs[signs == 0] = 1
    return PCAResult(scores * signs, weights * signs, values / (n - 1),
                     values / total_ss, total_ss / (n - 1))


def cross_correlations(a, b):
    """Pearson correlation between columns; constant columns return NaN."""
    if a.shape[0] != b.shape[0]:
        raise ValueError('Correlation requires aligned observations/features.')
    a = np.asarray(a, float) - np.mean(a, axis=0)
    b = np.asarray(b, float) - np.mean(b, axis=0)
    denominator = np.linalg.norm(a, axis=0)[:, None] * np.linalg.norm(b, axis=0)[None, :]
    return np.divide(a.T @ b, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0).clip(-1, 1)


def compare_to_ieeg(dataset, result, reference):
    """All-PC correlations and descriptive time-selected component assignment.

    Full MEG weights are sampled at an explicit electrode correspondence:
    nearest source for full average, paired subject's nearest source for full
    concatenation, electrode feature slots for the other three. Random-control
    slots deliberately have no anatomical correspondence.
    """
    time_corr = cross_correlations(result.scores, reference.scores)
    weight_corr = cross_correlations(result.weights[dataset.electrode_to_feature], reference.weights)
    rows, cols = linear_sum_assignment(-np.nan_to_num(np.abs(time_corr), nan=0.))
    records = []
    for i, j in zip(rows, cols):
        sign = -1 if time_corr[i, j] < 0 else 1
        records.append(dict(dataset=dataset.name, meg_pc=int(i + 1), ieeg_pc=int(j + 1),
                            time_r=float(time_corr[i, j]), abs_time_r=float(abs(time_corr[i, j])),
                            weight_r=float(weight_corr[i, j]),
                            time_aligned_weight_r=float(sign * weight_corr[i, j]),
                            sign=sign))
    return {'time': time_corr, 'weights': weight_corr, 'assignment': pd.DataFrame(records)}


def plot_coverage(datasets, electrode_positions, max_points=20000):
    """MNI scatter projections without atlas downloads; skip oversized sets."""
    figures = []
    for name, ds in datasets.items():
        if ds.n_features > max_points:
            print(f'Skipping coverage plot for {name}: {ds.n_features:,} features.')
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        pos = ds.metadata[['x', 'y', 'z']].to_numpy()
        for ax, (i, j), label in zip(axes, [(0, 1), (0, 2), (1, 2)], ['Axial', 'Coronal', 'Sagittal']):
            ax.scatter(pos[:, i], pos[:, j], s=8, alpha=.35, label='MEG feature')
            ax.scatter(electrode_positions[:, i], electrode_positions[:, j], s=10, c='crimson', alpha=.5, label='iEEG contact')
            ax.set(xlabel=f'{"xyz"[i]} (mm)', ylabel=f'{"xyz"[j]} (mm)', title=label, aspect='equal')
        axes[0].legend(fontsize=8)
        fig.suptitle(f'{name}: {ds.n_features:,} features, {len(np.unique(pos, axis=0)):,} unique plotted positions')
        figures.append((name, fig))
    return figures


def plot_pca(result, label, times, condition_mode='average', condition_labels=('1', '2')):
    k = result.scores.shape[1]
    fig, axes = plt.subplots(int(np.ceil(k / 2)), 2, figsize=(15, 2.2 * int(np.ceil(k / 2))), squeeze=False, constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        if i >= k:
            ax.set_visible(False)
            continue
        if condition_mode == 'average':
            ax.plot(times, result.scores[:, i])
        else:
            traces = result.scores[:, i].reshape(-1, len(times))
            for c, trace in enumerate(traces):
                ax.plot(times, trace, label=str(condition_labels[c]))
            ax.legend(fontsize=8)
        ax.set(title=f'PC {i+1} ({100 * result.explained_variance_ratio[i]:.1f}%)', xlabel='Time (s)', ylabel='Score')
        ax.grid(alpha=.2)
    fig.suptitle(label)
    return fig


def plot_explained_variance(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    for name, r in results.items():
        x = np.arange(1, len(r.explained_variance_ratio) + 1)
        axes[0].plot(x, 100 * r.explained_variance_ratio, '.-', label=name)
        axes[1].plot(x, 100 * np.cumsum(r.explained_variance_ratio), '.-', label=name)
    axes[0].set(xlabel='PC', ylabel='Explained variance (%)')
    axes[1].set(xlabel='Number of PCs', ylabel='Cumulative explained variance (%)', ylim=(0, 101))
    axes[1].legend(fontsize=8)
    return fig


def plot_correlations(comparisons):
    fig, axes = plt.subplots(len(comparisons), 2, figsize=(12, 4 * len(comparisons)), squeeze=False, constrained_layout=True)
    for row, (name, comparison) in enumerate(comparisons.items()):
        for col, key in enumerate(('time', 'weights')):
            matrix = comparison[key]
            im = axes[row, col].imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
            axes[row, col].set(title=f'{name}: {key}' + (' (random slots)' if name == 'random_control' and key == 'weights' else ''),
                               xlabel='iEEG PC', ylabel='MEG PC',
                               xticks=range(matrix.shape[1]), xticklabels=range(1, matrix.shape[1]+1),
                               yticks=range(matrix.shape[0]), yticklabels=range(1, matrix.shape[0]+1))
            fig.colorbar(im, ax=axes[row, col], label='Signed Pearson r')
    return fig


def plot_time_overlays(results, reference, times, condition_mode='average'):
    """Rank-wise, unit-SD overlays; sign alignment only, no PC reordering."""
    k = min(r.scores.shape[1] for r in [reference, *results.values()])
    fig, axes = plt.subplots(int(np.ceil(k/2)), 2, figsize=(16, 2.5 * int(np.ceil(k/2))), squeeze=False, constrained_layout=True)
    n_conditions = reference.scores.shape[0] // len(times)
    for pc, ax in enumerate(axes.flat):
        if pc >= k:
            ax.set_visible(False)
            continue
        ref = reference.scores[:, pc] / reference.scores[:, pc].std()
        # Stacked conditions are separated on a sample axis to avoid joining end to start at the same t.
        x = times if condition_mode == 'average' else np.arange(len(ref))
        ax.plot(x, ref, color='black', lw=2, label='iEEG')
        for name, r in results.items():
            score = r.scores[:, pc] / r.scores[:, pc].std()
            corr = cross_correlations(score[:, None], ref[:, None])[0, 0]
            ax.plot(x, score * (-1 if corr < 0 else 1), alpha=.75, lw=1, label=name)
        if condition_mode == 'stack':
            for boundary in range(1, n_conditions):
                ax.axvline(boundary * len(times) - .5, color='grey', ls=':')
        ax.set(title=f'PC {pc+1}', xlabel='Time (s)' if condition_mode == 'average' else 'Condition-major time sample', ylabel='Score / SD')
    axes.flat[0].legend(fontsize=7)
    return fig


def synthetic_inputs(seed=0):
    """Small, explicitly synthetic example for smoke tests; never study results."""
    rng = np.random.default_rng(seed)
    times = np.arange(100) / 250 - .1
    pos = rng.uniform([-60, -90, -40], [60, 70, 70], (40, 3))
    latent = rng.normal(size=(2, 5, len(times)))
    mixing = rng.normal(size=(40, 5))
    meg = [np.einsum('cf,kft->kct', mixing + rng.normal(scale=.2, size=mixing.shape), latent).astype(np.float32) + rng.normal(scale=.1, size=(2, 40, len(times))) for _ in range(4)]
    ix = np.array([1, 1, 6, 8, 10, 12, 15, 19, 22, 26, 29, 34])
    coords = pos[ix] + rng.normal(scale=.1, size=(len(ix), 3))
    owners = np.repeat(['I1', 'I2', 'I3'], [4, 3, 5])
    ieeg = np.take(np.mean(meg, axis=0), ix, axis=1) + rng.normal(scale=.2, size=(2, len(ix), len(times)))
    meta = pd.DataFrame(coords, columns=['x', 'y', 'z'])
    meta['subject'] = owners
    meta['channel_index'] = np.concatenate([np.arange(n) for n in [4, 3, 5]])
    return dict(meg=meg, meg_positions=[pos.copy() for _ in meg],
                meg_subjects=['M1', 'M2', 'M3', 'M4'], electrode_positions=coords,
                electrode_subjects=owners, ieeg=ieeg, electrode_metadata=meta,
                times=times, trial_counts=pd.DataFrame())


# Simple notebook API --------------------------------------------------------

def load_dataset(kind, *, reference=None, condition_mode=None, seed=2026,
                 pairing=None, **file_options):
    """Load 'ieeg' or one MEG setup and keep everything needed downstream.

    First call: load_dataset('ieeg', meg_dir=..., ieeg_dir=..., ...).
    Subsequent calls: load_dataset('paired_coverage', reference=ieeg).
    File options are those of load_project_data. A reference reuses the loaded
    arrays (no disk reload). Only the requested MEG dataset is constructed.
    Use the same seed/pairing for the paired and random-control setups.
    """
    if kind not in ('ieeg', 'iEEG', *SETUP_NAMES):
        raise ValueError(f'Choose ieeg or one of {SETUP_NAMES}.')
    if reference is None:
        inputs = load_project_data(**file_options)
    else:
        if file_options:
            raise ValueError('Pass file options on the first load only; reference reuses those data.')
        inputs = reference.source_data
        if inputs is None:
            raise ValueError('reference must come from load_dataset.')
    mode = condition_mode or (reference.condition_mode if reference is not None else 'average')
    if mode not in ('average', 'stack'):
        raise ValueError("condition_mode must be 'average' or 'stack'.")
    if kind.lower() == 'ieeg':
        dataset = make_ieeg_dataset(inputs['ieeg'], inputs['electrode_metadata'], mode)
    else:
        selected, audit, used_pairing = construct_five_datasets(
            inputs['meg'], inputs['meg_positions'], inputs['meg_subjects'],
            inputs['electrode_positions'], inputs['electrode_subjects'],
            seed=seed, pairing=pairing, condition_mode=mode, kinds=[kind],
        )
        dataset = selected[kind]
        dataset.matching = audit
        dataset.pairing = used_pairing
    dataset.source_data = inputs
    return dataset


def compute_variance(dataset):
    """Total/mean feature variance, feature count and observation count."""
    return variance_summary({dataset.name: dataset}).loc[dataset.name, [
        'n_observations', 'n_features', 'total_variance', 'mean_feature_variance',
    ]]


def show_coverage(dataset, max_points=20000):
    """Show MNI channel/source positions on a glass brain; skip large sets."""
    if dataset.n_features > max_points:
        print(f'Skipping {dataset.name} coverage: {dataset.n_features:,} features.')
        return None
    from nilearn import plotting
    pos = dataset.metadata[['x', 'y', 'z']].drop_duplicates().to_numpy()
    fig = plt.figure(figsize=(11, 3.5))
    plotting.plot_markers(
        np.ones(len(pos)), pos, node_size=8, node_cmap='Blues',
        node_vmin=0, node_vmax=1, node_threshold=None,
        figure=fig, colorbar=False,
        title=f'{dataset.name}: {dataset.n_features:,} features / {len(pos):,} locations',
    )
    plt.show()
    plt.close(fig)
    return fig


def compute_pca(dataset, n_components=10):
    """Compute centred, unwhitened PCA; retain dataset metadata for plots."""
    result = fit_block_pca(dataset, n_components=n_components)
    result.dataset = dataset
    return result


def plot_pca_timecourses(result):
    """Show all retained score time courses and individual/cumulative EVR."""
    data = result.dataset
    if data is None or data.source_data is None:
        raise ValueError('Use compute_pca on a dataset from load_dataset.')
    conditions = data.source_data.get('load_config', {}).get(
        'conditions', list(range(1, data.arrays[0].shape[0] + 1)))
    temporal = plot_pca(result, data.name, data.source_data['times'],
                        data.condition_mode, conditions)
    plt.show()
    plt.close(temporal)
    variance = plot_explained_variance({data.name: result})
    plt.show()
    plt.close(variance)
    return temporal, variance


def plot_pca_weights(result, n_components=3):
    """Signed PCA extraction weights on glass brains (MNI mm).

    Co-located weights are averaged ONLY for plotting, especially participant
    blocks in full_concatenated. Underlying PCA weights stay unchanged. Each
    component uses its own symmetric colour range, without magnitude threshold.
    """
    from nilearn import plotting
    if result.dataset is None:
        raise ValueError('Use compute_pca to attach source locations.')
    if n_components < 1:
        raise ValueError('n_components must be positive.')
    k = min(n_components, result.weights.shape[1])
    table = result.dataset.metadata[['x', 'y', 'z']].copy()
    if len(table) != len(result.weights):
        raise ValueError('Weight rows do not match feature coordinates.')
    columns = [f'PC{i+1}' for i in range(k)]
    table[columns] = result.weights[:, :k]
    grouped = table.groupby(['x', 'y', 'z'], sort=False)[columns].mean().reset_index()
    note = ' (mean at co-located features)' if len(grouped) < len(table) else ''
    fig, axes = plt.subplots(k, 1, figsize=(12, 3.3 * k), squeeze=False)
    for pc, ax in enumerate(axes[:, 0]):
        values = grouped[columns[pc]].to_numpy()
        limit = float(np.max(np.abs(values))) or 1.
        plotting.plot_markers(
            values, grouped[['x', 'y', 'z']].to_numpy(),
            node_size=7, node_cmap='RdBu_r', node_vmin=-limit, node_vmax=limit,
            node_threshold=None, colorbar=True, figure=fig, axes=ax,
            title=f'{result.dataset.name}: PC{pc+1}{note}',
        )
    plt.show()
    plt.close(fig)
    return fig


def _correlation_plot(matrix, name, title, plot):
    frame = pd.DataFrame(matrix,
                         index=[f'MEG PC{i+1}' for i in range(matrix.shape[0])],
                         columns=[f'iEEG PC{i+1}' for i in range(matrix.shape[1])])
    if plot:
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
        ax.set(title=f'{name}: {title}', xlabel='iEEG PC', ylabel='MEG PC',
               xticks=range(matrix.shape[1]), xticklabels=range(1, matrix.shape[1]+1),
               yticks=range(matrix.shape[0]), yticklabels=range(1, matrix.shape[0]+1))
        fig.colorbar(im, ax=ax, label='Signed Pearson r')
        plt.show()
        plt.close(fig)
    return frame


def _check_comparison(meg, ieeg):
    if meg.dataset is None or ieeg.dataset is None:
        raise ValueError('Both PCA results must come from compute_pca.')
    a, b = meg.dataset.source_data, ieeg.dataset.source_data
    if a is None or b is None:
        raise ValueError('Both datasets must come from load_dataset.')
    if ieeg.dataset.name != 'iEEG':
        raise ValueError('The second PCA must be the iEEG reference.')
    if meg.dataset.condition_mode != ieeg.dataset.condition_mode:
        raise ValueError('PCA condition modes differ.')
    if not np.array_equal(a['times'], b['times']):
        raise ValueError('Time axes differ.')
    if a.get('load_config', {}).get('conditions') != b.get('load_config', {}).get('conditions'):
        raise ValueError('Condition order differs.')
    if (not np.array_equal(a['electrode_subjects'], b['electrode_subjects']) or
            not np.array_equal(a['electrode_positions'], b['electrode_positions'])):
        raise ValueError('iEEG electrode reference/order differs.')


def correlate_timecourses(meg, ieeg, plot=True):
    """All MEG-PC x iEEG-PC score correlations; return a DataFrame and plot."""
    _check_comparison(meg, ieeg)
    matrix = cross_correlations(meg.scores, ieeg.scores)
    return _correlation_plot(matrix, meg.dataset.name, 'time courses', plot)


def correlate_weights(meg, ieeg, plot=True):
    """All PC weight correlations in the stored iEEG electrode correspondence.

    Full-source weights are sampled at nearest-source mappings. Random-control
    weights use non-anatomical random slots. No sign optimisation or p-values.
    """
    _check_comparison(meg, ieeg)
    weights = meg.weights[meg.dataset.electrode_to_feature]
    matrix = cross_correlations(weights, ieeg.weights)
    title = 'weights (random, non-anatomical slots)' if meg.dataset.name == 'random_control' else 'weights'
    return _correlation_plot(matrix, meg.dataset.name, title, plot)
