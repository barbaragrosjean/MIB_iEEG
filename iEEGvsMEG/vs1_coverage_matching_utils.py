from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import vs0_load_and_config as shared_data

# Existing sections may continue importing these shared helpers here.
coordinates_mm = shared_data.coordinates_mm
ieeg_getinfo_coordinates_m = shared_data.ieeg_getinfo_coordinates_m
_preprocess = shared_data._preprocess
load_project_data = shared_data.load_project_data


SETUP_NAMES = (
    'full_average', 'full_concatenated', 'coverage_average',
    'paired_coverage', 'random_control',
)


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
    scale: float = 1.
    block_scaling: bool = False

    def blocks(self):
        for a in self.arrays:
            values = observations(a, self.condition_mode)
            yield values if self.scale == 1. else values * self.scale

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

    Full concatenation is a list of participant blocks.
    Group coverage is one nearest-source feature per pooled iEEG electrode,
    selected separately in EVERY MEG participant, then participant-averaged.
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
    if 'full_average' in requested and any(a.shape != shape for a in meg):
        raise ValueError('full_average requires the same MEG source count, condition count and time-sample count.')
    # Average existing source rows directly; no coordinate alignment or reordering.
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
    """Spearman correlation between columns; constant columns return NaN."""
    from scipy.stats import rankdata

    a = np.asarray(a, float)
    b = np.asarray(b, float)

    if a.shape[0] != b.shape[0]:
        raise ValueError('Correlation requires aligned observations/features.')

    a = np.apply_along_axis(rankdata, 0, a)
    b = np.apply_along_axis(rankdata, 0, b)
    a -= np.mean(a, axis=0)
    b -= np.mean(b, axis=0)
    denominator = (np.linalg.norm(a, axis=0)[:, None]* np.linalg.norm(b, axis=0)[None, :])
    return np.divide(a.T @ b,denominator,out=np.full(denominator.shape, np.nan),where=denominator > 0).clip(-1, 1)


def load_dataset(kind, *, reference=None, condition_mode=None, seed=2026,
                 pairing=None, block_scaling=None, **file_options):
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
    enabled = reference.block_scaling if block_scaling is None and reference is not None else bool(block_scaling)
    return normalize_dataset(dataset, enabled)

def compute_variance(dataset):
    return variance_summary({dataset.name: dataset}).loc[dataset.name, [
        'n_observations', 'n_features', 'total_variance', 'mean_feature_variance',
    ]]


def compute_pca(dataset, n_components=10):
    """Compute centred, unwhitened PCA; retain dataset metadata for plots."""
    result = fit_block_pca(dataset, n_components=n_components)
    result.dataset = dataset
    return result


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

def correlate_timecourses(meg, ieeg, plot=False):
    """All MEG-PC x iEEG-PC score spearman correlations; return a DataFrame and plot."""
    _check_comparison(meg, ieeg)
    matrix = cross_correlations(meg.scores, ieeg.scores)
    matrix
    return _correlation_plot(matrix, meg.dataset.name, 'time courses', plot)

def correlate_weights(meg, ieeg, plot=False):
    """All PC weight correlations in the stored iEEG electrode correspondence.

    Full-source weights are sampled at nearest-source mappings. Random-control
    weights use non-anatomical random slots. No sign optimisation or p-values.
    """
    _check_comparison(meg, ieeg)
    weights = meg.weights[meg.dataset.electrode_to_feature]
    matrix = cross_correlations(weights, ieeg.weights)
    title = 'weights (random, non-anatomical slots)' if meg.dataset.name == 'random_control' else 'weights'
    return _correlation_plot(matrix, meg.dataset.name, title, plot)


def _basis(x):
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    tol = np.finfo(float).eps * max(x.shape) * s[0]
    if np.any(s <= tol):
        raise ValueError('Requested PCA scores are rank deficient; use fewer components.')
    return u


def compare_pca_to_ieeg(pca, reference='iEEG', n_components=(1, 2, 3, 5, 10), plot=False):
    """Return a ranking table and a time-selected component-pair table.

    pca is the notebook dictionary of PCAResult objects (with .scores).
    All scores must represent identical ordered times/conditions. No refitting.

    ieeg_variance_captured: fraction of the first k iEEG score variance captured
        by projection into the first k MEG temporal PCs (primary, higher better).
    subspace_overlap: mean squared cosine of principal angles; weights the k
        temporal dimensions equally (higher better, invariant to basis rotation).
    matched_abs_r: mean absolute Pearson r after one-to-one assignment among
        the first k PCs (sign/order invariant, not arbitrary-rotation invariant).

    These are descriptive in-sample metrics. Shared smooth/task-locked signals
    can yield high similarity. Larger k is more flexible: compare setups at the
    same k, and do not choose k solely because it gives the highest similarity.
    Weight geometry is a separate question; this ranks temporal resemblance.
    """
    if reference not in pca:
        raise KeyError(f'Missing reference {reference!r}.')
    reference_scores = np.asarray(pca[reference].scores, dtype=float)
    if reference_scores.ndim != 2 or not np.isfinite(reference_scores).all():
        raise ValueError('Reference scores must be a finite observations-by-PC matrix.')
    if np.isscalar(n_components):
        n_components = (n_components,)
    ks = sorted(set(n_components))
    if not ks or any(not isinstance(k, (int, np.integer)) or k < 1 for k in ks):
        raise ValueError('Component counts must be positive integers.')
    rows, pairs = [], []
    for name, result in pca.items():
        if name == reference:
            continue
        scores = np.asarray(result.scores, dtype=float)
        if scores.ndim != 2 or scores.shape[0] != reference_scores.shape[0] or not np.isfinite(scores).all():
            raise ValueError(f'{name}: scores must have the same observations as iEEG and be finite.')
        # Check available metadata, without requiring one particular PCA class.
        a, b = getattr(result, 'dataset', None), getattr(pca[reference], 'dataset', None)
        if a is not None and b is not None:
            if a.condition_mode != b.condition_mode:
                raise ValueError(f'{name}: condition modes differ.')
            da, db = getattr(a, 'source_data', None), getattr(b, 'source_data', None)
            if da is not None and db is not None:
                if not np.array_equal(da['times'], db['times']):
                    raise ValueError(f'{name}: time axes differ.')
                if da.get('load_config', {}).get('conditions') != db.get('load_config', {}).get('conditions'):
                    raise ValueError(f'{name}: condition orders differ.')
        for k in ks:
            if k > min(scores.shape[1], reference_scores.shape[1]):
                raise ValueError(f'{name}: {k} PCs requested but fewer are available. Choose a shared supported k.')
            x, y = scores[:, :k].copy(), reference_scores[:, :k].copy()
            x -= x.mean(axis=0)
            y -= y.mean(axis=0)
            qx, qy = _basis(x), _basis(y)
            captured = np.sum((qx.T @ y)**2) / np.sum(y**2)
            overlap = np.sum((qx.T @ qy)**2) / k
            corr = (x.T @ y) / np.outer(np.linalg.norm(x, axis=0), np.linalg.norm(y, axis=0))
            corr = np.clip(corr, -1, 1)
            ix, iy = linear_sum_assignment(-np.abs(corr))
            matched = np.abs(corr[ix, iy]).mean()
            rows.append(dict(dataset=name, k=k, ieeg_variance_captured=float(np.clip(captured, 0, 1)),
                             subspace_overlap=float(np.clip(overlap, 0, 1)), matched_abs_r=float(matched)))
            for i, j in zip(ix, iy):
                pairs.append(dict(dataset=name, k=k, meg_pc=int(i+1), ieeg_pc=int(j+1),
                                  signed_r=float(corr[i, j]), abs_r=float(abs(corr[i, j]))))
    if not rows:
        raise ValueError('Provide at least one MEG PCA in addition to iEEG.')
    table = pd.DataFrame(rows)
    table['rank'] = table.groupby('k')['ieeg_variance_captured'].rank(ascending=False, method='min').astype(int)
    table = table.sort_values(['k', 'rank', 'dataset']).reset_index(drop=True)
    if plot:
        from vs1_coverage_figures import plot_pca_comparison
        plot_pca_comparison(table)
    return table, pd.DataFrame(pairs)


def plot_pca(*args, **kwargs):
    """Compatibility entry point; plotting lives in vs1_coverage_figures."""
    from vs1_coverage_figures import plot_pca as draw
    return draw(*args, **kwargs)


def plot_explained_variance(*args, **kwargs):
    """Compatibility entry point; plotting lives in vs1_coverage_figures."""
    from vs1_coverage_figures import plot_explained_variance as draw
    return draw(*args, **kwargs)


def show_coverage(*args, **kwargs):
    """Compatibility entry point; plotting lives in vs1_coverage_figures."""
    from vs1_coverage_figures import show_coverage as draw
    return draw(*args, **kwargs)


def plot_pca_timecourses(*args, **kwargs):
    """Compatibility entry point; plotting lives in vs1_coverage_figures."""
    from vs1_coverage_figures import plot_pca_timecourses as draw
    return draw(*args, **kwargs)


def _brain_weights(*args, **kwargs):
    """Compatibility entry point; plotting lives in vs1_coverage_figures."""
    from vs1_coverage_figures import _brain_weights as draw
    return draw(*args, **kwargs)


def plot_voxel_weights(*args, **kwargs):
    """Compatibility entry point; plotting lives in vs1_coverage_figures."""
    from vs1_coverage_figures import plot_voxel_weights as draw
    return draw(*args, **kwargs)


def plot_glasser_weights(*args, **kwargs):
    """Compatibility entry point; plotting lives in vs1_coverage_figures."""
    from vs1_coverage_figures import plot_glasser_weights as draw
    return draw(*args, **kwargs)


def plot_pca_weights(*args, **kwargs):
    """Compatibility entry point; plotting lives in vs1_coverage_figures."""
    from vs1_coverage_figures import plot_pca_weights as draw
    return draw(*args, **kwargs)


def _correlation_plot(*args, **kwargs):
    """Compatibility entry point; plotting lives in vs1_coverage_figures."""
    from vs1_coverage_figures import _correlation_plot as draw
    return draw(*args, **kwargs)


def normalize_dataset(dataset, enabled=True):
    """Scale a whole constructed dataset to total sample variance one.

    The raw shared recordings are unchanged. Scaling follows condition handling.
    """
    dataset.scale = 1.
    dataset.block_scaling = bool(enabled)
    if enabled:
        variance = sum(np.var(x, axis=0, ddof=1, dtype=np.float64).sum() for x in dataset.blocks())
        if variance <= 0:
            raise ValueError(f'{dataset.name}: total variance must be positive.')
        dataset.scale = float(1 / np.sqrt(variance))
    return dataset
