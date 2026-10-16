"""MEG temporal variance retention, full-spectrum dimensionality and trial reliability.

No anatomical activation inference or iEEG signal comparison is performed here.
Split halves are independent trials within each participant and condition; source
reconstruction remains the upstream, fixed transform stored in the trial cache.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.linalg import eigh

MEG_KINDS = ('full_concatenated', 'full_average', 'coverage_average', 'paired_coverage')


def _dimensions(values):
    values = list(values)
    if not values or any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1 for k in values):
        raise ValueError('dimensions must contain positive integers.')
    return sorted(set(map(int, values)))


def temporal_spectrum(dataset, max_gram_gib=2.):
    """Exact observation Gram spectrum, without a feature covariance or weights.

    Participation ratio uses ALL eigenvalues, never just requested leading PCs.
    Feature chunks bound copying for full participant concatenation.
    """
    n = dataset.n_observations
    if n < 2 or 6*n*n*8 / 2**30 > max_gram_gib:
        raise ValueError('Too few observations or observation Gram exceeds memory budget.')
    gram = np.zeros((n, n), dtype=float)
    for block in dataset.blocks():
        if len(block) != n:
            raise ValueError('Inconsistent observation count.')
        for start in range(0, block.shape[1], 1024):
            x = np.array(block[:, start:start+1024], dtype=float, copy=True)
            if not np.isfinite(x).all():
                raise ValueError('Signals must be finite.')
            x -= x.mean(0)
            gram += x @ x.T
    values, vectors = eigh(gram, check_finite=False)
    values, vectors = np.maximum(values[::-1], 0.), vectors[:, ::-1]
    total = values.sum()
    if total <= 0:
        raise ValueError('Dataset has zero temporal variance.')
    rank = int(np.sum(values > values[0]*np.finfo(float).eps*max(n, dataset.n_features)))
    return dict(basis=vectors[:, :rank], eigenvalues=values/(n-1), rank=rank,
                effective_dimensionality=float(total**2 / np.sum(values**2)))


def reference_fraction(reference, modified, k):
    """Fraction of the first k reference-score variance captured by k new PCs."""
    if k > min(reference['rank'], modified['rank']):
        return np.nan
    cross = modified['basis'][:, :k].T @ reference['basis'][:, :k]
    variance = reference['eigenvalues'][:k]
    return float(np.clip(np.sum(cross**2 * variance[None, :])/variance.sum(), 0, 1))


def compare_meg_information(datasets, dimensions=(1, 2, 3, 5, 10), max_gram_gib=2.):
    """Describe the four constructions against full-concatenated MEG.

    Inputs must have identical time/condition rows. Unavailable k values are
    recorded explicitly; no silent rank truncation. Full reference is in sample.
    """
    dimensions = _dimensions(dimensions)
    selected = {name: datasets[name] for name in MEG_KINDS}
    ref = selected['full_concatenated']
    for ds in selected.values():
        if ds.n_observations != ref.n_observations or ds.condition_mode != ref.condition_mode:
            raise ValueError('MEG observation axes must match.')
        if ds.source_data is not None and ref.source_data is not None:
            if not np.array_equal(ds.source_data['times'], ref.source_data['times']):
                raise ValueError('MEG time axes must match.')
            if ds.source_data.get('load_config', {}).get('conditions') != ref.source_data.get('load_config', {}).get('conditions'):
                raise ValueError('MEG condition orders must match.')
    fits = {name: temporal_spectrum(ds, max_gram_gib) for name, ds in selected.items()}
    rows, spectra, summary = [], [], []
    for name, fit in fits.items():
        ratio = fit['eigenvalues']/fit['eigenvalues'].sum()
        cumulative = np.cumsum(ratio)
        summary.append(dict(dataset=name, n_features=selected[name].n_features,
                            rank=fit['rank'], effective_dimensionality=fit['effective_dimensionality']))
        spectra.extend(dict(dataset=name, component=j+1, variance_ratio=float(v),
                            cumulative_variance=float(cumulative[j])) for j, v in enumerate(ratio))
        for k in dimensions:
            captured = reference_fraction(fits['full_concatenated'], fit, k)
            rows.append(dict(dataset=name, k=k, reference_variance_captured=captured,
                             unrecovered_reference_variance=1-captured,
                             cumulative_variance=float(cumulative[k-1]) if k <= fit['rank'] else np.nan,
                             status='ok' if np.isfinite(captured) else 'insufficient_rank'))
    context = ref.source_data or {}
    return dict(config=dict(dimensions=dimensions, reference='full_concatenated',
                            condition_mode=ref.condition_mode, scope='in_sample',
                            preprocessing=context.get('load_config', {}),
                            meg_subjects=context.get('meg_subjects', []),
                            pairing=getattr(selected['paired_coverage'], 'pairing', None),
                            n_features={name: ds.n_features for name, ds in selected.items()}),
                metrics=pd.DataFrame(rows), spectra=pd.DataFrame(spectra), summary=pd.DataFrame(summary))


def _trial_halves(arrays, rng):
    halves = [[], []]
    audit = []
    for condition_index, array in enumerate(arrays):
        if array.ndim != 3 or len(array) < 2:
            raise ValueError('At least two individual trials per participant/condition are required.')
        order = rng.permutation(len(array))
        # Use every trial; an odd trial count differs by one between halves.
        for half, indices in enumerate(np.array_split(order, 2)):
            mean = np.zeros(array.shape[1:], dtype=float)
            for index in indices:
                mean += array[index]
                audit.append(dict(condition_index=condition_index, half=half, trial_index=int(index)))
            halves[half].append(mean/len(indices))
    return [np.stack(part) for part in halves], audit


def split_half_reliability(reference, cache_dir, *, pairing, dimensions=(1, 2, 3, 5, 10),
                           repeats=20, seed=2026, max_gram_gib=2.):
    """Independently preprocess/refit each trial half, using fixed source pairing.

    reference is the loaded iEEG Dataset used ONLY for coverage metadata and the
    MEG cohort/configuration. MEG trial arrays come from the existing trial cache.
    No iEEG recordings are loaded. Both halves use the same trials across all four
    constructions. Repeats overlap and their ranges are not confidence intervals.
    """
    from coverage_matching_utils import construct_five_datasets, _preprocess
    dimensions = _dimensions(dimensions)
    if isinstance(repeats, bool) or not isinstance(repeats, (int, np.integer)) or repeats < 1:
        raise ValueError('repeats must be a positive integer.')
    context = reference.source_data
    if context is None or pairing is None:
        raise ValueError('Supply loaded reference metadata and the existing paired-coverage assignment.')
    root = Path(cache_dir)
    manifest = json.loads((root/'manifest.json').read_text())
    conditions = list(context['load_config']['conditions'])
    if manifest['config']['conditions'] != conditions or not np.allclose(
            np.asarray(manifest['times']), context['times'], atol=1e-7, rtol=0):
        raise ValueError('Trial cache time/condition axes differ from the coverage analysis.')
    records = {r['subject']: r for r in manifest['records'] if r['modality'] == 'meg'}
    subjects = context['meg_subjects']
    arrays = []
    for subject, positions in zip(subjects, context['meg_positions']):
        record = records[subject]
        if np.shape(record['positions']) != positions.shape or not np.allclose(record['positions'], positions):
            raise ValueError(f'{subject}: cache source positions/order differ from coverage data.')
        data = [np.load(root/f, mmap_mode='r', allow_pickle=False) for f in record['files']]
        if len(data) != len(conditions) or any(a.ndim != 3 or a.shape[1:] != (len(positions), len(context['times'])) for a in data):
            raise ValueError(f'{subject}: trial cache dimensions differ from coverage data.')
        arrays.append(data)
    metrics, audit = [], []
    rng = np.random.default_rng(seed)
    scaling = context['load_config']['meg_scaling']
    for repeat in range(repeats):
        means = [[], []]
        for subject, data in zip(subjects, arrays):
            parts, split = _trial_halves(data, rng)
            audit.extend(dict(repeat=repeat, subject=subject, condition=conditions[r['condition_index']],
                              half=r['half'], trial_index=r['trial_index']) for r in split)
            for half in (0, 1):
                means[half].append(_preprocess(parts[half], scaling))
        fits = []
        for half in (0, 1):
            datasets, _, _ = construct_five_datasets(
                means[half], context['meg_positions'], subjects, context['electrode_positions'],
                context['electrode_subjects'], pairing=pairing, seed=seed,
                condition_mode=reference.condition_mode, kinds=MEG_KINDS)
            fits.append({name: temporal_spectrum(ds, max_gram_gib) for name, ds in datasets.items()})
        for name in MEG_KINDS:
            a, b = fits[0][name], fits[1][name]
            for k in dimensions:
                valid = k <= min(a['rank'], b['rank'])
                overlap = float(np.clip(np.sum((a['basis'][:, :k].T@b['basis'][:, :k])**2)/k, 0, 1)) if valid else np.nan
                metrics.append(dict(dataset=name, repeat=repeat, k=k, overlap=overlap,
                                    status='ok' if valid else 'insufficient_rank'))
        print(f'MEG split-half reliability: {repeat+1}/{repeats}', flush=True)
    return dict(config=dict(dimensions=dimensions, repeats=int(repeats), seed=seed,
                            pairing=dict(pairing), subjects=list(subjects), conditions=conditions,
                            meg_scaling=scaling, condition_mode=reference.condition_mode,
                            cache_dir=str(root.resolve()), preprocessing='independent per half',
                            scope='trial reliability conditional on upstream source reconstruction'),
                reliability=pd.DataFrame(metrics), split_audit=pd.DataFrame(audit))


def save_information(result, output_dir):
    """Refuse overwriting an existing run; write completion marker last."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError('Choose an empty output directory to avoid mixing configurations.')
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(root/f'{key}.csv', index=False)
    (root/'config.json').write_text(json.dumps(result['config'], indent=2))
    (root/'COMPLETE.json').write_text(json.dumps([k for k,v in result.items() if isinstance(v, pd.DataFrame)]))


def load_information(output_dir):
    root = Path(output_dir)
    tables = json.loads((root/'COMPLETE.json').read_text())
    return dict(config=json.loads((root/'config.json').read_text()),
                **{key: pd.read_csv(root/f'{key}.csv') for key in tables})


def plot_meg_information(result):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout='constrained')
    for name in MEG_KINDS:
        frame = result['metrics'].query('dataset == @name')
        axes[0].plot(frame.k, frame.reference_variance_captured, 'o-', label=name)
        spectrum = result['spectra'].query('dataset == @name')
        spectrum = spectrum[spectrum.component <= max(result['config']['dimensions'])]
        axes[1].plot(spectrum.component, spectrum.cumulative_variance, 'o-', label=name)
    for ax, title in zip(axes[:2], ['Full-concatenated reference variance captured', 'Cumulative explained variance']):
        ax.set(title=title, xlabel='Number of components', ylabel='Fraction', ylim=(0, 1.02))
        ax.grid(alpha=.2)
    summary = result['summary'].set_index('dataset').loc[list(MEG_KINDS)]
    axes[2].bar(range(len(summary)), summary.effective_dimensionality)
    axes[2].set(xticks=range(len(summary)), xticklabels=summary.index,
                title='Effective dimensionality (full spectrum)', ylabel='Participation ratio')
    axes[2].tick_params(axis='x', rotation=35)
    axes[0].legend(fontsize=8)
    fig.suptitle('MEG construction sensitivity · descriptive, in sample')
    return fig


def plot_split_half_reliability(result):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for name, frame in result['reliability'].groupby('dataset', sort=False):
        stats = frame.groupby('k').overlap.agg(['median', 'min', 'max'])
        ax.plot(stats.index, stats['median'], 'o-', label=name)
        ax.fill_between(stats.index, stats['min'], stats['max'], alpha=.15)
    ax.set(xlabel='Number of components', ylabel='Temporal subspace overlap', ylim=(0, 1.02),
           title='Independent trial halves: median and range across repeated splits')
    ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.supxlabel('Ranges describe split sensitivity, not confidence intervals. No Spearman–Brown correction.', fontsize=9)
    return fig
