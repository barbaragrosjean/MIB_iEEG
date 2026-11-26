"""Direct MEG manipulation comparisons and full-spectrum dimensionality.

No anatomical activation inference or iEEG signal comparison is performed here.

"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.linalg import eigh

MEG_KINDS = ('full_concatenated', 'full_average', 'coverage_average', 'paired_coverage')
COMPARISONS = (
    ('averaging', 'full_concatenated', 'full_average', 'Participant averaging'),
    ('coverage', 'full_average', 'coverage_average', 'Coverage restriction in averaged data'),
    ('paired_coverage', 'full_concatenated', 'paired_coverage', 'Coverage + participant contributions'),
)


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
    """Compare constructions directly by manipulation; none is ground truth.

    Inputs must have identical time/condition rows. Unavailable k values are
    recorded explicitly; no silent rank truncation. All comparisons are in sample.
    A_variance_captured_by_B targets the first k PCs of A using B as the basis;
    the reverse direction targets B. Temporal overlap is symmetric.
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
    for manipulation, a_name, b_name, title in COMPARISONS:
        a, b = fits[a_name], fits[b_name]
        for k in dimensions:
            valid = k <= min(a['rank'], b['rank'])
            overlap = float(np.clip(np.sum((a['basis'][:, :k].T@b['basis'][:, :k])**2)/k, 0, 1)) if valid else np.nan
            rows.append(dict(manipulation=manipulation, dataset_A=a_name, dataset_B=b_name,
                             k=k, temporal_overlap=overlap,
                             A_variance_captured_by_B=reference_fraction(a, b, k),
                             B_variance_captured_by_A=reference_fraction(b, a, k),
                             status='ok' if valid else 'insufficient_rank'))
    context = ref.source_data or {}
    return dict(config=dict(schema_version=2, dimensions=dimensions, comparisons=[list(c) for c in COMPARISONS],
                            condition_mode=ref.condition_mode, scope='in_sample; no ground-truth construction',
                            preprocessing=context.get('load_config', {}),
                            meg_subjects=context.get('meg_subjects', []),
                            pairing=getattr(selected['paired_coverage'], 'pairing', None),
                            n_features={name: ds.n_features for name, ds in selected.items()}),
                metrics=pd.DataFrame(rows), spectra=pd.DataFrame(spectra), summary=pd.DataFrame(summary))


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


