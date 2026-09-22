"""Compare separately fitted MEG and iEEG PCA temporal representations."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


def _basis(x):
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    tol = np.finfo(float).eps * max(x.shape) * s[0]
    if np.any(s <= tol):
        raise ValueError('Requested PCA scores are rank deficient; use fewer components.')
    return u


def compare_pca_to_ieeg(pca, reference='iEEG', n_components=(1, 2, 3, 5, 10), plot=True):
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
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        for ax, metric, title in zip(axes,
                ['ieeg_variance_captured', 'subspace_overlap', 'matched_abs_r'],
                ['iEEG retained-PC variance captured', 'Temporal subspace overlap', 'Matched component correlation']):
            for name, group in table.groupby('dataset', sort=False):
                group = group.sort_values('k')
                ax.plot(group.k, group[metric], 'o-', label=name)
            ax.set(title=title, xlabel='Number of PCs (k)', ylabel='Similarity (higher is closer)',
                   ylim=(-.02, 1.02), xticks=ks)
            ax.grid(alpha=.2)
        axes[-1].legend(fontsize=8)
        plt.show()
        plt.close(fig)
    return table, pd.DataFrame(pairs)
