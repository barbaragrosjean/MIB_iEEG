"""Numerical component alignment; plotting remains in the notebook cells."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def component_correlations(reference, candidate):
    """Signed Pearson correlations: reference columns by candidate columns."""
    a, b = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0] or a.shape[0] < 2:
        raise ValueError('Component scores need the same >=2 observation rows.')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Component scores must be finite.')
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    denominator = np.outer(np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0))
    return np.divide(a.T @ b, denominator,
                     out=np.full(denominator.shape, np.nan), where=denominator > 0).clip(-1, 1)


def match_components(reference, candidate, *, weights=None, n_components=None):
    """Match to fixed reference ranks using absolute temporal correlations.

    Candidate columns may be reordered and sign-corrected, but original ranks
    and signed correlations remain in the audit. Native electrode/source rows
    are never reordered. This is descriptive alignment, not held-out inference.
    """
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    correlation = component_correlations(reference, candidate)
    k = min(correlation.shape) if n_components is None else n_components
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= min(correlation.shape):
        raise ValueError('Requested reference components exceed available score columns.')
    selected = correlation[:k]
    # Reject undefined scores rather than manufacture a component match.
    if not np.isfinite(selected).all():
        raise ValueError('Constant component time courses cannot be matched.')
    rows, columns = linear_sum_assignment(-np.abs(selected))
    if not np.array_equal(rows, np.arange(k)):
        raise ValueError('A complete one-to-one component assignment is required.')
    signed_r = selected[rows, columns]
    signs = np.where(signed_r < 0, -1., 1.)
    aligned_weights = None
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 2 or weights.shape[1] != candidate.shape[1]:
            raise ValueError('Weights must have one column per candidate component.')
        aligned_weights = weights[:, columns] * signs
    audit = pd.DataFrame(dict(reference_rank=rows + 1, original_rank=columns + 1,
                              signed_r=signed_r, absolute_r=np.abs(signed_r),
                              sign_correction=signs.astype(int)))
    return dict(scores=candidate[:, columns] * signs, weights=aligned_weights,
                columns=columns, signs=signs, audit=audit)


def cumulative_matched_correlation(a, b, max_components=None):
    """Mean absolute one-to-one correlation within the first k ORIGINAL PCs.

    The assignment is solved independently at each k; no reference alignment
    or unrestricted candidate pool is used in this cumulative metric.
    """
    correlation = component_correlations(a, b)
    maximum = min(correlation.shape)
    if max_components is not None:
        maximum = min(maximum, int(max_components))
    records = []
    for k in range(1, maximum + 1):
        block = correlation[:k, :k]
        if not np.isfinite(block).all():
            records.append(dict(k=k, mean_absolute_r=np.nan, status='undefined_correlation'))
            continue
        rows, columns = linear_sum_assignment(-np.abs(block))
        records.append(dict(k=k, mean_absolute_r=float(np.abs(block[rows, columns]).mean()), status='ok'))
    return pd.DataFrame(records)
