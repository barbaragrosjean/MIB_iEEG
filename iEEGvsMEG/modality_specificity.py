"""Descriptive native-feature model weight magnitudes (no channel matching)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting


def threshold_weights(weights):
    """Threshold |W| independently per component at mean + population SD."""
    magnitude = np.abs(np.asarray(weights, dtype=float))
    if magnitude.ndim != 2 or 0 in magnitude.shape or not np.isfinite(magnitude).all():
        raise ValueError('Weights must be a nonempty finite channels × components matrix.')
    mean, std = magnitude.mean(axis=0), magnitude.std(axis=0, ddof=0)
    thresholds = mean + std
    retained = np.where(magnitude >= thresholds[None, :], magnitude, 0.)
    summary = pd.DataFrame(dict(component=np.arange(1, magnitude.shape[1]+1),
        mean_magnitude=mean, std_magnitude=std, threshold=thresholds,
        n_channels=magnitude.shape[0], n_retained=(retained > 0).sum(axis=0),
        fraction_retained=(retained > 0).mean(axis=0)))
    return retained, summary


def plot_weight_distributions(weights, summaries, bins=60):
    """Separate density panels preserve native magnitude scales and channel counts."""
    k = next(iter(weights.values())).shape[1]
    fig, axes = plt.subplots(k, len(weights), figsize=(11, 2.5*k), squeeze=False,
                             constrained_layout=True)
    for col, (modality, values) in enumerate(weights.items()):
        for component in range(k):
            ax = axes[component, col]
            ax.hist(np.abs(values[:, component]), bins=bins, density=True, color=('navy' if col == 0 else 'darkorange'), alpha=.65)
            ax.axvline(summaries[modality].threshold.iloc[component], color='crimson', linestyle='--', label='Mean + SD')
            ax.set(title=f'{modality} · component {component+1}', xlabel='Absolute projection weight', ylabel='Density')
    axes[0, 0].legend()
    return fig


def plot_thresholded_brains(datasets, retained):
    """MNI glass brains; exact co-locations use mean including thresholded zeros."""
    k = next(iter(retained.values())).shape[1]
    fig, axes = plt.subplots(k, len(datasets), figsize=(16, 3*k), squeeze=False)
    for col, (modality, dataset) in enumerate(datasets.items()):
        xyz = dataset.metadata[['x', 'y', 'z']].to_numpy(dtype=float)
        values = retained[modality]
        if xyz.shape != (len(values), 3) or not np.isfinite(xyz).all():
            raise ValueError(f'{modality}: weights require matching finite MNI coordinates in mm.')
        for component in range(k):
            table = pd.DataFrame(xyz, columns=['x', 'y', 'z'])
            table['weight'] = values[:, component]
            grouped = table.groupby(['x', 'y', 'z'], sort=False).weight.mean().reset_index()
            visible = grouped[grouped.weight > 0]
            title = f'{modality} · component {component+1}'
            if len(grouped) < len(table):
                title += ' · mean at co-locations'
            if visible.empty:
                plotting.plot_glass_brain(None, figure=fig, axes=axes[component, col], title=title+' (none retained)')
            else:
                plotting.plot_markers(visible.weight.to_numpy(), visible[['x', 'y', 'z']].to_numpy(),
                    node_size=7, node_cmap='Reds', node_vmin=0,
                    node_vmax=float(visible.weight.max()), node_threshold=None,
                    colorbar=True, figure=fig, axes=axes[component, col], title=title)
    return fig


def spatial_metadata(metadata):
    """Copy native-feature metadata with a canonical subject column.

    MEG ownership takes precedence over the paired iEEG subject label.
    Preserve row order and participant_average labels; never invent owners.
    """
    out = metadata.copy()
    owner = next((name for name in ('meg_subject', 'subject', 'ieeg_subject')
                  if name in out.columns), None)
    if owner is None:
        raise ValueError('Spatial analysis requires subject, meg_subject or ieeg_subject metadata.')
    if out[owner].isna().any():
        raise ValueError(f'Missing participant labels in {owner}.')
    out['subject'] = out[owner].astype(str)
    missing = {'x', 'y', 'z'} - set(out.columns)
    if missing:
        raise ValueError(f'Missing coordinate columns: {sorted(missing)}')
    return out


def spatial_weight_distance(weights, metadata, *, bin_edges=None, n_permutations=199,
                            seed=2026, block_size=512, progress=None):
    """Exact all-pair cosine-distance curves, group and individual subjects.

    Uses signed, unthresholded profiles from ONE group fit. Each unordered
    pair occurs once, including distinct features at identical coordinates.
    Null shuffles profiles within subjects against fixed coordinates, exactly
    equivalent to shuffling coordinates within subjects. No pair subsampling.
    Memory is bounded by block_size**2; runtime is quadratic in feature count
    and linear in permutation count. Null envelopes are pointwise, not CIs.
    """
    from scipy.spatial.distance import cdist
    metadata = spatial_metadata(metadata)
    w = np.asarray(weights, dtype=float)
    xyz = metadata[['x', 'y', 'z']].to_numpy(dtype=float)
    if w.ndim != 2 or w.shape[1] < 1 or xyz.shape != (len(w), 3):
        raise ValueError('Weights and coordinate rows must match.')
    if not np.isfinite(w).all() or not np.isfinite(xyz).all() or metadata.subject.isna().any():
        raise ValueError('Finite weights/coordinates and subject labels are required.')
    if not isinstance(n_permutations, (int, np.integer)) or n_permutations < 2:
        raise ValueError('Use at least two permutations.')
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError('block_size must be positive.')
    edges = np.asarray(np.arange(0, 310, 10) if bin_edges is None else bin_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or edges[0] != 0 or np.any(np.diff(edges) <= 0):
        raise ValueError('Bin edges must start at zero and increase strictly.')
    # Bounding-box diagonal ensures every pair is covered, including the final edge.
    if len(xyz) and np.linalg.norm(np.ptp(xyz, axis=0)) > edges[-1]:
        raise ValueError('Increase the last bin edge to cover the coordinate bounding-box diagonal.')
    norm = np.linalg.norm(w, axis=1)
    valid = norm > 0
    subjects = metadata.subject.astype(str).to_numpy()
    audit = pd.DataFrame([dict(subject=s, n_channels=int((subjects == s).sum()),
        n_zero_norm=int(((subjects == s) & ~valid).sum())) for s in pd.unique(subjects)])
    w, xyz, subjects = w[valid] / norm[valid, None], xyz[valid], subjects[valid]
    labels = list(pd.unique(subjects))
    groups = [np.flatnonzero(subjects == s) for s in labels]
    subject_index = np.array([labels.index(s) for s in subjects])
    nb = len(edges)-1
    counts = np.zeros((len(labels)+1, nb), dtype=np.int64)
    sums = np.zeros((n_permutations+1, len(labels)+1, nb))
    rng = np.random.default_rng(seed)
    permutations = [np.arange(len(w))]
    for _ in range(n_permutations):
        p = np.arange(len(w))
        for rows in groups:
            p[rows] = rng.permutation(rows)
        permutations.append(p)
    for start in range(0, len(w), block_size):
        ii = np.arange(start, min(start+block_size, len(w)))
        for other in range(start, len(w), block_size):
            jj = np.arange(other, min(other+block_size, len(w)))
            mask = ii[:, None] < jj[None, :]
            a, b = np.nonzero(mask)
            if not len(a):
                continue
            bins = np.minimum(np.searchsorted(edges, cdist(xyz[ii], xyz[jj])[a, b], side='right')-1, nb-1)
            same = subject_index[ii[a]] == subject_index[jj[b]]
            within_key = (subject_index[ii[a[same]]]+1)*nb + bins[same]
            counts[0] += np.bincount(bins, minlength=nb)
            counts += np.bincount(within_key, minlength=counts.size).reshape(counts.shape)
            for repeat, p in enumerate(permutations):
                distances = np.clip(1 - w[p[ii]] @ w[p[jj]].T, 0, 2)[a, b]
                sums[repeat, 0] += np.bincount(bins, weights=distances, minlength=nb)
                sums[repeat] += np.bincount(within_key, weights=distances[same], minlength=counts.size).reshape(counts.shape)
        if progress is not None:
            progress(f'Processed {min(start+block_size, len(w))}/{len(w)} channels')
    means = np.divide(sums, counts[None], out=np.full_like(sums, np.nan), where=counts[None]>0)
    rows, null_rows = [], []
    for group, label in enumerate(['all_channels']+labels):
        scope = 'group' if group == 0 else 'subject'
        for j in range(nb):
            observed, null = means[0, group, j], means[1:, group, j]
            populated = counts[group, j] > 0
            # Two-sided permutation tail test; ties included; Monte Carlo +1 correction.
            p = min(1., 2*min((1+np.sum(null <= observed))/(n_permutations+1),
                               (1+np.sum(null >= observed))/(n_permutations+1))) if populated else np.nan
            rows.append(dict(scope=scope, subject=label, distance_low_mm=edges[j], distance_high_mm=edges[j+1],
                distance_mid_mm=(edges[j]+edges[j+1])/2, n_pairs=counts[group,j],
                cosine_distance=observed, null_mean=float(null.mean()) if populated else np.nan,
                null_low=float(np.quantile(null,.025)) if populated else np.nan,
                null_high=float(np.quantile(null,.975)) if populated else np.nan, p_two_sided=p))
            for repeat, value in enumerate(null):
                null_rows.append(dict(scope=scope, subject=label, bin_index=j, permutation=repeat,
                                      cosine_distance=value))
    table = pd.DataFrame(rows)
    # Holm adjustment across every populated group/subject bin in this modality.
    table['p_holm'] = np.nan
    ix = table.index[table.p_two_sided.notna()]
    ordered = ix[np.argsort(table.loc[ix, 'p_two_sided'].to_numpy())]
    table.loc[ordered, 'p_holm'] = np.minimum(1, np.maximum.accumulate(
        table.loc[ordered, 'p_two_sided'].to_numpy()*np.arange(len(ordered),0,-1)))
    return {'curves': table, 'null_curves': pd.DataFrame(null_rows), 'channel_audit': audit}


def plot_spatial_weight_distance(result, title=''):
    """Group plus individual-subject curves with pointwise shuffle envelopes."""
    table = result['curves']
    figures = []
    for scope in ('group', 'subject'):
        groups = list(table[table.scope == scope].groupby('subject', sort=False))
        if not groups:
            continue
        cols = min(3, len(groups))
        fig, axes = plt.subplots(int(np.ceil(len(groups)/cols)), cols,
            figsize=(5*cols, 3.5*int(np.ceil(len(groups)/cols))), squeeze=False, constrained_layout=True)
        for ax, (subject, frame) in zip(axes.flat, groups):
            ax.fill_between(frame.distance_mid_mm, frame.null_low, frame.null_high, color='gray', alpha=.25, label='Null 95% pointwise envelope')
            ax.plot(frame.distance_mid_mm, frame.null_mean, '--', color='gray', label='Null mean')
            ax.plot(frame.distance_mid_mm, frame.cosine_distance, 'o-', label='Observed', markersize=3)
            significant = frame.p_holm < .05
            ax.scatter(frame.loc[significant,'distance_mid_mm'], frame.loc[significant,'cosine_distance'], marker='*', s=90, color='crimson', label='Holm p < .05')
            ax.set(title=subject, xlabel='MNI distance (mm)', ylabel='Cosine distance', ylim=(-.05,2.05))
        for ax in list(axes.flat)[len(groups):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f'{title} · {scope} · signed group-model weights')
        figures.append(fig)
    return figures
