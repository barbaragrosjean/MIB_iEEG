"""Electrode-anchored cross-modal weight distances from aligned pairing fits."""
import numpy as np
import pandas as pd


def analyze_cross_modal_profiles(aligned, *, n_components=2, normalize_weights=True,
                                 distance_bin_width_mm=5.):
    """Compare each fixed electrode with its paired subject's unique MEG sources.

    Cosine uses absolute component weights. Semivariance uses profile magnitudes,
    divided by the equally weighted mean of the two within-modality variances
    (computed over the whole fit, with MEG duplicates removed within owner).
    Optional RMS normalization is one scalar per modality and association.
    Native MNI coordinates in mm are used, including cross-hemisphere distances.
    """
    width = float(distance_bin_width_mm)
    if not np.isfinite(width) or width <= 0:
        raise ValueError('Shell width must be positive and finite.')
    if not isinstance(n_components, (int, np.integer)) or n_components < 2:
        raise ValueError('Use at least two components for cosine profiles.')
    anchor = aligned['anchor'].reset_index(drop=True).copy()
    xyz = anchor[['x', 'y', 'z']].to_numpy(float)
    owners = anchor.subject.astype(str).to_numpy()
    weights = {m: np.asarray(aligned['weights'][m], float) for m in ('iEEG', 'MEG')}
    n_iterations = len(aligned['meg_metadata'])
    for m, w in weights.items():
        if (w.ndim != 3 or w.shape[0] != n_iterations or w.shape[1] != len(anchor)
                or w.shape[2] < n_components or not np.isfinite(w).all()):
            raise ValueError(f'{m}: expected finite association × paired-row × component weights.')
    if not len(anchor) or not n_iterations or not np.isfinite(xyz).all():
        raise ValueError('Nonempty fits and finite electrode coordinates are required.')
    rows, audits, comparisons = [], [], []
    maximum = 0.
    for iteration, metadata in enumerate(aligned['meg_metadata']):
        mm = metadata.reset_index(drop=True)
        if (len(mm) != len(anchor) or 'ieeg_subject' not in mm
                or not np.array_equal(mm.ieeg_subject.astype(str).to_numpy(), owners)):
            raise ValueError('MEG rows must preserve paired-coverage electrode ownership.')
        if 'electrode_index' in mm and not np.array_equal(mm.electrode_index, np.arange(len(mm))):
            raise ValueError('MEG electrode correspondence was reordered.')
        meg_xyz = mm[['x', 'y', 'z']].to_numpy(float)
        if not np.isfinite(meg_xyz).all():
            raise ValueError('Finite MEG MNI coordinates in mm are required.')
        unique = ~mm.assign(owner=owners).duplicated(['owner', 'x', 'y', 'z']).to_numpy()
        profiles = {m: np.abs(w[iteration, :, :n_components]).copy() for m, w in weights.items()}
        scales = {}
        for m, p in profiles.items():
            sample = p if m == 'iEEG' else p[unique]
            rms = np.sqrt(np.mean(np.sum(sample**2, axis=1)))
            scales[m] = rms if normalize_weights else 1.
            if normalize_weights and rms <= 0:
                raise ValueError(f'{m}: RMS normalization is undefined for all-zero weights.')
            profiles[m] /= scales[m]
        magnitude = {m: np.linalg.norm(p, axis=1) for m, p in profiles.items()}
        variance = .5 * (np.var(magnitude['iEEG']) + np.var(magnitude['MEG'][unique]))
        audits.append(dict(iteration=iteration, ieeg_scale=scales['iEEG'], meg_scale=scales['MEG'],
                           pooled_variance=variance, n_unique_meg_sources=int(unique.sum())))
        for electrode, subject in enumerate(owners):
            sources = np.flatnonzero(unique & (owners == subject))
            distance = np.linalg.norm(meg_xyz[sources] - xyz[electrode], axis=1)
            maximum = max(maximum, float(distance.max()))
            bins = np.floor(distance / width).astype(int)
            denom = magnitude['iEEG'][electrode] * magnitude['MEG'][sources]
            cosine = np.full(len(sources), np.nan)
            valid = denom > 0
            cosine[valid] = np.clip(1 - profiles['MEG'][sources[valid]] @ profiles['iEEG'][electrode] / denom[valid], 0, 1)
            semivariance = ((magnitude['iEEG'][electrode] - magnitude['MEG'][sources])**2 / (2 * variance)
                            if variance > 0 else np.full(len(sources), np.nan))
            comparisons.append(pd.DataFrame(dict(iteration=iteration, subject=subject,
                electrode_row=electrode, meg_row=sources,
                meg_x=meg_xyz[sources, 0], meg_y=meg_xyz[sources, 1], meg_z=meg_xyz[sources, 2],
                source_distance_mm=distance, bin=bins, distance_mm=(bins+.5)*width,
                cosine_distance=cosine, magnitude_semivariance=semivariance)))
            for metric, values in [('cosine_distance', cosine), ('magnitude_semivariance', semivariance)]:
                for b in np.unique(bins):
                    selected = values[bins == b]
                    finite = selected[np.isfinite(selected)]
                    n = len(finite)
                    rows.append(dict(iteration=iteration, subject=subject, electrode_row=electrode,
                        bin=int(b), distance_mm=(b+.5)*width, metric=metric,
                        mean=float(finite.mean()) if n else np.nan,
                        within_sem=float(finite.std(ddof=1)/np.sqrt(n)) if n > 1 else np.nan,
                        n_sources=len(selected), n_valid=n))
    curves = pd.DataFrame(rows)
    keys = ['subject', 'electrode_row', 'bin', 'distance_mm', 'metric']
    summary = curves.groupby(keys)['mean'].agg(['mean', 'std', 'count']).reset_index()
    summary = summary.rename(columns={'count': 'n_associations', 'std': 'association_sd'})
    summary['across_sem'] = summary.association_sd / np.sqrt(summary.n_associations)
    # Pool individual comparisons BEFORE averaging: each valid comparison has equal weight.
    comparison_values = pd.concat(comparisons, ignore_index=True)
    pooled_tables = []
    for metric in ('cosine_distance', 'magnitude_semivariance'):
        group_keys = keys[:-1]
        grouped = comparison_values.groupby(group_keys)[metric]
        pooled = grouped.agg(['mean', 'std', 'count', 'size']).reset_index().rename(
            columns={'std': 'comparison_sd', 'count': 'n_comparisons', 'size': 'n_total_comparisons'})
        pooled['pooled_sem'] = pooled.comparison_sd / np.sqrt(pooled.n_comparisons)
        contributors = comparison_values.loc[np.isfinite(comparison_values[metric])].groupby(group_keys).iteration.nunique()
        pooled = pooled.merge(contributors.rename('n_associations'), on=group_keys, how='left')
        pooled['n_associations'] = pooled.n_associations.fillna(0).astype(int)
        pooled['metric'] = metric
        pooled_tables.append(pooled)
    return dict(comparison_values=comparison_values, pooled_summary=pd.concat(pooled_tables, ignore_index=True),
                electrode_curves=curves, summary=summary, anchor=anchor,
                normalization_audit=pd.DataFrame(audits), max_distance_mm=maximum,
                n_iterations=n_iterations, n_components=n_components,
                normalize_weights=normalize_weights, distance_bin_width_mm=width,
                provenance=aligned.get('provenance', []), source_dir=aligned.get('source_dir'))


def plot_cross_modal_profiles(result, *, sem_mode='pooled', association=1,
                              figsize=(18, 6), cmap='turbo'):
    """Yield one figure per subject; association is one-based for within SEM."""
    import matplotlib.pyplot as plt
    from nilearn import plotting
    if sem_mode not in ('pooled', 'within', 'across'):
        raise ValueError("sem_mode must be 'pooled', 'within', or 'across'.")
    if not isinstance(association, (int, np.integer)) or not 1 <= association <= result['n_iterations']:
        raise ValueError('Association must be a valid one-based integer.')
    if sem_mode == 'within':
        table = result['electrode_curves'].query('iteration == @association - 1')
        sem_column, title = 'within_sem', f'association {association}: source SEM'
    elif sem_mode == 'pooled':
        table = result['pooled_summary']
        sem_column, title = 'pooled_sem', 'pooled comparisons: source–association SEM'
    else:
        table = result['summary']
        sem_column, title = 'across_sem', 'association SEM'
    anchor = result['anchor']
    for subject in anchor.subject.astype(str).unique():
        indices = np.flatnonzero(anchor.subject.astype(str).to_numpy() == subject)
        colors = plt.get_cmap(cmap)(np.linspace(.03, .97, len(indices)))
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        for electrode, color in zip(indices, colors):
            label = f'E{electrode + 1}'
            for ax, metric in zip(axes[:2], ['cosine_distance', 'magnitude_semivariance']):
                frame = table[(table.electrode_row == electrode) & (table.metric == metric)].set_index('bin')
                # Preserve gaps rather than connecting across empty shells.
                frame = frame.reindex(range(int(np.floor(result['max_distance_mm']/result['distance_bin_width_mm']))+1))
                x = (frame.index.to_numpy()+.5)*result['distance_bin_width_mm']
                y, sem = frame['mean'].to_numpy(), frame[sem_column].to_numpy()
                ax.plot(x, y, color=color, linewidth=1, label=label)
                ax.fill_between(x, y-sem, y+sem, color=color, alpha=.12)
        for ax, ylabel in zip(axes[:2], ['Absolute-weight cosine distance', 'Normalized magnitude semivariance']):
            ax.set(xlabel='Electrode–MEG source distance (mm)', ylabel=ylabel)
            ax.grid(alpha=.2)
        brain = plotting.plot_glass_brain(None, display_mode='ortho', figure=fig, axes=axes[2],
                                          title='Fixed iEEG electrodes')
        brain.add_markers(anchor.iloc[indices][['x', 'y', 'z']].to_numpy(float),
                          marker_color=colors, marker_size=18)
        axes[0].legend(fontsize=6, ncol=max(1, int(np.ceil(len(indices)/20))))
        fig.suptitle(f'{subject} · {title} · RMS normalization: {result["normalize_weights"]}')
        fig.subplots_adjust(wspace=.3, top=.86)
        yield subject, fig
