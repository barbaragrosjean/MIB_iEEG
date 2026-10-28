"""Descriptive radial profiles around local maxima of fitted weight norms.

Distances are Euclidean MNI distances in mm, not cortical geodesic distances.
Pooling retains each measured feature; it does not equalize participant coverage.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.spatial import cKDTree


def gaussian_decay(distance, baseline, amplitude, sigma):
    return baseline + amplitude * np.exp(-0.5 * (distance / sigma) ** 2)


def fit_decay(distance, amplitude):
    """Fit nonnegative baseline/amplitude and positive width, excluding the peak.

    R² and RMSE are in-sample descriptive diagnostics, not normality tests.
    Excluding zero-distance points avoids forcing the fit through the selected
    maximum. Multiple starts reduce sensitivity to the initial width.
    """
    distance, amplitude = np.asarray(distance), np.asarray(amplitude)
    keep = (distance > 0) & np.isfinite(distance) & np.isfinite(amplitude)
    x, y = distance[keep], amplitude[keep]
    result = dict(fit_status='insufficient_data', n_fit=len(x), baseline=np.nan,
                  amplitude=np.nan, sigma_mm=np.nan, r_squared=np.nan, rmse=np.nan)
    if len(x) < 6 or len(np.unique(x)) < 4:
        return result
    if np.ptp(y) <= np.finfo(float).eps * max(1., np.max(y)):
        result['fit_status'] = 'constant_amplitude'
        return result
    scale = np.max(y)
    upper = 10 * np.max(x)
    candidates = []
    for width in (np.max(x) / 10, np.max(x) / 3, np.max(x)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)
                pars, _ = curve_fit(
                    gaussian_decay, x, y / scale,
                    p0=(np.min(y) / scale, np.ptp(y) / scale, width),
                    bounds=([0, 0, 1e-6], [np.inf, np.inf, upper]),
                    maxfev=20000)
            error = np.sum((y / scale - gaussian_decay(x, *pars)) ** 2)
            candidates.append((error, pars))
        except (RuntimeError, ValueError, OptimizeWarning):
            continue
    if not candidates:
        result['fit_status'] = 'fit_failed'
        return result
    error, pars = min(candidates, key=lambda item: item[0])
    baseline, height, sigma = pars
    status = 'ok'
    if sigma >= upper * .99 or sigma <= 1e-5 or height < 1e-6:
        status = 'weakly_identified_or_boundary'
    result.update(fit_status=status, baseline=baseline * scale,
                  amplitude=height * scale, sigma_mm=sigma,
                  r_squared=1 - error / np.sum((y / scale - np.mean(y / scale)) ** 2),
                  rmse=np.sqrt(error / len(y)) * scale)
    return result


def analyze_gradients(weights, metadata, *, n_components=3, spatial_scope='pooled',
                      neighborhood_mm=15., top_percent=5., max_peaks=None, min_neighbors=1,
                      fit_radius_mm=60.):
    """Find local maxima, rank them, and fit their radial amplitude profiles.

    A maximum has no larger norm within neighborhood_mm. Equal-norm neighbors
    resolve to the earlier feature row. min_neighbors counts other features.
    neighborhood_mm=None selects ceil(top_percent / 100 * n_features) highest
    norms instead. Ties resolve by feature row; zero norms are never peaks.
    max_peaks optionally limits plotted/fitted peaks per scope; None retains all.
    Distances to *all* other detected maxima are retained for red plot lines.
    Within-subject analysis restricts features from the same group fit; it does
    not refit the model separately. Input rows must remain in weight row order.
    """
    w = np.asarray(weights, float)
    if w.ndim != 2 or not isinstance(n_components, (int, np.integer)) or not 1 <= n_components <= w.shape[1]:
        raise ValueError('n_components must be between 1 and the fitted component count.')
    if spatial_scope not in ('pooled', 'within_subject'):
        raise ValueError("spatial_scope must be 'pooled' or 'within_subject'.")
    if ((neighborhood_mm is not None and (not np.isfinite(neighborhood_mm) or neighborhood_mm <= 0))
            or not np.isfinite(fit_radius_mm) or fit_radius_mm <= 0):
        raise ValueError('Neighborhood and fit radii must be finite and positive.')
    if not np.isfinite(top_percent) or not 0 < top_percent <= 100:
        raise ValueError('top_percent must be in (0, 100].')
    if max_peaks is not None and (not isinstance(max_peaks, int) or max_peaks < 1):
        raise ValueError('max_peaks must be a positive integer or None.')
    if not isinstance(min_neighbors, int) or min_neighbors < 1:
        raise ValueError('min_neighbors must be a positive integer.')
    table = metadata.copy().reset_index(drop=True)
    if len(table) != len(w):
        raise ValueError('Metadata and weights must have identical row counts.')
    xyz = table[['x', 'y', 'z']].to_numpy(float)
    if not np.isfinite(xyz).all() or not np.isfinite(w[:, :n_components]).all():
        raise ValueError('Selected weights and MNI coordinates must be finite.')
    table['feature_row'] = np.arange(len(table))
    table['weight_norm'] = np.linalg.norm(w[:, :n_components], axis=1)
    if spatial_scope == 'within_subject':
        # Native MEG ownership takes precedence; paired coverage is not a
        # subject-specific MEG measurement merely because it matches an electrode.
        owner = next((c for c in ('meg_subject', 'subject', 'ieeg_subject') if c in table), None)
        if owner is None or table[owner].isna().any():
            raise ValueError('Within-subject analysis requires valid participant labels.')
        labels = table[owner].astype(str)
        if labels.str.contains('average|pooled', case=False, regex=True).any():
            raise ValueError('Averaged data have no within-subject weights; use pooled scope or individual data.')
        table['scope'] = labels
    else:
        table['scope'] = 'pooled'
    results = {}
    for scope, group in table.groupby('scope', sort=False):
        group = group.reset_index(drop=True)
        coords = group[['x', 'y', 'z']].to_numpy(float)
        norms = group.weight_norm.to_numpy()
        tree = cKDTree(coords)
        maxima = []
        ranked = np.argsort(-norms, kind='stable')
        for row in ranked:
            if norms[row] <= 0:
                continue
            if neighborhood_mm is None:
                maxima.append(row)
                if len(maxima) >= int(np.ceil(len(group) * top_percent / 100)):
                    break
                continue
            neighbors = np.asarray(tree.query_ball_point(coords[row], neighborhood_mm))
            if len(neighbors) - 1 < min_neighbors:
                continue
            if np.any(norms[neighbors] > norms[row]):
                continue
            if np.any((norms[neighbors] == norms[row]) & (neighbors < row)):
                continue
            maxima.append(row)
        peaks = group.iloc[maxima].copy().reset_index(drop=True)
        peaks['peak_id'] = [f'P{i + 1}' for i in range(len(peaks))]
        selected = peaks if max_peaks is None else peaks.head(max_peaks)
        profiles, fits, other_peaks = {}, [], {}
        for peak in selected.itertuples():
            distance = np.linalg.norm(coords - np.array([peak.x, peak.y, peak.z]), axis=1)
            profile = group.copy()
            profile['distance_mm'] = distance
            profile['in_fit_radius'] = (distance > 0) & (distance <= fit_radius_mm)
            profiles[peak.peak_id] = profile
            fit = fit_decay(distance[distance <= fit_radius_mm], norms[distance <= fit_radius_mm])
            fits.append(dict(peak_id=peak.peak_id, feature_row=peak.feature_row, **fit))
            others = peaks.loc[peaks.peak_id != peak.peak_id].copy()
            others['distance_mm'] = np.linalg.norm(
                others[['x', 'y', 'z']].to_numpy(float) - [peak.x, peak.y, peak.z], axis=1)
            other_peaks[peak.peak_id] = others
        results[str(scope)] = dict(features=group, maxima=peaks, selected_maxima=selected,
                                   profiles=profiles, fits=pd.DataFrame(fits), other_maxima=other_peaks,
                                   fit_radius_mm=fit_radius_mm, n_components=n_components)
    return results


def plot_gradients(result, title='', *, include_profiles=True):
    """One brain overview and one radial scatter plot per selected maximum."""
    import matplotlib.pyplot as plt
    from nilearn import plotting

    figures = {}
    features, peaks = result['features'], result['selected_maxima']
    fig = plt.figure(figsize=(12, 4))
    brain = plotting.plot_glass_brain(None, figure=fig, display_mode='ortho',
                                     title=f'{title} · maxima of weight norm (k={result["n_components"]})')
    brain.add_markers(features[['x', 'y', 'z']].to_numpy(),
                      marker_color='gray', marker_size=5, alpha=.25)
    colors = plt.get_cmap('tab10')
    for index, peak in enumerate(peaks.itertuples()):
        brain.add_markers(np.array([[peak.x, peak.y, peak.z]]),
                          marker_color=colors(index % 10), marker='*', marker_size=140)
        # A figure-level key links the brain markers to the individual profiles.
        fig.text(.02 + .19 * (index % 5), .02 - .04 * (index // 5),
                 f'{peak.peak_id}: row {peak.feature_row}', color=colors(index % 10), fontsize=9)
    figures['brain'] = fig
    if not include_profiles:
        return figures
    for index, peak in enumerate(peaks.itertuples()):
        profile = result['profiles'][peak.peak_id]
        fit = result['fits'].set_index('peak_id').loc[peak.peak_id]
        fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
        ax.scatter(profile.distance_mm, profile.weight_norm, s=14, alpha=.4,
                   color=colors(index % 10), label='Electrodes / sources')
        ax.scatter([0], [peak.weight_norm], marker='*', s=160,
                   color=colors(index % 10), edgecolors='black', label=peak.peak_id, zorder=4)
        if np.isfinite(fit.sigma_mm):
            d = np.linspace(0, min(result['fit_radius_mm'], profile.distance_mm.max()), 200)
            ax.plot(d, gaussian_decay(d, fit.baseline, fit.amplitude, fit.sigma_mm),
                    color='black', linewidth=2,
                    label=f'Gaussian fit: σ={fit.sigma_mm:.1f} mm, R²={fit.r_squared:.2f}')
        others = result['other_maxima'][peak.peak_id]
        for j, other in enumerate(others.itertuples()):
            ax.axvline(other.distance_mm, color='red', alpha=.55, linestyle='--',
                       label='Distance to another maximum' if j == 0 else None)
            ax.text(other.distance_mm, 1, other.peak_id, transform=ax.get_xaxis_transform(),
                    color='red', fontsize=8, rotation=90, va='top')
        ax.axvspan(0, result['fit_radius_mm'], alpha=.05, color='black', label='Fit range (peak excluded)')
        ax.set(xlabel=f'Euclidean distance from {peak.peak_id} (mm)', ylabel='Component weight norm',
               title=f'{title} · {peak.peak_id} · feature row {peak.feature_row} · {fit.fit_status}',
               xlim=(-1, max(1, profile.distance_mm.max()) * 1.02),
               ylim=(0, 1.05 * profile.weight_norm.max()))
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
        figures[peak.peak_id] = fig
    return figures


def summarize_subject_gradients(scopes, peak_id, n_points=200):
    """Equal-subject mean and sample SD of same-rank Gaussian fits.

    Only fits with status 'ok' contribute. Predictions are restricted to the
    intersection of the subjects' observed positive-distance fitting ranges.
    Thus the contributing subjects are constant along the summary curve, with
    no extrapolation. Peak ranks do not establish anatomical correspondence.
    """
    fits, ranges, subjects = [], [], []
    for subject, result in scopes.items():
        if peak_id not in result['profiles']:
            continue
        fit = result['fits'].set_index('peak_id').loc[peak_id]
        if fit.fit_status != 'ok' or not np.isfinite(
                [fit.baseline, fit.amplitude, fit.sigma_mm]).all():
            continue
        profile = result['profiles'][peak_id]
        distances = profile.loc[profile.in_fit_radius, 'distance_mm']
        if distances.empty:
            continue
        subjects.append(subject)
        fits.append(fit)
        ranges.append((distances.min(), distances.max()))
    empty = dict(distance_mm=np.array([]), mean=np.array([]), std=np.array([]),
                 subjects=subjects, n_subjects=len(subjects))
    if not fits:
        return empty
    start, end = max(r[0] for r in ranges), min(r[1] for r in ranges)
    if start >= end:
        return empty
    distance = np.linspace(start, end, n_points)
    curves = np.array([gaussian_decay(distance, f.baseline, f.amplitude, f.sigma_mm)
                       for f in fits])
    return dict(distance_mm=distance, mean=curves.mean(axis=0),
                std=curves.std(axis=0, ddof=1) if len(curves) > 1 else np.full(n_points, np.nan),
                subjects=subjects, n_subjects=len(subjects))


def plot_subject_gradients(scopes, title='', *, show_other_maxima=True):
    """Overlay subjects by peak rank, with matching scatter/fit colors."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    subjects = list(scopes)
    cmap = plt.get_cmap('tab20' if len(subjects) <= 20 else 'turbo')
    colors = {subject: cmap(i if len(subjects) <= 20 else i / max(1, len(subjects) - 1))
              for i, subject in enumerate(subjects)}
    peak_ids = sorted({p for r in scopes.values() for p in r['profiles']},
                      key=lambda p: int(p[1:]))
    figures = {}
    for peak_id in peak_ids:
        fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
        ymax, xmax, n_valid, n_plotted = 0., 0., 0, 0
        has_other_maxima = False
        for subject, result in scopes.items():
            if peak_id not in result['profiles']:
                continue
            n_plotted += 1
            profile = result['profiles'][peak_id]
            fit = result['fits'].set_index('peak_id').loc[peak_id]
            color = colors[subject]
            valid = fit.fit_status == 'ok' and np.isfinite(
                [fit.baseline, fit.amplitude, fit.sigma_mm]).all()
            label = str(subject) if valid else f'{subject} ({fit.fit_status})'
            ax.scatter(profile.distance_mm, profile.weight_norm, color=color,
                       s=12, alpha=.3, label=label)
            ymax = max(ymax, profile.weight_norm.max())
            xmax = max(xmax, profile.distance_mm.max())
            distances = profile.loc[profile.in_fit_radius, 'distance_mm']
            if valid and len(distances):
                d = np.linspace(distances.min(), distances.max(), 200)
                ax.plot(d, gaussian_decay(d, fit.baseline, fit.amplitude, fit.sigma_mm),
                        color=color, linewidth=1.6, alpha=.85)
                n_valid += 1
            if show_other_maxima:
                for other in result['other_maxima'][peak_id].itertuples():
                    ax.axvline(other.distance_mm, color='red', alpha=.12,
                               linewidth=.7, linestyle='--', zorder=0)
                    has_other_maxima = True
        summary = summarize_subject_gradients(scopes, peak_id)
        if len(summary['distance_mm']):
            ax.plot(summary['distance_mm'], summary['mean'], color='black', linewidth=2.8,
                    label=f'Mean fitted curve (n={summary["n_subjects"]})', zorder=5)
            if summary['n_subjects'] >= 2:
                ax.fill_between(summary['distance_mm'], summary['mean'] - summary['std'],
                                summary['mean'] + summary['std'], color='black', alpha=.15,
                                label='Between-subject ±1 SD', zorder=1)
        else:
            ax.text(.98, .97, 'No valid fits with a shared distance range',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9)
        ax.set(xlabel=f'Euclidean distance from each subject’s {peak_id} (mm)',
               ylabel='Component weight norm',
               title=f'{title} · {peak_id} (within-subject rank) · {n_valid}/{n_plotted} valid fits',
               xlim=(-1, max(1, xmax) * 1.02), ylim=(0, 1.05 * ymax))
        ax.grid(alpha=.15)
        handles, labels = ax.get_legend_handles_labels()
        if has_other_maxima:
            handles.append(Line2D([], [], color='red', linestyle='--', linewidth=.7))
            labels.append('Other maxima: within-subject distances')
        ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.01, 1),
                  fontsize=8, frameon=False)
        figures[peak_id] = fig
    return figures
