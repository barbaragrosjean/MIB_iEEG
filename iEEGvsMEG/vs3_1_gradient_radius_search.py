"""Exploratory radius sensitivity for native model-weight magnitude gradients."""
import numpy as np
import pandas as pd

from vs3_1_gradients import analyze_gradients, fit_gradient_models


def radius_search(weights, metadata, *, neighborhood_radii, fit_radii,
                  metric='semivariance', n_components=2, spatial_scope='pooled',
                  fold_hemispheres=True, target_hemisphere='left', max_peaks=None,
                  min_points=6, min_peaks=3, min_valid_fraction=.8, progress=None):
    """Evaluate radius pairs with equal-peak / equal-subject aggregation.

    Semivariance is mean (Ai-Aj)^2/2 over all distinct pairs inside each
    peak-centered ball, normalized by population variance of A in its scope.
    Its exact value is sample variance inside the ball / scope variance.
    This is cumulative local dispersion, NOT a lag-binned semivariogram.
    Zero-distance observations are excluded as in the gradient fits.
    """
    neighborhoods = np.asarray(neighborhood_radii, float)
    radii = np.asarray(fit_radii, float)
    for values in (neighborhoods, radii):
        if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
                or np.any(values <= 0) or np.any(np.diff(values) <= 0)):
            raise ValueError('Radius grids must be finite, positive, and strictly increasing.')
    if metric not in ('semivariance', 'quadratic'):
        raise ValueError('metric must be semivariance or quadratic.')
    if not isinstance(min_points, int) or min_points < 6 or not isinstance(min_peaks, int) or min_peaks < 1:
        raise ValueError('Require integer min_points >= 6 and min_peaks >= 1.')
    if not 0 < min_valid_fraction <= 1:
        raise ValueError('min_valid_fraction must be in (0, 1].')
    rows, details = [], []
    for neighborhood in neighborhoods:
        # Select once per neighborhood. Quadratic fitting at the largest radius
        # also uses the standard geometry/ownership validation and peak handling.
        scopes = analyze_gradients(weights, metadata, n_components=n_components,
            spatial_scope=spatial_scope, neighborhood_mm=neighborhood,
            fit_radius_mm=float(radii[-1]), max_peaks=max_peaks,
            fold_hemispheres=fold_hemispheres, target_hemisphere=target_hemisphere,
            fit_model='quadratic')
        selected_count = sum(len(r['selected_maxima']) for r in scopes.values())
        for radius in radii:
            records = []
            for scope, result in scopes.items():
                variance = np.var(result['features'].weight_norm.to_numpy(float), ddof=0)
                for peak in result['selected_maxima'].itertuples():
                    profile = result['profiles'][peak.peak_id]
                    local = profile.loc[(profile.distance_mm > 0) & (profile.distance_mm <= radius)]
                    y = local.weight_norm.to_numpy(float)
                    value, status = np.nan, 'insufficient_data'
                    if len(y) >= (2 if metric == 'semivariance' else 3):
                        if metric == 'semivariance':
                            if variance > 0:
                                value, status = np.var(y, ddof=1)/variance, 'ok'
                            else:
                                status = 'constant_scope_magnitude'
                        else:
                            fit = fit_gradient_models(local.distance_mm, y, ['quadratic'], min_points=3, min_distances=2).iloc[0]
                            status = fit.fit_status
                            if status == 'ok':
                                value = fit.adjusted_r_squared
                    record = dict(neighborhood_mm=neighborhood, fit_radius_mm=radius,
                        scope=scope, peak_id=peak.peak_id, feature_row=peak.feature_row,
                        n_points=len(y), value=value, status=status)
                    records.append(record)
                    details.append(record)
            valid = [r for r in records if r['status'] == 'ok' and np.isfinite(r['value'])]
            fraction = len(valid)/selected_count if selected_count else 0.
            supported = [r for r in valid if r['n_points'] >= min_points]
            supported_fraction = len(supported)/selected_count if selected_count else 0.
            eligible = len(supported) >= min_peaks and supported_fraction >= min_valid_fraction
            if valid:
                frame = pd.DataFrame(valid)
                # Equal subject weight after within-subject peak means.
                values = (frame.groupby('scope').value.mean().to_numpy()
                          if spatial_scope == 'within_subject' else frame.value.to_numpy())
                value = float(values.mean())
                units = len(values)
            else:
                value, units = np.nan, 0
            rows.append(dict(neighborhood_mm=neighborhood, fit_radius_mm=radius,
                metric=metric, value=value, eligible=eligible, n_selected_peaks=selected_count,
                n_valid_peaks=len(valid), valid_fraction=fraction, n_units=units,
                n_supported_peaks=len(supported), supported_fraction=supported_fraction,
                mean_points=float(np.mean([r['n_points'] for r in valid])) if valid else np.nan))
        if progress:
            progress(f'{metric}: neighborhood {neighborhood:g} mm completed ({selected_count} selected peaks)')
    return {'grid': pd.DataFrame(rows), 'details': pd.DataFrame(details)}


def select_radius_candidates(grid, *, plateau_fraction=.9, plateau_tolerance=.1,
                             tail_bins=3, min_rise_fraction=.2):
    """Return supported heuristic candidates; never force a missing plateau.

    A stable tail has at least tail_bins eligible consecutive final grid values,
    each within tolerance of their mean. The plateau starts at the first tested
    radius whose entire eligible suffix lies between fraction*tail and
    (1+tolerance)*tail. Require a positive rise of min_rise_fraction*tail.
    Rank neighborhoods by relative rise, then smaller radii for exact ties.
    Quadratic candidates rank by mean adjusted R², then smaller radii.
    """
    if not (0 < plateau_fraction <= 1 and 0 < plateau_tolerance < 1
            and 0 <= min_rise_fraction < 1 and isinstance(tail_bins, int) and tail_bins >= 2):
        raise ValueError('Invalid plateau settings.')
    columns = ['neighborhood_mm', 'fit_radius_mm', 'score', 'selection_rule', 'boundary', 'selected']
    if grid.empty:
        return pd.DataFrame(columns=columns)
    metric = grid.metric.unique()
    if len(metric) != 1:
        raise ValueError('Select one metric/modality at a time.')
    candidates = []
    if metric[0] == 'quadratic':
        for row in grid.loc[grid.eligible & np.isfinite(grid.value)].itertuples():
            candidates.append(dict(neighborhood_mm=row.neighborhood_mm,
                fit_radius_mm=row.fit_radius_mm, score=row.value,
                selection_rule='maximum mean adjusted R²'))
    else:
        for neighborhood, frame in grid.groupby('neighborhood_mm', sort=True):
            frame = frame.sort_values('fit_radius_mm')
            values = frame.value.to_numpy(float)
            eligible = frame.eligible.to_numpy(bool) & np.isfinite(values)
            if len(frame) <= tail_bins or not eligible[-tail_bins:].all():
                continue
            tail = values[-tail_bins:].mean()
            if tail <= 0 or np.any(np.abs(values[-tail_bins:]-tail) > plateau_tolerance*tail):
                continue
            if not eligible[0] or (tail-values[0])/tail < min_rise_fraction:
                continue
            for i in range(1, len(frame)-tail_bins+1):
                if (eligible[i:].all() and np.all(values[i:] >= plateau_fraction*tail)
                        and np.all(values[i:] <= (1+plateau_tolerance)*tail)):
                    candidates.append(dict(neighborhood_mm=neighborhood,
                        fit_radius_mm=frame.iloc[i].fit_radius_mm,
                        score=(tail-values[0])/tail,
                        selection_rule='stable plateau; rank by relative rise'))
                    break
    if not candidates:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(candidates).sort_values(
        ['score', 'neighborhood_mm', 'fit_radius_mm'], ascending=[False, True, True]).reset_index(drop=True)
    result['boundary'] = (result.neighborhood_mm.isin([grid.neighborhood_mm.min(), grid.neighborhood_mm.max()])
                          | result.fit_radius_mm.isin([grid.fit_radius_mm.min(), grid.fit_radius_mm.max()]))
    result['selected'] = result.index == 0
    return result


def plot_radius_search(results, *, candidates=None, title=''):
    """One heatmap axis per modality, with shared metric color limits."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    def edges(x):
        x = np.asarray(x, float)
        if len(x) == 1:
            return np.array([x[0]-.5, x[0]+.5])
        mid = (x[:-1]+x[1:])/2
        return np.r_[x[0]-(x[1]-x[0])/2, mid, x[-1]+(x[-1]-x[-2])/2]

    fig, axes = plt.subplots(1, len(results), figsize=(7*len(results), 6),
                             squeeze=False, layout='constrained', sharex=True, sharey=True)
    metrics = {r['grid'].metric.iloc[0] for r in results.values()}
    if len(metrics) != 1:
        raise ValueError('Use one metric for all panels.')
    metric = next(iter(metrics))
    values = np.concatenate([r['grid']['value'].to_numpy() for r in results.values()])
    values = values[np.isfinite(values)]
    lo, hi = (float(values.min()), float(values.max())) if len(values) else (0., 1.)
    if lo == hi:
        hi = lo+max(1e-6, abs(lo)*.01)
    norm = Normalize(lo, hi)
    for ax, (modality, result) in zip(axes.flat, results.items()):
        grid = result['grid'].copy()
        matrix = grid.pivot(index='neighborhood_mm', columns='fit_radius_mm', values='value').sort_index().sort_index(axis=1)
        ax.set_facecolor('#eeeeee')
        mesh = ax.pcolormesh(edges(matrix.columns), edges(matrix.index),
                             np.ma.masked_invalid(matrix.to_numpy()), cmap='viridis', norm=norm)
        low_support = grid.loc[~grid.eligible & np.isfinite(grid.value)]
        if len(low_support):
            ax.scatter(low_support.fit_radius_mm, low_support.neighborhood_mm, marker='x',
                       s=30, color='black', linewidths=.8, label='Limited support')
        chosen = candidates.get(modality) if candidates else None
        if chosen is not None and not chosen.empty:
            if metric == 'semivariance':
                ax.plot(chosen.fit_radius_mm, chosen.neighborhood_mm, 'wo', markersize=4,
                        label='Plateau candidates')
            best = chosen.iloc[0]
            ax.scatter(best.fit_radius_mm, best.neighborhood_mm, marker='*', s=200,
                       facecolor='white', edgecolor='black', label='Suggested pair')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc='upper right')
        ax.set(title=modality, xlabel='Fitting / inclusion radius (mm)',
               ylabel='Peak-selection neighborhood radius (mm)',
               xticks=matrix.columns, yticks=matrix.index)
        if not np.isfinite(matrix.to_numpy()).any():
            ax.text(.5, .5, 'No computable estimates', transform=ax.transAxes, ha='center')
    label = ('Normalized local magnitude semivariance' if metric == 'semivariance'
             else 'Mean adjusted R² of quadratic fits')
    fig.colorbar(mesh, ax=list(axes.flat), label=label, shrink=.85)
    fig.suptitle(title or f'{label} · × = limited support; gray = undefined')
    return fig
