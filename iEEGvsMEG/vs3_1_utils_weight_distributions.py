"""Threshold-free, participant-balanced comparisons of fitted weight magnitudes."""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def compare_weight_distributions(aligned, *, components, grid_size=256):
    """Exact weighted W1; ECDF grids are used only for displaying the distributions.

    Each fixed iEEG coverage participant has equal total mass in both modalities.
    Within a participant, each coverage slot has equal mass. Repeated native MEG
    sources remain slots. RMS normalization divides magnitudes by their weighted
    root mean square, without centering or thresholding. Associations are repeated
    assignments of the same sample, not independent new participant cohorts.
    """
    owners = aligned['anchor'].subject.astype(str)
    counts = owners.value_counts()
    mass = 1. / (len(counts) * owners.map(counts).to_numpy(float))
    traces, distances, scales = [], [], []
    for component in components:
        raw = {m: np.abs(np.asarray(w[:, :, component-1], float)) for m, w in aligned['weights'].items()}
        rms = {m: np.sqrt(np.sum(values**2 * mass, axis=1)) for m, values in raw.items()}
        for modality in raw:
            scales.extend(dict(component=component, modality=modality, iteration=i, rms=value)
                          for i, value in enumerate(rms[modality]))
        for mode in ['raw', 'rms_normalized']:
            values = raw if mode == 'raw' else {
                m: np.divide(v, rms[m][:, None], out=np.full_like(v, np.nan), where=rms[m][:,None] > 0)
                for m, v in raw.items()}
            finite_maxima = [float(np.nanmax(v)) for v in values.values() if np.isfinite(v).any()]
            grid = np.linspace(0., max(finite_maxima, default=0.) or 1., grid_size)
            for iteration in range(len(raw['iEEG'])):
                x, y = values['iEEG'][iteration], values['MEG'][iteration]
                valid = np.isfinite(x).all() and np.isfinite(y).all()
                distance = wasserstein_distance(x, y, mass, mass) if valid else np.nan
                distances.append(dict(component=component, iteration=iteration, mode=mode,
                                      wasserstein=distance, status='ok' if valid else 'zero RMS'))
                for modality, stack in values.items():
                    vector = stack[iteration]
                    if np.isfinite(vector).all():
                        order = np.argsort(vector, kind='stable')
                        cumulative = np.r_[0., np.cumsum(mass[order])]
                        cdf = cumulative[np.searchsorted(vector[order], grid, side='right')]
                        cdf = np.clip(cdf, 0., 1.)
                    else:
                        cdf = np.full(len(grid), np.nan)
                    traces.extend(dict(component=component, iteration=iteration, mode=mode,
                        modality=modality, magnitude=value, cdf=fraction) for value, fraction in zip(grid, cdf))
    curves = pd.DataFrame(traces)
    summary = curves.groupby(['component','mode','modality','magnitude'], sort=False).cdf.agg(
        mean='mean', sd='std', sem='sem', n_associations='count').reset_index()
    return dict(distances=pd.DataFrame(distances), ecdf_iterations=curves, ecdf_summary=summary,
                rms=pd.DataFrame(scales), provenance=aligned['provenance'], source_dir=aligned['source_dir'])
