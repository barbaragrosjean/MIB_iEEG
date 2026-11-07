"""Electrode-to-MEG comparisons of absolute ERP sequences in MNI shells."""
import re
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata


def _condition_average(array, block_scaling):
    erp = np.asarray(array, dtype=float).mean(axis=0)
    total_variance = np.var(erp, axis=1, ddof=1).sum()
    scale = 1. / np.sqrt(total_variance) if block_scaling and total_variance > 0 else 1.
    return erp * scale, scale


def _bilateral_region(label):
    if pd.isna(label):
        return 'Unassigned'
    value = re.sub(r'^(?:left|right|lh|rh|L|R)[_ .-]+', '', str(label), flags=re.I)
    return re.sub(r'[_ .-]+(?:left|right|lh|rh|L|R)$', '', value, flags=re.I)


def absolute_spearman(reference, targets):
    """One reference and a row per shell; ties receive average ranks."""
    x = rankdata(reference)
    y = rankdata(targets, axis=1)
    x = x - x.mean()
    y = y - y.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(x) * np.linalg.norm(y, axis=1)
    values = np.divide(y @ x, denominator, out=np.full(len(y), np.nan), where=denominator > 0)
    return np.clip(np.abs(values), 0., 1.)


METRIC_LABELS = {
    'correlation': 'Absolute Spearman correlation of absolute ERPs',
    'rmsd': 'RMSD of absolute ERPs',
    'cosine': 'Cosine distance of absolute ERPs',
}


def sequence_comparison(reference, targets, metric='correlation'):
    """Compare |reference| with each |target|, preserving sample alignment.

    RMSD uses configured ERP units. Cosine is undefined for zero vectors;
    correlation is undefined for constant magnitudes. Nonfinite rows are omitted.
    """
    if metric not in METRIC_LABELS:
        raise ValueError(f'Unknown sequence metric: {metric}; choose {tuple(METRIC_LABELS)}.')
    x, y = np.abs(np.asarray(reference, float)), np.abs(np.asarray(targets, float))
    if x.ndim != 1 or y.ndim != 2 or y.shape[1] != len(x) or not len(x):
        raise ValueError('Expected one nonempty sequence and matching target rows.')
    result = np.full(len(y), np.nan)
    valid = np.isfinite(y).all(axis=1) & np.isfinite(x).all()
    if not valid.any():
        return result
    targets = y[valid]
    if metric == 'correlation':
        result[valid] = absolute_spearman(x, targets)
    elif metric == 'rmsd':
        result[valid] = np.sqrt(np.mean((targets-x)**2, axis=1))
    else:
        denominator = np.linalg.norm(x)*np.linalg.norm(targets, axis=1)
        similarity = np.divide(targets @ x, denominator,
                               out=np.full(len(targets), np.nan), where=denominator > 0)
        result[valid] = np.clip(1-similarity, 0., 1.)
    return result


def analyze_channel_space(data, *, shell_edges_mm, block_scaling=True,
                          time_window=None, roi_column='region', pool_hemispheres=True,
                          sequence_metric='correlation'):
    """Use all iEEG electrode × MEG participant comparisons, without random pairing.

    Whole-participant blocks have total temporal sample variance one when enabled,
    after condition averaging. Positive scalar scaling affects displayed ERPs but
    does not change Spearman correlations. Shells are [left, right), except that
    the last shell includes its right edge. ROI means give each iEEG participant
    equal weight after averaging available electrodes within that participant/ROI.
    """
    if sequence_metric not in METRIC_LABELS:
        raise ValueError(f'Unknown sequence metric: {sequence_metric}')
    edges = np.asarray(shell_edges_mm, dtype=float)
    if len(edges) < 2 or edges[0] < 0 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError('Shell edges must be finite, nonnegative, and strictly increasing.')
    times = np.asarray(data['times'])
    time_mask = np.ones(len(times), dtype=bool) if time_window is None else ((times >= time_window[0]) & (times <= time_window[1]))
    if time_mask.sum() < 3:
        raise ValueError('Select at least three time samples for ERP comparison.')
    metadata = data['electrode_metadata'].copy().reset_index(drop=True)
    metadata['subject'] = metadata.subject.astype(str)
    metadata['electrode_row'] = np.arange(len(metadata))
    metadata['region_analysis'] = (metadata[roi_column].map(_bilateral_region) if pool_hemispheres
                                   else metadata[roi_column].fillna('Unassigned').astype(str))
    if 'channel' not in metadata:
        metadata['channel'] = metadata.channel_index.map(lambda i: f'ch{int(i)+1}')
    ieeg_erp = np.empty((len(metadata), len(times)))
    scale_rows = []
    for subject, group in metadata.groupby('subject', sort=False):
        rows = group.index.to_numpy()
        ieeg_erp[rows], scale = _condition_average(data['ieeg'][:, rows, :], block_scaling)
        scale_rows.append(dict(modality='iEEG', subject=subject, block_scale=scale))
    n_shells = len(edges)-1
    centers = (edges[:-1]+edges[1:])/2
    shape = (len(metadata), n_shells, len(times))
    erp_sum = np.zeros(shape)
    erp_sum_squares = np.zeros(shape)
    erp_counts = np.zeros(shape[:2], dtype=int)
    rows_out = []
    coordinates = metadata[['x','y','z']].to_numpy(float)
    for subject, arrays, positions in zip(data['meg_subjects'], data['meg'], data['meg_positions']):
        meg_erp, scale = _condition_average(arrays, block_scaling)
        scale_rows.append(dict(modality='MEG', subject=str(subject), block_scale=scale))
        positions = np.asarray(positions, float)
        tree = cKDTree(positions)
        for electrode, xyz in enumerate(coordinates):
            nearby = np.asarray(tree.query_ball_point(xyz, np.nextafter(edges[-1], np.inf)), dtype=int)
            distances = np.linalg.norm(positions[nearby] - xyz, axis=1)
            shell_index = np.searchsorted(edges, distances, side='right')-1
            shell_index[distances == edges[-1]] = n_shells-1
            correlations = np.full(n_shells, np.nan)
            n_sources = np.zeros(n_shells, dtype=int)
            means, occupied = [], []
            for shell in range(n_shells):
                source_rows = nearby[shell_index == shell]
                n_sources[shell] = len(source_rows)
                if len(source_rows):
                    trace = meg_erp[source_rows].mean(axis=0)
                    erp_sum[electrode, shell] += trace
                    erp_sum_squares[electrode, shell] += trace**2
                    erp_counts[electrode, shell] += 1
                    means.append(trace[time_mask]); occupied.append(shell)
            if occupied:
                correlations[occupied] = sequence_comparison(ieeg_erp[electrode, time_mask], np.asarray(means), sequence_metric)
            rows_out.extend(dict(electrode_row=electrode, ieeg_subject=metadata.subject.iloc[electrode],
                meg_subject=str(subject), region=metadata.region_analysis.iloc[electrode],
                shell=shell, distance_mm=centers[shell], n_sources=int(n_sources[shell]),
                metric_value=correlations[shell]) for shell in range(n_shells))
        print(f'MEG {subject}: all electrode/shell comparisons complete', flush=True)
    correlations = pd.DataFrame(rows_out)
    keys = ['electrode_row','ieeg_subject','region','shell','distance_mm']
    electrode_summary = correlations.groupby(keys, sort=False).metric_value.agg(
        mean_value='mean', sd='std', sem='sem', n_meg='count').reset_index()
    support = correlations.assign(has_sources=correlations.n_sources.gt(0)).groupby(keys, sort=False).agg(
        n_meg_with_sources=('has_sources','sum'), mean_n_sources=('n_sources','mean')).reset_index()
    electrode_summary = electrode_summary.merge(support, on=keys)
    # Every MEG participant contributes equally within an electrode, then each
    # electrode equally within its iEEG participant/ROI, then participants equally.
    roi_subjects = electrode_summary.groupby(['ieeg_subject','region','shell','distance_mm'], sort=False).mean_value.agg(
        mean_value='mean', n_electrodes='count').reset_index()
    roi_summary = roi_subjects.groupby(['region','shell','distance_mm'], sort=False).mean_value.agg(
        mean_value='mean', sd='std', sem='sem', n_ieeg='count').reset_index()
    count = erp_counts[..., None]
    mean = np.divide(erp_sum, count, out=np.full(shape, np.nan), where=count > 0)
    ss = np.maximum(erp_sum_squares - np.divide(erp_sum**2, count, out=np.zeros(shape), where=count > 0), 0.)
    sem = np.sqrt(np.divide(ss, count*(count-1), out=np.full(shape, np.nan), where=count > 1))
    return dict(sequence_metric=sequence_metric, metadata=metadata, correlations=correlations, electrode_summary=electrode_summary,
        roi_subjects=roi_subjects, roi_summary=roi_summary, scaling=pd.DataFrame(scale_rows),
        shell_table=pd.DataFrame(dict(shell=np.arange(n_shells), left_mm=edges[:-1], right_mm=edges[1:], distance_mm=centers)),
        arrays=dict(times=times, correlation_time_mask=time_mask, ieeg_erp=ieeg_erp.astype('float32'),
                    meg_erp_mean=mean.astype('float32'), meg_erp_sem=sem.astype('float32'), n_meg_erp=erp_counts))


def roi_shell_source_maps(results, source_positions):
    """Project participant-balanced ROI shell means onto pooled MEG coordinates.

    Identical source coordinates are displayed once. For overlapping electrode
    shells, average finite ROI shell values within each iEEG participant first,
    then across participants. These are projected shell statistics, not source-
    specific sequence comparisons. No spatial interpolation or hemisphere folding occurs.
    """
    positions = np.unique(np.asarray(source_positions, float), axis=0)
    tree = cKDTree(positions)
    shells = results['shell_table'].sort_values('shell')
    edges = np.r_[shells.left_mm.to_numpy(), shells.right_mm.iloc[-1]]
    n_shells = len(shells)
    frames = []
    for region, electrodes in results['metadata'].groupby('region_analysis', sort=False):
        values = (results['roi_summary'].loc[lambda f: f.region.eq(region)]
                  .set_index('shell').mean_value.reindex(shells.shell).to_numpy())
        total = np.zeros(len(positions))
        n_subjects = np.zeros(len(positions), dtype=int)
        n_electrodes = np.zeros(len(positions), dtype=int)
        for _, group in electrodes.groupby('subject', sort=False):
            subtotal = np.zeros(len(positions))
            count = np.zeros(len(positions), dtype=int)
            for xyz in group[['x', 'y', 'z']].to_numpy(float):
                nearby = np.asarray(tree.query_ball_point(xyz, np.nextafter(edges[-1], np.inf)), dtype=int)
                distances = np.linalg.norm(positions[nearby] - xyz, axis=1)
                index = np.searchsorted(edges, distances, side='right') - 1
                index[distances == edges[-1]] = n_shells - 1
                valid = (index >= 0) & (index < n_shells)
                nearby, index = nearby[valid], index[valid]
                finite = np.isfinite(values[index])
                nearby, index = nearby[finite], index[finite]
                subtotal[nearby] += values[index]
                count[nearby] += 1
            available = count > 0
            total[available] += subtotal[available] / count[available]
            n_subjects[available] += 1
            n_electrodes += count
        mean = np.divide(total, n_subjects, out=np.full(len(positions), np.nan), where=n_subjects > 0)
        frame = pd.DataFrame(positions, columns=['x', 'y', 'z'])
        frame['region'] = region
        frame['mean_value'] = mean
        frame['n_ieeg_spatial_support'] = n_subjects
        frame['n_electrode_shells'] = n_electrodes
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
