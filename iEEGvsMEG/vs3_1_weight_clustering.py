"""Offline clustering of absolute model weights; notebook-facing export/load/plots.

Run: python vs3_1_weight_clustering.py --methods kmeans gmm birch minibatch" --project MINDLAB2021_MEG-TempSeqAges
"""
from pathlib import Path
import argparse
import hashlib
import json
import warnings

import numpy as np
import pandas as pd

METHODS = ('kmeans', 'minibatch', 'gmm', 'birch')


def export_inputs(weights, metadata, directory):
    """Export fitted weights and corresponding shared-region metadata, without fitting."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for modality, values in weights.items():
        frame = metadata[modality].reset_index(drop=True)
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] < 3 or len(frame) != len(values):
            raise ValueError('Fit at least three components and provide matching metadata.')
        np.save(directory / f'{modality}_weights.npy', values[:, :3])
        frame.to_csv(directory / f'{modality}_metadata.csv', index=False)
    (directory / 'inputs.json').write_text(json.dumps(list(weights)))


def input_signature(directory):
    directory = Path(directory)
    digest = hashlib.sha256()
    for filename in ['inputs.json'] + [f'{m}_{suffix}' for m in json.loads((directory / 'inputs.json').read_text())
                                      for suffix in ('weights.npy', 'metadata.csv')]:
        with (directory / filename).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def cluster_weights(weights, metadata, *, method='kmeans', n_clusters=5,
                    use_thresholded=False, seed=2026, birch_threshold=0.1,
                    k_min=2, k_max=10, silhouette_sample_size=2000):
    """Cluster |components 1–3| only, with no spatial input or axis scaling.

    BIRCH's radius is a fraction of the RMS input-vector norm, making its
    threshold interpretable across weight units without changing axis geometry.
    """
    if n_clusters == 'auto':
        return select_cluster_count(weights, metadata, method=method,
            use_thresholded=use_thresholded, seed=seed, birch_threshold=birch_threshold,
            k_min=k_min, k_max=k_max, silhouette_sample_size=silhouette_sample_size)
    from sklearn.cluster import KMeans, MiniBatchKMeans, Birch
    from sklearn.mixture import GaussianMixture
    from vs3_1_modality_specificity import spatial_metadata, threshold_weights, summarize_kmeans_regions_hemispheres
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}. Choose from {METHODS}.')
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, (int, np.integer)) or n_clusters < 2:
        raise ValueError('n_clusters must be an integer >= 2.')
    if not np.isfinite(birch_threshold) or birch_threshold <= 0:
        raise ValueError('birch_threshold must be positive.')
    members = spatial_metadata(metadata).reset_index(drop=True)
    if 'region_shared' not in members:
        raise ValueError('Export coordinate-based region_shared labels first.')
    members['region_shared'] = members.region_shared.fillna('Unassigned')
    w = np.asarray(weights, float)
    if w.ndim != 2 or w.shape[1] < 3 or len(w) != len(members) or not np.isfinite(w).all():
        raise ValueError('Finite weights with at least three components and matching metadata are required.')
    if not np.isfinite(members[['x','y','z']].to_numpy(float)).all():
        raise ValueError('Coordinates must be finite MNI millimeters.')
    thresholded, thresholds = threshold_weights(w[:, :3])
    x = thresholded if use_thresholded else np.abs(w[:, :3])
    active = np.any(x != 0, axis=1)
    if len(np.unique(x[active], axis=0)) < n_clusters:
        raise ValueError('Too few distinct nonzero profiles; reduce n_clusters or disable thresholding.')
    # A single scalar rescales numerical units only, preserving Euclidean geometry.
    scale = float(np.sqrt(np.mean(np.sum(x[active] ** 2, axis=1))))
    z = x[active] / scale
    if method == 'kmeans':
        estimator = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    elif method == 'minibatch':
        estimator = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=20, batch_size=1024)
    elif method == 'gmm':
        estimator = GaussianMixture(n_components=n_clusters, random_state=seed, n_init=5,
                                    covariance_type='full', reg_covar=1e-6)
    else:
        estimator = Birch(n_clusters=n_clusters, threshold=birch_threshold)
    labels = estimator.fit_predict(z)
    if method == 'gmm' and not estimator.converged_:
        warnings.warn('Gaussian mixture did not converge; inspect results or change parameters.')
    members['cluster_id'] = -1
    members.loc[active, 'cluster_id'] = labels
    members['feature_row'] = np.arange(len(w))
    for c in range(3):
        members[f'input_weight_{c+1}'] = x[:, c]
    summary = summarize_kmeans_regions_hemispheres({'memberships': members}, members,
                                                  fold_hemispheres=False)
    regions = (members[members.cluster_id >= 0].groupby(['cluster_id', 'region_shared'])
               .size().rename('n_sources').reset_index())
    regions['percent'] = 100 * regions.n_sources / regions.groupby('cluster_id').n_sources.transform('sum')
    summary['region_contributions'] = regions
    summary['thresholds'] = thresholds
    return summary


def select_cluster_count(weights, metadata, *, method='kmeans', use_thresholded=False,
                         seed=2026, birch_threshold=0.1, k_min=2, k_max=10,
                         silhouette_sample_size=2000):
    """Fit all candidates on all features; score a fixed seeded sample for memory bounds.

    Maximize Euclidean silhouette; exact ties favor smaller k. This assesses
    separation, not evidence against a single population or biological validity.
    """
    from sklearn.metrics import silhouette_score
    from vs3_1_modality_specificity import threshold_weights
    if not (isinstance(k_min, int) and isinstance(k_max, int) and 2 <= k_min <= k_max):
        raise ValueError('Require 2 <= k_min <= k_max.')
    if not isinstance(silhouette_sample_size, int) or silhouette_sample_size < 3:
        raise ValueError('silhouette_sample_size must be at least 3.')
    w = np.asarray(weights, float)
    if w.ndim != 2 or w.shape[1] < 3 or not np.isfinite(w).all():
        raise ValueError('Finite weights with at least three components are required.')
    x = threshold_weights(w[:, :3])[0] if use_thresholded else np.abs(w[:, :3])
    active = np.any(x != 0, axis=1)
    x = x[active]
    limit = min(len(x)-1, len(np.unique(x, axis=0)))
    sample = np.sort(np.random.default_rng(seed).choice(len(x), min(len(x), silhouette_sample_size), replace=False))
    rows, best, best_score = [], None, -np.inf
    for k in range(k_min, k_max+1):
        row = dict(k=k, actual_k=0, silhouette=np.nan, selected=False,
                   n_scored=len(sample), status='too few distinct/nonzero profiles')
        if k <= limit:
            result = cluster_weights(weights, metadata, method=method, n_clusters=k,
                use_thresholded=use_thresholded, seed=seed, birch_threshold=birch_threshold)
            labels = result['memberships'].loc[active, 'cluster_id'].to_numpy()
            row['actual_k'] = len(np.unique(labels))
            sampled_labels = labels[sample]
            if row['actual_k'] != k:
                row['status'] = 'method returned fewer clusters than requested'
            elif not 1 < len(np.unique(sampled_labels)) < len(sample):
                row['status'] = 'silhouette undefined on scoring sample'
            else:
                score = float(silhouette_score(x[sample], sampled_labels))
                row.update(silhouette=score, status='ok')
                if score > best_score:
                    best, best_score = result, score
                    best_k = k
        rows.append(row)
    if best is None:
        raise ValueError('No valid cluster-count candidate. Adjust range, sample size, thresholding or BIRCH threshold.')
    for row in rows:
        row['selected'] = row['k'] == best_k
    best['cluster_selection'] = pd.DataFrame(rows)
    if best_k in (k_min, k_max):
        warnings.warn(f'Selected k={best_k} lies at the search boundary; inspect the score curve.')
    print(f'{method}: selected k={best_k}, silhouette={best_score:.3f}', flush=True)
    return best


def plot_cluster_selection(result, title=''):
    """Plot saved candidate scores without fitting."""
    import matplotlib.pyplot as plt
    table = result['cluster_selection']
    fig, ax = plt.subplots(figsize=(7,4), layout='constrained')
    ax.plot(table.k, table.silhouette, 'o-')
    selected = table[table.selected]
    ax.scatter(selected.k, selected.silhouette, marker='*', s=180, label='Selected k')
    ax.set(xlabel='Number of clusters', ylabel='Mean silhouette', title=title, xticks=table.k)
    ax.legend()
    return fig


def run_clustering(directory, methods=METHODS, **options):
    """Persist results under parameter-specific folders; never refit the group model."""
    directory = Path(directory)
    signature = input_signature(directory)
    modalities = json.loads((directory / 'inputs.json').read_text())
    for method in methods:
        destination = result_directory(directory, method, options.get('n_clusters', 5),
                                       options.get('use_thresholded', False))
        destination.mkdir(parents=True, exist_ok=True)
        manifest = destination / 'configuration.json'
        manifest.unlink(missing_ok=True)  # Incomplete reruns must not appear loadable.
        for modality in modalities:
            result = cluster_weights(np.load(directory / f'{modality}_weights.npy'),
                                     pd.read_csv(directory / f'{modality}_metadata.csv'), method=method, **options)
            folder = destination / modality
            folder.mkdir(exist_ok=True)
            for name, table in result.items():
                table.to_csv(folder / f'{name}.csv', index=False)
            print(f'{method} · {modality}: {result["memberships"].query("cluster_id >= 0").cluster_id.nunique()} clusters', flush=True)
        manifest.write_text(json.dumps(dict(method=method, input_signature=signature,
                                            absolute=True, modalities=modalities, **options), indent=2))


def result_directory(directory, method, n_clusters=5, use_thresholded=False):
    return Path(directory) / f'{method}_k{n_clusters}_{"thresholded" if use_thresholded else "unthresholded"}'


def load_results(directory, method='kmeans', n_clusters=5, use_thresholded=False):
    folder = result_directory(directory, method, n_clusters, use_thresholded)
    config = json.loads((folder / 'configuration.json').read_text())
    if config['input_signature'] != input_signature(directory):
        raise ValueError('Clustering inputs changed. Rerun vs3_1_weight_clustering.py.')
    return {m: {name: pd.read_csv(folder / m / f'{name}.csv') for name in
                (('memberships', 'hemisphere_centroids', 'region_contributions', 'cluster_selection')
                 if n_clusters == 'auto' else ('memberships', 'hemisphere_centroids', 'region_contributions'))}
            for m in config['modalities']}


def plot_results(result, title=''):
    """Exactly three figures: weight-space cloud, bilateral brain centroids, regions."""
    import matplotlib.pyplot as plt
    from vs3_1_modality_specificity import plot_kmeans_hemisphere_centroids, ordered_regions
    members = result['memberships']
    clusters = sorted(members.loc[members.cluster_id >= 0, 'cluster_id'].unique())
    colors = {c: plt.get_cmap('turbo')(i / max(len(clusters)-1, 1)) for i, c in enumerate(clusters)}
    fig = plt.figure(figsize=(10, 8), layout='constrained')
    ax = fig.add_subplot(projection='3d')
    for c in clusters:
        xyz = members.loc[members.cluster_id == c, [f'input_weight_{i}' for i in (1,2,3)]].to_numpy()
        ax.scatter(*xyz.T, s=9, alpha=.4, color=colors[c], label=f'C{c} (n={len(xyz)})', rasterized=True)
        ax.scatter(*xyz.mean(axis=0), s=180, marker='*', color=colors[c], edgecolor='black')
    ax.set(xlabel='|Weight 1|', ylabel='|Weight 2|', zlabel='|Weight 3|', title=f'{title} · stars: member means')
    ax.legend()
    brain = plot_kmeans_hemisphere_centroids(result, title=title, cluster_label='Weight-space')
    table = result['region_contributions']
    regions = ordered_regions(members.region_shared.unique())
    hist, axes = plt.subplots(len(clusters), 1, figsize=(14, max(4, 3*len(clusters))),
                              squeeze=False, layout='constrained')
    for ax, c in zip(axes.flat, clusters):
        frame = table[table.cluster_id == c].set_index('region_shared').reindex(regions, fill_value=0)
        ax.bar(regions, frame.percent, color=colors[c])
        ax.set(title=f'Cluster {c}', ylabel='% of cluster sources', ylim=(0,100))
        ax.tick_params(axis='x', rotation=65, labelsize=8)
    hist.suptitle(f'{title} · regional composition (source counts, not weight magnitude)')
    return {'weights_3d': fig, 'brain_centroids': brain, 'region_contributions': hist}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default='/users/barbara/Desktop/MIB_iEEG/iEEGvsMEG/out/modality_specificity/paired_coverage/plssvd/absolute_weight_clustering' )
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--n-clusters', type=lambda value: 'auto' if value == 'auto' else int(value), default='auto')
    parser.add_argument('--k-min', type=int, default=2)
    parser.add_argument('--k-max', type=int, default=10)
    parser.add_argument('--silhouette-sample-size', type=int, default=2000)
    parser.add_argument('--threshold', action='store_true')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--birch-threshold', type=float, default=0.1)
    args = parser.parse_args()
    run_clustering(args.input_dir, args.methods, n_clusters=args.n_clusters,
                   use_thresholded=args.threshold, seed=args.seed, birch_threshold=args.birch_threshold,
                   k_min=args.k_min, k_max=args.k_max, silhouette_sample_size=args.silhouette_sample_size)
