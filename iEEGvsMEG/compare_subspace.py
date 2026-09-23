#!/usr/bin/env python3
"""Trial-held-out comparison of PCA, PLSSVD and joint PCA spaces.

python -u compare_subspace.py --meg-kind paired_coverage --models separate_pca plssvd joint_pca

Input: the resumable cache produced by plssvd_eval.py (ROOT/out/trial_cache).
Output: ROOT/out/compare_subspace/MEG_KIND. Read with compare_subspace.ipynb.
Independent trial partitions share condition/time rows and a fixed anatomical
mapping within each repetition. All repetitions use all cached participants.
Spatial tests concern new trials at the same locations, NOT new participants.
"""
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes, svd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import PolynomialFeatures

MODALITIES = ('ieeg', 'meg')
MODELS = ('separate_pca', 'plssvd', 'joint_pca')
MEG_KINDS = ('full_average', 'full_concatenated', 'coverage_average',
             'paired_coverage', 'random_control')


def subspace_metrics(a, b):
    """Centered column spaces, with equal nominal dimension and paired rows.

    Missing numerical dimensions contribute zero to overlap (denominator k).
    This avoids inflated overlap after rank loss. No n_locations² matrices.
    """
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError('Subspace inputs must have equal (paired rows, components) shape.')
    bases = []
    ranks = []
    for values in (a, b):
        x = values - values.mean(axis=0)
        u, s, _ = svd(x, full_matrices=False, check_finite=True)
        keep = s > (s[0] * np.finfo(float).eps * max(x.shape) if len(s) else 0)
        bases.append(u[:, keep]); ranks.append(int(keep.sum()))
    cosines = svd(bases[0].T @ bases[1], compute_uv=False) if min(ranks) else np.array([])
    cosines = np.clip(cosines, 0, 1)
    angles = np.degrees(np.arccos(cosines))
    return dict(overlap=float(np.sum(cosines**2) / a.shape[1]),
                rank_a=ranks[0], rank_b=ranks[1], angles_deg=angles.tolist())


def fit_models(train, names, k, block_scaling='equal_variance', max_gram_gib=2.):
    """Fit only requested models through observation Gram matrices."""
    from cov_models_utils import _gram, _spectrum, _weights
    from plssvd_eval_utils import _means
    n = train['ieeg'].n_observations
    if 12*n*n*8/2**30 > max_gram_gib:
        raise MemoryError('Observation Gram matrices exceed --max-gram-gib.')
    grams = {m: _gram(train[m]) for m in MODALITIES}
    scales = {m: (np.sqrt((n-1)/np.trace(grams[m])) if block_scaling == 'equal_variance'
                  else 1.) for m in MODALITIES}
    grams = {m: grams[m]*scales[m]**2 for m in MODALITIES}
    spectra = {m: _spectrum(grams[m], train[m].n_features) for m in MODALITIES}
    if any(len(d) < k for u, d in spectra.values()):
        raise ValueError(f'Training rank is below requested dimension {k}; reduce --dimensions.')
    means = {m: _means(train[m]) for m in MODALITIES}
    models = {}
    for name in names:
        if name == 'separate_pca':
            coefficients = {m: u[:, :k]/d[:k] for m, (u, d) in spectra.items()}
        elif name == 'joint_pca':
            u, d = _spectrum(grams['ieeg']+grams['meg'], sum(v.n_features for v in train.values()))
            coefficients = {m: u[:, :k]/d[:k] for m in MODALITIES}
        elif name == 'plssvd':
            ux, dx = spectra['ieeg']; uy, dy = spectra['meg']
            cross = dx[:, None]*(ux.T@uy)*dy[None, :]/(n-1)
            left, singular, right = svd(cross, full_matrices=False)
            if singular[k-1] <= singular[0]*np.finfo(float).eps*max(cross.shape):
                raise ValueError('PLSSVD cross-covariance rank is too low; reduce --dimensions.')
            coefficients = {'ieeg': (ux/dx)@left[:, :k], 'meg': (uy/dy)@right[:k].T}
        else:
            raise ValueError(f'Unknown model: {name}')
        model = {'k_max': k}
        for m in MODALITIES:
            model[m+'_weights'] = _weights(train[m], coefficients[m], scales[m])
            model[m+'_mean'] = means[m]
            model[m+'_scale'] = scales[m]
        models[name] = model
    return models


def normalize_from_train(parts):
    """One training center and scalar RMS per modality; preserve axis geometry."""
    train = parts['train']
    mean = train.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum((train-mean)**2, axis=1)))
    if scale <= np.finfo(float).tiny:
        raise ValueError('Representation has zero training spread.')
    return {p: (x-mean)/scale for p, x in parts.items()}, mean, scale


def _design(x, complexity):
    return PolynomialFeatures(degree=2, include_bias=False).fit_transform(x) if complexity == 'quadratic' else x


def fit_alignment(x, y, complexity, alpha=0.):
    if complexity == 'identity':
        return dict(complexity=complexity, coef=np.eye(x.shape[1]), intercept=np.zeros(y.shape[1]))
    if complexity == 'orthogonal':
        rotation, _ = orthogonal_procrustes(x, y)
        return dict(complexity=complexity, coef=rotation, intercept=np.zeros(y.shape[1]))
    design = _design(x, complexity)
    mean_x, mean_y = design.mean(0), y.mean(0)
    centered = design-mean_x
    u, s, vt = svd(centered, full_matrices=False)
    penalty = alpha * np.sum(s*s) / design.shape[1]
    coef = (vt.T * (s/(s*s+max(penalty, np.finfo(float).eps)))) @ u.T @ (y-mean_y)
    return dict(complexity=complexity, coef=coef, intercept=mean_y-mean_x@coef)


def predict_alignment(model, x):
    return _design(x, model['complexity'])@model['coef']+model['intercept']


def alignment_error(y, predicted):
    """Target already centered/scaled using TRAIN; zero is its train-mean baseline."""
    error = float(np.sum((predicted-y)**2))
    baseline = float(np.sum(y*y))
    return dict(nrmse=float(np.sqrt(error/baseline)) if baseline > 0 else np.nan,
                q2=1-error/baseline if baseline > 0 else np.nan)


def evaluate_alignment(representations, ridge_grid):
    normalized, centers, scales = {}, {}, {}
    for m in MODALITIES:
        normalized[m], centers[m], scales[m] = normalize_from_train(representations[m])
    rows, artifacts = [], {}
    for source, target in (('ieeg', 'meg'), ('meg', 'ieeg')):
        x, y = normalized[source], normalized[target]
        direction = source+'_to_'+target
        for complexity in ('identity', 'orthogonal', 'affine', 'quadratic'):
            candidates = ridge_grid if complexity in ('affine', 'quadratic') else [0.]
            selected = None
            best = np.inf
            for alpha in candidates:
                fitted = fit_alignment(x['train'], y['train'], complexity, alpha)
                error = alignment_error(y['tune'], predict_alignment(fitted, x['tune']))['nrmse']
                if np.isfinite(error) and error < best:
                    selected = (alpha, fitted); best = error
            if selected is None:
                raise ValueError('No finite tuning alignment error.')
            alpha, fitted = selected
            for part in ('train', 'tune', 'test_a', 'test_b', 'test'):
                rows.append(dict(direction=direction, complexity=complexity, partition=part,
                                 alpha=alpha, **alignment_error(y[part], predict_alignment(fitted, x[part]))))
            prefix = direction+'_'+complexity
            artifacts[prefix+'_coef'] = fitted['coef']
            artifacts[prefix+'_intercept'] = fitted['intercept']
            artifacts[prefix+'_alpha'] = alpha
    for m in MODALITIES:
        artifacts[m+'_center'] = centers[m]; artifacts[m+'_rms'] = scales[m]
    return rows, artifacts


def _silhouette(x, labels, seed):
    count = len(np.unique(labels))
    if count < 2 or count >= len(x):
        return np.nan
    # Bound the O(n²) distance allocation for large electrode collections.
    try:
        return float(silhouette_score(x, labels, sample_size=min(len(x), 2000), random_state=seed))
    except ValueError:
        return np.nan


def evaluate_clusters(representations, cluster_counts, seed):
    """Common cluster count selected by tuning compactness, never test agreement.

    K-means is fit in each modality's full k-dimensional spatial pattern space.
    Orthogonal alignment preserves within-space distances, so it is unnecessary
    for independent clustering/ARI. Fixed-centroid and refit stability are distinct.
    """
    normalized = {m: normalize_from_train(representations[m])[0] for m in MODALITIES}
    best = -np.inf; selected = None; tuning = []
    for count in sorted(set(cluster_counts)):
        if any(len(np.unique(normalized[m]['train'], axis=0)) < count for m in MODALITIES):
            continue
        fitted = {m: KMeans(n_clusters=count, n_init=20, random_state=seed).fit(normalized[m]['train'])
                  for m in MODALITIES}
        scores = [_silhouette(normalized[m]['tune'], fitted[m].predict(normalized[m]['tune']), seed)
                  for m in MODALITIES]
        value = float(np.mean(scores))
        tuning.append(dict(n_clusters=count, ieeg_silhouette=scores[0], meg_silhouette=scores[1],
                           selection_score=value))
        if np.isfinite(value) and value > best:
            selected = (count, fitted); best = value
    if selected is None:
        return [dict(metric='unavailable', modality='both', partition='test', value=np.nan, n_clusters=0)], {}, tuning
    count, fitted = selected
    labels = {m: {p: fitted[m].predict(x) for p, x in normalized[m].items()} for m in MODALITIES}
    rows = []
    for part in ('test_a', 'test_b', 'test'):
        rows.append(dict(metric='cross_modal_ari', modality='both', partition=part,
                         value=adjusted_rand_score(labels['ieeg'][part], labels['meg'][part])))
    for m in MODALITIES:
        rows.append(dict(metric='fixed_centroid_stability', modality=m, partition='test_a_vs_test_b',
                         value=adjusted_rand_score(labels[m]['test_a'], labels[m]['test_b'])))
        refit = {p: KMeans(n_clusters=count, n_init=20, random_state=seed).fit_predict(normalized[m][p])
                 for p in ('test_a', 'test_b')}
        rows.append(dict(metric='refit_stability', modality=m, partition='test_a_vs_test_b',
                         value=adjusted_rand_score(refit['test_a'], refit['test_b'])))
        rows.append(dict(metric='test_silhouette', modality=m, partition='test',
                         value=_silhouette(normalized[m]['test'], labels[m]['test'], seed)))
        for p in refit:
            labels[m]['refit_'+p] = refit[p]
    for row in rows:
        row['n_clusters'] = count
    artifacts = {m+'_'+p: lab for m in MODALITIES for p, lab in labels[m].items()}
    artifacts.update({m+'_centers': fitted[m].cluster_centers_ for m in MODALITIES})
    for m in MODALITIES:
        _, center, scale = normalize_from_train(representations[m])
        artifacts[m+'_normalization_center'] = center
        artifacts[m+'_normalization_rms'] = scale
    return rows, artifacts, tuning


def _representations(fold, scores, k):
    from plssvd_eval_utils import _patterns
    temporal = {m: {p: scores[p][m][:, :k] for p in fold} for m in MODALITIES}
    # All MEG setups expose an explicit electrode -> feature index mapping.
    # Full-source models are fitted on all sources, then patterns sampled here.
    spatial = {m: {p: _patterns(fold[p][m], scores[p][m][:, :k])[
        fold[p][m].electrode_to_feature] for p in fold} for m in MODALITIES}
    return {'temporal_scores': temporal, 'spatial_patterns': spatial}


def run_comparison(cache_dir, output_dir, meg_kind='paired_coverage', models=MODELS,
                   dimensions=(1, 2, 3, 5, 10), repeats=5, seed=2026,
                   block_scaling='equal_variance', split_unit='trial',
                   cluster_counts=(2, 3, 4, 5, 6), ridge_grid=(1e-4, 1e-2, 1., 100.),
                   max_gram_gib=2., scratch_dir=None):
    """Run/export comparison; reuse an already complete PLSSVD trial cache."""
    from plssvd_eval_utils import load_trial_cache, _split_subject, _temporary_fold, _project, _r
    dimensions = sorted(set(dimensions))
    if not dimensions or min(dimensions) < 1 or repeats < 1:
        raise ValueError('Positive dimensions and repeats are required.')
    if not models or not set(models) <= set(MODELS) or meg_kind not in MEG_KINDS:
        raise ValueError('Unknown model or MEG dataset.')
    if not cluster_counts or min(cluster_counts) < 2 or not ridge_grid or min(ridge_grid) <= 0:
        raise ValueError('Cluster counts must be >=2 and ridge penalties >0.')
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if (out/'config.json').exists():
        raise FileExistsError('Results already exist. Choose a new --output-dir or use --plot-only.')
    trials = load_trial_cache(cache_dir)
    manifest = json.loads((Path(cache_dir)/'manifest.json').read_text())
    for modality, subjects in (('ieeg', trials.ieeg), ('meg', trials.meg)):
        if not subjects or {s.subject for s in subjects} != set(map(str, manifest['config'][modality+'_subjects'])):
            raise ValueError('Trial cache is incomplete. Finish prepare_trial_cache first.')
    config = dict(cache_dir=str(Path(cache_dir).resolve()), meg_kind=meg_kind, models=list(models),
                  dimensions=dimensions, repeats=repeats, seed=seed, block_scaling=block_scaling,
                  split_unit=split_unit, cluster_counts=list(cluster_counts), ridge_grid=list(ridge_grid),
                  max_gram_gib=max_gram_gib, scope='new trials at fixed participant locations',
                  spatial_representation='forward patterns on the electrode grid',
                  normalization='training column means and one RMS per modality',
                  within_model_comparison='all model pairs within each modality; all native features',
                  within_model_matching='Pearson Hungarian assignment/signs on training temporal scores, frozen across spaces and partitions',
                  repetitions='overlapping sensitivity analyses; not independent confidence intervals')
    (out/'config.json').write_text(json.dumps(config, indent=2))
    (out/'cache_config.json').write_text(json.dumps(manifest['config'], indent=2))
    np.savez_compressed(out/'axes.npz', times=trials.times, conditions=trials.conditions)
    tables = {name: [] for name in ('overlap', 'alignment', 'reliability', 'clusters', 'cluster_selection', 'splits',
                                  'within_model_metrics', 'within_model_pairs', 'within_model_correlations')}
    for repeat, seq in enumerate(np.random.SeedSequence(seed).spawn(repeats)):
        rng = np.random.default_rng(seq)
        matching_seed = int(rng.integers(2**31-1))
        indices = {m: {s.subject: _split_subject(s, rng, split_unit) for s in getattr(trials, m)} for m in MODALITIES}
        for m in MODALITIES:
            for subject, partitions in indices[m].items():
                for part, conditions in partitions.items():
                    if part == 'test':
                        continue  # test is exactly the union of test_a and test_b
                    for condition, ix in zip(trials.conditions, conditions):
                        tables['splits'].extend(dict(repeat=repeat, modality=m, subject=subject,
                                                     partition=part, condition=condition, trial_index=int(i)) for i in ix)
        with _temporary_fold(trials, meg_kind, indices, matching_seed, scratch_dir) as prepared_fold:
            fold, scalers, audit, pairing = prepared_fold
            audit.to_csv(out/f'matching_{repeat:03d}.csv', index=False)
            (out/f'pairing_{repeat:03d}.json').write_text(json.dumps(pairing, indent=2))
            np.savez_compressed(out/f'preprocessing_{repeat:03d}.npz', **{
                f'{m}_{s}_{label}': value for m, subjects in scalers.items() for s, values in subjects.items()
                for label, value in zip(('mean', 'std', 'multiplier'), values)})
            for m in MODALITIES:
                fold['train'][m].metadata.to_csv(out/f'{m}_features_{repeat:03d}.csv', index=False)
                np.save(out/f'{m}_electrode_map_{repeat:03d}.npy', fold['train'][m].electrode_to_feature)
            fitted = fit_models(fold['train'], models, max(dimensions), block_scaling, max_gram_gib)
            for name, model in fitted.items():
                print(f'Repetition {repeat+1}/{repeats}: {meg_kind}, {name}', flush=True)
                scores = {p: {m: _project(fold[p][m], model, m) for m in MODALITIES} for p in fold}
                np.savez_compressed(out/f'{name}_{repeat:03d}_model.npz', **model,
                                    **{f'{p}_{m}_scores': a for p, modalities in scores.items() for m, a in modalities.items()})
                for k in dimensions:
                    base = dict(model=name, repeat=repeat, k=k)
                    representations = _representations(fold, scores, k)
                    for space, reps in representations.items():
                        prefix = f'{name}_{repeat:03d}_k{k}_{space}'
                        if space == 'spatial_patterns':
                            np.savez_compressed(out/f'{prefix}.npz', **{m+'_'+p: x for m, parts in reps.items() for p, x in parts.items()})
                        for part in ('train', 'tune', 'test_a', 'test_b', 'test'):
                            metrics = subspace_metrics(reps['ieeg'][part], reps['meg'][part])
                            metrics['angles_deg'] = json.dumps(metrics['angles_deg'])
                            tables['overlap'].append(dict(**base, space=space, partition=part, **metrics))
                        for m in MODALITIES:
                            a, b = reps[m]['test_a'], reps[m]['test_b']
                            metrics = subspace_metrics(a, b)
                            metrics['angles_deg'] = json.dumps(metrics['angles_deg'])
                            correlations = _r(a, b)
                            tables['reliability'].append(dict(**base, space=space, modality=m,
                                mean_signed_r=float(np.mean(correlations)),
                                component_r=json.dumps(correlations.tolist()), **metrics))
                        rows, artifacts = evaluate_alignment(reps, ridge_grid)
                        tables['alignment'].extend(dict(**base, space=space, **row) for row in rows)
                        np.savez_compressed(out/f'{prefix}_alignment.npz', **artifacts)
                    rows, artifacts, tuning = evaluate_clusters(representations['spatial_patterns'], cluster_counts, matching_seed)
                    tables['clusters'].extend(dict(**base, **row) for row in rows)
                    tables['cluster_selection'].extend(dict(**base, **row) for row in tuning)
                    np.savez_compressed(out/f'{name}_{repeat:03d}_k{k}_clusters.npz', **artifacts)
                del scores
            if len(fitted) > 1:
                from compare_models import compare_fitted_models
                within = compare_fitted_models(fold, fitted, dimensions, repeat=repeat)
                for key, table in within.items():
                    tables[key].extend(table.to_dict('records'))
        for table, rows in tables.items():
            frame = pd.DataFrame(rows)
            if table == 'cluster_selection' and frame.empty:
                frame = pd.DataFrame(columns=['model', 'repeat', 'k', 'n_clusters',
                                              'ieeg_silhouette', 'meg_silhouette', 'selection_score'])
            if not (table.startswith('within_model_') and frame.empty):
                frame.to_csv(out/f'{table}.csv', index=False)
    (out/'COMPLETE.json').write_text(json.dumps(dict(repeats=repeats, models=list(models))))
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}


def load_results(output_dir):
    root = Path(output_dir)
    config = json.loads((root/'config.json').read_text())
    if not (root/'COMPLETE.json').exists():
        warnings.warn('Run has not completed; tables may describe only finished repetitions.')
    names = ['overlap', 'alignment', 'reliability', 'clusters', 'cluster_selection']
    names += [name for name in ('within_model_metrics', 'within_model_pairs', 'within_model_correlations')
              if (root/f'{name}.csv').exists()]
    return config, {name: pd.read_csv(root/f'{name}.csv') for name in names}


def plot_results(output_dir, k=None, show=True):
    """Save cross-modal and available within-modal model comparison figures."""
    import matplotlib.pyplot as plt
    config, tables = load_results(output_dir)
    k = k or max(config['dimensions'])
    if k not in config['dimensions']:
        raise ValueError('Choose a dimension saved in this run.')
    out = Path(output_dir)
    spaces = ('temporal_scores', 'spatial_patterns')
    figures = []

    def finish(fig, filename):
        for ext in ('png', 'pdf'):
            fig.savefig(out/f'{filename}.{ext}', dpi=180, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)
        figures.append(str(out/f'{filename}.png'))

    def curve(ax, data, x, y, label):
        group = data.groupby(x)[y]
        summary = group.agg(['median', 'min', 'max'])
        ax.plot(summary.index, summary['median'], 'o-', label=label)
        ax.fill_between(summary.index.to_numpy(), summary['min'].to_numpy(), summary['max'].to_numpy(), alpha=.12)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    data = tables['overlap'].query("partition == 'test'")
    for ax, space in zip(axes, spaces):
        for name, rows in data[data.space == space].groupby('model'):
            curve(ax, rows, 'k', 'overlap', name)
        ax.set(title=space.replace('_', ' '), xlabel='Retained dimensions', xticks=config['dimensions'], ylabel='Mean squared cosine (overlap)', ylim=(0, 1.02))
        ax.legend()
    fig.suptitle('Held-out subspace overlap — median and range across repetitions')
    finish(fig, 'subspace_overlap')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    data = tables['alignment'].query("partition == 'test' and k == @k").copy()
    order = ['identity', 'orthogonal', 'affine', 'quadratic']
    data['complexity_index'] = data.complexity.map({v: i for i, v in enumerate(order)})
    for i, space in enumerate(spaces):
        for j, direction in enumerate(('ieeg_to_meg', 'meg_to_ieeg')):
            ax = axes[i, j]
            for name, rows in data[(data.space == space) & (data.direction == direction)].groupby('model'):
                curve(ax, rows, 'complexity_index', 'nrmse', name)
            ax.axhline(1, color='grey', linestyle=':', label='Training-mean baseline')
            ax.set(xticks=range(4), xticklabels=order, ylabel='Held-out normalized error', title=f'{space}: {direction}')
            ax.legend(fontsize=8)
    fig.suptitle(f'Alignment complexity, k={k} — ridge chosen on tuning trials; lower is better')
    finish(fig, f'alignment_complexity_k{k}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    data = tables['reliability']
    for ax, space in zip(axes, spaces):
        for (name, modality), rows in data[data.space == space].groupby(['model', 'modality']):
            curve(ax, rows, 'k', 'overlap', f'{name}: {modality}')
        ax.set(title=space.replace('_', ' '), xlabel='Retained dimensions', xticks=config['dimensions'], ylabel='Test-half subspace overlap', ylim=(0, 1.02))
        ax.legend(fontsize=7)
    fig.suptitle('Within-modality reliability — independent test halves, fixed trained axes')
    finish(fig, 'within_modality_reliability')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    data = tables['clusters']
    for ax, metric in zip(axes, ('cross_modal_ari', 'fixed_centroid_stability', 'refit_stability')):
        subset = data[data.metric == metric]
        if metric == 'cross_modal_ari':
            subset = subset[subset.partition == 'test']
        for (name, modality), rows in subset.groupby(['model', 'modality']):
            curve(ax, rows, 'k', 'value', f'{name}: {modality}')
        ax.axhline(0, color='grey', linestyle=':')
        ax.set(title=metric.replace('_', ' '), xlabel='Retained dimensions', xticks=config['dimensions'], ylabel='Adjusted Rand index', ylim=(-1, 1.02))
        if len(subset):
            ax.legend(fontsize=7)
        else:
            ax.text(.5, .5, 'No valid cluster solution', ha='center', transform=ax.transAxes)
    fig.suptitle('Spatial clusters — common cluster count selected on tuning trials')
    finish(fig, 'cluster_stability_agreement')
    if 'within_model_metrics' in tables:
        from compare_models import plot_within_models
        for index, fig in enumerate(plot_within_models(tables, k=k, partition='test')):
            finish(fig, f'within_models_k{k}_{index:02d}')
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--scratch-dir', type=Path, help='Temporary fold storage (default: TMPDIR/system temp).')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--meg-kind', choices=MEG_KINDS, default='paired_coverage')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--dimensions', nargs='+', type=int, default=[1, 2, 3, 5, 10])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--block-scaling', choices=['none', 'equal_variance'], default='equal_variance')
    parser.add_argument('--split-unit', choices=['trial', 'group'], default='trial')
    parser.add_argument('--cluster-counts', nargs='+', type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument('--ridge-grid', nargs='+', type=float, default=[1e-4, 1e-2, 1., 100.])
    parser.add_argument('--max-gram-gib', type=float, default=2.)
    parser.add_argument('--plot-only', action='store_true', help='Regenerate figures from saved result tables.')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    out = args.output_dir or args.root/'out'/'compare_subspace'/args.meg_kind
    if not args.plot_only:
        run_comparison(args.cache_dir or args.root/'out'/'trial_cache', out,
                       meg_kind=args.meg_kind, models=args.models, dimensions=args.dimensions,
                       repeats=args.repeats, seed=args.seed, block_scaling=args.block_scaling,
                       split_unit=args.split_unit, cluster_counts=args.cluster_counts,
                       ridge_grid=args.ridge_grid, max_gram_gib=args.max_gram_gib, scratch_dir=args.scratch_dir)
    plot_results(out, show=False)
    print(f'Outputs saved to: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
