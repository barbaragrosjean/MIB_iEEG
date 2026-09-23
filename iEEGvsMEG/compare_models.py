"""Within-modality model comparisons, with training-frozen component matching.

No fitting or test-driven component selection occurs here. Spatial inputs are
forward patterns on the same ordered features, not projection weights.
"""
from itertools import combinations
import json
import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment


def _dimensions(values):
    values = list(values)
    if not values or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v < 1 for v in values):
        raise ValueError('Dimensions must be positive integers.')
    return sorted(set(map(int, values)))


def correlation_matrix(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 2 or b.ndim != 2 or len(a) != len(b):
        raise ValueError('Correlation inputs require paired rows.')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Representations must be finite.')
    a, b = a-a.mean(0), b-b.mean(0)
    denominator = np.outer(np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0))
    return np.divide(a.T@b, denominator, out=np.full(denominator.shape, np.nan),
                     where=denominator > 0).clip(-1, 1)


def space_metrics(a, b):
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] < 1:
        raise ValueError('Subspaces need equal, nonempty (rows, components) shapes.')
    bases = []
    for x in (a, b):
        u, s, _ = svd(x-x.mean(0), full_matrices=False)
        keep = s > (s[0]*np.finfo(float).eps*max(x.shape) if len(s) else 0)
        bases.append(u[:, keep])
    cosines = svd(bases[0].T@bases[1], compute_uv=False)
    cosines = np.clip(cosines, 0, 1)
    return dict(overlap=float(np.sum(cosines**2)/a.shape[1]),
                rank_a=bases[0].shape[1], rank_b=bases[1].shape[1],
                angles_deg=json.dumps(np.degrees(np.arccos(cosines)).tolist()))


def compare_representations(representations, *, modality, k, repeat=0):
    """Compare model -> space -> partition -> paired-row arrays.

    Select ONE component assignment and orientation on training temporal scores
    per model pair, then reuse it for temporal scores, condition contrasts and
    spatial patterns in every partition. Undefined training correlations are
    excluded from matched summaries, with valid counts reported. No test
    rematching or test sign flips. Full signed Pearson matrices are exported.
    """
    metrics, pairs, correlations = [], [], []
    for model_a, model_b in combinations(representations, 2):
        a, b = representations[model_a], representations[model_b]
        train = correlation_matrix(a['temporal_scores']['train'], b['temporal_scores']['train'])
        if train.shape != (k, k):
            raise ValueError('Representations must have exactly k columns.')
        ia, ib = linear_sum_assignment(-np.nan_to_num(np.abs(train), nan=-1.))
        valid_train = np.isfinite(train[ia, ib])
        signs = np.where(train[ia, ib] < 0, -1., 1.)
        base = dict(modality=modality, model_a=model_a, model_b=model_b, k=k, repeat=repeat)
        for i, j, sign, valid in zip(ia, ib, signs, valid_train):
            pairs.append(dict(**base, component_a=int(i+1), component_b=int(j+1),
                              sign_b=float(sign), train_temporal_r=float(train[i, j]),
                              valid_training_pair=bool(valid)))
        if set(a) != set(b):
            raise ValueError('Models must expose the same representations.')
        for space in a:
            if set(a[space]) != set(b[space]):
                raise ValueError('Models must expose identical partitions.')
            for part in a[space]:
                x, y = a[space][part], b[space][part]
                r = correlation_matrix(x, y)
                values = r[ia, ib]*signs
                valid = valid_train & np.isfinite(values)
                metrics.append(dict(**base, space=space, partition=part,
                    **space_metrics(x, y), n_valid_pairs=int(valid.sum()),
                    matched_signed_r=float(values[valid].mean()) if valid.any() else np.nan,
                    matched_abs_r=float(np.abs(values[valid]).mean()) if valid.any() else np.nan))
                for i in range(k):
                    for j in range(k):
                        correlations.append(dict(**base, space=space, partition=part,
                            component_a=i+1, component_b=j+1, signed_r=float(r[i, j])))
    return {name: pd.DataFrame(rows) for name, rows in
            [('within_model_metrics', metrics), ('within_model_pairs', pairs),
             ('within_model_correlations', correlations)]}


def compare_fitted_models(fold, fitted, dimensions, repeat=0):
    """Held-out comparison on ALL native features within each modality.

    Pattern regression uses the retained k scores (not slices of a regression
    fitted with more components). Spatial patterns are evaluated one model pair
    and partition at a time, avoiding retention of all large full-source maps.
    """
    from plssvd_eval_utils import _project, _patterns
    dimensions = _dimensions(dimensions)
    if len(fitted) < 2:
        raise ValueError('Within-modality comparison requires at least two models.')
    collected = {}
    for modality in ('ieeg', 'meg'):
        scores = {name: {part: _project(datasets[modality], model, modality)
                        for part, datasets in fold.items()} for name, model in fitted.items()}
        for k in dimensions:
            reps = {}
            for name, parts in scores.items():
                temporal = {p: s[:, :k] for p, s in parts.items()}
                reps[name] = dict(temporal_scores=temporal)
                ds = fold['train'][modality]
                if ds.condition_mode == 'stack' and ds.arrays[0].shape[0] == 2:
                    nt = ds.arrays[0].shape[-1]
                    reps[name]['condition_contrast'] = {p: s[nt:]-s[:nt] for p, s in temporal.items()}
            result = compare_representations(reps, modality=modality, k=k, repeat=repeat)
            for key, value in result.items():
                collected.setdefault(key, []).append(value)
            for name_a, name_b in combinations(reps, 2):
                for part in fold:
                    spatial = {name: dict(
                        temporal_scores={'train': reps[name]['temporal_scores']['train']},
                        spatial_patterns={part: _patterns(fold[part][modality], reps[name]['temporal_scores'][part])})
                        for name in (name_a, name_b)}
                    result = compare_representations(spatial, modality=modality, k=k, repeat=repeat)
                    for key in ('within_model_metrics', 'within_model_correlations'):
                        collected[key].append(result[key].query("space == 'spatial_patterns'"))
                    del spatial
    return {key: pd.concat(values, ignore_index=True) for key, values in collected.items()}


def compare_cov_models_within(models, ieeg, meg, dimensions=(1, 2, 3, 5, 10)):
    """Descriptive in-sample counterpart for cov_models.ipynb (no holdout claim)."""
    from plssvd_eval_utils import _patterns
    dimensions = _dimensions(dimensions)
    if len(models) < 2:
        raise ValueError('Within-modality comparison requires at least two models.')
    collected = {}
    for modality, dataset in [('ieeg', ieeg), ('meg', meg)]:
        for k in dimensions:
            reps = {}
            for name, model in models.items():
                scores = getattr(model, modality+'_scores')
                if k < 1 or k > scores.shape[1]:
                    raise ValueError('Choose dimensions available in every fitted model.')
                scores = scores[:, :k]
                reps[name] = dict(temporal_scores={'train': scores},
                                 spatial_patterns={'train': _patterns(dataset, scores)})
                if dataset.condition_mode == 'stack' and dataset.arrays[0].shape[0] == 2:
                    nt = dataset.arrays[0].shape[-1]
                    reps[name]['condition_contrast'] = {'train': scores[nt:]-scores[:nt]}
            result = compare_representations(reps, modality=modality, k=k)
            for key, value in result.items():
                if 'partition' in value:
                    value['partition'] = 'in_sample'
                collected.setdefault(key, []).append(value)
    return {key: pd.concat(values, ignore_index=True) for key, values in collected.items()}


def plot_within_models(result, k, partition='test'):
    """Overlap/matching summaries and primary-repetition correlation heatmaps."""
    import matplotlib.pyplot as plt
    table = result['within_model_metrics']
    table = table[(table.k == k) & (table.partition == partition)]
    if table.empty:
        raise ValueError('No results for this dimension/partition.')
    figures = []
    for (modality, space), group in table.groupby(['modality', 'space'], sort=False):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        pairs = list(group.groupby(['model_a', 'model_b'], sort=False))
        for ax, metric in zip(axes, ['overlap', 'matched_signed_r']):
            for index, ((a, b), rows) in enumerate(pairs):
                values = rows[metric].dropna().to_numpy()
                ax.scatter(np.repeat(index, len(values)), values, alpha=.5)
                if len(values):
                    ax.plot(index, np.median(values), 'k_')
            labels = [a+' / '+b for (a, b), _ in pairs]
            ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel=metric,
                   ylim=(-1.05, 1.05) if metric.endswith('_r') else (-.05, 1.05))
            ax.tick_params(axis='x', labelrotation=20)
        fig.suptitle(f'{modality}: {space}, k={k}, {partition} (dots = repetitions)')
        figures.append(fig)
    corr = result['within_model_correlations']
    corr = corr[(corr.k == k) & (corr.partition == partition) & (corr.repeat == corr.repeat.min())]
    for (modality, space), group in corr.groupby(['modality', 'space'], sort=False):
        groups = list(group.groupby(['model_a', 'model_b'], sort=False))
        fig, axes = plt.subplots(1, len(groups), figsize=(5*len(groups), 4),
                                 squeeze=False, constrained_layout=True)
        for ax, ((a, b), pair) in zip(axes.flat, groups):
            matrix = pair.pivot(index='component_a', columns='component_b', values='signed_r')
            im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r')
            ax.set(title=f'{a} vs {b}', xlabel=b, ylabel=a,
                   xticks=range(k), xticklabels=range(1, k+1), yticks=range(k), yticklabels=range(1, k+1))
            fig.colorbar(im, ax=ax, label='Pearson r (native signs)')
        fig.suptitle(f'{modality}: {space}, {partition}, primary repetition')
        figures.append(fig)
    return figures
