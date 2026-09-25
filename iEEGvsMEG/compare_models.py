"""Within-modality model comparisons, with training-frozen component matching.

No fitting or test-driven component selection occurs here. Spatial inputs are
forward patterns on the same ordered features, not projection weights.
"""
from itertools import combinations
from dataclasses import replace
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
    per model pair, then reuse it for condition-averaged temporal scores and
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


def _condition_average(dataset, scores):
    """Average already fitted score contributions; never refit model weights."""
    if dataset.condition_mode == 'stack':
        scores = scores.reshape(dataset.arrays[0].shape[0], dataset.arrays[0].shape[-1], -1).mean(0)
    return replace(dataset, condition_mode='average'), scores


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
        scores = {name: {part: _condition_average(datasets[modality], _project(datasets[modality], model, modality))[1]
                        for part, datasets in fold.items()} for name, model in fitted.items()}
        for k in dimensions:
            reps = {}
            for name, parts in scores.items():
                temporal = {p: s[:, :k] for p, s in parts.items()}
                reps[name] = dict(temporal_scores=temporal)
            result = compare_representations(reps, modality=modality, k=k, repeat=repeat)
            for key, value in result.items():
                collected.setdefault(key, []).append(value)
            for name_a, name_b in combinations(reps, 2):
                for part in fold:
                    spatial = {name: dict(
                        temporal_scores={'train': reps[name]['temporal_scores']['train']},
                        spatial_patterns={part: _patterns(replace(fold[part][modality], condition_mode='average'), reps[name]['temporal_scores'][part])})
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
                averaged, scores = _condition_average(dataset, getattr(model, modality+'_scores'))
                if k < 1 or k > scores.shape[1]:
                    raise ValueError('Choose dimensions available in every fitted model.')
                scores = scores[:, :k]
                reps[name] = dict(temporal_scores={'train': scores},
                                 spatial_patterns={'train': _patterns(averaged, scores)})
            result = compare_representations(reps, modality=modality, k=k)
            for key, value in result.items():
                if 'partition' in value:
                    value['partition'] = 'in_sample'
                collected.setdefault(key, []).append(value)
    return {key: pd.concat(values, ignore_index=True) for key, values in collected.items()}


def plot_within_models(result, k, partition='test'):
    """One compact summary: rows=model pairs, columns=space and metric.

    Matching comes from temporal scores; absolute spatial correlations retain
    those assignments. Repeated runs are summarized by their median.
    """
    import matplotlib.pyplot as plt
    table = result['within_model_metrics']
    table = table[(table.k == k) & (table.partition == partition)]
    if table.empty:
        raise ValueError('No results for this dimension/partition.')
    modalities = [m for m in ('ieeg','meg') if m in set(table.modality)]
    labels = {'separate_pca':'PCA', 'plssvd':'PLSSVD', 'joint_pca':'Joint PCA'}
    columns = [(space,metric) for space in ('temporal_scores','spatial_patterns')
               for metric in ('overlap','matched_abs_r')]
    fig, axes = plt.subplots(1,len(modalities),figsize=(11,3.8),squeeze=False,layout='constrained')
    for ax, modality in zip(axes.flat,modalities):
        sub = table[table.modality == modality]
        pairs = list(sub[['model_a','model_b']].drop_duplicates().itertuples(index=False,name=None))
        matrix = np.full((len(pairs),len(columns)),np.nan)
        for i,(a,b) in enumerate(pairs):
            for j,(space,metric) in enumerate(columns):
                values = sub[(sub.model_a==a)&(sub.model_b==b)&(sub.space==space)][metric]
                matrix[i,j] = values.median()
        im = ax.imshow(np.ma.masked_invalid(matrix),vmin=0,vmax=1,cmap='viridis',aspect='auto')
        ax.set(title='iEEG' if modality=='ieeg' else 'MEG',
               xticks=range(4),xticklabels=['Time subspace\noverlap','Time components\nmean |r|','Spatial subspace\noverlap','Spatial components\nmean |r|'],
               yticks=range(len(pairs)),yticklabels=[labels.get(a,a)+' vs '+labels.get(b,b) for a,b in pairs])
        ax.axvline(1.5,color='white',lw=2)
        for i in range(len(pairs)):
            for j in range(4):
                v = matrix[i,j]
                ax.text(j,i,f'{v:.2f}' if np.isfinite(v) else '—',ha='center',va='center',
                        color='black' if v>.55 else 'white',fontsize=11)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Similarity (0–1)',shrink=.8)
    setting = 'Same-data comparison' if partition=='in_sample' else partition
    fig.suptitle(f'Within-modality model similarity · {k} components · {setting}')
    fig.supxlabel('Equal condition averages · component pairs chosen from time courses · median across runs',fontsize=9)
    return [fig]


def plot_within_components(result, k=10, partition='in_sample'):
    """Individual matched correlations; fixed matching among the first k axes.

    Labels A→B expose permutations. Uses the first recorded repetition, not
    an average across potentially different assignments in repeated fits.
    """
    import matplotlib.pyplot as plt
    corr = result['within_model_correlations']
    corr = corr[(corr.k == k) & (corr.partition == partition)]
    if corr.empty:
        raise ValueError('No component correlations for this dimension/partition.')
    repeat = corr.repeat.min()
    corr = corr[corr.repeat == repeat]
    pairs = result['within_model_pairs']
    pairs = pairs[(pairs.k == k) & (pairs.repeat == repeat) & pairs.valid_training_pair]
    keys = ['modality','model_a','model_b','k','repeat','component_a','component_b']
    matched = corr.merge(pairs[keys], on=keys, validate='many_to_one')
    labels = {'separate_pca':'PCA','joint_pca':'Joint PCA','plssvd':'PLSSVD'}
    fig, axes = plt.subplots(2,2,figsize=(max(11,k*1.05),6),layout='constrained')
    for row,modality in enumerate(('ieeg','meg')):
        sub = corr[corr.modality == modality]
        model_pairs = list(sub[['model_a','model_b']].drop_duplicates().itertuples(index=False,name=None))
        for col,space in enumerate(('temporal_scores','spatial_patterns')):
            ax = axes[row,col]
            matrix = np.full((len(model_pairs),k),np.nan)
            annotations = {}
            for i,(a,b) in enumerate(model_pairs):
                records = matched[(matched.modality==modality)&(matched.model_a==a)&(matched.model_b==b)&(matched.space==space)]
                for record in records.itertuples():
                    j = int(record.component_a)-1
                    matrix[i,j] = abs(record.signed_r)
                    annotations[i,j] = f'{abs(record.signed_r):.2f}\n{record.component_a}→{record.component_b}'
            im = ax.imshow(np.ma.masked_invalid(matrix),cmap='viridis',vmin=0,vmax=1,aspect='auto')
            ax.set(title=f'{"iEEG" if modality=="ieeg" else "MEG"} · {"Time courses" if col==0 else "Spatial patterns"}',
                   xticks=range(k),xticklabels=range(1,k+1),xlabel='Component in first model (A)',
                   yticks=range(len(model_pairs)),yticklabels=[f'{labels.get(a,a)} vs {labels.get(b,b)}' for a,b in model_pairs])
            for i in range(len(model_pairs)):
                for j in range(k):
                    ax.text(j,i,annotations.get((i,j),'—'),ha='center',va='center',fontsize=8,
                            color='black' if matrix[i,j]>.55 else 'white')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Individual matched |Pearson r|',shrink=.8)
    fig.suptitle(f'Individual components · matching within first {k} · {partition} · run {repeat}')
    fig.supxlabel('Each cell: |r| and component A→B. Spatial comparisons reuse temporal pairs; no spatial rematching.',fontsize=9)
    return fig
