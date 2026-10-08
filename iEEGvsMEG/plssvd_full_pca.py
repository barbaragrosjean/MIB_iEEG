"""Descriptive full-data PCA versus stored train/test PLSSVD time courses."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt


def fit_full_data_pca(cache_dir, n_components, scratch_dir=None):
    """Fit one PCA per modality to all trials, with concatenated MEG features.

    Trial means are computed within each condition, preprocessing is estimated
    from all trials, then conditions are averaged equally. Feature blocks are
    accumulated into a temporal Gram matrix rather than a dense concatenation.
    Inputs are unpermuted. No PLSSVD fitting, fold evaluation or sidecar writes.
    """
    from plssvd_eval_utils import load_trial_cache, _temporary_fold
    from plssvd_evaluation import selected_count
    from cov_models_utils import _gram, _spectrum
    trials=load_trial_cache(cache_dir)
    k=selected_count(n_components,n_components) if n_components is not None else None
    if k is None:raise ValueError('Select an explicit N_COMPONENTS for full-data PCA.')
    indices={m:{s.subject:{'train':[np.arange(len(a)) for a in s.data]}
                for s in getattr(trials,m)} for m in ('ieeg','meg')}
    scores={};singular_values={}
    with _temporary_fold(trials,'full_concatenated',indices,seed=2026,scratch_dir=scratch_dir,
                         condition_mode='average') as prepared:
        datasets=prepared[0]['train']
        for modality in ('ieeg','meg'):
            dataset=datasets[modality]
            u,d=_spectrum(_gram(dataset),dataset.n_features)
            if len(d)<k:raise ValueError(f'{modality}: full-data PCA rank {len(d)} is below {k}.')
            values=u[:,:k]*d[:k]
            signs=np.sign(values[np.argmax(abs(values),axis=0),np.arange(k)]);signs[signs==0]=1
            scores[modality]=values*signs
            singular_values[modality]=d
    return dict(scores=scores,singular_values=singular_values,times=trials.times,
                conditions=trials.conditions,n_components=k,
                subjects={m:[s.subject for s in getattr(trials,m)] for m in ('ieeg','meg')})


def correlate_full_pca(run_dir, pca, n_components=None, absolute=True):
    """Average correlations across iterations SEPARATELY for each fold/partition.

    PCA remains fixed. Every matrix contains all PLSSVD-by-PCA component pairs.
    Online moments keep memory independent of iteration count. Undefined
    correlations are excluded per cell; sample SD is NaN for fewer than 2 values.
    """
    from plssvd_eval_utils import _saved_scores
    from plssvd_evaluation import selected_count
    from plssvd_diagnostics import _audit
    from plssvd_postprocess import Moments
    from compare_models import correlation_matrix
    root=Path(run_dir);config=json.loads((root/'validation_options.json').read_text())
    if config.get('meg_kind')!='full_concatenated':
        raise ValueError('This comparison requires the full_concatenated MEG run.')
    k=selected_count(n_components,min(config['n_components'],pca['n_components']))
    ids=[i for i in range(config['n_iterations']) if (root/f'iteration_{i:03d}'/'COMPLETE.json').is_file()]
    if not ids:raise FileNotFoundError('No completed PLSSVD iterations are available.')
    axes=root/'trial_axes.npz'
    if not axes.exists():axes=root/f'iteration_{ids[0]:03d}'/'trial_axes.npz'
    with np.load(axes) as saved:
        if not np.array_equal(saved['times'],pca['times']) or tuple(saved['conditions'])!=tuple(pca['conditions']):
            raise ValueError('PCA and PLSSVD time/condition axes differ.')
    audit=_audit(root,ids[0])
    for m in ('ieeg','meg'):
        if set(audit.loc[audit.modality==m,'subject'])!=set(pca['subjects'][m]):
            raise ValueError(f'{m}: full-data PCA and saved PLSSVD cohorts differ; use the original trial cache.')
    stats={(m,p,f):Moments() for m in ('ieeg','meg') for p in ('train','test') for f in range(config['repeats'])}
    for iteration in ids:
        for fold in range(config['repeats']):
            with _saved_scores(root/f'iteration_{iteration:03d}',fold) as saved:
                for m in ('ieeg','meg'):
                    train=saved[f'train_{m}'][:,:k]
                    signs=np.sign(train[np.argmax(abs(train),axis=0),np.arange(k)]);signs[signs==0]=1
                    for part in ('train','test'):
                        values=saved[f'{part}_{m}'][:,:k]
                        if values.shape!=(len(pca['times']),k):raise ValueError('Saved PLSSVD scores have incompatible dimensions.')
                        r=correlation_matrix(values*signs,pca['scores'][m][:,:k])
                        stats[m,part,fold].add(abs(r) if absolute else r)
    shape=(2,2,config['repeats'],k,k)
    mean=np.empty(shape);sd=np.empty(shape);count=np.empty(shape,int)
    for mi,m in enumerate(('ieeg','meg')):
        for pi,p in enumerate(('train','test')):
            for f in range(config['repeats']):mean[mi,pi,f],sd[mi,pi,f],count[mi,pi,f]=stats[m,p,f].arrays()
    print(f'Full-data PCA reference versus PLSSVD: {len(ids)}/{config["n_iterations"]} iterations; '
          f'{config["repeats"]} folds kept separate, k={k}.')
    return dict(mean=mean,std=sd,count=count,iterations=ids,n_components=k,absolute=absolute,
                n_folds=config['repeats'],perm=config.get('perm'),perm_type=config.get('perm_type'))


def plot_full_pca_comparison(result):
    """Four rows (modality × partition), one column per fold; mean ± sample SD."""
    from plssvd_eval_utils import _annotate_matrix
    folds=result['n_folds'];k=result['n_components']
    fig,axes=plt.subplots(4,folds,figsize=(3.8*folds,14),squeeze=False,layout='constrained')
    for mi,m in enumerate(('ieeg','meg')):
        for pi,p in enumerate(('train','test')):
            for fold in range(folds):
                ax=axes[2*mi+pi,fold]
                mean=result['mean'][mi,pi,fold];sd=result['std'][mi,pi,fold]
                im=ax.imshow(mean,vmin=0 if result['absolute'] else -1,vmax=1,
                             cmap='viridis' if result['absolute'] else 'RdBu_r')
                if k<=15:_annotate_matrix(ax,mean,sd)
                ticks=np.arange(k) if k<=15 else np.arange(0,k,5)
                ax.set(title=f'{m.upper()} {p} · fold {fold+1}',xlabel='Full-data PCA component',
                       ylabel='PLSSVD component',xticks=ticks,xticklabels=ticks+1,yticks=ticks,yticklabels=ticks+1)
    stat='|Pearson r|' if result['absolute'] else 'Pearson r'
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,label=f'Mean {stat}')
    fig.suptitle(f'Full-data PCA versus saved PLSSVD scores · {len(result["iterations"])} iterations · k={k}\n'
                 f'Cells: mean ± SD across iterations, separately for each fold · PLSSVD permutation: {result["perm"]}/{result["perm_type"]}\n'
                 'Fixed unpermuted PCA fitted on all trials: descriptive comparison, not cross-validated PCA.')
    return fig
