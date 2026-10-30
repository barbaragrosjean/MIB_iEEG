"""Training-only, separate-modality PCA and held-out PLSSVD/PCA comparisons."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def independent_pca_scores(train, test, n_components, scale=1., train_gram=None):
    """Fit PCA on training time x feature data; transform held-out data.

    PCA is centered, not whitened. A modality-wide positive scale matches the
    PLSSVD preprocessing and cannot affect Pearson correlations. Work in the
    temporal Gram space and project feature chunks without storing PCA weights.
    Test data never enter the training eigendecomposition or sign convention.
    """
    from scipy.linalg import eigh
    from vs2_0_cov_models_utils import _gram
    gram = _gram(train)*scale**2 if train_gram is None else train_gram
    values,u = eigh(gram,check_finite=False)
    values,u = values[::-1],u[:,::-1]
    tol = values[0]*np.finfo(float).eps*max(len(gram),train.n_features)
    rank = int(np.sum(values>tol))
    if n_components>rank:
        raise ValueError(f'PCA training rank {rank} is smaller than requested {n_components}.')
    singular = np.sqrt(values[:n_components])
    temporal = u[:,:n_components]/singular
    train_scores = u[:,:n_components]*singular
    test_scores = np.zeros((test.n_observations,n_components))
    for train_block,test_block in zip(train.blocks(),test.blocks()):
        for start in range(0,train_block.shape[1],1024):
            x=np.asarray(train_block[:,start:start+1024],float)
            y=np.asarray(test_block[:,start:start+1024],float)
            mean=x.mean(0)
            weights=((x-mean)*scale).T@temporal
            test_scores+=((y-mean)*scale)@weights
    signs=np.sign(train_scores[np.argmax(np.abs(train_scores),axis=0),np.arange(n_components)])
    signs[signs==0]=1
    return dict(train=train_scores*signs,test=test_scores*signs,
                singular_values=np.sqrt(np.where(values>tol,values,0.)))


def temporal_singular_values(gram, n_features):
    """Exclude numerical null-space eigenvalues using the PCA rank tolerance."""
    from scipy.linalg import eigh
    values=eigh(gram,eigvals_only=True,check_finite=False)[::-1]
    tol=values[0]*np.finfo(float).eps*max(len(gram),n_features)
    return np.sqrt(np.where(values>tol,values,0.))


def _pca_artifact(child, fold):
    """Return a small PCA score sidecar or embedded future-fit scores."""
    for path in (child/f'pca_scores_{fold:03d}.npz',child/f'scores_{fold:03d}.npz'):
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:
                keys=[f'{p}_{m}_pca' for p in ('train','test') for m in ('ieeg','meg')]
                if all(key in saved.files for key in keys):
                    return {key:saved[key] for key in keys}
    return None


def plssvd_pca_comparison(runs_dir, perm=None, perm_type=None, n_components=None,
                          cache_dir=None, scratch_dir=None, absolute=True):
    """Held-out within-modality Pearson matrices, with matched baseline if selected.

    Rows are native PLSSVD components; columns are variance-ordered independent
    PCA components. No component matching or sign selection uses test data.
    Fold correlations are averaged within iteration, then across iterations.
    Default |r| avoids arbitrary model signs; signed mode orients each model's
    axes by its own largest absolute TRAIN score. It retains signed fold r too.
    Missing PCA scores require the original trial cache; PLSSVD is never refit.
    """
    from vs2_1_plssvd_eval_utils import load_plssvd_results, validation_run_name, load_trial_cache, _saved_scores
    from vs2_1_plssvd_diagnostics import _check_matching, _backfill_spectra
    from vs2_0_compare_models import correlation_matrix
    root=Path(runs_dir); name=validation_run_name(perm,perm_type)
    baseline=load_plssvd_results(root/'none',n_components=n_components)
    results={'Unpermuted':baseline}
    if name!='none':
        results['Permuted']=load_plssvd_results(root/name,n_components=n_components)
        ids=sorted(set(baseline['available_iterations'])&set(results['Permuted']['available_iterations']))
        if not ids:raise ValueError('No shared completed iterations for PLSSVD/PCA comparison.')
        _check_matching(baseline,results['Permuted'],ids)
    else:ids=baseline['available_iterations']
    k=baseline['n_components_evaluated']; folds=baseline['iteration_options']['repeats']
    trials=None; rows=[]
    for label,result in results.items():
        path=result['iterations_output_dir']
        for iteration in ids:
            child=path/f'iteration_{iteration:03d}'
            if any(_pca_artifact(child,f) is None for f in range(folds)):
                if cache_dir is None or not (Path(cache_dir)/'manifest.json').exists():
                    raise FileNotFoundError(
                        f'PCA scores are missing in {child}. Supply the original trial cache via cache_dir '
                        'to fit training-only PCA on the saved folds; existing PLSSVD fits are reused. '
                        'PCA cannot be recovered from PLSSVD scores or singular values alone.')
                if trials is None:trials=load_trial_cache(cache_dir)
                print(f'Recovering training-only PCA: {label}, iteration {iteration+1}; PLSSVD is unchanged.')
                _backfill_spectra(path,iteration,trials,scratch_dir,include_pca=True)
            for fold in range(folds):
                pca=_pca_artifact(child,fold)
                with _saved_scores(child,fold) as saved:
                    for modality in ('ieeg','meg'):
                        x=saved[f'test_{modality}'][:,:k].copy()
                        y=pca[f'test_{modality}_pca'][:,:k].copy()
                        if min(x.shape[1],y.shape[1])<k:raise ValueError('Saved PCA/PLSSVD scores have too few components.')
                        if not absolute:
                            for values,train in [(x,saved[f'train_{modality}'][:,:k]),
                                                 (y,pca[f'train_{modality}_pca'][:,:k])]:
                                signs=np.sign(train[np.argmax(np.abs(train),axis=0),np.arange(k)])
                                signs[signs==0]=1;values*=signs
                        r=correlation_matrix(x,y)
                        for i in range(k):
                            for j in range(k):
                                rows.append(dict(run=label,iteration=iteration,repeat=fold,modality=modality,
                                    pls_component=i+1,pca_component=j+1,pearson_r=r[i,j],
                                    value=abs(r[i,j]) if absolute else r[i,j]))
    fold_table=pd.DataFrame(rows)
    keys=['run','iteration','modality','pls_component','pca_component']
    iterations=fold_table.groupby(keys).value.agg(value='mean',valid_folds='count').reset_index()
    summary=iterations.groupby(['run','modality','pls_component','pca_component']).value.agg(
        mean='mean',std='std',count='count').reset_index()
    differences=pd.DataFrame();delta_summary=pd.DataFrame()
    if 'Permuted' in results:
        keys=['iteration','modality','pls_component','pca_component']
        a=iterations.query("run == 'Unpermuted'").set_index(keys).value
        b=iterations.query("run == 'Permuted'").set_index(keys).value
        differences=(b-a).rename('delta').reset_index()
        delta_summary=differences.groupby(keys[1:]).delta.agg(mean='mean',std='std',count='count').reset_index()
    print(f'PLSSVD/PCA: {len(ids)} {"matched " if name!="none" else ""}completed iterations, '
          f'{folds} folds, {k} components; test scores only.')
    return dict(fold_correlations=fold_table,iteration_correlations=iterations,summary=summary,
                differences=differences,delta_summary=delta_summary,iterations=ids,
                n_components=k,absolute=absolute,permutation=name)


def plot_plssvd_pca_comparison(data):
    """Annotated mean ± SD matrices; paired difference panels for a selected run."""
    from vs2_1_plssvd_eval_utils import _annotate_matrix
    k=data['n_components']; labels=[label for label in ('Unpermuted','Permuted') if label in set(data['summary'].run)]
    panels=labels+(['Permuted − unpermuted'] if len(labels)>1 else [])
    fig,axes=plt.subplots(2,len(panels),figsize=(6*len(panels),11),squeeze=False,layout='constrained')
    for row,modality in enumerate(('ieeg','meg')):
        for col,label in enumerate(panels):
            ax=axes[row,col];delta=col>=len(labels)
            table=(data['delta_summary'] if delta else data['summary'].query('run == @label'))
            table=table.query('modality == @modality')
            mean=table.pivot(index='pls_component',columns='pca_component',values='mean').to_numpy()
            sd=table.pivot(index='pls_component',columns='pca_component',values='std').to_numpy()
            if delta:
                limit=1 if data['absolute'] else 2
                vmin,vmax,cmap=-limit,limit,'RdBu_r'
            else:vmin,vmax,cmap=(0,1,'viridis') if data['absolute'] else (-1,1,'RdBu_r')
            im=ax.imshow(mean,vmin=vmin,vmax=vmax,cmap=cmap)
            if k<=20:_annotate_matrix(ax,mean,sd)
            ticks=np.arange(k) if k<=20 else np.arange(0,k,5)
            ax.set(title=f'{modality.upper()} · {label}',xlabel='Independent PCA component (training variance order)',
                   ylabel='PLSSVD component (native order)',xticks=ticks,xticklabels=ticks+1,
                   yticks=ticks,yticklabels=ticks+1)
            fig.colorbar(im,ax=ax,shrink=.7,label=('Δ ' if delta else '')+('Mean |Pearson r|' if data['absolute'] else 'Mean Pearson r'))
    statistic='|Pearson r|' if data['absolute'] else 'Pearson r (training-oriented signs)'
    fig.suptitle(f'Held-out PLSSVD versus independent PCA · {statistic}\n'
                 f'{data["permutation"]} · k={k} · {len(data["iterations"])} iterations · cells: mean ± SD of fold means\n'
                 'Each PCA is fitted on its own modality and training fold; no test-based component matching.')
    return fig


def fit_full_data_pca(cache_dir, n_components, scratch_dir=None, *, meg_kind='full_concatenated',
                      seed=2026, block_scaling='none'):
    """Fit a fixed PCA per modality to all trials using the selected MEG construction.

    Trial means are computed within each condition, preprocessing is estimated
    from all trials, then conditions are averaged equally. Feature blocks are
    accumulated into a temporal Gram matrix rather than a dense concatenation.
    Inputs are unpermuted. No PLSSVD fitting, fold evaluation or sidecar writes.
    """
    from vs2_1_plssvd_eval_utils import load_trial_cache, _temporary_fold
    from vs2_1_plssvd_evaluation  import selected_count
    from vs2_0_cov_models_utils import _gram, _spectrum
    trials=load_trial_cache(cache_dir)
    k=selected_count(n_components,n_components) if n_components is not None else None
    if k is None:raise ValueError('Select an explicit N_COMPONENTS for full-data PCA.')
    indices={m:{s.subject:{'train':[np.arange(len(a)) for a in s.data]}
                for s in getattr(trials,m)} for m in ('ieeg','meg')}
    scores={};singular_values={}
    with _temporary_fold(trials,meg_kind,indices,seed=seed,scratch_dir=scratch_dir,
                         condition_mode='average') as prepared:
        datasets=prepared[0]['train']
        for modality in ('ieeg','meg'):
            dataset=datasets[modality]
            from vs1_coverage_matching_utils import normalize_dataset
            normalize_dataset(dataset, block_scaling == 'equal_variance')
            u,d=_spectrum(_gram(dataset),dataset.n_features)
            if len(d)<k:raise ValueError(f'{modality}: full-data PCA rank {len(d)} is below {k}.')
            values=u[:,:k]*d[:k]
            signs=np.sign(values[np.argmax(abs(values),axis=0),np.arange(k)]);signs[signs==0]=1
            scores[modality]=values*signs
            singular_values[modality]=d
    return dict(scores=scores,singular_values=singular_values,times=trials.times,
                conditions=trials.conditions,n_components=k,meg_kind=meg_kind,block_scaling=block_scaling,seed=seed,
                subjects={m:[s.subject for s in getattr(trials,m)] for m in ('ieeg','meg')})


def correlate_full_pca(run_dir, pca, n_components=None, absolute=True, n_iterations=None):
    """Average correlations across iterations SEPARATELY for each fold/partition.

    PCA remains fixed. Every matrix contains all PLSSVD-by-PCA component pairs.
    Online moments keep memory independent of iteration count. Undefined
    correlations are excluded per cell; sample SD is NaN for fewer than 2 values.
    """
    from vs2_1_plssvd_eval_utils import _saved_scores
    from vs2_1_plssvd_evaluation import selected_count
    from vs2_1_plssvd_diagnostics import _audit
    from vs2_1_plssvd_postprocess import Moments
    from vs2_0_compare_models import correlation_matrix
    root=Path(run_dir);config=json.loads((root/'validation_options.json').read_text())
    if config.get('meg_kind') != pca.get('meg_kind', 'full_concatenated'):
        raise ValueError('PCA and PLSSVD MEG constructions differ.')
    k=selected_count(n_components,min(config['n_components'],pca['n_components']))
    ids=[i for i in range(config['n_iterations']) if (root/f'iteration_{i:03d}'/'COMPLETE.json').is_file()]
    if not ids:raise FileNotFoundError('No completed PLSSVD iterations are available.')
    from vs2_1_plssvd_results import selected_iterations
    ids = selected_iterations(ids, n_iterations)
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
                n_folds=config['repeats'],perm=config.get('perm'),perm_type=config.get('perm_type'),
                meg_kind=config['meg_kind'],block_scaling=config['block_scaling'])


def plot_full_pca_comparison(result):
    """Four rows (modality × partition), one column per fold; mean ± sample SD."""
    from vs2_1_plssvd_eval_utils import _annotate_matrix
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

