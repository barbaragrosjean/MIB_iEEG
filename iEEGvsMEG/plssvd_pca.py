"""Training-only, separate-modality PCA and held-out PLSSVD/PCA comparisons."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def independent_pca_scores(train, test, n_components, scale=1., train_gram=None):
    """Fit PCA on training time x feature data; transform held-out data.

    PCA is centered, not whitened. A modality-wide positive scale matches the
    PLSSVD preprocessing and cannot affect Pearson correlations. Work in the
    temporal Gram space and project feature chunks without storing PCA weights.
    Test data never enter the training eigendecomposition or sign convention.
    """
    from scipy.linalg import eigh
    from cov_models_utils import _gram
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
    from plssvd_eval_utils import load_plssvd_results, validation_run_name, load_trial_cache, _saved_scores
    from plssvd_diagnostics import _check_matching, _backfill_spectra
    from compare_models import correlation_matrix
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
    from plssvd_eval_utils import _annotate_matrix
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
