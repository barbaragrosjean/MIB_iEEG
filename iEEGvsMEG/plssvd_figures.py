"""Plot-only readers. Never open fitted models, trial caches, or fitting functions.

Missing evaluation snapshots raise actionable errors; computation is always an
explicit invocation of plssvd_postprocess.py outside the notebook.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plssvd_postprocess import snapshot_path


def load_evaluation(run_dir, n_components):
    folder=snapshot_path(run_dir,n_components)
    meta=json.loads((folder/'manifest.json').read_text())
    with np.load(folder/'arrays.npz') as z:arrays=dict(z)
    tables={name:pd.read_csv(folder/(name+'.csv')) for name in
            ('fold_metrics','components','iteration_metrics','metric_summary','fold_stability','iteration_stability')}
    first=meta['iterations'][0]
    with np.load(folder/f'iteration_{first:03d}.npz') as z:
        primary={p:{m:z[f'{p}_{m}'] for m in ('ieeg','meg')} for p in ('train','test')}
        first_corr=z['test_correlation_absolute']
    print(f"Prepared evaluation: {len(meta['iterations'])}/{meta['n_requested']} iterations; "
          f"{n_components}/{meta['options']['n_components']} fitted components. No fitting or evaluation is run.")
    if not meta['pca_available']:print('PCA comparison is not prepared. Run the evaluation job with --cache-dir to recover missing PCA scores.')
    if not meta['spectra_available']:print('Full input spectra are not prepared. Run the evaluation job with --cache-dir to recover them.')
    return dict(folder=folder,metadata=meta,arrays=arrays,primary_scores=primary,
                first_test_correlation=first_corr,**tables)


def plot_first_iteration(data):
    """Reuse existing figure layouts with compact first-iteration metrics/scores."""
    from plssvd_eval_utils import plot_plssvd_validation
    meta=data['metadata'];first=meta['iterations'][0];k=meta['n_components']
    folds=data['fold_metrics'].query('iteration == @first').drop(columns='iteration')
    original=dict(validation_options=dict(meta['options'],n_components=k,schema_version=5),
                  fold_metrics=folds,fold_stability=data['fold_stability'],primary_scores=data['primary_scores'],
                  times=data['arrays']['times'],null_tests=pd.DataFrame())
    plot_plssvd_validation(original)
    return None


def plot_correlation_matrix(mean, sd=None, title='Held-out mean |Pearson r|'):
    from plssvd_eval_utils import _annotate_matrix
    fig,ax=plt.subplots(figsize=(8,7),layout='constrained');k=len(mean)
    im=ax.imshow(mean,vmin=0,vmax=1,cmap='viridis')
    if k<=20:_annotate_matrix(ax,mean,sd)
    ax.set(title=title,xlabel='MEG component',ylabel='iEEG component',
           xticks=np.arange(k),xticklabels=np.arange(k)+1,yticks=np.arange(k),yticklabels=np.arange(k)+1)
    fig.colorbar(im,ax=ax);return fig


def plot_evaluation_summary(data):
    from plssvd_eval_utils import _annotate_matrix
    arrays=data['arrays'];meta=data['metadata'];k=meta['n_components'];n=len(meta['iterations'])
    fig=plt.figure(figsize=(20,12),layout='constrained')
    grid=fig.add_gridspec(3,2,width_ratios=[1,1.8],height_ratios=[1,1,.8])
    covgrid=grid[0,0].subgridspec(1,2)
    limit=max(np.nanmax(abs(arrays[p+'_covariance_mean'])) for p in ('train','test')) or 1.
    for col,part in enumerate(('train','test')):
        ax=fig.add_subplot(covgrid[0,col]);mean=arrays[part+'_covariance_mean'];sd=arrays[part+'_covariance_std']
        im=ax.imshow(mean,cmap='RdBu_r',vmin=-limit,vmax=limit)
        if k<=20:_annotate_matrix(ax,mean,sd,fmt='.1f')
        ax.set(title=f'Fold 1 {part}: mean ± SD',xlabel='MEG component',ylabel='iEEG component')
        ax.set(xticks=np.arange(k),xticklabels=np.arange(k)+1,yticks=np.arange(k),yticklabels=np.arange(k)+1)
        fig.colorbar(im,ax=ax,shrink=.7)
    ax=fig.add_subplot(grid[1:,0]);mean=arrays['test_correlation_absolute_mean'];sd=arrays['test_correlation_absolute_std']
    im=ax.imshow(mean,vmin=0,vmax=1,cmap='viridis')
    if k<=20:_annotate_matrix(ax,mean,sd)
    ax.set(title='Held-out |Pearson r|: mean ± SD of fold means',xlabel='MEG component',ylabel='iEEG component')
    ax.set(xticks=np.arange(k),xticklabels=np.arange(k)+1,yticks=np.arange(k),yticklabels=np.arange(k)+1)
    fig.colorbar(im,ax=ax,shrink=.7)
    tcgrid=grid[:2,1].subgridspec(min(k,3),1)
    for pc in range(min(k,3)):
        ax=fig.add_subplot(tcgrid[pc,0])
        for modality,color in [('ieeg','navy'),('meg','darkorange')]:
            for part,style in [('train','--'),('test','-')]:
                key=f'{part}_{modality}_normalized';mean=arrays[key+'_mean'][:,pc];sd=arrays[key+'_std'][:,pc]
                ax.plot(arrays['times'],mean,color=color,ls=style,label=f'{modality} {part}')
                if n>1:ax.fill_between(arrays['times'],mean-sd,mean+sd,color=color,alpha=.12)
        ax.set(title=f'Component {pc+1}: fold 1',xlabel='Time (s)',ylabel='Score / training SD')
        if pc==0:ax.legend(ncol=4,fontsize=8)
    mg=grid[2,1].subgridspec(1,3)
    for col,(metric,title) in enumerate([('crosscov_energy_fraction','Cross-covariance energy retained'),
        ('mean_paired_covariance','Paired score covariance'),('mean_r','Paired Pearson r')]):
        ax=fig.add_subplot(mg[0,col])
        for part in ('train','test'):
            g=data['fold_metrics'].query('partition == @part').groupby('repeat')[metric]
            ax.errorbar(g.mean().index+1,g.mean(),yerr=g.std() if n>1 else None,marker='o',capsize=3,label=part)
        ax.set(title=title,xlabel='Fold',ylabel='Mean ± SD');ax.legend()
    fig.suptitle(f'{n}/{meta["n_requested"]} evaluated iterations · k={k} · descriptive SD, not confidence intervals')
    return fig


def plot_prepared_permutations(runs_dir, n_components):
    from plssvd_eval_utils import PERMUTATION_TYPES, validation_run_name
    def read(name):
        try:folder=snapshot_path(Path(runs_dir)/name,n_components)
        except FileNotFoundError:return None
        table=pd.read_csv(folder/'iteration_metrics.csv')
        return table.query("partition == 'test'").mean_r.to_numpy(),json.loads((folder/'manifest.json').read_text())
    baseline=read('none');fig,axes=plt.subplots(3,len(PERMUTATION_TYPES),figsize=(4.5*len(PERMUTATION_TYPES),11),layout='constrained',sharex=True)
    for row,perm in enumerate(('ieeg','meg','both')):
        for col,kind in enumerate(PERMUTATION_TYPES):
            ax=axes[row,col];values=read(validation_run_name(perm,kind))
            if baseline is not None and values is not None:
                for key in ('seed','repeats','meg_kind','split_unit','block_scaling','ridge','condition_mode'):
                    if baseline[1]['options'].get(key)!=values[1]['options'].get(key):
                        raise ValueError(f'Incompatible prepared permutation overview: {key}.')
            if baseline is not None:ax.hist(baseline[0],bins=np.linspace(-1,1,31),density=True,histtype='step',lw=2,label=f'Unpermuted (n={len(baseline[0])})')
            if values is not None:ax.hist(values[0],bins=np.linspace(-1,1,31),density=True,alpha=.6,label=f'Permuted (n={len(values[0])})')
            else:ax.text(.5,.5,'Evaluation not prepared',transform=ax.transAxes,ha='center')
            ax.set(title=f'{perm.upper()} · {kind}',xlabel='Iteration mean held-out Pearson r',ylabel='Density',xlim=(-1,1))
            if baseline is not None or values is not None:ax.legend(fontsize=8)
    fig.suptitle(f'Prepared permutation evaluations · k={n_components} · all available iterations (unpaired overview)')
    return fig


def load_comparison(runs_dir, name, n_components):
    root=Path(runs_dir);folder=root/name/'evaluation'/f'k_{n_components:03d}'/'comparison'
    if not (folder/'CURRENT.json').exists():
        raise FileNotFoundError('Matched comparison not prepared. Run plssvd_postprocess.py for this permutation and component count.')
    path=folder/json.loads((folder/'CURRENT.json').read_text())['snapshot']
    meta=json.loads((path/'manifest.json').read_text())
    signatures=[json.loads((snapshot_path(root/n,n_components)/'manifest.json').read_text())['signature'] for n in ('none',name)]
    if signatures!=meta['source_signatures']:
        raise ValueError('Matched comparison is stale. Rerun plssvd_postprocess.py; cached fold evaluations will be reused.')
    with np.load(path/'arrays.npz') as z:arrays=dict(z)
    tables={n:pd.read_csv(path/(n+'.csv')) for n in ('metrics','components','differences','delta_summary')}
    print(f'Prepared matched comparison: {len(meta["iterations"])} shared iterations.')
    return dict(folder=path,metadata=meta,arrays=arrays,**tables)


def plot_prepared_diagnostics(data):
    from plssvd_diagnostics import plot_permutation_diagnostics
    meta=data['metadata'];components=data['components']
    wide=components.pivot(index=['run','iteration','component'],columns='partition',values=['r','covariance'])
    wide.columns=[part+'_'+metric for metric,part in wide.columns]
    summary={}
    if meta['spectra_available']:
        for label in ('Unpermuted','Permuted'):
            for p in ('train','test'):
                for m in ('ieeg','meg'):
                    for metric,key in [('singular_value',f'spectrum_{p}_{m}'),('energy_fraction',f'spectrum_energy_{p}_{m}')]:
                        summary[label,p,m,metric]=(data['arrays'][f'{label}_{key}_mean'],data['arrays'][f'{label}_{key}_std'])
    return plot_permutation_diagnostics(dict(components=wide.reset_index(),metrics=data['metrics'],
        differences=data['differences'],matched_iterations=meta['iterations'],n_components=meta['n_components'],
        permutation=meta['permutation'],full_spectra=meta['spectra_available'],spectrum_summaries=summary))


def plot_prepared_pca(data, absolute=True, comparison=False):
    from plssvd_pca import plot_plssvd_pca_comparison
    meta=data['metadata']
    if not meta['pca_available']:
        raise FileNotFoundError('PCA evaluation not prepared. Run plssvd_postprocess.py with --cache-dir outside the notebook.')
    stat='absolute' if absolute else 'signed';rows=[];deltas=[]
    labels=['Unpermuted','Permuted','Difference'] if comparison else ['Unpermuted']
    for label in labels:
        for modality in ('ieeg','meg'):
            key=f'pca_{modality}_{stat}';prefix=label+'_' if comparison else ''
            mean=data['arrays'][prefix+key+'_mean'];sd=data['arrays'][prefix+key+'_std']
            for (i,j),value in np.ndenumerate(mean):
                row=dict(run=label,modality=modality,pls_component=i+1,pca_component=j+1,mean=value,std=sd[i,j])
                (deltas if label=='Difference' else rows).append(row)
    return plot_plssvd_pca_comparison(dict(summary=pd.DataFrame(rows),delta_summary=pd.DataFrame(deltas),
        n_components=meta['n_components'],iterations=meta['iterations'],absolute=absolute,
        permutation=meta.get('permutation','none')))


def plot_phase_null_comparison(runs_dir, n_components, modes=('meg','ieeg','both')):
    """Matched-iteration held-out distributions from prepared snapshots only.

    Each value averages the folds of one iteration. Baseline spread reflects
    repeated trial splits; surrogate spread also varies phase. These overlapping
    iterations are not independent observations; no inferential p-value is shown.
    """
    modes = tuple(modes)
    if not modes or any(m not in ('meg','ieeg','both') for m in modes):
        raise ValueError('modes must contain meg, ieeg or both.')
    panels = [('mean_paired_covariance','Mean paired covariance'),
              ('mean_r','Mean signed Pearson correlation'),
              ('ieeg_reconstruction_fraction','iEEG reconstruction fraction'),
              ('meg_reconstruction_fraction','MEG reconstruction fraction')]
    fig, axes = plt.subplots(len(modes), len(panels), figsize=(18,3.4*len(modes)), squeeze=False, layout='constrained')
    for row, mode in enumerate(modes):
        try:
            data = load_comparison(runs_dir, mode+'__phase', n_components)
        except FileNotFoundError:
            for ax, (_, title) in zip(axes[row], panels):
                ax.text(.5,.5,'Phase comparison not prepared',ha='center',transform=ax.transAxes)
                ax.set_title(f'{mode}: {title}')
            continue
        frame = data['metrics'].query("partition == 'test'")
        for ax, (metric, title) in zip(axes[row], panels):
            values = {label: frame.loc[frame.run == label, metric].dropna().to_numpy()
                      for label in ('Unpermuted','Permuted')}
            values = {label: a[np.isfinite(a)] for label,a in values.items()}
            combined = np.concatenate(list(values.values()))
            if not len(combined):
                ax.text(.5,.5,'No finite values',ha='center',transform=ax.transAxes)
                continue
            bins = np.histogram_bin_edges(combined, bins=min(25,max(5,int(np.sqrt(len(combined))))))
            for label, color in [('Unpermuted','tab:blue'),('Permuted','tab:orange')]:
                a = values[label]
                if len(a):
                    ax.hist(a,bins=bins,histtype='step',linewidth=2,color=color,
                            label=f'{"Baseline" if label == "Unpermuted" else "Phase null"} (n={len(a)})')
                    ax.axvline(np.median(a),color=color,linestyle=':',alpha=.7)
            ax.set(title=f'{mode.upper()} randomized: {title}',xlabel='Held-out fold mean per iteration',ylabel='Iterations')
            ax.legend(fontsize=8)
    fig.suptitle(f'Coherent phase surrogates vs baseline · k={n_components}\nMatched trial splits; dotted lines: medians; descriptive distributions, not independent replicates')
    return fig
