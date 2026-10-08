"""Matched-iteration diagnostics for one selected permutation versus baseline."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METRICS = {
    'mean_r': 'Paired Pearson r',
    'mean_paired_covariance': 'Paired score covariance',
    'crosscov_energy_fraction': 'Cross-covariance energy fraction',
    'ieeg_reconstruction_fraction': 'iEEG reconstruction fraction',
    'meg_reconstruction_fraction': 'MEG reconstruction fraction',
    'predict_ieeg_q2': 'Predict iEEG Q²',
    'predict_meg_q2': 'Predict MEG Q²',
}


def _audit(root, iteration):
    child = root/f'iteration_{iteration:03d}'
    path = child/'split_audit.csv.gz'
    if not path.exists():
        path = child/'split_audit.csv'
    columns = ['repeat','modality','subject','condition','trial_index','partition']
    return pd.read_csv(path, dtype={'subject':str})[columns].sort_values(columns).reset_index(drop=True)


def _check_matching(baseline, selected, ids):
    a, b = baseline['fit_options'], selected['fit_options']
    for key in ('seed','repeats','meg_kind','split_unit','block_scaling','ridge','condition_mode'):
        if a.get(key) != b.get(key):
            raise ValueError(f'Cannot pair runs: different {key}.')
    if baseline['n_components_evaluated'] != selected['n_components_evaluated']:
        raise ValueError('Choose a component count supported by both runs.')
    if not np.array_equal(baseline['times'],selected['times']) or baseline['conditions'] != selected['conditions']:
        raise ValueError('Cannot pair runs: different time or condition axes.')
    for iteration in ids:
        if not _audit(baseline['iterations_output_dir'],iteration).equals(
                _audit(selected['iterations_output_dir'],iteration)):
            raise ValueError(f'Cannot pair iteration {iteration}: trial assignments or cohorts differ.')
    # Fixed matching is stored once in new runs, per fold in legacy runs.
    for filename in ('matching.csv','pairing.json'):
        paths = [r['iterations_output_dir']/filename for r in (baseline,selected)]
        if all(p.exists() for p in paths) and paths[0].read_bytes() != paths[1].read_bytes():
            raise ValueError(f'Cannot pair runs: different {filename}.')


def _backfill_spectra(root, iteration, trials, scratch_dir=None, include_pca=False, fold_indices=None):
    """Recreate folds for spectra/optional training PCA; never refit PLSSVD."""
    from plssvd_eval_utils import ValidationOptions, _temporary_fold, _project, _saved_scores
    from cov_models_utils import _gram
    child = root/f'iteration_{iteration:03d}'
    config = json.loads((root/'validation_options.json').read_text())
    child_config = json.loads((child/'validation_options.json').read_text())
    options = ValidationOptions(**{k:v for k,v in child_config.items() if k in ValidationOptions.__dataclass_fields__})
    matching_seq, _, permutation_seq = np.random.SeedSequence(config['seed']).spawn(3)
    matching_seed = int(np.random.default_rng(matching_seq).integers(2**31-1))
    perm_seed = int(np.random.default_rng(permutation_seq.spawn(iteration+1)[iteration]).integers(2**31-1))
    audit = _audit(root,iteration)
    for modality in ('ieeg','meg'):
        if set(audit.loc[audit.modality==modality,'subject']) != {s.subject for s in getattr(trials,modality)}:
            raise ValueError('Trial cache cohort differs from the saved run.')
    saved_axes = root/'trial_axes.npz'
    if not saved_axes.exists(): saved_axes = child/'trial_axes.npz'
    with np.load(saved_axes) as axes:
        if not np.array_equal(axes['times'],trials.times) or tuple(axes['conditions']) != tuple(trials.conditions):
            raise ValueError('Trial cache axes differ from the saved run.')
    for fold in (range(config['repeats']) if fold_indices is None else fold_indices):
        sidecar=child/f'temporal_spectra_{fold:03d}.npz'
        pca_sidecar=child/f'pca_scores_{fold:03d}.npz'
        if sidecar.exists() and (not include_pca or pca_sidecar.exists()):continue
        indices={}
        for modality in ('ieeg','meg'):
            indices[modality]={}
            for subject in getattr(trials,modality):
                rows=audit[(audit.repeat==fold)&(audit.modality==modality)&(audit.subject==subject.subject)]
                indices[modality][subject.subject]={part:[
                    rows.loc[(rows.partition==part)&(rows.condition==condition),'trial_index'].to_numpy(int)
                    for condition in trials.conditions] for part in ('train','test')}
        spectra={};pca_scores={}
        with _temporary_fold(trials,config['meg_kind'],indices,matching_seed,scratch_dir,
                             condition_mode='average',permutation_options=options,permutation_seed=perm_seed) as prepared:
            datasets,_,_,_=prepared
            with np.load(child/f'model_{fold:03d}.npz') as saved_model:
                model = {key:saved_model[key] for key in ('ieeg_mean','meg_mean','ieeg_scale','meg_scale')}
                model['k_max'] = min(3,int(saved_model['k_max']))
                for modality in ('ieeg','meg'):
                    model[modality+'_weights'] = saved_model[modality+'_weights'][:,:model['k_max']].copy()
            with _saved_scores(child,fold) as saved_scores:
                for part in ('train','test'):
                    for modality in ('ieeg','meg'):
                        expected=saved_scores[f'{part}_{modality}'][:,:model['k_max']]
                        actual=_project(datasets[part][modality],model,modality)
                        if not np.allclose(actual,expected,rtol=1e-4,atol=1e-5*max(1.,np.max(np.abs(expected)))):
                            raise ValueError('Reconstructed scores differ from saved scores; check the original trial cache and settings.')
                for part in ('train','test'):
                    for modality in ('ieeg','meg'):
                        gram=_gram(datasets[part][modality])*float(model[modality+'_scale'])**2
                        from plssvd_pca import temporal_singular_values
                        spectra[f'{part}_{modality}_temporal_singular_values']=temporal_singular_values(
                            gram,datasets[part][modality].n_features)
                        if part == 'train' and include_pca:
                            from plssvd_pca import independent_pca_scores
                            pca=independent_pca_scores(datasets['train'][modality],datasets['test'][modality],
                                config['n_components'],model[modality+'_scale'],train_gram=gram)
                            pca_scores[f'train_{modality}_pca']=pca['train']
                            pca_scores[f'test_{modality}_pca']=pca['test']
        if not sidecar.exists():np.savez_compressed(sidecar,**spectra)
        if include_pca:np.savez_compressed(pca_sidecar,**pca_scores)


def _spectral_rows(root, ids, folds, k, full):
    from plssvd_eval_utils import _saved_scores
    rows=[]
    for iteration in ids:
        child=root/f'iteration_{iteration:03d}'
        for fold in range(folds):
            with _saved_scores(child,fold) as saved:
                sidecar=child/f'temporal_spectra_{fold:03d}.npz'
                extra={}
                if sidecar.exists():
                    with np.load(sidecar) as z:extra=dict(z)
                for part in ('train','test'):
                    for modality in ('ieeg','meg'):
                        key=f'{part}_{modality}_temporal_singular_values'
                        if full:
                            values=saved[key] if key in saved.files else extra[key]
                        else:
                            scores=saved[f'{part}_{modality}'][:,:k]
                            values=np.linalg.svd(scores-scores.mean(0),compute_uv=False)
                        energy=values**2
                        fraction=energy/energy.sum() if energy.sum()>0 else np.full_like(energy,np.nan)
                        effective=1/np.sum(fraction**2)
                        for rank,(value,weight) in enumerate(zip(values,fraction),1):
                            rows.append(dict(iteration=iteration,repeat=fold,partition=part,modality=modality,
                                rank=rank,singular_value=value,energy_fraction=weight,effective_rank=effective))
    return pd.DataFrame(rows)


def _has_full(root, ids, folds):
    from plssvd_eval_utils import _saved_scores
    for iteration in ids:
        for fold in range(folds):
            child=root/f'iteration_{iteration:03d}'
            sidecar=child/f'temporal_spectra_{fold:03d}.npz'
            with _saved_scores(child,fold) as saved:
                extra=[]
                if sidecar.exists():
                    with np.load(sidecar) as spectra:extra=spectra.files
                for part in ('train','test'):
                    for m in ('ieeg','meg'):
                        if f'{part}_{m}_temporal_singular_values' not in saved.files+extra:return False
    return True


def permutation_diagnostics(runs_dir, perm, perm_type, n_components=None, cache_dir=None, scratch_dir=None):
    """Compare matched iterations; full input spectra require saved spectra/cache.

    If unavailable, score-subspace spectra are explicitly labeled as a fallback
    and are used for BOTH runs. They cannot establish full input dimensionality.
    Fold means are computed first; SDs describe variation across iterations.
    """
    from plssvd_eval_utils import load_plssvd_results, validation_run_name, load_trial_cache
    name=validation_run_name(perm,perm_type)
    if name=='none':raise ValueError('Select PERM and PERM_TYPE to compare a permutation with baseline.')
    root=Path(runs_dir)
    baseline=load_plssvd_results(root/'none',n_components=n_components)
    selected=load_plssvd_results(root/name,n_components=n_components)
    ids=sorted(set(baseline['available_iterations'])&set(selected['available_iterations']))
    if not ids:raise ValueError('No completed iteration IDs are shared by the two runs.')
    _check_matching(baseline,selected,ids)
    print(f'Matched {len(ids)} iterations: baseline has {baseline["n_iterations_loaded"]}, '
          f'{name} has {selected["n_iterations_loaded"]}. Only shared IDs are used.')
    k=baseline['n_components_evaluated'];folds=baseline['iteration_options']['repeats']
    runs={'Unpermuted':root/'none','Permuted':root/name}
    full=all(_has_full(path,ids,folds) for path in runs.values())
    if not full and cache_dir is not None and (Path(cache_dir)/'manifest.json').exists():
        print('Recovering full input spectra from cached trials; no PLSSVD refitting.')
        trials=load_trial_cache(cache_dir)
        for path in runs.values():
            if not _has_full(path,ids,folds):
                for iteration in ids:_backfill_spectra(path,iteration,trials,scratch_dir)
        full=True
    note=('Full preprocessed input spectra; independent of selected PLS component count.' if full else
          f'Full input spectra unavailable. Showing spectra of the first {k} PLS score time courses only; '
          'these cannot establish full input dimensionality. Supply the original trial cache to recover full spectra.')
    print(note)
    metrics=[];components=[];spectra=[]
    for label,result in [('Unpermuted',baseline),('Permuted',selected)]:
        metrics.append(result['iteration_metrics'].query('iteration in @ids').assign(run=label))
        table=result['iteration_components'].query('iteration in @ids')
        components.append(table.groupby(['iteration','component'])[
            ['train_r','test_r','train_covariance','test_covariance']].mean().reset_index().assign(run=label))
        spectrum=_spectral_rows(runs[label],ids,folds,k,full)
        spectra.append(spectrum.groupby(['iteration','partition','modality','rank'])[
            ['singular_value','energy_fraction','effective_rank']].mean().reset_index().assign(run=label))
    metrics=pd.concat(metrics,ignore_index=True);components=pd.concat(components,ignore_index=True)
    spectra=pd.concat(spectra,ignore_index=True)
    spectra['source']='full_input' if full else 'selected_pls_scores'
    a=metrics.query("run == 'Unpermuted'").set_index(['iteration','partition'])
    b=metrics.query("run == 'Permuted'").set_index(['iteration','partition'])
    differences=(b[list(METRICS)]-a[list(METRICS)]).reset_index()
    delta_summary=differences.melt(id_vars=['iteration','partition'],var_name='metric',value_name='delta').groupby(
        ['partition','metric']).delta.agg(['count','mean','std','median','min','max']).reset_index()
    return dict(metrics=metrics,components=components,spectra=spectra,differences=differences,
                delta_summary=delta_summary,matched_iterations=ids,n_components=k,permutation=name,
                full_spectra=full,spectrum_note=note)


def _line_sd(ax, frame, x, y, label, color, style='-'):
    grouped=frame.groupby(x)[y]
    mean,sd=grouped.mean(),grouped.std()
    ax.plot(mean.index,mean.values,style,color=color,label=label)
    if sd.notna().any():ax.fill_between(mean.index,mean-sd,mean+sd,color=color,alpha=.15)


def plot_permutation_diagnostics(data):
    """Four figures: components, paired metrics, paired differences, spectra."""
    figures={};colors={'Unpermuted':'tab:blue','Permuted':'tab:orange'}
    suffix=f'{data["permutation"]} · k={data["n_components"]} · {len(data["matched_iterations"])} matched iterations'
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for row,part in enumerate(('train','test')):
        for col,metric in enumerate(('r','covariance')):
            ax=axes[row,col]
            for label,color in colors.items():
                _line_sd(ax,data['components'].query('run == @label'),'component',f'{part}_{metric}',label,color)
            ax.set(title=f'{part.capitalize()}: '+('Pearson r' if metric=='r' else 'Score covariance'),
                   xlabel='Native PLS component index',ylabel='Mean ± SD across iteration fold means')
            ax.axhline(0,color='gray',ls=':',lw=.7);ax.legend()
    fig.suptitle('Component-wise comparison · '+suffix);figures['components']=fig
    fig,axes=plt.subplots(2,4,figsize=(18,9),layout='constrained')
    for ax,(metric,title) in zip(axes.flat,METRICS.items()):
        frame=data['metrics'].query("partition == 'test'")
        a=frame.query("run == 'Unpermuted'").set_index('iteration')[metric]
        b=frame.query("run == 'Permuted'").set_index('iteration')[metric]
        for iteration in data['matched_iterations']:
            ax.plot([0,1],[a[iteration],b[iteration]],color='gray',alpha=.3,lw=.8)
        for x,series,color in [(0,a,'tab:blue'),(1,b,'tab:orange')]:
            ax.scatter(np.full(len(series),x),series,s=15,color=color,alpha=.6)
            ax.errorbar(x,series.mean(),yerr=series.std() if len(series)>1 else None,fmt='ks',capsize=5)
        ax.set(title=title,xticks=[0,1],xticklabels=['Unpermuted','Permuted'],ylabel='Held-out fold mean')
    axes.flat[-1].axis('off')
    fig.suptitle('Held-out metrics · connected dots = matched iterations; black = mean ± SD\n'+suffix)
    figures['metrics']=fig
    fig,axes=plt.subplots(2,4,figsize=(18,8),layout='constrained')
    for ax,(metric,title) in zip(axes.flat,METRICS.items()):
        frame=data['differences'].query("partition == 'test'")
        ax.plot(frame.iteration+1,frame[metric],'o-',ms=3,lw=.6)
        ax.axhline(0,color='black',ls=':')
        ax.set(title=title,xlabel='Iteration (1-based)',ylabel='Permuted − unpermuted')
    axes.flat[-1].axis('off');fig.suptitle('Matched held-out differences · '+suffix)
    figures['differences']=fig
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for row,modality in enumerate(('ieeg','meg')):
        for col,metric in enumerate(('singular_value','energy_fraction')):
            ax=axes[row,col]
            for label,color in colors.items():
                for part,style in [('train','--'),('test','-')]:
                    if 'spectrum_summaries' in data:
                        values=data['spectrum_summaries'].get((label,part,modality,metric))
                        if values is not None:
                            mean,sd=values;rank=np.arange(len(mean))+1
                            ax.plot(rank,mean,ls=style,color=color,label=f'{label} {part}')
                            if np.isfinite(sd).any():ax.fill_between(rank,mean-sd,mean+sd,color=color,alpha=.15)
                    else:
                        frame=data['spectra'].query('run == @label and modality == @modality and partition == @part')
                        _line_sd(ax,frame,'rank',metric,f'{label} {part}',color,style)
            ax.set(title=modality.upper(),xlabel='Temporal singular-value rank',
                   ylabel='Singular value' if metric=='singular_value' else 'Fraction of squared singular values')
            ax.set_yscale('symlog',linthresh=1e-8)
            if ax.lines:ax.legend(fontsize=8)
            else:ax.text(.5,.5,'Full spectra not prepared; run the evaluation job with --cache-dir.',
                         transform=ax.transAxes,ha='center',wrap=True)
    source=('Full input' if data['full_spectra'] else
            'Full spectra not prepared' if 'spectrum_summaries' in data else
            'Projected PLS scores ONLY (full input unavailable)')
    fig.suptitle(source+' temporal spectra · mean ± SD\n'+suffix);figures['spectra']=fig
    return figures
