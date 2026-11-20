"""Result views for one saved match or equal-weight random-assignment summaries.

Signal arrays are never averaged before fitting. Internal folds/repetitions are
averaged within assignment before assignment means and sample SD are calculated.
All exported metric columns and display arrays remain available to the caller.
"""
from pathlib import Path
from dataclasses import dataclass
import json
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from vs0_run_pairings import pairing_context, assignment_config

KEYS = {'model','model_a','model_b','modality','dataset','dataset_A','dataset_B',
        'reference_dataset','comparison','manipulation','domain','space','partition',
        'direction','complexity','metric','run','level','roi','ieeg_roi','meg_roi',
        'condition','component','component_a','component_b','k','dimension','n_components',
        'n_clusters','n_sources','n_channels','channel_count','source_count','n_selected',
        'subject','ieeg_subject','meg_subject','coverage_row','feature','distance_mm','method'}
IGNORE = {'iteration','repeat','fold','association','pairing_iteration','iteration_a',
          'iteration_b','repeat_other','reference_iteration','status','reason','path'}


def alignment(reference, scores):
    """Training/fit-score Hungarian correspondence; no test/spatial optimization."""
    x, y = reference-reference.mean(0), scores-scores.mean(0)
    denominator = np.linalg.norm(x,axis=0)[:,None]*np.linalg.norm(y,axis=0)[None,:]
    r = np.divide(x.T@y,denominator,out=np.zeros((x.shape[1],y.shape[1])),where=denominator>0)
    a,b = linear_sum_assignment(-abs(r))
    if len(a) != reference.shape[1] or reference.shape[1] != scores.shape[1]:
        raise ValueError('Component counts differ across subject assignments.')
    order=b[np.argsort(a)]; signs=np.sign(r[np.arange(len(order)),order]);signs[signs==0]=1
    return order,signs


def aligned_arrays(reference_scores, scores, **arrays):
    order,signs=alignment(reference_scores,scores)
    return {name: value[:,order]*signs for name,value in dict(scores=scores,**arrays).items()}


def publish_arrays(folder, arrays, *, times=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'display_arrays.npz',**arrays)
    if times is not None:np.save(folder/'display_times.npy',times)


def numeric_summary(frames):
    """One contribution per assignment/key. Label columns are never averaged."""
    if not frames:raise ValueError('No assignment results supplied.')
    columns=set(frames[0])
    if any(set(f)!=columns for f in frames):raise ValueError('Metric schemas differ across assignments.')
    expanded=[]
    for frame in frames:
        frame=frame.copy()
        for column in list(frame):
            if frame[column].dtype.kind not in ('O','U','S') and str(frame[column].dtype) not in ('str','string'):
                continue
            nonmissing=frame[column].dropna()
            if nonmissing.empty or not all(isinstance(v,str) and v.startswith('[') for v in nonmissing):
                continue
            try:
                decoded=[json.loads(v) if isinstance(v,str) else [] for v in frame[column]]
                width=max(map(len,decoded),default=0)
                for j in range(width):frame[f'{column}_{j+1}']=[float(v[j]) if j<len(v) else np.nan for v in decoded]
                frame=frame.drop(columns=column)
            except (ValueError,TypeError):
                continue
        expanded.append(frame)
    # Pad ragged vector fields so each column has a consistent metric meaning.
    allcolumns=list(dict.fromkeys(c for frame in expanded for c in frame))
    frames=[frame.reindex(columns=allcolumns) for frame in expanded]
    first=frames[0]
    numeric=[c for c in first if pd.api.types.is_numeric_dtype(first[c]) or pd.api.types.is_bool_dtype(first[c])]
    keys=[c for c in first if (c in KEYS and not (c=='metric' and c in numeric)) or (c not in IGNORE and c not in numeric)]
    metrics=[c for c in numeric if c not in set(keys)|IGNORE]
    if not metrics:return dict(raw=frames,mean=first.iloc[:0].copy(),std=first.iloc[:0].copy(),count=first.iloc[:0].copy(),keys=keys,metrics=[])
    reduced=[]
    for frame in frames:
        values=frame[keys+metrics].copy()
        for c in metrics:values[c]=pd.to_numeric(values[c],errors='raise').astype(float).replace([np.inf,-np.inf],np.nan)
        reduced.append(values.groupby(keys,dropna=False)[metrics].mean() if keys else pd.DataFrame([values[metrics].mean()]))
    # Union of keys is explicit; undefined/missing cells have counts < N, never zero-imputed.
    stacked=pd.concat(reduced,keys=range(len(reduced)),names=['assignment'])
    levels=list(range(1,stacked.index.nlevels))
    grouped=stacked.groupby(level=levels,dropna=False)
    return dict(raw=frames,mean=grouped.mean().reset_index(drop=not bool(keys)),
                std=grouped.std(ddof=1).reset_index(drop=not bool(keys)),
                count=grouped.count().reset_index(drop=not bool(keys)),keys=keys,metrics=metrics)


@dataclass
class ResultView:
    view: str
    assignment_ids: list
    tables: dict
    arrays: dict
    times: np.ndarray = None
    metadata: dict = None


def aggregate_results(results, view, n_pairings):
    ids=[-1] if view=='matched' else list(range(n_pairings)) if view=='full' else None
    if ids is None:raise ValueError('PAIRING_VIEW must be matched or full.')
    if -1 not in results:raise FileNotFoundError('The saved matched assignment (-1) is required first.')
    missing=set(ids)-set(results)
    if missing:raise FileNotFoundError(f'Missing completed subject assignments: {sorted(missing)}')
    table_names=set(results[ids[0]].get('tables',{}));array_names=set(results[ids[0]].get('arrays',{}))
    for i in ids:
        if set(results[i].get('tables',{}))!=table_names or set(results[i].get('arrays',{}))!=array_names:
            raise ValueError(f'Result schema differs for assignment {i}.')
    tables={name:numeric_summary([results[i]['tables'][name] for i in ids]) for name in sorted(table_names)}
    arrays={}
    for name in sorted(array_names):
        values=[np.asarray(results[i]['arrays'][name],float) for i in ids]
        if any(v.shape!=values[0].shape for v in values):raise ValueError(f'Array shape differs: {name}')
        stack=np.stack(values)
        stack=np.where(np.isfinite(stack),stack,np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            mean=np.nanmean(stack,axis=0);std=np.nanstd(stack,axis=0,ddof=1) if len(ids)>1 else np.full_like(mean,np.nan)
        arrays[name]=dict(mean=mean,std=std,count=np.isfinite(stack).sum(0))
    times=results[-1].get('times')
    if times is not None and any(not np.array_equal(times,results[i].get('times')) for i in ids):
        raise ValueError('Time axes differ across assignments.')
    return ResultView(view,ids,tables,arrays,times,dict(sd_unit='subject assignment',matched_excluded_from_full=True))


def read_folder(folder, *, table_names=None):
    folder=Path(folder)
    if not any((folder/m).exists() for m in ('RUN_COMPLETE.json','COMPLETE.json','manifest.json','selection.json','WEIGHT_SUMMARY_COMPLETE.json')):
        raise FileNotFoundError(f'Analysis is not complete: {folder}')
    tables={}
    paths=[folder/f'{n}.csv' for n in table_names] if table_names is not None else sorted(folder.glob('*.csv'))
    for path in paths:
        if any(x in path.stem for x in ('metadata','features','matching','assignments','participants','source_mapping','splits','coordinate_sources','trial_counts','component_matching')):continue
        try:tables[path.stem]=pd.read_csv(path)
        except pd.errors.EmptyDataError:continue
    arrays={}
    if (folder/'display_arrays.npz').exists():
        with np.load(folder/'display_arrays.npz') as saved:arrays=dict(saved)
    times=np.load(folder/'display_times.npy') if (folder/'display_times.npy').exists() else None
    return dict(tables=tables,arrays=arrays,times=times)


def config_view(config, analysis, view, n_pairings=100):
    from vs1_run_coverage import analysis_directory
    results={}
    for iteration in [-1,*(range(n_pairings) if view=='full' else [])]:
        with pairing_context(iteration,config.get('pairing_seed',config.get('seed',2026))):
            effective=assignment_config(config,iteration)
            results[iteration]=read_folder(analysis_directory(effective,analysis))
    return aggregate_results(results,view,n_pairings)


def path_view(path, view, n_pairings=100, loader=read_folder):
    results={i:loader(Path(assignment_config(str(path),i))) for i in [-1,*(range(n_pairings) if view=='full' else [])]}
    return aggregate_results(results,view,n_pairings)


def show(result, *, max_components=5, tables=True, arrays=True):
    """Every scalar metric is displayed; vector/matrix displays retain mean and SD."""
    import matplotlib.pyplot as plt
    from IPython.display import display
    unit='saved match (-1)' if result.view=='matched' else f'{len(result.assignment_ids)} random assignments; mean ± sample SD'
    print(unit)
    if tables:
        for name,summary in result.tables.items():
            if not summary['metrics']:continue
            print(name)
            report=summary['mean'].copy()
            if result.view=='full':
                for c in summary['metrics']:
                    report[c+'_std']=summary['std'][c];report[c+'_n']=summary['count'][c]
            display(report)
            x=next((c for c in summary['keys'] if c in ('k','component','n_sources','n_channels','source_count','n_selected','complexity')),None)
            for metric in summary['metrics']:
                if x is None:continue  # The complete table above still contains every metric.
                groups=[k for k in summary['keys'] if k!=x]
                mean=summary['mean'];sd=summary['std']
                indices=mean.groupby(groups,dropna=False).indices if groups else {metric:np.arange(len(mean))}
                if len(indices)>24 or metric.startswith('angles_deg_'):
                    continue  # Detailed metrics remain in the complete mean/SD/count table.
                fig,ax=plt.subplots(figsize=(10,3.5))
                for label,rows in indices.items():
                    frame=mean.iloc[rows];order=np.argsort(frame[x].to_numpy());rows=np.asarray(rows)[order];frame=mean.iloc[rows]
                    ax.errorbar(frame[x],frame[metric],yerr=sd.iloc[rows][metric] if result.view=='full' else None,label=str(label),capsize=2)
                ax.set(xlabel=x,ylabel=metric,title=f'{name}: {unit}');ax.legend(fontsize=6);fig.tight_layout();plt.show()
    if arrays:
        for name,summary in result.arrays.items():
            mean,sd=summary['mean'],summary['std']
            if mean.ndim==2 and ('score' in name or 'normalized' in name):
                x=result.times if result.times is not None and len(result.times)==len(mean) else np.arange(len(mean))
                for component in range(min(max_components,mean.shape[1])):
                    fig,ax=plt.subplots(figsize=(10,3));ax.plot(x,mean[:,component])
                    if result.view=='full':ax.fill_between(x,mean[:,component]-sd[:,component],mean[:,component]+sd[:,component],alpha=.2)
                    ax.set(title=f'{name}, aligned component {component+1}: {unit}',xlabel='Time / observation',ylabel='Score');plt.show()
            elif mean.ndim>=2:
                for index in np.ndindex(mean.shape[:-2]):
                    m,s=mean[index],sd[index]
                    fig,axes=plt.subplots(1,2 if result.view=='full' else 1,figsize=(11,4),squeeze=False)
                    for ax,values,label in zip(axes.flat,[m,s],['Mean' if result.view=='full' else 'Matched','SD']):
                        image=ax.imshow(values,aspect='auto',cmap='viridis');ax.set(title=f'{name} {index}: {label}');fig.colorbar(image,ax=ax)
                    plt.show()
            elif mean.ndim==0:
                print(name)
                display(pd.DataFrame(dict(mean=[float(mean)],std=[float(sd)],count=[int(summary['count'])])))
            elif mean.ndim==1:
                fig,ax=plt.subplots();ax.errorbar(np.arange(1,len(mean)+1),mean,yerr=sd if result.view=='full' else None,capsize=2);ax.set(title=f'{name}: {unit}');plt.show()


def safe_load(loader, *args, **kwargs):
    """Report required computation; never substitute assignment zero or partial data."""
    try:return loader(*args,**kwargs)
    except FileNotFoundError as exc:
        print(exc);print('Run the printed terminal command, then reload this cell.');return None


def matched_analysis_directory(config, analysis):
    from vs1_run_coverage import analysis_directory
    with pairing_context(-1,config.get('pairing_seed',config.get('seed',2026))):
        return analysis_directory(assignment_config(config,-1),analysis)


def export_pca_metrics(config, folder, pca):
    from vs1_coverage_matching_utils import correlate_timecourses,correlate_weights,compare_pca_to_ieeg
    import vs0_load_and_config as shared
    folder=Path(folder);baseline=matched_analysis_directory(config,'pca');arrays={};aligned_all={}
    comparison,pairs=compare_pca_to_ieeg(pca,n_components=range(1,config['n_components']+1),plot=False)
    comparison.to_csv(folder/'pca_comparison.csv',index=False);pairs.to_csv(folder/'pca_component_pairs.csv',index=False)
    spectra=[]
    for name,result in pca.items():
        spectra.extend(dict(dataset=name,component=i+1,explained_variance=float(v),explained_variance_ratio=float(r))
                       for i,(v,r) in enumerate(zip(result.explained_variance,result.explained_variance_ratio)))
        if shared.subject_pairing_iteration()==-1:reference=result.scores
        else:
            with np.load(baseline/f'{name}.npz') as saved:reference=saved['scores']
        aligned=aligned_arrays(reference,result.scores,weights=result.weights)
        aligned_all[name]=aligned
        arrays[name+'/aligned_scores']=aligned['scores']
        slots=result.dataset.electrode_to_feature
        arrays[name+'/aligned_coverage_weights']=aligned['weights'][slots]
        arrays[name+'/absolute_coverage_weights']=abs(aligned['weights'][slots])
    from vs2_0_compare_models import correlation_matrix
    for name,result in pca.items():
        if name=='iEEG':continue
        arrays[name+'/time_correlations']=correlation_matrix(aligned_all[name]['scores'],aligned_all['iEEG']['scores'])
        arrays[name+'/weight_correlations']=correlation_matrix(arrays[name+'/aligned_coverage_weights'],arrays['iEEG/aligned_coverage_weights'])
    pd.DataFrame(spectra).to_csv(folder/'pca_spectra.csv',index=False)
    publish_arrays(folder,arrays,times=pca['iEEG'].dataset.source_data['times'])


def export_covariance_metrics(config, folder, models):
    from vs2_0_cov_models_utils import compare_cov_models
    import vs0_load_and_config as shared
    folder=Path(folder);baseline=matched_analysis_directory(config,'models');arrays={}
    comparison,pairs,reference=compare_cov_models(models,n_components=range(1,config['n_components']+1),plot=False)
    for name,table in [('model_comparison',comparison),('component_pairs',pairs),('reference_pairs',reference)]:table.to_csv(folder/f'{name}.csv',index=False)
    for name,model in models.items():
        x,y=model.ieeg_scores,model.meg_scores
        if shared.subject_pairing_iteration()==-1:rx,ry=x,y
        else:
            with np.load(baseline/f'{name}.npz') as saved:rx,ry=saved['ieeg_scores'],saved['meg_scores']
        if name=='separate_pca':
            x=aligned_arrays(rx,x)['scores'];y=aligned_arrays(ry,y)['scores']
        else:
            a,b=alignment(np.vstack([rx,ry]),np.vstack([x,y]));x,y=x[:,a]*b,y[:,a]*b
        arrays[name+'/aligned_ieeg_scores']=x;arrays[name+'/aligned_meg_scores']=y
        for modality,values in [('ieeg',x),('meg',y)]:
            sd=values.std(axis=0)
            arrays[name+'/normalized_'+modality+'_scores']=(values-values.mean(axis=0))/np.where(sd>0,sd,1.)
        x,y=x-x.mean(0),y-y.mean(0)
        arrays[name+'/ieeg_covariance']=x.T@x/(len(x)-1)
        arrays[name+'/meg_covariance']=y.T@y/(len(y)-1)
        arrays[name+'/signed_cross_covariance']=x.T@y/(len(x)-1)
        from vs2_0_compare_models import correlation_matrix
        arrays[name+'/signed_correlation']=correlation_matrix(x,y)
        arrays[name+'/score_variance_ieeg']=np.var(x,axis=0,ddof=1)
        arrays[name+'/score_variance_meg']=np.var(y,axis=0,ddof=1)
        arrays[name+'/twice_absolute_crosscovariance']=2*abs(np.diag(x.T@y/(len(x)-1)))
    publish_arrays(folder,arrays,times=next(iter(models.values())).times)


def evaluation_view(run_dir, k, view, n_pairings=100, n_iterations=None, **settings):
    from vs2_1_plssvd_results import load_summary
    def loader(folder):
        data=load_summary(folder,k,n_iterations,**settings)
        from vs2_1_plssvd_postprocess import snapshot_path
        snapshot=snapshot_path(folder,k)
        with np.load(snapshot/'arrays.npz') as saved:
            arrays={name[:-5]:saved[name] for name in saved.files if name.endswith('_mean')}
        arrays.update({name[:-5]:value for name,value in data['arrays'].items() if name.endswith('_mean')})
        tables={'fold_metrics':data['fold_metrics']}
        for name in ('components','iteration_metrics','metric_summary'):
            path=snapshot/f'{name}.csv'
            if path.exists():
                table=pd.read_csv(path)
                if n_iterations is not None and 'iteration' in table:table=table[table.iteration.isin(data['metadata']['iterations'])]
                tables[name]=table
        return dict(tables=tables,arrays=arrays,times=data['arrays']['times'])
    ids=[-1,*(range(n_pairings) if view=='full' else [])]
    results={i:loader(Path(assignment_config(str(run_dir),i))) for i in ids}
    reference=results[-1]['arrays']
    required=['train_ieeg_normalized','train_meg_normalized']
    if all(key in reference for key in required):
        ref=np.vstack([reference[key] for key in required])
        for i in ids:
            arrays=results[i]['arrays']
            order,signs=alignment(ref,np.vstack([arrays[key] for key in required]))
            for key,value in list(arrays.items()):
                if key in [f'{part}_{m}{suffix}' for part in ('train','test') for m in ('ieeg','meg') for suffix in ('','_normalized')]:
                    arrays[key]=value[:,order]*signs
                elif ('covariance' in key or 'correlation' in key) and value.shape==(k,k):
                    arrays[key]=value[np.ix_(order,order)]
                    if 'absolute' not in key:arrays[key]=arrays[key]*signs[:,None]*signs[None,:]
    return aggregate_results(results,view,n_pairings)


def pca_comparison_view(run_dir,k,view,n_pairings=100,**settings):
    from vs2_1_plssvd_fit_pca import load_comparison
    def loader(folder):
        data=load_comparison(folder,k,**settings)
        return dict(tables={},arrays={'full_data_pca_correlations':data['mean']})
    return path_view(run_dir,view,n_pairings,loader)


def permutation_view(runs_dir,name,k,view,n_pairings=100):
    from vs2_1_plssvd_figures import load_comparison
    def loader(folder):
        data=load_comparison(folder,name,k)
        # Every published numerical diagnostic, including spectra and PCA comparisons.
        arrays={n[:-5]:v for n,v in data.get('arrays',{}).items() if n.endswith('_mean')}
        return dict(tables={n:v for n,v in data.items() if isinstance(v,pd.DataFrame)},arrays=arrays)
    return path_view(runs_dir,view,n_pairings,loader)


def selection_view(runs_dir,k,perm,perm_type,view,n_pairings=100):
    from vs2_1_plssvd_select_components import load_selection
    def loader(folder):
        data=load_selection(folder,k,perm,perm_type)
        tables={n:v for n,v in data.items() if isinstance(v,pd.DataFrame)}
        scalar={n:v for n,v in data['summary'].items() if isinstance(v,(float,int,bool))}
        tables['selection_summary']=pd.DataFrame([scalar])
        return dict(tables=tables,arrays={})
    return path_view(runs_dir,view,n_pairings,loader)


def export_subspace_reference_matches(folder):
    """Compute the former notebook-only iEEG PCA-reference metrics for every fit."""
    from vs2_0_compare_models import correlation_matrix
    folder=Path(folder);config=json.loads((folder/'config.json').read_text());rows=[]
    if 'separate_pca' not in config['models']:raise ValueError('The fixed iEEG reference requires separate_pca fits.')
    for repeat in [-1,*range(config['repeats'])]:
        with np.load(folder/f'separate_pca_{repeat:03d}_model.npz') as saved:
            reference={part:saved[f'{part}_ieeg_scores'] for part in (['train'] if repeat==-1 else ['train','test'])}
        for model in config['models']:
            with np.load(folder/f'{model}_{repeat:03d}_model.npz') as saved:
                for k in config['dimensions']:
                    train=correlation_matrix(reference['train'][:,:k],saved['train_meg_scores'][:,:k])
                    a,b=linear_sum_assignment(-np.nan_to_num(abs(train),nan=-1.));valid=np.isfinite(train[a,b])
                    for part in reference:
                        r=correlation_matrix(reference[part][:,:k],saved[f'{part}_meg_scores'][:,:k])
                        values=abs(r[a[valid],b[valid]])
                        rows.append(dict(model=model,k=k,repeat=repeat,partition='in_sample' if repeat==-1 else part,
                                         matched_abs_r=float(np.nanmean(values)),n_valid_pairs=int(np.isfinite(values).sum())))
    pd.DataFrame(rows).to_csv(folder/'ieeg_pca_reference_matches.csv',index=False)


def finalize_weight_batch(output_dir,n_pairings,roi_level='region'):
    """Across-assignment stability, aligned to -1; inner trial fits averaged first."""
    import shutil
    from vs3_0_spatial_reliability import _load_bundle,align_bundle
    from vs3_0_weight_stability import summarize_weights
    matched=Path(assignment_config(str(output_dir),-1))
    for variation in ('pairing','trials','both'):
        original=matched/variation
        if not (original/'config.json').exists():continue
        config=json.loads((original/'config.json').read_text())
        reference_dir=original/'iteration_000'
        root=matched/'random_assignment_summary'/variation
        root.mkdir(parents=True,exist_ok=True)
        marker=root/'WEIGHT_SUMMARY_COMPLETE.json'
        if marker.exists():marker.unlink()
        batch_config=dict(config,iterations=n_pairings,pairing=None,subject_pairing_iteration=None,
                          assignment_ids=list(range(n_pairings)),reference_assignment=-1,
                          reference_dir=str(reference_dir),inner_variation=variation)
        (root/'config.json').write_text(json.dumps(batch_config,indent=2))
        for iteration in range(n_pairings):
            source=Path(assignment_config(str(original),iteration))
            source_config=json.loads((source/'config.json').read_text())
            dest=root/f'iteration_{iteration:03d}';dest.mkdir(exist_ok=True)
            for name in ('ieeg_features.csv','meg_features.csv'):
                shutil.copyfile(source/'iteration_000'/name,dest/name)
            anchor=pd.read_csv(reference_dir/'ieeg_features.csv')
            current_meta=pd.read_csv(dest/'ieeg_features.csv')
            identity=['subject','source_index','x','y','z']
            if not anchor[identity].equals(current_meta[identity]):raise ValueError('Fixed iEEG coverage changes across subject assignments.')
            for model in config['models']:
                reference=_load_bundle(reference_dir/f'{model}_A.npz');inner=[]
                for j in range(source_config['iterations']):
                    child=source/f'iteration_{j:03d}'
                    if not (child/'COMPLETE.json').exists():raise FileNotFoundError(f'Incomplete stability fit: {child}')
                    aligned,_=align_bundle(reference,_load_bundle(child/f'{model}_A.npz'),model);inner.append(aligned)
                required=['ieeg_weights','meg_weights','ieeg_train_scores','meg_train_scores']
                np.savez_compressed(dest/f'{model}_A.npz',**{key:np.mean([v[key] for v in inner],axis=0) for key in required})
            (dest/'COMPLETE.json').write_text(json.dumps(dict(subject_pairing_iteration=iteration,inner_fits=source_config['iterations'])))
        summarize_weights(root,roi_level=roi_level)
    return matched/'random_assignment_summary'
