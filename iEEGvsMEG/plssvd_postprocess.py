#!/usr/bin/env python3
"""Resumable PLSSVD evaluation. This script never refits PLSSVD.

python plssvd_postprocess.py --runs-dir out/plssvd_eval/full_concatenated \
    --components 5 10 25 --perm meg --perm-type time_point
Add --cache-dir out/trial_cache to recover missing PCA/spectra from old fits.
Use --all-runs to process all existing runs and prepare their matched comparisons.
The notebook reads published evaluation snapshots only.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import uuid
import numpy as np
import pandas as pd

CACHE_VERSION = 1
METRIC_NAMES = ['ieeg_reconstruction_fraction','predict_ieeg_q2','meg_reconstruction_fraction',
                'predict_meg_q2','crosscov_energy_fraction','paired_crosscov_energy_fraction',
                'mean_paired_covariance','total_crosscov_energy','captured_crosscov_energy','mean_r']


def _json(path):
    return json.loads(Path(path).read_text())


def _atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    pending.write_text(json.dumps(value,indent=2,default=str));os.replace(pending,path)


def _atomic_npz(path, **values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with pending.open('wb') as stream:np.savez_compressed(stream,**values)
    os.replace(pending,path)


def _fingerprint(paths):
    # Fit artifacts are immutable. Detect replacement/updates without reading weights.
    entries=[]
    for path in paths:
        path=Path(path)
        entries.append((str(path.resolve()),path.stat().st_size,path.stat().st_mtime_ns) if path.exists()
                       else (str(path.resolve()),None,None))
    return hashlib.sha256(json.dumps([CACHE_VERSION,entries]).encode()).hexdigest()


def _valid(path, fingerprint):
    if not path.exists():return False
    try:
        with np.load(path,allow_pickle=False) as saved:return str(saved['fingerprint'])==fingerprint
    except (ValueError,OSError,EOFError):return False


class Moments:
    """Array-valued online means and sample SD, excluding nonfinite cells."""
    def __init__(self):self.count=self.mean=self.m2=None
    def add(self, values):
        values=np.asarray(values,float);valid=np.isfinite(values)
        if self.count is None:
            self.count=np.zeros(values.shape,int);self.mean=np.zeros(values.shape);self.m2=np.zeros(values.shape)
        count=self.count+valid
        delta=np.where(valid,values-self.mean,0.)
        mean=self.mean+np.divide(delta,count,out=np.zeros_like(delta),where=count>0)
        self.m2+=np.where(valid,delta*(values-mean),0.)
        self.mean=mean;self.count=count
    def arrays(self):
        mean=np.where(self.count>0,self.mean,np.nan)
        sd=np.sqrt(np.divide(self.m2,self.count-1,out=np.full_like(self.mean,np.nan),where=self.count>1))
        return mean,sd,self.count


def _source_paths(run, iteration, fold):
    child=run/f'iteration_{iteration:03d}'
    return [run/'validation_options.json',child/'validation_options.json',child/'COMPLETE.json',
            child/f'scores_{fold:03d}.npz',child/f'model_{fold:03d}.npz',
            child/f'pca_scores_{fold:03d}.npz',child/f'temporal_spectra_{fold:03d}.npz',
            child/'fold_metrics.csv',child/'split_audit.csv.gz',child/'split_audit.csv']



def _metric_fingerprint(run, iteration, fold):
    paths=_source_paths(run,iteration,fold)
    # PCA/spectrum sidecars do not affect reconstruction or ridge prediction.
    return _fingerprint([p for p in paths if not p.name.startswith(('pca_scores_','temporal_spectra_'))])


def _scores(run, iteration, fold, k=None):
    from plssvd_eval_utils import _saved_scores
    with _saved_scores(run/f'iteration_{iteration:03d}',fold) as saved:
        return {p:{m:saved[f'{p}_{m}'][:,:k] for m in ('ieeg','meg')} for p in ('train','test')}


def _common_fold(run, iteration, fold, destination, fingerprint):
    """All-rank correlations are computed once, independently of metric k."""
    from plssvd_eval_utils import _saved_scores
    from plssvd_pca import _pca_artifact
    from compare_models import correlation_matrix
    child=run/f'iteration_{iteration:03d}'
    scores=_scores(run,iteration,fold)
    arrays={};x=scores['train']['ieeg'];k=x.shape[1]
    pair_signs=np.sign(x[np.argmax(np.abs(x),axis=0),np.arange(k)]);pair_signs[pair_signs==0]=1
    for part in ('train','test'):
        x,y=scores[part]['ieeg'],scores[part]['meg']
        covariance=(x-x.mean(0)).T@(y-y.mean(0))/(len(x)-1)
        arrays[part+'_covariance']=covariance*pair_signs[:,None]*pair_signs[None,:]
        arrays[part+'_component_r']=np.diag(correlation_matrix(x,y))
        if part=='test':
            r=correlation_matrix(x,y)
            arrays['test_correlation_signed']=r*pair_signs[:,None]*pair_signs[None,:]
            arrays['test_correlation_absolute']=abs(r)
        if fold==0:
            for modality in ('ieeg','meg'):
                values=scores[part][modality]
                sd=scores['train'][modality].std(0)
                arrays[f'{part}_{modality}']=values
                arrays[f'{part}_{modality}_normalized']=values*pair_signs/np.where(sd>0,sd,1.)
    pca=_pca_artifact(child,fold)
    if pca is not None:
        for modality in ('ieeg','meg'):
            transformed=[]
            for a,b in [(scores['test'][modality],scores['train'][modality]),
                        (pca[f'test_{modality}_pca'],pca[f'train_{modality}_pca'])]:
                signs=np.sign(b[np.argmax(abs(b),axis=0),np.arange(b.shape[1])]);signs[signs==0]=1
                transformed.append(a*signs)
            arrays[f'pca_{modality}_signed']=correlation_matrix(*transformed)
            arrays[f'pca_{modality}_absolute']=abs(arrays[f'pca_{modality}_signed'])
    sidecar=child/f'temporal_spectra_{fold:03d}.npz'
    with _saved_scores(child,fold) as saved:
        extra={}
        if sidecar.exists():
            with np.load(sidecar) as z:extra=dict(z)
        for part in ('train','test'):
            for modality in ('ieeg','meg'):
                key=f'{part}_{modality}_temporal_singular_values'
                if key in saved.files:arrays['spectrum_'+part+'_'+modality]=saved[key]
                elif key in extra:arrays['spectrum_'+part+'_'+modality]=extra[key]
    for part in ('train','test'):
        for modality in ('ieeg','meg'):
            key='spectrum_'+part+'_'+modality
            if key in arrays:
                energy=arrays[key]**2
                fraction=energy/energy.sum() if energy.sum()>0 else np.full_like(energy,np.nan)
                arrays['spectrum_energy_'+part+'_'+modality]=fraction
                arrays['spectrum_rank_'+part+'_'+modality]=1/np.sum(fraction**2)
    _atomic_npz(destination,fingerprint=fingerprint,**arrays)


def _metric_fold(run, iteration, fold, k, destination, fingerprint, options):
    from plssvd_evaluation import evaluate_statistics
    child=run/f'iteration_{iteration:03d}'
    if (child/f'scores_{fold:03d}.npz').exists():
        with np.load(child/f'scores_{fold:03d}.npz') as saved:
            _,metrics,_=evaluate_statistics(saved,k,options['ridge'])
        values=np.array([[metrics[p][name] for name in METRIC_NAMES] for p in ('train','test')])
    else:
        if k!=options['n_components']:
            raise ValueError('Legacy fits support metrics only at their original component count.')
        table=pd.read_csv(child/'fold_metrics.csv')
        values=np.array([table.loc[(table.repeat==fold)&(table.partition==p),METRIC_NAMES].iloc[0]
                         for p in ('train','test')])
    _atomic_npz(destination,fingerprint=fingerprint,metrics=values)


def _audit_hash(run, iteration):
    from plssvd_diagnostics import _audit
    return hashlib.sha256(_audit(run,iteration).to_csv(index=False).encode()).hexdigest()


def _stability(run, ids, folds, k, destination, signature):
    """Pairwise descriptions need only two folds in memory at once."""
    from plssvd_eval_utils import _across_split_stability
    marker=destination/'stability.json'
    if marker.exists() and _json(marker).get('signature')==signature:
        return
    fold_rows=[];iteration_rows=[]
    first=ids[0]
    for a in range(folds):
        sa=_scores(run,first,a,k)
        for b in range(a+1,folds):
            frame,_=_across_split_stability([sa,_scores(run,first,b,k)],k)
            fold_rows.append(frame.assign(fold_a=a,fold_b=b))
    ref=_scores(run,first,0,k)
    for iteration in ids[1:]:
        frame,_=_across_split_stability([ref,_scores(run,iteration,0,k)],k)
        iteration_rows.append(frame.assign(reference_iteration=first,iteration=iteration))
    columns=['fold_a','fold_b','partition','modality','n_components','matched_abs_r','status']
    for name,frames,cols in [('fold_stability',fold_rows,columns),
                            ('iteration_stability',iteration_rows,columns+['reference_iteration','iteration'])]:
        (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=cols)).to_csv(destination/(name+'.csv'),index=False)
    _atomic_json(marker,dict(signature=signature))


def _publish(run, ids, folds, k, options, fingerprint, audits):
    """Stream cached fold arrays into a self-contained, bounded-size snapshot."""
    folder=run/'evaluation'/f'k_{k:03d}'
    current=folder/'CURRENT.json'
    if current.exists():
        previous=folder/_json(current)['snapshot']
        if (previous/'manifest.json').exists() and _json(previous/'manifest.json').get('signature')==fingerprint:
            return previous
    snapshot=folder/('snapshot_'+uuid.uuid4().hex);snapshot.mkdir(parents=True)
    metric_rows=[];component_rows=[];moments={};pca_available=True;spectra_available=True
    def update(key, value):
        if key not in moments:moments[key]=Moments()
        moments[key].add(value)
    for iteration in ids:
        iteration_arrays={};fold_moments={};all_pca=True;all_spectra=True
        for fold in range(folds):
            common=run/'evaluation'/'shared'/f'iteration_{iteration:03d}'/f'fold_{fold:03d}.npz'
            with np.load(common) as saved:
                for part in ('train','test'):
                    r=saved[part+'_component_r'][:k];cov=np.diag(saved[part+'_covariance'])[:k]
                    for pc in range(k):component_rows.append(dict(iteration=iteration,repeat=fold,partition=part,
                        component=pc+1,r=r[pc],covariance=cov[pc]))
                    if fold==0:
                        update(part+'_covariance',saved[part+'_covariance'][:k,:k])
                        for modality in ('ieeg','meg'):
                            update(f'{part}_{modality}_normalized',saved[f'{part}_{modality}_normalized'][:,:k])
                            if iteration==ids[0]:iteration_arrays[f'{part}_{modality}']=saved[f'{part}_{modality}'][:,:k]
                for key in saved.files:
                    if key.startswith('test_correlation_') or key.startswith('pca_') or key.startswith('spectrum_'):
                        values=saved[key];values=values[:k,:k] if values.ndim==2 else values
                        if key not in fold_moments:fold_moments[key]=Moments()
                        fold_moments[key].add(values)
                all_pca &= all(f'pca_{m}_signed' in saved.files for m in ('ieeg','meg'))
                all_spectra &= all(f'spectrum_{p}_{m}' in saved.files for p in ('train','test') for m in ('ieeg','meg'))
            with np.load(folder/f'iteration_{iteration:03d}'/f'fold_{fold:03d}.npz') as saved:
                for index,part in enumerate(('train','test')):
                    metric_rows.append(dict(iteration=iteration,repeat=fold,partition=part,n_components=k,
                                            **dict(zip(METRIC_NAMES,saved['metrics'][index]))))
        for key,stat in fold_moments.items():
            if key.startswith('pca_') and not all_pca:continue
            if key.startswith('spectrum_') and not all_spectra:continue
            values=stat.arrays()[0];iteration_arrays[key]=values;update(key,values)
        pca_available &= all_pca;spectra_available &= all_spectra
        _atomic_npz(snapshot/f'iteration_{iteration:03d}.npz',**iteration_arrays)
    arrays={}
    for key,stat in moments.items():
        for suffix,value in zip(('mean','std','count'),stat.arrays()):arrays[key+'_'+suffix]=value
    axes=run/'trial_axes.npz'
    if not axes.exists():axes=run/f'iteration_{ids[0]:03d}'/'trial_axes.npz'
    with np.load(axes) as z:arrays.update(times=z['times'],conditions=z['conditions'])
    _atomic_npz(snapshot/'arrays.npz',**arrays)
    metrics=pd.DataFrame(metric_rows);metrics.to_csv(snapshot/'fold_metrics.csv',index=False)
    pd.DataFrame(component_rows).to_csv(snapshot/'components.csv',index=False)
    iteration_metrics=metrics.groupby(['iteration','partition'])[METRIC_NAMES].mean().reset_index()
    iteration_metrics.to_csv(snapshot/'iteration_metrics.csv',index=False)
    iteration_metrics.melt(id_vars=['iteration','partition'],var_name='metric',value_name='value').groupby(
        ['partition','metric']).value.agg(['count','mean','std','median','min','max']).reset_index().to_csv(snapshot/'metric_summary.csv',index=False)
    _stability(run,ids,folds,k,folder,fingerprint)
    for name in ('fold_stability','iteration_stability'):
        (snapshot/(name+'.csv')).write_bytes((folder/(name+'.csv')).read_bytes())
    metadata=dict(version=CACHE_VERSION,signature=fingerprint,options=options,n_components=k,
                  iterations=ids,n_requested=options['n_iterations'],audit_hashes=audits,
                  pca_available=pca_available,spectra_available=spectra_available)
    # Publish last: the notebook never sees a half-written snapshot.
    _atomic_json(snapshot/'manifest.json',metadata)
    _atomic_json(current,dict(snapshot=snapshot.name))
    return snapshot


def _evaluate_run(run_dir, components, cache_dir=None, scratch_dir=None):
    """Compute only missing/stale folds; save a new snapshot when inputs change."""
    from plssvd_evaluation import selected_count
    from plssvd_eval_utils import load_trial_cache
    from plssvd_pca import _pca_artifact
    from plssvd_diagnostics import _backfill_spectra
    run=Path(run_dir);options=_json(run/'validation_options.json')
    if options.get('schema_version') not in (4,6):raise ValueError('Expected a repeated PLSSVD fit directory.')
    counts=sorted({selected_count(k,options['n_components']) for k in components})
    ids=[i for i in range(options['n_iterations']) if (run/f'iteration_{i:03d}'/'COMPLETE.json').exists()]
    if not ids:raise FileNotFoundError(f'{run}: no completed fit iterations.')
    folds=options['repeats'];fingerprints=[];trials=None;audits={};computed=0
    for iteration in ids:
        child=run/f'iteration_{iteration:03d}';audits[str(iteration)]=_audit_hash(run,iteration)
        for fold in range(folds):
            if cache_dir is not None:
                missing_pca=_pca_artifact(child,fold) is None
                missing_spectra=not _has_full_fold(child,fold)
                if missing_pca or missing_spectra:
                    if trials is None:trials=load_trial_cache(cache_dir)
                    _backfill_spectra(run,iteration,trials,scratch_dir,include_pca=missing_pca,fold_indices=[fold])
            fingerprint=_fingerprint(_source_paths(run,iteration,fold));fingerprints.append(fingerprint)
            common=run/'evaluation'/'shared'/f'iteration_{iteration:03d}'/f'fold_{fold:03d}.npz'
            if not _valid(common,fingerprint):
                _common_fold(run,iteration,fold,common,fingerprint);computed+=1
            metric_fingerprint=_metric_fingerprint(run,iteration,fold)
            for k in counts:
                path=run/'evaluation'/f'k_{k:03d}'/f'iteration_{iteration:03d}'/f'fold_{fold:03d}.npz'
                if not _valid(path,metric_fingerprint):_metric_fold(run,iteration,fold,k,path,metric_fingerprint,options)
            print(f'Evaluated {run.name}: iteration {iteration+1}, fold {fold+1}/{folds} (cached results reused)',flush=True)
    signature=hashlib.sha256(json.dumps([fingerprints,audits]).encode()).hexdigest()
    for k in counts:_publish(run,ids,folds,k,options,signature,audits)
    print(f'{run.name}: {len(ids)}/{options["n_iterations"]} complete iterations; {computed} new/updated shared fold evaluations.',flush=True)



def publish_available(run_dir, components):
    """Publish fully evaluated iterations after an interrupted worker, no refitting."""
    from plssvd_evaluation import selected_count
    run=Path(run_dir);options=_json(run/'validation_options.json');folds=options['repeats']
    for requested in components:
        k=selected_count(requested,options['n_components']);ids=[];fingerprints=[];audits={}
        for iteration in range(options['n_iterations']):
            if not (run/f'iteration_{iteration:03d}'/'COMPLETE.json').exists():continue
            current=[]
            for fold in range(folds):
                fingerprint=_fingerprint(_source_paths(run,iteration,fold))
                common=run/'evaluation'/'shared'/f'iteration_{iteration:03d}'/f'fold_{fold:03d}.npz'
                metrics=run/'evaluation'/f'k_{k:03d}'/f'iteration_{iteration:03d}'/f'fold_{fold:03d}.npz'
                if not _valid(common,fingerprint) or not _valid(metrics,_metric_fingerprint(run,iteration,fold)):break
                current.append(fingerprint)
            if len(current)==folds:
                ids.append(iteration);fingerprints.extend(current);audits[str(iteration)]=_audit_hash(run,iteration)
        if not ids:
            print(f'{run.name}, k={k}: no fully evaluated iterations to publish.',flush=True)
            continue
        signature=hashlib.sha256(json.dumps([fingerprints,audits]).encode()).hexdigest()
        _publish(run,ids,folds,k,options,signature,audits)
        print(f'Published {len(ids)} evaluated iterations for {run.name}, k={k}.',flush=True)


def evaluate_run(run_dir, components, cache_dir=None, scratch_dir=None):
    try:
        return _evaluate_run(run_dir,components,cache_dir,scratch_dir)
    except (Exception,KeyboardInterrupt):
        # Preserve the original exception, but expose any fully evaluated work.
        try:publish_available(run_dir,components)
        except Exception as exc:print(f'Fold checkpoints retained; snapshot publication failed: {exc}',flush=True)
        raise


def _has_full_fold(child, fold):
    from plssvd_eval_utils import _saved_scores
    with _saved_scores(child,fold) as saved:
        keys=set(saved.files)
    sidecar=child/f'temporal_spectra_{fold:03d}.npz'
    if sidecar.exists():
        with np.load(sidecar) as z:keys.update(z.files)
    return all(f'{p}_{m}_temporal_singular_values' in keys for p in ('train','test') for m in ('ieeg','meg'))


def snapshot_path(run_dir, k):
    path=Path(run_dir)/'evaluation'/f'k_{k:03d}'/'CURRENT.json'
    if not path.exists():raise FileNotFoundError(f'No prepared evaluation for k={k} in {run_dir}. Run plssvd_postprocess.py --runs-dir <parent> --components {k} with the selected --perm/--perm-type. The notebook will not compute it.')
    folder=path.parent/_json(path)['snapshot']
    if not (folder/'manifest.json').exists():raise FileNotFoundError(f'Incomplete evaluation snapshot: {folder}')
    return folder


def prepare_comparison(runs_dir, name, k):
    """Pair prepared snapshots and publish compact comparison arrays/tables."""
    root=Path(runs_dir);a=snapshot_path(root/'none',k);b=snapshot_path(root/name,k)
    ma,mb=_json(a/'manifest.json'),_json(b/'manifest.json')
    for key in ('seed','repeats','meg_kind','split_unit','block_scaling','ridge','condition_mode'):
        if ma['options'].get(key)!=mb['options'].get(key):raise ValueError(f'Cannot pair runs: different {key}.')
    ids=sorted(set(ma['iterations'])&set(mb['iterations']))
    if not ids:raise ValueError('No shared evaluated iteration IDs.')
    for i in ids:
        if ma['audit_hashes'][str(i)]!=mb['audit_hashes'][str(i)]:raise ValueError(f'Cannot pair iteration {i}: trial assignments differ.')
    with np.load(a/'arrays.npz') as x,np.load(b/'arrays.npz') as y:
        if not np.array_equal(x['times'],y['times']) or not np.array_equal(x['conditions'],y['conditions']):
            raise ValueError('Cannot pair runs: time/condition axes differ.')
    for filename in ('matching.csv','pairing.json'):
        pa,pb=root/'none'/filename,root/name/filename
        if pa.exists() and pb.exists() and pa.read_bytes()!=pb.read_bytes():raise ValueError(f'Cannot pair runs: different {filename}.')
    folder=root/name/'evaluation'/f'k_{k:03d}'/'comparison';folder.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256((ma['signature']+mb['signature']).encode()).hexdigest()
    if (folder/'CURRENT.json').exists():
        old=folder/_json(folder/'CURRENT.json')['snapshot']
        if _json(old/'manifest.json')['signature']==signature:return
    out=folder/('snapshot_'+uuid.uuid4().hex);out.mkdir()
    moments={};pca_available=True;spectra_available=True
    for iteration in ids:
        with np.load(a/f'iteration_{iteration:03d}.npz') as za,np.load(b/f'iteration_{iteration:03d}.npz') as zb:
            for key in set(za.files)&set(zb.files):
                if not (key.startswith('pca_') or key.startswith('spectrum_')):continue
                for label,values in [('Unpermuted',za[key]),('Permuted',zb[key]),('Difference',zb[key]-za[key])]:
                    if (label,key) not in moments:moments[label,key]=Moments()
                    moments[label,key].add(values)
            pca_available &= all(f'pca_{m}_signed' in za.files and f'pca_{m}_signed' in zb.files for m in ('ieeg','meg'))
            spectra_available &= all(f'spectrum_{p}_{m}' in za.files and f'spectrum_{p}_{m}' in zb.files for p in ('train','test') for m in ('ieeg','meg'))
    arrays={}
    for (label,key),stat in moments.items():
        for suffix,value in zip(('mean','std','count'),stat.arrays()):arrays[f'{label}_{key}_{suffix}']=value
    _atomic_npz(out/'arrays.npz',**arrays)
    metrics=[];components=[]
    for label,source in [('Unpermuted',a),('Permuted',b)]:
        frame=pd.read_csv(source/'iteration_metrics.csv');metrics.append(frame[frame.iteration.isin(ids)].assign(run=label))
        frame=pd.read_csv(source/'components.csv');frame=frame[frame.iteration.isin(ids)]
        components.append(frame.groupby(['iteration','partition','component'])[['r','covariance']].mean().reset_index().assign(run=label))
    metrics=pd.concat(metrics,ignore_index=True);metrics.to_csv(out/'metrics.csv',index=False)
    pd.concat(components,ignore_index=True).to_csv(out/'components.csv',index=False)
    first=metrics.query("run == 'Unpermuted'").set_index(['iteration','partition'])
    second=metrics.query("run == 'Permuted'").set_index(['iteration','partition'])
    difference=(second[METRIC_NAMES]-first[METRIC_NAMES]).reset_index();difference.to_csv(out/'differences.csv',index=False)
    difference.melt(id_vars=['iteration','partition'],var_name='metric',value_name='delta').groupby(
        ['partition','metric']).delta.agg(['count','mean','std','median','min','max']).reset_index().to_csv(out/'delta_summary.csv',index=False)
    _atomic_json(out/'manifest.json',dict(signature=signature,source_signatures=[ma['signature'],mb['signature']],
        iterations=ids,n_components=k,permutation=name,pca_available=pca_available,spectra_available=spectra_available))
    _atomic_json(folder/'CURRENT.json',dict(snapshot=out.name))


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs-dir',type=Path,required=True)
    parser.add_argument('--components',type=int,nargs='+',required=True)
    parser.add_argument('--perm',choices=['none','ieeg','meg','both'],default='none')
    parser.add_argument('--perm-type',choices=['none','time_cirular_shift','time_circular_shift','time_point','time_block','space','phase'],default='none')
    parser.add_argument('--all-runs',action='store_true')
    parser.add_argument('--publish-only',action='store_true',help='Publish already evaluated iterations; do not evaluate additional folds.')
    parser.add_argument('--cache-dir',type=Path,help='Optional original trial cache for missing PCA and full spectra.')
    parser.add_argument('--scratch-dir',type=Path)
    args=parser.parse_args()
    from plssvd_eval_utils import validation_run_name
    if args.all_runs:
        names=sorted(p.name for p in args.runs_dir.iterdir() if (p/'validation_options.json').exists()
                     and any(p.glob('iteration_*/COMPLETE.json')))
        if not names:raise FileNotFoundError('No runs with completed fit iterations are available.')
    else:
        name=validation_run_name(args.perm,args.perm_type);names=['none'] if name=='none' else ['none',name]
    for name in names:
        if args.publish_only:publish_available(args.runs_dir/name,args.components)
        else:evaluate_run(args.runs_dir/name,args.components,args.cache_dir,args.scratch_dir)
    for name in names:
        if name!='none':
            for k in args.components:
                try:prepare_comparison(args.runs_dir,name,k)
                except FileNotFoundError as exc:
                    if not args.publish_only:raise
                    print(f'Comparison not yet available: {exc}',flush=True)


if __name__=='__main__':main()
