"""Trial-held-out PLSSVD validation. No time-sample train/test splitting.

All trainable normalisation, PLS weights and prediction maps are training-only.
Component count is fixed in advance. The same cohort and anatomical assignment
are evaluated on shuffled K-fold trial splits, without tuning.
"""
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from contextlib import contextmanager
import shutil
import hashlib
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from coverage_matching_utils import Dataset, coordinates_mm, construct_five_datasets
from cov_models_utils import _gram, _spectrum, _weights



@dataclass
class ValidationOptions:
    repeats: int = 5
    n_components: int = 5
    n_null: int = 199
    seed: int = 2026
    split_unit: str = 'trial'    # 'group' uses split_group from trial metadata
    block_scaling: str = 'none'
    ridge: float = 1e-3         # fixed, relative to training score covariance
    max_gram_gib: float = 2.


@dataclass
class TrialSubject:
    subject: str
    conditions: tuple
    data: list                 # one memmapped trial x channel x time array per condition
    positions: np.ndarray      # MNI mm
    metadata: pd.DataFrame
    split_groups: list
    permutation_blocks: list


@dataclass
class TrialData:
    ieeg: list
    meg: list
    times: np.ndarray
    conditions: tuple


def _signature(path):
    p = Path(path)
    return dict(path=str(p.resolve()), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)


def prepare_trial_cache(meg_raw_dir, ieeg_dir, cache_dir, meg_subjects, ieeg_subjects,
                        electrode_metadata, conditions=(1,2), ieeg_coordinate_unit='m',
                        meg_coordinate_unit='m', trial_metadata_csv=None):
    """Export/reuse unaveraged trials, not the old MEG *_source.p averages.

    Raw MEG layout follows LB10_extract_data.py: OUT.sources_ERFs[c][0] is
    source x time x trial, OUT.time is seconds, positions are in the stated unit.
    iEEG epochs are trial x channel x time with info.json event_id/time_epoch.

    Optional CSV keys: modality ('meg'/'ieeg'), subject, condition, trial_index
    (zero-based WITHIN condition). Optional columns: split_group (run or stimulus
    identity; never split across partitions), permutation_block (e.g. acoustically
    matched stratum; labels permuted only within subject/partition/block).
    All trials require metadata rows when CSV is supplied. No trial labels are
    inferred to mean memory or recognition. Existing caches are checked against
    source file signatures and configuration; stale caches fail explicitly.
    Completed subjects are checkpointed and skipped when resuming this directory.
    Without a manifest, source data rebuilds the metadata and validates existing
    trial files; matching files are preserved and missing/partial files written.
    """
    root=Path(cache_dir); root.mkdir(parents=True,exist_ok=True)
    meta=electrode_metadata.copy(); meta['subject']=meta.subject.astype(str)
    conditions=tuple(conditions)
    if len(conditions)!=2:
        raise ValueError('This validation currently uses exactly two conditions and their difference.')
    xyz=coordinates_mm(meta[['x','y','z']],ieeg_coordinate_unit)
    meta[['x','y','z']]=xyz
    config=dict(meg_subjects=list(meg_subjects),ieeg_subjects=list(ieeg_subjects),conditions=list(conditions),
        meg_unit=meg_coordinate_unit,
        metadata_hash=hashlib.sha256(meta.to_csv(index=False).encode()).hexdigest(),
        trial_metadata=_signature(trial_metadata_csv) if trial_metadata_csv else None)
    manifest=root/'manifest.json'
    times=None; records=[]; sources=[]
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        saved_config={k:v for k,v in saved['config'].items() if k not in ('meg_subjects','ieeg_subjects')}
        current_config={k:v for k,v in config.items() if k not in ('meg_subjects','ieeg_subjects')}
        if saved_config!=current_config:
            raise ValueError('Cache configuration changed. Choose a new cache directory.')
        for signature in saved['source_files']:
            if Path(signature['path']).exists() and _signature(signature['path'])!=signature:
                raise ValueError('A source file changed. Choose a new cache directory to re-export trials.')
        times=np.asarray(saved['times'],float)
        sources=saved['source_files']
        requested={(modality,str(subject)) for modality,subjects in
                   [('ieeg',ieeg_subjects),('meg',meg_subjects)] for subject in subjects}
        records=[row for row in saved['records']
                 if (row['modality'],row['subject']) in requested
                 and len(row['files'])==len(conditions)
                 and all((root/f).is_file() for f in row['files'])]
    completed={(row['modality'],row['subject']) for row in records}

    def save_progress():
        pending=root/'manifest.json.tmp'
        pending.write_text(json.dumps(dict(config=config,times=times.tolist(),records=records,
                                          source_files=sources),indent=2))
        pending.replace(manifest)

    if all((modality,str(subject)) in completed for modality,subjects in
           [('ieeg',ieeg_subjects),('meg',meg_subjects)] for subject in subjects) and times is not None:
        save_progress()
        print('cache computed')
        return load_trial_cache(root)
    extra=None
    if trial_metadata_csv:
        extra=pd.read_csv(trial_metadata_csv,dtype={'subject':str,'modality':str})
        keys=['modality','subject','condition','trial_index']
        if not set(keys)<=set(extra) or extra.duplicated(keys).any():
            raise ValueError('Trial metadata needs unique modality/subject/condition/trial_index keys.')
    for modality, subjects in [('ieeg',ieeg_subjects),('meg',meg_subjects)]:
        for subject in subjects:
            if (modality,str(subject)) in completed:
                continue
            if modality=='ieeg':
                info_path=Path(ieeg_dir)/f'{subject}_info.json'
                data_path=Path(ieeg_dir)/f'{subject}_epochs.p'
                info=json.loads(info_path.read_text());labels=np.asarray(info['event_id'],int).ravel()
                t=np.asarray(info['time_epoch'],float).ravel()
                with data_path.open('rb') as f: epochs=np.asarray(pickle.load(f))
                if epochs.ndim!=3 or len(labels)!=len(epochs) or epochs.shape[-1]!=len(t):
                    raise ValueError(f'{subject}: epoch/event/time mismatch.')
                arrays=[epochs[labels==c] for c in conditions]
                sm=meta.loc[meta.subject==str(subject)].sort_values('channel_index').reset_index(drop=True)
                if sm.channel_index.tolist()!=list(range(epochs.shape[1])):
                    raise ValueError(f'{subject}: metadata must cover every epoch channel in order.')
                pos=sm[['x','y','z']].to_numpy();sources.extend([_signature(info_path),_signature(data_path)])
            else:
                data_path=Path(meg_raw_dir)/f'{subject}_norm0_abs_0.mat'
                if not data_path.exists():
                    raise FileNotFoundError(f'Missing raw MEG trials: {data_path}. Averaged *_source.p files cannot be used.')
                try:
                    import mat73
                except ImportError as exc:
                    raise ImportError('Install mat73 in the notebook environment to export the original MEG .mat trials.') from exc
                out=mat73.loadmat(str(data_path))['OUT']
                t=np.asarray(out['time'],float).ravel()
                pos=coordinates_mm(out['pos_brainsources_MNI8'],meg_coordinate_unit)
                arrays=[np.asarray(out['sources_ERFs'][c][0]).transpose(2,0,1) for c in range(len(conditions))]
                sm=pd.DataFrame(pos,columns=['x','y','z']);sm['subject']=str(subject);sm['channel_index']=np.arange(len(pos))
                sources.append(_signature(data_path))
            if len(t)<2 or not np.isfinite(t).all() or not np.all(np.diff(t)>0):
                raise ValueError('Invalid time axis.')
            if times is None:times=t
            if len(t)==len(times)+1 and np.allclose(t[:-1],times,rtol=0,atol=1e-7):
                arrays=[a[...,:-1] for a in arrays];t=t[:-1]
            if len(t)!=len(times) or not np.allclose(t,times,rtol=0,atol=1e-7):
                raise ValueError(f'{subject}: independent MEG/iEEG time axes do not agree.')
            files=[];groups=[];blocks=[]
            for c,a in zip(conditions,arrays):
                if a.ndim!=3 or a.shape[1:]!=(len(pos),len(times)) or not np.isfinite(a).all():
                    raise ValueError(f'{subject}, condition {c}: invalid trial array.')
                if len(a)<2:
                    raise ValueError(f'{subject}, condition {c}: need >=2 trials; evaluation additionally requires at least n_splits trials per condition.')
                fname=f'{modality}_{subject}_condition{c}.npy'
                target=root/fname
                reuse=False
                if target.is_file():
                    try:
                        cached=np.load(target,mmap_mode='r',allow_pickle=False)
                        # Check one trial at a time to avoid copying a whole subject.
                        reuse=(cached.shape==a.shape and cached.dtype==np.float32
                               and all(np.array_equal(cached[i],np.asarray(a[i],dtype=np.float32))
                                       for i in range(len(a))))
                        del cached
                    except (ValueError,OSError,EOFError):
                        reuse=False
                if not reuse:
                    pending=target.with_suffix('.npy.tmp')
                    with pending.open('wb') as f:
                        np.save(f,np.asarray(a,dtype=np.float32))
                    pending.replace(target)
                files.append(fname)
                if extra is None:
                    groups.append(None);blocks.append(['all']*len(a))
                else:
                    rows=extra.loc[(extra.modality==modality)&(extra.subject==str(subject))&(extra.condition==c)].sort_values('trial_index')
                    if rows.trial_index.tolist()!=list(range(len(a))):
                        raise ValueError(f'Trial metadata incomplete for {modality}/{subject}/{c}.')
                    if 'split_group' in rows and rows.split_group.isna().any():raise ValueError('Missing split_group.')
                    if 'permutation_block' in rows and rows.permutation_block.isna().any():raise ValueError('Missing permutation_block.')
                    groups.append(rows.split_group.astype(str).tolist() if 'split_group' in rows else None)
                    blocks.append(rows.permutation_block.astype(str).tolist() if 'permutation_block' in rows else ['all']*len(a))
            records.append(dict(modality=modality,subject=str(subject),files=files,positions=pos.tolist(),
                metadata=sm.to_dict(orient='list'),split_groups=groups,permutation_blocks=blocks))
            save_progress()
            print(f'Cached {modality} {subject}: {[len(a) for a in arrays]} trials',flush=True)
            del arrays, a
            if modality=='meg':del out
            else:del epochs
    save_progress()
    return load_trial_cache(root)


def load_trial_cache(cache_dir):
    root=Path(cache_dir);m=json.loads((root/'manifest.json').read_text());modalities={'ieeg':[],'meg':[]}
    for row in m['records']:
        arrays=[np.load(root/f,mmap_mode='r') for f in row['files']]
        if any(a.ndim!=3 for a in arrays):raise ValueError('Cache must contain individual trials.')
        modalities[row['modality']].append(TrialSubject(row['subject'],tuple(m['config']['conditions']),arrays,
            np.asarray(row['positions']),pd.DataFrame(row['metadata']),row['split_groups'],row['permutation_blocks']))
    return TrialData(modalities['ieeg'],modalities['meg'],np.asarray(m['times']),tuple(m['config']['conditions']))


def _split_subject(subject,rng,unit):
    """Legacy split used by compare_subspace; fixed-k PLSSVD uses _subject_folds."""
    names=('train','tune','test_a','test_b')
    out={name:[] for name in names}
    if unit=='trial':
        for a in subject.data:
            n=len(a)
            # Half train, roughly one fifth tune, remainder divided into test halves.
            ntrain=max(2,n//2);ntune=max(2,n//5);nleft=n-ntrain-ntune
            if nleft<4: ntrain=n-6;ntune=2;nleft=4
            if ntrain<2:raise ValueError('Insufficient trials for independent partitions.')
            ix=rng.permutation(n);sizes=[ntrain,ntune,nleft//2,nleft-nleft//2]
            for name,part in zip(names,np.split(ix,np.cumsum(sizes)[:-1])):out[name].append(part)
    elif unit=='group':
        if any(g is None for g in subject.split_groups):raise ValueError('Group splitting requires split_group for every trial.')
        groups=np.unique(np.concatenate(subject.split_groups))
        if len(groups)<4:raise ValueError('At least four distinct split groups required.')
        for attempt in range(500):
            order=rng.permutation(groups);ng=len(order)
            cuts=[max(1,ng//2),max(1,ng//5)]
            if sum(cuts)>ng-2:cuts=[ng-3,1]
            remaining=ng-sum(cuts);sizes=cuts+[remaining//2,remaining-remaining//2]
            assignment=dict(zip(names,np.split(order,np.cumsum(sizes)[:-1])))
            proposal={name:[np.flatnonzero(np.isin(g,chosen)) for g in subject.split_groups] for name,chosen in assignment.items()}
            if all(len(ix)>=2 for parts in proposal.values() for ix in parts):out=proposal;break
        else:raise ValueError('Cannot split groups with >=2 trials per condition in every partition.')
    else:raise ValueError("split_unit must be 'trial' or 'group'.")
    out['test']=[np.concatenate([a,b]) for a,b in zip(out['test_a'],out['test_b'])]
    return out


def _subject_folds(subject, rng, unit, n_splits):
    """One shuffled partition: each trial is held out exactly once."""
    if n_splits < 2:
        raise ValueError('Need at least two folds.')
    if unit == 'trial':
        if any(len(a) < n_splits for a in subject.data):
            raise ValueError('Each condition needs at least n_splits trials.')
        tests = [np.array_split(rng.permutation(len(a)), n_splits) for a in subject.data]
    elif unit == 'group':
        if any(g is None for g in subject.split_groups):
            raise ValueError('Group splitting requires split_group for every trial.')
        groups = np.unique(np.concatenate(subject.split_groups))
        if len(groups) < n_splits:
            raise ValueError('Need at least n_splits distinct groups.')
        for _ in range(500):
            chunks = np.array_split(rng.permutation(groups), n_splits)
            tests = [[np.flatnonzero(np.isin(g, chunk)) for chunk in chunks]
                     for g in subject.split_groups]
            if all(len(ix) for condition in tests for ix in condition):
                break
        else:
            raise ValueError('Cannot assign groups with every condition in every test fold.')
    else:
        raise ValueError("split_unit must be 'trial' or 'group'.")
    return [dict(test=[c[f] for c in tests],
                 train=[np.setdiff1d(np.arange(len(a)), c[f]) for a,c in zip(subject.data,tests)])
            for f in range(n_splits)]


def _trial_mean(array,indices):
    # Accumulate without fancy indexing a whole subject's trial cube.
    mean=np.zeros(array.shape[1:],float)
    for i in indices:mean+=array[i]
    return mean/len(indices)


def _build_fold(trials,meg_kind,split_indices,root,seed,mmap_handles=None,condition_mode="stack"):
    prepared={m:{} for m in ('ieeg','meg')};scalers={m:{} for m in ('ieeg','meg')}
    for modality,subjects in [('ieeg',trials.ieeg),('meg',trials.meg)]:
        for s in subjects:
            ix=split_indices[modality][s.subject]
            train=np.stack([_trial_mean(a,i) for a,i in zip(s.data,ix['train'])])
            multiplier=1000. if modality=='ieeg' else 1.
            # Match original pipeline: MEG z-score based on TRAIN condition averages only.
            mu=train.mean(axis=(0,2),keepdims=True) if modality=='meg' else np.zeros((1,train.shape[1],1))
            sd=train.std(axis=(0,2),keepdims=True) if modality=='meg' else np.ones_like(mu)
            sd=np.where(sd>0,sd,1.)
            scalers[modality][s.subject]=(mu.ravel(),sd.ravel(),multiplier)
            prepared[modality][s.subject]={}
            for name,parts in ix.items():
                path=Path(root)/f'{modality}_{s.subject}_{name}.npy'
                out=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=train.shape)
                if mmap_handles is not None:
                    mmap_handles.append(out)
                for c,(a,indices) in enumerate(zip(s.data,parts)):
                    mean=train[c] if name=='train' else _trial_mean(a,indices)
                    out[c]=((mean-mu[0])/sd[0])*multiplier
                out.flush();prepared[modality][s.subject][name]=out
    metadata=pd.concat([s.metadata for s in trials.ieeg],ignore_index=True)
    coords=np.concatenate([s.positions for s in trials.ieeg]);owners=np.concatenate([np.repeat(s.subject,len(s.positions)) for s in trials.ieeg])
    context={'times':trials.times,'load_config':{'conditions':list(trials.conditions)}}
    datasets={};audit=None;pairing=None
    # Shared with compare_subspace, which still has a tuning partition.
    for part in next(iter(split_indices['ieeg'].values())):
        ieeg_arrays=[prepared['ieeg'][s.subject][part] for s in trials.ieeg]
        ieeg=Dataset('iEEG',ieeg_arrays,metadata,np.arange(len(coords)),condition_mode,source_data=context)
        selected,audit,pairing=construct_five_datasets(
            [prepared['meg'][s.subject][part] for s in trials.meg],
            [s.positions for s in trials.meg],[s.subject for s in trials.meg],coords,owners,
            seed=seed,pairing=pairing,condition_mode=condition_mode,kinds=[meg_kind])
        meg=selected[meg_kind];meg.source_data=context
        datasets[part]={'ieeg':ieeg,'meg':meg}
    return datasets,scalers,audit,pairing



@contextmanager
def _temporary_fold(trials, meg_kind, split_indices, seed, scratch_dir=None, condition_mode="stack"):
    """Own scratch mappings until computation finishes; close BEFORE unlinking.

    Only mappings created by _build_fold are closed, never the input trial cache.
    This also closes partially constructed folds if exporting or fitting fails.
    TMPDIR is honored by default; scratch_dir can select cluster-local storage.
    Cleanup failures warn without replacing an analysis exception or aborting
    completed computations (e.g. transient network-filesystem .nfs files).
    """
    if scratch_dir is not None:
        scratch_dir=Path(scratch_dir).expanduser()
        scratch_dir.mkdir(parents=True,exist_ok=True)
    scratch=Path(mkdtemp(prefix='fold_means_',dir=scratch_dir))
    handles=[]
    try:
        yield _build_fold(trials,meg_kind,split_indices,scratch,seed,mmap_handles=handles,condition_mode=condition_mode)
    finally:
        # Views may still reference these arrays, but no fold operations are
        # allowed after this context exits. Own each underlying mapping once.
        for array in handles:
            mapping=array._mmap
            if mapping is not None and not mapping.closed:
                try:
                    mapping.close()
                except (OSError,BufferError) as exc:
                    warnings.warn(f'Could not close scratch mapping in {scratch}: {exc}',RuntimeWarning)
        handles.clear()
        try:
            shutil.rmtree(scratch)
        except OSError as exc:
            warnings.warn(f'Could not remove temporary fold directory {scratch}: {exc}. '
                          'Only temporary files remain; analysis results are retained. '
                          'Use a local scratch directory via --scratch-dir if this persists.',RuntimeWarning)


def _means(dataset):return np.concatenate([x.mean(0) for x in dataset.blocks()])


def _fit(train,max_components,options):
    n=train['ieeg'].n_observations
    if 12*n*n*8/2**30>options.max_gram_gib:raise MemoryError('Increase max_gram_gib or shorten epochs.')
    gx,gy=_gram(train['ieeg']),_gram(train['meg'])
    if options.block_scaling=='none':sx=sy=1.
    elif options.block_scaling=='equal_variance':sx,sy=np.sqrt((n-1)/np.trace(gx)),np.sqrt((n-1)/np.trace(gy))
    else:raise ValueError('Unknown block scaling.')
    ux,dx=_spectrum(gx*sx*sx,train['ieeg'].n_features);uy,dy=_spectrum(gy*sy*sy,train['meg'].n_features)
    cross=dx[:,None]*(ux.T@uy)*dy[None,:]/(n-1)
    left,s,right=svd(cross,full_matrices=False,check_finite=False)
    tol=np.finfo(float).eps*max(cross.shape)*dx[0]*dy[0]/(n-1)
    k=min(max_components,int((s>tol).sum()))
    if k<1:raise ValueError('No stable training cross-covariance dimensions.')
    return {'ieeg_weights':_weights(train['ieeg'],(ux/dx)@left[:,:k],sx),
            'meg_weights':_weights(train['meg'],(uy/dy)@right[:k].T,sy),
            'ieeg_mean':_means(train['ieeg']),'meg_mean':_means(train['meg']),
            'ieeg_scale':sx,'meg_scale':sy,'k_max':k}


def _project(dataset,model,modality):
    # Float64 centering in bounded chunks keeps projections consistent with
    # the Gram-based covariance calculation, including float32 cached averages.
    out=np.zeros((dataset.n_observations,model['k_max']));offset=0
    for block in dataset.blocks():
        for start in range(0, block.shape[1], 1024):
            stop=min(start+1024,block.shape[1]);section=slice(offset+start,offset+stop)
            x=np.asarray(block[:,start:stop],dtype=float)-model[modality+'_mean'][section]
            out+=x@model[modality+'_weights'][section]*model[modality+'_scale']
        offset+=block.shape[1]
    return out


def _predictor(scores,target,mean,ridge):
    gram=scores.T@scores
    penalty=ridge*np.trace(gram)/scores.shape[1]
    inverse=np.linalg.solve(gram+max(penalty,np.finfo(float).eps)*np.eye(len(gram)),scores.T)
    coef=np.empty((scores.shape[1],target.n_features));offset=0
    for x in target.blocks():
        width=x.shape[1];coef[:,offset:offset+width]=inverse@(x-mean[offset:offset+width]);offset+=width
    return coef


def _q2(scores,target,mean,coef):
    error=baseline=0.;offset=0
    for x in target.blocks():
        width=x.shape[1];yc=x-mean[offset:offset+width]
        error+=np.sum((yc-scores@coef[:,offset:offset+width])**2);baseline+=np.sum(yc*yc);offset+=width
    return 1-error/baseline if baseline>0 else np.nan


def _r(a,b):
    a=np.asarray(a,float)-np.mean(a,axis=0);b=np.asarray(b,float)-np.mean(b,axis=0)
    den=np.linalg.norm(a,axis=0)*np.linalg.norm(b,axis=0)
    return np.divide(np.sum(a*b,axis=0),den,out=np.full(den.shape,np.nan),where=den>0).clip(-1,1)


def _association(a,b,absolute=False):
    r=_r(a,b)
    if absolute:r=np.abs(r)
    # A constant response has no evidence of association, not a missing null draw.
    return float(np.mean(np.nan_to_num(r,nan=0.)))


def _contrast(scores,ntime):return scores.reshape(2,ntime,-1)[1]-scores.reshape(2,ntime,-1)[0]


def _patterns(dataset,scores):
    centered=scores-scores.mean(0);inv=np.linalg.pinv(centered.T@centered)
    out=[]
    for x in dataset.blocks():out.append((x-x.mean(0)).T@centered@inv)
    return np.concatenate(out)


def _raw_weights(trials,fold,model,scalers,audit,kind,modality,k):
    weights=model[modality+'_weights'][:,:k]*model[modality+'_scale']
    subjects=getattr(trials,modality);out={s.subject:np.zeros((len(s.positions),k)) for s in subjects}
    if modality=='ieeg' or kind=='full_concatenated':
        offset=0
        for s in subjects:out[s.subject][:]=weights[offset:offset+len(s.positions)];offset+=len(s.positions)
    elif kind=='full_average':
        for s in subjects:out[s.subject][:]=weights/len(subjects)
    elif kind=='coverage_average':
        from scipy.spatial import cKDTree
        coords=np.concatenate([s.positions for s in trials.ieeg])
        for s in subjects:
            ix=cKDTree(s.positions).query(coords)[1];np.add.at(out[s.subject],ix,weights/len(subjects))
    else:
        column='random_source_index' if kind=='random_control' else 'source_index'
        for subject,rows in audit.groupby('meg_subject',sort=False):
            np.add.at(out[subject],rows[column].to_numpy(int),weights[rows.electrode_index.to_numpy(int)])
    for s in subjects:
        _,sd,mult=scalers[modality][s.subject]
        out[s.subject]*=(mult/sd)[:,None]
    return out


def _test_trial_projections(trials,indices,weights):
    out=[]
    for s in trials.meg:
        if not np.any(weights[s.subject]):continue
        values=[];labels=[];blocks=[]
        for c,(a,ix) in enumerate(zip(s.data,indices['meg'][s.subject]['test'])):
            for i in ix:
                values.append(a[i].T@weights[s.subject]);labels.append(c)
                block=str(s.permutation_blocks[c][i])
                if s.split_groups[c] is not None:block+='|'+str(s.split_groups[c][i])
                blocks.append(block)
        out.append((np.asarray(values),np.asarray(labels),np.asarray(blocks)))
    return out


def _permuted_contrast(projected,rng=None):
    total=None;movable=False
    for values,labels,blocks in projected:
        lab=labels.copy()
        for block in np.unique(blocks):
            ix=np.flatnonzero(blocks==block)
            if len(np.unique(lab[ix]))>1:movable=True
            if rng is not None:lab[ix]=rng.permutation(lab[ix])
        difference=values[lab==1].mean(0)-values[lab==0].mean(0)
        total=difference if total is None else total+difference
    return total,movable


def _tail(observed,null):
    null=np.asarray(null)
    if not np.isfinite(observed) or not np.isfinite(null).all():return np.nan
    return (1+np.sum(null>=observed))/(1+len(null))


def _null_tests(trials,fold,model,scalers,audit,kind,indices,scores,patterns,k,options,rng):
    x,y=scores['test']['ieeg'][:,:k],scores['test']['meg'][:,:k]
    cx,cy=_contrast(x,len(trials.times)),_contrast(y,len(trials.times))
    ix=fold['test']['meg'].electrode_to_feature
    px,py=patterns['test']['ieeg'],patterns['test']['meg'][ix]
    weights=_raw_weights(trials,fold,model,scalers,audit,kind,'meg',k)
    projected=_test_trial_projections(trials,indices,weights)
    direct,movable=_permuted_contrast(projected)
    # Confirms that the trial-level null uses exactly the same aggregation and
    # TRAINING-only normalisation/weights as the observed condition contrast.
    np.testing.assert_allclose(direct,cy,rtol=2e-4,atol=2e-4*max(1.,float(np.max(np.abs(cy)))))
    meta=fold['test']['ieeg'].metadata
    spatial_groups=meta['subject'].astype(str).to_numpy()
    if 'region' in meta:spatial_groups=spatial_groups+'|'+meta.region.fillna('unknown').astype(str).to_numpy()
    groups=[np.flatnonzero(spatial_groups==g) for g in np.unique(spatial_groups)]
    spatial_movable=any(len(g)>1 for g in groups)
    observed={'temporal_shift':_association(x,y),
              'condition_labels':_association(cx,cy,True),
              'spatial_correspondence':_association(px,py,True)}
    values={name:[] for name in observed}
    for _ in range(options.n_null):
        shift=int(rng.integers(1,len(trials.times)))
        ys=np.roll(y.reshape(2,len(trials.times),k),shift,axis=1).reshape(y.shape)
        values['temporal_shift'].append(_association(x,ys))
        if movable:
            perm,_=_permuted_contrast(projected,rng)
            values['condition_labels'].append(_association(cx,perm,True))
        if spatial_movable:
            permutation=np.arange(len(py))
            for group in groups:permutation[group]=rng.permutation(group)
            values['spatial_correspondence'].append(_association(px,py[permutation],True))
    rows=[]
    for name,null in values.items():
        rows.append(dict(test=name,observed=observed[name],tail_fraction=_tail(observed[name],null) if null else np.nan,
            n_null=len(null),interpretation='held-out conditional label permutation' if name=='condition_labels' else 'surrogate / correspondence diagnostic',
            note='' if null else 'No exchangeable groups, or null tests disabled.'))
    return pd.DataFrame(rows),values


def _average_null_tests(fold, scores, patterns, options, rng):
    """Diagnostics for the condition-averaged response; no condition-label null."""
    x, y = scores['test']['ieeg'], scores['test']['meg']
    ix = fold['test']['meg'].electrode_to_feature
    px, py = patterns['test']['ieeg'], patterns['test']['meg'][ix]
    meta = fold['test']['ieeg'].metadata
    labels = meta['subject'].astype(str).to_numpy()
    if 'region' in meta:
        labels = labels + '|' + meta.region.fillna('unknown').astype(str).to_numpy()
    groups = [np.flatnonzero(labels == g) for g in np.unique(labels)]
    values = dict(temporal_shift=[], spatial_correspondence=[])
    observed = dict(temporal_shift=_association(x,y), spatial_correspondence=_association(px,py,True))
    for _ in range(options.n_null):
        values['temporal_shift'].append(_association(x,np.roll(y,int(rng.integers(1,len(y))),axis=0)))
        if any(len(g)>1 for g in groups):
            perm = np.arange(len(py))
            for g in groups: perm[g] = rng.permutation(g)
            values['spatial_correspondence'].append(_association(px,py[perm],True))
    return pd.DataFrame([dict(test=name, observed=observed[name],
        tail_fraction=_tail(observed[name],null) if null else np.nan, n_null=len(null),
        interpretation='surrogate / correspondence diagnostic',
        note='' if null else 'No exchangeable spatial groups.') for name,null in values.items()]), values


def _reconstruction_fraction(dataset, scores, model, modality):
    """Own-modality orthogonal reconstruction, relative to TRAIN feature means."""
    error = baseline = 0.
    offset = 0
    for block in dataset.blocks():
        for start in range(0, block.shape[1], 1024):
            stop = min(start+1024, block.shape[1])
            section = slice(offset+start, offset+stop)
            centered = np.asarray(block[:, start:stop], dtype=float)-model[modality+'_mean'][section]
            predicted = (scores/model[modality+'_scale']) @ model[modality+'_weights'][section].T
            error += np.sum((centered-predicted)**2)
            baseline += np.sum(centered**2)
        offset += block.shape[1]
    return float(1-error/baseline) if baseline > 0 else np.nan


def _evaluate_fixed_model(datasets, scores, model, predictors):
    """Test covariance uses partition-centered scores, without refitting axes.

    Crosscov energy denominator covers ALL feature pairs, calculated with Gram
    matrices. Covariance is descriptive over condition/time rows, not trial-paired.
    """
    from cov_models_utils import _chunks
    n = datasets['ieeg'].n_observations
    centered = {m: scores[m]-scores[m].mean(0) for m in ('ieeg', 'meg')}
    covariance = centered['ieeg'].T@centered['meg']/(n-1)
    grams = {}
    metrics = {}
    for m in ('ieeg', 'meg'):
        gram = np.zeros((n, n))
        for block in _chunks(datasets[m]): gram += block@block.T
        grams[m] = gram*model[m+'_scale']**2
        metrics[m+'_reconstruction_fraction'] = _reconstruction_fraction(datasets[m], scores[m], model, m)
        other = 'meg' if m == 'ieeg' else 'ieeg'
        metrics['predict_'+m+'_q2'] = float(_q2(scores[other], datasets[m], model[m+'_mean'], predictors[m]))
    total_energy = float(np.sum(grams['ieeg']*grams['meg'])/(n-1)**2)
    captured = float(np.sum(covariance**2))
    metrics.update(crosscov_energy_fraction=float(np.clip(captured/total_energy, 0, 1)) if total_energy > 0 else np.nan,
        paired_crosscov_energy_fraction=float(np.clip(np.sum(np.diag(covariance)**2)/total_energy, 0, 1)) if total_energy > 0 else np.nan,
        mean_paired_covariance=float(np.mean(np.diag(covariance))),
        total_crosscov_energy=total_energy, captured_crosscov_energy=captured,
        mean_r=float(np.mean(_r(scores['ieeg'], scores['meg']))))
    return metrics, covariance


def _across_split_stability(all_scores, k):
    """Descriptive component matching across fits; never changes test evaluation."""
    from itertools import combinations
    from coverage_stability import matched_correlation
    rows, pairs = [], []
    for a, b in combinations(range(len(all_scores)), 2):
        for part in ('train', 'test'):
            for modality in ('ieeg', 'meg'):
                summary, assignment = matched_correlation(all_scores[a][part][modality], all_scores[b][part][modality], k)
                base = dict(fold_a=a, fold_b=b, partition=part, modality=modality, n_components=k)
                rows.append(dict(**base, matched_abs_r=summary['correlation'], status=summary['status']))
                pairs.extend(dict(**base, **pair) for pair in assignment)
    return (pd.DataFrame(rows, columns=['fold_a','fold_b','partition','modality','n_components','matched_abs_r','status']),
            pd.DataFrame(pairs, columns=['fold_a','fold_b','partition','modality','n_components','component_a','component_b','signed_r','abs_r','sign_b']))


def validate_plssvd(trials,meg_kind,options=None,output_dir='out/plssvd_eval_kfold',scratch_dir=None):
    """Fixed-k shuffled K-fold evaluation; cohort and matching never change.

    All folds (including fold 0) split trials. No tuning, subject subsampling,
    test-dependent component count, or test-dependent sign/component rematching.
    Test folds are disjoint; training folds overlap. Conditions are averaged
    with equal weight after trial averaging, before fitting and evaluation.
    """
    options = options or ValidationOptions()
    for name, minimum in [('n_components',1), ('repeats',2), ('n_null',0)]:
        value = getattr(options, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}.')
    if not np.isfinite(options.ridge) or options.ridge <= 0:
        raise ValueError('Require positive finite ridge.')
    if not trials.ieeg or not trials.meg or len(trials.conditions) != 2:
        raise ValueError('Evaluation requires both modalities and exactly two conditions.')
    for subjects in (trials.ieeg, trials.meg):
        if len({s.subject for s in subjects}) != len(subjects): raise ValueError('Participant IDs must be unique.')
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    if (out/'validation_options.json').exists() or any(out.glob('model_*.npz')):
        raise FileExistsError('Choose a new evaluation output directory; existing results are not overwritten.')
    settings = dict(**options.__dict__, schema_version=3, meg_kind=meg_kind,
        design='fixed-k shuffled disjoint test folds; condition average; no tuning',
        condition_mode='average',
        partitions=['train','test'],
        ieeg_preprocessing='fixed multiplier 1000', meg_preprocessing='zscore of training condition averages',
        cohort='all cached participants in every fold; fixed participant/source assignment',
        scope='new trials from the same participants; not held-out subjects, times or locations')
    (out/'validation_options.json').write_text(json.dumps(settings, indent=2))
    k = options.n_components
    matching_seq, split_seq, null_seq = np.random.SeedSequence(options.seed).spawn(3)
    matching_seed = int(np.random.default_rng(matching_seq).integers(2**31-1))
    summaries, components, split_rows, participants, fold_metrics, all_scores = [], [], [], [], [], []
    null_summary = pd.DataFrame(columns=['test','observed','tail_fraction','n_null','interpretation','note'])
    null_values = {}; primary_scores = {}
    rng = np.random.default_rng(split_seq)
    assignments = {m: {s.subject: _subject_folds(s, rng, options.split_unit, options.repeats)
                       for s in getattr(trials, m)} for m in ('ieeg','meg')}
    for repeat in range(options.repeats):
        indices = {m: {subject: folds[repeat] for subject, folds in subjects.items()}
                   for m, subjects in assignments.items()}
        for modality in ('ieeg','meg'):
            for subject in getattr(trials, modality):
                participants.append(dict(repeat=repeat, modality=modality, subject=subject.subject))
                for part in ('train','test'):
                    for ci, ix in enumerate(indices[modality][subject.subject][part]):
                        for trial in ix:
                            split_rows.append(dict(repeat=repeat, modality=modality, subject=subject.subject,
                                condition=trials.conditions[ci], trial_index=int(trial), partition=part,
                                split_group=subject.split_groups[ci][trial] if subject.split_groups[ci] is not None else ''))
        with _temporary_fold(trials, meg_kind, indices, matching_seed, scratch_dir, condition_mode="average") as prepared:
            fold, scalers, audit, pairing = prepared
            model = _fit(fold['train'], k, options)
            if model['k_max'] != k:
                raise ValueError(f'Fold {repeat}: training rank supports only {model["k_max"]} components, '
                                 f'but fixed n_components={k}. Choose a supported fixed count; no automatic selection.')
            scores = {part: {m: _project(ds, model, m) for m, ds in datasets.items()} for part, datasets in fold.items()}
            predictors = {m: _predictor(scores['train']['meg' if m == 'ieeg' else 'ieeg'],
                          fold['train'][m], model[m+'_mean'], options.ridge) for m in ('ieeg','meg')}
            evaluations, covariances = {}, {}
            for part in ('train','test'):
                evaluations[part], covariances[part] = _evaluate_fixed_model(fold[part], scores[part], model, predictors)
                fold_metrics.append(dict(repeat=repeat, partition=part, n_components=k, **evaluations[part]))
            patterns = {'test': {m: _patterns(ds, scores['test'][m]) for m, ds in fold['test'].items()}}
            train_r = _r(scores['train']['ieeg'], scores['train']['meg'])
            test_r = _r(scores['test']['ieeg'], scores['test']['meg'])
            row = dict(repeat=repeat, n_components=k, n_ieeg=len(trials.ieeg), n_meg=len(trials.meg),
                n_meg_contributing=audit.meg_subject.nunique() if meg_kind in ('paired_coverage','random_control') else len(trials.meg),
                train_mean_r=float(np.mean(train_r)), test_mean_r=float(np.mean(test_r)),
                predict_ieeg_q2=evaluations['test']['predict_ieeg_q2'], predict_meg_q2=evaluations['test']['predict_meg_q2'],
                **{part+'_'+key: value for part, vals in evaluations.items() for key, value in vals.items()
                   if key != 'mean_r'})
            train_cov = evaluations['train']['mean_paired_covariance']
            row['paired_covariance_retention'] = evaluations['test']['mean_paired_covariance']/train_cov if train_cov > 0 else np.nan
            summaries.append(row)
            for pc in range(k):
                components.append(dict(repeat=repeat, component=pc+1, train_r=train_r[pc], test_r=test_r[pc],
                    train_covariance=covariances['train'][pc,pc], test_covariance=covariances['test'][pc,pc]))
            all_scores.append({part: {m: scores[part][m].copy() for m in ('ieeg','meg')} for part in ('train','test')})
            if repeat == 0:
                primary_scores = {part: {m: v.copy() for m,v in values.items()} for part,values in scores.items()}
                if options.n_null:
                    null_summary, null_values = _average_null_tests(fold, scores, patterns, options, np.random.default_rng(null_seq))
            audit.to_csv(out/f'matching_{repeat:03d}.csv', index=False)
            (out/f'pairing_{repeat:03d}.json').write_text(json.dumps(pairing, indent=2))
            for m in ('ieeg','meg'): fold['train'][m].metadata.to_csv(out/f'{m}_features_{repeat:03d}.csv', index=False)
            np.savez_compressed(out/f'model_{repeat:03d}.npz', **model,
                predict_ieeg=predictors['ieeg'], predict_meg=predictors['meg'],
                **{f'{part}_{m}': value for part, vals in scores.items() for m, value in vals.items()},
                **{part+'_score_crosscovariance': value for part, value in covariances.items()})
            np.savez_compressed(out/f'preprocessing_{repeat:03d}.npz', **{
                f'{m}_{subject}_{label}': value for m, subjects in scalers.items() for subject, params in subjects.items()
                for label, value in zip(['mean','std','multiplier'], params)})
        print(f'Split {repeat+1}/{options.repeats}: fixed k={k}; test mean r={row["test_mean_r"]:.3f}; '
              f'test crosscov energy fraction={row["test_crosscov_energy_fraction"]:.3f}', flush=True)
    stability, stability_pairs = _across_split_stability(all_scores, k)
    fold_metrics = pd.DataFrame(fold_metrics)
    long = fold_metrics.melt(id_vars=['repeat','partition','n_components'], var_name='metric', value_name='value')
    metric_summary = long.groupby(['partition','metric'], sort=False)['value'].agg(['count','mean','std','median','min','max']).reset_index()
    result = dict(summary=pd.DataFrame(summaries), components=pd.DataFrame(components), fold_metrics=fold_metrics,
        metric_summary=metric_summary, fold_stability=stability, fold_component_pairs=stability_pairs,
        split_audit=pd.DataFrame(split_rows), participants=pd.DataFrame(participants), null_tests=null_summary,
        null_distributions=null_values, primary_scores=primary_scores, times=trials.times, conditions=trials.conditions,
        validation_options=settings)
    for name in ('summary','components','fold_metrics','metric_summary','fold_stability','fold_component_pairs','split_audit','participants'):
        result[name].to_csv(out/f'{name}.csv', index=False)
    if options.n_null:
        null_summary.to_csv(out/'primary_null_tests.csv', index=False)
        np.savez_compressed(out/'primary_null_distributions.npz', **null_values)
    np.savez_compressed(out/'trial_axes.npz', times=trials.times, conditions=trials.conditions)
    (out/'COMPLETE.json').write_text(json.dumps(dict(schema_version=3, repeats=options.repeats, n_components=k)))
    return result


def load_plssvd_results(output_dir, cache_dir=None):
    """Reload a completed plssvd_eval.py run without reading raw trials.

    Only result tables, axes, primary scores and null distributions are loaded.
    Large model weights/prediction maps remain on disk; ``artifacts`` provides
    paths for optional np.load/pd.read_csv access. For older output directories
    lacking trial_axes.npz, cache_dir may supply axes from manifest.json only.
    The returned dictionary works directly with plot_plssvd_validation.
    """
    root=Path(output_dir).expanduser().resolve()
    options=json.loads((root/'validation_options.json').read_text())
    fixed = options.get('schema_version', 1) >= 2
    if fixed and not (root/'COMPLETE.json').is_file():
        raise FileNotFoundError('Fixed-k evaluation is incomplete: COMPLETE.json is absent.')
    table_names=(('summary','components','fold_metrics','metric_summary','fold_stability',
                  'fold_component_pairs','split_audit','participants') if fixed else
                 ('summary','components','selection','split_audit','participants'))
    required=[root/f'{name}.csv' for name in table_names]
    required += [root/'validation_options.json',root/'model_000.npz']
    missing=[str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Incomplete PLSSVD output folder. Copy the completed run outputs: '
                                + ', '.join(missing))
    options=json.loads((root/'validation_options.json').read_text())
    result={name:pd.read_csv(root/f'{name}.csv') for name in table_names}
    if sorted(result['summary']['repeat'].unique().tolist())!=list(range(options['repeats'])):
        raise ValueError('Saved summary does not contain every configured repetition.')
    axes=root/'trial_axes.npz'
    if axes.is_file():
        with np.load(axes,allow_pickle=False) as saved:
            times=saved['times'];conditions=tuple(saved['conditions'].tolist())
    elif cache_dir is not None:
        manifest=json.loads((Path(cache_dir)/'manifest.json').read_text())
        times=np.asarray(manifest['times']);conditions=tuple(manifest['config']['conditions'])
    else:
        raise FileNotFoundError(f'Missing {axes}. Copy trial_axes.npz from the cluster run, '
                                'or provide cache_dir for an older run.')
    if times.ndim!=1 or not len(times) or len(conditions)!=2:
        raise ValueError('Invalid saved time/condition axes.')
    k_column = 'n_components' if fixed else 'selected_k'
    k=int(result['summary'].loc[result['summary']['repeat']==0,k_column].iloc[0])
    primary={}
    with np.load(root/'model_000.npz',allow_pickle=False) as saved:
        for part in options.get('partitions', ['train','tune','test_a','test_b','test']):
            primary[part]={}
            for modality in ('ieeg','meg'):
                scores=saved[f'{part}_{modality}']
                if scores.shape!=(len(times)*(1 if options.get('condition_mode') == 'average' else len(conditions)),k):
                    raise ValueError(f'Saved {part}/{modality} scores disagree with axes or component count.')
                primary[part][modality]=scores
    null_tests=pd.DataFrame();null_distributions={}
    if options.get('n_null',0):
        null_tests=pd.read_csv(root/'primary_null_tests.csv')
        with np.load(root/'primary_null_distributions.npz',allow_pickle=False) as saved:
            null_distributions={name:saved[name].tolist() for name in saved.files}
        if not set(null_tests['test'])<=set(null_distributions):
            raise ValueError('Missing saved distributions for one or more null tests.')
    result.update(times=times,conditions=conditions,primary_scores=primary,
                  null_tests=null_tests,null_distributions=null_distributions,
                  validation_options=options,output_dir=root,
                  artifacts={path.name:path for path in sorted(root.iterdir()) if path.is_file()})
    for name in ('trial_counts','electrode_metadata'):
        path=root/f'{name}.csv'
        result[name]=pd.read_csv(path) if path.is_file() else pd.DataFrame()
    result['run_config']=json.loads((root/'run_config.json').read_text()) if (root/'run_config.json').is_file() else {}
    return result


def plot_plssvd_validation(result, output_dir=None, show=True):
    """Plot validation; optionally save PNG/PDF figures and disable display."""
    def finish_figure(fig, name):
        if output_dir is not None:
            destination=Path(output_dir)
            destination.mkdir(parents=True,exist_ok=True)
            for extension in ('png','pdf'):
                fig.savefig(destination/f'{name}.{extension}',dpi=200)
        if show:
            plt.show()
        plt.close(fig)

    if result.get('validation_options', {}).get('schema_version', 1) >= 2:
        _plot_fixed_evaluation(result, finish_figure)
    else:
        s=result['summary'];fig,axes=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
        for col,label in [('train_mean_r','Training'),('test_mean_r','Held-out')]:axes[0].plot(s.repeat,s[col],'o-',label=label)
        axes[0].set(title='Cross-modal correspondence',ylabel='Mean paired Pearson r',xlabel='Repetition',ylim=(-1,1));axes[0].legend()
        for col,label in [('predict_ieeg_q2','Predict iEEG'),('predict_meg_q2','Predict MEG')]:axes[1].plot(s.repeat,s[col],'o-',label=label)
        axes[1].axhline(0,color='grey',ls=':');axes[1].set(title='Held-out prediction',ylabel='Q² vs training-mean baseline',xlabel='Repetition');axes[1].legend()
        for col,label in [('ieeg_split_half_r','iEEG'),('meg_split_half_r','MEG')]:axes[2].plot(s.repeat,s[col],'o-',label=label)
        axes[2].set(title='Test-half temporal reliability',ylabel='Pearson r',xlabel='Repetition',ylim=(-1,1));axes[2].legend()
        finish_figure(fig,'validation_summary')
    primary=result['primary_scores'];k=min(3,primary['test']['ieeg'].shape[1]);times=result['times']
    averaged = result.get('validation_options', {}).get('condition_mode') == 'average'
    fig,axes=plt.subplots(k,1 if averaged else 2,figsize=(14,3*k),squeeze=False,constrained_layout=True)
    for pc in range(k):
        for modality,color in [('ieeg','navy'),('meg','darkorange')]:
            for part,style in [('train','--'),('test','-')]:
                values=primary[part][modality].reshape(1 if averaged else 2,len(times),-1)
                mean=values.mean(0)[:,pc]
                # Scale displays by TRAIN score SD, never by test SD for modelling.
                sd=primary['train'][modality][:,pc].std() or 1.
                axes[pc,0].plot(times,mean/sd,color=color,ls=style,label=f'{modality} {part}')
                if not averaged: axes[pc,1].plot(times,(values[1,:,pc]-values[0,:,pc])/sd,color=color,ls=style,label=f'{modality} {part}')
        axes[pc,0].set(title=f'Component {pc+1}: condition mean',xlabel='Time (s)',ylabel='Score / training SD')
        if not averaged: axes[pc,1].set(title=f'Component {pc+1}: condition 2 − condition 1',xlabel='Time (s)',ylabel='Contrast / training SD')
    axes[0,0].legend(fontsize=8)
    finish_figure(fig,'primary_time_courses')
    if not result['null_tests'].empty:
        fig,axes=plt.subplots(1,len(result['null_tests']),figsize=(15,4),squeeze=False,constrained_layout=True)
        for ax,row in zip(axes.ravel(),result['null_tests'].itertuples()):
            values=result['null_distributions'][row.test]
            if values:ax.hist(values,bins=min(25,len(values)),color='grey',alpha=.6)
            ax.axvline(row.observed,color='crimson',label='Observed')
            ax.set(title=row.test,xlabel='Predeclared test statistic',ylabel='Null draws');ax.legend()
        finish_figure(fig,'primary_null_tests')


def _plot_fixed_evaluation(result, finish):
    """Train/test performance and descriptive consistency across random splits."""
    table = result['fold_metrics']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for column, modality in enumerate(('ieeg','meg')):
        for row, metric, title in [(0, modality+'_reconstruction_fraction', 'Own-signal reconstruction'),
                                   (1, 'predict_'+modality+'_q2', 'Cross-modal prediction')]:
            ax = axes[row,column]
            for part, style in [('train','--'),('test','-')]:
                values = table[table.partition == part]
                ax.plot(values.repeat+1, values[metric], 'o'+style, label=part)
            ax.axhline(0, color='grey', ls=':')
            ax.set(title=f'{modality.upper()}: {title}', xlabel='Random split',
                   ylabel='Retained signal fraction' if row == 0 else 'Q² vs training-mean baseline')
            ax.legend()
    fig.suptitle(f'Fixed {result["validation_options"]["n_components"]} components: do both datasets generalize?')
    finish(fig, 'heldout_model_goodness')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for ax, metric, title in zip(axes,
            ['crosscov_energy_fraction','mean_paired_covariance','mean_r'],
            ['Full cross-covariance energy retained','Mean paired score covariance','Mean paired score Pearson r']):
        for part, style in [('train','--'),('test','-')]:
            values = table[table.partition == part]
            ax.plot(values.repeat+1, values[metric], 'o'+style, label=part)
        ax.set(title=title, xlabel='Random split', ylabel=metric)
        ax.axhline(0, color='grey', ls=':'); ax.legend()
    finish(fig, 'heldout_crosscovariance')

    matrices = {}
    for part in ('train','test'):
        values = result['primary_scores'][part]
        x = values['ieeg']-values['ieeg'].mean(0)
        y = values['meg']-values['meg'].mean(0)
        matrices[part] = x.T@y/(len(x)-1)
    limit = max(float(np.max(np.abs(v))) for v in matrices.values()) or 1.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, (part, matrix) in zip(axes, matrices.items()):
        im = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit)
        k = len(matrix)
        ax.set(title=f'First split: {part}', xlabel='MEG component', ylabel='iEEG component',
               xticks=range(k), xticklabels=range(1,k+1), yticks=range(k), yticklabels=range(1,k+1))
        fig.colorbar(im, ax=ax, label='Native score cross-covariance')
    finish(fig, 'primary_crosscovariance')

    columns = ['ieeg_reconstruction_fraction','meg_reconstruction_fraction','predict_ieeg_q2',
               'predict_meg_q2','crosscov_energy_fraction','mean_r']
    labels = ['iEEG reconstruction','MEG reconstruction','Predict iEEG Q²','Predict MEG Q²','Crosscov energy','Paired score r']
    test = table[table.partition == 'test']
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for index, (metric, label) in enumerate(zip(columns, labels)):
        values = test[metric].dropna().to_numpy()
        if len(values):
            ax.scatter(index+np.linspace(-.12,.12,len(values)), values, alpha=.7)
            ax.plot(index, np.median(values), 'k_', markersize=20)
            ax.vlines(index, values.min(), values.max(), alpha=.4)
    ax.axhline(0, color='grey', ls=':')
    ax.set(xticks=range(len(labels)), xticklabels=labels, ylabel='Held-out metric value',
           title='Performance across splits: dots = splits, black mark = median, line = range (not CI)')
    ax.tick_params(axis='x', labelrotation=20)
    finish(fig, 'fold_performance_consistency')

    count = result['validation_options']['repeats']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, modality in zip(axes, ('ieeg','meg')):
        matrix = np.full((count,count), np.nan); np.fill_diagonal(matrix, 1.)
        data = result['fold_stability']
        for row in data[(data.modality == modality) & (data.partition == 'test')].itertuples():
            matrix[row.fold_a,row.fold_b] = matrix[row.fold_b,row.fold_a] = row.matched_abs_r
        im = ax.imshow(matrix, cmap='viridis', vmin=0, vmax=1)
        ax.set(title=modality.upper(), xlabel='Random split', ylabel='Random split',
               xticks=range(count), xticklabels=range(1,count+1), yticks=range(count), yticklabels=range(1,count+1))
        fig.colorbar(im, ax=ax, label='Matched mean |r| of test score time courses')
    fig.suptitle('Refitted temporal-pattern consistency; disjoint test folds, overlapping training folds'
                 if result['validation_options'].get('schema_version',1) >= 3 else
                 'Refitted temporal-pattern consistency; test sets can overlap across splits')
    finish(fig, 'fold_temporal_stability')
