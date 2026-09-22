"""Trial-held-out PLSSVD validation. No time-sample train/test splitting.

All trainable normalisation, PLS weights and prediction maps are training-only.
Component count is selected on an independent tuning partition. Test halves
are reserved for test performance and conditional split-half reliability.
"""
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from utils_updated import Dataset, coordinates_mm, construct_five_datasets
from cov_models_utils import _gram, _spectrum, _weights


@dataclass
class ValidationOptions:
    repeats: int = 5
    candidates: tuple = (1, 2, 3, 5, 10)
    n_null: int = 199
    seed: int = 2026
    subject_fraction: float = .8  # repeats after the primary full-cohort run
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
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved['config']!=config:
            raise ValueError('Cache configuration changed. Choose a new cache directory.')
        for signature in saved['source_files']:
            if Path(signature['path']).exists() and _signature(signature['path'])!=signature:
                raise ValueError('A source file changed. Choose a new cache directory to re-export trials.')
        return load_trial_cache(root)
    if any(root.iterdir()):
        raise ValueError('Incomplete/nonempty trial cache: choose a new directory.')
    extra=None
    if trial_metadata_csv:
        extra=pd.read_csv(trial_metadata_csv,dtype={'subject':str,'modality':str})
        keys=['modality','subject','condition','trial_index']
        if not set(keys)<=set(extra) or extra.duplicated(keys).any():
            raise ValueError('Trial metadata needs unique modality/subject/condition/trial_index keys.')
    times=None; records=[]; sources=[]
    for modality, subjects in [('ieeg',ieeg_subjects),('meg',meg_subjects)]:
        for subject in subjects:
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
                if len(a)<8:
                    raise ValueError(f'{subject}, condition {c}: need >=8 trials for train/tune/two test halves.')
                fname=f'{modality}_{subject}_condition{c}.npy'
                np.save(root/fname,np.asarray(a,dtype=np.float32));files.append(fname)
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
            print(f'Cached {modality} {subject}: {[len(a) for a in arrays]} trials',flush=True)
            del arrays, a
            if modality=='meg':del out
            else:del epochs
    manifest.write_text(json.dumps(dict(config=config,times=times.tolist(),records=records,source_files=sources),indent=2))
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
    """Return independent train/tune/test_a/test_b indices per condition."""
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


def _trial_mean(array,indices):
    # Accumulate without fancy indexing a whole subject's trial cube.
    mean=np.zeros(array.shape[1:],float)
    for i in indices:mean+=array[i]
    return mean/len(indices)


def _build_fold(trials,meg_kind,split_indices,root,seed):
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
                for c,(a,indices) in enumerate(zip(s.data,parts)):
                    mean=train[c] if name=='train' else _trial_mean(a,indices)
                    out[c]=((mean-mu[0])/sd[0])*multiplier
                out.flush();prepared[modality][s.subject][name]=out
    metadata=pd.concat([s.metadata for s in trials.ieeg],ignore_index=True)
    coords=np.concatenate([s.positions for s in trials.ieeg]);owners=np.concatenate([np.repeat(s.subject,len(s.positions)) for s in trials.ieeg])
    context={'times':trials.times,'load_config':{'conditions':list(trials.conditions)}}
    datasets={};audit=None;pairing=None
    for part in ('train','tune','test_a','test_b','test'):
        ieeg_arrays=[prepared['ieeg'][s.subject][part] for s in trials.ieeg]
        ieeg=Dataset('iEEG',ieeg_arrays,metadata,np.arange(len(coords)),'stack',source_data=context)
        selected,audit,pairing=construct_five_datasets(
            [prepared['meg'][s.subject][part] for s in trials.meg],
            [s.positions for s in trials.meg],[s.subject for s in trials.meg],coords,owners,
            seed=seed,pairing=pairing,condition_mode='stack',kinds=[meg_kind])
        meg=selected[meg_kind];meg.source_data=context
        datasets[part]={'ieeg':ieeg,'meg':meg}
    return datasets,scalers,audit,pairing


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
    out=np.zeros((dataset.n_observations,model['k_max']));offset=0
    for x in dataset.blocks():
        width=x.shape[1];section=slice(offset,offset+width)
        out+=(x-model[modality+'_mean'][section])@model[modality+'_weights'][section]*model[modality+'_scale']
        offset+=width
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


def validate_plssvd(trials,meg_kind,options=None,output_dir='out/plssvd_eval'):
    """Run nested repeated trial holdouts, test-half reliability and primary nulls.

    Primary repetition 0 uses all participants. Later repetitions use participant
    subsamples and new pairings. Participants are NEVER held out from training:
    all tests use new trials from the same selected participants. Resampling
    spread is not a population confidence interval. Nulls are performed only on
    the predeclared primary repetition, not combined across overlapping repeats.
    """
    options=options or ValidationOptions();out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if options.repeats<1 or options.n_null<0 or not 0<options.subject_fraction<=1 or options.ridge<=0:
        raise ValueError('Invalid repeat/null counts, participant fraction or ridge penalty.')
    candidates=sorted(set(options.candidates))
    if not candidates or any(int(k)!=k or k<1 for k in candidates):raise ValueError('Candidates must be positive integers.')
    summaries=[];components=[];selections=[];split_rows=[];participants=[]
    null_summary=pd.DataFrame();null_values={};primary_scores={}
    master=np.random.SeedSequence(options.seed)
    for repeat,seed_sequence in enumerate(master.spawn(options.repeats)):
        rng=np.random.default_rng(seed_sequence)
        def subset(subjects,n):return [subjects[i] for i in sorted(rng.choice(len(subjects),n,replace=False))]
        ni=len(trials.ieeg) if repeat==0 else max(1,int(np.ceil(len(trials.ieeg)*options.subject_fraction)))
        nm=len(trials.meg) if repeat==0 else max(ni,int(np.ceil(len(trials.meg)*options.subject_fraction)))
        if nm>len(trials.meg):raise ValueError('Need >= as many MEG as iEEG participants for one-to-one matching.')
        active=TrialData(subset(trials.ieeg,ni),subset(trials.meg,nm),trials.times,trials.conditions)
        indices={modality:{s.subject:_split_subject(s,rng,options.split_unit) for s in getattr(active,modality)} for modality in ('ieeg','meg')}
        for modality in ('ieeg','meg'):
            for s in getattr(active,modality):
                participants.append(dict(repeat=repeat,modality=modality,subject=s.subject))
                for part in ('train','tune','test_a','test_b'):
                    for ci,ix in enumerate(indices[modality][s.subject][part]):
                        for trial in ix:
                            split_rows.append(dict(repeat=repeat,modality=modality,subject=s.subject,condition=active.conditions[ci],
                                trial_index=int(trial),partition=part,
                                split_group=s.split_groups[ci][trial] if s.split_groups[ci] is not None else ''))
        matching_seed=int(rng.integers(0,2**31-1))
        with TemporaryDirectory(prefix='fold_means_',dir=out) as scratch:
            fold,scalers,audit,pairing=_build_fold(active,meg_kind,indices,scratch,matching_seed)
            model=_fit(fold['train'],max(candidates),options)
            allowed=[k for k in candidates if k<=model['k_max']]
            if not allowed:raise ValueError('No candidate component count supported by training data.')
            if len(allowed)<len(candidates):warnings.warn('Some component candidates exceed training rank and were excluded.')
            scores={part:{modality:_project(data,model,modality) for modality,data in datasets.items()} for part,datasets in fold.items()}
            best=None;best_value=-np.inf;predictors={}
            for k in allowed:
                bx=_predictor(scores['train']['meg'][:,:k],fold['train']['ieeg'],model['ieeg_mean'],options.ridge)
                by=_predictor(scores['train']['ieeg'][:,:k],fold['train']['meg'],model['meg_mean'],options.ridge)
                qx=_q2(scores['tune']['meg'][:,:k],fold['tune']['ieeg'],model['ieeg_mean'],bx)
                qy=_q2(scores['tune']['ieeg'][:,:k],fold['tune']['meg'],model['meg_mean'],by)
                value=(qx+qy)/2
                selections.append(dict(repeat=repeat,k=k,tune_predict_ieeg_q2=qx,tune_predict_meg_q2=qy,selection_score=value))
                if value>best_value+1e-12:
                    best,best_value=k,value;predictors={'ieeg':bx,'meg':by}
            if best is None:raise ValueError('No finite tuning prediction score.')
            k=best
            patterns={part:{modality:_patterns(data,scores[part][modality][:,:k]) for modality,data in fold[part].items()} for part in ('test_a','test_b','test')}
            train_r=_r(scores['train']['ieeg'][:,:k],scores['train']['meg'][:,:k])
            test_r=_r(scores['test']['ieeg'][:,:k],scores['test']['meg'][:,:k])
            contrast={modality:_contrast(scores['test'][modality][:,:k],len(trials.times)) for modality in ('ieeg','meg')}
            contrast_r=_r(contrast['ieeg'],contrast['meg'])
            temporal_reliability={m:_r(scores['test_a'][m][:,:k],scores['test_b'][m][:,:k]) for m in ('ieeg','meg')}
            contrast_reliability={m:_r(_contrast(scores['test_a'][m][:,:k],len(trials.times)),
                                      _contrast(scores['test_b'][m][:,:k],len(trials.times))) for m in ('ieeg','meg')}
            spatial_reliability={m:_r(patterns['test_a'][m],patterns['test_b'][m]) for m in ('ieeg','meg')}
            qx=_q2(scores['test']['meg'][:,:k],fold['test']['ieeg'],model['ieeg_mean'],predictors['ieeg'])
            qy=_q2(scores['test']['ieeg'][:,:k],fold['test']['meg'],model['meg_mean'],predictors['meg'])
            summaries.append(dict(repeat=repeat,selected_k=k,n_ieeg=ni,n_meg=nm,train_mean_r=float(np.mean(train_r)),
                test_mean_r=float(np.mean(test_r)),test_mean_abs_contrast_r=float(np.mean(np.abs(contrast_r))),
                predict_ieeg_q2=qx,predict_meg_q2=qy,
                ieeg_split_half_r=float(np.mean(temporal_reliability['ieeg'])),meg_split_half_r=float(np.mean(temporal_reliability['meg'])),
                ieeg_pattern_split_half_r=float(np.mean(spatial_reliability['ieeg'])),meg_pattern_split_half_r=float(np.mean(spatial_reliability['meg']))))
            for pc in range(k):
                components.append(dict(repeat=repeat,component=pc+1,train_r=train_r[pc],test_r=test_r[pc],contrast_r=contrast_r[pc],
                    ieeg_split_half_r=temporal_reliability['ieeg'][pc],meg_split_half_r=temporal_reliability['meg'][pc],
                    ieeg_contrast_split_half_r=contrast_reliability['ieeg'][pc],meg_contrast_split_half_r=contrast_reliability['meg'][pc],
                    ieeg_pattern_split_half_r=spatial_reliability['ieeg'][pc],meg_pattern_split_half_r=spatial_reliability['meg'][pc],
                    ieeg_contrast_rms=float(np.sqrt(np.mean(contrast['ieeg'][:,pc]**2))),meg_contrast_rms=float(np.sqrt(np.mean(contrast['meg'][:,pc]**2)))))
            if repeat==0:
                primary_scores={part:{m:s[:,:k].copy() for m,s in v.items()} for part,v in scores.items()}
                if options.n_null:
                    null_summary,null_values=_null_tests(active,fold,model,scalers,audit,meg_kind,indices,scores,patterns,k,options,rng)
                    null_summary.to_csv(out/'primary_null_tests.csv',index=False)
                    np.savez_compressed(out/'primary_null_distributions.npz',**null_values)
            audit.to_csv(out/f'matching_{repeat:03d}.csv',index=False)
            fold['train']['ieeg'].metadata.to_csv(out/f'ieeg_features_{repeat:03d}.csv',index=False)
            fold['train']['meg'].metadata.to_csv(out/f'meg_features_{repeat:03d}.csv',index=False)
            np.savez_compressed(out/f'model_{repeat:03d}.npz',ieeg_weights=model['ieeg_weights'][:,:k],meg_weights=model['meg_weights'][:,:k],
                ieeg_mean=model['ieeg_mean'],meg_mean=model['meg_mean'],ieeg_scale=model['ieeg_scale'],meg_scale=model['meg_scale'],
                predict_ieeg=predictors['ieeg'],predict_meg=predictors['meg'],
                **{f'{part}_{modality}':s[:,:k] for part,v in scores.items() for modality,s in v.items()})
            np.savez_compressed(out/f'preprocessing_{repeat:03d}.npz',
                **{f'{modality}_{subject}_{label}':value for modality,subjects in scalers.items() for subject,params in subjects.items()
                   for label,value in zip(['mean','std','multiplier'],params)})
        print(f'Repetition {repeat+1}/{options.repeats}: selected {k} components; held-out mean r={summaries[-1]["test_mean_r"]:.3f}',flush=True)
    result={'summary':pd.DataFrame(summaries),'components':pd.DataFrame(components),'selection':pd.DataFrame(selections),
            'split_audit':pd.DataFrame(split_rows),'participants':pd.DataFrame(participants),'null_tests':null_summary,
            'null_distributions':null_values,'primary_scores':primary_scores,'times':trials.times,'conditions':trials.conditions}
    for name in ('summary','components','selection','split_audit','participants'):result[name].to_csv(out/f'{name}.csv',index=False)
    (out/'validation_options.json').write_text(json.dumps(dict(**options.__dict__,meg_kind=meg_kind,
        ieeg_preprocessing='fixed multiplier 1000',meg_preprocessing='zscore of training condition averages',
        scope='new trials from selected training participants; not held-out participants'),indent=2))
    return result


def plot_plssvd_validation(result):
    """Primary train/test curves, condition contrasts, reliability and nulls."""
    s=result['summary'];fig,axes=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
    for col,label in [('train_mean_r','Training'),('test_mean_r','Held-out')]:axes[0].plot(s.repeat,s[col],'o-',label=label)
    axes[0].set(title='Cross-modal correspondence',ylabel='Mean paired Pearson r',xlabel='Repetition',ylim=(-1,1));axes[0].legend()
    for col,label in [('predict_ieeg_q2','Predict iEEG'),('predict_meg_q2','Predict MEG')]:axes[1].plot(s.repeat,s[col],'o-',label=label)
    axes[1].axhline(0,color='grey',ls=':');axes[1].set(title='Held-out prediction',ylabel='Q² vs training-mean baseline',xlabel='Repetition');axes[1].legend()
    for col,label in [('ieeg_split_half_r','iEEG'),('meg_split_half_r','MEG')]:axes[2].plot(s.repeat,s[col],'o-',label=label)
    axes[2].set(title='Test-half temporal reliability',ylabel='Pearson r',xlabel='Repetition',ylim=(-1,1));axes[2].legend()
    plt.show();plt.close(fig)
    primary=result['primary_scores'];k=min(3,primary['test']['ieeg'].shape[1]);times=result['times']
    fig,axes=plt.subplots(k,2,figsize=(14,3*k),squeeze=False,constrained_layout=True)
    for pc in range(k):
        for modality,color in [('ieeg','navy'),('meg','darkorange')]:
            for part,style in [('train','--'),('test','-')]:
                values=primary[part][modality].reshape(2,len(times),-1)
                mean=values.mean(0)[:,pc]
                # Scale displays by TRAIN score SD, never by test SD for modelling.
                sd=primary['train'][modality][:,pc].std() or 1.
                axes[pc,0].plot(times,mean/sd,color=color,ls=style,label=f'{modality} {part}')
                axes[pc,1].plot(times,(values[1,:,pc]-values[0,:,pc])/sd,color=color,ls=style,label=f'{modality} {part}')
        axes[pc,0].set(title=f'Component {pc+1}: condition mean',xlabel='Time (s)',ylabel='Score / training SD')
        axes[pc,1].set(title=f'Component {pc+1}: condition 2 − condition 1',xlabel='Time (s)',ylabel='Contrast / training SD')
    axes[0,0].legend(fontsize=8);plt.show();plt.close(fig)
    if not result['null_tests'].empty:
        fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
        for ax,row in zip(axes,result['null_tests'].itertuples()):
            values=result['null_distributions'][row.test]
            if values:ax.hist(values,bins=min(25,len(values)),color='grey',alpha=.6)
            ax.axvline(row.observed,color='crimson',label='Observed')
            ax.set(title=row.test,xlabel='Predeclared test statistic',ylabel='Null draws');ax.legend()
        plt.show();plt.close(fig)
