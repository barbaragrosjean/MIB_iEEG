"""Paired-coverage cosine profiles and normalized magnitude semivariance.

All inference is conditional on fitted group-model weights. Spatial shuffles
remove within-subject/hemisphere association, not MEG reconstruction smoothness.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


def _mean(values, axis=0):
    count = np.sum(np.isfinite(values), axis=axis)
    result = np.divide(np.nansum(values, axis=axis), count,
                     out=np.full(np.shape(count), np.nan, dtype=float), where=count > 0)
    return float(result) if result.ndim == 0 else result


def analyze_spatial_profiles(weights, ieeg_metadata, meg_metadata, *, bin_edges=None,
                             n_permutations=199, n_bootstrap=500, seed=2026,
                             min_norm_quantile=0.,
                             spatial_scope="within_participant", fold_hemispheres=False,
                             target_hemisphere="left"):
    """Matched-pair curves and paired participant bootstrap.

    weights has iEEG and MEG entries from the same selected group-model fit.
    Rows must retain paired_coverage electrode correspondence. Duplicate MEG
    source matches within a participant are removed jointly from both modalities.
    Curves use only pairs in the same distance bin in both modalities. All
    nonempty participant curves receive equal weight.
    spatial_scope='pooled' treats retained participants as one analysis block:
    cross-participant pairs are allowed, separately per hemisphere.
    Shuffling and magnitude variance use that pooled hemisphere. Every pair has
    equal weight; participant bootstrap intervals are unavailable. Deduplication
    remains within original participants because their measurements are distinct.
    fold_hemispheres reflects x to -abs(x) (left) or abs(x) (right), after
    original-coordinate matching exclusions and deduplication. Original sides
    then share distance, shuffle, variance groups within each block.
    Weights and input metadata are unchanged; zero folded distances are excluded.

    """
    from vs3_1_modality_specificity import spatial_metadata, fold_coordinates
    if target_hemisphere not in ('left', 'right'):
        raise ValueError("target_hemisphere must be 'left' or 'right'.")
    if spatial_scope not in ('within_participant', 'pooled'):
        raise ValueError("spatial_scope must be 'within_participant' or 'pooled'.")
    edges = np.asarray(np.arange(0, 101, 10) if bin_edges is None else bin_edges, float)
    if edges.ndim != 1 or len(edges)<2 or edges[0]!=0 or np.any(np.diff(edges)<=0) or not np.isfinite(edges).all():
        raise ValueError('Distance edges must be finite, start at zero, and strictly increase.')
    if n_permutations < 2 or n_bootstrap < 2:
        raise ValueError('Use at least two permutations and bootstrap draws.')
    if not 0 <= min_norm_quantile < 1:
        raise ValueError('Norm quantile must be in [0, 1).')
    im, mm = spatial_metadata(ieeg_metadata).reset_index(drop=True), spatial_metadata(meg_metadata).reset_index(drop=True)
    if len(im)!=len(mm) or 'ieeg_subject' not in mm or not np.array_equal(im.subject.astype(str),mm.ieeg_subject.astype(str)):
        raise ValueError('Use paired_coverage MEG in original electrode row order with ieeg_subject ownership.')
    if 'electrode_index' in mm and not np.array_equal(mm.electrode_index.to_numpy(), np.arange(len(mm))):
        raise ValueError('paired_coverage rows have been reordered.')
    w = {m: np.abs(np.asarray(weights[m], float)) for m in ('iEEG','MEG')}
    if any(a.ndim!=2 or len(a)!=len(im) or a.shape[1]<2 or not np.isfinite(a).all() for a in w.values()):
        raise ValueError('Finite matching weights with at least two components are required for profile similarity.')
    if w['iEEG'].shape!=w['MEG'].shape:raise ValueError('Use equal component counts.')
    xyz = {'iEEG':im[['x','y','z']].to_numpy(float),'MEG':mm[['x','y','z']].to_numpy(float)}
    if not all(np.isfinite(a).all() for a in xyz.values()):raise ValueError('Finite MNI coordinates in mm required.')
    norm = {m:np.linalg.norm(a,axis=1) for m,a in w.items()}
    keep = np.ones(len(im),bool)
    for m in w:
        positive = norm[m][norm[m]>0]
        if not len(positive):raise ValueError('All weight profiles are zero.')
        keep &= (norm[m]>0)&(norm[m]>=np.quantile(positive,min_norm_quantile))
    # Match hemispheres and eliminate duplicated MEG source measurements.
    keep &= np.sign(xyz['iEEG'][:,0])==np.sign(xyz['MEG'][:,0])
    keep &= xyz['iEEG'][:,0]!=0
    duplicate = mm.assign(pair_subject=im.subject.to_numpy()).duplicated(['pair_subject','x','y','z']).to_numpy()
    keep &= ~duplicate
    original = np.flatnonzero(keep)
    im,mm=im.loc[keep].reset_index(drop=True),mm.loc[keep].reset_index(drop=True)
    xyz={m:a[keep] for m,a in xyz.items()};norm={m:a[keep] for m,a in norm.items()}
    if fold_hemispheres:
        xyz={m:fold_coordinates(a, target_hemisphere) for m,a in xyz.items()}
    unit={m:w[m][keep]/norm[m][:,None] for m in w}
    owners=im.subject.to_numpy()
    subjects=owners if spatial_scope == 'within_participant' else np.full(len(im), 'pooled', dtype=object)
    hemisphere=np.where(xyz['iEEG'][:,0]<0,'left','right')
    names=list(pd.unique(subjects));nb=len(edges)-1
    if not names:raise ValueError('No valid matched features remain.')
    curves=np.full((2,len(names),n_permutations+1,2,nb),np.nan)
    counts=np.zeros((len(names),nb),int); rng=np.random.default_rng(seed)
    audit=[]
    contributing_subjects=[set() for _ in range(nb)]
    for si,subject in enumerate(names):
        total=np.zeros((2,n_permutations+1,2,nb));number=np.zeros(nb,int)
        for side in ('left','right'):
            rows=np.flatnonzero((subjects==subject)&(hemisphere==side))
            if len(rows)<2:continue
            a,b=np.triu_indices(len(rows),1)
            distances={m:pdist(xyz[m][rows]) for m in w}
            bins={m:np.searchsorted(edges,d,side='right')-1 for m,d in distances.items()}
            good=(bins['iEEG']==bins['MEG'])&(bins['iEEG']>=0)&(bins['iEEG']<nb)
            # Do not let co-located measurements inflate short-distance similarity.
            good &= (distances['iEEG']>0)&(distances['MEG']>0)
            a,b,bi=a[good],b[good],bins['iEEG'][good]
            number+=np.bincount(bi,minlength=nb)
            for bin_index in np.unique(bi):
                selected=bi==bin_index
                contributing_subjects[bin_index].update(owners[rows[np.r_[a[selected],b[selected]]]])
            for mi,m in enumerate(('iEEG','MEG')):
                u=unit[m][rows];mag=norm[m][rows];variance=np.var(mag)
                for repeat in range(n_permutations+1):
                    order=np.arange(len(rows)) if repeat==0 else rng.permutation(len(rows))
                    values=np.sum(u[order[a]]*u[order[b]],axis=1)
                    total[mi,repeat,0]+=np.bincount(bi,weights=values,minlength=nb)
                    if np.ptp(mag)>0 and variance>0:
                        values=(mag[order[a]]-mag[order[b]])**2/(2*variance)
                        total[mi,repeat,1]+=np.bincount(bi,weights=values,minlength=nb)
                    else:total[mi,repeat,1,:]=np.nan
        counts[si]=number
        curves[:,si]=np.divide(total,number[None,None,None,:],out=np.full_like(total,np.nan),where=number[None,None,None,:]>0)
        audit.append(dict(subject=subject,spatial_scope=spatial_scope,
            fold_hemispheres=bool(fold_hemispheres),target_hemisphere=target_hemisphere if fold_hemispheres else 'native',
            n_participants=len(pd.unique(owners[subjects==subject])),
            n_original_features=len(ieeg_metadata) if spatial_scope == 'pooled' else int((spatial_metadata(ieeg_metadata).subject.astype(str)==subject).sum()),
            n_matched_features=int((subjects==subject).sum()),n_matched_pairs=int(number.sum())))
    table=[];subject_table=[]
    excess=curves[:,:,0,0]-_mean(curves[:,:,1:,0],axis=2)
    boot=rng.integers(0,len(names),(n_bootstrap,len(names)))
    for mi,m in enumerate(('iEEG','MEG')):
        for b in range(nb):
            for si,sub in enumerate(names):
                subject_table.append(dict(modality=m,subject=sub,bin=b,distance_mm=(edges[b]+edges[b+1])/2,
                    n_pairs=counts[si,b],similarity=curves[mi,si,0,0,b],
                    null_mean=_mean(curves[mi,si,1:,0,b]),
                    magnitude_semivariance=curves[mi,si,0,1,b],
                    magnitude_null=_mean(curves[mi,si,1:,1,b]),excess_similarity=excess[mi,si,b]))
            observed=_mean(curves[mi,:,0,0,b]);null=_mean(curves[mi,:,1:,0,b],axis=0)
            bs=_mean(excess[mi,:,b][boot],axis=1);finite=bs[np.isfinite(bs)]
            table.append(dict(modality=m,distance_mm=(edges[b]+edges[b+1])/2,n_subjects=len(contributing_subjects[b]),
                n_analysis_units=int(np.isfinite(excess[mi,:,b]).sum()),
                similarity=observed,null_mean=_mean(null),excess_similarity=_mean(excess[mi,:,b]),
                ci_low=np.quantile(finite,.025) if len(finite) and np.isfinite(excess[mi,:,b]).sum()>1 else np.nan,
                ci_high=np.quantile(finite,.975) if len(finite) and np.isfinite(excess[mi,:,b]).sum()>1 else np.nan,
                magnitude_semivariance=_mean(curves[mi,:,0,1,b]),magnitude_null=_mean(_mean(curves[mi,:,1:,1,b],axis=1))))
    delta=excess[1]-excess[0];difference=[]
    for b in range(nb):
        bs=_mean(delta[:,b][boot],axis=1);finite=bs[np.isfinite(bs)]
        difference.append(dict(distance_mm=(edges[b]+edges[b+1])/2,meg_minus_ieeg=_mean(delta[:,b]),
            ci_low=np.quantile(finite,.025) if len(finite) and np.isfinite(delta[:,b]).sum()>1 else np.nan,
            ci_high=np.quantile(finite,.975) if len(finite) and np.isfinite(delta[:,b]).sum()>1 else np.nan))
    # Preserve signed weights from this exact fit, including on cache reload.
    weight_tables=[]
    for m in ('iEEG','MEG'):
        frame=pd.DataFrame(np.asarray(weights[m])[keep],
                           columns=[f'weight_{c+1}' for c in range(w[m].shape[1])])
        frame.insert(0,'feature_row',original)
        frame.insert(0,'modality',m)
        weight_tables.append(frame)
    return dict(weights=pd.concat(weight_tables,ignore_index=True),curves=pd.DataFrame(table),subject_curves=pd.DataFrame(subject_table),difference=pd.DataFrame(difference),
        audit=pd.DataFrame(audit))


def run_analysis(ieeg, *, model='plssvd', n_components=3, seed=2026,
                 block_scaling='equal_variance', output_dir=None, recompute=False, **options):
    """Construct paired_coverage, fit chosen model, compute/cache results outside notebook."""
    from vs1_coverage_matching_utils import load_dataset
    from vs2_0_cov_models_utils import fit_cov_models
    if model not in ('plssvd','separate_pca','joint_pca'):raise ValueError('Select plssvd, separate_pca or joint_pca.')
    if not isinstance(n_components,int) or n_components<2:raise ValueError('Select at least two components.')
    meg=load_dataset('paired_coverage',reference=ieeg,seed=seed,condition_mode='average')
    config=dict(model=model,n_components=n_components,seed=seed,block_scaling=block_scaling,options=options,meg_kind='paired_coverage',version=5)
    digest=hashlib.sha256(json.dumps(config,sort_keys=True,default=lambda v:np.asarray(v).tolist()).encode())
    for dataset in (ieeg,meg):
        digest.update(dataset.metadata.to_csv(index=False).encode())
        for block in dataset.blocks():digest.update(np.ascontiguousarray(block).tobytes())
    signature=digest.hexdigest();folder=Path(output_dir) if output_dir else None
    if folder and (folder/'manifest.json').exists() and not recompute:
        manifest=json.loads((folder/'manifest.json').read_text())
        if manifest['signature']!=signature:raise ValueError('Inputs/options changed; set RECOMPUTE_PROFILE_SPATIAL=True.')
        return {name:pd.read_csv(folder/f'{name}.csv') for name in manifest['tables']}
    fitted=fit_cov_models(ieeg,meg,n_components=n_components,block_scaling=block_scaling)[model]
    result=analyze_spatial_profiles({'iEEG':fitted.ieeg_weights,'MEG':fitted.meg_weights},ieeg.metadata,meg.metadata,seed=seed,**options)
    if folder:
        folder.mkdir(parents=True,exist_ok=True);(folder/'manifest.json').unlink(missing_ok=True)
        for name,table in result.items():table.to_csv(folder/f'{name}.csv',index=False)
        (folder/'manifest.json').write_text(json.dumps(dict(signature=signature,config=config,tables=list(result)),indent=2,default=lambda v:np.asarray(v).tolist()))
    return result

