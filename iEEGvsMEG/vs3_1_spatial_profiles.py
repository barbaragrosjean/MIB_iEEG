"""Paired-coverage spatial organization and descriptive absolute-weight patches.

All inference is conditional on fitted group-model weights. Spatial shuffles
remove within-subject/hemisphere association, not MEG reconstruction smoothness.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def _patch_labels(xyz, unit, radius, similarity):
    """Spatially connected merges with complete-link profile similarity.

    Every member pair must meet the similarity criterion, preventing chaining.
    Greedy edge order is deterministic; patches are exploratory, not optimized k.
    """
    distance = squareform(pdist(xyz))
    cosine = np.clip(unit @ unit.T, 0, 1)
    i, j = np.where(np.triu((distance > 0) & (distance <= radius), 1))
    labels = np.arange(len(xyz))
    groups = {k: [k] for k in labels}
    for edge in np.argsort(-cosine[i,j], kind='stable'):
        a, b = labels[i[edge]], labels[j[edge]]
        if a == b or cosine[i[edge],j[edge]] < similarity:
            continue
        if np.min(cosine[np.ix_(groups[a], groups[b])]) >= similarity:
            rows = groups.pop(b)
            groups[a].extend(rows)
            labels[rows] = a
    return labels, distance, cosine


def _mean(values, axis=0):
    count = np.sum(np.isfinite(values), axis=axis)
    result = np.divide(np.nansum(values, axis=axis), count,
                     out=np.full(np.shape(count), np.nan, dtype=float), where=count > 0)
    return float(result) if result.ndim == 0 else result


def analyze_spatial_profiles(weights, ieeg_metadata, meg_metadata, *, bin_edges=None,
                             n_permutations=199, n_bootstrap=500, seed=2026,
                             patch_radius_mm=15., patch_similarity=.9, min_patch_size=3,
                             patch_resamples=30, min_norm_quantile=0.):
    """Matched-pair curves, paired participant bootstrap, and hemisphere patches.

    weights has iEEG and MEG entries from the same selected group-model fit.
    Rows must retain paired_coverage electrode correspondence. Duplicate MEG
    source matches within a participant are removed jointly from both modalities.
    Curves use only pairs in the same distance bin in both modalities. All
    nonempty participant curves receive equal weight. Patch stability is source
    deletion stability (80% retained), NOT trial-bootstrap fit reproducibility.
    """
    from vs3_1_modality_specificity import spatial_metadata
    edges = np.asarray(np.arange(0, 101, 10) if bin_edges is None else bin_edges, float)
    if edges.ndim != 1 or len(edges)<2 or edges[0]!=0 or np.any(np.diff(edges)<=0) or not np.isfinite(edges).all():
        raise ValueError('Distance edges must be finite, start at zero, and strictly increase.')
    if n_permutations<2 or n_bootstrap<2 or patch_resamples<0 or min_patch_size<2:
        raise ValueError('Use >=2 permutations/bootstrap draws, >=2 patch members, and >=0 patch resamples.')
    if not 0<=min_norm_quantile<1 or not 0<patch_similarity<=1 or patch_radius_mm<=0:
        raise ValueError('Invalid norm quantile, cosine threshold, or patch radius.')
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
    unit={m:w[m][keep]/norm[m][:,None] for m in w}
    subjects=im.subject.to_numpy(); hemisphere=np.where(xyz['iEEG'][:,0]<0,'left','right')
    names=list(pd.unique(subjects));nb=len(edges)-1
    if not names:raise ValueError('No valid matched features remain.')
    curves=np.full((2,len(names),n_permutations+1,2,nb),np.nan)
    counts=np.zeros((len(names),nb),int); rng=np.random.default_rng(seed)
    audit=[];patches=[];membership=[]
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
                labels,d,cos=_patch_labels(xyz[m][rows],u,patch_radius_mm,patch_similarity)
                resampled=[]
                for repeat in range(patch_resamples):
                    subset=np.sort(rng.choice(len(rows),max(2,int(.8*len(rows))),replace=False))
                    sublabels,_,_=_patch_labels(xyz[m][rows][subset],u[subset],patch_radius_mm,patch_similarity)
                    resampled.append((subset,sublabels))
                for label in np.unique(labels):
                    local=np.flatnonzero(labels==label)
                    if len(local)<min_patch_size:continue
                    patch_id=f'{subject}_{side}_{int(label)}'
                    center=xyz[m][rows][local].mean(axis=0)
                    medoid=local[np.argmin(d[np.ix_(local,local)].sum(axis=1))]
                    other=np.flatnonzero(labels!=label)
                    boundary=(d[np.ix_(local,other)]<=patch_radius_mm)&(d[np.ix_(local,other)]>0)
                    inner=cos[np.ix_(local,local)][np.triu_indices(len(local),1)].mean()
                    outer=cos[np.ix_(local,other)][boundary].mean() if boundary.any() else np.nan
                    stability=[]
                    for subset,sl in resampled:
                        retained=np.intersect1d(local,subset)
                        if len(retained)<2:continue
                        stability.append(max(len(np.intersect1d(retained,subset[sl==l]))/len(np.union1d(retained,subset[sl==l])) for l in np.unique(sl)))
                    patches.append(dict(modality=m,subject=subject,hemisphere=side,patch_id=patch_id,
                        n_sources=len(local),coherence=inner,boundary_similarity=outer,contrast=inner-outer,
                        radius_rms_mm=np.sqrt(np.mean(np.sum((xyz[m][rows][local]-center)**2,axis=1))),
                        diameter_mm=d[np.ix_(local,local)].max(),stability=np.mean(stability) if stability else np.nan,
                        medoid_x=xyz[m][rows][medoid,0],medoid_y=xyz[m][rows][medoid,1],medoid_z=xyz[m][rows][medoid,2]))
                    for loc in local:
                        idx=rows[loc]
                        membership.append(dict(modality=m,subject=subject,patch_id=patch_id,feature_row=original[idx],
                            x=xyz[m][idx,0],y=xyz[m][idx,1],z=xyz[m][idx,2]))
        counts[si]=number
        curves[:,si]=np.divide(total,number[None,None,None,:],out=np.full_like(total,np.nan),where=number[None,None,None,:]>0)
        audit.append(dict(subject=subject,n_original_features=int((spatial_metadata(ieeg_metadata).subject.astype(str)==subject).sum()),n_matched_features=int((subjects==subject).sum()),n_matched_pairs=int(number.sum())))
    table=[];subject_table=[]
    excess=curves[:,:,0,0]-_mean(curves[:,:,1:,0],axis=2)
    boot=rng.integers(0,len(names),(n_bootstrap,len(names)))
    for mi,m in enumerate(('iEEG','MEG')):
        for b in range(nb):
            for si,sub in enumerate(names):
                subject_table.append(dict(modality=m,subject=sub,bin=b,n_pairs=counts[si,b],excess_similarity=excess[mi,si,b]))
            observed=_mean(curves[mi,:,0,0,b]);null=_mean(curves[mi,:,1:,0,b],axis=0)
            bs=_mean(excess[mi,:,b][boot],axis=1);finite=bs[np.isfinite(bs)]
            table.append(dict(modality=m,distance_mm=(edges[b]+edges[b+1])/2,n_subjects=int(np.isfinite(excess[mi,:,b]).sum()),
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
    return dict(curves=pd.DataFrame(table),subject_curves=pd.DataFrame(subject_table),difference=pd.DataFrame(difference),
        patches=pd.DataFrame(patches,columns=['modality','subject','hemisphere','patch_id','n_sources','coherence','boundary_similarity','contrast','radius_rms_mm','diameter_mm','stability','medoid_x','medoid_y','medoid_z']),
        memberships=pd.DataFrame(membership,columns=['modality','subject','patch_id','feature_row','x','y','z']),audit=pd.DataFrame(audit))


def run_analysis(ieeg, *, model='plssvd', n_components=3, seed=2026,
                 block_scaling='equal_variance', output_dir=None, recompute=False, **options):
    """Construct paired_coverage, fit chosen model, compute/cache results outside notebook."""
    from vs1_coverage_matching_utils import load_dataset
    from vs2_0_cov_models_utils import fit_cov_models
    if model not in ('plssvd','separate_pca','joint_pca'):raise ValueError('Select plssvd, separate_pca or joint_pca.')
    if not isinstance(n_components,int) or n_components<2:raise ValueError('Select at least two components.')
    meg=load_dataset('paired_coverage',reference=ieeg,seed=seed,condition_mode='average')
    config=dict(model=model,n_components=n_components,seed=seed,block_scaling=block_scaling,options=options,meg_kind='paired_coverage',version=1)
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


def plot_analysis(result, title=''):
    import matplotlib.pyplot as plt
    from nilearn import plotting
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for m,color in [('iEEG','navy'),('MEG','darkorange')]:
        t=result['curves'].query('modality == @m')
        axes[0,0].plot(t.distance_mm,t.similarity,'o-',color=color,label=m)
        axes[0,0].plot(t.distance_mm,t.null_mean,'--',color=color)
        axes[0,1].plot(t.distance_mm,t.excess_similarity,'o-',color=color,label=m)
        axes[0,1].fill_between(t.distance_mm,t.ci_low,t.ci_high,color=color,alpha=.15)
        axes[1,0].plot(t.distance_mm,t.magnitude_semivariance,'o-',color=color,label=m)
        axes[1,0].plot(t.distance_mm,t.magnitude_null,'--',color=color)
    t=result['difference'];axes[1,1].plot(t.distance_mm,t.meg_minus_ieeg,'o-',color='black')
    axes[1,1].fill_between(t.distance_mm,t.ci_low,t.ci_high,color='gray',alpha=.25)
    for ax,ylabel in zip(axes.flat,['Absolute-profile cosine similarity','Excess similarity over spatial shuffle','Normalized magnitude semivariance','Excess similarity: MEG − iEEG']):
        ax.set(xlabel='Distance (mm; matched bins)',ylabel=ylabel);ax.axhline(0,color='gray',lw=.7)
    axes[0,0].legend();fig.suptitle(title+' · dashed: shuffled baseline; bands: paired-participant bootstrap')
    brainfig=plt.figure(figsize=(16,7));brain=plotting.plot_glass_brain(None,figure=brainfig,display_mode='lyrz',title=title+' · MEG weight patches; stars: spatial medoids')
    members=result['memberships'].query("modality == 'MEG'");patches=result['patches'].query("modality == 'MEG'")
    from matplotlib.colors import to_hex
    for i,row in enumerate(patches.itertuples()):
        group=members[members.patch_id==row.patch_id];color=to_hex(plt.get_cmap('tab20')(i%20))
        brain.add_markers(group[['x','y','z']].to_numpy(),marker_color=color,marker_size=12,alpha=.6)
        brain.add_markers([[row.medoid_x,row.medoid_y,row.medoid_z]],marker_color=color,marker='*',marker_size=160)
    metrics,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,metric in zip(axes,['coherence','contrast','stability']):
        for m,color in [('iEEG','navy'),('MEG','darkorange')]:
            values=result['patches'].query('modality == @m')[metric].dropna()
            if len(values):ax.hist(values,bins=12,histtype='step',color=color,label=m)
        ax.set(xlabel=metric,ylabel='Patch count')
    if len(result['patches']):axes[0].legend()
    metrics.suptitle('Descriptive patches · stability = 80% source-retention Jaccard; no trial resampling')
    return dict(spatial_profiles=fig,meg_patches=brainfig,patch_metrics=metrics)
