"""Descriptive native-feature model weight magnitudes (no channel matching)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import plotting


def threshold_weights(weights):
    """Threshold |W| independently per component at mean + population SD."""
    magnitude = np.abs(np.asarray(weights, dtype=float))
    if magnitude.ndim != 2 or 0 in magnitude.shape or not np.isfinite(magnitude).all():
        raise ValueError('Weights must be a nonempty finite channels × components matrix.')
    mean, std = magnitude.mean(axis=0), magnitude.std(axis=0, ddof=0)
    thresholds = mean + std
    retained = np.where(magnitude >= thresholds[None, :], magnitude, 0.)
    summary = pd.DataFrame(dict(component=np.arange(1, magnitude.shape[1]+1),
        mean_magnitude=mean, std_magnitude=std, threshold=thresholds,
        n_channels=magnitude.shape[0], n_retained=(retained > 0).sum(axis=0),
        fraction_retained=(retained > 0).mean(axis=0)))
    return retained, summary


def plot_weight_distributions(weights, summaries, bins=60):
    """Separate density panels preserve native magnitude scales and channel counts."""
    k = next(iter(weights.values())).shape[1]
    fig, axes = plt.subplots(k, len(weights), figsize=(11, 2.5*k), squeeze=False,
                             constrained_layout=True)
    for col, (modality, values) in enumerate(weights.items()):
        for component in range(k):
            ax = axes[component, col]
            ax.hist(np.abs(values[:, component]), bins=bins, density=True, color=('navy' if col == 0 else 'darkorange'), alpha=.65)
            ax.axvline(summaries[modality].threshold.iloc[component], color='crimson', linestyle='--', label='Mean + SD')
            ax.set(title=f'{modality} · component {component+1}', xlabel='Absolute projection weight', ylabel='Density')
    axes[0, 0].legend()
    return fig


def plot_thresholded_brains(datasets, retained):
    """MNI glass brains; exact co-locations use mean including thresholded zeros."""
    k = next(iter(retained.values())).shape[1]
    fig, axes = plt.subplots(k, len(datasets), figsize=(16, 3*k), squeeze=False)
    for col, (modality, dataset) in enumerate(datasets.items()):
        xyz = dataset.metadata[['x', 'y', 'z']].to_numpy(dtype=float)
        values = retained[modality]
        if xyz.shape != (len(values), 3) or not np.isfinite(xyz).all():
            raise ValueError(f'{modality}: weights require matching finite MNI coordinates in mm.')
        for component in range(k):
            table = pd.DataFrame(xyz, columns=['x', 'y', 'z'])
            table['weight'] = values[:, component]
            grouped = table.groupby(['x', 'y', 'z'], sort=False).weight.mean().reset_index()
            visible = grouped[grouped.weight > 0]
            title = f'{modality} · component {component+1}'
            if len(grouped) < len(table):
                title += ' · mean at co-locations'
            if visible.empty:
                plotting.plot_glass_brain(None, figure=fig, axes=axes[component, col], title=title+' (none retained)')
            else:
                plotting.plot_markers(visible.weight.to_numpy(), visible[['x', 'y', 'z']].to_numpy(),
                    node_size=7, node_cmap='Reds', node_vmin=0,
                    node_vmax=float(visible.weight.max()), node_threshold=None,
                    colorbar=True, figure=fig, axes=axes[component, col], title=title)
    return fig


def spatial_metadata(metadata):
    """Copy native-feature metadata with a canonical subject column.

    MEG ownership takes precedence over the paired iEEG subject label.
    Preserve row order and participant_average labels; never invent owners.
    """
    out = metadata.copy()
    owner = next((name for name in ('meg_subject', 'subject', 'ieeg_subject')
                  if name in out.columns), None)
    if owner is None:
        raise ValueError('Spatial analysis requires subject, meg_subject or ieeg_subject metadata.')
    if out[owner].isna().any():
        raise ValueError(f'Missing participant labels in {owner}.')
    out['subject'] = out[owner].astype(str)
    missing = {'x', 'y', 'z'} - set(out.columns)
    if missing:
        raise ValueError(f'Missing coordinate columns: {sorted(missing)}')
    return out


def spatial_weight_distance(weights, metadata, *, bin_edges=None, n_permutations=199,
                            seed=2026, block_size=512, progress=None, metric="cosine"):
    """Exact all-pair cosine-distance curves, group and individual subjects.

    Uses signed, unthresholded profiles from ONE group fit. Each unordered
    pair occurs once, including distinct features at identical coordinates.
    Null shuffles profiles within subjects against fixed coordinates, exactly
    equivalent to shuffling coordinates within subjects. No pair subsampling.
    Memory is bounded by block_size**2; runtime is quadratic in feature count
    and linear in permutation count. Null envelopes are pointwise, not CIs.
    """
    from scipy.spatial.distance import cdist
    metadata = spatial_metadata(metadata)
    w = np.asarray(weights, dtype=float)
    xyz = metadata[['x', 'y', 'z']].to_numpy(dtype=float)
    if w.ndim != 2 or w.shape[1] < 1 or xyz.shape != (len(w), 3):
        raise ValueError('Weights and coordinate rows must match.')
    if not np.isfinite(w).all() or not np.isfinite(xyz).all() or metadata.subject.isna().any():
        raise ValueError('Finite weights/coordinates and subject labels are required.')
    if not isinstance(n_permutations, (int, np.integer)) or n_permutations < 2:
        raise ValueError('Use at least two permutations.')
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError('block_size must be positive.')
    edges = np.asarray(np.arange(0, 310, 10) if bin_edges is None else bin_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or edges[0] != 0 or np.any(np.diff(edges) <= 0):
        raise ValueError('Bin edges must start at zero and increase strictly.')
    # Bounding-box diagonal ensures every pair is covered, including the final edge.
    if len(xyz) and np.linalg.norm(np.ptp(xyz, axis=0)) > edges[-1]:
        raise ValueError('Increase the last bin edge to cover the coordinate bounding-box diagonal.')
    if metric not in ("cosine", "semivariance"):
        raise ValueError("Unknown spatial metric.")
    if metric == "semivariance" and w.shape[1] != 1:
        raise ValueError("Semivariance requires one component at a time.")
    norm = np.linalg.norm(w, axis=1)
    valid = norm > 0 if metric == "cosine" else np.ones(len(w), dtype=bool)
    subjects = metadata.subject.astype(str).to_numpy()
    audit = pd.DataFrame([dict(subject=s, n_channels=int((subjects == s).sum()),
        n_zero_norm=int(((subjects == s) & ~valid).sum())) for s in pd.unique(subjects)])
    w, xyz, subjects = (w[valid] / norm[valid, None] if metric == "cosine" else w[valid]), xyz[valid], subjects[valid]
    labels = list(pd.unique(subjects))
    groups = [np.flatnonzero(subjects == s) for s in labels]
    subject_index = np.array([labels.index(s) for s in subjects])
    nb = len(edges)-1
    counts = np.zeros((len(labels)+1, nb), dtype=np.int64)
    sums = np.zeros((n_permutations+1, len(labels)+1, nb))
    rng = np.random.default_rng(seed)
    permutations = [np.arange(len(w))]
    for _ in range(n_permutations):
        p = np.arange(len(w))
        for rows in groups:
            p[rows] = rng.permutation(rows)
        permutations.append(p)
    for start in range(0, len(w), block_size):
        ii = np.arange(start, min(start+block_size, len(w)))
        for other in range(start, len(w), block_size):
            jj = np.arange(other, min(other+block_size, len(w)))
            mask = ii[:, None] < jj[None, :]
            a, b = np.nonzero(mask)
            if not len(a):
                continue
            bins = np.minimum(np.searchsorted(edges, cdist(xyz[ii], xyz[jj])[a, b], side='right')-1, nb-1)
            same = subject_index[ii[a]] == subject_index[jj[b]]
            within_key = (subject_index[ii[a[same]]]+1)*nb + bins[same]
            counts[0] += np.bincount(bins, minlength=nb)
            counts += np.bincount(within_key, minlength=counts.size).reshape(counts.shape)
            for repeat, p in enumerate(permutations):
                if metric == "cosine":
                    distances = np.clip(1 - w[p[ii]] @ w[p[jj]].T, 0, 2)[a, b]
                else:
                    distances = .5 * (w[p[ii[a]], 0] - w[p[jj[b]], 0])**2
                sums[repeat, 0] += np.bincount(bins, weights=distances, minlength=nb)
                sums[repeat] += np.bincount(within_key, weights=distances[same], minlength=counts.size).reshape(counts.shape)
        if progress is not None:
            progress(f'Processed {min(start+block_size, len(w))}/{len(w)} channels')
    means = np.divide(sums, counts[None], out=np.full_like(sums, np.nan), where=counts[None]>0)
    if metric == "semivariance":
        variances = np.array([np.var(w[:, 0])] + [np.var(w[g, 0]) for g in groups])
        means = np.divide(means, variances[None, :, None], out=np.full_like(means, np.nan),
                          where=variances[None, :, None] > 0)
    rows, null_rows = [], []
    for group, label in enumerate(['all_channels']+labels):
        scope = 'group' if group == 0 else 'subject'
        for j in range(nb):
            observed, null = means[0, group, j], means[1:, group, j]
            populated = np.isfinite(observed)
            # Two-sided permutation tail test; ties included; Monte Carlo +1 correction.
            p = min(1., 2*min((1+np.sum(null <= observed))/(n_permutations+1),
                               (1+np.sum(null >= observed))/(n_permutations+1))) if populated else np.nan
            rows.append(dict(scope=scope, subject=label, distance_low_mm=edges[j], distance_high_mm=edges[j+1],
                distance_mid_mm=(edges[j]+edges[j+1])/2, n_pairs=counts[group,j],
                cosine_distance=observed, null_mean=float(null.mean()) if populated else np.nan,
                null_low=float(np.quantile(null,.025)) if populated else np.nan,
                null_high=float(np.quantile(null,.975)) if populated else np.nan, p_two_sided=p))
            for repeat, value in enumerate(null):
                null_rows.append(dict(scope=scope, subject=label, bin_index=j, permutation=repeat,
                                      cosine_distance=value))
    table = pd.DataFrame(rows)
    # Holm adjustment across every populated group/subject bin in this modality.
    table['p_holm'] = np.nan
    ix = table.index[table.p_two_sided.notna()]
    ordered = ix[np.argsort(table.loc[ix, 'p_two_sided'].to_numpy())]
    table.loc[ordered, 'p_holm'] = np.minimum(1, np.maximum.accumulate(
        table.loc[ordered, 'p_two_sided'].to_numpy()*np.arange(len(ordered),0,-1)))
    null_table = pd.DataFrame(null_rows)
    if metric == 'semivariance':
        table = table.rename(columns={'cosine_distance': 'semivariance'})
        null_table = null_table.rename(columns={'cosine_distance': 'semivariance'})
    return {'curves': table, 'null_curves': null_table, 'channel_audit': audit}


def plot_spatial_weight_distance(result, title=''):
    """Group plus individual-subject curves with pointwise shuffle envelopes."""
    table = result['curves']
    figures = []
    for scope in ('group', 'subject'):
        groups = list(table[table.scope == scope].groupby('subject', sort=False))
        if not groups:
            continue
        cols = min(3, len(groups))
        fig, axes = plt.subplots(int(np.ceil(len(groups)/cols)), cols,
            figsize=(5*cols, 3.5*int(np.ceil(len(groups)/cols))), squeeze=False, constrained_layout=True)
        for ax, (subject, frame) in zip(axes.flat, groups):
            ax.fill_between(frame.distance_mid_mm, frame.null_low, frame.null_high, color='gray', alpha=.25, label='Null 95% pointwise envelope')
            ax.plot(frame.distance_mid_mm, frame.null_mean, '--', color='gray', label='Null mean')
            ax.plot(frame.distance_mid_mm, frame.cosine_distance, 'o-', label='Observed', markersize=3)
            significant = frame.p_holm < .05
            ax.scatter(frame.loc[significant,'distance_mid_mm'], frame.loc[significant,'cosine_distance'], marker='*', s=90, color='crimson', label='Holm p < .05')
            ax.set(title=subject, xlabel='MNI distance (mm)', ylabel='Cosine distance', ylim=(-.05,2.05))
        for ax in list(axes.flat)[len(groups):]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f'{title} · {scope} · signed group-model weights')
        figures.append(fig)
    return figures


def component_semivariograms(weights, metadata, **options):
    """Per-component normalized semivariograms; Holm over all components/bins/scopes."""
    results = []
    for c in range(weights.shape[1]):
        result = spatial_weight_distance(weights[:, c:c+1], metadata, metric='semivariance', **options)
        for table in result.values():
            table['component'] = c+1
        results.append(result)
    combined = {key: pd.concat([r[key] for r in results], ignore_index=True) for key in results[0]}
    table = combined['curves']
    ix = table.index[table.p_two_sided.notna()]
    order = ix[np.argsort(table.loc[ix, 'p_two_sided'].to_numpy())]
    table.loc[order, 'p_holm'] = np.minimum(1, np.maximum.accumulate(
        table.loc[order, 'p_two_sided'].to_numpy()*np.arange(len(order), 0, -1)))
    return combined


def plot_component_semivariograms(result, title=''):
    """One figure per group/subject, panels for individual components."""
    figures = []
    for (scope, subject), group in result['curves'].groupby(['scope','subject'], sort=False):
        components = list(group.groupby('component'))
        cols = min(3, len(components))
        fig, axes = plt.subplots(int(np.ceil(len(components)/cols)), cols,
            figsize=(5*cols, 3.3*int(np.ceil(len(components)/cols))), squeeze=False, constrained_layout=True)
        for ax, (c, frame) in zip(axes.flat, components):
            ax.fill_between(frame.distance_mid_mm, frame.null_low, frame.null_high, color='gray', alpha=.25)
            ax.plot(frame.distance_mid_mm, frame.null_mean, '--', color='gray', label='Shuffle mean / 95% envelope')
            ax.plot(frame.distance_mid_mm, frame.semivariance, 'o-', markersize=3, label='Observed')
            sig = frame.p_holm < .05
            ax.scatter(frame.loc[sig,'distance_mid_mm'], frame.loc[sig,'semivariance'], marker='*', color='crimson', label='Holm p < .05')
            ax.set(title=f'Component {c}', xlabel='MNI distance (mm)', ylabel='Normalized semivariance', ylim=(0,None))
        for ax in list(axes.flat)[len(components):]: ax.set_visible(False)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f'{title} · {scope}: {subject}')
        figures.append(fig)
    return figures


def weight_clusters(weights, metadata, *, radius_mm=10., min_sources=3):
    """Group/component/sign/hemisphere clusters above group mean+SD.

    Euclidean MNI adjacency is a fallback, not cortical-surface adjacency.
    Midline x==0 points are isolated from both hemispheres. Feature rows and
    exact co-locations are preserved. Subjects do not constrain connectivity;
    their labels are retained only for membership and contribution counts.
    """
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    meta = spatial_metadata(metadata)
    w = np.asarray(weights, float)
    xyz = meta[['x','y','z']].to_numpy(float)
    if w.ndim != 2 or len(w) != len(meta) or not np.isfinite(w).all() or not np.isfinite(xyz).all():
        raise ValueError('Finite weights and coordinates with matching rows are required.')
    if not np.isfinite(radius_mm) or radius_mm <= 0 or not isinstance(min_sources, int) or min_sources < 1:
        raise ValueError('Use positive radius_mm and integer min_sources.')
    clusters, memberships, audit = [], [], []
    owners = meta.subject.to_numpy()
    hemi = np.where(xyz[:,0]<0, 'left', np.where(xyz[:,0]>0, 'right', 'midline'))
    rows = np.arange(len(w))
    for c in range(w.shape[1]):
        magnitudes = np.abs(w[rows,c])
        threshold = magnitudes.mean()+magnitudes.std(ddof=0)
        retained = rows[(magnitudes >= threshold) & (magnitudes > 0)]
        audit.append(dict(scope='group', component=c+1, threshold=threshold, n_sources=len(rows), n_selected=len(retained)))
        for sign in (-1,1):
            for hemisphere in ('left','right','midline'):
                selected = retained[(np.sign(w[retained,c]) == sign) & (hemi[retained] == hemisphere)]
                if not len(selected): continue
                pairs = cKDTree(xyz[selected]).query_pairs(radius_mm, output_type='ndarray')
                graph = coo_matrix((np.ones(len(pairs)), (pairs[:,0],pairs[:,1])), shape=(len(selected),len(selected)))
                _, labels = connected_components(graph, directed=False)
                for label in np.unique(labels):
                    members = selected[labels == label]
                    if len(members) < min_sources: continue
                    magnitude = np.abs(w[members,c])
                    center = np.average(xyz[members], axis=0, weights=magnitude)
                    distances = np.linalg.norm(xyz[members]-center, axis=1)
                    nearest = members[np.argmin(distances)]
                    cid = len(clusters)+1
                    # Spatial density counts unique locations to avoid inflation by
                    # repeated source grids; centroids retain every native feature.
                    unique_xyz = np.unique(xyz[members], axis=0)
                    n_unique = len(unique_xyz)
                    n_edges = (cKDTree(unique_xyz).count_neighbors(cKDTree(unique_xyz), radius_mm)-n_unique)/2
                    density = 2*n_edges/(n_unique*(n_unique-1)) if n_unique > 1 else np.nan
                    order = np.argsort(distances)
                    r90 = distances[order[np.searchsorted(np.cumsum(magnitude[order]), .9*magnitude.sum())]]
                    profiles = w[members]
                    norms = np.linalg.norm(profiles, axis=1)
                    directions = profiles/norms[:,None]
                    # Mean unit direction: amplitude does not dominate similarity.
                    mean_direction = directions.mean(axis=0)
                    direction_norm = np.linalg.norm(mean_direction)
                    similarities = directions @ (mean_direction/direction_norm) if direction_norm > 0 else np.full(len(members),np.nan)
                    clusters.append(dict(cluster_id=cid, scope='group', n_subjects=len(np.unique(owners[members])), component=c+1, sign=sign,
                        hemisphere=hemisphere, n_sources=len(members), n_unique_locations=len(np.unique(xyz[members],axis=0)),
                        total_magnitude=magnitude.sum(), threshold=threshold,
                        radius90_mm=r90, max_radius_mm=distances.max(),
                        edge_density_unique=density, mean_profile_cosine=float(np.mean(similarities)),
                        centroid_x=center[0],centroid_y=center[1],centroid_z=center[2],
                        spread_rms_mm=np.sqrt(np.average(distances**2, weights=magnitude)),
                        representative_row=int(nearest), representative_x=xyz[nearest,0],
                        representative_y=xyz[nearest,1],representative_z=xyz[nearest,2]))
                    memberships.extend(dict(cluster_id=cid,subject=owners[i],component=c+1,feature_row=int(i),weight=w[i,c],
                        distance_to_centroid_mm=distances[j], profile_cosine_to_cluster=similarities[j]) for j,i in enumerate(members))
    columns = ['cluster_id','scope','n_subjects','component','sign','hemisphere','n_sources','n_unique_locations','total_magnitude','threshold','radius90_mm','max_radius_mm','edge_density_unique','mean_profile_cosine',
               'centroid_x','centroid_y','centroid_z','spread_rms_mm','representative_row','representative_x','representative_y','representative_z']
    return dict(clusters=pd.DataFrame(clusters,columns=columns),
                memberships=pd.DataFrame(memberships,columns=['cluster_id','subject','component','feature_row','weight','distance_to_centroid_mm','profile_cosine_to_cluster']),
                thresholds=pd.DataFrame(audit))


def plot_weight_cluster_centroids(result, title='', *, metadata, figsize=(20, 12)):
    """All electrodes, cluster members and true centroids in MNI projections.

    Gray marks include every feature; colors indicate cluster identity, not
    magnitude. Stars are mathematical centroids, not snapped member sources.
    """
    xyz = spatial_metadata(metadata)[['x','y','z']].to_numpy(float)
    figures = []
    for component in sorted(result['thresholds'].component.unique()):
        table = result['clusters'].query('component == @component')
        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        cmap = plt.get_cmap('turbo')
        for ax, (u,v,name) in zip(axes.flat[:3], [(0,1,'Axial'),(0,2,'Coronal'),(1,2,'Sagittal')]):
            ax.scatter(xyz[:,u],xyz[:,v],s=5,c='lightgray',alpha=.35,rasterized=True,label='All electrodes')
            for j, row in enumerate(table.itertuples()):
                color = cmap(j/max(len(table)-1,1))
                members = result['memberships'].query('cluster_id == @row.cluster_id').feature_row.to_numpy(int)
                ax.scatter(xyz[members,u],xyz[members,v],s=14,color=color,alpha=.6,rasterized=True)
                center = np.array([row.centroid_x,row.centroid_y,row.centroid_z])
                ax.scatter(center[u],center[v],s=260,marker='*',color=color,edgecolor='black',linewidth=1.3,zorder=5)
                ax.annotate(str(row.cluster_id),center[[u,v]],xytext=(7,7),textcoords='offset points',fontsize=10,weight='bold')
            ax.set(title=name, xlabel=f'MNI {"xyz"[u]} (mm)',ylabel=f'MNI {"xyz"[v]} (mm)',aspect='equal')
            ax.grid(alpha=.15)
        ax = axes.flat[3]
        for j,row in enumerate(table.itertuples()):
            data = result['memberships'].query('cluster_id == @row.cluster_id')
            # Radial mean profiles, keeping all members; empty bins are omitted.
            edges = np.linspace(0,max(data.distance_to_centroid_mm.max(),1e-9),9)
            indices = np.minimum(np.searchsorted(edges,data.distance_to_centroid_mm,side='right')-1,7)
            radial = data.assign(bin=indices).groupby('bin').agg(
                distance=('distance_to_centroid_mm','mean'),similarity=('profile_cosine_to_cluster','mean'))
            ax.plot(radial.distance,radial.similarity,'o-',color=cmap(j/max(len(table)-1,1)),
                    label=f'C{row.cluster_id} ({"+" if row.sign>0 else "−"}), n={row.n_sources}, R90={row.radius90_mm:.1f} mm')
        ax.set(xlabel='Distance to spatial centroid (mm)',ylabel='Cosine similarity to mean cluster profile',ylim=(-1.05,1.05),title='Weight-profile coherence vs spatial distance')
        if len(table): ax.legend(fontsize=8)
        else: ax.text(.5,.5,'No retained clusters',transform=ax.transAxes,ha='center')
        fig.suptitle(f'{title} · component {component} · group clusters (stars = centroids)',fontsize=18)
        figures.append(fig)
    return figures


def kmeans_weight_space(weights, metadata, *, n_clusters=5, seed=2026,
                        bin_edges=None, block_size=512):
    """Group K-means in signed thresholded first-three-component weight space.

    No spatial coordinates enter the fit. All-zero thresholded rows are labelled
    -1 and excluded (cosine distance is undefined). K-means uses Euclidean
    distance, no additional feature standardization, and 20 initializations.
    Spatial centroids are unweighted means of member MNI coordinates and are
    descriptive only. Weight centroids are the actual fitted K-means centers.
    """
    from sklearn.cluster import KMeans
    from scipy.spatial.distance import cdist
    meta = spatial_metadata(metadata)
    w = np.asarray(weights, float)
    if w.ndim != 2 or w.shape[1] < 3 or len(w) != len(meta) or not np.isfinite(w).all():
        raise ValueError('Provide finite native weights with at least three components and matching metadata.')
    if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 2:
        raise ValueError('n_clusters must be an integer >= 2.')
    if not isinstance(block_size, (int, np.integer)) or block_size < 1:
        raise ValueError('block_size must be a positive integer.')
    xyz = meta[['x','y','z']].to_numpy(float)
    if not np.isfinite(xyz).all(): raise ValueError('MNI coordinates must be finite.')
    magnitude, thresholds = threshold_weights(w[:,:3])
    thresholded = np.sign(w[:,:3])*magnitude
    active = np.linalg.norm(thresholded,axis=1)>0
    if len(np.unique(thresholded[active],axis=0)) < n_clusters:
        raise ValueError('Fewer distinct nonzero thresholded profiles than clusters; reduce n_clusters.')
    fit = KMeans(n_clusters=int(n_clusters),random_state=seed,n_init=20).fit(thresholded[active])
    labels = np.full(len(w),-1,dtype=int)
    labels[active] = fit.labels_
    edges = np.asarray(np.arange(0,310,10) if bin_edges is None else bin_edges,float)
    if edges.ndim != 1 or len(edges)<2 or not np.isfinite(edges).all() or edges[0]!=0 or np.any(np.diff(edges)<=0):
        raise ValueError('Bin edges must start at zero and strictly increase.')
    if np.linalg.norm(np.ptp(xyz,axis=0)) > edges[-1]:
        raise ValueError('Increase final distance bin to cover the MNI bounding-box diagonal.')
    members = meta.copy().reset_index(drop=True)
    members['feature_row'] = np.arange(len(w))
    members['cluster_id'] = labels
    for c in range(3):
        members[f'weight_{c+1}'] = w[:,c]
        members[f'thresholded_weight_{c+1}'] = thresholded[:,c]
    clusters, curves = [], []
    for label in range(n_clusters):
        rows = np.flatnonzero(labels==label)
        spatial_center = xyz[rows].mean(axis=0)
        spatial_dist = np.linalg.norm(xyz[rows]-spatial_center,axis=1)
        center = fit.cluster_centers_[label]
        clusters.append(dict(cluster_id=label,n_sources=len(rows),n_subjects=meta.iloc[rows].subject.nunique(),
            weight_centroid_1=center[0],weight_centroid_2=center[1],weight_centroid_3=center[2],
            centroid_x=spatial_center[0],centroid_y=spatial_center[1],centroid_z=spatial_center[2],
            spatial_rms_mm=np.sqrt(np.mean(spatial_dist**2)),spatial_radius90_mm=np.quantile(spatial_dist,.9),
            weight_rms=np.sqrt(np.mean(np.sum((thresholded[rows]-center)**2,axis=1)))))
        unit = thresholded[rows]/np.linalg.norm(thresholded[rows],axis=1)[:,None]
        count, total = np.zeros(len(edges)-1,dtype=np.int64), np.zeros(len(edges)-1)
        for start in range(0,len(rows),block_size):
            ii = np.arange(start,min(start+block_size,len(rows)))
            for other in range(start,len(rows),block_size):
                jj = np.arange(other,min(other+block_size,len(rows)))
                a,b = np.nonzero(ii[:,None]<jj[None,:])
                if not len(a): continue
                d = cdist(xyz[rows[ii]],xyz[rows[jj]])[a,b]
                bins = np.minimum(np.searchsorted(edges,d,side='right')-1,len(edges)-2)
                cosine = np.clip(1-unit[ii]@unit[jj].T,0,2)[a,b]
                count += np.bincount(bins,minlength=len(count))
                total += np.bincount(bins,weights=cosine,minlength=len(count))
        for j in range(len(count)):
            curves.append(dict(cluster_id=label,distance_low_mm=edges[j],distance_high_mm=edges[j+1],
                distance_mid_mm=(edges[j]+edges[j+1])/2,n_pairs=count[j],
                cosine_distance=total[j]/count[j] if count[j] else np.nan))
    return dict(clusters=pd.DataFrame(clusters),memberships=members,thresholds=thresholds,
                distance_curves=pd.DataFrame(curves))


def plot_kmeans_weight_space(result, title=''):
    """3D weight space, full native brain cloud, histograms, within-cluster curves."""
    from matplotlib.lines import Line2D
    from matplotlib.colors import to_hex
    members, clusters = result['memberships'], result['clusters']
    colors = {int(row.cluster_id):plt.get_cmap('turbo')(j/max(len(clusters)-1,1))
              for j,row in enumerate(clusters.itertuples())}
    figures = []
    fig = plt.figure(figsize=(12,10),constrained_layout=True)
    ax = fig.add_subplot(111,projection='3d')
    for row in clusters.itertuples():
        data = members[members.cluster_id==row.cluster_id]
        color = colors[row.cluster_id]
        ax.scatter(data.thresholded_weight_1,data.thresholded_weight_2,data.thresholded_weight_3,
                   s=9,alpha=.4,color=color,rasterized=True,label=f'C{row.cluster_id} (n={row.n_sources})')
        ax.scatter(row.weight_centroid_1,row.weight_centroid_2,row.weight_centroid_3,
                   s=220,marker='*',color=color,edgecolor='black',linewidth=1.2)
        ax.text(row.weight_centroid_1,row.weight_centroid_2,row.weight_centroid_3,f' C{row.cluster_id}')
    ax.set(xlabel='Thresholded weight 1',ylabel='Thresholded weight 2',zlabel='Thresholded weight 3',
           title=f'{title} · K-means weight space (stars = fitted centers)')
    ax.legend(fontsize=8); figures.append(fig)
    fig = plt.figure(figsize=(22,8))
    brain = plotting.plot_glass_brain(None,figure=fig,display_mode='lyrz',
        title=f'{title} · brain locations of weight-space clusters (stars = spatial means)')
    coords = members[['x','y','z']].to_numpy()
    brain.add_markers(coords,marker_color='lightgray',marker_size=5,alpha=.2)
    for row in clusters.itertuples():
        selected = members.cluster_id.to_numpy()==row.cluster_id
        brain.add_markers(coords[selected],marker_color=to_hex(colors[row.cluster_id]),marker_size=12,alpha=.6)
        brain.add_markers(np.array([[row.centroid_x,row.centroid_y,row.centroid_z]]),
            marker_color=to_hex(colors[row.cluster_id]),marker_size=260,marker='*',edgecolors='black',linewidths=1.2)
    fig.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=colors[r.cluster_id],
                 label=f'C{r.cluster_id}: RMS {r.spatial_rms_mm:.1f} mm') for r in clusters.itertuples()],
               loc='lower center',ncol=min(5,len(clusters)),fontsize=10)
    figures.append(fig)
    fig,axes = plt.subplots(len(clusters),3,figsize=(15,2.6*len(clusters)),squeeze=False,constrained_layout=True)
    for i,row in enumerate(clusters.itertuples()):
        data = members[members.cluster_id==row.cluster_id]
        for c,ax in enumerate(axes[i]):
            # Identical edges for native and thresholded values within each panel.
            bins = np.histogram_bin_edges(data[f'weight_{c+1}'],bins=35)
            bins = np.unique(np.r_[bins,0.])
            ax.hist(data[f'weight_{c+1}'],bins=bins,density=True,histtype='step',color='black',label='Original')
            ax.hist(data[f'thresholded_weight_{c+1}'],bins=bins,density=True,alpha=.55,color=colors[row.cluster_id],label='Thresholded')
            ax.set(title=f'C{row.cluster_id} · component {c+1}',xlabel='Signed weight',ylabel='Density')
    axes[0,0].legend(fontsize=8)
    fig.suptitle(f'{title} · within-cluster weight distributions'); figures.append(fig)
    fig,axes = plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
    for label,frame in result['distance_curves'].groupby('cluster_id',sort=True):
        axes[0].plot(frame.distance_mid_mm,frame.cosine_distance,'o-',color=colors[label],label=f'C{label}')
        axes[1].plot(frame.distance_mid_mm,frame.n_pairs,'o-',color=colors[label],label=f'C{label}')
    axes[0].set(xlabel='MNI pair distance (mm)',ylabel='Mean cosine distance (thresholded 3D weights)',ylim=(-.05,2.05))
    axes[1].set(xlabel='MNI pair distance (mm)',ylabel='Number of unordered pairs',yscale='symlog')
    axes[0].legend(); fig.suptitle(f'{title} · within-cluster pairs (all clusters overlaid)')
    figures.append(fig)
    return figures
