"""Fixed-half-trial association summaries using compatible PAIRING_ONLY fits.

Association SD describes changes in subject assignment. Participant SD describes
heterogeneity across matched participant pairs within one assignment; neither
is a confidence interval. All component matching uses training time courses.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd

from vs3_0_spatial_reliability import align_bundle, _load_bundle, _fingerprint, fit_iterations
from vs3_0_weight_stability import controlled_seeds
from vs3_1_spatial_profiles import analyze_spatial_profiles


def compatible_run(root, fit, options):
    """Check saved fit design, completed iterations, fixed splits, and provenance."""
    root = Path(root)
    try:
        cfg = json.loads((root/'config.json').read_text())
        expected = dict(schema_version=2, variation='pairing', meg_kind='paired_coverage',
                        components=fit['n_components'], seed=fit['seed'],
                        split_unit=options['split_unit'], condition_mode='average',
                        block_scaling='equal_variance' if fit['block_scaling'] else 'none')
        if any(cfg.get(k) != v for k, v in expected.items()):
            return False, 'Fit settings differ.'
        if fit['model'] not in cfg['models'] or cfg['iterations'] < options['iterations']:
            return False, 'Model or requested iterations unavailable.'
        source_manifest = str((Path(options['cache_dir'])/'manifest.json').resolve())
        if not any(p['path'] == source_manifest for p in cfg['provenance']):
            return False, 'Different trial cache.'
        for key in ('atlas_image', 'atlas_labels'):
            if options.get(key) and str(Path(options[key]).resolve()) not in [p['path'] for p in cfg['provenance']]:
                return False, 'Requested atlas differs.'
        if any(_fingerprint(p['path']) != p for p in cfg['provenance']):
            return False, 'Input or coordinate provenance changed.'
        fixed_splits = None
        for i in range(options['iterations']):
            folder = root/f'iteration_{i:03d}'
            marker = json.loads((folder/'COMPLETE.json').read_text())
            split_seed, pair_seed = controlled_seeds(fit['seed'], i, 'pairing')
            if marker['split_seed'] != split_seed or marker['pair_seed'] != pair_seed:
                return False, 'Random streams differ.'
            splits = json.loads((folder/'splits.json').read_text())
            if fixed_splits is None:
                fixed_splits = splits
            if splits != fixed_splits:
                return False, 'Trial subset changes across associations.'
            for name in (f"{fit['model']}_A.npz", 'ieeg_features.csv', 'meg_features.csv'):
                if not (folder/name).exists():
                    return False, f'Missing {folder/name}.'
        return True, 'Compatible fixed-half-trial fits.'
    except (OSError, KeyError, ValueError):
        return False, 'Saved fits or provenance unavailable.'


def ensure_fits(config, fallback_dir):
    fit, options = config['fit'], config['pairing']
    # The established stability workflow applies these corrections to trial means.
    if (fit['condition_mode'] != 'average' or fit['data']['meg_scaling'] != 'channel_zscore'
            or fit['data']['ieeg_multiplier'] != 1000.):
        raise ValueError('PAIRING_ONLY reuse requires condition averaging, training-mean MEG z-scoring, '
                         'and iEEG multiplier 1000, as in the existing stability analysis.')
    manifest = json.loads((Path(options['cache_dir'])/'manifest.json').read_text())
    if list(manifest['config']['conditions']) != list(fit['data']['conditions']):
        raise ValueError('Notebook conditions differ from the selected trial cache.')
    if options['iterations'] < 2:
        raise ValueError('Use at least two associations for sample SD.')
    existing = Path(options['existing_dir'])
    valid, reason = compatible_run(existing, fit, options)
    if valid:
        print(f'Reusing PAIRING_ONLY fits: {existing}', flush=True)
        return existing
    print(f'{reason} Computing fixed-half-trial fits in {fallback_dir}.', flush=True)
    args = SimpleNamespace(cache_dir=Path(options['cache_dir']), output_dir=Path(fallback_dir),
        meg_kind='paired_coverage', models=[fit['model']], components=fit['n_components'],
        iterations=options['iterations'], seed=fit['seed'], split_unit=options['split_unit'],
        block_scaling='equal_variance' if fit['block_scaling'] else 'none',
        max_gram_gib=fit['max_gram_gib'], variation='pairing', weight_only=True,
        pool_hemispheres=False, scratch_dir=None,
        ieeg_coords_dir=None, meg_coords_dir=None, ieeg_coords_unit='mm', meg_coords_unit='mm',
        atlas_image=Path(options['atlas_image']) if options.get('atlas_image') else None,
        atlas_labels=Path(options['atlas_labels']) if options.get('atlas_labels') else None)
    fit_iterations(args)
    return Path(fallback_dir)


def load_aligned(root, model, iterations):
    root = Path(root)
    reference = _load_bundle(root/'iteration_000'/f'{model}_A.npz')
    anchor = pd.read_csv(root/'iteration_000/ieeg_features.csv', dtype={'subject': str})
    values = {'iEEG': [], 'MEG': []}
    meg_metadata, matches, provenance = [], [], [_fingerprint(root/'config.json')]
    for iteration in range(iterations):
        folder = root/f'iteration_{iteration:03d}'
        metadata = pd.read_csv(folder/'ieeg_features.csv', dtype={'subject': str})
        identity = ['subject', 'source_index', 'x', 'y', 'z']
        if not metadata[identity].equals(anchor[identity]):
            raise ValueError('Fixed iEEG coverage rows differ across associations.')
        bundle, transforms = align_bundle(reference, _load_bundle(folder/f'{model}_A.npz'), model)
        for label, m in [('iEEG', 'ieeg'), ('MEG', 'meg')]:
            values[label].append(bundle[m+'_weights'])
            columns, signs = transforms[m]
            matches.extend(dict(iteration=iteration, modality=label, component=c+1,
                                original_component=int(col)+1, sign=float(sign))
                           for c, (col, sign) in enumerate(zip(columns, signs)))
        meg_metadata.append(pd.read_csv(folder/'meg_features.csv', dtype={'subject': str, 'ieeg_subject': str}))
        provenance.extend(_fingerprint(folder/name) for name in
                          [f'{model}_A.npz','ieeg_features.csv','meg_features.csv','splits.json','COMPLETE.json'])
    cfg = json.loads((root/'config.json').read_text())
    provenance.extend(cfg['provenance'])
    return dict(weights={m: np.stack(v) for m, v in values.items()}, anchor=anchor,
                meg_metadata=meg_metadata, component_matching=pd.DataFrame(matches),
                provenance=provenance, source_dir=str(root), source_config=cfg)


def summarize_magnitudes(aligned, *, normalize, percentile, bins=50):
    """Threshold each aligned component in each association, then aggregate."""
    anchor = aligned['anchor']
    from vs3_0_weight_stability import bilateral_label
    regions = anchor['region'].map(bilateral_label).fillna('Unassigned')
    thresholds, roi_rows, histograms, maps = [], [], [], {}
    for modality, weights in aligned['weights'].items():
        if normalize:
            centered = weights - weights.mean(axis=1, keepdims=True)
            sd = weights.std(axis=1, keepdims=True)
            magnitude = np.abs(np.divide(centered, sd, out=np.zeros_like(centered), where=sd > 0))
        else:
            magnitude = np.abs(weights)
        limits = np.percentile(magnitude, percentile, axis=1)
        retained = np.where(magnitude >= limits[:, None, :], magnitude, 0.)
        maps[modality] = dict(mean=retained.mean(axis=0), sd=retained.std(axis=0, ddof=1))
        for c in range(weights.shape[2]):
            edges = np.linspace(0, max(float(magnitude[:, :, c].max()), np.finfo(float).eps), bins+1)
            for i in range(len(weights)):
                thresholds.append(dict(iteration=i, modality=modality, component=c+1,
                                       threshold=limits[i, c], percentile=percentile, normalized=normalize,
                                       n_above_threshold=int(np.count_nonzero(magnitude[i, :, c] > limits[i, c]))))
                counts, _ = np.histogram(magnitude[i, :, c], bins=edges)
                histograms.extend(dict(iteration=i, modality=modality, component=c+1,
                                       magnitude=(edges[b]+edges[b+1])/2, count=count)
                                  for b, count in enumerate(counts))
                frame = pd.DataFrame(dict(subject=anchor.subject, region=regions, weight=retained[i, :, c]))
                # Equal participant contribution within each region.
                roi = frame.groupby(['subject', 'region']).weight.mean().groupby('region').mean()
                roi_rows.extend(dict(iteration=i, modality=modality, component=c+1, region=r, magnitude=v)
                                for r, v in roi.items())
    return dict(thresholds=pd.DataFrame(thresholds), roi_iterations=pd.DataFrame(roi_rows),
                histogram_iterations=pd.DataFrame(histograms), maps=maps, **aligned)



def profile_distance_edges(aligned, n_components, *, width_mm=10., min_norm_quantile=0.,
                           fold_hemispheres=False, target_hemisphere='left'):
    """Common bins covering all eligible within-participant distances, including the maximum.

    Apply the same feature exclusions as analyze_spatial_profiles. Distance-bin
    agreement is imposed later, using these common edges for every association.
    """
    from scipy.spatial.distance import pdist
    anchor = aligned['anchor']
    owners = anchor.subject.astype(str).to_numpy()
    maximum = 0.
    for i, meg in enumerate(aligned['meg_metadata']):
        xyz = {'iEEG': anchor[['x','y','z']].to_numpy(float),
               'MEG': meg[['x','y','z']].to_numpy(float)}
        keep = np.ones(len(anchor), dtype=bool)
        for modality in xyz:
            norm = np.linalg.norm(aligned['weights'][modality][i, :, :n_components], axis=1)
            positive = norm[norm > 0]
            if not len(positive):
                raise ValueError('All weight profiles are zero.')
            keep &= (norm > 0) & (norm >= np.quantile(positive, min_norm_quantile))
        keep &= (np.sign(xyz['iEEG'][:,0]) == np.sign(xyz['MEG'][:,0])) & (xyz['iEEG'][:,0] != 0)
        keep &= ~meg.assign(pair_subject=owners).duplicated(['pair_subject','x','y','z']).to_numpy()
        xyz = {m: values[keep].copy() for m, values in xyz.items()}
        if fold_hemispheres:
            for values in xyz.values():
                values[:,0] = (-1 if target_hemisphere == 'left' else 1) * np.abs(values[:,0])
        for subject in pd.unique(owners[keep]):
            for side in (-1, 1):
                rows = (owners[keep] == subject) & (np.sign(xyz['iEEG'][:,0]) == side)
                distances = {m: pdist(values[rows]) for m, values in xyz.items()}
                valid = (distances['iEEG'] > 0) & (distances['MEG'] > 0)
                if valid.any():
                    maximum = max(maximum, *(float(d[valid].max()) for d in distances.values()))
    if not np.isfinite(width_mm) or width_mm <= 0:
        raise ValueError('Distance-bin width must be positive and finite.')
    # searchsorted(..., side="right") excludes the final edge: put it above the maximum.
    edges = np.arange(int(np.floor(maximum / width_mm)) + 2, dtype=float) * width_mm
    return edges, maximum


def summarize_profiles(aligned, options, seed):
    """Preserve participant-level curves before summarizing association means/SD."""
    options = dict(options)
    k = options.pop('n_components')
    if options.get('spatial_scope', 'within_participant') != 'within_participant':
        raise ValueError('Participant variability requires within_participant spatial scope.')
    width = options.pop('distance_bin_width_mm', 10.)
    maximum = None
    if options.get('bin_edges') is None:
        edges, maximum = profile_distance_edges(aligned, k, width_mm=width,
            min_norm_quantile=options.get('min_norm_quantile', 0.),
            fold_hemispheres=options.get('fold_hemispheres', False),
            target_hemisphere=options.get('target_hemisphere', 'left'))
        options['bin_edges'] = edges.tolist()
    subjects, audits = [], []
    for i in range(len(aligned['meg_metadata'])):
        result = analyze_spatial_profiles(
            {m: w[i, :, :k] for m, w in aligned['weights'].items()},
            aligned['anchor'], aligned['meg_metadata'][i], seed=seed, **options)
        subjects.append(result['subject_curves'].assign(iteration=i))
        audits.append(result['audit'].assign(iteration=i))
    participant = pd.concat(subjects, ignore_index=True)
    metrics = ['similarity','null_mean','excess_similarity','magnitude_semivariance','magnitude_null']
    groups = ['iteration','modality','distance_mm']
    means = participant.groupby(groups)[metrics].mean().reset_index()
    sd = participant.groupby(groups)[metrics].std(ddof=1).reset_index()
    sd = sd.rename(columns={m: m+'_subject_sd' for m in metrics})
    counts = participant.groupby(groups)[metrics].count().reset_index().rename(columns={m:m+'_n_subjects' for m in metrics})
    iterations = means.merge(sd, on=groups).merge(counts, on=groups)
    keys = ['modality','distance_mm']
    summary = iterations.groupby(keys)[metrics].agg(['mean','std','count'])
    summary.columns = [m+'_'+{'std':'association_sd','count':'n_associations'}.get(stat, stat) for m,stat in summary.columns]
    summary = summary.reset_index()
    pair_counts = participant.groupby(groups, as_index=False).n_pairs.sum()
    pair_summary = pair_counts.groupby(keys).n_pairs.agg(['mean','std','min','max','count']).reset_index()
    edges = np.asarray(options['bin_edges'], dtype=float)
    bins = pd.DataFrame(dict(bin_left_mm=edges[:-1], bin_right_mm=edges[1:],
                             distance_mm=(edges[:-1]+edges[1:])/2))
    return dict(pair_counts=pair_counts, pair_count_summary=pair_summary, distance_bins=bins,
                max_eligible_distance_mm=maximum, participant_curves=participant, iteration_curves=iterations, summary=summary,
                audit=pd.concat(audits, ignore_index=True), provenance=aligned['provenance'],
                source_dir=aligned['source_dir'])
