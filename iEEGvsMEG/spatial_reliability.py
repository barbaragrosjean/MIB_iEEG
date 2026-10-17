#!/usr/bin/env python3
"""Changing-pairing spatial reliability and participant-balanced regional comparisons.

python -u spatial_reliability.py --iterations 100 --components 3
python -u spatial_reliability.py --summarize-only

Uses the completed out/trial_cache. Fits PCA, PLSSVD and joint PCA on paired_coverage.
Iterations are repeated random splits, NOT 100 disjoint cross-validation folds.
"""
from pathlib import Path
import argparse
import hashlib
import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

from compare_subspace import MODELS, MODALITIES, fit_models
from compare_models import correlation_matrix

LEVELS = ('parcel', 'region', 'group')
SPACES = ('weights', 'patterns')


def matched_columns(a, b):
    """Hungarian matching on temporal scores only; no spatial optimization."""
    corr = correlation_matrix(a, b)
    rows, cols = linear_sum_assignment(-np.nan_to_num(abs(corr), nan=-1.))
    if not np.array_equal(rows, np.arange(a.shape[1])) or not np.isfinite(corr[rows, cols]).all():
        raise ValueError('Component matching requires nonconstant training scores.')
    return cols, np.where(corr[rows, cols] < 0, -1., 1.)


def align_bundle(reference, current, model):
    """Descriptive correspondence to iteration 0/A using training temporal scores.

    Coupled models use one shared permutation/sign for both blocks; separate PCA
    is matched independently within each modality. No test or spatial matching.
    """
    aligned = {key: np.array(value, copy=True) for key, value in current.items()}
    transforms = {}
    if model == 'separate_pca':
        transforms = {m: matched_columns(reference[m+'_train_scores'], current[m+'_train_scores'])
                      for m in MODALITIES}
    else:
        corr = sum(correlation_matrix(reference[m+'_train_scores'], current[m+'_train_scores'])
                   for m in MODALITIES) / 2
        rows, cols = linear_sum_assignment(-np.nan_to_num(abs(corr), nan=-1.))
        if not np.isfinite(corr[rows, cols]).all():
            raise ValueError('Undefined coupled-component correspondence.')
        transforms = dict.fromkeys(MODALITIES, (cols, np.where(corr[rows, cols] < 0, -1., 1.)))
    for m, (cols, signs) in transforms.items():
        for key, value in current.items():
            if key.startswith(m+'_') and (key.endswith('_weights') or key.endswith('_scores') or key.endswith('_patterns')):
                aligned[key] = value[:, cols]*signs
    return aligned, transforms


def diagonal_correlations(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected equal finite row-by-component matrices.')
    a, b = a-a.mean(0), b-b.mean(0)
    den = np.linalg.norm(a, axis=0)*np.linalg.norm(b, axis=0)
    return np.divide((a*b).sum(0), den, out=np.full(len(den), np.nan), where=den > 0).clip(-1, 1)


def correlation_rows(a, b, **labels):
    if a.shape != b.shape:
        raise ValueError('Correlation requires the same ordered rows and components.')
    signed = diagonal_correlations(a, b)
    magnitude = diagonal_correlations(abs(a), abs(b))
    return [dict(**labels, component=c+1, signed_r=float(r), magnitude_r=float(q), n_rows=len(a))
            for c, (r, q) in enumerate(zip(signed, magnitude))]


def iteration_splits(trials, seed, split_unit='trial'):
    """Two independent half fits; each opposite half also has two test quarters."""
    from plssvd_eval_utils import _subject_folds
    rng = np.random.default_rng(seed)
    result = {half: {m: {} for m in MODALITIES} for half in ('A', 'B')}
    for m in MODALITIES:
        for subject in getattr(trials, m):
            folds = _subject_folds(subject, rng, split_unit, 4)
            if any(len(ix) < 2 for f in folds for ix in f['test']):
                raise ValueError(f'{m}/{subject.subject}: need >=2 trials per condition in each quarter.')
            for half, training, testing in [('A', (0, 1), (2, 3)), ('B', (2, 3), (0, 1))]:
                merge = lambda ids: [np.concatenate([folds[i]['test'][c] for i in ids])
                                     for c in range(len(subject.conditions))]
                result[half][m][subject.subject] = dict(train=merge(training), test=merge(testing),
                    test_a=folds[testing[0]]['test'], test_b=folds[testing[1]]['test'])
    return result


def annotate_coordinates(xyz, atlas_image=None, atlas_labels=None, pool_hemispheres=False):
    """Common atlas at native coordinates; no nearest-labelled-voxel reassignment.

    Optional local atlas CSV columns: atlas_id, parcel, region, group. Project
    defaults use AAL parcels, existing AAL project regions, and REGION_GROUPS.
    Unmapped locations remain missing at each resolution.
    """
    xyz = np.asarray(xyz, float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Expected finite MNI coordinates in mm.')
    if (atlas_image is None) != (atlas_labels is None):
        raise ValueError('Provide both --atlas-image and --atlas-labels.')
    if atlas_image is None:
        from region_labels import get_region_label
        from modality_specificity import REGION_GROUPS
        parcel, region = get_region_label(pd.DataFrame(xyz)/1000., return_atlas_labels=True)
        grouping = {r: g for g, regions in REGION_GROUPS for r in regions}
        frame = pd.DataFrame(dict(parcel=parcel, region=region))
        frame['group'] = frame.region.map(grouping)
    else:
        import nibabel as nib
        atlas = nib.load(str(atlas_image))
        data = np.asarray(atlas.dataobj)
        if data.ndim != 3:
            raise ValueError('Atlas must be a 3D integer label image.')
        mapping = pd.read_csv(atlas_labels)
        if not {'atlas_id', *LEVELS} <= set(mapping):
            raise ValueError('Atlas CSV requires atlas_id, parcel, region, group.')
        if mapping.atlas_id.duplicated().any():
            raise ValueError('Atlas IDs must be unique.')
        ijk = np.rint(nib.affines.apply_affine(np.linalg.inv(atlas.affine), xyz)).astype(int)
        inside = ((ijk >= 0) & (ijk < np.asarray(data.shape))).all(1)
        ids = np.zeros(len(xyz), dtype=int)
        ids[inside] = data[tuple(ijk[inside].T)].astype(int)
        frame = mapping.set_index('atlas_id').reindex(ids)[list(LEVELS)].reset_index(drop=True)
        frame.loc[ids == 0, :] = np.nan
    frame = frame.replace({'N': np.nan, 'Background': np.nan, 'Unassigned': np.nan, '': np.nan})
    hemisphere = np.where(xyz[:, 0] < 0, 'L', np.where(xyz[:, 0] > 0, 'R', 'M'))
    for level in LEVELS:
        # Standardize the hemisphere convention at every resolution.
        frame[level] = frame[level].astype('string').str.replace(r'_[LR]$', '', regex=True)
        if not pool_hemispheres:
            frame[level] = frame[level] + '_' + hemisphere
    return frame


def subject_profiles(values, metadata, level):
    """Mean |coefficient| per unique source, then participant/parcel.

    Collapse repeated copies of one native source within a MEG participant so
    matching multiplicity cannot give it extra mass. No absent parcel is zero.
    """
    if len(values) != len(metadata):
        raise ValueError('Feature metadata and values differ in length.')
    columns = [f'c{i+1}' for i in range(values.shape[1])]
    frame = metadata[['subject', 'source_index', level]].copy()
    frame[columns] = abs(values)
    frame = frame.dropna(subset=[level])
    unique = frame.groupby(['subject', 'source_index', level], as_index=False)[columns].mean()
    profile = unique.groupby(['subject', level], as_index=False)[columns].mean()
    count = unique.groupby(['subject', level]).size().rename('n_unique_sources').reset_index()
    return profile.merge(count, on=['subject', level])


def row_spearman(a, b):
    """Vectorized row correlations; missing/constant rows stay NaN."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    result = np.full(len(a), np.nan)
    x, y = rankdata(a[valid], axis=1), rankdata(b[valid], axis=1)
    x -= x.mean(1, keepdims=True); y -= y.mean(1, keepdims=True)
    den = np.linalg.norm(x, axis=1)*np.linalg.norm(y, axis=1)
    result[valid] = np.divide((x*y).sum(1), den, out=np.full(len(den), np.nan), where=den > 0)
    return result


def parcel_correspondence(profiles, level, n_components, n_bootstrap, seed,
                          min_subjects=2, subject_ids=None):
    """Independent whole-participant bootstraps, conditional on fitted models.

    Hold the observed common-parcel set fixed. Bootstrap draws missing any of
    those parcels yield undefined correlation, not an altered comparison domain.
    """
    rng = np.random.default_rng(seed)
    counts = {m: profiles[m].groupby(level).subject.nunique() for m in MODALITIES}
    common = sorted(set(counts['ieeg'][counts['ieeg'] >= min_subjects].index) &
                    set(counts['meg'][counts['meg'] >= min_subjects].index))
    matrices, draws, means = {}, {}, {}
    for m in MODALITIES:
        subjects = list(subject_ids[m]) if subject_ids is not None else sorted(profiles[m].subject.unique())
        matrices[m] = np.stack([profiles[m].pivot(index='subject', columns=level, values=f'c{c+1}')
                                 .reindex(index=subjects, columns=common).to_numpy(float)
                                 for c in range(n_components)], axis=2)
        # Multinomial subject multiplicities are reused for all components/parcels.
        draws[m] = rng.multinomial(len(subjects), np.full(len(subjects), 1/len(subjects)), size=n_bootstrap)
        finite = np.isfinite(matrices[m])
        total = np.nansum(matrices[m], axis=0)
        means[m] = np.divide(total, finite.sum(0), out=np.full_like(total, np.nan), where=finite.sum(0) > 0)
    rows, region_rows = [], []
    for c in range(n_components):
        observed = np.nan
        bootstrap = np.full(n_bootstrap, np.nan)
        if len(common) >= 3:
            observed = row_spearman(means['ieeg'][:, c][None], means['meg'][:, c][None])[0]
            sampled = {}
            for m in MODALITIES:
                matrix = matrices[m][:, :, c]
                numerator = draws[m] @ np.nan_to_num(matrix)
                denominator = draws[m] @ np.isfinite(matrix).astype(float)
                sampled[m] = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
            bootstrap = row_spearman(sampled['ieeg'], sampled['meg'])
        valid = bootstrap[np.isfinite(bootstrap)]
        # Too few complete resamples should not create apparently precise CIs.
        low, high = np.quantile(valid, [.025, .975]) if len(valid) >= max(20, .8*n_bootstrap) else (np.nan, np.nan)
        rows.append(dict(component=c+1, level=level, spearman_r=float(observed), ci_low=low, ci_high=high,
                         n_common_parcels=len(common), n_valid_bootstrap=len(valid), n_bootstrap=n_bootstrap))
        for p, parcel in enumerate(common):
            for m in MODALITIES:
                region_rows.append(dict(component=c+1, level=level, parcel=parcel, modality=m,
                                        mean_magnitude=means[m][p, c], n_subjects=int(counts[m][parcel])))
    return rows, region_rows


def _fingerprint(path):
    p = Path(path)
    return dict(path=str(p.resolve()), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)


def _write_config(path, config):
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError(f'{path}: configuration/provenance changed; use a new output directory.')
    path.write_text(json.dumps(config, indent=2))


def inspect_comparison(directory):
    root = Path(directory)
    config = json.loads((root/'config.json').read_text())
    return dict(meg_kind=config['meg_kind'], repeats=config['repeats'],
                saved_model_files=len(list(root.glob('*_model.npz'))),
                saved_mapped_pattern_files=len(list(root.glob('*_spatial_patterns.npz'))),
                reuse='Weights/scores and mapped patterns support fixed-pairing analyses only. '
                'Changing pairing and independent held-out quarters require new fits; '
                'native patterns cannot generally be recovered from mapped patterns alone.',
                pairing=config.get('repetitions', 'Consult original configuration.'))


def fit_iterations(args):
    from plssvd_eval_utils import load_trial_cache, _temporary_fold, _project, _patterns
    from mni_coordinates import update_trial_coordinates
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    trials = load_trial_cache(args.cache_dir)
    manifest = json.loads((args.cache_dir/'manifest.json').read_text())
    for m in MODALITIES:
        if {str(s.subject) for s in getattr(trials, m)} != set(map(str, manifest['config'][m+'_subjects'])):
            raise ValueError('Trial cache is incomplete.')
    coordinate_sources = update_trial_coordinates(trials, ieeg_dir=args.ieeg_coords_dir,
        meg_dir=args.meg_coords_dir, ieeg_unit=args.ieeg_coords_unit, meg_unit=args.meg_coords_unit)
    records = [s for m in MODALITIES for s in getattr(trials, m)]
    labels = annotate_coordinates(np.concatenate([s.positions for s in records]),
                                 args.atlas_image, args.atlas_labels, args.pool_hemispheres)
    label_lookup, offset = {}, 0
    for m in MODALITIES:
        for s in getattr(trials, m):
            label_lookup[m, str(s.subject)] = labels.iloc[offset:offset+len(s.positions)].reset_index(drop=True)
            offset += len(s.positions)
    provenance = [_fingerprint(args.cache_dir/'manifest.json')]
    provenance += [_fingerprint(args.cache_dir/f) for record in manifest['records'] for f in record['files']]
    provenance += [_fingerprint(path) for path in coordinate_sources.path]
    provenance += [_fingerprint(p) for p in (args.atlas_image, args.atlas_labels) if p is not None]
    config = dict(schema_version=1, meg_kind='paired_coverage', iterations=args.iterations,
                  components=args.components, models=list(MODELS), seed=args.seed,
                  split_unit=args.split_unit, block_scaling=args.block_scaling,
                  pool_hemispheres=args.pool_hemispheres, condition_mode='average',
                  coordinate_units=dict(ieeg=args.ieeg_coords_unit, meg=args.meg_coords_unit),
                  provenance=provenance, label_hash=hashlib.sha256(labels.to_csv(index=False).encode()).hexdigest(),
                  design='Changing participant pairing; independent random trial halves A/B per iteration; '
                         'each half also provides two held-out quarters for axes fitted on the opposite half.',
                  weights='Projection weights in training-preprocessed feature coordinates; MEG train z-scored.',
                  across_iterations='MEG rows follow fixed electrode coverage slots, NOT identical participants/sources.',
                  bootstrap='Independent participant resampling by modality, conditional on each fitted half-A model; '
                            'not full-pipeline uncertainty or participant-held-out validation.')
    _write_config(root/'config.json', config)
    coordinate_sources.to_csv(root/'coordinate_sources.csv', index=False)
    for iteration in range(args.iterations):
        dest = root/f'iteration_{iteration:03d}'
        expected = [f'{model}_{half}.npz' for model in MODELS for half in ('A', 'B')] + [
            'ieeg_features.csv', 'meg_features.csv', 'pairing.json', 'matching.csv', 'splits.json',
            'preprocessing_A.npz', 'preprocessing_B.npz']
        if (dest/'COMPLETE.json').exists():
            if not all((dest/name).exists() for name in expected):
                raise ValueError(f'Incomplete files despite completion marker: {dest}')
            print(f'Iteration {iteration+1}/{args.iterations}: reuse saved fits', flush=True)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        split_seed, pair_seed = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence([args.seed, iteration]).spawn(2)]
        indices = iteration_splits(trials, split_seed, args.split_unit)
        (dest/'splits.json').write_text(json.dumps(indices, default=lambda a: a.tolist()))
        first_pairing = None
        for half in ('A', 'B'):
            with _temporary_fold(trials, 'paired_coverage', indices[half], pair_seed,
                                 args.scratch_dir, condition_mode='average') as prepared:
                fold, scalers, audit, pairing = prepared
                if half == 'A':
                    first_pairing = pairing
                    audit.to_csv(dest/'matching.csv', index=False)
                    (dest/'pairing.json').write_text(json.dumps(pairing, indent=2))
                    for m in MODALITIES:
                        meta = fold['train'][m].metadata.copy().reset_index(drop=True)
                        if m == 'ieeg':
                            meta['subject'] = np.concatenate([np.repeat(str(s.subject), len(s.positions)) for s in trials.ieeg])
                            meta['source_index'] = np.concatenate([np.arange(len(s.positions)) for s in trials.ieeg])
                        else:
                            meta['subject'] = meta.meg_subject.astype(str)
                        for level in LEVELS:
                            meta[level] = [label_lookup[m, str(s)].iloc[int(i)][level]
                                           for s, i in zip(meta.subject, meta.source_index)]
                        meta['coverage_row'] = np.arange(len(meta))
                        meta.to_csv(dest/f'{m}_features.csv', index=False)
                elif pairing != first_pairing:
                    raise AssertionError('A/B pairing must be identical.')
                fitted = fit_models(fold['train'], MODELS, args.components, args.block_scaling, args.max_gram_gib)
                for name, model in fitted.items():
                    bundle = {'k_max': args.components}
                    for m in MODALITIES:
                        bundle[m+'_weights'] = model[m+'_weights']
                        bundle[m+'_mean'] = model[m+'_mean']
                        bundle[m+'_scale'] = model[m+'_scale']
                        for part in fold:
                            scores = _project(fold[part][m], model, m)
                            bundle[f'{m}_{part}_scores'] = scores
                            bundle[f'{m}_{part}_patterns'] = _patterns(fold[part][m], scores)
                    np.savez_compressed(dest/f'{name}_{half}.npz', **bundle)
                np.savez_compressed(dest/f'preprocessing_{half}.npz', **{
                    f'{m}_{s}_{label}': v for m, subjects in scalers.items() for s, vals in subjects.items()
                    for label, v in zip(('mean', 'std', 'multiplier'), vals)})
            print(f'Iteration {iteration+1}/{args.iterations}, half {half}: fitted all three models', flush=True)
        (dest/'COMPLETE.json').write_text(json.dumps(dict(iteration=iteration, split_seed=split_seed, pair_seed=pair_seed)))
    return config


def _load_bundle(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def across_iteration_correlations(values, **labels):
    """All iteration pairs, using BLAS matrices one component at a time."""
    if any(v.shape != values[0].shape for v in values):
        raise ValueError('Across-iteration rows/components must have identical coverage ordering.')
    rows = []
    for c in range(values[0].shape[1]):
        matrix = np.column_stack([v[:, c] for v in values])
        signed = correlation_matrix(matrix, matrix)
        magnitude = correlation_matrix(abs(matrix), abs(matrix))
        for i, j in combinations(range(len(values)), 2):
            rows.append(dict(**labels, component=c+1, iteration_a=i, iteration_b=j,
                             signed_r=float(signed[i, j]), magnitude_r=float(magnitude[i, j]),
                             n_rows=len(matrix)))
    return rows


def summarize(root, n_bootstrap=1000, min_subjects=2, seed=2026):
    root = Path(root)
    config = json.loads((root/'config.json').read_text())
    k, n = config['components'], config['iterations']
    if n_bootstrap < 20 or min_subjects < 1:
        raise ValueError('Use >=20 bootstrap draws and >=1 participant per parcel.')
    for i in range(n):
        if not (root/f'iteration_{i:03d}'/'COMPLETE.json').exists():
            raise ValueError(f'Iteration {i} is incomplete; finish fitting before summarizing.')
    completion = root/'SUMMARY_COMPLETE.json'
    if completion.exists():
        completion.unlink()
    reliability, crossmodal, parcel_rows, region_rows, matches = [], [], [], [], []
    profiles_directory = root/'subject_profiles'
    profiles_directory.mkdir(exist_ok=True)
    across = []
    for model in config['models']:
        reference = _load_bundle(root/'iteration_000'/f'{model}_A.npz')
        iteration_weights, iteration_patterns = {m: [] for m in MODALITIES}, {m: [] for m in MODALITIES}
        for i in range(n):
            folder = root/f'iteration_{i:03d}'
            if not (folder/'COMPLETE.json').exists():
                raise ValueError(f'{folder} is incomplete; finish fitting before summarizing.')
            metadata = {m: pd.read_csv(folder/f'{m}_features.csv', dtype={'subject': str, 'meg_subject': str}) for m in MODALITIES}
            a, transforms = align_bundle(reference, _load_bundle(folder/f'{model}_A.npz'), model)
            b, _ = align_bundle(a, _load_bundle(folder/f'{model}_B.npz'), model)
            for m in MODALITIES:
                cols, signs = transforms[m]
                for c, (col, sign) in enumerate(zip(cols, signs)):
                    matches.append(dict(iteration=i, model=model, modality=m, component=c+1,
                                        original_component=int(col+1), sign=float(sign)))
                base = dict(iteration=i, model=model, modality=m)
                reliability += correlation_rows(a[m+'_weights'], b[m+'_weights'],
                    **base, measure='independent_half_weights', space='weights')
                reliability += correlation_rows(a[m+'_train_patterns'], b[m+'_train_patterns'],
                    **base, measure='independent_half_patterns', space='patterns')
                for half, bundle in [('A', a), ('B', b)]:
                    reliability += correlation_rows(bundle[m+'_test_a_patterns'], bundle[m+'_test_b_patterns'],
                        **base, measure=f'fixed_axes_{half}_heldout_quarters', space='patterns')
                iteration_weights[m].append(a[m+'_weights'])
                iteration_patterns[m].append(a[m+'_test_patterns'])
            # PCA component matching uses train time courses; coupled model axes
            # retain native cross-modal pairing. No spatial/test optimization.
            if model == 'separate_pca':
                columns, signs = matched_columns(a['ieeg_train_scores'], a['meg_train_scores'])
            else:
                columns, signs = np.arange(k), np.ones(k)
            for c, (col, sign) in enumerate(zip(columns, signs)):
                matches.append(dict(iteration=i, model=model, modality='crossmodal', component=c+1,
                                    original_component=int(col+1), sign=float(sign)))
            for part in ('train', 'test'):
                crossmodal += correlation_rows(a[f'ieeg_{part}_scores'], a[f'meg_{part}_scores'][:, columns]*signs,
                    iteration=i, model=model, space='temporal_scores', partition=part)
                crossmodal += correlation_rows(a[f'ieeg_{part}_patterns'], a[f'meg_{part}_patterns'][:, columns]*signs,
                    iteration=i, model=model, space='patterns', partition=part)
            crossmodal += correlation_rows(a['ieeg_weights'], a['meg_weights'][:, columns]*signs,
                iteration=i, model=model, space='weights', partition='train')
            for space in SPACES:
                suffix = 'weights' if space == 'weights' else 'test_patterns'
                values = {'ieeg': a['ieeg_'+suffix], 'meg': a['meg_'+suffix][:, columns]*signs}
                for level in LEVELS:
                    profiles = {m: subject_profiles(values[m], metadata[m], level) for m in MODALITIES}
                    for m in MODALITIES:
                        profiles[m].to_csv(profiles_directory/f'{model}_{i:03d}_{space}_{level}_{m}.csv', index=False)
                    rows, regions = parcel_correspondence(profiles, level, k, n_bootstrap,
                        seed + i, min_subjects, {m: metadata[m].subject.unique() for m in MODALITIES})
                    parcel_rows.extend(dict(iteration=i, model=model, space=space, **r) for r in rows)
                    region_rows.extend(dict(iteration=i, model=model, space=space, **r) for r in regions)
            print(f'Summarized {model}, iteration {i+1}/{n}', flush=True)
        for m in MODALITIES:
            for space, values in [('weights', iteration_weights[m]), ('patterns', iteration_patterns[m])]:
                across += across_iteration_correlations(values, model=model, modality=m, space=space,
                    axis='electrode_coverage_slots' if m == 'meg' else 'native_electrodes')
    tables = dict(reliability=reliability, crossmodal=crossmodal, parcels=parcel_rows,
                  regional_profiles=region_rows, across_iterations=across, component_matching=matches)
    tables = {name: pd.DataFrame(rows) for name, rows in tables.items()}
    if tables['regional_profiles'].empty:
        tables['regional_profiles'] = pd.DataFrame(columns=['iteration', 'model', 'space', 'component',
            'level', 'parcel', 'modality', 'mean_magnitude', 'n_subjects'])
    for name, rows in tables.items():
        rows.to_csv(root/f'{name}.csv', index=False)
    (root/'summary_config.json').write_text(json.dumps(dict(n_bootstrap=n_bootstrap, min_subjects=min_subjects,
        seed=seed, ci='95% percentile; conditional on fitted models, fixed common-parcel support; '
        'suppressed if <80% or <20 bootstrap draws are defined'), indent=2))
    completion.write_text(json.dumps(dict(iterations=n, tables=list(tables))))
    return tables


def load_results(root):
    root = Path(root)
    if not (root/'SUMMARY_COMPLETE.json').exists():
        raise ValueError('Summary is incomplete. Run spatial_reliability.py --summarize-only after fitting.')
    config = json.loads((root/'config.json').read_text())
    names = ('reliability', 'crossmodal', 'parcels', 'regional_profiles', 'across_iterations', 'component_matching')
    return config, {name: pd.read_csv(root/f'{name}.csv') for name in names}


def plot_results(root, component=1, iteration=0, space='patterns'):
    """Read-only plots: iteration ranges are descriptive, bootstrap CIs conditional."""
    import matplotlib.pyplot as plt
    config, tables = load_results(root)
    if component not in range(1, config['components']+1) or space not in SPACES:
        raise ValueError('Choose a saved component and weights or patterns.')
    if iteration not in range(config['iterations']):
        raise ValueError('Choose an existing zero-based iteration.')
    models = config['models']; figures = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    measures = (['independent_half_weights'] if space == 'weights' else
                ['independent_half_patterns', 'fixed_axes_A_heldout_quarters', 'fixed_axes_B_heldout_quarters'])
    for ax, m in zip(axes, MODALITIES):
        d = tables['reliability'].query('component == @component and modality == @m')
        arrays, labels = [], []
        for model in models:
            for measure in measures:
                arrays.append(d[(d.model == model) & (d.measure == measure)].magnitude_r.dropna())
                labels.append(model+'\n'+measure.replace('fixed_axes_', '').replace('_heldout_quarters', ' heldout'))
        ax.boxplot(arrays, tick_labels=labels, showfliers=False)
        ax.tick_params(axis='x', labelrotation=55)
        ax.set(title=m, ylabel='Pearson correlation of magnitudes', ylim=(-1.05, 1.05))
    fig.suptitle(f'Within-modality {space} reliability · component {component} · distributions across iterations')
    figures.append(fig)
    fig, axes = plt.subplots(len(models), 2, figsize=(11, 4*len(models)), layout='constrained', squeeze=False)
    d = tables['across_iterations'].query('component == @component and space == @space')
    for row, model in enumerate(models):
        for col, m in enumerate(MODALITIES):
            matrix = np.full((config['iterations'], config['iterations']), np.nan)
            for r in d[(d.model == model) & (d.modality == m)].itertuples():
                matrix[r.iteration_a, r.iteration_b] = matrix[r.iteration_b, r.iteration_a] = r.magnitude_r
            image = axes[row, col].imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r')
            axes[row, col].set(title=model+' / '+m, xlabel='Iteration', ylabel='Iteration')
    fig.colorbar(image, ax=axes.ravel().tolist(), label='Magnitude-pattern Pearson r')
    fig.suptitle(f'Across-iteration {space} stability · component {component}\nMEG: coverage slots change participants/sources; pairing and trial effects are combined')
    figures.append(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    d = tables['crossmodal'].query('component == @component')
    for ax, representation, partition, metric in [(axes[0], 'temporal_scores', 'test', 'signed_r'),
            (axes[1], space, 'test' if space == 'patterns' else 'train', 'magnitude_r')]:
        arrays = [d[(d.model == model) & (d.space == representation) & (d.partition == partition)][metric].dropna() for model in models]
        ax.boxplot(arrays, tick_labels=models, showfliers=False)
        ax.set(title=representation+' / '+partition, ylabel=metric, ylim=(-1.05, 1.05))
    fig.suptitle(f'Paired iEEG–MEG component correspondence · component {component}')
    figures.append(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    d = tables['parcels'].query('component == @component and space == @space')
    for mi, model in enumerate(models):
        offset = (mi-(len(models)-1)/2)*.2
        for li, level in enumerate(LEVELS):
            values = d[(d.model == model) & (d.level == level)].spearman_r.dropna()
            if len(values):
                lo, med, hi = np.quantile(values, [.05, .5, .95])
                axes[0].plot([li+offset]*2, [lo, hi], color=f'C{mi}')
                axes[0].plot(li+offset, med, 'o', color=f'C{mi}', label=model if li == 0 else None)
            chosen = d[(d.model == model) & (d.level == level) & (d.iteration == iteration)]
            if len(chosen):
                r = chosen.iloc[0]
                axes[1].plot(li+offset, r.spearman_r, 'o', color=f'C{mi}', label=model if li == 0 else None)
                axes[1].plot([li+offset]*2, [r.ci_low, r.ci_high], color=f'C{mi}')
    for ax in axes:
        ax.set(xticks=range(len(LEVELS)), xticklabels=LEVELS, ylabel='Regional magnitude Spearman r', ylim=(-1.05, 1.05))
        ax.axhline(0, color='grey', ls=':'); ax.legend(fontsize=8)
    axes[0].set_title('Median and 5–95% iteration range (not a CI)')
    axes[1].set_title(f'Iteration {iteration}: conditional participant-bootstrap 95% CI')
    fig.suptitle(f'Common anatomy at three scales · {space} · component {component}')
    figures.append(fig)
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path(__file__).resolve().parent
    parser.add_argument('--cache-dir', type=Path, default=base/'out/trial_cache')
    parser.add_argument('--output-dir', type=Path, default=base/'out/spatial_reliability/paired_coverage')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--components', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--split-unit', choices=['trial', 'group'], default='trial')
    parser.add_argument('--block-scaling', choices=['none', 'equal_variance'], default='equal_variance')
    parser.add_argument('--max-gram-gib', type=float, default=2.)
    parser.add_argument('--scratch-dir', type=Path)
    parser.add_argument('--ieeg-coords-dir', type=Path)
    parser.add_argument('--meg-coords-dir', type=Path)
    parser.add_argument('--ieeg-coords-unit', choices=['m', 'mm'], default='mm')
    parser.add_argument('--meg-coords-unit', choices=['m', 'mm'], default='mm')
    parser.add_argument('--atlas-image', type=Path, help='Local MNI label NIfTI; otherwise use cached/downloaded AAL.')
    parser.add_argument('--atlas-labels', type=Path, help='CSV: atlas_id,parcel,region,group. Requires --atlas-image.')
    parser.add_argument('--pool-hemispheres', action='store_true', help='Pool homologous labels; default keeps left/right separate.')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--min-subjects', type=int, default=2)
    parser.add_argument('--summarize-only', action='store_true', help='Reuse all saved weights/patterns; no trial cache or atlas required.')
    parser.add_argument('--inspect-comparison', type=Path, help='Audit an existing compare_subspace directory and exit.')
    args = parser.parse_args()
    if args.inspect_comparison:
        print(json.dumps(inspect_comparison(args.inspect_comparison), indent=2)); return
    if args.iterations < 2 or args.components < 1:
        parser.error('Need >=2 iterations and >=1 component.')
    if args.bootstrap < 20 or args.min_subjects < 1:
        parser.error('Need >=20 bootstrap draws and >=1 participant per parcel.')
    if not args.summarize_only:
        fit_iterations(args)
    summarize(args.output_dir, args.bootstrap, args.min_subjects, args.seed)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for i, figure in enumerate(plot_results(args.output_dir)):
        figure.savefig(args.output_dir/f'spatial_reliability_{i}.png', dpi=180, bbox_inches='tight')
        plt.close(figure)
    print(f'Saved results to {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
