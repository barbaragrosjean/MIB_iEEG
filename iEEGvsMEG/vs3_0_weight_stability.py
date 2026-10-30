#!/usr/bin/env python3
"""Controlled weight stability: trials only, pairing only, or both.

python -u vs3_0_weight_stability.py --iterations 100 --components 3

"""
from pathlib import Path
from copy import copy
import argparse
import json
import numpy as np
import pandas as pd
from vs3_0_spatial_reliability import (fit_iterations, align_bundle, _load_bundle,
                                 across_iteration_correlations, LEVELS)
from vs3_0_compare_subspace import MODELS, MODALITIES

CONDITIONS = ('trials', 'pairing', 'both')


def controlled_seeds(seed, iteration, variation):
    if variation not in CONDITIONS:
        raise ValueError('variation must be trials, pairing or both')
    split_i = 0 if variation == 'pairing' else iteration
    pair_i = 0 if variation == 'trials' else iteration
    # Separate random streams; iteration zero is identical in all conditions.
    split = np.random.SeedSequence([seed, split_i]).spawn(2)[0]
    pairing = np.random.SeedSequence([seed, pair_i]).spawn(2)[1]
    return int(split.generate_state(1)[0]), int(pairing.generate_state(1)[0])


def half_trial_indices(trials, seed, split_unit):
    from vs2_1_plssvd_eval_utils import _subject_folds
    rng = np.random.default_rng(seed)
    result = {m: {} for m in MODALITIES}
    for m in MODALITIES:
        for subject in getattr(trials, m):
            halves = _subject_folds(subject, rng, split_unit, 2)
            indices = halves[0]['train']
            if any(len(i) < 2 for i in indices):
                raise ValueError('Need at least two selected trials per condition.')
            result[m][subject.subject] = {'train': indices}
    return result


def bilateral_label(value):
    """Remove explicit terminal/prefix hemisphere tokens, preserving internal ROI names."""
    import re
    if pd.isna(value): return value
    value = re.sub(r'^(?:left|right|lh|rh|L|R)[_ .-]+', '', str(value), flags=re.I)
    return re.sub(r'[_ .-]+(?:left|right|lh|rh|L|R)$', '', value, flags=re.I)


def roi_pair_correlations(ieeg, meg, metadata, level='region', block_size=256):
    """Signed Pearson across iterations, then arithmetic mean/sample SD across ROI pairs.

    Arrays: iteration x fixed coverage location x aligned component.
    Each valid channel pair has equal weight. Constant trajectories are excluded.
    No dense all-channel correlation matrix is retained.
    """
    x, y = np.asarray(ieeg,float), np.asarray(meg,float)
    if x.shape != y.shape or x.ndim != 3 or x.shape[1] != len(metadata) or len(x)<2:
        raise ValueError('Need aligned iteration x coverage x component arrays.')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Weight trajectories must be finite.')
    labels = metadata[level].map(bilateral_label).reset_index(drop=True)
    groups = {r: np.flatnonzero(labels.eq(r)) for r in sorted(labels.dropna().unique())}
    rows=[]
    for c in range(x.shape[2]):
        standardized=[]
        for values in (x[:,:,c], y[:,:,c]):
            centered=values-values.mean(0)
            norm=np.linalg.norm(centered,axis=0)
            tolerance=32*np.finfo(float).eps*np.maximum(np.linalg.norm(values,axis=0),np.finfo(float).tiny)
            valid=norm>tolerance
            z=np.divide(centered,norm,out=np.zeros_like(centered),where=valid)
            standardized.append((z,valid))
        (a,va),(b,vb)=standardized
        for ra,ia in groups.items():
            for rb,ib in groups.items():
                aa=ia[va[ia]];bb=ib[vb[ib]]
                count=0;mean=0.;m2=0.
                for start in range(0,len(aa),block_size):
                    for stop in range(0,len(bb),block_size):
                        r=(a[:,aa[start:start+block_size]].T@b[:,bb[stop:stop+block_size]]).clip(-1,1).ravel()
                        n=len(r)
                        if not n:continue
                        delta=r.mean()-mean;total=count+n
                        m2+=np.sum((r-r.mean())**2)+delta**2*count*n/total
                        mean+=delta*n/total;count=total
                rows.append(dict(component=c+1,ieeg_roi=ra,meg_roi=rb,
                    mean_r=mean if count else np.nan,sd_r=np.sqrt(max(0.,m2)/(count-1)) if count>1 else np.nan,
                    n_pairs=count,n_possible_pairs=len(ia)*len(ib),n_iterations=len(x)))
    return rows


def roi_statistics(values, metadata, level='region'):
    """ROI means per iteration, then mean/SD across iterations (ddof=1).

    values: iterations x fixed coverage rows x components. Magnitudes first;
    participant-balanced using fixed iEEG coverage owners in both modalities.
    No zero-filling. Every coverage slot has equal mass within its owner/ROI.
    """
    values = np.abs(np.asarray(values, float))
    if values.ndim != 3 or values.shape[1] != len(metadata) or values.shape[0] < 2:
        raise ValueError('Need >=2 iterations and aligned fixed feature metadata.')
    metadata = metadata.reset_index(drop=True).copy()
    metadata[level] = metadata[level].map(bilateral_label)
    trajectories, summary = [], []
    for roi, rows in metadata.dropna(subset=[level]).groupby(level, sort=True):
        participant_means = [values[:, group.index.to_numpy(), :].mean(axis=1)
                             for _, group in rows.groupby('subject')]
        profile = np.mean(participant_means, axis=0)
        # Mean channel SD exposes variability that can cancel in an ROI average.
        feature_sd = np.mean([values[:, group.index.to_numpy(), :].std(axis=0, ddof=1).mean(axis=0)
                              for _, group in rows.groupby('subject')], axis=0)
        for c in range(values.shape[2]):
            summary.append(dict(roi=roi, component=c+1, mean_magnitude=profile[:, c].mean(),
                sd_across_iterations=profile[:, c].std(ddof=1),
                mean_channel_sd=feature_sd[c], n_channels=len(rows),
                n_subjects=rows.subject.nunique(), n_iterations=len(values)))
            trajectories.extend(dict(roi=roi, component=c+1, iteration=i, magnitude=v)
                                for i, v in enumerate(profile[:, c]))
    return summary, trajectories


def summarize_weights(root, roi_level='region'):
    root = Path(root)
    cfg = json.loads((root/'config.json').read_text())
    if cfg.get('schema_version') != 2 or 'variation' not in cfg:
        raise ValueError('Requires the controlled weight_stability.py fits, not the older combined workflow.')
    if roi_level not in LEVELS:
        raise ValueError('Unknown ROI level')
    for i in range(cfg['iterations']):
        if not (root/f'iteration_{i:03d}'/'COMPLETE.json').exists():
            raise ValueError(f'Iteration {i} is incomplete')
    marker = root/'WEIGHT_SUMMARY_COMPLETE.json'
    if marker.exists(): marker.unlink()
    anchor = pd.read_csv(root/'iteration_000/ieeg_features.csv', dtype={'subject': str}).reset_index(drop=True)
    matrices, roi_rows, trajectories, feature_rows, matches = [], [], [], [], []
    cross_rows, temporal_rows = [], []
    for model in cfg['models']:
        ref = _load_bundle(root/'iteration_000'/f'{model}_A.npz')
        values = {m: [] for m in MODALITIES}
        scores = {m: [] for m in MODALITIES}
        for i in range(cfg['iterations']):
            folder = root/f'iteration_{i:03d}'
            meta = pd.read_csv(folder/'ieeg_features.csv', dtype={'subject': str})
            if not meta[['subject', 'source_index', 'x', 'y', 'z']].equals(anchor[['subject', 'source_index', 'x', 'y', 'z']]):
                raise ValueError('iEEG coverage rows changed across iterations')
            current, transforms = align_bundle(ref, _load_bundle(folder/f'{model}_A.npz'), model)
            for m in MODALITIES:
                values[m].append(current[m+'_weights'])
                scores[m].append(current[m+'_train_scores'])
                cols, signs = transforms[m]
                matches.extend(dict(model=model, modality=m, iteration=i, component=c+1,
                    original_component=int(col+1), sign=float(sign)) for c, (col, sign) in enumerate(zip(cols, signs)))
        cross_rows.extend(dict(model=model, **row) for row in roi_pair_correlations(
            np.stack(values['ieeg']), np.stack(values['meg']), anchor, roi_level))
        for m in MODALITIES:
            matrices.extend(across_iteration_correlations(values[m], model=model, modality=m))
            temporal_rows.extend(across_iteration_correlations(scores[m], model=model, modality=m))
            stack = np.stack(values[m])
            summaries, traces = roi_statistics(stack, anchor, roi_level)
            roi_rows.extend(dict(model=model, modality=m, **row) for row in summaries)
            trajectories.extend(dict(model=model, modality=m, **row) for row in traces)
            means, sds = abs(stack).mean(0), abs(stack).std(0, ddof=1)
            for j in range(len(anchor)):
                for c in range(cfg['components']):
                    feature_rows.append(dict(model=model, modality=m, coverage_row=j,
                        subject=anchor.subject.iloc[j], roi=anchor[roi_level].iloc[j], component=c+1,
                        mean_magnitude=means[j, c], sd_across_iterations=sds[j, c]))
    frames = dict(timecourse_correlations=pd.DataFrame(temporal_rows), roi_cross_correlations=pd.DataFrame(cross_rows), iteration_correlations=pd.DataFrame(matrices),
        roi_stability=pd.DataFrame(roi_rows, columns=['model', 'modality', 'roi', 'component',
            'mean_magnitude', 'sd_across_iterations', 'mean_channel_sd', 'n_channels', 'n_subjects', 'n_iterations']),
        roi_trajectories=pd.DataFrame(trajectories, columns=['model','modality','roi','component','iteration','magnitude']),
        feature_stability=pd.DataFrame(feature_rows), component_matching=pd.DataFrame(matches))
    for name, frame in frames.items(): frame.to_csv(root/f'{name}.csv', index=False)
    marker.write_text(json.dumps(dict(roi_level=roi_level, summary_version=4,
        hemispheres='pooled before aggregation',
        cross_roi='Signed-weight Pearson across iterations; mean and sample SD across valid channel pairs; equal pair weighting.',
        error_bars='Sample SD across iterations of participant-balanced ROI mean magnitudes; not a CI.',
        roi_definition='Fixed iEEG coverage-location labels for BOTH modalities; MEG native source/owner may change.',
        duplicate_policy='Each coverage slot has equal mass within its coverage participant/ROI; repeated native MEG sources remain slots.')))
    return frames



def load_weight_summary(root, *, model, component, meg_kind, block_scaling, n_components):
    """Load compatible saved tables only; never fit or summarize from a plot cell."""
    root = Path(root)
    config = json.loads((root/'config.json').read_text())
    marker = json.loads((root/'WEIGHT_SUMMARY_COMPLETE.json').read_text())
    if marker.get('summary_version', 0) < 4:
        raise ValueError('Temporal summary missing; run --summarize-only.')
    if (config['meg_kind'] != meg_kind or config['block_scaling'] != block_scaling
            or config['components'] != n_components or model not in config['models']):
        raise ValueError('Saved stability settings differ; select matching results or a new output directory.')
    if not 1 <= component <= n_components:
        raise ValueError('Choose a component within N_COMPONENTS.')
    names = ['iteration_correlations', 'timecourse_correlations', 'roi_stability', 'roi_cross_correlations']
    tables = {}
    for name in names:
        frame = pd.read_csv(root/f'{name}.csv')
        tables[name] = frame[(frame.model == model) & (frame.component == component)].copy()
    return dict(config=config, summary=marker, **tables)



def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path(__file__).resolve().parent
    p.add_argument('--cache-dir', type=Path, default=base/'out/trial_cache')
    p.add_argument('--output-dir', type=Path, default=base/'out/weight_stability/paired_coverage')
    p.add_argument('--meg-kind', choices=['full_average','full_concatenated','coverage_average','paired_coverage','random_control'], default='paired_coverage')
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--components', type=int, default=3)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--split-unit', choices=['trial','group'], default='trial')
    p.add_argument('--block-scaling', choices=['none','equal_variance'], default='equal_variance')
    p.add_argument('--max-gram-gib', type=float, default=2.)
    for name in ('scratch-dir','ieeg-coords-dir','meg-coords-dir','atlas-image','atlas-labels'):
        p.add_argument('--'+name, type=Path)
    for m in MODALITIES: p.add_argument('--'+m+'-coords-unit', choices=['mm','m'], default='mm')
    p.add_argument('--pool-hemispheres', action='store_true')
    p.add_argument('--roi-level', choices=LEVELS, default='region')
    p.add_argument('--summarize-only', action='store_true')
    args = p.parse_args()
    if args.iterations < 2 or args.components < 1: p.error('Need >=2 iterations and >=1 component')
    for condition in args.conditions:
        run = copy(args); run.variation = condition; run.weight_only = True
        run.output_dir = args.output_dir/condition
        if not args.summarize_only: fit_iterations(run)
        summarize_weights(run.output_dir, args.roi_level)
        print(f'Completed {condition}: {run.output_dir}', flush=True)


if __name__ == '__main__': main()
