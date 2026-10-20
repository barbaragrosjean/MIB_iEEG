#!/usr/bin/env python3
"""Controlled weight stability: trials only, pairing only, or both.

python -u weight_stability.py --iterations 100 --components 3
Repeat with --summarize-only to reuse fitted weights. No held-out-half or
multiscale bootstrap analyses are performed by this workflow.
"""
from pathlib import Path
from copy import copy
import argparse
import json
import numpy as np
import pandas as pd
from spatial_reliability import (fit_iterations, align_bundle, _load_bundle,
                                 across_iteration_correlations, LEVELS)
from compare_subspace import MODELS, MODALITIES

CONDITIONS = ('trials', 'pairing', 'both')
TITLES = dict(trials='Trials only', pairing='Pairing only', both='Trials and pairing')


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
    from plssvd_eval_utils import _subject_folds
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
    cross_rows = []
    for model in cfg['models']:
        ref = _load_bundle(root/'iteration_000'/f'{model}_A.npz')
        values = {m: [] for m in MODALITIES}
        for i in range(cfg['iterations']):
            folder = root/f'iteration_{i:03d}'
            meta = pd.read_csv(folder/'ieeg_features.csv', dtype={'subject': str})
            if not meta[['subject', 'source_index', 'x', 'y', 'z']].equals(anchor[['subject', 'source_index', 'x', 'y', 'z']]):
                raise ValueError('iEEG coverage rows changed across iterations')
            current, transforms = align_bundle(ref, _load_bundle(folder/f'{model}_A.npz'), model)
            for m in MODALITIES:
                values[m].append(current[m+'_weights'])
                cols, signs = transforms[m]
                matches.extend(dict(model=model, modality=m, iteration=i, component=c+1,
                    original_component=int(col+1), sign=float(sign)) for c, (col, sign) in enumerate(zip(cols, signs)))
        cross_rows.extend(dict(model=model, **row) for row in roi_pair_correlations(
            np.stack(values['ieeg']), np.stack(values['meg']), anchor, roi_level))
        for m in MODALITIES:
            matrices.extend(across_iteration_correlations(values[m], model=model, modality=m))
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
    frames = dict(roi_cross_correlations=pd.DataFrame(cross_rows), iteration_correlations=pd.DataFrame(matrices),
        roi_stability=pd.DataFrame(roi_rows, columns=['model', 'modality', 'roi', 'component',
            'mean_magnitude', 'sd_across_iterations', 'mean_channel_sd', 'n_channels', 'n_subjects', 'n_iterations']),
        roi_trajectories=pd.DataFrame(trajectories, columns=['model','modality','roi','component','iteration','magnitude']),
        feature_stability=pd.DataFrame(feature_rows), component_matching=pd.DataFrame(matches))
    for name, frame in frames.items(): frame.to_csv(root/f'{name}.csv', index=False)
    marker.write_text(json.dumps(dict(roi_level=roi_level, summary_version=3,
        hemispheres='pooled before aggregation',
        cross_roi='Signed-weight Pearson across iterations; mean and sample SD across valid channel pairs; equal pair weighting.',
        error_bars='Sample SD across iterations of participant-balanced ROI mean magnitudes; not a CI.',
        roi_definition='Fixed iEEG coverage-location labels for BOTH modalities; MEG native source/owner may change.',
        duplicate_policy='Each coverage slot has equal mass within its coverage participant/ROI; repeated native MEG sources remain slots.')))
    return frames


def plot_weight_stability(root, component=1, roi_error='sd_across_iterations'):
    """Iteration heatmaps, bilateral ROI means, and cross-modal ROI correlation mean/SD."""
    import matplotlib.pyplot as plt
    root = Path(root)
    cfg = json.loads((root/'config.json').read_text())
    if not (root/'WEIGHT_SUMMARY_COMPLETE.json').exists():
        raise ValueError('Run weight_stability.py --summarize-only first')
    marker = json.loads((root/'WEIGHT_SUMMARY_COMPLETE.json').read_text())
    if marker.get('summary_version', 0) < 3:
        summarize_weights(root, marker.get('roi_level', 'region'))
    if component not in range(1, cfg['components']+1): raise ValueError('Choose a saved component')
    if roi_error not in ('sd_across_iterations', 'mean_channel_sd'): raise ValueError('Unknown ROI variability measure')
    data = pd.read_csv(root/'iteration_correlations.csv')
    roi = pd.read_csv(root/'roi_stability.csv')
    fig, axes = plt.subplots(3, 2, figsize=(11, 13), layout='constrained')
    for row, model in enumerate(cfg['models']):
        for col, m in enumerate(MODALITIES):
            matrix = np.full((cfg['iterations'], cfg['iterations']), np.nan)
            subset = data.query('model == @model and modality == @m and component == @component')
            for r in subset.itertuples():
                matrix[r.iteration_a, r.iteration_b] = matrix[r.iteration_b, r.iteration_a] = r.magnitude_r
            im = axes[row, col].imshow(matrix, vmin=-1, vmax=1, cmap='RdBu_r')
            axes[row, col].set(title=f'{model} / {m}', xlabel='Iteration', ylabel='Iteration')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Pearson r between spatial weight magnitudes')
    title = TITLES[cfg['variation']]+f' · component {component}'
    fig.suptitle(title+'\nMEG rows are fixed electrode coverage locations, not fixed native sources')
    rois = sorted(roi.roi.dropna().unique())
    dots, axes = plt.subplots(3, 2, figsize=(max(13, len(rois)*.38), 12), layout='constrained')
    for row, model in enumerate(cfg['models']):
        for col, m in enumerate(MODALITIES):
            ax = axes[row, col]
            sub = roi.query('model == @model and modality == @m and component == @component').set_index('roi').reindex(rois)
            ax.errorbar(np.arange(len(rois)), sub.mean_magnitude, yerr=sub[roi_error], fmt='o',
                        capsize=3, markersize=4, color=f'C{col}')
            ax.set(title=f'{model} / {m}', ylabel='Mean |weight|', xticks=np.arange(len(rois)), xticklabels=rois)
            ax.tick_params(axis='x', labelrotation=90)
            ax.axhline(0, color='grey', lw=.5)
            if not rois: ax.text(.5, .5, 'No labelled coverage locations', transform=ax.transAxes, ha='center')
    error_label = ('SD across iterations of ROI means' if roi_error == 'sd_across_iterations'
                   else 'mean of individual-channel SDs across iterations')
    dots.suptitle(title+'\nParticipant-balanced ROI mean magnitude ± '+error_label+' (not confidence intervals)')
    cross = pd.read_csv(root/'roi_cross_correlations.csv')
    cross_fig, axes = plt.subplots(len(cfg['models']), 2,
        figsize=(max(14,len(rois)*.65),max(6,len(cfg['models'])*5)), squeeze=False, layout='constrained')
    for row, model in enumerate(cfg['models']):
        selected = cross.query('model == @model and component == @component')
        for col, (metric,label) in enumerate([('mean_r','Mean Pearson r'),('sd_r','SD across channel-pair r')]):
            ax=axes[row,col]
            matrix=selected.pivot(index='ieeg_roi',columns='meg_roi',values=metric).reindex(index=rois,columns=rois)
            if not rois:
                ax.text(.5,.5,'No labelled coverage locations',ha='center');continue
            cmap=__import__('matplotlib').colormaps['RdBu_r' if col==0 else 'viridis'].copy()
            cmap.set_bad('lightgrey')
            im=ax.imshow(np.ma.masked_invalid(matrix.to_numpy()),cmap=cmap,
                         vmin=-1 if col==0 else 0,vmax=1 if col==0 else np.sqrt(2))
            ax.set(title=f'{model}: {label}',xlabel='MEG ROI',ylabel='iEEG ROI',
                xticks=np.arange(len(rois)),yticks=np.arange(len(rois)),xticklabels=rois,yticklabels=rois)
            ax.tick_params(axis='x',labelrotation=90)
            cross_fig.colorbar(im,ax=ax)
    cross_fig.suptitle(title+'\nSigned weights correlated across iterations, then grouped by bilateral ROI pairs\n'
                      'Fixed coverage locations; grey = undefined; SD is pair heterogeneity, not a confidence interval')
    return fig, dots, cross_fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    base = Path(__file__).resolve().parent
    p.add_argument('--cache-dir', type=Path, default=base/'out/trial_cache')
    p.add_argument('--output-dir', type=Path, default=base/'out/weight_stability/paired_coverage')
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
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for condition in args.conditions:
        run = copy(args); run.variation = condition; run.weight_only = True
        run.output_dir = args.output_dir/condition
        if not args.summarize_only: fit_iterations(run)
        summarize_weights(run.output_dir, args.roi_level)
        for name, figure in zip(('iteration_heatmaps','roi_weight_stability','roi_cross_correlations'), plot_weight_stability(run.output_dir)):
            figure.savefig(run.output_dir/f'{name}.png', dpi=180, bbox_inches='tight')
            plt.close(figure)
        print(f'Completed {condition}: {run.output_dir}', flush=True)


if __name__ == '__main__': main()
