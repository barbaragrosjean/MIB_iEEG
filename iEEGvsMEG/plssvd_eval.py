#!/usr/bin/env python3
"""Batch version of plssvd_eval.ipynb (Python 3.9+).

Run from a cluster checkout containing the project helpers and src.setting:
    python -u plssvd_eval.py --root /path/to/iEEGvsMEG

Defaults preserve the notebook analysis and exclude MEG SUBJ_0038. Outputs
are saved under ROOT/out/plssvd_eval; trials use ROOT/out/trial_cache.
Requires the notebook's Python dependencies, including mat73 for raw MEG.
Tests concern new trials from the same participants, not new participants.
"""
from pathlib import Path
import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent,
                        help='Project/data root (default: directory containing this script).')
    parser.add_argument('--meg-dir', type=Path, help='Subject discovery directory (default: ROOT/MEG/dataMEG).')
    parser.add_argument('--ieeg-dir', type=Path, help='Epoch directory (default: ROOT/ieeg_shortWOBS_fs250).')
    parser.add_argument('--meg-raw-dir', type=Path, default=Path(
        '/projects/MINDLAB2025_MEG-Auditory_Cognitive_Maps/scratch/APR2020_Block3_SingleTrial_BarbaraNikita'))
    parser.add_argument('--cache-dir', type=Path, help='Default: ROOT/out/trial_cache.')
    parser.add_argument('--scratch-dir', type=Path,
                        help='Temporary fold storage; default: system temporary directory (honors TMPDIR).')
    parser.add_argument('--output-dir', type=Path, help='Default: ROOT/out/plssvd_eval; use a different directory for each configuration.')
    parser.add_argument('--trial-metadata-csv', type=Path)
    parser.add_argument('--meg-kind', default='paired_coverage', choices=[
        'full_average', 'full_concatenated', 'coverage_average', 'paired_coverage', 'random_control'])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--n-null', type=int, default=199)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--candidates', type=int, nargs='+', default=[1, 2, 3, 5, 10])
    parser.add_argument('--subject-fraction', type=float, default=0.8)
    parser.add_argument('--split-unit', choices=['trial', 'group'], default='trial')
    parser.add_argument('--max-gram-gib', type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    for directory in (root, root.parent, root.parent / 'LB'):
        sys.path.insert(0, str(directory))

    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from src.setting import GetInfo, PROJECT_PATH
    from plssvd_eval_utils import ValidationOptions, prepare_trial_cache, validate_plssvd, plot_plssvd_validation

    meg_dir = args.meg_dir or root / 'MEG' / 'dataMEG'
    ieeg_dir = args.ieeg_dir or root / 'ieeg_shortWOBS_fs250'
    cache_dir = args.cache_dir or root / 'out' / 'trial_cache'
    output_dir = args.output_dir or root / 'out' / 'plssvd_eval'
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
    options = ValidationOptions(repeats=args.repeats, candidates=tuple(args.candidates),
                                n_null=args.n_null, seed=args.seed,
                                subject_fraction=args.subject_fraction, split_unit=args.split_unit,
                                block_scaling='none', max_gram_gib=args.max_gram_gib)

    meg_subjects = sorted(p.name.removesuffix('_source.p') for p in meg_dir.glob('*_source.p')
                          if p.name != 'SUBJ_0038_source.p')
    ieeg_subjects = sorted(p.name.removesuffix('_epochs.p') for p in ieeg_dir.glob('*_epochs.p'))
    if not meg_subjects or not ieeg_subjects:
        raise FileNotFoundError(f'No subjects found: check {meg_dir} and {ieeg_dir}.')
    print(f'MEG subjects: {len(meg_subjects)}; iEEG subjects: {len(ieeg_subjects)}', flush=True)
    coord, _, electrodes, subjects, regions = GetInfo(
        ieeg_subjects, data_path=str(ieeg_dir), project_path=str(root.parent / PROJECT_PATH))
    # Preserve the notebook coordinate conversion; exporter receives metres.
    coord = np.asarray(coord)
    coord = np.where(abs(coord) > 100, coord / 1000, coord)
    coord = np.where(abs(coord) > 100, coord / 1000, coord)
    coord = coord / 1000
    metadata = pd.DataFrame(coord, columns=['x', 'y', 'z'])
    metadata['subject'] = subjects
    metadata['channel'] = electrodes
    metadata['region'] = regions
    metadata['channel_index'] = metadata.groupby('subject', sort=False).cumcount()
    metadata.to_csv(output_dir / 'electrode_metadata.csv', index=False)
    configuration = dict(vars(args), root=root, meg_dir=meg_dir, ieeg_dir=ieeg_dir,
                         cache_dir=cache_dir, output_dir=output_dir,
                         meg_subjects=meg_subjects, ieeg_subjects=ieeg_subjects)
    (output_dir / 'run_config.json').write_text(json.dumps(configuration, default=str, indent=2))

    trials = prepare_trial_cache(args.meg_raw_dir, ieeg_dir, cache_dir, meg_subjects,
                                 ieeg_subjects, metadata, conditions=(1, 2),
                                 ieeg_coordinate_unit='m', meg_coordinate_unit='m',
                                 trial_metadata_csv=args.trial_metadata_csv)
    trial_counts = pd.DataFrame([
        dict(modality=modality, subject=subject.subject, condition=condition,
             n_trials=len(array), n_channels=array.shape[1], n_times=array.shape[2])
        for modality in ('ieeg', 'meg') for subject in getattr(trials, modality)
        for condition, array in zip(trials.conditions, subject.data)])
    trial_counts.to_csv(output_dir / 'trial_counts.csv', index=False)
    print(trial_counts.to_string(index=False), flush=True)
    np.savez_compressed(output_dir / 'trial_axes.npz', times=trials.times, conditions=trials.conditions)

    # Saves tables, partition/matching audits, model weights, scores, prediction
    # maps, preprocessing parameters and primary null distributions.
    result = validate_plssvd(trials, args.meg_kind, options, output_dir=output_dir, scratch_dir=args.scratch_dir)
    for name in ('summary', 'selection', 'components', 'null_tests'):
        print(f'\n{name}\n{result[name].round(3).to_string(index=False)}', flush=True)
    plot_plssvd_validation(result, output_dir=output_dir, show=False)

    primary = result['components'].query('repeat == 0')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for modality in ('ieeg', 'meg'):
        axes[0].plot(primary.component, primary[f'{modality}_pattern_split_half_r'], 'o-', label=modality)
        axes[1].plot(primary.component, primary[f'{modality}_contrast_split_half_r'], 'o-', label=modality)
    for ax, title in zip(axes, ('Forward-pattern reliability', 'Condition-contrast reliability')):
        ax.set(title=title, xlabel='Component', ylabel='Test-half Pearson r', ylim=(-1, 1))
        ax.legend()
    for extension in ('png', 'pdf'):
        fig.savefig(output_dir / f'primary_reliability.{extension}', dpi=200)
    plt.close(fig)
    print(f'Outputs saved to: {output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
