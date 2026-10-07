#!/usr/bin/env python3
"""Batch version of plssvd_eval.ipynb (Python 3.9+).

Run from a cluster checkout containing the project helpers and src.setting:
    python -u plssvd_eval.py --root /path/to/iEEGvsMEG

Defaults preserve the notebook analysis and exclude MEG SUBJ_0038. Outputs
are saved under ROOT/out/plssvd_eval/MEG_KIND/PERMUTATION; trials use ROOT/out/trial_cache.
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
    parser.add_argument('--project-path', type=Path, help='GetInfo project directory; default: PROJECT_PATH as configured, without ROOT prefix.')
    parser.add_argument('--cache-dir', type=Path, help='Default: ROOT/out/trial_cache.')
    parser.add_argument('--scratch-dir', type=Path,
                        help='Temporary fold storage; default: system temporary directory (honors TMPDIR).')
    parser.add_argument('--output-dir', type=Path, help='Default: ROOT/out/plssvd_eval/{MEG_KIND}/{PERMUTATION}')
    parser.add_argument('--trial-metadata-csv', type=Path)
    parser.add_argument('--meg-kind', default='full_concatenated', choices=[
        'full_average', 'full_concatenated', 'coverage_average', 'paired_coverage', 'random_control'])
    parser.add_argument('--repeats', '--n-splits', dest='repeats', type=int, default=5,
                        help='Number of shuffled folds with disjoint test trials (default: 5).')
    parser.add_argument('--n-iterations', type=int, default=100,
                        help='Number of independently shuffled K-fold evaluations (default: 100).')
    parser.add_argument('--perm', choices=['ieeg','meg','both','none'], default=None)
    parser.add_argument('--perm-type', choices=['time_cirular_shift','time_circular_shift',
                        'time_block','time_point','space','none'], default=None)
    parser.add_argument('--block-seconds', type=float, default=0.36)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-components', '--n-components', dest='n_components', type=int, default=100,
                        help='Maximum fitted dimension; select fewer later in the notebook (default: 100).')
    parser.add_argument('--split-unit', choices=['trial', 'group'], default='trial')
    parser.add_argument('--max-gram-gib', type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    for directory in (root, root.parent, root.parent / 'LB'):
        sys.path.insert(0, str(directory))

    import numpy as np
    import pandas as pd
    from src.setting import GetInfo, PROJECT_PATH
    from plssvd_eval_utils import ValidationOptions, prepare_trial_cache, fit_plssvd, validation_run_name

    meg_dir = args.meg_dir or root / 'MEG' / 'dataMEG'
    ieeg_dir = args.ieeg_dir or root / 'ieeg_shortWOBS_fs250'
    cache_dir = args.cache_dir or root / 'out' / 'trial_cache'
    output_dir = args.output_dir or root / 'out' / 'plssvd_eval' / args.meg_kind / validation_run_name(args.perm, args.perm_type)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / 'validation_options.json').exists() or any(output_dir.glob('model_*.npz')):
        raise FileExistsError('Choose a new --output-dir; existing evaluations are not overwritten.')
    options = ValidationOptions(repeats=args.repeats, n_components=args.n_components,
                                n_iterations=args.n_iterations, perm=args.perm, perm_type=args.perm_type,
                                block_seconds=args.block_seconds, seed=args.seed,
                                split_unit=args.split_unit,
                                block_scaling='none', max_gram_gib=args.max_gram_gib)

    meg_subjects = sorted(p.name.removesuffix('_source.p') for p in meg_dir.glob('*_source.p')
                          if p.name != 'SUBJ_0038_source.p')
    ieeg_subjects = sorted(p.name.removesuffix('_epochs.p') for p in ieeg_dir.glob('*_epochs.p'))
    if not meg_subjects or not ieeg_subjects:
        raise FileNotFoundError(f'No subjects found: check {meg_dir} and {ieeg_dir}.')
    print(f'MEG subjects: {len(meg_subjects)}; iEEG subjects: {len(ieeg_subjects)}', flush=True)
    coord, _, electrodes, subjects, regions = GetInfo(
        ieeg_subjects, data_path=str(ieeg_dir), project_path=str(args.project_path if args.project_path is not None else PROJECT_PATH))
    # Preserve the notebook coordinate conversion; exporter receives metres.
    from coverage_matching_utils import ieeg_getinfo_coordinates_m
    coord = ieeg_getinfo_coordinates_m(coord)
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

    # Fit artifacts only. All metrics and figures are computed later in the notebook.
    fit_plssvd(trials, args.meg_kind, options, output_dir=output_dir, scratch_dir=args.scratch_dir)

    print(f'Outputs saved to: {output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
