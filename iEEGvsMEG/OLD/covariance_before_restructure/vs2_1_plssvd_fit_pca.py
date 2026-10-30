#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

from vs2_1_plssvd_pca import fit_full_data_pca, correlate_full_pca, plot_full_pca_comparison

ROOT = Path.cwd()
PCA_ABSOLUTE = True 

def main():
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
    parser.add_argument('--meg-kind', default='full_concatenated', choices=[
        'full_average', 'full_concatenated', 'coverage_average', 'paired_coverage', 'random_control'])
    parser.add_argument('--repeats', '--n-splits', dest='repeats', type=int, default=5,
                        help='Number of shuffled folds with disjoint test trials (default: 5).')
    parser.add_argument('--perm', choices=['ieeg','meg','both','none'], default=None)
    parser.add_argument('--perm-type', choices=['time_cirular_shift','time_circular_shift',
                        'time_block','time_point','space','phase','none'], default=None,
                        help='Surrogate transform; phase uses coherent Fourier phases within each modality.')
    parser.add_argument('--block-seconds', type=float, default=0.36)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--components', '--n-components', dest='n_components', type=int, default=100,
                        help='Maximum fitted dimension; select fewer later in the notebook (default: 100).')
    parser.add_argument('--split-unit', choices=['trial', 'group'], default='trial')
    parser.add_argument('--max-gram-gib', type=float, default=2.0)
    args=parser.parse_args()

    from vs2_1_plssvd_eval_utils import validation_run_name
    root = args.root.expanduser().resolve()
    meg_dir = args.meg_dir or root / 'MEG' / 'dataMEG'
    ieeg_dir = args.ieeg_dir or root / 'ieeg_shortWOBS_fs250'
    cache_dir = args.cache_dir or root / 'out' / 'trial_cache'
    output_dir = args.output_dir or root / 'out' / 'pca_comparison' / args.meg_kind / validation_run_name(args.perm, args.perm_type)
    
    full_pca = fit_full_data_pca(cache_dir, n_components=args.components)

    full_pca_result = correlate_full_pca(output_dir, full_pca, n_components=args.components, absolute=PCA_ABSOLUTE)
    fig = plot_full_pca_comparison(full_pca_result)

    pca_destination = output_dir
    pca_destination.mkdir(exist_ok=True)
    statistic = 'absolute' if PCA_ABSOLUTE else 'signed'
    fig.savefig(pca_destination / f'comparison_k{args.components}_{statistic}.png', dpi=180, bbox_inches='tight')
    
    np.savez_compressed(pca_destination / f'correlations_k{args.components}_{statistic}.npz',
        mean=full_pca_result['mean'], std=full_pca_result['std'], count=full_pca_result['count'],
        iterations=full_pca_result['iterations'])


if __name__=='__main__':main()
