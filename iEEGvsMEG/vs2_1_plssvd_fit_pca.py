#!/usr/bin/env python3
"""Compare saved PLSSVD scores with fixed, full-data PCA of the selected MEG dataset."""
from pathlib import Path
import argparse
import json
import numpy as np
from vs2_1_plssvd_pca import fit_full_data_pca, correlate_full_pca


def comparison_path(run_dir, components, absolute=True, n_iterations=None):
    count = 'all' if n_iterations is None else str(n_iterations)
    statistic = 'absolute' if absolute else 'signed'
    return Path(run_dir)/'full_pca'/f'k_{components:03d}_{statistic}_iterations_{count}.npz'


def run(run_dir, cache_dir, components, *, meg_kind, block_scaling, absolute=True,
        n_iterations=None, scratch_dir=None):
    config = json.loads((Path(run_dir)/'validation_options.json').read_text())
    if config['meg_kind'] != meg_kind or config['block_scaling'] != block_scaling:
        raise ValueError('Requested MEG construction/scaling differs from the saved PLSSVD run.')
    # Reproduce fit_plssvd's fixed random association and random-source selection.
    matching_sequence = np.random.SeedSequence(config['seed']).spawn(3)[0]
    matching_seed = int(np.random.default_rng(matching_sequence).integers(2**31-1))
    reference = fit_full_data_pca(cache_dir, components, scratch_dir, meg_kind=meg_kind,
                                 seed=matching_seed, block_scaling=block_scaling)
    result = correlate_full_pca(run_dir, reference, components, absolute, n_iterations)
    destination = comparison_path(run_dir, components, absolute, n_iterations)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, mean=result['mean'], std=result['std'], count=result['count'])
    metadata = {key:value for key,value in result.items() if key not in ('mean','std','count')}
    metadata['fit_options'] = config
    destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    return destination


def load_comparison(run_dir, components, *, meg_kind, block_scaling, absolute=True, n_iterations=None):
    from vs2_1_plssvd_results import selected_iterations
    destination = comparison_path(run_dir, components, absolute, n_iterations)
    metadata = json.loads(destination.with_suffix('.json').read_text())
    config = json.loads((Path(run_dir)/'validation_options.json').read_text())
    ids = [i for i in range(config['n_iterations']) if (Path(run_dir)/f'iteration_{i:03d}'/'COMPLETE.json').exists()]
    if (metadata['fit_options'] != config or metadata['meg_kind'] != meg_kind
        or metadata['block_scaling'] != block_scaling or metadata['iterations'] != selected_iterations(ids, n_iterations)):
        raise ValueError('PCA comparison differs from the requested run/settings; recompute it.')
    with np.load(destination) as saved: return dict(metadata, **dict(saved))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--meg-kind', required=True, choices=['full_average','full_concatenated','coverage_average','paired_coverage','random_control'])
    parser.add_argument('--block-scaling', choices=['none','equal_variance'], default='equal_variance')
    parser.add_argument('--components', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--absolute', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--scratch-dir', type=Path)
    args = parser.parse_args()
    print(run(args.run_dir, args.cache_dir, args.components, meg_kind=args.meg_kind,
              block_scaling=args.block_scaling, absolute=args.absolute,
              n_iterations=args.iterations, scratch_dir=args.scratch_dir))


if __name__ == '__main__': main()
