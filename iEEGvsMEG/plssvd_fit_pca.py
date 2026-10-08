#!/usr/bin/env python3
"""Backfill independent training PCA scores for existing PLSSVD folds only.

python -u plssvd_fit_pca.py --run-dir out/plssvd_eval/full_concatenated/none \
    --cache-dir out/trial_cache

Reconstruct original folds/preprocessing/permutations from the trial cache and
saved assignments. Fit each modality's PCA on training data, transform train and
test, and save pca_scores_XXX.npz. Existing PCA scores are reused. No PLSSVD fit,
metric evaluation, spectrum export, comparison publication, or plotting is run.
"""
from pathlib import Path
import argparse
import json


def backfill_pca(run_dir, cache_dir, scratch_dir=None):
    from plssvd_eval_utils import load_trial_cache
    from plssvd_pca import _pca_artifact
    from plssvd_diagnostics import _backfill_spectra
    root=Path(run_dir).expanduser().resolve()
    options=json.loads((root/'validation_options.json').read_text())
    if options.get('schema_version') not in (4,6):
        raise ValueError('Expected a repeated PLSSVD run directory containing iteration_XXX folders.')
    ids=[i for i in range(options['n_iterations'])
         if (root/f'iteration_{i:03d}'/'COMPLETE.json').is_file()]
    if not ids:raise FileNotFoundError('No completed fit iterations are available.')
    trials=None;computed=skipped=0
    for iteration in ids:
        child=root/f'iteration_{iteration:03d}'
        for fold in range(options['repeats']):
            if _pca_artifact(child,fold) is not None:
                skipped+=1
                continue
            if trials is None:trials=load_trial_cache(Path(cache_dir).expanduser())
            print(f'Fitting PCA: iteration {iteration+1}, fold {fold+1}/{options["repeats"]}',flush=True)
            _backfill_spectra(root,iteration,trials,scratch_dir,include_pca=True,
                              fold_indices=[fold],compute_spectra=False)
            computed+=1
    print(f'PCA scores: {computed} folds computed, {skipped} reused; '
          f'{len(ids)}/{options["n_iterations"]} completed iterations. Evaluation snapshots unchanged.',flush=True)
    return dict(computed_folds=computed,reused_folds=skipped,iterations=ids)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir',type=Path,required=True,help='One existing run, e.g. .../none or .../meg__time_point.')
    parser.add_argument('--cache-dir',type=Path,required=True,help='Original trial cache used for this run.')
    parser.add_argument('--scratch-dir',type=Path,help='Optional local scratch directory for reconstructed fold averages.')
    args=parser.parse_args()
    backfill_pca(args.run_dir,args.cache_dir,args.scratch_dir)


if __name__=='__main__':main()
