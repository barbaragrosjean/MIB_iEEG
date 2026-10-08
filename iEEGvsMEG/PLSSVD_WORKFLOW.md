
## Optional PCA-only backfill

To add missing PCA scores to one existing run without evaluating metrics,
exporting spectra, or updating figure snapshots:

```bash
python -u plssvd_fit_pca.py \
  --run-dir out/plssvd_eval/full_concatenated/none \
  --cache-dir out/trial_cache
```

For a permutation, replace `none` with its run folder (e.g., `meg__time_point`).
Add `--scratch-dir /path/to/local/scratch` if appropriate. The command reconstructs
saved folds and preprocessing, verifies them against saved PLSSVD scores, fits
separate training-only PCA models, and saves training/test PCA scores at the
original fitted component count. It processes one fold at a time and atomically
writes `iteration_XXX/pca_scores_YYY.npz`. Completed PCA folds are skipped on rerun.
No PLSSVD model, metric checkpoint, spectrum file, or notebook snapshot is changed.
Stored PLSSVD scores alone cannot supply the original feature-space PCA input;
the original trial cache is required only for folds whose PCA scores are missing.
