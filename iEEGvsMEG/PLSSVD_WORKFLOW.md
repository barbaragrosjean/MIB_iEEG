## Full-data PCA comparison in the notebook

The final section of `plssvd_eval.ipynb` fits separate PCA models to the full
condition-averaged iEEG and concatenated MEG data from `out/trial_cache`.
Run its first cell once to create the fixed PCA reference. Run the following
cell to compare it with the selected run's saved training and test PLSSVD scores.
Select the run using `PERM` and `PERM_TYPE`, and the component count using
`N_COMPONENTS`. `PCA_ABSOLUTE` selects absolute or signed Pearson correlations.

The figure has four rows (iEEG train/test, MEG train/test) and one column per
fold. Each matrix contains every PLSSVD–PCA component pair, with mean and sample
SD across completed iterations, separately for each fold. The available
iteration count is printed; SD is undefined with only one iteration. The figure
and numerical summaries are saved under the selected run's
`full_data_pca_comparison` directory.

This section requires the original trial cache and saved PLSSVD scores, but
does not require `plssvd_fit_pca.py` or recompute PLSSVD fits or metrics. The PCA
reference always uses unpermuted data. Because it includes all trials, this is
a descriptive comparison, not a held-out assessment of PCA generalization.

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
