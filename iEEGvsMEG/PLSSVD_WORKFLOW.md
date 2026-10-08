# PLSSVD: fit once, evaluate later

## Cluster fitting

```bash
python plssvd_eval.py --max-components 100 --n-iterations 100 --n-splits 5
```

`--n-components` remains an alias for `--max-components`. Both default to 100.
Every training fold must support this rank; the job fails explicitly if it does not.
Use a new output directory when changing fit settings. Existing fits are never overwritten.
Permutation selection is unchanged: `--perm ieeg|meg|both|none` and
`--perm-type time_cirular_shift|time_block|time_point|space|none`.
Both default to no permutation; time blocks default to 0.36 seconds.

The fitting job does not evaluate metrics or create figures. `fit_plssvd(...)`
is the Python entry point; `validate_plssvd` is retained as an alias and now
returns fit metadata, not evaluation tables.

## Notebook evaluation

Set `N_COMPONENTS` near the beginning of `plssvd_eval.ipynb`, then rerun the
loading and plotting cells. Any integer from 1 to the fitted maximum works;
`None` uses the full fitted rank.

```python
result = load_plssvd_results(OUTPUT_DIR, n_components=10)
plot_plssvd_validation(result)  # first available iteration
plot_iteration_summary(OUTPUT_DIR, n_components=10)
plot_permutation_comparison(RUNS_DIR, n_components=10)
```

The loader reports completed/requested iterations and selected/fitted components.
It excludes unfinished iterations and supports interrupted jobs with at least
one completed iteration. Summaries are computed in memory. Figure filenames
include k so results for different selections do not overwrite each other.
The selected count applies to correlations, covariance, reconstruction,
prediction, stability, time courses, and the permutation comparison.

## Saved data

- Per run: settings, time/condition axes, fixed matching, pairing and feature metadata.
- Per iteration: settings, compressed trial split audit, completion marker.
- Per fold: model weights/means/scales; training preprocessing; component scores
  and compact sufficient statistics in a separate `scores_XXX.npz` file.

No feature-sized prediction maps or redundant metric CSVs are written by fitting.
Evaluation reads the scores/statistics rather than decompressing the large weight
arrays. Raw trials are not needed. Keep the run directory structure when copying.

For prediction, training score-target products are reduced to small component-space
matrices. These reproduce squared prediction errors at any leading k, with a
training-only ridge predictor and k-specific ridge penalty. Reconstruction retains
training-centered signal energy and the fitted weight Gram matrix. The full
cross-covariance denominator remains independent of selected k. None of these
statistics changes the fitted model or uses test data for calibration.

## Existing output compatibility

Older runs still load at their original component count, including interrupted
repeated runs. They do not contain all statistics required to recompute prediction
and reconstruction for a smaller k, so the loader rejects that request explicitly.
New-format fits enable the full flexible workflow. Existing outputs are not migrated
or modified automatically.

Changing k after examining held-out results is exploratory model comparison;
it does not create an independent estimate of the performance of the selected k.

## Selected permutation diagnostics

The final notebook section uses the same `PERM`, `PERM_TYPE`, and `N_COMPONENTS`
settings as the earlier sections. It compares the selected run with `none`,
restricting both to their shared completed iteration IDs and verifying actual
trial assignments. It displays component-wise train/test correlations and
covariances, paired held-out metrics, within-iteration differences, and temporal
spectra. Figures and numerical tables are exported to `diagnostics_kK` inside the
selected run. Fold means precede across-iteration means and SDs.

Future fits also save full temporal input spectra in the score artifacts. Existing
runs can recover these from the original trial cache without refitting PLSSVD;
small `temporal_spectra_XXX.npz` sidecars are saved after checking reconstructed
scores against the fitted scores. This reconstruction can be computationally
expensive. If the original cache is unavailable, the section explicitly shows
selected-score spectra for both runs instead. This fallback does not measure
full input dimensionality. The spectrum table includes singular values, normalized
squared singular values, and participation-ratio effective rank.

## Independent PCA comparison

The final PLSSVD/PCA section fits PCA independently to the training iEEG and
training MEG matrices, then correlates each modality's held-out PLSSVD scores
with its own held-out PCA scores across time. PCA uses the same preprocessing
and permutation as that run, without whitening or test-based component alignment.
The plotted rows are PLSSVD components and columns are PCA components ordered by
training variance. Equal indices need not correspond to the same direction.

`PERM`, `PERM_TYPE`, and `N_COMPONENTS` select the analysis. `PCA_ABSOLUTE=True`
shows mean absolute Pearson correlations; False shows signed correlations with
signs determined independently from each model's training scores. The matrices
show fold-averaged iteration means and across-iteration SDs. Selected permutations
are compared with baseline on matched iterations, including difference panels;
selecting no permutation shows baseline alone. Tables also retain fold-level
signed Pearson correlations and valid fold/iteration counts.

Future fit jobs save training/test PCA scores in `scores_XXX.npz`, for every
fitted component. Existing fits require the original trial cache to reconstruct
their folds and fit PCA once. This writes `pca_scores_XXX.npz` sidecars and never
refits PLSSVD. Without that cache or saved PCA scores, the comparison fails with
an explicit message: independent PCA scores cannot be inferred from PLSSVD scores
or singular values alone. Subsequent component-count changes reuse saved scores.
