# PLSSVD: fit, evaluate, plot

The notebook now draws figures from prepared evaluation snapshots. It does not
load fitted weights, reconstruct trial averages, fit PCA, calculate prediction
metrics, or invoke evaluation automatically. Existing fit directories are reused.

## 1. Fit once on the cluster

```bash
python -u plssvd_eval.py --max-components 100 --n-iterations 100 --n-splits 5
```

Skip this stage if the fits already exist. `--n-components` remains an alias for
`--max-components`. Every training fold must support the fitted rank. Fit files
are never overwritten. Participant matching and preprocessing are unchanged.

## 2. Prepare evaluations on the cluster

For baseline and one selected permutation:

```bash
python -u plssvd_postprocess.py \
  --runs-dir out/plssvd_eval/full_concatenated \
  --components 5 10 25 \
  --perm meg --perm-type time_point
```

For every existing run:

```bash
python -u plssvd_postprocess.py \
  --runs-dir out/plssvd_eval/full_concatenated \
  --components 5 10 25 --all-runs
```

Older fits may lack saved PCA scores or full input spectra. To recover them once,
add these options to the evaluation command:

```bash
--cache-dir /path/to/out/trial_cache --scratch-dir /path/to/local/scratch
```

This reconstructs one fold at a time and fits independent **training-only PCA**;
it does not refit PLSSVD. Reconstructed scores are checked against saved scores.
Without the cache, the worker still evaluates the available metrics and explicitly
marks missing PCA/spectra. The notebook does not substitute projected spectra for
missing full input spectra. A worker still needs sufficient RAM for one fold's
PCA/Gram calculation; moving it outside the notebook does not eliminate that cost.

### Reuse and resume

- Each shared fold evaluation and each k-specific metric calculation is saved
  atomically. Rerunning the same command skips valid completed work.
- Component and PCA correlation matrices are calculated for all fitted components
  once. Different k selections reuse their prefixes.
- Reconstruction and ridge-prediction metrics are cached separately for each k.
  The k-specific predictor uses only compact training statistics, without refitting
  PLSSVD or rereading raw trials.
- Cache signatures include the evaluator version and source file paths, sizes,
  and modification times. Changed inputs invalidate dependent evaluations.
- Only complete fit iterations enter published figures. Partially evaluated
  iterations retain their fold checkpoints but are excluded from summaries.
- On a handled interruption (including a Python `MemoryError`), the worker attempts
  to publish fully evaluated iterations before propagating the error. A forcibly
  killed process retains atomic checkpoints and the previous published snapshot.

To publish fully evaluated iterations after a forced stop, without evaluating
additional folds, repeat the relevant command with `--publish-only`. To continue
work, repeat it without that option. Neither command reruns PLSSVD fitting.

Published snapshots are immutable. The `CURRENT.json` pointer changes only after
all files are written. Earlier snapshots are retained; their folders can be
archived when no longer needed. This prevents a plotting session from observing
partially overwritten summary files. Do not delete the shared fold checkpoints
if you want subsequent evaluation jobs to reuse them.

## 3. Plot locally

Copy each run's `evaluation` directory, keeping this layout:

```text
out/plssvd_eval/full_concatenated/
  none/evaluation/...
  meg__time_point/evaluation/...
```

The original fit files and trial cache are not required on the plotting machine.
Open `plssvd_eval.ipynb` and set `PERM`, `PERM_TYPE`, and an explicitly prepared
`N_COMPONENTS`. `PCA_ABSOLUTE` selects signed versus absolute PCA correlations.
Changing plot settings never launches computation. An unprepared k produces an
instruction to run the separate evaluation command.

The notebook preserves first-iteration plots, the mean/SD iteration overview,
the 3-by-4 permutation overview, matched diagnostics, and the PLSSVD/PCA matrices.
The overview uses all prepared iterations per run. Matched diagnostics use only
shared iteration IDs, verified against actual trial-assignment hashes, compatible
settings, and fixed participant/source matching. When an underlying snapshot is
updated, stale matched comparisons are rejected until the evaluation command
republishes them.

## Evaluation files

Under each run:

```text
evaluation/
  shared/iteration_000/fold_000.npz   # reusable all-rank summaries
  k_010/
    iteration_000/fold_000.npz        # k-specific metrics
    CURRENT.json                     # atomically published snapshot pointer
    snapshot_<id>/                    # compact arrays and metric tables
    comparison/CURRENT.json          # selected run versus baseline
    comparison/snapshot_<id>/
```

Only the pointed-to snapshots are read for figures. They contain means, sample SDs,
component correlations, metric tables, and first-iteration score time courses.
The evaluator processes source folds sequentially and uses online array summaries;
no collection of model weights or all-iteration score cubes is loaded at once.

The older computational helpers (`load_plssvd_results`, `permutation_diagnostics`,
`plssvd_pca_comparison`) remain available for scripts and compatibility, but are
not called by the notebook. For figures use `plssvd_figures.py`.

## Interpretation and compatibility

All metrics use held-out condition-averaged time courses. Prediction calibration
and PCA use training data only. Fold means precede across-iteration SDs; repeated
iterations reuse trials and are not independent population replicates.

The PCA matrices compare PLSSVD with independently fitted PCA within each modality.
They preserve native ranks rather than matching axes using test data. Absolute
correlations avoid arbitrary signs; signed correlations orient axes using training
scores. Full temporal spectra use partition-centered input data; numerical
null-space eigenvalues are excluded using the PCA rank tolerance.

Legacy fits lacking sufficient statistics support reconstruction/prediction only
at the original fitted component count. New-format fits support all smaller k.
Examining multiple k values on held-out results is exploratory analysis, not an
independent evaluation of a model selected using those results.
