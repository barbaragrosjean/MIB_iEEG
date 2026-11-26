# Analysis workflow with explicit participant pairings

Executable Python scripts contain `_run_` after their `vs` stage code; helper modules contain `_utils_`. Notebook filenames and analysis output-folder names are unchanged.

Run the commands below from the repository root (the directory containing `iEEGvsMEG`). The recording and demographic inputs require the cluster. Edit the existing JSON files before running; notebooks read them without overwriting them.

## 1. Select participants and create the 22 pairings

Edit `out/vs0_pairs/config.json`, including the cluster input paths, then run:

```bash
python -u iEEGvsMEG/vs0_run_create_pairs.py --config iEEGvsMEG/out/vs0_pairs/config.json
```

This creates 44 CSV files: demographic and electrode/source tables for `match`, `order`, and `random_001` through `random_020`. iEEG participants remain fixed. Each pairing samples distinct MEG participants; participants may recur across different pairings. Demographic matching follows the notebook's greedy standardized-age/sex cost, with no instrument term. Equal-distance source ties are seeded and saved explicitly.

MEG eligibility requires accuracy >=0.5 separately in old and new conditions, using actual condition-specific trial denominators, and >=24 trials across those conditions. Missing/invalid responses count as incorrect. Available behavioral Blocks 3–6 are combined within each condition. Missing age, sex, behavior, or required recording/metadata files excludes the participant. The iEEG demographic roster is already behaviorally screened. The selection audit records exclusions.

Existing pairing files are protected; deliberate regeneration requires `--overwrite`. Regenerating pairings invalidates analyses referencing changed CSVs.

## 2. Review configurations and the run plan

Every analysis has an editable `out/<analysis>/config.json`. Its first field is the execution command. The default pairing selection is `all` (22 pairings).

```bash
python iEEGvsMEG/vs0_run_pipeline.py --dry-run
```

This lists 25 analysis types × 22 pairings = 550 jobs. Dependencies are reused rather than rerun when their manifests match. This is a plan, not evidence of completed computations.

## 3. Execute

One analysis across all pairings:

```bash
python -u iEEGvsMEG/vs0_run_analysis.py --config iEEGvsMEG/out/vs1_pca/config.json
```

One selected pairing:

```bash
python -u iEEGvsMEG/vs0_run_analysis.py --config iEEGvsMEG/out/vs1_pca/config.json --pairing random_001
```

All analyses across all pairings, sequentially:

```bash
python -u iEEGvsMEG/vs0_run_pipeline.py
```

Existing entry-point scripts also accept these managed configs. For example, `vs1_run_coverage.py --config .../vs1_pca/config.json` dispatches to the same runner. Legacy command-line interfaces remain available for existing results; use managed configs for this organization.

The runner checks dependencies, pairing identities, input provenance, and result completion. Changed settings raise an explicit stale-results error. Use `--recompute` for the affected analysis; prior completed outputs move to its `archive/` directory. Dependencies must be recomputed separately if they are stale. Interrupted PLSSVD fit jobs resume their saved iterations. Shared caches with changed source data require explicit invalidation or a separate output root; they are not silently overwritten.

## Storage

Each result lives in `out/<analysis>/<pairing>/`, with a manifest recording its actual configuration, pairing files, source signatures, code hashes, dependencies, and output inventory. `summary/manifest.json` indexes completed pairings without duplicating their results. `match` and `order` remain distinct from the 20 random pairings; random pairings are not independent cohort samples.

`vs1_construct/shared/` holds common iEEG condition means, per-MEG condition averages, and a union trial cache. Trial-cache reads select the exact participants in the requested pairing without rewriting or truncating that cache. Source correspondence is read from the electrode CSV and checked against recording coordinates.

PLSSVD fits, evaluation, PCA comparison, permutation comparison, and selection outputs reside in their respective analysis folders. Fit variants (baseline and nulls), trial iterations, and folds remain nested below their outer participant pairing.

Within each outer pairing, participant correspondence stays fixed through trial folds. `vs3_weight_stability` defaults to trial-only repetitions; participant-association variation is represented by the 20 outer random pairings. `vs3_pairing_weights` uses one fixed-half-trial fit per outer pairing, avoiding repeated identical association fits. SD across associations is undefined within a single pairing; cross-pair summaries should use the outer random results. The dedicated `vs1_stability` analysis retains its original within-selected-cohort assignment perturbations around each explicit baseline pairing.

Canonical modality-specificity results use CSV tables, compressed numerical arrays, and `result_index.json`, without a duplicate pickle copy. Existing legacy completion/configuration records may remain inside result subdirectories for compatibility; only the analysis-root `config.json` is editable.

## Notebooks

Select `PAIRING_ID` in each notebook header. Header/loading cells display the saved JSON parameters, validate results, and load the selected pairing. Figure cells, stored figure outputs, and plotting definitions are preserved. Computations are run by the printed commands, not by notebook loading cells.

## Validation

`tests/test_pairing_workflow.py` uses synthetic recordings and local atlas-label stubs. It exercises all analysis types, nearest-source ties, unequal behavioral denominators, explicit participant/source correspondence, PLSSVD output routing, and actual PCA runs for all 22 pairings. Full cluster validation still requires the real inputs. Packages include NumPy, pandas, SciPy, scikit-learn, Matplotlib, nilearn, nibabel, openpyxl, and mat73 for MATLAB v7.3 files.
