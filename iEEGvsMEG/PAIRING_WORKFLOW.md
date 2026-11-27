# Analysis workflow with explicit participant pairings

Executable Python scripts contain `_run_` after their `vs` stage code; helper modules contain `_utils_`. Notebook filenames and analysis output-folder names are unchanged.

Run the commands below from the repository root (the directory containing `iEEGvsMEG`). The recording and demographic inputs require the cluster. Edit the existing JSON files before running; notebooks read them without overwriting them.

## 1. Select participants and create the 22 pairings

Edit `out/vs0_pairs/config.json`, including the cluster input paths, then run:

```bash
python -u iEEGvsMEG/vs0_run_create_pairs.py --config iEEGvsMEG/out/vs0_pairs/config.json
```

This creates 44 CSV files: demographic and electrode/source tables for `match`, `order`, and `random_001` through `random_020`. iEEG participants remain fixed. Each pairing samples distinct MEG participants; participants may recur across different pairings. Demographic matching follows the notebook's greedy standardized-age/sex cost, with no instrument term. Equal-distance source ties are seeded and saved explicitly. Each electrode CSV also stores `coverage_source_indices` (a JSON dictionary of source indices for every MEG participant selected in that pairing) and `random_source_index` for the control. These are resolved only by pair creation; analyses never regenerate participant associations or nearest-source matches.

MEG eligibility requires accuracy >=0.5 separately in old and new conditions, using actual condition-specific trial denominators, and >=24 trials across those conditions. Missing/invalid responses count as incorrect. Available behavioral Blocks 3–6 are combined within each condition. Missing age, sex, behavior, or required recording/metadata files excludes the participant. The iEEG demographic roster is already behaviorally screened. The selection audit records exclusions.

Electrode CSVs generated before `coverage_source_indices` and `random_source_index` were added must be regenerated once with `--overwrite`; incomplete correspondence now raises an error instead of triggering rematching. Existing pairing files are protected; deliberate regeneration requires `--overwrite`. Regenerating pairings invalidates analyses referencing changed CSVs.

## 2. Review an analysis configuration and its run plan

Every analysis has an editable `out/<analysis>/config.json`. Its first field is the execution command. The default pairing selection is `all` (22 pairings).

```bash
python iEEGvsMEG/vs0_run_analysis.py --config iEEGvsMEG/out/vs1_pca/config.json --dry-run
```

This lists the selected analysis and its pairing-specific output paths. Dependencies are reused when their manifests match. Each analysis is launched explicitly using its own configuration.

## 3. Execute

One analysis across all pairings:

```bash
python -u iEEGvsMEG/vs0_run_analysis.py --config iEEGvsMEG/out/vs1_pca/config.json
```

One selected pairing:

```bash
python -u iEEGvsMEG/vs0_run_analysis.py --config iEEGvsMEG/out/vs1_pca/config.json --pairing random_001
```

Existing entry-point scripts also accept these managed configs. For example, `vs1_run_coverage.py --config .../vs1_pca/config.json` dispatches to the same runner. Legacy command-line interfaces remain available for existing results; use managed configs for this organization.

The runner checks dependencies, pairing identities, input provenance, and result completion. Changed settings raise an explicit stale-results error. Use `--recompute` for the affected analysis; prior completed outputs move to its `archive/` directory. Dependencies must be recomputed separately if they are stale. Interrupted PLSSVD fit jobs resume their saved iterations. Shared caches with changed source data require explicit invalidation or a separate output root; they are not silently overwritten.

## vs4 decoding

Edit `out/vs4_decoding/config.json`. Its top-level `pairings` field accepts `"match"`, `"order"`, `"random_001"` through `"random_020"`, or `"all"` (all 22). A list of selected pairing IDs is also accepted. The initial selection is `"match"`.

```bash
python -u iEEGvsMEG/vs4_1_run_decoding.py --config iEEGvsMEG/out/vs4_decoding/config.json --dry-run
python -u iEEGvsMEG/vs4_1_run_decoding.py --config iEEGvsMEG/out/vs4_decoding/config.json
```

The equivalent shared entry point is `vs0_run_analysis.py --config iEEGvsMEG/out/vs4_decoding/config.json`. There is no all-analysis pipeline launcher.

Configure the projection (`method`: PCA or PLSSVD), zero-based component index (`pc_use`), classifier (`model`), condition representation (`method_pls`), augmentation (`data_aug_method`), decoding iterations, classifier-label permutations, pseudo-trial count, block scaling, seed, and optional classifier parameters in `parameters`. These are passed to the existing `mainTS`/`decodingTS` functions without changing their computations. `bs_decoding` retains the existing pre-stimulus cropping option. `band` labels the output; it does not filter data.

A prepared **individual-trial** cache is required at `cache_dir`, initially `{OUT}/vs1_construct/shared/trials`. The vs4 runner does not rerun condition-average construction or export/map the full union cache. Only participants in the selected pairing are loaded, and the saved source/control/coverage indices are used. Missing cache files or pairing correspondence raise explicit errors. The decoder's preprocessing and classifier controls are unchanged; label shuffling remains a conditional classifier control rather than a refitted full-analysis null.

Outputs retain the decoder's existing filenames inside:

```text
out/vs4_decoding/
├── config.json
├── match/
│   ├── manifest.json
│   ├── ieeg/<band>/fullmodel/*.csv
│   ├── ieeg/<band>/fullmodel/*_iteration0_time_courses.npz
│   ├── meg/<band>/fullmodel/*.csv
│   └── meg/<band>/fullmodel/*_iteration0_time_courses.npz
├── order/
├── random_001/ ... random_020/
└── summary/manifest.json
```

Only selected pairing folders are created. A changed numerical configuration requires `--recompute`; previous outputs are archived. Changing only the selected pairing list or command reuses compatible completed pairing outputs. The manifest also tracks the selected trial-cache files. The vs4 notebook and the currently empty behavior helper are not modified by this runner adaptation.

## Storage

Each result lives in `out/<analysis>/<pairing>/`, with a manifest recording its actual configuration, pairing files, source signatures, code hashes, dependencies, and output inventory. `summary/manifest.json` indexes completed pairings without duplicating their results. `match` and `order` remain distinct from the 20 random pairings; random pairings are not independent cohort samples.

`vs1_construct/shared/` holds common iEEG condition means, per-MEG condition averages, and a union trial cache. Trial-cache reads select the exact participants in the requested pairing without rewriting or truncating that cache. Source correspondence is read from the electrode CSV and checked against recording coordinates.

PLSSVD fits, evaluation, PCA comparison, permutation comparison, and selection outputs reside in their respective analysis folders. Fit variants (baseline and nulls), trial iterations, and folds remain nested below their outer participant pairing.

Within each outer pairing, participant correspondence stays fixed through trial folds. `vs3_weight_stability` defaults to trial-only repetitions; participant-association variation is represented by the 20 outer random pairings. `vs3_pairing_weights` uses one fixed-half-trial fit per outer pairing, avoiding repeated identical association fits. SD across associations is undefined within a single pairing; cross-pair summaries should use the outer random results. The dedicated `vs1_stability` analysis compares each explicit baseline against the random pairing CSVs listed in its `comparison_pairings` configuration. It excludes the baseline itself when it is in that list. It does not generate further permutations. Spatial comparisons with a baseline full-concatenated PCA are omitted when a comparison pairing contains MEG participants absent from that baseline; time-course comparisons remain available.

Canonical modality-specificity results use CSV tables, compressed numerical arrays, and `result_index.json`, without a duplicate pickle copy. Existing legacy completion/configuration records may remain inside result subdirectories for compatibility; only the analysis-root `config.json` is editable.

## Notebooks

For vs1, use `vs1_coverage_visualization.ipynb` and select `MODE` (`match`, `order`, or `random`); the earlier notebook is preserved at `OLD/vs1_coverage_matching.ipynb`. In the other notebooks, select `PAIRING_ID` in the header. Header/loading cells display the saved JSON parameters, validate results, and load the selected pairing. Figure cells, stored figure outputs, and plotting definitions are preserved. Computations are run by the printed commands, not by notebook loading cells.

## Validation

`tests/test_pairing_workflow.py` uses synthetic recordings and local atlas-label stubs. It exercises all analysis types, nearest-source ties, unequal behavioral denominators, explicit participant/source correspondence, analysis execution with nearest-source matching disabled, saved-association stability comparisons, PLSSVD output routing, and actual PCA runs for all 22 pairings. Full cluster validation still requires the real inputs. Packages include NumPy, pandas, SciPy, scikit-learn, Matplotlib, nilearn, nibabel, openpyxl, and mat73 for MATLAB v7.3 files.
