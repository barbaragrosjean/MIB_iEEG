# Subject-pairing workflow

## Computation and display are independent

The four notebooks `vs1_coverage_matching`, `vs2_0_cov_models`,
`vs2_1_plssvd_eval`, and `vs3_0_compare_modality` use:

```python
PAIRING_VIEW = 'matched'  # saved demographic assignment (-1)
# PAIRING_VIEW = 'full'   # mean ± sample SD across random assignments
N_PAIRINGS = 100
PAIRING_SEED = 2026
```

A computational batch always saves **101 assignments in order**: `-1`, then
`0` through `99`. Changing `PAIRING_VIEW` does not change what is computed,
create a different result set, or refit completed assignments. It changes only
which outputs the notebook loads. Successful command receipts skip completed commands on rerun;
per-analysis rules for interrupted, incomplete outputs still apply. Increasing `N_PAIRINGS` requests more random assignments.

Full summaries exclude `-1`. Each random assignment contributes one mean for
each metric after its internal folds/repetitions are averaged. The displayed
SD is the sample SD (`ddof=1`) of those assignment means, not a pooled fold SD,
a standard error, or a confidence interval. Valid counts accompany undefined
or missing metric cells. An absent assignment or incompatible result schema
prevents loading a full summary; assignment zero is never used as a substitute.
Categorical component identities are labels, never averaged indices.

## Prepare all eligible MEG recordings once

Set the raw/preprocessing configuration in the coverage or covariance notebook.
The preparation cell only saves that configuration and prints:

```text
python -u vs0_prepare_data.py --config /path/to/config.json
```

Run this command in a terminal. Preparation is implemented in
`vs0_prepare_data.py`, not in a notebook. It reads **every eligible MEG recording**
and the retained iEEG cohort, preserving exact matched CSV recording paths.
All subject assignments reuse the same prepared signal cache; loading does not
fit models or average different input assignments into one signal dataset.
Changing an assignment or the display view does not reprocess raw signals.
Preparation and result loading require the demographic match CSV to define the
retained iEEG cohort and the matched reference.

`vs0_run_pairings.py` handles computational batches. Commands printed by the
notebooks already include it. Relevant analysis entry points also dispatch the
complete batch when invoked directly from a terminal. Only internal batch children or the explicit debugging flag
`SUBJECT_PAIRING_SINGLE_RUN=1` use a single assignment. A notebook's display
context alone never bypasses the batch. Default batch size is 100 random assignments; configured
coverage/covariance runners read `n_pairings` and `pairing_seed` from their JSON.
Restart notebook kernels after updating the code. Other direct terminal entry points accept `SUBJECT_PAIRING_COUNT` and
`SUBJECT_PAIRING_SEED` environment variables; printed notebook commands supply
the selected count and seed explicitly.

Outputs are isolated under
`out/subject_pairings/<protocol/CSV/behavior/seed hash>/iteration_<ID>/`.
The protocol namespace separates the new batches from earlier partial workflows.

## Coverage notebook policies

| Section | Computation | Display |
| --- | --- | --- |
| Preparation | All eligible MEG participants, once | Preparation command only |
| Load | Reuse the common prepared cache | Select fitted results from CONFIG and PAIRING_VIEW |
| Metadata, variance, feature counts, brain coverage | Matched assignment | Always matched, even in full view |
| Direct MEG comparisons by manipulation | -1 plus 100 random assignments | Matched or random-assignment mean ± SD |
| MEG source-count effect | Complete eligible MEG pool, once | Independent of PAIRING_VIEW |
| PCA for all datasets | -1 plus 100 random assignments | Matched or random-assignment mean ± SD |
| Stability under random reassignment | Matched reference plus 100 random assignments from the full eligible pool | Full view only |

Non-pairwise constructions use the same full eligible pool across assignments;
their assignment SD is consequently zero when their metrics do not depend on
the pairing map. Source-count repetitions remain a distinct subsampling factor.
Stability uses the same seeded random subject assignments as the other sections.
For its random-source control, native source permutations are fixed across
assignments. Between-assignment comparisons share fitted models and are not
independent observations; their pair dispersion is labeled separately.

## Covariance, PLSSVD evaluation and modality comparisons

All exported scalar metrics are available in matched or full view. Model runners
also export metrics formerly computed only inside notebooks, including fixed
full-data iEEG-PCA reference comparisons, covariance/correlation matrices, score
variances, and component matching diagnostics. Vector-valued principal angles
are expanded into separate numerical metrics before aggregation.

Covariance/PCA time courses and coverage-slot weights are aligned to the saved
matched fit using fitted temporal scores only. Separate PCA aligns modalities
independently; coupled models retain a common permutation and sign. Full means
and SD are calculated after alignment. Sign/permutation matching does not resolve
rotations within nearly degenerate subspaces. Native rank-based scalar metrics and
spectra retain their rank interpretation. Coverage-slot spatial summaries do
not imply that the same native MEG source was sampled in every assignment.
Matrix displays show separate mean and SD panels; all numeric table metrics
remain available in each `ResultView` even when only selected components are
plotted (`DISPLAY_COMPONENTS`).

The modality-comparison stability command publishes an additional
`random_assignment_summary` after all assignments complete. Inner trial fits
are aligned and averaged within each subject assignment before across-assignment
weight/time-course stability and participant-balanced ROI summaries. The matched
fit is the alignment reference and is excluded from these random summaries.
ROI cross-correlation SD describes channel-pair heterogeneity, not assignment SD.

## Independent trial and temporal-permutation axes

`N_PAIRINGS` controls subject assignments. `N_FIT_ITERATIONS` in PLSSVD controls
repeated trial-fold evaluations **within one subject assignment**. `N_SPLITS`
controls its trial folds. `TEMPORAL_PERMUTATION_TYPE = 'time_point'` controls
surrogate time samples and does not shuffle subjects. The notebook defaults to
two inner trial iterations because the existing component-selection procedure
requires at least two; increase this independently when required by the analysis.

The baseline and each temporal-null run use the same outer pairing IDs and seed.
For each assignment, the existing component-selection rule is applied separately.
Full view reports the mean/SD of resulting numerical diagnostics and per-rank
retention frequencies. A mean selected dimension is a descriptive statistic,
not a new integer selection rule. Full-data PCA comparisons, temporal-null
comparisons, spectral diagnostics and selection tables all use the same view.

The independent trial cache includes the full eligible MEG pool. Its default
location is `out/trial_cache_full_meg/<behavior-content-hash>/`. Run the PLSSVD
preparation/fitting command before trial-based modality comparisons. Restricted
legacy caches cannot be used as a complete random-assignment pool.

## Behavioral eligibility

The default source is `MEG/Final_xlsx`, overridable with `MEG_BEHAVIOR_DIR`.
Strict and standard OOXML workbooks are read without modifying them. Project_PD
workbooks, macOS sidecars and Excel lock files are ignored. `bis`/`bis2` labels
map to the corresponding participant number; duplicate files for the same
participant/block are rejected rather than pooled. Alternate CSV directories
are supported explicitly, with no automatic fallback to older CSV files.

Accuracy pools old/new trials over available Blocks 3–6. Old stimuli require
response 1; new stimuli require response 2. Missing/invalid/zero responses count
as incorrect. Other stimuli and export footers are ignored. Accuracy below 0.50,
missing behavioral files, or no scorable old/new trials exclude a participant.
Exactly 0.50 is retained. Missing-file exclusions produce warnings.

This rule is shared by demographic matching, preparation, random assignment,
and trial-cache loading. An older saved match containing an excluded MEG
participant loses that pair and reports it; rerun demographic matching to retain
the full iEEG cohort with eligible replacements. Random analyses use the same
retained iEEG cohort until the match is regenerated. Eligibility audits are saved
with prepared data and demographic-matching outputs. Behavioral content hashes
and the eligibility rule are included in cache/result provenance.

## Standalone metadata and decoding helpers

Active scripts and notebook code no longer import the external legacy `src`
package or add the `LB` tree to their module search path. `GetInfo`, `FindRegion`,
`REGION`, `PROJECT_PATH`, the decoding CSV cleaner, and post-projection data
augmentation are defined in `vs0_load_and_config.py`.

The default `PROJECT_PATH` retains the legacy relative project location, resolved
against this module's directory rather than the terminal working directory.
Override it with `data.project_path` in a JSON configuration, `--project-path`
where supported, or the `IEEG_PROJECT_PATH` environment variable. Coordinate CSVs
remain at `<ieeg_dir>/<subject>_coords.csv`; electrode labels are read from
`<PROJECT_PATH>/misc/electrodes/<subject>_electrodes.csv`. Preparation fingerprints
both inputs and the shared helper code, so metadata edits select a fresh cache.
Coordinate corrections and the original first-contact region mapping are
preserved. When deploying, synchronize the shared module and its updated callers
together before rerunning the preparation command.
