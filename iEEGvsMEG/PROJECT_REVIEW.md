# iEEG–MEG latent-space comparison: methods and implementation review

Reviewed against the current source and notebook cells on 23 September 2026. This is a methods and implementation audit, not a rerun or confirmation of the saved results. Raw data, trial caches and batch result directories are absent from this checkout. The coverage-sampling and within-modality model-comparison extensions described below have subsequently been implemented and checked with synthetic data; project recordings have not been rerun.

## Scientific aim and unit of comparison

The project asks which task-related temporal and spatial structures are reproducible within iEEG and MEG, and which structures correspond across modalities after accounting for sampling coverage and the objective of dimensionality reduction. Latent spaces provide a way to compare structure despite different measurement processes; they do not remove modality-specific sensitivity, source leakage, reference effects or anatomical sampling bias.

The current analyses align **condition × time averages**, not simultaneously recorded or individually paired trials. Participants are pooled within each modality. Random one-to-one participant pairing is a sampling construction, not a biological pairing. Held-out evaluation currently concerns new trials from the selected participants at the same time points and locations.

Distinguish three objects throughout the text: projection **weights** define the latent axes; **scores** are projected time courses; **forward patterns** describe how observed features covary with the scores. Spatial weights and forward patterns are not interchangeable, especially for PLS. The current forward-pattern estimator is the multivariate regression `A = X_centered.T @ T_centered @ pinv(T_centered.T @ T_centered)`.

## 1. Coverage matching and group composition

**Question.** How do anatomical coverage, participant averaging and feature concatenation alter MEG PCA structure and its correspondence with iEEG PCA?

**Method used.** Independently fit centered PCA to pooled iEEG and to five MEG compositions, using the same condition/time rows. PCA is computed through the observation Gram matrix, avoiding a large feature covariance matrix.

| Composition | Construction | Main comparison it supports |
|---|---|---|
| `full_average` | Average participants at corresponding source indices | Whole-source group-average reference |
| `full_concatenated` | Concatenate participant source features | Retain participant-specific source features |
| `coverage_average` | Match every pooled electrode to each participant's nearest source, then average participants | Coverage restriction with group averaging |
| `paired_coverage` | Assign each iEEG participant a distinct sampled MEG participant and retain nearest sources for their electrodes | Coverage and participant composition resembling pooled iEEG |
| `random_control` | Use the same pairing and feature counts, but random source locations; preserve duplicate-source multiplicities | Sensitivity to anatomical source selection |

Nearest-source mapping uses Euclidean coordinate distance. MEG sources are z-scored across condition/time before aggregation; iEEG has a fixed ×1000 multiplier. Notebook defaults average conditions before PCA; `stack` preserves them as separate observation rows.

**Outputs.** Coverage plots, feature/variance summaries, PCA spectra, component maps and time courses; full component-by-component **Spearman** correlation matrices for time courses and anatomically mapped weights; and three temporal metrics at each prespecified common dimension k:

1. **Retained iEEG variance captured:** `||Q_MEG.T @ T_iEEG,k||² / ||T_iEEG,k||²`, where Q is an orthonormal basis for the first k MEG score columns. This is the fraction of variance in the retained iEEG scores captured by the MEG temporal subspace. It is not the cumulative PCA variance explained in all original channels.
2. **Temporal subspace overlap:** `||Q_MEG.T @ Q_iEEG||² / k`, the mean squared cosine of principal angles.
3. **Matched component correlation:** mean absolute **Pearson** correlation under optimal one-to-one Hungarian matching of the first k components.

The first two are invariant to a change of basis within the retained subspace; the third handles sign/order ambiguity but not arbitrary rotations. Cumulative PCA variance explained is an additional, distinct output.

**Implemented extension.** `coverage_sampling.py` and the final section of `coverage_matching.ipynb` repeat coordinated participant/source draws and cross MEG participant-pool counts with common feature budgets. The complete iEEG reference stays fixed. Participant pools and feature-budget subsets are nested within repetition; matched/random controls share electrode slots and preserve source duplication. Full-source feature budgets uniformly sample native features. Pair-based setups keep one MEG participant per iEEG participant before feature sampling, so a larger pool is not a larger averaged group. Actual contributor counts are exported. Unsupported k values are explicitly marked `skipped_rank`.

Outputs include all three temporal metrics, cumulative PCA variance fractions, component pairs, exact selections (compact ranges for complete full-source blocks), anatomical mapping distances, unique-source/duplicate summaries and within-draw paired-minus-random differences. Plots show median/range against participant pool and feature count; these ranges are not confidence intervals. Saved complete runs can be loaded without recordings using `load_coverage_sampling`.

**Remaining scope.** Participant-pool counts must be at least the number of iEEG participants to support one-to-one pairing. No participant-balanced feature weighting or distance-rejection threshold is imposed. The five compositions still differ in aggregation and coverage; common feature counts control size, not all confounds. This is descriptive resampling of condition averages, not trial-held-out validation.

## 2. Alternative objectives and PLSSVD validation

**Question.** Does maximizing cross-modal covariance produce reproducible shared patterns, and how much modality-specific variance do those patterns retain?

**Method used.** `cov_models_utils.py` fits separate PCA, joint PCA on feature-concatenated modalities, and exact PLSSVD on `X.T @ Y/(n−1)`. Separate PCA maximizes each modality's variance; joint PCA maximizes variance of the concatenated data; PLSSVD selects paired unit-norm weight vectors that maximize cross-covariance. The implementation uses full nonzero sample-space factors, not a preliminary truncation to k PCs.

PLSSVD itself is a covariance decomposition. The validation adds ridge regression from one modality's latent scores to the other modality's original features. Thus model fitting, cross-modal association and prediction are related but distinct evaluations. PCA also needs held-out evaluation for a fair comparison.

**Validation method.** `plssvd_eval` partitions trials separately within modality, participant and condition into approximately 50% training, 20% tuning and two 15% test halves, with minimum-count adjustments. Group splitting is available when run/stimulus metadata is supplied. Condition averages are calculated separately within each partition and stacked by condition/time. MEG normalization, PLS weights and ridge prediction maps are learned using training data only. Anatomical matching stays fixed across partitions.

Select k by mean bidirectional tuning Q² in the original target feature spaces, with a fixed relative ridge penalty and smaller k winning ties. Freeze the model before evaluating test trials. The primary repetition uses all participants; subsequent repetitions change participant subsets, trial splits and pairing. These repetitions measure sensitivity, not population confidence intervals.

**Outputs.** Selected k and tuning curves; signed held-out paired score correlations; bidirectional prediction Q² against the training-mean baseline; condition-difference correlations and amplitudes; test-half temporal, contrast and forward-pattern reliability; saved scores, weights, preprocessing and split/matching audits. Primary-fit null outputs include conditional condition-label permutations, temporal circular-shift diagnostics and spatial-correspondence diagnostics.

**Missing.** Predeclare the primary endpoint and what constitutes a useful/reliable pattern; selecting k alone is not component-level validation. No component acceptance rule, independent-refit component/subspace stability analysis or multiplicity procedure is implemented. Group splits and exchangeability blocks need actual task metadata. Verify upstream preprocessing/source filters for dependence on held-out trials. Temporal shifts and spatial row shuffles are not automatically valid inferential p-values. A stable shared evoked response does not establish condition-specific or memory-related information.

The current validation selects k for PLSSVD only. A fair predictive model comparison needs identical partitions, targets, preprocessing and tuning opportunities for PCA and joint PCA as well. `compare_subspace.py` refits all models on common partitions but uses a prespecified dimension grid and a different prediction endpoint (latent representations).

## 3. Within-modality comparison across models

**Question.** Within iEEG, and separately within MEG, are the dominant PCA patterns the same as the patterns selected by PLSSVD or joint PCA?

**Implemented.** `compare_models.py` supplies descriptive comparisons in `cov_models.ipynb` and held-out comparisons within the existing `compare_subspace.py` batch pipeline. All model pairs are compared separately within iEEG and MEG at common dimensions. Temporal scores, two-condition contrasts (when stacked), and forward patterns on **all native features** are compared. Pattern regression is recomputed for each retained k.

A one-to-one Hungarian assignment and sign orientation are learned from **training temporal scores**, then reused for every representation and partition. Thus spatial and temporal metrics concern the same component pairs. There is no test rematching or test-based sign orientation. Signed and absolute matched correlations, valid-pair counts, full signed Pearson matrices, principal angles, ranks and rank-aware subspace overlap are exported. Undefined training pairs are excluded from matched summaries; valid-pair counts expose this limitation.

**Outputs.** `within_model_metrics.csv`, `within_model_pairs.csv`, `within_model_correlations.csv`; summary plots and primary-repetition correlation heatmaps. Batch results identify modality, model pair, repetition, k, representation and partition. The descriptive notebook labels its rows `in_sample`; batch comparisons use the existing independent trial partitions. Old result folders remain readable but require a new run to acquire these tables.

**Remaining scope.** Independently refitted component stability, participant-level inference, anatomical pattern displays and explicit component-acceptance criteria remain separate work. Test-half correspondence is conditional on the fitted axes. Near-degenerate components can rotate, making subspace metrics more stable than componentwise matching.

**Already available context.** `evaluate_cov_models()` reports reconstruction variance explained and retained cross-covariance energy. These are descriptive in-sample quantities with different objectives. Joint PCA reconstructs with the joint score, which uses both modalities; this is not an own-modality-only prediction benchmark. Cross-covariance energy is not a fraction of uniquely identified shared biological variance.

## 4. Cross-modality geometry, transformations and clustering

**Question.** For a fixed dimensionality-reduction method, how similar are iEEG and MEG latent structures, and what transformation is required to relate them?

**Method used.** `compare_subspace.py` fits separate PCA, PLSSVD and joint PCA using shared train/tune/test partitions. It compares condition/time score matrices and forward-pattern matrices sampled at the electrode correspondences. It computes principal angles and rank-aware overlap. Each representation is centered and divided by one scalar RMS estimated on training data, preserving relative axis geometry.

Fit identity, orthogonal rotation/reflection, regularized affine and regularized quadratic mappings in both directions. Fit coefficients on training representations and select ridge penalties on tuning error. Report held-out normalized RMSE and Q². Orthogonal improvement can reflect sign, permutation or rotation ambiguity; it is not by itself a physiological transformation. More flexible mappings can capture additional differences, but the code reports all complexities rather than selecting a single winning complexity.

**Clustering used.** Independently fit K-means to each modality's spatial forward patterns using all retained dimensions. Select a common cluster count by mean tuning silhouette. Compare test assignments to frozen training centroids using cross-modal adjusted Rand index (ARI). Report within-modality test-half assignment stability, independently refitted test-half clustering stability and test silhouette. This is spatial clustering; temporal-state clustering is not implemented.

**Outputs.** `overlap.csv`, `alignment.csv`, `reliability.csv`, `clusters.csv`, `cluster_selection.csv`, `splits.csv`; models/scores, spatial patterns, transformation coefficients, cluster assignments/centroids, anatomical mappings, preprocessing and diagnostic figures.

**Missing.** No inferential test for improvements across models/transformations or for cluster agreement is provided. There is no held-out participant/location/time generalization. Spatial mappings are learned and tested at recurring electrode slots using independent trial estimates. Random-control slots are non-anatomical and must not be described as a common anatomical grid. Clustering is exploratory: stable partitions do not establish discrete biological networks. Add anatomical cluster displays and a defensible spatially structured null only if discrete spatial organization is a primary scientific claim. Reliability supplies context, not a calibrated noise ceiling.

## Implementation and reporting gaps to resolve

| Priority | Location | Finding and required action |
|---|---|---|
| High | Coverage/covariance notebook coordinate cells; `plssvd_eval.py: main` | Magnitude-based repeated coordinate division can alter individual axes before the explicit unit checker sees them. Establish the native `GetInfo` unit and apply one documented conversion; verify against known anatomical locations. Do not infer correctness from plausible ranges alone. |
| Resolved | `construct_five_datasets` | Full-source averaging now requires equal array shapes and matching registered source coordinates/order. Nonmatching grids must be aligned upstream. |
| High | Coverage/covariance loader calls | MEG epoch origin is set from the iEEG epoch origin. Supply independently verified MEG time metadata; matching constructed vectors alone does not verify acquisition alignment. |
| High | Across stages | Coverage/covariance default to condition averaging; validation/comparison stack conditions. Choose a primary representation and rerun comparable stages consistently. Condition averaging cannot address condition differences. |
| High | Across entry points | PLSSVD batch excludes `SUBJ_0038`; coverage/covariance discovery does not. Record one participant manifest and exclusion rationale. Comparison uses the cache cohort. |
| Implemented | Model comparison | Descriptive and trial-held-out within-modality comparisons now exist; run the updated pipeline before interpreting project results. |
| Medium | Scaling defaults | Covariance/validation use `none`; subspace comparison defaults to `equal_variance`. Record and harmonize this choice, especially for joint PCA. Scalar modality scaling alone does not change ideal PLSSVD directions, but changes joint PCA's balance. MEG channel z-scoring versus unstandardized iEEG also changes what “dominant variance” means. |
| Medium | Correlation reporting | Coverage matrices are Spearman; matching summaries and validation use Pearson. Label both explicitly. A notebook TODO requests Spearman native matching, but it is not implemented; do not describe it as completed. |
| Implemented | Coverage design | Repeated coordinated draws and crossed participant-pool/feature-budget sweeps are available; run them on the project recordings. |
| Medium | Validation/results | Test-half reliability fixes trained axes; independently refit models and compare subspaces if the claim is stability of learned patterns. Do not average PC1 across repetitions as if component identity were guaranteed. |
| Medium | Result lifecycle | PLSSVD output directories can be reused without a run-completion guard; comparison writes a completion marker but rejects any existing configuration, including interrupted runs. Add run identity, completion checks and a deliberate resume policy. |
| Medium | Reproducibility | A synthetic unittest suite now covers the extensions; a dependency manifest is still absent. `src.setting/GetInfo`, data and cluster paths are external. Add environment versions, input provenance and focused numerical/leakage tests before final analysis. |
| Text | Methods/results | Supply task/condition meanings, participant/trial/electrode counts, exclusions, filtering, baseline/reference, MEG source reconstruction, coordinate frame, epoch/window choice, source polarity handling, acquisition differences and trial dependence. These cannot be inferred reliably from numeric condition codes. |

Small maintenance fixes made during this review: replaced the obsolete `utils_updated` notebook import/path checks with `coverage_matching_utils`; corrected the PLSSVD notebook's stale claim that a `RUN_ANALYSIS` switch exists; annotated the missing within-modality analysis and the covariance interpretation placeholder. The subsequent implementation adds the two requested analyses and a common-grid guard; existing scientific preprocessing defaults remain unchanged.

## Project map and recommended sequence

| Files | Role |
|---|---|
| `coverage_matching.ipynb`, `coverage_matching_utils.py` | Data loading, anatomical sampling, five MEG compositions and descriptive PCA comparisons |
| `coverage_sampling.py` | Repeated MEG participant-pool/feature-budget sensitivity, audits, reload and plots |
| `compare_models.py` | Within-modality model-pair comparisons with training-frozen component matching |
| `tests/test_comparison_extensions.py` | Synthetic numerical, sampling and batch integration checks |
| `cov_models.ipynb`, `cov_models_utils.py` | Descriptive separate/joint PCA and PLSSVD objectives, reconstruction and cross-covariance metrics |
| `plssvd_eval.py`, `plssvd_eval_utils.py`, `plssvd_eval.ipynb` | Trial export/cache, PLSSVD selection/validation, persisted outputs and result reader |
| `compare_subspace.py`, `compare_subspace.ipynb` | Held-out cross-modality geometry, alignment and spatial clustering for all three models |
| `OLD/LB10*` | Earlier extraction, concatenation and randomized matching work |
| `OLD/LB11*`, `OLD/LB12*`, `OLD/LB13*`, `OLD/LB14*`, `OLD/LB_Summary.ipynb`, `OLD/utils.py` | Earlier embedding, decomposition, interpretation, manifold and frequency explorations; historical context, not the current validation pipeline |

The active modules and all current notebook source cells were inspected. Legacy files were inventoried and entry points inspected; their analyses were not revalidated.

Recommended order: establish task/data provenance and alignment → assess coverage/composition sensitivity → fit competing objectives → validate on independent trials → compare models within modality → compare modalities within model → optionally interpret stable spatial clusters. Freeze scientific choices before inspecting final test outcomes. If exploratory coverage/model results already used the eventual test trials, a later split does not retrospectively make the entire study confirmatory; describe it as conditional validation or obtain independent confirmation.


## Running the extensions

Run the added final section of `coverage_matching.ipynb` after loading `ieeg`. It defaults to 20 sampling repetitions and three participant-pool sizes with two common feature budgets plus native counts. Review these settings for runtime and the intended study design. A completed output directory is loaded with its saved configuration; choose a new directory to change settings. Interrupted runs are not resumed automatically.

Run the final section of `cov_models.ipynb` for descriptive within-modality comparisons. For held-out results, use the existing batch command with at least two models and a new output directory:

```bash
python -u compare_subspace.py --cache-dir /path/to/trial_cache \
  --output-dir /path/to/new_comparison \
  --models separate_pca plssvd joint_pca --meg-kind paired_coverage \
  --dimensions 1 2 3 5 10 --repeats 5 --block-scaling equal_variance
```

Inspect the new within-modality section of `compare_subspace.ipynb`. The batch plotter also exports within-modality summaries and heatmaps. The synthetic suite runs with `MPLBACKEND=Agg python -m unittest discover -s tests -v` in an environment containing the project's scientific Python dependencies.
