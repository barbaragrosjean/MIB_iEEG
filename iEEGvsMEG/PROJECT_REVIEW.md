# iEEG–MEG latent-space comparison: methods and implementation review

## Current status

| Analysis | Implementation | Evaluation scope | Output |  Results | 
|---|---|---|---|---| 
| Coverage matching and composition sensitivity | Implemented in `coverage_stability.py` and `coverage_matching.ipynb` | Compute different MEG dataset organisation and test how much the information is shared with iEEG using PCA on both dataset individually |  
| PLSSVD held-out evaluation | Implemented in `plssvd_eval` modules | Fixed **5 components by default**, repeated random train/test splits, all participants and matching fixed; no tuning |
| PCA versus PLSSVD/joint PCA within each modality | Implemented in `compare_models.py`, `cov_models.ipynb` and the `compare_subspace` batch pipeline | Descriptive notebook comparisons and held-out comparisons with training-frozen component assignments/signs |
| Cross-modality geometry and spatial clustering | Implemented in `compare_subspace` | Held-out trial representations at recurring times and locations; exploratory clustering |

## Scientific aim and unit of comparison

The project asks which task-related temporal and spatial structures are reproducible within iEEG and MEG, and which structures correspond across modalities. We explore the different ways of arenging MEG datatset before applying dimension reduction and different model with 2 different opbjectives, the fisr PCA only focus on covariance within each modality and the second PLSSVD focus on covariance tha tis shared bevtween modality. Componants are then use to evaluate how much we can get from both modality. We first compare the models within the same modality (Q: Does the componant that are mainly share are comparable with the one that represent most of the variance ?) Then we compare within the same model how well each modeality can explain the other, what si it shared. We decidede to work with dimensionality components as Latent spaces provide a way to compare structure despite different measurement processes. Still the information on which the dimensionnality reduction operate are different so in a secodn part of the project we will investigate that despite finding the comparable latent structures the two recordings are modality specific.

The current analyses align **condition × time averages**, not simultaneously recorded or individually paired trials. 

**Method and language:** \
Distinguish three objects throughout the text: projection **weights** define the latent axes; **scores** are projected time courses; **forward patterns** describe how observed features covary with the scores. Spatial weights and forward patterns are not interchangeable, especially for PLS. The current forward-pattern estimator is the multivariate regression `A = X_centered.T @ T_centered @ pinv(T_centered.T @ T_centered)`.

## 1. Coverage matching and group composition

**Question.** How do anatomical coverage, participant averaging and feature concatenation alter MEG PCA structure and its correspondence with iEEG PCA?

**Method used.** Independently fit centered PCA to pooled iEEG and to five MEG compositions, using the same condition/time rows. PCA is computed through the observation Gram matrix (XX.T inseqd of X.TX), avoiding a large feature covariance matrix.

| Composition | Construction | Main comparison it supports |
|---|---|---|
| `full_average` | Average participants at corresponding source indices | Whole-source group-average reference |
| `full_concatenated` | Concatenate participant source features | Retain participant-specific source features |
| `coverage_average` | Match every pooled electrode to each participant's nearest source, then average participants | Coverage restriction with group averaging |
| `paired_coverage` | Assign each iEEG participant a distinct sampled MEG participant and retain nearest sources for their electrodes | Coverage and participant composition resembling pooled iEEG |
| `random_control` | Use the same pairing and feature counts, but random source locations; preserve duplicate-source multiplicities | Sensitivity to anatomical source selection |

For pair association: Nearest-source mapping uses Euclidean coordinate distance. 

**Outputs.** Coverage plots, feature/variance summaries, PCA spectra, component maps and time courses; full component-by-component **Spearman** correlation matrices for time courses and anatomically mapped weights; and three temporal metrics at each prespecified common dimension k:

1. **Retained iEEG variance captured:** `||Q_MEG.T @ T_iEEG,k||² / ||T_iEEG,k||²`, where Q is an orthonormal basis for the first k MEG score columns. This is the fraction of variance in the retained iEEG scores captured by the MEG temporal subspace. Given the k first iEEG componant we ask we the k MEG componant can recontruct and how well they can recontruct iEEG time courses using a linear combination of the time courses. Coefficient extracted are constent over time but different for each iEEg target: E_hat = MB amd wonder if E == E_hat. We can see this probelam as a a transformation from one basis to an other with Qm an orthogonal basis of M. E_hat = QmQm.TE this projects the iEEG time course (E) into the space of the MEG (M). The capture recontruct fraction is norm(E_hat) - norm(E) (Frobenius norm).
It is variance weighted : Suppose the three iEEG components contain 60%, 30% and 10% of the retained variance, and MEG reconstructs 90%, 50% and 20% of their respective variances:
\[
F=0.60(0.90)+0.30(0.50)+0.10(0.20)=0.71.
\]
2. **Temporal subspace overlap:** `||Q_MEG.T @ Q_iEEG||² / k`, the mean squared cosine of principal angles. We remove the amplitude and keep only the direction (orthonormal). H=Qm.TQe and then compute the overlap if 1 --> overlap if 0 --> orthogonal. 
3. **Matched component correlation:** mean absolute **Pearson** correlation under optimal one-to-one Hungarian matching of the first k components.

The first two are invariant to a change of basis within the retained subspace; the third handles sign/order ambiguity but not arbitrary rotations. 

**Two separate sensitivity analyses.** The final notebook sections now use `coverage_stability.py`. Both use only the mean absolute Pearson correlation after one-to-one matching of the first k PCA time courses. Matching handles component order; absolute values handle sign flips. Per-component signed/absolute correlations are retained in `component_pairs.csv`.

**A. Actual subject-count stability (5, 10, 20, 30).** Independently sample MEG and iEEG participants without replacement. Within a repetition, smaller subsets are nested in larger ones. Counts exceeding the available cohort are explicitly marked as unavailable, not replaced by nearby counts. MEG uses `full_average`, `full_concatenated` and `coverage_average`; no participant assignment is needed, so five MEG participants are allowed even with thirty iEEG participants. iEEG retains all electrodes of each sampled subject in original order. Full averaging uses MEG row indices without grid alignment. Native feature counts are kept, so subject-count effects in concatenated MEG/iEEG also include increased feature coverage.

For each modality/composition/count, compare each fitted PCA to the corresponding full-cohort PCA, and compare every pair of resampled PCAs at that count. For cross-modal correspondence, compare each MEG subset to fixed full-cohort iEEG PCA, and each iEEG subset to each fixed full-cohort MEG composition. This separates PCA stability from cross-modal agreement. Plots show median/range versus actual selected subject count. Full-cohort subset stability is one by construction when all participants are selected; this is not independent validation.

**B. Fixed-cohort pairing sensitivity.** Keep participant identities, subject counts and iEEG electrode slots fixed. Reuse the notebook's existing `paired_coverage` assignment as the baseline when available. Otherwise select a fixed roster once and use its initial assignment as the baseline. Randomly permute MEG-to-iEEG assignments across repetitions, computing only `paired_coverage` and `random_control`. The same selected iEEG PCA is the reference throughout.

Random-control source randomization is fixed separately for each MEG subject: draw one permutation of that subject's source indices once, and apply it to the anatomical source indices in every assignment. This preserves repeated-source multiplicities while avoiding fresh source draws as an additional varying factor. It deliberately changes the control-randomization design from the old mixed sweep.

Compare each randomized PCA with the baseline-assignment PCA, compare all pairs of randomized PCAs, and correlate every result with the fixed iEEG PCA. Plots show the correlation distributions and mark baseline cross-modal correspondence. Export paired-minus-control correlation differences for the same assignment. High stability supports robustness to arbitrary pairing; it does not validate biological identity or true correspondence between different participants. Neither analysis produces inferential p-values or population confidence intervals. Component matching is descriptive and reoptimized separately for every comparison.

**Outputs.** Both runs save `config.json`, `metrics.csv`, `component_pairs.csv`, `participants.csv`, `scores.npz` and `COMPLETE.json`. The subject-count run also saves `availability.csv`. The pairing run adds `assignments.csv`, `source_mapping.csv`, `ieeg_features.csv`, `control_source_permutations.npz` and `paired_control_differences.csv`. The notebook saves separate PNG figures in `out/subject_count_stability` and `out/pairing_stability`. Use `load_stability_results` to reload either completed run; existing results retain their saved settings, and changing settings requires a new output directory.

### Results and interpretations
TODO


## 2. Alternative objectives and PLSSVD validation
**Question1.** Does maximizing cross-modal covariance produce reproducible shared patterns, and how much modality-specific variance do those patterns retain?

**Model :** PLSSVD selects paired unit-norm weight vectors that maximize cross-covariance. The implementation uses full nonzero sample-space factors. PLSSVD itself is a covariance decomposition. The validation adds ridge regression from one modality's latent scores to the other modality's original features..

**PLSSVD Validation:** 
- Trials are partitioned separately within participant, modality and condition before condition averaging. 
- Approximately 70% train the model; the remaining 30% form the test set. 
- At least six trials per condition are required, with small-count adjustments to keep two training trials and two per test half. 
- Group splitting is available for dependent trials, requiring at least three valid groups with sufficient trials in each condition. 
- Both conditions are average for the fit.
- 5 fix components are always retained. insufficient training rank raises an error rather than selecting a smaller k. 
- Every split uses the same complete cohort, not an 80% subset. The evaluation concerns new trials from the same participants. 
- Split 0 is special only for optional null diagnostics and example time-course plots.

**Goodness of fit on each test set.** Three complementary questions are computed:

| Question | Metric | Interpretation |
|---|---|---|
| Do the learned spaces represent both modalities? | `ieeg_reconstruction_fraction`, `meg_reconstruction_fraction` | Own-modality test reconstruction using fixed training PLS weights, relative to training feature means |
| Can each modality predict the other? | `predict_ieeg_q2`, `predict_meg_q2` | Ridge cross-modal prediction learned on training data; positive test Q² beats the training-mean baseline |
| Is shared covariance retained? | `crosscov_energy_fraction`, `mean_paired_covariance`, paired score r | Test cross-covariance captured by training spaces, with signed covariance and correlation to show magnitude/direction |

For one modality, with training mean \(\mu\), orthonormal training weights W and \(X_c=X_{test}-\mu\), reconstruction is \(\widehat X_c=X_cWW^\top\). The retained fraction is \(1-\|X_c-\widehat X_c\|_F^2/\|X_c\|_F^2\). Training whole-modality scaling is undone for this reconstruction. Test signals supply their own scores, so this measure is not cross-modal prediction. The latter uses the training ridge map and reports \(Q^2=1-\mathrm{SSE}_{prediction}/\mathrm{SSE}_{training\ mean}\), which can be negative.

For cross-covariance, let \(C_{test}\) be the full feature cross-covariance after partition centering and fixed training preprocessing/scaling. The retained energy fraction is

\[
\frac{\|W_I^\top C_{test}W_M\|_F^2}{\|C_{test}\|_F^2}.
\]

The denominator is evaluated through observation Gram matrices. The test latent cross-covariance need not be diagonal, so its full matrix contributes to the primary energy fraction. `paired_crosscov_energy_fraction` separately measures the diagonal energy. Per-component train/test covariance and signed paired correlations are also saved. `paired_covariance_retention` in `summary.csv` compares the mean paired test covariance with training; it can exceed one or become negative. Covariance magnitude depends on preprocessing and units, and a high retained fraction alone does not demonstrate a strong biological shared signal. Main test metrics never rematch components or flip signs based on test outcomes.

**Stability across splits.** `metric_summary.csv` gives count, mean, SD, median, minimum and maximum for each train/test metric. `fold_stability.csv` compares temporal scores from every pair of independently refitted splits, separately by modality and train/test partition, using one-to-one matched mean absolute Pearson r. `fold_component_pairs.csv` records those descriptive assignments. Matching handles sign/order variation only, not arbitrary rotations; it is not used to optimize the main test metrics. Test averages may share trials across splits, so stability and performance spread are descriptive, not independent replicates or confidence intervals. Test-half temporal/contrast/forward-pattern reliability remains available as a distinct conditional-on-model measure.

**Outputs and plots.** `summary.csv`, `components.csv`, `fold_metrics.csv`, `metric_summary.csv`, `fold_stability.csv`, `fold_component_pairs.csv`, participant/trial/matching audits, preprocessing, trained models, score covariance matrices and projected scores. The reader requires a completion marker for new-format runs. New figures are `heldout_model_goodness`, `heldout_crosscovariance`, `primary_crosscovariance`, `fold_performance_consistency` and `fold_temporal_stability` (PNG and PDF). Training/test time courses and optional split-0 null diagnostics remain available. There is no new `selection.csv`; older tuning-based outputs are readable through the loader but must not be interpreted as the fixed-k experiment.

**Remaining scope.** Define scientifically meaningful acceptance criteria and supply group/exchangeability metadata. Audit upstream preprocessing for leakage. A stable shared evoked response can reflect common stimulus timing rather than condition-specific information. Optional temporal-shift and spatial-row-shuffle results remain diagnostics unless their exchangeability assumptions are justified. Across-split score consistency does not establish independent spatial-weight stability, subject-level generalization or a calibrated noise ceiling.


## 3. Comapre models within modality

**Question.** Within iEEG, and separately within MEG, are the dominant PCA patterns the same as the patterns selected by PLSSVD or joint PCA?

**Method used.** `cov_models_utils.py` fits separate PCA, joint PCA on feature-concatenated modalities, and exact PLSSVD on `X.T @ Y/(n−1)`. Separate PCA maximizes each modality's variance; joint PCA maximizes variance of the concatenated data; PLSSVD selects paired unit-norm weight vectors that maximize cross-covariance. The implementation uses full nonzero sample-space factors.

**Implemented.** `compare_models.py` supplies descriptive comparisons in `cov_models.ipynb` and held-out comparisons within the existing `compare_subspace.py` batch pipeline. All model pairs are compared separately within iEEG and MEG at common dimensions. Temporal scores, two-condition contrasts (when stacked), and forward patterns on **all native features** are compared. Pattern regression is recomputed for each retained k.

In the held-out pipeline, a one-to-one Hungarian assignment and sign orientation are learned from **training temporal scores**, then reused for every representation and partition. The descriptive notebook learns its assignment on the same data it summarizes and labels the results `in_sample`. Thus spatial and temporal metrics concern the same component pairs. There is no test rematching or test-based sign orientation. `matched_signed_r` retains the training orientation, so a test-time sign reversal lowers it; `matched_abs_r` ignores the sign of the resulting correlations without changing assignments. Valid-pair counts, full signed Pearson matrices, principal angles, ranks and rank-aware subspace overlap are also exported. Undefined training pairs are excluded from matched summaries; valid-pair counts expose this limitation.

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
| Resolved | `construct_five_datasets` | Full-source averaging requires equal signal-array shapes and uses existing row order, without coordinate matching or reordering, as requested. |
| High | Coverage/covariance loader calls | MEG epoch origin is set from the iEEG epoch origin. Supply independently verified MEG time metadata; matching constructed vectors alone does not verify acquisition alignment. |
| High | Across stages | Coverage/covariance default to condition averaging; validation/comparison stack conditions. Choose a primary representation and rerun comparable stages consistently. Condition averaging cannot address condition differences. |
| High | Across entry points | PLSSVD batch excludes `SUBJ_0038`; coverage/covariance discovery does not. Record one participant manifest and exclusion rationale. Comparison uses the cache cohort. |
| Implemented | Model comparison | Descriptive and trial-held-out within-modality comparisons now exist; run the updated pipeline before interpreting project results. |
| Medium | Scaling defaults | Covariance/validation use `none`; subspace comparison defaults to `equal_variance`. Record and harmonize this choice, especially for joint PCA. Scalar modality scaling alone does not change ideal PLSSVD directions, but changes joint PCA's balance. MEG channel z-scoring versus unstandardized iEEG also changes what “dominant variance” means. |
| Medium | Correlation reporting | Coverage matrices are Spearman; matching summaries and validation use Pearson. Label both explicitly. A notebook TODO requests Spearman native matching, but it is not implemented; do not describe it as completed. |
| Implemented | Coverage design | Separate actual subject-count (MEG and iEEG) and fixed-cohort pairing analyses are available; run them on the project recordings. |
| Medium | Validation/results | Test-half reliability fixes trained axes; PLSSVD now also reports cross-split refitted temporal-score correlations. Independent spatial-weight/subspace stability remains separate. Do not average PC1 across repetitions as if component identity were guaranteed. |
| Medium | Result lifecycle | Fixed-k PLSSVD runs now reject existing model/config outputs and write a completion marker. Interrupted runs require a new directory; automatic resume remains unimplemented. |
| High | Reproducibility | The six-test synthetic suite passed on 23 September, but `tests/test_comparison_extensions.py` is absent on 24 September; only its compiled cache remains. Restore the test source before rerunning validation. A dependency manifest is also absent; `src.setting/GetInfo`, data and cluster paths remain external. |
| Text | Methods/results | Supply task/condition meanings, participant/trial/electrode counts, exclusions, filtering, baseline/reference, MEG source reconstruction, coordinate frame, epoch/window choice, source polarity handling, acquisition differences and trial dependence. These cannot be inferred reliably from numeric condition codes. |

Completed maintenance includes the obsolete notebook-import fix, corrected PLSSVD execution instructions, covariance interpretation text, implemented within-modality notebook cells and the requested index-based full-source averaging. Existing scientific preprocessing defaults remain unchanged. The remaining code gaps include independent-refit spatial-weight stability, inferential model/cluster comparisons, participant-balanced sampling options and robust run-resume handling. Task semantics, valid exchangeability groups and verified coordinate/time metadata require study-specific information.

## Project map and recommended sequence

| Files | Role |
|---|---|
| `coverage_matching.ipynb`, `coverage_matching_utils.py` | Data loading, anatomical sampling, five MEG compositions and descriptive PCA comparisons |
| `coverage_stability.py` | Separate actual subject-count and fixed-cohort pairing sensitivity, matched correlations, saved results and plots |
| `coverage_sampling.py` | Legacy mixed pool/feature-budget sweep; no longer called by the coverage notebook |
| `tests/test_coverage_stability.py` | Five synthetic tests for sign/order invariance, both-modality count sweeps, fixed pairing rosters/control maps, persistence and plotting |
| `compare_models.py` | Within-modality model-pair comparisons with training-frozen component matching |
| `tests/` | `test_meg_grids.py` checks index-based averaging with differing coordinates and incompatible array shapes; restore the missing `test_comparison_extensions.py` to rerun the original six-test suite |
| `cov_models.ipynb`, `cov_models_utils.py` | Descriptive separate/joint PCA and PLSSVD objectives, reconstruction and cross-covariance metrics |
| `plssvd_eval.py`, `plssvd_eval_utils.py`, `plssvd_eval.ipynb` | Trial export/cache, fixed-k repeated train/test evaluation, goodness/stability metrics, persisted outputs and reader |
| `compare_subspace.py`, `compare_subspace.ipynb` | Held-out cross-modality geometry, alignment and spatial clustering for all three models |
| `OLD/LB10*` | Earlier extraction, concatenation and randomized matching work |
| `OLD/LB11*`, `OLD/LB12*`, `OLD/LB13*`, `OLD/LB14*`, `OLD/LB_Summary.ipynb`, `OLD/utils.py` | Earlier embedding, decomposition, interpretation, manifold and frequency explorations; historical context, not the current validation pipeline |

The active modules and all current notebook source cells were inspected. Legacy files were inventoried and entry points inspected; their analyses were not revalidated.

Recommended order: establish task/data provenance and alignment → assess coverage/composition sensitivity → fit competing objectives → validate on independent trials → compare models within modality → compare modalities within model → optionally interpret stable spatial clusters. Freeze scientific choices before inspecting final test outcomes. If exploratory coverage/model results already used the eventual test trials, a later split does not retrospectively make the entire study confirmatory; describe it as conditional validation or obtain independent confirmation.


## Running the extensions

Run the final two sections of `coverage_matching.ipynb` after loading `ieeg` and the initial datasets. Defaults are 20 repetitions, 3 retained components, and actual subject counts `(5, 10, 20, 30)` for both modalities. Pairing sensitivity fixes the existing paired-coverage roster/assignment when available and then permutes assignments without changing subject membership. The two analyses have separate output directories. Completed runs reload their saved configuration; choose new directories for changed settings. Interrupted runs are not automatically resumed.

For fixed-k PLSSVD generalization, run into a new output directory:

```bash
python -u plssvd_eval.py --root /path/to/iEEGvsMEG \
  --meg-kind full_concatenated --n-components 5 --repeats 5 \
  --train-fraction 0.7 --output-dir /path/to/new_plssvd_eval_fixed
```

Then open `plssvd_eval.ipynb` with that output directory. Its displayed defaults do not launch a computation; the notebook reads completed batch results. The default output location is `out/plssvd_eval_fixed`, keeping older tuning-based runs separate.

Run the final section of `cov_models.ipynb` for descriptive within-modality comparisons. For held-out results, use the existing batch command with at least two models and a new output directory:

```bash
python -u compare_subspace.py --cache-dir /path/to/trial_cache \
  --output-dir /path/to/new_comparison \
  --models separate_pca plssvd joint_pca --meg-kind paired_coverage \
  --dimensions 1 2 3 5 10 --repeats 5 --block-scaling equal_variance
```

Inspect the new within-modality section of `compare_subspace.ipynb`. The batch plotter also exports within-modality summaries and heatmaps. 

## Verification record

On **23 September 2026**, all six synthetic tests passed in the project's `ieeg` Python environment:

1. End-to-end trial-cache batch execution, result reload and figure exports.
2. Coverage rank skipping and rejection of mismatched full-average source grids.
3. Reproducible/nested coverage sampling, coordinated paired/random controls and a fixed iEEG variance reference.
4. Agreement between descriptive and held-out comparison paths when given identical partitions.
5. Frozen component matching/signs, including an adversarial test-time sign reversal and changed test data.
6. Rotation-invariant overlap, rank loss and undefined component correlations.

All active Python modules and notebook code cells also passed syntax checks during that implementation. These checks used synthetic data and establish neither real-data performance nor the validity of upstream preprocessing.

On **24 September 2026**, the implementation and notebook wiring were inspected for this documentation update. The test source `tests/test_comparison_extensions.py` is missing; only `tests/__pycache__/test_comparison_extensions.cpython-311.pyc` remains. Tests were **not rerun** for this update. Restore the source before using:

```bash
MPLBACKEND=Agg python -m unittest discover -s tests -v
```

Verify that six tests are actually discovered; a zero-test run is not validation. Real-data results, independent-refit stability and participant-level generalization remain unverified.


## Full-average dataset construction

At the user's request, `full_average` averages MEG sources by their existing row index. Participants must have the same source count and compatible condition/time dimensions. Coordinate equality is not required, and no source-grid alignment, reordering or interpolation is performed. The first participant's coordinates remain the coordinate metadata for the averaged dataset. iEEG signals and electrode order are not aligned or reordered. The existing nearest-source sampling used by coverage-specific MEG compositions remains part of those analyses.

The coordinate diagnostics/reordering helpers and their notebook cells have been removed. Both the dataset builder and repeated coverage sampler use shape compatibility checks only for full averaging.


Verification of the separated analyses: all seven currently available tests passed (five new stability tests and two index-based averaging tests). The new tests exercise the exact 5/10/20/30 grid for MEG and iEEG, small MEG subsets with a larger iEEG cohort, unavailable-count reporting, sign/order invariance, fixed participant membership and control-source mappings across assignments, reproducibility, result reload and plot exports. Project recordings remain unavailable in this checkout, so these are synthetic checks rather than real-data results.


Verification of fixed-k PLSSVD evaluation: all **13 currently available tests passed**, including six new evaluation tests. Checks cover train/test disjointness and complete trial assignment, group integrity, the unchanged tuning split used by `compare_subspace`, fixed cohort/pairing across repetitions, strict enforcement of k, dense-reference covariance/reconstruction algebra, result reload, figure exports, and a test-only data perturbation that leaves all trained weights/means/predictors unchanged. The cache exporter now accepts the six-trial minimum of the new design. Command-line help and all active Python/notebook code cells were also checked. No project recordings were evaluated locally.
