# iEEG–MEG latent-space comparison: methods and implementation review

## Current status

| Analysis | Implementation | Evaluation scope | Output |  Results | 
|---|---|---|---|---| 
| Coverage matching and composition sensitivity | Implemented in `coverage_stability.py` and `coverage_matching.ipynb` | Compute different MEG dataset organisation and test how much the information is shared with iEEG using PCA on both dataset individually |  
| PLSSVD held-out evaluation | Implemented in `plssvd_eval` modules | Fixed **5 components by default**, shuffled folds with disjoint test trials and condition-averaged responses, all participants and matching fixed; no tuning |
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
- Shuffle once within subject and condition; divide trials into five nearly equal test folds. Each trial is held out exactly once; other folds provide training (approximately 80/20 per fold).
- No test A/B subdivisions. At least n_splits trials per condition are required. Group splitting keeps whole groups together and requires at least n_splits groups with each condition represented in every fold.
- Average trials within each condition separately for train and test, then average conditions equally. Models receive time × features matrices, not individual trials or condition-stacked observations.
- Concatenate all iEEG electrodes across subjects. MEG full_concatenated concatenates sources; full_average averages source indices across subjects; coverage_average averages matched coverage; paired_coverage/random_control retain their paired sampled-feature construction.
- The PCA/sample-space factors inside PLSSVD operate on these averaged matrices. Keep 5 configurable components; insufficient training rank raises an error.
- All folds use the same cohort and participant/source assignment. No tuning. Fold 0 is special only for example plots and optional temporal/spatial null diagnostics.
- Condition-label nulls and condition-contrast/test-half outputs are removed from this condition-averaged evaluation. Legacy compare_subspace remains separate and unchanged.

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

**Stability across splits.** `metric_summary.csv` gives count, mean, SD, median, minimum and maximum for each train/test metric. `fold_stability.csv` compares temporal scores from every pair of independently refitted splits, separately by modality and train/test partition, using one-to-one matched mean absolute Pearson r. `fold_component_pairs.csv` records those descriptive assignments. Matching handles sign/order variation only, not arbitrary rotations; it is not used to optimize the main test metrics. Test folds share no trials, but training folds overlap. Stability and performance spread remain descriptive, not independent replicates or confidence intervals. No test-half reliability is computed in this workflow.

**Outputs and plots.** `summary.csv`, `components.csv`, `fold_metrics.csv`, `metric_summary.csv`, `fold_stability.csv`, `fold_component_pairs.csv`, participant/trial/matching audits, preprocessing, trained models, score covariance matrices and projected scores. The reader requires a completion marker for new-format runs. New figures are `heldout_model_goodness`, `heldout_crosscovariance`, `primary_crosscovariance`, `fold_performance_consistency` and `fold_temporal_stability` (PNG and PDF). Training/test time courses and optional split-0 null diagnostics remain available. There is no new `selection.csv`; older tuning-based outputs are readable through the loader but must not be interpreted as the fixed-k experiment.

**Remaining scope.** Define scientifically meaningful acceptance criteria and supply group/exchangeability metadata. Audit upstream preprocessing for leakage. A stable shared evoked response can reflect common stimulus timing rather than condition-specific information. Optional temporal-shift and spatial-row-shuffle results remain diagnostics unless their exchangeability assumptions are justified. Across-split score consistency does not establish independent spatial-weight stability, subject-level generalization or a calibrated noise ceiling.


## 3. Comapre models

### 3.1. Run the 3 and observe the how the cov and corr behave
**Question.** What information does each objective emphasize, and how much cross-modal correspondence accompanies it?

notebook cov_model.ipynb

**Method used.** `cov_models_utils.py` fits separate PCA, joint PCA on feature-concatenated modalities, and exact PLSSVD on `X.T @ Y/(n−1)`. Separate PCA maximizes each modality's variance; joint PCA maximizes variance of the concatenated data; PLSSVD selects paired unit-norm weight vectors that maximize cross-covariance. The implementation uses full nonzero sample-space factors.

### 3.2 Compare within modality
**Question.** Within iEEG, and separately within MEG, are the dominant PCA patterns the same as the patterns selected by PLSSVD or joint PCA?

Use fitted models from the previous section.

We compare temporal scores and spatial forward patterns over all native features.
- **Subspace overlap** compares all retained patterns together and permits rotations/mixing of their axes. 
- **Mean matched |r|** compares individual component pairs and averages their absolute Pearson correlations. 

High overlap with lower matched correlation means similar collective patterns represented by different mixtures.

Allignement and matching: one-to-one Hungarian assignment and sign orientation are learned from **training temporal scores**, then reused for every representation and partition.

**Remaining scope.** Independently refitted component stability, participant-level inference, anatomical pattern displays and explicit component-acceptance criteria remain separate work. Test-half correspondence is conditional on the fitted axes. Near-degenerate components can rotate, making subspace metrics more stable than componentwise matching.

## 4. Cross-modal geometry

**Question.** For a fixed method and number of components, how similar are iEEG and MEG latent structures, and what transformation relates them?

**Data and models.** `compare_subspace.py` reuses the trial cache and fits separate PCA, PLSSVD and joint PCA. Within each subject, average trials separately by condition, then average conditions equally. Model inputs are time × features. 

/!\ : Default whole-modality scaling remains `equal_variance` (configurable), unlike PLSSVD evaluation's `none`.

**All-data description.** Fit preprocessing, model weights, transformations and cluster centroids using every trial. Score on the same data. These rows are explicitly labelled `analysis=in_sample`, `partition=in_sample`, `repeat=-1`; they are not test results.

**Train/Test splits: five disjoint test folds.** Shuffle trials once within each subject and condition and divide into five nearly equal groups (approximately 80/20 Train/test). Each condition needs at least five trials. Optional group splitting keeps whole groups together and requires every condition in each fold. Refit preprocessing, dimensionality reduction, transformations and centroids from scratch for every fold. There is one test partition, no test A/B and no tuning partition. Training folds overlap, so fold ranges are descriptive rather than confidence intervals.

**Analysis 1 — subspace similarity.** Compute principal angles and overlap for temporal scores and spatial patterns separately. Overlap is the sum of squared principal-angle cosines divided by the requested k; missing numerical dimensions count as zero. Values range from 0 (orthogonal) to 1 (same full-rank subspace). This permits component rotations/mixing and does not require equal component order. Test overlap describes the two held-out representations directly.

**Analysis 2 — transformations.** Center each representation and divide by one scalar RMS using training statistics. Fit identity, orthogonal rotation/reflection, affine and quadratic maps in both directions. Identity uses no learned map; orthogonal preserves distances; affine allows general linear deformation; quadratic adds squared terms and interactions. Ridge is fixed in advance (`--ridge-alpha`, default 0.01), not chosen on test data. All four complexities are reported. Test NRMSE is prediction error norm divided by the target's distance from its training-mean baseline; Q² = 1 − NRMSE². NRMSE below 1 / Q² above 0 beats that baseline. Improvements with complexity must be assessed on test folds; rotations alone need not have physiological meaning.

**Analysis 3 — exploratory spatial grouping.** Fit K-means separately in each modality for every requested cluster count (default 2–6), using all k spatial-pattern coordinates. Report every count without selecting a winner. Training centroids assign held-out locations. Cross-modal adjusted Rand index compares group membership (1 identical, approximately 0 chance-level); silhouette describes separation/compactness.

**Outputs.** `overlap.csv`, `alignment.csv`, `clusters.csv`, `splits.csv`, within-modality comparison tables, and saved preprocessing, model scores/weights, patterns, transformations, centroids and mapping audits. Every result table separates all-data description from cross-validation with an `analysis` label. All-data artifacts use repeat -1 (`-01` in filenames); fold artifacts use 000–004. `COMPLETE.json` marks successful completion. New runs no longer produce `reliability.csv` or `cluster_selection.csv`; the reader retains older-format support.

**Plots.** Dashed lines show the all-data descriptive values. Solid lines show test-fold medians, with minimum–maximum shading. Figures cover overlap versus k, transformation error by complexity/direction for a selected k, and spatial ARI versus predeclared cluster count. The notebook separately displays full-data tables before test results.


## Implementation and reporting gaps to resolve

| Priority | Location | Finding and required action |
|---|---|---|
| High | Coverage/covariance notebook coordinate cells; `plssvd_eval.py: main` | Magnitude-based repeated coordinate division can alter individual axes before the explicit unit checker sees them. Establish the native `GetInfo` unit and apply one documented conversion; verify against known anatomical locations. Do not infer correctness from plausible ranges alone. |
| High | Coverage/covariance loader calls | MEG epoch origin is set from the iEEG epoch origin. Supply independently verified MEG time metadata; matching constructed vectors alone does not verify acquisition alignment. |
| High | Across stages | Coverage/covariance default to condition averaging; validation/comparison stack conditions. Choose a primary representation and rerun comparable stages consistently. Condition averaging cannot address condition differences. |
| High | Across entry points | PLSSVD batch excludes `SUBJ_0038`; coverage/covariance discovery does not. Record one participant manifest and exclusion rationale. Comparison uses the cache cohort. |
| Implemented | Model comparison | Descriptive and trial-held-out within-modality comparisons now exist; run the updated pipeline before interpreting project results. |
| Medium | Scaling defaults | Covariance/validation use `none`; subspace comparison defaults to `equal_variance`. Record and harmonize this choice, especially for joint PCA. Scalar modality scaling alone does not change ideal PLSSVD directions, but changes joint PCA's balance. MEG channel z-scoring versus unstandardized iEEG also changes what “dominant variance” means. |
| Medium | Correlation reporting | Coverage matrices are Spearman; matching summaries and validation use Pearson. Label both explicitly. A notebook TODO requests Spearman native matching, but it is not implemented; do not describe it as completed. |
| Implemented | Coverage design | Separate actual subject-count (MEG and iEEG) and fixed-cohort pairing analyses are available; run them on the project recordings. |
| Medium | Validation/results | Legacy compare_subspace test-half reliability fixes trained axes; PLSSVD evaluation reports cross-fold refitted temporal-score correlations without test halves. Independent spatial-weight/subspace stability remains separate. Do not average PC1 across repetitions as if component identity were guaranteed. |
| Medium | Result lifecycle | Fixed-k PLSSVD runs now reject existing model/config outputs and write a completion marker. Interrupted runs require a new directory; automatic resume remains unimplemented. |
| High | Reproducibility | The six-test synthetic suite passed on 23 September, but `tests/test_comparison_extensions.py` is absent on 24 September; only its compiled cache remains. Restore the test source before rerunning validation. A dependency manifest is also absent; `src.setting/GetInfo`, data and cluster paths remain external. |



Within-modality display update: compare_models now always compares equal-condition-average temporal scores and corresponding forward patterns; condition-contrast comparisons are removed. Existing stacked fitted scores are averaged without refitting weights. The covariance notebook fits on condition averages. Its final cell exports detailed CSVs and one annotated two-panel similarity summary (iEEG/MEG; temporal/spatial overlap and matched absolute correlation), saved as PNG/PDF. PLSSVD batch output defaults to ROOT/out/plssvd_eval_{MEG_KIND}, with --output-dir still available.
