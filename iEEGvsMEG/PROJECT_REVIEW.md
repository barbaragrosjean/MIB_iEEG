# iEEG–MEG latent-space comparison: methods and implementation review

## Current status

Reviewed against the local implementation on October 10, 2026. “Implemented” describes available code, not an established empirical finding. Saved notebook outputs exist, but standalone run tables and their provenance were not available for this review; numerical conclusions therefore remain unverified.

| Analysis | Implementation | Evaluation scope | Principal outputs | Results status |
|---|---|---|---|---|
| Coverage and composition sensitivity | `coverage_stability.py`, `coverage_matching.ipynb` | In-sample PCA correspondence, participant-count and pairing sensitivity | Metrics, component pairs, assignments and coverage figures | Requires verification of saved runs |
| MEG channel-count sensitivity | `elect_number_effect.py` | Nested feature-count sweep against a fixed iEEG PCA reference | Channel-count curves, metrics and feature orders | Descriptive; numerical findings unverified |
| PLSSVD validation | `plssvd_eval.py`, `plssvd_postprocess.py` | Repeated trial-held-out fits; default 100 iterations × 5 folds | Saved fits, evaluated component prefixes and figure snapshots | Generalization claims require matched run results |
| Temporal surrogate validation | `plssvd_phase_null.py`, `plssvd_diagnostics.py` | Refit baseline/null models with matched splits | Covariance, correlation, reconstruction, prediction and spectral diagnostics | Diagnostic null comparisons; no phase-null p-value |
| Within-modality model comparison | `compare_models.py`, `cov_models.ipynb`, `compare_subspace.py` | PCA versus PLSSVD/joint PCA; descriptive and held-out comparisons | Temporal/spatial overlap and matched correlations | Numerical findings unverified |
| Cross-modal geometry | `compare_subspace.py`, `spatial_clouds.py` | Held-out representations, transformations, clustering and unpaired spatial distributions | Overlap, alignment, ARI and cloud metrics | Exploratory; numerical findings unverified |
| Modality-specific spatial organization | `modality_specificity.py`, `modality_specificity.ipynb` | Full-data native projection weights, spatial nulls and anatomical summaries | Distance curves, semivariograms, clusters, regional and hemisphere summaries | Descriptive fits; spatial inference conditional on exchangeability |

## Scientific aim and unit of comparison

The project asks which temporal and spatial structures are reproducible within iEEG and MEG and which correspond across modalities. Separate PCA identifies dominant within-modality variance; PLSSVD identifies axes that maximize cross-modal covariance; joint PCA identifies dominant variance in feature-concatenated data. Comparing these objectives tests whether shared structure coincides with each modality’s dominant structure. Complementary spatial analyses examine modality-specific organization without assuming identical sensors or measurement processes.

The datasets contain corresponding conditions and time axes, not simultaneously recorded or individually paired trials. The current primary model inputs are **time × features**: trials are averaged within each condition, then conditions receive equal weight in a combined response. This representation cannot establish a difference between known and unknown sequences.

**Method and language.**
Distinguish three objects throughout the text: projection **weights** define the latent axes; **scores** are projected time courses; **forward patterns** describe how observed features covary with the scores. Spatial weights and forward patterns are not interchangeable, especially for PLS. The current forward-pattern estimator is the multivariate regression `A = X_centered.T @ T_centered @ pinv(T_centered.T @ T_centered)`.

## 1. Coverage matching and group composition

**Question.** How do anatomical coverage, participant averaging and feature concatenation alter MEG PCA structure and its correspondence with iEEG PCA?

**Method used.** Independently fit centered PCA to pooled iEEG and to five MEG compositions, using the same condition/time rows. PCA is computed through the observation Gram matrix (`XX.T` rather than `X.T @ X`), avoiding a large feature covariance matrix.

| Composition | Construction | Main comparison it supports |
|---|---|---|
| `full_average` | Average participants at corresponding source indices | Whole-source group-average reference |
| `full_concatenated` | Concatenate participant source features | Retain participant-specific source features |
| `coverage_average` | Match every pooled electrode to each participant's nearest source, then average participants | Coverage restriction with group averaging |
| `paired_coverage` | Assign each iEEG participant a distinct sampled MEG participant and retain nearest sources for their electrodes | Coverage and participant composition resembling pooled iEEG |
| `random_control` | Use the same pairing and feature counts, but random source locations; preserve duplicate-source multiplicities | Sensitivity to anatomical source selection |

For pair association: Nearest-source mapping uses Euclidean coordinate distance. 

**Outputs.** Coverage plots, feature/variance summaries, PCA spectra, component maps and time courses; full component-by-component **Spearman** correlation matrices for time courses and anatomically mapped weights; and three temporal metrics at each prespecified common dimension k:

1. **Retained iEEG variance captured:** `||Q_MEG.T @ T_iEEG,k||² / ||T_iEEG,k||²`, where Q is an orthonormal basis for the first k MEG score columns. This is the fraction of variance in the retained iEEG scores captured by the MEG temporal subspace. Let E and M denote the retained iEEG and MEG score matrices, and let Q_M be an orthonormal basis for M. The least-squares projection is E_hat = Q_M Q_M.T E. Its captured fraction is ||E_hat||_F² / ||E||_F² = 1 − ||E − E_hat||_F² / ||E||_F², not a difference between norms. This is an in-sample projection with coefficients constant over time.
It is variance weighted : Suppose the three iEEG components contain 60%, 30% and 10% of the retained variance, and MEG reconstructs 90%, 50% and 20% of their respective variances:
\[
F=0.60(0.90)+0.30(0.50)+0.10(0.20)=0.71.
\]
2. **Temporal subspace overlap:** `||Q_MEG.T @ Q_iEEG||² / k`, the mean squared cosine of principal angles. We remove the amplitude and keep only the direction (orthonormal). H=Qm.TQe and then compute the overlap if 1 --> overlap if 0 --> orthogonal. 
3. **Matched component correlation:** mean absolute **Pearson** correlation under optimal one-to-one Hungarian matching of the first k components.

Subspace overlap is invariant to any full-rank change of basis within either retained space. Captured iEEG variance is invariant to the MEG basis and to orthogonal rotations of the iEEG scores, but not arbitrary rescaling of the target scores. Matched correlation handles sign/order ambiguity but not arbitrary rotations. 

**Two separate sensitivity analyses.** The final notebook sections now use `coverage_stability.py`. Both use only the mean absolute Pearson correlation after one-to-one matching of the first k PCA time courses. Matching handles component order; absolute values handle sign flips. Per-component signed/absolute correlations are retained in `component_pairs.csv`.

**A. Actual subject-count stability (5, 10, 20, 30).** Independently sample MEG and iEEG participants without replacement. Within a repetition, smaller subsets are nested in larger ones. Counts exceeding the available cohort are explicitly marked as unavailable, not replaced by nearby counts. MEG uses `full_average`, `full_concatenated` and `coverage_average`; no participant assignment is needed, so five MEG participants are allowed even with thirty iEEG participants. iEEG retains all electrodes of each sampled subject in original order. Full averaging uses MEG row indices without grid alignment. Native feature counts are kept, so subject-count effects in concatenated MEG/iEEG also include increased feature coverage.

For each modality/composition/count, compare each fitted PCA to the corresponding full-cohort PCA, and compare every pair of resampled PCAs at that count. For cross-modal correspondence, compare each MEG subset to fixed full-cohort iEEG PCA, and each iEEG subset to each fixed full-cohort MEG composition. This separates PCA stability from cross-modal agreement. Plots show median/range versus actual selected subject count. Full-cohort subset stability is one by construction when all participants are selected; this is not independent validation.

**B. Fixed-cohort pairing sensitivity.** Keep participant identities, subject counts and iEEG electrode slots fixed. Reuse the notebook's existing `paired_coverage` assignment as the baseline when available. Otherwise select a fixed roster once and use its initial assignment as the baseline. Randomly permute MEG-to-iEEG assignments across repetitions, computing only `paired_coverage` and `random_control`. The same selected iEEG PCA is the reference throughout.

Random-control source randomization is fixed separately for each MEG subject: draw one permutation of that subject's source indices once, and apply it to the anatomical source indices in every assignment. This preserves repeated-source multiplicities while avoiding fresh source draws as an additional varying factor. It deliberately changes the control-randomization design from the old mixed sweep.

Compare each randomized PCA with the baseline-assignment PCA, compare all pairs of randomized PCAs, and correlate every result with the fixed iEEG PCA. Plots show the correlation distributions and mark baseline cross-modal correspondence. Export paired-minus-control correlation differences for the same assignment. High stability supports robustness to arbitrary pairing; it does not validate biological identity or true correspondence between different participants. Neither analysis produces inferential p-values or population confidence intervals. Component matching is descriptive and reoptimized separately for every comparison.

**Outputs.** Both runs save `config.json`, `metrics.csv`, `component_pairs.csv`, `participants.csv`, `scores.npz` and `COMPLETE.json`. The subject-count run also saves `availability.csv`. The pairing run adds `assignments.csv`, `source_mapping.csv`, `ieeg_features.csv`, `control_source_permutations.npz` and `paired_control_differences.csv`. The notebook saves separate PNG figures in `out/subject_count_stability` and `out/pairing_stability`. Use `load_stability_results` to reload either completed run; existing results retain their saved settings, and changing settings requires a new output directory.

### Results and interpretation criteria

No verified numerical result is added here. High retained-iEEG variance with lower overlap would indicate that correspondence is concentrated in the larger-variance retained directions. High overlap with lower matched component correlation would indicate similar subspaces expressed through different component mixtures. Stable pairing results would support robustness to participant assignment, not biological equivalence of paired individuals. Report cohort sizes, feature counts, selected k, repetitions and metric distributions together; increasing participant count also changes coverage in concatenated datasets.

### Channel-count sensitivity

`elect_number_effect.py` uses nested random **ascending** subsets of full-concatenated MEG features. The default step is 100 channels, up to the largest complete step no greater than min(20,000, available channels). Five-component MEG PCA is refitted at each count and compared with a fixed five-component full-iEEG temporal reference. Outputs include individual/cumulative explained variance, total remaining-feature variance, retained-iEEG variance captured, temporal overlap and matched absolute Pearson correlation.

Sampling is global across features, not balanced by participant. Each repetition has one saved channel order whose prefixes define the subsets. Results describe sensitivity to channel count and changing coverage; they do not isolate participant effects or provide independent validation. `out/elect_number_effect` contains configuration, metrics, component pairs, feature metadata, saved orders and PNG/PDF plots. The notebook loads completed results; `--plot-only` provides the same behavior from the command line.


## 2. Alternative objectives and PLSSVD validation
**Question.** Does maximizing cross-modal covariance produce reproducible shared patterns, and how much modality-specific variance do those patterns retain?

**Model.** PLSSVD selects paired unit-norm weight vectors that maximize cross-covariance. The implementation uses full nonzero sample-space factors. PLSSVD itself is a covariance decomposition. The validation adds ridge regression from one modality's latent scores to the other modality's original features.

**PLSSVD Validation:**
- Shuffle once within subject and condition; divide trials into five nearly equal test folds. Each trial is held out exactly once; other folds provide training (approximately 80/20 per fold).
- No test A/B subdivisions. At least n_splits trials per condition are required. Group splitting keeps whole groups together and requires at least n_splits groups with each condition represented in every fold.
- Average trials within each condition separately for train and test, then average conditions equally. Models receive time × features matrices, not individual trials or condition-stacked observations.
- Concatenate all iEEG electrodes across subjects. MEG full_concatenated concatenates sources; full_average averages source indices across subjects; coverage_average averages matched coverage; paired_coverage/random_control retain their paired sampled-feature construction.
- The PCA/sample-space factors operate on these averaged matrices. Fit up to `--max-components` (default 100), then evaluate a prespecified leading k without refitting PLSSVD. The phase launcher inherits an existing baseline’s fitted count when this option is omitted. Insufficient training rank raises an error.
- Repeat the shuffled five-fold procedure for 100 iterations by default. All folds and iterations use the same cohort and participant/source assignment. There is no tuning; repeated trial splits are not independent biological samples.
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

**Fit, evaluation and output separation.** Fits are saved under `out/plssvd_eval/{MEG_KIND}/{PERMUTATION}`, with `none` for baseline. Each `iteration_XXX` contains fold models, scores, sufficient statistics, preprocessing and split/matching audits. `plssvd_postprocess.py` evaluates requested component prefixes and prepares saved snapshots; `plssvd_figures.py` reads prepared results. Reconstruction and k-specific ridge prediction are evaluated from saved information rather than refitting the PLS axes. Missing PCA/spectral artifacts can require reconstruction from the original trial cache. Completed iterations can be evaluated even when a run-level completion marker is absent; incomplete iterations are excluded. The phase launcher reuses these iterations but does not resume interrupted fits.

Prepared tables include fold/component metrics, iteration means and descriptive stability summaries. Figures cover held-out reconstruction, prediction, covariance, correlations and stability. Selecting k after inspecting test results makes that selection exploratory.

### 2.1. Phase-randomization null

**Question.** Does held-out cross-modal correspondence exceed that obtained after altering temporal phase relationships while preserving each modality’s internal second-order structure?

For each Fourier frequency f and channel j in a randomized modality, the surrogate is

\[
\widetilde X_j(f)=X_j(f)e^{i\phi_f},\qquad
\phi_f\sim\mathrm{Uniform}(-\pi,\pi).
\]

One phase vector is shared across channels, participants, conditions and train/test folds within an iteration. DC and, for even-length signals, Nyquist coefficients are unchanged. Inverse real FFT returns the surrogate signal. Consequently power spectra and cross-spectra within that modality are preserved: the common phase cancels in the product of one channel’s Fourier coefficient and another’s complex conjugate. This also preserves circular second-order temporal structure and spatial covariance, but not event timing or every local temporal property of an evoked response.

`--perm meg` randomizes only MEG; `--perm ieeg` only iEEG; `--perm both` uses **independent** phase vectors for the two modalities. `--perm all` runs all three. Using the same phase vector in both modalities would preserve their cross-spectrum and would not provide this null. New phase vectors are drawn across iterations, and PLSSVD is refitted for each surrogate with the baseline’s matching and trial-split design.

```bash
python -u plssvd_phase_null.py --perm all --components 10
```

Outputs are `none`, `meg__phase`, `ieeg__phase` and `both__phase` below `out/plssvd_eval/full_concatenated` by default. The final notebook cell and exported `phase_null_comparison_k*.png/.pdf` compare matched iteration means of held-out covariance, signed Pearson correlation and each modality’s reconstruction. Matching uses available shared iteration IDs. The component count is inherited from the saved baseline; other explicit/default settings must still match it.

**Interpretation.** A null correlation need not be zero: smooth, low-dimensional signals can remain alignable, and the model optimizes covariance again on every surrogate. Evidence against this surrogate model requires the observed metrics to exceed its corresponding distribution, not merely to be positive. Strong own-modality reconstruction alone is insufficient because the surrogate intentionally preserves within-modality covariance. Conditions are kept intact; this does not test a known-versus-unknown contrast. These plots do not report a calibrated permutation p-value. Reused trials, event-locked nonstationarity and the circular Fourier construction limit an exact exchangeability interpretation.

### 2.2. Other surrogates and spectral diagnostics

Existing temporal circular shifts, block permutations and point permutations use independent channel transforms shared across conditions and partitions within an iteration. Spatial permutations shuffle channels separately at each time point. They change different aspects of within-modality structure and should not be pooled into one null. A fixed channel relabeling alone would leave an unconstrained PLS refit equivalent.

`plssvd_diagnostics.py` and prepared comparisons report signed component correlations/covariances, held-out reconstruction and prediction, and paired null-minus-baseline differences. Temporal singular-value spectra of the preprocessed inputs diagnose changes in dimensionality. For normalized spectral energies p_j = sigma_j² / sum_l sigma_l², the participation-ratio effective rank is 1 / sum_j p_j². Spectra use all temporal dimensions, not just the selected PLS prefix. These diagnostics help distinguish destroyed cross-modal alignment from altered marginal structure; they do not by themselves establish physiological specificity.

### 2.3. PCA reference comparisons

Training-only PCA sidecars support fold-specific comparisons conditional on training preprocessing. Separately, `plssvd_full_pca.py` fits unpermuted PCA to all cached trials and correlates its score time courses with saved PLS scores, reporting component-pair means and sample SD across iterations for each fold and partition. The latter is descriptive because its PCA reference includes test trials. Output is saved under the selected run’s `full_data_pca_comparison` directory.

The current `plssvd_fit_pca.py` is a hard-coded full-data comparison script (`full_concatenated`, 10 components, `ieeg__time_point`), not the configurable PCA-only backfill CLI described in `PLSSVD_WORKFLOW.md`. That workflow documentation needs reconciliation before its example command is used.

**Remaining scope.** Define scientifically meaningful acceptance criteria and supply group/exchangeability metadata. Audit upstream preprocessing for leakage. A stable shared evoked response can reflect common stimulus timing rather than condition-specific information. Optional temporal-shift and spatial-row-shuffle results remain diagnostics unless their exchangeability assumptions are justified. Across-split score consistency does not establish independent spatial-weight stability, subject-level generalization or a calibrated noise ceiling.


## 3. Compare models

### 3.1. Compare variance and covariance objectives
**Question.** What information does each objective emphasize, and how much cross-modal correspondence accompanies it?

Notebook: `cov_models.ipynb`.

**Method used.** `cov_models_utils.py` fits separate PCA, joint PCA on feature-concatenated modalities, and exact PLSSVD on `X.T @ Y/(n−1)`. Separate PCA maximizes each modality's variance; joint PCA maximizes variance of the concatenated data; PLSSVD selects paired unit-norm weight vectors that maximize cross-covariance. The implementation uses full nonzero sample-space factors.

### 3.2 Compare within modality
**Question.** Within iEEG, and separately within MEG, are the dominant PCA patterns the same as the patterns selected by PLSSVD or joint PCA?

Use fitted models from the previous section.

We compare temporal scores and spatial forward patterns over all native features.
- **Subspace overlap** compares all retained patterns together and permits rotations/mixing of their axes. 
- **Mean matched |r|** compares individual component pairs and averages their absolute Pearson correlations. 

High overlap with lower matched correlation means similar collective patterns represented by different mixtures.

Alignment and matching: one-to-one Hungarian assignment and sign orientation are learned from **training temporal scores**, then reused for every representation and partition.

**Remaining scope.** Independently refitted component stability, participant-level inference, anatomical pattern displays and explicit component-acceptance criteria remain separate work. Legacy test-half correspondence is conditional on the fitted axes; the current workflow uses disjoint test folds. Near-degenerate components can rotate, making subspace metrics more stable than componentwise matching.

## 4. Cross-modal geometry

**Question.** For a fixed method and number of components, how similar are iEEG and MEG latent structures, and what transformation relates them?

**Data and models.** `compare_subspace.py` reuses the trial cache and fits separate PCA, PLSSVD and joint PCA. Within each subject, average trials separately by condition, then average conditions equally. Model inputs are time × features. 

**Scaling.** Default whole-modality scaling remains `equal_variance` (configurable), unlike PLSSVD evaluation's `none`.

**All-data description.** Fit preprocessing, model weights, transformations and cluster centroids using every trial. Score on the same data. These rows are explicitly labelled `analysis=in_sample`, `partition=in_sample`, `repeat=-1`; they are not test results.

**Train/Test splits: five disjoint test folds.** Shuffle trials once within each subject and condition and divide into five nearly equal groups (approximately 80/20 Train/test). Each condition needs at least five trials. Optional group splitting keeps whole groups together and requires every condition in each fold. Refit preprocessing, dimensionality reduction, transformations and centroids from scratch for every fold. There is one test partition, no test A/B and no tuning partition. Training folds overlap, so fold ranges are descriptive rather than confidence intervals.

**Analysis 1 — subspace similarity.** Compute principal angles and overlap for temporal scores and spatial patterns separately. Overlap is the sum of squared principal-angle cosines divided by the requested k; missing numerical dimensions count as zero. Values range from 0 (orthogonal) to 1 (same full-rank subspace). This permits component rotations/mixing and does not require equal component order. Test overlap describes the two held-out representations directly.

**Analysis 2 — transformations.** Center each representation and divide by one scalar RMS using training statistics. Fit identity, orthogonal rotation/reflection, affine and quadratic maps in both directions. Identity uses no learned map; orthogonal preserves distances; affine allows general linear deformation; quadratic adds squared terms and interactions. Ridge is fixed in advance (`--ridge-alpha`, default 0.01), not chosen on test data. All four complexities are reported. Test NRMSE is prediction error norm divided by the target's distance from its training-mean baseline; Q² = 1 − NRMSE². NRMSE below 1 / Q² above 0 beats that baseline. Improvements with complexity must be assessed on test folds; rotations alone need not have physiological meaning.

**Analysis 3 — exploratory spatial grouping.** Fit K-means separately in each modality for every requested cluster count (default 2–6), using all k spatial-pattern coordinates. Report every count without selecting a winner. Training centroids assign held-out locations. Cross-modal adjusted Rand index compares group membership (1 identical, approximately 0 chance-level); silhouette describes separation/compactness.

**Outputs.** `overlap.csv`, `alignment.csv`, `clusters.csv`, `splits.csv`, within-modality comparison tables, and saved preprocessing, model scores/weights, patterns, transformations, centroids and mapping audits. Every result table separates all-data description from cross-validation with an `analysis` label. All-data artifacts use repeat -1 (`-01` in filenames); fold artifacts use 000–004. `COMPLETE.json` marks successful completion. New runs no longer produce `reliability.csv` or `cluster_selection.csv`; the reader retains older-format support.

**Plots.** Dashed lines show the all-data descriptive values. Solid lines show test-fold medians, with minimum–maximum shading. Figures cover overlap versus k, transformation error by complexity/direction for a selected k, and spatial ARI versus predeclared cluster count. The notebook separately displays full-data tables before test results.


### 4.1. Unpaired spatial distributions

`spatial_clouds.py` complements mapped-row comparisons with all-native-feature forward-pattern distributions. Temporal overlap uses scores only and does not depend on spatial matching. Spatial overlap, transformations and ARI still use mapped rows; these are different estimands.

Training temporal scores determine an orthogonal Procrustes alignment. Each modality’s spatial patterns are centered and RMS-normalized using training statistics. Training K-means prototypes (default 32) compress the spatial distribution; prototype mass equals the fraction of native features assigned to it. Training memberships aggregate corresponding test features without refitting.

The saved `cloud_metrics.csv` reports Wasserstein W2, Gromov–Wasserstein squared internal-distance loss and Gaussian MMD². W2 compares aligned locations in component space; GW compares internal geometry; MMD compares kernel distributions using a training-derived bandwidth. All are feature-weighted compressed approximations, not participant-balanced estimates. The GW solver uses two local starts with convergence diagnostics, not a global-optimality guarantee. Shared-region comparison is deferred in `compare_subspace`; its regional output is no longer part of the current pipeline.

Full-data k=3 cluster displays separately select the highest silhouette within each modality and the highest cross-modal ARI at a common candidate count; ties favor fewer clusters. These selections are descriptive and do not alter cross-validation. `full_data_clusters_3d/` contains plots and `display_selections.csv`. Cluster colors are matched in the ARI view, but component axes are not thereby aligned.

## 5. Modality-specific spatial organization

**Question.** How are the native projection weights distributed anatomically within each modality, even when temporal latent structure is similar?

`modality_specificity.ipynb` uses a full-data, equal-condition-average fit of separate PCA, joint PCA or PLSSVD. Current notebook selectors are separate PCA, `paired_coverage`, five components and no whole-modality scaling. Outputs are saved to `out/modality_specificity/{MEG_KIND}/{MODEL}`. These analyses use **projection weights**, not forward patterns, and do not constitute held-out model validation.

### 5.1. Weight magnitudes, coverage and brain displays

For each modality and component, the descriptive threshold is

\[
\tau_c=\operatorname{mean}_i|w_{ic}|+\operatorname{SD}_i|w_{ic}|,
\]

using population SD. Coefficients below threshold are zeroed for display; model fitting is unchanged. Raw magnitudes depend on model normalization, feature count and preprocessing and cannot establish that one modality has greater physiological activity.

Brain markers average retained magnitudes, including zeros, at co-located MEG features and within iEEG complete-linkage spatial groups. The default iEEG radius is estimated from native MEG nearest-neighbor grid spacing. Grouping is for visualization, not model fitting. The Glasser/HCP display uses Gaussian interpolation followed by surface projection; it is a continuous surface map, not a Glasser parcel average, and requires the corresponding mesh resources.

Regional counts and mean absolute weights use the same coordinate-based atlas labeling for both modalities. Every native feature contributes once, including co-located features from different participants. Coverage counts should accompany regional weight plots. Missing regions remain distinct from observed zero values; neither unequal coverage nor participant sampling is corrected by regional averaging. This regional description exists in the modality-specificity notebook even though regional cross-modal comparisons are deferred in `compare_subspace`.

### 5.2. Weight-profile similarity versus anatomical distance

For signed, unthresholded first-k weight profiles w_i at MNI coordinates r_i, compute

\[
d_{ij}=\|r_i-r_j\|_2,\qquad
D_{ij}=1-\frac{w_i^\top w_j}{\|w_i\|_2\|w_j\|_2}.
\]

Mean cosine distance is reported in anatomical-distance bins for all pooled pairs and for each participant separately. All distinct unordered pairs are included, including co-locations; zero-norm profiles are excluded and audited. The pooled curve includes between-participant pairs and is pair-weighted, not participant-balanced.

The null shuffles whole weight profiles within each participant against fixed coordinates, retaining that participant’s empirical weight distribution. The implementation uses 199 permutations by default, pointwise 2.5–97.5% null envelopes and two-sided Monte Carlo tail probabilities with a +1 correction. Holm correction spans populated group/subject bins within each modality. These envelopes are not confidence intervals. Valid inference requires within-participant spatial exchangeability; source smoothing, anatomical heterogeneity and nonuniform coverage can violate it. A significant result indicates departure from random spatial assignment under this null, not unique neural specificity.

Exact blockwise computation limits pairwise working memory, but runtime remains quadratic in channel count and linear in permutation count. `spatial_distance/{modality}` saves curves, null curves, channel audits, configuration and figures.

### 5.3. Component-wise normalized semivariograms

For component c and distance bin B, the estimator is

\[
\gamma_c(B)=\frac{1}{|B|}\sum_{(i,j)\in B}
\frac{(w_{ic}-w_{jc})^2}{2s_c^2}.
\]

The population variance s_c² is computed over the relevant group or participant. Low values indicate nearby coefficients with similar signed values; a global component sign reversal leaves the statistic unchanged. Constant components yield undefined normalized values. The same within-participant coordinate null is used; Holm correction spans all populated bins, components and group/subject scopes within each modality. With 199 permutations, adjusted significance has limited resolution. Results and configuration are saved under `component_spatial/{modality}`.

### 5.4. Clusters and centroid definitions

Two distinct methods are available and should not be conflated:

| Method | Inputs and grouping | Centroid | Scope |
|---|---|---|---|
| `weight_clusters` helper | Above-threshold coefficients, separated by component, sign and hemisphere; connected components within an MNI radius (default 10 mm), minimum three feature rows | MNI mean weighted by absolute coefficient; nearest recorded feature also retained | Available anatomical grouping helper; Euclidean adjacency is not cortical adjacency |
| `kmeans_weight_space` notebook analysis | First three signed component weights; Euclidean K-means with 20 seeded initializations, default five clusters; optional thresholding | Fitted weight-space mean; separate unweighted MNI mean for display | Exploratory full-data grouping; coordinates do not enter K-means |

The current notebook disables K-means thresholding, although the helper defaults to enabling it. All-zero selected profiles receive label −1 and are excluded from fitting. Cluster labels are arbitrary and are not matched across modalities. Spatial centroids can lie outside sampled tissue when membership is dispersed; RMS radius and participant contributions should accompany them. Similar within-cluster profiles are expected by construction and are not independent validation. Neither a fixed count of five nor a visually compact result establishes an optimal biological partition.

`weight_space_kmeans/{original|thresholded}/{modality}` stores cluster summaries, memberships, thresholds, distance curves, configuration and figures. Regional polar charts then summarize absolute selected-input weights within these same clusters. Hemisphere centroids are computed separately for x<0 and x>0 without reclustering; x=0 features are counted separately. Missing hemispheres have undefined centroids. These additional outputs are under `regional_hemisphere_summary`.

## 6. Evidence needed for substantive conclusions

Before writing empirical conclusions, associate each figure/table with its saved configuration, participant manifest, preprocessing, component count and completed iteration IDs. Report baseline and matched surrogate metrics together, including the number of usable iterations. Do not infer population uncertainty from trial-split spread or treat the same component index across independently fitted models as established physiological identity.

The implemented analyses support separate statements about trial generalization, robustness to coverage/composition, temporal correspondence and spatial organization. They do not yet establish generalization to new participants, a calibrated noise ceiling or a known-versus-unknown condition effect. Modality-specific spatial claims additionally require sensitivity checks for coverage, feature density, threshold, cluster count and source smoothing. No numerical biological conclusion is asserted in this review without verified result tables.

## Implementation and reporting gaps to resolve

| Priority | Location | Finding and required action |
|---|---|---|
| High | Coverage/covariance notebook coordinate cells; `plssvd_eval.py: main` | Magnitude-based repeated coordinate division can alter individual axes before the explicit unit checker sees them. Establish the native `GetInfo` unit and apply one documented conversion; verify against known anatomical locations. Do not infer correctness from plausible ranges alone. |
| High | Coverage/covariance loader calls | MEG epoch origin is set from the iEEG epoch origin. Supply independently verified MEG time metadata; matching constructed vectors alone does not verify acquisition alignment. |
| Medium | Across stages | Current primary workflows use equal-condition averages. Older stacked fits may still be present; verify saved settings before comparison. Condition averaging cannot address condition differences. |
| High | Across entry points | PLSSVD batch excludes `SUBJ_0038`; coverage/covariance discovery does not. Record one participant manifest and exclusion rationale. Comparison uses the cache cohort. |
| Implemented | Model comparison | Descriptive and trial-held-out within-modality comparisons now exist; run the updated pipeline before interpreting project results. |
| Medium | Scaling defaults | Covariance/validation use `none`; subspace comparison defaults to `equal_variance`. Record and harmonize this choice, especially for joint PCA. Scalar modality scaling alone does not change ideal PLSSVD directions, but changes joint PCA's balance. MEG channel z-scoring versus unstandardized iEEG also changes what “dominant variance” means. |
| Medium | Correlation reporting | Coverage matrices are Spearman; matching summaries and validation use Pearson. Label both explicitly. A notebook TODO requests Spearman native matching, but it is not implemented; do not describe it as completed. |
| Implemented | Coverage design | Separate actual subject-count (MEG and iEEG) and fixed-cohort pairing analyses are available; run them on the project recordings. |
| Medium | Validation/results | Legacy compare_subspace test-half reliability fixes trained axes; PLSSVD evaluation reports cross-fold refitted temporal-score correlations without test halves. Independent spatial-weight/subspace stability remains separate. Do not average PC1 across repetitions as if component identity were guaranteed. |

### Additional implementation checks

- The phase launcher now distinguishes dataset mismatch from unavailable completed iterations. It accepts per-iteration completion markers without a run-level marker and excludes unfinished iterations; it does not resume partial fits.
- Coordinate overrides in `compare_subspace` preserve feature order and record coordinate sources. The shared GetInfo correction still uses magnitude-dependent divisions before conversion to internal millimeters; an independent anatomical/unit audit remains necessary. Explicit metadata uses its declared units.
- GetInfo callers use the configured project directory directly. Its electrode directory is appended internally; cluster filesystem paths still need validation.
- The PCA-only backfill instructions in `PLSSVD_WORKFLOW.md` do not match the current `plssvd_fit_pca.py`; use the actual implementation as the reference until reconciled.
- Existing synthetic tests cover phase spectral/covariance preservation, completed-iteration reuse, model evaluation, spatial-cloud metrics and modality-specific analyses. Test presence is not evidence that the current recordings satisfy timing, anatomical or exchangeability assumptions; this documentation update does not claim a new full-data validation run.
