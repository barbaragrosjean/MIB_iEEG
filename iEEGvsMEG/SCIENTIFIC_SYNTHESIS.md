# MEG–iEEG comparison: scientific synthesis and evidence audit

**Provisional conclusion:** The available outputs support:
- temporal correspondence between low-dimensional temporal dynamics, accompanied by weak correspondence between mapped spatial patterns. 
- Dataset construction changes the temporal comparison. 
- These observations do not identify whether spatial disagreement arises from measurement operators, preprocessing, anatomical registration, sampling, participant differences, or genuinely different neural recruitment.

**Primary interpretation:** 
- A common stimulus-locked response can be expressed through different measurement and sampling operators. 
- MEG source mixing is a plausible explanation for part of the spatial discrepancy, but is not established as its dominant cause.

## 1. Group and coverage when comparing MEG and iEEG
### 1.1 Anatomical sampling and averaging

- **Hypothesis:** Coverage restriction preserves much of the dominant temporal structure, whereas averaging participants changes the structure being estimated.
- **Analysis done:** 
- Implement dataset construction: full average, full concatenation, coverage average, paired coverage and random control. 
- Metrics include retained-iEEG-score variance captured, temporal subspace overlap, and Hungarian-matched mean absolute Pearson and Spearman correlation.

- **Results:** Embedded output at k = 4:

| MEG construction | Retained iEEG variance captured | Temporal overlap | Matched absolute r |
|---|---:|---:|---:|
| Full concatenation | 0.790 | 0.639 | 0.676 |
| Random control | 0.747 | 0.559 | 0.614 |
| Paired coverage | 0.688 | 0.492 | 0.529 |
| Coverage average | 0.572 | 0.441 | 0.428 |
| Full average | 0.512 | 0.450 | 0.478 |

- **Conclusion:** 
- The concatenated representation corresponds more closely to iEEG than the averaged representations
- Coverage-average versus full-average overlap is similar. However, full-concatenated versus paired-coverage overlap differs by approximately 0.148.
- Correspondence with an iEEG reference is not itself a direct measurement of the difference between two MEG spaces.


**TODO: TO CHECK HOW THE CENTER IS DONE ?**
For centered participant matrices X_s with corresponding feature columns,

\[
X_{cat}X_{cat}^{\top}=\sum_sX_sX_s^{\top},\qquad
\bar X\bar X^{\top}=S^{-2}\sum_{s,r}X_sX_r^{\top}.
\]

These follow by block multiplication and expanding the participant average. Averaging introduces cross-participant terms and can cancel responses with differing signs or timing; concatenation retains each participant's temporal covariance contribution. Consequently, these approaches estimate different objects, rather than one being a neutral substitute for the other.

**TODO We should answer the question how information we loose/gain on MEG data by doing the average and the limited coverage and the subject subsampling.**

### 1.2 Anatomical matching versus random source selection

- **Hypothesis:** Anatomical coverage matching improves temporal correspondence beyond matching participant composition and feature counts.
- **Analysis done:** `coverage_stability.py:run_pairing_stability` fixes participant identities and electrode slots, permutes assignments, and compares paired coverage with a random-source control. Duplicate-source multiplicity is preserved. The stability control fixes a source permutation per participant; this differs from the initial construction's random draws.
- **Results:** At k = 3, the 20 displayed randomized assignments give mean matched r approximately **0.515 for paired coverage** and **0.550 for control**. The mean paired-minus-control difference is **−0.035**, ranging from **−0.193 to +0.063**. 
- **Conclusion:** 
- Anatomical matching does not consistently improve this temporal metric in the displayed sensitivity analysis. 
- This is compatible with broadly distributed or spatially mixed task-locked information.


**TODO : Additional analysis:** 
- Use repeated location samples while fixing participant assignment, 
- examine temporal redundancy versus effective rank, and compare held-out performance after region removal. 
- Match controls on participant, number of unique sources, duplicate multiplicity, depth and spatial dispersion where feasible. 
- An equivalence claim requires a justified equivalence margin and appropriate uncertainty.

### 1.3 Participant-count and channel-count robustness

- **Hypothesis:** Temporal correspondence is robust to cohort composition and recording density.
- **Analysis done:** `coverage_stability.py:run_subject_count_stability` implements nested participant subsets; `elect_number_effect.py` implements nested random full-concatenated MEG feature subsets against a fixed iEEG PCA reference. The channel-count display cell has no saved output; the participant-count section contains no result table.
- **Results:**
- **Conclusion:** Not established by the available outputs. Arbitrary-assignment stability does not establish robustness to changing participant identities.

**TODO RUN**


## 2. Model of covariance, extracting common components
### 2.1 Shared covariance versus dominant within-modality structure

- **Hypothesis:** Cross-modal covariance is substantially expressed within each modality's dominant variance subspace.
- **Analysis done:** `cov_models_utils.py:fit_cov_models` fits separate PCA, joint PCA and exact PLSSVD. `compare_models.py` compares time courses and forward patterns within each modality.
- **Results:** In the paired-coverage notebook at k = 3, PCA/PLSSVD retain respectively **57.1%/55.6%** of iEEG variance and **29.8%/27.4%** of MEG variance. Their captured cross-covariance fractions are **76.7%/88.7%**. The saved within-model figure gives PCA–PLSSVD temporal overlap approximately **0.89 for iEEG and 0.84 for MEG**, and spatial-pattern overlap **0.93 and 0.89**. These figure values are rounded and in sample. At k = 6, native paired PLSSVD mean absolute r is **0.888**, also in sample.
- **Conclusion:** Shared covariance overlaps substantially with dominant within-modality structure. “Especially PLSSVD” is metric dependent: the MEG temporal overlap with joint PCA is approximately 0.93, exceeding its overlap with PLSSVD. Similarity to PCA does not establish identical physiological generators or the same cross-modal spatial covariance direction.

For centered X and Y, PLSSVD decomposes C = XᵀY/(n−1) and selects unit vectors maximizing uᵀCv. This optimizes covariance, not correlation or equality of anatomical maps. Joint PCA maximizes variance of the concatenated block [aX, bY]; the modality scaling a,b matters. 

/!\ - Whole-block scaling does not remove channelwise preprocessing differences.

### 2.2 Reproducibility and overfitting -- Validate Common Components

- **Hypothesis:** A small leading shared subspace generalizes to new trials from the same cohort.
- **Analysis done:** `plssvd_eval_utils.py`, `plssvd_evaluation.py`, and `plssvd_postprocess.py` fit on trial means from training trials and evaluate fixed axes on test-trial means. MEG standardization parameters are estimated on training data. The workflow reports own-modality reconstruction, signed paired correlation, cross-covariance capture and ridge prediction in both directions.

- **Results:** The displayed 100-component evaluation, averaged over 50 iteration summaries, reports test mean paired r **0.092**, cross-covariance energy fraction **0.346**, iEEG/MEG reconstruction **0.246/0.601**, and prediction Q² **−0.301/−0.002**. These do not support generalization of the full 100-component representation. They do not rule out stronger leading components. \

- **Conclusion:** 


## 3. Analysis of the components extracted
## 3.1 Temporal correspondence without spatial-pattern correspondence

- **Hypothesis:** Temporal correspondence can coexist with weak correspondence between anatomical expressions of the latent scores.
- **Analysis done:** `compare_subspace.py:_representations` constructs temporal scores and forward patterns; spatial rows are mapped to electrode/source correspondences. Principal angles, orthogonal/affine/quadratic alignment and clustering assess complementary aspects of agreement.
- **Results:** The **full-concatenated**, k = 3, all-data PLSSVD output has temporal overlap **0.881** and mapped spatial-pattern overlap **0.000800**. Spatial affine Q² is approximately **0.00079** in either direction; quadratic Q² is approximately **0.0102** iEEG→MEG and **0.00292** MEG→iEEG. Cross-modal clustering ARI is approximately **−0.00087 to 0.00256** across 2–6 clusters. These particular values are in sample and are not from a demonstrated validated component set.
- **Conclusion:** This configuration shows a pronounced descriptive dissociation between temporal and spatial correspondence. Failure of these mappings does not prove that different brain regions generate the responses, or that no spatial relationship exists at any resolution.

- **Additional analysis:** 
- Repeat on paired coverage with a separately validated k. 
- Quantify within-modality spatial-pattern reliability first. 
- Compare common parcels at several resolutions, using participant-balanced summaries and bootstrap uncertainty. 
- Include spatially appropriate nulls and check mapping distances. 
- Assess held-out participants through a common parcel or temporal representation, because native feature axes cannot simply be transferred to different electrode layouts.

The code estimates forward patterns as

\[
A=X_c^{\top}T_c(T_c^{\top}T_c)^+.
\]

This is the least-squares solution of X_c ≈ T_c Aᵀ. Setting the derivative of the squared reconstruction loss to zero yields T_cᵀT_c Aᵀ = T_cᵀX_c; the pseudoinverse gives the minimum-norm solution. Test patterns are estimated from test data with training-derived scores: this is a pattern-reproducibility assessment, not a frozen forward-map prediction test. Patterns characterize measured feature–score relationships and do not undo MEG source leakage. Projection weights and activation patterns must be distinguished; see [Haufe et al., 2014](https://pubmed.ncbi.nlm.nih.gov/24239590/).

### 3.2 Spatial concentration and smoothness

- **Hypothesis:** iEEG and MEG latent representations differ in spatial concentration and spatial autocorrelation.
- **Analysis done:** `modality_specificity.py` examines native projection weights, absolute-weight thresholds, regional means, signed-profile cosine distances, normalized component semivariograms, and exploratory clustering. Thresholds are descriptive mean(|w|) + SD(|w|), not significance thresholds.
- **Results:** Historical separate-PCA distance plots show short-distance cosine dissimilarity around **0.769 for MEG versus 0.981 for iEEG** at 0–10 mm; at 10–20 mm these are **0.890 versus 0.996**. Lower dissimilarity indicates more similar nearby profiles. The earlier, currently joint-PCA-configured threshold table retains **8.5–12.7%** of iEEG coefficients and **14.7–16.7%** of MEG coefficients across five components. These observations come from different notebook states and must not be combined into a single model-specific result.
- **Conclusion:** The saved curves are consistent with greater local smoothness of MEG projection profiles.

- **Additional analysis:** Regenerate weights and forward patterns under one configuration. Compare within-participant distance curves, then summarize participants equally. Remove or collapse repeated copies of the same MEG source within participant for sensitivity analysis; duplicates can create zero-distance identical profiles. Match distance distributions and trial counts. Repeat with defensible harmonized channel scaling, because MEG is channel-z-scored whereas iEEG is only multiplied by 1000. For concentration, supplement arbitrary threshold counts with normalized energy profiles, participation ratio or Gini measures.

- **Inference limitation:** The present group curves include cross-participant pairs and are dominated by participants contributing many channels/pairs. Separate within-modality shuffle significance is not a direct test of a modality difference. With 199 permutations, the implemented two-sided minimum p-value is 2/200 = 0.01; Holm correction over hundreds of tests cannot yield any rejection. Use a small prespecified participant-level smoothness statistic or a suitable omnibus/max-statistic procedure, and sufficient permutations. Exchangeability must be justified.

- **Clustering limitation:** Weight-space clusters are fitted separately and IDs are not cross-modally aligned. Low within-cluster cosine distance is partly imposed by clustering; it is not independent evidence. 

## CCL of evidences and take home message:
- how grouping affect and subsampling subject to mimick iEEG network analysis subject change and actually fits
- Cov model and strong common dynamics 
- despite a strong temporal correspondance different spatial activation
- intrinsic difference between modality.
- possible further instrisic difference: 
    - Type of signal
    - Type of referencing: Uniform referencing method across participants does not make pair spacing, orientation or tissue placement uniform

## Addendum: confirmed bipolar referencing and MEG reconstruction method

MEG reconstruction method and confirms bipolar iEEG referencing, consistent across participants. The paper describes a single-shell forward model, beamforming, an 8-mm grid with 3,559 voxels, and sensor covariance estimated from concatenated trials across conditions. Its displayed inverse-filter equation has the linearly constrained minimum-variance form. These are paper-level details; exact reconstruction settings and implementation in this dataset have not been independently audited. The grid spacing is not a measured spatial resolution.

- **Bipolar mechanism:** For contacts at r₁ and r₂, b(t) = v(r₁,t) − v(r₂,t). If both contain the same component a·s(t), that component cancels exactly. Under a locally smooth-field approximation and small separation d = r₂ − r₁, Taylor expansion gives b(t) ≈ −dᵀ∇v(r₁,t). Thus sensitivity depends on spatial gradient, pair direction and separation, not simply the amplitude at a point. The approximation need not hold near abrupt spatial changes; the subtraction identity always holds. Reference-distance dependence is empirically documented in [Caldwell et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12648405/).

- **Additional analysis 2 — reference sensitivity:** 
- Try another referencing 


## TODO: 

### 1.1
**TODO: TO CHECK HOW THE CENTER IS DONE ?**
**TODO We should answer the question how information we loose/gain on MEG data by doing the average and the limited coverage and the subject subsampling.**
### 1.2
**TODO : Additional analysis:** 
- Use repeated location samples while fixing participant assignment, 
- examine temporal redundancy versus effective rank, and compare held-out performance after region removal. 
- Match controls on participant, number of unique sources, duplicate multiplicity, depth and spatial dispersion where feasible. 
- An equivalence claim requires a justified equivalence margin and appropriate uncertainty.
### 1.3
**TODO RUN**

### 2.2
**TODO: rerun teh stat using the other null models to validate the compoents and their number**

### 3.1
- **Additional analysis:** 
- Repeat on paired coverage with a separately validated k. 
- Quantify within-modality spatial-pattern reliability first. 
- Compare common parcels at several resolutions, using participant-balanced summaries and bootstrap uncertainty. 
- Include spatially appropriate nulls and check mapping distances. 
- Assess held-out participants through a common parcel or temporal representation, because native feature axes cannot simply be transferred to different electrode layouts.

### 3.2
- **Additional analysis 1 — pair metadata and sign sensitivity:** 
- Preserve both contact coordinates, ordered contact IDs, midpoint, direction, spacing and tissue/region labels. Flag pairs crossing regions; a midpoint is a plotting convention, not a point-source measurement. 
- Recalculate spatial comparisons using sign-invariant profile similarity (e.g., absolute cosine) alongside signed metrics and magnitude patterns. 
- These address pattern shape irrespective of polarity and sacrifice sign information; do not select sign flips to maximize cross-modal agreement. 

- **Additional analysis:** Regenerate weights and forward patterns under one configuration. Compare within-participant distance curves, then summarize participants equally. Remove or collapse repeated copies of the same MEG source within participant for sensitivity analysis; duplicates can create zero-distance identical profiles. Match distance distributions and trial counts. Repeat with defensible harmonized channel scaling, because MEG is channel-z-scored whereas iEEG is only multiplied by 1000. For concentration, supplement arbitrary threshold counts with normalized energy profiles, participation ratio or Gini measures.

try the treshold using the log of iEEG weights
