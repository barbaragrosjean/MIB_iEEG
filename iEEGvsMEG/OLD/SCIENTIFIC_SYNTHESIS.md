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

Additionnal: 
- Within the MEG dataset comparison accounting for average and subsampling effect.
- Pair vs random: redondunt info spread over the sources
- channel count in concat and iEEG matching : in favor of informaiton redundancy

-- Taking the paring did'nt remove much info and ensure the comparability between weights

## 2. Model of covariance, extracting common components
### 2.1 Shared covariance versus dominant within-modality structure

- **Hypothesis:** Cross-modal covariance is substantially expressed within each modality's dominant variance subspace.
- **Analysis done:** `cov_models_utils.py:fit_cov_models` fits separate PCA, joint PCA and exact PLSSVD. `compare_models.py` compares time courses and forward patterns within each modality.
- **Results:** In the paired-coverage notebook at k = 3, PCA/PLSSVD retain respectively **57.1%/55.6%** of iEEG variance and **29.8%/27.4%** of MEG variance. Their captured cross-covariance fractions are **76.7%/88.7%**. The saved within-model figure gives PCA–PLSSVD temporal overlap approximately **0.89 for iEEG and 0.84 for MEG**, and spatial-pattern overlap **0.93 and 0.89**. These figure values are rounded and in sample. At k = 6, native paired PLSSVD mean absolute r is **0.888**, also in sample.
- **Conclusion:** Shared covariance overlaps substantially with dominant within-modality structure. “Especially PLSSVD” is metric dependent: the MEG temporal overlap with joint PCA is approximately 0.93, exceeding its overlap with PLSSVD. Similarity to PCA does not establish identical physiological generators or the same cross-modal spatial covariance direction.

For centered X and Y, PLSSVD decomposes C = XᵀY/(n−1) and selects unit vectors maximizing uᵀCv. This optimizes covariance, not correlation or equality of anatomical maps. Joint PCA maximizes variance of the concatenated block [aX, bY]; the modality scaling a,b matters for them. 

- The tree method show nice corresponding temporal pattern for the first components (two first very close), 
- PCA joint doesn't show big similarity of wight but that make sense

-- PLS SVD for the shared part lets go with it 


### 2.2 Reproducibility and overfitting -- Validate Common Components

- **Hypothesis:** A small leading shared subspace generalizes to new trials from the same cohort.
- **Analysis done:** `plssvd_eval_utils.py`, `plssvd_evaluation.py`, and `plssvd_postprocess.py` fit on trial means from training trials and evaluate fixed axes on test-trial means. MEG standardization parameters are estimated on training data. The workflow reports own-modality reconstruction, signed paired correlation, cross-covariance capture and ridge prediction in both directions.

- **Results:** The displayed 100-component evaluation, averaged over 50 iteration summaries, reports test mean paired r **0.092**, cross-covariance energy fraction **0.346**, iEEG/MEG reconstruction **0.246/0.601**, and prediction Q² **−0.301/−0.002**. These do not support generalization of the full 100-component representation. They do not rule out stronger leading components. \

- **Conclusion:** Take the null time point make us think that are is 10 important components. The time block null show that 2 are very very strong


## 3. Analysis of the components extracted
## 3.1 Temporal correspondence without spatial-pattern correspondence

- **Hypothesis:** Temporal correspondence can coexist with weak correspondence between anatomical expressions of the latent scores.
- **Analysis done:** `compare_subspace.py:_representations` constructs temporal scores and forward patterns; spatial rows are mapped to electrode/source correspondences. Principal angles, orthogonal/affine/quadratic alignment and clustering assess complementary aspects of agreement.
- **Results:** The **full-concatenated**, k = 3, all-data PLSSVD output has temporal overlap **0.881** and mapped spatial-pattern overlap **0.000800**. Spatial affine Q² is approximately **0.00079** in either direction; quadratic Q² is approximately **0.0102** iEEG→MEG and **0.00292** MEG→iEEG. Cross-modal clustering ARI is approximately **−0.00087 to 0.00256** across 2–6 clusters. These particular values are in sample and are not from a demonstrated validated component set.
- **Conclusion:** This configuration shows a pronounced descriptive dissociation between temporal and spatial correspondence. Failure of these mappings does not prove that different brain regions generate the responses, or that no spatial relationship exists at any resolution.

### 3.2 Spatial concentration and smoothness

- **Hypothesis:** iEEG and MEG latent representations differ in spatial concentration and spatial autocorrelation.

- histogram of weights for diffferent organisation. **Try to treshold differently**

- clustering with kmean doesn't work even by taking the Ari as a criterion **Try other clustering?**

- abs cosinus as a function fo the distance show difference. **Can we find clusters or centroid with this?**

- spatial patterns with mixing trials are stable but when we change the pairing they are not (in MEG) but still give the same dynamics **How much it is different grouped by ROIs**. No Rois seems to be similars.


## 4. Physiology of the common networks


### 4.1 Decoding


## CCL of evidences and take home message:
- how grouping affect and subsampling subject to mimick iEEG network analysis subject change and actually fits
- Cov model and strong common dynamics 
- despite a strong temporal correspondance different spatial activation
- intrinsic difference between modality.


## Addendum: confirmed bipolar referencing and MEG reconstruction method

MEG reconstruction method and confirms bipolar iEEG referencing, consistent across participants. The paper describes a single-shell forward model, beamforming, an 8-mm grid with 3,559 voxels, and sensor covariance estimated from concatenated trials across conditions. Its displayed inverse-filter equation has the linearly constrained minimum-variance form. These are paper-level details; exact reconstruction settings and implementation in this dataset have not been independently audited. The grid spacing is not a measured spatial resolution.

- **Bipolar mechanism:** For contacts at r₁ and r₂, b(t) = v(r₁,t) − v(r₂,t). If both contain the same component a·s(t), that component cancels exactly. Under a locally smooth-field approximation and small separation d = r₂ − r₁, Taylor expansion gives b(t) ≈ −dᵀ∇v(r₁,t). Thus sensitivity depends on spatial gradient, pair direction and separation, not simply the amplitude at a point. The approximation need not hold near abrupt spatial changes; the subtraction identity always holds. Reference-distance dependence is empirically documented in [Caldwell et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12648405/).

- **Additional analysis 2 — reference sensitivity:** 
- Try another referencing 
