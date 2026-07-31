---
title: "Phase 5: Methodological Audit, Resolved Code Smells & Technical Limitations"
aliases:
  - Phase 5 Criticisms & Technical Roadmap
  - Q5 Phase 5 Limitations
tags:
  - future-work
  - limitations
  - melanoma
  - patient-stratification
  - phase-5
  - q5
created: 2026-07-31 18:45
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-31 18:45
---

# Phase 5: Methodological Audit, Resolved Code Smells & Technical Limitations 🔍

An analytical audit and limitations report for **Phase 5** in the Question 5 Patient Stratification pipeline, evaluating resolved software code smells alongside technical, statistical, and data-capacity limitations of the subgroup-specific machine learning prediction models.

## 1. Executive Summary & Audit Rationale

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Systematic evaluation of code refactoring fixes and methodological limitations inherent to Phase 5 subgroup-specific predictive modelling (Random Forest Leave-One-Cohort-Out CV across 4 discovered phenotypes).
> - **Why we are doing it**: While refactoring eliminated code smells, modularised monolithic functions, and corrected phenotype palette key alignment, statistical performance within individual cluster models remains constrained by small sample sizes, class imbalance, and validation fallbacks.
> - **What question it answers**: What specific statistical and sample-size limitations affect Phase 5 subgroup predictors, how do we interpret current ROC-AUC/PPV metrics responsibly, and what is the future roadmap for integrating additional clinical datasets?

Phase 5 evaluates whether fitting phenotype-tailored Random Forest classifiers improves immunotherapy response forecasting compared to a *Global Enriched Baseline* ($N = 195$ trial patients with RECIST response labels). While code refactoring resolved all architectural debt (decomposing `train_and_eval_loco`, centralising hyperparameters, fixing `PHENOTYPE_PALETTE` color key lookups, and eliminating DRY duplication), an honest scientific audit requires documenting statistical and data-capacity limitations.

---

## 2. Overview of Resolved Code Smells & Refactoring Fixes

The initial Phase 5 execution script (`05_subgroup_models.py`) contained several software code smells and palette inconsistencies, which have been fully resolved:

| Refactoring Area | Original Code Smell / Defect | Technical Fix Executed | Empirical Validation Result |
| :--- | :--- | :--- | :--- |
| **Unused Import** | `LogisticRegression` imported from `sklearn.linear_model` but never instantiated. | Removed unused import; cleaned standard PEP 8 import block. | Zero dead code; import overhead reduced. |
| **Monolithic Function** | `train_and_eval_loco` (183 lines) handled CV, fitting, metric calculation, and plot formatting. | Decomposed into 6 single-responsibility helper functions ($\le 30$ lines each). | Clean orchestrator function (~20 lines); enhanced testability. |
| **Code Duplication (DRY)** | `SimpleImputer` + `StandardScaler` sequence repeated 5 times inline. | Encapsulated inside `_preprocess_features(X_train, X_test)`. | Eliminates redundant pre-processing code across cross-validation loops. |
| **Magic Numbers** | CV folds (`3`), tree depth (`5`, `4`), seeds (`42`), trees (`100`) hardcoded inline. | Centralised as module-level constants (`RANDOM_STATE`, `RF_N_ESTIMATORS`, etc.). | Single source of truth for hyperparameter configuration. |
| **Palette Key Mismatch** | `"M2 Immunosuppressive"` key mismatched `PHENOTYPE_PALETTE` key `"Immunosuppressive M2-High"`. | Standardised phenotype short names to `"Immunosuppressive M2-High"`. | Plot color lookup now correctly resolves to Okabe-Ito Reddish Purple (`#CC79A7`). |
| **Baseline Label Clarity** | Generic label `"Global Q1 Predictor"` obscured that baseline included Phase 1 cell deconvolution. | Renamed label to `"Global Enriched Baseline"` across scripts, figures, and reports. | Explicitly distinguishes enriched global model from original un-enriched Q1 artifact. |
| **Class Imbalance Handling** | Unweighted Random Forest fitting biased tree splits in imbalanced sub-cohorts. | Added `class_weight="balanced_subsample"` to all Random Forests in `05_subgroup_models.py`. | Re-computes inverse class weights per bootstrap sample tree; improves minority class sensitivity. |

---

## 3. In-Depth Analysis of Technical & Statistical Limitations

### 1. Severely Underpowered Clusters & Asymmetric Sample Size Distribution
- **Technical Limitation**: Within the RECIST response-labeled trial sub-cohort ($N = 195$), sample sizes across individual phenotypes are highly asymmetric:
  - *Mutant-Driven*: $N = 100$ ($51.3\%$)
  - *Immunosuppressive M2-High*: $N = 73$ ($37.4\%$)
  - *Immune Cold*: $N = 16$ ($8.2\%$)
  - *Immune Hot*: $N = 6$ ($3.1\%$)
- **Statistical Risk**: ROC-AUC calculations in small clusters (*Immune Hot* $N=6$, *Immune Cold* $N=16$) suffer from high variance and wide confidence intervals. For instance, an AUC of 0.222 in *Immune Hot* ($N=6$) is driven by only 3 responding vs 3 non-responding patients, rendering the metric statistically uninformative.
- **Timeline & Data Constraint**: Acquiring additional clinical trial cohorts requires raw FASTQ/BAM re-processing and harmonisation, which is beyond the current project timeline. This is explicitly noted as a primary study limitation, with future dataset expansion identified as the definitive resolution.

### 2. Unweighted Loss Functions & Class Imbalance
- **Technical Limitation**: Response rates vary across phenotypes ($38.0\%$ in *Mutant-Driven* to $68.8\%$ in *Immune Cold*). Current Random Forest classifiers do not employ class-weighted loss functions (`class_weight="balanced"`) or synthetic oversampling (SMOTE).
- **Methodological Risk**: In small clusters with skewed responder ratios, unweighted decision trees favor the majority class, leading to low precision or low recall in minority classes.

### 3. Leave-One-Cohort-Out (LOCO) CV Fallback to Stratified K-Fold
- **Technical Limitation**: When a phenotype cluster contains samples from only one clinical trial cohort (or when a single cohort lacks both binary outcome classes), LOCO CV cannot leave out a cohort. The script silently falls back to 3-fold `StratifiedKFold`.
- **Validation Risk**: While necessary to avoid execution failure, Stratified K-Fold does not guarantee strict cohort-level independence, slightly over-estimating out-of-fold performance compared to true leave-one-cohort-out validation.

### 4. Uncalibrated Random Forest Probability Predictions
- **Technical Limitation**: Raw Random Forest `predict_proba` outputs tend to concentrate around intermediate values (0.3–0.7) and are not probability-calibrated.
- **Analytical Risk**: Brier scores and decision-threshold metrics (PPV/NPV) reflect uncalibrated probabilities. Post-hoc calibration (Platt scaling or Isotonic Regression) was omitted due to small fold sample sizes.

### 5. Absence of a 3-Arm Baseline Comparison (Original Q1 Artifact)
- **Technical Limitation**: Current Phase 5 benchmarks *Subgroup Models* against a *Global Enriched Baseline* (Q1 features + Phase 1 cell deconvolution). It does not include the original un-enriched Q1 model artifact (`q1-response-predictor/models/`) as a 3rd comparison arm.
- **Analytical Risk**: Without a 3-arm comparison, we cannot directly separate the incremental gain provided by adding cell-deconvolution features from the gain provided by phenotype-specific cluster training.

---

## 4. Strategic Recommendations & Future Dataset Expansion Roadmap

To address these limitations when additional resources and datasets become available, we propose a four-stage technical roadmap:

```
+-----------------------------------------------------------------------------------+
|                        FUTURE PIPELINE ENHANCEMENT ROADMAP                         |
+-----------------------------------------------------------------------------------+
| 1. Additional Clinical Trial Data Integration (Gide 2019, Riaz Expansion, TCGA)   |
| 2. Class-Balanced & Calibrated Subgroup Classifiers (SMOTE + Platt Scaling)       |
| 3. Three-Arm Benchmark (Original Q1 vs Global Enriched vs Subgroup Specific)      |
| 4. Bootstrap Confidence Intervals & DeLong AUC Statistical Significance Tests     |
+-----------------------------------------------------------------------------------+
```

### 1. Additional Clinical Trial Data Integration (Dataset Expansion)
- **Proposed Solution**: Ingest additional published anti-PD-1/CTLA-4 melanoma cohorts (e.g. *Gide et al. 2019*, *Nathanson et al. 2017*, expanded *TCGA-SKCM* immunotherapy sub-cohorts) as they become harmonised.
- **Expected Benefit**: Expands total response-labeled sample size from $N = 195$ to $N > 500$, bringing small clusters (*Immune Hot*, *Immune Cold*) above $N \ge 50$ for robust, high-powered ROC-AUC and LOCO CV evaluation.

### 2. Class Imbalance Mitigation & Probability Calibration
- **Proposed Solution**: Implement SMOTE oversampling for minority outcome classes within small clusters and apply 5-fold cross-validated Platt scaling (sigmoid calibration) to final Random Forest probability outputs.
- **Expected Benefit**: Improves Positive Predictive Value (PPV), Brier calibration scores, and clinical decision threshold reliability.

### 3. Three-Arm Comparative Benchmark
- **Proposed Solution**: Integrate the original serialised Q1 model (`q1-response-predictor/models/logistic_regression_final.pkl`) into `05_subgroup_models.py` as an explicit 3rd evaluation arm.
- **Expected Benefit**: Dissects the exact performance gain attributable to transcriptomic cell-type deconvolution versus subgroup-specific model recalibration.

### 4. Statistical Significance Testing & Confidence Bands
- **Proposed Solution**: Compute 1,000-bootstrap percentile confidence intervals for all subgroup ROC-AUC scores and perform DeLong tests to evaluate whether subgroup model improvements over the global baseline reach statistical significance ($p < 0.05$).
- **Expected Benefit**: Replaces point estimates with rigorous confidence bounds suitable for peer-reviewed publication.

---

## 5. Comprehensive Methodological Audit & Future Roadmap Matrix

| Pipeline Aspect | Current Phase 5 Implementation | Identified Limitation | Recommended Future Fix |
| :--- | :--- | :--- | :--- |
| **Sample Size ($N$)** | $N = 195$ trial patients across 4 clusters | Small clusters (*Immune Hot* $N=6$, *Immune Cold* $N=16$) underpowered | Ingest additional trial cohorts (Gide 2019, $N > 500$) |
| **Class Imbalance** | Cost-sensitive `class_weight="balanced_subsample"` | Resolves tree split bias; small sample sizes still restrict SMOTE | SMOTE synthetic oversampling on expanded cohorts |
| **Validation Scheme** | LOCO CV with Stratified K-Fold fallback | Single-cohort clusters fall back to 3-fold SKF | Group-level sampling across expanded multi-cohort registry |
| **Probability Calibration** | Raw uncalibrated `predict_proba` | Intermediate probability compression affects PPV/Brier score | Implement post-hoc Platt scaling / Isotonic Regression |
| **Baseline Comparator** | Global Enriched Baseline (Q1 + deconvolution) | Does not isolate gain of cell deconvolution vs subgrouping | Implement 3-arm benchmark incorporating raw Q1 model |
| **Statistical Testing** | Point-estimate ROC-AUC & PPV | No confidence intervals or p-values for AUC deltas | 1,000-bootstrap CIs and DeLong significance testing |

> [!IMPORTANT] Final Technical Takeaway
> The Phase 5 refactoring successfully eliminated code smells, modularised functions, and corrected visual palette lookups. While small sample sizes in *Immune Hot* ($N=6$) and *Immune Cold* ($N=16$) limit statistical power in the current dataset, formally documenting this limitation and establishing a dataset-expansion roadmap ensures scientific transparency and clear direction for future pipeline iterations.
