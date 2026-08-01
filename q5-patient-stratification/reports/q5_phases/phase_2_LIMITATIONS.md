---
title: "Phase 2: Methodological Limitations, Biological Criticisms & Future Iterations"
aliases:
  - Phase 2 Criticisms & Future Roadmap
  - Q5 Phase 2 Limitations
tags:
  - future-work
  - limitations
  - melanoma
  - patient-stratification
  - phase-2
  - q5
created: 2026-07-31 12:21
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-31 12:21
---

# Phase 2: Methodological Limitations, Biological Criticisms & Future Iterations 🔍

An analytical audit of **Phase 2** in the Question 5 Patient Stratification pipeline, detailing methodological constraints, biological criticisms, and actionable technical solutions for future iterations.

## 1. Executive Summary & Audit Rationale

While Phase 2 successfully establishes univariate feature rankings, optimal Youden decision thresholds, and genomic $\times$ immune interaction effect sizes, several statistical and biological assumptions constrain clinical translation. This audit identifies key limitations in threshold discretization, statistical multiple testing power, and spatial B-cell structure, outlining a technical roadmap for future iterations.

## 2. In-Depth Analysis of Limitations & Criticisms

### 1. Methodological Limitation: Information Loss from Step-Function Threshold Discretisation
- **Current Approach**: Youden's $J$ statistic calculates single rigid numerical thresholds (such as `B_cells` $\ge 0.430$ or `TIS` $\ge 0.191$) to classify continuous score distributions into binary High vs Low risk categories.
- **Scientific Criticism**: Converting continuous biological variables into step-function binary categories discards quantitative variance. Patients scoring $0.429$ and $0.010$ are treated identically as "Low", despite $0.429$ lying immediately adjacent to the $0.430$ cutoff threshold.
- **Clinical Impact**: Rigid step-functions misclassify borderline patients and fail to reflect the smooth, continuous gradient of immune activation observed in real-world human tumours.

### 2. Statistical Limitation: Uncorrected Multiple Testing Across Permutation Matrices
- **Current Approach**: Logistic regression interaction models test 21 driver mutation $\times$ microenvironment signature pairs independently, reporting raw unadjusted $p$-values.
- **Scientific Criticism**: Evaluating 21 simultaneous statistical tests without Benjamini-Hochberg False Discovery Rate (FDR) correction increases the probability of Type I errors (false positives). While `BRAF` $\times$ `TIS` reaches $p = 0.040$, borderline interactions ($p \approx 0.06 - 0.09$) require strict multi-testing control to confirm true biological validity.
- **Statistical Impact**: Risks over-interpreting exploratory secondary interactions in smaller subgroup cohorts.

### 3. Biological Limitation: Bulk RNA-Seq Blindness to Tertiary Lymphoid Structures (TLS)
- **Current Approach**: Deconvolution identifies `B_cells` as the top individual standalone predictor ($\text{AUC} = 0.632$).
- **Scientific Criticism**: Bulk transcriptomics measures total B-cell RNA abundance, but cannot determine whether B cells are organised into mature **Tertiary Lymphoid Structures (TLS)** with germinal centers and plasma cell antibody production, versus unorganized, non-functional naive B-cell infiltrates.
- **Biological Impact**: Tertiary Lymphoid Structures (TLS) are the true functional drivers of anti-PD-1 response. Two tumours with identical bulk `B_cells` transcript levels may exhibit completely opposite clinical outcomes depending on spatial TLS maturation.

### 4. Sampling Limitation: Driver Subgroup Sample Size Imbalance
- **Current Approach**: Interaction testing is conducted across available driver mutation categories (`mut_BRAF`, `mut_NRAS`, `mut_NF1`).
- **Scientific Criticism**: Sample size varies substantially across genomic driver subgroups (`BRAF` $N = 130$, `NRAS` $N = 78$, `NF1` $N = 34$).
- **Data Impact**: Lower statistical power in `NF1` and `NRAS` cohorts makes it difficult to detect subtle interaction effect sizes ($\beta_{\text{interaction}}$), potentially creating an artificial appearance of `BRAF` dominance due to subgroup sample size advantages.

### 5. Clinical Limitation: Unmeasured Prior Targeted Therapy Exposure
- **Current Approach**: Baseline pre-treatment biopsies from clinical trials (_Liu 2019_, _Riaz 2017_, _Hugo 2016_) are evaluated without detailed records of prior targeted therapy duration.
- **Scientific Criticism**: Prior treatment with BRAF/MEK inhibitors (such as Dabrafenib + Trametinib) induces profound, transient changes in microenvironmental immune infiltration and antigen presentation.
- **Clinical Impact**: Unmeasured prior targeted therapy exposure introduces unobserved confounding variables into pre-treatment biomarker evaluations.

> [!WARNING] Summary of Impact on Downstream Phases  
> Discretisation information loss and uncorrected multiple testing can distort feature selection inputs for Phase 5 subgroup predictive models and Phase 7 decision trees. Soft continuous scoring and spatial TLS validation are required for robust clinical translation.

## 3. Proposed Fixes & Strategic Roadmap for Future Project Iterations

To address these limitations in future iterations of Question 5, we propose five concrete technical enhancements:

### Proposed Fix 1: Transition to Generalized Additive Models (GAMs) & Cubic Splines
- **Technical Solution**: Replace binary Youden step-functions with Generalized Additive Models (GAMs) and restricted cubic splines.
- **Expected Benefit**: Models continuous, non-linear risk gradients while retaining threshold inflection points for clinical reporting, eliminating arbitrary boundary misclassification.

### Proposed Fix 2: Benjamini-Hochberg FDR Control & Hierarchical Bayesian Interaction Shrinkage
- **Technical Solution**: Apply Benjamini-Hochberg False Discovery Rate (FDR $q < 0.05$) correction across all 21 interaction pairs and fit Hierarchical Bayesian shrinkage models.
- **Expected Benefit**: Prevents Type I false positive errors and yields stable interaction effect size estimates across small driver subgroups.

### Proposed Fix 3: Spatial TLS Characterisation via H&E Deep Learning & Multiplex Imaging
- **Technical Solution**: Combine bulk transcriptomics with H&E histology slide deep learning (QuPath / StarDist) or 7-colour multiplex immunofluorescence (mIF).
- **Expected Benefit**: Formally quantifies **TLS Density and Maturation State** (CD20+ B cells co-localised with CD21+ follicular dendritic cells and CD8+ T cells), distinguishing functional TLS from unorganised infiltrates.

### Proposed Fix 4: Power-Weighted Interaction Testing & Subgroup Resampling
- **Technical Solution**: Implement balanced bootstrap resampling and power-weighted regression adjustments for non-`BRAF` driver cohorts.
- **Expected Benefit**: Ensures equal statistical sensitivity for detecting genomic interactions in `NF1` and `NRAS` melanomas.

### Proposed Fix 5: Prior Therapy Annotation & Washout Tracking
- **Technical Solution**: Annotate patient metadata with prior systemic therapy exposure, targeted therapy duration, and washout intervals.
- **Expected Benefit**: Adjusts statistical interaction models for prior treatment history, eliminating confounding effects from prior BRAF/MEK inhibition.

## 4. Comprehensive Roadmap Comparison Matrix

| Limitation Category | Current Phase 2 Method | Proposed Technical Fix | Primary Tool / Platform | Expected Clinical & Analytical Benefit |
| :--- | :--- | :--- | :--- | :--- |
| **Score Discretisation** | Binary Youden step-function cutoffs | GAMs & restricted cubic splines | `pyGAM` / `statsmodels` | Preserves continuous risk gradients; avoids boundary misclassification. |
| **Multiple Testing** | Unadjusted univariate $p$-values | Benjamini-Hochberg FDR ($q < 0.05$) & Bayesian shrinkage | `statsmodels.stats.multitest` / `PyMC` | Eliminates false positive interaction terms across 21 test pairs. |
| **B-Cell Functionality** | Bulk transcriptomic `B_cells` deconvolution | Spatial TLS quantification via H&E / mIF | QuPath / 10x Xenium / mIF | Distinguishes functional Germinal Centre TLS from naive infiltrates. |
| **Driver Imbalance** | Unweighted interaction regression | Power-weighted regression & bootstrap resampling | Custom Resampling Pipelines | Equalises statistical power across `BRAF`, `NRAS`, and `NF1` subgroups. |
| **Treatment History** | Pre-treatment biopsy snapshot | Prior therapy & washout annotation | Clinical Metadata Curation | Eliminates confounding from prior BRAF/MEK inhibitor exposure. |

> [!INSIGHT] Key Takeaways & Strategic Future Roadmap
> - **Continuous Modelling**: Moving from rigid Youden binary step-functions to continuous GAM splines preserves quantitative biological information.
> - **Multi-Testing Rigour**: Implementing Benjamini-Hochberg FDR control ensures that only true biological interactions guide downstream clinical decision trees.
> - **Spatial Structural Validation**: Combining bulk B-cell scores with spatial TLS image quantification resolves the functional state of humoral microenvironmental immunity.
