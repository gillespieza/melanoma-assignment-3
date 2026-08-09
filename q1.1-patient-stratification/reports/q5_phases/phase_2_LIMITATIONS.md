---
title: "Phase 2: Methodological Limitations, Statistical Weaknesses & Strategic Roadmap"
aliases:
  - Phase 2 Limitations
  - Q5 Phase 2 Audit
tags:
  - melanoma
  - patient-stratification
  - phase-2
  - limitations
  - q5
created: 2026-08-01 12:41
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-01 12:41
---

# Phase 2: Methodological Limitations, Statistical Weaknesses & Strategic Roadmap

> [!NOTE] Analytical Audit Context & Objectives
> - **What is being evaluated**: Methodological assumptions, statistical power constraints, biological abstractions, and computational limits across Phase 2 (Univariate Associations, Youden Cutoffs, and Genomic Interaction Modelling).
> - **Why**: Critical assessment of standalone biomarker thresholds and interaction models is necessary to establish realistic boundaries for clinical translation and prevent over-interpretation of univariate metrics.
> - **What question it answers**: What statistical, biological, and computational constraints limit the standalone utility of Phase 2 cutoffs, and how can future pipeline iterations resolve them?

## 1. Executive Summary & Audit Overview

Phase 2 evaluates biomarker discriminative power across $N_{\text{ICI}} = 326$ immunotherapy-treated melanoma patients using non-parametric Mann-Whitney U testing, Cohen's $d$ effect sizes, Youden's $J$ optimal decision thresholds, and 21 driver mutation $\times$ immune signature logistic regression interaction terms.

While Phase 2 establishes key empirical benchmarks—such as `B_cells` operating as the top standalone predictor (AUC $= 0.632$, cutoff $= 0.430$) and `BRAF` oncogenic signalling significantly dampening T-cell inflammation ($\text{TIS} \times \text{BRAF } \beta = -0.65, p = 0.040$)—several statistical, biological, and data-structure constraints limit direct clinical application. This audit details these constraints in the current pipeline state and outlines actionable, prioritised technical solutions.

> [!WARNING] Summary of Core Phase 2 Analytical Limitations
> - **Step-Function Information Loss**: Discretising continuous biomarker distributions into binary cutoffs via Youden's $J$ discards continuous biological variance and creates artificial boundary misclassifications.
> - **Uncorrected Multiple Testing**: Evaluating 21 interaction pairs simultaneously without False Discovery Rate (FDR) control risks Type I errors among secondary interaction signals.
> - **Bulk RNA-seq Spatial Blindness**: Bulk transcriptomic `B_cells` scoring cannot determine spatial organisation into Tertiary Lymphoid Structures (TLS).
> - **Subgroup Sample Size Imbalance**: Statistical power is heavily skewed toward `mut_BRAF` ($N = 128$), leaving smaller subgroups (`mut_NF1` $N = 40$, `mut_NRAS` $N = 82$) underpowered for interaction detection.

---

## 2. Code-Quality & Architectural Considerations

Evaluating the current Phase 2 software architecture (`02_feature_analysis.py` and `src/feature_analysis.py`) highlights several design considerations for scalable pipeline development:

1. **Tight Coupling of Execution Script and Output Paths**:
   - `02_feature_analysis.py` hardcodes output CSV export paths (`univariate_feature_associations.csv`, `youden_cutoffs.csv`, `genomic_immune_interactions.csv`) directly to `data/processed/q5/`.
   - *Impact*: Hinders programmatic parameter sweeps or multi-dataset benchmarking without overwriting baseline processed artifacts.

2. **In-Memory Transformation Overhead**:
   - Univariate Mann-Whitney U testing and Youden ROC iterations recreate subset DataFrames in memory per feature loop rather than vectorising operations across feature blocks.
   - *Impact*: Increases execution overhead when expanding the feature panel beyond the core baseline markers.

3. **Fallback Error Handling in Interaction Fits**:
   - When statsmodels `Logit` fails to converge on sparse feature subsets, the exception handler falls back to returning $\beta_{\text{interaction}} = 0.0$ and $p = 1.0$.
   - *Impact*: While preventing pipeline crashes, silent fallback masks convergence issues caused by severe feature collinearity or extreme class imbalance in smaller driver subgroups.

---

## 3. Statistical Weaknesses & Methodological Constraints

> [!WARNING] Methodological & Statistical Constraints
> The following statistical properties constrain the diagnostic precision and generalisability of Phase 2 cutoffs:

### 1. Information Loss from Step-Function Threshold Discretisation
- **Current Approach**: Youden's $J$ statistic calculates single rigid numerical cutoffs (such as `B_cells` $\ge 0.430$ or `TIS` $\ge 0.191$) to classify continuous score distributions into binary High vs Low risk categories.
- **Statistical Criticism**: Converting continuous biological variables into step-function binary categories discards quantitative variance. A patient scoring $0.429$ is classified as Low risk alongside a patient scoring $-1.500$, despite $0.429$ lying immediately adjacent to the $0.430$ cutoff threshold.
- **Impact**: Step-functions misclassify borderline patients and fail to capture the smooth, continuous gradient of immune activation in real-world tumours.

### 2. Uncorrected Multiple Testing Across Interaction Matrices
- **Current Approach**: Logistic regression interaction models test 21 driver mutation $\times$ microenvironment signature pairs independently, reporting raw unadjusted $p$-values.
- **Statistical Criticism**: Testing 21 hypotheses simultaneously without Benjamini-Hochberg False Discovery Rate (FDR) correction inflates the family-wise Type I error rate. While `TIS` $\times$ `BRAF` reaches strict significance ($p = 0.040$), borderline interactions (`B_cells` $\times$ `BRAF` $p = 0.073$, `CD8_T_cells` $\times$ `BRAF` $p = 0.074$, `IFN_gamma` $\times$ `BRAF` $p = 0.087$) require strict multi-testing control to confirm true biological validity.
- **Impact**: Risks over-interpreting exploratory secondary interaction terms in unpowered subgroups.

### 3. Modest Standalone Diagnostic Accuracy (AUC $\approx 0.58 - 0.63$)
- **Current Approach**: Evaluating continuous biomarkers as standalone univariate response predictors.
- **Statistical Criticism**: Even the top individual feature (`B_cells`) achieves an AUC of only $0.632$ (Sensitivity $= 53.7\%$, Specificity $= 77.0\%$), while core T-cell signatures (`TIS` AUC $= 0.585$, `CYT` AUC $= 0.583$, `CD8_T_cells` AUC $= 0.584$) yield modest separation.
- **Impact**: Demonstrates mathematically that single-biomarker triage is insufficient for clinical selection, providing the empirical rationale for multi-dimensional clustering in Phase 3.

---

## 4. Biological Assumptions & Clinical Translation Limits

> [!WARNING] Biological Abstractions & Clinical Translation Boundaries
> Biological simplifications inherent in bulk sequencing and retrospective trial data limit direct clinical translation:

### 1. Bulk RNA-seq Blindness to Tertiary Lymphoid Structures (TLS)
- **Biological Constraint**: Transcriptomic cell deconvolution identifies `B_cells` as the top standalone predictor ($d = 0.373$, AUC $= 0.632$). However, bulk RNA sequencing measures total tissue RNA abundance and cannot determine whether B cells are organized into functional **Tertiary Lymphoid Structures (TLS)** with germinal centres versus unorganised, non-functional naive infiltrates.
- **Clinical Implication**: Mature TLS containing co-localised B cells (`CD20`), T cells (`CD8A`), and follicular dendritic cells (`CD21`) drive anti-PD-1 response. Two tumours with identical bulk `B_cells` scores may exhibit opposite clinical outcomes based on spatial TLS maturation.

### 2. MAPK Signalling Abstraction & `BRAF` Interaction Mechanism
- **Biological Constraint**: The interaction model confirms that `BRAF` mutation dampens T-cell predictive value ($\beta = -0.65, p = 0.040$). However, linear interaction terms abstract the underlying biochemistry—constitutive BRAF V600E signalling suppresses antigen presentation (`B2M`, `HLA-A/B/C`) and secretes immunosuppressive cytokines (`VEGF`, `IL-6`, `IL-10`).
- **Clinical Implication**: A static linear interaction coefficient misses non-linear threshold dynamics where high MAPK flux completely suppresses immune recognition below a critical T-cell infiltration level.

### 3. Unmeasured Prior Systemic Therapy Confounding
- **Biological Constraint**: Pre-treatment biopsies from clinical trials (*Liu 2019*, *Riaz 2017*, *Hugo 2016*) are evaluated without detailed records of prior targeted kinase inhibitor exposure or washout duration.
- **Clinical Implication**: Prior BRAF/MEK inhibition (Dabrafenib/Trametinib) transiently increases T-cell infiltration and HLA expression. Unmeasured prior therapy introduces unobserved confounding into baseline biomarker evaluations.

---

## 5. Computational & Subgroup Data Constraints

> [!WARNING] Data Structure & Subgroup Imbalances
> Subgroup sample size disparities restrict statistical detection power across driver subtypes:

### 1. Driver Subgroup Sample Size Imbalance
- **Data Constraint**: ICI cohort sample sizes vary substantially across driver mutation subgroups:
  - `mut_BRAF`: $N = 128$ ($39.3\%$ of cohort)
  - `mut_NRAS`: $N = 82$ ($25.2\%$ of cohort)
  - `mut_NF1`: $N = 40$ ($12.3\%$ of cohort)
- **Computational Impact**: High sample size in `mut_BRAF` provides strong power to detect interaction terms ($\text{TIS} \times \text{BRAF } \beta = -0.65, p = 0.040$). Conversely, `mut_NF1` displays the largest positive interaction effect size with macrophage polarisation (`M1_M2_Ratio` $\beta = +0.94$), but yields $p = 0.654$ due to statistical underpowering ($N = 40$).

### 2. Missing Co-stimulatory Mappings in `IMPRES` Calculation
- **Data Constraint**: `IMPRES` evaluates 15 pairwise boolean comparisons between co-stimulatory and co-inhibitory genes. Several key co-stimulatory genes (`CD28`, `CD86`, `CD80`, `CD40`, `CD200`) are missing or unmapped in legacy cohort matrices.
- **Computational Impact**: Reduces `IMPRES` calculation to a truncated pair subset, systematically underestimating checkpoint regulation in specific historical trial arms.

---

## 6. Actionable Prioritised Technical Fixes & Strategic Roadmap

To resolve these statistical and biological constraints in future iterations, we propose five concrete technical enhancements:

| Priority | Constraint Category | Proposed Technical Solution | Implementation Tool / Library | Expected Analytical Benefit |
| :--- | :--- | :--- | :--- | :--- |
| **P1** | **Threshold Discretisation** | Replace binary Youden step-functions with Generalized Additive Models (GAMs) and restricted cubic splines. | `pyGAM` / `statsmodels.gam` | Preserves continuous risk gradients; eliminates artificial boundary misclassification. |
| **P2** | **Multiple Testing** | Implement Benjamini-Hochberg FDR control ($q < 0.05$) and Bayesian interaction shrinkage. | `statsmodels.stats.multitest` | Controls false discovery rates across multi-permutation interaction matrices. |
| **P3** | **Spatial B-Cell Function** | Combine transcriptomic deconvolution with spatial TLS image quantification via multiplex immunofluorescence. | QuPath / 10x Xenium / StarDist | Distinguishes functional Germinal Centre TLS from unorganised B-cell infiltrates. |
| **P4** | **Subgroup Imbalance** | Apply balanced bootstrap resampling and power-weighted interaction regression adjustments. | Scikit-learn Resampling / Custom PyTorch | Equalises statistical detection sensitivity across `BRAF`, `NRAS`, and `NF1` cohorts. |
| **P5** | **Prior Therapy Confounding** | Annotate clinical metadata with prior targeted therapy duration and washout intervals. | Pandas Clinical Pipeline | Eliminates unobserved confounding from prior BRAF/MEK inhibitor exposure. |

---

## 7. Key Takeaways

> [!INSIGHT] Key Takeaways: Phase 2 Limitations & Strategic Future Roadmap
> - **Step-Functions Discretise Variance**: Binary Youden cutoffs provide useful clinical triage numbers (e.g. `B_cells` $\ge 0.430$) but discard continuous biological information, necessitating GAM spline modelling in future work.
> - **Univariate Tests Are Modest**: Single-biomarker predictive power tops out at AUC $= 0.632$ (`B_cells`), proving mathematically that single-gene tests fail in complex tumours and validating Phase 3 multidimensional clustering.
> - **Genomic Interactions Require Power**: `BRAF` oncogenic signalling significantly dampens T-cell inflammation ($\beta = -0.65, p = 0.040$), but smaller driver subgroups (`NF1` $N = 40$) require power-weighted resampling to confirm high effect-size synergies ($\beta = +0.94$).
> - **Spatial Resolution Is Essential**: Bulk transcriptomics cannot confirm TLS spatial maturation; future clinical translation requires integrating spatial histology image quantification.
