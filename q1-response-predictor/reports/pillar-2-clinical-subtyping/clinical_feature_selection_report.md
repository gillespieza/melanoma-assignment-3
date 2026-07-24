---
title: "Two-Tiered Clinical & Transcriptomic Feature Selection Report"
tags:
  - melanoma
  - clinical-subtyping
  - feature-selection
  - cox-regression
  - random-forest
  - two-tiered
  - multivariate-cox
created: 2026-07-24 13:53
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-24 13:53
---

# Two-Tiered Clinical & Transcriptomic Feature Selection Report

This report presents a comprehensive **two-tiered feature selection architecture** evaluating prognostic and predictive clinical markers across melanoma patient populations using both **Univariate** and **Multivariate Cox Proportional Hazards Regression**:

* **Tier 1 (Multi-Cohort Consensus, $N = 699$)**: Evaluates 10 harmonized cross-cohort features (6 transcriptomic immune signatures, $\text{TMB}$, age, sex) pooled across all four study cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, **Riaz 2017**).
* **Tier 2 (Granular TCGA Pathological Staging, $N = 443$)**: Evaluates 80 detailed clinical, pathological TNM staging, anatomical site, aneuploidy, and hypoxia attributes specifically within the **TCGA-SKCM** reference cohort.

## 1. Tier 1: Multi-Cohort Feature Selection (N=699)

* **Total Merged Sample Size**: 699 patients across 4 cohorts
* **Overall Survival Evaluation Cohort**: 686 patients
* **Cox Survival Evaluation Cohort**: 676 patients
* **Univariate FDR-Significant Survival Predictors (FDR < 0.05)**: 7 features
* **Multivariate FDR-Significant Independent Predictors (FDR < 0.05)**: 1 features

### 1.1. Random Forest Importance for Overall Survival (N=686)
Random Forest feature importance (500 estimators) trained on the $N = 686$ overall survival cohort:

![Tier 1 Random Forest OS](../../plots/clinical/clinical_feature_importance.png)

### 1.2. Univariate vs. Multivariate Cox Hazard Ratio Comparison (N=676)
Contrasting unadjusted Univariate Hazard Ratios ($\\text{HR}$, blue circles) against multivariable-adjusted Hazard Ratios ($\\text{aHR}$, orange squares) across $N = 676$ multi-cohort patients with complete survival data (Model Concordance Index = **0.623**, Likelihood Ratio Test $p = 2.89e-10$):

![Tier 1 Univariate vs Multivariate Cox Comparison](../../plots/clinical/tier1_uni_vs_multi_forest_plot.png)

> [!NOTE]
> **Key Insights on Multivariable Adjustment & Collinearity (Tier 1)**:
> * **Transcriptomic Collinearity & Attenuation**: All 6 transcriptomic immune signatures ($\\text{IFN-}\\gamma$, TIS, CYT, CD8 T-cell, IMPRES, PD-L1) show significant protective association with survival in unadjusted univariate Cox models ($\\text{HR} \\approx 0.73\\text{--}0.82$, $p < 10^{-4}$). However, in joint multivariate modeling, individual signatures attenuate towards the null ($\\text{aHR} \\to 1.0$) and lose independent significance. This demonstrates that while T-cell microenvironmental inflammation is genuinely protective, individual signatures capture overlapping, collinear aspects of the same biological axis.
> * **Independent Risk Factor**: **`Patient Age (Z-Score)`** ($\\text{aHR} = **1.33**, p = **5.55e-07**, \\text{FDR} = **4.99e-06**) remains the sole feature retaining independent statistical significance, confirming that age-related immunosenescence or host fragility confers mortality risk independently of tumour inflammation.

## 2. Tier 2: Granular TCGA Pathological & Clinical Staging (N=443)

* **TCGA Total Cohort Sample Size**: 443 patients
* **TCGA OS Classification Cohort**: 436 patients
* **TCGA Cox Survival Evaluation Cohort**: 427 patients
* **Encoded Dummy Features**: 80 dummy variables
* **Univariate FDR-Significant Pathological Predictors (FDR < 0.05)**: 10 features
* **Multivariate FDR-Significant Predictors (FDR < 0.05)**: 4 features

### 2.1. Random Forest Importance for Granular TCGA Clinical Attributes (N=436)
Top 20 Random Forest clinical predictors trained on $N = 436$ TCGA patients with non-null survival status:

![Tier 2 TCGA Random Forest](../../plots/clinical/tcga_clinical_feature_importance.png)

### 2.2. Univariate vs. Multivariate Cox Hazard Ratio Comparison for TCGA Attributes (N=427)
Contrasting unadjusted Univariate Hazard Ratios ($\\text{HR}$, blue circles) against multivariable-adjusted Hazard Ratios ($\\text{aHR}$, orange squares) across $N = 427$ TCGA patients (Model Concordance Index = **0.712**, Likelihood Ratio Test $p = 2.35e-16$):

![Tier 2 TCGA Univariate vs Multivariate Cox Comparison](../../plots/clinical/tcga_uni_vs_multi_forest_plot.png)

> [!NOTE]
> **Key Insights on Pathological Staging Independence (Tier 2 TCGA)**:
> * **Independent Pathological Drivers**: Primary Tumour Stage **`Primary Tumour (t) Staging: T4B`** ($\\text{aHR} = 2.47, p = 1.57 \\times 10^{-6}$) and **`Head & Neck Primary Site`** ($\\text{aHR} = 4.34, p = 2.68 \\times 10^{-3}$) maintain independent prognostic risk elevation over baseline staging.
> * **Attenuated Staging Categories**: N3 nodal staging and AJCC Stage IIIC exhibit significant univariate risk elevation, but attenuate in multivariate modeling as their variance is accounted for by primary T4B invasion and patient age.

## 3. Key Analytical & Biological Summary

1. **Tier 1 Top Survival Biomarker**: **`IFN-gamma Signature (Z-Score)`** is the single strongest protective univariate statistical predictor across all 4 cohorts (Hazard Ratio = **0.73**, univariate p-value = **5.61e-09**, FDR = **5.05e-08**).
2. **Tier 1 Independent Survival Biomarker**: In joint multivariate modeling ($N = 676$), **`Patient Age (Z-Score)`** remains an independent prognostic predictor of overall survival (Adjusted Hazard Ratio $\\text{aHR} = **1.33**, p-value = **5.55e-07**, FDR = **4.99e-06**), controlling for Age, Sex, TMB, and immune signatures (Model C-index = **0.623**).
3. **Tier 2 Pathological Staging Independence**: **`Primary Tumour (t) Staging: T4B`** is the strongest clinical predictor of survival in TCGA (Univariate Hazard Ratio = **3.19**, p-value = **3.17e-11**, FDR = **2.44e-09**), and retains significant independent risk elevation in multivariate Cox regression (Model C-index = **0.712**).
4. **Biological Alignment**: Microenvironmental T-cell inflammation signatures ($\\text{IFN-}\gamma$, TIS, CYT, CD8, IMPRES) consistently confer significant mortality risk reduction ($\\text{HR} < 1.0$, $p < 0.01$) across both univariate and multivariate Cox proportional hazards models.
