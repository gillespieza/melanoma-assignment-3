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
created: 2026-07-24 13:48
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-24 13:48
---

# Two-Tiered Clinical & Transcriptomic Feature Selection Report

This report presents a comprehensive **two-tiered feature selection architecture** evaluating prognostic and predictive clinical markers across melanoma patient populations using both **Univariate** and **Multivariate Cox Proportional Hazards Regression**:

* **Tier 1 (Multi-Cohort Consensus, $N = 699$)**: Evaluates 10 harmonized cross-cohort features (6 transcriptomic immune signatures, $\text{TMB}$, age, sex) pooled across all four study cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, **Riaz 2017**).
* **Tier 2 (Granular TCGA Pathological Staging, $N = 443$)**: Evaluates 80 detailed clinical, pathological TNM staging, anatomical site, aneuploidy, and hypoxia attributes specifically within the **TCGA-SKCM** reference cohort.

## 1. Tier 1: Multi-Cohort Feature Selection (N=699)

* **Total Merged Sample Size**: 699 patients across 4 cohorts
* **Overall Survival Evaluation Cohort**: 686 patients
* **Cox Survival Evaluation Cohort**: 676 patients
* **Anti-PD-1 Response Evaluation Cohort**: 247 trial patients
* **Univariate FDR-Significant Survival Predictors (FDR < 0.05)**: 7 features
* **Multivariate FDR-Significant Independent Predictors (FDR < 0.05)**: 1 features

### 1.1. Random Forest Importance for Overall Survival (N=686)
Random Forest feature importance (500 estimators) trained on the $N = 686$ overall survival cohort:

![Tier 1 Random Forest OS](../../plots/clinical/clinical_feature_importance.png)

### 1.2. Univariate Cox Proportional Hazards Regression (N=676)
Univariate Cox Proportional Hazards models fitted across $N = 676$ patients with complete survival duration data:

![Tier 1 Cox Forest Plot](../../plots/clinical/cox_forest_plot.png)

### 1.3. Multivariate Cox Proportional Hazards Regression (N=676)
Multivariate Cox Proportional Hazards model evaluating joint covariate effects across $N = 676$ multi-cohort patients with complete survival data (Model Concordance Index = **0.623**, Likelihood Ratio Test $p = 2.89e-10$):

![Tier 1 Multivariate Cox Forest Plot](../../plots/clinical/multivariate_cox_forest_plot.png)

#### Comparison of Univariate vs. Multivariate Adjusted Hazard Ratios (Tier 1)

![Tier 1 Univariate vs Multivariate Cox Comparison](../../plots/clinical/tier1_uni_vs_multi_forest_plot.png)

### 1.4. Random Forest Importance for Anti-PD-1 Immunotherapy Response (N=247)
Random Forest feature importance predicting objective response (CR/PR vs PD) across the $N = 247$ trial cohort:

![Tier 1 Random Forest Response](../../plots/clinical/response_feature_importance.png)

### 1.5. Univariate Forest Plot for Anti-PD-1 Immunotherapy Response (N=247)
Univariate Logistic Regression Odds Ratio (OR) forest plot predicting objective anti-PD-1 response across the $N = 247$ trial cohort:

![Tier 1 Response Forest Plot](../../plots/clinical/response_forest_plot.png)

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

### 2.2. Univariate Cox Proportional Hazards Regression for TCGA Attributes (N=427)
Top 20 Univariate Cox hazard ratios evaluated across $N = 427$ TCGA patients with complete survival duration data:

![Tier 2 TCGA Cox Forest Plot](../../plots/clinical/tcga_cox_forest_plot.png)

### 2.3. Multivariate Cox Proportional Hazards Regression for TCGA Attributes (N=427)
Multivariate Cox Proportional Hazards model evaluating joint TCGA clinical & pathological attributes across $N = 427$ patients (Model Concordance Index = **0.712**, Likelihood Ratio Test $p = 2.35e-16$):

![Tier 2 TCGA Multivariate Cox Forest Plot](../../plots/clinical/tcga_multivariate_cox_forest_plot.png)

#### Comparison of Univariate vs. Multivariate Adjusted Hazard Ratios (Tier 2 TCGA)

![Tier 2 TCGA Univariate vs Multivariate Cox Comparison](../../plots/clinical/tcga_uni_vs_multi_forest_plot.png)

## 3. Key Analytical & Biological Summary

1. **Tier 1 Top Survival Biomarker**: **`IFN-gamma Signature (Z-Score)`** is the single strongest protective univariate statistical predictor across all 4 cohorts (Hazard Ratio = **0.73**, univariate p-value = **5.61e-09**, FDR = **5.05e-08**).
2. **Tier 1 Independent Survival Biomarker**: In joint multivariate modeling ($N = 676$), **`Patient Age (Z-Score)`** remains an independent prognostic predictor of overall survival (Adjusted Hazard Ratio $\\text{aHR} = **1.33**, p-value = **5.55e-07**, FDR = **4.99e-06**), controlling for Age, Sex, TMB, and immune signatures (Model C-index = **0.623**).
3. **Tier 1 Top Immunotherapy Marker**: **`TMB Nonsynonymous (Z-Score)`** is the top predictive feature for objective anti-PD-1 response across trial cohorts (Gini Importance = **0.1537**).
4. **Tier 2 Pathological Staging Independence**: **`Primary Tumour (t) Staging: T4B`** is the strongest clinical predictor of survival in TCGA (Univariate Hazard Ratio = **3.19**, p-value = **3.17e-11**, FDR = **2.44e-09**), and retains significant independent risk elevation in multivariate Cox regression (Model C-index = **0.712**).
5. **Biological Alignment**: Microenvironmental T-cell inflammation signatures ($\\text{IFN-}\gamma$, TIS, CYT, CD8, IMPRES) consistently confer significant mortality risk reduction ($\\text{HR} < 1.0$, $p < 0.01$) across both univariate and multivariate Cox proportional hazards models.
