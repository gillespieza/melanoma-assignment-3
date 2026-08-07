---
title: "Two-Tiered Clinical & Transcriptomic Feature Selection Report"
aliases:
  - Feature Selection Report
  - Clinical Feature Selection
tags:
  - melanoma
  - clinical-subtyping
  - feature-selection
  - cox-regression
  - random-forest
  - two-tiered
  - multivariate-cox
created: 2026-08-07 10:22
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-07 10:22
---

# Two-Tiered Clinical & Transcriptomic Feature Selection Report

This report presents a comprehensive **two-tiered feature selection architecture** evaluating prognostic and predictive clinical markers across melanoma patient populations using both **Univariate** and **Multivariate Cox Proportional Hazards Regression**:

- **Tier 1 (Multi-Cohort Consensus, $N = 699$)**: Evaluates 6 transcriptomic immune signatures, $\text{TMB}$, age, and sex — pooled across all four study cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, **Riaz 2017**).
- **Tier 2 (Granular TCGA Pathological Staging, $N = 443$)**: Evaluates 77 detailed clinical, pathological TNM staging, anatomical site, aneuploidy, and hypoxia attributes specifically within the **TCGA-SKCM** reference cohort.

## 1. Tier 1: Multi-Cohort Feature Selection ($N = 699$)

> [!INFO] What, Why & Questions — Tier 1
> **What We Are Doing**: Evaluating transcriptomic immune features — 6 immune gene expression signatures (IFN-$\gamma$, TIS, CYT, CD8 T-cell, IMPRES, `PD-L1`), `TMB`, age, and sex — against overall survival across all four cohorts pooled ($N = 699$). We apply two complementary survival models: **Random Forest** importance (to rank features without distributional assumptions) and **Cox Proportional Hazards regression** (univariate then multivariate, to estimate hazard ratios and test for independent effects after adjusting for confounders).
> **Why We Are Doing It**: A feature that looks predictive in isolation may simply be correlated with a stronger feature (collinearity). Multivariate Cox regression disentangles this — only features that remain significant after mutual adjustment carry truly independent prognostic signal, protecting downstream models from redundant features.
> **Questions**:
>   1. *Which baseline clinical and transcriptomic features are individually associated with overall survival across all four cohorts?*
>   2. *After mutual adjustment, which features retain independent prognostic significance?*
>   3. *Do the 6 immune expression signatures collapse into one another due to collinearity?*

- **Total Merged Sample Size**: 699 patients across 4 cohorts
- **Overall Survival Evaluation Cohort**: 686 patients
- **Cox Survival Evaluation Cohort**: 676 patients
- **Univariate FDR-Significant Survival Predictors (FDR < 0.05)**: 7 features
- **Multivariate FDR-Significant Independent Predictors (FDR < 0.05)**: 1 feature(s)

### 1.1. Random Forest Importance for Overall Survival ($N = 686$)
Random Forest feature importance (500 estimators) trained on the $N = 686$ overall survival cohort:
![Tier 1 Random Forest OS](../../plots/clinical/clinical_feature_importance.png)

### 1.2. Univariate vs. Multivariate Cox Hazard Ratio Comparison ($N = 676$)
Contrasting unadjusted Univariate Hazard Ratios ($\text{HR}$, blue circles) against multivariable-adjusted Hazard Ratios ($\text{aHR}$, orange squares) across $N = 676$ multi-cohort patients with complete survival data (Model Concordance Index = **0.623**, Likelihood Ratio Test $p = 2.82 \times 10^{-10}$):

![Tier 1 Univariate vs Multivariate Cox Comparison](../../plots/clinical/tier1_uni_vs_multi_forest_plot.png)

> [!INSIGHT] Key Insights: Multivariable Adjustment & Collinearity (Tier 1)
> - **Transcriptomic Collinearity & Attenuation**: All 6 transcriptomic immune signatures (IFN-$\gamma$, TIS, CYT, CD8 T-cell, IMPRES, `PD-L1`) show significant protective association with survival in unadjusted univariate Cox models ($\text{HR} \approx 0.73\text{--}0.82$, all $p \leq 4.86 \times 10^{-4}$). However, in joint multivariate modelling, individual signatures attenuate towards the null ($\text{aHR} \to 1.0$) and lose independent significance — demonstrating that while T-cell microenvironmental inflammation is genuinely protective, individual signatures capture overlapping, collinear aspects of the same biological axis.
> - **Sole Independent Risk Factor**: **`Patient Age (Z-Score)`** ($\text{aHR} = 1.33$, $p = 5.42 \times 10^{-7}$, FDR $= 4.88 \times 10^{-6}$) retains independent statistical significance after joint adjustment, confirming that age-related immunosenescence or host fragility confers mortality risk independently of tumour inflammation.

## 2. Tier 2: Granular TCGA Pathological & Clinical Staging ($N = 443$)

> [!INFO] What, Why & Questions — Tier 2
> **What We Are Doing**: Evaluating 77 granular clinical and pathological attributes available exclusively in **TCGA-SKCM** ($N = 443$) — including TNM staging (T1–T4, N0–N3, metastasis), AJCC stage groupings, primary anatomical sites, aneuploidy score, and hypoxia scores — against overall survival using the same Random Forest + Cox regression two-step framework.
> **Why We Are Doing It**: Clinical trial datasets record only basic demographics. The TCGA reference cohort contains rich pathological staging data not available in the trial cohorts — answering which pathological features independently determine prognosis in the general melanoma population.
> **Questions**:
>   1. *Which TNM staging features carry the strongest independent prognostic signal?*
>   2. *Does primary tumour invasion depth (T staging) dominate over nodal spread (N staging) as an independent predictor?*
>   3. *Do anatomical primary sites carry independent survival differences beyond TNM stage?*

- **TCGA Total Cohort Sample Size**: 443 patients
- **TCGA OS Classification Cohort**: 436 patients
- **TCGA Cox Survival Evaluation Cohort**: 427 patients
- **Encoded Dummy Features**: 77 dummy variables
- **Univariate FDR-Significant Pathological Predictors (FDR < 0.05)**: 10 features
- **Multivariate FDR-Significant Predictors (FDR < 0.05)**: 4 feature(s)

### 2.1. Random Forest Importance for Granular TCGA Clinical Attributes ($N = 436$)
Top 20 Random Forest clinical predictors trained on $N = 436$ TCGA patients with non-null survival status:

![Tier 2 TCGA Random Forest](../../plots/clinical/tcga_clinical_feature_importance.png)

### 2.2. Univariate vs. Multivariate Cox Hazard Ratio Comparison for TCGA Attributes ($N = 427$)
Contrasting unadjusted Univariate Hazard Ratios ($\text{HR}$, blue circles) against multivariable-adjusted Hazard Ratios ($\text{aHR}$, orange squares) across $N = 427$ TCGA patients (Model Concordance Index = **0.712**, Likelihood Ratio Test $p = 2.35 \times 10^{-16}$):

![Tier 2 TCGA Univariate vs Multivariate Cox Comparison](../../plots/clinical/tcga_uni_vs_multi_forest_plot.png)

> [!INSIGHT] Key Insights: Pathological Staging Independence (Tier 2 TCGA)
> - **Top Independent Risk Factor**: **`Primary Tumour (t) Staging: T4B`** ($\text{aHR} = 2.47$, $p = 1.57 \times 10^{-6}$) maintains the strongest independent prognostic risk elevation in multivariate modelling.
> - **Second Independent Risk Factor**: **`Patient Age`** ($\text{aHR} = 1.02$, $p = 5.09 \times 10^{-6}$) also retains independent prognostic significance after mutual adjustment.
> - **Attenuated Staging Categories**: Other staging categories (e.g. N3 nodal staging, AJCC Stage IIIC) exhibit significant univariate risk elevation but attenuate in multivariate modelling as their variance is explained by the top independent predictors.

## 3. Key Analytical & Biological Summary

> [!INSIGHT] Key Insights: Overall Findings
> 1. **Tier 1 Top Survival Biomarker**: **`IFN-gamma Signature (Z-Score)`** is the single strongest protective univariate predictor across all 4 cohorts ($\text{HR} = 0.73$, $p = 5.61 \times 10^{-9}$, FDR $= 5.05 \times 10^{-8}$).
> 2. **Tier 1 Independent Survival Biomarker**: In joint multivariate modelling ($N = 676$), **`Patient Age (Z-Score)`** remains an independent prognostic predictor ($\text{aHR} = 1.33$, $p = 5.42 \times 10^{-7}$, FDR $= 4.88 \times 10^{-6}$; Model C-index = **0.623**).
> 3. **Tier 2 Pathological Staging Independence**: **`Primary Tumour (t) Staging: T4B`** is the strongest univariate predictor in TCGA ($\text{HR} = 3.19$, $p = 3.17 \times 10^{-11}$, FDR $= 2.35 \times 10^{-9}$), and retains significant independent risk elevation in multivariate Cox regression (Model C-index = **0.712**).
> 4. **Biological Alignment**: Microenvironmental T-cell inflammation signatures (IFN-$\gamma$, TIS, CYT, CD8, IMPRES) consistently confer significant mortality risk reduction ($\text{HR} < 1.0$) in univariate Cox models across all 699 patients, confirming the prognostic value of the tumour immune microenvironment.

## 4. Methodological Limitations & Future Directions

> [!WARNING] Analytical Scope & Limitations
> - **Proportional Hazards Assumption**: Cox regression assumes hazard ratios remain constant over time. This assumption was not formally tested (e.g., Schoenfeld residuals) and may be violated for some features, particularly those with time-varying effects.
> - **Collinearity Among Immune Signatures**: The six transcriptomic immune signatures share overlapping gene sets and are highly collinear. Multivariate Cox estimates for individual signatures are unstable and should not be over-interpreted in isolation.
> - **Tier 2 TCGA Staging Scope**: Granular TNM staging and anatomical site data are available only in TCGA-SKCM ($N = 443$) and cannot be transferred to clinical trial cohorts, limiting the generalisability of Tier 2 findings.
> - **High-Dimensional Dummy Encoding**: One-hot encoding of 77 categorical staging variables creates sparse, high-dimensional feature matrices. Multivariate models in Tier 2 are particularly susceptible to overfitting and convergence instability at low per-category sample counts.
> - **OS as Outcome Proxy**: Overall Survival reflects diverse treatment histories (surgery, targeted therapy, immunotherapy) rather than response to a single agent, making it a weaker endpoint than progression-free survival under checkpoint blockade.

> [!formula]+ Clinical Feature Selection Script Execution & Software Module Architecture
> - **Primary Pipeline Execution Scripts**:
>   - [`run_clinical_feature_selection.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-2-clinical-subtyping/run_clinical_feature_selection.py): Evaluates two-tiered clinical and transcriptomic feature selection across multi-cohort ($N = 699$) and TCGA ($N = 443$) datasets using Random Forest Gini importance and Univariate/Multivariate Cox Proportional Hazards regression, and outputs `clinical_feature_selection_report.md`.
> - **Data Preprocessing & Loading Modules**:
>   - [`clean_data.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/clean_data.py): Preprocesses raw cohort clinical metadata and RNA-seq expression profiles into cleaned CSV matrices.
>   - [`merge_datasets.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/merge_datasets.py): Merges processed expression matrices across cohorts into harmonised pooled matrices (`expr_merged.csv`, `clin_merged.csv`).
>   - [`signatures.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/signatures.py): Computes transcriptomic immune signatures (IFN-$\gamma$, TIS, CYT, CD8 T-cell, IMPRES, `PD-L1`) across cohort expression matrices.
> - **Shared Cross-Question & Pipeline Modules**:
>   - [`run_pipeline.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/run_pipeline.py): Master Q1 pipeline orchestrator executing downstream modeling and evaluation.
>   - [`styles.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/src/styles.py): Single source of truth for Okabe-Ito colour palettes (`COHORT_PALETTE`, `RESPONSE_PALETTE`, `MODEL_TYPE_PALETTE`) and visualisation presentation style.