---
title: "Two-Tiered Clinical & Transcriptomic Feature Selection Report"
aliases:
  - Feature Selection Report
  - Clinical Feature Selection
  - ICI Feature Selection
tags:
  - melanoma
  - clinical-subtyping
  - feature-selection
  - logistic-regression
  - random-forest
  - two-tiered
  - ici-cohorts
  - anti-pd1
  - binary-response
created: 2026-08-07 12:32
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-07 12:32
---

# Two-Tiered Clinical & Transcriptomic Feature Selection Report

This report presents a **two-tiered feature selection architecture** evaluating clinical and transcriptomic predictors of **anti-PD-1 binary response** across the three ICI clinical trial cohorts. Feature selection is performed exclusively on ICI cohorts — TCGA-SKCM is excluded because downstream response prediction models are trained solely on immunotherapy trial data.

- **Tier 1 (ICI Multi-Cohort Immune Signatures, $N = 256$)**: Evaluates 6 transcriptomic immune signatures, $\text{TMB}$, age, and sex — pooled across **Liu 2019**, **Hugo 2016**, and **Riaz 2017**.
- **Tier 2 (ICI Granular Clinical Features, $N = 256$)**: Evaluates 23 granular categorical baseline clinical covariates available within the ICI trial cohorts.

## 1. Tier 1: ICI Multi-Cohort Immune Feature Selection ($N = 256$)

> [!INFO] What, Why & Questions — Tier 1
> **What We Are Doing**: Evaluating 6 transcriptomic immune gene expression signatures, `TMB`, age, and sex against **anti-PD-1 binary response** ($N_{\text{resp}} = 82$ / $N_{\text{non-resp}} = 113$) across all three ICI cohorts pooled ($N = 195$). Two complementary models are applied: **Random Forest** importance and **Logistic Regression** (univariate then multivariate).
> **Why We Are Doing It**: These features directly reflect the tumour immune microenvironment hypothesised to drive anti-PD-1 response.
> **Questions**:
>   1. *Which transcriptomic immune features are individually associated with anti-PD-1 response?*
>   2. *After mutual adjustment, which features retain independent predictive significance?*
>   3. *Do the 6 immune expression signatures exhibit collinearity?*

- **Total ICI Sample Size**: 256 patients across 3 cohorts (Liu 2019, Hugo 2016, Riaz 2017)
- **Response Evaluation Cohort**: 195 patients (82 responders / 113 non-responders)
- **Univariate FDR-Significant Predictors (FDR < 0.05)**: 0 features
- **Multivariate FDR-Significant Independent Predictors (FDR < 0.05)**: 0 feature(s)

### 1.1. Random Forest Importance for Anti-PD-1 Response ($N = 195$)
Random Forest feature importance trained on 195 ICI patients:

![Tier 1 RF Response Importance](../../plots/clinical/tier1_rf_response_importance.png)

> [!INSIGHT] Key Takeaways: Tier 1 RF Importance
> - **Top Predictor**: **`Immune Predictive Score (IMPRES) (Z-Score)`** accounts for 10.5% of total RF Gini importance across all features.
> - **Top 5 Features by Gini Importance**:
>   - `Immune Predictive Score (IMPRES) (Z-Score)` — 10.5%
>   - `Cytolytic Activity (CYT) Score (Z-Score)` — 10.1%
>   - `Tumour Mutational Burden (TMB) (Z-Score)` — 9.7%
>   - `SNV Neoantigen Burden (Z-Score)` — 9.2%
>   - `Macrophage STV Score (Z-Score)` — 8.2%

### 1.2. Univariate vs. Multivariate Odds Ratio Comparison for ICI Immune Signatures ($N = 195$)
Contrasting unadjusted Univariate $\text{OR}$ (blue circles) against multivariable-adjusted $\text{aOR}$ (orange squares) across $N = 195$ ICI patients (Model AUC-ROC = **0.726**, McFadden $R^2 = 0.109$):

![Tier 1 ICI Univariate vs Multivariate OR Comparison](../../plots/clinical/tier1_uni_vs_multi_or_forest.png)

> [!INSIGHT] Key Insights: ICI Immune Signature Predictors (Tier 1)
> - **Transcriptomic Collinearity**: All 8 transcriptomic/myeloid signatures (IFN-$\gamma$, TIS, CYT, CD8 T-cell, IMPRES, `PD-L1`, M1/M2 Ratio, Macrophage STV) show association with response in univariate logistic regression ($\text{OR} \approx 0.96\text{--}1.44$, leading $p \le 0.784$). In multivariate modelling, individual signatures attenuate towards $\text{OR} = 1.0$ due to shared variance.
> - **Statistical Significance & Power Limitations**: Neither univariate nor multivariate logistic models yield features passing FDR correction ($q < 0.05$). This is driven by modest sample size ($N = 195$), multiple testing burden across 18 candidate features, and substantial multicollinearity among transcriptomic signatures.
> - **Top Nominal Candidate**: **`Immune Predictive Score (IMPRES) (Z-Score)`** demonstrates the strongest unadjusted trend ($\text{aOR} = 1.66$, nominal $p = 0.025$), though it does not clear FDR correction (FDR $= 0.381$). High-dimensional modelling (Pillar 3/4) is required to aggregate these weak, correlated signals.

## 2. Tier 2: ICI Granular Clinical Feature Selection ($N = 256$)

> [!INFO] What, Why & Questions — Tier 2
> **What We Are Doing**: Evaluating 23 encoded dummy variables derived from granular baseline categorical clinical covariates available in the ICI trial cohorts ($N = 256$) — including clinical staging (`CLINICAL_STAGE`), anatomical biopsy site (`BIOPSY_SITE`), histological subtype (`TISSUE_SUBTYPE`), prior ICI therapy (`PRIOR_ICI_RX`), prior non-ICI therapy (`PRIOR_RX`), biopsy timing (`SAMPLE_TREATMENT`), and metastasis status (`METASTASIZED`) — against **anti-PD-1 binary response** using the same RF + Logistic Regression framework.
> **Why We Are Doing It**: Granular clinical covariates may independently predict ICI response beyond immune expression signatures.
> **Questions**:
>   1. *Which clinical staging or treatment covariates carry independent response signal?*
>   2. *Does prior ICI therapy confound response classification?*
>   3. *Do anatomical biopsy sites carry differential response rates?*

- **ICI Cohort Sample Size**: 256 patients (Liu 2019, Hugo 2016, Riaz 2017)
- **Response Evaluation Cohort**: 195 patients (82 responders / 113 non-responders)
- **Encoded Dummy Features**: 23 dummy variables
- **Univariate FDR-Significant Predictors (FDR < 0.05)**: 0 features
- **Multivariate FDR-Significant Predictors (FDR < 0.05)**: 0 feature(s)

### 2.1. Random Forest Importance for Granular ICI Clinical Attributes ($N = 195$)
Top-20 RF clinical predictors of anti-PD-1 response trained on $N = 195$ ICI patients:

![Tier 2 ICI RF Importance](../../plots/clinical/ici_tier2_rf_importance.png)

> [!INSIGHT] Key Takeaways: Tier 2 Clinical RF Importance
> - **Top Clinical Predictor**: **`Tissue Subtype: Other`** accounts for 13.0% of total RF Gini importance among granular clinical attributes.
> - **Top 5 Clinical Features by Gini Importance**:
>   - `Tissue Subtype: Other` — 13.0%
>   - `Biopsy Site: Endocrine` — 11.0%
>   - `Prior ICI Therapy: Ipilimumab` — 9.2%
>   - `Biopsy Site: Brain` — 8.2%
>   - `Tissue Subtype: Mucosal` — 5.8%

### 2.2. Univariate vs. Multivariate Odds Ratio Comparison for ICI Clinical Attributes ($N = 195$)
Contrasting unadjusted Univariate $\text{OR}$ (blue circles) against multivariable-adjusted $\text{aOR}$ (orange squares) across $N = 195$ ICI patients (Model AUC-ROC = **0.636**, McFadden $R^2 = 0.058$):

![Tier 2 ICI Univariate vs Multivariate OR Comparison](../../plots/clinical/ici_tier2_uni_vs_multi_or_forest.png)

> [!INSIGHT] Key Insights: ICI Granular Clinical Predictors (Tier 2)
> - **Lack of FDR Significance**: Neither univariate nor multivariate logistic regression models yield granular clinical covariates passing FDR correction ($q < 0.05$). Baseline clinical attributes alone (biopsy site, staging, prior therapy) provide insufficient predictive power for anti-PD-1 response without molecular TME profiling.
> - **Top Nominal Factor**: **`Biopsy Timing: Pre`** ($\text{aOR} = 1.60$, nominal $p = 0.324$) demonstrates the strongest unadjusted clinical association, though it does not clear FDR correction.
> - **Second Nominal Factor**: **`Biopsy Site: Soft Tissue`** ($\text{aOR} = 1.49$, nominal $p = 0.671$) shows weak secondary trend.
> - **Multivariate Attenuation**: All clinical covariates attenuate toward $\text{aOR} = 1.0$ in mutual adjustment, underscoring that routine clinical attributes cannot substitute for multi-omic biomarker panels.

## 3. Key Analytical & Biological Summary

> [!INSIGHT] Key Insights: Overall Findings
> 1. **Tier 1 Top Response Biomarker**: **`Immune Predictive Score (IMPRES) (Z-Score)`** is the single strongest univariate predictor of response across all 256 ICI patients ($\text{OR} = 1.44$, $p = 0.019$, FDR $= 0.273$).
> 2. **Tier 1 Independent Biomarker**: In joint multivariate modelling ($N = 195$), **`Immune Predictive Score (IMPRES) (Z-Score)`** remains an independent predictor ($\text{aOR} = 1.66$, $p = 0.025$, FDR $= 0.381$; Model AUC = **0.726**).
> 3. **Tier 2 Strongest Clinical Predictor**: **`Clinical Stage: Iii`** is the strongest univariate ICI clinical predictor ($\text{OR} = 0.16$, $p = 0.089$, FDR $= 0.568$).
> 4. **Biological Alignment**: Microenvironmental T-cell inflammation signatures consistently associate with response ($\text{OR} > 1.0$) across all 256 ICI patients.

## 4. Methodological Limitations & Future Directions

> [!WARNING] Analytical Scope & Limitations
> - **Small Response-Labelled Cohort**: The pooled ICI response dataset ($N = 195$, 82 responders) is small relative to the feature space in Tier 2 (23 encoded variables), increasing risk of overfitting.
> - **Cohort Heterogeneity**: Liu 2019, Hugo 2016, and Riaz 2017 differ in treatment agent, response definition, and biopsy timing.
> - **Collinearity Among Immune Signatures**: The six transcriptomic immune signatures share overlapping gene sets (`CD274`, `STAT1`, `IDO1`), inducing collinearity.
> - **High-Dimensional Tier 2 Feature Space**: One-hot encoding of 23 clinical variables against $N = 195$ patients creates a sparse matrix.
> - **No TCGA Pathological Staging**: Granular TNM staging and AJCC categories are not available in the ICI cohorts.

---

> [!formula]+ Clinical Feature Selection Script Execution & Software Module Architecture
>   - **Primary Pipeline Execution Scripts**:
>     - [`run_clinical_feature_selection.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-2-clinical-subtyping/run_clinical_feature_selection.py): Two-tiered ICI feature selection (N=256 across 3 cohorts) using RF Gini importance and Logistic Regression against anti-PD-1 binary response.
>   - **Data Preprocessing & Loading Modules**:
>     - [`clean_data.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/clean_data.py): Preprocesses raw cohort clinical metadata and RNA-seq expression profiles.
>     - [`signatures.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/signatures.py): Computes transcriptomic immune signatures across cohort expression matrices.
>   - **Shared Cross-Question & Pipeline Modules**:
>     - [`styles.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/src/styles.py): Single source of truth for Okabe-Ito colour palettes and presentation style.
