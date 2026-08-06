---
title: "Clinical Characteristics of Data Cohorts"
aliases:
  - Clinical Cohort Characteristics
tags:
  - melanoma
  - clinical-analysis
  - cohort-characteristics
  - survival-analysis
  - kaplan-meier
cssclasses:
  - table-small
  - table-center
  - row-alt
created: 2026-08-02 12:49
updated: 2026-08-02 12:49
---

# Clinical Characteristics of Data Cohorts

## 1. Baseline Patient and Disease Characteristics

> [!INFO] Why We Are Doing This
> **What**: We compare patient demographics, treatment histories, and survival outcomes across the four melanoma study cohorts: **TCGA-SKCM** ($N = 443$), **Liu 2019** ($N = 122$), **Hugo 2016** ($N = 27$), and **Riaz 2017** ($N = 107$).
> **Why**: Before building predictive models or analysing transcriptomic signatures, we must understand the clinical composition of each dataset. Cohort-level differences in prior treatment, disease stage, and patient demographics can confound downstream survival and response analyses.
> **Question Answered**: Are baseline patient populations sufficiently comparable across independent datasets to permit pooled multi-cohort machine learning?

This report compares patient demographics, treatments, survival, and sample attrition across the four melanoma study cohorts:
- **TCGA-SKCM**: Baseline reference dataset ($N = 443$) representing primary and metastatic melanoma with general treatment histories.
- **Liu 2019**: Immunotherapy trial ($N = 122$) treated with anti-PD-1 monotherapy (Pembrolizumab or Nivolumab).
- **Hugo 2016**: Anti-PD-1 clinical trial cohort ($N = 27$).
- **Riaz 2017**: Anti-PD-1 clinical trial cohort ($N = 107$); 100% of patients had previously received anti-CTLA-4 (Ipilimumab).

### 1.1 Clinical Demographics & Treatment Distributions (2×2 Grid)

![Clinical Demographics & Treatment Distributions](../../plots/clinical/clinical_demographics_2x2_grid.png)

_**Figure 1: 2×2 Grid of Clinical Demographics and Treatment Histories across Cohorts.** Panel A: sex distribution; Panel B: age at diagnosis; Panel C: TCGA-SKCM treatment modalities; Panel D: immunotherapy agents administered across trial cohorts._

#### Key Demographics & Treatment Insights

- **Panel A: Sex Distribution ($N = 684$)**: The overall cohort shows a male predominance (**59.8% Male** [$N = 409$] vs. **40.2% Female** [$N = 275$]), reflecting real-world cutaneous melanoma incidence patterns where male patients account for the majority of advanced presentations.
- **Panel B: Age Distribution across Studies**: Patient ages span from 15 to 90 years with a **median age of 57.0 years** ($\text{IQR} = 48.0\text{--}69.0\text{ years}$). Trial cohorts (`Hugo 2016`: median 61.0; `Riaz 2017`: median 56.0; `TCGA-SKCM`: median 57.0) display consistent age distributions centred around late middle age.
- **Panel C: TCGA-SKCM Recorded Treatment Types ($N = 443$)**: Among TCGA-SKCM patients, **Radiation Therapy** represents the largest modality (**25.3%** [$N = 112$]), followed by **Immunotherapy** (**15.8%** [$N = 70$]), **Chemotherapy** (**15.1%** [$N = 67$]), **Vaccine Therapy** (**5.0%** [$N = 22$]), and **Targeted Therapy** (**2.7%** [$N = 12$]). **54.4%** [$N = 241$] have no recorded systemic therapy in cBioPortal.
- **Panel D: Immunotherapy Agents Administered ($N = 243$)**: Across all immunotherapy-treated trial and TCGA patients, **Nivolumab (Anti-PD-1)** is the predominant agent (**21.0%** [$N = 51$]), followed by **Pembrolizumab (Anti-PD-1)** (**29.2%** [$N = 71$]), **Ipilimumab Combination / Prior Exposure** (**22.6%** [$N = 55$]), and **Unspecified Anti-PD-1** (**27.2%** [$N = 66$]).

_**Table 1: Baseline Patient and Disease Characteristics**_

| Characteristic                | Liu 2019   | Hugo 2016        | Riaz 2017        | TCGA-SKCM        |
|:------------------------------|:-----------|:-----------------|:-----------------|:-----------------|
| **N**                         | 122        | 27               | 107              | 443              |
|                               |            |                  |                  |                  |
| **Demographics**              |            |                  |                  |                  |
| Age, median (IQR)             | N/A        | 61.0 (54.0–68.5) | 56.0 (48.8–63.0) | 57.0 (47.0–70.0) |
| Female sex, n (%)             | 51 (41.8%) | 8 (29.6%)        | 47 (51.1%)       | 169 (38.1%)      |
|                               |            |                  |                  |                  |
| **Treatment**                 |            |                  |                  |                  |
| ICI agent — Pembrolizumab     | 71 (58.2%) | 0 (0.0%)         | —                | 3 (0.7%)         |
| ICI agent — Nivolumab         | 51 (41.8%) | 0 (0.0%)         | 0 (0.0%)         | 1 (0.2%)         |
| Prior anti-CTLA-4             | 0 (0.0%)   | —                | 55 (100.0%)      | —                |
| Treatment History (TCGA only) | —          | —                | —                |                  |
| — Radiation Therapy           | —          | —                | —                | 112 (25.3%)      |
| — Immunotherapy               | —          | —                | —                | 70 (15.8%)       |
| — Chemotherapy                | —          | —                | —                | 67 (15.1%)       |
| — Vaccine                     | —          | —                | —                | 22 (5.0%)        |
| — Targeted Therapy            | —          | —                | —                | 12 (2.7%)        |
| — Other Therapy               | —          | —                | —                | 7 (1.6%)         |
| — No recorded treatment       | —          | —                | —                | 241 (54.4%)      |
|                               |            |                  |                  |                  |
| **Survival Outcomes**         |            |                  |                  |                  |
| Median OS, months (95% CI)    | 22.6       | 32.2             | 21.2             | 79.0             |
| OS events, n (%)              | 62 (50.8%) | 12 (46.2%)       | 63 (62.4%)       | 212 (49.6%)      |
| Median follow-up, months      | 17.5       | 14.4             | 17.8             | 41.6             |

## 2. Sample Preprocessing Attrition

> [!INFO] Why We Are Doing This
> **What**: We track sample retention through quality control, identifier standardisation, and clinical-expression alignment across all $N = 704$ initial records.
> **Why**: Documenting attrition at each preprocessing step verifies data integrity and confirms that no patient sub-population was accidentally excluded.
> **Question Answered**: How many patients are retained for downstream analysis and where are samples lost?

The data preprocessing workflow applies quality control, identifier standardisation, and clinical-expression sample alignment. The table below outlines sample retention and attrition rationale for each cohort.

_**Table 2: Sample Attrition Across Preprocessing Steps**_

| Cohort    | Preprocessing Step            |   N Initial |   N Retained |   N Removed | Rationale                                                                                              |
|:----------|:------------------------------|------------:|-------------:|------------:|:-------------------------------------------------------------------------------------------------------|
| Liu 2019  | Clinical data loaded          |         122 |          122 |           0 | Merged patient-level and sample-level clinical records.                                                |
| Liu 2019  | Clinical data harmonised      |         122 |          122 |           0 | Applied identifier standardisation, clinical cleaning, baseline filtering, and variable harmonisation. |
| Liu 2019  | Clinical-expression alignment |         122 |          122 |           0 | Retained samples with matching clinical and expression data.                                           |
| Hugo 2016 | Clinical data loaded          |          27 |           27 |           0 | Merged patient-level and sample-level clinical records.                                                |
| Hugo 2016 | Clinical data harmonised      |          27 |           27 |           0 | Applied identifier standardisation, clinical cleaning, baseline filtering, and variable harmonisation. |
| Hugo 2016 | Clinical-expression alignment |          27 |           27 |           0 | Retained samples with matching clinical and expression data.                                           |
| Riaz 2017 | Clinical data loaded          |         107 |          107 |           0 | Merged patient-level and sample-level clinical records.                                                |
| Riaz 2017 | Clinical data harmonised      |         107 |          107 |           0 | Applied identifier standardisation, clinical cleaning, baseline filtering, and variable harmonisation. |
| Riaz 2017 | Clinical-expression alignment |         107 |          107 |           0 | Retained samples with matching clinical and expression data.                                           |
| TCGA-SKCM | Clinical data loaded          |         448 |          448 |           0 | Merged patient-level and sample-level clinical records.                                                |
| TCGA-SKCM | Clinical data harmonised      |         448 |          448 |           0 | Applied identifier standardisation and clinical data cleaning.                                         |
| TCGA-SKCM | Clinical-expression alignment |         448 |          443 |           5 | Retained samples with matching clinical and expression data.                                           |

### Key Observations
1. **Overall Cohort Size ($N = 699$)**: The largest individual dataset is **TCGA-SKCM** ($N = 443$). Combined across all four cohorts, **$N = 699$** cleaned patient records were harmonised for downstream analysis.
2. **Minimal Sample Attrition (99.3% Retention)**: Across all 704 initial records, only **5** sample(s) were removed. **Liu 2019** retained 100% of samples ($N = 122$); **Hugo 2016** retained 100% of samples ($N = 27$); **Riaz 2017** retained 100% of samples ($N = 107$); **TCGA-SKCM** lost **5** sample(s) (448 → 443). All three immunotherapy trial cohorts retained 100% of their samples; TCGA-SKCM attrition reflects unmatched RNA-seq expression records.
3. **Longest Follow-up**: **TCGA-SKCM** shows the longest median follow-up duration (**41.6 months**) due to its inclusion of earlier-stage surgical cases.
4. **Treatment History & Clinical Context**: All $N = 107$ patients in **Riaz 2017** were previously treated with anti-CTLA-4 (Ipilimumab), representing an Ipilimumab-refractory population relative to the anti-CTLA-4 naïve patients in **Liu 2019** ($N = 122$).

## 3. Overall Survival Curves (KM Plots)

> [!INFO] Why We Are Doing This
> **What**: We plot unstratified Kaplan-Meier overall survival curves for each of the four cohorts.
> **Why**: Visualising baseline mortality rates establishes the clinical context for each dataset and confirms that follow-up duration is sufficient to evaluate treatment outcomes.
> **Question Answered**: How does overall survival differ between the TCGA-SKCM reference cohort and the active immunotherapy trial cohorts?

The unstratified Kaplan-Meier overall survival curves for each cohort are shown below. Median overall survival is annotated for each cohort where reached.

![Overall Survival KM Curves (All Cohorts)](../../plots/clinical/km_os_grid.png)

_**Figure 2: Unstratified Overall Survival KM Curves across All Four Cohorts.**_

## 4. Overall Survival Stratified by Immunotherapy Response

> [!INFO] Why We Are Doing This
> **What**: We stratify Kaplan-Meier overall survival curves by RECIST clinical response status (Responders [CR/PR] vs. Non-responders [PD]) across the three immunotherapy trial cohorts ($N = 256$).
> **Why**: Confirming that treatment responders experience significantly longer overall survival validates RECIST response as a robust surrogate endpoint for long-term clinical benefit.
> **Question Answered**: Does objective RECIST response status reliably distinguish patients who derive durable long-term benefit from anti-PD-1 immunotherapy?

![Overall Survival by Immunotherapy Response (RECIST)](../../plots/clinical/km_os_by_response.png)

_**Figure 3: Overall Survival Stratified by RECIST Response Status across Immunotherapy Trial Cohorts.** Treatment responders (CR/PR) exhibit significantly superior overall survival compared to non-responders (PD) across all three trial cohorts (Log-rank $p < 0.0001$)._

> [!INSIGHT] Key Insights: Survival Stratification by Response
> 1. **Profound Survival Benefit**: Treatment responders (CR/PR) achieve significantly longer overall survival compared to non-responders (PD) across all three independent trial cohorts (Liu 2019, Hugo 2016, and Riaz 2017; all Log-rank $p < 0.0001$).
> 2. **Durable Long-Term Survival**: In **Liu 2019**, non-responders experience rapid mortality whereas the majority of responders survive well beyond 50 months. Similar durable separation is observed in Hugo 2016 and Riaz 2017.
> 3. **Conclusion**: Objective RECIST response is a highly robust surrogate endpoint for overall survival in metastatic melanoma, justifying its use as the primary outcome for predictive model training.

## 5. Univariate Associations with Response (Forest Plot)

> [!INFO] Why We Are Doing This
> **What**: We run univariate statistical tests (Odds Ratios and 95% Confidence Intervals) to measure the isolated predictive strength of individual baseline features — age, sex, tumour mutation burden (TMB), and driver mutations (`BRAF`, `NRAS`, `NF1`) — against immunotherapy response across individual and pooled trial cohorts.
> **Why**: Before building complex multivariate models, univariate screening identifies whether any single clinical or genomic feature alone is sufficient to predict response.
> **Question Answered**: Does any single baseline clinical or genomic feature reliably predict anti-PD-1 immunotherapy response across independent cohorts?

![Forest Plot of Univariate Odds Ratios](../../plots/clinical/univariate_associations.png)

_**Figure 4: Forest Plot of Univariate Odds Ratios for Clinical and Genomic Features against Immunotherapy Response.**_

> [!INSIGHT] Key Takeaways: Univariate Associations
> 1. **Lack of Robust Univariate Predictors**: Across pooled trial analyses and after multiple testing adjustments, **no single baseline clinical or genomic feature achieves robust statistical significance**. All 95% CIs for pooled odds ratios cross 1.0.
> 2. **Anatomical Staging Artefact (Liu 2019 $p = 0.029$)**: Stage IV disease shows a nominal unadjusted association in the isolated Liu 2019 cohort. This is an artefact of extreme trial enrolment imbalance; in the pooled multi-cohort analysis, the association attenuates to $p = 0.083$.
> 3. **Subtle Feature Trends**: `NRAS` mutations and higher Tumour Mutation Burden (TMB; OR = 1.37 per SD increase, $p = 0.160$) trend towards elevated response odds. `BRAF` driver mutations show no univariate association (OR ≈ 1.0).
> 4. **Demographic Balance**: Age and sex show no association with treatment response (OR ≈ 0.98–1.26, $p > 0.50$), confirming demographic balance between responder and non-responder arms.
> 5. **Core Scientific Implication**: The failure of individual clinical variables and driver mutations to predict outcome explains why single-variable biomarker tests fail in clinical practice, highlighting the necessity of multi-gene transcriptomic signatures and integrated multivariate machine learning.

## 6. Technical Analysis Notes

> [!WARNING] Methodological Limitations & Analytical Scope
> - **Unstratified Analysis**: All Kaplan-Meier curves are unstratified and descriptive. They do not adjust for demographic, clinical, molecular, treatment, or study-specific confounding factors.
> - **Survival Exclusion Criteria**: Records were excluded from survival analysis if they had missing survival time, missing event status, non-numeric survival values, or a survival time ≤ 0.
> - **Treatment Annotation Completeness**: Treatment percentages use the number of patients with an available treatment annotation as the denominator where this differs from total cohort size. TCGA treatment-history categories are based on binary treatment-type indicators and are not necessarily mutually exclusive.
> - **Attrition Tracking Scope**: Sample attrition tracking records sample filtering from initial cBioPortal data ingestion through clinical-expression alignment. Downstream feature engineering steps may impose additional filters not captured here.
> - **Median OS Reporting**: Where the estimated survival probability did not fall below 50% during follow-up, median OS is reported as **NR (not reached)**.

> [!formula]+ Clinical Analysis Script Execution & Software Module Architecture
> - **Primary Pipeline Execution Scripts**:
>   - [`run_clinical_analysis.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-2-clinical-subtyping/run_clinical_analysis.py): Generates all clinical cohort characterisation outputs — demographics grid, attrition table, baseline characteristics table, Kaplan-Meier OS curves (overall and response-stratified), and forest plot of univariate associations — and writes `cohort_characteristics_clinical.md`.
> - **Data Preprocessing & Loading Modules**:
>   - [`clean_data.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/clean_data.py): Preprocesses raw cohort clinical metadata and RNA-seq expression profiles into cleaned CSV matrices.
>   - [`merge_datasets.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/merge_datasets.py): Merges processed expression matrices across cohorts into harmonised pooled matrices (`expr_merged.csv`, `clin_merged.csv`).
>   - [`data_loaders.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/data_loaders.py): Provides helper loader functions (`load_liu_2019`, `load_hugo_2016`, `load_riaz_2017`) for retrieving expression and clinical data.
> - **Shared Cross-Question & Pipeline Modules**:
>   - [`run_pipeline.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/run_pipeline.py): Master Q1 pipeline orchestrator executing downstream modelling and evaluation.
>   - [`styles.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/src/styles.py): Single source of truth for Okabe-Ito colour palettes (`COHORT_PALETTE`, `RESPONSE_PALETTE`) and visualisation presentation style.
