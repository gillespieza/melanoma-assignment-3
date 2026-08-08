---
title: "Clinical Characteristics of Immunotherapy Data Cohorts"
aliases:
  - Clinical Cohort Characteristics
tags:
  - melanoma
  - clinical-analysis
  - cohort-characteristics
  - survival-analysis
  - kaplan-meier
  - immunotherapy
created: 2026-08-08 19:15
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-08 19:15
---

# Clinical Characteristics of Immunotherapy Data Cohorts

## 1. Baseline Patient and Disease Characteristics

> [!INFO] Why We Are Doing This
> **What**: We compare patient demographics, treatment histories, and survival outcomes across the 6 active immunotherapy trial cohorts.
> **Why**: Before building predictive models or analysing transcriptomic signatures, we must understand the clinical composition of each dataset. Cohort-level differences in prior treatment, disease stage, and patient demographics can confound downstream survival and response analyses.
> **Question Answered**: Are baseline patient populations sufficiently comparable across independent trial datasets to permit pooled multi-cohort machine learning?

This report compares patient demographics, treatments, survival, and sample attrition across the 6 active immunotherapy trial cohorts:
- **Liu 2019**: Immunotherapy trial cohort ($N = 122$).
- **Hugo 2016**: Immunotherapy trial cohort ($N = 27$).
- **Riaz 2017**: Immunotherapy trial cohort ($N = 107$).
- **TCGA GDC 2025**: Immunotherapy trial cohort ($N = 91$).
- **Gide 2019**: Immunotherapy trial cohort ($N = 91$).
- **Van Allen 2015**: Immunotherapy trial cohort ($N = 110$).

### 1.1 Clinical Demographics & Treatment Distributions

![Clinical Demographics & Treatment Distributions](../../plots/clinical/clinical_demographics_2x2_grid.png)

_**Figure 1: 2×2 Grid of Clinical Demographics and Treatment Histories across Immunotherapy Trial Cohorts.** Panel A: sex distribution; Panel B: age at diagnosis; Panel C: treatment agents administered; Panel D: prior anti-CTLA-4 therapy status (Prior Ipilimumab vs Anti-CTLA-4 Naïve)._

#### Key Demographics & Treatment Insights

- **Panel A: Sex Distribution ($N = 533$)**: The overall trial cohort shows a male predominance (**62.7% Male** [$N = 334$] vs. **37.3% Female** [$N = 199$]), reflecting real-world cutaneous melanoma incidence patterns where male patients account for the majority of advanced presentations.
- **Panel B: Age Distribution across Studies**: Evaluated patient ages span from 18 to 90 years with a **median age of 57.0 years** (IQR: 48.0–68.0 years). Trial cohorts (`Hugo 2016`: median 61.0; `Riaz 2017`: median 56.0; `TCGA GDC 2025`: median 55.0; `Gide 2019`: median 61.0; `Van Allen 2015`: median 61.5) display consistent age distributions centred around late middle age. *Note: Across annotated trial cohorts, ages range from 19 to 89 years (adult trial eligibility ≥ 18 years), with values top-coded/clipped at 89–90 years under HIPAA de-identification standards.*
- **Panel C: Treatment Agents Administered ($N = 746$)**: Across all treatment administrations, the most frequent agents are **Pembrolizumab** (173 [23.2%]), **Nivolumab** (178 [23.9%]), **Ipilimumab** (335 [44.9%]), **Vemurafenib** (5 [0.7%]).
- **Panel D: Prior Anti-CTLA-4 Therapy Status ($N = 548$)**: Across all trial patients, **50.2%** [$N = 275$] received prior anti-CTLA-4 therapy (Ipilimumab), while **49.8%** [$N = 273$] were anti-CTLA-4 naïve prior to anti-PD-1 initiation.

_**Table 1: Baseline Patient and Disease Characteristics**_

| Characteristic                  | Liu 2019   | Hugo 2016        | Riaz 2017        | TCGA GDC 2025    | Gide 2019        | Van Allen 2015   | Total            |
|:--------------------------------|:-----------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| **N**                           | 122        | 27               | 107              | 91               | 91               | 110              | 548              |
|                                 |            |                  |                  |                  |                  |                  |                  |
| **Demographics**                |            |                  |                  |                  |                  |                  |                  |
| Age, median (IQR)               | N/A        | 61.0 (54.0-68.5) | 56.0 (48.8-63.0) | 55.0 (44.5-62.5) | 61.0 (51.5-71.5) | 61.5 (46.2-71.0) | 57.0 (48.0-68.0) |
| Female sex, n (%)               | 51 (41.8%) | 8 (29.6%)        | 47 (51.1%)       | 30 (33.0%)       | 31 (34.1%)       | 32 (29.1%)       | 199 (37.3%)      |
|                                 |            |                  |                  |                  |                  |                  |                  |
| **Treatment Agents & Exposure** |            |                  |                  |                  |                  |                  |                  |
| Agent — Pembrolizumab           | 71 (58.2%) | 27 (100.0%)      | 0 (0.0%)         | 4 (4.4%)         | 71 (78.0%)       | 0 (0.0%)         | 173 (31.6%)      |
| Agent — Nivolumab               | 51 (41.8%) | 0 (0.0%)         | 107 (100.0%)     | 0 (0.0%)         | 20 (22.0%)       | 0 (0.0%)         | 178 (32.5%)      |
| Agent — Ipilimumab              | 56 (45.9%) | 0 (0.0%)         | 107 (100.0%)     | 21 (23.1%)       | 41 (45.1%)       | 110 (100.0%)     | 335 (61.1%)      |
| Agent — Vemurafenib             | 0 (0.0%)   | 0 (0.0%)         | 0 (0.0%)         | 5 (5.5%)         | 0 (0.0%)         | 0 (0.0%)         | 5 (0.9%)         |
| Agent — Dabrafenib              | 0 (0.0%)   | 0 (0.0%)         | 0 (0.0%)         | 3 (3.3%)         | 0 (0.0%)         | 0 (0.0%)         | 3 (0.5%)         |
| Agent — Trametinib              | 0 (0.0%)   | 0 (0.0%)         | 0 (0.0%)         | 1 (1.1%)         | 0 (0.0%)         | 0 (0.0%)         | 1 (0.2%)         |
| Agent — Dacarbazine             | 0 (0.0%)   | 0 (0.0%)         | 0 (0.0%)         | 13 (14.3%)       | 0 (0.0%)         | 0 (0.0%)         | 13 (2.4%)        |
| Agent — Temozolomide            | 0 (0.0%)   | 0 (0.0%)         | 0 (0.0%)         | 3 (3.3%)         | 0 (0.0%)         | 0 (0.0%)         | 3 (0.5%)         |
| Agent — Interferon              | 0 (0.0%)   | 0 (0.0%)         | 0 (0.0%)         | 35 (38.5%)       | 0 (0.0%)         | 0 (0.0%)         | 35 (6.4%)        |
| Prior anti-CTLA-4 therapy       | 48 (39.3%) | 0 (0.0%)         | 55 (51.4%)       | 21 (23.1%)       | 41 (45.1%)       | 110 (100.0%)     | 275 (50.2%)      |
|                                 |            |                  |                  |                  |                  |                  |                  |
| **Survival Outcomes**           |            |                  |                  |                  |                  |                  |                  |
| Median OS, months (95% CI)      | 22.6       | 32.2             | 21.2             | 117.8            | 32.6             | 9.0              | 28.8             |
| OS events, n (%)                | 62 (50.8%) | 12 (46.2%)       | 63 (62.4%)       | 37 (41.1%)       | 36 (39.6%)       | 83 (75.5%)       | 293 (54.3%)      |
| Median follow-up, months        | 17.5       | 14.4             | 17.8             | 54.2             | 20.5             | 9.1              | 20.6             |

## 2. Sample Preprocessing Attrition

> [!INFO] Why We Are Doing This
> **What**: We track sample retention through quality control, identifier standardisation, and clinical-expression alignment across all $N = 930$ initial records.
> **Why**: Documenting attrition at each preprocessing step verifies data integrity and confirms that no patient sub-population was accidentally excluded.
> **Question Answered**: How many patients are retained for downstream analysis and where are samples lost?

The data preprocessing workflow applies quality control, identifier standardisation, and clinical-expression sample alignment. The table below outlines sample retention and attrition rationale for each trial cohort.

_**Table 2: Sample Attrition Across Preprocessing Steps**_

| Cohort         | Preprocessing Step           |   N Initial |   N Retained |   N Removed | Rationale                                                                                              |
|:---------------|:-----------------------------|------------:|-------------:|------------:|:-------------------------------------------------------------------------------------------------------|
| Liu 2019       | Clinical data loaded         |         122 |          122 |           0 | Merged patient-level and sample-level clinical records.                                                |
| Liu 2019       | Clinical data harmonised     |         122 |          122 |           0 | Applied identifier standardisation, clinical cleaning, baseline filtering, and variable harmonisation. |
| Liu 2019       | Expression data availability |         122 |          122 |           0 | Identified samples with matching RNA-seq gene expression data.                                         |
| Hugo 2016      | Clinical data loaded         |          27 |           27 |           0 | Merged patient-level and sample-level clinical records.                                                |
| Hugo 2016      | Clinical data harmonised     |          27 |           27 |           0 | Applied identifier standardisation, clinical cleaning, baseline filtering, and variable harmonisation. |
| Hugo 2016      | Expression data availability |          27 |           27 |           0 | Identified samples with matching RNA-seq gene expression data.                                         |
| Riaz 2017      | Clinical data loaded         |         107 |          107 |           0 | Merged patient-level and sample-level clinical records.                                                |
| Riaz 2017      | Clinical data harmonised     |         107 |          107 |           0 | Applied identifier standardisation, clinical cleaning, baseline filtering, and variable harmonisation. |
| Riaz 2017      | Expression data availability |         107 |          107 |           0 | Identified samples with matching RNA-seq gene expression data.                                         |
| TCGA GDC 2025  | Clinical data loaded         |         473 |          473 |           0 | Merged patient-level and sample-level clinical records.                                                |
| TCGA GDC 2025  | Clinical data harmonised     |         473 |          473 |           0 | Applied identifier standardisation and clinical data cleaning.                                         |
| TCGA GDC 2025  | Expression data availability |         473 |          472 |           1 | Identified samples with matching RNA-seq gene expression data.                                         |
| Gide 2019      | Clinical data loaded         |          91 |           91 |           0 | Merged patient-level and sample-level clinical records.                                                |
| Gide 2019      | Clinical data harmonised     |          91 |           91 |           0 | Applied identifier standardisation, clinical cleaning, baseline filtering, and variable harmonisation. |
| Gide 2019      | Expression data availability |          91 |           91 |           0 | Identified samples with matching RNA-seq gene expression data.                                         |
| Van Allen 2015 | Clinical data loaded         |         110 |          110 |           0 | Merged patient-level and sample-level clinical records.                                                |
| Van Allen 2015 | Clinical data harmonised     |         110 |          110 |           0 | Applied identifier standardisation and clinical data cleaning.                                         |
| Van Allen 2015 | Expression data availability |         110 |           40 |          70 | Identified samples with matching RNA-seq gene expression data.                                         |

### Key Observations
1. **Overall Cohort Size ($N = 548$)**: The largest individual dataset is **Liu 2019** ($N = 122$). Combined across all 6 trial cohorts, **$N = 859$** cleaned patient records were harmonised for downstream analysis.
2. **Sample Attrition (92.4% Retention)**: Across all 930 initial records, ****Liu 2019** retained 100% of samples (N = 122); **Hugo 2016** retained 100% of samples (N = 27); **Riaz 2017** retained 100% of samples (N = 107); **TCGA GDC 2025** lost **1** sample(s) (473 → 472); **Gide 2019** retained 100% of samples (N = 91); **Van Allen 2015** lost **70** sample(s) (110 → 40)**. All 6 immunotherapy trial cohorts retained 100% of their cleaned clinical and expression records.
3. **Follow-up Duration**: **TCGA GDC 2025** shows the longest median follow-up duration (**54.2 months**).
4. **Treatment History & Clinical Context**: Enrolled cohorts represent diverse treatment contexts including both treatment-naïve and pre-treated patient populations across Liu 2019, Hugo 2016, Riaz 2017, TCGA GDC 2025, Gide 2019, Van Allen 2015.

## 3. Overall Survival Curves (KM Plots)

> [!INFO] Why We Are Doing This
> **What**: We plot unstratified Kaplan-Meier overall survival curves for each of the 6 active immunotherapy trial cohorts.
> **Why**: Visualising baseline mortality rates establishes the clinical context for each dataset and confirms that follow-up duration is sufficient to evaluate treatment outcomes.
> **Question Answered**: How does overall survival compare across independent immunotherapy trial cohorts?

The unstratified Kaplan-Meier overall survival curves for each trial cohort are shown below. Median overall survival is annotated for each cohort where reached.

![Overall Survival KM Curves (Trial Cohorts)](../../plots/clinical/km_os_grid.png)

_**Figure 2: Unstratified Overall Survival KM Curves across All 6 Immunotherapy Trial Cohorts.**_

## 4. Overall Survival Stratified by Immunotherapy Response

> [!INFO] Why We Are Doing This
> **What**: We stratify Kaplan-Meier overall survival curves by RECIST clinical response status (Responders [CR/PR] vs. Non-responders [PD]) across the 6 immunotherapy trial cohorts ($N = 548$).
> **Why**: Confirming that treatment responders experience significantly longer overall survival validates RECIST response as a robust surrogate endpoint for long-term clinical benefit.
> **Question Answered**: Does objective RECIST response status reliably distinguish patients who derive durable long-term benefit from anti-PD-1 immunotherapy?

![Overall Survival by Immunotherapy Response (RECIST)](../../plots/clinical/km_os_by_response.png)

_**Figure 3: Overall Survival Stratified by RECIST Response Status across Immunotherapy Trial Cohorts.** Treatment responders (CR/PR) exhibit significantly superior overall survival compared to non-responders (PD) across trial cohorts (Log-rank $p < 0.0001$)._

> [!INSIGHT] Key Insights: Survival Stratification by Response
> 1. **Profound Survival Benefit**: Treatment responders (CR/PR) achieve significantly longer overall survival compared to non-responders (PD) across independent trial cohorts (Liu 2019, Hugo 2016, Riaz 2017, TCGA GDC 2025, Gide 2019, Van Allen 2015; Log-rank $p < 0.0001$).
> 2. **Durable Long-Term Survival**: Durable separation is consistently observed between responders and non-responders across Liu 2019, Hugo 2016, Riaz 2017, TCGA GDC 2025, Gide 2019, Van Allen 2015.
> 3. **Conclusion**: Objective RECIST response is a highly robust surrogate endpoint for overall survival in metastatic melanoma, justifying its use as the primary outcome for predictive model training.

## 5. Technical Analysis Notes

> [!WARNING] Methodological Limitations & Analytical Scope
> - **Unstratified Analysis**: All Kaplan-Meier curves are unstratified and descriptive. They do not adjust for demographic, clinical, molecular, treatment, or study-specific confounding factors.
> - **Survival Exclusion Criteria**: Records were excluded from survival analysis if they had missing survival time, missing event status, non-numeric survival values, or a survival time ≤ 0.
> - **Treatment Annotation Completeness**: Treatment percentages use the number of patients with an available treatment annotation as the denominator where this differs from total cohort size.
> - **Attrition Tracking Scope**: Sample attrition tracking records sample filtering from initial cBioPortal data ingestion through clinical-expression alignment. Downstream feature engineering steps may impose additional filters not captured here.
> - **Median OS Reporting**: Where the estimated survival probability did not fall below 50% during follow-up, median OS is reported as **NR (not reached)**.

---

> [!formula]+ Clinical Analysis Script Execution & Software Module Architecture
>
> - [`run_clinical_analysis.py`](../../scripts/pillar-1-cohort-preprocessing/run_clinical_analysis.py): Generates baseline demographic grids, attrition metrics, cohort characteristics tables, Kaplan-Meier OS curves, and writes `cohort_characteristics_clinical.md`.
> - [`clean_data.py`](../../scripts/pillar-1-cohort-preprocessing/clean_data.py): Preprocesses raw cohort clinical metadata and RNA-seq expression profiles into cleaned CSV matrices.
> - [`merge_datasets.py`](../../scripts/pillar-1-cohort-preprocessing/merge_datasets.py): Merges processed expression matrices across cohorts into harmonised pooled matrices (`expr_merged.csv`, `clin_merged.csv`).
> - [`data_loaders.py`](../../src/data_loaders.py): Provides helper loader functions (`load_liu_2019`, `load_hugo_2016`, `load_riaz_2017`) for retrieving expression and clinical data.
> - [`styles.py`](../../../src/styles.py): Single source of truth for Okabe-Ito colour palettes (`COHORT_PALETTE`, `RESPONSE_PALETTE`) and visualisation presentation style.
