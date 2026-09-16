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
created: 2026-09-16 13:01
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-09-16 13:01
---

# Clinical Characteristics of Immunotherapy Data Cohorts

## 1. Baseline Patient and Disease Characteristics

> [!INFO] Why We Are Doing This
> - **What**: Compare patient demographics and survival across 4 trial cohorts.
> - **Why**: Identify potential demographic confounders before predictive modeling.
> - **Questions**: Are patient populations comparable across cohorts?

This report compares patient demographics across active trial cohorts:
- **[Liu 2019](https://www.cbioportal.org/study/summary?id=mel_iatlas_liu_2019)**: Anti-PD-1 (pembrolizumab / nivolumab) ($N = 122$).
- **[Hugo 2016](https://www.cbioportal.org/study/summary?id=mel_iatlas_hugo_ucla_2016)**: Anti-PD-1 (pembrolizumab) ($N = 26$).
- **[Riaz 2017](https://www.cbioportal.org/study/summary?id=mel_iatlas_riaz_nivolumab_2017)**: Anti-PD-1 (nivolumab) ($N = 51$).
- **[Gide 2019](https://www.cbioportal.org/study/summary?id=mel_iatlas_gide_2019)**: Anti-PD-1 +/- anti-CTLA-4 (pembrolizumab / nivolumab +/- ipilimumab) ($N = 73$).

### 1.1 Clinical Demographics & Treatment Distributions

![Clinical Demographics](../../plots/clinical/clinical_demographics_2x2_grid.png)

_**Figure 1: 2×2 Grid of Clinical Demographics and Treatment Histories.**_

#### Key Demographics & Treatment Insights

- **Panel A: Sex Distribution ($N = 267$)**: The overall trial cohort shows a male predominance (**58.8% Male** [$N = 157$] vs. **41.2% Female** [$N = 110$]), reflecting real-world melanoma incidence.
- **Panel B: Age Distribution across Studies**: Evaluated patient ages span from 22 to 90 years with a **median age of 59.0 years** (IQR: 52.0–69.0 years). Trial cohorts (`Hugo 2016`: median 61.5; `Riaz 2017`: median 56.0; `Gide 2019`: median 62.0) display consistent age distributions.
- **Panel C: Treatment Agents Administered ($N = 622$)**: Most frequent agents are **Nivolumab** (246 [39.5%]), **Pembrolizumab** (221 [35.5%]), **Ipilimumab** (155 [24.9%]).
- **Panel D: Prior Anti-CTLA-4 Therapy Status ($N = 272$)**: Across all patients, **54.0%** [$N = 147$] received prior ipilimumab, while **46.0%** [$N = 125$] were anti-CTLA-4 naïve.

_**Table 1: Baseline Patient Characteristics**_

| Characteristic                  | Liu 2019     | Hugo 2016        | Riaz 2017        | Gide 2019        | Total            |
|:--------------------------------|:-------------|:-----------------|:-----------------|:-----------------|:-----------------|
| **N**                           | 122          | 26               | 51               | 73               | 272              |
|                                 |              |                  |                  |                  |                  |
| **Demographics**                |              |                  |                  |                  |                  |
| Age, median (IQR)               | N/A          | 61.5 (55.0-68.8) | 56.0 (49.0-62.8) | 62.0 (52.0-71.0) | 59.0 (52.0-69.0) |
| Female sex, n (%)               | 51 (41.8%)   | 8 (30.8%)        | 25 (54.3%)       | 26 (35.6%)       | 110 (41.2%)      |
|                                 |              |                  |                  |                  |                  |
| **Treatment Agents & Exposure** |              |                  |                  |                  |                  |
| Agent — Pembrolizumab           | 122 (100.0%) | 26 (100.0%)      | 0 (0.0%)         | 73 (100.0%)      | 221 (81.2%)      |
| Agent — Nivolumab               | 122 (100.0%) | 0 (0.0%)         | 51 (100.0%)      | 73 (100.0%)      | 246 (90.4%)      |
| Agent — Ipilimumab              | 56 (45.9%)   | 0 (0.0%)         | 26 (51.0%)       | 73 (100.0%)      | 155 (57.0%)      |
| Prior anti-CTLA-4 therapy       | 48 (39.3%)   | 0 (0.0%)         | 26 (51.0%)       | 73 (100.0%)      | 147 (54.0%)      |
|                                 |              |                  |                  |                  |                  |
| **Survival Outcomes**           |              |                  |                  |                  |                  |
| Median OS, months (95% CI)      | 22.6         | 32.2             | 20.8             | 35.0             | 27.1             |
| OS events, n (%)                | 62 (50.8%)   | 11 (44.0%)       | 34 (66.7%)       | 29 (39.7%)       | 136 (50.2%)      |
| Median follow-up, months        | 17.5         | 14.4             | 15.9             | 20.5             | 18.1             |

## 2. Sample Preprocessing Attrition

> [!INFO] Why We Are Doing This
> - **What**: We track sample retention through quality control across $N = 347$ initial records.
> - **Why**: Documenting attrition at each step verifies data integrity.
> - **Questions**: How many patients are retained for downstream analysis?

_**Table 2: Sample Attrition Across Preprocessing Steps**_

| Cohort    | Preprocessing Step           |   N Initial |   N Retained |   N Removed | Rationale                                                                                              |
|:----------|:-----------------------------|------------:|-------------:|------------:|:-------------------------------------------------------------------------------------------------------|
| Liu 2019  | Clinical data loaded         |         122 |          122 |           0 | Merged patient-level and sample-level clinical records.                                                |
| Liu 2019  | Clinical data harmonised     |         122 |          122 |           0 | Applied identifier standardisation, clinical cleaning, baseline filtering, and variable harmonisation. |
| Liu 2019  | Expression data availability |         122 |          122 |           0 | Identified samples with matching RNA-seq gene expression data.                                         |
| Hugo 2016 | Clinical data loaded         |          27 |           27 |           0 | Merged patient-level and sample-level clinical records.                                                |
| Hugo 2016 | Clinical data harmonised     |          27 |           26 |           1 | Applied identifier standardisation, clinical cleaning, baseline filtering, and variable harmonisation. |
| Hugo 2016 | Expression data availability |          26 |           26 |           0 | Identified samples with matching RNA-seq gene expression data.                                         |
| Riaz 2017 | Clinical data loaded         |         107 |          107 |           0 | Merged patient-level and sample-level clinical records.                                                |
| Riaz 2017 | Clinical data harmonised     |         107 |           51 |          56 | Applied identifier standardisation, clinical cleaning, baseline filtering, and variable harmonisation. |
| Riaz 2017 | Expression data availability |          51 |           51 |           0 | Identified samples with matching RNA-seq gene expression data.                                         |
| Gide 2019 | Clinical data loaded         |          91 |           91 |           0 | Merged patient-level and sample-level clinical records.                                                |
| Gide 2019 | Clinical data harmonised     |          91 |           73 |          18 | Applied identifier standardisation, clinical cleaning, baseline filtering, and variable harmonisation. |
| Gide 2019 | Expression data availability |          73 |           73 |           0 | Identified samples with matching RNA-seq gene expression data.                                         |

### Key Observations
1. **Overall Cohort Size ($N = 272$)**: Largest dataset is **Liu 2019** ($N = 122$). Combined across all 4 cohorts, **$N = 272$** cleaned records were harmonised.
2. **Sample Attrition (78.4% Retention)**: Across all 347 initial records, ****Liu 2019** retained 100% of samples (N = 122); **Hugo 2016** lost **1** sample(s) (27 → 26); **Riaz 2017** lost **56** sample(s) (107 → 51); **Gide 2019** lost **18** sample(s) (91 → 73)**.
3. **Follow-up Duration**: **Gide 2019** displays median follow-up of **20.5 months**.
4. **Treatment History**: Enrolled cohorts represent diverse treatment contexts across Liu 2019, Hugo 2016, Riaz 2017, Gide 2019.

## 3. Overall Survival Curves (KM Plots)

> [!INFO] Why We Are Doing This
> - **What**: We plot unstratified Kaplan-Meier OS curves for each of the 4 active trial cohorts.
> - **Why**: Visualising baseline mortality rates establishes the clinical context for each dataset.
> - **Questions**: How does overall survival compare across independent immunotherapy trial cohorts?

![Overall Survival KM Curves](../../plots/clinical/km_os_grid.png)

_**Figure 2: Unstratified Overall Survival KM Curves across All 4 Immunotherapy Trial Cohorts.**_

## 4. Overall Survival Stratified by Reported Immunotherapy Response

> [!INFO] Why We Are Doing This
> - **What**: We stratify KM overall survival curves by the reported immunotherapy response classification ($N = 272$).
> - **Why**: We compare survival patterns between response-defined groups while preserving each cohort's original response terminology.
> - **Questions**: Do reported response groups show different overall survival patterns across cohorts?

![Overall Survival by Response](../../plots/clinical/km_os_by_response.png)

_**Figure 3: Overall Survival Stratified by Reported Immunotherapy Response Classification.**_

> [!INSIGHT] Key Insights: Survival Stratification by Response
> 1. **Response-stratified analysis**: Curves are shown for the available responder and non-responder classifications across Liu 2019, Hugo 2016, Riaz 2017, Gide 2019.
> 2. **Terminology note**: Response definitions are cohort-specific; Hugo 2016 does not use RECIST terminology, so these curves should not be interpreted as a pooled RECIST analysis.
> 3. **Interpretation**: Statistical significance and surrogate-endpoint claims must be taken from the companion analysis results.

## 5. Technical Analysis Notes

> [!WARNING] Methodological Limitations & Analytical Scope
> - **Unstratified Analysis**: All KM curves are unstratified and descriptive.
> - **Exclusion Criteria**: Excluded missing survival time, missing event status, or survival time ≤ 0.
> - **Median OS Reporting**: Median OS is reported as **NR (not reached)** where survival probability remained above 50%.

---

> [!formula]+ Clinical Analysis Script Execution & Software Module Architecture
>
> - [`run_clinical_analysis.py`](../../scripts/pillar_1_cohort_preprocessing/run_clinical_analysis.py): Generates baseline demographic grids, attrition metrics, cohort characteristics tables, Kaplan-Meier OS curves, and writes `cohort_characteristics_clinical.md`.
> - [`clean_data.py`](../../scripts/pillar_1_cohort_preprocessing/clean_data.py): Preprocesses raw cohort clinical metadata and RNA-seq expression profiles into cleaned CSV matrices.
> - [`merge_datasets.py`](../../scripts/pillar_1_cohort_preprocessing/merge_datasets.py): Merges processed expression matrices across cohorts into harmonised pooled matrices (`expr_merged.csv`, `clin_merged.csv`).
> - [`data_loaders.py`](../../src/data_loaders.py): Provides `load_cohort_by_name` (canonical dynamic entry point — resolves any cohort from `datasets.yaml` without code changes), `load_dataset_by_config`, `load_all_active_cohorts`, and `load_merged_immunotherapy`. Legacy per-cohort shims are retained for backwards compatibility.
> - [`styles.py`](../../../src/styles.py): Single source of truth for Okabe-Ito colour palettes (`COHORT_PALETTE`, `RESPONSE_PALETTE`) and visualisation presentation style.
