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
created: 2026-08-09 15:15
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-09 15:15
---

# Clinical Characteristics of Immunotherapy Data Cohorts

## 1. Baseline Patient and Disease Characteristics

> [!INFO] Why We Are Doing This
> - **What**: Compare patient demographics and survival across 6 trial cohorts.
> - **Why**: Identify potential demographic confounders before predictive modeling.
> - **Questions**: Are patient populations comparable across cohorts?

This report compares patient demographics across active trial cohorts:
- **Liu 2019**: Anti-PD-1 (pembrolizumab / nivolumab) ($N = 122$).
- **Hugo 2016**: Anti-PD-1 (pembrolizumab) ($N = 27$).
- **Riaz 2017**: Anti-PD-1 (nivolumab) ($N = 107$).
- **TCGA GDC 2025**: Heterogeneous IT-treated cohort (ipilimumab, vaccines, interferon +/- chemotherapy) ($N = 91$).
- **Gide 2019**: Anti-PD-1 +/- anti-CTLA-4 (pembrolizumab / nivolumab +/- ipilimumab) ($N = 91$).
- **Van Allen 2015**: Anti-CTLA-4 (ipilimumab) ($N = 110$).

### 1.1 Clinical Demographics & Treatment Distributions

![Clinical Demographics](../../plots/clinical/clinical_demographics_2x2_grid.png)

_**Figure 1: 2×2 Grid of Clinical Demographics and Treatment Histories.**_

#### Key Demographics & Treatment Insights

- **Panel A: Sex Distribution ($N = 533$)**: The overall trial cohort shows a male predominance (**62.7% Male** [$N = 334$] vs. **37.3% Female** [$N = 199$]), reflecting real-world melanoma incidence.
- **Panel B: Age Distribution across Studies**: Evaluated patient ages span from 18 to 90 years with a **median age of 57.0 years** (IQR: 48.0–68.0 years). Trial cohorts (`Hugo 2016`: median 61.0; `Riaz 2017`: median 56.0; `TCGA GDC 2025`: median 55.0; `Gide 2019`: median 61.0; `Van Allen 2015`: median 61.5) display consistent age distributions.
- **Panel C: Treatment Agents Administered ($N = 957$)**: Most frequent agents are **Pembrolizumab** (244 [25.5%]), **Nivolumab** (320 [33.4%]), **Ipilimumab** (333 [34.8%]), **Vemurafenib** (5 [0.5%]).
- **Panel D: Prior Anti-CTLA-4 Therapy Status ($N = 548$)**: Across all patients, **59.3%** [$N = 325$] received prior ipilimumab, while **40.7%** [$N = 223$] were anti-CTLA-4 naïve.

_**Table 1: Baseline Patient Characteristics**_

| Characteristic                  | Liu 2019     | Hugo 2016        | Riaz 2017        | TCGA GDC 2025    | Gide 2019        | Van Allen 2015   | Total            |
|:--------------------------------|:-------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| **N**                           | 122          | 27               | 107              | 91               | 91               | 110              | 548              |
|                                 |              |                  |                  |                  |                  |                  |                  |
| **Demographics**                |              |                  |                  |                  |                  |                  |                  |
| Age, median (IQR)               | N/A          | 61.0 (54.0-68.5) | 56.0 (48.8-63.0) | 55.0 (44.5-62.5) | 61.0 (51.5-71.5) | 61.5 (46.2-71.0) | 57.0 (48.0-68.0) |
| Female sex, n (%)               | 51 (41.8%)   | 8 (29.6%)        | 47 (51.1%)       | 30 (33.0%)       | 31 (34.1%)       | 32 (29.1%)       | 199 (37.3%)      |
|                                 |              |                  |                  |                  |                  |                  |                  |
| **Treatment Agents & Exposure** |              |                  |                  |                  |                  |                  |                  |
| Agent — Pembrolizumab           | 122 (100.0%) | 27 (100.0%)      | 0 (0.0%)         | 4 (4.4%)         | 91 (100.0%)      | 0 (0.0%)         | 244 (44.5%)      |
| Agent — Nivolumab               | 122 (100.0%) | 0 (0.0%)         | 107 (100.0%)     | 0 (0.0%)         | 91 (100.0%)      | 0 (0.0%)         | 320 (58.4%)      |
| Agent — Ipilimumab              | 56 (45.9%)   | 0 (0.0%)         | 55 (51.4%)       | 21 (23.1%)       | 91 (100.0%)      | 110 (100.0%)     | 333 (60.8%)      |
| Agent — Vemurafenib             | 0 (0.0%)     | 0 (0.0%)         | 0 (0.0%)         | 5 (5.5%)         | 0 (0.0%)         | 0 (0.0%)         | 5 (0.9%)         |
| Agent — Dabrafenib              | 0 (0.0%)     | 0 (0.0%)         | 0 (0.0%)         | 3 (3.3%)         | 0 (0.0%)         | 0 (0.0%)         | 3 (0.5%)         |
| Agent — Trametinib              | 0 (0.0%)     | 0 (0.0%)         | 0 (0.0%)         | 1 (1.1%)         | 0 (0.0%)         | 0 (0.0%)         | 1 (0.2%)         |
| Agent — Dacarbazine             | 0 (0.0%)     | 0 (0.0%)         | 0 (0.0%)         | 13 (14.3%)       | 0 (0.0%)         | 0 (0.0%)         | 13 (2.4%)        |
| Agent — Temozolomide            | 0 (0.0%)     | 0 (0.0%)         | 0 (0.0%)         | 3 (3.3%)         | 0 (0.0%)         | 0 (0.0%)         | 3 (0.5%)         |
| Agent — Interferon              | 0 (0.0%)     | 0 (0.0%)         | 0 (0.0%)         | 35 (38.5%)       | 0 (0.0%)         | 0 (0.0%)         | 35 (6.4%)        |
| Prior anti-CTLA-4 therapy       | 48 (39.3%)   | 0 (0.0%)         | 55 (51.4%)       | 21 (23.1%)       | 91 (100.0%)      | 110 (100.0%)     | 325 (59.3%)      |
|                                 |              |                  |                  |                  |                  |                  |                  |
| **Survival Outcomes**           |              |                  |                  |                  |                  |                  |                  |
| Median OS, months (95% CI)      | 22.6         | 32.2             | 21.2             | 117.8            | 32.6             | 9.0              | 28.8             |
| OS events, n (%)                | 62 (50.8%)   | 12 (46.2%)       | 63 (62.4%)       | 37 (41.1%)       | 36 (39.6%)       | 83 (75.5%)       | 293 (54.3%)      |
| Median follow-up, months        | 17.5         | 14.4             | 17.8             | 54.2             | 20.5             | 9.1              | 20.6             |

## 2. Sample Preprocessing Attrition

> [!INFO] Why We Are Doing This
> - **What**: We track sample retention through quality control across $N = 930$ initial records.
> - **Why**: Documenting attrition at each step verifies data integrity.
> - **Questions**: How many patients are retained for downstream analysis?

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
1. **Overall Cohort Size ($N = 548$)**: Largest dataset is **Liu 2019** ($N = 122$). Combined across all 6 cohorts, **$N = 859$** cleaned records were harmonised.
2. **Sample Attrition (92.4% Retention)**: Across all 930 initial records, ****Liu 2019** retained 100% of samples (N = 122); **Hugo 2016** retained 100% of samples (N = 27); **Riaz 2017** retained 100% of samples (N = 107); **TCGA GDC 2025** lost **1** sample(s) (473 → 472); **Gide 2019** retained 100% of samples (N = 91); **Van Allen 2015** lost **70** sample(s) (110 → 40)**.
3. **Follow-up Duration**: **TCGA GDC 2025** displays median follow-up of **54.2 months**.
4. **Treatment History**: Enrolled cohorts represent diverse treatment contexts across Liu 2019, Hugo 2016, Riaz 2017, TCGA GDC 2025, Gide 2019, Van Allen 2015.

## 3. Overall Survival Curves (KM Plots)

> [!INFO] Why We Are Doing This
> - **What**: We plot unstratified Kaplan-Meier OS curves for each of the 6 active trial cohorts.
> - **Why**: Visualising baseline mortality rates establishes the clinical context for each dataset.
> - **Questions**: How does overall survival compare across independent immunotherapy trial cohorts?

![Overall Survival KM Curves](../../plots/clinical/km_os_grid.png)

_**Figure 2: Unstratified Overall Survival KM Curves across All 6 Immunotherapy Trial Cohorts.**_

## 4. Overall Survival Stratified by Immunotherapy Response

> [!INFO] Why We Are Doing This
> - **What**: We stratify KM overall survival curves by RECIST response status ($N = 548$).
> - **Why**: Confirming that responders experience significantly longer OS validates RECIST response as a surrogate endpoint.
> - **Questions**: Does RECIST response reliably distinguish durable long-term benefit?

![Overall Survival by Response](../../plots/clinical/km_os_by_response.png)

_**Figure 3: Overall Survival Stratified by RECIST Response Status.**_

> [!INSIGHT] Key Insights: Survival Stratification by Response
> 1. **Survival Benefit**: Responders (CR/PR) achieve significantly longer OS vs non-responders (PD) (Liu 2019, Hugo 2016, Riaz 2017, TCGA GDC 2025, Gide 2019, Van Allen 2015; Log-rank $p < 0.0001$).
> 2. **Surrogate Validation**: Objective RECIST response is a robust surrogate endpoint for overall survival.

## 5. Technical Analysis Notes

> [!WARNING] Methodological Limitations & Analytical Scope
> - **Unstratified Analysis**: All KM curves are unstratified and descriptive.
> - **Exclusion Criteria**: Excluded missing survival time, missing event status, or survival time ≤ 0.
> - **Median OS Reporting**: Median OS is reported as **NR (not reached)** where survival probability remained above 50%.

---

> [!formula]+ Clinical Analysis Script Execution & Software Module Architecture
>
> - [`run_clinical_analysis.py`](../../scripts/pillar-1-cohort-preprocessing/run_clinical_analysis.py): Generates baseline demographic grids, attrition metrics, cohort characteristics tables, Kaplan-Meier OS curves, and writes `cohort_characteristics_clinical.md`.
> - [`clean_data.py`](../../scripts/pillar-1-cohort-preprocessing/clean_data.py): Preprocesses raw cohort clinical metadata and RNA-seq expression profiles into cleaned CSV matrices.
> - [`merge_datasets.py`](../../scripts/pillar-1-cohort-preprocessing/merge_datasets.py): Merges processed expression matrices across cohorts into harmonised pooled matrices (`expr_merged.csv`, `clin_merged.csv`).
> - [`data_loaders.py`](../../src/data_loaders.py): Provides `load_cohort_by_name` (canonical dynamic entry point — resolves any cohort from `datasets.yaml` without code changes), `load_dataset_by_config`, `load_all_active_cohorts`, and `load_merged_immunotherapy`. Legacy per-cohort shims are retained for backwards compatibility.
> - [`styles.py`](../../../src/styles.py): Single source of truth for Okabe-Ito colour palettes (`COHORT_PALETTE`, `RESPONSE_PALETTE`) and visualisation presentation style.
