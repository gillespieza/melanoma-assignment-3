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
created: 2026-09-15 11:50
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-09-15 11:50
---

# Clinical Characteristics of Immunotherapy Data Cohorts

## 1. Baseline Patient and Disease Characteristics

> [!INFO] Why We Are Doing This
> - **What**: Compare patient demographics and survival across 5 trial cohorts.
> - **Why**: Identify potential demographic confounders before predictive modeling.
> - **Questions**: Are patient populations comparable across cohorts?

This report compares patient demographics across active trial cohorts:
- **[Liu 2019](https://www.cbioportal.org/study/summary?id=mel_iatlas_liu_2019)**: Anti-PD-1 (pembrolizumab / nivolumab) ($N = 122$).
- **[Hugo 2016](https://www.cbioportal.org/study/summary?id=mel_iatlas_hugo_ucla_2016)**: Anti-PD-1 (pembrolizumab) ($N = 27$).
- **[Riaz 2017](https://www.cbioportal.org/study/summary?id=mel_iatlas_riaz_nivolumab_2017)**: Anti-PD-1 (nivolumab) ($N = 107$).
- **[TCGA GDC 2025](https://www.cbioportal.org/study/summary?id=skcm_tcga_gdc)**: Heterogeneous IT-treated cohort (ipilimumab, vaccines, interferon +/- chemotherapy) ($N = 91$).
- **[Gide 2019](https://www.cbioportal.org/study/summary?id=mel_iatlas_gide_2019)**: Anti-PD-1 +/- anti-CTLA-4 (pembrolizumab / nivolumab +/- ipilimumab) ($N = 91$).

### 1.1 Clinical Demographics & Treatment Distributions

![Clinical Demographics](../../plots/clinical/clinical_demographics_2x2_grid.png)

_**Figure 1: 2×2 Grid of Clinical Demographics and Treatment Histories.**_

#### Key Demographics & Treatment Insights

- **Panel A: Sex Distribution ($N = 423$)**: The overall trial cohort shows a male predominance (**60.5% Male** [$N = 256$] vs. **39.5% Female** [$N = 167$]), reflecting real-world melanoma incidence.
- **Panel B: Age Distribution across Studies**: Evaluated patient ages span from 19 to 90 years with a **median age of 56.0 years** (IQR: 48.0–66.0 years). Trial cohorts (`Hugo 2016`: median 61.0; `Riaz 2017`: median 56.0; `TCGA GDC 2025`: median 55.0; `Gide 2019`: median 61.0) display consistent age distributions.
- **Panel C: Treatment Agents Administered ($N = 847$)**: Most frequent agents are **Pembrolizumab** (244 [28.8%]), **Nivolumab** (320 [37.8%]), **Ipilimumab** (223 [26.3%]), **Vemurafenib** (5 [0.6%]).
- **Panel D: Prior Anti-CTLA-4 Therapy Status ($N = 438$)**: Across all patients, **49.1%** [$N = 215$] received prior ipilimumab, while **50.9%** [$N = 223$] were anti-CTLA-4 naïve.

_**Table 1: Baseline Patient Characteristics**_

| Characteristic                  | Liu 2019     | Hugo 2016        | Riaz 2017        | TCGA GDC 2025    | Gide 2019        | Total            |
|:--------------------------------|:-------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| **N**                           | 122          | 27               | 107              | 91               | 91               | 438              |
|                                 |              |                  |                  |                  |                  |                  |
| **Demographics**                |              |                  |                  |                  |                  |                  |
| Age, median (IQR)               | N/A          | 61.0 (54.0-68.5) | 56.0 (48.8-63.0) | 55.0 (44.5-62.5) | 61.0 (51.5-71.5) | 56.0 (48.0-66.0) |
| Female sex, n (%)               | 51 (41.8%)   | 8 (29.6%)        | 47 (51.1%)       | 30 (33.0%)       | 31 (34.1%)       | 167 (39.5%)      |
|                                 |              |                  |                  |                  |                  |                  |
| **Treatment Agents & Exposure** |              |                  |                  |                  |                  |                  |
| Agent — Pembrolizumab           | 122 (100.0%) | 27 (100.0%)      | 0 (0.0%)         | 4 (4.4%)         | 91 (100.0%)      | 244 (55.7%)      |
| Agent — Nivolumab               | 122 (100.0%) | 0 (0.0%)         | 107 (100.0%)     | 0 (0.0%)         | 91 (100.0%)      | 320 (73.1%)      |
| Agent — Ipilimumab              | 56 (45.9%)   | 0 (0.0%)         | 55 (51.4%)       | 21 (23.1%)       | 91 (100.0%)      | 223 (50.9%)      |
| Agent — Vemurafenib             | 0 (0.0%)     | 0 (0.0%)         | 0 (0.0%)         | 5 (5.5%)         | 0 (0.0%)         | 5 (1.1%)         |
| Agent — Dabrafenib              | 0 (0.0%)     | 0 (0.0%)         | 0 (0.0%)         | 3 (3.3%)         | 0 (0.0%)         | 3 (0.7%)         |
| Agent — Trametinib              | 0 (0.0%)     | 0 (0.0%)         | 0 (0.0%)         | 1 (1.1%)         | 0 (0.0%)         | 1 (0.2%)         |
| Agent — Dacarbazine             | 0 (0.0%)     | 0 (0.0%)         | 0 (0.0%)         | 13 (14.3%)       | 0 (0.0%)         | 13 (3.0%)        |
| Agent — Temozolomide            | 0 (0.0%)     | 0 (0.0%)         | 0 (0.0%)         | 3 (3.3%)         | 0 (0.0%)         | 3 (0.7%)         |
| Agent — Interferon              | 0 (0.0%)     | 0 (0.0%)         | 0 (0.0%)         | 35 (38.5%)       | 0 (0.0%)         | 35 (8.0%)        |
| Prior anti-CTLA-4 therapy       | 48 (39.3%)   | 0 (0.0%)         | 55 (51.4%)       | 21 (23.1%)       | 91 (100.0%)      | 215 (49.1%)      |
|                                 |              |                  |                  |                  |                  |                  |
| **Survival Outcomes**           |              |                  |                  |                  |                  |                  |
| Median OS, months (95% CI)      | 22.6         | 32.2             | 21.2             | 117.8            | 32.6             | 36.2             |
| OS events, n (%)                | 62 (50.8%)   | 12 (46.2%)       | 63 (62.4%)       | 37 (41.1%)       | 36 (39.6%)       | 210 (48.8%)      |
| Median follow-up, months        | 17.5         | 14.4             | 17.8             | 54.2             | 20.5             | 21.4             |

## 2. Sample Preprocessing Attrition

> [!INFO] Why We Are Doing This
> - **What**: We track sample retention through quality control across $N = 820$ initial records.
> - **Why**: Documenting attrition at each step verifies data integrity.
> - **Questions**: How many patients are retained for downstream analysis?

_**Table 2: Sample Attrition Across Preprocessing Steps**_

| Cohort        | Preprocessing Step           |   N Initial |   N Retained |   N Removed | Rationale                                                                                              |
|:--------------|:-----------------------------|------------:|-------------:|------------:|:-------------------------------------------------------------------------------------------------------|
| Liu 2019      | Clinical data loaded         |         122 |          122 |           0 | Merged patient-level and sample-level clinical records.                                                |
| Liu 2019      | Clinical data harmonised     |         122 |          122 |           0 | Applied identifier standardisation, clinical cleaning, baseline filtering, and variable harmonisation. |
| Liu 2019      | Expression data availability |         122 |          122 |           0 | Identified samples with matching RNA-seq gene expression data.                                         |
| Hugo 2016     | Clinical data loaded         |          27 |           27 |           0 | Merged patient-level and sample-level clinical records.                                                |
| Hugo 2016     | Clinical data harmonised     |          27 |           27 |           0 | Applied identifier standardisation, clinical cleaning, baseline filtering, and variable harmonisation. |
| Hugo 2016     | Expression data availability |          27 |           27 |           0 | Identified samples with matching RNA-seq gene expression data.                                         |
| Riaz 2017     | Clinical data loaded         |         107 |          107 |           0 | Merged patient-level and sample-level clinical records.                                                |
| Riaz 2017     | Clinical data harmonised     |         107 |          107 |           0 | Applied identifier standardisation, clinical cleaning, baseline filtering, and variable harmonisation. |
| Riaz 2017     | Expression data availability |         107 |          107 |           0 | Identified samples with matching RNA-seq gene expression data.                                         |
| TCGA GDC 2025 | Clinical data loaded         |         473 |          473 |           0 | Merged patient-level and sample-level clinical records.                                                |
| TCGA GDC 2025 | Clinical data harmonised     |         473 |          473 |           0 | Applied identifier standardisation and clinical data cleaning.                                         |
| TCGA GDC 2025 | Expression data availability |         473 |          472 |           1 | Identified samples with matching RNA-seq gene expression data.                                         |
| Gide 2019     | Clinical data loaded         |          91 |           91 |           0 | Merged patient-level and sample-level clinical records.                                                |
| Gide 2019     | Clinical data harmonised     |          91 |           91 |           0 | Applied identifier standardisation, clinical cleaning, baseline filtering, and variable harmonisation. |
| Gide 2019     | Expression data availability |          91 |           91 |           0 | Identified samples with matching RNA-seq gene expression data.                                         |

### Key Observations
1. **Overall Cohort Size ($N = 438$)**: Largest dataset is **Liu 2019** ($N = 122$). Combined across all 5 cohorts, **$N = 819$** cleaned records were harmonised.
2. **Sample Attrition (99.9% Retention)**: Across all 820 initial records, ****Liu 2019** retained 100% of samples (N = 122); **Hugo 2016** retained 100% of samples (N = 27); **Riaz 2017** retained 100% of samples (N = 107); **TCGA GDC 2025** lost **1** sample(s) (473 → 472); **Gide 2019** retained 100% of samples (N = 91)**.
3. **Follow-up Duration**: **TCGA GDC 2025** displays median follow-up of **54.2 months**.
4. **Treatment History**: Enrolled cohorts represent diverse treatment contexts across Liu 2019, Hugo 2016, Riaz 2017, TCGA GDC 2025, Gide 2019.

## 3. Overall Survival Curves (KM Plots)

> [!INFO] Why We Are Doing This
> - **What**: We plot unstratified Kaplan-Meier OS curves for each of the 5 active trial cohorts.
> - **Why**: Visualising baseline mortality rates establishes the clinical context for each dataset.
> - **Questions**: How does overall survival compare across independent immunotherapy trial cohorts?

![Overall Survival KM Curves](../../plots/clinical/km_os_grid.png)

_**Figure 2: Unstratified Overall Survival KM Curves across All 5 Immunotherapy Trial Cohorts.**_

## 4. Overall Survival Stratified by Immunotherapy Response

> [!INFO] Why We Are Doing This
> - **What**: We stratify KM overall survival curves by RECIST response status ($N = 438$).
> - **Why**: Confirming that responders experience significantly longer OS validates RECIST response as a surrogate endpoint.
> - **Questions**: Does RECIST response reliably distinguish durable long-term benefit?

![Overall Survival by Response](../../plots/clinical/km_os_by_response.png)

_**Figure 3: Overall Survival Stratified by RECIST Response Status.**_

> [!INSIGHT] Key Insights: Survival Stratification by Response
> 1. **Survival Benefit**: Responders (CR/PR) achieve significantly longer OS vs non-responders (PD) (Liu 2019, Hugo 2016, Riaz 2017, TCGA GDC 2025, Gide 2019; Log-rank $p < 0.0001$).
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
