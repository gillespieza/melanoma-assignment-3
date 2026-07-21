---
title: Clinical Characteristics of Data Cohorts
aliases:
  - Clinical Cohort Characteristics
tags:
  - melanoma
  - clinical-analysis
  - cohort-characteristics
  - survival-analysis
created: 2026-07-21 19:30
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-21 19:30
---

# Clinical Characteristics of Data Cohorts

This report provides a comparative summary of the patient demographic, treatment and survival characteristics across the four cohorts analysed in this study:

*   **TCGA-SKCM**: Baseline reference cohort with recorded treatment history.
*   **Liu 2019**: Advanced melanoma trial cohort treated with anti-PD-1 monotherapy.
*   **Hugo 2016**: Anti-PD-1 clinical trial cohort.
*   **Riaz 2017**: Anti-PD-1 clinical trial cohort treated with nivolumab.

The demographic and treatment characteristics were calculated from the available clinical cohort data. Survival characteristics were calculated from the cleaned clinical data used in the Kaplan-Meier analysis.

_**Table 1: Baseline Patient and Disease Characteristics**_

| Characteristic                | Liu 2019   | Hugo 2016        | Riaz 2017        | TCGA-SKCM        |
|:------------------------------|:-----------|:-----------------|:-----------------|:-----------------|
| **N**                         | 122        | 27               | 107              | 448              |
|                               |            |                  |                  |                  |
| **Demographics**              |            |                  |                  |                  |
| Age, median (IQR)             | N/A        | 61.0 (54.0–68.5) | 56.0 (48.8–63.0) | 57.0 (47.0–70.0) |
| Female sex, n (%)             | 51 (41.8%) | 8 (29.6%)        | 47 (51.1%)       | 171 (38.2%)      |
|                               |            |                  |                  |                  |
| **Treatment**                 |            |                  |                  |                  |
| ICI agent — Pembrolizumab     | 71 (58.2%) | 0 (0.0%)         | —                | 3 (0.7%)         |
| ICI agent — Nivolumab         | 51 (41.8%) | 0 (0.0%)         | 0 (0.0%)         | 1 (0.2%)         |
| Prior anti-CTLA-4             | 0 (0.0%)   | —                | 55 (100.0%)      | —                |
| Treatment History (TCGA only) | —          | —                | —                |                  |
| — Radiation Therapy           | —          | —                | —                | 114 (25.4%)      |
| — Immunotherapy               | —          | —                | —                | 71 (15.8%)       |
| — Chemotherapy                | —          | —                | —                | 68 (15.2%)       |
| — Vaccine                     | —          | —                | —                | 22 (4.9%)        |
| — Targeted Therapy            | —          | —                | —                | 12 (2.7%)        |
| — Other Therapy               | —          | —                | —                | 7 (1.6%)         |
| — No recorded treatment       | —          | —                | —                | 244 (54.5%)      |
|                               |            |                  |                  |                  |
| **Survival Outcomes**         |            |                  |                  |                  |
| Median OS, months (95% CI)    | 22.6       | 32.2             | 21.2             | 74.7             |
| OS events, n (%)              | 62 (50.8%) | 12 (46.2%)       | 63 (62.4%)       | 215 (49.8%)      |
| Median follow-up, months      | 17.5       | 14.4             | 17.8             | 40.1             |

## Key Observations

1. **Cohort Size**: The largest cohort is **TCGA-SKCM** with **N = 448** patients. The TCGA-SKCM cohort serves as a genomic reference population with detailed treatment history. The three IO cohorts are anti-PD-1 clinical trial cohorts.

2. **Overall Survival**: Median OS was estimated using the Kaplan-Meier survival function. Where the estimated survival probability did not fall below 50% during follow-up, median OS was not reached.

3. **Follow-up**: **TCGA-SKCM** had the longest median follow-up at **40.1 months**.

4. **Event Rates**: Observed OS event rates varied between cohorts, reflecting differences in cohort composition, disease stage, treatment context, follow-up duration, and censoring.

5. **Missing Data**: Age, sex and treatment variables have different levels of availability across cohorts. These differences should be considered when comparing clinical characteristics across datasets.

6. **Interpretation**: These unstratified survival estimates provide a descriptive baseline for subsequent molecular and predictive analyses. Cross-cohort comparisons should be interpreted cautiously because the cohorts differ in clinical context and study design.

---

## Overall Survival Curves (KM Plots)

The unstratified Kaplan-Meier overall survival curves for each cohort are shown below. Median overall survival is annotated for each cohort where reached.

![[km_os_grid.png]]

_**Overall Survival KM Curves (All Cohorts)**_

---

## Analysis Notes

Overall survival was analysed using the available cohort-specific survival duration and event-status variables.

Records were excluded from the survival analysis if they had:

*   Missing survival time.
*   Missing survival event status.
*   Non-numeric survival values.
*   A survival time less than or equal to zero.

Median overall survival was estimated using the Kaplan-Meier survival function. Where the estimated survival probability did not fall below 50% during follow-up, median OS was reported as **NR (not reached)**.

Treatment percentages use the number of patients with an available treatment annotation as the denominator where this differs from the total cohort size. TCGA treatment-history categories are based on binary treatment-type indicators and are not necessarily mutually exclusive.

The analysis is descriptive and unstratified. It does not adjust for demographic, clinical, molecular, treatment, or study-specific confounding factors.

