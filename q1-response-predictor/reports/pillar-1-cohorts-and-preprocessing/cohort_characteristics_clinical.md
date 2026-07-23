---
title: Clinical Characteristics of Data Cohorts
aliases:
  - Clinical Cohort Characteristics
tags:
  - melanoma
  - clinical-analysis
  - cohort-characteristics
  - survival-analysis
created: 2026-07-23 18:43
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-23 18:43
---

# Clinical Characteristics of Data Cohorts

This report provides a comparative summary of patient demographic, treatment, survival, and sample attrition characteristics across the four cohorts analysed in this study:

*   **TCGA-SKCM**: Baseline reference cohort with recorded treatment history.
*   **Liu 2019**: Advanced melanoma trial cohort treated with anti-PD-1 monotherapy.
*   **Hugo 2016**: Anti-PD-1 clinical trial cohort.
*   **Riaz 2017**: Anti-PD-1 clinical trial cohort treated with nivolumab.

Demographic and treatment characteristics were calculated dynamically from the processed clinical datasets. Survival characteristics were calculated from cleaned clinical records evaluated in the Kaplan-Meier analysis. Sample attrition details track cohort retention across data preprocessing stages.

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

---

## Sample Preprocessing Attrition

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

---

## Key Observations

1. **Cohort Size**: The largest individual cohort is **TCGA-SKCM** with **N = 443** patients. Across all four cohorts, **N = 699** cleaned clinical samples were harmonised.

2. **Sample Attrition**: Preprocessing quality control and clinical-expression alignment evaluated **704** initial records and removed **5** sample(s) across cohorts (**Liu 2019** retained 100% of samples (N = 122); **Hugo 2016** retained 100% of samples (N = 27); **Riaz 2017** retained 100% of samples (N = 107); **TCGA-SKCM** lost **5** sample(s) (448 → 443)).

3. **Overall Survival**: Median overall survival was estimated using the Kaplan-Meier survival function. Where the estimated survival probability did not fall below 50% during follow-up, median OS was reported as not reached.

4. **Follow-up Duration**: **TCGA-SKCM** demonstrated the longest median follow-up duration at **41.6 months**.

5. **Event Rates**: Observed OS event rates varied between cohorts, reflecting differences in patient composition, disease staging, treatment regimens, follow-up duration, and censoring.

6. **Variable Completeness**: Demographic (age, sex) and treatment annotations vary in availability across datasets, requiring careful consideration during multi-cohort synthesis.

---

## Overall Survival Curves (KM Plots)

The unstratified Kaplan-Meier overall survival curves for each cohort are shown below. Median overall survival is annotated for each cohort where reached.

![[km_os_grid.png]]

_**Overall Survival KM Curves (All Cohorts)**_

---

## Univariate Associations with Response (Forest Plot)

To evaluate whether individual baseline clinical and genomic features predict response to anti-PD-1 therapy, univariate Odds Ratios (OR) and 95% Confidence Intervals (95% CI) were calculated across individual trial cohorts (Liu 2019, Hugo 2016, Riaz 2017) and the pooled immunotherapy trial cohort. Categorical variables were evaluated via Fisher's Exact test, and continuous variables (Age, TMB, SNV Neoantigens, Indel Neoantigens) were evaluated per 1 SD increase via univariate logistic regression.

![Forest Plot of Univariate Odds Ratios](../../plots/clinical/univariate_associations.png)

### Key Takeaways
1. **Driver Mutations**: *NF1* mutated tumours show elevated odds ratios for response in the pooled trial cohort, while *BRAF* mutations show virtually no univariate association with response.
2. **Anatomical Stage Trend**: Clinical Stage IV was associated with an OR of 6.06 relative to Stage III in pooled analysis (p = 0.083), reflecting small Stage III representation in checkpoint blockade trial cohorts.
3. **TMB & Neoantigen Trends**: TMB shows a positive trend with response in the pooled trial cohort (OR = 1.37 per SD increase, p = 0.160), with consistent positive point estimates across Liu 2019 and Hugo 2016.
4. **Demographics**: Age and Sex show no significant association with immunotherapy response (OR ≈ 0.98 – 1.26), confirming demographic balance across treatment groups.

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

Sample attrition tracking records sample filtering from initial cBioPortal data ingestion through clinical-expression alignment.

The analysis is descriptive and unstratified. It does not adjust for demographic, clinical, molecular, treatment, or study-specific confounding factors.
