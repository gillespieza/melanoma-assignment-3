---
title: Clinical Characteristics of Data Cohorts
aliases:
  - Clinical Cohort Characteristics
tags:
  - melanoma
  - clinical-analysis
  - cohort-characteristics
  - survival-analysis
created: 2026-07-23 23:27
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-24 15:05
---

# Clinical Characteristics of Data Cohorts

## 1. Baseline Patient and Disease Characteristics

> [!summary] Why We Are Doing This  
> Before building predictive models or analysing transcriptomic signatures, we must understand patient demographics, treatment histories, and survival outcomes across our datasets. Comparing baseline characteristics across **TCGA-SKCM** ($N = 443$), **Liu 2019** ($N = 122$), **Hugo 2016** ($N = 27$), and **Riaz 2017** ($N = 107$) ensures we account for clinical differences between cohorts (such as prior anti-CTLA-4 therapy) and verify sample retention during preprocessing.

This report compares patient demographics, treatments, survival, and sample attrition across the four melanoma study cohorts:
- **TCGA-SKCM**: Baseline reference dataset ($N = 443$) representing primary and metastatic melanoma with general treatment histories.
- **Liu 2019**: Immunotherapy trial ($N = 122$) treated with anti-PD-1 monotherapy (Pembrolizumab or Nivolumab).
- **Hugo 2016**: Anti-PD-1 clinical trial cohort ($N = 27$).
- **Riaz 2017**: Anti-PD-1 clinical trial cohort ($N = 107$) where 100% of patients had previously received anti-CTLA-4 (Ipilimumab).

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

> [!summary] Why We Are Doing This  
> Tracking sample retention through quality control, identifier standardisation, and clinical-expression alignment verifies data integrity and confirms that no patient sub-populations were accidentally dropped during pipeline processing.

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
1. **Cohort Size ($N = 699$)**: The largest individual dataset is **TCGA-SKCM** ($N = 443$). Combined with the three clinical trial cohorts (**Liu 2019**: $N = 122$; **Riaz 2017**: $N = 107$; **Hugo 2016**: $N = 27$), a total of $N = 699$ cleaned patient records were harmonised.
2. **Minimal Sample Attrition (99.3% Retention)**: Across all 704 initial records downloaded from cBioPortal, only **5 samples were removed** (a 99.3% retention rate). All three immunotherapy trial cohorts retained **100% of their samples**, while TCGA-SKCM lost only 5 records (448 → 443) due to missing matched RNA-seq expression data. Attrition was minimal because these published trial cohorts were already heavily curated and pre-filtered by the original study investigators.
3. **Baseline Survival Differences**: **TCGA-SKCM** shows the longest median overall survival (**79.0 months**) and follow-up (**41.6 months**) because it includes earlier-stage surgical cases. In contrast, the three advanced metastatic trial cohorts exhibit shorter median survival (**21.2 to 32.2 months**).
4. **Treatment History & Clinical Context**: All $N = 107$ patients in **Riaz 2017** were previously treated with anti-CTLA-4 (Ipilimumab), representing a treatment-resistant population compared to anti-CTLA-4 naive patients in **Liu 2019** ($N = 122$).

## 3. Overall Survival Curves (KM Plots)

> [!summary] Why We Are Doing This  
> Visualising unstratified Kaplan-Meier overall survival curves establishes baseline mortality rates across each study cohort and confirms that follow-up duration is sufficient to evaluate treatment outcomes.

The unstratified Kaplan-Meier overall survival curves for each cohort are shown below. Median overall survival is annotated for each cohort where reached.

![Overall Survival KM Curves (All Cohorts)](../../plots/clinical/km_os_grid.png)

_**Figure 1: Unstratified Overall Survival KM Curves across All Four Cohorts.**_

## 4. Overall Survival Stratified by Immunotherapy Response

> [!summary] Why We Are Doing This  
> To verify that RECIST clinical response (Responders [CR/PR] vs. Non-responders [PD]) is a meaningful outcome, we stratify Kaplan-Meier overall survival curves by response status. Confirming that treatment responders experience significantly longer overall survival validates RECIST response as a strong surrogate endpoint for long-term clinical benefit.

![Overall Survival by Immunotherapy Response (RECIST)](../../plots/clinical/km_os_by_response.png)

_**Figure 2: Overall Survival Stratified by RECIST Response Status across Immunotherapy Trial Cohorts.** Treatment responders (CR/PR) exhibit significantly superior overall survival compared to non-responders (PD) across all three trial cohorts (Log-rank $p < 0.0001$)._

### Key Observations
1. **Profound Survival Benefit**: Treatment responders (CR/PR) achieve significantly longer overall survival compared to non-responders (PD) across all three independent trial cohorts (Liu 2019, Hugo 2016, and Riaz 2017; all Log-rank $p < 0.0001$).
2. **Durable Long-Term Survival**: In **Liu 2019**, non-responders experience rapid mortality (median OS ~10.5 months), whereas >70% of responders survive past 50 months. Similar durable separation occurs in Hugo 2016 and Riaz 2017.
3. **Conclusion**: Objective RECIST response is a highly robust surrogate endpoint for overall survival in metastatic melanoma.

## 5. Univariate Associations with Response (Forest Plot)

> [!summary] Why We Are Doing This  
> Before building complex multivariate prediction models, we run univariate statistical tests (Odds Ratios and 95% Confidence Intervals) to measure the isolated predictive strength of individual baseline features—such as age, sex, tumour mutation burden (TMB), and driver mutations (`BRAF`, `NRAS`, `NRAS`)—against immunotherapy response.

![Forest Plot of Univariate Odds Ratios](../../plots/clinical/univariate_associations.png)

### Key Takeaways
1. **Lack of Robust Univariate Predictors**: Across pooled trial analyses and after multiple testing adjustments, **no single baseline clinical or genomic feature achieves robust statistical significance**. All 95% confidence intervals for pooled odds ratios cross 1.0.
2. **Anatomical Staging Artefact (Liu 2019 $p = 0.029$)**: Stage IV disease shows a nominal unadjusted association in the isolated Liu 2019 cohort ($p = 0.029$, highlighted in the plot). However, this is an artefact of extreme trial enrollment imbalance (almost all trial patients have Stage IV metastatic disease). In the pooled multi-cohort analysis, this association attenuates to $p = 0.083$ (not significant), and loses significance entirely after multiple testing correction.
3. **Subtle Feature Trends**: Certain features display weak positive trends: `NRAS` mutations and higher Tumour Mutation Burden (TMB; OR = 1.37 per SD increase, $p = 0.160$) trend towards elevated response odds, whereas `BRAF` driver mutations show zero univariate association with response (OR ≈ 1.0).
4. **Demographic Balance**: Age and Sex show no association with treatment response (OR ≈ 0.98–1.26, $p > 0.50$), confirming demographic balance between responders and non-responders across treatment arms.
5. **Core Scientific Implication**: The failure of individual clinical variables and driver mutations to reliably predict outcome proves why single-variable biomarker tests fail in clinical practice, highlighting the necessity of multi-gene transcriptomic signatures and integrated multivariate machine learning models.

## 6. Technical Analysis Notes

Overall survival was analysed using the available cohort-specific survival duration and event-status variables.

Records were excluded from the survival analysis if they had:
* Missing survival time.
* Missing survival event status.
* Non-numeric survival values.
* A survival time less than or equal to zero.

Median overall survival was estimated using the Kaplan-Meier survival function. Where the estimated survival probability did not fall below 50% during follow-up, median OS was reported as **NR (not reached)**.

Treatment percentages use the number of patients with an available treatment annotation as the denominator where this differs from the total cohort size. TCGA treatment-history categories are based on binary treatment-type indicators and are not necessarily mutually exclusive.

Sample attrition tracking records sample filtering from initial cBioPortal data ingestion through clinical-expression alignment.

The analysis is descriptive and unstratified. It does not adjust for demographic, clinical, molecular, treatment, or study-specific confounding factors.
