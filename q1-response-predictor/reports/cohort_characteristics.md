# Clinical & Genomic Characteristics of Data Cohorts

This report provides a comparative summary of the patient demographic and clinicopathological features across the four cohorts analyzed in this study:
*   **TCGA-SKCM**: Baseline reference cohort with adjuvant systemic treatment annotations.
*   **Liu 2019**: Advanced melanoma trial of patients treated with anti-PD-1 monotherapy.
*   **Hugo 2016**: Anti-PD-1 clinical trial cohort.
*   **Riaz 2017**: Anti-PD-1 clinical trial cohort (nivolumab-treated).

## Summary Table

| Characteristic               | TCGA-SKCM    | Liu 2019     | Hugo 2016   | Riaz 2017   |
|:-----------------------------|:-------------|:-------------|:------------|:------------|
| Total cohort                 | 426          | 103          | 26          | 33          |
| Treatment received           |              |              |             |             |
| - Immunotherapy              | 68 (16.0%)   | 103 (100.0%) | 26 (100.0%) | 33 (100.0%) |
| - Chemotherapy               | 66 (15.5%)   | 0 (0.0%)     | 0 (0.0%)    | N/A         |
| - Targeted Therapy           | 12 (2.8%)    | 0 (0.0%)     | 12 (46.2%)  | N/A         |
| - Radiation Therapy          | 107 (25.1%)  | N/A          | 0 (0.0%)    | N/A         |
| Response Rate (RECIST CR/PR) | N/A          | 47 (45.6%)   | 13 (50.0%)  | 10 (30.3%)  |
| Age (Years, Mean ± SD)       | 57.5 ± 15.7  | N/A          | 58.7 ± 14.4 | N/A         |
| Gender (Male)                | 264 (62.0%)  | 61 (59.2%)   | 18 (69.2%)  | N/A         |
| Specimen Type (Primary)      | 80 (18.8%)   | 0 (0.0%)     | 0 (0.0%)    | 0 (0.0%)    |
| Survival Data Follow-up      | 426 (100.0%) | 103 (100.0%) | 25 (96.2%)  | 0 (0.0%)    |

## Key Observations
1.  **Cohort Size**: The TCGA-SKCM cohort is by far the largest ($N=426$), providing a strong baseline for mapping genetic and clinical features associated with overall survival. The clinical trial cohorts are smaller ($N=26$ to $N=103$), reflecting the typical sizes of single-agent immunotherapy clinical studies.
2.  **Survival Data**: High-quality overall survival follow-up is available for **TCGA-SKCM** (100%), **Liu 2019** (100%), and **Hugo 2016** (100%), which enables rigorous log-rank testing and Cox proportional hazards regression. Riaz 2017 processed clinical data lacks survival follow-up and is used exclusively for predicting binary RECIST response.
3.  **Treatment Context**: TCGA-SKCM contains baseline genomic profiling accompanied by detailed adjuvant systemic treatment histories (including chemotherapy, immunotherapy, targeted molecular therapies, and radiation therapy). In contrast, the clinical trial cohorts represent patients enrolled in specific active anti-PD-1 interventional trials where pre-treatment biopsies were matched to subsequent checkpoint inhibitor response.
4.  **Specimen Sites**: While TCGA-SKCM contains a mix of primary cutaneous melanomas (18.8%) and regional/distant metastases (81.2%), the clinical trial cohorts consist almost entirely of pre-treatment resections of metastatic lesions.