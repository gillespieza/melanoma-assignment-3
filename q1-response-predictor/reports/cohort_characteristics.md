# Clinical Characteristics of Data Cohorts

This report provides a comparative summary of the patient demographic and clinicopathological features across the four cohorts analyzed in this study:
*   **TCGA-SKCM**: Baseline reference cohort with adjuvant systemic treatment annotations.
*   **Liu 2019**: Advanced melanoma trial of patients treated with anti-PD-1 monotherapy.
*   **Hugo 2016**: Anti-PD-1 clinical trial cohort.
*   **Riaz 2017**: Anti-PD-1 clinical trial cohort (nivolumab-treated).

## Summary Table

| Characteristic               | TCGA-SKCM    | Liu 2019     | Hugo 2016   | Riaz 2017   |
|:-----------------------------|:-------------|:-------------|:------------|:------------|
| Total cohort                 | 426          | 104          | 26          | 20          |
| Treatment received           |              |              |             |             |
| - Immunotherapy              | 68 (16.0%)   | 104 (100.0%) | 26 (100.0%) | 20 (100.0%) |
| - Chemotherapy               | 66 (15.5%)   | 0 (0.0%)     | 0 (0.0%)    | 0 (0.0%)    |
| - Targeted Therapy           | 12 (2.8%)    | 0 (0.0%)     | 12 (46.2%)  | 0 (0.0%)    |
| - Radiation Therapy          | 107 (25.1%)  | 0 (0.0%)     | 0 (0.0%)    | 0 (0.0%)    |
| Response Rate (RECIST CR/PR) | N/A          | 48 (46.2%)   | 13 (50.0%)  | 5 (25.0%)   |
| Age (Years, Mean ± SD)       | 57.5 ± 15.7  | N/A          | 59.3 ± 15.1 | 56.1 ± 11.0 |
| Gender (Male)                | 264 (62.0%)  | 61 (58.7%)   | 18 (69.2%)  | 9 (45.0%)   |
| Specimen Type (Primary)      | 77 (18.1%)   | 0 (0.0%)     | 0 (0.0%)    | 0 (0.0%)    |
| Survival Data Follow-up      | 426 (100.0%) | 104 (100.0%) | 26 (100.0%) | 20 (100.0%) |

## Key Observations
1.  **Cohort Size**: The TCGA-SKCM cohort is by far the largest ($N=426$), providing a strong baseline for mapping genetic and clinical features associated with overall survival. The clinical trial cohorts are smaller ($N=20$ to $N=104$), reflecting the typical sizes of single-agent immunotherapy clinical studies.
2.  **Survival Data**: High-quality overall survival follow-up is available for all four cohorts: **TCGA-SKCM** (100%), **Liu 2019** (100%), **Hugo 2016** (100%), and **Riaz 2017** (100%). This enables overall survival characterization across cohorts.
3.  **Treatment Context**: TCGA-SKCM contains baseline genomic profiling accompanied by detailed adjuvant systemic treatment histories (including chemotherapy, immunotherapy, targeted molecular therapies, and radiation therapy). In contrast, the clinical trial cohorts represent patients enrolled in specific active anti-PD-1 interventional trials where pre-treatment biopsies were matched to subsequent checkpoint inhibitor response.
4.  **Specimen Sites**: While TCGA-SKCM contains a mix of primary cutaneous melanomas (18.1%) and regional/distant metastases (81.9%), the clinical trial cohorts consist almost entirely of pre-treatment resections of metastatic lesions.

---

## Overall Survival Curves (KM Plots)

The unstratified Kaplan-Meier overall survival curves for each cohort are shown in the 2x2 grid below. The median overall survival is annotated for each cohort where reached.

![Overall Survival KM Curves (All Cohorts)](../plots/clinical/km_os_grid.png)

---

## Patient Selection (CONSORT Flowcharts)

The flowcharts below document the cohort attrition and selection process from raw downloads to final cleaned datasets:

### Liu 2019 Attrition Flowchart
```mermaid
graph TD
    A["Raw Merged Cohort (N = 122)"] --> B{"Response Filter"}
    B -->|"Excluded SD, MR, or Missing Response (n = 18)"| C("Excluded (n = 18)")
    B -->|"Included CR, PR, PD (n = 104)"| D["Response-Aligned Cohort (N = 104)"]
    D --> E{"Expression Data Alignment"}
    E -->|"No Expression Matching Mismatch (n = 0)"| F["Final Cleaned Cohort (N = 104)"]
```

### Hugo 2016 Attrition Flowchart
```mermaid
graph TD
    A["Raw Merged Cohort (N = 27)"] --> B{"Clinical QC Check<br>(Duplicate Patients & OS data)"}
    B -->|"Excluded Duplicate/Invalid OS (n = 1)"| C("Excluded (n = 1)")
    B -->|"Valid Patient OS Records (n = 26)"| D["QC-Passed Cohort (N = 26)"]
    D --> E{"Response Filter"}
    E -->|"Excluded SD, MR, or Missing Response (n = 0)"| F["Response-Aligned Cohort (N = 26)"]
    F --> G{"Expression Data Alignment"}
    G -->|"No Expression Matching Mismatch (n = 0)"| H["Final Cleaned Cohort (N = 26)"]
```

### Riaz 2017 Attrition Flowchart
```mermaid
graph TD
    A["Raw Merged Cohort (N = 107)<br>64 Patients"] --> B{"Clinical QC Check<br>(Duplicate Patient IDs)"}
    B -->|"Excluded Duplicate Patient Records (n = 43)"| C("Excluded (n = 43)")
    B -->|"Unique Patient Records (n = 64)"| D["QC-Passed Cohort (N = 64)"]
    D --> E{"Timepoint Filter"}
    E -->|"Excluded On-Treatment Samples (n = 30)"| F("Excluded (n = 30)")
    E -->|"Pre-Treatment Baseline Samples (n = 34)"| G["Baseline Cohort (N = 34)"]
    G --> H{"Response Filter"}
    H -->|"Excluded SD, MR, or Missing Response (n = 14)"| I("Excluded (n = 14)")
    H -->|"Included CR, PR, PD (n = 20)"| J["Response-Aligned Cohort (N = 20)"]
    J --> K{"Expression Data Alignment"}
    K -->|"No Expression Matching Mismatch (n = 0)"| L["Final Cleaned Baseline Cohort (N = 20)"]
```

### TCGA-SKCM Attrition Flowchart
```mermaid
graph TD
    A["Raw Merged Cohort (N = 448)<br>442 Patients"] --> B{"Clinical QC Check<br>(Duplicate Patient IDs)"}
    B -->|"Excluded Duplicate Patient Records (n = 6)"| C("Excluded (n = 6)")
    B -->|"Unique Patient Records (n = 442)"| D["QC-Passed Cohort (N = 442)"]
    D --> E{"Survival Data QC Check<br>(Invalid/Missing OS)"}
    E -->|"Excluded Invalid OS Data (n = 16)"| F("Excluded (n = 16)")
    E -->|"Valid Survival Records (n = 426)"| G["Final Cleaned Cohort (N = 426)"]
```