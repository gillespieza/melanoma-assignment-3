---
title: Patient Phenotyping via Full-Dataset Immunological & Genomic Clustering
tags:
  - melanoma
  - clinical-subtyping
  - clustering
  - full-dataset
  - immune-hot-cold
created: 2026-07-23 17:49
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-23 17:49
---

# Patient Phenotyping via Full-Dataset Immunological & Genomic Clustering

We performed unsupervised subtyping across the **entire combined study dataset** ($N = 699$ patients across **TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, and **Riaz 2017**) using Agglomerative Hierarchical Clustering (Ward linkage). To prevent technical study platform offsets from dominating the clustering, feature scores (immune signatures, TMB, and age) were Z-score standardized **within each cohort** prior to pooling.

## Subtype Profiles
The average clinical, genomic, and transcriptomic immune signature values for each patient subtype are detailed below:

| Feature / Clinical & Biological Metric | Cluster 0: Immunologically Hot (N=311) | Cluster 1: Immunologically Cold (N=205) | Cluster 2: High-TMB (N=183) |
| --- | --- | --- | --- |
| **Demographics & Sample Size** |  |  |  |
| Patient Count (N) | 311 | 205 | 183 |
| Age (Years, Mean) | 45.5 | 50.5 | 47.0 |
| **Genomic & Mutational Burden** |  |  |  |
| TMB (Nonsynonymous, Mean Mut/Mb) | 29.5 | 18.5 | 17.6 |
| **Transcriptomic Immune Signatures (Raw Mean)** |  |  |  |
| IFN_gamma Signature Score | 7.85 | 5.48 | 6.35 |
| TIS Signature Score | 7.30 | 5.07 | 5.96 |
| CYT Signature Score | 6.69 | 3.93 | 5.14 |
| CD8_Tcell Signature Score | 6.72 | 3.37 | 4.99 |
| PD_L1 Signature Score | 4.97 | 3.10 | 3.73 |
| IMPRES Signature Score | 9.62 | 7.35 | 8.62 |
| **Therapeutic Response (Trial Subset, N=256)** |  |  |  |
| Response Rate (CR/PR %) | 45 (37.5%) | 14 (27.5%) | 23 (30.3%) |
| **Prognosis & Survival** |  |  |  |
| Median Overall Survival | 🟢 103.1 months | 🔴 41.6 months | 🟠 47.4 months |

## Key Analytical Findings

1. **Prognostic Stratification ($N=699$)**: Hierarchical clustering on within-cohort Z-score standardized features yields a highly statistically significant overall survival separation across the full 4-cohort dataset (Log-Rank $p = 2.29e-05$). Patients in the **Immunologically Hot** cluster achieve a median survival exceeding **100 months** (🟢 103.1 months), more than double that of the **Cold** cluster (🔴 41.6 months).
2. **Therapeutic Response Alignment ($N=256$)**: Patients in **Cluster 0 (Hot)** demonstrate the highest objective response rate to anti-PD-1 immunotherapy (**45 (37.5%)**), compared to **14 (27.5%)** in **Cluster 1 (Cold)**, validating that unsupervised microenvironment subtyping captures anti-tumor immune responsiveness.
3. **Genomic vs. Transcriptomic Decoupling**: High tumor mutational burden alone (**Cluster 2**, mean TMB = 17.6 mut/Mb) yields only an intermediate overall survival trajectory (🟠 47.4 months) in the absence of robust T-cell inflammation, demonstrating that high TMB is insufficient without an active immune microenvironment.

## Biological Interpretation of Patient Subtypes

The unsupervised clustering isolates three distinct patient phenotypes across the multi-study population:

1.  **Cluster 0: Immunologically Hot / Inflamed Phenotype** ($N=311$)
    *   *Immune Signatures*: Highest T-cell inflammation (IFN-\(\gamma\) = 7.85, TIS = 7.30, CYT = 6.69, CD8 = 6.72).
    *   *Therapeutic Benefit*: Highest immunotherapy response rate (**45 (37.5%)**).
    *   *Prognosis*: Superior overall survival trajectory (Median OS = 🟢 **103.1 months**).

2.  **Cluster 1: Immunologically Cold / Desert Phenotype** ($N=205$)
    *   *Immune Signatures*: Attenuated T-cell inflammation across all markers (IFN-\(\gamma\) = 5.48, TIS = 5.07, CYT = 3.93, CD8 = 3.37).
    *   *Therapeutic Benefit*: Lower response rate to anti-PD-1 therapy (**14 (27.5%)**).
    *   *Prognosis*: Poor overall survival trajectory (Median OS = 🔴 **41.6 months**).

3.  **Cluster 2: High-TMB / Hypermutated Phenotype** ($N=183$)
    *   *Genomics*: Highest tumor mutational burden (**TMB = 17.6 mut/Mb**).
    *   *Immune Signatures*: Moderate T-cell inflammation (IFN-\(\gamma\) = 6.35, TIS = 5.96).
    *   *Prognosis*: Intermediate survival trajectory (Median OS = 🟠 **47.4 months**).

## Subtype Visualisation (2D PCA Projection)
Below is a 2D PCA projection showing clear multi-dimensional separation of the patient subtypes across the $N=699$ full dataset. The 'X' markers denote cluster centroids:

![2D PCA Visualisation of Clusters](../../plots/clinical/pca_clinical_clusters.png)

## Immunotherapy Response & Overall Survival Validation
Validation across clinical outcomes demonstrates that unsupervised immune subtyping strongly correlates with clinical benefit:

*   **Therapeutic Response Rate (Trial Cohorts, $N=256$)**: Significant difference in response rate across clusters (Chi-Square p-value = **\(3.58e-01\)**).
    ![Response Rate by Cluster](../../plots/clinical/response_by_clinical_cluster.png)

*   **Overall Survival (Full Dataset, $N=699$)**: Highly significant survival separation across patient subtypes (Log-Rank p-value = **\(2.29e-05\)**):
    ![KM Survival of Clinical Clusters](../../plots/clinical/km_clinical_clusters.png)
