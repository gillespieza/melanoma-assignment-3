# Patient Phenotyping via Clinical & Genomic Clustering

We performed unsupervised phenotyping on the **TCGA-SKCM** cohort ($N = 426$ patients) using Agglomerative Hierarchical Clustering (Ward linkage). Patient profiles were constructed across demographics, tumor genetics, microenvironmental stress, and adjuvant therapy classes.

## Cluster Profiles
The average clinical and genomic values for each patient cluster are detailed below in a transposed summary table:

| Feature / Clinicopathological Metric   | Cluster 0 (N=312)   | Cluster 1 (N=112)   | Cluster 2 (N=24)   |
|:---------------------------------------|:--------------------|:--------------------|:-------------------|
| **Demographics & Baseline**            |                     |                     |                    |
| Patient Count (N)                      | 312                 | 112                 | 24                 |
| Age (Years, Mean)                      | 58.8                | 54.3                | 54.9               |
| **Genomics**                           |                     |                     |                    |
| TMB (Nonsynonymous, Mean Mut/Mb)       | 24.9                | 31.4                | 14.0               |
| Aneuploidy Score (Mean)                | 13.1                | 12.9                | 11.8               |
| **Microenvironment**                   |                     |                     |                    |
| Winter Hypoxia Score (Mean)            | -2.57               | -1.61               | -2.33              |
| **Adjuvant Treatment History**         |                     |                     |                    |
| Chemotherapy Received (%)              | 0.0%                | 58.0%               | 12.5%              |
| Immunotherapy Received (%)             | 0.0%                | 63.4%               | 0.0%               |
| Radiation Therapy Received (%)         | 21.5%               | 37.5%               | 20.8%              |
| **Clinical Presentation**              |                     |                     |                    |
| Primary Specimen Type (%)              | 19.9%               | 14.3%               | 12.5%              |
| Stage IV Metastatic Disease (%)        | 0.0%                | 0.0%                | 100.0%             |
| **Prognosis & Outcomes**               |                     |                     |                    |
| Median Overall Survival                | 93.0 months         | 66.5 months         | 28.1 months        |

## Clinical Interpretation of Clusters

Based on the multi-dimensional profiles, the three clusters represent distinct disease states:

1.  **Cluster 0: Baseline Disease / Primary Specimen Phenotype** ($N=312$)
    *   *Genomics*: Moderate chromosomal instability (aneuploidy score = 13.1) and moderate TMB (24.9 mut/Mb).
    *   *Microenvironment*: Low-moderate Winter hypoxia score (-2.57).
    *   *Clinical*: Stage IV rate = 0.0%. Highest rate of primary specimens (19.9%). No immunotherapy (0.0%).
    *   *Prognosis*: Better overall survival trajectory.

2.  **Cluster 1: High Mutational Load & Immunotherapy Phenotype** ($N=112$)
    *   *Genomics*: Elevated copy-number alterations (aneuploidy score = 12.9) and the highest mutational load (**TMB = 31.4 mut/Mb**).
    *   *Microenvironment*: Low-moderate Winter hypoxia score (-1.61).
    *   *Clinical*: Stage IV rate = 0.0%. Lower primary tumor rate (14.3%). High rate of immunotherapy (63.4%).
    *   *Prognosis*: Moderate survival trajectory.

3.  **Cluster 2: Stage IV / Advanced Metastatic Disease Phenotype** ($N=24$)
    *   *Genomics*: Lower mutational load (TMB = 14.0 mut/Mb) and lowest copy-number alterations (aneuploidy score = 11.8).
    *   *Clinical*: **100% of these patients have Stage IV disease**. No patients received immunotherapy (0.0%).
    *   *Prognosis*: Poor overall survival trajectory (Stage IV baseline) (Median Overall Survival = **28.1 months**).

## Cluster Visualisation (2D PCA Projection)
Below is a 2D PCA projection of the multi-dimensional patient profiles, showing the distinct separation of the three clinical-genomic patient groups. The 'X' markers show the cluster centroids:

![2D PCA Visualization of Clusters](../plots/clinical/pca_clinical_clusters.png)

## Kaplan-Meier Survival Analysis
The unsupervised patient clusters show a highly statistically significant separation in overall survival duration (Log-Rank p-value = **\(1.82e-02\)**):

![KM Survival of Clinical Clusters](../plots/clinical/km_clinical_clusters.png)
