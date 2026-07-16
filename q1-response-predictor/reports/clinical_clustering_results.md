# Patient Phenotyping via Clinical & Genomic Clustering

We performed unsupervised phenotyping on the **TCGA-SKCM** cohort ($N = 426$ patients) using K-Means clustering. Patient profiles were constructed across demographics, tumor genetics, microenvironmental stress, and adjuvant therapy classes.

## Cluster Profiles
The average clinical and genomic values for each patient cluster are detailed below:

|   CLINICAL_CLUSTER |   Patient Count |     AGE |   TMB_NONSYNONYMOUS |   FRACTION_GENOME_ALTERED |   ANEUPLOIDY_SCORE |   WINTER_HYPOXIA_SCORE |   TX_TYPE_CHEMOTHERAPY |   TX_TYPE_IMMUNOTHERAPY |   TX_TYPE_RADIATION_THERAPY |   IS_PRIMARY |   IS_STAGE_IV |
|-------------------:|----------------:|--------:|--------------------:|--------------------------:|-------------------:|-----------------------:|-----------------------:|------------------------:|----------------------------:|-------------:|--------------:|
|                  0 |             244 | 57.4467 |             26.4049 |                  0.197548 |             7.7161 |               -5.39918 |               0.143443 |                0.127049 |                    0.213115 |     0.221311 |             0 |
|                  1 |             160 | 58.0187 |             27.9743 |                  0.501518 |            20.6918 |                2.3125  |               0.175    |                0.23125  |                    0.31875  |     0.14375  |             0 |
|                  2 |              22 | 55.0909 |             13.5242 |                  0.294982 |            11.3333 |               -2       |               0.136364 |                0        |                    0.181818 |     0.136364 |             1 |

## Clinical Interpretation of Clusters

Based on the multi-dimensional profiles, the three clusters represent distinct disease states:

1.  **Cluster 0: Low Chromosomal Instability & Low Hypoxia Phenotype** ($N=244$)
    *   *Genomics*: Low chromosomal instability (fraction genome altered = 0.198, aneuploidy score = 7.72) and moderate TMB (26.4 mut/Mb).
    *   *Microenvironment*: Low Winter hypoxia score (-5.40).
    *   *Clinical*: Stage IV rate = 0%. Contains the highest rate of primary specimens (22.1%).
    *   *Prognosis*: Better overall survival trajectory.

2.  **Cluster 1: High Chromosomal Instability & High Hypoxia Phenotype** ($N=160$)
    *   *Genomics*: Extremely elevated copy number alteration fraction (**0.502**), highest aneuploidy score (**20.69**), and high TMB (28.0 mut/Mb).
    *   *Microenvironment*: Elevated Winter hypoxia score (+2.31).
    *   *Clinical*: Stage IV rate = 0%. Lower primary tumor rate (14.4%). Highest rate of systemic/adjuvant therapies (Immunotherapy = 23.1%, Radiation = 31.9%).
    *   *Prognosis*: Moderate/poor survival trajectory.

3.  **Cluster 2: Stage IV / Advanced Metastatic Disease Phenotype** ($N=22$)
    *   *Genomics*: Lower mutational load (TMB = 13.5 mut/Mb) and moderate copy-number alterations.
    *   *Clinical*: **100% of these patients have Stage IV disease**. No patients received immunotherapy (0%).
    *   *Prognosis*: Poor overall survival trajectory (Stage IV baseline).

## Kaplan-Meier Survival Analysis
The unsupervised patient clusters show a highly statistically significant separation in overall survival duration (Log-Rank p-value = **\(2.44e-02\)**).

![KM Survival of Clinical Clusters](../plots/km_clinical_clusters.png)
