---
cssclasses: table-small
---
# Batch Effect Assessment & Dimensionality Reduction Analysis

When combining transcriptomic datasets across independent clinical studies, technical variations (e.g. sequencing platforms, RNA extraction methods, and library preparation) typically dominate the biological signals. This report documents how technical batch effects were identified and corrected across our melanoma cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, and **Riaz 2017**) and whether global expression profiles separate patients based on therapeutic response. Plot aesthetics and palettes are aligned with the Okabe-Ito color guidelines used across other reports.

## 1. Full Cohort Batch Assessment (N = 699)

To evaluate overall batch effects across all samples, we performed a Principal Component Analysis (PCA) on the top 559 most variable genes (selected from 559 genes common to all four cohorts) across the $N=699$ patients in the full merged cohort, comparing the uncorrected concatenated matrix against the individually Z-score standardized matrix.

![[batch_effect_pca.png]]

### Key Observations
- **Panel A: Before Batch Correction**: The uncorrected PCA reveals a strong cohort-associated structure. TCGA-SKCM samples form a relatively distinct cluster, while the Liu 2019, Hugo 2016, and Riaz 2017 trial cohorts occupy a broadly overlapping region of the projection space. This suggests that the dominant variation in the integrated expression matrix is strongly influenced by the distinction between the TCGA dataset and the clinical-trial datasets. However, the PCA does not support the conclusion that each study forms a completely isolated cluster, nor that the first two principal components are exclusively technical in origin.
- **Panel B: After Cohort-Specific Z-Score Standardisation**: Following within-cohort Z-score standardisation, the strong separation between TCGA-SKCM and the clinical-trial cohorts is substantially reduced. The increased overlap suggests that cohort-level differences in gene-wise means and variances contributed to the original separation. However, the resulting PCA should be interpreted as evidence of reduced cohort-associated structure rather than proof that all technical batch effects have been completely eliminated.

## 2. Immunotherapy Trial Dimensionality Reduction (N = 256)

To evaluate technical batch effects and patient response separation in the clinical trials specifically, we performed PCA and Uniform Manifold Approximation and Projection (UMAP) on the $N=256$ response-aligned trial patients (Liu 2019, Hugo 2016, and Riaz 2017) using the 559 common genes.

### PCA Projections (Raw vs. Standardized)
![[pca_dimensionality_reduction.png]]

### UMAP Projections (Raw vs. Standardized)
![[umap_dimensionality_reduction.png]]

### Key Observations
- **Cohort Structure:** The PCA and UMAP projections provide different views of the relationship between the three trial cohorts. The effect of cohort-wise Z-score standardisation is relatively limited in the global PCA structure, while the UMAP embedding shows a more substantial reorganisation of local sample neighbourhoods. This indicates that the transformation affects local relationships between samples more strongly than the dominant global variance structure captured by PCA.
- **Response Distribution:** When coloured by treatment response, responders (CR/PR) and non-responders (PD) do not form clearly separated global groups in either the PCA or UMAP projections. Although local patterns or partial enrichment may be present, there is no obvious response-defined boundary that separates the two response groups across the overall transcriptomic space.
- **Interpretation:** The absence of clear global separation by response suggests that treatment response is not captured by a simple, dominant axis of variation in the reduced transcriptomic space. However, PCA and UMAP are exploratory visualisation methods and cannot establish whether response is predictable from gene expression. Predictive modelling and pathway-level analyses are therefore required to determine whether response-associated molecular patterns exist that are not apparent in these two-dimensional projections.

## 3. Gene-Level Expression Heatmaps (Top 50 Highly Variable Genes)

To evaluate batch correction at the individual gene level, we selected the **top 50 genes by variance** (calculated on raw log2-TPM expression data across trial patients) and performed hierarchical clustering on both uncorrected and standardized expression values.

### Raw Expression (Top 50 Genes)
![[heatmap_top_variance_genes_raw.png]]

### Standardized Expression (Top 50 Genes)
![[heatmap_top_variance_genes_standardized.png]]

### Key Observations
*   **Before Batch Correction (Raw log2-TPM)**: Patient columns cluster heavily by cohort source.
*   **After Batch Correction (Individual Z-scoring)**: Perfect cohort mixing across patient dendrograms.

## 4. Cross-Validation Rigor & Data Leakage Prevention

### The Hazard of Global Batch Correction (e.g. ComBat)
Algorithms like ComBat pool all samples together to estimate batch correction parameters. When applied to a multi-study cohort prior to Leave-One-Cohort-Out (LOCO) cross-validation, the test cohort is included in parameter estimation, introducing data leakage.

### The Z-score Scaling Solution
By applying Z-score standardization cohort-independently, zero leakage is guaranteed during model cross-validation.
