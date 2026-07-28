---
title:
aliases: 
tags: 
created: 2026-07-23 17:21
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-24 14:59
---

# Batch Effect Assessment & Dimensionality Reduction Analysis

When combining transcriptomic datasets across independent clinical studies, technical variations (e.g. sequencing platforms, RNA extraction methods, and library preparation) typically dominate the biological signals. This report documents how technical batch effects were identified and corrected across our melanoma cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, and **Riaz 2017**) and whether global expression profiles separate patients based on therapeutic response. Plot aesthetics and palettes are aligned with the Okabe-Ito colour guidelines used across other reports.

## 1. Full Cohort Batch Assessment (N = 699)

> [!summary] Why We Are Doing This  
> When combining transcriptomic data collected by different research centres, technical variations—such as differences in sequencing machinery, RNA extraction kits, and laboratory protocols—create unwanted noise known as **batch effects**. If left uncorrected, a machine learning algorithm will learn to identify which laboratory processed a tissue sample rather than detecting true underlying biological signals related to patient treatment response. To evaluate and correct these technical distortions, we performed Principal Component Analysis (PCA) across all $N = 699$ patients from four combined melanoma cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, and **Riaz 2017**) using $559$ genes common to all datasets.

![[batch_effect_pca.png]]

### Key Observations
- **Panel A: Before Batch Correction (Raw Data)**: The uncorrected PCA projection reveals a strong artificial separation between the TCGA-SKCM reference study and the three clinical trial cohorts (Liu 2019, Hugo 2016, and Riaz 2017). This separation demonstrates that raw measurement differences between laboratories dominate the uncorrected expression matrix.
- **Panel B: After Cohort-Specific Z-Score Standardisation**: Applying Z-score standardisation independently within each cohort (rescaling each gene's expression to a mean of $0$ and standard deviation of $1$ per study) removes baseline laboratory shifts. The TCGA-SKCM samples now overlap smoothly with the immunotherapy trial cohorts, confirming that study-level batch effects have been effectively harmonised.

## 2. Immunotherapy Trial Dimensionality Reduction (N = 256)

> [!summary] Why We Are Doing This  
> While Section 1 evaluated overall batch effects across all samples (including non-trial reference tissue), Section 2 focuses exclusively on the $N = 256$ patients across the three clinical trials (Liu 2019, Hugo 2016, and Riaz 2017) who received anti-PD-1 immunotherapy and have known clinical response outcomes. We evaluate two complementary dimensionality reduction techniques—**PCA** (which captures global linear variance) and **UMAP** (which preserves local non-linear sample clusters)—to determine whether patient gene expression profiles naturally separate by treatment outcome prior to building supervised predictive models.

### PCA Projections (Raw vs. Standardised)
![[pca_dimensionality_reduction.png]]

### UMAP Projections (Raw vs. Standardised)
![[umap_dimensionality_reduction.png]]

### Key Observations
- **Cohort Structure**: Cohort-wise Z-score standardisation subtly adjusts global PCA axes while substantially reorganising local sample neighbourhoods in UMAP embeddings, confirming effective baseline harmonisation across trial sites.
- **Response Distribution**: When samples are coloured by therapeutic outcome (Responders vs. Non-Responders), patients do not form distinct global clusters in either PCA or UMAP projections. Responders and non-responders mix homogeneously throughout the transcriptomic projection space.
- **Biological Interpretation**: The lack of visual 2D clustering demonstrates that immunotherapy response is not governed by a single, dominant axis of gene expression variance. Simple exploratory projections are insufficient on their own to predict patient outcomes, confirming the necessity for targeted gene signatures, pathway-level scoring, and supervised classification algorithms to detect subtle predictive signals.

## 3. Gene-Level Expression Heatmaps (Top 50 Highly Variable Genes)

> [!summary] Why We Are Doing This  
> While PCA and UMAP evaluate overall patient groupings in reduced 2D space, heatmaps allow us to inspect batch effects directly at the individual gene level. By selecting the top 50 most variable genes across trial patients ($N = 256$) and performing hierarchical clustering, we check whether individual gene expression signals group patients by laboratory source or allow them to mix naturally.

### Raw Expression (Top 50 Genes)
![[heatmap_top_variance_genes_raw.png]]

### Standardised Expression (Top 50 Genes)
![[heatmap_top_variance_genes_standardized.png]]

### Key Observations
- **Before Batch Correction (Raw log2-TPM)**: Patient columns cluster heavily by study source. Entire blocks of the dendrogram align strictly with individual trial datasets, demonstrating that raw high-variance gene signals are dominated by laboratory site.
- **After Batch Correction (Cohort-Specific Z-Scoring)**: Within-cohort Z-score standardisation eliminates study-based grouping. Patients from Liu 2019, Hugo 2016, and Riaz 2017 mix completely across the hierarchical tree, confirming that gene-level technical baseline shifts have been successfully removed.

## 4. Cross-Validation Rigor & Data Leakage Prevention

> [!summary] Why We Are Doing This  
> Predictive models must be tested on unseen patient cohorts to prove their real-world clinical utility. If a batch correction algorithm uses the test cohort to calculate its transformation parameters, information from the test set "leaks" into the training phase. We use cohort-independent Z-score standardisation to prevent data leakage during Leave-One-Cohort-Out (LOCO) cross-validation.

### The Hazard of Global Batch Correction (e.g. ComBat)
Popular batch correction tools like ComBat pool all samples across all studies together to estimate correction parameters. When performing Leave-One-Cohort-Out cross-validation, including the held-out test cohort in these calculations allows the model to indirectly "peek" at test set distributions. This introduces **data leakage**, producing artificially inflated accuracy scores that fail to generalise to new clinical cohorts.

### The Cohort-Independent Z-Score Solution
Standardising gene expression independently within each cohort (using only that study's internal mean and standard deviation) ensures that zero information crosses cohort boundaries during cross-validation. Each held-out test cohort remains completely isolated, guaranteeing strict evaluation of true model generalisability.
