---
title: "Batch Effect Assessment & Dimensionality Reduction Analysis"
aliases:
  - Q1 Batch Correction Report
  - Batch Effect Assessment
tags:
  - melanoma
  - batch-correction
  - pca
  - umap
  - tme
  - transcriptomics
cssclasses:
  - table-small
  - table-center
  - row-alt
created: 2026-08-02 12:03
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-02 12:03
---

# Batch Effect Assessment & Dimensionality Reduction Analysis

When combining transcriptomic datasets across independent clinical studies, technical variations (e.g. sequencing platforms, RNA extraction methods, and library preparation protocols) typically dominate the underlying biological signals. This report documents how technical batch effects were evaluated and harmonised across four melanoma cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, and **Riaz 2017**) and tests whether global transcriptomic profiles naturally separate patients based on therapeutic response. Visualisations follow the Okabe-Ito colour guidelines used throughout the study.

## 1. Cohort Batch Assessment

### 1.1 Full Cohort Batch Assessment (N = 699)

> [!INFO] Why We Are Doing This
> **What**: We perform Principal Component Analysis (PCA) across all $N = 699$ patients from four combined melanoma cohorts (**TCGA-SKCM** [$N = 443$], **Liu 2019** [$N = 122$], **Hugo 2016** [$N = 27$], and **Riaz 2017** [$N = 107$]) using the top 1,000 most variable genes selected from the 19,757 common genes across all datasets.
> **Why**: Combining transcriptomic data from diverse sequencing centres introduces technical distortions (batch effects). Uncorrected models risk classifying sequencing centres rather than patient biology.
> **Question Answered**: Does cohort-independent Z-score standardisation eliminate macro-level technical separation between reference tissue (TCGA-SKCM) and active clinical trial cohorts?

![[batch_effect_pca.png]]

### Key Observations
- **Panel A: Before Batch Correction (Raw Data)**: The uncorrected PCA projection reveals a strong separation between the TCGA-SKCM reference dataset and the three clinical trial cohorts. Uncorrected PC1 (95.3% variance) and PC2 (0.7% variance) reflect laboratory platform shifts.
- **Panel B: After Cohort-Specific Z-Score Standardisation**: Cohort-wise Z-score standardisation (centering each gene to $\mu = 0, \sigma = 1$ within each study) aligns the TCGA-SKCM reference with trial cohorts. Post-correction PC1 (13.2% variance) and PC2 (9.6% variance) show homogeneous distribution across datasets.

### 1.2 ICI Trial Cohort Batch Assessment (N = 256)

> [!INFO] Why We Are Doing This
> **What**: We evaluate technical batch effects specifically between the three active anti-PD-1 training cohorts (**Liu 2019** [$N = 122$], **Hugo 2016** [$N = 27$], and **Riaz 2017** [$N = 107$]; $N = 256$) across all 58,954 common trial genes before and after cohort-wise Z-score standardisation.
> **Why**: These trials vary by platform (Illumina HiSeq 2500 vs HiSeq 2000), tissue state (fresh-frozen vs FFPE), and prior treatment. We must verify baseline offsets are eliminated before Leave-One-Cohort-Out (LOCO) cross-validation.
> **Question Answered**: Are inter-trial technical offsets harmonised across the model training cohorts without leaking test-set information?

![[batch_effect_ici_pca.png]]

### Key Observations
- **Panel A: Before Batch Correction (Uncorrected Raw Expression)**: In uncorrected $\log_2(\text{TPM})$ space across all 58,954 trial genes, `Liu 2019` ($N = 122$, HiSeq 2500) separates along PC1 (28.9% variance) from `Riaz 2017` ($N = 107$, HiSeq 2000 / FFPE) and `Hugo 2016` ($N = 27$, HiSeq 2000 / fresh-frozen). This confirms that sequencing depth and platform chemistry dominate raw expression signals.
- **Panel B: After Cohort-Wise Z-Score Standardisation**: Standardising gene expression independently within each cohort completely removes artificial study-level separation. The distributions for Liu 2019, Hugo 2016, and Riaz 2017 overlap smoothly across PC1 (7.0% variance) and PC2 (4.2% variance), ensuring unbiased model training.

## 2. Immunotherapy Trial Dimensionality Reduction (N = 256)

> [!INFO] Why We Are Doing This
> **What**: We apply linear (PCA) and non-linear (UMAP) dimensionality reduction to the $N = 256$ response-annotated trial patients (Liu 2019, Hugo 2016, Riaz 2017) using the top 1,000 variable genes.
> **Why**: To test whether baseline gene expression profiles naturally segregate treatment responders from non-responders prior to supervised machine learning.
> **Question Answered**: Can therapeutic response be predicted directly from global 2D expression clusters, or are targeted biomarker signatures required?

### PCA Projections (Raw vs. Standardised)
![[pca_dimensionality_reduction.png]]

### UMAP Projections (Raw vs. Standardised)
![[umap_dimensionality_reduction.png]]

> [!INSIGHT] Key Insights: Dimensionality Reduction & Patient Distribution
- **Cohort Harmonisation**: Z-score scaling successfully integrates `Liu 2019` ($N = 122$), `Riaz 2017` ($N = 107$), and `Hugo 2016` ($N = 27$) across both PCA and UMAP embeddings.
- **Homogeneous Response Mixing**: Responders (CR/PR) and non-responders (PD) mix homogeneously throughout PCA and UMAP projections, with zero global cluster separation by clinical outcome.
- **Biological Rationale**: Immunotherapy response is driven by multi-pathway immune microenvironment features (e.g. `CD274`, `PDCD1`, `IFNG` signalling) rather than global transcriptomic variance. Simple 2D projections cannot separate response groups, proving the necessity for supervised multivariate classifiers.

## 3. Gene-Level Expression Heatmaps (Top 50 Highly Variable Genes)

> [!INFO] Why We Are Doing This
> **What**: We inspect individual gene expression heatmaps for the top 50 most variable genes across trial patients ($N = 256$) with hierarchical clustering.
> **Why**: Dimensionality reduction aggregates thousands of genes into single axes. Heatmaps allow direct inspection of batch effects at individual gene resolutions.
> **Question Answered**: Does within-cohort Z-score standardisation prevent individual high-variance genes from clustering patients by study origin?

### Raw Expression (Top 50 Genes)
![[heatmap_top_variance_genes_raw.png]]

### Standardised Expression (Top 50 Genes)
![[heatmap_top_variance_genes_standardized.png]]

### Key Observations
- **Before Batch Correction (Raw log2-TPM)**: Patient columns cluster strongly by cohort source, with distinct blocks corresponding to individual clinical studies.
- **After Batch Correction (Cohort Z-Scoring)**: Within-cohort Z-score standardisation eliminates study-based clustering, producing complete cohort mixing across the hierarchical dendrogram.

## 4. Cross-Validation Rigor & Data Leakage Prevention

> [!INFO] Why We Are Doing This
> **What**: We compare cohort-independent Z-score standardisation against global batch correction algorithms (such as ComBat).
> **Why**: Data preprocessing methods used in cross-validation must strictly preserve test-set independence.
> **Question Answered**: How does cohort-independent Z-score scaling prevent data leakage during Leave-One-Cohort-Out (LOCO) evaluation?

### 4.1 The Hazard of Global Batch Correction (e.g. ComBat)
Global batch correction algorithms like ComBat estimate location and scale transformation parameters using all samples pooled across all available cohorts. When performing Leave-One-Cohort-Out (LOCO) cross-validation, including the held-out test cohort in parameter estimation allows information from the test set to leak into the training phase. This **data leakage** produces artificially inflated performance metrics that fail to generalise to external clinical validation sets.

### 4.2 The Cohort-Independent Z-Score Solution
Standardising gene expression independently within each cohort (rescaling each gene using only that cohort's internal mean $\mu$ and standard deviation $\sigma$) guarantees zero data leakage. Each held-out study remains completely unobserved during model training, ensuring robust, generalisable estimates of real-world predictive performance.

> [!WARNING] Methodological Limitations & Future Rationale
- **Sample Size Constraints**: The smallest training cohort (`Hugo 2016`, $N = 27$) has reduced statistical power compared to `Liu 2019` ($N = 122$) and `Riaz 2017` ($N = 107$).
- **Platform Heterogeneity**: Z-score scaling harmonises gene-wise means and variances but does not alter relative non-linear gene correlations within a single study.
- **Pipeline Scope**: Unsupervised projections confirm that single-gene thresholds are insufficient for response prediction, motivating the 12-feature multimodal ensemble (incorporating TMB, TIS, CYT, and driver mutations like `BRAF`, `NRAS`, `NF1`) evaluated in downstream Q1 phases.
