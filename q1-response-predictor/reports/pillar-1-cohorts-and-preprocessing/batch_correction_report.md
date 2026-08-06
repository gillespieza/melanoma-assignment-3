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
created: 2026-08-05 22:58
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-05 22:58
---

# Batch Effect Assessment & Dimensionality Reduction Analysis

When combining transcriptomic datasets across independent clinical studies, technical variations typically dominate underlying biological signals. Macro-level cohort batch assessment and Z-score alignment across the four melanoma cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, and **Riaz 2017**) are detailed in [curated_signatures_report.md](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/reports/pillar-3-transcriptomic-signatures/curated_signatures_report.md#32-cohort-batch-assessment--visualising-batch-correction-impact). This report focuses on unsupervised dimensionality reduction embeddings, high-variance gene heatmaps, and cross-validation data leakage prevention. Visualisations follow the Okabe-Ito colour guidelines used throughout the study.

## 1. Immunotherapy Trial Dimensionality Reduction (N = 256)

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

## 2. Gene-Level Expression Heatmaps (Top 50 Highly Variable Genes)

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

## 3. Cross-Validation Rigor & Data Leakage Prevention

> [!INFO] Why We Are Doing This
> **What**: We compare cohort-independent Z-score standardisation against global batch correction algorithms (such as ComBat).
> **Why**: Data preprocessing methods used in cross-validation must strictly preserve test-set independence.
> **Question Answered**: How does cohort-independent Z-score scaling prevent data leakage during Leave-One-Cohort-Out (LOCO) evaluation?

### 3.1 The Hazard of Global Batch Correction (e.g. ComBat)
Global batch correction algorithms like ComBat estimate location and scale transformation parameters using all samples pooled across all available cohorts. When performing Leave-One-Cohort-Out (LOCO) cross-validation, including the held-out test cohort in parameter estimation allows information from the test set to leak into the training phase. This **data leakage** produces artificially inflated performance metrics that fail to generalise to external clinical validation sets.

### 3.2 The Cohort-Independent Z-Score Solution
Standardising gene expression independently within each cohort (rescaling each gene using only that cohort's internal mean $\mu$ and standard deviation $\sigma$) guarantees zero data leakage. Each held-out study remains completely unobserved during model training, ensuring robust, generalisable estimates of real-world predictive performance.

> [!WARNING] Methodological Limitations & Future Rationale
- **Sample Size Constraints**: The smallest training cohort (`Hugo 2016`, $N = 27$) has reduced statistical power compared to `Liu 2019` ($N = 122$) and `Riaz 2017` ($N = 107$).
- **Platform Heterogeneity**: Z-score scaling harmonises gene-wise means and variances but does not alter relative non-linear gene correlations within a single study.
- **Pipeline Scope**: Unsupervised projections confirm that single-gene thresholds are insufficient for response prediction, motivating the 12-feature multimodal ensemble (incorporating TMB, TIS, CYT, and driver mutations like `BRAF`, `NRAS`, `NF1`) evaluated in downstream Q1 phases.

> [!formula]+ Batch Correction Script Execution & Software Module Architecture
> - **Primary Pipeline Execution Scripts**:
>   - [`run_dimensionality_reduction.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/exploratory_plots/run_dimensionality_reduction.py): Evaluates technical batch effects across four melanoma cohorts (TCGA-SKCM, Liu 2019, Hugo 2016, Riaz 2017), computes uncorrected vs. cohort Z-score standardised PCA/UMAP projections, generates top 50 variable gene heatmaps, and outputs `batch_correction_report.md`.
> - **Data Preprocessing & Loading Modules**:
>   - [`clean_data.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/clean_data.py): Preprocesses raw cohort clinical metadata and RNA-seq expression profiles into cleaned CSV matrices.
>   - [`merge_datasets.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/merge_datasets.py): Merges processed expression matrices across cohorts into harmonised pooled matrices (`expr_merged.csv`, `clin_merged.csv`).
>   - [`data_loaders.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/data_loaders.py): Provides helper loader functions (`load_liu_2019`, `load_hugo_2016`, `load_riaz_2017`) for retrieving expression and clinical data.
> - **Shared Cross-Question & Pipeline Modules**:
>   - [`run_pipeline.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/run_pipeline.py): Master Q1 pipeline orchestrator executing downstream modeling and evaluation.
>   - [`styles.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/src/styles.py): Single source of truth for Okabe-Ito colour palettes (`COHORT_PALETTE`, `RESPONSE_PALETTE`) and visualization presentation style.
