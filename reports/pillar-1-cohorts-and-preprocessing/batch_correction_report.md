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
created: 2026-08-08 19:16
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-08 19:16
---

# Batch Effect Assessment & Dimensionality Reduction Analysis

This report documents how technical batch effects were evaluated and harmonised across 6 melanoma cohorts (**Liu 2019**, **Hugo 2016**, **Riaz 2017**, **TCGA GDC 2025**, **Gide 2019**, **Van Allen 2015**) and tests if global profiles separate therapeutic responses.

## 1. Cohort Batch Assessment

### 1.1 Full Cohort Batch Assessment (N = 478)

> [!INFO] Why We Are Doing This
>
> **What**: PCA across $N = 478$ patients from cohorts (**Liu 2019** [$N = 122$], **Hugo 2016** [$N = 27$], **Riaz 2017** [$N = 107$], **TCGA GDC 2025** [$N = 91$], **Gide 2019** [$N = 91$], **Van Allen 2015** [$N = 40$]) using 1,000 genes from 20,918 common genes.
> **Why**: Combining transcriptomic data introduces batch effects. Uncorrected models risk classifying sequencing centres rather than patient biology.
> **Question Answered**: Does cohort-independent Z-score standardisation eliminate technical separation between reference and trial cohorts?

![[batch_effect_pca.png]]

### Key Observations
- **Raw**: Separation between reference and trial cohorts. Uncorrected PC1 (66.3%) and PC2 (16.4%) reflect platform shifts.
- **Corrected**: Standardisation ($\mu=0, \sigma=1$ per study) aligns datasets. Post-correction PC1 (13.9%) and PC2 (8.8%) show homogeneous spread.

## 2. Immunotherapy Trial Dimensionality Reduction (N = 478)

> [!INFO] Why We Are Doing This
>
> **What**: Linear (PCA) and non-linear (UMAP) reduction to $N = 478$ response-annotated patients (Liu 2019, Hugo 2016, Riaz 2017, TCGA GDC 2025, Gide 2019, Van Allen 2015) using 1,000 genes.
> **Why**: Test if baseline expression profiles naturally segregate responders.
> **Question Answered**: Can therapeutic response be predicted directly from global 2D expression clusters?

### Projections
![[pca_dimensionality_reduction.png]]
![[umap_dimensionality_reduction.png]]

> [!INSIGHT] Key Insights
>
> - **Harmonisation**: Z-score scaling integrates `Liu 2019` ($N = 122$), `Hugo 2016` ($N = 27$), `Riaz 2017` ($N = 107$), `TCGA GDC 2025` ($N = 91$), `Gide 2019` ($N = 91$), `Van Allen 2015` ($N = 40$) in embeddings.
> - **Mixing**: Responders (CR/PR) and non-responders (PD) mix homogeneously.
> - **Biological Rationale**: Response is driven by multi-pathway immune features, not global variance. Simple 2D projections cannot separate response groups.

## 3. Gene-Level Expression Heatmaps (Top 50 Genes)

> [!INFO] Why We Are Doing This
>
> **What**: Inspect heatmaps for top 50 genes by average within-cohort variance across trial patients ($N = 478$). Samples are sorted by cohort, then by responder status (CR/PR before PD) within each cohort, with no hierarchical column clustering, to make cohort-level baseline differences and response group separation directly readable.
> **Why**: Validate batch effects at individual gene resolution and assess whether per-cohort Z-score correction removes study-level baseline shifts while preserving responder vs. non-responder biological contrast.
> **Question Answered**: Are cohort expression baselines visibly harmonised after per-cohort Z-score standardisation, and does the response group signal become more consistent across cohorts?

### Raw & Standardised
![[heatmap_top_variance_genes_raw.png]]
![[heatmap_top_variance_genes_standardized.png]]

### Key Observations
- **Raw** (`log2(TPM+1)`, robust 2nd–98th percentile scaling): Cohort blocks are visible in the Cohort colour bar. Van Allen 2015 shows a notably elevated baseline due to its different normalisation pipeline. The four iAtlas cohorts (Liu 2019, Hugo 2016, Riaz 2017, Gide 2019) share similar expression scales, reflecting their common preprocessing.
- **Corrected** (per-cohort Z-score, $\mu=0$, $\sigma=1$): Study-level baseline offsets are removed. Gene expression patterns now reflect within-cohort biological variation rather than technical platform differences, with the responder/non-responder contrast becoming more consistent across cohorts.

## 4. Cross-Validation Rigour & Data Leakage Prevention

> [!INFO] Why We Are Doing This
>
> **What**: Compare cohort-independent Z-score against global batch correction.
> **Question Answered**: How does independent Z-score prevent leakage in LOCO?

### 4.1 The Hazard of Global Correction (e.g. ComBat)
Global algorithms pool all samples to estimate parameters. In LOCO CV, including the test set in parameter estimation leaks information into the training phase, inflating metrics artificially.

### 4.2 The Z-Score Solution
Standardising independently per cohort (using internal $\mu, \sigma$) guarantees zero leakage. Each held-out study remains unobserved during training.

> [!WARNING] Limitations
>
> - **Constraints**: Sample sizes vary across trial cohorts (`Liu 2019` ($N = 122$), `Hugo 2016` ($N = 27$), `Riaz 2017` ($N = 107$), `TCGA GDC 2025` ($N = 91$), `Gide 2019` ($N = 91$), `Van Allen 2015` ($N = 40$)).
> - **Scope**: Projections confirm single-gene thresholds are insufficient, motivating a multimodal ensemble feature approach.

---

> [!formula]+ Dimensionality Reduction Script Execution & Software Module Architecture
>
> - [`run_dimensionality_reduction.py`](../../scripts/pillar-1-cohort-preprocessing/run_dimensionality_reduction.py): Performs PCA and UMAP dimensionality reduction across all active ICI trial cohorts, evaluates cohort-independent Z-score standardisation against raw expression profiles, and produces `batch_correction_report.md`.
> - [`run_expression_heatmap.py`](../../scripts/pillar-1-cohort-preprocessing/run_expression_heatmap.py): Generates raw log2(TPM+1) and per-cohort Z-score heatmap visualisations for top high-variance genes across trial cohorts (`heatmap_top_variance_genes_raw.png`, `heatmap_top_variance_genes_standardized.png`).
> - [`data_loaders.py`](../../src/data_loaders.py): Provides `load_merged_immunotherapy()` to retrieve aligned raw and standardised expression matrices and clinical metadata across ICI trial cohorts.
> - [`styles.py`](../../../src/styles.py): Central definition of Okabe-Ito colour palettes (`COHORT_PALETTE`, `RESPONSE_PALETTE`).
