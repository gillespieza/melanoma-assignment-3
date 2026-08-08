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
created: 2026-08-08 15:39
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-08 15:39
---

# Batch Effect Assessment & Dimensionality Reduction Analysis

This report documents how technical batch effects were evaluated and harmonised across four melanoma cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, and **Riaz 2017**) and tests if global profiles separate therapeutic responses.

## 1. Cohort Batch Assessment

### 1.1 Full Cohort Batch Assessment (N = 699)

> [!INFO] Why We Are Doing This
>
> **What**: PCA across $N = 699$ patients from four cohorts (**TCGA-SKCM** [$N = 443$], **Liu 2019** [$N = 122$], **Hugo 2016** [$N = 27$], **Riaz 2017** [$N = 107$]) using 1,000 genes from 19,757 common genes.
> **Why**: Combining transcriptomic data introduces batch effects. Uncorrected models risk classifying sequencing centres rather than patient biology.
> **Question Answered**: Does cohort-independent Z-score standardisation eliminate technical separation between reference (TCGA) and trial cohorts?

![[batch_effect_pca.png]]

### Key Observations
- **Raw**: Separation between TCGA and trial cohorts. Uncorrected PC1 (95.3%) and PC2 (0.7%) reflect platform shifts.
- **Corrected**: Standardisation ($\mu=0, \sigma=1$ per study) aligns datasets. Post-correction PC1 (13.2%) and PC2 (9.6%) show homogeneous spread.

### 1.2 ICI Trial Cohort Batch Assessment (N = 256)

> [!INFO] Why We Are Doing This
>
> **What**: Technical effects between 3 training cohorts (**Liu 2019** [$N = 122$], **Hugo 2016** [$N = 27$], **Riaz 2017** [$N = 107$]; $N = 256$) across 58,954 trial genes.
> **Why**: Trials vary by platform, tissue state, and treatment. Verify baseline offsets are eliminated before LOCO cross-validation.
> **Question Answered**: Are inter-trial offsets harmonised without leaking test data?

![[batch_effect_ici_pca.png]]

### Key Observations
- **Raw**: In $\log_2(\text{TPM})$, `Liu 2019` ($N = 122$) separates from `Riaz 2017` ($N = 107$) and `Hugo 2016` ($N = 27$) along PC1 (28.9%), confirming sequencing depth/platform dominate signals.
- **Corrected**: Standardisation removes study-level separation. distributions overlap smoothly across PC1 (7.0%) and PC2 (4.2%).

## 2. Immunotherapy Trial Dimensionality Reduction (N = 256)

> [!INFO] Why We Are Doing This
>
> **What**: Linear (PCA) and non-linear (UMAP) reduction to $N = 256$ response-annotated patients (Liu 2019, Hugo 2016, Riaz 2017) using 1,000 genes.
> **Why**: Test if baseline expression profiles naturally segregate responders.
> **Question Answered**: Can therapeutic response be predicted directly from global 2D expression clusters?

### Projections
![[pca_dimensionality_reduction.png]]
![[umap_dimensionality_reduction.png]]

> [!INSIGHT] Key Insights
>
> - **Harmonisation**: Z-score scaling integrates `Liu 2019` ($N = 122$), `Riaz 2017` ($N = 107$), `Hugo 2016` ($N = 27$) in embeddings.
> - **Mixing**: Responders (CR/PR) and non-responders (PD) mix homogeneously.
> - **Biological Rationale**: Response is driven by multi-pathway immune features, not global variance. Simple 2D projections cannot separate response groups.

## 3. Gene-Level Expression Heatmaps (Top 50 Genes)

> [!INFO] Why We Are Doing This
>
> **What**: Inspect individual gene heatmaps for top 50 genes across trial patients ($N = 256$) with hierarchical clustering.
> **Why**: Validate batch effects at individual gene resolutions.
> **Question Answered**: Does Z-score prevent gene-based cohort clustering?

### Raw & Standardised
![[heatmap_top_variance_genes_raw.png]]
![[heatmap_top_variance_genes_standardized.png]]

### Key Observations
- **Raw**: Columns cluster by cohort source, showing distinct blocks.
- **Corrected**: Within-cohort Z-score standardisation eliminates study-based clustering, producing complete cohort mixing.

## 4. Cross-Validation Rigor & Data Leakage Prevention

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
> - **Constraints**: `Hugo 2016` ($N = 27$) has lower power than `Liu 2019` ($N = 122$) and `Riaz 2017` ($N = 107$).
> - **Scope**: Projections confirm single-gene thresholds are insufficient, motivating a 12-feature multimodal ensemble approach.

