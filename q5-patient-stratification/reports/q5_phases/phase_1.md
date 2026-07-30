---
title: "Phase 1: Feature Engineering & Baseline Signature Distribution"
aliases:
  - Q5 Phase 1
tags:
  - melanoma
  - patient-stratification
  - phase-1
  - q5
created: 2026-07-30 14:27
cssclasses:
  - table-small
  - table-center
  - row-alt
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-30 14:27
---

## 1. Phase 1: Multi-Modal Feature Matrix & Microenvironment Deconvolution (N = 326)

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Loading preprocessed clinical, expression, and genomic data ($N = 326$) and engineering core immune signatures, Macrophage STV ratios, and cell deconvolution scores.
> - **Why we are doing it**: Raw gene expression matrices containing ~19,757 genes suffer from the curse of dimensionality. Dimensionality reduction into validated signature scores and cell-type fractions provides interpretable biological features.
> - **What question it answers**: What baseline immune and microenvironmental features best capture the state of tumour-infiltrating lymphocytes and immunosuppressive stroma?

Phase 1 integrates harmonised data from four clinical trials (*Liu 2019*, *Riaz 2017*, *Hugo 2016*, *TCGA-SKCM*). Rather than evaluating 19,757 genes independently, Phase 1 projects expression profiles onto curated biological axes:
- **Core Immune Signatures**: Tumour Inflammation Signature (`TIS`), Cytolytic Index (`CYT`, mean of `PRF1` and `GZMA`), Interferon-gamma (`IFN_gamma`), and `CD274` (PD-L1) expression.
- **Macrophage STV (`M1_M2_Ratio`)**: Computed using a linear Signature Transcript Vector ($W_g$, 14,837 genes) to quantify the balance between pro-inflammatory M1 macrophages ($W_g > 0$) and pro-tumour M2 macrophages ($W_g < 0$).
- **Transcriptomic Deconvolution**: Marker-based signature scores estimating the relative abundance of CD8+ T cells, CD4+ T cells, NK cells, B cells, M1 Macrophages, M2 Macrophages, and Cancer-Associated Fibroblasts (CAFs).

### Baseline Biomarker Feature Distributions

![Baseline Biomarker Feature Distributions](q5-patient-stratification/plots/phenotypes/baseline_response_violins.png)

### Key Takeaways
- **Dimensionality Reduction**: Successfully compressed ~19,757 transcriptomic features into 37 standardized, clinically interpretable biomarkers.
- **M1/M2 Polarisation**: The Macrophage STV score captures microenvironmental suppression that operates independently of total T-cell density.
