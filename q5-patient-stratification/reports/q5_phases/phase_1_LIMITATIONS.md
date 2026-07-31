---
title: "Phase 1: Methodological Limitations, Biological Criticisms & Future Iterations"
aliases:
  - Phase 1 Criticisms & Future Roadmap
  - Q5 Phase 1 Limitations
tags:
  - future-work
  - limitations
  - melanoma
  - patient-stratification
  - phase-1
  - q5
created: 2026-07-31 11:18
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-31 11:19
---

# Phase 1: Methodological Limitations, Biological Criticisms & Future Iterations 🔍

An analytical audit of **Phase 1** in the Question 5 Patient Stratification pipeline, detailing methodological constraints, biological criticisms, and actionable technical solutions for future iterations.

## 1. Executive Summary & Audit Rationale
While Phase 1 successfully compresses ~19,757 raw transcriptomic features into an interpretable 26-feature multi-modal panel, several implicit assumptions constrain diagnostic sensitivity and spatial accuracy.

## 2. In-Depth Analysis of Limitations & Criticisms

### 1. Methodological Limitation: Marker Averaging vs. Constrained Linear Unmixing
- **Current Approach**: Transcriptomic cell deconvolution (`compute_cell_deconvolution`) computes relative infiltration scores by taking the sample-wise mean log-expression across 3–5 marker genes per cell type.
- **Scientific Criticism**: It does not solve a formal constrained linear unmixing or support vector regression problem (such as CIBERSORTx or EPIC: $Y = X \cdot B$, subject to $\sum B = 1, B \ge 0$). Marker averaging produces **unbounded, arbitrary score units** rather than true, non-negative cell-type percentage proportions ($0–100\%$).
- **Biological Impact**: Marker genes such as `TNF` or `TGFB1` are expressed across multiple cell types (e.g. tumour cells, endothelial cells, or activated lymphocytes), inducing cross-cell-type expression spillover and potential correlation artifacts.

### 2. Biological Limitation: Spatial Blindness of Bulk RNA-Seq
- **Current Approach**: Bulk RNA-seq homogenises the entire tumour biopsy core into a single average expression profile prior to sequencing.
- **Scientific Criticism**: Bulk transcriptomics cannot distinguish between an **Immune-Excluded** tumour (where CD8+ T cells and `CAFs` are both abundant, but T cells are physically trapped in the stroma surrounding the tumour margin) vs. an **Immune-Infiltrated (Hot)** tumour (where T cells physically infiltrate the malignant cell nest).
- **Clinical Impact**: Two patients with identical bulk `CD8_T_cells` and `CAFs` scores may experience completely opposite clinical outcomes due to spatial microenvironmental architecture that bulk sequencing inherently erases.

### 3. Technical Limitation: Absence of Direct Copy Number Alteration (CNA) Maps
- **Current Approach**: Trial cohorts (_Liu 2019_, _Riaz 2017_, _Hugo 2016_) lack processed GISTIC arm-level Copy Number Alteration (CNA) files.
- **Scientific Criticism**: Chromosomal instability (high aneuploidy burden) is a major driver of immune exclusion and anti-PD-1 (`CD274`) resistance. Phase 1 relies on Whole Exome Sequencing (WES) mutational burden (`TMB_NONSYNONYMOUS`) and single-gene transcript levels (`PTEN`, `CDKN2A`) as indirect proxies.
- **Data Impact**: Tumour Mutational Burden (TMB) and CNA burden can be decoupled (e.g. high-CNA, low-TMB tumours exist), leaving a potential blind spot for chromosomal structural variation.

### 4. Biological Oversimplification: 1D Macrophage Polarisation Axis
- **Current Approach**: The Macrophage STV score collapses macrophage biology into a 1D scalar balance (`M1_M2_Ratio`).
- **Scientific Criticism**: In human tumours, Tumour-Associated Macrophages (TAMs) exist along a continuous, multi-dimensional functional spectrum (e.g. M2a, M2b, M2c, tissue-resident, lipid-laden) rather than a strict binary M1 (antitumour) vs. M2 (pro-tumour) axis.

### 5. Sampling Limitation: Static Pre-Treatment Snapshots
- **Current Approach**: Phase 1 feature matrices are constructed exclusively from baseline, pre-treatment biopsies.
- **Scientific Criticism**: Tumour microenvironments are dynamic and evolve rapidly under therapeutic pressure (e.g. on-treatment T-cell recruitment, acquired resistance mutations, and adaptive `CD274` / PD-L1 upregulation). A static pre-treatment snapshot cannot capture early on-treatment immune dynamics occurring 2–4 weeks post-infusion.

> [!WARNING] Summary of Impact on Downstream Phases  
> Unbounded deconvolution scores and spatial blindness can propagate unmeasured variance into Phase 3 unsupervised clustering and Phase 5 subgroup predictive models, leading to potential misclassification of stroma-excluded tumours as immune-hot.

## 3. Proposed Fixes & Strategic Roadmap for Future Project Iterations

To address these limitations in future iterations of Question 5, we propose five concrete technical enhancements:

### Proposed Fix 1: Transition to CIBERSORTx / Single-Cell Reference Deconvolution
- **Technical Solution**: Replace marker gene averaging with CIBERSORTx or EPIC support vector regression using single-cell RNA-seq (scRNA-seq) melanoma reference matrices (e.g. Jerby-Arnon et al. or Tirosh et al.).
- **Expected Benefit**: Yields true, absolute cell-type fractions ($0–100\%$) subject to sum-to-one constraints, eliminating marker spillover and arbitrary scaling units.

### Proposed Fix 2: Integration of Spatial Transcriptomics & Multiplex Immunofluorescence (mIF)
- **Technical Solution**: Incorporate spatial transcriptomics (10x Visium / Xenium) or 7-colour multiplex immunofluorescence (mIF) image quantification.
- **Expected Benefit**: Formally calculates a **Spatial Infiltration Distance Metric** (distance from CD8+ T cells to malignant cells vs distance to `CAFs`), enabling the model to explicitly separate _Immune-Excluded_ from _Immune-Infiltrated_ phenotypes.

### Proposed Fix 3: Direct WGS / SNP-Array Processing for Arm-Level CNA & Aneuploidy
- **Technical Solution**: Process raw BAM/CRAM alignment files using CNVkit or ASCAT to generate explicit arm-level Copy Number Alteration (CNA) fractional loss/gain vectors and whole-genome duplication (WGD) indicators.
- **Expected Benefit**: Directly quantifies chromosomal instability independent of TMB, closing the proxy gap for immune exclusion driven by aneuploidy.

### Proposed Fix 4: Multi-State Macrophage Deconvolution via Single-Cell Signatures
- **Technical Solution**: Expand the 1D STV score into a 4-dimensional macrophage vector capturing distinct subtypes: $\text{TAM}_{\text{Pro-inflammatory}}$ (M1), $\text{TAM}_{\text{Angiogenic}}$ (M2a), $\text{TAM}_{\text{Immunosuppressive}}$ (M2c), and $\text{TAM}_{\text{Lipid-Laden}}$.
- **Expected Benefit**: Captures functional macrophage plasticity and resolves fine-grained immunosuppressive mechanisms.

### Proposed Fix 5: Longitudinal Paired Baseline & On-Treatment Biopsy Modelling
- **Technical Solution**: Integrate paired baseline (Day 0) and early on-treatment (Day 14–28) biopsy transcriptomics.
- **Expected Benefit**: Calculates dynamic **Delta Signatures** ($\Delta \text{TIS} = \text{TIS}_{\text{Day28}} - \text{TIS}_{\text{Day0}}$), capturing early T-cell expansion and adaptive immune reactivation that static baseline biopsies cannot observe.

## 4. Comprehensive Roadmap Comparison Matrix

| Limitation Category | Current Phase 1 Method | Proposed Technical Fix | Primary Tool / Platform | Expected Clinical & Analytical Benefit |
| :--- | :--- | :--- | :--- | :--- |
| **Cell Deconvolution** | Mean marker log-expression averaging | Constrained SVR linear unmixing | CIBERSORTx / EPIC | Bounded $0–100\%$ absolute cell fractions; no marker spillover. |
| **Spatial Architecture** | Bulk RNA-seq homogenisation | Spatial transcriptomics & mIF | 10x Visium / Xenium / mIF | Accurately separates stroma-excluded from tumour-infiltrated T cells. |
| **Genomic Instability** | Indirect TMB & single-gene expression proxies | Direct WGS / SNP-array copy number callouts | CNVkit / ASCAT / GISTIC 2.0 | Direct measurement of arm-level CNA and aneuploidy-driven exclusion. |
| **Macrophage Subtypes** | 1D M1/M2 binary ratio (`M1_M2_Ratio`) | 4-state single-cell macrophage vector | scRNA-seq reference matrices | Resolves M2a/M2c subtype heterogeneity and therapeutic drug targets. |
| **Biopsy Timing** | Static pre-treatment snapshot (Day 0) | Longitudinal paired biopsy modeling ($\Delta$ scores) | Paired Day 0 & Day 14–28 RNA-seq | Observes early on-treatment T-cell expansion and adaptive resistance. |

> [!IMPORTANT] Key Takeaways & Strategic Future Roadmap
> - **Methodological Evolution**: Moving from marker averaging to CIBERSORTx SVR unmixing will eliminate arbitrary score scaling and cross-cell spillover.
> - **Spatial Resolution**: Integrating spatial transcriptomics (10x Visium) resolves the spatial blindness of bulk RNA-seq, allowing precise detection of stroma-excluded tumours.
> - **Multi-Omics Precision**: Processing raw WGS/SNP-arrays for direct arm-level CNA calls will eliminate reliance on indirect TMB proxies for chromosomal instability.
