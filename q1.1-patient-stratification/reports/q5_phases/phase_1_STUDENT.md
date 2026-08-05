---
title: "Phase 1: Multi-Modal Feature Engineering & TME Characterisation"
aliases:
  - Phase 1 Presentation Guide
  - Q5 Phase 1 Student Report
tags:
  - melanoma
  - patient-stratification
  - phase-1
  - presentation
  - q5
created: 2026-08-01 12:15
cssclasses:
  - row-alt
  - table-center
  - table-small
updated: 2026-08-01 12:15
---

# Phase 1: Multi-Modal Feature Engineering & TME Characterisation 🧬

A slide-ready presentation summary of **Phase 1** in the Question 5 Patient Stratification pipeline. This report provides an intuitive biological, clinical, and data-driven overview of how multi-modal patient profiles are constructed from bulk transcriptomics, genomics, and clinical metadata. All reported sample counts and feature metrics are derived directly from live pipeline outputs.

> [!NOTE] What, Why & What It Answers
> - **What**: Engineering a 33-feature multi-modal dataset spanning transcriptomics, cell-type deconvolution, spatial microenvironment proxies, and driver genomics.
> - **Why**: Individual biomarkers (such as single-gene expression or mutation status alone) fail to capture the complex, multi-cellular interactions that determine whether a melanoma patient will respond to immune checkpoint blockade.
> - **What it answers**: How can we integrate disparate molecular data streams into a unified, high-dimensional representation of each patient's tumour microenvironment?

---

## 1. Phase Overview

Phase 1 serves as the foundational data integration engine for the entire Question 5 pipeline. It harmonises clinical, gene expression, and genomic mutation data from four independent cutaneous melanoma studies (*Liu 2019*, *Riaz 2017*, *Hugo 2016*, and *TCGA-SKCM*) into two structured feature matrices:

1. **Immunotherapy Cohort Matrix** ($N = 326$ patients): Contains anti-PD-1 and anti-CTLA-4 treated patients with verified RECIST clinical response labels (42.1% responder rate). This dataset powers downstream feature discovery (Phase 2), unsupervised patient clustering (Phase 5), and clinical utility modelling (Phase 6).
2. **Full Melanoma Reference Matrix** ($N = 699$ patients): Merges the immunotherapy cohort with the broader TCGA-SKCM reference cohort ($N = 443$). This expanded dataset provides statistical power for cohort-wide phenotype profiling (Phases 3 and 4) and subpopulation routing (Phase 7).

Each patient is characterised across **33 engineered features** (19 transcriptomic and 14 genomic), accompanied by 12 clinical and administrative metadata columns.

---

## 2. Upstream Project Integration

Phase 1 unifies biological insights and analytical pipelines developed across earlier stages of the melanoma research project:

- **Integration with Question 1 (Predictive Immunotherapy Signatures)**: Incorporates established bulk expression signatures, including the Tumor Inflammation Signature (TIS), Cytolytic Activity Score (CYT), Interferon-Gamma (IFN-γ) 6-gene panel, PD-L1 transcript proxy, and the Immune Predictive Score (IMPRES).
- **Integration with Question 2 & Question 3 (Targeted Therapy & Driver Genomics)**: Integrates driver mutation status for the three canonical cutaneous melanoma subtypes (`BRAF`, `NRAS`, and `NF1`), connecting targeted therapy genomic landscapes to immune microenvironment phenotypes.
- **Integration with Question 4 (Pathway & Microenvironment Profiling)**: Extends pathway-level analysis by deconvoluting bulk RNA-seq into seven distinct cell types (CD8+ T cells, CD4+ T cells, NK cells, B cells, M1 Macrophages, M2 Macrophages, and Cancer-Associated Fibroblasts) and calculating Macrophage Polarisation (STV score and M1/M2 ratio).

---

## 3. Clinical Relevance: Why This Matters for Patient Stratification (Q5)

Immune checkpoint inhibitors (such as anti-PD-1) have transformed melanoma care, yet over half of treated patients fail to achieve durable clinical benefit. A primary challenge in clinical oncology is that single biomarkers — such as PD-L1 immunohistochemistry or Tumor Mutational Burden (TMB) alone — exhibit limited predictive accuracy when used in isolation.

Phase 1 addresses this clinical bottleneck by constructing a **multi-dimensional landscape** of the tumour microenvironment:

- **Capturing Immune Infiltration**: Quantifying cytotoxic CD8+ T cells and natural killer cells measures existing anti-tumour immunity.
- **Measuring Immunosuppressive Barriers**: Deconvoluting M2-like macrophages and cancer-associated fibroblasts identifies active physical and chemical resistance mechanisms.
- **Quantifying Mutational Antigenicity**: TMB and neoantigen burden quantify the tumour's visibility to the host immune system.
- **Modeling Spatial Infiltration**: Engineered proxy ratios approximate whether cytotoxic lymphocytes are actively infiltrating the tumour core or trapped in surrounding stroma.

By combining these complementary biological dimensions into a single matrix, Phase 1 creates the substrate required to discover distinct biological subtypes of melanoma that respond differently to therapy.

---

## 4. Key Phase Results & Biomarker Profiles

### Multi-Modal Feature Composition (33 Features)

- **Transcriptomic Features (19 features)**:
  - *Core Signatures (6)*: TIS, CYT, IFN-γ 6-gene panel, CD8 T-cell signature, PD-L1 proxy (`CD274`), and IMPRES score.
  - *Deconvoluted Cell Types (7)*: Relative abundance scores for CD8+ T cells, CD4+ T cells, NK cells, B cells, M1 Macrophages, M2 Macrophages, and CAFs.
  - *Macrophage Polarisation (4)*: M1 score, M2 score, M1/M2 ratio, and Macrophage Signature Transcript Vector (STV).
  - *Spatial Microenvironment Proxies (2)*: CD8-to-CAF distance ratio and Tumour Infiltration Index.
- **Genomic Features (14 features)**:
  - *Tumor Burden & Stability (4)*: Nonsynonymous TMB, Aneuploidy Score, MSI Mantis Score, and MSI Sensor Score.
  - *Neoantigen Subtypes (7)*: Single Nucleotide Variant (SNV), Indel, Fusion, Splice, Cancer-Testis Antigen (CTA), Viral, and Endogenous Retrovirus (ERV) neoantigens.
  - *Driver Mutation Status (3)*: Binary flags for `BRAF`, `NRAS`, and `NF1` mutations.

### Empirical Cohort Characteristics

- In the $N = 326$ immunotherapy cohort, 137 patients (42.1%) achieved complete or partial response (CR/PR), while 189 patients (57.9%) exhibited progressive disease (PD).
- Responders display significantly elevated baseline expression across all core immune signatures (TIS, CYT, IFN-γ) and higher infiltration of CD8+ T cells and B cells.
- Non-responders exhibit higher baseline M2 macrophage polarisation and elevated Cancer-Associated Fibroblast (CAF) abundance, confirming the role of stromal exclusion and myeloid suppression in treatment failure.

---

## 5. Limitations & Future Directions

> [!WARNING] Methodological Limitations & Analytical Constraints
> Derived directly from the Phase 1 Limitations Audit (`phase_1_LIMITATIONS.md`), key analytical considerations for downstream interpretation include:
> 
> 1. **Uncontrolled Inter-Cohort Batch Effects**: Merging four independent studies without explicit batch correction means study-of-origin could act as a confounding variable in downstream clustering. *Hugo 2016* represents a smaller sub-cohort ($N = 27$) compared to *Liu 2019* ($N = 122$).
> 2. **Genomic Feature Missingness**: While transcriptomic features achieve 100% completeness, specific genomic metrics exhibit high missingness (such as Aneuploidy Score at 79.4% missingness and MSI scores at 97.9% missingness across trial cohorts), requiring careful handling during machine learning.
> 3. **Spatial Proxies as Algebraic Approximations**: The CD8-to-CAF distance ratio and Tumour Infiltration Index are mathematical log-ratios of bulk expression, not physical spatial measurements from tissue imaging.
> 4. **One-Dimensional Macrophage Axis**: The M1/M2 polarisation ratio simplifies a complex, continuous spectrum of macrophage functional states into a single numerical scale.
> 
> **Future Iterations**: Priority improvements include implementing ComBat-seq batch correction across source cohorts, transitioning from marker-averaging to constrained single-cell RNA-seq deconvolution (CIBERSORTx), and integrating multiplex immunofluorescence to validate physical cell-cell distances.

---

> [!INSIGHT] Key Takeaways
> - **Unified Matrix Engine**: Phase 1 successfully integrates 33 multi-modal features across 326 immunotherapy patients (ICI cohort) and 699 total melanoma patients (full reference cohort).
> - **Multi-Cellular Perspective**: Combines adaptive immune activation (CD8+ T cells, IFN-γ, TIS) with stromal and myeloid resistance mechanisms (CAFs, M2 macrophages) and genomic antigenicity (TMB, driver mutations).
> - **Substrate for Stratification**: Provides the normalized input matrix powering all subsequent clustering, predictive modeling, and clinical decision-routing phases in the Q5 pipeline.
