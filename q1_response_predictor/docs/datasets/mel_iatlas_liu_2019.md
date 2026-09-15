---
title: "Cohort Specification: Liu 2019 (mel_iatlas_liu_2019)"
aliases:
  - Liu 2019
  - Liu Nature Medicine 2019
  - mel_iatlas_liu_2019
tags:
  - clinical-trials
  - dataset
  - iatlas
  - immunotherapy
  - liu-2019
  - melanoma
created: 2026-09-15 12:32
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-09-15 12:35
---

# Dataset Overview

`mel_iatlas_liu_2019` is the iAtlas-harmonised version of the metastatic melanoma cohort described by Liu et al. (2019). The original study investigated molecular and clinical predictors of response to anti-PD-1 blockade in patients with metastatic melanoma.

> [!abstract] Primary Reference  
> Liu D, Schilling B, Liu D, et al. (2019). _Integrative molecular and clinical modeling of clinical outcomes to PD1 blockade in patients with metastatic melanoma_. Nature Medicine, 25(12):1916–1927. [DOI: 10.1038/s41591-019-0654-5](https://doi.org/10.1038/s41591-019-0654-5)

> [!note] cBioPortal Study Page  
> [https://www.cbioportal.org/study/summary?id=mel_iatlas_liu_2019](https://www.cbioportal.org/study/summary?id=mel_iatlas_liu_2019)

The dataset serves as one of the four primary development and training cohorts in this project for predicting response to anti-PD-1 therapy using a curated panel of multimodal biological and genomic features.

# Clinical Context

Patients presented with advanced/metastatic cutaneous melanoma and were treated with anti-PD-1 immune checkpoint blockade:
- **Nivolumab**
- **Pembrolizumab**

The cohort directly aligns with the core clinical question of this project: baseline prediction of objective response to **anti-PD-1 therapy**.

Clinical metadata include treatment history and outcome variables:
- RECIST v1.1 objective response criteria (CR, PR, SD, PD)
- Progression-Free Survival (`PFS_MONTHS`, `PFS_STATUS`)
- Overall Survival (`OS_MONTHS`, `OS_STATUS`)
- Prior exposure to systemic therapy (including prior anti-CTLA-4 ipilimumab)

For this project, patients must satisfy predefined quality and clinical eligibility criteria before entering the modelling pipeline. The raw cBioPortal cohort size should therefore not be conflated with the final modelled sample size.

# Molecular Data

The Liu 2019 cohort is particularly valuable because it captures multi-omics profiles across the same patient tumours.

## Transcriptomic Data (RNA-seq)

Pre-treatment RNA sequencing ($\log_2(\text{TPM} + 1)$) enables the calculation of canonical immune and microenvironmental signatures:
1. `IFN_gamma` (Interferon-gamma signalling score; 6 genes)
2. `TIS` (Tumour Inflammation Score; 18 genes)
3. `CYT` (Cytolytic Activity Score; `PRF1`, `GZMA`)
4. `CD8_Tcell` (CD8+ T-cell infiltration proxy; `CD8A`, `CD8B`)
5. `IMPRES` (Immune Predictive Score; 15 checkpoint pair relations)
6. `PD_L1` (Direct checkpoint ligand expression proxy; `CD274`)
7. Macrophage deconvolution metrics (Macrophage STV and M1/M2 polarisation ratios)

## Somatic Genomic Data (WES)

Matched whole-exome sequencing supports patient-level genomic profiling:
- Non-synonymous Tumour Mutational Burden (`TMB_NONSYNONYMOUS`)
- Driver mutation hotspots in `BRAF` (Class 1 `mut_BRAF_V600` vs Class 2/3 `mut_BRAF_nonV600`)
- Oncogenic mutations in `NRAS` (`mut_NRAS`)
- Loss-of-function somatic alterations in `NF1` (`mut_NF1`)
- Other non-silent somatic mutations and copy-number alterations (CNAs)

This combination of transcriptomic immune infiltration and genomic driver status provides complete feature coverage for the project's 12-feature multimodal Random Forest classifier.

# Cohort Size & Attrition

- **Original Liu et al. Publication**: 144 tumour biopsies collected across 121 unique patients (accounting for longitudinal and on-treatment re-biopsies).
- **iAtlas-Harmonised cBioPortal Asset**: 122 patient/sample records.
- **Project Preprocessed Modelling Cohort ($N = 121$)**:
  - Sample filtering excludes 1 record lacking an evaluable binary RECIST endpoint.
  - **47 Responders** (CR/PR; **38.8%**)
  - **74 Non-Responders** (SD/PD; **61.2%**)

> [!tip] Cohort Attrition Rule  
> The cBioPortal sample count ($N = 122$) is not the final modelled sample size ($N = 121$). Attrition steps and exclusions are tracked programmatically in `data/processed/liu_2019/attrition.csv`.

# Role in This Project

`mel_iatlas_liu_2019` is a **primary development and cross-validation cohort**, rather than a hold-out test set.

It is one of four core anti-PD-1 melanoma clinical trial cohorts utilised for model development:
1. **Hugo 2016** ($N = 26$ evaluable)
2. **Liu 2019** ($N = 121$ evaluable)
3. **Riaz 2017** ($N = 68$ evaluable baseline)
4. **Gide 2019** ($N = 91$ evaluable)

The project employs a Leave-One-Cohort-Out (LOCO) cross-validation framework to assess whether the 12-feature panel generalises across independent trial populations, avoiding overly optimistic performance estimates from random in-cohort splits.

# Important Distinction: iAtlas-Harmonised Data

This dataset is not simply the raw Liu publication data re-hosted on cBioPortal:
- The iAtlas consortium normalised and harmonised expression counts, gene identifiers, and clinical annotations to common schemas.
- To prevent feature leakage and inconsistencies, feature provenance is strictly managed:
  1. **Directly supplied**: Clinical identifiers, RECIST labels, OS/PFS outcomes.
  2. **Derived by project pipeline**: Z-score standardised gene expression matrices, canonical immune signatures (`IFN_gamma`, `TIS`, `CYT`, `CD8_Tcell`, `IMPRES`, `PD_L1`), and binary driver flags (`mut_BRAF_V600`, `mut_NRAS`, `mut_NF1`).

# Modelling Considerations

- **Pre-treatment Tumours Only**: Restrict analyses to baseline pre-treatment samples to ensure valid predictive rather than pharmacodynamic modelling.
- **Anti-PD-1 Homogeneity**: Patient population is restricted to anti-PD-1 monotherapy (nivolumab or pembrolizumab).
- **Endpoint Harmonisation**: Binary response is harmonised to RECIST v1.1 criteria (CR/PR = 1, SD/PD = 0).
- **Deduplication**: Ensure each patient contributes exactly one pre-treatment baseline sample.

# Summary
> [!summary] Dataset Summary Card
> - **Study**: Liu et al. (2019)
> - **Disease**: Metastatic cutaneous melanoma
> - **Regimen**: Anti-PD-1 checkpoint blockade (nivolumab / pembrolizumab)
> - **Modalities**: RNA-seq (TPM), WES somatic mutations (MAF), clinical outcomes
> - **cBioPortal Study ID**: `mel_iatlas_liu_2019`
> - **Original Study Population**: 144 biopsies / 121 patients
> - **cBioPortal Cohort Size**: 122 samples
> - **Model-Eligible Cohort Size**: $N = 121$ (47 responders, 74 non-responders)
> - **Pipeline Role**: Core training & LOCO cross-validation cohort
