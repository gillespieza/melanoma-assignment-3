---
title: "Cohort Specification: Gide 2019 (mel_iatlas_gide_2019)"
aliases:
  - Gide 2019
  - Gide Cancer Cell 2019
  - mel_iatlas_gide_2019
  - PRJEB23709
tags:
  - clinical-trials
  - dataset
  - gide-2019
  - iatlas
  - immunotherapy
  - melanoma
created: 2026-09-15 12:41
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-09-15 12:42
---

# Dataset Overview

`mel_iatlas_gide_2019` is the iAtlas-harmonised version of the metastatic melanoma clinical trial cohort described by Gide et al. (2019). The study investigated immune-cell subsets, transcriptomic profiles, and spatial architectures associated with response and resistance to anti-PD-1 monotherapy versus combined anti-PD-1 + anti-CTLA-4 blockade in advanced cutaneous melanoma.

> [!abstract] Primary Reference  
> Gide TN, Quek C, Menzies AM, et al. (2019). _Distinct Immune Cell Populations Define Response to Anti-PD-1 Monotherapy and Anti-PD-1/Anti-CTLA-4 Combined Therapy_. Cancer Cell, 35(2):238–255.e6. [DOI: 10.1016/j.ccell.2019.01.003](https://doi.org/10.1016/j.ccell.2019.01.003)

> [!note] cBioPortal Study Page  
> [https://www.cbioportal.org/study/summary?id=mel_iatlas_gide_2019](https://www.cbioportal.org/study/summary?id=mel_iatlas_gide_2019)

> [!info] ENA BioProject Accession  
> [PRJEB23709](https://www.ebi.ac.uk/ena/browser/view/PRJEB23709) (Transcriptomic RNA sequencing of immunotherapy-treated metastatic melanoma biopsies)

The dataset serves as one of the four primary development and cross-validation cohorts in this project for predicting response to anti-PD-1 therapy using a curated panel of multimodal biological and genomic features.

# Clinical Context

The original Melanoma Institute Australia trial enrolled patients into two therapeutic arms:
1. **Anti-PD-1 Monotherapy**: Nivolumab or Pembrolizumab
2. **Combination Checkpoint Blockade**: Anti-PD-1 + Anti-CTLA-4 (Nivolumab + Ipilimumab)

For the project's primary baseline anti-PD-1 response predictor, patients receiving anti-PD-1 monotherapy represent the target population. Combining monotherapy and dual-agent cohorts without accounting for regimen differences can confound baseline predictive signals, as combination regimens exhibit higher baseline response rates through complementary biological mechanisms.

# Molecular Data

The Gide study generated deep transcriptomic sequencing and multiparametric immunophenotyping from tumour biopsies.

## Transcriptomic Data (RNA-seq)

Pre-treatment RNA sequencing ($\log_2(\text{TPM} + 1)$) across 59,409 genes provides the primary molecular layer for computing the project's canonical immune signatures:
1. `IFN_gamma` (Interferon-gamma signalling score; 6 genes)
2. `TIS` (Tumour Inflammation Score; 18 genes)
3. `CYT` (Cytolytic Activity Score; `PRF1`, `GZMA`)
4. `CD8_Tcell` (CD8+ T-cell infiltration proxy; `CD8A`, `CD8B`)
5. `IMPRES` (Immune Predictive Score; 15 checkpoint pair relations)
6. `PD_L1` (Direct checkpoint ligand expression proxy; `CD274`)
7. Macrophage deconvolution metrics (Macrophage STV and M1/M2 polarisation ratios)

## Immune-Cell Subpopulation Profiling

A hallmark finding of Gide et al. was that distinct microenvironmental architectures govern response to monotherapy versus combination blockade. Responders to anti-PD-1 monotherapy were characterised by pre-existing CD8+ T-cell infiltration and high baseline interferon-gamma signalling, whereas combination therapy was capable of rescuing tumours with lower baseline immune infiltration.

## Somatic Genomic Data (WES Status)

Unlike Liu, Hugo, and Riaz, public somatic mutation files (MAF) are **not available** in the cBioPortal asset for Gide 2019. In the project's multimodal data pipeline, missing driver mutations (`mut_BRAF`, `mut_NRAS`, `mut_NF1`) are safely initialised to zero (or handled as missing values), and the cohort contributes primarily to transcriptomic validation folds.

# Cohort Subsetting & Attrition

The headline study numbers must be carefully distinguished from the baseline modelling subset:

| Cohort Stratification | Sample Count ($N$) | Description |
|:---|:---:|:---|
| ENA BioProject Experiments | 91 | Total sequencing experiments covering monotherapy and combination |
| cBioPortal Asset Samples | 91 | Harmonised samples in `mel_iatlas_gide_2019` |
| Primary Anti-PD-1 Monotherapy Cohort | 41 | Benchmark baseline subset receiving anti-PD-1 monotherapy |
| Responders in Monotherapy Subset | 19 | Objective responders (CR/PR; 46.3%) |
| Non-Responders in Monotherapy Subset | 22 | Objective non-responders (SD/PD; 53.7%) |
| Combination Arm (Nivo + Ipi) | 50 | Patients receiving dual-agent blockade (investigated separately) |

> [!tip] Cohort Attrition & Monotherapy Filtering Rule  
> The 91 samples in the cBioPortal release encompass both monotherapy and combination-treated patients. For strict anti-PD-1 monotherapy modelling, the benchmark cohort comprises $N = 41$ pre-treatment samples. In pooled exploratory multi-cohort analyses, the treatment regimen (`TREATMENT_TYPE`) is explicitly tracked to avoid conflating monotherapy with combination response biology. Attrition is tracked programmatically in `data/processed/gide_2019/attrition.csv`.

# Role in This Project

`mel_iatlas_gide_2019` is a **primary development and cross-validation cohort**, rather than an external validation-only cohort.

It represents one of the four core anti-PD-1 melanoma trial cohorts evaluated in this project:
1. **Hugo 2016** ($N = 26$ evaluable)
2. **Liu 2019** ($N = 121$ evaluable)
3. **Riaz 2017** ($N = 68$ evaluable baseline)
4. **Gide 2019** ($N = 41$ monotherapy / $N = 91$ overall evaluable)

The project employs Leave-One-Cohort-Out (LOCO) cross-validation to evaluate whether the 12-feature biological panel generalises across independent trial populations, testing true inter-study transportability.

# Methodological & Modelling Considerations

## Combination Therapy vs Monotherapy Stratification

> [!caution] Combination Blockade Stratification Warning  
> The Gide cohort contains both anti-PD-1 monotherapy and dual anti-PD-1 + anti-CTLA-4 treated patients.  
>
> **Methodological Rule**: Combination blockade must not be pooled into anti-PD-1 monotherapy training sets without explicit stratification. Dual checkpoint blockade remodels tumours through distinct immunological mechanisms and produces higher baseline objective response rates (~70% overall across the combined Gide asset vs 46.3% in monotherapy).

## Longitudinal Timing

- **Pre-treatment Samples Only**: Filter strictly to pre-treatment biopsies. On-treatment biopsies reflect early pharmacodynamic response rather than pre-existing predictive biomarkers.
- **Harmonised Response Definition**: Response criteria are standardised to RECIST v1.1 (CR/PR = 1, SD/PD = 0) matching Liu, Hugo, and Riaz.
- **Handling Absent WES**: Accommodate the absence of somatic mutation MAF data by evaluating transcriptomic sub-models and recording feature provenance.

# Key Limitations

- **Absent Somatic Mutation Data**: Lack of public WES calls precludes direct patient-level TMB and driver mutation evaluation for this specific cohort.
- **Moderate Monotherapy Sample Size**: At $N = 41$ pre-treatment monotherapy samples (19 responders, 22 non-responders), single-cohort statistical power is limited, reinforcing the necessity of multi-cohort LOCO benchmarking.
- **Regimen Heterogeneity**: Co-existence of monotherapy and combination therapy requires rigorous clinical metadata filtering during data ingestion.

# Summary
> [!summary] Dataset Summary Card  
> - **Study**: Gide et al. (2019)  
> - **Disease**: Advanced/metastatic cutaneous melanoma  
> - **Regimens**: Anti-PD-1 monotherapy (pembrolizumab / nivolumab) and combination (ipilimumab + nivolumab)  
> - **Primary Modality**: RNA sequencing (TPM, `PRJEB23709`)  
> - **cBioPortal Study ID**: `mel_iatlas_gide_2019`  
> - **cBioPortal Cohort Size**: 91 samples  
> - **Core Anti-PD-1 Monotherapy Cohort**: $N = 41$ (19 responders [46.3%], 22 non-responders [53.7%])  
> - **Overall Evaluated Asset**: $N = 91$ (monotherapy + combination)  
> - **Pipeline Role**: Core training & LOCO cross-validation cohort  
