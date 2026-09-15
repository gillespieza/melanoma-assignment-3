---
title: "Cohort Specification: Riaz 2017 (mel_iatlas_riaz_nivolumab_2017)"
aliases:
  - CheckMate-038
  - mel_iatlas_riaz_nivolumab_2017
  - Riaz 2017
  - Riaz Cell 2017
tags:
  - clinical-trials
  - dataset
  - iatlas
  - immunotherapy
  - melanoma
  - riaz-2017
created: 2026-09-15 12:40
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-09-15 12:42
---

# Dataset Overview

`mel_iatlas_riaz_nivolumab_2017` is the iAtlas-harmonised version of the metastatic melanoma clinical trial cohort described by Riaz et al. (2017). The study investigated how melanoma tumours and their immune microenvironment evolve during treatment with the anti-PD-1 monoclonal antibody nivolumab, combining whole-exome sequencing (WES), transcriptome sequencing (RNA-seq), and T-cell receptor (TCR) sequencing across longitudinal biopsies.

> [!abstract] Primary Reference  
> Riaz N, Havel JJ, Makarov V, et al. (2017). _Tumor and Microenvironment Evolution during Immunotherapy with Nivolumab_. Cell, 171(4):934–949.e16. [DOI: 10.1016/j.cell.2017.09.028](https://doi.org/10.1016/j.cell.2017.09.028)

> [!note] cBioPortal Study Page  
> [https://www.cbioportal.org/study/summary?id=mel_iatlas_riaz_nivolumab_2017](https://www.cbioportal.org/study/summary?id=mel_iatlas_riaz_nivolumab_2017)

The dataset serves as one of the four primary development and cross-validation cohorts in this project for predicting response to anti-PD-1 therapy using a curated panel of multimodal biological and genomic features.

# Clinical Context

All patients presented with advanced/metastatic cutaneous melanoma and received anti-PD-1 checkpoint blockade:
- **Nivolumab** (3 mg/kg every 2 weeks as part of the CheckMate-038 trial)

The study specifically enrolled two distinct clinical sub-cohorts:
1. **Ipilimumab-Naïve**: Patients who had not previously received anti-CTLA-4 therapy.
2. **Ipilimumab-Progressors**: Patients who had received prior ipilimumab and subsequently experienced disease progression before initiating nivolumab.

This clinical setting directly addresses the primary focus of this project: baseline prediction of objective response to **anti-PD-1 therapy**, while introducing clinically realistic pre-treatment heterogeneity.

# Molecular Data

The Riaz study contains rich multimodal profiling across pre-treatment and on-treatment tumour specimens.

## Transcriptomic Data (RNA-seq)

Pre-treatment RNA sequencing ($\log_2(\text{TPM} + 1)$) across 59,383 genes enables calculation of the project's canonical immune and microenvironmental signatures:
1. `IFN_gamma` (Interferon-gamma signalling score; 6 genes)
2. `TIS` (Tumour Inflammation Score; 18 genes)
3. `CYT` (Cytolytic Activity Score; `PRF1`, `GZMA`)
4. `CD8_Tcell` (CD8+ T-cell infiltration proxy; `CD8A`, `CD8B`)
5. `IMPRES` (Immune Predictive Score; 15 checkpoint pair relations)
6. `PD_L1` (Direct checkpoint ligand expression proxy; `CD274`)
7. Macrophage deconvolution metrics (Macrophage STV and M1/M2 polarisation ratios)

The original study demonstrated extensive microenvironmental remodelling during nivolumab therapy, with objective responders showing early infiltration of CD8+ T cells and natural killer (NK) cells alongside a decrease in M1-polarised macrophages.

## Somatic Genomic Data (WES)

Matched whole-exome sequencing supports patient-level genomic profiling:
- Non-synonymous Tumour Mutational Burden (`TMB_NONSYNONYMOUS`)
- Driver mutation hotspots in `BRAF` (Class 1 `mut_BRAF_V600` vs Class 2/3 `mut_BRAF_nonV600`)
- Oncogenic mutations in `NRAS` (`mut_NRAS`)
- Loss-of-function somatic alterations in `NF1` (`mut_NF1`)
- Clonal neoantigen load and mutation tracking

The original publication revealed that responding tumours selectively lost candidate neoantigens and somatic mutations during therapy, reflecting immune-mediated clonal editing.

## T-Cell Receptor (TCR) Sequencing

The study also performed deep TCR sequencing to track the clonal expansion and diversity of tumour-infiltrating lymphocytes. While biologically informative, TCR repertoire metrics are **not part of the project's current 12-feature prediction panel**, which focuses strictly on widely accessible bulk RNA-seq and WES biomarkers.

# Cohort Subsetting & Attrition

Because the CheckMate-038 design included longitudinal collection (pre-treatment baseline and on-treatment Week 4 biopsies), raw repository counts must be distinguished from baseline predictive modelling instances:

| Cohort Stratification | Sample Count ($N$) | Description |
|:---|:---:|:---|
| cBioPortal Asset Samples | 107 | Total pre-treatment and on-treatment biopsy records |
| Unique Patients | 68 | Unique patient clinical trial participants |
| Pre-treatment Baseline Samples | 68 | Biopsies collected prior to first nivolumab dose |
| Objective Responders (CR/PR) | 24 | Evaluated responders in pre-treatment cohort (35.3%) |
| Objective Non-Responders (SD/PD) | 44 | Evaluated non-responders in pre-treatment cohort (64.7%) |
| Prior Ipilimumab Progressors | 35 | Patients with prior anti-CTLA-4 failure |
| Prior Ipilimumab Naïve | 33 | Checkpoint inhibitor-naïve patients |

> [!tip] Cohort Attrition Rule  
> For predictive modelling, only pre-treatment baseline biopsies ($N = 68$) are eligible. On-treatment samples reflect post-exposure pharmacodynamic remodelling rather than baseline predictive biology and are excluded from model training. Attrition is tracked programmatically in `data/processed/riaz_2017/attrition.csv`.

# Role in This Project

`mel_iatlas_riaz_nivolumab_2017` is a **primary development and cross-validation cohort**, rather than a hold-out test set.

It represents one of the four core anti-PD-1 melanoma trial cohorts evaluated in this project:
1. **Hugo 2016** ($N = 26$ evaluable)
2. **Liu 2019** ($N = 121$ evaluable)
3. **Riaz 2017** ($N = 68$ evaluable baseline)
4. **Gide 2019** ($N = 91$ evaluable)

The project employs Leave-One-Cohort-Out (LOCO) cross-validation to assess whether the 12-feature biological panel generalises across trial cohorts, testing true inter-study transportability.

# Methodological & Modelling Considerations

## Prior Ipilimumab Treatment Heterogeneity

> [!warning] Key Clinical Confounder: Prior Anti-CTLA-4 Exposure  
> Approximately half of the Riaz cohort ($35/68$, 51.5%) had previously progressed on ipilimumab before receiving nivolumab. Riaz et al. demonstrated that prior ipilimumab exposure induces clonal selection and alters baseline immune microenvironments.  
>
> In predictive modelling, treatment history must be explicitly tracked (via `PRIOR_CTLA4_STATUS` in `clin_cleaned.csv`) to test whether predictive models trained predominantly on checkpoint-naïve cohorts transfer robustly to pre-treated populations.

## Longitudinal Sampling & Leakage Prevention

> [!caution] Longitudinal Sampling & Data Leakage Risk  
> The cBioPortal asset contains paired pre-treatment and on-treatment specimens from the same individuals (`riaz_*_PRE` vs `riaz_*_ON`).  
>
> **Leakage Safeguard**: Multiple specimens from the same individual must never be treated as independent observations. In cross-validation folds, patient identity must serve as the grouping unit, and only pre-treatment samples are admitted to the response predictor to prevent data leakage.

## Preprocessing & Provenance

- **Pre-treatment Samples Only**: Restrict model inputs strictly to pre-treatment biopsies.
- **Harmonised Response Definition**: Align RECIST criteria with Liu, Hugo, and Gide (CR/PR = 1, SD/PD = 0).
- **Feature Provenance Tracking**: Distinguish iAtlas-supplied metadata from pipeline-derived Z-scored expression signatures and binary mutation flags.

# Key Limitations

- **Prior Treatment Heterogeneity**: Mixed checkpoint-naïve and ipilimumab-refractory biology introduces confounding that can attenuate single-biomarker predictive power.
- **Moderate Baseline Sample Size**: At $N = 68$ evaluable baseline tumours, subgroup analyses (e.g. evaluating Ipi-Naïve vs Ipi-Progressor performance independently) operate with reduced statistical power.
- **Longitudinal Attrition**: Not all patients contributed paired pre- and on-treatment tissue, requiring careful alignment across multi-omic layers.

# Summary
> [!summary] Dataset Summary Card  
> - **Study**: Riaz et al. (2017), CheckMate-038  
> - **Disease**: Advanced/metastatic cutaneous melanoma  
> - **Regimen**: Nivolumab monotherapy (anti-PD-1)  
> - **Primary Modalities**: RNA-seq (TPM), matched WES, TCR-seq  
> - **cBioPortal Study ID**: `mel_iatlas_riaz_nivolumab_2017`  
> - **cBioPortal Cohort Size**: 107 samples  
> - **Model-Eligible Cohort Size**: $N = 68$ (24 responders [35.3%], 44 non-responders [64.7%])  
> - **Subgroups**: 35 ipilimumab-progressors, 33 ipilimumab-naïve  
> - **Pipeline Role**: Core training & LOCO cross-validation cohort  
