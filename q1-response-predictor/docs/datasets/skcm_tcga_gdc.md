---
title: "Cohort Specification: TCGA-SKCM GDC 2025 (skcm_tcga_gdc)"
aliases:
  - skcm_tcga_gdc
  - TCGA GDC 2025
  - TCGA-SKCM
  - TCGA-SKCM Cell 2015
tags:
  - dataset
  - gdc
  - genomics
  - melanoma
  - reference-cohort
  - tcga
  - transcriptomics
created: 2026-09-15 12:43
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-09-15 12:44
---

# Dataset Overview

`skcm_tcga_gdc` represents the harmonised **Skin Cutaneous Melanoma (SKCM)** cohort from The Cancer Genome Atlas (TCGA), retrieved and processed through the **NCI Genomic Data Commons (GDC)** data portal.

TCGA-SKCM is a comprehensive multi-omic repository generated primarily to characterise the somatic genomic landscape, transcriptomic architecture, and molecular subtypes of cutaneous melanoma, rather than to evaluate prospectively defined response to immune checkpoint blockade.

> [!abstract] Primary Reference  
> Cancer Genome Atlas Network. (2015). _Genomic Classification of Cutaneous Melanoma_. Cell, 161(7):1681–1696. [DOI: 10.1016/j.cell.2015.05.044](https://doi.org/10.1016/j.cell.2015.05.044)

> [!note] GDC Data Portal & cBioPortal Study Page  
> - **GDC Publication Page**: [https://gdc.cancer.gov/about-data/publications/skcm_2015](https://gdc.cancer.gov/about-data/publications/skcm_2015)  
> - **cBioPortal Study Page**: [https://www.cbioportal.org/study/summary?id=skcm_tcga_gdc](https://www.cbioportal.org/study/summary?id=skcm_tcga_gdc)

The original landmark publication established the canonical four-way genomic classification of cutaneous melanoma based on mutually exclusive driver alterations: `BRAF`-mutant, `NRAS`-mutant, `NF1`-mutant, and Triple-Wild-Type.

# Clinical Context & Structural Differences

Unlike the project's four core clinical trial cohorts (Hugo 2016, Liu 2019, Riaz 2017, and Gide 2019), TCGA-SKCM is **not a dedicated anti-PD-1 response cohort**.

Patients were enrolled across multiple academic medical centres to characterise untreated or surgically resected primary and metastatic melanoma:
- **Treatment Heterogeneity**: Treatment histories are highly variable, spanning adjuvant interferon, dacarbazine chemotherapy, surgical resection alone, radiation, and diverse post-progression regimens.
- **Incomplete Longitudinal Records**: As the NCI GDC explicitly highlights, treatment data were not systematically captured across all TCGA participating sites.
- **Absence of Standard RECIST anti-PD-1 Endpoint**: Prospective RECIST v1.1 objective response criteria linked to a defined baseline anti-PD-1 start date are unavailable.

> [!warning] Strict Training Exclusion Mandate (`merge_enabled: false`)  
> **TCGA GDC 2025 is strictly excluded from predictive model training**. Pooling TCGA cases into anti-PD-1 classifier training would introduce severe clinical confounding and label ambiguity. In `config/datasets.yaml`, this cohort is designated with `merge_enabled: false` and is reserved strictly for independent biological validation, downstream survival benchmarking, and mechanistic simulation.

# Molecular Data

TCGA-SKCM provides an exceptionally rich, multidimensional molecular profiling layer across matched tumours:

## Transcriptomic Data (RNA-seq)

Harmonised GDC RNA sequencing ($\log_2(\text{TPM} + 1)$) across 40,761 genes supports calculation of the project's canonical immune signatures:
1. `IFN_gamma` (Interferon-gamma signalling score; 6 genes)
2. `TIS` (Tumour Inflammation Score; 18 genes)
3. `CYT` (Cytolytic Activity Score; `PRF1`, `GZMA`)
4. `CD8_Tcell` (CD8+ T-cell infiltration proxy; `CD8A`, `CD8B`)
5. `IMPRES` (Immune Predictive Score; 15 checkpoint pair relations)
6. `PD_L1` (Direct checkpoint ligand expression proxy; `CD274`)
7. Macrophage STV deconvolution and M1/M2 polarisation ratios

## Somatic Genomic Data (WES)

Matched whole-exome sequencing supports comprehensive somatic alteration profiling across 17,238 genes:
- Non-synonymous Tumour Mutational Burden (`TMB_NONSYNONYMOUS`)
- Driver mutation classification: Class 1 `mut_BRAF_V600` vs Class 2/3 `mut_BRAF_nonV600`
- Oncogenic mutations in `NRAS` (`mut_NRAS`)
- Loss-of-function somatic alterations in `NF1` (`mut_NF1`)
- Copy-number alterations (GISTIC2) and aneuploidy scores

# Retrospective Immunotherapy Subsets

Through detailed timeline mining of supplementary treatment files (`data_timeline_treatment.txt`), a retrospective subset of **$N = 91$ patients** who received systemic immunotherapy (primarily ipilimumab, therapeutic vaccines, or interferon) can be identified.

> [!caution] Retrospective Immunotherapy Subset Confounding  
> While the $N = 91$ immunotherapy-treated TCGA subset is valuable for retrospective exploration, it remains fundamentally distinct from the prospective trial cohorts:  
> - Regimens were administered at varying disease stages (adjuvant vs post-progression).  
> - The primary treatment was rarely anti-PD-1 monotherapy (predating routine pembrolizumab/nivolumab approval).  
> - Measurable baseline tumour scans and irRECIST evaluations were not prospectively logged.  
> 
> Therefore, this subset is evaluated for **overall survival (OS) stratification**, but never combined into the binary anti-PD-1 response training set.

# Appropriate Pipeline Roles Across Subprojects

Although excluded from training, `skcm_tcga_gdc` plays vital roles across the project ecosystem:

## 1. Q1 Downstream Survival Validation
Model-predicted response probabilities (trained on the 4 trial cohorts) are projected onto TCGA patients to test whether high predicted response correlates with significantly improved real-world Overall Survival (Kaplan-Meier log-rank testing and Cox proportional hazards modelling).

## 2. Q1.1 Patient Stratification (Two-Stage GMM Phenotyping)
TCGA provides the large-scale ($N = 473$) reference population used to discover the four clinical phenotypes:
- **Immunosuppressive M2-High** ($36.6\%$)
- **Immune Cold** ($6.4\%$)
- **Immune Hot** ($48.8\%$)
- **Mutant-Driven** ($8.2\%$)

## 3. Q3 ODE Tumour-Immune Digital Twin
Patient-specific expression of `BRAF`, `MAP2K1`, `CD8A`, `CD274`, and `PRF1` parameterises the 4-module ordinary differential equation (ODE) kinetic simulations (vemurafenib and anti-PD-1 dose sweeps).

## 4. Q5 Clinical Decision Support Dashboard (OncoTwin)
Powers the individual patient digital twins, virtual patient passports, and multi-modal recommendation engines in the React workbench.

# Cohort Size & Attrition

| Cohort Layer | Sample Count ($N$) | Notes |
|:---|:---:|:---|
| Cleaned Clinical Records | 473 | Unique patient participants |
| Gene Expression Profiles | 472 | Aligned $\log_2(\text{TPM}+1)$ profiles (40,761 genes) |
| Somatic Mutation Profiles | 473 | Non-synonymous MAF calls (17,238 genes) |
| Retrospective Immunotherapy Subset | 91 | Documented exposure to checkpoint or immune agents |
| Primary Anti-PD-1 Training Cohort | 0 | **0 samples included (strictly excluded via `merge_enabled: false`)** |

> [!tip] Provenance & GDC Pipeline Standards  
> The GDC reprocessed all TCGA raw RNA-seq reads through standardized bioinformatic pipelines (STAR 2-pass alignment and GENCODE v36 gene annotations). GDC TPM counts are numerically distinct from older legacy TCGA Pan-Cancer Atlas 2018 RSEM estimates. Attrition and pipeline provenance are recorded in `data/processed/skcm_tcga_gdc/attrition.csv`.

# Key Limitations

- **No Prospective Anti-PD-1 RECIST Endpoint**: Principal reason for exclusion from Q1 training.
- **Incomplete Treatment Timeline Logging**: Systemic therapy duration and dosing are missing for a substantial fraction of patients.
- **Specimen Type Heterogeneity**: Includes both primary melanomas and regional/distant metastatic lymph node and visceral metastases.

# Summary
> [!summary] Dataset Summary Card  
> - **Project**: The Cancer Genome Atlas (TCGA) / NCI Genomic Data Commons (GDC)  
> - **Cancer Type**: Skin Cutaneous Melanoma (SKCM)  
> - **Study ID**: `skcm_tcga_gdc`  
> - **Dataset Scope**: $N = 473$ clinical patients ($472$ RNA-seq, $473$ WES)  
> - **Immunotherapy Subset**: $N = 91$ patients (heterogeneous immune regimens)  
> - **Q1 Role**: Independent biological reference and overall survival validation benchmark  
> - **Training Status**: Strictly excluded (`merge_enabled: false`)  
> - **Downstream Subprojects**: Primary patient cohort for Q1.1 GMM clustering, Q3 ODE digital twin, and Q5 OncoTwin dashboard  
