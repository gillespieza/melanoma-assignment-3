---
title: "Cohort Specification: TCGA-SKCM Pan-Cancer Atlas 2018 (skcm_tcga_pan_can_atlas_2018)"
aliases:
  - skcm_tcga_pan_can_atlas_2018
  - TCGA PanCan 2018
  - TCGA-SKCM PanCan Atlas
tags:
  - dataset
  - genomics
  - melanoma
  - pancan-atlas
  - reference-cohort
  - rppa
  - tcga
  - transcriptomics
created: 2026-09-15 12:45
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-09-15 12:46
---

# Dataset Overview

`skcm_tcga_pan_can_atlas_2018` is the Pan-Cancer Atlas (PanCanAtlas 2018) harmonised release of the **TCGA Skin Cutaneous Melanoma (SKCM)** cohort. The Pan-Cancer Atlas initiative reprocessed 33 tumour types through unified bioinformatic pipelines to eliminate cross-batch technical artefacts and characterise pan-cancer somatic drivers, immune infiltrates, and cell-of-origin patterns.

> [!abstract] Primary Reference  
> Hoadley KA, Yau C, Hinoue T, et al. (2018). _Cell-of-Origin Patterns Dominate the Molecular Classification of 10,000 Tumors from 33 Types of Cancer_. Cell, 173(2):291–304.e6. [DOI: 10.1016/j.cell.2018.03.022](https://doi.org/10.1016/j.cell.2018.03.022)

> [!note] cBioPortal Study Page  
> [https://www.cbioportal.org/study/summary?id=skcm_tcga_pan_can_atlas_2018](https://www.cbioportal.org/study/summary?id=skcm_tcga_pan_can_atlas_2018)

In this project, `skcm_tcga_pan_can_atlas_2018` serves as an **archival reference and quality-control benchmark**, specifically providing Reverse Phase Protein Array (RPPA) proteomic validation and batch-comparison data.

# Clinical Context & Pipeline Status

Like the modern GDC release, the Pan-Cancer Atlas SKCM cohort is a molecular landscape dataset rather than a prospective anti-PD-1 trial:
- **Heterogeneous Treatment Histories**: Captured across 448 patients, with varied adjuvant therapies, chemotherapy, and surgical interventions.
- **Incomplete Immunotherapy Endpoints**: Lacks prospective RECIST v1.1 response criteria linked to anti-PD-1 baseline timing.

> [!warning] Archival QC & Superseded Status (`merge_enabled: false`)  
> **Superseded for General Use**: For standard transcriptomic analyses and patient phenotyping, this dataset is superseded by the modern `skcm_tcga_gdc` (TCGA GDC 2025) release. In `config/datasets.yaml`, it is set to `merge_enabled: false` (strictly excluded from training and merging) and is retained exclusively for:  
> 1. Backward-compatible quality assurance and pipeline verification.  
> 2. RPPA phospho-protein kinetic validation in Q3 ODE digital twin modelling.  
> 3. Historical replication of earlier project findings.

# Molecular Data

The Pan-Cancer Atlas 2018 dataset is distinguished by its specific multi-omic data standards:

## Transcriptomic Data (RNA-seq RSEM)

Quantified using batch-normalised, upper-quartile normalised RSEM values ($\log_2(\text{RSEM} + 1)$) across 20,505 genes:
- Supports calculation of the 6 canonical immune signatures (`IFN_gamma`, `TIS`, `CYT`, `CD8_Tcell`, `IMPRES`, `PD_L1`).
- Entrez Gene IDs are systematically mapped to current Hugo Gene Symbols using `entrez_to_symbol_cache.json`.

## Somatic Genomic Data (MC3 Consensus Mutations)

Somatic mutations were derived from the Multi-Center Mutation Calling (MC3) consensus callset across 18,180 genes:
- High-confidence non-synonymous variant filtering.
- Somatic driver mutation classification for `BRAF` (`mut_BRAF_V600` vs `mut_BRAF_nonV600`), `NRAS` (`mut_NRAS`), and `NF1` (`mut_NF1`).
- Non-synonymous Tumour Mutational Burden (`TMB_NONSYNONYMOUS`).

## Unique Proteomic & Supplementary Layers

> [!tip] Distinct Modality Value: RPPA Proteomics & Hypoxia  
> 1. **Reverse Phase Protein Array (RPPA)**: Quantifies total and phospho-protein abundance (e.g. p-ERK `MAPK1`/`MAPK3`, p-MEK, AKT). This is indispensable for **Q3 Phase 5 RPPA validation** (`phase5_rppa_validation.py`), where simulated ODE steady-state pERK levels are correlated directly against physical RPPA measurements.  
> 2. **Hypoxia Annotations**: Integrates supplementary Winter hypoxia scores (`WINTER_HYPOXIA_SCORE`) from `data_clinical_supp_hypoxia.txt` to characterise microenvironmental oxygen deprivation.

# Differences: PanCan 2018 vs GDC 2025

| Feature / Attribute | `skcm_tcga_pan_can_atlas_2018` | `skcm_tcga_gdc` |
|:---|:---|:---|
| **Release Framework** | Pan-Cancer Atlas (Cell 2018) | NCI Genomic Data Commons (GDC 2025) |
| **Expression Metric** | Normalised $\log_2(\text{RSEM}+1)$ | Standardised $\log_2(\text{TPM}+1)$ |
| **Total Cleaned Samples** | $N = 448$ | $N = 473$ |
| **Gene Identifier Format** | Entrez Gene IDs (mapped via cache) | Direct Hugo Symbols / Ensembl |
| **Mutation Caller** | MC3 Consensus Callset | GDC Somatic Aggregation Pipeline |
| **Proteomic Profiling** | **Available (RPPA)** | Not natively bundled in GDC core |
| **Pipeline Role** | Archival reference & RPPA validation | Primary TCGA validation & Q1.1 clustering |

# Cohort Size & Attrition

| Cohort Layer | Sample Count ($N$) | Notes |
|:---|:---:|:---|
| Cleaned Clinical Records | 448 | Unique clinical sample records |
| Gene Expression Profiles | 443 | RSEM expression profiles across 20,505 genes |
| Somatic Mutation Profiles | 448 | MC3 consensus mutation calls across 18,180 genes |
| Primary Anti-PD-1 Training Cohort | 0 | **0 samples included (strictly excluded via `merge_enabled: false`)** |

> [!tip] Attrition Logging  
> Attrition steps and sample alignment details are logged in `data/processed/skcm_tcga_pan_can_atlas_2018/attrition.csv`.

# Key Limitations

- **Legacy Expression Metric**: RSEM counts are not directly comparable to modern TPM counts without rank transformation or cross-cohort Z-scoring.
- **Absence of Prospective Response Data**: Cannot evaluate binary anti-PD-1 response prediction.
- **Superseded by GDC**: Contains fewer total patients ($448$ vs $473$) than the updated GDC 2025 release.

# Summary
> [!summary] Dataset Summary Card  
> - **Project**: TCGA Pan-Cancer Atlas 2018  
> - **Cancer Type**: Skin Cutaneous Melanoma (SKCM)  
> - **Study ID**: `skcm_tcga_pan_can_atlas_2018`  
> - **Dataset Scope**: $N = 448$ clinical samples ($443$ RNA-seq, $448$ WES)  
> - **Distinct Capabilities**: RPPA phospho-protein quantification and Winter hypoxia scoring  
> - **Q1 Role**: Archival QC benchmark and historical reference  
> - **Q3 Role**: Primary dataset for RPPA pERK digital twin validation (`phase5_rppa_validation.py`)  
> - **Training Status**: Strictly excluded (`merge_enabled: false`)  
