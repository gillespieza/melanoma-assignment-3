---
title: "Q1 Response Predictor: Dataset Inventory & Harmonisation Specification"
aliases:
  - "Q1 Datasets"
  - "Immunotherapy Cohorts"
  - "Cohort Specifications"
tags:
  - datasets
  - data-harmonisation
  - melanoma
  - clinical-trials
  - transcriptomics
  - genomics
  - q1-response-predictor
created: 2026-09-15 12:28
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-09-15 12:28
---

# 🧬 Q1 Response Predictor: Dataset Inventory & Harmonisation Specification

> [!NOTE]
> **What**: Comprehensive technical and clinical inventory of all active immunotherapy trial cohorts and reference datasets utilised in the Q1 response prediction pipeline.  
> **Why**: Ensures complete provenance, methodological transparency, and biological comparability across heterogeneous clinical trials and genomic screens.  
> **Question**: How are clinical endpoints, transcriptomic profiles, and somatic mutations standardised across disparate multicentre cohorts for predictive modelling and out-of-cohort benchmarking?

---

## 1. Master Dataset Summary

The Q1 pipeline dynamically configures datasets via [`config/datasets.yaml`](090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/config/datasets.yaml). Cohorts are stratified into **Active Immunotherapy Trial Cohorts** (used for model training and Leave-One-Cohort-Out [LOCO] cross-validation), **Independent Validation & Reference Cohorts** (held out from training to avoid data leakage), and **Prospective Expansion Candidates**.

| Cohort Name | Study Identifier (`study_id`) | Primary Regimen / Treatment Target | Sample Size ($N$) | Data Modalities Available | Pipeline Role | cBioPortal Study Summary |
|:---|:---|:---|:---:|:---|:---|:---:|
| **Liu 2019** | `mel_iatlas_liu_2019` | Anti-PD-1 (Nivolumab / Pembrolizumab) | 122 | RNA-seq (TPM), WES Mutations, RECIST, OS/PFS | Training & LOCO CV | [cBioPortal](https://www.cbioportal.org/study/summary?id=mel_iatlas_liu_2019) |
| **Hugo 2016** | `mel_iatlas_hugo_ucla_2016` | Anti-PD-1 (Pembrolizumab) | 27 | RNA-seq (TPM), WES Mutations, RECIST, OS | Training & LOCO CV | [cBioPortal](https://www.cbioportal.org/study/summary?id=mel_iatlas_hugo_ucla_2016) |
| **Riaz 2017** | `mel_iatlas_riaz_nivolumab_2017` | Anti-PD-1 (Nivolumab; Ipi-Naïve & Ipi-Progressor) | 107 | RNA-seq (TPM), WES Mutations, RECIST, OS/PFS | Training & LOCO CV | [cBioPortal](https://www.cbioportal.org/study/summary?id=mel_iatlas_riaz_nivolumab_2017) |
| **Gide 2019** | `mel_iatlas_gide_2019` | Anti-PD-1 +/- Anti-CTLA-4 (Ipilimumab + Nivolumab) | 91 | RNA-seq (TPM), RECIST, PFS/OS | Training & LOCO CV | [cBioPortal](https://www.cbioportal.org/study/summary?id=mel_iatlas_gide_2019) |
| **TCGA GDC 2025** | `skcm_tcga_gdc` | Heterogeneous Cohort ($N=91$ IT-treated subset) | 473 (472 RNA) | RNA-seq (TPM), WES Mutations, Clinical Timeline | External Validation & Survival Benchmark | [cBioPortal](https://www.cbioportal.org/study/summary?id=skcm_tcga_gdc) |
| **TCGA PanCan 2018** | `skcm_tcga_pan_can_atlas_2018` | Treatment-naïve primary & metastatic reference | 448 (443 RNA) | RNA-seq (RSEM), WES Mutations, RPPA Proteomics | Legacy Reference & QC | [cBioPortal](https://www.cbioportal.org/study/summary?id=skcm_tcga_pan_can_atlas_2018) |

---

## 2. Active Immunotherapy Trial Cohorts (Training & LOCO CV)

### 2.1 Liu et al. (2019, Nature Medicine)
* **Study Overview**: Pre-treatment tumours from patients with metastatic melanoma treated with anti-PD-1 monotherapy (nivolumab or pembrolizumab). Part of the cancer immunotherapy research consortium (iAtlas).
* **cBioPortal Access**: [`https://www.cbioportal.org/study/summary?id=mel_iatlas_liu_2019`](https://www.cbioportal.org/study/summary?id=mel_iatlas_liu_2019)
* **Sample Numbers**:
  * Cleaned clinical samples: $N = 122$
  * Gene expression matrix: 122 samples $\times$ 59,409 genes
  * Somatic mutation matrix: 122 samples $\times$ 15,019 genes
  * Binary response evaluated: $N = 121$ ($47$ Responders [CR/PR], $74$ Non-responders [SD/PD], Response rate: 38.8%)
* **Clinical Endpoints**: RECIST v1.1 response criteria, Overall Survival (`OS_MONTHS`, `OS_STATUS`), Progression-Free Survival (`PFS_MONTHS`, `PFS_STATUS`), and prior systemic therapies.
* **Key Findings in Pipeline**: Serves as the primary anchor training cohort with rich somatic mutation depth, high sequencing quality, and balanced clinical demographics.

### 2.2 Hugo et al. (2016, Cell)
* **Study Overview**: Landmark whole-exome and transcriptomic study from UCLA examining pre-treatment and on-treatment biopsies from metastatic melanoma patients receiving pembrolizumab.
* **cBioPortal Access**: [`https://www.cbioportal.org/study/summary?id=mel_iatlas_hugo_ucla_2016`](https://www.cbioportal.org/study/summary?id=mel_iatlas_hugo_ucla_2016)
* **Sample Numbers**:
  * Cleaned clinical samples: $N = 27$
  * Gene expression matrix: 27 samples $\times$ 59,409 genes
  * Somatic mutation matrix: 27 samples $\times$ 12,628 genes
  * Binary response evaluated: $N = 26$ ($14$ Responders [CR/PR], $12$ Non-responders [SD/PD], Response rate: 53.8%)
* **Clinical Endpoints**: RECIST response and Overall Survival.
* **Key Findings in Pipeline**: Enriched for high baseline response; served as the original discovery benchmark for innate anti-PD-1 resistance signatures (IPRES).

### 2.3 Riaz et al. (2017, Cell)
* **Study Overview**: CheckMate-038 prospective trial investigating nivolumab monotherapy in metastatic melanoma, explicitly divided into two clinical cohorts: patients who progressed on prior ipilimumab (Ipi-Progressors) and patients who were ipilimumab-naïve (Ipi-Naïve).
* **cBioPortal Access**: [`https://www.cbioportal.org/study/summary?id=mel_iatlas_riaz_nivolumab_2017`](https://www.cbioportal.org/study/summary?id=mel_iatlas_riaz_nivolumab_2017)
* **Sample Numbers**:
  * Cleaned clinical samples: $N = 107$
  * Gene expression matrix: 107 samples $\times$ 59,383 genes
  * Somatic mutation matrix: 107 samples $\times$ 8,114 genes
  * Pre-treatment baseline with RECIST response: $N = 68$ ($24$ Responders, $44$ Non-responders, Response rate: 35.3%)
* **Clinical Endpoints**: RECIST response, prior anti-CTLA-4 exposure status, Overall Survival, and mutation load dynamics.
* **Key Findings in Pipeline**: Critical for evaluating resistance mechanisms induced by prior checkpoint exposure and separating treatment-naïve biology from therapy-induced immune editing.

### 2.4 Gide et al. (2019, Cancer Cell)
* **Study Overview**: Multicentre Australian study (Melanoma Institute Australia) examining baseline and early on-treatment biopsies from metastatic melanoma patients treated with anti-PD-1 monotherapy (pembrolizumab or nivolumab) versus combination anti-PD-1 + anti-CTLA-4 (ipilimumab + nivolumab).
* **cBioPortal Access**: [`https://www.cbioportal.org/study/summary?id=mel_iatlas_gide_2019`](https://www.cbioportal.org/study/summary?id=mel_iatlas_gide_2019)
* **Sample Numbers**:
  * Cleaned clinical samples: $N = 91$
  * Gene expression matrix: 91 samples $\times$ 59,409 genes
  * Somatic mutation matrix: Empty in public cBioPortal asset (handled safely via zero-initialisation in pipeline)
  * Binary response evaluated: $N = 91$ ($64$ Responders, $27$ Non-responders, Response rate: 70.3%)
* **Clinical Endpoints**: RECIST response, monotherapy vs combination therapy indicator, PFS, and OS.
* **Key Findings in Pipeline**: Provides critical evidence on combined checkpoint blockade efficacy; demonstrates the highest objective response rate among all trial cohorts due to dual-agent administration.

---

## 3. Reference & Independent Validation Cohorts

### 3.1 TCGA-SKCM GDC 2025 (`skcm_tcga_gdc`)
* **Study Overview**: Modern harmonised release from the NCI Genomic Data Commons (GDC) representing the largest public cutaneous melanoma collection ($N=473$).
* **cBioPortal Access**: [`https://www.cbioportal.org/study/summary?id=skcm_tcga_gdc`](https://www.cbioportal.org/study/summary?id=skcm_tcga_gdc)
* **Sample Numbers**:
  * Cleaned clinical samples: $N = 473$
  * Gene expression matrix: 472 samples $\times$ 40,761 genes ($\log_2(\text{TPM}+1)$)
  * Somatic mutation matrix: 473 samples $\times$ 17,238 genes
  * Immunotherapy-treated subset: $N = 91$ patients (identified via detailed treatment timelines receiving ipilimumab, interferon, or therapeutic vaccines)
* **Training Exclusion Policy**:
  > [!WARNING]
  > **Training Exclusion Mandate**: In accordance with clinical trial protocol integrity, **TCGA GDC 2025 is strictly excluded from predictive model training (`merge_enabled: false`)**. TCGA treatment protocols were non-randomised, heterogeneous, and frequently post-progression or adjuvant. Pooling TCGA into training would introduce severe clinical confounding. It is reserved exclusively as an independent validation set for overall survival stratification and downstream digital twin ODE kinetic simulation (Q3).
* **Role in Research Questions**:
  * **Q1 Downstream Survival Validation**: Independent validation of model-predicted response classes against true overall survival.
  * **Q1.1 (Patient Stratification)**: Unsupervised two-stage GMM phenotyping ($N=473$).
  * **Q3 (ODE Digital Twin)**: Parameterises 4-module ODE tumour-immune trajectories across individual patient profiles.
  * **Q5 (Dashboard / OncoTwin)**: Powers the digital patient passport and virtual clinical trial demonstrators.

### 3.2 TCGA-SKCM Pan-Cancer Atlas 2018 (`skcm_tcga_pan_can_atlas_2018`)
* **Study Overview**: Legacy Cell 2018 Pan-Cancer Atlas harmonisation. Expression quantified using batch-normalised RSEM values.
* **cBioPortal Access**: [`https://www.cbioportal.org/study/summary?id=skcm_tcga_pan_can_atlas_2018`](https://www.cbioportal.org/study/summary?id=skcm_tcga_pan_can_atlas_2018)
* **Sample Numbers**: 448 clinical, 443 expression, 448 mutation.
* **Pipeline Role**: Retained for backward-compatible quality assurance, RPPA phospho-protein validation in Q3, and historical reproduction.

---

## 4. Harmonised Multi-Cohort Matrices (`data/processed/merged/`)

Data harmonisation is executed by [`scripts/pillar-1-cohort-preprocessing/merge_datasets.py`](090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/merge_datasets.py):

```
Raw Downloads (cBioPortal / GDC)
              │
              ▼
   clean_data.py (Standardisation, log2-transformation, ID prefixes, MAF parsing)
              │
              ▼
   data/processed/<cohort>/ (clin_cleaned.csv, expr_cleaned.csv, mutations_cleaned.csv)
              │
              ▼
   merge_datasets.py (Gene intersection: 20,918 common genes; Cohort-wise Z-score)
              │
      ┌───────┴────────────────────────┐
      ▼                                ▼
merged/full/ (N=859)          merged/immunotherapy/ (N=347)
- expr_merged.csv             - expr_merged.csv
- clin_merged.csv             - clin_merged.csv
- merged_genomic.csv          - merged_genomic.csv
```

### 4.1 Gene Intersection & Z-Score Normalisation
1. **Gene Filtering**: The script extracts the intersection of all available Hugo Gene Symbols across active cohorts, yielding a high-confidence common panel of **20,918 genes**.
2. **Batch Normalisation**: To account for technical differences across sequencing platforms without losing biological variance, expression values for each gene are **Z-score standardised within each cohort independently**:
   $$Z_{i,j,c} = \frac{x_{i,j,c} - \mu_{j,c}}{\sigma_{j,c}}$$
   where $x_{i,j,c}$ is the $\log_2$-transformed expression of gene $j$ in sample $i$ within cohort $c$.

### 4.2 Genomic Feature Integration (`merged_genomic.csv`)
Maintains patient-level driver gene mutations and tumour characteristics:
* `mut_BRAF_V600`: Binary indicator for canonical Class 1 monomeric hotspots (V600E/K).
* `mut_BRAF_nonV600`: Binary indicator for Class 2/3 dimeric `BRAF` variants.
* `mut_NRAS`: Somatic hotspot mutations in `NRAS` (Q61, G12, G13).
* `mut_NF1`: Loss-of-function somatic alterations in neurofibromin 1 (`NF1`).
* `TMB_NONSYNONYMOUS`: Total non-synonymous mutation count per megabase.

---

## 5. Prospective Expansion Candidates (Under Investigation)

> [!INFO]
> Candidate cohorts identified for potential integration into the multi-omics evaluation pool, pending data availability and annotation quality audits.

### 5.1 Campbell / MORRISON-1 Dataset
* **Cohort Identity**: Pre-treatment and on-treatment transcriptomic and whole-exome sequencing from metastatic melanoma immunotherapy trials (Morrison et al. / Campbell et al.).
* **Investigation Objectives**:
  * Verify availability of raw FASTQ or normalised $\log_2(\text{TPM}+1)$ count matrices in public repositories (dbGaP / GEO / SRA).
  * Confirm RECIST v1.1 clinical response labels (CR, PR, SD, PD) and survival tracking.
  * Assess gene intersection with the current 20,918-gene harmonised panel.
  * Evaluate potential as an additional out-of-cohort benchmark in the LOCO pipeline.

### 5.2 GEM / Spanish Melanoma Group (*Grupo Español Multidisciplinar de Melanoma*)
* **Cohort Identity**: Large-scale prospective registry and genomic repository of advanced melanoma patients treated across Spanish academic centres with targeted therapy (`BRAF`/MEK inhibitors) and checkpoint blockade.
* **Investigation Objectives**:
  * Determine data governance and controlled-access requirements via the European Genome-phenome Archive (EGA).
  * Examine clinical annotation completeness (prior lines of therapy, brain metastasis status, LDH levels).
  * Evaluate sample size feasibility for independent sub-phenotype validation within the Q1.1 two-stage GMM framework.

---

## 6. Canonical Immune Biomarker Signatures Computed Across Cohorts

All cohorts are evaluated across the 6 canonical immune signature modalities defined in `.agents/AGENTS.md` (computed via [`src/signatures.py`](090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/signatures.py)):

1. **`IFN_gamma`** (Interferon-gamma Signature, 6 genes): `IFNG`, `STAT1`, `IDO1`, `CXCL9`, `CXCL10`, `HLA-DRA`. Reflects active Th1-mediated cytotoxicity and cellular immune recruitment.
2. **`TIS`** (Tumour Inflammation Score, 18 genes): Nanostring-derived clinical signature measuring suppressed adaptive immunity, antigen presenting machinery, and chemokine release.
3. **`CYT`** (Cytolytic Activity Score, 2 genes): Geometric mean of perforin (`PRF1`) and granzyme A (`GZMA`), directly quantifying CD8+ T-cell and NK-cell lytic competence.
4. **`CD8_Tcell`** (CD8+ T-cell Abundance proxy): Normalised expression of `CD8A` and `CD8B`.
5. **`IMPRES`** (Immune Predictive Score, 15 pairwise relations): Logical checkpoint gene pair expression ratios (`CD274`, `PDCD1`, `CTLA4`, etc.) trained specifically on spontaneous neuroblastoma and melanoma regressions.
6. **`PD_L1`** (Checkpoint Ligand Expression): Direct log2 expression of `CD274`.
