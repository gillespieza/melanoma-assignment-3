---
title: "Cohort Specification: Hugo 2016 (mel_iatlas_hugo_ucla_2016)"
aliases:
  - GSE78220
  - Hugo 2016
  - Hugo Cell 2016
  - mel_iatlas_hugo_ucla_2016
tags:
  - clinical-trials
  - dataset
  - hugo-2016
  - iatlas
  - immunotherapy
  - melanoma
created: 2026-09-15 12:38
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-09-15 12:39
---

# Dataset Overview

`mel_iatlas_hugo_ucla_2016` is the iAtlas-harmonised version of the metastatic melanoma cohort described by Hugo et al. (2016). The landmark study investigated genomic and transcriptomic characteristics associated with response and innate resistance to anti-PD-1 therapy in patients with metastatic melanoma.

> [!abstract] Primary Reference  
> Hugo W, Zaretsky JM, Sun L, et al. (2016). _Genomic and Transcriptomic Features of Response to Anti-PD-1 Therapy in Metastatic Melanoma_. Cell, 165(1):35–44. [DOI: 10.1016/j.cell.2016.02.065](https://doi.org/10.1016/j.cell.2016.02.065)

> [!note] cBioPortal Study Page  
> [https://www.cbioportal.org/study/summary?id=mel_iatlas_hugo_ucla_2016](https://www.cbioportal.org/study/summary?id=mel_iatlas_hugo_ucla_2016)

> [!info] GEO Accession  
> [GSE78220](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE78220) (Pre-treatment and on-treatment transcriptomic profiling of melanoma anti-PD-1 biopsies)

The dataset serves as one of the four core development and cross-validation cohorts in this project for predicting response to anti-PD-1 therapy using a curated panel of multimodal biological and genomic features.

# Clinical Context

Patients presented with advanced/metastatic melanoma and received anti-PD-1 checkpoint blockade:
- **Pembrolizumab**
- **Nivolumab**

The clinical setting directly addresses the primary focus of this project: baseline prediction of objective response to **anti-PD-1 therapy**.

Response evaluation was based on immune-related RECIST (irRECIST). Responding tumours included complete response (CR) and partial response (PR), while progressive disease (PD) tumours were classified as non-responders. Stable disease (SD) and mixed responses were evaluated per study criteria.

# Molecular Data

The Hugo 2016 study is of foundational importance because it contains matched **whole-exome sequencing (WES)** and **RNA sequencing (RNA-seq)** within the same cohort.

## Transcriptomic Data (RNA-seq)

Pre-treatment RNA-seq profiles ($\log_2(\text{TPM} + 1)$) derived from `GSE78220` enable the calculation of the project's canonical immune and microenvironmental signatures:
1. `IFN_gamma` (Interferon-gamma signalling score; 6 genes)
2. `TIS` (Tumour Inflammation Score; 18 genes)
3. `CYT` (Cytolytic Activity Score; `PRF1`, `GZMA`)
4. `CD8_Tcell` (CD8+ T-cell infiltration proxy; `CD8A`, `CD8B`)
5. `IMPRES` (Immune Predictive Score; 15 checkpoint pair relations)
6. `PD_L1` (Direct checkpoint ligand expression proxy; `CD274`)

The original study also discovered the **IPRES (innate anti-PD-1 resistance)** signature, characterised by transcriptional upregulation of mesenchymal transition, cell adhesion, extracellular matrix remodelling, angiogenesis, and wound healing. IPRES demonstrates that biological microenvironmental features provide critical orthogonal signals beyond mutation burden alone.

## Somatic Genomic Data (WES)

Whole-exome sequencing supports patient-level genomic profiling:
- Non-synonymous Tumour Mutational Burden (`TMB_NONSYNONYMOUS`)
- Driver mutation hotspots in `BRAF` (Class 1 `mut_BRAF_V600` vs Class 2/3 `mut_BRAF_nonV600`)
- Oncogenic mutations in `NRAS` (`mut_NRAS`)
- Loss-of-function somatic alterations in `NF1` (`mut_NF1`)
- Total non-synonymous mutation counts

The original study reported a median of 489 non-synonymous somatic mutations across the 38 WES tumours (range: 73 to 3,985). Importantly, while higher mutation load correlated with prolonged overall survival, TMB alone did not reliably discriminate responders from non-responders, strongly motivating the project's multimodal feature integration strategy.

# Cohort Subsetting & Attrition

The sample numbers across published modalities must be carefully distinguished:

| Modality / Subset | Sample Count ($N$) | Description |
|:---|:---:|:---|
| Total WES specimens | 38 | 34 pre-treatment + 4 early on-treatment specimens |
| Total RNA-seq patients | 28 | Matched patients with successful transcriptomic libraries |
| Pre-treatment RNA-seq | 27 | Baseline tumour biopsies eligible for response prediction |
| Responders (CR/PR) | 14 | Evaluated objective responders in pre-treatment cohort (53.8%) |
| Non-Responders (SD/PD) | 12 | Evaluated objective non-responders in pre-treatment cohort (46.2%) |
| Excluded from Modelling | 1 | Pre-treatment sample lacking evaluable binary RECIST endpoint |

> [!tip] Cohort Attrition Rule  
> Any model requiring RNA-derived expression features cannot utilize all 38 WES specimens. The final modelling cohort ($N = 26$) is strictly defined by the intersection of baseline pre-treatment timing, valid anti-PD-1 regimen, RNA-seq availability, and binary RECIST response. Attrition is tracked programmatically in `data/processed/hugo_2016/attrition.csv`.

# Role in This Project

`mel_iatlas_hugo_ucla_2016` is a **primary development and cross-validation cohort**, rather than a hold-out test set.

It represents one of the four core anti-PD-1 trial cohorts evaluated in this project:
1. **Hugo 2016** ($N = 26$ evaluable)
2. **Liu 2019** ($N = 121$ evaluable)
3. **Riaz 2017** ($N = 68$ evaluable baseline)
4. **Gide 2019** ($N = 91$ evaluable)

The project employs Leave-One-Cohort-Out (LOCO) cross-validation to assess whether the 12-feature biological panel generalises across trial cohorts, testing true inter-study transportability.

# Non-Independence: `mel_ucla_2016` vs `mel_iatlas_hugo_ucla_2016`

> [!caution] Cohort Non-Independence Warning  
> `mel_ucla_2016` and `mel_iatlas_hugo_ucla_2016` represent the **same underlying patient population** (the UCLA Hugo et al. trial), with `mel_ucla_2016` containing genomic/WES calls and `mel_iatlas_hugo_ucla_2016` containing the harmonised transcriptomic RNA-seq data.  
>
> They must **never be treated as two independent cohorts** in cross-validation or statistical pooling. Genomic information must be joined at the patient/sample level (`hugo_*` identifier mapping) rather than appended as independent patient instances.

# Modelling Considerations

- **Pre-treatment Samples Only**: Filter strictly to pre-treatment biopsies; exclude on-treatment or post-progression samples.
- **Anti-PD-1 Homogeneity**: Restrict analysis to patients treated with anti-PD-1 monotherapy (pembrolizumab or nivolumab).
- **Standardised Response Definition**: Harmonise irRECIST response to binary classification (CR/PR = 1, SD/PD = 0) matching the other three core cohorts.
- **Patient Identifier Alignment**: Map sample identifiers across WES and RNA-seq without double-counting patients.

# Key Limitations

> [!warning] Sample Size Vulnerability  
> With only $N = 26$ evaluable pre-treatment patients with matched RNA-seq (14 responders, 12 non-responders), the Hugo cohort is small. This confers heightened vulnerability to:  
> - Sensitivity to individual high-leverage patient profiles  
> - Overfitting in high-dimensional feature spaces  
> - Wide variance in performance estimates  
> 
> For these reasons, Hugo 2016 is evaluated strictly within a **pooled multimodal pipeline and Leave-One-Cohort-Out (LOCO) validation framework**, rather than as an isolated single-cohort training model.

# Summary
> [!summary] Dataset Summary Card  
> - **Study**: Hugo et al. (2016)  
> - **Disease**: Metastatic cutaneous melanoma  
> - **Regimen**: Anti-PD-1 checkpoint blockade (pembrolizumab / nivolumab)  
> - **Primary Modalities**: RNA-seq (`GSE78220`) and matched WES  
> - **cBioPortal Study ID**: `mel_iatlas_hugo_ucla_2016`  
> - **WES Population**: 38 specimens (34 pre-treatment, 4 on-treatment)  
> - **Pre-treatment RNA-seq Cohort**: 27 samples  
> - **Model-Eligible Cohort Size**: $N = 26$ (14 responders [53.8%], 12 non-responders [46.2%])  
> - **Pipeline Role**: Core training & LOCO cross-validation cohort  
