---
title: "Phase 1: Feature Engineering & Baseline Signature Distribution"
aliases:
  - Q5 Phase 1
tags:
  - melanoma
  - patient-stratification
  - phase-1
  - q5
created: 2026-07-30 18:40
cssclasses:
  - table-small
  - table-center
  - row-alt
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-31 11:11
---

## 1. Phase 1: Multi-Modal Feature Matrix & Microenvironment Deconvolution

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Loading preprocessed clinical, transcriptomic, and genomic data and constructing dual feature matrices: an ICI-treated cohort ($N_{\text{ICI}} = 326$) and a full melanoma cohort ($N_{\text{Full}} = 699$) with engineered immune signatures, Macrophage STV ratios, and cell deconvolution metrics.
> - **Why we are doing it**: Raw RNA-seq gene expression matrices containing ~19,757 genes suffer from severe dimensionality challenges. Transforming high-dimensional transcriptomics into validated signature scores and cell-type fractions provides interpretable, non-redundant biological features for both response prediction and unsupervised stratification.
> - **What question it answers**: How are raw multi-modal datasets harmonised and structured into dual feature matrices to power downstream response-supervised modelling and unsupervised patient stratification?

Phase 1 establishes the foundational dataflow architecture by integrating harmonised multi-modal data across 4 melanoma studies (*Liu 2019*, *Riaz 2017*, *Hugo 2016*, and *TCGA-SKCM*). To support distinct analytical requirements across downstream phases, Phase 1 outputs two standardised feature matrices:

### Dual Feature Matrix Dataflow Architecture
1. **ICI-Treated Feature Matrix (`feature_matrix.csv`, $N_{\text{ICI}} = 326$)**:
   - **Composition**: Comprises immunotherapy-treated patients across 4 studies (*Liu 2019*, *Riaz 2017*, *Hugo 2016*, and ICI-treated *TCGA-SKCM* subset) with complete RECIST clinical response labels (`RESPONSE_BINARY`).
   - **Downstream Routing**: Powers response-supervised analyses: **Phase 2** (Feature Analysis & Youden Cutoffs), **Phase 5** (Subgroup Predictive Modelling), and **Phase 6** (Clinical Utility & Decision Curve Analysis).
2. **Full Melanoma Feature Matrix (`feature_matrix_full.csv`, $N_{\text{Full}} = 699$)**:
   - **Composition**: Merges ICI trial cohorts with the complete reference cohort (*TCGA-SKCM*, $N = 443$), expanding the dataset to capture overall population-level biological heterogeneity.
   - **Downstream Routing**: Powers response-agnostic biological discovery and decision support: **Phase 3** (Unsupervised Patient Stratification & Manifold Projections), **Phase 4** (Phenotype Characterisation & Dynamic Trajectories), and **Phase 7** (3-Arm Decision Support & Treatability Index Scoring).

### Biological Feature Engineering & Microenvironment Deconvolution Determination

Rather than evaluating ~19,757 genes independently, Phase 1 projects patient expression profiles onto curated biological axes. Transcriptomic cell deconvolution is determined via two complementary quantitative methodologies in `deconvolution.py`:

#### 1. Marker-Based Relative Cell Abundance Averaging
For 7 distinct immune and stromal cell types, relative infiltration abundance is calculated as the sample-wise mean $\log_2$-transformed expression across curated marker gene panels:

$$\text{Deconvolution Score}_{\text{CellType}, i} = \frac{1}{|M|} \sum_{g \in M} E_{i, g}$$

Where $M$ represents the set of verified marker genes present in the normalized expression matrix for sample $i$:

| Cell-Type Feature | Marker Genes ($M$) | Primary Biological Function |
| :--- | :--- | :--- |
| **`CD8_T_cells`** | `CD8A`, `CD8B`, `CD3D`, `CD3E` | Cytotoxic T-cell effector density |
| **`CD4_T_cells`** | `CD4`, `IL7R`, `FOXP3` | Helper and regulatory T-cell infiltrates |
| **`NK_cells`** | `NCAM1`, `KLRB1`, `NCR1` | Innate natural killer cell abundance |
| **`B_cells`** | `CD19`, `MS4A1`, `CD79A` | Humoral immune infiltrate density |
| **`M1_Macrophages`** | `TNF`, `IL12B`, `CXCL10`, `NOS2`, `IRF5` | Pro-inflammatory antitumour macrophages |
| **`M2_Macrophages`** | `CD163`, `MRC1`, `MSR1`, `TGFB1`, `ARG1` | Immunosuppressive pro-tumour macrophages |
| **`CAFs`** | `FAP`, `PDGFRB`, `COL1A1`, `ACTA2` | Cancer-Associated Fibroblast stromal walls |

#### 2. Macrophage Signature Transcript Vector (STV) Polarisation Scoring
To resolve the functional balance between M1 (pro-inflammatory) and M2 (immunosuppressive) macrophages, a continuous **Signature Transcript Vector (STV)** is computed using the 14,837-gene scorecard `m1_m2_stv.csv`:

- **M1 Score Calculation**: Weighted dot-product across positive weight genes ($W_{g, \text{M1}} > 0$):
  $$\text{M1\_score}_i = \sum_{g \in \text{M1}} E_{i,g} \cdot W_{g, \text{M1}}$$
- **M2 Score Calculation**: Weighted dot-product across absolute negative weight genes ($W_{g, \text{M2}} < 0$):
  $$\text{M2\_score}_i = \sum_{g \in \text{M2}} E_{i,g} \cdot |W_{g, \text{M2}}|$$
- **Normalized M1/M2 Ratio**:
  $$\text{M1\_M2\_Ratio}_i = \frac{\text{M1\_score}_i}{\text{M1\_score}_i + \text{M2\_score}_i}$$

### Baseline Biomarker Feature Distributions

![Baseline Biomarker Feature Distributions](q5-patient-stratification/plots/phenotypes/baseline_response_violins.png)

> [!IMPORTANT] Key Takeaways
> - **Dual-Matrix Dataflow**: Established a dual dataflow pipeline isolating response-labeled ICI trials ($N_{\text{ICI}} = 326$) for predictive modelling while embedding the full cohort ($N_{\text{Full}} = 699$) for unsupervised manifold learning.
> - **Dimensionality Reduction**: Compressed ~19,757 transcriptomic features into 17 engineered biological signatures (part of a 26-feature multi-modal panel, centered on 9 core baseline biomarkers).
> - **M1/M2 Polarisation**: The Macrophage STV score captures stromal microenvironmental suppression that operates independently of total T-cell density.

> [!INFO] Phase 1 Feature Matrix Architecture & Complete Feature Inventory
> - **Transcriptomic Features (17)**:
>   - **Core Immune Signatures (6)**: 
>      1. `TIS` (Tumour Inflammation Signature)
>      2. `CYT` (Cytolytic Index)
>      3. `IFN_gamma` (Interferon-gamma signalling)
>      4. `CD8_Tcell` ($CD8A/B$)
>      5. `IMPRES`
>      6. `PD_L1` (`CD274`).
>   - **Macrophage STV Metrics (4)**: 
>      1. `M1_score`
>      2. `M2_score`
>      3. `M1_M2_Ratio`
>      4. `Macrophage_STV_Score` (Signature Transcript Vector balance).
>   - **Transcriptomic Cell Deconvolution (7)**: 
>      1. `CD8_T_cells`
>      2. `CD4_T_cells`
>      3. `NK_cells`
>      4. `B_cells`
>      5. `M1_Macrophages`
>      6. `M2_Macrophages`
>      7. `CAFs` (Cancer-Associated Fibroblasts).
> - **9 Genomic, TMB & Neoantigen Features**:
>   - **Driver Mutations (3)**: 
>      1. `mut_BRAF`
>      2. `mut_NRAS`
>      3. `mut_NF1` (binary oncogenic driver status).
>   - **TMB & Neoantigen Burden (6)**: 
>      1. `TMB_NONSYNONYMOUS`
>      2. `SNV_NEOANTIGEN`
>      3. `INDEL_NEOANTIGEN`
>      4. `FUSION_NEOANTIGEN`
>      5. `SPLICE_NEOANTIGEN`
>      6. `CTA_SELF_NEOANTIGEN`.
> - **9 Core Baseline Biomarkers (Primary Subset)**: 
>      1. `TIS`
>      2. `CYT`
>      3. `PD_L1`
>      4. `Macrophage_STV_Score`
>      5. `CD8_T_cells`
>      6. `CD4_T_cells`
>      7. `NK_cells`
>      8. `B_cells`
>      9. `CAFs` (used for primary volcano, Youden ROC, and radar visualisations).
