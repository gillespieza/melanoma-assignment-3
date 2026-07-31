---
title: "Phase 1: Feature Engineering & Baseline Signature Distribution"
aliases:
  - Q5 Phase 1
tags:
  - melanoma
  - patient-stratification
  - phase-1
  - q5
created: 2026-08-01 00:13
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-01 00:13
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

### Biological Feature Engineering & Microenvironment Deconvolution
Rather than evaluating ~19,757 genes independently, Phase 1 projects patient expression profiles onto curated biological axes:
- **Core Immune Signatures**: Tumour Inflammation Signature (`TIS`), Cytolytic Index (`CYT`, mean of `PRF1` and `GZMA`), Interferon-gamma (`IFN_gamma`), and `CD274` (`PD-L1`) expression.
- **Macrophage STV (`M1_M2_Ratio`)**: Computed using a linear Signature Transcript Vector ($W_g$, 14,835 genes) to quantify the microenvironmental balance between pro-inflammatory M1 macrophages ($W_g > 0$) and pro-tumour M2 macrophages ($W_g < 0$).
- **Transcriptomic Cell Deconvolution**: Marker-based signature scores estimating the relative infiltration abundance of CD8+ T cells (`CD8_Tcell`), CD4+ T cells, NK cells, B cells, M1 Macrophages, M2 Macrophages, and Cancer-Associated Fibroblasts (`CAFs`).

### Baseline Biomarker Feature Distributions

![Baseline Biomarker Feature Distributions](q5-patient-stratification/plots/phenotypes/baseline_response_violins.png)

> [!IMPORTANT] Key Takeaways
> - **Dual-Matrix Dataflow**: Established a dual dataflow pipeline isolating response-labeled ICI trials ($N_{\text{ICI}} = 326$) for predictive modelling while embedding the full cohort ($N_{\text{Full}} = 699$) for unsupervised manifold learning.
> - **Dimensionality Reduction**: Compressed ~19,757 transcriptomic features into 23 engineered biological signatures (part of a 32-feature multi-modal panel, centered on 9 core baseline biomarkers).
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

> [!formula] Phase 1 Script Execution & Software Module Architecture
> - **Primary Pipeline Execution Scripts**:
>   - [`01_load_and_prepare.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/01_load_and_prepare.py): Loads preprocessed clinical, expression, and genomic data from `data/processed/merged/`, computes immune signatures (`TIS`, `CYT`, `IFN_gamma`, `CD8_Tcell`), calculates Macrophage STV (`M1_M2_Ratio`), runs transcriptomic cell deconvolution, and exports dual feature matrices (`feature_matrix.csv`, $N_{\text{ICI}} = 326$; `feature_matrix_full.csv`, $N_{\text{Full}} = 699$).
> - **Core Supporting Python Modules**:
>   - [`deconvolution.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/deconvolution.py): Implements Signature Transcript Vector (`compute_macrophage_stv`) and cell-type abundance estimation (`compute_cell_deconvolution`).
>   - [`phenotyping.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/phenotyping.py): Renders 2x3 facetted biomarker distribution violins (`plot_baseline_signature_boxplots`).
>   - [`q5_constants.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/q5_constants.py): Single source of truth for cell marker panels (`CELL_TYPE_MARKERS`), immune signature definitions (`IMMUNE_SIGNATURE_MARKERS`), and neutral STV fallback ratios.
> - **Shared Cross-Question & Pipeline Modules**:
>   - [`run_q5_pipeline.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/run_q5_pipeline.py): Master pipeline orchestrator executing `01_load_and_prepare.py` as Step 1.
>   - [`generate_q5_report.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/generate_q5_report.py): Compiles live statistical summaries and generates phase markdown reports.
