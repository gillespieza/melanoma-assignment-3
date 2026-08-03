# Project Architecture Map

> **Purpose**: Living reference document for agent orientation. Read this FIRST before
> exploring the codebase. Eliminates redundant file-discovery across conversations.
>
> **Last updated**: 2026-08-02 (Consolidated duplicate `q1-response-predictor/src/` utilities and `styles.py` to re-export directly from root `src/` single source of truth via explicit file-path loading; refactored `generate_loco_heatmap.py` to eliminate code smells with sub-30 line functions, explicit type annotations, module constants, and `TeeStream` logging to `logs/generate_loco_heatmap.log`).

## Repository Overview

**Domain**: Melanoma immunotherapy – predicting anti-PD-1/CTLA-4 response, drug sensitivity, tumour dynamics, and patient stratification across multi-cohort clinical trial data (Liu 2019, Hugo 2016, Riaz 2017, TCGA-SKCM).

**Structure**: 5 research questions (Q1–Q5), each in its own subproject directory, plus shared infrastructure in the root `src/` and `data/` directories. Q5 is the master synthesis engine that integrates Q1–Q4 outputs.

```
melanoma-assignment-3/
├── .agents/AGENTS.md       <- Style rules, palettes, code conventions
├── PROJECT_MAP.md          <- THIS FILE – architecture reference
├── src/                    <- Shared source (styles, utils, biology constants)
├── data/                   <- All raw + processed data
├── docs/                   <- Assignment brief, data dictionary, references
├── plots/                  <- Top-level cross-cohort visualisations
├── logs/                   <- Top-level pipeline logs
├── q1-response-predictor/  <- Q1: Immunotherapy response prediction
├── q2-viability-predictor/ <- Q2: Cell-line drug sensitivity modelling
├── q3-ode-model/           <- Q3: ODE tumour-immune dynamics
├── Q4_dep_map/             <- Q4: DepMap CRISPR + LINCS L1000 target discovery
├── q5-patient-stratification/ <- Q5: Patient clustering, clinical utility, treatability
└── dashboard/                 <- React clinical decision-support dashboard (OncoTwin)
```

## Shared Infrastructure (`src/`)

### Core Modules

| Module | Purpose |
|--------|---------|
| `src/styles.py` | All Okabe-Ito colour palettes (`COHORT_PALETTE`, `RESPONSE_PALETTE`, `PHENOTYPE_PALETTE`, `MUTATION_PALETTE`, etc.), `set_presentation_style()`, `get_cohort_color()`, `get_phenotype_color()`. Single source of truth for visual identity. |
| `src/biology_constants.py` | Centralised biological domain constants: `NON_SILENT` variant classes, gene panels (IFN-γ, antigen presentation, checkpoint, `M1_MACROPHAGE_GENES`, `M2_MACROPHAGE_GENES`), `COHORT_DIRS` mappings. |

### Utilities (`src/utils/`)

| Module | Key Exports |
|--------|------------|
| `src/utils/paths.py` | `PROJECT_ROOT`, `CONFIG_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `PLOTS_DIR`, `REPORTS_DIR`, `DATA_DIR`, `rel_path()` |
| `src/utils/formatting.py` | `generate_obsidian_frontmatter()` (supports `extra_css_classes` param), `format_count_percentage()`, `format_median()`, `format_median_iqr()` |
| `src/utils/plotting.py` | `save_fig()`, `resolve_colors()` |
| `src/utils/logging.py` | `TeeStream` (dual stdout/log-file stream) |
| `src/utils/io.py` | `safe_save_csv()` – Windows/Dropbox-safe atomic CSV write with `.tmp.csv` fallback |

*Note*: Dataset metadata is stored in `data/config/datasets.yaml`. Q1-specific dataset configuration and feature definitions are located in `q1-response-predictor/src/config/` (`datasets.py`, `constants.py`).

## Data Directory (`data/`)

### Raw Data (`data/raw/`)

| Subdirectory | Contents |
|-------------|----------|
| `iatlas/` | iAtlas harmonised RNA-seq and clinical data (Liu, Hugo, Riaz) |
| `tcga/` | TCGA-SKCM expression and clinical |
| `gdsc/` | Genomics of Drug Sensitivity in Cancer (cell-line viability) |
| `ccle/` | Cancer Cell Line Encyclopedia expression |

### Configuration (`data/config/`)

| File | Purpose |
|------|---------|
| `datasets.yaml` | Master dataset metadata (cohort names, paths, sizes) |
| `m1_m2_stv.csv` | 14,837-gene M1/M2 macrophage Signature Transcript Vector matrix |
| `q2_drug_list.csv` | Curated drug list for Q2 modelling |
| `q4_depmap_targets.csv` | DepMap CRISPR essentiality targets |
| `q4_lincs_perturbagens.csv` | LINCS L1000 perturbagen recommendations |

### Processed Data (`data/processed/`)

| Subdirectory | Contents |
|-------------|----------|
| `liu_2019/`, `hugo_2016/`, `riaz_2017/` | Per-cohort cleaned CSVs (`clin_cleaned.csv`, `expr_cleaned.csv`, `mutations_cleaned.csv`) |
| `tcga_skcm/` | TCGA-SKCM processed data |
| `merged/immunotherapy/` | **Key merged files**: `clin_merged.csv`, `expr_merged.csv`, `merged_genomic.csv` |
| `merged/full/` | All cohorts merged (including TCGA-SKCM reference) |
| `q1/` | Q1 feature matrices, model predictions (`pResponse`, `pResponsePct`) |
| `q2/` | Q2 cleaned viability matrices, patient drug-sensitivity predictions |
| `q5/` | Q5 outputs: `feature_matrix.csv` (ICI-only, N≈326), `feature_matrix_full.csv` (all cohorts, N≈699), `patient_clusters.csv`, `kmeans_model.pkl`, `clustering_feature_cols.json` |

## Q1: Response Predictor (`q1-response-predictor/`)

**Question**: Can we predict immunotherapy response from baseline tumour profiles?

### Script Structure (`q1-response-predictor/scripts/`)

| Directory / Script | Purpose |
|-------------------|---------|
| `clean_data.py` | Clean raw cohort clinical, expression, and mutation data |
| `download_data.py` | Retrieve and structure raw cohort files |
| `merge_datasets.py` | Merge processed cohort matrices into harmonised immunotherapy datasets |
| `biomarkers/run_extended_biomarkers.py` | Univariate biomarker association testing (Mann-Whitney U, ROC AUC) across signatures & genes |
| `biomarkers/run_genomic_characterisation.py` | TMB calculation, driver mutation prevalence (`BRAF`, `NRAS`, `NF1`), Fisher's exact co-occurrence |
| `biomarkers/run_merged_comut_plot.py` | Generates co-mutation oncoprint visualisations |
| `biomarkers/generate_threshold_plot.py` | Youden's J biomarker threshold optimisation plot |
| `biomarkers/generate_loco_heatmap.py` | Leave-One-Cohort-Out (LOCO) performance heatmap generator |
| `feature_selection/run_transcriptomic_feature_selection.py` | 12-feature multimodal Random Forest classifier (8 signature modalities + driver mutation flags) & signature vs raw gene benchmarks |
| `reports/run_executive_summary.py` | Executive summary report generator |
| `run_pipeline.py` | Master Q1 pipeline orchestrator |

### Source Modules (`q1-response-predictor/src/`)

| Module | Purpose |
|--------|---------|
| `signatures.py` | Signature computation (TIS, CYT, IFN-γ, Macrophage STV, M1/M2 Ratio, IMPRES) |
| `models.py` | Random Forest & Logistic Regression modeling pipelines |
| `evaluation.py` | ROC AUC, classification metrics, LOCO CV evaluation |
| `data_loaders.py` | Merged dataset loading helpers |
| `styles.py` | Local style re-exports and helper bindings |
| `config/constants.py` | Feature definitions, signature gene lists, and threshold grids |
| `config/datasets.py` | Dataset metadata and loading utilities |

### Key Outputs

- `models/final_rf_model.pkl` (trained 12-feature Random Forest model)
- `plots/` subdirs: `biomarkers/`, `feature_selection/`, `models/`
- `reports/` subdirs: `pillar-1-cohorts-and-preprocessing/`, `pillar-2-genomic-landscape/`, `pillar-3-transcriptomic-signatures/`, `pillar-4-out-of-cohort-benchmarks/`, `executive_summary.md`

## Q2: Viability Predictor (`q2-viability-predictor/`)

**Question**: Can we predict cell-line drug sensitivity and translate to patient tumours?

> **Note**: Q2 uses an **R-based pipeline** (`Question2.R`), unlike Q1/Q3/Q5 which use Python scripts.

### Script

| File | Purpose |
|------|---------|
| `scripts/Question2.R` | Complete R pipeline: trains LASSO regression viability models across 5 drugs (Dabrafenib, PLX-4720, Trametinib, Temozolomide, Dacarbazine), conducts stability checks, scores TCGA-SKCM patients, and correlates predicted viability with Q1 immunotherapy response |

### Key Input Data

- `Model.csv` (DepMap) – cell line ID → Oncotree lineage mapping
- `GDSC2_fitted_dose_response.csv` – cell-line drug AUC response data
- `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` (DepMap) – Log2(TPM+1) expression
- `data_mrna_seq_v2_rsem.txt` (TCGA-SKCM) – patient RSEM expression
- `clin_merged.csv` – merged clinical metadata
- `patient_predicted_response_scores.csv` (from Q1) – predicted IO response probabilities

### Key Outputs (`q2-viability-predictor/outputs/`)

- `important_genes_{Drug}.csv` – LASSO gene weights per drug
- `q2_model_coefficients.csv` – master non-zero LASSO weights across all 5 drugs
- `q2_patient_drug_predictions.csv` – predicted viability AUC for TCGA-SKCM patients
- `q2_correlation_results.csv` – merged drug viability + Q1 response scores

## Q3: ODE Model (`q3-ode-model/`)

**Question**: Can we simulate tumour-immune dynamics under therapy (targeted and immunotherapy)?

### Phase → Script Mapping

| Phase | Script | Purpose |
|-------|--------|---------|
| 1 | `phase1_check_environment.py` | Verifies required packages, TCGA data files, CPU cores |
| 2 | `phase2_preprocess_data.py` | Preprocesses TCGA clinical/expression/mutation data → `melanoma_params_full.csv` |
| 3 | `phase3_ode_simulation.py` | Solves 4-module ODE per patient across BRAFi dose sweep AND anti-PD-1 dose sweep |
| 4 | `phase4_survival_analysis.py` | KM + Cox PH survival stratification across ODE readouts |
| 5 | `phase5_rppa_validation.py` | Validates ODE pERK vs TCGA RPPA phospho-protein data |
| 6 | `phase6_ml_vs_ode.py` | 5-fold CV AUC benchmark: ODE digital twin vs ML classifiers |
| – | `run_all.sh` | Shell orchestrator for phases 1–6 |

### 4-Module ODE Architecture (`phase3_ode_simulation.py`)

All four modules operate on **per-patient inputs only** – kinetic rate constants are universal (literature-sourced). Only protein expression levels, mutation status, infiltration scores, and drug doses vary per patient.

| Module | Timescale | What It Models | Per-Patient Inputs |
|--------|-----------|----------------|--------------------|
| **A** – RAF dimerisation + BRAFi binding | Fast (equilibrium) | Vemurafenib binding to RAF dimers/monomers; RAF-inhibitor paradox (paradoxical ERK activation in `NRAS`-mutant/WT tumours emerges from binding equilibria, not hard-coded); `BRAF V600E` monomer directly inhibited (monotonic suppression) | `BRAF` expression, `BRAF_MUT`, `NRAS_MUT` → RAS-GTP level |
| **B** – MAPK cascade with ERK feedback | Fast (minutes) | 8-state Raf/MEK/ERK phosphorylation cascade; di-phospho-ERK feeds back to inhibit cascade top (Ki = 9 nM, universal). Readout: steady-state pERK | `BRAF`, `MAP2K1`, `MAP2K2`, `MAPK1`, `MAPK3` expression (scales protein pool totals) |
| **C** – Melanoma tumour / immune dynamics | Slow (days) | Single ODE: logistic tumour growth scaled by pERK from Module B → targeted drug effect propagates here; minus CD8+ T-cell killing gated by checkpoint state (Module D) and capped by patient CYT; minus natural death. Published Lai et al. 2017 rate constants | `CD8A`, `PRF1`, `GZMA` (effector density); `CYT` (killing ceiling, Rooney et al. 2015); pERK from Module B; f_kill from Module D |
| **D** – PD-1/PD-L1 checkpoint + anti-PD-1 | Steady state | Minimal PD-1/PD-L1 binding equilibrium; anti-PD-1 competitively occupies PD-1 before it can form the inhibitory complex with PD-L1; f_kill = 1 − Q/P_tot gates Module C killing (0 = fully suppressed, 1 = fully unleashed). At zero drug, f_kill reflects patient's own baseline checkpoint burden | `PDCD1` (PD-1 pool), `CD274` (PD-L1 pool), anti-PD-1 dose |

### Simulation Grid & Outputs

`simulate_patient()` runs **two independent dose sweeps per patient**:

| Sweep | Drug | Dose Range | Output CSV |
|-------|------|------------|------------|
| BRAFi sweep | Vemurafenib | 0–1000 nM (10 dose steps) | `results/pERK_simulations.csv`, `results/tumour_burden_simulations.csv` |
| Anti-PD-1 sweep | Anti-PD-1 | 0–500 nM (10 dose steps) | `results/checkpoint_tumour_simulations.csv` |

### Source Modules (`q3-ode-model/src/`)

| Module | Purpose |
|--------|---------|
| `ode_models.py` | ODE system equations (`tumour_immune_ode`), `run_simulation()`, `DEFAULT_PARAMS` |

### Key Data Dependencies

- **Reads**: `q1-response-predictor/data/raw/skcm_tcga_pan_can_atlas_2018/` (clinical, expression, mutation, RPPA data), `data/processed/q5/patient_clusters.csv` (Q5 phenotype integration)
- **Outputs**: `q3-ode-model/data/melanoma_params_full.csv`, `outputs/results/pERK_simulations.csv`, `outputs/results/tumour_burden_simulations.csv`, `outputs/results/checkpoint_tumour_simulations.csv`, `outputs/plots/` (KM, Cox dose-scan, RPPA validation, ML comparison figures)

## Q4: DepMap & LINCS (`Q4_dep_map/`)

**Question**: Can we identify novel drug targets from functional genomics screens?

### Structure

| Path | Contents |
|------|----------|
| `scripts/melanoma_depmap_v2.py` | Main DepMap CRISPR essentiality analysis pipeline |
| `scripts/sox10_proxy_plot_clean.py` | SOX10 proxy target visualisation |
| `scripts/sox10_proxy_targets.py` | SOX10 proxy target identification |
| `figures/dependency_selectivity/` | Dependency selectivity visualisations |
| `figures/proxy/` | SOX10 proxy analysis figures |

## Q5: Patient Stratification (`q5-patient-stratification/`)

**Question**: Can we identify clinically distinct patient subgroups requiring different treatments?

**Role**: Master synthesis engine integrating Q1–Q4 outputs into a clinical decision framework.

### Phase → Script Mapping

| Phase | Script | Purpose |
|-------|--------|---------|
| 1 | `01_load_and_prepare.py` | Load merged data, compute signatures (TIS, CYT, IFN-γ, CD8_Tcell, IMPRES), M1/M2 STV deconvolution, cell-type estimates, and spatial proxy indicators across 33 multi-modal features (19 transcriptomic + 14 genomic) |
| 2 | `02_feature_analysis.py` | Mann-Whitney U, Fisher's exact, Youden cutoffs, interaction terms, feature credibility |
| 3 | `03_cluster_patients.py` | **Two-Stage GMM**: Stage 1 GMM (K=3, full covariance) on 6 continuous immune/stromal features (TIS, CYT, CD8_T_cells, M1/M2_Macrophages, CAFs); Stage 2 deterministic NF1 split produces 4 final phenotypes. Exports named probability columns (`P_Immune_Hot`, `P_Immune_Cold`, `P_Immunosuppressive_M2_High`, `P_Mutant_Driven`) mapped via runtime-safe label-to-index resolution. PCA/t-SNE/UMAP projections, TMB violin, spatial violins, model persistence. |
| 4 | `04_phenotype_characterisation.py` | Cluster profiling, phenotype labelling, KM survival. Dual-arm Q3-parameterised Kuznetsov 2-state ODE trajectories: Panel A = Immunotherapy (Anti-PD-1 monotherapy + M2 CAF-rescue combination), Panel B = Targeted Therapy (Vemurafenib BRAFi 500 nM). |
| 5 | `05_subgroup_models.py` | Soft-weighted GMM probability Random Forest models trained on 10 non-circular features, evaluated via LOCO CV vs Global Enriched Baseline |
| 6 | `06_clinical_utility.py` | Decision Curve Analysis, Net Benefit, NNT, PPV, clinical benchmarks |
| 7 | `07_treatability_scoring.py` | Treatability Index, Q2 drug integration, Q4 DepMap/LINCS target nominations |
| – | `08_compare_clustering_algorithms.py` | Algorithmic comparison: K-Means vs Ward vs GMM vs DBSCAN |
| – | `09_run_consensus_clustering.py` | 1,000-bootstrap Consensus Clustering ensemble across patients & features ($K \in [2, 8]$) |
| – | `generate_q5_report.py` | Comprehensive Q5 markdown report generator |
| – | `run_q5_pipeline.py` | Master pipeline orchestrator (runs Phases 1–7) |

### Source Modules (`q5-patient-stratification/src/`)

| Module | Purpose |
|--------|---------|
| `q5_constants.py` | Cell-type markers, immune signature genes, clustering features, `PHENOTYPE_PROB_COL` mapping, resistance pathways, treatability features, phenotype labels, Q3 ODE params, Q4 target nominations |
| `clustering.py` | `prepare_clustering_features()`, `run_gmm()` (Two-stage GMM, K=3 continuous features + Stage 2 NF1 split), `plot_2d_cluster_projection()` (PCA, t-SNE, UMAP). |
| `phenotyping.py` | `profile_clusters()`, `assign_phenotype_labels()`, `plot_baseline_signature_boxplots()` |
| `deconvolution.py` | Transcriptomic cell-type deconvolution from marker panels |
| `clinical_utility.py` | DCA net benefit and NNT calculators |
| `reporting.py` | Obsidian frontmatter (re-exported from `src.utils.formatting`), markdown table formatting |

### Four Discovered Phenotypes (Two-Stage Stratification, N=699)

| Phenotype | Cluster ID | N (%) | Key Signatures & Driver Composition | Primary Therapeutic Routing |
|-----------|------------|-------|------------------------------------|----------------------------|
| **Immunosuppressive M2-High** | Cluster 0 | 256 (36.6%) | High M2 macrophages, CAF stromal exclusion, low TIS; 44.1% `BRAF`+, 29.3% `NRAS`+, 0% `NF1` | Targeted `BRAF`/MEK inhibition (for `BRAF`+) or stromal remodeling |
| **Immune Cold** | Cluster 1 | 45 (6.4%) | Low TIS, low CYT, T-cell desert; 53.3% `BRAF`+, 17.8% `NRAS`+, 13.3% `NF1`+ | Dual M2-depleting agent + checkpoint combination |
| **Immune Hot** | Cluster 2 | 341 (48.8%) | High TIS, high CYT, inflamed microenvironment; 49.6% `BRAF`+, 25.2% `NRAS`+, 12.6% `NF1`+ | Primary immune checkpoint blockade (ICI monotherapy) |
| **Mutant-Driven** | Cluster 3 | 57 (8.2%) | 100% `NF1` loss-of-function, RAS hyperactivation, high TMB (Med = 41 mut/Mb) | Immune checkpoint blockade + MEK adjunct for RAS suppression |

## Dashboard Subproject (`dashboard/`)

**Technology**: React 18 + Vite + TypeScript + TailwindCSS. Static SPA – no server required.

**Purpose**: Clinical decision-support demonstrator. Loads `public/cohort.json` (built from Q3–Q5 outputs) and renders TCGA-SKCM patient digital twin, Q1 ML predictor, Q2 validation, Q3 ODE simulation, Q4 resistance, and integrated treatment recommendations.

### Key Files

| File / Directory | Purpose |
|------------------|---------|
| `scripts/build_cohort.mjs` | Ingests Q3 ODE CSVs + TCGA clinical + Q1 predictions + Q5 scores → `public/cohort.json`. |
| `src/data/cohort.ts` | TypeScript interfaces for the cohort JSON (`CohortPatient`, `Q5Phenotype`, `CohortMeta`). |
| `src/data/model.ts` | Clinical decision rules, drug target metadata, and model definitions. |
| `src/data/types.ts` | Core UI and patient data type definitions. |
| `src/lib/integrationEngine.ts` | Integration engine – derives treatment recommendations and reasoning chains. |
| `src/lib/scoring.ts` | Score calculation and risk stratification utilities. |
| `src/lib/forecast.ts` | ODE tumour trajectory forecasting helpers. |
| `src/lib/whatIf.ts` | Interactive counterfactual simulation logic. |
| `src/components/Q1Lane.tsx` | Q1 multimodal ML predictor panel & signature breakdown. |
| `src/components/Q1ValidationPanel.tsx` | Cross-cohort LOCO validation metrics & ROC AUC display. |
| `src/components/Q2Evidence.tsx` | Q2 cell-line drug sensitivity & LASSO viability evidence panel. |
| `src/components/Q4Resistance.tsx` | Q4 DepMap essentiality & LINCS perturbagen resistance panel. |
| `src/components/RecommendationPanel.tsx` | Integrated multi-modal treatment recommendation panel. |
| `src/components/PatientView.tsx` | Patient workbench with 5 tabs (`Q1 · ML predictor`, `Q2 · Validation`, `Q3 · Digital twin`, `Q4 · Resistance`, `Decision path`). |
| `src/components/CohortTable.tsx` | Searchable/filterable patient table with Q5 Phenotype, TI, and Confidence Band columns. |
| `src/components/FeaturedCards.tsx` | Landing page archetype selection cards. |
| `src/components/PatientPassport.tsx` | Patient header pills (`BRAF`, `PD-L1`, `TMB`, `Q5`). |
| `src/components/TumourForecastChart.tsx` | ODE tumour burden forecast chart. |
| `src/components/DoseResponseChart.tsx` | ODE drug dose response chart. |
| `src/components/DecisionTree.tsx` | Interactive clinical decision logic visualiser. |

### Build Pipeline

```
npm run build
  └── prebuild
        └── node scripts/build_cohort.mjs       → public/cohort.json
  └── tsc --noEmit && vite build
```

## Known Technical Debt & Recent Resolution History

| Area | Description | Status |
|------|-------|--------|
| `q1-response-predictor` pipeline & scripts | Refactored Q1 pipeline structure into modular directories (`scripts/biomarkers/`, `scripts/feature_selection/`, `scripts/reports/`), updated to 12-multimodal feature panel (8 signatures + driver mutation flags), added `pResponsePct` cohort-relative percentile, generated LOCO heatmaps & feature selection benchmarks. | **Resolved** (2026-08-02) |
| Dashboard terminology & validation sync | Updated confusion matrix labels to True positive / True negative, updated Q1 user guide to 12 multimodal features state, updated Q1 validation panel calibration descriptions, and synced hard-blocking contraindication messaging. | **Resolved** (2026-08-02) |
| `src/styles.py` L53 & L178 | Duplicate `get_phenotype_color()` definitions | **Resolved** (2026-08-01) |
| `q5/src/reporting.py` | Local `generate_obsidian_frontmatter()` duplicated `src/utils/formatting.py` | **Resolved** (2026-08-01) |
| `q5/src/phenotyping.py` | Uncalled exports (`plot_radar_chart()`, `plot_cluster_heatmap()`) | Unresolved |
| Subproject log routing (`q1`, `q3`, `q5`) | Standardised `LOG_DIR` and `LOG_PATH` across all subproject scripts (`q1-response-predictor`, `q3-ode-model`, `q5-patient-stratification`) to output logs to each subproject's dedicated `logs/` directory (`<subproject>/logs/`) instead of top-level `PROJECT_ROOT/logs/`. Fixed root bootstrap `BASE_DIR` resolution in Q1 biomarker/exploratory scripts and `src/utils/paths.py` `find_subproject_root()` recognition. | **Resolved** (2026-08-03) |

## Conventions Quick Reference

- **Pipeline pattern**: Each Q has `run_qN_pipeline.py` or `run_pipeline.py` → sequential scripts
- **Logging**: All scripts use `TeeStream` to dual-write stdout to `logs/` directory
- **Figures**: 300 DPI PNG via `save_fig()`, 16:9 aspect ratio preferred
- **Reports**: Obsidian markdown with YAML frontmatter, callout boxes, British English
- **Colours**: Okabe-Ito palette only, accessed via `src/styles.py` helpers
- **Paths**: Always use `src/utils/paths.py` constants, log relative paths via `rel_path()`
