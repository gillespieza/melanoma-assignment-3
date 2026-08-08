# Project Architecture Map

> **Purpose**: Living reference document for agent orientation. Read this FIRST before
> exploring the codebase. Eliminates redundant file-discovery across conversations.
>
> **Last updated**: 2026-08-08 (Routed `run_response_km_curves.py` log output to subproject directory `q1-response-predictor/logs/run_response_km_curves.log` per `.agents/AGENTS.md` Rule 10; verified clean execution with exit code 0).

## Repository Overview

**Domain**: Melanoma immunotherapy – predicting anti-PD-1/CTLA-4 response, targeted therapy drug sensitivity, tumour dynamics, and patient stratification across multi-cohort clinical trial data (Liu 2019, Hugo 2016, Riaz 2017, TCGA-SKCM).

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
├── q1.1-patient-stratification/ <- Q5: Patient clustering, clinical utility, treatability
├── q2-viability-predictor/ <- Q2: Cell-line drug sensitivity modelling
├── q3-ode-model/           <- Q3: ODE tumour-immune dynamics
├── Q4_dep_map/             <- Q4: DepMap CRISPR + LINCS L1000 target discovery
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
| `pillar-1-cohort-preprocessing/download_data.py` | Retrieve and structure raw cohort files |
| `pillar-1-cohort-preprocessing/clean_data.py` | Clean raw cohort clinical, expression, and mutation data |
| `pillar-1-cohort-preprocessing/merge_datasets.py` | Merge processed cohort matrices into harmonised immunotherapy datasets |
| `pillar-1-cohort-preprocessing/run_dimensionality_reduction.py` | PCA / UMAP projections, top 50 variable gene heatmaps, and batch correction evaluation |
| `pillar-1-cohort-preprocessing/run_genomic_characterisation.py` | TMB calculation, driver mutation prevalence (`BRAF`, `NRAS`, `NF1`), Fisher's exact co-occurrence |
| `pillar-1-cohort-preprocessing/run_clinical_analysis.py` | Clinical feature distributions, Kaplan-Meier OS curves, and univariate log-rank tests |
| `pillar-2-clinical-subtyping/run_clinical_clustering.py` | Exploratory clinical phenotyping, cluster-based patient stratification, and PCA / t-SNE / UMAP 2D projection comparisons |
| `pillar-2-clinical-subtyping/run_clinical_feature_selection.py` | Clinical feature selection and univariate association benchmarking |
| `pillar-2-clinical-subtyping/run_univariate_associations.py` | Statistical testing of baseline clinical covariates against ICI response |
| `pillar-2-clinical-subtyping/plot_cluster_profile_visualizations.py` | Radar and violin plot visualisations for clinical patient clusters |
| `pillar-3-transcriptomic-signatures/run_extended_biomarkers.py` | Fast exploratory biomarker analysis (TMB vs Neoantigen, Pathway Mutations, Aneuploidy/CNA, TCGA OS curves) |
| `pillar-3-transcriptomic-signatures/train_multimodal_predictor.py` | Multimodal ML model training, 5-fold CV hyperparameter search across 5 feature permutation tiers, Section 5 report update |
| `pillar-4-out-of-cohort-benchmarks/run_transcriptomic_feature_selection.py` | 12-feature multimodal Random Forest classifier (8 signature modalities + driver mutation flags) & signature vs raw gene benchmarks |
| `pillar-4-out-of-cohort-benchmarks/generate_5f_cv_comparison_heatmap.py` | 5-Fold Stratified CV benchmark heatmap: Curated Signatures vs SelectKBest |
| `exploratory_plots/run_merged_comut_plot.py` | Generates co-mutation oncoprint visualisations |
| `exploratory_plots/generate_threshold_plot.py` | Youden's J biomarker threshold optimisation plot |
| `exploratory_plots/generate_loco_heatmap.py` | Leave-One-Cohort-Out (LOCO) performance heatmap (Curated Signatures, per-cohort) |
| `exploratory_plots/generate_5f_cv_heatmap.py` | 5-Fold Stratified CV per-fold AUROC heatmap |
| `exploratory_plots/generate_combined_cv_loco_heatmap.py` | **Primary evaluation figure**: 1×2 panel — Left: 5-fold CV AUROC (Curated Signatures vs SelectKBest); Right: LOCO AUROC per held-out cohort. Output: `plots/models/cv_loco_1x2_heatmap.png` |
| `exploratory_plots/generate_loco_feature_comparison_heatmap.py` | **LOCO feature comparison figure**: 1×2 panel — Left: LOCO per-cohort (Curated Signatures); Right: LOCO by feature representation. Output: `plots/models/loco_dual_1x2_heatmap.png` |
| `exploratory_plots/run_comparison.py` | LOCO cross-validation benchmark: Curated Multimodal Features vs SelectKBest |
| `exploratory_plots/run_clustering.py` | Exploratory clustering of pooled cohort expression data |
| `exploratory_plots/run_expression_heatmap.py` | Signature expression heatmap across cohorts |
| `exploratory_plots/run_extra_plots.py` | Supplementary exploratory plots |
| `exploratory_plots/run_forest_plot.py` | Forest plot of univariate biomarker associations |
| `exploratory_plots/run_km_curves.py` | Kaplan-Meier survival curves |
| `exploratory_plots/run_response_distribution.py` | Response rate distribution across cohorts |
| `exploratory_plots/run_response_km_curves.py` | KM curves stratified by predicted response |
| `exploratory_plots/run_waffle_chart.py` | Cohort composition waffle chart |
| `models/predictors.py` | Classifier evaluation wrappers (LR, RF, XGB, SVM, ElasticNet) |
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
| SVM Tuning, 16:9 Heatmaps & Dashboard Sync | Enhanced `tune_svc` with `class_weight='balanced'`, expanded grid (`C=0.01-100`, `gamma=['scale', 'auto']`), and adaptive CV fold counts for small splits. Fixed 16:9 canvas rendering in LOCO and 5-fold CV heatmaps by replacing `tight_layout` with explicit `subplots_adjust` margin preservation. Re-trained pooled model pickles in `q1-response-predictor/models/`, updated `q1_predictions.csv`, and rebuilt `dashboard/public/cohort.json`. | **Resolved** (2026-08-03) |
| Q1 script organisation — plot scripts in `biomarkers/` | Plot-generating scripts were incorrectly placed in `scripts/biomarkers/`. Moved to: `scripts/exploratory_plots/` (`generate_loco_heatmap.py`, `generate_5f_cv_heatmap.py`, `generate_combined_cv_loco_heatmap.py`, `generate_loco_feature_comparison_heatmap.py`, `generate_threshold_plot.py`, `run_merged_comut_plot.py`) and `scripts/feature_selection/` (`generate_5f_cv_comparison_heatmap.py`). `biomarkers/` now contains analysis scripts only. | **Resolved** (2026-08-05) |
| Q1 log path routing | `generate_combined_cv_loco_heatmap.py` and `generate_loco_feature_comparison_heatmap.py` were writing logs to `PROJECT_ROOT/logs/` instead of `q1-response-predictor/logs/`. Fixed by replacing imported `LOG_DIR` with `get_subproject_log_dir(Path(__file__))` in both scripts. | **Resolved** (2026-08-05) |
| `q1-response-predictor/scripts/clean_data.py` code smell refactoring | Conducted 4-pass code smell remediation per `AGENTS.md` guidelines: 100% of 44 functions decomposed to $\le 30$ lines, extracted 10 domain constants (`_COL_VARIANT_CLASSIFICATION`, `_COL_TREATMENT_TYPE`, `_STRATEGY_IATLAS`, `_STRATEGY_TCGA`, etc.), introduced `CleanedDataBundle` parameter object to shrink function signatures, restored `_build_treatment_summary_features()`, added full type annotations & docstrings, eliminated long lines & long ternaries, and added traceback logging to broad exception handler. | **Resolved** (2026-08-06) |
| Data ingestion & pipeline scripts refactoring (`download_data.py`, `clean_data.py`, `merge_datasets.py`) | Audited and refactored all 3 pipeline data scripts: 100% of 91 functions decomposed to $\le 30$ lines (16 in `download_data.py`, 45 in `clean_data.py`, 30 in `merge_datasets.py`), eliminated cross-script DRY path ambiguities by deriving `CONFIG_PATH` via `SCRIPT_DIR.parent`, integrated project-root `src/` utilities (`paths.py`, `io.py`, `logging.py`) and biological constants (`src/biology_constants.py`), zero lines > 100 chars, 100% docstring & type hint coverage. Verified full sequential pipeline run (`download` → `clean` → `merge`) with 0 errors. | **Resolved** (2026-08-06) |
| `q1-response-predictor/scripts/biomarkers/run_extended_biomarkers.py` code smell & dead code cleanup | Audited and refactored `run_extended_biomarkers.py` per `AGENTS.md` guidelines & user instructions: 100% of 26 active functions decomposed to $\le 30$ lines, zero lines $>100$ chars, extracted private `_COL_*` and `_CURATED_IMMUNE_SIGNATURES` constants, removed orphaned report generator stubs (`_generate_aneuploidy_tmb_report_lines`, `_build_spearman_table_rows`, `_get_aneuploidy_tmb_headers`) and unused model helpers (`TunedCalibratedModel`, `evaluate_auc_cv`), removed unused imports, added 100% type hint & docstring coverage, and verified execution with 0 errors. | **Resolved** (2026-08-06) |
| Q1 script folder organisation — Pillar alignment | Restructured `q1-response-predictor/scripts/` to mirror `reports/` pillar structure: moved `run_genomic_characterisation.py` → `pillar-1-cohort-preprocessing/`, `clinical_analysis/` → `pillar-2-clinical-subtyping/`, `run_extended_biomarkers.py` & `train_multimodal_predictor.py` → `pillar-3-transcriptomic-signatures/`, `feature_selection/` → `pillar-4-out-of-cohort-benchmarks/`. Added `__init__.py` to `pillar-3-transcriptomic-signatures/` package, updated cross-script imports (`scripts.pillar_3_transcriptomic_signatures`), updated report callout deep links in `curated_signatures_report.md`, and verified full script execution with exit code 0. | **Resolved** (2026-08-06) |
| Relocating batch correction & data scripts to Pillar 1 | Moved `download_data.py`, `clean_data.py`, `merge_datasets.py`, and `run_dimensionality_reduction.py` to `q1-response-predictor/scripts/pillar-1-cohort-preprocessing/`. Updated `SUBPROJECT_ROOT` and `CONFIG_PATH` resolution, updated file links in `batch_correction_report.md`, `Pipeline.md`, `README.md`, `data/README.md`, and verified execution of all relocated scripts with exit code 0. | **Resolved** (2026-08-06) |
| `run_genomic_characterisation.py` refactoring & report enhancements | Refactored `q1-response-predictor/scripts/pillar-1-cohort-preprocessing/run_genomic_characterisation.py` per `AGENTS.md` guidelines: 100% of functions decomposed to $\le 30$ lines, zero lines $>100$ chars, removed unused/invalid `CONFIG_DIR` and data loader imports, added KM confidence shading (`ci_show=True`, `ci_alpha=0.15`) for `tcga_survival_by_mutation.png`, removed Figure 7 from report generator, appended script reference callout box to `cohort_characteristics_genomic.md`, added private `_COL_*` constants, type hints, docstrings, and verified 100% clean pipeline execution. | **Resolved** (2026-08-06) |
| `run_dimensionality_reduction.py` code smell & report callout refactoring | Conducted multi-pass audit and refactoring of `run_dimensionality_reduction.py` per `AGENTS.md` guidelines: 100% of 22 active functions decomposed to $\le 30$ lines (including `main()` decomposed into `_run_pca_batch_projections` and `_run_trial_reduction_grids`), zero lines $> 100$ characters, extracted `N_TOP_HEATMAP_GENES` constant, extracted `_assign_pca_coords`, `_assign_umap_coords`, and `_grid_col_names` helpers, removed dead `PLOT_DIR` constant, added `>\n` lines after callout headers to fix Obsidian callout container rendering in `batch_correction_report.md`, added 100% type hint & docstring coverage, and verified clean execution with exit code 0. | **Resolved** (2026-08-06) |
| `plot_cluster_profile_visualizations.py` code smell & DRY refactoring | Remediation of `plot_cluster_profile_visualizations.py` per `AGENTS.md` guidelines: eliminated hardcoded magic radar values & N-counts by exporting `data/processed/merged/clinical_clusters.csv` from `run_clinical_clustering.py` and calculating values dynamically from live DataFrame; extracted reusable `build_radar_angles()` to `src/utils/plotting.py`; aligned cluster palette to `PHENOTYPE_PALETTE`; corrected log routing to `get_subproject_log_dir`; decomposed functions to $\le 30$ lines; cleaned unused imports; fixed pre-existing `CONFIG_DIR` path error in `run_clinical_clustering.py`; verified 100% clean execution with exit code 0. | **Resolved** (2026-08-07) |
| `run_clinical_feature_selection.py` two-tiered redesign & code smell remediation | Redesigned feature selection to evaluate anti-PD-1 binary response exclusively on ICI cohorts ($N=256$, excluding TCGA-SKCM); cleaned Tier 2 features to baseline pre-treatment clinical covariates; removed non-baseline variables (`CLINICAL_BENEFIT`, `PROGRESSION`, `RACE`, etc.); integrated `get_model("rf", X, y)` factory; added standard error bounds ($\text{SE} > 10.0$) and rank deficiency safeguards (`np.linalg.matrix_rank`) for logistic regression; decomposed 100% of 49 functions to $\le 30$ lines in AST; zero lines $>100$ characters; 100% type hint and docstring coverage; verified clean execution with exit code 0. | **Resolved** (2026-08-07) |
| `run_univariate_associations.py` code smell remediation & exact CI calculation | Refactored `q1-response-predictor/scripts/pillar-2-clinical-subtyping/run_univariate_associations.py` per `AGENTS.md` guidelines: fixed subproject log routing (`get_subproject_log_dir`), removed 5 unused imports, extracted domain constants, decomposed 100% of functions to $\le 30$ lines in AST, zero lines $>100$ characters, 100% docstring & type hint coverage. In Second Pass: enriched clinical DataFrames with `mutations_cleaned.csv` driver mutation flags (`mut_BRAF`, `mut_NRAS`, `mut_NF1`), enabling complete univariate association testing across all trial cohorts and pooled benchmark (identifying significant pooled `NF1` response association $\text{OR}=2.41, p=0.0386^*$). Resolved 2x2 contingency table zero-cell Wald CI anomaly for Liu 2019 Stage (IV vs III) by switching `_calc_categorical_or()` to `scipy.stats.contingency.odds_ratio` exact hypergeometric CIs ($95\% \text{ CI} = [1.0587, \infty]$), eliminating artificial Haldane-Anscombe Wald variance inflation below 1.0 and ensuring 100% consistency between plotted CIs and Fisher exact p-values ($p=0.0295^*$), verifying clean execution with exit code 0. | **Resolved** (2026-08-07) |

## Conventions Quick Reference

- **Pipeline pattern**: Each Q has `run_qN_pipeline.py` or `run_pipeline.py` → sequential scripts
- **Logging**: All scripts use `TeeStream` to dual-write stdout to `logs/` directory
- **Figures**: 300 DPI PNG via `save_fig()`, 16:9 aspect ratio preferred
- **Reports**: Obsidian markdown with YAML frontmatter, callout boxes, British English
- **Colours**: Okabe-Ito palette only, accessed via `src/styles.py` helpers
- **Paths**: Always use `src/utils/paths.py` constants, log relative paths via `rel_path()`
