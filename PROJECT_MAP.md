# Project Architecture Map

> **Purpose**: Living reference document for agent orientation. Read this FIRST before
> exploring the codebase. Eliminates redundant file-discovery across conversations.
>
> **Last updated**: 2026-08-02 (Commit `e2c3a0f`: Expanded Q1 ensemble to 12 multimodal features including TMB; created `tcga_immune` processed cohort; generated 2×2 clinical demographics grid `clinical_demographics_2x2_grid.png` and 3-cohort PCA `batch_effect_ici_pca.png`; updated Section 1.2 in `batch_correction_report.md` and Section 1.1 in `cohort_characteristics_clinical.md`).

## Repository Overview

**Domain**: Melanoma immunotherapy – predicting anti-PD-1/CTLA-4 response, drug sensitivity, tumour dynamics, and patient stratification across multi-cohort clinical trial data (Liu 2019, Hugo 2016, Riaz 2017, TCGA-SKCM).

**Structure**: 5 research questions (Q1–Q5), each in its own subproject directory, plus shared infrastructure in the root `src/` and `data/` directories. Q5 is the master synthesis engine that integrates Q1–Q4 outputs.

```
melanoma-assignment-3/
├── .agents/AGENTS.md       <- Style rules, palettes, code conventions
├── PROJECT_MAP.md          <- THIS FILE – architecture reference
├── src/                    <- Shared source (styles, utils, config, biology constants)
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
| `src/utils/dataframes.py` | DataFrame manipulation helpers |
| `src/utils/preprocessing.py` | Data cleaning and transformation |

### Configuration (`src/config/`)

| Module | Key Exports |
|--------|------------|
| `src/config/datasets.py` | `load_dataset_config()` – loads `data/config/datasets.yaml` |
| `src/config/constants.py` | Central domain/pathway definitions |

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
| `q1/` | Q1 feature matrices, model predictions |
| `q2/` | Q2 cleaned viability matrices, patient drug-sensitivity predictions |
| `q5/` | Q5 outputs: `feature_matrix.csv` (ICI-only, N≈326), `feature_matrix_full.csv` (all cohorts, N≈699), `patient_clusters.csv`, `kmeans_model.pkl`, `clustering_feature_cols.json` |

## Q1: Response Predictor (`q1-response-predictor/`)

**Question**: Can we predict immunotherapy response from baseline tumour profiles?

### Phase → Script Mapping

| Phase | Script | Purpose |
|-------|--------|---------|
| 1 | `01_load_and_prepare.py` | Load multi-cohort data, compute immune signatures (TIS, CYT, IFN-γ) |
| 2 | `02_univariate_biomarkers.py` | Mann-Whitney U, Fisher's exact, ROC AUC per biomarker |
| 3 | `03_multivariate_model.py` | Logistic Regression + Random Forest with stratified CV |
| 4 | `04_threshold_optimisation.py` | Youden's J optimal cutoff selection |
| 5 | `05_cohort_validation.py` | Leave-One-Cohort-Out (LOCO) cross-validation |
| 6 | `06_summary_report.py` | Final summary report compilation |
| – | `generate_q1_report.py` | Comprehensive Q1 markdown report generator |
| – | `run_pipeline.py` | Master pipeline orchestrator |

### Source Modules (`q1-response-predictor/src/`)

| Module | Purpose |
|--------|---------|
| `data_loading.py` | Loads merged clinical, expression, genomic CSVs |
| `feature_engineering.py` | TIS, CYT, IFN-γ signature computation; TMB; response binarisation |
| `modelling.py` | Model training pipelines and hyperparameter grids |
| `evaluation.py` | ROC AUC, classification reports, confusion matrices |
| `preprocessing.py` | Missing-value imputation, feature scaling |
| `visualisation.py` | ROC curves, confusion matrices, forest plots |
| `reporting.py` | Obsidian frontmatter and markdown helpers |

### Key Outputs

- `models/logistic_regression_final.pkl`, `random_forest_final.pkl`
- `models/threshold_optimisation_results.json`, `feature_importance_rankings.csv`
- `plots/` subdirs: `biomarkers/`, `models/`, `validation/`, `threshold/`

## Q2: Viability Predictor (`q2-viability-predictor/`)

**Question**: Can we predict cell-line drug sensitivity and translate to patient tumours?

> **Note**: Q2 uses an **R-based pipeline** (`Question2.R`), unlike Q1/Q3/Q5 which use Python scripts.

### Script

| File | Purpose |
|------|---------||
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
- No `models/` directory – coefficients saved as CSV artifacts

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

> **Note**: A full BRAFi × anti-PD-1 combination grid is explicitly deferred – see `docs/Q3_Refactor_Proposal.md`.

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

> **Note**: Q4 uses a script-based workflow. Integration into Q5 is via `data/config/q4_depmap_targets.csv` and `q4_lincs_perturbagens.csv`.

## Q5: Patient Stratification (`q5-patient-stratification/`)

**Question**: Can we identify clinically distinct patient subgroups requiring different treatments?

**Role**: Master synthesis engine integrating Q1–Q4 outputs into a clinical decision framework.

### Phase → Script Mapping

| Phase | Script | Purpose |
|-------|--------|---------|
| 1 | `01_load_and_prepare.py` | Load merged data, compute signatures (TIS, CYT, IFN-γ, CD8_Tcell, IMPRES), M1/M2 STV deconvolution, cell-type estimates, and spatial proxy indicators across 33 multi-modal features (19 transcriptomic + 14 genomic) |
| 2 | `02_feature_analysis.py` | Mann-Whitney U, Fisher's exact, Youden cutoffs, interaction terms, feature credibility |
| 3 | `03_cluster_patients.py` | **Two-Stage GMM**: Stage 1 GMM (K=3, full covariance) on 6 continuous immune/stromal features (TIS, CYT, CD8_T_cells, M1/M2_Macrophages, CAFs); Stage 2 deterministic NF1 split produces 4 final phenotypes. Exports named probability columns (P_Immune_Hot, P_Immune_Cold, P_Immunosuppressive_M2_High, P_Mutant_Driven) mapped via runtime-safe label-to-index resolution. PCA/t-SNE/UMAP projections, TMB violin, spatial violins, model persistence. |
| 4 | `04_phenotype_characterisation.py` | Cluster profiling, phenotype labelling, KM survival. Dual-arm Q3-parameterised Kuznetsov 2-state ODE trajectories: Panel A = Immunotherapy (Anti-PD-1 monotherapy + M2 CAF-rescue combination), Panel B = Targeted Therapy (Vemurafenib BRAFi 500 nM). Per-patient r derived from Q3 pERK/pERK_ref coupling (Module A→B); per-patient c from Q3 checkpoint f_kill (Module D) and CYT. Phenotype-level pheno_r_mult applied in targeted arm to separate NF1-loss (0.68×), M2-High (1.25×), and NRAS-paradox (1.00×) cohorts. |
| 5 | `05_subgroup_models.py` | Soft-weighted GMM probability Random Forest models trained on 10 non-circular features (excluding 9 Phase 3 clustering features), evaluated via LOCO CV vs Global Enriched Baseline |
| 6 | `06_clinical_utility.py` | Decision Curve Analysis, Net Benefit, NNT, PPV, clinical benchmarks |
| 7 | `07_treatability_scoring.py` | Treatability Index, Q2 drug integration, Q4 DepMap/LINCS target nominations |
| – | `08_compare_clustering_algorithms.py` | Algorithmic comparison: K-Means vs Ward vs GMM vs DBSCAN |
| – | `09_run_consensus_clustering.py` | 1,000-bootstrap Consensus Clustering ensemble across patients & features ($K \in [2, 8]$), Consensus CDF, Delta Area $\Delta(K)$ |
| – | `generate_q5_report.py` | Comprehensive Q5 markdown report generator |
| – | `run_q5_pipeline.py` | Master pipeline orchestrator (runs Phases 1–7) |

### Source Modules (`q5-patient-stratification/src/`)

| Module | Purpose |
|--------|---------|
| `q5_constants.py` | Cell-type markers, immune signature genes, clustering features, `PHENOTYPE_PROB_COL` mapping, resistance pathways, treatability features, phenotype labels, Q3 ODE params, Q4 target nominations |
| `clustering.py` | `prepare_clustering_features()`, `run_gmm()` (Two-stage GMM, K=3 continuous features + Stage 2 NF1 split), `plot_2d_cluster_projection()` (PCA, t-SNE, UMAP). `run_kmeans()` is a deprecated legacy alias. |
| `phenotyping.py` | `profile_clusters()`, `assign_phenotype_labels()`, `plot_baseline_signature_boxplots()` (implemented). `plot_radar_chart()` and `plot_cluster_heatmap()` are implemented but **not called** by any current pipeline script – they are uncalled exports. |
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

### Cross-Question Data Flow

```
Q1 models (LR/RF .pkl) ──────────────┐
Q2 drug predictions (viability CSV) ──┤
Q3 ODE params (per phenotype) ────────┼──► Q5 Phase 7: Treatability Scoring
Q4 DepMap/LINCS targets ──────────────┘           │
                                                   ▼
Q5 patient_clusters.csv ──────────────► Q3 Phase 1: Phenotype ODE trajectories
                                                   │
data/processed/q5/treatability_scores.csv ─────────┼──► dashboard/scripts/build_cohort.mjs
data/processed/q5/phenotype_characterisation.csv ──┤      │
data/processed/q5/ode_trajectory_summary.json ─────┘      ▼
                                                      dashboard/public/cohort.json
                                                           │
                                                           ▼
                                                   React frontend (OncoTwin)
                                                   Q5PhenotypePanel · CohortTable
                                                   FeaturedCards · PatientPassport
```

## Module Dependency Graph

```
Shared src/
├── src/styles.py            ← imported by ALL scripts and Q-specific src/ modules
├── src/biology_constants.py ← imported by preprocessing scripts, Q5 constants
├── src/utils/paths.py       ← imported by ALL scripts
├── src/utils/logging.py     ← imported by ALL pipeline runners
├── src/utils/plotting.py    ← imported by ALL plotting code
├── src/utils/formatting.py  ← imported by report generators
└── src/config/datasets.py   ← imported by data-loading scripts

Q5 internal dependency chain:
  q5_constants.py ← clustering.py, phenotyping.py, clinical_utility.py
  clustering.py   ← 03_cluster_patients.py, 08_compare_clustering_algorithms.py
  phenotyping.py  ← 03_cluster_patients.py, 04_phenotype_characterisation.py
  deconvolution.py ← 01_load_and_prepare.py
```

### Known Technical Debt & Recent Resolution History

| Area | Description | Status |
|------|-------|--------|
| `src/styles.py` L53 & L178 | Duplicate `get_phenotype_color()` definitions (first is legacy, second added later) | **Resolved** (2026-08-01: duplicate definition at L53 removed) |
| `q5/src/reporting.py` | Local `generate_obsidian_frontmatter()` duplicates `src/utils/formatting.py` version | **Resolved** (2026-08-01: deleted local copy; `reporting.py` now re-exports from `src.utils.formatting`; shared version extended with `extra_css_classes` param) |
| `q5/src/phenotyping.py` | `plot_radar_chart()` and `plot_cluster_heatmap()` are fully implemented but are **not called by any pipeline script**. They are uncalled exports – dead code in the execution path. | Unresolved |
| `q5_constants.py` | `PHENOTYPE_FEATURES` defined but never used | **Resolved** (replaced with central `CLUSTERING_FEATURES` & `PHENOTYPE_PROFILE_FEATURES`) |
| `run_pipeline.py` (Q1) | Still writes log to project root instead of `logs/` directory | Unresolved |
| `q5/scripts/04_phenotype_characterisation.py` | Phase 4 now renders a dual-arm Kuznetsov 2-state ODE figure (Panel A: Immunotherapy, Panel B: Targeted Therapy) without IQR confidence shading. Per-patient r derived from Q3 pERK coupling; c from Q3 checkpoint f_kill and CYT. Targeted arm uses phenotype-level pheno_r_mult to achieve visual separation (NF1-loss=0.68×, M2-High=1.25×). All changes confined to `04_phenotype_characterisation.py`; q3-ode-model/ unchanged. | **Resolved** |
| `q5/scripts/05_subgroup_models.py` | Unused `LogisticRegression` import; magic numbers inline; 183-line `train_and_eval_loco` monolith; `SimpleImputer`+`StandardScaler` duplicated 5×; `PHENOTYPE_SHORT_NAMES` key `"M2 Immunosuppressive"` mismatched `PHENOTYPE_PALETTE` (silent wrong colour); missing docstrings on helpers. | **Resolved** (Phase 5 refactored 2026-07-31) |
| `q5/scripts/05_subgroup_models.py` & `06_clinical_utility.py` | Feature circularity (53% overlap with Phase 3 clustering features), GMM index instability, missing `TMB_NONSYNONYMOUS`. Resolved by excluding clustering features, establishing `PHENOTYPE_PROB_COL` in `q5_constants.py`, implementing soft GMM mixture weighting, and restoring `TMB_NONSYNONYMOUS` in clinical merge. | **Resolved** (2026-07-31) |
| `q5/scripts/06_clinical_utility.py` & `07_treatability_scoring.py` | Legacy markdown generation functions (`generate_phase6_markdown` / `generate_phase7_markdown`) wrote loose duplicate files in `reports/`. Unified report generation entirely into `generate_q5_report.py`, removed legacy report writers from Phase 6 & 7 scripts, and added `generate_q5_report` as Step 08 in `run_q5_pipeline.py`. | **Resolved** (2026-08-01: clean single source of truth for Q5 reporting) |
| `q5/scripts/05_subgroup_models.py` – single-class calibration, OOF fallbacks & sample inflation | Resolved single-class calibration failure via `_IdentityPredictor` fallback and guards (`len(y_eff)<2`, `len(np.unique(y_cal))<2`, `np.isnan(raw_cal_prob)`). Resolved degenerate zero-weight OOF predictions by pre-filtering training features/labels to non-zero weight samples (`w_tr_k > MIN_PROB_WEIGHT`) in `_build_subgroup_k_pred`, eliminating Scikit-Learn tree bootstrap division-by-zero (`0 NaNs generated`). Resolved GMM sample inflation by updating `MIN_PROB_WEIGHT` from `1e-6` to `0.05` ($5\%$ minimum membership threshold), pruning 156 noisy non-member patients from the *Immune Cold* training array (count reduced from 192 to 36) while retaining $98\%$ of effective sample weight ($N_{\text{eff}} = 25.5$). | **Resolved** (2026-08-01: zero NaNs in LOCO CV, noise pruned) |
| `q5/scripts/03_cluster_patients.py` – code smells (Phase 3 audit 2026-08-01) | Seven issues found and fixed: (1) **Critical** hardcoded GMM component indices 0/1/2 in `_export_cluster_outputs` replaced with runtime-derived `stage1_short_labels` map; `_assign_labels` returns 3-tuple. (2) `import seaborn as sns` inside `_plot_tmb_by_phenotype` body moved to module level. (3) Dead function `_format_prob_col_name` (defined, never called) removed. (4) Unused import `CLUSTERING_FEATURES` removed. (5) Inline 10-line per-cluster summary loop in `main()` extracted to `_print_stratification_summary()`. (6) Module docstring output list updated to include all 6 generated plots and the metrics CSV. (7) Stray extra blank line removed. Second and third pass refactored functions to script-private, added `mkdir` guards, fixed Seaborn deprecations, and consolidated duplicate `get_phenotype_color()` in `src/styles.py`. | **Resolved** (2026-08-01) |
| `q5/scripts/generate_q5_report.py` – Phase 3 report overhaul | Overhauled Phase 3 report generation in `generate_q5_report.py`: updated methodology text to Two-Stage Stratification ($K=3$ GMM continuous immune features + Stage 2 deterministic `NF1` split), eliminated hardcoded static TMB medians/percentages in favor of live DataFrame computations, corrected the false "100% BRAF" claim in `Immune Hot` to live driver percentages (49.6% `BRAF`+, 25.2% `NRAS`+, 12.6% `NF1`+), and updated UMAP manifold figure descriptions. | **Resolved** (2026-08-01) |
| `q5/scripts/06_clinical_utility.py` – code smells (Phase 6 audit 2026-08-01) | Evaluated and refactored Phase 6 clinical utility script across a 3-pass audit: (1) Fixed tuple return size mismatch in `compute_net_benefit()` (5 vs 6 elements). (2) Decomposed overlength functions (100% of functions in module are strictly $\le 30$ lines). (3) Purged unused imports (`generate_obsidian_frontmatter`, `PHENOTYPE_PALETTE`, `PROJECT_ROOT`) and unused constant (`REPORTS_DIR`). (4) Integrated `safe_save_csv()` for atomic I/O. (5) Centralized uppercase constants for features, DCA threshold grids, and plot limits. (6) Added zero-division ($p_t=1.0$) and single-class `LogisticRegression` safeguards. | **Resolved** (2026-08-01: 3-pass audit completed) |
| `q5/scripts/07_treatability_scoring.py` – code smells, 1st pass (Phase 7 audit 2026-08-01) | (1) **Critical bug**: `plot_treatability_distributions()` incomplete – figure initialised but never rendered/saved. Fixed. (2) Decomposed 5 overlength functions into 8 helpers; all functions ≤ 30 lines. (3) Purged 4 unused imports/constants. (4) Centralised 16 magic numbers as named constants. | **Resolved** (2026-08-01) |
| `q5/scripts/07_treatability_scoring.py` – DRY & residual smells, 2nd pass (Phase 7 audit 2026-08-01) | (1) **DRY**: Added `_safe_z_score(df, col)` to replace 6× repeated `_z_score` guards. (2) **DRY**: Added `_safe_minmax(df, col)` to replace 4× repeated `_minmax` guards. (3) **DRY**: Added `_add_phenotype_column(df)` to replace 3× repeated `get_cluster_name_map` + `.map()` pattern. (4) **DRY**: Extracted `_build_alignment_scores(df)`. (5) **Dead code**: Removed no-op fallback path in `load_q2_dabrafenib_weights()`. (6) **Magic numbers**: Centralised 12 residual inline constants (`AG_PRES_*_WEIGHT`, `MUT_STRENGTH_*`, `ALIGN_SCORE_*`, `STD_EPSILON`, `BAR_ANNOTATION_MIN_HEIGHT`, `KDE_MIN_*`). (7) **Naming**: Renamed `ax1`/`ax2` panel params to `ax`. 32 functions total, all strictly ≤ 30 lines. | **Resolved** (2026-08-01: DRY, dead code, residual magic numbers) |
| `q5/scripts/07_treatability_scoring.py` – Empirical Weight Optimisation (Phase 7 enhancement 2026-08-01) | Implemented data-driven Treatability Index sub-score weighting via `fit_empirical_treatability_weights(df)`. Replaced static heuristic constants ($0.35/0.35/0.15/-0.15$) with L2-regularised logistic regression coefficients fitted on $N = 195$ response-annotated patients (`RESPONSE_BINARY`). Derived empirical weights: AgPres = $-0.0247$, IFN = $+0.3751$, Effector = $+0.1096$, Barrier = $-0.2966$. Extracted `_get_default_treatability_weights()` and `_build_treatability_feature_matrix()` to maintain 35/35 functions strictly $\le 30$ lines. | **Resolved** (2026-08-01: dynamic L2 empirical weight fitting) |
| `q5/scripts/07_treatability_scoring.py` – Linear Min-Max Rescaling & AST Code Smells (Phase 7 fix 2026-08-01) | (1) **Quantile Scaling**: Replaced linear min-max with `QuantileTransformer(output_distribution='uniform')` in `_rank_scale_treatability_index()`, expanding Treatability Index IQR span from $21.45$ to $50.00$ units ($+133\%$) and eliminating boundary outlier compression while preserving ROC-AUC ($0.5956$). (2) **AST Code Smells**: Resolved 12 AST-detected smells by moving `sklearn` imports to module top level, adding `Optional[Tuple[...]]` type hints, centralising phenotype display strings (`PHENO_NAME_*`), and passing `cluster_map` down to avoid redundant `get_cluster_name_map()` calls. 35/35 functions pass AST check $\le 30$ lines, 0 AST code smells remaining. | **Resolved** (2026-08-01: quantile scaling & 100% clean AST audit) |
| `q5/scripts/07_treatability_scoring.py` – Sigmoidal Boundary Smoothing (Phase 7 fix 2026-08-01) | Resolved deterministic cutoff boundaries and cliff-edge effects in Arm C sub-arm selection by implementing logistic sigmoidal transition weighting ($w_{\text{AXL}} = 1 / (1 + \exp(-0.2 \cdot (\text{TI} - 40.0)))$) and an explicit Equipoise Buffer Zone ($[35.0, 45.0]$) in `_evaluate_arm_c_therapy()`. 36/36 functions pass AST check $\le 30$ lines, 0 AST code smells remaining. | **Resolved** (2026-08-01: sigmoidal boundary smoothing) |
| `q5/scripts/07_treatability_scoring.py` – NRAS Confidence Capping Removal (Phase 7 fix 2026-08-01) | Removed hard capping rule in `_assign_confidence_band()` that forced `NRAS`-mutant Arm B patients to `Moderate` status regardless of Q2 sensitivity score. Retained continuous `MUT_STRENGTH_NRAS` ($0.70$) weighting in `_compute_arm_b_confidence()`; $+10$ `NRAS`-mutant patients with exceptional Q2 sensitivity ($>88/100$) now reach `High` confidence ($N_{\text{High}} = 173$). 36/36 functions pass AST check $\le 30$ lines, 0 AST code smells remaining. | **Resolved** (2026-08-01: NRAS confidence capping removal) |
| Dashboard BRAF × PD-L1 heuristic | `FeaturedCards` and `CohortTable` previously used hard BRAF/PD-L1 cutoffs to define 3 clinical archetypes. Replaced with Q5 Two-Stage GMM phenotype labels (Immune Hot / Cold / M2-High / Mutant-Driven) from `treatability_scores.csv`. `PatientPassport` pills now show Q5 phenotype + Confidence + TI. Integration engine decision path replaces the `"pdl1"` node with a `"phenotype"` node when `patient.q5` is present. | **Resolved** (2026-08-01: Q5 dashboard integration) |
| Dashboard Q5 Enhanced ML Predictor & Score Relabeling | Added `Q5 Enhanced ML Predictor` tab to patient workbench navigation; relabeled Treatability Index score in recommendation cards to `TI Score` (`/100`) to eliminate confusion with statistical confidence; centralized `M1_MACROPHAGE_GENES` and `M2_MACROPHAGE_GENES` in `src/biology_constants.py`; updated `q1-response-predictor` signature extraction to 8 signatures (`+ Macrophage_STV_Score, + M1_M2_Ratio`). | **Resolved** (2026-08-02) |

## Dashboard Subproject (`dashboard/`)

**Technology**: React 18 + Vite + TypeScript + TailwindCSS. Static SPA – no server required.

**Purpose**: Clinical decision-support demonstrator. Loads `public/cohort.json` (built from Q3–Q5 outputs) and renders every TCGA-SKCM patient's digital twin, Q5 stratification, and integrated treatment recommendation.

### Key Files

| File / Directory | Purpose |
|------------------|---------|
| `scripts/build_cohort.mjs` | Ingests Q3 ODE CSVs + TCGA clinical + Q1 predictions + Q5 scores → `public/cohort.json`. Re-run whenever upstream Q3/Q5 outputs change. |
| `scripts/extract_palette.py` | Reads `src/styles.py` → auto-generates `src/data/palette.ts` so React inherits Okabe-Ito colours exactly. |
| `src/data/cohort.ts` | TypeScript interfaces for the cohort JSON: `CohortPatient`, `Q5Phenotype`, `CohortMeta`, etc. |
| `src/data/palette.ts` | Auto-generated colour helpers (`getPhenotypeColor`, `getCohortColor`, `getArmColor`). Do **not** edit manually. |
| `src/lib/integrationEngine.ts` | Q5 integration – derives ranked options from `patient.q5` when present, falls back to `scoreArms()`. |
| `src/components/Q5PhenotypePanel.tsx` | Per-patient Q5 panel: phenotype badge, GMM probability bar, Treatability Index, ODE trajectory. |
| `src/components/Q5MlPredictorPanel.tsx` | Per-patient enhanced ML predictor panel: subgroup Random Forest model, 33-feature panel, head-to-head comparison, feature importances, cross-validation metrics. |
| `src/components/FeaturedCards.tsx` | Landing page archetype cards – 4 phenotype examples (Immune Hot / Cold / M2-High / Mutant-Driven). |
| `src/components/CohortTable.tsx` | Searchable/filterable patient table with Q5 Phenotype, TI, and Confidence Band columns. |
| `src/components/PatientPassport.tsx` | Patient header pills – Q5 phenotype + confidence + TI when `q5` present; falls back to BRAF/PD-L1 pills. |
| `src/components/PatientView.tsx` | Full patient workbench with 7 tabs (`Q5 · Stratification`, `Q1 · ML predictor`, `Q2 · Validation`, `Q3 · Digital twin`, `Q4 · Resistance`, `Q5 Enhanced ML Predictor`, `Decision path`). |

### Build Pipeline

```
npm run build
  └── prebuild
        ├── python scripts/extract_palette.py   → src/data/palette.ts
        └── node scripts/build_cohort.mjs       → public/cohort.json
  └── tsc --noEmit && vite build
```

### Cohort Scope

Currently `TCGA_ONLY = true` in `build_cohort.mjs` – only TCGA-SKCM rows from `treatability_scores.csv` are joined.
To extend to all 4 ICI-trial cohorts (Option B): set `TCGA_ONLY = false` and add a cohort-selector UI component.

### Q5 → Dashboard Data Flow

| Q5 Output File | Dashboard Usage |
|----------------|-----------------|
| `data/processed/q5/treatability_scores.csv` | Per-patient `q5` block (phenotype, TI, GMM probs, therapy arm) |
| `data/processed/q5/phenotype_characterisation.csv` | `meta.q5PhenotypeStats` (proportion bar, summary strip) |
| `data/processed/q5/ode_trajectory_summary.json` | `meta.q5OdeTrajectory` (ODE endpoint minicard in Q5PhenotypePanel) |
| `data/processed/q5/subgroup_models_evaluation.csv` | `meta.q5SubgroupAuc` (per-phenotype AUC for future display) |

## Conventions Quick Reference

- **Pipeline pattern**: Each Q has `run_qN_pipeline.py` → sequential `01_...py` through `0N_...py` scripts
- **Logging**: All scripts use `TeeStream` to dual-write stdout to `logs/` directory
- **Figures**: 300 DPI PNG via `save_fig()`, 16:9 aspect ratio preferred
- **Reports**: Obsidian markdown with YAML frontmatter, callout boxes, British English
- **Colours**: Okabe-Ito palette only, accessed via `src/styles.py` helpers
- **Paths**: Always use `src/utils/paths.py` constants, log relative paths via `rel_path()`
