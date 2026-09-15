---
title: "Project Task List & Backlog"
aliases:
  - "TODO"
  - "Task Backlog"
tags:
  - project-management
  - backlog
  - tasks
created: 2026-09-14 15:15
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-09-15 16:32
---

# 📋 Project Task List & Backlog

> [!NOTE]
> **Purpose**: Central living task tracker for the `melanoma-assignment-3` research repository and the `additional-datasets` branch refactoring.
> 
> **Status Legend**:
> - `[ ]` Not Started
> - `[/]` In Progress
> - `[x]` Completed

---

## 🚀 Active Sprint: `additional-datasets` Branch Refactoring

- [ ] **Working Tree Commit & Verification**
  - [ ] Finalise and verify `q1-response-predictor/scripts/exploratory_plots/generate_combined_cv_loco_heatmap.py` execution (per-cohort patient/responder breakdown logging, matrix NaN checks, OpenMP single-threaded worker flags).
  - [ ] Commit working tree changes with conventional commit message (`feat(q1-viz): add detailed cohort sample audit and OpenMP thread limits to CV/LOCO heatmap generator`).
  - [ ] Push local commits to `origin/additional-datasets`.
- [ ] **Cohort Evaluation & Expansion Candidates**
  - [ ] Investigate suitability of adding **Campbell / MORRISON-1** dataset (sample size, RNA-seq availability, anti-PD-1 response labels).
  - [ ] Investigate suitability of adding **GEM / Spanish Melanoma Group** (Grupo Español Multidisciplinar de Melanoma) dataset (data access, clinical annotations, harmonisation feasibility).
- [ ] **Full Q1 Pipeline Smoke Test**
  - [ ] Execute `q1-response-predictor/run_pipeline.py` end-to-end to verify 6-cohort data flow, 12-feature Random Forest model training, and report generation.
  - [ ] Inspect generated plots in `q1-response-predictor/plots/` for visual quality and compliance with Okabe-Ito palettes and KM numbers-at-risk standards.
- [ ] **Branch Sync & Merge Readiness**
  - [ ] Review diff against `main` (`rtk git log main..additional-datasets`).
  - [ ] Ensure all pillar reports in `q1-response-predictor/reports/` are synchronised with live dataset figures.

---

## 🔬 Subproject Workstreams

### Q1: Immunotherapy Response Predictor (`q1-response-predictor/`)
- [x] Dynamic YAML dataset configuration ingestion (`config/datasets.yaml` via `load_dataset_config()`).
- [x] Expansion to 6 active datasets ($N = 473$ response-labelled / $N = 478$ IT-treated subcohort).
- [x] Resolution of scikit-learn 1.8/1.9 deprecations (`penalty`, `CalibratedClassifierCV`, feature name tracking).
- [x] Pillar directory reorganisation and AST code smell refactoring ($\le 30$ line functions, $\le 100$ char line width).
- [x] Parallelisation of 5-fold CV and LOCO heatmaps (`generate_combined_cv_loco_heatmap.py`).
- [ ] Audit TMB comparability across cohorts: calculate TMB from mutation data where feasible, and document included mutation types, germline filtering, synonymous-variant filtering, mutation-count denominator, minimum coverage, and variant-level filtering criteria.
- [ ] Retain all eligible patients while deriving a patient-level CTLA-4 exposure flag; preserve unknown treatment history explicitly, keep current regimen separate from prior exposure, and report overall plus CTLA-4-stratified model performance.
- [ ] Verify `reports/pillar_4_out_of_cohort_benchmarks/` Markdown reports against the parallelised LOCO heatmap metrics.
- [ ] Investigate suitability of adding **Campbell / MORRISON-1** and **GEM / Spanish Melanoma Group** datasets.

### Q1.1: Patient Stratification (`q1.1-patient-stratification/`)
- [x] Two-Stage GMM clustering ($K=3$ continuous + Stage 2 `NF1` split) generating 4 biological phenotypes:
  - *Immunosuppressive M2-High*, *Immune Cold*, *Immune Hot*, *Mutant-Driven*.
- [x] Soft-weighted GMM Random Forest sub-models and Decision Curve Analysis (DCA).
- [ ] Clean up uncalled/orphaned exports in `src/phenotyping.py` (`plot_radar_chart()`, `plot_cluster_heatmap()`).
- [ ] Audit dataset loader consistency against `config/datasets.yaml` across all 7 phase scripts.

### Q2: Viability Predictor (`q2-viability-predictor/`)
- [x] R LASSO regression pipeline across 5 targeted therapies (`Question2.R`).
- [ ] Verify TCGA-SKCM viability predictions correlate consistently with updated Q1 multi-cohort response probabilities.

### Q3: ODE Tumour-Immune Dynamics (`q3-ode-model/`)
- [x] 4-Module kinetic simulation (RAF dimerisation, MAPK cascade, tumour/immune ODE, PD-1/PD-L1 equilibrium).
- [x] Two independent dose sweeps per patient (Vemurafenib 0–1000 nM, Anti-PD-1 0–500 nM).
- [ ] Verify `results/pERK_simulations.csv` and tumour burden trajectories against updated Q1.1 phenotype assignments.

### Q4: Functional Genomics & Target Discovery (`Q4_dep_map/`)
- [x] DepMap CRISPR essentiality screening & SOX10 proxy target discovery.
- [ ] Ensure LINCS L1000 perturbagen recommendations are referenced consistently in Q5 dashboard feeds.

### Q5: Clinical Decision Support Dashboard (`dashboard/`)
- [x] Static React 18 + Vite + TypeScript + TailwindCSS platform (OncoTwin).
- [x] 5-Tab workbench (`Q1 · ML predictor`, `Q2 · Validation`, `Q3 · Digital twin`, `Q4 · Resistance`, `Decision path`).
- [ ] Run `npm run build` (prebuild script `build_cohort.mjs` $\rightarrow$ `tsc --noEmit` $\rightarrow$ Vite export) to ensure `public/cohort.json` is synced with latest model outputs.

---

## 🛠️ Architecture, Code Quality & Documentation Debt

- [ ] **`PROJECT_MAP.md` Synchronisation**
  - [ ] Update resolution status for completed Q1 refactoring milestones.
  - [ ] Verify line number cross-references and script tables.
- [ ] **Observability & Logging Standards**
  - [x] Standardise `TeeStream` log directory routing to `<subproject>/logs/` across Q1, Q3, and Q1.1.
  - [ ] Migrate root-level `run_pipeline.py` log (`q1_pipeline.log`) to `q1-response-predictor/logs/`.
- [ ] **Documentation & Report Guidelines**
  - [x] Enforce British English spelling across all Markdown reports and UI labels.
  - [x] Wrap gene symbols in backticks across all documentation (`CD274`, `PDCD1`, `BRAF`, `NRAS`, `NF1`).
  - [x] Maintain canonical immune signature and ML model ordering across all report outputs.
