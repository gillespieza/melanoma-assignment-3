# Q5: Patient Stratification, Cell Deconvolution & Clinical Utility Pipeline

Translating Q1 predictive response models into clinically actionable patient subtypes, cell-type deconvolution metrics, Q2 drug sensitivity predictions, Q3 ODE dynamic simulations, Q4 DepMap/LINCS drug target nominations, and treatability index scoring for advanced melanoma immunotherapy.

---

## Executive Summary

This subproject implements **Question 5 (Q5)** of the Melanoma Immunotherapy Assignment (*"Can we identify clinically distinct patient subgroups? Do subgroups require different treatments?"*). It serves as the **master synthesis engine** uniting all 5 project questions (Q1–Q5) into a single bench-to-bedside clinical decision framework:

```
   Q1: Patient IO Predictor ──┐
                              ├──► Q5: Patient Stratification ──► Discovers 4 Patient Phenotypes
   Q2: Cell Line Sensitivity──┘    (Immune Hot, Cold, M2-High, Mutant-Driven)
   (Dabrafenib/Dacarbazine)              │
                                         ┌────────────────────────────┴────────────────────────────┐
                                         ▼                                                         ▼
                                 Q3: ODE Dynamic Models                                   Q4: DepMap & LINCS
                                 Simulates Tumour Volume T(t)                             Identifies Novel Targets & Drug Perturbagens
                                 Trajectories for Each Phenotype                          to Overcome Non-Response
                                         │                                                         │
                                         └────────────────────────────┬────────────────────────────┘
                                                                      ▼
                                                       Q5: Treatability Index & Decision Support
                                                       (Routes Patient to Arm A, Arm B, or Arm C)
```

Key features & multi-question integrations in Q5:
* **Q1 Immunotherapy Prediction**: Inputting patient baseline expression/clinical profiles to predict anti-PD-1/CTLA-4 response probability.
* **Q2 Cell Viability & Drug Sensitivity**: Integrating Q2 drug response models for standard-of-care targeted therapies (*Dabrafenib*, *Trametinib*) and chemotherapy (*Dacarbazine*) to select specific agents for non-responders.
* **Cell Count & Deconvolution Analysis**: Quantitative transcriptomic deconvolution of immune cell fractions and integration of a professor-provided **M1/M2 Macrophage Signature Transcript Vector (STV)** ([m1_m2_stv.csv](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/data/config/m1_m2_stv.csv)).
* **Q3 ODE Dynamic Trajectories**: Parameterising ODE tumour-immune differential equations for each discovered phenotype to plot simulated 180-day tumour volume regression ($T(t)$) under monotherapy vs. combination therapy.
* **Q4 DepMap & LINCS Target Nominations**: Mapping DepMap CRISPR essentiality targets (`AXL`, `MDM2`, `CSF1R`) and LINCS L1000 perturbational gene signatures to overcome non-response in therapy-resistant phenotypes.
* **Clinical Utility & Treatability Scoring**: Decision Curve Analysis (DCA), NNT calculations, and scoring non-responders for reversible biological barriers to recommend combination interventions.

---

## Workflow & Implementation Architecture

The analysis is structured into 7 sequential phases executed via standalone scripts and orchestrated by a master pipeline:

```
                  ┌──────────────────────────────────────────────────────────┐
                  │                    INPUT DATASETS                        │
                  │  • clin_merged.csv, expr_merged.csv, merged_genomic.csv  │
                  │  • Q1 Serialised Models & Q2 Cell Line Viability Scores  │
                  │  • Professor M1/M2 STV Matrix (data/config/m1_m2_stv.csv)│
                  └────────────────────────────┬─────────────────────────────┘
                                               │
                                               ▼
    Phase 1 ──► [01_load_and_prepare.py] ────► Load data, extract signatures & M1/M2 STV
                                               │
    Phase 2 ──► [02_feature_analysis.py] ───► Univariate associations & interaction tests
                                               │
    Phase 3 ──► [03_cluster_patients.py] ───► Unsupervised K-Means/Ward & UMAP projection
                                               │
    Phase 4 ──► [04_phenotype_characterisation.py] ─► Annotate phenotypes & Q3 ODE Trajectories
                                               │
    Phase 5 ──► [05_subgroup_models.py] ───► Train subgroup-specific predictive models
                                               │
    Phase 6 ──► [06_clinical_utility.py] ───► Decision Curve Analysis (DCA) & NNT
                                               │
    Phase 7 ──► [07_treatability_scoring.py] ─► Treatability Index, Q2 Drugs & Q4 Targets
                                               │
                                               ▼
                  ┌──────────────────────────────────────────────────────────┐
                  │                    OUTPUT ARTIFACTS                      │
                  │  • Consolidated Feature Matrix & Patient Clusters CSVs   │
                  │  • 300 DPI Publication Visualisations & Plots             │
                  │  • Detailed Markdown Reports in reports/                 │
                  └────────────────────────────┴─────────────────────────────┘
```

---

## Step-by-Step Analytical Phases

### Phase 1: Feature Matrix & Deconvolution Preparation
* **Script**: [01_load_and_prepare.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/01_load_and_prepare.py)
* **Tasks**:
  * Loads multi-cohort processed data from `data/processed/merged/immunotherapy/`.
  * Computes core immune signatures: IFN-$\gamma$, TIS, CYT, CD8 T-cell, IMPRES, and `CD274` (PD-L1) expression.
  * Calculates M1 and M2 macrophage scores using the professor's 14,837-gene STV matrix ([m1_m2_stv.csv](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/data/config/m1_m2_stv.csv)) and derives the M1/M2 ratio (`M1 / (M1 + M2)`).
  * Performs transcriptomic cell-type deconvolution (estimating CD8+ T cells, CD4+ T cells, M1/M2 macrophages, NK cells, B cells, and CAFs).
  * Exports unified matrix to `data/processed/q5/feature_matrix.csv`.

### Phase 2: Deep Feature Interpretation
* **Script**: [02_feature_analysis.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/02_feature_analysis.py)
* **Tasks**:
  * Conducts Mann-Whitney U testing (continuous features) and Fisher's exact / Chi-Square testing (categorical mutations/subtypes like `BRAF`, `NRAS`, `NF1`) against immunotherapy response.
  * Determines optimal decision cutoffs using Youden's J statistic.
  * Tests pairwise logistic regression feature interaction terms (e.g. `TIS` $\times$ `BRAF`).
  * Generates feature credibility matrices combining Q1 model importance, effect sizes, and cohort consistency.

### Phase 3: Unsupervised Patient Stratification
* **Script**: [03_cluster_patients.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/03_cluster_patients.py)
* **Tasks**:
  * Scales continuous features via `StandardScaler`.
  * Evaluates K-Means and Hierarchical Agglomerative (Ward linkage) clustering across $K = 2 \dots 8$.
  * Computes silhouette coefficients, elbow inertias, and GAP statistics to select optimal $K$.
  * Generates 2D UMAP projections coloured by cluster, cohort, and response status.
  * Exports cluster assignments to `data/processed/patient_clusters.csv`.

### Phase 4: Biological Phenotype Characterisation & Q3 ODE Integration
* **Script**: [04_phenotype_characterisation.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/04_phenotype_characterisation.py)
* **Tasks**:
  * Profiles per-cluster feature means, response rates, and M1/M2 ratios.
  * Assigns biological phenotype labels:
    * **Immune Hot**: High TIS, high CD8, high M1/M2 ratio ($\sim 65\%$ response).
    * **Immune Cold**: Low TIS, low CD8, low infiltrate ($\sim 20\%$ response).
    * **Immunosuppressive M2-High**: High M2 macrophage abundance, high CAF infiltration ($\sim 15\%$ response).
    * **Mutant-Driven**: High TMB, high neoantigen burden ($\sim 50\%$ response).
  * Performs Kaplan-Meier overall survival (OS) analyses and log-rank tests across phenotypes.
  * **Q3 ODE Integration**: Simulates differential equation tumour growth/regression trajectories ($T(t)$ over 180 days) using phenotype-specific initial effector cell counts ($E_0$) and kill rates ($a$).

### Phase 5: Subgroup-Specific Predictive Models
* **Script**: [05_subgroup_models.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/05_subgroup_models.py)
* **Tasks**:
  * Trains separate predictive models (Logistic Regression, Random Forest) within each phenotype subgroup ($N \ge 15$).
  * Evaluates subgroup Leave-One-Cohort-Out (LOCO) CV performance against the global Q1 model to quantify biological heterogeneity.

### Phase 6: Clinical Utility & Decision Impact Analysis
* **Script**: [06_clinical_utility.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/06_clinical_utility.py)
* **Tasks**:
  * Calculates Net Benefit across threshold probabilities ($0.1 - 0.9$) for Decision Curve Analysis (DCA).
  * Computes Number Needed to Treat (NNT) and Positive Predictive Value (PPV).
  * Compares Q1 model performance against standard clinical strategies (*Treat All*, *High TMB*, `CD274` / PD-L1+).

### Phase 7: Treatability Index, Q2 Drugs & Q4 Target Nominations
* **Script**: [07_treatability_scoring.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/07_treatability_scoring.py)
* **Tasks**:
  * Evaluates non-responders for reversible barriers (antigen presentation integrity, IFN-$\gamma$ pathway mutations, CNA burden, and targetable `BRAF`, `NRAS`, `PTEN` mutations).
  * Computes per-patient Treatability Index scores ($0 - 1$).
  * **Q2 Drug Integration**: Connects Q2 cell-line viability and drug sensitivity predictions to select specific targeted agents (Arm B: *Dabrafenib/Trametinib*) or chemotherapy agents (Arm C: *Dacarbazine*).
  * **Q4 DepMap/LINCS Integration**: Maps CRISPR essentiality targets (`AXL`, `MDM2`, `CSF1R`) and LINCS L1000 perturbagens to supply specific combination therapy recommendations for resistant phenotypes.

---

## Directory & File Map

```
q5-patient-stratification/
├── README.md                      <- Project documentation (this file)
├── README_STUDENT_GUIDE.md        <- Biological & statistical study guide
├── requirements.txt               <- Q5 additions (umap-learn, dcurves)
├── logs/                          <- Pipeline TeeStream execution logs
├── models/                        <- Serialised subgroup classifier models
├── plots/                         <- 300 DPI publication-ready figures
│   ├── clustering/                <- UMAP, silhouette, and elbow plots
│   ├── phenotypes/                <- Radar charts, KM curves, and Q3 ODE T(t) plots
│   ├── clinical_utility/          <- Decision curves and NNT charts
│   └── feature_analysis/          <- Univariate volcano & interaction plots
├── reports/                       <- Detailed markdown analysis reports
├── scratch/                       <- Temporary exploratory scripts
├── scripts/
│   ├── 01_load_and_prepare.py     <- Data loading, signature & STV computation
│   ├── 02_feature_analysis.py     <- Association testing & Youden thresholds
│   ├── 03_cluster_patients.py     <- K-Means/Ward clustering & UMAP
│   ├── 04_phenotype_characterisation.py <- Subtype annotation & Q3 ODE trajectories
│   ├── 05_subgroup_models.py     <- Subgroup-specific predictive modeling
│   ├── 06_clinical_utility.py     <- Decision Curve Analysis (DCA)
│   ├── 07_treatability_scoring.py <- Treatability index, Q2 drugs & Q4 targets
│   └── run_q5_pipeline.py         <- Master pipeline orchestrator
└── src/                           <- Q5-specific Python source package
    ├── __init__.py
    ├── q5_constants.py            <- Feature sets, Q3 ODE params & Q4 target maps
    ├── clustering.py              <- Clustering & GAP statistic algorithms
    ├── phenotyping.py             <- Subtype profiling & radar chart utilities
    ├── feature_analysis.py        <- MW-U, Fisher's, & interaction statistics
    └── clinical_utility.py        <- DCA net benefit & NNT calculators
```

---

## Execution & Verification

Test end-to-end pipeline execution (including Q2, Q3, and Q4 integration stubs):

```bash
python q5-patient-stratification/scripts/run_q5_pipeline.py
```
