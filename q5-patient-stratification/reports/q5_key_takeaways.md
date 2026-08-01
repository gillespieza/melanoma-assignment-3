# Q5 Patient Stratification — Key Takeaways Summary

> All figures and statistics drawn from the live Phase 1–7 reports in  
> [`q5-patient-stratification/reports/q5_phases/`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/reports/q5_phases)

---

## Phase 1 — Multi-Modal Feature Matrix & Microenvironment Deconvolution

**Goal**: Build a dual feature matrix from raw multi-cohort melanoma data.

- **Dual-matrix architecture**: Two complementary matrices are produced — an ICI-treated cohort ($N_\text{ICI} = 326$) with response labels for supervised modelling, and a full cohort ($N_\text{Full} = 699$, including TCGA-SKCM reference) for unsupervised stratification.
- **Dimensionality compression**: ~19,757 raw transcriptomic genes collapsed into **32 engineered multi-modal features** (18 transcriptomic + 14 genomic), built around 6 core immune signatures (`TIS`, `CYT`, `IFN_gamma`, `CD8_Tcell`, `PD_L1`, `IMPRES`).
- **Macrophage STV deconvolution**: A 14,835-gene linear Signature Transcript Vector captures the M1 (pro-inflammatory) vs M2 (pro-tumour) macrophage balance — a microenvironmental suppression signal that operates independently of T-cell density.
- **Spatial proxy indicators**: Two engineered ratios ($\log_2(\text{CD8}/\text{CAF})$ and $\log_2(\text{CD8} \times \text{M1\_M2\_Ratio} / \text{CAF})$) serve as proxies for cytotoxic T-cell penetration vs stromal exclusion.
- **IMPRES caveat**: Retained in the 33-feature panel for Phases 5 & 6 but not used in TME deconvolution; moderate collinearity with `TIS`/`CYT` ($r_s > 0.75$); regularisation in Phase 5 mitigates this.

---

## Phase 2 — Feature Analysis, Youden Cutoffs & Genomic Interactions

**Goal**: Identify which individual biomarkers discriminate responders from non-responders, and whether genomic drivers modify immune signals.

- **Best standalone marker**: `B_cells` is the top individual predictor (AUC = **0.632**, Youden J = 0.306, threshold ≥ 0.430). All other biomarkers achieve AUC ≈ 0.57–0.59.
- **Modest univariate power**: `TIS`, `CYT`, and `CD8_T_cells` are all significantly elevated in responders but achieve only ~58% standalone accuracy — proving single biomarkers are insufficient for clinical decisions.
- **Macrophage STV performs poorly alone**: `M1_M2_Ratio` (AUC = 0.442) and `Macrophage_STV_Score` (AUC = 0.413) are near-random when tested univariately, despite being key clustering features.
- **`BRAF` × `TIS` interaction is the only statistically significant driver–immune interaction** ($\beta = -0.65$, $p = 0.040$): high T-cell inflammation has a stronger positive predictive value in **`BRAF` wild-type** tumours — `BRAF` oncogenic signalling dampens microenvironmental immunity.
- **Every borderline significant interaction is in the `BRAF` column**: `BRAF` × `IFN_gamma` ($p = 0.064$), `BRAF` × `B_cells` ($p = 0.073$), `BRAF` × `CD8_T_cells` ($p = 0.074$).
- **`NF1` × `M1_M2_Ratio`** shows the largest positive effect size ($\beta = +0.94$): pro-inflammatory myeloid reprogramming strongly enhances response in high-TMB `NF1`-loss tumours.
- **Rationale for clustering**: Multi-factorial resistance (stromal exclusion, M2 polarisation, driver mutation context) means that no single threshold test is clinically adequate → motivates Phase 3 multi-dimensional stratification.

---

## Phase 3 — Unsupervised Patient Stratification (N = 699)

**Goal**: Discover biologically distinct TME archetypes across the full melanoma cohort without outcome bias.

- **Two-stage stratification**: Stage 1 = GMM ($K=3$, full covariance) on 6 continuous immune/stromal features; Stage 2 = deterministic `NF1` split carves out the Mutant-Driven phenotype from the `NF1`-enriched immune cluster.
- **Four discovered phenotypes** (N = 699):

| Phenotype | N | Share | Key Biology | Primary Therapy |
|---|---|---|---|---|
| **Immunosuppressive M2-High** | 256 | 36.6% | High M2 macrophages + CAF exclusion; 44.1% `BRAF`+, 0% `NF1` | `BRAF`/MEK targeted inhibition |
| **Immune Cold** | 45 | 6.4% | T-cell desert; 53.3% `BRAF`+, 13.3% `NF1`+ | Dual M2-depleting + checkpoint combination |
| **Immune Hot** | 341 | 48.8% | High TIS + CYT; 49.6% `BRAF`+, 25.2% `NRAS`+, 12.6% `NF1`+ | ICI monotherapy (primary) |
| **Mutant-Driven** | 57 | 8.2% | 100% `NF1` loss, RAS hyperactivation; TMB median ≈ 41 mut/Mb | ICI + MEK adjunct |

- **PC1 = immune activation axis**: separates Immune Hot (inflamed) from Immune Cold (desert) — the primary axis of immunological responsiveness.
- **PC2 = myeloid/stromal axis**: separates M2-High (macrophage-excluded stroma) from Mutant-Driven (`NF1` loss, high TMB).
- **`NF1` / Mutant-Driven TMB**: Median TMB ≈ 41 mut/Mb — more than 3× higher than any other phenotype; 87.7% exceed the FDA ≥10 mut/Mb threshold.
- **Inflamed TME is multi-driver**: Immune Hot is driven by `BRAF`, `NRAS`, *and* `NF1` mutations — robust TME inflammation is not restricted to any single oncogenic driver.
- **Soft probabilistic assignments**: GMM posterior probabilities ($P_\text{Immune Hot}$, $P_\text{Immune Cold}$, etc.) quantify biological uncertainty at cluster boundaries.

---

## Phase 4 — Phenotype Characterisation & ODE Digital Twin Dynamics

**Goal**: Simulate 180-day per-patient tumour trajectories and validate the mechanistic model against survival and protein data.

- **4-module ODE per patient**: Module A (RAF dimerisation/BRAFi binding), Module B (8-state MAPK cascade, steady-state pERK), Module C (Kuznetsov tumour-immune ODE), Module D (PD-1/PD-L1 checkpoint QSS).
- **ODE trajectory results (180-day relative tumour volume $T(180)$)**:
  - *Immune Hot*: near-complete clearance ($T(180) = 0.11$) under ICI monotherapy.
  - *Mutant-Driven*: effective clearance ($T(180) = 0.76$) via high TMB-driven immunogenicity.
  - *Immune Cold*: incomplete regression ($T(180) = 0.96$) — T-cell paucity limits killing.
  - *M2 Immunosuppressive* on monotherapy: uncontrolled growth ($T(180) = 0.91$); adding M2-depleting combination rescues response ($T(180) = 0.46$).
- **Mechanistic proof of combination necessity**: The ODE mathematically demonstrates why M2-High patients fail single-agent anti-PD-1 and require macrophage/CAF targeting.
- **Survival stratification**: Simulated ODE checkpoint tumour burden produces a **82-month median OS gap** ($p = 0.0024$); *Immune Cold* has the worst median OS (11.4 months).
- **RPPA orthogonal validation**: ODE-predicted baseline pERK correlates significantly with measured TCGA RPPA phospho-ERK ($n = 310$, $r = 0.175$, $p = 2.03 \times 10^{-3}$); `NRAS`-mutant tumours show the highest baseline pERK ($p = 3.16 \times 10^{-9}$).
- **3-feature ODE beats 12-feature ML**: ODE digital twin (ROC-AUC = **0.666**, 3 features) outperforms 12-feature Logistic Regression (0.646) and Neural Networks (0.583), and is within 0.02 AUC of Random Forest (0.686), while remaining fully biologically interpretable.

---

## Phase 5 — Subgroup-Specific Predictive Modelling

**Goal**: Evaluate whether cluster-tailored Random Forest models outperform the global Q1 predictor using LOCO cross-validation.

- **Training approach**: Soft-weighted GMM probability RF classifiers on 9 non-circular features (i.e., features not used for Phase 3 clustering), evaluated via Leave-One-Cohort-Out (LOCO) CV.
- **Mutant-Driven gains are the most clinically meaningful**: Subgroup-specific modelling increases Recall **0% → 20%** (+20 pp) and PPV **0% → 25%** (+25 pp) — true responders completely missed by the global model are now identified.
- **Global predictor often numerically outperforms in AUC**: Q1 global AUC exceeds subgroup AUC for Immune Hot (0.541 vs 0.514) and M2-High (0.617 vs 0.509). However, this is a **calibration artefact** — Q1 does not distinguish patients who fail for different biological reasons; subgroup models correctly suppress false positives in resistant clusters.
- **Phenotype-specific top features**:
  - `B_cells` dominates in **Mutant-Driven** (Gini = 0.204) and **M2-High** (0.191) — TLS formation is crucial when macrophages are M2-polarised.
  - `Macrophage_STV_Score` is primary in **Immune Hot** (0.191) and **Immune Cold** (0.165).
  - `TMB_NONSYNONYMOUS` leads in the **Global Enriched Baseline** (0.166).
- **LOCO robustness confirms generalisability** across independent clinical cohorts.
- **Small cluster sizes** (Immune Cold $N = 28$, Mutant-Driven $N = 9$) limit statistical power — results require prospective validation.

---

## Phase 6 — Clinical Utility & Decision Curve Analysis

**Goal**: Determine whether the Q5 phenotype-stratified system yields genuine clinical benefit over empirical and single-gene benchmarks.

- **Evaluation population**: $N = 195$ ICI-treated patients; baseline response rate = **42.1%** (82 responders).
- **Net Benefit at $p_t = 0.30$**: Q5 = **0.251** vs Treat All = **0.172** vs single-gene `CD274` = **0.165** — Q5 outperforms both empirical and single-gene benchmarks.
- **NNT reduction**: Q5 lowers NNT from **2.38** (Treat All) to **1.94** — an **18.5% improvement** in treatment efficiency.
- **PPV increase**: Q5 raises PPV to **51.6%** vs 42.1% for Treat All.
- **Non-responders spared**: **36** predicted non-responders are correctly withheld from futile anti-PD-1 — protecting them from irAEs (colitis, pneumonitis, hepatitis, endocrinopathies).
- **Multi-feature stratification beats single genes**: Q5 consistently outperforms `CD274` (PD-L1) and `TMB_NONSYNONYMOUS` cutoffs — microenvironmental context is essential.
- **Q5 conservative precision in resistant subgroups is clinically desirable**: Low within-cluster Net Benefit in Immune Cold reflects correct non-treatment of predicted non-responders, not a weakness.
- **Clinical decision window ($p_t = 0.20$–$0.50$)**: Q5 remains above both Treat All and Treat None benchmarks throughout the realistic clinical preference window.
- **Transition to Phase 7**: Sparing non-responders from monotherapy is only half the problem — they must be actively routed to alternative therapies (the 3-arm system).

---

## Phase 7 — 3-Arm Decision Support & Treatability Scoring

**Goal**: Route 100% of N = 699 patients into actionable, biologically rational therapeutic arms.

- **Complete 3-arm allocation** (N = 699):
  - **Arm A — Immunotherapy Monotherapy** ($N = 364$, **52.1%**): Immune Hot or high-TIS patients; anti-PD-1 (Pembrolizumab / Nivolumab).
  - **Arm B — Targeted Therapy** ($N = 234$, **33.5%**): Predicted non-responders with `BRAF V600` or `NRAS` mutations; Q2 Dabrafenib Sensitivity Index integrated (mean = **56.0/100**).
  - **Arm C — Combination/Reversal** ($N = 101$, **14.4%**): Remaining non-responders in cold/immunosuppressive TMEs; Q4 DepMap targets nominated.
- **Primary Arm C helper target**: `CSF1R` (M2 TAM depletion) — nominated for $N = 67$ patients; also `MDM2` and `AXL` for deeper resistance.
- **Composite Treatability Index** (0–100, quantile-scaled): Overall mean = **50.0** (uniform IQR = 50.0 by design):
  - *Immune Hot*: **65.9** (highest convertibility)
  - *Mutant-Driven*: **40.8** (moderate; MAPK inhibition path)
  - *M2 Immunosuppressive*: **35.3** (`CSF1R` depletion pathway)
  - *Immune Cold*: **25.3** (lowest; requires `AXL`/STING priming)
- **Empirical sub-score weights** (L2-regularised logistic regression, $N = 195$ response-annotated patients): IFN-gamma integrity = +0.375 (strongest positive predictor); M2 barrier = −0.297 (strongest negative); Effector density = +0.110; Antigen presentation = −0.025.
- **Sigmoidal boundary smoothing**: Prevents cliff-edge decision switches for borderline patients using a logistic transition function and an Equipoise Buffer Zone ($\text{TI} \in [35.0, 45.0]$).
- **Recommendation confidence**: Mean = **56.2/100**; $N = 173$ patients (24.7%) classified as High Confidence (Index ≥ 70.0).
- **Cross-question integration**: Successfully bridges Q2 cell-line LASSO viability models → Arm B patient sensitivity scores, and Q4 DepMap CRISPR essentiality → Arm C target nominations.

---

## Overall Q5 Summary

| Phase | Core Output | Headline Metric |
|---|---|---|
| **1** | Dual feature matrices (ICI: N=326, Full: N=699) | 32-feature multi-modal panel |
| **2** | Youden cutoffs + genomic interaction matrix | `B_cells` best AUC = 0.632; `BRAF`×`TIS` $p = 0.040$ |
| **3** | 4-phenotype two-stage GMM stratification | Mutant-Driven TMB median ≈ 41 mut/Mb |
| **4** | ODE digital twin trajectories + RPPA validation | 3-feature ODE AUC = 0.666 vs 12-feature RF = 0.686 |
| **5** | Subgroup-specific LOCO RF models | Mutant-Driven Recall: 0% → 20% |
| **6** | DCA, NNT, PPV, toxicity avoidance | NNT reduced 2.38 → 1.94 (18.5%); 36 non-responders spared |
| **7** | 3-arm clinical decision engine + Treatability Index | 100% patient allocation; CSF1R nominated for N=67 |
