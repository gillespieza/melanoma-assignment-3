---
title: "Phase 4 Limitations Audit"
aliases:
  - "q5-phase4-limitations"
  - "Phase 4 Limitations"
tags:
  - report
  - q5
  - limitations
  - phase-4
created: "2026-07-31 17:32"
updated: "2026-07-31 17:32"
cssclasses:
  - "table-small"
  - "table-center"
  - "row-alt"
obsidianEditingMode: "preview"
obsidianUIMode: "source"
---

## Overview

> [!NOTE]
> **What is being done**: A rigorous critical audit evaluating the statistical weaknesses, biological assumptions, and computational constraints of Q5 Phase 4 phenotype characterisation and Ordinary Differential Equation (ODE) trajectory modelling.
> **Why we are doing it**: To systematically document the methodological boundaries of our current patient stratification pipeline and identify vulnerabilities that could impact clinical interpretation.
> **What question it answers**: What statistical, biological, and computational limitations restrict the predictive validity of Phase 4, and how can they be resolved in future refactoring iterations?

Phase 4 couples soft Gaussian Mixture Model (GMM) clustering of N=699 patients across four biological phenotypes (Immune Cold, Mutant-Driven, Immune Hot, Immunosuppressive M2-High) with dynamic 180-day ODE trajectory simulations (`ode_trajectories.png`). While this framework connects static multi-omic profiling to temporal treatment response, several critical statistical, biological, and data architecture limitations remain.

## Statistical Limitations

> [!WARNING]
> **What is being done**: Evaluation of sample size asymmetries, multiple testing risks, and parameter approximation uncertainties within the Phase 4 analytical framework.
> **Why we are doing it**: Unbalanced cluster distributions and unadjusted survival statistics can generate overconfident clinical conclusions for underrepresented patient subsets.
> **What question it answers**: Where are the statistical findings most vulnerable to sampling noise, confounding, or parameter approximation errors?

### Extreme Cluster Size Asymmetry and Low Subgroup Power
The Immune Cold cluster (Cluster 0) represents a tiny fraction of the pooled patient cohort, containing only N=22 patients (3.1% of N=699). When matched to the Q3 genomic parameter matrices for ODE trajectory simulation, this subpopulation shrinks further to just N=13 patients. Consequently, empirical response rates (50.0%) and mean biomarker signature Z-scores calculated for this group carry very wide confidence intervals, making statistical inference fragile and highly sensitive to sampling variation.

### Unadjusted Survival Log-Rank Testing
Kaplan-Meier survival stratification across the four phenotypes (`km_survival_by_phenotype.png`) relies on unadjusted multivariate log-rank tests. These tests do not adjust for key clinical confounders, including patient age, prior systemic therapies, disease stage, or treatment cohort origin. Furthermore, because GMM clustering features (`TIS`, `CYT`, `CD8A`, `CD163`) partially overlap with established prognostic survival factors, there is an inherent risk of circularity in the survival evaluation.

### Cluster-Level Multiplier Approximation (`pheno_r_mult`)
Although baseline proliferation rates ($r$) and immune killing coefficients ($c$) are parameterised per-patient via Q3 pERK coupling and checkpoint $f_{\text{kill}}$ equations, the targeted therapy ODE arm relies on a cluster-level scalar (`pheno_r_mult = 0.68` for Mutant-Driven, `1.25` for M2-High). Applying a single scalar multiplier across an entire phenotype cluster reintroduces a cluster-mean approximation, partially masking intra-cluster heterogeneity in drug response.

### Literature-Derived Targeted Kinetic Rates
The kinetic rate constants governing tumour proliferation suppression under `BRAF` inhibition in the ODE targeted arm are literature-derived approximations. They have not been formally calibrated or fitted to the empirical cell-line drug viability AUC scores generated in Q2, limiting the numerical precision of simulated tumour regression rates.

## Biological Assumptions and Model Boundaries

> [!WARNING]
> **What is being done**: Critical examination of biological mechanisms, cell types, and pharmacology missing from the current ODE trajectory formulation.
> **Why we are doing it**: To clarify the biological boundaries of the differential equation model and prevent over-interpretation of simplified cell dynamics.
> **What question it answers**: Which physiological tumour microenvironment features and drug resistance mechanisms are currently absent from the Phase 4 simulation?

### Absence of an Explicit Stromal Compartment
The 2-state Kuznetsov ODE system models tumour cells ($T$) and immune effector cells ($E$), but lacks an explicit Cancer-Associated Fibroblast (CAF) or M2 macrophage compartment. Although M2-High tumours are defined transcriptomically by high CAF (`TGFB1`) and M2 macrophage (`CD163`, `ARG1`) scores, the ODE represents stromal exclusion indirectly by lowering the killing coefficient ($c$) and elevating the proliferation multiplier (`pheno_r_mult = 1.25`), rather than modelling physical T-cell barrier mechanics through explicit cell-cell interaction equations.

### Lack of Secondary Resistance and Phenotype Switching
Dynamic ODE simulations assume static drug sensitivity over the entire 180-day treatment period. The model does not incorporate acquired resistance mechanisms, such as secondary `NRAS` or `MAP2K1` mutations, MEK bypass reactivation, or microenvironmental phenotype switching (e.g. an Immune Hot tumour transitioning to an Immune Cold state under anti-`PDCD1` treatment pressure).

### Drug Compound Mismatch Between Q2 and Phase 4
The targeted therapy arm of the ODE models Vemurafenib pharmacology based on Q3 Module A parameters ($K_D = 50\text{ nM}$). However, Q2 cell-line viability screens evaluated Dabrafenib and PLX-4720, but not Vemurafenib directly. Individual patient drug sensitivity AUC scores predicted in Q2 are not currently linked to the Phase 4 targeted therapy ODE, leaving an unresolved compound mismatch between the drug sensitivity and trajectory modules.

### Targeted Arm Parameterisation Disconnect for `NF1`-Loss Tumours
In the targeted therapy arm (Panel B of `ode_trajectories.png`), Mutant-Driven tumours are simulated under Vemurafenib `BRAF` inhibition, yielding partial regression ($T(180) = 0.79$) via `pheno_r_mult = 0.68`. However, empirical profiling reveals this cluster is **0% `BRAF` V600E mutant and 100% `NF1` loss-of-function**. `NF1` loss eliminates RasGAP activity, resulting in high constitutive RAS-GTP levels. Under high RAS-GTP conditions, `BRAF` inhibitors induce RAF monomer-dimer transitions that paradoxically *activate* ERK signalling — the established RAF-inhibitor paradox. Simulating `BRAF` inhibitor sensitivity for `NF1`-loss tumours is mechanistically inaccurate; the biologically appropriate targeted agent for this cluster is MEK inhibition (`MAPK1` / `MAP2K1` targeting via Trametinib), which acts downstream of RAS.

## Computational and Data Constraints

> [!WARNING]
> **What is being done**: Audit of data integration paths, file dependencies, and cohort subset mismatches between Phase 4 and upstream modules.
> **Why we are doing it**: To streamline computational data flows and eliminate architectural redundancies across project modules.
> **What question it answers**: What technical data flow constraints exist between Q3 simulation outputs and Phase 4 reporting scripts?

### Re-Simulated ODE Trajectories vs Direct Pre-Computed Ingestion
Phase 4 re-simulates dynamic trajectories locally by coupling Q3 steady-state pERK and checkpoint $f_{\text{kill}}$ functions to 2-state Kuznetsov equations, rather than ingesting Q3's pre-computed 4-module patient simulation files (`tumour_burden_simulations.csv`). This creates architectural code duplication between `q3-ode-model/` and `q5-patient-stratification/`.

### Cohort Subset Discrepancy
While GMM soft clustering operates on the full pooled dataset of N=699 patients, dynamic ODE trajectory plotting in Phase 4 is restricted to the N=421 patients matched to Q3 expression and genomic parameter matrices (`melanoma_params_full.csv`). This creates a sample size discrepancy between baseline cluster characterisation (N=699) and trajectory plotting (N=421).

## Actionable Fixes and Improvement Roadmap

> [!NOTE]
> **What is being done**: Presenting a prioritised, actionable roadmap to resolve identified statistical, biological, and computational limitations in future pipeline iterations.
> **Why we are doing it**: To provide concrete engineering steps for enhancing model accuracy, clinical fidelity, and pipeline efficiency.
> **What question it answers**: What specific modifications should be prioritised in the next refactoring cycle?

| Priority | Proposed Improvement | Targeted Limitation | Expected Impact |
| :--- | :--- | :--- | :--- |
| **P1** (Highest) | Direct Ingestion of Q3 Simulation CSVs | Re-Simulated ODE Trajectories | Eliminates code duplication by reading pre-computed Q3 `tumour_burden_simulations.csv` files directly into Phase 4 reporting. |
| **P1** | Re-parameterise Mutant-Driven Targeted Arm for MEK Inhibition | `NF1`-Loss Targeted Disconnect | Replaces `BRAF` inhibitor simulation with Trametinib MEK inhibitor kinetics for `NF1`-loss tumours, resolving the RAF paradox error. |
| **P1** | Add an Explicit CAF / M2 Stromal ODE Compartment | Absence of Stromal Compartment | Introduces a 3rd differential equation for stromal density, enabling mechanistic simulation of physical T-cell exclusion in M2-High tumours. |
| **P2** | Calibrate Proliferation Rates to Empirical Q2 Viability Scores | Literature-Derived Targeted Rates | Fits targeted therapy $r$ parameters directly to patient-specific Q2 Dabrafenib / PLX-4720 viability AUC predictions. |
| **P2** | Adjusted Cox Proportional Hazards and Multiple-Testing Correction | Unadjusted Survival Testing | Implements FDR-adjusted log-rank tests and multivariate Cox regression adjusted for age, stage, and cohort origin. |
| **P3** | Model Time-Varying Drug Efficacy $r(t)$ for Acquired Resistance | Lack of Secondary Resistance | Introduces time-decaying drug efficacy after Day 60 to simulate secondary `NRAS` mutations and MEK bypass resistance. |

## Key Takeaways

*   **Subgroup Power Deficit**: The Immune Cold phenotype is severely underpowered ($N=22$ full cohort, $N=13$ ODE subset), requiring cautious clinical interpretation of its empirical response rate.
*   **Targeted Arm Disconnect**: Simulating `BRAF` inhibition on the Mutant-Driven cluster (100% `NF1` loss, 0% `BRAF` V600E) is biologically inaccurate due to the RAF paradox; future iterations must re-parameterise this arm for MEK inhibition (Trametinib).
*   **Implicit Stromal Dynamics**: The 2-state ODE lacks a dedicated CAF compartment, relying on cluster-level multipliers (`pheno_r_mult = 1.25`) to approximate M2-High stromal exclusion rather than modelling cell-cell interaction physics.
*   **Unlinked Q2 Viability Profiles**: Targeted therapy kinetic rates are literature approximations unlinked to Q2 per-patient drug sensitivity AUC predictions, leaving a compound mismatch between Vemurafenib and Dabrafenib.
*   **Clear Refactoring Roadmap**: Prioritised improvements focusing on direct Q3 CSV ingestion, MEK inhibitor re-parameterisation, and 3-state stromal ODE expansion will significantly elevate Phase 4 model rigor.
