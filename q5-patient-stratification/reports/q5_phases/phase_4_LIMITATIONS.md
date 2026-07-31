---
title: Phase 4 Limitations Audit
aliases: [q5-phase4-limitations, Phase 4 Limitations]
tags: [report, q5, limitations, phase-4]
created: 2026-07-31 16:09
cssclasses: [table-small, table-center, row-alt]
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-31 16:09
---

## Overview

> [!NOTE] 
> **What is being done:** An audit of the limitations remaining in Q5 Phase 4 following the recent pipeline refactoring.
> **Why we are doing it:** To systematically document the statistical, biological, and computational constraints of the current Ordinary Differential Equation (ODE) framework.
> **What question it answers:** What are the known vulnerabilities in the Phase 4 phenotype characterisation and trajectory modelling, and how can they be improved in future iterations?

This report outlines the structural and methodological limitations of the Phase 4 analysis. Phase 4 characterises patient phenotypes, generating survival plots (`km_survival_by_phenotype.png`), biomarker signature distributions (`baseline_signature_boxplots.png`), and ODE trajectories (`ode_trajectories.png`). A total of N=699 patients were stratified into four clusters: Immune Cold (N=22, 3.1%, Response Rate=50.0%), Mutant-Driven (N=65, 9.3%, Response Rate=68.8%), Immune Hot (N=304, 43.5%, Response Rate=41.1%), and Immunosuppressive M2-High (N=308, 44.1%, Response Rate=38.0%). 

## What Was Fixed: The Dual-Arm ODE Integration

> [!INFO]
> **What is being done:** Documentation of the resolved limitations regarding therapeutic arms.
> **Why we are doing it:** To acknowledge the recent pipeline refactoring that successfully expanded the mechanistic modelling scope.
> **What question it answers:** Which major limitations were successfully addressed in the latest update?

- **[Resolved] Targeted Therapy Integration:** Phase 4 now correctly renders a dual-arm ODE trajectory figure. Panel A visualises Immunotherapy (Anti-`PDCD1` / Anti-`CD274` + Combination Rescue), whilst Panel B visualises Targeted Therapy (`BRAF` / MEK Inhibitor monotherapy). Previously, the targeted therapy arm was entirely absent from the Phase 4 ODE visualisations.

## Statistical Limitations

> [!WARNING]
> **What is being done:** Identification of statistical vulnerabilities in the clustering and survival analyses.
> **Why we are doing it:** To prevent overconfidence in the model's predictions, particularly for underrepresented patient subpopulations.
> **What question it answers:** Where are the statistical findings most likely to fail or lack power?

### Extreme Cluster Size Asymmetry
The Immune Cold cluster (Cluster 0) is severely underpowered, containing only N=22 patients (3.1% of the cohort). Consequently, any response rate or biomarker mean calculated for this subpopulation is statistically unreliable due to very wide confidence intervals. Furthermore, Kaplan-Meier survival curves for this cluster possess very low statistical power.

### Unadjusted Log-Rank Kaplan-Meier Test
The multivariate log-rank test is not adjusted for critical clinical confounders such as patient age, treatment cohort, or prior therapy history. Furthermore, because clustering was performed using biomarker features that partially overlap with known survival predictors, there is a risk of circularity in the survival analysis.

### Mean-Feature Parameterisation of ODE
The current 2-state ODE uses cluster-mean biomarker values (e.g., mean `CD8A`, mean cytolytic activity via `PRF1` and `GZMA`) to define parameters. This aggregation eliminates all intra-cluster heterogeneity. As a result, all patients within a given phenotype cluster receive identical predicted trajectories, failing to provide true per-patient Digital Twins.

### Unfitted Targeted Therapy Parameters
The ODE parameters governing the targeted therapy arm (specifically the tumour proliferation rate and kill coefficient for `BRAF` inhibition) rely on literature-motivated approximations. They are not formally fitted to our empirical GDSC dose-response data, limiting their predictive accuracy.

## Biological Assumptions

> [!WARNING]
> **What is being done:** Examination of biological mechanisms absent or oversimplified in the current ODE formulation.
> **Why we are doing it:** To contextualise the model's biological fidelity and highlight missing tumour microenvironment factors.
> **What question it answers:** What biological realities are currently ignored by the mathematical model?

### Absence of Stromal Compartment
The 2-state ODE (tumour and immune effector cells) lacks a Cancer-Associated Fibroblast (CAF) or stromal compartment. The M2-High phenotype is characterised by very high CAF scores, yet the ODE cannot explicitly model this stromal exclusion. It merely approximates the effect via a reduced kill coefficient, thereby obscuring the true biological mechanism.

### Lack of Resistance Emergence Modelling
The ODE does not incorporate resistance emergence mechanisms. It fails to model `BRAF` inhibitor resistance via secondary `NRAS` mutations, `MAP2K1` / `MAPK1` bypass signalling, or phenotype switching (e.g., from Mutant-Driven to Immune Cold). The assumption that drug efficacy remains constant over 180 days is biologically unrealistic.

### Drug Compound Mismatch
The targeted therapy arm ODE utilises vemurafenib pharmacology parameters (derived from Q3 Module A, where KD_RAF=50 nM). However, Q2 tested Dabrafenib and PLX-4720, but not vemurafenib directly. No per-patient drug sensitivity scores from Q2 are linked to the targeted therapy ODE arm, leaving the drug compound mismatch unresolved.

### Ignored NF1 Loss-of-Function Biology
The ODE ignores the unique biology of `NF1` loss-of-function. `NF1` loss activates RAS signalling through the loss of GAP activity — a mechanism distinct from `NRAS` activating mutations, though both drive the MAPK pathway. The ODE currently sets the RAS-GTP level to RASGTP_NRAS = 0.9 for both, failing to differentiate them.

## Computational & Data Constraints

> [!WARNING]
> **What is being done:** Review of data flow and computational architecture limitations within Phase 4.
> **Why we are doing it:** To identify inefficiencies and structural disconnects between the Q3 simulation outputs and Phase 4 reporting.
> **What question it answers:** How does Phase 4 fail to utilise the full computational output of preceding modules?

### Disconnected Q3 Simulation Data
Phase 4 still does not read the per-patient simulation CSVs generated by Q3 (`tumour_burden_simulations.csv`, `checkpoint_tumour_simulations.csv`). Instead, Phase 4 re-simulates trajectories from scratch using a simplified 2-state equation, rather than the full 4-module Q3 system. Consequently, mechanistic phenomena such as the `BRAF` paradox, pERK-proliferation coupling, and checkpoint binding equilibria are entirely absent from Phase 4's visualisations.

### Absence of Confidence Shading on ODE Trajectories
The Phase 4 ODE plots currently render a single mean curve per phenotype. There is no interquartile range (IQR) or confidence shading, which hides the extensive intra-cluster variability and presents a misleadingly deterministic view of patient trajectories.

## Proposed Improvements

> [!NOTE]
> **What is being done:** Outlining a prioritised roadmap for addressing the identified limitations.
> **Why we are doing it:** To guide the next iteration of refactoring and analytical enhancements.
> **What question it answers:** What are the most impactful technical steps to improve the Phase 4 pipeline?

| Priority | Improvement | Expected Impact |
| :--- | :--- | :--- |
| **P1** (Highest) | Replace Phase 4's local 2-state ODE with a loader that reads Q3's output CSVs, joins on SAMPLE_ID with `patient_clusters.csv`, and aggregates by Phenotype_Label to plot mean +/- IQR bands. | Transforms Phase 4 into a true Digital Twin aggregation, incorporating the full Q3 mechanistic model. |
| **P1** | Add a CAF/M2 macrophage compartment to the Phase 4 ODE. | Allows mechanistic modelling of stromal exclusion for M2-High patients. |
| **P2** | Add Bonferroni or False Discovery Rate (FDR) adjustment to Kaplan-Meier log-rank tests and bootstrap confidence intervals for Cluster 0 (N=22). | Improves statistical rigour and prevents false positive conclusions from the underpowered cluster. |
| **P2** | Fit `BRAF` inhibitor ODE proliferation parameters to empirical Q2 PLX-4720 viability AUC scores per patient. | Links Q2 drug sensitivity directly to the Phase 4 targeted therapy ODE arm. |
| **P3** | Model resistance emergence via a time-varying parameter that increases after Day 60-90. | Better reflects the typical `BRAF` V600E melanoma resistance window. |

## Key Takeaways

- **Statistical Fragility:** The Immune Cold cluster (N=22) is severely underpowered, and unadjusted log-rank tests risk confounding, necessitating rigorous multiple testing correction in future iterations.
- **Architectural Disconnect:** Phase 4 re-simulates a simplified 2-state ODE rather than aggregating the high-fidelity, 4-module per-patient simulations generated in Q3, losing critical mechanistic depth.
- **Missing Stromal Mechanics:** The lack of a distinct CAF compartment in the ODE obscures the true mechanism of immune exclusion, relying on mathematical approximation rather than biological modelling.
- **Unrealistic Temporal Dynamics:** The model currently assumes constant drug efficacy over 180 days, completely ignoring the inevitable emergence of acquired resistance via bypass signalling or phenotype switching.
- **Drug Calibration Gap:** Targeted therapy parameters rely on generic literature approximations rather than being calibrated to the empirical Q2 dose-response viability scores.
