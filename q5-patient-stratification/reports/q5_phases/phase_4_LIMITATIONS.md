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

## What Was Fixed: Dual-Arm ODE & Q3 4-Module Digital Twin Integration

> [!INFO]
> **What is being done:** Documentation of the resolved limitations regarding therapeutic arms and mechanistic ODE architecture.
> **Why we are doing it:** To acknowledge the recent pipeline refactoring that successfully expanded the mechanistic modelling scope and resolved structural disconnects.
> **What question it answers:** Which major limitations were successfully addressed in the latest update?

- **[Resolved] Targeted Therapy Integration:** Phase 4 now correctly renders a dual-arm ODE trajectory figure (`ode_trajectories.png`). Panel A visualises Immunotherapy (Anti-`PDCD1` / Anti-`CD274` + Combination Rescue), whilst Panel B visualises Targeted Therapy (`BRAF` / MEK Inhibitor monotherapy).
- **[Resolved] Direct Q3 4-Module ODE Integration:** Phase 4 has been refactored to import and execute Q3's exact 4-module ODE system (`active_raf_signal`, `steady_pERK`, `checkpoint_kill_factor`, `cancer_rhs`) across all $N=699$ patients. The simplified local 2-state ODE has been completely eliminated.
- **[Resolved] Per-Patient Digital Twin Aggregation & IQR Shading:** Trajectories are now computed per-patient and aggregated by phenotype cluster, rendering the Mean curve alongside 25th–75th Interquartile Range (IQR) shaded bands to capture intra-cluster heterogeneity.
- **[Resolved] Surfacing the RAF-Inhibitor Paradox:** By running Q3 Module A across all patient genomes, the targeted therapy arm dynamically surfaces the RAF-inhibitor paradox for `NF1`-loss (Mutant-Driven) and `NRAS`-mutant (Immune Cold / M2-High) tumours, showing primary resistance under BRAFi monotherapy.

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

### Mutant-Driven Targeted Therapy Parameterisation Mismatch
The most significant biological error in the current Phase 4 ODE is the parameterisation of the Mutant-Driven cluster's targeted therapy arm. The ODE uses a dramatically reduced proliferation rate (r=0.04) to simulate BRAFi sensitivity — yet the empirical `phenotype_characterisation.csv` shows this cluster is **0% `BRAF` V600E and 100% `NF1` loss-of-function**. `NF1` loss activates RAS signalling through the loss of GAP activity, resulting in constitutively elevated RAS-GTP. In a high-RAS-GTP context, applying a BRAFi such as vemurafenib drives RAF monomer formation and paradoxical ERK *activation* — the well-characterised **RAF-inhibitor paradox** captured in Q3 Module A. The ODE therefore models a BRAFi sensitivity that is mechanistically inapplicable to `NF1`-loss tumours. The biologically correct targeted agent for this cluster is MEK inhibition (Trametinib), which acts downstream of RAS and suppresses ERK regardless of RAS-GTP level. This aligns with Q4's DepMap recommendation of Dabrafenib + Trametinib combination. The 68.8% empirical response rate for this cluster is better explained by strong immunotherapy response (high TMB, neoantigenic burden from `NF1`-loss) than by BRAFi sensitivity.

### Ignored NF1 Loss-of-Function Biology
Beyond the parameterisation mismatch above, the ODE does not differentiate `NF1` loss-of-function from `NRAS` activating mutations at the mechanistic level. Both are set to RASGTP_NRAS = 0.9 in Q3 Module A, yet they are biologically distinct: `NF1` loss eliminates a tumour suppressor, whereas `NRAS` mutations constitutively activate an oncogene. The distinction has implications for sensitivity to MEK inhibition vs. direct `NRAS` targeting strategies.

## Computational & Data Constraints

> [!WARNING]
> **What is being done:** Review of data flow and computational architecture constraints within Phase 4.
> **Why we are doing it:** To identify remaining data integration bounds between the Q3 simulation outputs and Phase 4 reporting.
> **What question it answers:** What computational constraints remain after the Q3 4-module ODE refactoring?

### Intra-Cluster Variance & Extreme Outliers
While Phase 4 now plots 25th–75th IQR shaded bands for every phenotype cluster, extreme outliers (e.g. patients with exceptionally high baseline `CD274` or `CYT`) are compressed into the summary IQR interval, which may obscure long-tail responders within resistant phenotypes.

## Proposed Improvements

> [!NOTE]
> **What is being done:** Outlining a prioritised roadmap for addressing the identified limitations.
> **Why we are doing it:** To guide the next iteration of refactoring and analytical enhancements.
> **What question it answers:** What are the most impactful technical steps to improve the Phase 4 pipeline?

| Priority | Improvement | Expected Impact |
| :--- | :--- | :--- |
| **P1** (Highest) | Replace Phase 4's local 2-state ODE with a loader that reads Q3's output CSVs, joins on SAMPLE_ID with `patient_clusters.csv`, and aggregates by Phenotype_Label to plot mean +/- IQR bands. | Transforms Phase 4 into a true Digital Twin aggregation, incorporating the full Q3 mechanistic model including the RAF paradox for `NF1`-loss and `NRAS`-mutant clusters. |
| **P1** | Re-parameterise the Mutant-Driven targeted therapy ODE arm to use MEK inhibitor kinetics (Trametinib) rather than BRAFi (vemurafenib). The cluster is 0% `BRAF` V600E and 100% `NF1`-loss; BRAFi monotherapy is paradoxically stimulatory in high-RAS-GTP tumours. | Removes the most significant biological error in the current Phase 4 ODE and correctly reflects Q4's MEK inhibitor recommendation for this subtype. |
| **P1** | Add a CAF/M2 macrophage compartment to the Phase 4 ODE. | Allows mechanistic modelling of stromal exclusion for M2-High patients. |
| **P2** | Add Bonferroni or False Discovery Rate (FDR) adjustment to Kaplan-Meier log-rank tests and bootstrap confidence intervals for Cluster 0 (N=22). | Improves statistical rigour and prevents false positive conclusions from the underpowered cluster. |
| **P2** | Fit targeted therapy ODE parameters to empirical Q2 PLX-4720 viability AUC scores per patient, and resolve the vemurafenib/Dabrafenib compound mismatch. | Links Q2 drug sensitivity directly to the Phase 4 targeted therapy ODE arm. |
| **P3** | Model resistance emergence via a time-varying parameter that increases after Day 60-90. | Better reflects the typical `BRAF` V600E melanoma resistance window and NF1-loss bypass mechanisms. |

## Key Takeaways

- **Statistical Fragility:** The Immune Cold cluster (N=22) is severely underpowered, and unadjusted log-rank tests risk confounding, necessitating rigorous multiple testing correction in future iterations.
- **Unified Q3 Architecture:** Phase 4 directly executes the full 4-module per-patient Q3 ODE system, accurately surfacing the RAF paradox and checkpoint equilibria across all $N=699$ Digital Twins with IQR confidence bands.
- **Mechanistic Precision (Mutant-Driven & Immune Cold):** The targeted therapy arm demonstrates primary resistance for `NF1`-loss (Mutant-Driven) and `NRAS`-mutant (Immune Cold) tumours due to drug-promoted dimerisation and paradoxical ERK activation, correctly pointing to MEK inhibition (Trametinib) as the appropriate target.
- **Missing Stromal Mechanics:** The lack of a distinct CAF compartment in the ODE obscures the true mechanism of immune exclusion for M2-High patients, relying on mathematical approximation rather than explicit cell-cell interaction modelling.
- **Drug Calibration Gap:** Targeted therapy parameters rely on generic literature approximations rather than being calibrated to the empirical Q2 dose-response viability scores, and the vemurafenib/Dabrafenib compound mismatch between Q2 and the ODE remains unresolved.
