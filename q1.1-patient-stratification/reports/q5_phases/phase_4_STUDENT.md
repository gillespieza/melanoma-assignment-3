---
title: "Phase 4: Phenotype Characterisation & ODE Digital Twin Dynamics"
aliases:
  - Q5 Phase 4 Student Guide
  - Phase 4 Phenotype Characterisation
tags:
  - melanoma
  - patient-stratification
  - phase-4
  - ode-dynamics
  - q5
created: 2026-08-01 17:30
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-01 17:30
---

## Phase Overview

> [!NOTE] Phase 4 Overview
> - **What is being done**: We integrate baseline multi-omic signatures with a four-module literature-parameterised Ordinary Differential Equation (ODE) digital twin system to simulate 180-day dynamic tumour growth and regression trajectories across 699 melanoma patients.
> - **Why we are doing it**: Static biomarker snapshots cannot predict how a tumour will dynamically evolve under drug treatment over time; kinetic ODE simulation allows us to model treatment response, resistance, and combination therapy rescue.
> - **What question it answers**: How do distinct biological phenotypes dictate dynamic tumour regression or primary resistance under targeted and immune therapies over time?

Phase 4 bridges static biomarker characterisation and dynamic mathematical modelling. By stratifying the pooled patient cohort of 699 patients into four biological phenotypes (*Immunosuppressive M2-High*, *Immune Cold*, *Immune Hot*, and *Mutant-Driven*), we map baseline transcriptomic and mutational signatures to ODE parameters. This enables 180-day relative tumour volume trajectory simulations under both Immunotherapy (anti-`PDCD1` monotherapy and combination rescue) and Targeted Therapy (Vemurafenib `BRAF` inhibitor 500 nM).


## Upstream Integration

> [!INFO] Cross-Question Synthesis
> - **What is being done**: Phase 4 acts as a core integration bridge, connecting signature discovery from Question 1, drug sensitivity predictions from Question 2, differential equation kinetics from Question 3, and CRISPR essentiality targets from Question 4.
> - **Why we are doing it**: Effective patient stratification requires translating isolated molecular findings into an executable digital twin model that simulates clinical therapeutic response.
> - **What question it answers**: How do baseline immune signatures, driver mutations, and drug mechanisms interact to dictate patient-specific tumour regression curves?

This phase represents a major synthesis milestone across the assignment:
- **Question 1 Integration**: Leverages inflammatory gene expression signatures including `TIS`, `CYT`, `CD8A`, `PRF1`, `GZMA`, and `CD274` (`PD-L1`) to parameterise baseline immune infiltration and cytotoxic killing capacity.
- **Question 2 Integration**: Incorporates patient-level drug sensitivity concepts and `BRAF` / MEK pathway dependency signals.
- **Question 3 Integration**: Direct ingestion of the four-module ODE architecture (Module A: RAF dimerisation and paradoxical re-activation; Module B: 8-state MAPK cascade; Module C: Kuznetsov-de Pillis tumour-immune kinetics; Module D: `PDCD1` / `CD274` checkpoint binding).
- **Question 4 Integration**: Incorporates CRISPR essentiality insights and macrophage-mediated resistance pathways (`TGFB1`, `CD163`) to evaluate combination rescue protocols.


## Why This Matters for Patient Stratification (Q5)

> [!INSIGHT] Clinical Decision Relevance
> Understanding the temporal trajectory of each phenotype provides actionable guidance for precision oncology treatment routing:
> 1. **Immune Hot Patients (48.8% of cohort, N = 341)**: Display dense cytotoxic T-cell infiltration and robust baseline immune activation. They achieve near-complete tumour regression under single-agent anti-`PDCD1` checkpoint blockade (final relative tumour volume of 0.11 at Day 180).
> 2. **Immunosuppressive M2-High Patients (36.6% of cohort, N = 256)**: Characterised by high M2 macrophage and fibroblast exclusion. They experience treatment failure under anti-`PDCD1` monotherapy (final relative volume of 0.91), but achieve effective tumour regression (final relative volume of 0.46) when combined with an M2 macrophage-depleting rescue agent.
> 3. **Mutant-Driven Patients (8.2% of cohort, N = 57)**: Defined by `NF1` loss-of-function and high tumour mutational burden (TMB). They respond well to immunotherapy (final relative volume of 0.76) due to high neoantigen immunogenicity, but require MEK inhibitors (`MAPK1` / `MAP2K1` targeting via Trametinib) rather than `BRAF` inhibitors to prevent paradoxical RAF activation.
> 4. **Immune Cold Patients (6.4% of cohort, N = 45)**: Represent an immune desert with severe T-cell paucity. They exhibit primary resistance to immunotherapy (final relative volume of 0.96) and require experimental immune-priming protocols to recruit effector T cells before checkpoint administration.


## Key Phase Results

The pooled patient cohort of 699 patients is categorised into four distinct clinical phenotypes based on soft Gaussian Mixture Model (GMM) clustering of mutational and microenvironmental profiles:

| Biological Phenotype | Sample Size N (%) | Day 180 Anti-PD-1 Monotherapy | Day 180 M2-Rescue Combination | Key Molecular & Clinical Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **Immune Hot** | 341 (48.8%) | **0.11** (Clearance) | N/A | High baseline `TIS`, `CYT`, `CD8A`, and `CD274` expression; prime candidates for monotherapy |
| **Immunosuppressive M2-High** | 256 (36.6%) | **0.91** (Failure) | **0.46** (Regression) | Elevated M2 macrophages (`CD163`) and CAFs (`TGFB1`); requires combination macrophage depletion |
| **Mutant-Driven** | 57 (8.2%) | **0.76** (Regression) | N/A | `NF1` loss-of-function, high TMB and neoantigen load; highly immunogenic under checkpoint blockade |
| **Immune Cold** | 45 (6.4%) | **0.96** (Desert) | N/A | Deeply suppressed immune signatures across all markers; lowest median survival (11.4 months) |

### Major Dynamic & Validation Findings:
- **Kaplan-Meier Survival Stratification**: Stratifying N = 677 patients with overall survival metadata demonstrates statistically significant survival separation across the four phenotypes (log-rank p-value less than 0.001). *Immune Cold* exhibits the worst prognosis (median overall survival of 11.4 months), whereas *Immune Hot* and *M2-High* achieve longer median survival (27 to 29 months).
- **Q3 Checkpoint Survival Separation**: Stratifying TCGA-SKCM patients by ODE-simulated checkpoint tumour burden yields an 82-month median survival gap (148 months for low burden vs 66 months for high burden, p-value = 0.0024).
- **Orthogonal Protein Validation (RPPA)**: In N = 310 patients with Reverse-Phase Protein Array data, ODE-predicted baseline phospho-ERK correlates significantly with experimentally measured protein levels (r = 0.175, p-value = 0.00203). `NRAS`-mutant tumours exhibit the highest baseline phospho-ERK activation (p-value = 3.16 × 10⁻⁹).
- **Mechanistic ODE vs Black-Box Machine Learning Benchmark**: Operating on just 3 interpretable dynamic features (baseline phospho-ERK, `BRAF` inhibitor tumour burden, and anti-`PDCD1` checkpoint burden), the ODE digital twin achieves a 5-fold cross-validated ROC-AUC of **0.666** (± 0.074). This outperforms 12-feature Logistic Regression (0.646) and 12-feature Neural Networks (0.583), while approaching complex 12-feature Random Forests (0.686).


## Limitations & Future Directions

> [!WARNING] Methodological & Biological Boundaries
> Derived directly from the Phase 4 Limitations Audit (`phase_4_LIMITATIONS.md`):
> - **Subgroup Sample Size Deficit**: The *Immune Cold* phenotype represents a minor subset (45 patients, 6.4% of cohort), yielding wider confidence intervals around empirical response estimates compared to the dominant *Immune Hot* subgroup.
> - **Unadjusted Survival Testing**: Kaplan-Meier survival stratification relies on univariable log-rank testing; multivariate Cox Proportional Hazards regression is required to control for clinical covariates such as patient age, disease stage, and prior therapies.
> - **2-State Biophysical Simplification**: The ODE digital twin models tumour and immune effector cell dynamics accurately, but represents M2 macrophage stromal exclusion via parameter scale factors rather than explicit differential equations for Cancer-Associated Fibroblasts and M2 macrophages.
> - **Fixed Pharmacokinetic Exposure**: Trajectory simulations assume constant drug trough concentrations (500 nM Vemurafenib, 250 nM pembrolizumab) over 180 days, omitting peak-trough pharmacokinetic clearance dynamics.
> - **Lack of Acquired Resistance**: Trajectory models assume constant drug sensitivity over 180 days, omitting secondary `NRAS` / `MAP2K1` mutations or phenotype switching that can emerge during prolonged therapy.


## Key Takeaways

> [!INSIGHT] Summary Takeaways
> - Phase 4 successfully stratifies 699 melanoma patients into four biologically distinct phenotypes with unique dynamic therapeutic responses.
> - **Immune Hot** tumours achieve marked regression under anti-`PDCD1` monotherapy (Day 180 volume of 0.11) due to dense baseline cytotoxic T-cell infiltration.
> - **Immunosuppressive M2-High** tumours fail anti-`PDCD1` monotherapy (Day 180 volume of 0.91) due to stromal exclusion, but achieve effective regression (Day 180 volume of 0.46) when paired with M2-depleting rescue therapy.
> - **Mutant-Driven** (`NF1`-loss) tumours respond to immunotherapy due to high TMB, but require MEK inhibitors rather than `BRAF` inhibitors to avoid paradoxical RAF activation.
> - Operating on only 3 mechanistically derived features, the ODE digital twin achieves an ROC-AUC of 0.666, outperforming 12-feature machine learning baselines while maintaining complete biological interpretability.
