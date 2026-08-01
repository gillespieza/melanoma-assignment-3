---
title: "Phase 4: Phenotype Stratification and ODE Simulation"
aliases:
  - "Phase 4 Stratification"
  - "ODE Trajectory Simulation"
tags:
  - Phase4
  - PatientStratification
  - ODESimulation
  - Phenotypes
created: "2026-07-31 17:27"
updated: "2026-07-31 17:27"
cssclasses:
  - "table-small"
  - "table-center"
  - "row-alt"
obsidianEditingMode: "preview"
obsidianUIMode: "source"
---

## Phase Overview

> [!NOTE]
> **What is being done**: We synthesise multi-omic transcriptomic and genomic data to stratify melanoma patients into four biologically distinct phenotypes, simulating their longitudinal response to therapy using dynamic ordinary differential equations (ODEs).
> **Why we are doing it**: Static biomarkers cannot capture the temporal interplay between tumour growth, cytotoxic immune clearance, and therapeutic drug mechanisms, making it difficult to select optimal treatment regimens.
> **What question it answers**: How do distinct biological phenotypes dictate dynamic tumour regression or primary resistance under targeted and immune therapies over time?

Phase 4 bridges static biomarker characterisation and dynamic mathematical modelling. By grouping the pooled patient cohort (N=699) into four robust biological phenotypes, we map transcriptomic and mutational profiles to ODE parameters to simulate relative tumour volume trajectories over a 180-day treatment course. This dual-arm simulation framework evaluates therapeutic efficacy tailored to each phenotype's unique immune microenvironment and driver mutation profile.

## Upstream Integration

> [!INFO]
> **What is being done**: We integrate findings from upstream predictive modelling, drug viability profiling, ODE kinetic systems, and functional genomic screens into a unified patient stratification framework.
> **Why we are doing it**: A systems-level understanding of treatment response requires connecting single-gene biomarkers, microenvironmental cell states, and drug mechanisms into a comprehensive clinical decision pipeline.
> **What question it answers**: How do multi-omic biological signals identified across earlier research questions interact to govern overall patient response and treatment resistance?

This phase represents the core synthesis step of the project. It builds upon foundational biomarker signatures identified in Q1, such as `CD274`, `PDCD1`, and `CD8A` infiltration scores. We contextualise these signatures using cell-line drug sensitivity predictions from Q2 and CRISPR essentiality targets from Q4 (DepMap). Finally, these empirical profiles parameterise the Mechanistic ODE framework from Q3, which models `BRAF` inhibitor dynamics, the paradoxical RAF activation mechanism, and anti-`PDCD1` checkpoint blockade, creating a direct link between static baseline biology and dynamic therapeutic response.

## The Four Patient Phenotypes

> [!NOTE]
> **What is being done**: We categorise the pooled patient cohort (N=699) into four distinct clinical phenotypes based on soft Gaussian Mixture Model (GMM) clustering of mutational and microenvironmental profiles.
> **Why we are doing it**: Moving beyond single-biomarker thresholds allows us to define distinct biological subgroups with shared mechanisms of immune evasion and oncogenesis.
> **What question it answers**: What are the primary biological subsets within the melanoma patient population, and what are their defining clinical and molecular profiles?

| Phenotype | N (%) | Empirical Response Rate | Defining Molecular Biology | Therapeutic Signal & Vulnerability |
| :--- | :--- | :--- | :--- | :--- |
| **Immune Cold (Cluster 0)** | 22 (3.1%) | 50.0% | Deeply suppressed immune signatures; 90.9% `NRAS` mutant, 31.0% `BRAF` mutant | Primary resistance on both therapy arms; no baseline T cells for anti-`PDCD1` to unleash; `NRAS` mutation triggers RAF paradox under `BRAF` inhibition |
| **Mutant-Driven (Cluster 1)** | 65 (9.3%) | 68.8% | 100% `NF1` loss-of-function, 0% `BRAF V600E`, 24.0% `NRAS` mutant; high neoantigen burden | Strong response to immunotherapy due to high tumour mutational burden; MEK inhibition (`MAP2K1` / `MAPK1` targeted) is required rather than `BRAF` monotherapy |
| **Immune Hot (Cluster 2)** | 304 (43.5%) | 41.1% | 100% `BRAF V600` mutant; high baseline `CD8A`, `PRF1`, `GZMA`, and `CD274` expression | Highly responsive to anti-`PDCD1` checkpoint blockade; sensitive to `BRAF` inhibitor monotherapy due to oncogene addiction suppression |
| **M2-High (Cluster 3)** | 308 (44.1%) | 38.0% | High M2 macrophage (`CD163`, `ARG1`) and CAF (`TGFB1`) infiltration; 47.7% `NRAS` mutant | Refractory to anti-`PDCD1` monotherapy due to stromal exclusion; requires M2-depleting combination rescue therapy to enable T-cell clearance |

> **Note**: Total cohort sizes reflect the full GMM clustering (N=699). Dynamic ODE trajectory simulations use the patient subset matched to Q3 baseline parameter files: Immune Hot N=208, M2-High N=154, Mutant-Driven N=46, Immune Cold N=13.

## Dynamic ODE Trajectory Simulation

> [!INFO]
> **What is being done**: We simulate longitudinal relative tumour volume trajectories over 180 days across the four phenotypes under two distinct therapeutic modalities: Immunotherapy (Panel A) and Targeted Therapy (Panel B), visualised in a two-panel trajectory figure (ode_trajectories.png).
> **Why we are doing it**: To mechanistically demonstrate why specific phenotypes succeed or fail under monotherapy versus combination rescue treatment.
> **What question it answers**: Which therapeutic strategy achieves optimal, sustained tumour regression for each biological phenotype?

![Q3 ODE Tumour Trajectories](q5-patient-stratification/plots/phenotypes/ode_trajectories.png)

### Panel A: Immunotherapy (Anti-PD-1 Monotherapy vs M2 Combination Rescue)

Immunotherapy is modelled by unleashing CD8+ T-cell cytotoxic killing capacity through competitive inhibition of the `PDCD1` and `CD274` checkpoint binding complex.

*   **Immune Hot**: Demonstrates rapid and marked tumour regression, reaching a low relative tumour volume by Day 180 (final volume 0.14). The high baseline infiltration of `CD8A`+ T cells and cytolytic factors (`PRF1`, `GZMA`) is fully unblocked when anti-`PDCD1` therapy prevents checkpoint suppression.
*   **Immunosuppressive M2-High**: Fails anti-`PDCD1` monotherapy (dotted line), experiencing sustained high tumour burden (final volume 0.84). The dense microenvironmental barrier of M2 macrophages and cancer-associated fibroblasts physically excludes T cells from the tumour core. However, when simulated with an M2-depleting combination rescue agent (dashed line), stromal exclusion is breached, driving robust tumour regression (final volume 0.36).
*   **Mutant-Driven**: Shows moderate tumour regression under checkpoint blockade (final volume 0.65), supported by elevated baseline T-cell infiltration and high neoantigen burden resulting from `NF1` loss-of-function.
*   **Immune Cold**: Remains refractory to immunotherapy (final volume 0.91). Severe T-cell paucity means that unblocking the `PDCD1` checkpoint provides no cytotoxic clearance mechanism.

### Panel B: Targeted Therapy (BRAF Inhibitor Vemurafenib 500 nM)

Targeted therapy is modelled by directly suppressing intrinsic tumour cell proliferation through inhibition of the `MAPK1` / `MAP2K1` (pERK) signalling cascade.

*   **Immune Hot**: Achieves substantial tumour regression (final volume 0.41). Driven by 100% `BRAF V600` mutation prevalence, these tumours display strong oncogene addiction, making `BRAF` inhibition highly effective at suppressing pERK signalling.
*   **Mutant-Driven**: Exhibits partial resistance (final volume 0.79). Characterised by 100% `NF1` loss-of-function, these tumours exhibit elevated RAS-GTP levels. While residual RasGAP dampening provides partial suppression, direct `BRAF` monotherapy triggers paradoxical RAF dimerisation. The clinically optimal targeted strategy for this cluster is MEK inhibition (`MAPK1` targeting via Trametinib), which acts downstream of RAS.
*   **Immune Cold**: Shows primary resistance (final volume 0.81). A high `NRAS` mutation prevalence (90.9%) triggers the paradoxical RAF activation mechanism under `BRAF` inhibition, maintaining elevated pERK proliferation despite targeted treatment.
*   **M2-High**: Demonstrates maximal resistance (final volume 0.92). High `NRAS` mutation frequency (47.7%) combined with CAF-secreted growth factors (`HGF`, `FGF`) provides an additional proliferative drive independent of `BRAF` inhibition.

## Why This Matters for Patient Stratification (Q5)
Understanding the biological boundaries of each phenotype enables precise clinical decision-making:

> [!insight] Clinical Relevance
> 1.  **Immune Hot Patients**: Prime candidates for immediate anti-`PDCD1` monotherapy or standard `BRAF` / MEK inhibitor combination therapy. High baseline T-cell infiltration ensures robust immune clearance once checkpoint restraint is removed.
> 2.  **Immunosuppressive M2-High Patients**: Should not receive anti-`PDCD1` monotherapy alone. Stratification identifies the urgent need for front-line combination protocols pairing checkpoint inhibitors with M2 macrophage-depleting or CAF-targeting agents to breach stromal exclusion.
> 3.  **Mutant-Driven (`NF1`-Loss) Patients**: Benefit significantly from immunotherapy due to high neoantigen burden. When targeted therapy is required, treatment must utilise MEK inhibitors (Trametinib) rather than `BRAF` inhibitor monotherapy to avoid paradoxical ERK activation.
> 4.  **Immune Cold Patients**: Represent a critical unmet need. Primary resistance to both checkpoint blockade and `BRAF` inhibition necessitates enrollment in experimental immune-priming trials designed to recruit T cells into the tumour desert prior to checkpoint administration.

## Key Phase Outputs

| Artifact | Type | Clinical & Analytical Purpose |
| :--- | :--- | :--- |
| `phenotype_characterisation.csv` | Summary Data Table | Contains cluster-level biomarker means, mutation rates, response rates, and ODE parameter mappings across all N=699 patients. |
| `baseline_signature_boxplots.png` | Visualisation Plot | Displays Z-score distributions across `TIS`, `CYT`, `CD8A`, `M1_Macrophages`, `M2_Macrophages`, and CAFs for all four phenotypes. |
| `ode_trajectories.png` | Dual-Panel Visualisation Plot | Illustrates 180-day relative tumour volume trajectories under Immunotherapy (Panel A) and Targeted Therapy (Panel B). |
| `km_survival_by_phenotype.png` | Survival Analysis Plot | Kaplan-Meier overall survival curves evaluating empirical survival differences across the four GMM phenotype clusters. |

> [!warning] Limitations & Future Directions
> *   **Subgroup Sample Size Asymmetry**: The Immune Cold cluster (Cluster 0) contains only N=22 patients (3.1% of cohort, N=13 in the ODE subset), yielding wide confidence intervals around its empirical response rate (50.0%) and mean signature Z-scores.
> *   **Targeted Therapy Disconnect for `NF1`-Loss Tumours**: Mutant-Driven tumours (100% `NF1` loss-of-function, 0% `BRAF` V600E) are simulated under `BRAF` inhibitor treatment. In an `NF1`-loss context, elevated RAS-GTP levels trigger paradoxical RAF activation. The biologically appropriate targeted agent for this subtype is MEK inhibition (Trametinib), acting downstream of RAS.
> *   **Stromal Exclusion Approximation**: The 2-state ODE represents M2-High stromal exclusion via a reduced killing coefficient and scalar multiplier (`pheno_r_mult = 1.25`) rather than an explicit differential equation for Cancer-Associated Fibroblasts (CAFs) and M2 macrophages.
> *   **Targeted Rate Calibration & Compound Mismatch**: Proliferation suppression rates rely on literature parameters for Vemurafenib rather than being fitted directly to per-patient Dabrafenib or PLX-4720 viability AUC scores from Q2.
> *   **Lack of Acquired Resistance Modelling**: Trajectory simulations assume constant drug efficacy over 180 days, omitting secondary `NRAS` / `MAP2K1` mutations or phenotype switching that emerge during prolonged targeted or immune therapy.

> [!insight] Key Takeaways
> *   Phase 4 successfully stratifies N=699 melanoma patients into four biologically distinct phenotypes with unique therapeutic vulnerabilities.
> *   **Immune Hot** tumours achieve marked regression under anti-`PDCD1` immunotherapy (final volume 0.14) and `BRAF` inhibition (final volume 0.41) due to high T-cell infiltration and `BRAF V600E`  oncogene addiction.
> *   **M2-High** tumours demonstrate stromal T-cell exclusion, failing anti-`PDCD1` monotherapy (final volume 0.84) but achieving regression when combined with M2-depleting rescue therapy (final volume 0.36).
> *   **Mutant-Driven** tumours (100% `NF1` loss-of-function) respond to immunotherapy but require MEK inhibition rather than `BRAF` monotherapy to avoid paradoxical RAF activation.
> *   **Immune Cold** tumours exhibit dual resistance across both therapeutic arms, underscoring an urgent clinical need for novel immune-priming combination strategies.

