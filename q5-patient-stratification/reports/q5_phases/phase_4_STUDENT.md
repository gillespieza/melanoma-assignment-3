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
created: "2026-07-31 16:09"
updated: "2026-07-31 16:09"
cssclasses:
  - "table-small"
  - "table-center"
  - "row-alt"
obsidianEditingMode: "preview"
obsidianUIMode: "source"
---

## Phase Overview

> [!NOTE] 
> **What is being done**: We are synthesising transcriptomic and genomic data to classify melanoma patients into distinct biological phenotypes and simulating their longitudinal response to therapy using ordinary differential equations (ODEs).
> **Why we are doing it**: Static biomarkers often fail to capture the dynamic interplay between the tumour, the immune system, and therapeutic interventions, making it difficult to predict clinical outcomes reliably.
> **What question it answers**: How do underlying biological phenotypes dictate dynamic tumour regression or resistance under different targeted and immune therapies?

Phase 4 bridges static multi-omic characterisation and dynamic mathematical modelling by classifying the pooled patient cohort (N=699) into four distinct biological phenotypes. By mapping these transcriptomic and mutational profiles to ODE initial conditions, we simulate tumour volume trajectories over time. This approach evaluates the efficacy of specific therapeutic interventions tailored to each phenotype's unique immune and oncogenic state.

## Upstream Integration

> [!INFO]
> **What is being done**: We integrate upstream findings from biomarker discovery, drug sensitivity profiling, and targeted pathway analysis into a cohesive predictive framework.
> **Why we are doing it**: A systems-level understanding of patient response requires linking isolated insights—such as single-gene mutations and immune signatures—into a unified model.
> **What question it answers**: How do the disparate biological signals identified in earlier phases interact to drive overall patient response?

This phase represents the culmination of the analytical pipeline. It builds upon the foundational biomarker signatures established in Q1, incorporating key genes such as `CD274` and `CD8A`. We contextualise these signatures using the drug sensitivity profiles (Q2) and specific genetic vulnerabilities identified in the DepMap screening (Q4). Finally, the phenotypes parameterise the advanced ODE framework (Q3), which explicitly models both `BRAF`/`MAP2K1` inhibitor dynamics and anti-`PDCD1` immune checkpoint blockade, providing a mechanistic link between static biology and dynamic therapeutic response.

## The Four Patient Phenotypes

> [!NOTE]
> **What is being done**: We categorise the patient cohort (N=699) into four distinct clinical phenotypes based on clustering of mutational and transcriptomic profiles.
> **Why we are doing it**: To move beyond single-biomarker stratification and define robust biological subgroups that reflect distinct mechanisms of immune evasion and oncogenesis.
> **What question it answers**: What are the primary biological subsets within the melanoma patient population, and what are their defining characteristics?

| Phenotype | N (%) | Response Rate | Defining Biology | Therapy Signal |
| :--- | :--- | :--- | :--- | :--- |
| **Immune Cold (Cluster 0)** | 22 (3.1%) | 50.0% | `BRAF` (86.4%), `NRAS` (90.9%) mutant, zero immune infiltration | Primary resistance to monotherapies; requires priming |
| **Mutant-Driven (Cluster 1)** | 65 (9.3%) | 68.8% | 100% `NF1` mutant, strong oncogene addiction | Sensitive to targeted pathway inhibition |
| **Immune Hot (Cluster 2)** | 304 (43.5%) | 41.1% | 100% `BRAF` mutant, high `CD8A`, `PRF1`, `GZMA` | Strong response to immune checkpoint blockade |
| **M2-High (Cluster 3)** | 308 (44.1%) | 38.0% | 47.7% `NRAS` mutant, high `CD163`, `ARG1`, `TGFB1` | Stromal exclusion of T cells; requires combination therapy |

## Two Therapy Arms: ODE Trajectory Simulation

> [!INFO]
> **What is being done**: We simulate dynamic tumour volume over time across the four phenotypes under two therapeutic modalities: Immunotherapy and Targeted Therapy, visualised in a two-panel trajectory figure (ode_trajectories.png).
> **Why we are doing it**: To mechanically validate why certain phenotypes respond to specific treatments by tracing the modelled interactions between tumour cells, immune effectors, and the drug mechanism.
> **What question it answers**: Which therapeutic modality is optimally matched to each biological phenotype to achieve sustained tumour regression?

### Panel A: Immunotherapy (Anti-PD-1 + M2 Combination)

Immunotherapy acts mathematically by increasing the T-cell kill gate (f_kill), unleashing the cytotoxic potential of existing immune cells. 
*   **Immune Hot**: Responds optimally to immunotherapy because the tumour microenvironment is already rich in `CD8A`+ T cells and cytolytic factors (`PRF1`, `GZMA`), but restrained by high `CD274` (`PDCD1` ligand) expression. Unleashing the checkpoint releases this pre-existing killing machinery.
*   **Immunosuppressive M2-High**: Fails anti-`PDCD1` monotherapy due to a dense stromal barrier of cancer-associated fibroblasts and M2 macrophages (marked by `CD163`, `ARG1`, `TGFB1`) that physically excludes T cells. However, when simulated with an M2-depleting combination rescue, the barrier is breached, allowing anti-PD-1 to effectively clear the tumour.

### Panel B: Targeted Therapy (BRAF/MEK Inhibitor)

Targeted therapy is modelled to act by directly reducing the intrinsic proliferation rate (r) of oncogene-dependent tumour cells via suppression of the `MAPK1`/`MAP2K1` (pERK) pathway.
*   **Mutant-Driven**: Achieves near-complete regression. The defining `BRAF` and `NF1` mutations drive profound oncogene-addiction. Inhibiting this pathway directly suppresses proliferation (resulting in a very low proliferation rate r=0.04), causing rapid tumour collapse.
*   **Immune Cold**: Presents the greatest clinical challenge. The absence of T cells means anti-`PDCD1` has no immune response to unleash. Furthermore, because these tumours lack the specific oncogenic drivers targeted by `BRAF` inhibitors, targeted therapy is also ineffective (resulting in no change in the proliferation rate r), demonstrating primary resistance. These patients require novel priming strategies.

## Why This Matters for Patient Stratification

> [!NOTE]
> **What is being done**: We map our simulated and empirical findings directly to clinical decision-making strategies.
> **Why we are doing it**: To translate complex systems biology models into actionable guidance for selecting first-line and combination therapies.
> **What question it answers**: How can we use these four phenotypes to select the right drug for the right patient in the clinic?

Understanding the biological constraints of each phenotype allows for precise treatment matching. Immune Hot patients are prime candidates for immediate checkpoint blockade, while Mutant-Driven patients benefit heavily from targeted kinase inhibitors. Recognising the M2-High phenotype prevents futile monotherapy treatment, highlighting the absolute necessity for combination trials. Finally, identifying the Immune Cold phenotype early avoids exposing patients to toxicities from ineffective standard treatments, directing them instead towards experimental immune-priming clinical trials.

## Key Phase Outputs

> [!INFO]
> **What is being done**: We document the primary analytical artifacts generated by Phase 4.
> **Why we are doing it**: To maintain a clear inventory of generated models, tables, and visualisations for reporting and verification.
> **What question it answers**: What specific deliverables does Phase 4 produce for the final project report?

| Output File | Description | Purpose |
| :--- | :--- | :--- |
| phenotype_characterisation.csv | Empirical metadata for all 699 patients | Defines the exact size, mutation frequencies, and response rates of the four clusters. |
| ode_trajectories.png | Two-panel longitudinal simulation figure | Visually demonstrates tumour regression vs resistance under the two therapeutic arms. |
| cluster_biomarkers.csv | Differential expression results per cluster | Identifies the key genes (e.g., `CD8A`, `CD163`) defining each phenotype's biology. |

## Limitations

> [!NOTE]
> **What is being done**: We briefly acknowledge the constraints of the current ODE model and stratification approach.
> **Why we are doing it**: To demonstrate scientific rigour and contextualise the certainty of our simulated predictions.
> **What question it answers**: What are the boundaries of our current analytical framework?

While the ODE framework provides powerful mechanistic insights, it relies on aggregate parameters derived from bulk RNA sequencing, which cannot fully capture the spatial heterogeneity of the tumour microenvironment. Furthermore, the model assumes uniform drug penetration across all phenotypes. A complete discussion of model assumptions, spatial limitations, and sensitivity analysis is provided in phase_4_LIMITATIONS.md.

## Key Takeaways

*   The analytical pipeline successfully stratifies N=699 patients into four distinct phenotypes with unique therapeutic vulnerabilities.
*   The **Immune Hot** phenotype leverages high intrinsic T-cell infiltration, resulting in optimal responses to anti-`PDCD1` immunotherapy.
*   The **Mutant-Driven** phenotype is characterised by oncogene addiction, making it highly sensitive to targeted therapy that directly suppresses proliferation.
*   The **M2-High** phenotype demonstrates stromal T-cell exclusion, failing monotherapy but responding to simulated M2-depleting combination therapies.
*   The **Immune Cold** phenotype lacks both immune infiltration and targetable drivers, highlighting a critical unmet need for novel immune-priming strategies.
