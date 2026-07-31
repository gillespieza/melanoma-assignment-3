---
title: "Phase 3 Presentation Report: Unsupervised Patient Stratification & Phenotype Manifolds"
aliases:
  - Q5 Phase 3 Presentation Summary
  - Phase 3 Graduate Overview
tags:
  - melanoma
  - patient-stratification
  - phase-3
  - presentation
  - q5
created: 2026-07-31 14:20
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-31 14:20
---

# Phase 3: Unsupervised Patient Stratification & Phenotype Manifolds 🔬

A presentation guide evaluating Phase 3 unsupervised patient stratification in the Question 5 Melanoma Pipeline, detailing multi-modal biological clustering, clinical relevance, and key empirical findings across 699 patients.

## 1. Phase Overview

> [!NOTE] Analytical Summary & Rationale
> - **What is being done**: Grouping 699 melanoma patients into natural biological subgroups using multi-modal machine learning (Gaussian Mixture Models and 1,000-bootstrap consensus clustering) across 9 immunologic and genomic feature axes without outcome bias.
> - **Why we are doing it**: Patients respond differently to anti-PD-1 checkpoint immunotherapy. Rather than relying on single gene markers, unsupervised multi-dimensional stratification groups patients by shared tumour microenvironment biology.
> - **What question it answers**: What distinct biological phenotypes exist within advanced melanoma, how are patients soft-partitioned across these phenotypes, and how stable are these patient subgroups?

Phase 3 groups patients along 9 multi-dimensional biological traits, combining key inflammatory gene signatures (`TIS` Tumor Inflammation Score, `CYT` Cytolytic Activity), immune cell deconvolution estimates (CD8+ T cells, M1 and M2 Macrophages, Cancer-Associated Fibroblasts `CAFs`), driver mutations (`BRAF`, `NRAS`, `NF1`), and spatial microenvironment metrics.

Rather than forcing patients into rigid binary boxes, Phase 3 assigns continuous soft membership probabilities to reflect intermediate biological states.

## 2. Upstream Integration & Pipeline Architecture

Phase 3 acts as the central biological foundation of the Question 5 stratification engine, directly integrating discovery modules from Question 1 through Question 4:

* **Integration with Question 1 (Predictive Immune Signatures)**: Incorporates the 18-gene Tumor Inflammation Score (`TIS`), cytolytic granzyme/perforin index (`CYT`), and cell-type deconvolution metrics developed in Question 1 to quantify baseline microenvironmental inflammation.
* **Integration with Question 2 (Drug Combination Viability)**: Uses targeted driver mutation profiles (`BRAF V600`, `NRAS`, `NF1` loss) to link transcriptomic immune features with kinase inhibitor sensitivity.
* **Integration with Question 3 (Digital Twin ODE Dynamics)**: Provides baseline biological parameters (initial tumour burden, effector T-cell density, macrophage polarization) to initialize 180-day ordinary differential equation (ODE) simulation models for individual patient phenotypes.
* **Integration with Question 4 (Target Nomination)**: Maps distinct non-responsive phenotypes to DepMap essentiality targets and LINCS perturbagen recommendations (such as CSF1R inhibitors for M2-high tumours and MEK inhibitors for driver-mutated tumours).

## 3. Why This Matters for Patient Stratification (Q5)

> [!IMPORTANT] Clinical Relevance & Precision Medicine Rationale
> Applying a uniform single-agent checkpoint blockade protocol across all melanoma patients leads to non-response in over 50% of cases. Stratifying patients into distinct biological phenotypes enables:
> 1. **Avoiding One-Size-Fits-All Therapy**: Identifying which patients possess inflamed microenvironments versus stroma-excluded or driver-mutated barriers.
> 2. **Quantifying Classification Uncertainty**: Soft probabilistic GMM clustering provides continuous membership scores, alerting clinicians when a patient sits on the boundary between two biological phenotypes.
> 3. **Guiding Rational Combination Therapies**: Directing non-responsive phenotypes to targeted combination strategies (such as combining anti-PD-1 with stromal modifiers or kinase inhibitors).

## 4. Key Phase Results & Empirical Findings

### Cohort Phenotype Subtype Distribution (N = 699)

Across the full patient matrix (N = 699) and the immunotherapy-treated cohort (N = 326), unsupervised stratification identified four distinct biological subgroups:

| Biological Phenotype Subtype | Patient Count (N) | Cohort Share | Key Microenvironmental Profile | Response Rate (ICI Cohort) |
| :--- | :---: | :---: | :--- | :---: |
| **`Immunosuppressive M2-High`** | 305 | 43.6% | Depleted T-cell infiltrate, M2 macrophage dominance, and high CAF stromal exclusion walls. | **41.4%** |
| **`Immune Hot`** | 266 | 38.1% | High `TIS` and `CYT` signatures, dense CD8+ T-cell infiltration, inflamed microenvironment. | **37.1%** |
| **`Mutant-Driven`** | 106 | 15.2% | 100% `NF1` loss driver mutation subtype with moderate immune background. | **37.5%** |
| **`Immune Cold`** | 22 | 3.1% | Severe immune desert with uniformly low T-cell infiltration and low signature scores. | **66.7%** |

### Visualizing Patient Subgroups in 2D Space

To evaluate how cleanly patient groups separate, Phase 3 generates both linear and non-linear 2D projections:

![2D Principal Component Cluster Projection](q5-patient-stratification/plots/clustering/pca_clusters.png)

> [!INFO] Figure Interpretation: 2D PCA Cluster Projection
> - **Axis 1 (Horizontal)**: Principal Component 1 captures immune activation and lymphocytic T-cell density, separating inflamed hot tumours from cold deserts.
> - **Axis 2 (Vertical)**: Principal Component 2 captures macrophage polarization (M1 vs M2 balance) and stromal CAF exclusion.

![Non-Linear t-SNE Cluster Manifold](q5-patient-stratification/plots/clustering/tsne_clusters.png)

> [!INFO] Figure Interpretation: Non-Linear t-SNE Cluster Manifold
> - **Non-Linear Topology**: t-SNE preserves local patient neighbourhood relationships across all 9 multi-modal features. Natural within-cluster spread reflects genuine continuous variation rather than rigid artificial boundaries.

### Key Biological Takeaways
* **Dominance of Stromal Exclusion**: Data-driven probabilistic clustering reveals that **true immune deserts (Immune Cold, 3.1%) are rare** in advanced melanoma. Instead, **stromal exclusion (M2-High, 43.6%) is the primary non-hot phenotype**, where T cells are trapped by fibrotic fibroblast walls.
* **Continuous Biological Gradients**: Patient microenvironments form a continuous spectrum rather than isolated islands. Assigning soft probability scores allows precise monitoring of hybrid patient states.

## 5. Limitations & Future Directions

> [!WARNING] Summary of Methodological & Biological Limitations
> 1. **Bioinformatic Spatial Proxies**: Spatial distance ratios (`Spatial_CD8_CAF_Distance_Ratio`) are calculated from transcriptomic cell deconvolution estimates rather than direct micrometer measurements on physical tissue slides.
> 2. **High-Dimensional Parameter Scale**: Fitting full covariance matrices for 4 clusters across 9 features requires estimating 180 parameters. Expanding to 20+ features risks parameter inflation without covariance shrinkage.
> 3. **Asymmetric Subgroup Sizes**: True immune cold deserts comprise a small minority of cases (N = 22, 3.1%), requiring specialized sampling methods during predictive model training.

### Strategic Future Roadmap
* **Physical Spatial Slide Validation**: Validating transcriptomic distance ratios using physical 10x Visium, Xenium, or multiplex immunofluorescence (mIF) tissue slides.
* **1,000-Bootstrap Consensus Stability**: Validating that 4 clusters represents the optimal mathematical elbow (Delta Area score = 0.1499) across 1,000 randomized patient and feature resamplings.
* **Self-Tuning Graph Manifolds**: Implementing adaptive local distance metrics for Spectral Manifold Clustering to automatically capture complex non-linear tumour shapes.
