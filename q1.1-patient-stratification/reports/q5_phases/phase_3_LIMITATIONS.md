---
title: "Phase 3 Limitations & Methodological Evaluation: Unsupervised Patient Stratification"
aliases:
  - Q5 Phase 3 Limitations
  - Phase 3 Methodological Audit
tags:
  - melanoma
  - patient-stratification
  - phase-3
  - limitations
  - q5
created: 2026-08-01 16:06
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-01 16:06
---

# Phase 3 Methodological Limitations & Critical Evaluation

> [!NOTE] Evaluation Scope & Objectives
> - **What is being evaluated**: Comprehensive methodological audit of Phase 3 Two-Stage Patient Stratification (Stage 1 GMM $K=3$ on 6 continuous immune features + Stage 2 deterministic `NF1` split).
> - **Why we are doing it**: Rigorous identification of remaining statistical, biological, architectural, and data limitations without relying on outcome labels or discussing past refactoring iterations.
> - **What question it answers**: What fundamental statistical uncertainties, biological assumptions, and data constraints bound the generalisability of the four discovered TME phenotypes ($N = 699$), and how should future pipeline iterations refine stratification accuracy?

> [!INSIGHT] Key Takeaways & Evaluation Summary
> - **Moderate Cluster Separation**: Stage 1 GMM achieves modest internal cluster separation (Silhouette = $0.2043$, Calinski-Harabasz = $229.9$, Davies-Bouldin = $1.7377$), reflecting genuine continuous transitions across tumour microenvironment archetypes rather than discrete isolated clusters.
> - **Spectral Benchmark Superiority**: Spectral Manifold clustering achieved superior geometric separation (Silhouette = $0.2737$, Calinski-Harabasz = $463.7$, Davies-Bouldin = $1.1209$), indicating non-linear Graph Laplacian manifold structure that Gaussian ellipsoids only partially capture.
> - **Heuristic Two-Stage NF1 Coupling**: Stage 2 carves out Cluster 3 (`Mutant-Driven`, $N = 57$) via a hard deterministic `NF1` split from Cluster 0 (`Immunosuppressive M2-High`), introducing a hybrid continuous-discrete boundary where `P_Mutant_Driven` is zeroed for `NF1`-negative patients rather than modelled via joint probabilistic estimation.
> - **Bulk Deconvolution & Spatial Proxy Limits**: Microenvironment characterisation relies on bulk RNA-seq deconvolution and two-cell spatial distance ratios, omitting high-multiplex spatial protein imaging (e.g. CODEX/IMC) and single-cell resolution required to resolve cellular contact networks.

## 1. Code-Quality & Architectural Evaluation

> [!WARNING] Architectural Limitations & Pipeline Coupling
> - **Modular Pipeline Decoupling**: Phase 3 exports `patient_clusters.csv` and `gmm_posterior_probabilities.csv` with named columns (`P_Immune_Hot`, `P_Immune_Cold`, `P_Immunosuppressive_M2_High`, `P_Mutant_Driven`). However, downstream scripts (Phase 5 subgroup models, Phase 6 clinical utility) maintain direct dependencies on these specific column keys, creating tight schema coupling across pipeline stages.
> - **Model Persistence Scope**: The persisted GMM artefact (`gmm_model.pkl`) serialises only the Stage 1 $K=3$ Gaussian mixture model. The Stage 2 deterministic `NF1` split logic is not encapsulated inside a unified scikit-learn `Pipeline` or `Transformer` object, requiring custom wrapper logic to predict on unseen validation samples.
> - **Global State Resolution**: Dynamic label-to-index mapping resolves GMM component reordering at runtime, but relies on string matching against `stage1_short_labels`. If feature distributions shift significantly in small external datasets, profile-based phenotype assignment could misclassify cluster identities.

## 2. Statistical & Algorithmic Weaknesses

> [!WARNING] Statistical Vulnerabilities & Clustering Quality
> - **Sub-optimal Convex Clustering**: Internal validation metrics confirm that standard GMM (Silhouette = $0.2043$, Calinski-Harabasz = $229.9$, Davies-Bouldin = $1.7377$) yields lower spatial separation than Spectral Manifold clustering (Silhouette = $0.2737$, Calinski-Harabasz = $463.7$). GMM assumes multi-variate Gaussian components with full covariance matrices, which struggle to model non-convex, ribbon-like manifold structures in transcriptomic feature space.
> - **Mahalanobis Space Degeneracy**: Transforming feature space via regularised inverse covariance ($\mathbf{X}_{\text{Mahalanobis}} = \mathbf{X}_{\text{scaled}} \mathbf{\Sigma}^{-1/2}$) reduced clustering quality (Silhouette = $0.2025$, Calinski-Harabasz = $49.8$, Davies-Bouldin = $2.3818$, Log-Likelihood = $-7.9196$). Linear decorrelation distorts density gradients between `Immune Hot` ($N = 341$, $48.8\%$) and `Immunosuppressive M2-High` ($N = 256$, $36.6\%$).
> - **Posterior Boundary Overlap**: Mean Stage 1 posterior probabilities across all four phenotypes average $\bar{P} \approx 0.840 - 0.849$, indicating that $\approx 15-18\%$ of patients reside in ambiguous boundary regions ($P < 0.70$), representing intermediate transition states rather than distinct biological subtypes.

## 3. Biological Assumptions & Domain Constraints

> [!WARNING] Biological Simplifications & Phenotype Definitions
> - **Disjoint Stage 2 Genomic Split**: Excluding binary mutation indicators (`mut_BRAF`, `mut_NRAS`, `mut_NF1`) from Stage 1 prevents covariance matrix collapse, but forces `Mutant-Driven` phenotype assignment ($N = 57$, $8.2\%$) to occur via a post-hoc deterministic split on `NF1` loss-of-function ($100.0\%$). This assumes `NF1` loss acts as a binary switch rather than interacting continuously with immune microenvironment density.
> - **Bulk RNA-seq Deconvolution Limits**: Macrophage polarisation (`M1_Macrophages`, `M2_Macrophages`) and cancer-associated fibroblast (`CAFs`) estimates derive from signature transcript vectors (STV) and cell deconvolution algorithms. Bulk deconvolution averages expression across heterogeneous tissue regions, obscuring localized cell-cell interactions.
> - **Simplified Spatial Proxies**: Spatial microenvironment indicators (`Spatial_CD8_CAF_Distance_Ratio`, `Spatial_Tumour_Infiltration_Index`) are inferred from transcriptomic proxy models rather than direct multiplexed spatial proteomics, limiting structural microenvironment interpretation.

## 4. Computational & Data Constraints

> [!WARNING] Sample Size Disparities & Cohort Heterogeneity
> - **Full Cohort vs ICI Subset Disparity**: Clustering is anchored on the full cohort ($N = 699$, including TCGA-SKCM biological reference), but downstream clinical utility (Phase 6) and ICI-specific response evaluation are constrained to the ICI-treated subset ($N = 326$). On the ICI subset, GMM log-likelihood drops to $-4.1582$ (AIC = $2877.1$, BIC = $3191.5$), reflecting sample size contraction.
> - **Class Imbalance in Phenotype Sizes**: Subgroup sample sizes range from $N = 341$ ($48.8\%$, `Immune Hot`) down to $N = 45$ ($6.4\%$, `Immune Cold`) and $N = 57$ ($8.2\%$, `Mutant-Driven`). This imbalance reduces statistical power for subgroup-specific predictive modelling in Phase 5.
> - **Missing Neoantigen & Genomic Features**: Genomic mutation profiles rely on binary flags without incorporating continuous mutation copy-number alterations, allele frequencies (VAF), or structural variants across all cohorts.

## 5. Prioritised Actionable Fixes & Future Directions

> [!TIP] Recommendations for Future Pipeline Iterations
> 1. **High Priority — Unified Custom Estimator Class**: Encapsulate Stage 1 GMM and Stage 2 `NF1` split into a custom scikit-learn `BaseEstimator` class (`TwoStageGMMPhenotyper`) implementing `.fit()` and `.transform()` to enable seamless model persistence and external cross-validation without custom export scripts.
> 2. **High Priority — Variational Bayesian GMM or Graph Neural Networks**: Replace fixed $K=3$ GMM with Variational Bayesian Inference (DPGMM) or Graph Neural Network (GNN) embeddings to learn non-convex manifold topologies and dynamic cluster counts automatically.
> 3. **Medium Priority — Soft Probabilistic NF1 Integration**: Replace the binary Stage 2 split with a joint Bayesian probabilistic model incorporating continuous `NF1` variant allele fraction (VAF) and TMB load directly into posterior probability estimation.
> 4. **Medium Priority — Multiplexed Spatial Proteomics Integration**: Integrate single-cell spatial proteomics (CODEX/MIBI) data to validate whether transcriptomic spatial proxy indicators accurately reflect physical cell-cell contact distances between CD8+ T-cells and M2 CAFs.
