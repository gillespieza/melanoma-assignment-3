---
title: "Phase 3: Unsupervised Patient Stratification & Manifold Projections"
aliases:
  - Q5 Phase 3
tags:
  - melanoma
  - patient-stratification
  - phase-3
  - q5
created: 2026-07-30 18:04
cssclasses:
  - table-small
  - table-center
  - row-alt
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-30 18:04
---

## 3. Phase 3: Unsupervised Phenotype Stratification (N = 699)

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Applying K-Means clustering ($K=4$) to feature matrices and generating 2D Principal Component projections.
> - **Why we are doing it**: Unsupervised clustering discovers natural biological patient subgroups without outcome bias. Embedding visual projections is preferred over raw tables for slide presentation.
> - **What question it answers**: What distinct patient clusters emerge from multi-dimensional biological profiling?

### Unsupervised Phenotype Cluster Summary

| Cluster ID   | Biological Phenotype Subtype                                       |   Patient Count (N) | Cohort Share   | Response Rate   |
|:-------------|:-------------------------------------------------------------------|--------------------:|:---------------|:----------------|
| Cluster 0    | `Mutant-Driven (NF1 Loss & High Response Subtype)`                 |                  89 | 12.7%          | **61.5%**       |
| Cluster 1    | `Immune Cold (Low TIS & Infiltration, Desert)`                     |                 198 | 28.3%          | **35.4%**       |
| Cluster 2    | `Immune Hot (High TIS & CYT, Inflamed Microenvironment)`           |                 289 | 41.3%          | **38.9%**       |
| Cluster 3    | `Immunosuppressive M2-High (Depleted T-cells & Stromal Exclusion)` |                 123 | 17.6%          | **45.2%**       |

### Unsupervised Phenotype Cluster Projection (2D PCA)

![Unsupervised Patient Phenotype Clusters PCA](q5-patient-stratification/plots/clustering/pca_clusters.png)

> [!INFO] Figure Interpretation: 2D Principal Component Cluster Projection
> - **What this plot shows**: 2D Principal Component Projection of $N = 699$ patients colour-coded by their multi-modal K-Means phenotype cluster ($K=4$). Shaded confidence ellipses mark cluster boundaries.
> - **Axis 1 (Horizontal)**: Principal Component 1 captures immune activation and lymphocytic T-cell density (separating Inflamed Hot vs Desert Cold tumours).
> - **Axis 2 (Vertical)**: Principal Component 2 captures macrophage polarisation (M1/M2 ratio) and stromal CAF exclusion.
> - **Clinical Value**: Discovers discrete patient subgroups with distinct treatment response profiles without relying on biased outcome labels.

### Unsupervised Phenotype Manifold (t-SNE Projection)

![Unsupervised Patient Phenotype Clusters t-SNE](q5-patient-stratification/plots/clustering/umap_clusters.png)

> [!INFO] Figure Interpretation: Non-Linear t-SNE Cluster Manifold
> - **What this plot shows**: 2D t-SNE non-linear manifold projection of the 9-feature patient space ($N = 699$), colour-coded by the K-Means cluster labels assigned in full 9-dimensional feature space.
> - **Non-Linear Topology**: t-SNE (perplexity=50) preserves local patient neighbourhood structure and non-linear biomarker interactions across the 9 multi-modal clustering features (`TIS`, `CYT`, CD8 T-cells, M1/M2 Macrophages, CAFs, `BRAF`/`NRAS`/`NF1` mutations). Natural within-cluster scatter reflects genuine continuous variation within each immune phenotype.

### Key Takeaways & Student Summary
- **Distinct Patient Groups**: K-Means clustering partitioned $N = 699$ patients (full cohort) into four biological subgroups; within the ICI-treated sub-cohort ($N = 326$), response rates range from **35.4% to 61.5%**.
- **Highest Response Group**: The **Mutant-Driven (NF1 Loss & High Response Subtype)** subgroup achieves the highest response rate (61.5%), benefiting from favorable immune activation and high driver mutation burden.
- **Treatment-Resistant Subgroup**: The **Immune Cold (Low TIS & Infiltration, Desert)** subgroup exhibits the lowest response rate (35.4%), highlighting the need for targeted combination therapies beyond single-agent PD-1 blockade.

> [!NOTE] Student-Friendly Phase 3 Summary
> Phase 3 performed unsupervised multi-dimensional clustering to discover natural biological patient subgroups without relying on outcome labels:
> 1. **Four Distinct Phenotypes**: K-Means clustering ($K=4$) partitioned patients into *Immune Hot (High TIS & CYT, Inflamed Microenvironment)*, *Immune Cold (Low TIS & Infiltration, Desert)*, *Immunosuppressive M2-High (Depleted T-cells & Stromal Exclusion)*, and *Mutant-Driven (NF1 Loss & High Response Subtype)* phenotypes across 9 biomarker axes.
> 2. **Wide Response Rate Divergence**: Clinical response rates varied markedly across clusters, demonstrating that unselected cohort averages mask distinct biological subgroups.
> 3. **Dimensionality Projections**: 2D PCA and non-linear t-SNE projections confirm clear spatial separation, with PC1 capturing T-cell inflammation and PC2 capturing myeloid/stromal exclusion.
> 4. **Clinical Takeaway**: Identifying a patient's biological phenotype provides the foundation for targeted routing rather than applying a single uniform treatment protocol.
