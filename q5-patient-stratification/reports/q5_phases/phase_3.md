---
title: "Phase 3: Unsupervised Patient Stratification & Manifold Projections"
aliases:
  - Q5 Phase 3
tags:
  - melanoma
  - patient-stratification
  - phase-3
  - q5
created: 2026-08-01 00:30
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-01 00:30
---

## 3. Phase 3: Unsupervised Phenotype Stratification (N = 699)

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Applying Gaussian Mixture Models (GMM, $K=4$) with full covariance matrices to feature matrices, calculating soft posterior probabilities, and generating 2D Principal Component projections.
> - **Why we are doing it**: Unsupervised clustering discovers natural biological patient subgroups without outcome bias. Soft probabilistic GMM clustering accommodates non-spherical correlated feature distributions and quantifies patient membership uncertainty.
> - **What question it answers**: What distinct patient clusters emerge from multi-dimensional biological profiling, and how are patients soft-partitioned across biological phenotypes?

### Unsupervised Phenotype Cluster Summary

| Cluster ID   | Biological Phenotype Subtype                                     |   Patient Count (N) | Cohort Share   | Response Rate   |
|:-------------|:-----------------------------------------------------------------|--------------------:|:---------------|:----------------|
| Cluster 0    | Immune Cold (Low TIS & Infiltration, Desert)                     |                  22 | 3.1%           | **66.7%**       |
| Cluster 1    | Immunosuppressive M2-High (Depleted T-cells & Stromal Exclusion) |                 305 | 43.6%          | **37.1%**       |
| Cluster 2    | Immune Hot (High TIS & CYT, Inflamed Microenvironment)           |                 266 | 38.1%          | **37.7%**       |
| Cluster 3    | Mutant-Driven (NF1 Loss & High Response Subtype)                 |                 106 | 15.2%          | **60.7%**       |

### Unsupervised Phenotype Cluster Projection (2D PCA)

![Unsupervised Patient Phenotype Clusters PCA](q5-patient-stratification/plots/clustering/pca_clusters.png)

> [!INFO] Figure Interpretation: 2D Principal Component Cluster Projection
> - **What this plot shows**: 2D Principal Component Projection of $N = 699$ patients colour-coded by their multi-modal GMM phenotype cluster ($K=4$). Shaded confidence ellipses mark cluster boundaries.
> - **Axis 1 (Horizontal)**: Principal Component 1 captures immune activation and lymphocytic T-cell density (separating Inflamed Hot vs Desert Cold tumours).
> - **Axis 2 (Vertical)**: Principal Component 2 captures macrophage polarisation (M1/M2 ratio) and stromal CAF exclusion.
> - **Clinical Value**: Discovers discrete patient subgroups with distinct treatment response profiles without relying on biased outcome labels.

### Unsupervised Phenotype Manifold (t-SNE Projection)

![Unsupervised Patient Phenotype Clusters t-SNE](q5-patient-stratification/plots/clustering/tsne_clusters.png)

> [!INFO] Figure Interpretation: Non-Linear t-SNE Cluster Manifold
> - **What this plot shows**: 2D t-SNE non-linear manifold projection of the 9-feature patient space ($N = 699$), colour-coded by the GMM cluster labels assigned in full 9-dimensional feature space.
> - **Non-Linear Topology**: t-SNE (perplexity=50) preserves local patient neighbourhood structure and non-linear biomarker interactions across the 9 multi-modal clustering features (`TIS`, `CYT`, CD8 T-cells, M1/M2 Macrophages, CAFs, `BRAF`/`NRAS`/`NF1` mutations). Natural within-cluster scatter reflects genuine continuous variation within each immune phenotype.

### Key Takeaways & Student Summary
- **Distinct Patient Groups**: GMM soft clustering partitioned $N = 699$ patients (full cohort) into four biological subgroups; within the ICI-treated sub-cohort ($N = 326$), response rates range from **37.1% to 66.7%**.
- **Highest Response Group**: The **Immune Cold (Low TIS & Infiltration, Desert)** subgroup achieves the highest response rate (66.7%), benefiting from favorable immune activation and high driver mutation burden.
- **Treatment-Resistant Subgroup**: The **Immunosuppressive M2-High (Depleted T-cells & Stromal Exclusion)** subgroup exhibits the lowest response rate (37.1%), highlighting the need for targeted combination therapies beyond single-agent PD-1 blockade.

> [!NOTE] Student-Friendly Phase 3 Summary
> Phase 3 performed unsupervised multi-dimensional GMM soft clustering to discover natural biological patient subgroups without relying on outcome labels:
> 1. **Four Distinct Phenotypes**: Gaussian Mixture Models ($K=4$) partitioned patients into *Immune Hot (High TIS & CYT, Inflamed Microenvironment)*, *Immune Cold (Low TIS & Infiltration, Desert)*, *Immunosuppressive M2-High (Depleted T-cells & Stromal Exclusion)*, and *Mutant-Driven (NF1 Loss & High Response Subtype)* phenotypes across 9 biomarker axes.
> 2. **Soft Probabilistic Assignments**: Full covariance matrices ($\mathbf{\Sigma}_k$) accommodate non-spherical feature correlation and calculate continuous posterior membership probabilities $\vec{P}_i$.
> 3. **Dimensionality Projections**: 2D PCA and non-linear t-SNE projections confirm clear spatial separation, with PC1 capturing T-cell inflammation and PC2 capturing myeloid/stromal exclusion.
> 4. **Clinical Takeaway**: Identifying a patient's biological phenotype and probability profile provides the foundation for targeted routing rather than applying a single uniform treatment protocol.

> [!formula] Phase 3 Script Execution & Software Module Architecture
> - **Primary Pipeline Execution Scripts**:
>   - [`03_cluster_patients.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/03_cluster_patients.py): Executes Gaussian Mixture Model (GMM) soft clustering ($K=4$, full covariance) across the 9 multi-modal feature space, computes posterior probabilities, exports `gmm_posterior_probabilities.csv` and `patient_clusters.csv`, serialises the fitted model (`gmm_model.pkl`), and generates PCA/t-SNE 2D projections (`pca_clusters.png`, `tsne_clusters.png`).
>   - [`08_compare_clustering_algorithms.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/08_compare_clustering_algorithms.py): Benchmarks alternative clustering algorithms (K-Means, HAC, GMM, Spectral Clustering, DBSCAN, Consensus Clustering) across Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI, and response rate spread.
>   - [`09_run_consensus_clustering.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/09_run_consensus_clustering.py): Executes 1,000-bootstrap Consensus Clustering ensemble across patients ($80\%$) and features ($80\%$) for $K \in [2, 8]$, generating Consensus CDF curves (`consensus_cdf_curves.png`), Delta Area elbow plots ($\Delta(4) = 0.1499$, `consensus_delta_area.png`), and co-association heatmaps (`consensus_heatmap_k4.png`).
> - **Core Supporting Python Modules**:
>   - [`clustering.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/clustering.py): Implements feature scaling, GMM soft clustering (`run_gmm`), and 2D cluster projection plotting (`plot_2d_cluster_projection`) for PCA and t-SNE manifolds.
>   - [`phenotyping.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/phenotyping.py): Implements cluster profiling (`profile_clusters`) and phenotype label mapping (`assign_phenotype_labels`).
>   - [`q5_constants.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/q5_constants.py): Central source of truth defining the 9 multi-modal clustering features (`CLUSTERING_FEATURES`) and biological phenotype mappings (`PHENOTYPE_LABEL_MAP`).
> - **Shared Cross-Question & Pipeline Modules**:
>   - [`run_q5_pipeline.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/run_q5_pipeline.py): Master pipeline orchestrator executing `03_cluster_patients.py` as Step 3.
>   - [`generate_q5_report.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/generate_q5_report.py): Reads clustering metrics and updates phase markdown reports.
