---
title: "Phase 3: Unsupervised Patient Stratification & Manifold Projections"
aliases:
  - Q5 Phase 3
tags:
  - melanoma
  - patient-stratification
  - phase-3
  - q5
created: 2026-08-01 13:58
cssclasses:
  - table-small
  - table-center
  - row-alt
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-01 13:58
---

## 3. Phase 3: Unsupervised Phenotype Stratification (N = 699)

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Applying Gaussian Mixture Models (GMM, $K=4$) with full covariance matrices to the 9-feature multi-modal immune and genomic feature space, calculating soft posterior probabilities, and generating 2D Principal Component and t-SNE manifold projections.
> - **Why we are doing it**: Unsupervised clustering discovers natural tumour microenvironment archetypes without outcome bias. Grounding phenotype discovery in the full cohort ($N = 699$, including TCGA-SKCM biological reference) ensures that the resulting phenotypes reflect the complete biological landscape rather than a trial-selected population.
> - **What question it answers**: What distinct tumour microenvironment phenotypes emerge from multi-dimensional immune and genomic profiling across $N = 699$ patients, and which therapeutic modality — `BRAF`/MEK targeted inhibition, immune checkpoint blockade, or combination strategies — does each phenotype indicate?

### Unsupervised Phenotype Cluster Summary

| Cluster ID   | Biological Phenotype Subtype                                     |   N (Total) | Cohort Share   | `BRAF` Mut   | `NRAS` Mut   | `NF1` Mut   | Therapeutic Routing Rationale                                                                                                                                                                                                                                                                    |
|:-------------|:-----------------------------------------------------------------|------------:|:---------------|:-------------|:-------------|:------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Cluster 0    | Immune Cold (Low TIS & Infiltration, Desert)                     |          22 | 3.1%           | 72.7%        | 77.3%        | 0.0%        | Desert/excluded TME: absent T-cell infiltration, low CYT, low TIS. High `BRAF` (72.7%) + `NRAS` (77.3%) co-mutation drives constitutive MAPK activation. Primary routing: **`BRAF`/MEK targeted inhibition**; ICI monotherapy unlikely to engage without prior immune priming.                   |
| Cluster 1    | Immunosuppressive M2-High (Depleted T-cells & Stromal Exclusion) |         305 | 43.6%          | 0.0%         | 47.9%        | 0.0%        | M2-polarised macrophages and CAF-mediated stromal exclusion block effector T-cell entry. `NRAS`-mutated (47.9%); no `BRAF` driver. Primary routing: **dual M2-depleting agent + checkpoint combination** to remodel the immunosuppressive stroma.                                                |
| Cluster 2    | Immune Hot (High TIS & CYT, Inflamed Microenvironment)           |         266 | 38.1%          | 100.0%       | 0.0%         | 0.0%        | Inflamed TME with high TIS and CYT, but 100% `BRAF`-mutated. MAPK oncogenic signalling counteracts T-cell activation (`TIS` $\times$ `BRAF` $\beta = -0.65$). Primary routing: **sequential `BRAF`/MEK inhibition → checkpoint therapy** to exploit both MAPK debulking and immune reactivation. |
| Cluster 3    | Mutant-Driven (NF1 Loss & RAS Hyperactivation, High TMB)         |         106 | 15.2%          | 38.7%        | 22.6%        | 100.0%      | `NF1` loss-of-function (100.0%) drives RAS hyperactivation with elevated TMB and neoantigen burden. Primary routing: **immune checkpoint blockade** leveraging high immunogenicity; MEK inhibition as adjunct for RAS pathway suppression.                                                       |

### Unsupervised Phenotype Cluster Projection (2D PCA)

![Unsupervised Patient Phenotype Clusters PCA](q5-patient-stratification/plots/clustering/pca_clusters.png)

> [!INFO] Figure Interpretation: 2D Principal Component Cluster Projection
> - **What this plot shows**: 2D Principal Component Projection of $N = 699$ patients colour-coded by their multi-modal GMM phenotype cluster ($K=4$). Shaded confidence ellipses mark cluster boundaries.
> - **Axis 1 (Horizontal)**: Principal Component 1 captures immune activation and lymphocytic T-cell density, separating Immune Hot (inflamed) from Immune Cold (desert/excluded) tumour microenvironments.
> - **Axis 2 (Vertical)**: Principal Component 2 captures myeloid polarisation and stromal architecture — separating M2-macrophage/CAF-excluded phenotypes from `NF1`-driven mutant phenotypes.
> - **Biological Value**: Confirms that the four GMM phenotypes occupy distinct regions of the biological feature space, validating that the clustering captures genuine TME archetypes rather than algorithmic artefacts.

### Unsupervised Phenotype Manifold (t-SNE Projection)

![Unsupervised Patient Phenotype Clusters t-SNE](q5-patient-stratification/plots/clustering/tsne_clusters.png)

> [!INFO] Figure Interpretation: Non-Linear t-SNE Cluster Manifold
> - **What this plot shows**: 2D t-SNE non-linear manifold projection of the 9-feature patient space ($N = 699$), colour-coded by the GMM cluster labels assigned in full 9-dimensional feature space.
> - **Non-Linear Topology**: t-SNE (perplexity=50) preserves local patient neighbourhood structure and non-linear biomarker interactions across the 9 multi-modal clustering features (`TIS`, `CYT`, CD8 T-cells, M1/M2 Macrophages, CAFs, `BRAF`/`NRAS`/`NF1` mutations). Natural within-cluster scatter reflects genuine continuous variation within each immune phenotype.

### Key Takeaways & Biological Insights
> [!INSIGHT] Key Insights: Phase 3 Unsupervised Phenotype Stratification
> - **Four Distinct TME Archetypes**: GMM soft clustering partitioned $N = 699$ patients into 4 tumour microenvironment phenotypes defined by T-cell infiltration, macrophage polarisation, stromal architecture, and oncogenic driver mutation signature — not by treatment outcome.
> - **Immune Activation Axis (PC1)**: Principal Component 1 separates *Immune Hot* (high TIS & CYT, inflamed) from *Immune Cold* (desert/excluded, absent T-cell infiltration) phenotypes — the primary axis of immunological responsiveness.
> - **Myeloid/Stromal Axis (PC2)**: Principal Component 2 separates *M2-High* (macrophage-polarised, CAF-excluded stroma) from *Mutant-Driven* (`NF1` loss, high TMB neoantigen load) phenotypes — the oncogenic and stromal axis.
> - **`BRAF`–Immunity Paradox**: The *Immune Hot* cluster is 100% `BRAF`-mutated — the most inflammatory TME is paradoxically driven by constitutive MAPK signalling. This creates a dual oncogenic–immune target amenable to sequential `BRAF`/MEK inhibition followed by checkpoint re-engagement.
> - **Treatment Routing Foundation**: These 4 phenotypes define the biological basis for precision therapeutic routing — `BRAF`/MEK targeted therapy, immune checkpoint blockade, or combination strategies — evaluated quantitatively in Phase 5.

> [!NOTE] Phase 3 Methodological Summary
> Phase 3 performed unsupervised multi-dimensional GMM soft clustering across the full $N = 699$ cohort to discover biological patient subgroups without outcome bias:
> 1. **Four Distinct Phenotypes**: Gaussian Mixture Models ($K=4$, full covariance) partitioned patients into *Immune Hot (High TIS & CYT, Inflamed Microenvironment)*, *Immune Cold (Low TIS & Infiltration, Desert)*, *Immunosuppressive M2-High (Depleted T-cells & Stromal Exclusion)*, and *Mutant-Driven (`NF1` Loss & High TMB)* phenotypes across 9 biomarker axes.
> 2. **Soft Probabilistic Assignments**: Full covariance matrices ($\mathbf{\Sigma}_k$) accommodate non-spherical feature correlation and compute continuous posterior membership probabilities $\vec{P}_i$ — quantifying biological uncertainty at patient level.
> 3. **Full-Cohort Grounding**: Clustering on $N = 699$ (including TCGA-SKCM biological reference) anchors phenotype definitions to the complete melanoma TME landscape rather than a trial-selected subset.
> 4. **Dimensionality Projections**: 2D PCA and non-linear t-SNE projections confirm clear spatial separation, validating that the four GMM phenotypes capture genuine TME archetypes.

> [!formula]+ Phase 3 Script Execution & Software Module Architecture
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
