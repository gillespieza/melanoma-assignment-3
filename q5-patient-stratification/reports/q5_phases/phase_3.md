---
title: "Phase 3: Unsupervised Patient Stratification & Manifold Projections"
aliases:
  - Q5 Phase 3
tags:
  - melanoma
  - patient-stratification
  - phase-3
  - q5
created: 2026-08-01 17:43
cssclasses:
  - table-small
  - table-center
  - row-alt
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-01 17:43
---

## 3. Phase 3: Unsupervised Phenotype Stratification (N = 699)

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Executing Two-Stage Patient Stratification: Stage 1 GMM ($K=3$, full covariance) on 6 continuous immune/stromal features (`TIS`, `CYT`, `CD8_T_cells`, `M1_Macrophages`, `M2_Macrophages`, `CAFs`), followed by Stage 2 deterministic `NF1` split on the `NF1`-enriched immune cluster to carve out the `Mutant-Driven` phenotype. Exports continuous posterior probabilities and named columns (`P_Immune_Hot`, `P_Immune_Cold`, `P_Immunosuppressive_M2_High`, `P_Mutant_Driven`), generating 2D PCA, t-SNE, and UMAP projection maps.
> - **Why we are doing it**: Unsupervised clustering discovers natural tumour microenvironment archetypes without outcome bias. Grounding phenotype discovery in the full cohort ($N = 699$, including TCGA-SKCM biological reference) ensures that the resulting phenotypes reflect the complete biological landscape rather than a trial-selected population.
> - **What question it answers**: What distinct tumour microenvironment phenotypes emerge from two-stage immune and genomic profiling across $N = 699$ patients, and which therapeutic modality — `BRAF`/MEK targeted inhibition, immune checkpoint blockade, or combination strategies — does each phenotype indicate?

### Unsupervised Phenotype Cluster Summary

| Cluster ID   | Biological Phenotype Subtype                                     |   N (Total) | Cohort Share   | `BRAF` Mut   | `NRAS` Mut   | `NF1` Mut   | Therapeutic Routing Rationale                                                                                                                                                                                                                                                                                                         |
|:-------------|:-----------------------------------------------------------------|------------:|:---------------|:-------------|:-------------|:------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Cluster 0    | Immunosuppressive M2-High (Depleted T-cells & Stromal Exclusion) |         256 | 36.6%          | 44.1%        | 29.3%        | 0.0%        | Desert/excluded TME: depleted T-cell infiltration, low CYT, low TIS. `BRAF` (44.1%) and `NRAS` (29.3%) driver mutations present; `NF1` loss-of-function is 0.0% (reassigned to Cluster 3). Primary routing: **`BRAF`/MEK targeted inhibition** for `BRAF`+ patients; ICI monotherapy unlikely to engage without prior immune priming. |
| Cluster 1    | Immune Cold (Low TIS & Infiltration, Desert)                     |          45 | 6.4%           | 53.3%        | 17.8%        | 13.3%       | M2-polarised macrophages and CAF-mediated stromal exclusion block effector T-cell entry. `BRAF`-mutated (53.3%), `NRAS`-mutated (17.8%), `NF1`-mutated (13.3%). Primary routing: **dual M2-depleting agent + checkpoint combination** to remodel the immunosuppressive stroma.                                                        |
| Cluster 2    | Immune Hot (High TIS & CYT, Inflamed Microenvironment)           |         341 | 48.8%          | 49.6%        | 25.2%        | 12.6%       | Inflamed TME with high TIS and CYT. Heterogeneous oncogenic driver profile (`BRAF` 49.6%, `NRAS` 25.2%, `NF1` 12.6%). High baseline immunogenicity favors **primary immune checkpoint blockade**; `BRAF`/MEK inhibition reserved for `BRAF`+ sub-cohort.                                                                              |
| Cluster 3    | Mutant-Driven (NF1 Loss & RAS Hyperactivation, High TMB)         |          57 | 8.2%           | 29.8%        | 31.6%        | 100.0%      | `NF1` loss-of-function (100.0%) drives RAS hyperactivation with elevated TMB and neoantigen burden. Primary routing: **immune checkpoint blockade** leveraging high immunogenicity; MEK inhibition as adjunct for RAS pathway suppression.                                                                                            |

### Unsupervised Phenotype Cluster Projection (2D PCA)

![Unsupervised Patient Phenotype Clusters PCA](q5-patient-stratification/plots/clustering/pca_clusters.png)

> [!INFO] Figure Interpretation: 2D Principal Component Cluster Projection
> - **What this plot shows**: 2D Principal Component Projection of $N = 699$ patients colour-coded by their multi-modal GMM phenotype cluster ($K=4$). Shaded confidence ellipses mark cluster boundaries.
> - **Axis 1 (Horizontal)**: Principal Component 1 captures immune activation and lymphocytic T-cell density, separating Immune Hot (inflamed) from Immune Cold (desert/excluded) tumour microenvironments.
> - **Axis 2 (Vertical)**: Principal Component 2 captures myeloid polarisation and stromal architecture — separating M2-macrophage/CAF-excluded phenotypes from `NF1`-driven mutant phenotypes.
> - **Biological Value**: Confirms that the four GMM phenotypes occupy distinct regions of the biological feature space, validating that the clustering captures genuine TME archetypes rather than algorithmic artefacts.

### Unsupervised Phenotype Manifold (UMAP Projection)

![Unsupervised Patient Phenotype Clusters UMAP](q5-patient-stratification/plots/clustering/umap_clusters.png)

> [!INFO] Figure Interpretation: Non-Linear UMAP Cluster Manifold
> - **What this plot shows**: 2D UMAP non-linear manifold projection of the patient space ($N = 699$), colour-coded by the two-stage biological phenotype cluster labels.
> - **Non-Linear Topology**: UMAP preserves local patient neighbourhood structure and non-linear biomarker interactions across continuous immune microenvironment features (`TIS`, `CYT`, CD8 T-cells, M1/M2 Macrophages, CAFs) and driver mutation axes. Natural within-cluster scatter reflects genuine continuous variation within each immune phenotype.

### Tumour Mutational Burden (TMB) Across Biological Phenotypes

![TMB Distribution by Phenotype](q5-patient-stratification/plots/clustering/tmb_by_phenotype_comparison.png)

> [!INFO] Figure Interpretation: TMB Distribution & High-TMB Prevalence
> - **What this plot shows**: Violin and strip plot summary of nonsynonymous TMB across the 699-patient cohort stratified by biological phenotype.
> - **Mutant-Driven Dominance**: The *Mutant-Driven* (`NF1` Loss & RAS Hyperactivation) phenotype has a median TMB of $\approx 41$ mut/Mb — more than 3× higher than any other phenotype — consistent with replication-repair deficiency secondary to `NF1`/RAS pathway dysregulation.
> - **High-TMB Enrichment**: 87.7\% of *Mutant-Driven* patients exceed the $\geq 10$ mut/Mb FDA threshold vs 58.9\% in *Immune Hot*, 52.5\% in *M2-High*, and 48.8\% in *Immune Cold* clusters — confirming that neoantigen load is a phenotype-specific biological property.
> - **Biological Relevance**: High TMB generates immunogenic neoantigens that can engage adaptive immunity; however, the `NF1`-loss TME is not inherently inflamed (TIS is low-to-moderate), suggesting neoantigen presentation is suppressed — a context where combined checkpoint + TMB-directed therapeutic routing is biologically motivated.

### Key Takeaways & Biological Insights
> [!INSIGHT] Key Insights: Phase 3 Unsupervised Phenotype Stratification
> - **Four Distinct TME Archetypes**: Two-Stage Patient Stratification (Stage 1 GMM $K=3$ on 6 continuous immune features + Stage 2 deterministic `NF1` split) partitioned $N = 699$ patients into 4 tumour microenvironment phenotypes defined by T-cell infiltration, macrophage polarisation, stromal architecture, and driver mutation signature.
> - **Immune Activation Axis (PC1)**: Principal Component 1 separates *Immune Hot* (high TIS & CYT, inflamed) from *Immune Cold* (desert/excluded, absent T-cell infiltration) phenotypes — the primary axis of immunological responsiveness.
> - **Myeloid/Stromal Axis (PC2)**: Principal Component 2 separates *M2-High* (macrophage-polarised, CAF-excluded stroma) from *Mutant-Driven* (`NF1` loss, high TMB neoantigen load) phenotypes — the oncogenic and stromal axis.
> - **Multi-Driver Inflamed TME**: The *Immune Hot* cluster features diverse oncogenic driver mutations (49.6% `BRAF`+, 25.2% `NRAS`+, 12.6% `NF1`+) — demonstrating that robust TME inflammation (high TIS & CYT) develops across multiple driver mutation subtypes.
> - **Treatment Routing Foundation**: These 4 phenotypes define the biological basis for precision therapeutic routing — `BRAF`/MEK targeted therapy, immune checkpoint blockade, or combination strategies — evaluated quantitatively in Phase 5.

> [!NOTE] Phase 3 Methodological Summary
> Phase 3 performed two-stage patient stratification across the full $N = 699$ cohort to discover biological patient subgroups without outcome bias:
> 1. **Four Distinct Phenotypes**: Stage 1 GMM ($K=3$, full covariance) on 6 continuous immune features (`TIS`, `CYT`, `CD8_T_cells`, `M1_Macrophages`, `M2_Macrophages`, `CAFs`), followed by Stage 2 deterministic `NF1` split on the `NF1`-enriched cluster, partitioned patients into *Immune Hot*, *Immune Cold*, *Immunosuppressive M2-High*, and *Mutant-Driven* phenotypes.
> 2. **Soft Probabilistic Assignments**: Stage 1 GMM full covariance matrices ($\mathbf{\Sigma}_k$) compute continuous posterior membership probabilities $\vec{P}_i$ — quantifying biological uncertainty at cluster boundaries.
> 3. **Full-Cohort Grounding**: Clustering on $N = 699$ (including TCGA-SKCM biological reference) anchors phenotype definitions to the complete melanoma TME landscape rather than a trial-selected subset.
> 4. **Dimensionality Projections**: 2D PCA, t-SNE, and non-linear UMAP projections confirm clear spatial separation, validating that the four phenotypes capture genuine TME archetypes.

> [!formula]+ Phase 3 Script Execution & Software Module Architecture
> - **Primary Pipeline Execution Scripts**:
>   - [`03_cluster_patients.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/03_cluster_patients.py): Executes Two-Stage Patient Stratification (Stage 1 GMM $K=3$ continuous immune features + Stage 2 deterministic `NF1` split), exports `gmm_posterior_probabilities.csv` and `patient_clusters.csv` with runtime-mapped named probability columns (`P_Immune_Hot`, `P_Immune_Cold`, `P_Immunosuppressive_M2_High`, `P_Mutant_Driven`), serialises the fitted model (`gmm_model.pkl`), and generates 2D PCA, t-SNE, and UMAP projections (`pca_clusters.png`, `tsne_clusters.png`, `umap_clusters.png`).
>   - [`08_compare_clustering_algorithms.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/08_compare_clustering_algorithms.py): Benchmarks alternative clustering algorithms (K-Means, HAC, GMM, Spectral Clustering, DBSCAN, Consensus Clustering) across Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI, and response rate spread.
>   - [`09_run_consensus_clustering.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/09_run_consensus_clustering.py): Executes 1,000-bootstrap Consensus Clustering ensemble across patients ($80\%$) and features ($80\%$) for $K \in [2, 8]$, generating Consensus CDF curves (`consensus_cdf_curves.png`), Delta Area elbow plots (`consensus_delta_area.png`), and co-association heatmaps (`consensus_heatmap_k4.png`).
> - **Core Supporting Python Modules**:
>   - [`clustering.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/clustering.py): Implements feature scaling, Stage 1 GMM soft clustering (`run_gmm`), and 2D cluster projection plotting (`plot_2d_cluster_projection`) for PCA, t-SNE, and UMAP manifolds.
>   - [`phenotyping.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/phenotyping.py): Implements cluster profiling (`profile_clusters`) and phenotype label mapping (`assign_phenotype_labels`).
>   - [`q5_constants.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/q5_constants.py): Central source of truth defining Stage 1 GMM continuous features (`GMM_CONTINUOUS_FEATURES`) and biological phenotype mappings (`PHENOTYPE_LABEL_MAP`).
> - **Shared Cross-Question & Pipeline Modules**:
>   - [`run_q5_pipeline.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/run_q5_pipeline.py): Master pipeline orchestrator executing `03_cluster_patients.py` as Step 3.
>   - [`generate_q5_report.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/generate_q5_report.py): Reads clustering metrics and generates phase markdown reports.
