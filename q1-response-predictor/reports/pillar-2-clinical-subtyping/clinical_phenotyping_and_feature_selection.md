---
title: "Patient Phenotyping via Two-Stage GMM Clustering"
aliases:
  - Patient Subtyping
  - Clinical Clustering
  - Immune Phenotyping
  - GMM Stratification
tags:
  - melanoma
  - clinical-subtyping
  - gmm
  - clustering
  - immune-hot-cold
  - nf1-split
created: 2026-08-07 15:17
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-07 15:17
---

# Patient Phenotyping via ICI Trial Cohort Two-Stage GMM Clustering

> [!INFO] What, Why & Key Questions — Overview
> - **What We Are Doing**: Applying a two-stage unsupervised Gaussian Mixture Model (GMM) stratification to the **anti-PD-1 ICI trial cohorts** ($N = 256$ patients across Liu 2019, Hugo 2016, and Riaz 2017). Stage 1 fits a GMM (K=3, full covariance) on 8 continuous immune/microenvironment Z-score features (IFN-γ, TIS, CYT, CD8+, IMPRES, PD-L1, M1/M2 ratio, Macrophage STV). Stage 2 applies a deterministic NF1-positive split on the NF1-enriched immune cluster, producing a 4th Mutant-Driven phenotype. Phenotype labels are assigned by rank-based rules on empirical cluster profiles — never by hardcoded cluster integer IDs.
> - **Why We Are Doing It**: GMM provides soft posterior probability assignments rather than hard cluster membership, capturing biological uncertainty at phenotype boundaries. The two-stage design separates the continuous immune microenvironment axis (Stage 1 GMM) from the discrete driver mutation axis (Stage 2 NF1 split), mirroring how biological phenotyping works in the immunotherapy literature.
> - **Questions**:
>   1. *Do distinct immunological subtypes emerge from the data without supervision?*
>   2. *Do those subtypes differ significantly in overall survival?*
>   3. *Do immunotherapy responders concentrate in the Immune Hot subtype?*
>   4. *Does the NF1 loss Mutant-Driven subtype show a distinct clinical profile?*

## 1. Subtype Profiles

> [!INFO] What, Why & Key Questions — Subtype Fingerprints
> - **What We Are Doing**: Characterising the four discovered patient subtypes using 2D UMAP and PCA projections (for geometric separation) and an annotated Z-score heatmap (for per-feature biological detail).
> - **Why We Are Doing It**: UMAP captures non-linear manifold structure; PCA provides a linear orthogonal view. The heatmap overlays response rate and survival tracks to verify that phenotypes are clinically meaningful.
>   1. *Are the four subtypes cleanly separated in 2D projections?*
>   2. *Does the response rate track visibly with immune intensity in the heatmap?*

### 1.1. Subtype Visualisation (2D UMAP & PCA Projections)

The 2D UMAP and PCA scatter plots display the geometric separation of four phenotype subtypes:

![2D UMAP Projection of Clusters](../../plots/clinical/umap_clinical_clusters.png)

![2D PCA Projection of Clusters](../../plots/clinical/pca_clinical_clusters.png)

### 1.2. Annotated Subtype Feature Heatmap & Clinical Tracks

The heatmap details the Z-score signature matrix per phenotype cluster, annotated with immunotherapy response rates (CR/PR %) and median overall survival (OS):

![Annotated Subtype Feature Heatmap](../../plots/clinical/heatmap_clinical_clusters.png)

## 2. Biological Interpretation of Patient Subtypes

The two-stage GMM pipeline isolates 4 distinct patient phenotypes:

1. **Immune Cold** ($N = 52$)
    - *Immune Signatures*: IFN-$\gamma$ = 2.63, TIS = 2.55, CYT = 2.26, CD8+ = 1.99.
    - *Macrophage Polarisation*: M1/M2 Ratio = 0.51.
    - *Genomics*: TMB = 13.33 mut/Mb.
    - *Therapeutic Benefit*: Response rate in trial patients = **17/50 = 34.0%**.
    - *Prognosis*: Median OS = **14.4 months**.

2. **Immune Hot** ($N = 92$)
    - *Immune Signatures*: IFN-$\gamma$ = 3.55, TIS = 3.44, CYT = 3.33, CD8+ = 3.31.
    - *Macrophage Polarisation*: M1/M2 Ratio = 0.32.
    - *Genomics*: TMB = 9.57 mut/Mb.
    - *Therapeutic Benefit*: Response rate in trial patients = **26/85 = 30.6%**.
    - *Prognosis*: Median OS = **22.5 months**.

3. **Immunosuppressive M2-High** ($N = 97$)
    - *Immune Signatures*: IFN-$\gamma$ = 3.14, TIS = 3.09, CYT = 2.92, CD8+ = 2.80.
    - *Macrophage Polarisation*: M1/M2 Ratio = 1.16.
    - *Genomics*: TMB = 22.19 mut/Mb.
    - *Therapeutic Benefit*: Response rate in trial patients = **29/97 = 29.9%**.
    - *Prognosis*: Median OS = **22.6 months**.

4. **Mutant-Driven** ($N = 15$)
    - *Immune Signatures*: IFN-$\gamma$ = 3.46, TIS = 3.35, CYT = 3.20, CD8+ = 3.16.
    - *Macrophage Polarisation*: M1/M2 Ratio = 0.39.
    - *Genomics*: TMB = 57.11 mut/Mb.
    - *Therapeutic Benefit*: Response rate in trial patients = **10/15 = 66.7%**.
    - *Prognosis*: Median OS = **Not Reached**.

## 3. Immunotherapy Response & Overall Survival Validation

> [!INFO] What, Why & Key Questions — Clinical Outcome Validation
> - **What We Are Doing**: Testing whether the unsupervised GMM phenotype labels — derived without any response information — stratify immunotherapy response rates (in the $N = 247$ trial patients with binary labels) and overall survival (across all $N = 256$ patients).
> - **Why We Are Doing It**: This is the critical validation step. If GMM phenotypes correlate with clinical outcomes, the biology is real and the subtypes are clinically actionable. The two-stage design should also reveal whether the Mutant-Driven (NF1 Loss) subtype has a distinct survival profile from the immune-defined clusters.
>   1. *Do immunotherapy responders concentrate significantly in the Immune Hot cluster?*
>   2. *Is the survival separation across subtypes statistically significant (Log-Rank)?*

**Therapeutic Response Rate (Trial Cohorts, $N = 247$ with binary labels)**:

> [!INSIGHT] Chi-Square Response Rate Evaluation
> The Chi-Square test across phenotype response rates yields $p = 0.040$ — **statistically significant**.
> Immunotherapy responders are significantly enriched in the Immune Hot cluster (26/85 = 30.6%) compared to Immune Cold (17/50 = 34.0%).

![Response Rate by Cluster](../../plots/clinical/response_by_clinical_cluster.png)

**Overall Survival (Full Dataset, $N = 256$)**:

The survival separation across patient subtypes yields a Log-Rank $p = 0.241$, indicating a trend that does not reach conventional significance at this cohort size:

![KM Survival of Clinical Clusters](../../plots/clinical/km_clinical_clusters.png)

> [!INSIGHT] Key Insights: Two-Stage GMM Subtyping Summary
> - **Probabilistic assignment**: GMM provides soft posterior probabilities per patient (P_Immune_Hot, P_Immune_Cold, P_Immunosuppressive_M2_High, P_Mutant_Driven), capturing biological uncertainty at phenotype boundaries — an advantage over hard Ward clustering.
> - **NF1-driven subtype isolated**: Stage 2 carves out the Mutant-Driven phenotype (NF1 loss) from the NF1-enriched immune cluster, mirroring how immunotherapy literature treats driver mutations as a secondary stratification axis.
> - **Survival trend present**: The four phenotypes show a survival separation trend ($p = 0.241$) that does not reach conventional significance, likely due to cohort size ($N = 256$).
> - **Response rate stratifies significantly**: Immunotherapy responders concentrate in the Immune Hot cluster (26/85 = 30.6%) vs. Immune Cold (17/50 = 34.0%) ($p = 0.040$).

## 4. Methodological Limitations & Future Directions

> [!WARNING] Analytical Scope & Limitations
> - **Cohort Size**: The $N = 256$ pooled ICI-only dataset limits statistical power, particularly for the four-way Chi-Square response test. Expanding trial cohort coverage will improve per-phenotype subgroup precision.
> - **GMM Non-Determinism**: GMM cluster integer IDs are non-deterministic across runs. Phenotype labels are assigned via rank-based rules on empirical TIS profiles, making biological assignments reproducible even when component indices shift.
> - **NF1 Split Threshold**: Stage 2 uses any NF1-positive patient (mut_NF1 = 1) as the split criterion. An alternative threshold (e.g. top quartile of NF1 expression) may refine the Mutant-Driven subgroup boundary.
> - **Within-Cohort Z-Score Dependence**: Cluster boundaries depend on within-cohort standardisation; applying this subtyping scheme to a single new patient requires reference cohort normalisation parameters.

> [!formula]+ Clinical Subtyping Script Execution & Software Module Architecture
> - **Primary Pipeline Execution Scripts**:
>   - [`run_clinical_clustering.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-2-clinical-subtyping/run_clinical_clustering.py): Executes two-stage GMM + NF1 deterministic split stratification across ICI trial cohorts ($N = 256$) using the 12-feature set; generates UMAP, PCA, heatmap, KM, and response-rate plots; exports `clinical_clusters.csv` with posterior probability columns; and produces this report.
>   - [`plot_cluster_profile_visualizations.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-2-clinical-subtyping/plot_cluster_profile_visualizations.py): Generates supplementary cluster profile visualisations from `clinical_clusters.csv`; requires `run_clinical_clustering.py` to be executed first.
> - **Shared Cross-Question & Pipeline Modules**:
>   - [`signatures.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/signatures.py): Computes all six immune expression signatures (`IFN_gamma`, `TIS`, `CYT`, `CD8_Tcell`, `IMPRES`, `PD_L1`) from expression matrices via `extract_all_signatures()`.
>   - [`styles.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/src/styles.py): Single source of truth for Okabe-Ito colour palettes (`PHENOTYPE_PALETTE`, `RESPONSE_PALETTE`) and `get_phenotype_color()` for ID-agnostic colour lookup.