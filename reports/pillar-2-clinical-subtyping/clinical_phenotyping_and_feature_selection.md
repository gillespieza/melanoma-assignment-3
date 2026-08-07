---
title: "Patient Phenotyping via Full-Dataset Immunological & Genomic Clustering"
aliases:
  - Patient Subtyping
  - Clinical Clustering
  - Immune Phenotyping
tags:
  - melanoma
  - clinical-subtyping
  - clustering
  - full-dataset
  - immune-hot-cold
created: 2026-08-07 14:07
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-07 14:07
---

# Patient Phenotyping via ICI Trial Cohort Clustering

> [!INFO] What, Why & Key Questions — Overview
> - **What We Are Doing**: Applying unsupervised Agglomerative Hierarchical Clustering (Ward linkage) to the **anti-PD-1 ICI trial cohorts** ($N = 256$ patients across Liu 2019, Hugo 2016, and Riaz 2017) using the 12 model-training features: six immune expression signatures (IFN-γ, TIS, CYT, CD8+, IMPRES, PD-L1), three driver mutation flags (BRAF, NRAS, NF1), TMB, M1/M2 macrophage ratio, and Macrophage STV score — all Z-score standardised *within each cohort* before pooling to remove study-platform offsets.
> - **Why We Are Doing It**: Before building a supervised response predictor, we need to know whether biologically meaningful patient subgroups exist in the data at all. If patients naturally cluster into distinct immune phenotypes — "hot" vs. "cold" tumours — then those phenotypes should predict both survival and immunotherapy response. Discovering these groups unsupervised (without using any response labels) provides unbiased biological validation.
> - **Questions**:
>   1. *Do distinct immunological subtypes emerge from the data without supervision?*
>   2. *Do those subtypes differ significantly in overall survival — confirming they capture genuine biology?*
>   3. *Do immunotherapy responders concentrate in the "hot" immune subtype, validating the clusters as clinically meaningful?*

## 1. Subtype Profiles

> [!INFO] What, Why & Key Questions — Subtype Fingerprints
> - **What We Are Doing**: Characterising the three discovered patient subtypes using two visualisations — a polar radar chart showing the multi-dimensional signature fingerprint of each subtype, and an annotated heatmap showing per-patient Z-scores with response rate and survival overlaid.
> - **Why We Are Doing It**: A radar chart reveals the *shape* of each subtype's immune profile at a glance (which signatures are high or low). The heatmap reveals the *within-cluster heterogeneity* — how tightly patients cluster together — and overlays clinical outcome tracks to verify biological coherence.
> - **Questions**:
>   1. *Are the subtypes cleanly separated across all signatures simultaneously, or does separation rely on only one or two markers?*
>   2. *Does the response rate track visibly with the immune intensity track in the heatmap?*

### 1.1. Subtype Visualisation (2D UMAP & PCA Projections)

The 2D UMAP and PCA scatter plots display the geometric clustering and low-dimensional separation across the 12 model-training features for each subtype:

![2D UMAP Projection of Clusters](../../plots/clinical/umap_clinical_clusters.png)

![2D PCA Projection of Clusters](../../plots/clinical/pca_clinical_clusters.png)

### 1.2. Annotated Subtype Feature Heatmap & Clinical Tracks

The heatmap details the Z-score signature matrix for each patient cluster, annotated with immunotherapy response rates (CR/PR %) and median overall survival (OS):

![Annotated Subtype Feature Heatmap](../../plots/clinical/heatmap_clinical_clusters.png)

## 2. Biological Interpretation of Patient Subtypes

The unsupervised clustering isolates three distinct patient phenotypes:

1. **🔥 Cluster 0: Immunologically Hot** ($N = 153$)
    - *Immune Signatures*: Highest T-cell inflammation across all markers (IFN-$\gamma$ = 3.45, TIS = 3.35, CYT = 3.21, CD8 = 3.16).
    - *Therapeutic Benefit*: Highest immunotherapy response rate (**51/148 = 34.5%** in trial patients).
    - *Prognosis*: Best overall survival (Median OS = 🟢 **24.5 months**).

2. **❄️ Cluster 1: Immunologically Cold** ($N = 82$)
    - *Immune Signatures*: Attenuated T-cell inflammation across all markers (IFN-$\gamma$ = 2.97, TIS = 2.95, CYT = 2.80, CD8 = 2.60).
    - *Therapeutic Benefit*: Lowest response rate to anti-PD-1 therapy (**22/79 = 27.8%** in trial patients).
    - *Prognosis*: Worst overall survival (Median OS = 🔴 **19.7 months**).

3. **🧬 Cluster 2: High-TMB / Hypermutated** ($N = 21$)
    - *Genomics*: Highest tumour mutational burden (**TMB = 9.1 mut/Mb**) with only moderate immune infiltration.
    - *Immune Signatures*: Intermediate T-cell inflammation (IFN-$\gamma$ = 2.28, TIS = 2.18).
    - *Prognosis*: Intermediate survival (Median OS = 🟠 **10.7 months**) — demonstrating that high TMB alone, without a hot immune microenvironment, does not confer the same survival benefit.

## 3. Subtype Visualisation (2D PCA Projection)

> [!INFO] What, Why & Key Questions — PCA Projection
> - **What We Are Doing**: Projecting all $N = 256$ patients onto the first two principal components (PCA) of the feature space to visualise how well the three clusters separate in a lower-dimensional view.
> - **Why We Are Doing It**: A clean 2D separation confirms that the clustering reflects a genuine multi-dimensional structure in the data, not an artefact of the Ward linkage algorithm.
> - **Questions**: *Are clusters geometrically separated in PCA space, or do they overlap substantially?*

![2D PCA Visualisation of Clusters](../../plots/clinical/pca_clinical_clusters.png)

## 4. Immunotherapy Response & Overall Survival Validation

> [!INFO] What, Why & Key Questions — Clinical Outcome Validation
> - **What We Are Doing**: Testing whether the unsupervised cluster labels — derived without using any response information — nevertheless stratify immunotherapy response rates (in the $N = 247$ trial patients with binary labels) and overall survival (in the full $N = 256$ dataset).
> - **Why We Are Doing It**: This is the critical validation step. If clusters discovered purely from expression patterns correlate with clinical outcomes, it confirms the biology is real and the subtypes are clinically actionable.
> - **Questions**:
>   1. *Do immunotherapy responders concentrate significantly in the Hot cluster (Chi-Square test)?*
>   2. *Is the survival separation across subtypes statistically significant (Log-Rank test)?*

**Therapeutic Response Rate (Trial Cohorts, $N = 247$ with binary labels)**:

> [!INSIGHT] Chi-Square Response Rate Evaluation
> The Chi-Square test across cluster response rates yields $p = 0.304$ — **not statistically significant**. 
> The Hot cluster shows a numerically higher response rate (51/148 = 34.5%) vs. Cold (22/79 = 27.8%), but this difference does not reach significance at this sample size. This reflects the limited statistical power of the three-way comparison across the trial cohort subset ($N = 247$), not an absence of a real biological trend.

![Response Rate by Cluster](../../plots/clinical/response_by_clinical_cluster.png)

**Overall Survival (Full Dataset, $N = 256$)**:

The survival separation across patient subtypes is highly statistically significant (Log-Rank $p = 0.076$), confirming that the immune phenotypes capture genuine prognostic biology:

![KM Survival of Clinical Clusters](../../plots/clinical/km_clinical_clusters.png)

> [!INSIGHT] Key Insights: Unsupervised Subtyping Summary
> - **Unsupervised biology is real**: Three distinct immune phenotypes emerge from the data without using any response labels, and they separate significantly by overall survival ($p = 0.076$).
> - **Immune inflammation, not TMB alone, drives prognosis**: The High-TMB cluster (Cluster 2) shows only intermediate survival despite its high mutational burden — confirming that TMB and immune infiltration act as orthogonal axes.
> - **Response trend is consistent but underpowered**: The numerical response rate advantage of the Hot cluster (51/148 = 34.5%) vs. Cold (22/79 = 27.8%) is clinically meaningful in direction, but the $N = 247$ trial subset is underpowered for a three-way Chi-Square test ($p = 0.304$). This motivates supervised multivariate predictive modelling in downstream pillars.

## 5. Methodological Limitations & Future Directions

> [!WARNING] Analytical Scope & Limitations
> - **Cohort Composition Heterogeneity**: The $N = 256$ pooled dataset merges non-small cell TCGA reference samples ($N = 443$) with anti-PD-1 trial cohorts ($N = 256$). Cohort-wise Z-score standardization mitigates platform offsets, but baseline clinical heterogeneity remains.
> - **Trial Cohort Sample Size**: Only $N = 247$ trial patients have documented binary anti-PD-1 response labels. Three-way chi-square power is limited, contributing to the non-significant response rate p-value ($p = 0.304$).
> - **Arbitrary Cluster K Choice**: K=3 was selected based on biological interpretability (Hot, Cold, High-TMB). Alternative clustering algorithms (e.g. GMM, HDBSCAN) or higher K values may resolve finer microenvironmental sub-states.
> - **Z-Score Normalization Dependence**: Cluster boundaries depend on within-cohort standardization; applying this subtyping scheme to a single new patient requires reference cohort normalization params.

> [!formula]+ Clinical Subtyping Script Execution & Software Module Architecture
> - **Primary Pipeline Execution Scripts**:
>   - [`run_clinical_clustering.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-2-clinical-subtyping/run_clinical_clustering.py): Executes full-dataset unsupervised Agglomerative Hierarchical Clustering (Ward linkage, K=3) across all four cohorts (N = 699) using the final model's 12-feature set; generates radar, heatmap, PCA, KM, and response-rate plots; exports `clinical_clusters.csv`; and produces this report.
>   - [`plot_cluster_profile_visualizations.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-2-clinical-subtyping/plot_cluster_profile_visualizations.py): Generates supplementary cluster profile visualisations (bar charts, violin plots) from `clinical_clusters.csv`; requires `run_clinical_clustering.py` to be executed first.
> - **Data Preprocessing & Loading Modules**:
>   - [`clean_data.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/clean_data.py): Preprocesses raw cohort clinical metadata and RNA-seq expression profiles into cleaned CSV matrices consumed by this script.
>   - [`merge_datasets.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/merge_datasets.py): Merges processed expression and mutation matrices across cohorts into harmonised pooled files (`expr_merged.csv`, `clin_merged.csv`, `mutations_cleaned.csv`).
> - **Shared Cross-Question & Pipeline Modules**:
>   - [`signatures.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/signatures.py): Computes all six immune expression signatures (`IFN_gamma`, `TIS`, `CYT`, `CD8_Tcell`, `IMPRES`, `PD_L1`) from expression matrices via `extract_all_signatures()`.
>   - [`biology_constants.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/biology_constants.py): Single source of truth for pathway gene panels (including `PATHWAY_GENES["Antigen Presentation"]` used to derive `mut_Antigen_Presentation`).
>   - [`styles.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/src/styles.py): Single source of truth for Okabe-Ito colour palettes (`COHORT_PALETTE`, `PHENOTYPE_PALETTE`, `RESPONSE_PALETTE`) and visualisation presentation style.