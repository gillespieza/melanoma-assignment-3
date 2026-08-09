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
created: 2026-08-09 14:08
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-09 14:08
---

# Patient Phenotyping via ICI Trial Cohort Two-Stage GMM Clustering

> [!INFO] What, Why & Key Questions — Overview
> - **What**: Applying a two-stage unsupervised Gaussian Mixture Model (GMM) stratification to the **immunotherapy (ICI) trial cohorts** ($N = 478$ patients across Liu 2019, Hugo 2016, Riaz 2017, TCGA GDC 2025, Gide 2019, and Van Allen 2015). Stage 1 fits a GMM ($K = 3$, full covariance) on 6 continuous immune/microenvironment Z-score features (TIS Signature, Cytolytic (CYT) Score, CD8+ T-cell Score, M1/M2 Macrophage Ratio, Macrophage STV Score, and PD-L1 Expression Score). Stage 2 applies a deterministic NF1-positive split on the NF1-enriched immune cluster, producing a 4th Mutant-Driven phenotype. Phenotype labels are assigned by rank-based rules on empirical cluster profiles — never by hardcoded cluster integer IDs.
> - **Why**: GMM provides soft posterior probability assignments rather than hard cluster membership, capturing biological uncertainty at phenotype boundaries. The two-stage design separates the continuous immune microenvironment axis (Stage 1 GMM) from the discrete driver mutation axis (Stage 2 NF1 split), mirroring how biological phenotyping works in the immunotherapy literature.
> - **Questions**:
>   1. *Do distinct immunological subtypes emerge from the data without supervision?*
>   2. *Do those subtypes differ significantly in overall survival?*
>   3. *Do immunotherapy responders concentrate in the Immune Hot subtype?*
>   4. *Does the NF1 loss Mutant-Driven subtype show a distinct clinical profile?*

## 1. Subtype Profiles

> [!INFO] What, Why & Key Questions — Subtype Fingerprints
> - **What**: Characterising the four discovered patient subtypes using 2D UMAP and PCA projections (for geometric separation) and an annotated Z-score heatmap (for per-feature biological detail).
> - **Why**: UMAP captures non-linear manifold structure; PCA provides a linear orthogonal view. The heatmap overlays response rate and survival tracks to verify that phenotypes are clinically meaningful.
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

### 2.1 Stage 1 GMM: How Phenotype Labels Are Assigned

Stage 1 fits a Gaussian Mixture Model ($K = 3$, full covariance) simultaneously on 6 continuous Z-score features: TIS Signature, Cytolytic (CYT) Score, CD8+ T-cell Score, M1/M2 Macrophage Ratio, Macrophage STV Score, and PD-L1 Expression Score. No phenotype label is assumed during fitting — the GMM discovers structure from the data. Labels are then assigned post-hoc via a single rank-based rule anchored on each cluster's empirical mean TIS (Tumour Inflammation Score), the most validated composite immune score in the ICI literature (Ayers et al., 2017 *J Clin Invest*):

| Assignment Rule | Phenotype Label |
|---|---|
| Cluster with the **highest** mean TIS | **Immune Hot** |
| Cluster with the **lowest** mean TIS | **Immune Cold** |
| The **remaining** cluster | **Immunosuppressive M2-High** |

This rank-based assignment is reproducible across GMM restarts: the biological phenotype label is always tied to the empirical cluster profile, never to a non-deterministic integer cluster ID.

> [!INFO] What, Why & Key Questions — Stage 1 Phenotype Biology
> - **What**: Characterising the biological meaning of the three Stage 1 immune archetypes and explaining how each relates to ICI response mechanisms.
> - **Why**: The phenotype labels must be grounded in the immunotherapy literature to be clinically interpretable. Each archetype corresponds to a distinct tumour-immune microenvironment (TME) state with a different predicted ICI response mechanism and therapeutic implication.

**Immune Hot** — highest TIS cluster. Characterised by high TIS, CYT, and CD8+ T-cell scores, and elevated PD-L1 expression. Active T-cell infiltration is present with functional cytolytic machinery. PD-L1 is elevated as an adaptive resistance response to IFN-γ secreted by tumour-infiltrating lymphocytes (TILs) — precisely the mechanism anti-PD-1 agents are designed to reverse. **These patients are the primary ICI responders.** BRAF V600E patients in this cluster retain their Immune Hot label: their immune microenvironment, not their mutation alone, drives ICI eligibility.

**Immune Cold** — lowest TIS cluster. Characterised by low TIS, CYT, CD8+, and PD-L1. Two mechanistic subtypes underlie this phenotype: (a) *immune desert* — T cells were never primed against tumour antigens due to low mutational burden or antigen presentation defects; or (b) *immune excluded* — T cells are primed but physically barred from the tumour parenchyma by stromal or vascular barriers. In either case, PD-1 blockade has no infiltrating effector T cells to unleash. **These patients are the poorest ICI responders** and may require priming strategies (STING agonists, cancer vaccines, anti-VEGF) before ICI is effective.

**Immunosuppressive M2-High** — intermediate TIS cluster (assigned by exclusion after Hot and Cold are identified). T cells are present but suppressed by M2-polarised tumour-associated macrophages (TAMs) secreting IL-10, TGF-β, and VEGF, producing a low M1/M2 ratio and elevated Macrophage STV score. NF1 loss enriches in this cluster because RAS/MAPK hyperactivation (from NF1 loss) drives M2 macrophage recruitment. NF1-positive patients are subsequently carved out as the Stage 2 Mutant-Driven phenotype. **ICI response is intermediate** — present but attenuated by active immunosuppression. Macrophage repolarisation strategies (anti-CSF1R, anti-IL-10) combined with ICI may improve outcomes in this group.

> [!NOTE] On the M2-High Label
> The Immunosuppressive M2-High phenotype is defined algorithmically as the **residual** cluster after Immune Hot and Immune Cold are identified. This is biologically motivated — intermediate TIS with elevated macrophage suppression signal is the canonical M2 TME signature — but it means the cluster boundary is defined partly by what it *is not*. The exported GMM posterior probabilities (`P_Immunosuppressive_M2_High`) capture patients near these boundaries with soft probability assignments rather than hard binary membership.

### 2.2 Per-Phenotype Profile Summary

Empirical feature profiles across all 4 patient phenotypes ($N = 478$):

1. **Immunosuppressive M2-High** ($N = 148$)
    - *Immune Signatures*: IFN-$\gamma$ = 3.41, TIS = 3.33, CYT = 3.19, CD8+ = 3.15.
    - *Macrophage Polarisation*: M1/M2 Ratio = 0.42.
    - *Genomics*: TMB = 10.23 mut/Mb.
    - *Therapeutic Benefit*: Response rate in trial patients = **43/141 = 30.5%**.
    - *Prognosis*: Median OS = **23.3 months**.

2. **Immune Cold** ($N = 84$)
    - *Immune Signatures*: IFN-$\gamma$ = 2.76, TIS = 2.70, CYT = 2.46, CD8+ = 2.19.
    - *Macrophage Polarisation*: M1/M2 Ratio = 1.20.
    - *Genomics*: TMB = 12.64 mut/Mb.
    - *Therapeutic Benefit*: Response rate in trial patients = **25/82 = 30.5%**.
    - *Prognosis*: Median OS = **21.2 months**.

3. **Immune Hot** ($N = 222$)
    - *Immune Signatures*: IFN-$\gamma$ = 4.31, TIS = 3.77, CYT = 3.47, CD8+ = 2.84.
    - *Macrophage Polarisation*: M1/M2 Ratio = 0.00.
    - *Genomics*: TMB = 0.03 mut/Mb.
    - *Therapeutic Benefit*: Response rate in trial patients = **49/91 = 53.8%**.
    - *Prognosis*: Median OS = **66.4 months**.

4. **Mutant-Driven** ($N = 24$)
    - *Immune Signatures*: IFN-$\gamma$ = 3.37, TIS = 3.28, CYT = 3.12, CD8+ = 3.10.
    - *Macrophage Polarisation*: M1/M2 Ratio = 0.44.
    - *Genomics*: TMB = 83.30 mut/Mb.
    - *Therapeutic Benefit*: Response rate in trial patients = **14/24 = 58.3%**.
    - *Prognosis*: Median OS = **32.2 months**.

## 3. Immunotherapy Response & Overall Survival Validation

> [!INFO] What, Why & Key Questions — Clinical Outcome Validation
> - **What**: Testing whether the unsupervised GMM phenotype labels — derived without any response information — stratify immunotherapy response rates (in the $N = 338$ trial patients with binary labels) and overall survival (across all $N = 478$ patients).
> - **Why**: This is the critical validation step. If GMM phenotypes correlate with clinical outcomes, the biology is real and the subtypes are clinically actionable. The two-stage design should also reveal whether the Mutant-Driven (NF1 Loss) subtype has a distinct survival profile from the immune-defined clusters.
>   1. *Do immunotherapy responders concentrate significantly in the Immune Hot cluster?*
>   2. *Is the survival separation across subtypes statistically significant (Log-Rank)?*

**Therapeutic Response Rate (Trial Cohorts, $N = 338$ with binary labels)**:

> [!INSIGHT] Chi-Square Response Rate Evaluation
> The Chi-Square test across phenotype response rates yields $p = 2.71 \times 10^{-4}$ — **statistically significant**.
> Immunotherapy responders are significantly enriched in the Immune Hot cluster (49/91 = 53.8%) compared to Immune Cold (25/82 = 30.5%).

![Response Rate by Cluster](../../plots/clinical/response_by_clinical_cluster.png)

**Overall Survival (Full Dataset, $N = 478$)**:

The survival separation across patient subtypes yields a Log-Rank $p = 2.18 \times 10^{-6}$, confirming that the GMM immune phenotypes capture genuine prognostic biology:

![KM Survival of Clinical Clusters](../../plots/clinical/km_clinical_clusters.png)

> [!INSIGHT] Key Insights: Two-Stage GMM Subtyping Summary
> - **Probabilistic assignment**: GMM provides soft posterior probabilities per patient (P_Immune_Hot, P_Immune_Cold, P_Immunosuppressive_M2_High, P_Mutant_Driven), capturing biological uncertainty at phenotype boundaries — an advantage over hard Ward clustering.
> - **NF1-driven subtype isolated**: Stage 2 carves out the Mutant-Driven phenotype (NF1 loss) from the NF1-enriched immune cluster, mirroring how immunotherapy literature treats driver mutations as a secondary stratification axis.
> - **Survival separation is significant**: The four phenotypes separate significantly by overall survival ($p = 2.18 \times 10^{-6}$), validating that GMM captures real prognostic biology.
> - **Response rate stratifies significantly**: Immunotherapy responders concentrate in the Immune Hot cluster (49/91 = 53.8%) vs. Immune Cold (25/82 = 30.5%) ($p = 2.71 \times 10^{-4}$).

## 4. Methodological Limitations & Future Directions

> [!WARNING] Analytical Scope & Limitations
> - **Cohort Size**: The $N = 478$ pooled ICI-only dataset limits statistical power, particularly for the four-way Chi-Square response test. Expanding trial cohort coverage will improve per-phenotype subgroup precision.
> - **GMM Non-Determinism**: GMM cluster integer IDs are non-deterministic across runs. Phenotype labels are assigned via rank-based rules on empirical TIS profiles, making biological assignments reproducible even when component indices shift.
> - **NF1 Split Threshold**: Stage 2 uses any NF1-positive patient (mut_NF1 = 1) as the split criterion. An alternative threshold (e.g. top quartile of NF1 expression) may refine the Mutant-Driven subgroup boundary.
> - **Within-Cohort Z-Score Dependence**: Cluster boundaries depend on within-cohort standardisation; applying this subtyping scheme to a single new patient requires reference cohort normalisation parameters.
---

> [!formula]+ Clinical Subtyping Script Execution & Software Module Architecture
>
> - [`run_clinical_clustering.py`](../../scripts/pillar-2-clinical-subtyping/run_clinical_clustering.py): Executes two-stage GMM + NF1 deterministic split stratification across ICI trial cohorts ($N = 478$) using the 12-feature set; generates UMAP, PCA, heatmap, KM, and response-rate plots; exports `clinical_clusters.csv` with posterior probability columns; and produces this report.
> - [`plot_cluster_profile_visualizations.py`](../../scripts/pillar-2-clinical-subtyping/plot_cluster_profile_visualizations.py): Generates supplementary cluster profile visualisations from `clinical_clusters.csv`; requires `run_clinical_clustering.py` to be executed first.
> - [`signatures.py`](../../src/signatures.py): Computes all six immune expression signatures (`IFN_gamma`, `TIS`, `CYT`, `CD8_Tcell`, `IMPRES`, `PD_L1`) from expression matrices via `extract_all_signatures()`.
> - [`styles.py`](../../../src/styles.py): Single source of truth for Okabe-Ito colour palettes (`PHENOTYPE_PALETTE`, `RESPONSE_PALETTE`) and `get_phenotype_color()` for ID-agnostic colour lookup.