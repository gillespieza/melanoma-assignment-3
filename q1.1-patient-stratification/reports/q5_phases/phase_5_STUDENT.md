---
title: "Phase 5: Subgroup-Specific Predictive Modelling Presentation"
aliases:
  - Q5 Phase 5 Presentation
  - Phase 5 Student Presentation
tags:
  - melanoma
  - patient-stratification
  - phase-5
  - predictive-modelling
  - machine-learning
  - student-presentation
created: 2026-08-01 19:56
cssclasses:
  - table-small
  - table-center
  - row-alt
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-01 19:56
---

# Phase 5: Subgroup-Specific Predictive Modelling

> [!NOTE] Phase 5 Context & Objectives
> - **What is being done**: Training phenotype-tailored Random Forest classifiers using soft Gaussian Mixture Model (GMM) posterior probability sample weighting across four distinct biological microenvironments (*Immune Hot*, *Immune Cold*, *Immunosuppressive M2-High*, and *Mutant-Driven*), and evaluating them against a global cohort baseline using Leave-One-Cohort-Out (LOCO) cross-validation.
> - **Why we are doing it**: Standard machine learning models trained on whole-cohort data risk averaging out phenotype-specific biomarker rules, potentially missing targeted signals that govern response within distinct tumor microenvironments.
> - **What question it answers**: Does training separate predictive models tailored to specific biological subgroups improve immune checkpoint inhibitor (ICI) response prediction compared to a single global model?

## 1. Phase Overview

Phase 5 evaluates the central hypothesis of patient stratification: whether tailoring machine learning models to distinct biological phenotypes improves predictive accuracy over a global one-size-fits-all classifier.

Using the four biological phenotypes identified in Phase 3 (*Immune Hot*, *Immune Cold*, *Immunosuppressive M2-High*, and *Mutant-Driven*), Phase 5 trains subgroup-specific Random Forest models using soft GMM posterior probability sample weighting (`sample_weight = P_k`). To ensure unbiased evaluation, feature selection strictly excludes the six continuous clustering features (`TIS`, `CYT`, `CD8_T_cells`, `M1_Macrophages`, `M2_Macrophages`, `CAFs`), focusing instead on non-circular biomarker and genomic predictors including `IFN_gamma`, `CD8_Tcell`, `PD_L1` (`CD274`), `M1_M2_Ratio`, `Macrophage_STV_Score`, `CD4_T_cells`, `NK_cells`, `B_cells`, and `TMB_NONSYNONYMOUS`. Performance is rigorously validated using Leave-One-Cohort-Out (LOCO) cross-validation across 195 patients with ground-truth clinical response labels.

## 2. Upstream Integration & Pipeline Synergy

Phase 5 serves as the predictive synthesis bridging biological characterisation and clinical decision support across the assignment pipeline:

- **Upstream Integration with Q1 & Q5 Phase 3**: Builds directly on the global enriched baseline framework from Q1 and the two-stage continuous GMM stratification from Q5 Phase 3. The continuous posterior probability vector for each patient (`P_Hot`, `P_Cold`, `P_M2`, `P_Mutant`) defines the soft sample weights for subgroup training.
- **Synergy with Q2 & Q4 Targeted Profiling**: Incorporates tumor mutational burden (`TMB_NONSYNONYMOUS`) alongside driver mutation status (`BRAF`, `NRAS`, `NF1`). This connects immune-oriented features with driver-targeted pathways explored in Q2 (Dabrafenib response) and Q4 (DepMap target essentiality).
- **Foundation for Phase 6 & Phase 7**: Out-of-fold probability predictions generated in Phase 5 provide the direct decision threshold inputs for Decision Curve Analysis (DCA) in Phase 6 and the 3-Arm Treatability Index in Phase 7.

## 3. Why This Matters for Patient Stratification (Q5)

In heterogeneous cancer cohorts, a single global predictive model often prioritises broad immune markers (such as high `CD8_Tcell` infiltration) while missing niche subpopulation dynamics. For instance:

- In an *Immune Hot* microenvironment, responsiveness depends on fine-grained balance between `PD_L1` suppression and interferon-gamma signaling (`IFN_gamma`).
- In an *Immunosuppressive M2-High* microenvironment, response is constrained by stromal exclusion barrier mechanisms (`M1_M2_Ratio` and `Macrophage_STV_Score`).
- In a *Mutant-Driven* tumor microenvironment, high mutational workload (`TMB_NONSYNONYMOUS`) drives neoantigen creation despite low baseline T-cell infiltration.

Evaluating subgroup-tailored models reveals whether clinical decision support systems should deploy modular, phenotype-specific classifiers or rely on pooled global models.

## 4. Key Phase Results & Empirical Findings

Performance evaluation across 195 ICI-treated trial patients across four clinical cohorts (Hugo 2016, Riaz 2017, Liu 2019) reveals critical trade-offs between sample size pooling and local model tailoring:

- **Overall Cohort Ensemble vs Global Baseline**: Across all 195 patients (82 responders, 42.1% response rate), the Global Enriched Baseline achieves an ROC-AUC of 0.550 (PPV = 53.8%), whereas the Subgroup Ensemble achieves an ROC-AUC of 0.488 (PPV = 48.6%). Global sample pooling maintains superior overall discrimination due to larger training sample sizes.
- **Immune Hot Subgroup (N = 103, 46 Responders, 44.7% Response Rate)**: The phenotype-tailored subgroup model achieves a Positive Predictive Value (PPV) of 73.7% compared to 63.0% for the global baseline (a 10.7% improvement in precision). While ROC-AUC is comparable (0.514 vs 0.541), tailoring feature weights improves the identification of true responders in inflamed tumors.
- **Immunosuppressive M2-High Subgroup (N = 55, 20 Responders, 36.4% Response Rate)**: The global model achieves an ROC-AUC of 0.617 (PPV = 37.5%), whereas the subgroup model achieves an ROC-AUC of 0.509 (PPV = 33.3%). Stromal exclusion signatures require broader cohort sample size to stabilize predictive weights.
- **Immune Cold Subgroup (N = 28, 11 Responders, 39.3% Response Rate)**: Both global (ROC-AUC = 0.374) and subgroup models (ROC-AUC = 0.380) exhibit low discrimination, reflecting the non-immunogenic desert phenotype where baseline RNA biomarkers offer minimal predictive traction.
- **Mutant-Driven Subgroup (N = 9, 5 Responders, 55.6% Response Rate)**: Small sample size partitioned across four LOCO trial folds yields extreme split fragmentation (1-2 patients per fold). While the global model achieves ROC-AUC of 0.900 (with 0.0% PPV and 0.0% Recall), the subgroup model achieves positive identification (Recall = 20.0%, PPV = 25.0%).

> [!INSIGHT] Key Takeaways: Subgroup Modelling Performance
> - **Precision Gains in Inflamed Tumors**: Subgroup-specific modeling boosts Positive Predictive Value in *Immune Hot* tumors from 63.0% to 73.7% (+10.7% gain), demonstrating that local feature re-weighting enhances precision when sample size is sufficient.
> - **Sample Fragmentation Trade-Off**: Partitioning trial cohorts into small subgroups reduces sample size per fold, causing overall cohort ROC-AUC to favor global sample pooling (0.550 vs 0.488).
> - **Zero Fallback Robustness**: Pre-filtering training arrays to non-zero posterior weights eliminates division-by-zero during tree bootstrap sampling, achieving zero out-of-fold fallbacks across all cross-validation folds.

## 5. Limitations & Future Directions

> [!WARNING] Methodological Limitations & Recommended Future Iterations
> - **Sample Size Bottleneck in Rare Subgroups**: Partitioning 195 trial patients into four phenotypes reduces training sample sizes to N = 28 in *Immune Cold* and N = 9 in *Mutant-Driven*. Leave-One-Cohort-Out cross-validation across small subgroups leads to high variance in metric estimation.
> - **Exclusion of Continuous Clustering Features**: To prevent circularity, Stage 1 GMM clustering features (`TIS`, `CYT`, `CD8_T_cells`, `M1_Macrophages`, `M2_Macrophages`, `CAFs`) are strictly excluded from Phase 5 prediction. While methodologically essential, this prevents classifiers from directly utilizing core cytotoxic expression levels.
> - **Static Pre-Treatment Profiling**: Models rely on baseline, pre-treatment bulk transcriptomic signatures and cannot capture dynamic immune cell recruitment, adaptive `PD-L1` upregulation, or clonal evolution during therapy.
> - **Future Iteration Roadmap**:
>   1. **Hierarchical / Transfer Learning**: Implement hierarchical Bayesian or multi-task neural networks that share global parameters across all patients while learning phenotype-specific adapter layers.
>   2. **Dynamic Genomic Feature Expansion**: Incorporate non-circular genomic predictors such as `HLA-A/B/C` antigen presentation expression and `JAK1`/`STAT1` loss-of-function variants.
>   3. **Synthetic Sampling for Small Folds**: Apply adaptive synthetic sampling within small LOCO cross-validation folds to stabilize model fitting in rare phenotypes like *Mutant-Driven*.
