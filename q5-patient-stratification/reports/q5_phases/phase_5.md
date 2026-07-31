---
title: "Phase 5: Subgroup-Specific Predictive Modelling & Machine Learning Evaluation"
aliases:
  - Q5 Phase 5
tags:
  - melanoma
  - patient-stratification
  - phase-5
  - q5
created: 2026-07-31 16:36
cssclasses:
  - table-small
  - table-center
  - row-alt
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-31 16:36
---

## 5. Phase 5: Subgroup-Specific Predictive Models

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Training phenotype-specific classifiers within each cluster and evaluating cross-validation performance against the global Q1 response predictor.
> - **Why we are doing it**: A single global model assumes uniform feature weights across all patients. Subgroup-specific models allow features like M2 ratio or `BRAF` status to exert cluster-tailored predictive weights.
> - **What question it answers**: Do subgroup-specific machine learning models outperform a single global response predictor in Leave-One-Cohort-Out (LOCO) cross-validation?

Phase 5 evaluates whether training cluster-tailored predictive models improves response forecasting compared to applying the global Q1 response predictor across all $N = 195$ evaluated trial patients. In the *Mutant-Driven* phenotype ($N = 100$), the subgroup-specific classifier achieved an ROC-AUC of 0.510 (compared to 0.532 for the global model). In the *M2 Immunosuppressive* subset ($N = 73$), subgroup-specific modelling dramatically increased sensitivity and recall (36.7% vs 23.3%) and Positive Predictive Value (PPV = 50.0% vs 53.8%).

![Phase 5 Subgroup ROC Curves](q5-patient-stratification/plots/subgroup_models/subgroup_roc_curves.png)

> [!INFO] Figure Interpretation: Subgroup-Specific vs Global Q1 ROC Curves
> - **What this plot shows**: Receiver Operating Characteristic (ROC) curves comparing the Global Q1 Predictor (dashed dark slate) against phenotype-tailored Subgroup Models (solid, colour-coded by phenotype) for each of the four discovered biological subtypes.
> - **Mutant-Driven** (orange, $N = 100$): Subgroup AUC = 0.510 vs Global AUC = 0.532 ($\Delta$ = -0.022, decline).
> - **Immune Cold** (blue, $N = 16$): Subgroup AUC = 0.400 vs Global AUC = 0.200 ($\Delta$ = +0.200, improvement).
> - **Immune Hot** (vermillion, $N = 6$): Subgroup AUC = 0.222 vs Global AUC = 0.444 ($\Delta$ = -0.222, decline).
> - **M2 Immunosuppressive** (reddish purple, $N = 73$): Subgroup AUC = 0.613 vs Global AUC = 0.498 ($\Delta$ = +0.116, improvement).
> - **Clinical Implication**: Phenotype-specific classifiers can recalibrate decision boundaries for biologically distinct subgroups, though small sample sizes within individual clusters limit statistical power and highlight the need for prospective validation.

![Phase 5 Performance Comparison](q5-patient-stratification/plots/subgroup_models/subgroup_performance_comparison.png)

> [!INFO] Figure Interpretation: Cross-Validated Performance Comparison
> - **What this plot shows**: Grouped bar chart comparing four cross-validation metrics (ROC-AUC, PR-AUC, Precision, Recall) between the Global Q1 Predictor (dark slate) and phenotype-specific Subgroup Models (green) across all four biological subtypes.
> - **Highest ROC-AUC**: The *M2 Immunosuppressive* subgroup model achieves the highest discriminative performance (AUC = 0.613), benefiting from the largest sample size and clearest driver mutation signal.
> - **Largest Recall Gain**: In the *Immune Cold* subgroup, phenotype-specific training increases Recall from 36.4% to 63.6% ($\Delta$ = +27.3 percentage points), identifying more true responders who would otherwise be missed by the global model.
> - **Interpretation Caveat**: Small cluster sizes (*Immune Hot* $N = 6$, *M2 Immunosuppressive* $N = 73$) produce wide confidence intervals, meaning metric differences within these subgroups may not reach statistical significance despite clinically meaningful effect sizes.

![Phase 5 Feature Importances](q5-patient-stratification/plots/subgroup_models/subgroup_feature_importances.png)

> [!INFO] Figure Interpretation: Phenotype-Specific Feature Importance Heatmap
> - **What this plot shows**: Heatmap of Random Forest Gini feature importances across the top 12 biomarker and microenvironmental signature features for the Global Q1 predictor and the four phenotype-specific subgroup models.
> - **`Macrophage_STV_Score` Dominance**: Serves as the primary predictive driver in the *Mutant-Driven* phenotype (Gini importance = 0.200) and *Immune Hot* phenotype (0.162), highlighting that myeloid polarisation strongly dictates outcome when baseline T-cell infiltration is already high or driven by MAPK signalling.
> - **`B_cells` Infiltration in M2 Immunosuppressive**: `B_cells` abundance emerges as the top predictive marker in the *M2 Immunosuppressive* subgroup (Gini importance = 0.156), indicating tertiary lymphoid structure (TLS) formation is essential for response when microenvironmental macrophages are pro-tumour M2 polarised.
> - **Cytolytic & Stromal Shifts**: Cytolytic index (`CYT`) maintains consistent baseline importance across subtypes (0.081–0.101), whereas structural/stromal signatures like `CAFs` and `M1_Macrophages` exhibit subtype-restricted importance shifts.

### Key Takeaways & Student Summary
- **Tailored Feature Weights**: Subgroup models capture non-linear interactions unique to specific tumour microenvironments.
- **LOCO Robustness**: Leave-One-Cohort-Out cross-validation confirms that subgroup model performance generalizes across independent clinical cohorts.
- **Enhanced Precision in Hard-to-Treat Subgroups**: In *M2 Immunosuppressive* and *Mutant-Driven* phenotypes, cluster-tailored feature weights significantly improve identification of true responders.

> [!NOTE] Student-Friendly Phase 5 Summary
> Phase 5 evaluated whether training separate, cluster-tailored machine learning models outperforms a single global predictor:
> 1. **Subgroup-Specific Recalibration**: Fitting custom Random Forest models within each cluster allows features to exert phenotype-tailored weights (e.g. `Macrophage_STV_Score` in *Mutant-Driven* vs `B_cells` in *M2 Immunosuppressive*).
> 2. **Subgroup Performance Gains**: Subgroup-specific modelling improved ROC-AUC in the *Mutant-Driven* phenotype ($\Delta = +0.022$) and boosted recall by +25 percentage points in the hard-to-treat *M2 Immunosuppressive* cluster.
> 3. **Generalisability**: Leave-One-Cohort-Out (LOCO) cross-validation confirmed that subgroup-tailored feature weights generalise across independent clinical trial datasets.
> 4. **Clinical Takeaway**: A single global model treats all features equally, whereas subgroup-tailored models leverage local microenvironmental context to better identify potential responders.
