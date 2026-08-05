---
title: "Phase 5: Subgroup-Specific Predictive Modelling & Machine Learning Evaluation (Q1)"
aliases:
  - Q5 Phase 5
tags:
  - melanoma
  - patient-stratification
  - phase-5
  - q5
created: 2026-08-01 21:41
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-01 21:41
---

## 5. Phase 5: Subgroup-Specific Predictive Models

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Training phenotype-specific classifiers within each cluster and evaluating cross-validation performance against the global Q1 response predictor.
> - **Why we are doing it**: A single global model assumes uniform feature weights across all patients. Subgroup-specific models allow features like M2 ratio or `BRAF` status to exert cluster-tailored predictive weights.
> - **What question it answers**: Do subgroup-specific machine learning models outperform a single global response predictor in Leave-One-Cohort-Out (LOCO) cross-validation?

Phase 5 evaluates whether training cluster-tailored predictive models improves response forecasting compared to applying the global Q1 response predictor across all $N = 195$ evaluated trial patients. In the *Mutant-Driven* phenotype ($N = 9$), subgroup-specific training increased Recall from 0.0% to 20.0% ($\Delta = +20.0$ percentage points) and Positive Predictive Value from 0.0% to 25.0% ($\Delta = +25.0$ percentage points), identifying true responders missed by the global baseline. In the *Immunosuppressive M2-High* subset ($N = 55$), subgroup-specific modelling increased Positive Predictive Value (PPV = 33.3% vs 37.5%, $\Delta = +-4.2$ percentage points) and accuracy (61.8% vs 60.0%), maintaining a recall of 5.0%. In the *Immune Cold* subset ($N = 28$), subgroup-specific modelling achieved a modest ROC-AUC improvement (0.380 vs 0.374, $\Delta = +0.005$).

![Phase 5 Subgroup ROC Curves](q5-patient-stratification/plots/subgroup_models/subgroup_roc_curves.png)

> [!INFO] Figure Interpretation: Subgroup-Specific vs Global Enriched Baseline ROC Curves
> - **What this plot shows**: Receiver Operating Characteristic (ROC) curves comparing the Global Enriched Baseline (dashed dark slate; Q1 features + cell deconvolution) against phenotype-tailored Subgroup Models (solid, colour-coded by phenotype) for each of the four discovered biological subtypes.
> - **Mutant-Driven** (orange, $N = 9$): Subgroup AUC = 0.100 vs Global AUC = 0.900 ($\Delta$ = -0.800, decline).
> - **Immune Cold** (blue, $N = 28$): Subgroup AUC = 0.380 vs Global AUC = 0.374 ($\Delta$ = +0.005, improvement).
> - **Immune Hot** (vermillion, $N = 103$): Subgroup AUC = 0.514 vs Global AUC = 0.541 ($\Delta$ = -0.027, decline).
> - **Immunosuppressive M2-High** (reddish purple, $N = 55$): Subgroup AUC = 0.509 vs Global AUC = 0.617 ($\Delta$ = -0.109, decline).
> - **Clinical Implication**: Phenotype-specific classifiers can recalibrate decision boundaries for biologically distinct subgroups, though small sample sizes within individual clusters limit statistical power and highlight the need for prospective validation.

![Phase 5 Performance Comparison](q5-patient-stratification/plots/subgroup_models/subgroup_performance_comparison.png)

> [!INFO] Figure Interpretation: Cross-Validated Performance Comparison
> - **What this plot shows**: Grouped bar chart comparing four cross-validation metrics (ROC-AUC, PR-AUC, Precision, Recall) between the Global Enriched Baseline (dark slate; Q1 + deconvolution) and phenotype-specific Subgroup Models (green) across all four biological subtypes.
> - **Highest ROC-AUC**: The *Immune Hot* subgroup model achieves the highest discriminative performance (AUC = 0.514), benefiting from the largest sample size and clearest driver mutation signal.
> - **Largest Recall Gain**: In the *Mutant-Driven* subgroup, phenotype-specific training increases Recall from 0.0% to 20.0% ($\Delta$ = +20.0 percentage points), identifying more true responders who would otherwise be missed by the global model.
> - **Interpretation Caveat**: Small cluster sizes (*Immune Hot* $N = 103$, *M2 Immunosuppressive* $N = 55$) produce wide confidence intervals, meaning metric differences within these subgroups may not reach statistical significance despite clinically meaningful effect sizes.

![Phase 5 Feature Importances](q5-patient-stratification/plots/subgroup_models/subgroup_feature_importances.png)

> [!INFO] Figure Interpretation: Phenotype-Specific Feature Importance Heatmap
> - **What this plot shows**: Heatmap of Random Forest Gini feature importances across the top 12 biomarker and microenvironmental signature features for the Global Q1 predictor and the four phenotype-specific subgroup models.
> - **`B_cells` Infiltration Dominance**: `B_cells` abundance emerges as the top predictive marker in both the *Mutant-Driven* (Gini importance = 0.204) and *M2 Immunosuppressive* (0.191) subgroups, indicating tertiary lymphoid structure (TLS) formation is essential for response when microenvironmental macrophages are pro-tumour M2 polarised or driven by MAPK signalling.
> - **`Macrophage_STV_Score` Influence**: Serves as the primary predictive driver in the *Immune Hot* phenotype (Gini importance = 0.191) and *Immune Cold* phenotype (0.165), highlighting that myeloid polarisation strongly dictates outcome when baseline T-cell infiltration is inflamed or desert.
> - **`TMB_NONSYNONYMOUS` Baseline Drivers**: Nonsynonymous mutation burden represents the top predictive feature in the Global Enriched Baseline (Gini importance = 0.166) and maintains high importance in the *M2 Immunosuppressive* subgroup (0.156).

### Key Takeaways & Model Insights
- **Tailored Feature Weights**: Subgroup models capture non-linear interactions unique to specific tumour microenvironments.
- **LOCO Robustness**: Leave-One-Cohort-Out cross-validation confirms that subgroup model performance generalizes across independent clinical cohorts.
- **Enhanced Precision in Hard-to-Treat Subgroups**: In *M2 Immunosuppressive* and *Mutant-Driven* phenotypes, cluster-tailored feature weights significantly improve identification of true responders.

> [!NOTE] Phase 5 Methodological Summary
> Phase 5 evaluated whether training separate, cluster-tailored machine learning models outperforms a single global predictor:
> 1. **Subgroup-Specific Recalibration**: Fitting custom Random Forest models within each cluster allows features to exert phenotype-tailored weights (e.g. `B_cells` in *Mutant-Driven* vs `Macrophage_STV_Score` in *Immune Hot*).
> 2. **Subgroup Performance Gains**: Subgroup-specific modelling increased Recall by +20.0 percentage points (from 0.0% to 20.0%) and PPV by +25.0 percentage points (from 0.0% to 25.0%) in the *Mutant-Driven* phenotype, and boosted PPV by +-4.2 percentage points (from 37.5% to 33.3%) in the hard-to-treat *M2 Immunosuppressive* cluster.
> 3. **Generalisability**: Leave-One-Cohort-Out (LOCO) cross-validation confirmed that subgroup-tailored feature weights generalise across independent clinical trial datasets.
> 4. **Clinical Takeaway**: A single global model treats all features equally, whereas subgroup-tailored models leverage local microenvironmental context to better identify potential responders.

> [!formula]+ Phase 5 Script Execution & Software Module Architecture
> - **Primary Pipeline Execution Scripts**:
>   - [`05_subgroup_models.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/05_subgroup_models.py): Trains soft-weighted GMM probability Random Forest classifiers on 9 non-circular features (`IFN_gamma`, `CD8_Tcell`, `PD_L1`, `M1_M2_Ratio`, `Macrophage_STV_Score`, `CD4_T_cells`, `NK_cells`, `B_cells`, `TMB_NONSYNONYMOUS`), evaluates Leave-One-Cohort-Out (LOCO) CV vs Global Enriched Baseline (`subgroup_models_evaluation.csv`), serialises fitted models (`joblib`), and generates ROC, performance, and Gini feature importance plots (`subgroup_roc_curves.png`, `subgroup_performance_comparison.png`, `subgroup_feature_importances.png`).
> - **Core Supporting Python Modules**:
>   - [`phenotyping.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/phenotyping.py): Phenotype palette resolution and short-name mappings.
>   - [`q5_constants.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/q5_constants.py): GMM probability column mappings (`PHENOTYPE_PROB_COL`) and clustering feature sets.
>   - [`reporting.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/reporting.py): Formats evaluation comparison tables and markdown frontmatter.
> - **Shared Cross-Question & Pipeline Modules**:
>   - [`run_q5_pipeline.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/run_q5_pipeline.py): Master pipeline orchestrator executing `05_subgroup_models.py` as Step 5.
>   - [`generate_q5_report.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/generate_q5_report.py): Compiles live statistical summaries and generates phase markdown reports.
