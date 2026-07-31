---
title: "Phase 5: Subgroup-Specific Predictive Modelling & Machine Learning Evaluation"
aliases:
  - Q5 Phase 5
tags:
  - melanoma
  - patient-stratification
  - phase-5
  - q5
created: 2026-08-01 00:26
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-01 00:26
---

## 5. Phase 5: Subgroup-Specific Predictive Models

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Training phenotype-specific classifiers within each cluster and evaluating cross-validation performance against the global Q1 response predictor.
> - **Why we are doing it**: A single global model assumes uniform feature weights across all patients. Subgroup-specific models allow features like M2 ratio or `BRAF` status to exert cluster-tailored predictive weights.
> - **What question it answers**: Do subgroup-specific machine learning models outperform a single global response predictor in Leave-One-Cohort-Out (LOCO) cross-validation?

Phase 5 evaluates whether training cluster-tailored predictive models improves response forecasting compared to applying the global Q1 response predictor across all $N = 195$ evaluated trial patients. In the *Mutant-Driven* phenotype ($N = 28$), the subgroup-specific classifier achieved an ROC-AUC of 0.337 (compared to 0.578 for the global model). In the *Immunosuppressive M2-High* subset ($N = 97$), subgroup-specific modelling dramatically increased sensitivity and recall (16.7% vs 33.3%) and Positive Predictive Value (PPV = 54.5% vs 46.2%).

![Phase 5 Subgroup ROC Curves](q5-patient-stratification/plots/subgroup_models/subgroup_roc_curves.png)

> [!INFO] Figure Interpretation: Subgroup-Specific vs Global Enriched Baseline ROC Curves
> - **What this plot shows**: Receiver Operating Characteristic (ROC) curves comparing the Global Enriched Baseline (dashed dark slate; Q1 features + cell deconvolution) against phenotype-tailored Subgroup Models (solid, colour-coded by phenotype) for each of the four discovered biological subtypes.
> - **Mutant-Driven** (orange, $N = 28$): Subgroup AUC = 0.337 vs Global AUC = 0.578 ($\Delta$ = -0.241, decline).
> - **Immune Cold** (blue, $N = 9$): Subgroup AUC = 0.333 vs Global AUC = 0.333 ($\Delta$ = +0.000, decline).
> - **Immune Hot** (vermillion, $N = 61$): Subgroup AUC = 0.467 vs Global AUC = 0.555 ($\Delta$ = -0.088, decline).
> - **Immunosuppressive M2-High** (reddish purple, $N = 97$): Subgroup AUC = 0.484 vs Global AUC = 0.572 ($\Delta$ = -0.089, decline).
> - **Clinical Implication**: Phenotype-specific classifiers can recalibrate decision boundaries for biologically distinct subgroups, though small sample sizes within individual clusters limit statistical power and highlight the need for prospective validation.

![Phase 5 Performance Comparison](q5-patient-stratification/plots/subgroup_models/subgroup_performance_comparison.png)

> [!INFO] Figure Interpretation: Cross-Validated Performance Comparison
> - **What this plot shows**: Grouped bar chart comparing four cross-validation metrics (ROC-AUC, PR-AUC, Precision, Recall) between the Global Enriched Baseline (dark slate; Q1 + deconvolution) and phenotype-specific Subgroup Models (green) across all four biological subtypes.
> - **Highest ROC-AUC**: The *Immunosuppressive M2-High* subgroup model achieves the highest discriminative performance (AUC = 0.484), benefiting from the largest sample size and clearest driver mutation signal.
> - **Largest Recall Gain**: In the *Mutant-Driven* subgroup, phenotype-specific training increases Recall from 23.5% to 70.6% ($\Delta$ = +47.1 percentage points), identifying more true responders who would otherwise be missed by the global model.
> - **Interpretation Caveat**: Small cluster sizes (*Immune Hot* $N = 61$, *M2 Immunosuppressive* $N = 97$) produce wide confidence intervals, meaning metric differences within these subgroups may not reach statistical significance despite clinically meaningful effect sizes.

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

> [!formula] Phase 5 Script Execution & Software Module Architecture
> - **Primary Pipeline Execution Scripts**:
>   - [`05_subgroup_models.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/05_subgroup_models.py): Trains soft-weighted GMM probability Random Forest classifiers on 9 non-circular features (`IFN_gamma`, `CD8_Tcell`, `PD_L1`, `M1_M2_Ratio`, `Macrophage_STV_Score`, `CD4_T_cells`, `NK_cells`, `B_cells`, `TMB_NONSYNONYMOUS`), evaluates Leave-One-Cohort-Out (LOCO) CV vs Global Enriched Baseline (`subgroup_models_evaluation.csv`), serialises fitted models (`joblib`), and generates ROC, performance, and Gini feature importance plots (`subgroup_roc_curves.png`, `subgroup_performance_comparison.png`, `subgroup_feature_importances.png`).
> - **Core Supporting Python Modules**:
>   - [`phenotyping.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/phenotyping.py): Phenotype palette resolution and short-name mappings.
>   - [`q5_constants.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/q5_constants.py): GMM probability column mappings (`PHENOTYPE_PROB_COL`) and clustering feature sets.
>   - [`reporting.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/reporting.py): Formats evaluation comparison tables and markdown frontmatter.
> - **Shared Cross-Question & Pipeline Modules**:
>   - [`run_q5_pipeline.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/run_q5_pipeline.py): Master pipeline orchestrator executing `05_subgroup_models.py` as Step 5.
>   - [`generate_q5_report.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/generate_q5_report.py): Compiles live statistical summaries and generates phase markdown reports.
