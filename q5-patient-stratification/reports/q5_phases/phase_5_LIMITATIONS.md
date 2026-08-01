---
title: "Phase 5: Limitations, Methodological Critique & Future Directions"
aliases:
  - Q5 Phase 5 Limitations
tags:
  - melanoma
  - patient-stratification
  - phase-5
  - limitations
  - q5
created: 2026-08-01 18:48
cssclasses:
  - table-small
  - table-center
  - row-alt
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-01 18:48
---

## 5. Phase 5: Methodological Critique & Limitations Evaluation

> [!NOTE] Purpose & Scope of Phase 5 Critique
> - **Objective**: Critical evaluation of Phase 5 Subgroup-Specific Machine Learning Modelling in its current state across software architecture, statistical validation, biological assumptions, and data constraints.
> - **Source Data**: Grounded in live execution logs (`logs/05_subgroup_models.log`) and quantitative Leave-One-Cohort-Out (LOCO) evaluation metrics (`data/processed/q5/subgroup_models_evaluation.csv`).

---

### 5.1 Code-Quality & Software Architecture Evaluation

> [!WARNING] Software Architecture & Pipeline Edge-Case Constraints
> - **Degenerate Out-of-Fold (OOF) Prediction Fallbacks**: In the current pipeline, 27 out-of-fold patient slots (primarily in the *Mutant-Driven* $N=9$ and *Immune Cold* $N=28$ subgroups) encounter single-class training folds during Leave-One-Cohort-Out cross-validation. When a LOCO test fold removes an entire clinical trial containing all positive responders, Platt scaling (`LogisticRegression`) and Isotonic Regression fail to fit. The architecture safely catches these edge cases using pass-through calibrators (`_IdentityPredictor`) and replaces the 27 degenerate OOF slots with predictions from the global model. While defensively robust, this fallback introduces hybrid prediction logic into the evaluation matrix.
> - **Soft GMM Weighting vs Sample Inflation**: Subgroup Random Forest models are fitted using GMM posterior probabilities ($\vec{P}_i$) as sample weights (`sample_weight = P_k`). While sound in principle, samples with low posterior probability ($P_k \in [10^{-6}, 0.10]$) still exert weak sample-weight influence, effectively inflating the sample count during tree splitting without contributing strong phenotype-specific signal.

---

### 5.2 Statistical Weaknesses & Methodological Limitations

> [!WARNING] Statistical Power & Cross-Validation Degradation
> - **Severe Sample Size Imbalance**: While the overall evaluated trial cohort contains $N = 195$ patients with ground-truth immunotherapy response labels, partitioning patients into four GMM biological phenotypes yields extreme sample imbalance:
>   - *Immune Hot*: $N = 103$ ($46$ Responders, $44.7\%$ response rate)
>   - *Immunosuppressive M2-High*: $N = 55$ ($20$ Responders, $36.4\%$ response rate)
>   - *Immune Cold*: $N = 28$ ($11$ Responders, $39.3\%$ response rate)
>   - *Mutant-Driven*: $N = 9$ ($5$ Responders, $55.6\%$ response rate)
> - **LOCO CV Performance Degradation**: Training separate classifiers within small subgroups reduces effective sample size, leading to lower overall cohort ROC-AUC for the Subgroup Ensemble ($0.501$) compared to the Global Enriched Baseline ($0.550$, $\Delta = -0.049$).
> - **Extremely Small Subgroup CV Folds**: In the *Mutant-Driven* phenotype ($N = 9$), Leave-One-Cohort-Out CV splits 9 patients across 4 clinical cohorts, leaving folds with as few as 1 or 2 patients. This results in an artificially depressed subgroup ROC-AUC of $0.300$ (vs $0.900$ for the global baseline), despite achieving improvements in Recall ($20.0\%$ vs $0.0\%$) and Positive Predictive Value ($33.3\%$ vs $0.0\%$).

---

### 5.3 Biological Assumptions & Domain Constraints

> [!WARNING] Biological Modeling & Feature Selection Constraints
> - **Strict Exclusion of Clustering Features**: To prevent circular reasoning, the candidate feature set for Phase 5 subgroup models ($9$ features: `IFN_gamma`, `CD8_Tcell`, `PD_L1`, `M1_M2_Ratio`, `Macrophage_STV_Score`, `CD4_T_cells`, `NK_cells`, `B_cells`, `TMB_NONSYNONYMOUS`) strictly excludes the 6 Stage 1 GMM continuous clustering features (`TIS`, `CYT`, `CD8_T_cells`, `M1_Macrophages`, `M2_Macrophages`, `CAFs`). While methodologically rigorous, excluding `TIS` and `CYT` prevents subgroup classifiers from exploiting subtle non-linear interactions within core cytotoxic signatures.
> - **Assumption of Discrete Decision Boundaries**: Fitting separate Random Forest classifiers within distinct GMM phenotypes assumes that biological microenvironments operate on decoupled predictive hyperplanes. In reality, tumour microenvironments exist on a continuous spectrum, and hard phenotype partitioning can misclassify patients situated near GMM cluster boundaries.
> - **Static Pre-Treatment Biomarkers**: All features are derived from baseline, pre-treatment RNA-seq and WES profiling. Static baseline measurements cannot capture dynamic immune cell recruitment, adaptive immune resistance (`PD-L1` upregulation), or clonal evolution occurring under active anti-PD-1 therapy.

---

### 5.4 Computational & Data Constraints

> [!WARNING] Data Coverage & Measurement Resolution Constraints
> - **Outcome Label Bottleneck**: Although Phase 3 stratification establishes biological phenotype assignments for all $N = 699$ patients across the combined dataset (including TCGA-SKCM biological reference), Phase 5 predictive model evaluation is constrained to the $N = 195$ ICI-treated patients with curated $CR/PR/PD$ clinical response labels across the Hugo 2016, Riaz 2017, and Liu 2019 trials.
> - **Lack of Spatial & Single-Cell Resolution**: Bulk RNA-seq transcriptomic cell deconvolution estimates total cell type fractions but lacks spatial resolution. It cannot determine whether CD8+ T-cells are physically in contact with tumour cells (inflamed) or trapped in peri-tumoural stroma by CAFs (excluded).

---

### 5.5 Actionable Improvements & Future Iterations

> [!INSIGHT] Prioritised Roadmap for Future Subgroup Modelling Iterations
> 1. **Hierarchical / Transfer Learning Architectures**: Instead of fitting completely independent Random Forest models per subgroup, adopt a hierarchical Bayesian or multi-task neural network architecture. This allows global features to share statistical strength across the full cohort while permitting phenotype-specific adapter layers to fine-tune local feature weights.
> 2. **Continuous GMM Weight-Regularised Classifiers**: Replace hard subgroup partitioning with soft sample-weighted ensemble classifiers that dynamically weight predictions based on each patient's full continuous GMM posterior probability vector $\vec{P}_i = [P_{\text{Hot}}, P_{\text{Cold}}, P_{\text{M2}}, P_{\text{Mut}}]$.
> 3. **Nested Feature Selection & Feature Expansion**: Incorporate additional non-circular genomic and microenvironmental features (e.g. `HLA-A/B/C` antigen presentation expression, `JAK1`/`STAT1` loss-of-function variants, tertiary lymphoid structure signatures) to provide broader signal for small subgroups.
> 4. **Synthetic Minority Oversampling (SMOTE) for Small LOCO Folds**: Apply SMOTE or adaptive synthetic sampling within small cross-validation training folds (such as *Mutant-Driven* $N=9$) to mitigate single-class calibration failures during LOCO CV.

---

### Key Takeaways

> [!INSIGHT] Key Takeaways: Phase 5 Methodological Critique
> - **Trade-Off Between Global Pooling & Local Tailoring**: Global models benefit from large sample pooling ($N = 195$, AUC = $0.550$), whereas subgroup models suffer from sample fragmentation in small clusters (*Mutant-Driven* $N = 9$, AUC = $0.300$), despite improving localized precision (PPV = $50.0\%$ in *M2-High* vs $37.5\%$).
> - **Robust Edge-Case Fallbacks**: The current implementation handles LOCO CV single-class fold failures via 27 global model fallbacks, ensuring pipeline stability.
> - **Path to Optimization**: Future iterations should transition from isolated subgroup classifiers to hierarchical multi-task models that combine global feature pooling with phenotype-specific adaptation.
