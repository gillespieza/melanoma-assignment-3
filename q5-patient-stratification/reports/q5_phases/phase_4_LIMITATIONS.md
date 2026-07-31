---
title: "Phase 4: Methodological Audit, Biological Criticisms & Technical Limitations"
aliases:
  - Phase 4 Criticisms & Technical Roadmap
  - Q5 Phase 4 Limitations
tags:
  - future-work
  - limitations
  - melanoma
  - patient-stratification
  - phase-4
  - q5
  - ode-modelling
created: 2026-07-31 15:29
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-31 15:29
---

# Phase 4: Methodological Audit, Biological Criticisms & Technical Limitations 🔍

An analytical audit and limitations report for **Phase 4** in the Question 5 Patient Stratification pipeline, evaluating statistical weaknesses, biological assumptions, and computational constraints of cluster profiling, phenotype characterisation, Kaplan–Meier overall survival analysis, and Question 3 Ordinary Differential Equation (ODE) tumour-immune dynamic simulations.

## 1. Executive Summary & Audit Rationale

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Critical evaluation of methodological weaknesses, biological simplifications, and statistical constraints across Phase 4 phenotype characterisation, Kaplan–Meier survival stratification, and Q3 2-state ODE dynamic simulations.
> - **Why we are doing it**: While Phase 4 successfully mapped discovered GMM clusters into four biological phenotypes (*Immune Hot*, *Immune Cold*, *Immunosuppressive M2-High*, *Mutant-Driven*) and integrated dynamic tumour burden trajectories ($T(t)$), rigorous scientific audit requires identifying where aggregated cluster profiles and deterministic ODE systems simplify complex clinical biology.
> - **What question it answers**: What specific statistical and biological limitations constrain Phase 4 phenotype characterisation, and what concrete methodological improvements are required for future pipeline iterations?

Phase 4 bridges unsupervised clustering (Phase 3) with dynamic ODE mechanics (Q3) and subgroup predictive modelling (Phase 5). By profiling clusters across baseline biomarkers (`TIS`, `CYT`, `CD8_T_cells`, `M1_Macrophages`, `M2_Macrophages`, `CAFs`), driver mutations (`BRAF`, `NRAS`, `NF1`), and 180-day simulated tumour volume trajectories $T(t)$, Phase 4 establishes the biological interpretation of the patient stratification framework. However, a rigorous audit reveals critical statistical and biological limitations.

---

## 2. Statistical Weaknesses & Data Constraints ([STATISTICAL_LIMITATIONS])

| Statistical Limitation | Primary Cause | Clinical Impact | Empirical Observation / Risk |
| :--- | :--- | :--- | :--- |
| **Cohort Sample Size Asymmetry** | Imbalanced cluster sizes ($N=22$ to $N=308$). | Reduced statistical power for minority phenotype comparisons. | Cluster 0 (*Immune Cold*, $N=22, 3.1\%$) and Cluster 1 (*Mutant-Driven*, $N=65, 9.3\%$) have far fewer patients than Cluster 2 (*Immune Hot*, $N=304, 43.5\%$) and Cluster 3 (*Immunosuppressive M2-High*, $N=308, 44.1\%$). |
| **Profile Mean Aggregation** | Summarising clusters using central mean Z-scores. | Masks continuous within-cluster variance and heavy-tailed outliers. | Single mean Z-score vectors collapse broad multi-modal distributions into point estimates, ignoring intra-cluster heterogeneity. |
| **Survival Data Censoring & Sub-cohort Loss** | Missing Overall Survival (OS) follow-up across trial cohorts. | Reduces effective $N$ for Kaplan–Meier log-rank testing. | Only a subset of the $N=699$ cohort has complete `OS_MONTHS` and `OS_STATUS` metadata, increasing vulnerability to right-censoring bias. |

### Detailed Statistical Audit

1. **Sample Size Imbalance Across Phenotype Subtypes**:
   - The stratified patient dataset ($N=699$) exhibits extreme group size imbalance: Cluster 0 (*Immune Cold*, $N=22, 3.1\%$) and Cluster 1 (*Mutant-Driven*, $N=65, 9.3\%$) comprise only $12.4\%$ of the total cohort combined, whereas Cluster 2 (*Immune Hot*, $N=304, 43.5\%$) and Cluster 3 (*Immunosuppressive M2-High*, $N=308, 44.1\%$) dominate $87.6\%$ of all patients.
   - This imbalance reduces statistical power when conducting non-parametric Mann–Whitney U or log-rank survival tests on minority clusters, increasing the risk of Type II errors for rare subtype characterisation.

2. **Information Loss from Central Profile Aggregation**:
   - Computing mean feature values per cluster (`compute_phenotype_profiles`) compresses continuous Z-score distributions into single scalar metrics.
   - For heterogeneous features like `CAFs` or `M2_Macrophages`, mean aggregation hides bi-modal sub-populations within a cluster, treating patients with extreme stromal exclusion identically to those with moderate values.

3. **Unadjusted Kaplan–Meier Log-Rank Testing**:
   - The multivariate log-rank test evaluates survival probability differences across phenotype clusters without adjusting for key clinical covariates (age, sex, baseline lactate dehydrogenase [LDH], tumour stage, or prior targeted therapy).
   - Confounding clinical covariates may distort apparent survival differences between phenotypes.

---

## 3. Biological Assumptions & Model Simplifications ([BIOLOGICAL_LIMITATIONS])

> [!WARNING] Biological Simplifications in Dynamic ODE Trajectories
> The current Q5 Phase 4 integration uses a **2-state deterministic ODE system** parameterised by static, phenotype-level average rate constants. This assumes homogeneous dynamic responses within each phenotype arm and omits microenvironmental spatial barriers, antigen presentation loss (`B2M`/`TAP1`), and adaptive therapy resistance.

```
+-----------------------------------------------------------------------------------+
|                        CURRENT PHASE 4 ODE SYSTEM (2-STATE)                      |
|                                                                                   |
|    dT/dt = r * T * (1 - T/K) - c * E * T       [Tumour Growth & Effector Kill]    |
|    dE/dt = s + (p * E * T)/(g + T) - d_E * E - mu * E * T   [Effector Kinetics]   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       BIOLOGICAL CRITICISMS & OMITTED AXES                        |
|                                                                                   |
|  1. Phenotype Homogenisation: Phenotype arms use identical kinetic parameters.    |
|  2. Omitted Stromal Barrier: No CAF physical trapping or spatial exclusion term. |
|  3. Omitted Myeloid Axis: No M1/M2 macrophage immunosuppressive switching.        |
|  4. Omitted Genomic Escape: No `B2M`/`TAP1` antigen presentation loss term.        |
|  5. Static Kinetics: No adaptive resistance, clonal evolution, or dynamic decay.  |
+-----------------------------------------------------------------------------------+
```

### Detailed Biological Audit

1. **Phenotype-Level Parameter Homogenisation**:
   - ODE simulations use fixed scalar parameter vectors ($[r, K, c, s, p, g, d_E, \mu]$) per phenotype cluster (e.g. `[0.18, 1.0, 0.45, 0.10, 0.15, 0.30, 0.05, 0.02]` for *Immune Hot*).
   - This treats every patient in a phenotype cluster as an identical "digital clone", failing to capture individual-level kinetic variation or personalized drug clearance rates.

2. **Omission of Spatial Microenvironment & Myeloid Dynamics**:
   - The 2-state differential system models effector cell activation ($E$) and tumour burden ($T$), but omits explicit state variables for Cancer-Associated Fibroblasts ($CAF$) and M1/M2 macrophage polarisation.
   - Consequently, the observed clinical failure of *Immunosuppressive M2-High* tumours under anti-PD-1 monotherapy is modelled indirectly via a reduced kill rate ($c = 0.08$) rather than explicitly via physical T-cell trapping or IL-10/TGF-$\beta$ cytokine suppression.

3. **Absence of Genomic Escape Mechanisms**:
   - The model assumes effector cells ($E$) retain constant killing efficiency against tumour cells ($T$).
   - In clinical melanoma, acquired or primary resistance frequently arises from genomic antigen presentation loss (mutations in `B2M`, `TAP1`, `TAP2`, or `HLA-A/B/C`) or IFN-$\gamma$ pathway insensitivity (`JAK1`, `JAK2`, `STAT1` loss). The 2-state ODE cannot distinguish between immune cell exhaustion and tumour-intrinsic antigen invisibility.

4. **Static Non-Adaptive Rate Constants**:
   - Kinetic parameters remain static over the entire 180-day simulation window.
   - Real anti-PD-1 monotherapy or combination regimens induce dynamic transcriptomic shifts, clonal selection, and adaptive up-regulation of secondary checkpoints (`HAVCR2`/TIM-3, `LAG3`).

---

## 4. Prioritised Methodological Improvements ([PROPOSED_IMPROVEMENTS])

To address these statistical weaknesses and biological simplifications, future iterations of Phase 4 should execute the following prioritized enhancements:

> [!IMPORTANT] Actionable Improvement Roadmap for Future Iterations
> 1. **Patient-Specific Bayesian ODE Parameterisation**: Transition from cluster-level static ODE parameters to individualised Bayesian parameter estimation ($r_i, c_i, s_i$) derived from each patient's baseline transcriptomic (`TIS`, `CYT`, `M1_M2_Ratio`) and genomic (`mut_BRAF`, `mut_NF1`) profile.
> 2. **Multi-State 4-Module ODE System**: Expand the differential system to incorporate explicit Cancer-Associated Fibroblast density ($C(t)$), M1/M2 macrophage polarisation ratio ($M(t)$), and antigen presentation intactness ($\alpha_{\text{AP}}$).
> 3. **Multivariable Cox Proportional Hazards Regression**: Replace unadjusted Kaplan–Meier log-rank tests with stratified Cox PH modelling adjusting for clinical covariates (age, stage, LDH, TMB).
> 4. **Uncertainty Quantification for Profile Means**: Report interquartile ranges (IQR) and 95% bootstrap confidence intervals alongside cluster feature means.

```
+-----------------------------------------------------------------------------------+
|                        PROPOSED 4-MODULE ODE EXPANSION                            |
|                                                                                   |
|  dT/dt = r * T * (1 - T/K) - c * alpha_AP * E * T / (1 + beta_CAF * C)            |
|  dE/dt = s + (p * E * T)/(g + T) - d_E * E - mu_M2 * M2 * E                       |
|  dC/dt = r_C * C * (1 - C/K_C) + gamma_T * T                                      |
|  dM2/dt = r_M * M2 * (1 - M2/K_M) + theta_supp * T - phi_rescue * D_rescue        |
+-----------------------------------------------------------------------------------+
```

### Prioritised Action Items

| Priority | Improvement Item | Target Module / Script | Rationale & Expected Benefit |
| :---: | :--- | :--- | :--- |
| **P1** | **Patient-Specific ODE Parameterisation** | `04_phenotype_characterisation.py` / `ode_models.py` | Replaces cluster-wide static ODE trajectories with $N=699$ individual patient digital twin simulations based on empirical expression scores. |
| **P2** | **4-Module ODE System Expansion** | `q3-ode-model/src/ode_models.py` | Explicitly models CAF physical exclusion ($C$), M2 macrophage suppression ($M_2$), and `B2M`/`TAP1` antigen intactness ($\alpha_{\text{AP}}$). |
| **P3** | **Multivariable Stratified Cox PH Regression** | `04_phenotype_characterisation.py` | Adjusts phenotype survival hazard ratios ($HR$) for age, tumour stage, LDH, and driver mutation status. |
| **P4** | **Bootstrap Confidence Intervals for Profiles** | `src/phenotyping.py` | Provides 95% bootstrap confidence intervals for cluster biomarker means, resolving profile mean point-estimate limitations. |

---

## 5. Key Takeaways & Student Summary 🎓

- **Cluster Profiling Characterises Biological Phenotypes**: Phase 4 successfully mapped $N=699$ patients into four actionable phenotype profiles (*Immune Hot*, *Immune Cold*, *Immunosuppressive M2-High*, *Mutant-Driven*), demonstrating distinct response rates ($19.4\%$ to $60.4\%$) and overall survival trends.
- **2-State ODE Systems Model Phenotype Tumour Burden**: Phenotype-specific 180-day ODE simulations capture tumour clearance in inflamed subtypes versus uncontrolled growth in immunosuppressive subtypes.
- **Statistical Imbalance & Profile Aggregation Constraints**: Imbalanced cluster sizes ($3.1\%$ to $44.1\%$) and mean scalar Z-score aggregation collapse intra-cluster continuous variance.
- **Biological Simplifications Require Multi-Module Extensions**: 2-state ODE models omit explicit CAF physical barriers, M1/M2 macrophage switching, and `B2M`/`TAP1` antigen presentation loss, highlighting the need for 4-module patient-specific digital twin extensions in future iterations.

---

> [!formula] Phase 4 Script Execution & Software Module Architecture
> - **Primary Pipeline Execution Scripts**:
>   - [`04_phenotype_characterisation.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/04_phenotype_characterisation.py): Annotates and characterises discovered patient phenotypes, calculates summary profiles across immune signatures and cell deconvolution, integrates Q3 ODE tumour dynamics simulations ($T(t)$ trajectories) over 180 days, performs log-rank Kaplan–Meier survival analysis, and exports `phenotype_characterisation.csv`, `baseline_signature_boxplots.png`, `ode_trajectories.png`, and `km_survival_by_phenotype.png`.
> - **Core Supporting Python Modules**:
>   - [`phenotyping.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/phenotyping.py): Implements cluster profiling (`profile_clusters`), rank-based phenotype label assignment (`assign_phenotype_labels`), facetted biomarker violin plots (`plot_baseline_signature_boxplots`), radar chart comparison (`plot_radar_chart`), and annotated patient heatmap (`plot_cluster_heatmap`).
>   - [`q5_constants.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/q5_constants.py): Central source of truth defining clustering features (`CLUSTERING_FEATURES`), phenotype profile features (`PHENOTYPE_PROFILE_FEATURES`), and phenotype label mappings (`PHENOTYPE_LABELS`).
> - **Shared Cross-Question & Pipeline Modules**:
>   - [`run_q5_pipeline.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/run_q5_pipeline.py): Master pipeline orchestrator executing `04_phenotype_characterisation.py` as Step 4.
>   - [`generate_q5_report.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/generate_q5_report.py): Reads phenotype characterisation summaries and updates phase markdown reports.
