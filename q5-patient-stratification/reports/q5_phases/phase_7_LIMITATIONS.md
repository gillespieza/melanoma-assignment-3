---
title: "Phase 7 Methodological Evaluation & Limitations Report"
aliases:
  - Phase 7 Limitations
  - Q5 Phase 7 Methodological Appraisal
tags:
  - melanoma
  - patient-stratification
  - phase-7
  - limitations
  - treatability-scoring
created: 2026-08-01 21:33
cssclasses:
  - table-small
  - table-center
  - row-alt
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-01 21:33
---

# Phase 7: 3-Arm Decision Support & Treatability Scoring — Methodological Appraisal & Limitations Report

> [!WARNING] Methodological Scope & Evaluation Notice
> This document provides a rigorous, graduate-level critical appraisal of Phase 7 (`07_treatability_scoring.py`) in its current state. It evaluates the statistical assumptions, biological simplifications, dataset constraints, and software architecture boundary conditions across $N = 699$ patients ($N = 326$ retrospective ICI-treated patients with known response annotations, $N = 373$ prospective un-annotated patients).

## 1. Statistical & Methodological Weaknesses

> [!WARNING] Mathematical Formulation & Decision Tree Weaknesses
> - **What is being evaluated**: The mathematical properties of the empirical Treatability Index ($0 – 100$), decision tree routing rules, and recommendation confidence scoring.
> - **Why it matters**: Clinical decision support frameworks must maintain robust statistical separation without introducing artificial boundary penalties or over-interpreting non-linear biomarker interactions.

### 1.1 Non-Linear Quantile Scaling Distribution Flattening
The Treatability Index utilizes a rank-preserving uniform quantile transformation (`QuantileTransformer(output_distribution='uniform')`) across raw composite scores. While this eliminates single-patient boundary outlier compression and forces the interquartile range ($\text{IQR}$) to span exactly $50.00$ units ($\text{Q1} = 25.00, \text{Q3} = 75.00$, $\text{Median} = 50.00, \text{Mean} = 50.00$), uniform quantile scaling transforms continuous raw score distances into a flat uniform density $U[0, 100]$. Consequently, metric distance between patients in the central distribution reflects relative sample rank order rather than absolute biological difference in immune infiltrate intensity.

### 1.2 Binary L2-Logistic Regression Weight Estimation
Empirical weights ($w_{\text{AgPres}} = -0.0247, w_{\text{IFN}} = 0.3751, w_{\text{Effector}} = 0.1096, w_{\text{Barrier}} = -0.2966$) are estimated using L2-regularised logistic regression ($C = 1.0$) trained on $N = 195$ response-annotated ICI-treated patients (`RESPONSE_BINARY`). While this significantly outperforms static heuristic weights ($\text{ROC-AUC} = 0.5956, p = 0.0228$), fitting weights against a binary response outcome ignores time-to-event censorship in overall survival ($\text{OS}$) and progression-free survival ($\text{PFS}$). Furthermore, negative weighting on antigen presentation ($-0.0247$) indicates collinearity with `IFN_gamma` signaling ($+0.3751$) in unadjusted multi-variable models.

### 1.3 Soft Sigmoidal Boundary Transition & Equipoise Zone Implementation
To resolve deterministic cliff-edge effects in Arm C sub-arm selection, a logistic sigmoid transition function ($w_{\text{AXL}} = \frac{1}{1 + \exp(-0.2 \cdot (\text{TI} - 40.0))}$) and an explicit equipoise buffer zone ($[35.0, 45.0]$) have been implemented. For patients within the buffer zone, recommendations report dual candidate probabilities rather than a sharp binary switch. However, discrete boundaries remain at the top level between Arm A ($q = 0.60$ `TIS` threshold) and Arm B driver mutation overrides.

### 1.4 Continuous Evidence-Based Confidence Scoring (NRAS Cap Removed)
Recommendation confidence scores ($0 – 100$) are categorised into three continuous bands: *High* ($\ge 0.70$), *Moderate* ($0.45 – 0.70$), and *Low* ($< 0.45$). To eliminate hard step penalties, the artificial ceiling capping `NRAS`-mutant patients at *Moderate* status was removed. `NRAS` status is weighted continuously via `MUT_STRENGTH_NRAS` ($0.70$ vs $1.00$ for `BRAF` V600). Patients with `NRAS` mutations presenting with exceptional Q2 target sensitivity ($>88/100$) can now reach *High* confidence status ($N = 173$ high-confidence patients total, $+10$ pts), reflecting evidence-based patient-specific scoring.

---

## 2. Biological & Clinical Assumptions

> [!WARNING] Biological Simplifications & Clinical Translation Assumptions
> - **What is being evaluated**: Theoretical assumptions regarding microenvironmental reversal, driver mutation hierarchy, and cell line target translation.
> - **Why it matters**: Translating computational subtype nominations into clinical decision support requires accounting for in vivo resistance mechanisms and drug toxicity constraints.

### 2.1 Monotherapy vs Combination Synergy Assumptions
Arm C ($N = 101$, $14.4\%$ of total cohort) nominates combination reversal strategies:
- *Immunosuppressive M2-High*: Anti-PD-1 + `CSF1R` inhibitor (Pexidartinib) for TAM reprogramming.
- *Mutant-Driven*: Anti-PD-1 + `MDM2` antagonist (Idasanutlin) for `p53` reactivation.
- *Immune Cold*: Anti-PD-1 + `AXL` inhibitor (Bemcentinib) or `HDAC` inhibitor + chemotherapy.

This framework assumes additive therapeutic efficacy when pairing immune checkpoint blockade with microenvironmental modulators. In clinical settings, combination immunotherapies frequently encounter overlapping toxicity profiles, Grade 3–4 immune-related adverse events (irAEs), and counter-regulatory immunosuppressive feedbacks (such as compensatory upregulation of `TIM-3` or `LAG3`).

### 2.2 Hierarchical Driver Mutation Override
The decision tree prioritises driver mutation status (`BRAF` V600 and `NRAS`) in Arm B ($N = 234$) over underlying microenvironmental phenotype features for non-Arm A candidates. Among $N = 256$ patients in the *Immunosuppressive M2-High* phenotype, $N = 173$ ($67.6\%$) are routed to Arm B due to co-occurring `BRAF` or `NRAS` mutations. Routing M2-macrophage-dense tumours primarily to targeted kinase inhibitors assumes oncogenic MAPK signaling overrides TAM-mediated immunosuppression, ignoring potential resistance driven by dense stromal barriers.

### 2.3 Cell-Line DepMap Target Translation
Arm C target nominations (`CSF1R`, `MDM2`, `AXL`) rely on Q4 DepMap cancer dependency screens and LINCS perturbational signatures. DepMap essentiality scores are derived from in vitro monoculture cell lines lacking an intact immune system, functional vasculature, or spatial tissue architecture. Translating monoculture essentialities to human in vivo tumour microenvironments assumes bulk transcriptomic proxy scores reflect cell-type-specific protein expression and cell-cell spatial interactions.

---

## 3. Computational & Data Constraints

> [!WARNING] Dataset Resolution & Cohort Heterogeneity Constraints
> - **What is being evaluated**: Cohort distribution, prospective dataset integration, and confidence score proportions across $N = 699$ patients.
> - **Why it matters**: Clinical utility estimates are constrained by sample sizes, retrospective sequencing batch effects, and un-annotated prospective cohorts.

### 3.1 Retrospective Cohort Heterogeneity ($N = 699$)
Phase 7 evaluates patient stratification across $N = 699$ total samples pooled from four clinical cohorts:
- $N = 326$ Retrospective ICI-Treated Patients (known clinical response annotations): Arm A = $185$ ($56.7\%$), Arm B = $85$ ($26.1\%$), Arm C = $56$ ($17.2\%$).
- $N = 373$ Prospective Un-annotated Patients (TCGA-SKCM benchmark): Arm A = $179$ ($48.0\%$), Arm B = $149$ ($39.9\%$), Arm C = $45$ ($12.1\%$).

Integrating multi-study transcriptomic datasets introduces technical batch effects, variation in sequencing depth, and biopsy timing differences (pre-treatment baseline vs on-treatment).

### 3.2 Moderate-to-Low Recommendation Confidence Proportion ($75.3\%$)
Across the $N = 699$ cohort, recommendation confidence scoring yields the following distribution:
- **High Confidence** ($\ge 0.70$): $N = 173$ patients ($24.7\%$)
- **Moderate Confidence** ($0.45 – 0.70$): $N = 306$ patients ($43.8\%$)
- **Low Confidence** ($< 0.45$): $N = 220$ patients ($31.5\%$)

Three-quarters ($75.3\%$) of all evaluated patients receive *Moderate* or *Low* confidence recommendations. This highlights biological ambiguity for patients presenting with intermediate `TIS` scores, borderline driver mutation VAFs, or mixed M1/M2 macrophage infiltrate ratios.

---

## 4. Code Quality & Software Architecture Evaluation

> [!WARNING] Code Quality & Software Engineering Appraisal
> - **What is being evaluated**: Module structure, function complexity, type safety, and architectural decoupling in `07_treatability_scoring.py`.
> - **Why it matters**: Clean code guarantees maintainability, reproducibility, and seamless integration with downstream pipelines and dashboards.

### 4.1 Modular Quality & AST Conformance
The current implementation in `07_treatability_scoring.py` satisfies strict software engineering standards:
- **Function Line-Length Compliance**: All $36$ functions in the script are strictly $\le 30$ lines long.
- **Import Organisation**: All imports (`sklearn`, `pandas`, `numpy`, `matplotlib`) are positioned at the module top level; no deferred lazy imports remain inside function bodies.
- **Type Safety**: Function signatures carry complete Python type annotations, including `Optional[Tuple[...]]` for defaulted parameters.
- **Centralised Constants**: Phenotype display strings (`PHENO_NAME_M2_SHORT`, `PHENO_NAME_M2_HIGH`, `PHENO_NAME_IMMUNE_COLD`, `PHENO_NAME_IMMUNE_HOT`), confidence thresholds (`CONF_HIGH_THRESHOLD`, `CONF_MOD_THRESHOLD`), and sigmoidal parameters (`SIGMOID_MIDPOINT`, `SIGMOID_STEEP_K`, `EQUIPOL_LOWER_BOUND`, `EQUIPOL_UPPER_BOUND`) are defined in the module-level constants block.

### 4.2 Architectural Decoupling Opportunity
Currently, `07_treatability_scoring.py` combines core decision rules (`assign_treatment_arms`, `compute_recommendation_confidence`, `calculate_treatability_index`) with standalone CLI script orchestration and file I/O. Extracting decision functions into a dedicated module within `src/treatability.py` would allow external interactive web dashboards or sensitivity engines to import and execute scoring logic without triggering script-level file reads.

---

## 5. Key Takeaways & Prioritised Future Directions

> [!INSIGHT] Key Insights & Prioritised Actionable Improvements
> 1. **Survival-Based Cox Proportional Hazards Weighting (High Priority)**: Upgrade binary L2 logistic regression weights ($N = 195$) to a multi-variable Cox proportional hazards survival model incorporating time-to-event overall survival ($\text{OS}$) and progression-free survival ($\text{PFS}$).
> 2. **Probabilistic Soft Multi-Arm Assignment (High Priority)**: Extend sigmoidal sub-arm smoothing to top-level Arm A/B/C routing, replacing deterministic conditional branching (`if/else`) with continuous multi-arm membership probabilities that integrate Phase 3 GMM posterior probabilities ($P_{\text{Immune\_Hot}}$, $P_{\text{M2\_High}}$, $P_{\text{Mutant\_Driven}}$, $P_{\text{Immune\_Cold}}$) with Q2 Dabrafenib sensitivity scores.
> 3. **Non-Linear Sigmoid / Rank Transformation Hybrids (Medium Priority)**: Replace flat uniform quantile transformation ($U[0, 100]$) with a hybrid logistic-sigmoid transformation that preserves relative biological distance near clinical decision boundaries while capping extreme outliers.
> 4. **Decoupled Engine Module (`src/treatability.py`) (Medium Priority)**: Move core decision logic from `07_treatability_scoring.py` into a reusable `src/treatability.py` package to support lightweight real-time prediction in interactive dashboards.
