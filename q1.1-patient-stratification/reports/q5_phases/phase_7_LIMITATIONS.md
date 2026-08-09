---
title: "Phase 7: Limitations, Assumptions & Future Directions"
aliases:
  - Q5 Phase 7 Limitations
tags:
  - melanoma
  - patient-stratification
  - phase-7
  - q5
  - limitations
created: 2026-08-01 21:44
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-01 21:44
---

## Phase 7: Limitations, Assumptions & Future Directions

> [!NOTE] Scope & Purpose of This Appraisal
> - **What is being appraised**: The current state of Phase 7 (`07_treatability_scoring.py`), which routes $N = 699$ patients into three therapeutic arms and produces Treatability Index scores, Q4 target nominations, and Recommendation Confidence bands.
> - **Why**: No analytical pipeline is free of assumptions and constraints. A rigorous self-appraisal is essential for academic transparency, for calibrating downstream clinical confidence, and for prioritising the highest-impact improvements in future iterations.
> - **What question it answers**: What are the current methodological, statistical, biological, and data constraints of Phase 7, and where should effort be focused next?

---

## 1. Statistical Weaknesses

### 1.1 Moderate Treatability Index Discriminative Power

The Treatability Index achieves a ROC-AUC of **0.5956** on the $N = 195$ response-annotated patients used for L2-regularised logistic regression weight fitting. This is only marginally above the null (AUC = 0.50), indicating that the current four-component index (`B2M`/`TAP1` antigen presentation, `IFN_gamma`, `CD8_Tcell`, `M2_score`) has limited independent predictive value for binary immunotherapy response. The index is useful for biological stratification and rank-ordering, but should not be interpreted as a clinically validated response predictor.

> [!WARNING] Limitation 1.1
> A Treatability Index ROC-AUC of **0.5956** on $N = 195$ patients is insufficient for clinical decision support in isolation. External prospective validation in an independent cohort is required before any clinical translation is attempted.

### 1.2 Small Empirical Weight Fitting Sample

The L2-regularised logistic regression that derives sub-score weights is fitted on only **195 response-annotated patients** out of the full $N = 699$ cohort. The remaining 504 patients ($326$ ICI-treated with known response but filtered on data availability, $373$ prospective TCGA-SKCM) do not contribute to weight estimation. This is a small, potentially selection-biased training set given the 4-dimensional feature space and the modest signal in individual components.

> [!WARNING] Limitation 1.2
> The 4-component logistic regression is fitted on $N = 195$ samples. This yields approximately 49 samples per predictor variable — within acceptable range, but borderline for L2-regularised logistic regression. Cross-validated weight stability estimates have not been computed; it is unknown whether the derived weights ($w_{\text{AgPres}} = -0.0247$, $w_{\text{IFN}} = +0.3751$, $w_{\text{Effector}} = +0.1096$, $w_{\text{Barrier}} = -0.2966$) are stable across bootstrap resamples.

### 1.3 Confidence Index Asymmetry Across Arms

The Recommendation Confidence Index is structurally asymmetric across the three arms. Per the live output:

| Arm | Mean Confidence | High ($N$) | Moderate ($N$) | Low ($N$) |
|-----|----------------|-----------|--------------|---------|
| Arm A: Immunotherapy ($N = 364$) | 47.6 / 100 | 37 | 137 | 190 |
| Arm B: Targeted Therapy ($N = 234$) | 69.0 / 100 | 119 | 111 | 4 |
| Arm C: Combination/Reversal ($N = 101$) | 57.0 / 100 | 17 | 58 | 26 |

Arm A patients — the largest arm at 52.1% of the cohort — receive a mean confidence of only **47.6/100**, with **190 patients (52.2% of Arm A)** rated Low confidence. This reflects that high-confidence immunotherapy recommendation currently requires both strong *Immune Hot* phenotype membership and a high Treatability Index, conditions many *Immune Hot* patients satisfy by phenotype but not index score. This asymmetry may paradoxically assign the lowest confidence to the arm with the strongest biological rationale.

> [!WARNING] Limitation 1.3
> Arm A has the lowest mean confidence index (47.6/100) despite immunotherapy being the most evidence-backed intervention for *Immune Hot* melanoma. The confidence formula should be re-examined to ensure that phenotype strength (GMM posterior $P_{\text{Immune Hot}}$) contributes more directly to Arm A confidence scoring.

---

## 2. Biological Assumptions

### 2.1 Q2 Dabrafenib Sensitivity Generalisation to Patient Transcriptomics

Arm B Dabrafenib Sensitivity Index scores (mean = **56.0/100**, std = **14.2**, range = **11.1–100.0**) are derived from a Q2 LASSO model trained on 24 gene features from CCLE cancer cell line viability data. Applying cell-line-derived gene expression weights to bulk patient RNA-seq transcriptomics assumes that the transcriptomic predictors of Dabrafenib sensitivity in controlled cell culture transfer directly to the complex, heterogeneous in vivo tumour microenvironment. This is a strong assumption: stromal contamination, tumour purity variation, and cell-line-specific growth conditions all reduce the validity of this transfer.

> [!WARNING] Limitation 2.1
> The Q2 Dabrafenib Sensitivity Index applies cell-line-derived LASSO weights to bulk patient RNA-seq. No direct validation against patient clinical response to Dabrafenib has been performed. Arm B scores should be treated as a relative rank-ordering of drug sensitivity, not an absolute sensitivity prediction.

### 2.2 Arm C Q4 Target Nominations Are Not Patient-Personalised

`CSF1R` (M2 TAM Depletion) is nominated for **67/101 (66.3%)** of Arm C patients, `MDM2` for **21/101 (20.8%)**, `HDAC`/Epigenetic Remodeling for **10/101 (9.9%)**, and `AXL`/STING Pathway for only **3/101 (3.0%)**. These nominations derive from Q4 DepMap CRISPR essentiality data mapped to tumour phenotype, not from individual patient CRISPR screens or personalised genomic data. The same `CSF1R` nomination applies uniformly to all *M2 Immunosuppressive* patients in Arm C regardless of their individual M2 macrophage burden, `CSF1R` expression level, or co-occurring genetic alterations.

> [!WARNING] Limitation 2.2
> Q4 target nominations are phenotype-level rather than patient-level. A patient in the *M2 Immunosuppressive* sub-group receives `CSF1R` as their nominated target irrespective of their individual `CSF1R` expression, M2 macrophage score, or co-occurring driver mutation status. Patient-level `CSF1R` expression quantiles are not currently incorporated into nomination logic.

### 2.3 Arm Boundaries Are Deterministic on Phenotype, Not Probability-Weighted

Arm assignment uses a deterministic decision tree based on GMM-derived `Phenotype_Label` (the modal cluster assignment). However, many patients have non-trivial posterior probabilities across multiple phenotypes — particularly patients near cluster boundaries. The live arm summary confirms that **23 non–Immune-Hot patients are assigned to Arm A** (16 *M2 Immunosuppressive*, 6 *Mutant-Driven*, 1 *Immune Cold*), and similarly **173 M2-Immunosuppressive patients go to Arm B**. These allocations may be appropriate given mutation status, but the routing logic ignores uncertainty in the GMM posterior: a patient with $P_{\text{Immune Hot}} = 0.52$ and $P_{\text{M2}} = 0.48$ is treated identically to one with $P_{\text{Immune Hot}} = 0.99$.

> [!WARNING] Limitation 2.3
> Arm routing is deterministic on modal phenotype label. GMM posterior uncertainty is not propagated into arm assignment probability or confidence scoring. Patients with borderline phenotype membership are allocated with the same confidence as high-certainty members.

### 2.4 Immune Cold Sub-Arm Sigmoidal Parameters Are Heuristic

The sigmoidal boundary transition for Arm C *Immune Cold* sub-arm selection (midpoint $= 40.0$, steepness $k = 0.2$, equipoise zone $[35.0, 45.0]$) was calibrated by analytical reasoning rather than from empirical data. No optimisation of these parameters against patient outcome data has been performed. The shape and midpoint of the sigmoidal curve are therefore approximations that impose a smooth transition where the true biological decision boundary is unknown.

---

## 3. Computational & Data Constraints

### 3.1 Weight Fitting Uses Only ICI-Treated, Response-Annotated Patients

The L2 logistic regression is fitted exclusively on the **195 patients** with a non-null `RESPONSE_BINARY` label. These are drawn from the three ICI clinical trial cohorts (Liu 2019, Hugo 2016, Riaz 2017) and represent a specific subpopulation: patients enrolled in immune checkpoint inhibitor trials, predominantly anti-PD-1 monotherapy. The **373 TCGA-SKCM prospective patients** — who comprise **53.4%** of the full cohort — contribute no information to weight fitting despite being the majority. The fitted weights may be systematically biased towards trial-eligible patient characteristics (performance status, prior treatment, tissue biopsy timing).

> [!WARNING] Limitation 3.1
> Weight fitting on $N = 195$ trial-enrolled patients risks systematic bias: TCGA-SKCM ($N = 373$), which constitutes the majority of the cohort, is entirely excluded from weight calibration. Treatability Index weights are calibrated on a trial-selected, survival-enriched, ICI-treated population and may not generalise to the broader TCGA-SKCM prospective subset.

### 3.2 No Cross-Validated Confidence Interval on Treatability Weights

The empirical logistic regression weights are reported as point estimates ($w_{\text{AgPres}} = -0.0247$, $w_{\text{IFN}} = +0.3751$, $w_{\text{Effector}} = +0.1096$, $w_{\text{Barrier}} = -0.2966$) with no bootstrap confidence intervals or cross-validated stability metrics. Given the small sample ($N = 195$), the width of the 95% confidence interval around these weights could be substantial, particularly for `AgPres` which is near zero. A weight whose confidence interval straddles zero does not provide reliable directional guidance.

### 3.3 No Survival Outcome Integration

Phase 7 routes patients based on immunotherapy response prediction (binary CR/PR vs PD), Dabrafenib sensitivity index, and TME phenotype. It does not incorporate overall survival (OS) or progression-free survival (PFS) data, despite OS months and OS status columns being present in the dataset. Treatment arm allocation that maximises short-term response probability may not maximise survival benefit, particularly for Arm B (`BRAF`-targeted) patients where acquired resistance to BRAF inhibition is a well-established clinical challenge.

> [!WARNING] Limitation 3.3
> Arm assignments are optimised for predicted immunotherapy response (binary), not survival. OS and PFS columns are available in `treatability_scores.csv` but are not integrated into the routing or confidence logic. A survival-weighted routing objective would provide a more clinically meaningful decision criterion.

### 3.4 Arm C `AXL`/STING Pathway Arm Is Critically Under-Populated

Only **3/101 (3.0%)** of Arm C patients are nominated for `AXL`/STING pathway priming. This small subgroup is insufficient to draw any statistically robust conclusions about the effectiveness of this nomination, and the sigmoidal Equipoise Buffer Zone ($[35.0, 45.0]$) that governs `AXL` vs `HDAC` sub-arm selection affects only the minority of Arm C patients (those with Treatability Index in the transition zone). The practical clinical utility of the `AXL` nomination cannot be assessed with $N = 3$.

---

## 4. Code Quality

> [!NOTE] Code Quality Status
> Phase 7 (`07_treatability_scoring.py`) has undergone four refactoring passes as documented in `PROJECT_MAP.md`. The current state is:
> - **Function length compliance**: 36/36 functions $\le 30$ lines ✅
> - **AST smell audit**: 0 code smells detected ✅
> - **DRY compliance**: Shared z-score, min-max, and phenotype-column helpers in place ✅

### 4.1 `treatability.py` Src Module Is a Stub

The software architecture callout in `phase_7.md` references `src/treatability.py` as implementing `assign_treatment_arms`, `compute_treatability_index`, and plotting functions. In the current pipeline, all of this logic lives directly in `07_treatability_scoring.py`. The `src/treatability.py` module either does not exist or is an empty stub, meaning the architecture documentation overstates the modularity of the current implementation.

### 4.2 No Unit Tests for Arm Routing Logic

The 3-arm decision tree, sigmoidal boundary weights, and confidence banding thresholds are not covered by any automated unit tests. Refactoring or future parameter changes (e.g., adjusting `CONF_HIGH_THRESHOLD`, `SIGMOID_MIDPOINT`, or `EQUIPOL_LOWER_BOUND`) carry an unquantified risk of silently altering patient routing outcomes without triggering any test failure.

### 4.3 Arm B `NRAS` Confidence Lacks Biological Sub-Stratification

`NRAS`-mutant patients in Arm B ($N = 96$) are routed to targeted therapy with a single continuous confidence penalty (`MUT_STRENGTH_NRAS = 0.70`) applied uniformly to all `NRAS` subtypes. `NRAS` mutations are clinically and biologically heterogeneous — `NRAS Q61` activating mutations confer different MEK/ERK signalling intensities than rarer `NRAS` variants. No sub-stratification by `NRAS` codon is currently applied, meaning confidence weighting conflates clinically distinct `NRAS` disease subsets.

---

> [!INSIGHT] Key Insights & Prioritised Future Directions
> The following improvements are ranked by estimated scientific impact:
>
> **Priority 1 — Statistical**
> - **Bootstrap weight stability analysis**: Compute 1,000-sample bootstrap confidence intervals on the four logistic regression weights. Any weight whose 95% CI crosses zero should be flagged as unreliable and considered for removal from the index.
> - **Survival-weighted routing objective**: Integrate OS/PFS data as a secondary routing criterion to move beyond binary response prediction towards survival-optimised arm allocation.
>
> **Priority 2 — Biological**
> - **Patient-level `CSF1R` expression integration**: Replace phenotype-level Q4 target nominations with patient-level nomination logic using quantile-stratified `CSF1R` / `AXL` / `MDM2` expression from the patient RNA-seq matrix.
> - **GMM posterior uncertainty propagation**: Replace deterministic phenotype-label-based arm routing with a probability-weighted allocation (e.g., expected arm assignment $= \sum_k P_k \cdot \text{Arm}(k)$) to propagate cluster uncertainty into routing decisions.
> - **Arm A confidence recalibration**: Incorporate `P_Immune_Hot` GMM posterior probability directly into Arm A confidence scoring to prevent *Immune Hot* patients with high phenotype certainty from receiving Low confidence assignments.
>
> **Priority 3 — Computational**
> - **External cohort validation**: Validate the Treatability Index on an independent ICI trial cohort not included in this analysis (e.g., Hellmann 2018 or Snyder 2014) to assess generalisation beyond the three training cohorts.
> - **Unit test coverage**: Implement parametric unit tests for arm routing, confidence banding, and sigmoidal weighting functions to ensure refactoring safety.
> - **`src/treatability.py` modularisation**: Migrate arm assignment and scoring logic from `07_treatability_scoring.py` into `src/treatability.py` to match the documented architecture and enable reuse across pipeline steps.
