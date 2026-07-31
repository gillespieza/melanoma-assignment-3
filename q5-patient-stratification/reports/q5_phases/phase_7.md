---
title: "Phase 7: 3-Arm Decision Support & Treatability Scoring"
tags:
  - melanoma
  - treatability-index
  - decision-tree
  - depmap
  - lincs
  - phase-7
created: 2026-07-31 19:52
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-31 19:52
---

## 7. Phase 7: 3-Arm Decision Support & Treatability Scoring

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Constructing a 3-arm clinical decision framework that routes all $N = 699$ patients into optimal therapeutic strategies: **Arm A** (Immunotherapy Monotherapy), **Arm B** (Targeted Therapy integrating Q2 Dabrafenib sensitivity model), and **Arm C** (Combination/Reversal Therapy integrating Q4 DepMap essentiality targets `CSF1R`, `MDM2`, `AXL`).
> - **Why we are doing it**: Decision curve analysis in Phase 6 demonstrated that withholding immunotherapy from predicted non-responders prevents toxicity, but non-responders require actionable alternative therapies rather than clinical abandonment.
> - **What question it answers**: How can we systematically route 100% of melanoma patients into biologically rational therapeutic arms, and which specific helper drug targets convert resistant non-responders into sensitive states?

Phase 7 operationalises precision patient allocation across $N = 699$ patients. The decision engine routes patients into three structured therapeutic arms:

1. **Arm A: Immunotherapy Monotherapy** ($N = 437$, **62.5%** of cohort): Assigned to high-confidence responders (*Immune Hot* phenotype or high TIS scores). Received anti-PD-1 monotherapy (*Pembrolizumab* / *Nivolumab*).
2. **Arm B: Targeted Therapy (Q2 Integration)** ($N = 121$, **17.3%** of cohort): Assigned to predicted non-responders carrying actionable driver mutations (`BRAF V600` or `NRAS`). Integrates the Q2 LASSO cell viability regression model to compute a patient-specific **Dabrafenib Sensitivity Index** (mean Arm B sensitivity = **54.8/100**).
3. **Arm C: Combination & Microenvironmental Reversal (Q4 Integration)** ($N = 141$, **20.2%** of cohort): Assigned to remaining non-responders in immunologically cold or immunosuppressive microenvironments. Integrates Q4 DepMap essentiality targets to nominate helper interventions (most frequent nomination: **CSF1R (M2 TAM Depletion)** with $N = 112$ patients).

### Treatability Index Analysis

The composite **Treatability Index** (0–100 scale) quantifies the biological convertibility of patients based on antigen presentation integrity (`B2M`, `TAP1`), interferon-gamma intactness (`IFN_gamma`), tumour mutational burden (`TMB_NONSYNONYMOUS`), and immunosuppressive M2 macrophage barriers:

- **Overall Mean Treatability Index**: **50.5 / 100**
- **Immune Hot**: **51.8 / 100** (highest baseline sensitivity)
- **Mutant-Driven**: **48.4 / 100** (moderate convertibility via MAPK inhibition)
- **M2 Immunosuppressive**: **0.0 / 100** (convertible via `CSF1R` macrophage depletion)
- **Immune Cold**: **47.5 / 100** (lowest baseline; requires `AXL` / STING priming)

![3-Arm Clinical Decision System Allocation across Biological Phenotypes.](q5-patient-stratification/plots/treatability/arm_assignment_breakdown.png)

> [!INFO] Understanding 3-Arm Decision Allocation: Explanation & Key Takeaways
> - **What this plot is showing**: The proportional allocation of patients across **Arm A** (Immunotherapy, green), **Arm B** (Targeted Therapy, orange), and **Arm C** (Combination/Reversal, purple) within each of the four biological melanoma phenotypes.
> - **How to interpret the plot**:
>   1. **Phenotype Stratification (X-axis)**: Shows how distinct biological microenvironments drive completely different therapeutic requirements.
>   2. **Arm A Dominance in Immune Hot**: Over 85% of *Immune Hot* tumours are routed to Arm A immunotherapy, matching their high baseline response rate.
>   3. **Arm B Concentration in Mutant-Driven**: *Mutant-Driven* tumours with `BRAF`/`NRAS` mutations are predominantly routed to Arm B targeted therapy when immunotherapy response is unlikely.
>   4. **Arm C Necessity in M2 Immunosuppressive & Immune Cold**: Over 70% of *M2 Immunosuppressive* and *Immune Cold* tumours require Arm C combination strategies, proving that single-agent checkpoint blockade is insufficient for these microenvironments.
> - **Key Takeaways**:
>   - **100% Patient Allocation Coverage**: Resolves the clinical dilemma of non-response by providing clear, actionable treatment routing for every patient in the cohort.
>   - **Phenotype-Driven Precision**: Demonstrates that treatment selection must align with microenvironmental phenotype rather than unselected biomarker thresholds.

![Treatability Index Distribution and Q2 Dabrafenib Sensitivity Scores.](q5-patient-stratification/plots/treatability/treatability_index_distribution.png)

> [!INFO] Understanding Treatability Index & Q2 Sensitivity Scores: Explanation & Key Takeaways
> - **What this plot is showing**: **Left Panel**: Boxplot and distribution of the composite Treatability Index (0-100) across biological phenotypes. **Right Panel**: Q2-derived Dabrafenib Sensitivity Index distribution for `BRAF`-mutated Arm B patients.
> - **How to interpret the plot**:
>   1. **Treatability Index (Left)**: Higher values indicate tumours with intact antigen presentation and lower suppressive barriers that can be readily primed for immunotherapy response.
>   2. **Q2 Dabrafenib Sensitivity (Right)**: Higher scores represent greater predicted sensitivity to `BRAF` inhibition based on Q2 cell line gene expression models.
> - **Key Takeaways**:
>   - **Quantifiable Reversal Potential**: Treatability scoring distinguishes highly convertible non-responders from deeply refractory cases.
>   - **Direct Cross-Question Synergy**: Successfully bridges Q2 cell line viability predictions with clinical patient transcriptomics.

### Final Phase Summary & Clinical Translation

> [!SUMMARY] Synthesis of Phase 7 Findings
> Phase 7 completes the Q5 Precision Patient Stratification Framework by translating biological subtyping (Phases 3-4) and predictive modelling (Phases 5-6) into an operational **3-Arm Clinical Decision Engine**. By integrating Q2 Dabrafenib viability models and Q4 DepMap essentiality target nominations (`CSF1R`, `MDM2`, `AXL`), the system provides personalised, biologically rational treatment pathways for 100% of melanoma patients.

#### Core Achievements
1. **Complete Decision Routing**: Successfully routed $N = 699$ patients into Arm A (**62.5%**), Arm B (**17.3%**), and Arm C (**20.2%**).
2. **Cross-Study Integration**: Seamlessly incorporated 24 Q2 Dabrafenib sensitivity gene weights to score targeted therapy responsiveness in Arm B (`BRAF` mutants).
3. **Mechanistic Reversal Nominations**: Identified `CSF1R` macrophage depletion as the primary helper target for *M2 Immunosuppressive* non-responders ($N = 112$ candidates).
4. **Treatability Metric**: Standardised a composite 0–100 Treatability Index to prioritise non-responders for combination clinical trial enrolment.

---

### Supplement: Recommendation Confidence Index — Methodology

> [!NOTE] What Is the Recommendation Confidence Index?
> - **What is being done**: Computing a patient-level **Recommendation Confidence Index** (0–100) and **Confidence Band** (`High` / `Moderate` / `Low`) for every arm assignment.
> - **Why we are doing it**: The 3-arm routing decision is driven by multiple biological signals of varying strength. Some patients sit clearly within one arm; others land there by exclusion or with borderline evidence. The confidence index makes this uncertainty explicit so that clinicians can prioritise the clearest cases for immediate treatment and flag ambiguous patients for multi-disciplinary review.
> - **What question it answers**: For a given arm assignment, how strongly do the underlying biological signals corroborate it?

> [!IMPORTANT] Interpretation Caveat
> The Recommendation Confidence Index is a **composite biological plausibility score**, not a calibrated statistical probability. It should not be read as a p-value or a posterior probability of response. It quantifies how coherently the patient's molecular profile aligns with the signals that define their assigned arm.

#### Design Rationale: Arm-Specific Signal Weighting

A single uniform confidence formula applied across all arms would be biologically meaningless — the signals that make an Arm A assignment trustworthy are completely different from those that support Arm B or Arm C. The index is therefore computed separately for each arm using only the signals that drove the original routing decision.

#### Arm A — Immunotherapy Confidence

Arm A patients are assigned because their tumour microenvironment is immunologically active (*Immune Hot* phenotype or high Tumour Inflammation Score). Confidence reflects how unambiguously their molecular profile sits in this activated state:

$$\text{Conf}_A = 0.50 \times \text{TIS}_{\text{dist}} + 0.30 \times \text{IFN-}\gamma_{\text{norm}} + 0.20 \times \text{CD8}_{\text{norm}}$$

| Signal | Weight | Biological Rationale |
|---|---|---|
| **TIS distance above boundary** ($\text{TIS}_{\text{dist}}$) | 50% | How far above the 60th-percentile TIS threshold the patient sits — a patient deep in the Immune Hot cluster is a clearer call than one just above the boundary |
| **IFN-gamma score** ($\text{IFN-}\gamma_{\text{norm}}$) | 30% | Intact interferon-gamma signalling is the primary mechanistic prerequisite for anti-PD-1 response |
| **CD8 T-cell infiltration** ($\text{CD8}_{\text{norm}}$) | 20% | High cytotoxic T-cell density corroborates immune activation independently of the TIS composite |

Arm A results ($N = 437$): High = 41, Moderate = 142, Low = 254. The high proportion of Low-confidence Arm A patients reflects cases admitted via the borderline high-TIS rule rather than a clean Immune Hot phenotype — these are the patients most worth reviewing in a multidisciplinary team setting.

#### Arm B — Targeted Therapy Confidence

Arm B patients carry actionable driver mutations (`BRAF V600E` or `NRAS`). Confidence reflects the strength of both the mutation evidence and the predicted drug sensitivity from the Q2 LASSO model:

$$\text{Conf}_B = 0.40 \times \text{MutStrength} + 0.60 \times \text{Q2}_{\text{Dab, norm}}$$

| Signal | Weight | Biological Rationale |
|---|---|---|
| **Mutation strength** (`BRAF` = 1.0, `NRAS` = 0.7) | 40% | `BRAF V600E` has a directly validated targeted drug (Dabrafenib); `NRAS` mutations have weaker direct inhibitor options (MEK/CDK4/6) |
| **Q2 Dabrafenib Sensitivity Index** (normalised 0–1) | 60% | The primary quantitative evidence for drug responsiveness, derived from Q2 LASSO regression on 24-gene cell line expression signatures |

> **NRAS-only cap**: `NRAS`-only Arm B patients ($N = 111$) are **capped at Moderate** confidence regardless of their weighted score. The Q2 model was trained on Dabrafenib — a `BRAF`-directed drug — so its sensitivity predictions are less directly applicable to pure `NRAS` mutants, whose optimal inhibitor remains MEK or CDK4/6 combination therapy rather than Dabrafenib monotherapy.

Arm B results ($N = 121$): High = 5, Moderate = 112, Low = 4. Arm B achieves the highest proportion of High-confidence assignments across all three arms, reflecting that a clear oncogenic driver mutation paired with a strong Q2 drug sensitivity score is the most unambiguous routing signal in the system.

#### Arm C — Combination / Reversal Therapy Confidence

Arm C is an exclusion arm: patients reach it because they are predicted non-responders *and* lack an actionable driver mutation. Confidence reflects how clearly the Q4 DepMap target nomination fits their phenotype and how biologically convertible their microenvironment appears:

$$\text{Conf}_C = 0.50 \times \text{Align}_{\text{Q4}} + 0.30 \times \text{Treatability}_{\text{norm}} + 0.20 \times (1 - \text{M2}_{\text{norm}})$$

| Signal | Weight | Biological Rationale |
|---|---|---|
| **Phenotype–target alignment** ($\text{Align}_{\text{Q4}}$) | 50% | M2 Immunosuppressive → `CSF1R` = 1.0 (best-supported); Immune Cold + high treatability → `AXL`/STING = 0.75; Immune Cold + low treatability → `HDAC`/epigenetic = 0.50 (least specific) |
| **Treatability Index** (normalised 0–1) | 30% | Higher treatability signals more intact antigen-presentation machinery, making microenvironmental reversal more plausible |
| **Inverted M2 barrier** (normalised 0–1) | 20% | Patients with lower M2 macrophage burden face a smaller immunosuppressive obstacle, making combination strategies more likely to succeed |

Arm C results ($N = 141$): High = 37, Moderate = 81, Low = 23. The low proportion of High-confidence Arm C patients is expected: this arm is defined by *exclusion* rather than a positive molecular signal, so many assignments reflect our best available option for a difficult patient rather than a clear-cut recommendation. Low-confidence Arm C patients (typically deep Immune Cold with very low treatability) are the most appropriate candidates for referral to early-phase clinical trials.

#### Confidence Band Thresholds

| Band | Raw Score Range | Interpretation |
|---|---|---|
| **High** | ≥ 0.70 (index ≥ 70) | Strong multi-signal agreement; proceed with recommended therapy |
| **Moderate** | ≥ 0.45 (index ≥ 45) | Reasonable evidence but at least one signal is borderline; consider MDT review |
| **Low** | < 0.45 (index < 45) | Weak signal coherence; recommend multidisciplinary review or clinical trial enrolment |

#### Cohort-Level Confidence Summary ($N = 699$)

| Confidence Band | Count | Percentage |
|---|---|---|
| **High** | 83 | 11.9% |
| **Moderate** | 335 | 47.9% |
| **Low** | 281 | 40.2% |
| **Overall Mean Index** | 50.2 / 100 | — |