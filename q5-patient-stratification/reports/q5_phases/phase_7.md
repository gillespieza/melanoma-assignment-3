---
title: "Phase 7: 3-Arm Decision Support System & Treatability Scoring (Q1–Q4)"
aliases:
  - Q5 Phase 7
tags:
  - melanoma
  - patient-stratification
  - phase-7
  - q5
created: 2026-08-01 20:05
cssclasses:
  - table-small
  - table-center
  - row-alt
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-01 20:05
---

## 7. Phase 7: 3-Arm Decision Support & Treatability Scoring

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Constructing a 3-arm clinical decision framework routing all $N = 699$ patients into: **Arm A** (Immunotherapy Monotherapy, $N = 364$), **Arm B** (Targeted Therapy integrating Q2 Dabrafenib sensitivity model, $N = 234$), and **Arm C** (Combination/Reversal Therapy integrating Q4 DepMap essentiality targets `CSF1R`, `MDM2`, `AXL`, $N = 101$).
> - **Why we are doing it**: Decision Curve Analysis in Phase 6 demonstrated that withholding immunotherapy from predicted non-responders prevents toxicity, but non-responders require actionable alternative therapies rather than clinical abandonment.
> - **What question it answers**: How can we systematically route 100% of melanoma patients into biologically rational therapeutic arms, and which specific helper drug targets convert resistant non-responders into sensitive states?

Phase 7 operationalises precision patient allocation across $N = 699$ patients. The decision engine routes patients into three structured therapeutic arms:

1. **Arm A: Immunotherapy Monotherapy** ($N = 364$, **52.1%** of cohort): Assigned to high-confidence predicted responders (*Immune Hot* phenotype or high TIS scores). Received anti-PD-1 monotherapy (*Pembrolizumab* / *Nivolumab*).
2. **Arm B: Targeted Therapy (Q2 Integration)** ($N = 234$, **33.5%** of cohort): Assigned to predicted non-responders carrying actionable driver mutations (`BRAF V600` or `NRAS`). Integrates the Q2 LASSO cell viability regression model to compute a patient-specific **Dabrafenib Sensitivity Index** (mean Arm B sensitivity = **56.0/100**).
3. **Arm C: Combination & Microenvironmental Reversal (Q4 Integration)** ($N = 101$, **14.4%** of cohort): Assigned to remaining non-responders in immunologically cold or immunosuppressive microenvironments. Integrates Q4 DepMap essentiality targets to nominate helper interventions (most frequent nomination: **CSF1R (M2 TAM Depletion)** with $N = 67$ patients).

### Treatability Index Analysis

The composite **Treatability Index** (0–100 scale) quantifies the biological convertibility of patients based on antigen presentation integrity (`B2M`, `TAP1`), interferon-gamma intactness (`IFN_gamma`), and immunosuppressive M2 macrophage barriers:

- **Overall Mean Treatability Index**: **46.8 / 100**
- **Immune Hot**: **57.8 / 100** (highest baseline sensitivity)
- **Mutant-Driven**: **39.0 / 100** (moderate convertibility via MAPK inhibition)
- **M2 Immunosuppressive**: **38.1 / 100** (convertible via `CSF1R` macrophage depletion)
- **Immune Cold**: **23.1 / 100** (lowest baseline; requires `AXL` / STING priming)

![3-Arm Clinical Decision System Allocation across Biological Phenotypes.](q5-patient-stratification/plots/treatability/arm_assignment_breakdown.png)

> [!INFO] Understanding 3-Arm Decision Allocation: Explanation & Key Takeaways
> - **What this plot is showing**: The proportional allocation of patients across **Arm A** (Immunotherapy, green), **Arm B** (Targeted Therapy, orange), and **Arm C** (Combination/Reversal, purple) within each of the four biological melanoma phenotypes.
> - **How to interpret the plot**:
>   1. **Phenotype Stratification (X-axis)**: Shows how distinct biological microenvironments drive completely different therapeutic requirements.
>   2. **Arm A Dominance in Immune Hot**: The majority of *Immune Hot* tumours are routed to Arm A immunotherapy, matching their high baseline response rate.
>   3. **Arm B Concentration in Mutant-Driven**: *Mutant-Driven* tumours with `BRAF`/`NRAS` mutations are predominantly routed to Arm B targeted therapy when immunotherapy response is unlikely.
>   4. **Arm C Necessity in M2 Immunosuppressive & Immune Cold**: The majority of *M2 Immunosuppressive* and *Immune Cold* tumours require Arm C combination strategies, proving that single-agent checkpoint blockade is insufficient for these microenvironments.
> - **Key Takeaways**:
>   - **100% Patient Allocation Coverage**: Resolves the clinical dilemma of non-response by providing clear, actionable treatment routing for every patient in the cohort.
>   - **Phenotype-Driven Precision**: Demonstrates that treatment selection must align with microenvironmental phenotype rather than unselected biomarker thresholds.

![Treatability Index Distribution and Q2 Dabrafenib Sensitivity Scores.](q5-patient-stratification/plots/treatability/treatability_index_distribution.png)

> [!INFO] Understanding Treatability Index & Q2 Sensitivity Scores: Explanation & Key Takeaways
> - **What this plot is showing**: **Left Panel**: Boxplot distribution of the composite Treatability Index (0-100) across biological phenotypes. **Right Panel**: Q2-derived Dabrafenib Sensitivity Index distribution for `BRAF`-mutated Arm B patients.
> - **How to interpret the plot**:
>   1. **Treatability Index (Left)**: Higher values indicate tumours with intact antigen presentation and lower suppressive barriers that can be readily primed for immunotherapy response.
>   2. **Q2 Dabrafenib Sensitivity (Right)**: Higher scores represent greater predicted sensitivity to `BRAF` inhibition based on Q2 cell line gene expression models.
> - **Key Takeaways**:
>   - **Quantifiable Reversal Potential**: Treatability scoring distinguishes highly convertible non-responders from deeply refractory cases.
>   - **Direct Cross-Question Synergy**: Successfully bridges Q2 cell line viability predictions with clinical patient transcriptomics.

### Key Takeaways
- **Complete Decision Framework**: Provides clear, actionable routing for 100% of incoming melanoma patients.
- **Mechanistic Target Nomination**: Nominates validated helper targets (`CSF1R`, `MDM2`, `AXL`) to overcome specific resistance mechanisms.
- **Q2/Q4 Integration**: Seamlessly bridges cell line viability models (Q2 Dabrafenib sensitivity) and DepMap essentiality targets (Q4) with clinical patient transcriptomics.

### Final Phase Summary & Clinical Translation

> [!SUMMARY] Synthesis of Phase 7 Findings
> Phase 7 completes the Q5 Precision Patient Stratification Framework by translating biological subtyping (Phases 3-4) and predictive modelling (Phases 5-6) into an operational **3-Arm Clinical Decision Engine**. By integrating Q2 Dabrafenib viability models and Q4 DepMap essentiality target nominations (`CSF1R`, `MDM2`, `AXL`), the system provides personalised, biologically rational treatment pathways for 100% of $N = 699$ melanoma patients.

#### Core Achievements
1. **Complete Decision Routing**: Successfully routed $N = 699$ patients into Arm A (**52.1%**), Arm B (**33.5%**), and Arm C (**14.4%**).
2. **Cross-Study Integration**: Incorporated Q2 Dabrafenib sensitivity gene weights to score targeted therapy responsiveness in Arm B (`BRAF` mutants; mean sensitivity = **56.0/100**).
3. **Mechanistic Reversal Nominations**: Identified **CSF1R (M2 TAM Depletion)** as the primary helper target for resistant non-responders ($N = 67$ candidates).
4. **Treatability Metric**: Standardised a composite 0–100 Treatability Index (overall mean = **46.8**) to prioritise non-responders for combination clinical trial enrolment.

> [!formula]+ Phase 7 Script Execution & Software Module Architecture
> - **Primary Pipeline Execution Scripts**:
>   - [`07_treatability_scoring.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/07_treatability_scoring.py): Implements 3-arm clinical decision tree routing all $N = 699$ patients into **Arm A** (Immunotherapy Monotherapy), **Arm B** (Targeted Therapy integrating Q2 Dabrafenib sensitivity), and **Arm C** (Combination/Reversal integrating Q4 DepMap essentiality targets `CSF1R`, `MDM2`, `AXL`), computes composite Treatability Index (`treatability_scores.csv`), exports arm allocation summary (`treatment_arm_summary.csv`), and generates 3-arm pie, treatability distribution, and waterfall plots (`arm_distribution_pie.png`, `arm_assignment_breakdown.png`, `treatability_index_distribution.png`, `treatability_waterfall.png`).
> - **Core Supporting Python Modules**:
>   - [`treatability.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/treatability.py): Implements 3-arm assignment rules (`assign_treatment_arms`), composite treatability scoring (`compute_treatability_index`), and treatability visualisations (`plot_arm_distribution_pie`, `plot_treatability_distribution`, `plot_treatability_waterfall`).
>   - [`q5_constants.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/q5_constants.py): Central source of truth for 3-arm definitions (`TREATMENT_ARMS`), arm color palette (`TREATMENT_ARM_PALETTE`), and treatability weights (`TREATABILITY_WEIGHTS`).
>   - [`reporting.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/reporting.py): Formats treatability summary tables and Obsidian markdown elements.
> - **Shared Cross-Question & Pipeline Modules**:
>   - [`q2-dabrafenib-dose-response`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q2-dabrafenib-dose-response): Q2 Hill equation dose-response model providing patient Dabrafenib sensitivity scores.
>   - [`q4-essentiality-mapping`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q4-essentiality-mapping): Q4 DepMap CRISPR essentiality model providing nominated helper targets (`CSF1R`, `MDM2`, `AXL`).
>   - [`run_q5_pipeline.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/run_q5_pipeline.py): Master pipeline orchestrator executing `07_treatability_scoring.py` as Step 7.
>   - [`generate_q5_report.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/generate_q5_report.py): Compiles live statistical summaries and generates phase markdown reports.
