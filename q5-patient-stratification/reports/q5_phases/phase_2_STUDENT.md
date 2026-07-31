---
title: "Phase 2: Biomarker Feature Analysis, Decision Cutoffs & Genomic Interactions"
aliases:
  - Phase 2 Conceptual Overview
  - Q5 Phase 2 Conceptual Guide
tags:
  - melanoma
  - patient-stratification
  - phase-2
  - presentation
  - q5
created: 2026-07-31 11:54
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-31 12:24
---

# Phase 2: Biomarker Feature Analysis, Clinical Decision Cutoffs & Genomic Interactions

> [!NOTE] Concept Overview
> - **What is being done**: Testing individual immune biomarkers, finding optimal clinical decision thresholds, and evaluating whether underlying gene mutations alter immune biomarker effectiveness.
> - **Why we are doing it**: To determine if a single biological test can accurately predict anti-PD-1 immunotherapy response, or if patient tumours are too complex for single-gene tests.
> - **What question it answers**: Which individual biomarkers best separate Responders from Non-Responders, where should clinical cutoffs be set, and how do driver mutations like `BRAF` interact with microenvironmental immunity?

## 1. Introduction & Clinical Motivation

When treating advanced melanoma with anti-PD-1 immune checkpoint inhibitors, clinicians face a fundamental challenge: some patient tumours respond remarkably well, while others experience rapid disease progression.

In Phase 2, we evaluate whether single biological measurements (such as tumour-infiltrating immune cell counts or interferon gene expression signatures) can serve as reliable standalone predictors of treatment success.

```
       [ Patient Tumour Biopsy ]
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
[ Single Biomarker ]   [ Multi-Gene Signature ]
  e.g., `CD8_T_cells`     e.g., `TIS` Score
         │                   │
         └─────────┬─────────┘
                   ▼
     [ Can a Single Test Predict
     Immunotherapy Response? ]
```

To answer this question, Phase 2 carries out three core analytical steps:
1. **Univariate Feature Screening**: Ranking individual biomarkers by their statistical effect size (Cohen's $d$).
2. **Clinical Decision Cutoff Optimisation**: Using Youden's $J$ statistic to find optimal numerical cutoffs that balance sensitivity and specificity.
3. **Genomic x Immune Interaction Modelling**: Measuring how oncogenic driver mutations (such as `BRAF V600`) modify the predictive value of microenvironmental immune inflammation.

## 1.1 Building Beyond Question 1

A natural question when reviewing Phase 2 is: _"How does this analysis build upon Question 1?"_

In **Question 1**, we harmonised data across 326 metastatic melanoma patients (Hugo 2016, Riaz 2017, Liu 2019) and built a **multimodal feature vector** ($D = 11$) combining transcriptomic immune signatures (`TIS`, `CYT`), Tumour Mutational Burden (`TMB`), and somatic driver mutation flags (`mut_BRAF`, `mut_NRAS`, `mut_NF1`). Question 1 used these combined multimodal features to train supervised classifiers (such as Random Forests and XGBoost) to predict patient response ($CR/PR$ vs $PD$).

In Question 1, driver mutations and immune signatures were combined as **additive main effects**. **Phase 2 extends Question 1 in three fundamental ways**:

1. **From Additive Multimodal Inputs to Multiplicative Interaction Terms**: While Question 1 included both mutations and signatures in the same feature vector, it evaluated them as independent additive features. Phase 2 fits explicit **logistic regression interaction terms** ($\beta_{\text{interaction}}$) across all 21 driver $\times$ immune pairs ($\text{Signature} \times \text{Driver}$), proving that oncogenic `BRAF` mutations actively dampen the predictive strength of T-cell inflammation ($\beta = -0.65, p = 0.040$).
2. **From Continuous Risk Probabilities to Concrete Decision Cutoffs**: Question 1 generated continuous probability scores for response prediction. Phase 2 uses **Youden's $J$ statistic** to establish exact, practical decision thresholds (such as `B_cells` $\ge 0.430$) to discretise continuous scores for clinical triage.
3. **From Supervised Classification to Stratification Rationale**: Phase 2 provides the mathematical proof that single biomarkers yield modest overall diagnostic accuracy ($\text{AUC} \approx 0.58–0.63$), demonstrating why single-gene tests fail in practice and establishing the essential rationale for **Phase 3 (Unsupervised Multidimensional Clustering)**.

## 2. Ranking Biomarker Associations (Cohen's d Effect Size)

To compare continuous biomarkers on a standardised scale, we compute Cohen's $d$ effect size between Responders ($CR/PR$) and Non-Responders ($PD$). Cohen's $d$ measures how many standard deviations separate the mean scores of Responders from Non-Responders.

![Ranked Biomarker Feature Associations](q5-patient-stratification/plots/feature_analysis/biomarker_volcano_plot.png)

> [!INFO] Understanding the Effect Size Chart
> - **Right Side ($d > 0$, Green)**: Features enriched in Responders. Higher scores indicate a higher probability of treatment success (e.g. `B_cells`, `TIS`, `CD8_T_cells`).
> - **Left Side ($d < 0$, Red)**: Features enriched in Non-Responders. Higher scores indicate primary resistance (e.g. `Macrophage_STV_Score`).
> - **Dashed Cutoff Lines ($|d| = 0.20$)**: Represents a small biological effect size threshold. Features within these lines have over 92% overlap between patient groups and lack strong standalone predictive power.

### Key Finding
While inflammatory signatures (`TIS`, `CYT`) and lymphocyte markers (`B_cells`, `CD8_T_cells`) display statistically significant elevation in Responders ($p < 0.05$), their effect sizes remain modest ($d \approx 0.25 - 0.40$). This indicates that individual microenvironmental markers exhibit substantial overlap between outcome groups.

## 3. Finding Clinical Decision Cutoffs (Youden ROC Analysis)

> [!insight] Clinical triage  
> In clinical practice, doctors need concrete numerical cutoffs rather than continuous probabilities. For example: _"If a patient's B-cell score exceeds $0.430$, classify their tumour as inflamed."_

We use Receiver Operating Characteristic (ROC) curves and **Youden's $J$ statistic** to determine the single mathematical threshold that maximises diagnostic accuracy. Youden's $J$ calculates the point on the ROC curve that provides the largest combined advantage of sensitivity (true positive rate) minus false positive rate.

This plot analyses **how accurately individual biological markers separate Responders (CR/PRCR/PR) from Non-Responders (PDPD)** across every possible cutoff threshold.

![Youden ROC Curves](q5-patient-stratification/plots/feature_analysis/youden_roc_curves.png)

> [!INFO] Understanding the ROC Curves & Youden Cutoffs
> - **The Curves**: Show trade-offs between Sensitivity (detecting true responders) and 1 - Specificity (false alarm rate).
> 	- As you move along a curve from left to right, you are changing the numerical score threshold needed to classify a tumour as "inflamed".
> 	- Curves arching higher toward the **top-left corner** demonstrate superior diagnostic accuracy. A perfect test reaches the top-left corner (AUC = 1.0).
> - **The Marked Dots**: Represent Youden's optimal decision threshold for each biomarker.
> - **Diagonal Dashed Line**: Represents random chance guessing (AUC = 0.50).

### Clinical Threshold Performance Summary

| Biomarker Feature | Optimal Cutoff | Youden J | Sensitivity (%) | Specificity (%) | Area Under Curve (AUC-ROC) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `B_cells` | **0.430** | **0.306** | **53.7%** | **77.0%** | **0.632** |
| `TIS` | 0.191 | 0.184 | 57.3% | 61.1% | 0.585 |
| `CD8_T_cells` | 0.069 | 0.184 | 67.1% | 51.3% | 0.584 |
| `CYT` | 0.621 | 0.205 | 32.9% | 87.6% | 0.583 |
| `IFN_gamma` | 0.439 | 0.170 | 42.7% | 74.3% | 0.575 |
| `M1_M2_Ratio` | 1.076 | 0.034 | 6.1% | 97.3% | 0.442 |

> [!INSIGHT] Core Scientific Takeaway: Modest Standalone Accuracy  
> `B_cells` emerges as the best single standalone marker with an AUC of $0.632$.
>
> However, most single biomarkers yield modest diagnostic accuracy ($\text{AUC} \approx 0.58$).
>
> This modest performance proves that **single-gene tests are insufficient** for patient selection and demonstrates why multi-dimensional clustering (Phase 3) is biologically essential.

## 4. Genomic Synergy: How Driver Mutations Alter Immune Response

A central discovery of Phase 2 is that immune microenvironment inflammation cannot be interpreted in isolation from the tumour's underlying genomic driver mutations.

We fit logistic regression interaction models to test whether the predictive value of immune signatures depends on whether a patient harbours a `BRAF` mutation. By adding a multiplicative interaction term ($\text{Immune Score} \times \text{Driver Mutation}$) to our statistical model, we measure whether the presence of a mutation alters the slope or strength of the immune biomarker's response prediction.

### Why TIS x BRAF Was Selected as the Primary Benchmark

Before evaluating all 21 driver mutation $\times$ immune microenvironment combinations, we selected `TIS` $\times$ `BRAF` as our primary benchmark pair for three clinical and biological reasons:

1. **`TIS` is the Clinical Gold Standard**: The Tumour Inflammation Signature (`TIS`, Ayers et al.) is an 18-gene interferon-gamma responsive score validated extensively in clinical trials as the clinical gold standard for measuring pre-existing T-cell inflammation.
2. **`BRAF V600` Dictates First-Line Treatment**: `BRAF` mutations occur in 40–50% of cutaneous melanomas. In clinical oncology, knowing a patient's `BRAF` status determines whether they receive Targeted Kinase Inhibitors (such as Dabrafenib + Trametinib) versus Checkpoint Immunotherapy (anti-PD-1).
3. **Biological & Statistical Anchor**: Testing `TIS` $\times$ `BRAF` evaluates whether oncogenic MAPK signalling dampens T-cell inflammation before expanding to all driver gene permutations. Mathematically, `TIS` $\times$ `BRAF` was confirmed as the single statistically significant interaction ($\beta = -0.65, p = 0.040$), validating this pair as the primary benchmark.

![Genomic Interaction TIS x BRAF](q5-patient-stratification/plots/feature_analysis/genomic_interaction_tis_braf.png)

> [!INFO] Understanding the TIS x BRAF Interaction Bar Chart
> - **In `BRAF` Wild-Type Tumours**: High T-cell inflammation (`TIS` High) dramatically boosts the objective response rate (from 22% up to over 58%).
> - **In `BRAF` Mutated Tumours**: Oncogenic `BRAF` activation blunts the benefit of high T-cell inflammation, resulting in a significantly attenuated response increase.

### Why This Matters for Patient Stratification

Understanding that `BRAF` mutations alter microenvironmental immunity explains why single-biomarker tests are insufficient and provides the exact rationale for **Question 5's Multi-Arm Stratification & Treatability Scoring**:

```
                  [ BRAF-Mutated Melanoma Patient ]
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
[ Microenvironment Inflamed ]                     [ Microenvironment Cold / M2-High ]
         │                                                 │
         ▼                                                 ▼
  [ Anti-PD-1 Immunotherapy ]                   [ Combination / Triplet Strategy ]
 (Durable long-term response)                  (BRAF/MEK Inhibitor Priming 
                                                + Anti-PD-1 Blockade)
```

1. **`BRAF` Mutations Alter Predictive Value**: In `BRAF` wild-type tumours, high T-cell inflammation (`TIS` High) dramatically boosts anti-PD-1 response rate (from 22% up to over 58%). In `BRAF`-mutated tumours, constitutive oncogenic MAPK signalling causes immunosuppression (downregulating antigen presentation via `B2M`/HLA), blunting the predictive boost of T-cell inflammation.
2. **Clinical Guidance for Stratification (Phase 7)**:
   - **Inflamed `BRAF` Tumours**: First-line anti-PD-1 immunotherapy provides durable, long-term survival.
   - **Cold/Suppressed `BRAF` Tumours**: Single-agent immunotherapy is insufficient. Patients require **combination strategies**—using short-course BRAF/MEK targeted inhibitors to release neoantigens and un-blind the immune system, followed by anti-PD-1 checkpoint blockade.

## 5. Comprehensive Genomic x Immune Interaction Matrix

Expanding our interaction analysis across all 21 driver mutation $\times$ microenvironment signature combinations reveals broader genomic-immune wiring rules:

![Genomic Immune Interaction Matrix](q5-patient-stratification/plots/feature_analysis/genomic_immune_interaction_matrix.png)

> [!INFO] Understanding the Interaction Heatmap
> - **Cell Values ($\beta_{\text{interaction}}$)**: Numerical interaction effect sizes. Positive values (green) indicate positive synergy; negative values (vermillion) indicate blunted efficacy.
> - **White Box Highlights**: Indicates statistically significant interaction terms ($p < 0.05$).
> - **`BRAF` Column Dominance**: `BRAF` $\times$ `TIS` ($\beta = -0.65, p = 0.040$) is the primary statistically significant interaction. Furthermore, all lymphocytic signatures (`TIS`, `IFN_gamma`, `CD8_T_cells`, `B_cells`) exhibit negative interaction terms ($\beta \approx -0.57 \text{ to } -0.65, p < 0.10$) specifically in `BRAF` melanomas.
> - **`NF1` Microenvironment Synergy**: `NF1` loss tumours demonstrate strong positive synergy with macrophage polarisation (`M1_M2_Ratio`, $\beta = +0.94$), suggesting pro-inflammatory myeloid targeted therapy may be uniquely effective in `NF1`-mutated tumours.

---

> [!insight] Key Takeaways & Summary
> ### 1. Single Biomarkers Have Modest Power
> While inflammatory signatures (`TIS`, `CYT`) and lymphocyte markers (`B_cells`, `CD8_T_cells`) are elevated in responders, single markers achieve modest overall accuracy ($\text{AUC} \approx 0.58–0.63$). No single test reliably segregates all patients.
> ### Decision Thresholds Provide Clinical Triage Cutoffs
> Youden's $J$ statistic establishes concrete numerical cutoffs (such as `B_cells` threshold $\ge 0.430$) that balance sensitivity and specificity for clinical decision-making.
> ### Genomic Drivers Modify Microenvironmental Immunity
> Immune inflammation behaves differently depending on underlying driver mutations. Specifically, oncogenic `BRAF` mutations dampen the positive predictive value of T-cell inflammation ($\beta_{\text{interaction}} = -0.65, p = 0.040$).
> ### Rationale for Unsupervised Stratification
> Because single biomarkers perform modestly and interact strongly with underlying driver mutations, robust clinical decision-making requires multi-dimensional patient clustering (Phase 3) rather than rigid single-gene tests.

> [!warning] Key Phase 2 Limitations & Future Improvements
> 1. **Rigid Cutoffs Lose Fine Information**:
>    - _Limitation_: Youden cutoffs split patients into binary categories (High vs Low). A patient scoring just below the threshold (e.g. $0.429$ vs $0.430$) is classified as "Low", even though their biological score is nearly identical to a "High" patient.
>    - _Future Fix_: Transition to smooth, continuous probability curves (Generalized Additive Models) to avoid artificial boundary misclassifications.
> 
> 2. **Bulk Sequencing Lacks Spatial Resolution**:
>    - _Limitation_: `B_cells` emerged as our top individual marker, but bulk RNA sequencing cannot confirm whether B cells are organised into functional **Tertiary Lymphoid Structures (TLS)** versus un-engaged naive cells scattered in the tissue.
>    - _Future Fix_: Integrate spatial transcriptomics (10x Visium) or multiplex tissue imaging to visualize B-cell spatial organisation directly.
> 
> 3. **Subgroup Sample Size Imbalance**:
>    - _Limitation_: `BRAF`-mutated tumours formed our largest subgroup ($N = 130$), providing high statistical power to detect interactions, whereas smaller subgroups (`NF1` $N = 34$) had lower statistical power.
>    - _Future Fix_: Apply power-weighted statistical models and larger multi-centre cohorts to ensure equal detection sensitivity across all driver mutation subgroups.
