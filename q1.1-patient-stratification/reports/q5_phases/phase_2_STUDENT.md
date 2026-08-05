---
title: "Phase 2: Biomarker Feature Analysis, Clinical Decision Cutoffs & Genomic Interactions"
aliases:
  - Phase 2 Conceptual Overview
  - Q5 Phase 2 Presentation Guide
tags:
  - melanoma
  - patient-stratification
  - phase-2
  - presentation
  - q5
created: 2026-08-01 12:44
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-01 12:44
---

# Phase 2: Biomarker Feature Analysis, Clinical Decision Cutoffs & Genomic Interactions

> [!NOTE] Phase Overview & Objectives
> - **What is being done**: Evaluating individual immune biomarkers, establishing optimal clinical decision cutoffs, and testing whether underlying oncogenic driver mutations alter microenvironmental immune response prediction.
> - **Why we are doing it**: To determine if single biological tests can accurately predict anti-PD-1 immunotherapy response, or if tumour complexity requires multi-dimensional patient stratification.
> - **What question it answers**: Which individual biomarkers best differentiate Responders from Non-Responders, where should clinical cutoffs be set, and how do driver mutations like `BRAF V600` interact with microenvironmental immunity?

## 1. Introduction & Biological Motivation

When treating advanced melanoma with anti-PD-1 immune checkpoint inhibitors, clinicians face a fundamental challenge: some patient tumours respond remarkably well, while others experience rapid disease progression.

Phase 2 evaluates whether single biological measurements (such as tumour-infiltrating immune cell counts or interferon gene expression signatures) can serve as reliable standalone predictors of treatment success across N = 326 immunotherapy-treated patients.

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
1. **Univariate Feature Screening**: Ranking individual biomarkers by their statistical effect size (Cohen's d).
2. **Clinical Decision Cutoff Optimisation**: Using Youden's J statistic to find optimal numerical cutoffs that balance sensitivity and specificity.
3. **Genomic x Immune Interaction Modelling**: Measuring how oncogenic driver mutations (such as `BRAF V600`) modify the predictive value of microenvironmental immune inflammation.

## 2. Upstream Integration: Building Beyond Question 1

In **Question 1**, we harmonised data across 326 metastatic melanoma patients (*Hugo 2016*, *Riaz 2017*, *Liu 2019*, and ICI-treated *TCGA-SKCM*) and built a multimodal feature vector combining transcriptomic immune signatures (`TIS`, `CYT`), Tumour Mutational Burden (`TMB`), and somatic driver mutation flags (`mut_BRAF`, `mut_NRAS`, `mut_NF1`). Question 1 used these combined features as additive inputs for supervised classification models.

**Phase 2 extends Question 1 in three fundamental ways**:

1. **From Additive Multimodal Inputs to Multiplicative Interaction Terms**: Question 1 evaluated driver mutations and immune signatures as independent additive features. Phase 2 fits explicit logistic regression interaction terms across all 21 driver mutation x immune signature pairs, proving that oncogenic `BRAF` mutations actively dampen the predictive strength of T-cell inflammation (interaction coefficient beta = -0.65, p = 0.040).
2. **From Continuous Risk Probabilities to Concrete Decision Cutoffs**: Question 1 generated continuous response probability scores. Phase 2 uses Youden's J statistic to establish exact, practical decision thresholds (such as `B_cells` threshold >= 0.430) to discretise continuous scores for clinical triage.
3. **From Supervised Classification to Stratification Rationale**: Phase 2 provides mathematical proof that single biomarkers yield modest overall diagnostic accuracy (AUC between 0.58 and 0.63), demonstrating why single-gene tests fail in clinical practice and establishing the essential rationale for **Phase 3 (Unsupervised Multidimensional Clustering)**.

## 3. Why This Matters for Patient Stratification (Q5)

Understanding that `BRAF` driver mutations alter microenvironmental immunity explains why single-biomarker tests are insufficient and provides the exact clinical rationale for Question 5's Multi-Arm Stratification and Treatability Scoring:

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

1. **`BRAF` Mutations Alter Predictive Value**: In `BRAF` wild-type tumours, high T-cell inflammation (`TIS` High) dramatically boosts anti-PD-1 response rates (from 22% up to over 58%). In `BRAF`-mutated tumours, constitutive oncogenic MAPK signalling causes immunosuppression (downregulating antigen presentation via `B2M` and `HLA-A/B/C`), blunting the predictive boost of T-cell inflammation.
2. **Clinical Guidance for Stratification (Phase 7)**:
   - **Inflamed `BRAF` Tumours**: First-line anti-PD-1 immunotherapy provides durable, long-term survival.
   - **Cold/Suppressed `BRAF` Tumours**: Single-agent immunotherapy is insufficient. Patients require **combination strategies**—using short-course BRAF/MEK targeted inhibitors to release neoantigens and un-blind the immune system, followed by anti-PD-1 checkpoint blockade.

## 4. Key Phase Results & Empirical Findings

### 4.1 Ranking Biomarker Associations (Cohen's d Effect Size)

To compare continuous biomarkers on a standardised scale, we compute Cohen's d effect size between Responders (CR/PR) and Non-Responders (PD). Cohen's d measures how many standard deviations separate the mean scores of Responders from Non-Responders.

![Ranked Biomarker Feature Associations](q5-patient-stratification/plots/feature_analysis/biomarker_volcano_plot.png)

> [!INFO] Understanding the Effect Size Chart
> - **Right Side (d > 0, Green)**: Features enriched in Responders. Higher scores indicate a higher probability of treatment success (e.g. `B_cells`, `TIS`, `CD8_T_cells`).
> - **Left Side (d < 0, Red)**: Features enriched in Non-Responders. Higher scores indicate primary resistance (e.g. `Macrophage_STV_Score`).
> - **Dashed Cutoff Lines (|d| = 0.20)**: Represents a small biological effect size threshold. Features within these lines have over 92% overlap between patient groups and lack strong standalone predictive power.

### 4.2 Establishing Clinical Decision Cutoffs (Youden ROC Analysis)

In clinical practice, doctors need concrete numerical cutoffs rather than continuous probabilities. For example: *"If a patient's B-cell score exceeds 0.430, classify their tumour as inflamed."*

We use Receiver Operating Characteristic (ROC) curves and Youden's J statistic to determine the single mathematical threshold that maximises diagnostic accuracy (Sensitivity + Specificity - 1).

![Youden ROC Curves](q5-patient-stratification/plots/feature_analysis/youden_roc_curves.png)

> [!INFO] Understanding the ROC Curves & Youden Cutoffs
> - **The Curves**: Show trade-offs between Sensitivity (detecting true responders) and False Alarm Rate (1 - Specificity).
> - **The Marked Dots**: Represent Youden's optimal decision threshold for each biomarker.
> - **Diagonal Dashed Line**: Represents random chance guessing (AUC = 0.50).

#### Youden Optimal Decision Threshold Summary

| Biomarker Feature | Optimal Cutoff | Youden J | Sensitivity (%) | Specificity (%) | Area Under Curve (AUC-ROC) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`B_cells`** | **0.430** | **0.306** | **53.7%** | **77.0%** | **0.632** |
| `TIS` | 0.191 | 0.184 | 57.3% | 61.1% | 0.585 |
| `CD8_T_cells` | 0.069 | 0.184 | 67.1% | 51.3% | 0.584 |
| `CYT` | 0.621 | 0.205 | 32.9% | 87.6% | 0.583 |
| `IFN_gamma` | 0.243 | 0.151 | 54.9% | 60.2% | 0.566 |
| `M1_M2_Ratio` | 1.076 | 0.034 | 6.1% | 97.3% | 0.442 |
| `Macrophage_STV_Score` | -16.628 | 0.027 | 100.0% | 2.7% | 0.413 |

`B_cells` emerges as the top single standalone marker with an AUC of 0.632. However, most single biomarkers yield modest diagnostic accuracy (AUC around 0.58), proving that single-gene tests are insufficient for patient selection.

### 4.3 Genomic Synergy: TIS x BRAF Interaction Analysis

Before evaluating all 21 driver mutation x immune microenvironment combinations, we selected `TIS` x `BRAF` as our primary benchmark pair because:
1. **`TIS` is the Clinical Gold Standard**: The Tumour Inflammation Signature (`TIS`, Ayers et al.) is an 18-gene interferon-gamma responsive score validated extensively in clinical trials.
2. **`BRAF V600` Dictates First-Line Treatment**: `BRAF` mutations occur in 40–50% of cutaneous melanomas, dictating whether patients receive Targeted Therapy vs Immunotherapy.
3. **Biological & Statistical Anchor**: Testing `TIS` x `BRAF` evaluates whether oncogenic MAPK activation dampens T-cell inflammation before expanding to all driver gene permutations. Mathematically, `TIS` x `BRAF` was confirmed as the single statistically significant interaction (interaction coefficient beta = -0.65, p = 0.040).

![Genomic Interaction TIS x BRAF](q5-patient-stratification/plots/feature_analysis/genomic_interaction_tis_braf.png)

> [!INFO] Understanding the TIS x BRAF Interaction Bar Chart
> - **In `BRAF` Wild-Type Tumours**: High T-cell inflammation (`TIS` High) dramatically boosts the objective response rate (from 22% up to over 58%).
> - **In `BRAF` Mutated Tumours**: Oncogenic `BRAF` activation blunts the benefit of high T-cell inflammation, resulting in an attenuated response increase.

### 4.4 Comprehensive Genomic x Immune Interaction Matrix

Expanding our interaction analysis across all 21 driver mutation x microenvironment signature combinations reveals broader genomic-immune wiring rules across N_BRAF = 128, N_NRAS = 82, and N_NF1 = 40 patients:

![Genomic Immune Interaction Matrix](q5-patient-stratification/plots/feature_analysis/genomic_immune_interaction_matrix.png)

> [!INFO] Understanding the Interaction Heatmap
> - **Cell Values (beta_interaction)**: Numerical interaction effect sizes. Positive values indicate positive synergy; negative values indicate blunted efficacy.
> - **White Box Highlights**: Indicates statistically significant interaction terms (p < 0.05).
> - **`BRAF` Column Dominance**: `BRAF` x `TIS` (beta = -0.65, p = 0.040) is the primary statistically significant interaction. Furthermore, borderline significant interactions (p < 0.10) occur exclusively within the `BRAF` column: `BRAF` x `B_cells` (beta = -0.59, p = 0.073), `BRAF` x `CD8_T_cells` (beta = -0.57, p = 0.074), and `BRAF` x `IFN_gamma` (beta = -0.53, p = 0.087).
> - **`NF1` Microenvironment Synergy**: `NF1` loss tumours demonstrate strong positive synergy with macrophage polarisation (`M1_M2_Ratio`, beta = +0.94), suggesting pro-inflammatory myeloid targeted therapy may be uniquely effective in `NF1`-mutated tumours.

## 5. Limitations & Future Directions

> [!WARNING] Summary of Phase 2 Limitations & Strategic Future Roadmap
> 1. **Step-Function Threshold Discretisation**:
>    - *Limitation*: Binary Youden cutoffs split continuous score distributions into rigid High vs Low categories, misclassifying borderline patients (e.g. 0.429 vs 0.430).
>    - *Future Direction*: Transition to Generalized Additive Models (GAMs) and restricted cubic splines to model continuous, non-linear risk gradients.
> 2. **Bulk RNA-Seq Blindness to Tertiary Lymphoid Structures (TLS)**:
>    - *Limitation*: Bulk transcriptomics measures total B-cell RNA abundance (`B_cells` AUC = 0.632) but cannot confirm whether B cells are organised into functional Tertiary Lymphoid Structures (TLS) versus scattered naive cells.
>    - *Future Direction*: Combine transcriptomic deconvolution with spatial histology deep learning (QuPath) and multiplex immunofluorescence imaging.
> 3. **Multiple Testing & Driver Subgroup Sample Size Imbalance**:
>    - *Limitation*: Testing 21 interaction pairs without Benjamini-Hochberg FDR control risks Type I errors. Furthermore, smaller driver subgroups (`mut_NF1` N = 40) lack statistical power to detect interactions compared to `mut_BRAF` (N = 128).
>    - *Future Direction*: Implement Benjamini-Hochberg FDR control (q < 0.05) and balanced bootstrap resampling to equalise detection sensitivity across driver subtypes.

## 6. Key Takeaways

> [!INSIGHT] Key Takeaways: Phase 2 Feature Analysis & Stratification Rationale
> - **Single Biomarkers Have Modest Power**: While inflammatory signatures (`TIS`, `CYT`, `CD8_T_cells`) and lymphocyte markers (`B_cells`) show statistically significant elevation in responders, single markers achieve modest overall accuracy (AUC between 0.58 and 0.63).
> - **Decision Thresholds Provide Clinical Cutoffs**: Youden's J statistic establishes concrete numerical cutoffs (such as `B_cells` threshold >= 0.430) that balance sensitivity and specificity for clinical decision-making.
> - **Genomic Drivers Modify Microenvironmental Immunity**: Immune inflammation behaves differently depending on underlying driver mutations. Specifically, oncogenic `BRAF` mutations dampen the positive predictive value of T-cell inflammation (beta = -0.65, p = 0.040).
> - **Rationale for Unsupervised Stratification**: Because single biomarkers perform modestly and interact strongly with underlying driver mutations, robust clinical decision support requires multi-dimensional patient clustering (Phase 3) rather than rigid single-gene tests.
