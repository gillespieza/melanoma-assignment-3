---
title: "Phase 7: Translating Biology into Personalised Treatment Decisions"
aliases:
  - Q5 Phase 7 Student
tags:
  - melanoma
  - patient-stratification
  - phase-7
  - q5
  - presentation
created: 2026-08-01 21:46
cssclasses:
  - table-small
  - table-center
  - row-alt
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-01 21:46
---

## Phase 7: Translating Biology into Personalised Treatment Decisions

> [!NOTE] What is Phase 7 Doing and Why?
> - **What**: Phase 7 is the final, clinical decision-making layer of the Q5 pipeline. It takes everything discovered in the previous six phases — immune phenotypes, predictive models, drug sensitivity scores, and essentiality targets — and converts that information into a concrete, personalised treatment recommendation for every single patient in the cohort.
> - **Why**: Knowing that a patient is unlikely to respond to immunotherapy is only useful if you can also tell their clinician *what to do instead*. Phase 7 answers that "what next?" question for all 699 patients.
> - **Core output**: Every patient is routed into one of three clinical arms, receives a Treatability Index score (0–100), and is assigned a recommendation confidence level (High, Moderate, or Low).

---

## Phase Overview

Imagine you have 699 melanoma patients and a battery of biological measurements — immune cell infiltration, tumour mutation burden, driver gene status, gene expression profiles. You now need to hand each patient a treatment plan. That is exactly what Phase 7 does.

The phase operates as a **3-Arm Clinical Decision Engine**. It does not rely on a single biomarker or a simple cut-off rule. Instead, it synthesises the tumour's immune microenvironment phenotype (from Phase 3), the patient's predicted immunotherapy response probability (from Phase 5), and cross-study drug sensitivity and target data (from Q2 and Q4) to produce a ranked, confidence-annotated treatment pathway.

The three arms represent distinct therapeutic philosophies:

1. **Arm A — Immunotherapy Monotherapy**: For patients whose tumours show a strongly inflamed microenvironment (*Immune Hot* phenotype). These tumours already have the biological machinery — activated T-cells, intact interferon signalling, antigen presentation — needed to mount an anti-tumour immune response when checkpoint inhibition is applied. Recommended therapy: anti-PD-1 agents (*Pembrolizumab* or *Nivolumab*).

2. **Arm B — Targeted Therapy**: For patients who are unlikely immunotherapy responders but carry an actionable oncogenic driver mutation — specifically `BRAF V600` or `NRAS` mutations. Rather than forcing an ineffective immune response, the strategy pivots to targeting the underlying driver directly. A patient-specific Dabrafenib Sensitivity Index (derived from Q2 drug viability modelling) is computed to rank how responsive each Arm B patient is likely to be to `BRAF` inhibition.

3. **Arm C — Combination and Microenvironmental Reversal**: For the remaining non-responders who lack an actionable mutation for direct targeting. These patients sit in immunologically cold or heavily immunosuppressive tumour microenvironments. The strategy here is to *remodel the tumour microenvironment* using a helper agent before or alongside checkpoint blockade — essentially trying to convert a cold or suppressed tumour into one that can be engaged immunologically.

---

## Upstream Integration

> [!INFO] How Phase 7 Draws on Every Prior Phase
> Phase 7 is the convergence point of the entire Q5 pipeline. It would not be possible without each preceding analysis:
>
> - **Phase 1 & 2 (Q1)**: Established which immune and stromal features — TIS, CYT, interferon-gamma signalling, T-cell infiltration — are the most informative for predicting melanoma immunotherapy response. These same features feed directly into the Treatability Index score.
> - **Phase 3**: Performed unsupervised GMM clustering to discover the four biological phenotype subtypes (*Immune Hot*, *Immune Cold*, *Immunosuppressive M2-High*, *Mutant-Driven*). These phenotype labels are the primary routing variable in Phase 7 — they determine which arm a patient enters first.
> - **Phase 4 (Q3)**: ODE-based digital twin modelling characterised each phenotype's dynamic immune behaviour. This provides the biological rationale for why *Immune Cold* tumours need priming agents rather than direct checkpoint blockade.
> - **Phase 5**: Subgroup-specific machine learning models provide individualised immunotherapy response probabilities per patient. Phase 7 uses these predictions, combined with phenotype assignment, to decide whether a patient qualifies for Arm A.
> - **Phase 6**: Decision Curve Analysis confirmed that selectively withholding immunotherapy from predicted non-responders yields net clinical benefit. Phase 7 is the operationalisation of that conclusion: non-responders need not just exclusion from immunotherapy, but an active alternative plan.
> - **Q2 (Dabrafenib Dose-Response)**: A LASSO regression model trained on CCLE cancer cell line viability data identified 24 gene predictors of Dabrafenib sensitivity. Phase 7 applies those gene weights to each patient's tumour RNA expression profile to generate a Dabrafenib Sensitivity Index — the key scoring metric for Arm B.
> - **Q4 (DepMap Essentiality Mapping)**: CRISPR genome-wide essentiality screens identified which genes are selectively lethal in different tumour phenotype contexts. Phase 7 maps these findings onto Arm C patients to nominate biologically rational helper targets (`CSF1R`, `MDM2`, `AXL`, `HDAC`).

---

## Clinical Relevance

> [!INFO] Why This Matters for Patient Stratification (Q5)
> The clinical problem that motivates Q5 is a fundamental one in oncology: **not all melanoma patients benefit equally from the same treatment**. Immune checkpoint inhibitors like anti-PD-1 are transformative for a subset of patients, but around 60–70% of patients do not respond, and exposing non-responders to these therapies introduces immune-related toxicity without benefit.
>
> Phase 7 addresses this directly by moving beyond "treat everyone with immunotherapy and see who responds" to a **precision allocation framework** with three key properties:
>
> 1. **100% patient coverage**: Every patient receives a recommended pathway — no patient is simply labelled "unlikely to respond" and left without an alternative plan.
> 2. **Biologically motivated routing**: Treatment selection is driven by the patient's tumour microenvironment biology and validated cross-study drug sensitivity data, not arbitrary thresholds.
> 3. **Transparent confidence scoring**: Each recommendation comes with an explicit confidence rating (High, Moderate, or Low) that communicates to the clinician how much biological evidence supports the routing decision.
>
> In a real-world clinical trial setting, a framework like this would underpin patient pre-stratification before enrolment, reducing the proportion of patients receiving ineffective first-line therapy.

---

## Key Phase Results

### Arm Allocation Across 699 Patients

| Treatment Arm | Strategy | Patients | Cohort Share |
|---|---|---|---|
| **Arm A: Immunotherapy** | Anti-PD-1 monotherapy (*Pembrolizumab* / *Nivolumab*) | 364 | 52.1% |
| **Arm B: Targeted Therapy** | Dabrafenib (`BRAF`/`NRAS`-driven; mean sensitivity = 56.0/100) | 234 | 33.5% |
| **Arm C: Combination/Reversal** | Helper agent + checkpoint blockade (primary target: `CSF1R`) | 101 | 14.4% |

The majority of patients (52.1%) qualify for immunotherapy — largely driven by the large *Immune Hot* phenotype subgroup ($N = 341$, who make up 93.7% of Arm A). The remaining 335 patients receive a targeted or combination strategy.

### Treatability Index: Quantifying Biological Convertibility

The **Treatability Index** is a 0–100 composite score that captures how biologically amenable a patient's tumour is to immune-mediated killing. It integrates four molecular sub-scores: antigen presentation (`B2M`, `TAP1`), interferon-gamma pathway integrity (`IFN_gamma`), effector T-cell density (`CD8_Tcell`), and immunosuppressive macrophage burden (`M2_score`). The weights assigned to each component are derived empirically from 195 response-annotated patients using regularised logistic regression.

By design, the score is uniformly spread across the cohort (overall mean = **50.0/100**, interquartile range spanning the full 50-point window), ensuring no patient is artificially compressed near the boundaries. Key phenotype-level averages from the live data:

| Biological Phenotype | Mean Treatability Index | Interpretation |
|---|---|---|
| *Immune Hot* | 65.9 / 100 | Highest baseline immune convertibility |
| *Mutant-Driven* | 40.8 / 100 | Moderate; MAPK inhibition may prime immunity |
| *M2 Immunosuppressive* | 35.3 / 100 | Suppressive stroma; `CSF1R` depletion needed |
| *Immune Cold* | 25.3 / 100 | Desert TME; requires `AXL`/STING priming first |

### Q4 Arm C Target Nominations

For patients in Arm C — those with non-inflamed, non-mutation-driven microenvironments — Q4 DepMap CRISPR essentiality data nominates a specific helper drug target:

| Helper Target | Biological Rationale | Arm C Patients ($N = 101$) |
|---|---|---|
| `CSF1R` (M2 TAM Depletion) | Blocks M2 macrophage survival, remodels immunosuppressive stroma | 67 (66.3%) |
| `MDM2` (p53 Activation) | Restores p53 tumour suppression in `TP53`-intact tumours | 21 (20.8%) |
| `HDAC` Epigenetic Remodelling | Reverses epigenetic silencing of immune recognition genes | 10 (9.9%) |
| `AXL` / STING Pathway Priming | Activates innate immune sensing in deeply cold tumours | 3 (3.0%) |

### Recommendation Confidence Distribution

Each recommendation is assigned a confidence band based on the strength of the supporting biological evidence. Across the full cohort of 699 patients:

- **High Confidence** (score ≥ 70/100): **173 patients (24.7%)**
- **Moderate Confidence** (45–70/100): **306 patients (43.8%)**
- **Low Confidence** (score < 45/100): **220 patients (31.5%)**

Arm B (targeted therapy) patients receive the highest average confidence (mean = 69.0/100), reflecting that mutation-driven routing with a quantified drug sensitivity index provides stronger individualised evidence than phenotype-based immunotherapy assignment alone.

---

> [!INSIGHT] Key Takeaways
> - **100% patient coverage is achievable**: By combining immune phenotyping, driver mutation status, drug sensitivity scores, and essentiality targets, the 3-arm framework generates a biologically motivated treatment plan for every single patient — leaving no one without an actionable pathway.
> - **Phenotype drives treatment philosophy**: The *Immune Hot* phenotype almost exclusively populates Arm A (341 of 364 patients), confirming that a patient's tumour microenvironment biology — not just their mutation profile — is the strongest determinant of which therapeutic strategy to pursue.
> - **Cross-study integration is feasible and informative**: Q2 cell-line drug viability data and Q4 DepMap CRISPR screens can be meaningfully projected onto patient transcriptomics to generate individual drug sensitivity scores and ranked target nominations, even without patient-specific experimental data.
> - **Recommendation confidence reveals where we are certain — and where we are not**: Only 24.7% of patients (N = 173) receive a High confidence recommendation, signalling that the field still lacks sufficient biological resolution to make high-confidence precision treatment decisions for the majority of patients.
> - **Non-response is not a dead end**: Phase 7 reframes immunotherapy non-response as an opportunity for microenvironmental remodelling and targeted combination therapy, directly nominating actionable agents (`CSF1R`, `MDM2`, `AXL`) that address the specific resistance mechanism in each patient's tumour.

## Limitations & Future Directions

> [!WARNING] Current Limitations of the Phase 7 Framework
> The following limitations are derived directly from the Phase 7 Limitations Report. They should be kept in mind when interpreting the recommendations produced by this pipeline.
>
> **Statistical Constraints**
> - The Treatability Index achieves a discriminative accuracy (ROC-AUC) of approximately **0.60** on the 195 training patients — only modestly above chance. It is a useful biological stratification tool, but cannot yet function as a standalone clinical predictor. Independent validation on a held-out cohort is essential before clinical use.
> - The empirical weights used in the index are point estimates derived from a relatively small sample ($N = 195$). We do not yet know how stable these weights are: a weight close to zero (such as the antigen presentation component) may be statistically indistinguishable from having no effect. Bootstrap stability analysis is a priority next step.
> - There is a notable paradox in confidence scoring: Arm A — the largest arm and the one with the strongest clinical evidence base — has the *lowest* average confidence score (47.6/100), with over half of Arm A patients rated Low confidence. This reflects a design tension in the confidence formula and should be re-examined.
>
> **Biological Assumptions**
> - The Q2 Dabrafenib Sensitivity Index is derived from cancer cell line experiments, not patient tumours. Stromal contamination, tumour heterogeneity, and in vivo drug pharmacokinetics are not captured by cell-line models. Arm B scores should be interpreted as a relative ranking, not an absolute prediction of clinical response.
> - Arm C target nominations (`CSF1R`, `MDM2`, etc.) are assigned at the phenotype level — all patients of the same phenotype in Arm C receive the same nomination. Individual patient gene expression levels for these targets are not yet incorporated into the nomination logic.
> - Arm routing is based on the most likely phenotype label, ignoring the uncertainty in cluster assignment. A patient almost equally likely to be *Immune Hot* or *M2 Immunosuppressive* is treated identically to a patient who is unambiguously *Immune Hot*.
>
> **Data Constraints**
> - Overall survival and progression-free survival data are available in the dataset but are not currently integrated into routing or confidence scoring. Recommendations are optimised for predicted response (tumour shrinkage), not for maximising patient survival — a clinically important distinction.
> - The `AXL`/STING priming arm within Arm C is allocated to only 3 patients, which is far too small to draw any clinical conclusions about the effectiveness of this strategy.

> [!INSIGHT] Top Priorities for Future Iterations
> 1. **Validate the Treatability Index externally** on an independent melanoma ICI trial cohort (e.g., Hellmann 2018) not included in this analysis.
> 2. **Incorporate survival outcomes** (OS, PFS) as a routing criterion alongside binary response, so that treatment recommendations are optimised for patient longevity rather than short-term response alone.
> 3. **Move from phenotype-level to patient-level target nomination** in Arm C — using each patient's own `CSF1R`, `AXL`, and `MDM2` expression levels to personalise the helper drug recommendation.
> 4. **Recalibrate Arm A confidence scoring** to weight GMM cluster certainty (how strongly the patient belongs to *Immune Hot*) more directly, resolving the paradox of low-confidence immunotherapy recommendations for genuinely inflamed tumours.
