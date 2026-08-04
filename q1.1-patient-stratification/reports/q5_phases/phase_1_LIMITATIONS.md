---
title: "Phase 1: Statistical Weaknesses, Biological Assumptions & Computational Constraints"
aliases:
  - Phase 1 Limitations Audit
  - Q5 Phase 1 Limitations
tags:
  - future-work
  - limitations
  - melanoma
  - patient-stratification
  - phase-1
  - q5
created: 2026-08-01 12:06
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-01 12:06
---

# Phase 1: Statistical Weaknesses, Biological Assumptions & Computational Constraints 🔍

A rigorous analytical audit of **Phase 1** in the Question 5 Patient Stratification pipeline. This document evaluates the current state of Phase 1's statistical methodology, biological assumptions, and data constraints. All metrics are sourced directly from `PROJECT_MAP.md`, the `phase_1.md` report, the `01_load_and_prepare.log` runtime output, and direct inspection of the live output CSVs (`feature_matrix.csv`, `feature_matrix_full.csv`).

> [!NOTE] Audit Scope & Rationale
> - **What is being evaluated**: The statistical rigour, biological validity, and computational constraints of Phase 1's multi-modal feature engineering pipeline as it currently operates.
> - **Why this matters**: Phase 1 outputs (`feature_matrix.csv`, $N_{\text{ICI}} = 326$; `feature_matrix_full.csv`, $N_{\text{Full}} = 699$) propagate directly into all seven downstream phases. Any systematic bias, unmeasured variance, or violated assumption introduced here compounds across the entire Q5 analytical stack.
> - **What question it answers**: Where does Phase 1's methodology introduce statistical noise, biological misrepresentation, or data gaps that could distort downstream clustering, predictive modelling, and clinical utility analysis?

---

## 1. Statistical Weaknesses

> [!NOTE] What, Why & What It Answers
> - **What**: Evaluating where Phase 1's quantitative methodology departs from statistical best practice.
> - **Why**: Unbounded or non-comparable score units, uncontrolled inter-cohort batch effects, and missing imputation diagnostics can all introduce systematic bias that survives standardisation and distorts downstream dimensionality reduction.
> - **What it answers**: Which statistical properties of the Phase 1 feature matrix are weakly justified and require explicit validation before predictive claims can be made?

### 1.1 Unbounded, Arbitrary Score Units from Marker Averaging

Phase 1 estimates relative cell-type infiltration by computing the sample-wise mean log-expression across a fixed panel of 3–5 marker genes per cell type (e.g. `CD8_T_cells` from `CD8A`, `CD8B`). This produces scores in arbitrary log-expression units that are:

- **Unbounded**: Unlike constrained deconvolution (e.g. CIBERSORTx), there is no sum-to-one constraint ($\sum \hat{f}_k = 1$) enforcing non-negative fractional proportions. Scores can exceed biologically meaningful ranges without triggering any warning.
- **Not Proportional**: The mean of two log-expression values is not equivalent to a weighted linear mixture model. A doubling of `CD8A` expression does not correspond to a doubling of CD8+ T-cell proportion.
- **Cross-Cell Spillover**: Marker genes such as `TGFB1` and `TNF` are expressed across multiple cell types (tumour cells, endothelial cells, myeloid cells). Their inclusion in a single-cell-type marker panel introduces cross-contamination signal without correction.

### 1.2 Absence of Batch Effect Correction Between Cohorts

The ICI feature matrix ($N_{\text{ICI}} = 326$) merges RNA-seq data from four independent studies with heterogeneous per-cohort sizes:

| Cohort | $N$ | Platform Source |
| :--- | :---: | :--- |
| *Liu 2019* | 122 | iAtlas harmonised |
| *Riaz 2017* | 107 | iAtlas harmonised |
| *TCGA-SKCM* (ICI subset) | 70 | TCGA RSEM |
| *Hugo 2016* | 27 | iAtlas harmonised |

These cohorts differ in:

- **Sequencing Platform**: iAtlas harmonised data (Liu, Riaz, Hugo) vs. TCGA RSEM (TCGA-SKCM), introducing systematic expression scaling differences even after log-transformation.
- **Library Preparation**: Poly-A selection vs. rRNA depletion protocols alter the relative abundance of immune cell transcripts in bulk RNA-seq.
- **Biopsy Site & Timing**: Pre-treatment surgical resections (TCGA-SKCM) vs. pre-treatment fine needle aspirates or core needle biopsies (trial cohorts) yield structurally different TME compositions.
- **Cohort Imbalance**: *Hugo 2016* contributes only 27 patients (8.3% of the ICI matrix), introducing severe sample size imbalance. Its per-cohort mean estimates carry wide confidence intervals that may disproportionately distort cohort-normalised features.

Phase 1 applies no explicit inter-cohort batch correction (e.g. ComBat-seq, limma `removeBatchEffect`). The Macrophage STV gene weights (`m1_m2_stv.csv`, 14,835 genes) were derived from a separate reference and may not be calibrated to any of the four study-specific expression distributions.

**Statistical consequence**: Cohort identity may function as an uncontrolled confounding variable in unsupervised clustering (Phase 3), causing phenotype clusters to partially reflect study-of-origin rather than true biological microenvironment states.

### 1.3 Severe and Undocumented Missingness in Genomic Features

Direct inspection of `feature_matrix.csv` reveals substantial missingness concentrated in genomic features, while all 19 transcriptomic features have zero missing values:

| Feature(s) | Missing ($N$ / 326) | Missing (%) | Likely Cause |
| :--- | :---: | :---: | :--- |
| `MSI_SCORE_MANTIS`, `MSI_SENSOR_SCORE` | 319 | **97.9%** | Microsatellite instability scoring was not routinely computed for melanoma trial cohorts |
| `ANEUPLOIDY_SCORE` | 259 | **79.4%** | Computed systematically for TCGA-SKCM only; absent in iAtlas-harmonised trial data |
| 7 neoantigen features (`SNV_NEOANTIGEN`, `INDEL_NEOANTIGEN`, `FUSION_NEOANTIGEN`, `SPLICE_NEOANTIGEN`, `CTA_SELF_NEOANTIGEN`, `VIRUS_NEOANTIGEN`, `ERV_NEOANTIGEN`) | 104 each | **31.9%** | Neoantigen prediction not uniformly available across all trial cohorts |
| `TMB_NONSYNONYMOUS` | 36 | **11.0%** | Incomplete mutation calling files for a subset of patients |

This missingness pattern has three downstream consequences:

1. **Silent Zero-Inflation**: If missing genomic values are zero-imputed (the default `NaN` → 0 behaviour in many scikit-learn pipelines), 97.9% of MSI scores become artificial zeroes — creating a feature that is nearly constant and contributes negligible variance to Phase 3 clustering, despite microsatellite instability being a biologically meaningful predictor of immunotherapy response.
2. **Subgroup Power Asymmetry**: Only 67 patients (20.6%) have complete `ANEUPLOIDY_SCORE` data. Any Phase 2 or Phase 5 analysis that conditions on this feature implicitly selects a TCGA-SKCM-enriched subset, introducing cohort selection bias.
3. **No Imputation Strategy Documentation**: The pipeline log does not record which imputation strategy (zero-fill, median-fill, complete-case exclusion, or carry-forward) is applied to these features before export. The absence of a missingness flag column means downstream phases cannot distinguish "measured zero" from "unmeasured, imputed to zero".

### 1.4 No Normalisation Equivalence Validation Across the Dual Matrix Split

Phase 1 outputs two matrices derived from overlapping but non-identical patient sets: the ICI matrix ($N_{\text{ICI}} = 326$) and the full-cohort matrix ($N_{\text{Full}} = 699$). Features shared between both matrices (e.g. `TIS`, `CYT`, `M1_M2_Ratio`) are computed independently on each patient set, but **no validation is performed** to confirm that their distributional properties are equivalent between the two matrices.

If TCGA-SKCM patients (the additional $N = 373$ in the full-cohort matrix) systematically differ in expression scale from the ICI cohort — as is expected given platform differences — then the full-cohort feature matrix will have different marginal distributions for every transcriptomic feature. Any inter-cohort comparison or visualisation that combines statistics from both matrices without acknowledging this divergence is potentially misleading.

---

## 2. Biological Assumptions

> [!NOTE] What, Why & What It Answers
> - **What**: Examining where Phase 1 encodes biological simplifications that may not generalise across patient subgroups or tumour contexts.
> - **Why**: Biologically incorrect assumptions embedded in feature construction propagate into phenotype labels and clinical recommendations, creating a false sense of mechanistic interpretability.
> - **What it answers**: Which biological model choices in Phase 1 are unsupported oversimplifications that limit the validity of downstream phenotype characterisation?

### 2.1 Spatial Proxy Ratios Are Algebraic Heuristics, Not Physical Distances

Phase 1 introduces two engineered spatial microenvironment indicators:

- `Spatial_CD8_CAF_Distance_Ratio`: $\log_2(\text{CD8\_T\_cells} / \text{CAFs})$
- `Spatial_Tumour_Infiltration_Index`: $\log_2(\text{CD8\_T\_cells} \times \text{M1\_M2\_Ratio} / \text{CAFs})$

These are algebraic ratios of bulk deconvolution scores, not measurements of physical intercellular distances. They assume that a high `CD8_T_cells / CAFs` bulk RNA-seq ratio *implies* cytotoxic T-cell penetration into the malignant nest — an inference that is biologically unjustified. In a stroma-excluded tumour, CD8+ T cells accumulate at the tumour margin without penetrating the core; both `CD8_T_cells` and `CAFs` bulk scores would be elevated simultaneously, potentially producing a ratio close to unity despite a physically excluded architecture.

Furthermore, the logarithmic ratio is undefined (or $-\infty$) when `CAFs` → 0, requiring an implicit numerical floor that is not documented.

### 2.2 One-Dimensional Macrophage Polarisation (M1/M2 Binary Axis)

The `M1_M2_Ratio` and `Macrophage_STV_Score` collapse the entire landscape of tumour-associated macrophage (TAM) biology into a single signed scalar: positive values represent M1 pro-inflammatory dominance, negative values represent M2 pro-tumour dominance.

In human melanoma, TAMs occupy a continuous, multi-dimensional functional spectrum encompassing distinct subtypes (e.g. angiogenic M2a, immunosuppressive M2c, tissue-resident FOLR2+ macrophages, and lipid-laden macrophages) that exert mechanistically distinct effects on CD8+ T-cell function and anti-PD-1 efficacy. Collapsing this heterogeneity onto a 1D axis discards functional subtype information that is potentially critical for distinguishing the *Immunosuppressive M2-High* phenotype (Cluster 3, $N = 308$, 44.1% of full cohort) from the *Immune Hot* phenotype (Cluster 2, $N = 304$, 43.5%) — the two most prevalent and most clinically important groups, separated by only a 3.2-percentage-point response rate difference (38.0% vs. 41.1%).

### 2.3 Static Pre-Treatment Snapshots Cannot Capture Dynamic TME Remodelling

All Phase 1 features are derived from baseline, pre-treatment biopsies. The tumour microenvironment undergoes rapid and substantial remodelling within 2–4 weeks of anti-PD-1 infusion, including:

- T-cell proliferative bursts and spatial redistribution into the tumour core
- Adaptive `CD274` (PD-L1) upregulation as a tumour-intrinsic resistance mechanism
- Macrophage repolarisation from M2-like to inflammatory states driven by interferon signalling

A baseline `TIS` score cannot distinguish a patient whose immune response is genuinely quiescent (true non-responder) from one whose immune response is suppressed but remains inducible (pseudo-non-responder who would respond with adequate co-stimulation). This ambiguity is a primary source of false-negative predictions for patients who exhibit delayed responses.

### 2.4 Marker Gene Panels Assume Cross-Cohort Gene Coverage Uniformity

The 3–5 marker genes per cell type defined in `q5_constants.py` (`CELL_TYPE_MARKERS`) are assumed to be present and reliably quantified in all four source cohorts. The `IMPRES` score (*Auslander et al., 2018*) — retained in the 33-feature panel for Phase 5/6 predictive modelling — highlights a concrete example of this risk: key co-stimulatory partners (`CD28`, `CD86`, `CD80`, `CD40`, `CD200`, `TNFRSF4`, `VSIR`) are absent or unmapped in a subset of merged multi-study expression matrices. Where genes are missing, the `IMPRES` score is computed over a reduced pair set, producing a systematically lower value for those samples without any missingness flag.

No systematic audit has been conducted to verify that all marker genes across all seven deconvolved cell types (`CD8_T_cells`, `CD4_T_cells`, `NK_cells`, `B_cells`, `M1_Macrophages`, `M2_Macrophages`, `CAFs`) achieve 100% coverage across all four cohorts. If one or more marker genes are missing in a subset of samples, the cell-type marker mean is silently computed over a reduced gene set, producing a systematically lower score for those samples without any missingness flag in the exported feature matrix.

---

## 3. Computational & Data Constraints

> [!NOTE] What, Why & What It Answers
> - **What**: Identifying data availability gaps and computational modelling choices that limit Phase 1's completeness and reproducibility.
> - **Why**: Constraints in input data propagate into feature completeness, and computational fallback behaviours risk silent result divergence between execution environments.
> - **What it answers**: What are the hard data and infrastructure boundaries within which Phase 1 currently operates, and where do they introduce analytical blind spots?

### 3.1 Absence of Direct Copy Number Alteration Maps

Trial cohorts (*Liu 2019*, *Riaz 2017*, *Hugo 2016*) lack processed GISTIC arm-level Copy Number Alteration (CNA) files. Phase 1 relies on `ANEUPLOIDY_SCORE` and single-gene expression levels as indirect proxies for chromosomal instability. Critically:

- `TMB_NONSYNONYMOUS` and CNA burden are biologically decoupled: high-CNA, low-TMB tumours exist (particularly in `NF1`-loss and chromosomally unstable melanomas) and would be misclassified as genomically stable by TMB alone.
- `ANEUPLOIDY_SCORE` is available for only 67 of 326 ICI patients (20.6%), rendering it effectively unusable as a standalone feature without cohort-biased imputation.

### 3.2 Full-Cohort Matrix ($N_{\text{Full}} = 699$) Lacks RECIST Response Labels for TCGA-SKCM

The full-cohort feature matrix (`feature_matrix_full.csv`, $N_{\text{Full}} = 699$) incorporates the complete TCGA-SKCM reference cohort ($N = 443$), the majority of whom were **not treated with anti-PD-1 immunotherapy** and therefore lack RECIST clinical response labels (`RESPONSE_BINARY`).

This creates a fundamental interpretability constraint for any Phase 3 or Phase 4 analysis derived from the full-cohort matrix:

- Phenotype labelling (e.g. *Immune Hot*, *Immunosuppressive M2-High*) that was validated against immunotherapy response rates in the ICI matrix ($N_{\text{ICI}} = 326$) cannot be directly extended to the TCGA-SKCM patients without an explicit biological validation step.
- Kaplan-Meier survival analysis in Phase 4 uses overall survival (OS) as a response surrogate for TCGA-SKCM patients, but OS conflates immunotherapy benefit with the effects of surgery, targeted therapy, and natural disease history in an unselected population.

### 3.3 STV Reference Matrix Provenance & Calibration Gap

The Macrophage Signature Transcript Vector is computed against `m1_m2_stv.csv` (14,835 genes). The biological provenance of this reference — the original study, cell line, or patient population from which M1/M2 gene weights were derived — is not documented in any accessible report or configuration file.

Without knowing the reference population, it is impossible to assess whether the STV weights are calibrated for the expression scale of the iAtlas-harmonised melanoma trial cohorts, or whether they were derived from a reference population with systematically different baseline expression levels that would systematically skew the `Macrophage_STV_Score` distributions.

### 3.4 Near-Complete Missingness Renders Three Genomic Features Effectively Unusable

Two MSI features (`MSI_SCORE_MANTIS`, `MSI_SENSOR_SCORE`) have only 7 of 326 patients (2.1%) with non-null values, and `ANEUPLOIDY_SCORE` has only 67 of 326 (20.6%). These three features are carried through the full pipeline as columns of the 33-feature panel but contribute negligible analytical value in their current state:

- In Phase 3 unsupervised clustering, near-constant (zero-imputed) features absorb a GMM covariance dimension without encoding real biological signal.
- In Phase 5 predictive modelling, they add noise variables that can degrade Random Forest importance estimates through spurious splits on imputed values.

These features should either be excluded from the panel with an explicit rationale, or flagged with a per-patient missingness indicator so downstream phases can handle them appropriately.

---

## 4. Prioritised Roadmap for Future Iterations

| Priority | Limitation | Proposed Fix | Analytical Benefit |
| :---: | :--- | :--- | :--- |
| **1 — High** | No batch effect correction | Apply ComBat-seq or limma `removeBatchEffect` across 4 cohorts prior to feature extraction | Removes cohort-of-origin as a confound in Phase 3 clustering |
| **2 — High** | 3 genomic features with >79% missingness carried silently | Exclude `MSI_SCORE_MANTIS`, `MSI_SENSOR_SCORE`, and `ANEUPLOIDY_SCORE` from the active panel or add per-patient missingness flags | Prevents zero-inflated noise from contaminating clustering and predictive models |
| **3 — High** | No missingness audit logged at Phase 1 exit | Log per-feature missingness rates in `_print_completion_summary()`; document and justify the imputation strategy for each feature class | Enables downstream phases to make informed decisions about feature handling |
| **4 — Medium** | Unbounded marker-averaging deconvolution | Transition to CIBERSORTx or EPIC constrained SVR unmixing with single-cell melanoma reference matrices | Yields bounded $0–100\%$ cell fractions; eliminates cross-cell marker spillover |
| **5 — Medium** | 1D macrophage polarisation axis | Expand to a 4-dimensional TAM subtype vector (M1, M2a, M2c, lipid-laden) using single-cell reference signatures | Resolves functional subtype heterogeneity in the M2-High phenotype (44.1% of full cohort) |
| **6 — Medium** | Spatial proxies are algebraic heuristics | Integrate multiplex immunofluorescence (mIF) or spatial transcriptomics (10x Visium) to measure physical T-cell to tumour distances | Correctly separates stroma-excluded from tumour-infiltrated microenvironments |
| **7 — Lower** | Static pre-treatment biopsy only | Integrate paired Day-0 / Day-14–28 transcriptomics where trial data permit | Enables $\Delta\text{TIS}$ dynamic signatures capturing early on-treatment immune reactivation |
| **8 — Lower** | Absent CNA maps for trial cohorts | Process alignment files via CNVkit or ASCAT for arm-level CNA and whole-genome duplication indicators | Directly quantifies chromosomal instability independent of TMB |
| **9 — Lower** | STV reference provenance undocumented | Add `source`, `population`, and `normalisation` metadata fields to `m1_m2_stv.csv` configuration | Enables calibration audit and cross-study STV score comparability |

---

> [!INSIGHT] Key Takeaways
> - **Batch Effects Are Uncontrolled**: Merging four studies with distinct sequencing platforms and biopsy protocols without explicit batch correction means cohort-of-origin remains a confounding variable that may partially drive the Phase 3 phenotype clusters. *Hugo 2016* ($N = 27$) is severely underrepresented.
> - **Severe Genomic Missingness**: Three genomic features (`MSI_SCORE_MANTIS`, `MSI_SENSOR_SCORE`, `ANEUPLOIDY_SCORE`) have 79–98% missing values, and seven neoantigen features are 31.9% missing. These features are carried in the 33-feature panel but contribute negligible analytical value without explicit handling of their missingness.
> - **Spatial Features Are Heuristics**: The two engineered spatial proxy ratios are algebraic approximations of physical tumour architecture — they cannot distinguish stroma-excluded from tumour-infiltrated microenvironments, and the logarithmic form is undefined when `CAFs` → 0.
> - **M2-High vs. Immune Hot Clinical Overlap**: The two largest phenotypes — *Immunosuppressive M2-High* ($N = 308$, response rate 38.0%) and *Immune Hot* ($N = 304$, response rate 41.1%) — are separated by only 3.2 percentage points. The 1D macrophage STV axis driving their separation encodes a biologically oversimplified model that may not reliably distinguish these phenotypes at the level of individual patients.
> - **`IMPRES` Gene Coverage Gap**: The Immune Predictive Score is retained in the panel but computed over a reduced co-stimulatory pair set for samples where `CD28`, `CD86`, or `CD80` are absent — producing systematically underestimated scores for those samples with no missingness flag in the output CSV.
