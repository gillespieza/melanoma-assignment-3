# Melanoma BRAF-Inhibitor Viability Predictor (q2-viability-predictor)

## Overview

This module builds a predictor of melanoma cell viability in response to standard-of-care drug treatment, then applies it to real patient data and checks how it relates to predicted immunotherapy response (see `q1-response-predictor`).

The core analysis (and the required answer to Question 2) uses two drugs as BRAF-inhibitor proxies:
- **Dabrafenib** – FDA-approved clinical BRAF inhibitor
- **PLX-4720** – research-grade BRAF inhibitor (preclinical analogue of vemurafenib)

The drug panel was later expanded to include three additional drugs, for a fuller group presentation:
- **Trametinib** – MEK inhibitor
- **Temozolomide** – chemotherapy agent
- **Dacarbazine** – chemotherapy agent

## Data sources

- **GDSC2** – drug response data (AUC) for melanoma cell lines
- **DepMap** – gene expression data (log2(TPM+1)) for the same cell lines
- **TCGA-SKCM** (cBioPortal) – real patient gene expression (RSEM) and clinical data, used to apply the trained model to patients

## Method

1. Filtered GDSC2 + DepMap to melanoma cell lines with both drug response and expression data (~24–39 lines depending on drug)
2. Trained a LASSO regression model (`glmnet`, alpha = 1) per drug to predict AUC from gene expression
3. Ran a stability check across 5 random train/test splits (Dabrafenib) to identify genes consistently selected by the model
4. Applied the two core BRAF-inhibitor models to 70 real TCGA-SKCM patients (excluding cohorts with formal immunotherapy response tracking: Liu 2019, Hugo 2016, Riaz 2017)
5. Correlated predicted patient viability scores against predicted immunotherapy response scores from `q1-response-predictor`
6. Expanded the drug panel to 3 additional drugs (Trametinib, Temozolomide, Dacarbazine) confirmed present in the GDSC2 data, using the same LASSO approach
7. Combined gene coefficients from all 5 drug models into a single reference file (`q2_model_coefficients.csv`)

## Results

**Core BRAF-inhibitor models (applied to patients + correlated with immunotherapy response):**

| Drug | Cell-line model correlation (train/test) | Correlation with predicted immunotherapy response (n=70 patients) |
|---|---|---|
| Dabrafenib | 0.751 | -0.178 |
| PLX-4720 | 0.136 | -0.004 |

**Interpretation:** Predicted Dabrafenib viability shows a weak negative correlation with predicted immunotherapy response, suggesting a mild trend where better predicted BRAF-inhibitor response coincides with better predicted immunotherapy response — though the relationship is not strong. PLX-4720 shows no meaningful correlation, consistent with its much weaker underlying model.

**Expanded drug panel (cell-line models only, not applied to patients):**

| Drug | Cell lines used | Test-set actual viability range | Correlation |
|---|---|---|---|
| Trametinib | 39 | 0.23 – 0.83 (0.60) | 0.524 |
| Temozolomide | 39 | 0.953 – 0.985 (0.03) | 0.288* |
| Dacarbazine | 24 | 0.954 – 0.984 (0.03) | 0.797 |

\* *Temozolomide's default train/test split (seed 123) produced a test set with zero prediction variance, so no correlation could be calculated (NA). A different random split (seed 456) was used instead to obtain a valid result; its gene coefficients are the ones saved in `important_genes_Temozolomide.csv`.*

**Important caveat on comparing these correlations.** Dacarbazine's 0.797 is the highest value in the panel but is *not* the strongest model. Its test-set viability values span only 0.03 AUC units (all cell lines are near-uniformly resistant), across just 5 test cell lines — so the correlation measures the ordering of near-identical values rather than genuine predictive ability. The same applies to Temozolomide. This is consistent with melanoma's known poor in vitro response to both chemotherapy agents. **Dabrafenib (0.751, viability range 0.60) is the most trustworthy model in the panel**, with Trametinib (0.524, same wide range) a credible second. See `reports/Q2_report.md` for the full discussion.

**Stability check (Dabrafenib, 5 random seeds):** Mean correlation 0.276, range -0.293 to 0.751 — one seed also produced NA for the same zero-variance reason described above, and was excluded from the mean/range. Genes selected in at least 3 of 5 seeds: **ABRA, NINL, PTER, TRPM3** — these are the most consistently informative genes across random splits, independent of which specific cell lines happened to be used for training.

## Known limitations

- Small training sample (24–39 melanoma cell lines) relative to ~19,000 candidate genes, leaving test sets of only 5–8 lines; LASSO gene selection can vary across random splits (see stability check output)
- Correlation is not comparable across drugs with very different outcome ranges — see the caveat above on Dacarbazine and Temozolomide
- Predicted viability spans a narrower range than actual viability in every model, an expected effect of LASSO shrinkage; the models are more informative about relative ordering than absolute values
- PLX-4720's model is unreliable (0.136 train/test correlation) — its patient-level predictions should be treated as low-confidence
- Both core-drug correlation values are between two *predicted* scores, not measured outcomes — this is an exploratory comparison, not causal evidence
- One selected gene (CEP104) required renaming to its older symbol (KIAA0562) to match TCGA's gene annotation
- The 3 expanded-panel drugs (Trametinib, Temozolomide, Dacarbazine) have cell-line models only — they have not been applied to TCGA patients or correlated with immunotherapy response, since this was outside the required Question 2 scope
- Temozolomide's correlation depends on which random train/test split is used (see note above); the reported value (0.288) came from a non-default seed chosen specifically because the default split produced an undefined result
- Patients were selected at cohort level rather than by individual treatment history, due to a disagreement between the `ACTUAL_RESPONSE` and `TREATMENT_RECEIVED` fields in the Q1 response-score file — see `reports/Q2_report.md`

## Folder structure

```
q2-viability-predictor/
├── outputs/    → important_genes_Dabrafenib.csv, important_genes_PLX_4720.csv,
│                 important_genes_Trametinib.csv, important_genes_Temozolomide.csv,
│                 important_genes_Dacarbazine.csv, q2_model_coefficients.csv,
│                 q2_patient_drug_predictions.csv, q2_correlation_results.csv
├── plots/      → 5 predicted vs actual viability plots (one per drug, points
│                 colour-coded by prediction error) + model_performance_comparison.png
│                 (bar chart comparing all 5 model correlations)
├── reports/    → Q2_report.md and Q2_report.pdf (full results and interpretation)
├── scripts/    → Question2.R (full pipeline, Steps 1–12)
└── README.md
```

## How to run

Requires R with the following packages:
```
glmnet
readxl
```

Run `scripts/Question2.R` top to bottom. Update the file paths in Steps 1 and 8 (`data_path`, `tcga_path`, `clin_path`, `amanda_path`) to match your local folder locations before running.

## Output files

- `q2_patient_drug_predictions.csv` – predicted Dabrafenib/PLX-4720 viability scores per patient
- `q2_correlation_results.csv` – merged viability + immunotherapy response scores, used for the core-drug correlation above
- `q2_model_coefficients.csv` – combined gene coefficients (Drug_Name, Gene_Symbol, Lasso_Weight) across all 5 drug models
