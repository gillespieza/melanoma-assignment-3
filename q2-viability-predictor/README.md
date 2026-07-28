# Question 2: Cell Viability Predictor for BRAF Inhibitors

## What this does

For this question I built a model that predicts how melanoma cells respond
to BRAF inhibitor drugs, since these are the standard-of-care treatment for
melanoma patients with the BRAF V600E/K mutation. It uses gene expression
data to predict how sensitive a given melanoma cell line is to a drug
(measured as AUC).

## Drugs used

- **Dabrafenib**
- **PLX-4720** (used in place of vemurafenib)

**Note on drug substitution:** Vemurafenib was the other standard-of-care
drug named in the brief, but it wasn't available in either the GDSC2 or
PRISM Repurposing datasets - I checked both. PLX-4720 is vemurafenib's
direct research analogue (same drug class, same mechanism), and it's
commonly used as a substitute for vemurafenib in melanoma cell line
studies, so I used that instead.

## Method

1. Loaded DepMap cell line metadata, filtered to melanoma cell lines
2. Loaded GDSC2 drug response data, filtered to Dabrafenib and PLX-4720
3. Loaded DepMap gene expression data (TPM, ~19,000 genes)
4. Merged drug response with matching gene expression per cell line
5. Split data into training (80%) and test (20%) sets
6. Trained a LASSO regression model per drug to predict AUC from gene
   expression
7. Checked prediction accuracy via correlation on the held-out test set
8. Saved the genes each model found most predictive to CSV

## Results

| Drug        | Cell lines used | Correlation (predicted vs actual) |
|-------------|------------------|-------------------------------------|
| Dabrafenib  | 37               | 0.751                              |
| PLX-4720    | 39               | 0.136                              |

Gene expression was a fairly strong predictor of Dabrafenib response, but
much weaker for PLX-4720. This is likely partly down to the smaller sample
size for PLX-4720, and some missing expression values that came up as
warnings during model fitting.

## Data requirements

To run this script, you'll need a folder called `depmap_data` with these
files inside (not included here due to file size):

- `Model.csv` — DepMap cell line metadata
  (download from [depmap.org/portal/download](https://depmap.org/portal/download))
- `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` — DepMap gene expression data
  (same source as above)
- `GDSC2_fitted_dose_response.csv` — GDSC2 drug response data
  (download from [Cell Model Passports](https://cellmodelpassports.sanger.ac.uk/downloads))

The script currently points to `~/Downloads/depmap_data/`. If yours is
somewhere else, just update the `data_path` line at the top of
`Question2.R`.

## Files in this folder

- `Question2.R` — main analysis script
- `important_genes_Dabrafenib.csv` — genes identified as predictive of Dabrafenib response
- `important_genes_PLX_4720.csv` — genes identified as predictive of PLX-4720 response
- `predicted_vs_actual_dabrafenib.png` — plot of predicted vs actual viability for Dabrafenib
- `predicted_vs_actual_plx4720.png` — plot of predicted vs actual viability for PLX-4720
