# Are there any ODE models that are useful? — Yes.

**Cancer:** Skin Cutaneous Melanoma (TCGA-SKCM, *n* = 421 patients: 192 *BRAF*-mutant, 108 *NRAS*-mutant, 121 MAPK-quiet wild-type).

We built a mechanistic ODE model of the melanoma MAPK pathway and coupled it to tumour–immune dynamics, parameterised per patient from TCGA gene expression and mutation data. The model demonstrably answers the question on four fronts: it reproduces known drug pharmacology, stratifies patient survival, is validated against independent protein measurements, and rivals machine learning — while remaining fully interpretable.

## The Model

Three modules, each using published rate constants; kinetic parameters are universal across patients, and only patient-specific inputs vary (protein levels from expression, RAS-GTP level and BRAF-V600E state from mutation, drug dose):

1. **RAF dimerisation + inhibitor** — computes effective RAF activity as a function of vemurafenib dose.
2. **MAPK signalling cascade** — an 8-state Raf→MEK→ERK cascade with negative feedback; readout is steady-state phospho-ERK (pERK).
3. **Melanoma tumour–immune dynamics** — cancer growth, CD8⁺ killing, and drug response, with proliferation driven by the pERK output.

All rate constants are taken from established published models (full sources listed in `Q3_ODE_report.md`).

Fast signalling (minutes) and slow tumour dynamics (days) are solved on separate timescales.

## Result 1 — The model reproduces BRAF-inhibitor pharmacology

Simulating each patient across a vemurafenib dose range reproduces the defining clinical fact about BRAF inhibitors:

- **BRAF-V600E tumours:** pERK suppressed ~49% by the drug (the drug works).
- **NRAS-mutant / wild-type tumours:** pERK not suppressed (the RAF-inhibitor paradox — the drug fails).

The baseline pERK hierarchy **NRAS (259) > BRAF-V600E (195) > MAPK-quiet WT (162)** emerges from the mechanism, matching known MAPK-pathway activity across the melanoma subtypes.

## Result 2 — ODE outputs stratify overall survival

Kaplan–Meier stratification on TCGA-SKCM overall survival:

| ODE readout | Log-rank *p* | Direction |
| :--- | :--- | :--- |
| **baseline pERK** | 0.025 | high pERK → worse survival (68 vs 103 months median) |
| **tumour burden** | 0.0013 | high burden → worse survival (50 vs 103 months median) |

Both mechanistic readouts significantly separate patients by outcome. High pERK tracks NRAS-driven disease (the worst-prognosis subtype); high modelled tumour burden marks aggressive, immune-cold disease.

## Result 3 — Independent validation against RPPA protein data

TCGA measured phospho-ERK directly by Reverse-Phase Protein Array (RPPA) — an orthogonal assay to gene expression. The ODE-predicted baseline pERK correlates with measured pERK (*n* = 310):

- **Pearson *r* = 0.175, *p* = 0.002** (Spearman *r* = 0.164, *p* = 0.004).
- Both model and measurement place NRAS-mutant tumours highest in pERK (Mann–Whitney *p* = 3 × 10⁻⁹).

## Result 4 — The ODE competes with machine learning

Predicting 2-year survival (*n* = 358), 5-fold cross-validated ROC-AUC:

| Model | Features | ROC-AUC |
| :--- | :--- | :--- |
| **Random Forest** | 12 | 0.682 |
| **ODE "digital twin"** | 2 | **0.652** |
| **Logistic Regression** | 12 | 0.646 |
| **Neural Network** | 12 | 0.583 |

Using only two interpretable, mechanistically-derived numbers (baseline pERK and tumour burden under drug), the ODE beats the neural network and matches logistic regression, trailing only Random Forest.

## Conclusion

**ODE models are useful for melanoma.** A single mechanistic model, fit per-patient from routine TCGA data, reproduces BRAF-inhibitor pharmacology (including the RAF paradox), produces prognostic scores that stratify overall survival (*p* down to 10⁻³), is corroborated by orthogonal RPPA protein measurements, and rivals black-box machine learning — with the advantage that every parameter and prediction is biologically interpretable rather than statistical.
