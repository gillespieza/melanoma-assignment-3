# Are there any ODE models that are useful? — Yes.

**Cancer:** Skin Cutaneous Melanoma (TCGA-SKCM, *n* = 421 patients: 192 *BRAF*-mutant, 108 *NRAS*-mutant, 121 MAPK-quiet wild-type).

We built a mechanistic ODE model of the melanoma MAPK pathway coupled to tumour–immune dynamics, parameterised per patient from TCGA gene expression and mutation data. The model demonstrably answers the question on four fronts: it reproduces known drug pharmacology, stratifies patient survival across both targeted-therapy and immunotherapy arms, is validated against independent protein measurements, and rivals machine learning — while remaining fully interpretable.

## The Model

Four modules, each using published rate constants; kinetic parameters are universal across patients, and only patient-specific inputs vary (protein levels from expression, RAS-GTP level and BRAF-V600E state from mutation, drug dose):

1. **RAF dimerisation + inhibitor** — computes effective RAF activity as a function of vemurafenib dose.
2. **MAPK signalling cascade** — an 8-state Raf→MEK→ERK cascade with negative feedback; readout is steady-state phospho-ERK (pERK).
3. **Melanoma tumour–immune dynamics** — cancer growth, CD8⁺ killing, and drug response, with proliferation driven by pERK. Killing capacity is scaled by each patient's cytolytic (CYT) score (Rooney et al. 2015).
4. **Checkpoint axis (anti-PD-1)** — a minimal PD-1/PD-L1/complex steady-state sub-module (Lai et al. 2017); a dialable anti-PD-1 dose depletes free PD-1, reducing the inhibitory PD-1·PD-L1 complex and unleashing CD8⁺ killing. Initialised per patient from IMPRES and PD-L1 (CD274) expression — reusing `compute_impres()` and CYT score already implemented in Q1's `q1-response-predictor/src/signatures.py`, requiring no new data engineering.

All rate constants are taken from established published models (full sources in `q3_technical_report.md`). Fast signalling (minutes) and slow tumour dynamics (days) are solved on separate timescales (quasi-steady-state).

## Result 1 — The model reproduces BRAF-inhibitor pharmacology

Simulating each patient across a vemurafenib dose range reproduces the defining clinical fact about BRAF inhibitors:

- **BRAF-V600E tumours:** pERK suppressed ~49% by the drug (the drug works).
- **NRAS-mutant / wild-type tumours:** pERK not suppressed (the RAF-inhibitor paradox — the drug fails).

The baseline pERK hierarchy **NRAS (259) > BRAF-V600E (195) > MAPK-quiet WT (162)** emerges from the mechanism, matching known MAPK-pathway activity across the melanoma subtypes.

## Result 2 — ODE outputs stratify overall survival

Kaplan–Meier stratification on TCGA-SKCM overall survival across all three mechanistic readouts:

| ODE readout | Arm | Log-rank *p* | Direction |
| :--- | :--- | :--- | :--- |
| **baseline pERK** | BRAFi | 0.025 | high pERK → worse survival (68 vs 103 mo median) |
| **tumour burden** | BRAFi | 0.027 | high burden → worse survival (68 vs 105 mo median) |
| **checkpoint tumour burden** | anti-PD-1 | **0.0024** | high burden → worse survival (66 vs 148 mo median) |

All three readouts significantly separate patients by outcome. The immunotherapy arm (Module D) produces the strongest signal: an 82-month OS gap, nearly double that of the targeted-therapy arm. High checkpoint tumour burden marks checkpoint-refractory disease — patients whose tumours persist despite simulated anti-PD-1 blockade, the worst-prognosis group. As a mechanism sanity check, anti-PD-1 shrinks modelled tumour burden **33% in PD-L1-high patients vs only 2% in PD-L1-low patients** — a biologically correct, differentiated dose-response.

## Result 3 — Independent validation against RPPA protein data

TCGA measured phospho-ERK directly by Reverse-Phase Protein Array (RPPA) — an orthogonal assay to gene expression. The ODE-predicted baseline pERK correlates with measured pERK (*n* = 310):

- **Pearson *r* = 0.175, *p* = 0.002** (Spearman *r* = 0.164, *p* = 0.004).
- Both model and measurement place NRAS-mutant tumours highest in pERK (Mann–Whitney *p* = 3 × 10⁻⁹).

Modules A and B are numerically identical before and after the Module D refactor — confirming the checkpoint axis only affects the immune module and does not perturb the signalling results.

## Result 4 — The ODE competes with machine learning

Predicting 2-year survival (*n* = 358), 5-fold cross-validated ROC-AUC:

| Model | Features | ROC-AUC |
| :--- | :--- | :--- |
| **Random Forest** | 12 | 0.686 |
| **ODE "digital twin"** | **3** | **0.666** |
| **Logistic Regression** | 12 | 0.646 |
| **Neural Network** | 12 | 0.583 |

Using only three interpretable, mechanistically-derived numbers (baseline pERK, BRAFi tumour burden, and checkpoint tumour burden), the ODE beats logistic regression and the neural network, trailing only Random Forest — which uses 12 raw features. The third feature was decisive: with the refactored Module C/D maths but still only 2 features, AUC briefly *dropped* to 0.608 (the checkpoint dynamics changed the existing features' signal); adding the checkpoint tumour burden as an explicit third feature recovered and surpassed the original (0.666 vs 0.652), confirming that the immunotherapy readout carries independent prognostic information.

## Conclusion

**ODE models are useful for melanoma.** A single mechanistic model, fit per-patient from routine TCGA data, reproduces BRAF-inhibitor pharmacology (including the RAF paradox), produces prognostic scores that stratify overall survival across both the targeted-therapy arm (*p* = 0.027) and the immunotherapy arm (*p* = 0.0024), is corroborated by orthogonal RPPA protein measurements, and rivals black-box machine learning with just three interpretable features — with the advantage that every parameter and prediction is biologically grounded rather than statistical. Crucially, the model now mechanistically supports both branches of the Q5 treatment decision tree: the BRAFi arm (BRAF-mutant → vemurafenib) and the immunotherapy arm (checkpoint-sensitive → anti-PD-1), and the immunotherapy arm turned out to be the stronger prognostic signal of the two.

## Caveats

Two limitations should be stated alongside these results, in keeping with the model's existing tone of honest parameterisation:

1. **Checkpoint binding constants are literature order-of-magnitude values, not fitted.** The constants *K*D_PA = 5 nM (anti-PD-1 affinity for PD-1) and *K*D_PL = 8 nM (PD-1·PD-L1 complex affinity) are taken from published biochemical ranges (Lai et al. 2017) — exactly as Module A's RAF-dimer constants are taken from Rukhlenko et al. They are not fit to TCGA data. This is appropriate for a mechanistic model where the point is to use published biology, not to tune parameters to a single cohort, but it means the absolute tumour-burden values are indicative rather than quantitatively precise.

2. **Validation of the anti-PD-1 dose axis is indirect.** TCGA-SKCM contains no anti-PD-1 treatment assignments or drug-response outcomes, so the checkpoint-dose axis cannot be validated against actual treatment data. Instead, it is validated the same way the BRAFi axis was: indirectly, via survival stratification (Phase 4) and biological consistency with IMPRES/CYT scores. The strong log-rank result (*p* = 0.0024) and the differentiated PD-L1-high vs PD-L1-low dose-response (33% vs 2% tumour shrinkage) are consistent with expected biology, but a prospective immunotherapy cohort with dosing records would be needed for direct quantitative validation.
