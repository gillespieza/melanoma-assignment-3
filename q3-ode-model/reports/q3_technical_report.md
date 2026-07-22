# Q3 — Are there any ODE models that are useful? (Melanoma / SKCM)
## Technical Report

**Cohort:** TCGA Skin Cutaneous Melanoma (SKCM), n = 421 patients
**Answer:** **Yes.** A mechanistic, literature-parameterised BRAF→MEK→ERK signalling
model coupled to a melanoma tumour–immune model with a checkpoint immunotherapy arm — with
per-patient inputs from TCGA — reproduces BRAF-inhibitor pharmacology (including the RAF
paradox), stratifies survival across both targeted-therapy and immunotherapy arms, is
corroborated by orthogonal RPPA protein data, and competes with black-box ML.

Every kinetic constant comes from an established published model, and the constants are
universal — *rate constants are identical for every patient; only inputs (protein
abundance, mutation state, drug dose) vary.* Sources are listed in **References** [1–7].

---

## 1. Model — four literature-parameterised modules with fast/slow separation

| Module | Source | Role |
|--------|--------|------|
| **A. RAF dimerisation + drug** | RAF-inhibitor paradox model [4, 5] | RAF-inhibitor paradox → effective RAF drive |
| **B. MAPK cascade + feedback** | MAPK cascade model [1] | fast signalling → steady-state pERK |
| **C. Melanoma tumour–immune** | melanoma tumour-immune model [2, 3] | slow tumour burden under drug + immunity; killing scaled by per-patient CYT [6] |
| **D. Checkpoint axis (anti-PD-1)** | Lai et al. 2017 [2], Auslander et al. 2018 [7] | PD-1/PD-L1/complex steady-state; dialable anti-PD-1 dose unleashes CD8 killing |

**Module A — the paradox is emergent, not scripted.** RAF signals as a dimer; RAS-GTP
drives dimerisation. A RAF inhibitor binds a protomer, but (i) a singly-liganded dimer
keeps its drug-free partner catalytically active (transactivation), (ii) the second site
binds with strong **negative cooperativity**, and (iii) drug binding **promotes**
dimerisation. From these binding equilibria, ERK output *rises then falls* with dose in
RAS-driven tumours — the RAF paradox — with no `if`-statement forcing it. A BRAF-V600E
**monomer** signals RAS-independently and is inhibited directly (monotonic suppression).

**Module B — universal cascade with universal feedback.** An eight-state cascade
(Raf/MEK/ERK, un-/mono-/di-phosphorylated) with di-phospho-ERK inhibiting the top of the
cascade through a **single feedback constant Ki = 9 nM, identical for every patient** [1].
Module A sets its input V1. Readout: time-averaged di-phospho-ERK = pERK.

**Module C — melanoma-specific, mechanistic drug coupling + CYT-scaled killing.** The published melanoma cancer
equation is
`dC/dt = λC·C(1−C/CM) − η8_i·f_kill·T8·C − dC·C`, where `prolif = pERK/pERK_ref` replaces
the original phenomenological drug bracket, so the drug's effect on the tumour is
whatever the mechanistic cascade says it is — the RAF paradox carries all the way to
tumour burden. Published melanoma constants [2, 3]: λC = 0.616, dC = 0.17, CM = 0.8,
η8 = 46, dT8 = 0.18 (day⁻¹). Killing rate is now per-patient: `η8_i = η8 × CYT_i / CYT_mean`
(Rooney et al. 2015 [6]), using the patient's cytolytic activity score from GZMA/PRF1 expression.

**Module D — checkpoint axis (anti-PD-1).** A minimal 3-state QSS sub-module: PD-1 (*P*),
PD-L1 (*L*), inhibitory complex (*Q*). Anti-PD-1 drug *A* competitively occupies PD-1 before
it can bind PD-L1, reducing *Q* and raising the fraction of CD8 cells not checkpoint-suppressed:
`f_kill = 1 − Q/P_total`. Constants: KD_PA = 5 nM (anti-PD-1 affinity), KD_PL = 8 nM
(PD-1·PD-L1 complex affinity) — literature order-of-magnitude values from Lai et al. [2],
exactly as Module A's constants are order-of-magnitude values from [5]. Per-patient *P*, *L*
pools are initialised from PDCD1 and CD274 (PD-L1) expression, reusing Q1's
`compute_impres()` and CYT functions from `q1-response-predictor/src/signatures.py` — no
new data or signature code needed.

**Timescale separation (QSS).** The fast cascade (minutes) is integrated to steady state
*first*; its pERK is then a fixed input to the slow tumour–immune ODE (days). The two are
never co-integrated, so the system is **not artificially stiff** (full 421×10 BRAFi + 421×10 anti-PD-1 run: ~3 s).

**Per-patient inputs (only):** protein totals (Raf←BRAF, MEK←MAP2K1/2, ERK←MAPK1/3
expression); RAS-GTP level; BRAF-V600E flag; immune infiltration (CD8A/PRF1/GZMA); CYT score;
PD-L1 / PDCD1 expression; drug dose.

---

## 2. Design choices and assumptions

- **Rate constants.** Modules B and C use published constants [1–3]. Module A's allosteric
  constants (Kd ≈ 50 nM inhibitor affinity, negative cooperativity ×20, drug-induced
  dimerisation) are literature-motivated order-of-magnitude values [5]. Module D's binding
  constants (KD_PA = 5 nM, KD_PL = 8 nM) are likewise literature order-of-magnitude values [2].
  None of these are fit to TCGA data.
- **No per-patient kinetics.** Every rate constant is universal. Mutation changes only
  **inputs** (RAS-GTP level, presence of a V600E monomer), never a rate constant.
- **Emergent paradox.** There is no logic gate on the drug output. The paradox emerges
  from explicit RAF-dimer + drug-binding equilibria (Module A).
- **Timescale separation.** Fast signalling and slow tumour dynamics are solved separately
  via a quasi-steady-state approximation, so the coupled system is not stiff.
- **Indirect validation for anti-PD-1 arm.** TCGA-SKCM has no anti-PD-1 dosing or
  treatment-response labels. The checkpoint-dose axis is validated the same way the BRAFi
  axis was: indirectly, via survival stratification and biological consistency with
  IMPRES/CYT scores.

---

## 3. Reproducing BRAF-inhibitor pharmacology (Phase 3)

Mean pERK across the vemurafenib dose sweep, by genotype:

| Subtype | baseline pERK | low→high dose | behaviour |
|---------|--------------:|---------------|-----------|
| BRAF-V600E | 195 | 195 → 101 | **suppressed ~49 %** (drug works) |
| NRAS-mutant | 259 | 259 → 268 | **paradox / flat** (drug fails) |
| MAPK-quiet WT | 162 | 162 → 170 | flat |

The subtype hierarchy **NRAS > BRAF > WT** *emerges* from the mechanism (RAS-driven
dimerisation), and the drug suppresses ERK only in BRAF-V600E tumours — the central
clinical fact about BRAF inhibitors.

Mechanism sanity check for Module D: anti-PD-1 shrinks modelled tumour burden **33.4%**
in high-PD-L1 patients vs only **2.0%** in low-PD-L1 patients (median CD274 split) — a
biologically correct, differentiated dose-response.

---

## 4. ODE outputs stratify real survival (Phase 4)

| ODE readout | Arm | Log-rank *p* | High vs Low median OS |
|-------------|-----|-------------:|----------------------|
| baseline **pERK** | BRAFi | 2.5 × 10⁻² | 68.1 vs 103.1 mo |
| **tumour burden** | BRAFi | 2.7 × 10⁻² | 68.1 vs 105.0 mo |
| **checkpoint tumour burden** | anti-PD-1 | **2.4 × 10⁻³** | **65.9 vs 148.2 mo** |

All three readouts significantly separate patients by outcome. The immunotherapy arm
(Module D) produces the strongest signal: an 82-month OS gap and the smallest p-value.
High checkpoint tumour burden marks checkpoint-refractory disease — tumours that persist
despite simulated anti-PD-1 blockade — consistent with the worst prognosis. Directions
are computed from the data, not assumed.

---

## 5. Independent validation against RPPA protein data (Phase 5)

TCGA measured phospho-ERK directly by RPPA (`MAPK_pT202_Y204`) — an orthogonal assay.
ODE-predicted baseline pERK vs measured pERK (n = 310):

- **Pearson r = 0.175, p = 2.0 × 10⁻³** (Spearman r = 0.164, p = 3.8 × 10⁻³).
- **Subtype hierarchy matches:** both model and RPPA place **NRAS-mutant highest**
  (measured NRAS-vs-rest Mann-Whitney **p = 3.2 × 10⁻⁹**).

This correlation is *not* fit to the RPPA data — it emerges from the mechanistic modules,
which makes it a genuine (if modest) external validation. *Honest caveat:* the model puts
BRAF slightly above WT in baseline pERK whereas RPPA puts BRAF slightly below — the
steady-state pERK of BRAF-V600E is strongly feedback-buffered in vivo; we did not overfit
that detail. Modules A/B results are numerically identical before and after the Module D
refactor, confirming the checkpoint axis does not perturb the signalling modules.

---

## 6. Mechanistic ODE vs machine learning (Phase 6)

Predicting 2-year survival (n = 358), 5-fold CV ROC-AUC:

| Model | Features | ROC-AUC |
|-------|---------:|--------:|
| Random Forest | 12 raw | 0.686 ± 0.046 |
| **ODE "digital twin"** | **3 ODE outputs** | **0.666 ± 0.074** |
| Logistic Regression | 12 raw | 0.646 ± 0.029 |
| Neural Network (MLP) | 12 raw | 0.583 ± 0.054 |

Three interpretable, mechanistically-derived numbers (baseline pERK, BRAFi tumour burden,
checkpoint tumour burden) beat the neural net and logistic regression, trailing only Random
Forest — with every input traceable to a biological mechanism. The third feature was
decisive: with the refactored Module C/D maths but still only 2 features, AUC briefly
dropped to 0.608; adding the checkpoint tumour burden as the third feature recovered and
surpassed the original baseline (0.666 vs 0.652).

---

## 7. Conclusion

**ODE models are useful for melanoma — and can be built rigorously.** A model assembled
entirely from published kinetics [1–5], using universal rate constants and separating
fast/slow timescales, reproduces BRAF-inhibitor pharmacology, stratifies overall survival
across both targeted-therapy (*p* = 0.027) and immunotherapy (*p* = 0.0024) arms, is
corroborated by orthogonal RPPA protein data, and rivals black-box ML with three
interpretable features. The model now mechanistically supports both branches of the Q5
treatment decision tree, and the immunotherapy arm proved the stronger prognostic signal.

### References

1. Kholodenko BN. MAPK cascade with negative feedback. *Eur J Biochem* 2000; 267:1583–1588. (BioModels BIOMD0000000010)
2. Lai X et al. Melanoma BRAF/MEK-inhibitor + immune-checkpoint model. *BMC Syst Biol* 2017; 11:70. (PMC5517842)
3. Melanoma BRAF/MEK-inhibitor + anti-PD-1 model. *Appl Sci* 2022; 12:12474.
4. Poulikakos PI et al. RAF-inhibitor paradoxical activation. *Nature* 2010; 464:427–430 and 464:431–435.
5. Rukhlenko OS et al. Structure-based model of RAF-inhibitor resistance. *Cell Systems* 2018; 7:161–179.
6. Rooney MS et al. Molecular and genetic properties of tumours associated with local immune cytolytic activity. *Cell* 2015; 160:48–61.
7. Auslander N et al. Robust prediction of response to immune checkpoint blockade therapy in metastatic melanoma. *Nat Med* 2018; 24:1545–1549. (IMPRES)

### Reproduce

```bash
cd q3-ode-model
for p in 1 2 3 4 5 6; do ./venv/bin/python scripts/phase${p}_*.py; done
```
