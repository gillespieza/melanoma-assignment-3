# Q5 — Which treatment when multiple options are available? (Melanoma / SKCM)

**Cohort:** TCGA Skin Cutaneous Melanoma (SKCM), n = 421 patients
**Answer:** **Yes** — the choice between immunotherapy, BRAF/MEK-targeted therapy, and
chemotherapy can be made systematically by combining two orthogonal response axes that
this project already modelled, gated by mutation status. Applied to TCGA-SKCM, the
resulting first-line recommendations fall into four groups whose overall survival is
strongly and correctly ordered (log-rank **p = 2.7 × 10⁻⁵**).

---

## 1. The idea — two response axes, one decision

Melanoma first-line therapy is a genuine multi-option decision: checkpoint immunotherapy,
BRAF/MEK-targeted therapy (for BRAF-V600 tumours only), or — increasingly rarely —
chemotherapy. The two modern options are driven by **independent biology**, and this
project built a predictor for each:

| Axis | Predicts benefit from | Model / feature |
|------|-----------------------|-----------------|
| **Targeted axis** | BRAF/MEK inhibitors | Q3 ODE — predicted vemurafenib pERK suppression (high only in BRAF-V600; NRAS/WT show the RAF paradox) |
| **Immune axis** | Checkpoint immunotherapy | Q1 immune-response signatures — here proxied by a CD8A/PRF1/GZMA cytotoxic score; in deployment, Q1's calibrated response predictor |
| **Eligibility gate** | — | BRAF-V600 vs NRAS vs triple-wild-type mutation status |

Because the two axes are biologically distinct (MAPK oncogene addiction vs. pre-existing
anti-tumour immunity), every patient can be placed on a 2 × 2 grid and routed to the
option their tumour biology actually supports.

---

## 2. The triage rule (first-line)

```
BRAF-V600 mutant
   immune-hot, low burden   -> IMMUNOTHERAPY FIRST      (durable; targeted kept in reserve)
   immune-hot, high burden  -> TARGETED -> IMMUNOTHERAPY (rapid control, then switch)
   immune-cold              -> TARGETED THERAPY FIRST     (BRAF/MEK inhibitor)
BRAF wild-type
   immune-hot               -> IMMUNOTHERAPY
   immune-cold (NRAS / WT)  -> TRIAL / CHEMOTHERAPY       (hardest group)
```

Chemotherapy (dacarbazine/temozolomide) sits as a fallback throughout, matching current
practice where it is essentially a last resort.

This logic is consistent with the clinical evidence base: in BRAF-mutant melanoma,
sequencing trials have shown **immunotherapy-first improves overall survival vs.
targeted-first** for most patients, while targeted therapy is favoured when rapid disease
control is needed (high burden / symptomatic) or when the tumour is immunologically cold.

---

## 3. Applying it to TCGA-SKCM

The four groups and their outcomes:

| First-line recommendation | n (%) | Median OS |
|---------------------------|------:|----------:|
| Immunotherapy first | 198 (47.0%) | **117.2 mo** |
| Targeted → Immunotherapy | 13 (3.1%) | 111.1 mo |
| Targeted therapy first | 94 (22.3%) | 71.8 mo |
| Trial / Chemotherapy | 116 (27.6%) | **50.1 mo** |

By molecular subtype: all 94 BRAF-V600/immune-cold patients route to targeted therapy;
BRAF-V600/immune-hot split between immunotherapy-first (85) and targeted→IO (13, high
burden); every NRAS and triple-WT patient routes to immunotherapy (if immune-hot) or the
trial/chemo group (if immune-cold).

**The survival ordering is exactly what the biology predicts** — immune-hot patients
(routed to immunotherapy) have the best prognosis (~117 months), and the BRAF-wild-type
immune-cold "hardest group" the worst (~50 months), with the targeted-therapy group in
between.

**Honest interpretation (important).** This survival gradient is an *internal consistency
check*, not proof that following the algorithm *causes* better outcomes. TCGA-SKCM is a
mostly pre-modern-therapy, mixed-treatment cohort, so patients did not actually receive
algorithm-guided first-line therapy. What the gradient shows is that the triage strata
capture real prognostic biology — a necessary condition for a useful triage — but a
prospective or trial-based validation (recommended-vs-received) would be needed to prove
benefit. The recommendations themselves rest on the established mechanism of each drug
class and published sequencing evidence, not on this correlation.

---

## 4. Where this fits the group's story

Q5 is the integration point of the whole project. Q3 (ODE) supplies the **targeted-therapy
axis** — who benefits from BRAF/MEK inhibition — and Q1 supplies the **immunotherapy axis**
— who benefits from checkpoint blockade. Neither answers the treatment-selection question
alone; together, gated by mutation status, they route each patient to the option their
tumour biology supports. The clinical triage tool in `clinical_triage_guide.md` turns this
into a bedside-style decision aid.

**Limitations.** Research-grade decision support only. The immune axis here is a
three-gene proxy (the full Q1 predictor is stronger); thresholds are cohort-relative
(median splits) and would be fixed against a reference population for deployment; and real
first-line choice also weighs LDH, disease stage, brain-metastasis status, performance
status, prior therapy, and patient preference, none of which are modelled here.

## 5. Using it on new patients (interactive predictor)

The triage is also packaged as a predictor that takes a **new dataset** and returns a
recommendation per patient — `scripts/predict_dataset.py`. Given a CSV with mutation
status (BRAF-V600, NRAS) and expression of eight genes (BRAF, MAP2K1/2, MAPK1/3, CD8A,
PRF1, GZMA), it normalises expression to the input dataset's own per-gene mean, runs the
Q3 ODE per patient, computes the immune score, and routes each patient on the frozen
reference thresholds:

```
python scripts/predict_dataset.py new_cohort.csv        # -> new_cohort_predictions.csv
python scripts/predict_dataset.py --demo                # built-in example patients
```

Example output (demo patients):

| Patient | Profile | Recommendation |
|---------|---------|----------------|
| BRAF-V600, immune-hot | targetable + inflamed | Immunotherapy first |
| BRAF-V600, immune-cold | targetable + excluded | Targeted therapy first |
| NRAS, immune-hot | not targetable + inflamed | Immunotherapy |
| triple-WT, immune-cold | not targetable + cold | Trial / Chemotherapy |

Thresholds are frozen in `reference/reference.json` (from the TCGA reference cohort).
**Caveat:** a single isolated patient cannot be self-normalised — run patients as a batch
or against a reference panel, and batch-harmonise genuinely cross-platform datasets first
(the same harmonisation care that applies to the Q1 predictor).

### Outputs
- `scripts/predict_dataset.py` — interactive predictor for new datasets
- `reference/reference.json` — frozen thresholds + reference constants
- `results/triage_assignments.csv` — per-patient recommendation + rationale + features
- `results/triage_summary.txt` — cohort distribution and survival by group
- `plots/triage_decision_matrix.png` — the BRAF × immune decision grid
- `plots/triage_km_by_recommendation.png` — survival by recommended strategy

### Reproduce
```bash
../q3-ode-model/venv/bin/python scripts/triage.py                     # cohort analysis
../q3-ode-model/venv/bin/python scripts/predict_dataset.py --demo     # new-patient prediction
```
