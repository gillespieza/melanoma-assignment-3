# 01 · Project Context

## The academic project

UCD "AI in Personalised Medicine" group assignment: **Melanoma Digital Twin** — a
computational-biology pipeline predicting how melanoma patients respond to
treatment. Group presentation date: **August 4th**. The dashboard is the visual
representation of **Q5**.

The project is split into five questions, each a *different method*:

| Q | Owner | Method type | What it produces |
|---|-------|-------------|------------------|
| **Q1** | Amanda | Statistical / ML | Immunotherapy-response predictor. 5 trained models (LogReg, RandomForest, XGBoost, SVM, ElasticNet) over a curated 6-signature feature set. Predicts CR/PR vs PD to anti-PD-1. AUC ≈ 0.67. |
| **Q2** | Gift | Experimental | Applies the Q1 signature to melanoma **cell lines** to validate against real drug response. |
| **Q3** | Jinish (BRAF part) & Anthony | Mechanistic | An **ODE digital twin**: 4 coupled modules (RAF-dimer, MAPK cascade, tumour–immune, PD-1/PD-L1 checkpoint). Simulates tumour burden + survival under BRAF/MEK vs anti-PD-1. |
| **Q4** | Hena | Target/resistance | Drug targets & resistance mechanisms (e.g. SOX10 knockout). Second-line / salvage logic. |
| **Q5** | Whole team | Integration | Combine Q1–Q4 into a **treatment decision tree**. **← the dashboard is this.** |

The key insight driving the current design: **the ODE (Q3) is only one method.**
The dashboard must tell the *integrated* story — multiple methods converging on one
recommendation, with a "methods agree / disagree" signal. That agreement signal is
the most impressive thing to show a clinical panel (e.g. "the ML predictor *and* the
mechanistic model both favour immunotherapy → high-confidence call").

## The clinical decision framing (non-negotiable)

The app is **decision support**, not an oracle. It presents ranked, evidence-anchored
options; the **consultant oncologist confirms the final plan and is accountable**.
Every screen should reinforce this. There is a consultant sign-off step. The
signed-off consultant persona is **Dr. Aoife Gallagher** (fictional Irish name — do
NOT use real names).

### How real melanoma treatment decisions actually work (use this for the logic)

- Order of assessment: **stage → LDH → BRAF mutation test → lane choice.**
- **BRAF V600 status** (a DNA test) decides whether targeted therapy is even possible.
- Current guidelines favour **immunotherapy first** for most advanced patients
  (DREAMseq: 72% vs 52% alive at 2y immuno-first; SECOMBIT confirms), even for
  BRAF-mutant disease — because immunotherapy responses are *durable*, while targeted
  responses are fast but hit *resistance*.
- **Targeted therapy (BRAF/MEK)** is reserved for BRAF-mutant patients needing rapid
  control (high LDH, bulky/symptomatic) or when immunotherapy is unsuitable.
- **PD-L1** in melanoma is a *weak* selector (unlike lung); immunotherapy is given even
  to PD-L1-low patients. Q1's expression signature is the project's *better-than-PD-L1*
  contribution.
- Gene-expression signatures (Q1) are currently a **research tool**, not routine
  clinical practice — frame the app as a near-future decision aid. This is honest and
  defensible.

## The user

Jinish — nursing/clinical background, not a developer. Explanations should be plain.
Values scientific honesty ("is this real?" must have a good answer). Prefers concise,
direct communication. The dashboard must look **premium / clinical-grade** ("like $100M
software"), clinical-white aesthetic, highly professional.

## What must stay honest

- Numbers on screen should trace to real model output wherever possible (see `03`).
- Where a value is a projection/derivation (e.g. the 12-month *shape* of a curve), label
  it as such. Never overclaim. The v1 already does this on the forecast chart footnote.
