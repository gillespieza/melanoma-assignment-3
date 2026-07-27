# OncoTwin Dashboard — Handoff & Build Docs

**Read this first.** This folder is a self-contained brief for building the
OncoTwin clinical decision-support dashboard (the visual deliverable for **Q5**
of the Melanoma Digital Twin project). It is written so a *fresh session* — even
with a smaller model — can produce high-quality output without re-deriving
context. Follow the docs in order; each is focused and concrete.

## The one-paragraph summary

We are building a premium, clinician-facing web dashboard that presents a
melanoma patient's **full multi-method story** — the Q1 gene-expression ML
predictor, Q2 cell-line validation, the Q3 ODE digital twin, and Q4 resistance
analysis — all converging into a single ranked **Q5 treatment recommendation**
that a consultant oncologist confirms. It runs over a browsable cohort of ~421
real TCGA-SKCM patients. It is front-end only (React + TS + Vite + Tailwind +
Recharts); all model outputs are baked into data files, no live backend.

## Document map

| File | What it covers |
|------|----------------|
| `01_CONTEXT.md` | The science (Q1–Q5), the clinical decision framing, who the user is |
| `02_ARCHITECTURE.md` | Target file tree, component breakdown, data flow, the Q5 integration engine design |
| `03_DATA_SCHEMAS.md` | Every data source, exact schemas for `cohort.json` + `q1_predictions.csv`, and how to generate them |
| `04_DESIGN_SYSTEM.md` | Clinical-white theme tokens, component conventions, the "$100M professional" bar |
| `05_BUILD_PLAN.md` | Ordered task list with acceptance criteria + verification steps |
| `06_DECISIONS.md` | Every decision locked so far (so nothing gets re-litigated) |

## Current status (as of this handoff)

- **Branch:** `the-dashboard` (off `q5-treatment-triage`). Work is committed locally, not pushed.
- **Built and working:** a v1 dashboard that tells the **Q3-only** story — 3 hardcoded
  patients, live intake form, ODE simulation overlay, ranked options, tumour-burden
  forecast, survival chart, decision tree, consultant sign-off. It typechecks, builds,
  and runs (`npm run dev`). This is the foundation to extend, NOT to throw away.
- **The pivot (this is the job):** rebuild it into the **multi-method, all-421-patient**
  version described in these docs.
- **Q1 data — PARTIALLY DELIVERED.** `dashboard/public/q1_predictions.csv` now EXISTS with
  **256 genuine predictions** across the ICI trial cohorts (Liu 122, Riaz 107, Hugo 27),
  each WITH a real `actual_response` label → enables a real "model accuracy / AUC" panel.
  **BUT the 421 TCGA twin cohort did NOT score** (TCGA cleaning didn't complete), and the
  trial IDs (`LIU_PATIENT1`…) don't join to the Q3 TCGA IDs (`TCGA-3N-A9WB`). So:
    - Q1 as a **validation/accuracy panel** (trial cohorts, real labels) → ready to build now.
    - Q1 as a **per-TCGA-patient probability** in the twin cohort → still pending; needs the
      user to finish cleaning TCGA (`data/processed/skcm_tcga_pan_can_atlas_2018/expr_cleaned.csv`)
      and re-run `q1_infer.py`. See `03_DATA_SCHEMAS.md` and `06_DECISIONS.md`.

## How to work in the next session

1. Read `01`–`06` in order (they're short).
2. Confirm whether `dashboard/public/q1_predictions.csv` exists yet.
   - If yes → wire the real Q1 numbers in.
   - If no → build everything else; leave the Q1 slot gated (it auto-lights-up when the file lands).
3. Work through `05_BUILD_PLAN.md` top to bottom. Each task has an acceptance check.
4. Always finish with the verification step (typecheck + build + engine sanity + dev-server smoke test).

## Run commands

```bash
cd dashboard
npm install       # first time
npm run dev       # http://localhost:5173
npm run build     # tsc --noEmit + vite build (must pass before done)
npm run typecheck
```
