# OncoTwin Dashboard – Handoff & Build Docs

**Read this first.** This folder is a self-contained brief for building the
OncoTwin clinical decision-support dashboard (the visual deliverable for **Q5**
of the Melanoma Digital Twin project). It is written so a *fresh session* – even
with a smaller model – can produce high-quality output without re-deriving
context. Follow the docs in order; each is focused and concrete.

## The one-paragraph summary

We are building a premium, clinician-facing web dashboard that presents a
melanoma patient's **full multi-method story** – the Q1 gene-expression ML
predictor, Q2 cell-line validation, the Q3 ODE digital twin, and Q4 resistance
analysis – all converging into a single ranked **Q5 treatment recommendation**
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
| `07_SESSION_LOG.md` | **Current state, git/deploy setup, and how to wire in the Q2 cell-line data** |
| `08_USER_GUIDE.md` | Complete, zero-assumed-background explanation of every screen, field, and term for anyone using the dashboard (not building it) |

## Current status – BUILD PLAN COMPLETE

The multi-method v2 described in these docs is **built, verified and committed** on
`the-dashboard`. Steps 0–7 of `05_BUILD_PLAN.md` are all done and their acceptance
checks pass. What exists now:

- `scripts/build_cohort.mjs` → `public/cohort.json` (421 patients, real Q3 + TCGA clinical).
- Three views: **Cohort** (filterable table + featured real cases + Q1 accuracy panel),
  **Patient** (Q1→Q2→Q3→Q4→Q5 lanes), **Archetypes** (the original editable v1 workbench).
- `lib/scoring.ts` shared core, with `decisionEngine` (archetypes) and
  `integrationEngine` (cohort + methods agreement) both running through it.
- Typecheck clean, production build succeeds, engine sanity-checked over all 421 patients
  (8/8 assertions pass), rendering verified by screenshot in a real browser engine.

**Q1 is still cohort-level only, and that is now understood rather than pending.**
Re-running `q1_infer.py` against the (now present) TCGA expression matrix still fails:
the matrix is missing every IMPRES co-stimulatory gene partner, and its normalisation
scale differs from the training cohorts. See the root-cause section in `06_DECISIONS.md`.
The Q1 lane shows a designed "awaiting inference" state and lights up automatically if
real per-patient rows ever appear in `q1_predictions.csv`.

## How to work in the next session

1. Read `06_DECISIONS.md` first – it now carries the root-cause analysis and decisions
   D10–D15 taken during the build.
2. If the goal is genuine per-patient Q1: that is **Q1-pipeline work**, not dashboard work
   (fix the TCGA gene panel + normalisation, then re-run `q1_infer.py` and
   `node scripts/build_cohort.mjs`). The front end needs no changes.
3. If the goal is dashboard polish: run the app, then work from the quality checklist in
   `04_DESIGN_SYSTEM.md`.
4. Always finish with the verification step (typecheck + build + engine sanity + dev-server smoke test).

## Run commands

```bash
cd dashboard
npm install       # first time
npm run dev       # http://localhost:5173
npm run build     # tsc --noEmit + vite build (must pass before done)
npm run typecheck
```
