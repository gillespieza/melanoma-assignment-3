# 05 · Build Plan

Work top to bottom. Each step has an **acceptance check**. Do not mark a step done
until its check passes. Finish the session with the Verification block.

## Step 0 — Orient (5 min)
- Read `01`–`04` and `06`.
- `cd dashboard && npm install && npm run dev` — confirm the v1 app runs.
- Check if `dashboard/public/q1_predictions.csv` exists.
  - **Acceptance:** you can state current status and whether Q1 data is present.

## Step 1 — Generate `cohort.json`
- Extract the two Q3 CSVs from the `q3-ode-model` branch (see `03_DATA_SCHEMAS.md §A`).
- Write `dashboard/scripts/build_cohort.mjs` (Node) to produce `public/cohort.json`
  per the schema in `03 §C`. Include the Q1 merge (guarded on file presence) and heuristic Q4 flags.
- Run it; spot-check 3 patients (one BRAF-mut/PD-L1-low, one WT/PD-L1-high, one BRAF-mut/PD-L1-high).
  - **Acceptance:** `public/cohort.json` has ~421 patients; reductions match the archetype
    pattern (mut/low → high brafiReduction, low antipd1Reduction; WT/high → opposite).

## Step 2 — Cohort loader + types
- `src/data/cohort.ts`: `CohortPatient` type + `loadCohort()` (fetch `/cohort.json`).
- Load once in `App.tsx`; hold in state.
  - **Acceptance:** cohort count logs to console; no fetch errors.

## Step 3 — Cohort view (table + featured cards)
- `CohortTable.tsx`: columns per `02`. Search (by ID), filter (BRAF, PD-L1 band, agreement),
  sort (Q1 P(response), Q3 reduction, recommendation). Row click → Patient view.
- Featured archetype cards (A/B/C) on top for the demo.
- `ViewToggle` in `Header.tsx`.
  - **Acceptance:** table lists 421 patients, filters/sorts work, clicking opens a patient.

## Step 4 — Integration engine (`lib/integrationEngine.ts`)
- Implement `integrate(patient)` per `02` (blend Q1 pResponse + Q3 antipd1Reduction;
  targeted from brafiReduction gated on BRAF; combo; agreement/concordance; tiering; path).
- Refactor `decisionEngine.triage()` to share the core so archetypes stay consistent.
  - **Acceptance:** run the engine over all patients in a Node scratch script; print the
    distribution of primary recommendations + agreement; sanity-check a few by hand
    (WT/high → immuno; mut/low/high-LDH → targeted). Delete the scratch file after
    (may need the cowork file-delete tool — the mount blocks `rm`).

## Step 5 — Patient view lanes
- `PatientPassport.tsx` (demographics + molecular chips).
- `Q1Lane.tsx`: P(response) gauge + per-model bars + 6 signature scores + provenance.
  **If `patient.q1 == null`, render a designed "awaiting Q1 inference — run q1_infer.py"
  state** (not an error).
- `Q2Evidence.tsx`: cohort-level validation chips.
- Reuse `TumourForecastChart`/`SurvivalChart` for the **Q3 lane** (feed per-patient curves/reductions).
- `Q4Resistance.tsx`: resistance risk + reserve targets.
- `AgreementBadge.tsx` + Q5 lane: ranked options (`RecommendationPanel`), decision path
  (`DecisionTree`), sign-off (`ConsultantSignoff`).
  - **Acceptance:** every lane renders for a real cohort patient; Q1 gracefully handles null.

## Step 6 — Wire real Q1 (only if `q1_predictions.csv` present)
- Re-run `build_cohort.mjs` so `cohort.json` includes `q1`.
- Add a "model accuracy" mini-panel using the trial cohorts' `actual_response` vs `prob_ensemble`
  (real AUC / confusion) — strong "our AI is genuinely predictive" evidence.
  - **Acceptance:** Q1 lane shows real probabilities; agreement badge reflects Q1×Q3.

## Step 7 — Polish
- Apply `04_DESIGN_SYSTEM.md` quality checklist. Loading/empty states. Projector legibility.
- Update `dashboard/README.md` to describe the multi-method version.

## Verification (ALWAYS run before finishing)
```bash
cd dashboard
npm run typecheck          # must be clean
npm run build              # tsc --noEmit + vite build must succeed
# engine sanity: node scratch over cohort.json (recommendation distribution + a few hand checks)
npx vite --port 5199 &     # smoke test: curl -s -o /dev/null -w "%{http_code}" localhost:5199/  → 200
```
- Confirm: table loads 421, a WT/PD-L1-high patient → immunotherapy primary, a
  BRAF-mut/PD-L1-low/high-LDH patient → targeted primary, agreement badge renders,
  Q1 lane handles both present/absent data.

## Guardrails
- Front-end only. No backend, no localStorage/sessionStorage (breaks artifacts/build).
- Don't fabricate Q1 numbers — gate on the real file (decision locked, see `06`).
- Reuse v1 components; don't rewrite working charts.
- The mounted folder blocks `rm` from bash — use the cowork file-delete tool to remove temp files.
- Keep everything on branch `the-dashboard`. Commit locally; don't push unless asked.
