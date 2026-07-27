# 02 · Target Architecture

## Tech stack (already chosen, don't change)

React 18 + TypeScript + Vite 5 + Tailwind CSS 3 + Recharts 2 + Framer Motion +
lucide-react. Front-end only. No backend, no browser storage APIs. Node 18+.

## What exists today (v1, Q3-only)

```
dashboard/src/
├── App.tsx                      # orchestrator: patient state, run/sim, results reveal
├── data/
│   ├── types.ts                 # PatientInput, RankedOption, TriageResult, etc.
│   ├── patients.ts              # 3 hardcoded archetypes (A/B/C) + BLANK_PATIENT
│   └── model.ts                 # REAL q3 dose-response group means + KM facts + ODE modules
├── lib/
│   ├── decisionEngine.ts        # triage(): ranks immuno/targeted/combo from a patient
│   └── forecast.ts              # buildForecast() 12-mo curves + buildSurvival() KM curves
└── components/
    ├── Header.tsx  ui.tsx  PatientIntake.tsx  SimulationOverlay.tsx
    ├── RecommendationPanel.tsx  TumourForecastChart.tsx  SurvivalChart.tsx
    ├── DecisionTree.tsx  ConsultantSignoff.tsx
```

Keep all of this. The rebuild *extends* it.

## Target v2 (multi-method, all-421-patient)

### New top-level information architecture

Two views, toggled in the Header:

1. **Cohort view** (new default landing): a searchable/sortable/filterable table of
   ~421 real TCGA patients. Columns: ID, BRAF, NRAS, PD-L1 band, Q1 P(response),
   Q3 predicted best lane, Q5 recommendation, methods-agreement badge. Clicking a row
   opens that patient in the Patient view.
2. **Patient view** (the "full patient story"): the multi-lane layout below.

Keep the existing 3 archetypes as **featured quick-pick cards** on the Cohort view for
the scripted live demo.

### Patient view — the multi-lane story

Vertical stack (or responsive grid) of method "lanes", each a `Panel`:

```
┌─ Patient passport ─────────────────────────────────────────┐
│ demographics + molecular chips (BRAF, NRAS, PD-L1, stage…)  │
├─ Q1 · Gene-Expression Response Predictor (ML) ─────────────┤
│ P(response) gauge + per-model bars (LR/RF/XGB/SVM/ENet) +   │
│ the 6 signature scores + AUC/provenance. Reads cohort.json  │
│ (from q1_predictions.csv). If absent → "awaiting inference".│
├─ Q2 · Cell-line Validation (evidence chip row) ────────────┤
│ cohort-level supporting evidence; not per-patient numbers.  │
├─ Q3 · ODE Digital Twin ────────────────────────────────────┤
│ tumour-burden forecast + survival (REUSE existing charts).  │
├─ Q4 · Resistance & Salvage Targets ────────────────────────┤
│ resistance-risk flags + reserve options (SOX10 etc).        │
├─ Q5 · Integrated Recommendation ───────────────────────────┤
│ ranked options + METHODS-AGREEMENT signal + decision path + │
│ consultant sign-off. This is the synthesis of Q1+Q3(+Q4).   │
└────────────────────────────────────────────────────────────┘
```

### Data flow

```
cohort.json  ──►  cohortStore (in-memory, loaded once in App)
   │                      │
   │                      ├─► CohortTable (list/filter/sort)
   │                      └─► PatientView(selectedId)
   │                                  │
q1_predictions.csv ─(merged at build)─┘
                                       │
                          integrationEngine.ts  (Q5)
                                       │
             ┌───────────┬────────────┼───────────┬───────────┐
           Q1 lane     Q3 lane     agreement   ranked opts  decision path
```

- `cohort.json` is the single source of truth, generated once (see `03_DATA_SCHEMAS.md`).
  It already contains each patient's real Q3 dose-response + molecular fields, and — once
  the user runs inference — the merged Q1 probabilities.
- App loads `cohort.json` from `public/` via `fetch` on mount (or imports a bundled JSON).
- No network calls beyond that static file.

### The Q5 integration engine (`lib/integrationEngine.ts`) — NEW

This generalises the current `decisionEngine.triage()`. Signature:

```ts
integrate(patient: CohortPatient): IntegratedResult
```

Inputs per patient (from cohort.json):
- Q1: `pResponse` (ensemble immunotherapy-response probability, 0–1) + per-model probs.
- Q3: real per-patient `brafiReduction`, `antipd1Reduction` (from dose-response), BRAF/NRAS, PD-L1.
- Q4: derived resistance flags.

Logic:
1. **Immunotherapy score** = blend of Q1 `pResponse` (statistical) and Q3
   `antipd1Reduction` (mechanistic). Weight ~0.5/0.5; expose both contributions.
2. **Targeted score** = Q3 `brafiReduction`, gated on BRAF-mutant; boosted by high LDH.
3. **Combination score** = as today, needs BRAF-mut + immuno-responsive biology.
4. **Methods-agreement**: compare the *direction* of Q1 and Q3 for immunotherapy.
   - Both high → `Concordant · high confidence`.
   - Split → `Discordant · consultant review` (great talking point).
   Compute a simple concordance score = 1 − |normalise(Q1) − normalise(Q3_immuno)|.
5. Rank options; assign primary/alternative/not-recommended (reuse tiering from v1).
6. Produce the decision-tree path (Stage → BRAF → PD-L1/signature → Q1×Q3 agreement → rec).

Keep the existing `decisionEngine.triage()` working for the 3 editable archetypes (which
have no real Q1 row); `integrate()` is the cohort path. Consider refactoring `triage()` to
call a shared core so behaviour is consistent.

### Reuse map (do NOT rewrite these)

| Need | Reuse |
|------|-------|
| Tumour forecast chart | `TumourForecastChart.tsx` (feed per-patient reductions) |
| Survival chart | `SurvivalChart.tsx` |
| Ranked option cards | `RecommendationPanel.tsx` |
| Decision path | `DecisionTree.tsx` |
| Sign-off | `ConsultantSignoff.tsx` |
| Panels/pills/stats | `ui.tsx` |
| Simulation overlay | `SimulationOverlay.tsx` |

### New components to add

`CohortTable.tsx`, `PatientPassport.tsx`, `Q1Lane.tsx`, `Q2Evidence.tsx`,
`Q4Resistance.tsx`, `AgreementBadge.tsx`, and a `ViewToggle` in `Header.tsx`.
New lib: `integrationEngine.ts`, `cohort.ts` (loader/types).

## Performance note

421 patients × (2 curves × 10 pts + scalar fields) is small (~200–400 KB JSON). Fine to
bundle or fetch. Virtualise the table only if scroll feels heavy (probably unnecessary).
Recharts is the heaviest dep; the v1 bundle is ~690 KB (203 KB gzip) — acceptable for a demo.
