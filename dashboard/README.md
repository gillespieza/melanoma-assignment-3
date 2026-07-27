# OncoTwin™ — Melanoma Digital-Twin Decision Support (Q5)

A clinician-facing decision-support cockpit for the Melanoma Digital Twin project
(UCD AI in Personalised Medicine). It is the **visual representation of Q5**: the
treatment-triage decision tree, turned into a "doctor's dashboard".

The framing is deliberate: the app **presents ranked, evidence-anchored options**
and the **consultant oncologist confirms the final plan**. It is decision *support*,
not a directive.

## Run it

```bash
cd dashboard
npm install      # first time only
npm run dev      # starts Vite on http://localhost:5173
```

Other scripts:

```bash
npm run build      # type-check + production build into dist/
npm run preview    # serve the production build
npm run typecheck  # tsc --noEmit
```

Requires Node 18+ (tested on Node 22).

## The demo flow

1. **Patient intake & molecular workup** (left panel) — pick Patient A/B/C or edit any
   field (age, ECOG, AJCC stage, LDH, BRAF, PD-L1, Q1 signature). The twin re-simulates live.
2. **"Running ODE Digital-Twin Simulation"** — an animated loading state that names the four
   real model modules (RAF-dimer → MAPK cascade → tumour–immune → checkpoint axis).
3. **Results cockpit** — ranked treatment options with confidence, predicted median OS and
   12-month burden reduction; a **Tumour Burden Forecast** chart; a **Predicted Survival**
   projection; the **Q5 decision pathway**; and a **consultant sign-off** bar.

## The three demo patients

| Patient | Molecular profile | What the model recommends |
|---------|-------------------|---------------------------|
| **A** — Ms R. Doyle | BRAF V600E · PD-L1 3% · high LDH, symptomatic | **Targeted therapy** (rapid control; immunotherapy predicted ~2% response) |
| **B** — Mr T. Okafor | BRAF WT · PD-L1 65% | **Immunotherapy** (targeted correctly ineligible — no BRAF target) |
| **C** — Ms L. Bianchi | BRAF V600E · PD-L1 55% | **Immunotherapy** primary, **Combination/sequencing** close behind (the debate) |

## Where the numbers come from (this is real, not invented)

The charts and rankings are anchored to the group-averaged tumour-burden dose-response
sweeps produced by the **q3 ODE model** on TCGA-SKCM (n=421):

- `q3-ode-model/outputs/results/tumour_burden_simulations.csv` (BRAF-inhibitor sweep)
- `q3-ode-model/outputs/results/checkpoint_tumour_simulations.csv` (anti-PD-1 sweep)
- Kaplan–Meier medians from `q3-ode-model/outputs/results/survival_summary.txt`
  (checkpoint tumour burden: median OS 148.2 vs 65.9 mo, log-rank p = 0.0024).

These values live in `src/data/model.ts`. The 12-month forecast *endpoints* come from that
real dose-response; the month-by-month *dynamics* encode the well-known clinical pattern
(fast BRAF/MEK nadir with resistance rebound vs slower, durable checkpoint response).

## Project structure

```
dashboard/
├── index.html
├── src/
│   ├── App.tsx                     # orchestrator + layout + live re-simulation
│   ├── data/
│   │   ├── types.ts                # clinical + molecular domain types
│   │   ├── patients.ts             # 3 hardcoded demo patients
│   │   └── model.ts                # REAL q3 ODE output (dose-response + KM facts)
│   ├── lib/
│   │   ├── decisionEngine.ts       # transparent triage / ranking logic
│   │   └── forecast.ts             # 12-month forecast + survival curves
│   └── components/                 # Header, PatientIntake, SimulationOverlay,
│                                   # RecommendationPanel, charts, DecisionTree, sign-off
```

## Tech

React + TypeScript + Vite · Tailwind CSS · Recharts · Framer Motion · lucide-react.
All front-end; no backend, no live Python — the model outputs are baked in.

> Research demonstrator. Not a medical device. Outputs are model projections, not clinical directives.
