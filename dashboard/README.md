# OncoTwin™ — Melanoma Digital-Twin Decision Support (Q5)

A clinician-facing decision-support cockpit for the Melanoma Digital Twin project
(UCD AI in Personalised Medicine). It is the **visual representation of Q5**: the
point where every method in the project — the Q1 ML predictor, Q2 experimental
validation, the Q3 ODE digital twin and Q4 resistance analysis — converges on one
ranked treatment recommendation.

The framing is deliberate: the app **presents ranked, evidence-anchored options**
and the **consultant oncologist confirms the final plan**. It is decision *support*,
not a directive.

## Run it

```bash
cd dashboard
npm install                     # first time only
node scripts/build_cohort.mjs   # regenerate public/cohort.json (already committed)
npm run dev                     # Vite on http://localhost:5173
```

Other scripts:

```bash
npm run build      # type-check + production build into dist/
npm run preview    # serve the production build
npm run typecheck  # tsc --noEmit
```

Requires Node 18+ (tested on Node 22).

## The three views

Switch between them in the header; the URL hash tracks the view, so a patient is
linkable (`#/patient/TCGA-3N-A9WC`) and survives a refresh.

### 1 · Cohort — the landing view

All **421 real TCGA-SKCM patients** that have a complete ODE digital twin,
searchable by ID and filterable by BRAF status, PD-L1 band and methods-agreement.
Every row is scored by the Q5 engine up front, so you can filter straight to the
**discordant** cases — the ones where the methods disagree and a consultant
genuinely has to decide.

Above the table, three **featured cases** are picked at render time from the real
cohort because they cleanly exhibit the classic archetypes (BRAF-mut/PD-L1-low,
BRAF-WT/PD-L1-high, BRAF-mut/PD-L1-high). Below it sits the Q1 model-accuracy panel.

### 2 · Patient — run the twin, then explore it

Open any patient and press **Run Digital-Twin Simulation**. The ODE module sequence
plays, then the integrated recommendation, the methods-agreement badge and the ranked
options appear.

Above them sits the **what-if explorer**: change BRAF, NRAS, stage or PD-L1 and every
method re-runs instantly, so you can show what *would* have been recommended had the
biology been different. An edited patient is badged **Modified** throughout, and because
their real ODE curves no longer apply, the twin falls back to the matching cohort-average
sweep — stated on screen, never silently.

Method detail sits behind tabs so only one is on screen at a time:

| Tab | What it shows |
|-----|---------------|
| **Q1** · ML predictor | P(response) gauge, cohort percentile, per-model bars where available |
| **Q2** · Validation | RPPA proteomic validation + the ML-vs-ODE benchmark |
| **Q3** · Digital twin | The patient's own dose-response sweep, 12-month forecast, survival |
| **Q4** · Resistance | Escape mechanisms flagged and salvage targets to hold in reserve |
| **Decision path** | Stage → BRAF → PD-L1 → methods agreement → recommendation |

### What is computed live, and what is not

Worth being precise about, because it is a fair question to be asked:

- **Live, in the browser:** the entire Q5 layer — arm scoring, ranking, tiering, the
  methods-agreement calculation, the decision path, and the forecast/survival curves.
  Every what-if edit genuinely re-runs all of it.
- **Precomputed:** the Q3 ODE solutions (solved in Python for all 421 patients) and the
  Q1 model predictions (scikit-learn models that cannot run in a web page).

That split is normal for deployed clinical software — you serve model outputs, you do not
re-solve an ODE in a browser tab.

### 3 · Archetypes — the editable workbench

The original three hand-built patients with live-editable clinical and molecular
fields. Editing any field re-runs the logic immediately. Kept for the scripted demo.

## Where the numbers come from

Everything on screen traces to real project output. The app reads exactly one
generated file, `public/cohort.json`, built by `scripts/build_cohort.mjs` from:

| Source | Supplies |
|--------|----------|
| `q3-ode-model/outputs/results/tumour_burden_simulations.csv` | BRAF-inhibitor dose sweep |
| `q3-ode-model/outputs/results/checkpoint_tumour_simulations.csv` | Anti-PD-1 sweep, CD274, PDCD1 |
| `data/processed/skcm_tcga_pan_can_atlas_2018/clin_cleaned.csv` | Age, sex, stage, TMB, survival |
| `public/q1_predictions.csv` | Q1 per-model + ensemble probabilities (trial cohorts) |
| `public/q1_tcga_scores.csv` | Q1 per-patient TCGA response scores (from `origin/main`) |
| `q3-ode-model/outputs/results/survival_summary.txt` | KM medians (in `src/data/model.ts`) |

The generator is idempotent — re-run it any time.

### What is real, and what is a projection

Being precise about this matters more than the numbers looking good:

- **Real:** every molecular and clinical field; both dose-response curves per patient;
  the Kaplan–Meier medians; the Q1 accuracy figures (AUC 0.593 over 195 held-out trial
  patients with true response labels — 0.766 in Riaz 2017, 0.451 in the small Hugo set);
  the RPPA validation (n=310, r=0.175, p=0.002).
- **Projection:** the *shape* of the 12-month forecast between endpoints, and the
  survival curves, which are Weibull fits anchored to the real KM medians. Labelled
  as such on the charts.
- **Heuristic:** the Q4 resistance flags. Rule-based, and the panel says so.

### Two honest caveats the UI surfaces

1. **Q1 supplies one ensemble score per TCGA patient, not the per-model breakdown.**
   All 421 patients carry a real Q1 P(response) from the Q1 workstream's
   `patient_predicted_response_scores.csv`. That file has a single score per patient, so
   the Q1 lane shows the gauge and cohort percentile but not the five per-model bars.
   If `q1_infer.py` ever produces TCGA rows with the full breakdown, the generator prefers
   those automatically and the bars appear with no code change.
2. **~21% of twins are uninformative.** For 93 of 421 patients the ODE settles at a
   numerically-zero tumour compartment. That is *not* drug resistance — the model has
   nothing to say — so those patients are flagged rather than shown as 0% response, and
   the agreement badge reports "single method" instead of inventing a comparison.

## Project structure

```
dashboard/
├── scripts/build_cohort.mjs        # generates public/cohort.json from real outputs
├── public/
│   ├── cohort.json                 # the single file the app consumes
│   └── q1_predictions.csv          # Q1 model output
├── src/
│   ├── App.tsx                     # view routing + cohort loading
│   ├── data/
│   │   ├── types.ts                # clinical + molecular domain types
│   │   ├── cohort.ts               # CohortPatient types + loadCohort()
│   │   ├── patients.ts             # the 3 editable archetypes
│   │   └── model.ts                # REAL q3 group dose-response + KM facts
│   ├── lib/
│   │   ├── scoring.ts              # shared scoring core (both engines run through it)
│   │   ├── decisionEngine.ts       # archetype path
│   │   ├── integrationEngine.ts    # Q5 cohort path + methods agreement
│   │   └── forecast.ts             # 12-month forecast + survival curves
│   └── components/                 # cohort table, lanes, charts, agreement badge, sign-off
```

Both engines share `scoring.ts`, so a cohort patient and a hand-edited archetype with
the same profile always produce the same recommendation.

## Tech

React + TypeScript + Vite · Tailwind CSS · Recharts · Framer Motion · lucide-react.
All front-end; no backend, no live Python, no browser storage — model outputs are baked in.

> Research demonstrator. Not a medical device. Outputs are model projections, not clinical directives.
