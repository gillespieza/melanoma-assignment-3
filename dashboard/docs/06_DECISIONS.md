# 06 · Decisions Log

Locked decisions — do not re-litigate in the next session.

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Clinical-white SaaS** aesthetic (not dark, not playful) | User chose it; reads as trustworthy to clinicians |
| D2 | **Fully interactive branching** — editing molecular fields live re-runs the logic | User chose it; strongest live-demo behaviour |
| D3 | **Integrate all 5 methods** (Q1 ML, Q2 cell-line, Q3 ODE, Q4 resistance → Q5) | User's insight: ODE is only one method; Q5 is the integration |
| D4 | **Genuine Q1 only** — no fabricated ML numbers | User was explicit: run the real models, wire in real output. Q1 lane gates on `q1_predictions.csv`. |
| D5 | **All ~421 real TCGA patients**, browsable cohort + 3 featured archetypes | User chose full cohort; archetypes remain for the scripted demo |
| D6 | Consultant persona = **Dr. Aoife Gallagher** (fictional Irish) | User asked to replace the earlier placeholder; never use real names |
| D7 | **Decision support, not oracle** — consultant confirms & signs off | Clinical correctness; the app presents options, human decides |
| D8 | Branch **`the-dashboard`**, commit locally, don't push | User instruction |
| D9 | Keep v1 (Q3-only) components and extend, don't rewrite | They typecheck/build/run; save tokens |

## Known constraints / gotchas discovered

- **cBioPortal is blocked in the build sandbox** (403). The Q1 pipeline must run on the
  user's Mac. That's why Q1 data is a handoff dependency.
- **Q1 models were pickled with an older scikit-learn**; the user's venv now has 1.9.0.
  `q1_infer.py` includes a `compat_fix()` shim (adds `LogisticRegression.multi_class`) and
  per-model try/except. If models still error, pin sklearn to the training version.
- **Nested-repo path bug:** `q1-response-predictor` is nested inside the larger
  `melanoma-assignment-3` git repo, so `find_project_root` resolves `data/` and `logs/` to
  the PARENT. Fixed in `download_data.py` + `clean_data.py` (added `ROOT_DIR`). The full
  pipeline scripts (`merge_datasets.py`, `run_pipeline.py`) still have the same bug — patch
  them the same way if the user runs them.
- **The mounted workspace blocks `rm` from bash** ("Operation not permitted"). Use the
  cowork file-delete tool to remove temp files.
- **Q3 checkpoint CSV** (with CD274/PDCD1) is only on the `q3-ode-model` branch under
  `outputs/results/` — extract via `git show` when building `cohort.json`.
- **Two cohorts don't fully overlap:** Q1 models were trained/validated on ICI trial
  cohorts (Liu/Hugo/Riaz, ~195 patients with real response labels); Q3 twin is 421 TCGA
  (survival-only, no response labels). Join Q1's TCGA rows to Q3; use the trial cohorts for
  the "real model accuracy" evidence panel.

## Q1 status update (resolved during handoff session)
- `q1_infer.py` RAN successfully → `dashboard/public/q1_predictions.csv` committed with
  **256 real trial-cohort predictions** (Liu 122, Riaz 107, Hugo 27), each with a real
  `actual_response` label. Compute real AUC/confusion from `prob_ensemble` vs `actual_response`.
- **TCGA did NOT score** → no per-patient Q1 for the 421 twin cohort yet. Trial IDs
  (`LIU_PATIENT1`) don't join to Q3 TCGA IDs.

## Open questions for the user (ask early next session)
- Do you want to finish cleaning TCGA and re-run `q1_infer.py` so the twin cohort gets
  per-patient Q1 probabilities? (Needed for a true per-TCGA-patient Q1×Q3 agreement badge.)
  If not, Q1 becomes a **cohort-level validation/accuracy panel** and the per-patient
  agreement is computed from Q3 + molecular signature only.
- Feature the "our AI predicts response at AUC ≈ 0.6–0.7 on held-out trials" panel prominently?
