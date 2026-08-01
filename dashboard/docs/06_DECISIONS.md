# 06 · Decisions Log

Locked decisions – do not re-litigate in the next session.

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Clinical-white SaaS** aesthetic (not dark, not playful) | User chose it; reads as trustworthy to clinicians |
| D2 | **Fully interactive branching** – editing molecular fields live re-runs the logic | User chose it; strongest live-demo behaviour |
| D3 | **Integrate all 5 methods** (Q1 ML, Q2 cell-line, Q3 ODE, Q4 resistance → Q5) | User's insight: ODE is only one method; Q5 is the integration |
| D4 | **Genuine Q1 only** – no fabricated ML numbers | User was explicit: run the real models, wire in real output. Q1 lane gates on `q1_predictions.csv`. |
| D5 | **All ~421 real TCGA patients**, browsable cohort + 3 featured archetypes | User chose full cohort; archetypes remain for the scripted demo |
| D6 | Consultant persona = **Dr. Aoife Gallagher** (fictional Irish) | User asked to replace the earlier placeholder; never use real names |
| D7 | **Decision support, not oracle** – consultant confirms & signs off | Clinical correctness; the app presents options, human decides |
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
  pipeline scripts (`merge_datasets.py`, `run_pipeline.py`) still have the same bug – patch
  them the same way if the user runs them.
- **The mounted workspace blocks `rm` from bash** ("Operation not permitted"). Use the
  cowork file-delete tool to remove temp files.
- **Q3 checkpoint CSV** (with CD274/PDCD1) is only on the `q3-ode-model` branch under
  `outputs/results/` – extract via `git show` when building `cohort.json`.
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

## Q1 × TCGA – root cause found (dashboard build session)

The earlier assumption that TCGA "just needs cleaning to finish" is **wrong**.
`data/processed/skcm_tcga_pan_can_atlas_2018/expr_cleaned.csv` now exists (443 samples ×
3003 genes) and `q1_infer.py` was re-run against it. It still reports *"no scorable
samples"*, for two independent reasons:

1. **Missing genes.** The matrix was reduced to ~3000 features, and while 25 of the 40
   signature genes survive (as unmapped Entrez IDs – `entrez_to_symbol_cache.json` does
   not cover them), **all of the IMPRES co-stimulatory partners are absent**: CD28, CD86,
   CD80, CD40, CD200, CD276, TNFRSF4, TNFRSF14, TNFSF9, HAVCR2, VSIR (plus STAT1, CMKLR1,
   HLA-E, PSMB10). Zero of the 15 IMPRES pairs are computable, so IMPRES – one of the six
   required model features – cannot be produced at all.
2. **Scale mismatch.** TCGA values run 0–21.7 (median 4.61); the trial cohorts the scaler
   was fit on run 0–4.4 (median 1.74). Even with every gene present, pushing TCGA through
   `final_scaler.pkl` would produce meaningless probabilities.

**To actually unblock it:** re-run TCGA cleaning retaining the full signature gene panel
(mapping Entrez → symbol), and harmonise normalisation with the training cohorts – or
re-fit the scaler/models on a jointly-normalised matrix. This is Q1-pipeline work, not
dashboard work. Until then, fabricating per-patient Q1 would violate D4.

## Decisions taken during the dashboard build

| # | Decision | Rationale |
|---|----------|-----------|
| D10 | Q1 appears as a **cohort-level accuracy panel**; the per-patient lane shows a designed "awaiting inference" state | D4 – no fabricated Q1 numbers. Lane auto-fills when real data lands. |
| D11 | The statistical side of the agreement badge falls back to a **measured checkpoint-biomarker composite** (PD-L1 + PD-1 + TMB percentiles), always labelled as such | Keeps the multi-method story real using measured data, never implying it is Q1 output |
| D12 | Patients whose ODE baseline is < 1e-3 are flagged **"twin below model resolution"**, not shown as 0% response | 93/421 patients. "The model can't say" and "the drug won't work" are clinically different claims |
| D13 | Reduction percentiles ranked among informative twins put the mechanistic read-out on the same 0–100 scale as the statistical one | Makes the concordance number principled rather than an arbitrary normalisation |
| D14 | Chart/entry animations are CSS or disabled, not JS-driven visibility | Content that only becomes visible once a JS animation completes can render blank; unacceptable for a projected live demo |
| D15 | Hash routing (`#/patient/<id>`) | Linkable patients + survives refresh mid-demo. No storage APIs, no router dependency. |

## Open questions for the user
- Do you want to invest in fixing the TCGA expression matrix (see root cause above) so the
  twin cohort gets genuine per-patient Q1 probabilities and a true Q1×Q3 agreement badge?
- Feature the Q1 accuracy panel prominently? It is currently on the cohort landing view.
  Note the honest reading: AUC 0.593 overall, 0.766 in Riaz, 0.451 in the small Hugo set,
  and the models are poorly calibrated (most patients pushed above 0.5 against a 42%
  true response rate). The panel states this plainly rather than showing only the best number.
