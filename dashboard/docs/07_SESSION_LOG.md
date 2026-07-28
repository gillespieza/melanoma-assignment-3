# 07 · Session Log & Next Steps

Written 28 July 2026. Captures what was built, what was decided, the git state,
and exactly what to do when the Q2 cell-line data arrives.

Presentation date: **Thursday 30 July 2026**.

---

## 1 · What was built

The v1 Q3-only dashboard was rebuilt into the multi-method Q5 version described in
`02_ARCHITECTURE.md`, then reshaped again around how the demo actually needs to run.

### Data layer
- `dashboard/scripts/build_cohort.mjs` generates `public/cohort.json` (421 patients)
  from the real Q3 sweeps, TCGA clinical data, and the Q1 outputs. Idempotent —
  re-run any time with `node scripts/build_cohort.mjs`.

### Engines
- `src/lib/scoring.ts` — the scoring core (arm confidence, OS anchoring, tiering).
- `src/lib/integrationEngine.ts` — Q5 integration + the methods-agreement signal.
- `src/lib/whatIf.ts` — applies hypothetical edits and swaps in cohort-average
  curves when an edit invalidates the patient's real ODE run.
- `src/lib/forecast.ts` — 12-month forecast + survival curves from ranked options.

### Views (two, not three)
- **Cohort** — headline + stat tiles, the five-method strip, three featured real
  patients, filterable 421-row table, and the Q1 accuracy panel.
- **Patient** — passport → what-if bar → run gate → recommendation + agreement
  badge + ranked options → method detail behind Q1/Q2/Q3/Q4/decision-path tabs.

The old **Archetypes** view was removed once the what-if explorer made it
redundant; `PatientIntake.tsx`, `data/patients.ts` and `lib/decisionEngine.ts`
were deleted with it.

### Current numbers (regenerate to confirm)
```
421 patients · 192 BRAF V600 · 109 NRAS mutant
recommendations   344 immuno / 71 targeted / 6 combo
agreement         164 concordant / 119 discordant / 45 partial / 93 single-method
informative twins 332 BRAFi / 328 anti-PD-1  (93 below model resolution)
Q1 per-patient    421/421 scored
Q1 validation     n=195, AUC 0.593 (Riaz 0.766, Liu 0.584, Hugo 0.451)
```

---

## 2 · Decisions taken (see also `06_DECISIONS.md` D10–D15)

| # | Decision | Why |
|---|----------|-----|
| D16 | Q1 per-patient scores come from `patient_predicted_response_scores.csv` (Q1 workstream, on `main`) | 443 TCGA rows, all 421 cohort patients join. Replaced the biomarker-composite fallback. |
| D17 | Q1 lane shows the gauge + cohort percentile only, not per-model bars | That file supplies one ensemble score per patient. The bars appear automatically if `q1_infer.py` ever produces TCGA rows with the full breakdown. |
| D18 | Every patient opens on a **run gate**; the what-if explorer edits real patients live | The demo needed "pick a patient, run it, watch it change", not a static read-out. |
| D19 | Editing a patient swaps the ODE curves for the cohort-average sweep and badges them **Modified** | Their real curves no longer describe the edited profile. Stated on screen, never silently. |
| D20 | Removed the Archetypes view | The what-if explorer supersedes it. LDH was added to the what-if bar first, since that was the one thing archetypes could show that a TCGA patient could not. |
| D21 | Renamed from "OncoTwin™" to **Melanoma Digital Twin** | Invented product branding read wrong on a student project. |
| D22 | Entry animations are CSS, chart series animation disabled | Content whose visibility depends on a JS animation completing can render blank. Unacceptable for a projected demo. |

### The live-vs-precomputed split — say this if asked
- **Live in the browser:** all Q5 logic — scoring, ranking, tiering, methods
  agreement, decision path, forecast/survival curves. Every what-if edit re-runs it.
- **Precomputed:** the Q3 ODE solutions (Python) and the Q1 predictions (sklearn).

That split is normal for deployed clinical software. Do not claim the ODE solves live.

---

## 3 · Git state

| Branch | State |
|--------|-------|
| `main` | Q3 merged and **pushed to `origin`** (gillespieza). Merge commit `e7e1a4c` — 47 additions under `q3-ode-model/` + 2 lines in `.gitignore`. Nothing of anyone else's modified. |
| `the-dashboard` | Merged up to date with `main` (`36d704c`). Pushed to **`personal`** only. |

```
origin    → gillespieza/melanoma-assignment-3        (team repo)
personal  → Dublindeveloper/melanoma-digital-twin    (private; the-dashboard tracks this)
```

**The dashboard must not go to `origin`.** It is not in main's tree, and
`the-dashboard` tracks `personal`. Just never run `git add .` while on `main`.

Watch the `.gitignore` `/lib/` rule — it is anchored deliberately. Unanchored,
`lib/` also matches `dashboard/src/lib/` and silently drops the engine source
from every commit. This already happened once.

### Deployment
Vercel/Cloudflare must set **Root Directory = `dashboard`** (monorepo; the repo
root has no `package.json`). Build `npm run build`, output `dist`. Framework Vite.
`dashboard/vercel.json` already carries the SPA rewrite needed for `#/patient/...`.

---

## 4 · NEXT: wiring in the Q2 cell-line data

Q2 (cell-line drug-response validation of the Q1 signature) is the one method not
yet wired in. It currently renders as an honest "pending" card in the Q2 tab.

### Where it plugs in
`dashboard/src/components/Q2Evidence.tsx` — the `EVIDENCE` array near the top.
The last entry has `status: "pending"`. That is the only thing to change.

### What to ask Gift for
Ideally a small CSV, one row per cell line:

```
cell_line, signature_score, drug, drug_response, response_metric
A375,      2.41,            Vemurafenib, 0.82,   AUC
SK-MEL-28, 1.87,            Vemurafenib, 0.31,   AUC
```

Plus the headline statistic: **the correlation between signature score and drug
sensitivity, with n and a p-value.** That single number is what the panel needs.

### Two ways to wire it, depending on what arrives

**A · Just a headline statistic (fastest, ~5 minutes).**
Flip the pending entry to `status: "established"` and fill in the real numbers:

```ts
{
  status: "established",
  claim: "Q1 signature tracks drug sensitivity in melanoma cell lines",
  detail: "The Q1 expression signature was applied to melanoma cell lines with "
        + "measured drug response — an orthogonal test of the same signature.",
  stat: "n = __ · Spearman r = __ · p = __",
},
```

**B · A per-cell-line CSV (better, ~1 hour).**
1. Drop the file at `dashboard/public/q2_cell_lines.csv`.
2. In `build_cohort.mjs`, read it alongside the other inputs and compute the
   correlation at build time (there is already a rank/AUC helper in that file to
   copy the pattern from). Put the result on `meta.q2Validation`.
3. Add the type to `src/data/cohort.ts` next to `Q1Validation`.
4. In `Q2Evidence.tsx`, take `meta` as a prop and render the real stat; optionally
   add a small scatter of signature score vs drug response.
5. `node scripts/build_cohort.mjs && npm run build`.

Prefer **A** if it lands the night before. Prefer **B** if there is a clear day.

### Do not
- Invent a correlation or a p-value. If the data does not arrive, the pending card
  is the honest outcome and it is fine to present it that way — it shows you knew
  what was missing.
- Change the Q2 tab to claim per-patient cell-line evidence. Q2 is cohort-level.

---

## 5 · Other loose ends

- **`stash@{0}`** holds 90 Q1 plots/models + literature PDFs from before a branch
  switch. Most now come from `main` as tracked files, so it is probably redundant —
  check, then `git stash drop`.
- **Q1 TCGA per-model breakdown.** `q1_infer.py` still cannot score TCGA (missing
  IMPRES co-stimulatory genes + normalisation mismatch — root cause in
  `06_DECISIONS.md`). Not blocking; only affects the per-model bars.
- **Q1 calibration.** The models push most patients above 0.5 against a 42% true
  response rate. The accuracy panel states this plainly. Do not quote the raw
  probability as a clinical probability — the *ranking* carries the signal.

### Verification before any demo
```bash
cd dashboard
node scripts/build_cohort.mjs   # should report 421 patients, 421 with Q1
npm run typecheck               # clean
npm run build                   # succeeds
npm run dev                     # click a patient, run it, edit BRAF, see it change
```
