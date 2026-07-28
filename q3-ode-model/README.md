# Q3 — Melanoma BRAF/MEK/ERK ODE Model + Checkpoint Immunotherapy

Answers the assignment question **"Are there any ODE models that are useful?"** for
Skin Cutaneous Melanoma (TCGA-SKCM, n=421).

**Bottom line:** yes — a mechanistic, literature-parameterised model reproduces
BRAF-inhibitor pharmacology (incl. the RAF paradox), stratifies survival, is validated
by orthogonal RPPA protein data, and matches ML with interpretable features. Full
write-up (with references) in [`reports/Q3_ODE_report.md`](reports/Q3_ODE_report.md).
Refactor rationale and results in [`docs/Q3_Refactor_Proposal.md`](docs/Q3_Refactor_Proposal.md).

**Model (Phase 3) — four published modules; kinetic constants are identical for every
patient, only inputs (expression, mutation, drug dose) vary:**
- **A. RAF-dimer + drug** — the RAF paradox *emerges* from binding equilibria, no logic gate.
- **B. MAPK cascade** — published constants, universal negative feedback → steady-state pERK.
- **C. Melanoma tumour–immune** — published melanoma constants; drug→tumour coupling driven mechanistically by pERK. Killing scaled by per-patient CYT score (Rooney et al. 2015).
- **D. Checkpoint axis (anti-PD-1)** — PD-1/PD-L1/complex steady-state sub-module (Lai et al. 2017); dialable anti-PD-1 dose depletes PD-1 and unleashes CD8 killing. Initialised per-patient from IMPRES/PD-L1 expression.
- Fast (signalling) and slow (tumour) timescales solved separately (quasi-steady-state).

## Pipeline

| Phase | Script | Output |
|-------|--------|--------|
| 1 | `phase1_check_environment.py` | dependency + data-file check |
| 2 | `phase2_preprocess_data.py` | `data/melanoma_params_full.csv` (genes, BRAF/NRAS status, survival, IMPRES/CYT) |
| 3 | `phase3_ode_simulation.py` | `outputs/results/pERK_simulations.csv`, `outputs/results/tumour_burden_simulations.csv`, `outputs/results/checkpoint_tumour_simulations.csv` |
| 4 | `phase4_survival_analysis.py` | `outputs/plots/km_ode_stratified.pdf`, `outputs/results/survival_summary.txt` |
| 5 | `phase5_rppa_validation.py` | `outputs/plots/ode_vs_rppa_*.pdf`, `outputs/results/rppa_validation_summary.txt` |
| 6 | `phase6_ml_vs_ode.py` | `outputs/plots/ml_vs_ode_comparison.pdf`, `outputs/results/ml_vs_ode_comparison.csv` |

## Directory Structure

```
q3-ode-model/
├── scripts/          ← phase1–6 pipeline scripts
├── reports/          ← written reports and final answers
├── docs/             ← refactor proposal and run logs
├── outputs/
│   ├── plots/        ← current plots (BRAFi + immunotherapy arms)
│   ├── results/      ← current CSVs and summaries
│   └── archive/      ← original BRAFi-only outputs (pre-immunotherapy)
└── README.md
```

## Run

```bash
python3 -m venv venv
./venv/bin/pip install numpy pandas scipy lifelines matplotlib seaborn scikit-learn
for p in 1 2 3 4 5 6; do ./venv/bin/python scripts/phase${p}_*.py; done
```

## Key Results (n=421, TCGA-SKCM)

- **Pharmacology:** BRAF-V600E pERK suppressed ~49% by vemurafenib; NRAS/WT show the RAF paradox (not suppressed). Baseline hierarchy NRAS > BRAF > WT emerges mechanistically.
- **Survival (BRAFi arm):** pERK log-rank p = 2.5e-2; tumour burden p = 2.7e-2 (both high → worse OS).
- **Survival (immunotherapy arm):** checkpoint tumour burden log-rank p = **2.4e-3**, 66 vs 148 mo median OS — strongest prognostic signal Q3 produces.
- **Mechanism check:** anti-PD-1 shrinks modelled tumour burden 33% in PD-L1-high patients vs only 2% in PD-L1-low.
- **Validation:** ODE pERK vs measured RPPA pERK r = 0.175, p = 2.0e-3 (emergent, not fit); NRAS-highest hierarchy matched (p = 3e-9).
- **ODE vs ML:** 3-feature digital-twin AUC **0.666** beats NN (0.583) and LogReg (0.646), below RF (0.686/12 features).

## Data

Reads TCGA-SKCM from
`../q1-response-predictor/data/raw/skcm_tcga_pan_can_atlas_2018/` (read-only; not modified).
