# Q3 — Melanoma BRAF/MEK/ERK ODE Model

Answers the assignment question **"Are there any ODE models that are useful?"** for
Skin Cutaneous Melanoma (TCGA-SKCM).

**Bottom line:** yes — a mechanistic, literature-parameterised model reproduces
BRAF-inhibitor pharmacology (incl. the RAF paradox), stratifies survival, is validated
by orthogonal RPPA protein data, and matches ML with two interpretable features. Full
write-up (with references) in [`reports/Q3_ODE_report.md`](reports/Q3_ODE_report.md).

**Model (Phase 3) — three published modules; kinetic constants are identical for every
patient, only inputs (expression, mutation, drug dose) vary:**
- **A. RAF-dimer + drug** — the RAF paradox *emerges* from binding equilibria, no logic gate.
- **B. MAPK cascade** — published constants, universal negative feedback → steady-state pERK.
- **C. Melanoma tumour–immune** — published melanoma constants; drug→tumour coupling
  driven mechanistically by pERK.
- Fast (signalling) and slow (tumour) timescales solved separately (quasi-steady-state).

## Pipeline

| Phase | Script | Output |
|-------|--------|--------|
| 1 | `phase1_check_environment.py` | dependency + data-file check |
| 2 | `phase2_preprocess_data.py` | `data/melanoma_params_full.csv` (genes, BRAF/NRAS status, survival) |
| 3 | `phase3_ode_simulation.py` | `results/pERK_simulations.csv`, `results/tumour_burden_simulations.csv` |
| 4 | `phase4_survival_analysis.py` | `plots/km_ode_stratified.pdf`, `results/survival_summary.txt` |
| 5 | `phase5_rppa_validation.py` | `plots/ode_vs_rppa_*.pdf`, `results/rppa_validation_summary.txt` |
| 6 | `phase6_ml_vs_ode.py` | `plots/ml_vs_ode_comparison.pdf`, `results/ml_vs_ode_comparison.csv` |

## Run

```bash
python3 -m venv venv
./venv/bin/pip install numpy pandas scipy lifelines matplotlib seaborn scikit-learn
for p in 1 2 3 4 5 6; do ./venv/bin/python scripts/phase${p}_*.py; done
```

## Key results

- **Pharmacology:** BRAF-V600E pERK suppressed ~49% by vemurafenib; NRAS/WT show the RAF paradox (not suppressed). Baseline hierarchy NRAS > BRAF > WT emerges mechanistically.
- **Survival:** pERK log-rank p = 2.5e-2; tumour burden p = 1.3e-3 (both high → worse OS).
- **Validation:** ODE pERK vs measured RPPA pERK r = 0.175, p = 2.0e-3 (emergent, not fit); NRAS-highest hierarchy matched (p = 3e-9).
- **ODE vs ML:** ODE digital-twin AUC 0.652 (2 features) beats NN (0.583), matches LogReg (0.646), below RF (0.682).

## Data

Reads TCGA-SKCM from
`../q1-response-predictor/data/raw/skcm_tcga_pan_can_atlas_2018/` (read-only; not modified).
