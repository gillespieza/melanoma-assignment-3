#!/usr/bin/env python3
"""
q1_infer.py — Generate genuine per-patient immunotherapy-response predictions
from the trained Q1 models, for the OncoTwin dashboard (Q5 integration).

WHY THIS EXISTS
---------------
The dashboard needs a real per-patient probability from the Q1 gene-expression
predictor. The trained models (models/final_*.pkl) expect six signature scores:
    IFN_gamma, TIS, CYT, CD8_Tcell, IMPRES, PD_L1
These are computed from each cohort's expression matrix with the project's own
src/signatures.py, then standardised with models/final_scaler.pkl.

This script reuses that exact code path so the output is faithful to training.
It must be run LOCALLY (where cBioPortal / your processed data is available) —
it cannot run in the dashboard's network-restricted sandbox.

WHAT IT DOES
------------
1. Loads each cohort's cleaned expression matrix (samples x genes).
2. Computes the six signatures via src.signatures.extract_all_signatures().
3. Standardises with final_scaler.pkl and runs all five models.
4. Writes q1_predictions.csv with per-model + ensemble probabilities
   (+ the real response label where the cohort has one).

HOW TO RUN
----------
    cd q1-response-predictor
    # 1) make sure the processed data exists (your normal pipeline):
    #    python scripts/download_data.py         # raw cBioPortal tarballs
    #    python scripts/clean_data.py             # -> data/processed/<cohort>/expr_cleaned.csv
    # 2) then:
    python q1_infer.py

    # optional: point at a specific cleaned TCGA matrix (samples x genes, index=SAMPLE_ID)
    python q1_infer.py --tcga-expr data/processed/tcga_skcm/expr_cleaned.csv

OUTPUT
------
    q1_predictions.csv   (also copied to ../dashboard/public/q1_predictions.csv
                          if that folder exists, so the app picks it up directly)

Columns:
    SAMPLE_ID, PATIENT_ID, cohort,
    IFN_gamma, TIS, CYT, CD8_Tcell, IMPRES, PD_L1,
    prob_lr, prob_rf, prob_xgb, prob_svm, prob_enet, prob_ensemble,
    pred_label, actual_response, os_months
"""

from __future__ import annotations
import argparse
import pickle
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # so `import src.signatures` works

from src.signatures import extract_all_signatures  # noqa: E402


def _project_root(start: Path) -> Path:
    """Locate the folder that actually holds data/ (this project is nested inside
    a larger repo, so data/ lives one level up from q1-response-predictor)."""
    for p in [start, *start.parents]:
        if (p / "data").is_dir() and (p / "src").is_dir():
            return p
    return start


DATA_DIR = _project_root(HERE) / "data"
MODELS_DIR = HERE / "models"
FEATURES = [
    "IFN_gamma",
    "TIS",
    "CYT",
    "CD8_Tcell",
    "IMPRES",
    "PD_L1",
    "Macrophage_STV_Score",
    "M1_M2_Ratio",
    "mut_BRAF",
    "mut_NRAS",
    "mut_NF1",
    "TMB_NONSYNONYMOUS",
]

MODEL_FILES = {
    "lr": "final_lr_model.pkl",
    "rf": "final_rf_model.pkl",
    "xgb": "final_xgb_model.pkl",
    "svm": "final_svm_model.pkl",
    "enet": "final_elasticnet_model.pkl",
}

# Cohorts with cleaned expression matrices produced by the standard pipeline.
# (SAMPLE_ID index, genes as columns.) Response label lives in clin_cleaned.csv.
ICI_COHORTS = {
    "liu_2019": "Liu 2019",
    "hugo_2016": "Hugo 2016",
    "riaz_2017": "Riaz 2017",
    "tcga_immune": "TCGA-Immunotherapy",
}


def load_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def compat_fix(model):
    """Patch estimators pickled with an older scikit-learn so they run on newer
    versions. The main break between sklearn <1.5 and >=1.5 is the removed
    `multi_class` attribute on LogisticRegression."""
    try:
        from sklearn.linear_model import LogisticRegression

        targets = [model]
        # CalibratedClassifierCV / pipelines may wrap an estimator
        for attr in ("estimator", "base_estimator"):
            if hasattr(model, attr):
                targets.append(getattr(model, attr))
        for t in targets:
            if isinstance(t, LogisticRegression) and not hasattr(t, "multi_class"):
                t.multi_class = "auto"
    except Exception:
        pass
    return model


def read_expr(path: Path) -> pd.DataFrame | None:
    """Read an expression matrix (samples x genes) indexed by SAMPLE_ID."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    idcol = "SAMPLE_ID" if "SAMPLE_ID" in df.columns else df.columns[0]
    return df.set_index(idcol)


def read_clin(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    idcol = "SAMPLE_ID" if "SAMPLE_ID" in df.columns else df.columns[0]
    return df.set_index(idcol)


def predict_block(df_expr: pd.DataFrame, cohort_label: str, scaler, models) -> pd.DataFrame:
    """Compute signatures + run all models for one cohort's expression matrix."""
    sig = extract_all_signatures(df_expr)
    sig = sig.reindex(columns=FEATURES)
    # Rows with any missing feature can't be scored faithfully — drop + warn.
    n_before = len(sig)
    sig = sig.dropna(how="any")
    if len(sig) < n_before:
        print(f"  [{cohort_label}] dropped {n_before - len(sig)} samples with missing signature genes")
    if sig.empty:
        print(f"  [{cohort_label}] no scorable samples — skipping")
        return pd.DataFrame()

    X = scaler.transform(sig[FEATURES].values)
    out = sig.copy()
    probs = []
    for key, mdl in models.items():
        try:
            p = compat_fix(mdl).predict_proba(X)[:, 1]
            out[f"prob_{key}"] = p
            probs.append(p)
        except Exception as e:
            print(f"  [{cohort_label}] model '{key}' skipped ({type(e).__name__}: {str(e)[:80]})")
    if not probs:
        print(f"  [{cohort_label}] all models failed — check scikit-learn version")
        return pd.DataFrame()
    out["prob_ensemble"] = np.mean(np.vstack(probs), axis=0)
    out["pred_label"] = (out["prob_ensemble"] >= 0.5).astype(int)
    out["cohort"] = cohort_label
    out.index.name = "SAMPLE_ID"
    return out.reset_index()


def attach_clinical(df_pred: pd.DataFrame, df_clin: pd.DataFrame | None) -> pd.DataFrame:
    df_pred["PATIENT_ID"] = df_pred["SAMPLE_ID"]
    df_pred["actual_response"] = np.nan
    df_pred["os_months"] = np.nan
    if df_clin is None:
        return df_pred
    clin = df_clin.copy()
    clin.index.name = "SAMPLE_ID"
    lookup = clin.reset_index()
    for src_col, dst_col in [
        ("PATIENT_ID", "PATIENT_ID"),
        ("RESPONSE_BINARY", "actual_response"),
        ("response", "actual_response"),
        ("OS_MONTHS", "os_months"),
        ("os_months", "os_months"),
    ]:
        if src_col in lookup.columns:
            m = lookup.set_index("SAMPLE_ID")[src_col]
            df_pred[dst_col] = df_pred["SAMPLE_ID"].map(m).fillna(df_pred[dst_col])
    return df_pred


def main() -> int:
    ap = argparse.ArgumentParser(description="Q1 per-patient inference for the dashboard")
    ap.add_argument(
        "--tcga-expr",
        type=str,
        default=None,
        help="Path to a cleaned TCGA expression matrix (samples x genes, index=SAMPLE_ID). "
        "If omitted, tries data/processed/tcga_skcm/expr_cleaned.csv.",
    )
    ap.add_argument("--out", type=str, default=str(HERE / "q1_predictions.csv"))
    args = ap.parse_args()

    # --- load models -------------------------------------------------------
    if not (MODELS_DIR / "final_scaler.pkl").exists():
        print("ERROR: models/final_scaler.pkl not found. Run the Q1 training pipeline first.")
        return 1
    scaler = load_pickle(MODELS_DIR / "final_scaler.pkl")
    models = {}
    for key, fname in MODEL_FILES.items():
        p = MODELS_DIR / fname
        if p.exists():
            models[key] = load_pickle(p)
        else:
            print(f"  WARNING: {fname} missing — skipping {key}")
    if not models:
        print("ERROR: no models found in models/. Aborting.")
        return 1
    print(f"Loaded {len(models)} models: {', '.join(models)}")

    blocks: list[pd.DataFrame] = []

    # --- ICI trial cohorts (have real response labels) ---------------------
    for folder, label in ICI_COHORTS.items():
        expr = read_expr(DATA_DIR / "processed" / folder / "expr_cleaned.csv")
        if expr is None:
            print(f"  [{label}] processed/{folder}/expr_cleaned.csv not found — skipping")
            continue
        clin = read_clin(DATA_DIR / "processed" / folder / "clin_cleaned.csv")
        block = predict_block(expr, label, scaler, models)
        if not block.empty:
            block = attach_clinical(block, clin)
            blocks.append(block)
            print(f"  [{label}] scored {len(block)} patients")

    # --- TCGA reference cohort (survival only, matches the Q3 twin) ---------
    tcga_path = (
        Path(args.tcga_expr)
        if args.tcga_expr
        else DATA_DIR / "processed" / "skcm_tcga_pan_can_atlas_2018" / "expr_cleaned.csv"
    )
    expr = read_expr(tcga_path)
    if expr is None:
        print(
            f"  [TCGA-SKCM] no cleaned expression at {tcga_path}. "
            "Provide --tcga-expr to include the 421 twin cohort."
        )
    else:
        clin = read_clin(tcga_path.parent / "clin_cleaned.csv")
        block = predict_block(expr, "TCGA-SKCM", scaler, models)
        if not block.empty:
            block = attach_clinical(block, clin)
            blocks.append(block)
            print(f"  [TCGA-SKCM] scored {len(block)} patients")

    if not blocks:
        print("\nNo cohorts scored. Make sure the processed expression matrices exist.")
        return 1

    result = pd.concat(blocks, ignore_index=True)
    cols = (
        ["SAMPLE_ID", "PATIENT_ID", "cohort"]
        + FEATURES
        + [f"prob_{k}" for k in models]
        + ["prob_ensemble", "pred_label", "actual_response", "os_months"]
    )
    result = result[[c for c in cols if c in result.columns]]
    out_path = Path(args.out)
    result.to_csv(out_path, index=False)
    print(f"\nWrote {len(result)} predictions -> {out_path}")

    # Copy into the dashboard so the app can read it directly.
    dash_public = HERE.parent / "dashboard" / "public"
    if dash_public.exists():
        shutil.copy(out_path, dash_public / "q1_predictions.csv")
        print(f"Copied -> {dash_public / 'q1_predictions.csv'}")

    # Quick sanity summary
    print("\nSummary by cohort:")
    for c, g in result.groupby("cohort"):
        line = f"  {c:12s} n={len(g):4d}  mean P(response)={g['prob_ensemble'].mean():.3f}"
        if g["actual_response"].notna().any():
            try:
                from sklearn.metrics import roc_auc_score

                y = g["actual_response"].dropna()
                auc = roc_auc_score(y, g.loc[y.index, "prob_ensemble"])
                line += f"  AUC={auc:.3f}"
            except Exception:
                pass
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
