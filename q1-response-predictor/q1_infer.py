#!/usr/bin/env python3
"""
q1_infer.py — Generate genuine per-patient immunotherapy-response predictions
from trained Q1 models for the OncoTwin dashboard (Q5 integration).

METHODOLOGY:
- Active ICI cohorts are loaded dynamically from config/datasets.yaml (all cohorts
  with processing_strategy='iatlas' or cohort_immunotherapy=True). Currently:
  Liu 2019, Hugo 2016, Riaz 2017, Gide 2019, TCGA GDC 2025, Van Allen 2015.
- For each labelled cohort, predictions are generated using strict Leave-One-Cohort-Out
  (LOCO) Cross-Validation folds. Models are fit exclusively on the remaining cohorts,
  with the StandardScaler fit on training cohorts only. This guarantees out-of-fold,
  zero-leakage evaluation metrics for the dashboard validation panel. Patients with
  missing RESPONSE_BINARY labels are silently dropped via valid_mask before LOCO.
- The default reference cohort for pooled-model scoring is resolved dynamically
  (prefers skcm_tcga_gdc; falls back to skcm_tcga_pan_can_atlas_2018 if absent).
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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.signatures import extract_all_signatures, zscore_df
from src.models import get_model
from src.config.datasets import load_dataset_config_or_empty

CONFIG_PATH = HERE / "config" / "datasets.yaml"

def _project_root(start: Path) -> Path:
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

def get_active_ici_cohorts() -> dict[str, str]:
    """Loads active immunotherapy trial cohort folder -> display name mapping from datasets.yaml.

    Includes all cohorts that carry per-patient immunotherapy response labels, i.e.:
    - iAtlas-sourced cohorts (processing_strategy == 'iatlas'): Liu 2019, Hugo 2016,
      Riaz 2017, Gide 2019 — always have RESPONSE_BINARY from the iAtlas curation.
    - Non-iAtlas cohorts flagged cohort_immunotherapy=True: TCGA GDC 2025 IT-subcohort,
      Van Allen 2015 — have partial RESPONSE_BINARY coverage.

    Patients with NaN RESPONSE_BINARY are already dropped by the valid_mask filter in
    main(), so partial coverage cohorts contribute only their labelled rows to LOCO CV.
    """
    configs = load_dataset_config_or_empty(CONFIG_PATH)
    if configs:
        return {
            cfg.processed_directory: cfg.cohort_name
            for cfg in configs
            if cfg.processing_strategy == "iatlas" or cfg.cohort_immunotherapy
        }
    return {
        "liu_2019": "Liu 2019",
        "hugo_2016": "Hugo 2016",
        "riaz_2017": "Riaz 2017",
        "gide_2019": "Gide 2019",
        "skcm_tcga_gdc": "TCGA GDC 2025",
        "van_allen_2015": "Van Allen 2015",
    }

def resolve_reference_tcga_path(custom_path: str | None) -> tuple[Path, str]:
    """Resolves reference cohort expression path and display name dynamically."""
    if custom_path:
        p = Path(custom_path)
        return p, p.parent.name

    configs = load_dataset_config_or_empty(CONFIG_PATH)
    for cfg in configs:
        if cfg.study_id == "skcm_tcga_gdc" or cfg.processed_directory == "skcm_tcga_gdc":
            p = DATA_DIR / "processed" / cfg.processed_directory / "expr_cleaned.csv"
            if p.exists():
                return p, cfg.cohort_name

    fallback = DATA_DIR / "processed" / "skcm_tcga_pan_can_atlas_2018" / "expr_cleaned.csv"
    if fallback.exists():
        return fallback, "TCGA-SKCM"

    return DATA_DIR / "processed" / "skcm_tcga_gdc" / "expr_cleaned.csv", "TCGA GDC 2025"

def compat_fix(model):
    try:
        from sklearn.linear_model import LogisticRegression
        targets = [model]
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
    ap = argparse.ArgumentParser(description="Q1 per-patient inference for the dashboard (LOCO CV out-of-fold for trial cohorts)")
    ap.add_argument(
        "--tcga-expr",
        type=str,
        default=None,
        help="Path to a cleaned TCGA expression matrix (samples x genes, index=SAMPLE_ID).",
    )
    ap.add_argument("--out", type=str, default=str(HERE / "q1_predictions.csv"))
    args = ap.parse_args()

    print("==================================================")
    print("Q1 Per-Patient Inference (LOCO CV Out-of-Fold)")
    print("==================================================")

    # 1. Load feature matrices and labels for active trial cohorts (loaded dynamically from datasets.yaml)
    ici_cohorts = get_active_ici_cohorts()
    trial_data = {}
    for folder, label in ici_cohorts.items():
        expr = read_expr(DATA_DIR / "processed" / folder / "expr_cleaned.csv")
        clin = read_clin(DATA_DIR / "processed" / folder / "clin_cleaned.csv")
        if expr is None or clin is None:
            print(f"  [{label}] missing expr/clin cleaned files in processed/{folder}")
            continue

        # Extract features
        sig = extract_all_signatures(expr)
        sig = sig.reindex(columns=FEATURES)

        # Attach response label
        resp_col = "RESPONSE_BINARY" if "RESPONSE_BINARY" in clin.columns else "response"
        if resp_col in clin.columns:
            resp = clin[resp_col].dropna()
            common = sig.index.intersection(resp.index)
            sig = sig.loc[common]
            y = resp.loc[common].astype(int)
        else:
            y = pd.Series(np.nan, index=sig.index)

        # Drop rows with missing features
        valid_mask = sig.notna().all(axis=1) & y.notna()
        sig = sig.loc[valid_mask]
        y = y.loc[valid_mask]

        if not sig.empty:
            trial_data[label] = {"X": sig, "y": y, "clin": clin}
            print(f"  [{label}] Loaded {len(sig)} scorable patients (Responders: {sum(y==1)}, Non-Responders: {sum(y==0)})")

    blocks: list[pd.DataFrame] = []

    # 2. Perform LOCO CV inference for each trial cohort
    trial_cohort_names = list(trial_data.keys())
    for test_cohort in trial_cohort_names:
        train_cohorts = [c for c in trial_cohort_names if c != test_cohort]
        print(f"\n[LOCO Split] Testing on '{test_cohort}' (Trained on {', '.join(train_cohorts)})...")

        # Training set: combine other cohorts
        X_train = pd.concat([trial_data[c]["X"] for c in train_cohorts], axis=0)
        y_train = pd.concat([trial_data[c]["y"] for c in train_cohorts], axis=0)

        # Test set
        X_test = trial_data[test_cohort]["X"].copy()
        y_test = trial_data[test_cohort]["y"].copy()

        # Scaler fit strictly on training set
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURES, index=X_test.index)

        out = X_test.copy()
        probs = []
        for mkey in ["lr", "rf", "xgb", "svm", "elasticnet"]:
            try:
                mdl = get_model(mkey, X_train_scaled, y_train)
                p = compat_fix(mdl).predict_proba(X_test_scaled)[:, 1]
                out[f"prob_{mkey}"] = p
                if mkey == "elasticnet":
                    out["prob_enet"] = p
                probs.append(p)
            except Exception as e:
                print(f"  [{test_cohort}] model '{mkey}' failed ({e})")

        if probs:
            out["prob_ensemble"] = np.mean(np.vstack(probs), axis=0)
            out["pred_label"] = (out["prob_ensemble"] >= 0.5).astype(int)
            out["cohort"] = test_cohort
            out.index.name = "SAMPLE_ID"
            block = out.reset_index()
            block = attach_clinical(block, trial_data[test_cohort]["clin"])
            blocks.append(block)
            print(f"  [{test_cohort}] Generated out-of-fold predictions for {len(block)} patients.")

    # 3. Reference cohort inference (TCGA) using pooled trial model
    tcga_path, tcga_label = resolve_reference_tcga_path(args.tcga_expr)
    expr_tcga = read_expr(tcga_path)
    if expr_tcga is not None and trial_data:
        print(f"\n[Pooled Model Inference] Scoring reference cohort {tcga_label} ({tcga_path.parent.name})...")
        clin_tcga = read_clin(tcga_path.parent / "clin_cleaned.csv")
        sig_tcga = extract_all_signatures(expr_tcga).reindex(columns=FEATURES)
        sig_tcga = sig_tcga.dropna(how="any")

        # Pooled training data across all trial cohorts
        X_train_full = pd.concat([trial_data[c]["X"] for c in trial_data], axis=0)
        y_train_full = pd.concat([trial_data[c]["y"] for c in trial_data], axis=0)

        scaler_full = StandardScaler()
        X_train_full_scaled = pd.DataFrame(scaler_full.fit_transform(X_train_full), columns=FEATURES, index=X_train_full.index)
        X_tcga_scaled = pd.DataFrame(scaler_full.transform(sig_tcga), columns=FEATURES, index=sig_tcga.index)

        out_tcga = sig_tcga.copy()
        probs_tcga = []
        for mkey in ["lr", "rf", "xgb", "svm", "elasticnet"]:
            try:
                mdl = get_model(mkey, X_train_full_scaled, y_train_full)
                p = compat_fix(mdl).predict_proba(X_tcga_scaled)[:, 1]
                out_tcga[f"prob_{mkey}"] = p
                if mkey == "elasticnet":
                    out_tcga["prob_enet"] = p
                probs_tcga.append(p)
            except Exception as e:
                print(f"  [{tcga_label}] model '{mkey}' failed ({e})")

        if probs_tcga:
            out_tcga["prob_ensemble"] = np.mean(np.vstack(probs_tcga), axis=0)
            out_tcga["pred_label"] = (out_tcga["prob_ensemble"] >= 0.5).astype(int)
            out_tcga["cohort"] = tcga_label
            out_tcga.index.name = "SAMPLE_ID"
            block_tcga = out_tcga.reset_index()
            block_tcga = attach_clinical(block_tcga, clin_tcga)
            blocks.append(block_tcga)
            print(f"  [{tcga_label}] Scored {len(block_tcga)} reference patients.")

    if not blocks:
        print("\nERROR: No cohorts scored.")
        return 1

    result = pd.concat(blocks, ignore_index=True)
    cols = (
        ["SAMPLE_ID", "PATIENT_ID", "cohort"]
        + FEATURES
        + [
            "prob_lr",
            "prob_rf",
            "prob_xgb",
            "prob_svm",
            "prob_elasticnet",
            "prob_enet",
            "prob_ensemble",
            "pred_label",
            "actual_response",
            "os_months",
        ]
    )
    result = result[[c for c in cols if c in result.columns]]
    out_path = Path(args.out)
    result.to_csv(out_path, index=False)
    print(f"\nWrote {len(result)} total predictions -> {out_path}")

    # Copy to dashboard/public/q1_predictions.csv
    dash_public = HERE.parent / "dashboard" / "public"
    if dash_public.exists():
        shutil.copy(out_path, dash_public / "q1_predictions.csv")
        print(f"Copied predictions to -> {dash_public / 'q1_predictions.csv'}")

    # Print out-of-fold validation AUC summary by cohort
    print("\n==================================================")
    print("Out-of-Fold LOCO CV Validation Summary (Dashboard Input):")
    print("==================================================")
    for c, g in result.groupby("cohort"):
        line = f"  {c:12s} n={len(g):4d}  mean P(response)={g['prob_ensemble'].mean():.3f}"
        if g["actual_response"].notna().any():
            try:
                from sklearn.metrics import roc_auc_score
                sub = g.dropna(subset=["actual_response"])
                if len(np.unique(sub["actual_response"])) > 1:
                    auc_val = roc_auc_score(sub["actual_response"], sub["prob_ensemble"])
                    line += f"  LOCO Out-of-Fold AUC={auc_val:.3f}"
            except Exception as e:
                line += f"  (AUC error: {e})"
        print(line)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
