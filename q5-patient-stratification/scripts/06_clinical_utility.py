#!/usr/bin/env python3
"""Script 06: Clinical utility and Decision Curve Analysis (DCA).

Evaluates net clinical benefit via Decision Curve Analysis, calculates NNT and PPV at optimal decision thresholds,
and compares Q5 phenotype-stratified predictive model utility against global Q1 and standard clinical benchmarks (Treat All, High TMB, CD274/PD-L1+).
"""

import contextlib
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class _LRPredictor:
    """Thin picklable wrapper around LogisticRegression for Platt scaling."""

    def __init__(self, lr: LogisticRegression) -> None:
        self.lr = lr

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(p.reshape(-1, 1))[:, 1]


class _CalibratedModel:
    """Lightweight wrapper combining a fitted base classifier with a probability calibrator."""

    base_clf: RandomForestClassifier
    calibrator: object

    def __init__(self, base_clf: RandomForestClassifier, calibrator: object) -> None:
        self.base_clf = base_clf
        self.calibrator = calibrator

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.base_clf.feature_importances_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_prob = self.base_clf.predict_proba(X)[:, 1]
        cal_prob = np.clip(self.calibrator.predict(raw_prob), 0.0, 1.0)
        return np.column_stack([1.0 - cal_prob, cal_prob])

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SUBPROJECT_ROOT = SCRIPT_DIR.parent
for parent in [SCRIPT_DIR] + list(SCRIPT_DIR.parents):
    if (parent / "src").is_dir() and (parent / "data").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

if str(SUBPROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(SUBPROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from reporting import generate_obsidian_frontmatter
from phenotyping import get_cluster_name_map
from src.styles import (
    DARK_SLATE_CHARCOAL,
    PHENOTYPE_PALETTE,
    RESPONSE_PALETTE,
    STRATEGY_PALETTE,
    set_presentation_style,
)
from q5_constants import CLUSTERING_FEATURES, PHENOTYPE_PROB_COL
from src.utils.io import safe_save_csv
from src.utils.logging import TeeStream
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, rel_path
from src.utils.plotting import save_fig

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "06_clinical_utility.log"

INPUT_FILE = PROCESSED_DIR / "q5" / "patient_clusters.csv"
OUTPUT_DIR = PROCESSED_DIR / "q5"
MODELS_DIR = SUBPROJECT_ROOT / "models"
PLOTS_DIR = SUBPROJECT_ROOT / "plots" / "clinical_utility"
REPORTS_DIR = SUBPROJECT_ROOT / "reports" / "q5_phases"

set_presentation_style()


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Identify prediction features for subgroup modelling, excluding clustering features."""
    candidate_features = [
        "IFN_gamma",
        "TIS",
        "CYT",
        "CD8_Tcell",
        "IMPRES",
        "PD_L1",
        "M1_M2_Ratio",
        "Macrophage_STV_Score",
        "CD8_T_cells",
        "CD4_T_cells",
        "NK_cells",
        "B_cells",
        "M1_Macrophages",
        "M2_Macrophages",
        "CAFs",
        "TMB_NONSYNONYMOUS",
        "mut_BRAF",
        "mut_NRAS",
        "mut_NF1",
    ]
    clustering_feature_set = set(CLUSTERING_FEATURES)
    return [
        col for col in candidate_features
        if col not in clustering_feature_set
        and col in df.columns
        and df[col].notna().any()
    ]


def compute_net_benefit(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Tuple[float, float, float, float, float, float]:
    """Compute Net Benefit, TP, FP, TN, FN, and Unnecessary Treatment Reduction at threshold pt."""
    N = len(y_true)
    if N == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # Net Benefit formula: TP/N - (FP/N) * (pt / (1 - pt))
    weight = threshold / (1.0 - threshold)
    net_benefit = (tp / N) - (fp / N) * weight

    return net_benefit, float(tp), float(fp), float(tn), float(fn)


def calculate_dca_curves(
    df: pd.DataFrame, probability_cols: Dict[str, np.ndarray], thresholds: np.ndarray
) -> pd.DataFrame:
    """Calculate Net Benefit across decision threshold probabilities for all strategies."""
    y_true = df["RESPONSE_BINARY"].values
    N = len(y_true)
    prevalence = np.mean(y_true)

    rows = []
    for pt in thresholds:
        # Treat None
        rows.append(
            {
                "Threshold": pt,
                "Strategy": "Treat None",
                "Net_Benefit": 0.0,
                "TP": 0,
                "FP": 0,
                "TN": int(np.sum(y_true == 0)),
                "FN": int(np.sum(y_true == 1)),
                "PPV": np.nan,
                "NNT": np.nan,
                "Unnecessary_Treatments_Avoided": int(np.sum(y_true == 0)),
            }
        )

        # Treat All
        tp_all = np.sum(y_true == 1)
        fp_all = np.sum(y_true == 0)
        weight = pt / (1.0 - pt)
        nb_all = (tp_all / N) - (fp_all / N) * weight
        ppv_all = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
        nnt_all = 1.0 / ppv_all if ppv_all > 0 else np.nan

        rows.append(
            {
                "Threshold": pt,
                "Strategy": "Treat All",
                "Net_Benefit": nb_all,
                "TP": int(tp_all),
                "FP": int(fp_all),
                "TN": 0,
                "FN": 0,
                "PPV": ppv_all,
                "NNT": nnt_all,
                "Unnecessary_Treatments_Avoided": 0,
            }
        )

        # Model and biomarker strategies
        for strat_name, y_prob in probability_cols.items():
            nb, tp, fp, tn, fn = compute_net_benefit(y_true, y_prob, pt)
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            nnt = 1.0 / ppv if ppv > 0 else np.nan

            rows.append(
                {
                    "Threshold": pt,
                    "Strategy": strat_name,
                    "Net_Benefit": nb,
                    "TP": int(tp),
                    "FP": int(fp),
                    "TN": int(tn),
                    "FN": int(fn),
                    "PPV": ppv,
                    "NNT": nnt,
                    "Unnecessary_Treatments_Avoided": int(tn),
                }
            )

    return pd.DataFrame(rows)


def _predict_single_gene_benchmarks(
    df: pd.DataFrame, y_true: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit logistic regression on single-gene benchmarks (PD-L1 and High TMB)."""
    if "PD_L1" in df.columns and df["PD_L1"].notna().any():
        med_val = df["PD_L1"].median()
        x_pdl1 = df[["PD_L1"]].copy().fillna(med_val if pd.notna(med_val) else 0.0)
        lr_pdl1 = LogisticRegression(C=1.0, max_iter=1000).fit(x_pdl1, y_true)
        pred_pdl1 = lr_pdl1.predict_proba(x_pdl1)[:, 1]
    else:
        pred_pdl1 = np.full(len(df), np.mean(y_true))

    if "TMB_NONSYNONYMOUS" in df.columns and df["TMB_NONSYNONYMOUS"].notna().any():
        med_val = df["TMB_NONSYNONYMOUS"].median()
        x_tmb = df[["TMB_NONSYNONYMOUS"]].copy().fillna(med_val if pd.notna(med_val) else 0.0)
        lr_tmb = LogisticRegression(C=1.0, max_iter=1000).fit(x_tmb, y_true)
        pred_tmb = lr_tmb.predict_proba(x_tmb)[:, 1]
    else:
        pred_tmb = np.full(len(df), np.mean(y_true))

    return pred_pdl1, pred_tmb


def generate_predictions(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Generate predicted response probabilities for Global, Stratified, and Benchmark strategies."""
    feature_cols = get_feature_columns(df)
    X_raw = df[feature_cols].copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X_raw)
    X_scaled = scaler.fit_transform(X_imp)

    y_true = df["RESPONSE_BINARY"].values

    global_model_file = MODELS_DIR / "subgroup_model_global.joblib"
    if global_model_file.exists():
        global_clf = joblib.load(global_model_file)
        pred_global = global_clf.predict_proba(X_scaled)[:, 1]
    else:
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(X_scaled, y_true)
        pred_global = clf.predict_proba(X_scaled)[:, 1]

    pred_stratified = np.zeros(len(df))
    prob_sum = np.zeros(len(df))

    for p_name, prob_col in PHENOTYPE_PROB_COL.items():
        clean_name = p_name.lower().replace(" ", "_").replace("-", "_")
        sub_file = MODELS_DIR / f"subgroup_model_{clean_name}.joblib"
        if sub_file.exists() and prob_col in df.columns:
            sub_clf = joblib.load(sub_file)
            p_probs = df[prob_col].values
            pred_stratified += p_probs * sub_clf.predict_proba(X_scaled)[:, 1]
            prob_sum += p_probs
        else:
            p_probs = df[prob_col].values if prob_col in df.columns else np.zeros(len(df))
            pred_stratified += p_probs * pred_global
            prob_sum += p_probs

    nonzero_mask = prob_sum > 0
    pred_stratified[nonzero_mask] = pred_stratified[nonzero_mask] / prob_sum[nonzero_mask]
    pred_stratified[~nonzero_mask] = pred_global[~nonzero_mask]

    pred_pdl1, pred_tmb = _predict_single_gene_benchmarks(df, y_true)

    return {
        "Phenotype-Stratified (Q5)": pred_stratified,
        "Global Predictor (Q1)": pred_global,
        "CD274 (PD-L1+)": pred_pdl1,
        "High TMB": pred_tmb,
    }


def plot_dca_curves(df_dca: pd.DataFrame, out_path: Path) -> None:
    """Plot Decision Curve Analysis (DCA) Net Benefit across threshold probabilities."""
    fig, ax = plt.subplots(figsize=(11, 6))

    strategies = [
        "Phenotype-Stratified (Q5)",
        "Global Predictor (Q1)",
        "CD274 (PD-L1+)",
        "High TMB",
        "Treat All",
        "Treat None",
    ]

    styles = {
        "Phenotype-Stratified (Q5)": {"linestyle": "-", "linewidth": 2.8},
        "Global Predictor (Q1)": {"linestyle": "--", "linewidth": 2.2},
        "CD274 (PD-L1+)": {"linestyle": "-.", "linewidth": 1.8},
        "High TMB": {"linestyle": ":", "linewidth": 1.8},
        "Treat All": {"linestyle": "-", "linewidth": 1.5},
        "Treat None": {"linestyle": "--", "linewidth": 1.2},
    }

    for strat in strategies:
        sub = df_dca[df_dca["Strategy"] == strat]
        if sub.empty:
            continue
        color = STRATEGY_PALETTE.get(strat, DARK_SLATE_CHARCOAL)
        st = styles.get(strat, {"linestyle": "-", "linewidth": 2.0})

        ax.plot(
            sub["Threshold"],
            sub["Net_Benefit"],
            label=strat,
            color=color,
            linestyle=st["linestyle"],
            linewidth=st["linewidth"],
        )

    # Set clinical y-axis limits (y >= -0.05)
    ax.set_ylim([-0.05, max(0.40, df_dca["Net_Benefit"].max() * 1.1)])
    ax.set_xlim([0.05, 0.80])

    ax.set_title("Decision Curve Analysis (DCA): Clinical Net Benefit of Q5 Patient Stratification", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Threshold Probability (p_t)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Net Clinical Benefit", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=10)
    ax.grid(True, color="#E5E7EB", linewidth=0.5, alpha=0.6)

    # Highlight clinical decision range 0.2 - 0.5
    ax.axvspan(0.20, 0.50, color=RESPONSE_PALETTE["CR/PR"], alpha=0.08, label="Clinical Decision Window (0.20-0.50)")

    plt.tight_layout()
    save_fig(fig, out_path, dpi=300)
    plt.close(fig)


def _plot_ppv_panel(ax1: plt.Axes, df_sub: pd.DataFrame) -> None:
    """Render Positive Predictive Value (PPV) panel."""
    sns.barplot(
        data=df_sub,
        x="Threshold_Label",
        y="PPV",
        hue="Strategy",
        palette=STRATEGY_PALETTE,
        ax=ax1,
        edgecolor="black",
        linewidth=0.8,
    )
    ax1.set_title("Positive Predictive Value (PPV) at Key Thresholds", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Decision Threshold Probability", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Positive Predictive Value (Precision)", fontsize=10, fontweight="bold")
    ax1.set_ylim([0.0, 1.05])
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=9)
    ax1.grid(True, axis="y", color="#E5E7EB", linewidth=0.5, alpha=0.6)

    for p in ax1.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax1.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )


def _plot_nnt_panel(ax2: plt.Axes, df_sub: pd.DataFrame) -> None:
    """Render Number Needed to Treat (NNT) panel."""
    sns.barplot(
        data=df_sub,
        x="Threshold_Label",
        y="NNT",
        hue="Strategy",
        palette=STRATEGY_PALETTE,
        ax=ax2,
        edgecolor="black",
        linewidth=0.8,
    )
    ax2.set_title("Number Needed to Treat (NNT) to Achieve 1 Response", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Decision Threshold Probability", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Number Needed to Treat (NNT = 1 / PPV)", fontsize=10, fontweight="bold")
    ax2.set_ylim([0.0, max(5.0, df_sub["NNT"].max() * 1.15)])
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=9)
    ax2.grid(True, axis="y", color="#E5E7EB", linewidth=0.5, alpha=0.6)

    for p in ax2.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax2.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )


def plot_nnt_ppv_comparison(df_dca: pd.DataFrame, target_thresholds: List[float], out_path: Path) -> None:
    """Generate bar chart comparing NNT and PPV across strategies at target decision thresholds."""
    df_sub = df_dca[
        (df_dca["Threshold"].isin(target_thresholds))
        & (df_dca["Strategy"].isin(["Phenotype-Stratified (Q5)", "Global Predictor (Q1)", "CD274 (PD-L1+)", "High TMB", "Treat All"]))
    ].copy()

    df_sub["Threshold_Label"] = df_sub["Threshold"].apply(lambda p: f"p_t = {p:.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    _plot_ppv_panel(ax1, df_sub)
    _plot_nnt_panel(ax2, df_sub)

    plt.tight_layout()
    save_fig(fig, out_path, dpi=300)
    plt.close(fig)


def plot_unnecessary_treatments_avoided(df_dca: pd.DataFrame, out_path: Path) -> None:
    """Plot net reduction in unnecessary treatments / toxicities spared across decision thresholds."""
    fig, ax = plt.subplots(figsize=(11, 6))

    strategies = ["Phenotype-Stratified (Q5)", "Global Predictor (Q1)", "CD274 (PD-L1+)", "High TMB"]

    for strat in strategies:
        sub = df_dca[df_dca["Strategy"] == strat].copy()
        if sub.empty:
            continue
        color = STRATEGY_PALETTE.get(strat, DARK_SLATE_CHARCOAL)

        ax.plot(
            sub["Threshold"],
            sub["Unnecessary_Treatments_Avoided"],
            label=strat,
            color=color,
            linewidth=2.5,
        )

    ax.set_title("Reduction in Unnecessary Monotherapy Exposures (Non-Responders Spared Toxicities)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Threshold Probability (p_t)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count of Non-Responders Correctly Identified (TN)", fontsize=11, fontweight="bold")
    ax.set_xlim([0.05, 0.80])
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=10)
    ax.grid(True, color="#E5E7EB", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    save_fig(fig, out_path, dpi=300)
    plt.close(fig)


def plot_net_benefit_by_phenotype(
    df: pd.DataFrame, prob_cols: Dict[str, np.ndarray], pt: float, out_path: Path
) -> None:
    """Plot Net Benefit across individual biological phenotypes at fixed decision threshold pt."""
    rows = []
    cluster_id_to_name = get_cluster_name_map(df)
    for p_id, p_name in cluster_id_to_name.items():
        mask = df["Cluster_ID"] == p_id
        if not mask.any():
            continue
        df_sub = df[mask]
        y_true_sub = df_sub["RESPONSE_BINARY"].values

        for strat_name in ["Phenotype-Stratified (Q5)", "Global Predictor (Q1)", "CD274 (PD-L1+)", "High TMB", "Treat All"]:
            if strat_name == "Treat All":
                y_prob_sub = np.ones(len(df_sub))
            else:
                y_prob_sub = prob_cols[strat_name][mask]

            nb, tp, fp, tn, fn = compute_net_benefit(y_true_sub, y_prob_sub, pt)
            rows.append(
                {
                    "Phenotype": p_name,
                    "Strategy": strat_name,
                    "Net_Benefit": nb,
                    "N": len(df_sub),
                }
            )

    df_pheno = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df_pheno,
        x="Phenotype",
        y="Net_Benefit",
        hue="Strategy",
        palette=STRATEGY_PALETTE,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title(f"Clinical Net Benefit Breakdown across Phenotypes (Decision Threshold p_t = {pt:.2f})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Biological Phenotype Subgroup", fontsize=11, fontweight="bold")
    ax.set_ylabel("Net Clinical Benefit", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=10)
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.5, alpha=0.6)

    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and abs(h) > 0.001:
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=8,
                xytext=(0, 2 if h >= 0 else -8),
                textcoords="offset points",
            )

    plt.tight_layout()
    save_fig(fig, out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Orchestrate Phase 6: Decision Curve Analysis and Clinical Utility Assessment.

    Loads the clustered patient dataset produced by Phase 3, generates predicted
    response probabilities for each clinical strategy (phenotype-stratified Q5 model,
    global Q1 predictor, single-gene benchmarks), computes Net Benefit across decision
    thresholds via Decision Curve Analysis, and saves all evaluation plots and metrics.
    """
    print("\n" + "=" * 80)
    print("Starting Phase 6: Clinical Utility Assessment (Project root: .)")
    print("=" * 80)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load patient cluster data
    # ------------------------------------------------------------------
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Patient cluster file not found: {rel_path(INPUT_FILE)}. "
            "Run Phase 3 (03_cluster_patients.py) first."
        )
    df = pd.read_csv(INPUT_FILE)
    df_valid = df.dropna(subset=["RESPONSE_BINARY"]).copy()
    df_valid["RESPONSE_BINARY"] = df_valid["RESPONSE_BINARY"].astype(int)
    print(f"Loaded {len(df_valid)} patients with known response status.")

    # ------------------------------------------------------------------
    # Generate predicted probabilities for all strategies
    # ------------------------------------------------------------------
    prob_cols = generate_predictions(df_valid)

    # ------------------------------------------------------------------
    # Decision Curve Analysis across threshold range
    # ------------------------------------------------------------------
    thresholds = np.linspace(0.01, 0.99, 99)
    df_dca = calculate_dca_curves(df_valid, prob_cols, thresholds)

    dca_csv_path = OUTPUT_DIR / "dca_results.csv"
    df_dca.to_csv(dca_csv_path, index=False)
    print(f"Saved DCA results to {rel_path(dca_csv_path)}")

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    dca_plot_file = PLOTS_DIR / "dca_net_benefit_curves.png"
    nnt_plot_file = PLOTS_DIR / "nnt_ppv_comparison.png"
    tox_plot_file = PLOTS_DIR / "unnecessary_treatments_avoided.png"
    pheno_plot_file = PLOTS_DIR / "net_benefit_by_phenotype.png"

    target_thresholds = [0.20, 0.30, 0.40]
    fixed_pt = 0.30

    plot_dca_curves(df_dca, dca_plot_file)
    plot_nnt_ppv_comparison(df_dca, target_thresholds, nnt_plot_file)
    plot_unnecessary_treatments_avoided(df_dca, tox_plot_file)
    plot_net_benefit_by_phenotype(df_valid, prob_cols, fixed_pt, pheno_plot_file)

    print("\nGenerated Plots:")
    print(f"  1. {rel_path(dca_plot_file)}")
    print(f"  2. {rel_path(nnt_plot_file)}")
    print(f"  3. {rel_path(tox_plot_file)}")
    print(f"  4. {rel_path(pheno_plot_file)}")

    print("\n" + "=" * 80)
    print("PHASE 6 CLINICAL UTILITY ASSESSMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()

