#!/usr/bin/env python3
"""Script 05: Train subgroup-specific predictive models.

Evaluates whether training separate predictive models within discovered patient phenotypes
improves response prediction performance compared to applying the global Q1 predictive
model across all subgroups.
"""

# ---------------------------------------------------------------------------
# Standard Library & Third-Party Imports
# ---------------------------------------------------------------------------

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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

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
    sys.path.insert(0, str(SUBPROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from src.styles import (
    PHENOTYPE_PALETTE,
    RESPONSE_PALETTE,
    STRATEGY_PALETTE,
    set_presentation_style,
)
from src.utils.logging import TeeStream
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, rel_path
from src.utils.plotting import save_fig

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "05_subgroup_models.log"

INPUT_FILE = PROCESSED_DIR / "q5" / "patient_clusters.csv"
OUTPUT_DIR = PROCESSED_DIR / "q5"
MODELS_DIR = SUBPROJECT_ROOT / "models"
PLOTS_DIR = SUBPROJECT_ROOT / "plots" / "subgroup_models"

# Hyperparameter & CV Constants
RANDOM_STATE = 42
RF_N_ESTIMATORS = 100
GLOBAL_MAX_DEPTH = 5
SUBGROUP_MAX_DEPTH = 4
SKF_N_SPLITS = 3

set_presentation_style()

PHENOTYPE_COLORS: Dict[int, str] = {
    0: PHENOTYPE_PALETTE["Immune Hot"],                 # Okabe-Ito Vermillion (Immune Hot)
    1: PHENOTYPE_PALETTE["Immune Cold"],                # Okabe-Ito Blue (Immune Cold)
    2: PHENOTYPE_PALETTE["Immunosuppressive M2-High"],  # Okabe-Ito Reddish Purple (Immunosuppressive M2-High)
    3: PHENOTYPE_PALETTE["Mutant-Driven"],              # Okabe-Ito Orange (Mutant-Driven / NF1 Loss)
}

PHENOTYPE_SHORT_NAMES: Dict[int, str] = {
    0: "Immune Hot",
    1: "Immune Cold",
    2: "Immunosuppressive M2-High",
    3: "Mutant-Driven",
}

# ---------------------------------------------------------------------------
# Feature Selection & Preprocessing Utilities
# ---------------------------------------------------------------------------


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Identify available numerical and binary features for modelling.

    Args:
        df: Input patient DataFrame.

    Returns:
        List of feature column names present in the DataFrame with valid data.
    """
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
    return [col for col in candidate_features if col in df.columns and df[col].notna().any()]


def _preprocess_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Impute missing values using median and scale features via StandardScaler.

    Args:
        X_train: Raw feature matrix for training fold.
        X_test: Raw feature matrix for test fold.

    Returns:
        Tuple of (processed X_train, processed X_test).
    """
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr_proc = scaler.fit_transform(imp.fit_transform(X_train))
    X_te_proc = scaler.transform(imp.transform(X_test))
    return X_tr_proc, X_te_proc


# ---------------------------------------------------------------------------
# Model Evaluation & Cross-Validation Helpers
# ---------------------------------------------------------------------------


def evaluate_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted positive-class probabilities.
        threshold: Decision threshold for binarising predictions.

    Returns:
        Dictionary of computed metric names and values.
    """
    y_pred = (y_prob >= threshold).astype(int)
    has_two_classes = len(np.unique(y_true)) > 1

    auc = roc_auc_score(y_true, y_prob) if has_two_classes else np.nan
    pr_auc = average_precision_score(y_true, y_prob) if has_two_classes else np.nan
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "ROC_AUC": auc,
        "PR_AUC": pr_auc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "Accuracy": acc,
        "Brier_Score": brier,
        "PPV": ppv,
        "NPV": npv,
    }


def _fit_predict_fold(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, max_depth: int
) -> np.ndarray:
    """Impute, scale, train a Random Forest model, and return test predictions.

    Args:
        X_tr: Training features.
        y_tr: Training labels.
        X_te: Test features.
        max_depth: Maximum tree depth for Random Forest.

    Returns:
        Array of predicted probabilities for the test fold.
    """
    X_tr_proc, X_te_proc = _preprocess_features(X_tr, X_te)
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        max_depth=max_depth,
        class_weight="balanced_subsample",
    )
    clf.fit(X_tr_proc, y_tr)
    return clf.predict_proba(X_te_proc)[:, 1]


def _compute_global_oof_predictions(
    X_raw: np.ndarray, y_raw: np.ndarray, cohorts: np.ndarray
) -> np.ndarray:
    """Compute out-of-fold probability predictions using Leave-One-Cohort-Out CV.

    Args:
        X_raw: Full raw feature matrix.
        y_raw: Full target label array.
        cohorts: Cohort identifiers per sample.

    Returns:
        Out-of-fold predicted probabilities for global model.
    """
    global_oof_prob = np.zeros(len(y_raw))
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X_raw, y_raw, groups=cohorts):
        X_tr, y_tr = X_raw[train_idx], y_raw[train_idx]
        X_te = X_raw[test_idx]
        global_oof_prob[test_idx] = _fit_predict_fold(X_tr, y_tr, X_te, max_depth=GLOBAL_MAX_DEPTH)
    return global_oof_prob


def _compute_subgroup_oof_predictions(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    cohorts: np.ndarray,
    clusters: np.ndarray,
    global_oof_prob: np.ndarray,
) -> np.ndarray:
    """Compute out-of-fold predictions per phenotype cluster using LOCO or SKF.

    Args:
        X_raw: Full raw feature matrix.
        y_raw: Full target label array.
        cohorts: Cohort identifiers per sample.
        clusters: Phenotype cluster IDs per sample.
        global_oof_prob: Global model OOF predictions (used as fallback).

    Returns:
        Out-of-fold predicted probabilities for subgroup ensemble.
    """
    subgroup_oof_prob = np.zeros(len(y_raw))
    unique_clusters = sorted(np.unique(clusters))

    for cid in unique_clusters:
        c_mask = clusters == cid
        c_indices = np.where(c_mask)[0]
        X_c, y_c, groups_c = X_raw[c_indices], y_raw[c_indices], cohorts[c_indices]

        if len(np.unique(groups_c)) >= 2:
            logo_sub = LeaveOneGroupOut()
            for tr_sub, te_sub in logo_sub.split(X_c, y_c, groups=groups_c):
                real_te = c_indices[te_sub]
                if len(np.unique(y_c[tr_sub])) < 2:
                    subgroup_oof_prob[real_te] = global_oof_prob[real_te]
                else:
                    subgroup_oof_prob[real_te] = _fit_predict_fold(
                        X_c[tr_sub], y_c[tr_sub], X_c[te_sub], max_depth=SUBGROUP_MAX_DEPTH
                    )
        else:
            skf = StratifiedKFold(n_splits=SKF_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            for tr_sub, te_sub in skf.split(X_c, y_c):
                real_te = c_indices[te_sub]
                subgroup_oof_prob[real_te] = _fit_predict_fold(
                    X_c[tr_sub], y_c[tr_sub], X_c[te_sub], max_depth=SUBGROUP_MAX_DEPTH
                )

    return subgroup_oof_prob


def _fit_production_models(
    df_valid: pd.DataFrame, X_raw: np.ndarray, y_raw: np.ndarray, feature_cols: List[str]
) -> Dict[str, RandomForestClassifier]:
    """Train production models on complete dataset and per-cluster subsets.

    Args:
        df_valid: Filtered dataset containing valid response and cluster IDs.
        X_raw: Complete raw feature matrix.
        y_raw: Complete target label array.
        feature_cols: List of feature names.

    Returns:
        Dictionary mapping model scope/phenotype name to trained RandomForestClassifier.
    """
    imp_final = SimpleImputer(strategy="median")
    scaler_final = StandardScaler()
    X_full_proc = scaler_final.fit_transform(imp_final.fit_transform(X_raw))

    global_final_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        max_depth=GLOBAL_MAX_DEPTH,
        class_weight="balanced_subsample",
    )
    global_final_model.fit(X_full_proc, y_raw)

    final_subgroup_models: Dict[str, RandomForestClassifier] = {"Global": global_final_model}

    for cid in sorted(df_valid["Cluster_ID"].unique()):
        p_name = PHENOTYPE_SHORT_NAMES.get(cid, f"Cluster {cid}")
        c_sub = df_valid[df_valid["Cluster_ID"] == cid]
        X_sub_raw = c_sub[feature_cols].values
        y_sub = c_sub["RESPONSE_BINARY"].values

        imp_sub = SimpleImputer(strategy="median")
        scaler_sub = StandardScaler()
        X_sub_proc = scaler_sub.fit_transform(imp_sub.fit_transform(X_sub_raw))

        clf_final = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            max_depth=SUBGROUP_MAX_DEPTH,
            class_weight="balanced_subsample",
        )
        clf_final.fit(X_sub_proc, y_sub)
        final_subgroup_models[p_name] = clf_final

    return final_subgroup_models


def _compile_evaluation_metrics(
    df_valid: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """Compile per-phenotype performance metrics and ROC curve data.

    Args:
        df_valid: Dataset with true labels, predicted probabilities, and cluster IDs.

    Returns:
        Tuple of (evaluation metrics DataFrame, ROC curve points dictionary).
    """
    results_list = []
    roc_data: Dict[str, Dict[str, np.ndarray]] = {}

    g_all = evaluate_predictions(df_valid["RESPONSE_BINARY"].values, df_valid["Global_OOF_Prob"].values)
    s_all = evaluate_predictions(df_valid["RESPONSE_BINARY"].values, df_valid["Subgroup_OOF_Prob"].values)

    n_total = len(df_valid)
    n_resp_total = int(df_valid["RESPONSE_BINARY"].sum())
    resp_rate_total = df_valid["RESPONSE_BINARY"].mean() * 100.0

    results_list.append(
        {
            "Cluster_ID": -1,
            "Phenotype": "Overall Cohort",
            "Model_Scope": "Global Enriched Baseline",
            "N": n_total,
            "Responders": n_resp_total,
            "Response_Rate": resp_rate_total,
            **g_all,
        }
    )
    results_list.append(
        {
            "Cluster_ID": -1,
            "Phenotype": "Overall Cohort",
            "Model_Scope": "Subgroup Ensemble",
            "N": n_total,
            "Responders": n_resp_total,
            "Response_Rate": resp_rate_total,
            **s_all,
        }
    )

    for cid in sorted(df_valid["Cluster_ID"].unique()):
        p_name = PHENOTYPE_SHORT_NAMES.get(cid, f"Cluster {cid}")
        c_sub = df_valid[df_valid["Cluster_ID"] == cid]
        y_sub = c_sub["RESPONSE_BINARY"].values
        n_sub = len(c_sub)
        n_resp = int(y_sub.sum())

        g_prob = c_sub["Global_OOF_Prob"].values
        s_prob = c_sub["Subgroup_OOF_Prob"].values

        g_met = evaluate_predictions(y_sub, g_prob)
        s_met = evaluate_predictions(y_sub, s_prob)

        results_list.append(
            {
                "Cluster_ID": cid,
                "Phenotype": p_name,
                "Model_Scope": "Global Enriched Baseline",
                "N": n_sub,
                "Responders": n_resp,
                "Response_Rate": y_sub.mean() * 100.0,
                **g_met,
            }
        )
        results_list.append(
            {
                "Cluster_ID": cid,
                "Phenotype": p_name,
                "Model_Scope": "Subgroup Specific",
                "N": n_sub,
                "Responders": n_resp,
                "Response_Rate": y_sub.mean() * 100.0,
                **s_met,
            }
        )

        fpr_g, tpr_g, _ = roc_curve(y_sub, g_prob)
        fpr_s, tpr_s, _ = roc_curve(y_sub, s_prob)
        roc_data[p_name] = {
            "fpr_global": fpr_g,
            "tpr_global": tpr_g,
            "fpr_subgroup": fpr_s,
            "tpr_subgroup": tpr_s,
        }

    return pd.DataFrame(results_list), roc_data


def train_and_eval_loco(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]], Dict[str, RandomForestClassifier]]:
    """Perform Leave-One-Cohort-Out CV for Global vs Subgroup models.

    Args:
        df: Input patient dataset with cluster IDs and response status.
        feature_cols: Selected feature column names.

    Returns:
        Tuple containing evaluation DataFrame, ROC curve data, and trained production models.
    """
    df_valid = df.dropna(subset=["RESPONSE_BINARY"]).copy()
    df_valid["RESPONSE_BINARY"] = df_valid["RESPONSE_BINARY"].astype(int)

    cohorts = df_valid["COHORT"].values
    clusters = df_valid["Cluster_ID"].values

    X_raw = df_valid[feature_cols].values
    y_raw = df_valid["RESPONSE_BINARY"].values

    # 1. Out-of-fold probability predictions
    global_oof_prob = _compute_global_oof_predictions(X_raw, y_raw, cohorts)
    subgroup_oof_prob = _compute_subgroup_oof_predictions(
        X_raw, y_raw, cohorts, clusters, global_oof_prob
    )

    df_valid["Global_OOF_Prob"] = global_oof_prob
    df_valid["Subgroup_OOF_Prob"] = subgroup_oof_prob

    # 2. Fit production models on full dataset & clusters
    final_subgroup_models = _fit_production_models(df_valid, X_raw, y_raw, feature_cols)

    # 3. Compile summary statistics & ROC curve points
    df_eval, roc_data = _compile_evaluation_metrics(df_valid)

    return df_eval, roc_data, final_subgroup_models


# ---------------------------------------------------------------------------
# Plotting Helpers
# ---------------------------------------------------------------------------


def _plot_single_roc_panel(
    ax: plt.Axes,
    p_name: str,
    r_dict: Dict[str, np.ndarray],
    g_row: pd.Series,
    s_row: pd.Series,
) -> None:
    """Helper to render a single ROC curve subplot panel for a phenotype."""
    n_pts = int(g_row["N"])
    g_auc = g_row["ROC_AUC"]
    s_auc = s_row["ROC_AUC"]

    ax.plot(
        r_dict["fpr_global"],
        r_dict["tpr_global"],
        color="#37474F",
        linestyle="--",
        linewidth=2,
        label=f"Global Enriched (AUC = {g_auc:.3f})",
    )

    phenotype_color = PHENOTYPE_PALETTE.get(p_name, RESPONSE_PALETTE["CR/PR"])
    ax.plot(
        r_dict["fpr_subgroup"],
        r_dict["tpr_subgroup"],
        color=phenotype_color,
        linestyle="-",
        linewidth=2.5,
        label=f"Subgroup Model (AUC = {s_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], color="#9E9E9E", linestyle=":", linewidth=1)

    ax.set_title(f"{p_name} (N = {n_pts})", fontsize=13, fontweight="bold")
    ax.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=11)
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=10)
    ax.grid(True, color="#E5E7EB", linewidth=0.5, alpha=0.6)


def plot_subgroup_roc_curves(
    roc_data: Dict[str, Dict[str, np.ndarray]], df_eval: pd.DataFrame, out_path: Path
) -> None:
    """Generate 2x2 multi-panel ROC curves comparing Global Enriched Baseline vs Subgroup Models.

    Args:
        roc_data: Dictionary mapping phenotype to ROC points.
        df_eval: DataFrame containing evaluation metrics.
        out_path: File path to save output figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    for idx, (p_name, r_dict) in enumerate(roc_data.items()):
        ax = axes_flat[idx]
        g_row = df_eval[(df_eval["Phenotype"] == p_name) & (df_eval["Model_Scope"] == "Global Enriched Baseline")].iloc[0]
        s_row = df_eval[(df_eval["Phenotype"] == p_name) & (df_eval["Model_Scope"] == "Subgroup Specific")].iloc[0]
        _plot_single_roc_panel(ax, p_name, r_dict, g_row, s_row)

    plt.tight_layout()
    save_fig(fig, out_path, dpi=300)
    plt.close(fig)


def _annotate_bars(ax: plt.Axes) -> None:
    """Annotate bar values on top of matplotlib patches."""
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )


def plot_performance_comparison(df_eval: pd.DataFrame, out_path: Path) -> None:
    """Generate comparative bar chart of metrics across phenotypes.

    Args:
        df_eval: DataFrame containing evaluation metrics.
        out_path: File path to save output figure.
    """
    df_plot = df_eval[df_eval["Phenotype"] != "Overall Cohort"].copy()
    metrics = ["ROC_AUC", "PR_AUC", "Precision", "Recall"]

    df_melt = df_plot.melt(
        id_vars=["Phenotype", "Model_Scope"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )
    df_melt["Metric"] = df_melt["Metric"].replace({"ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC"})

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=df_melt,
        x="Phenotype",
        y="Score",
        hue="Model_Scope",
        palette=STRATEGY_PALETTE,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title(
        "Subgroup Model vs Global Enriched Baseline Performance across Melanoma Phenotypes",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Biological Phenotype Subgroup", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cross-Validation Score (LOCO CV)", fontsize=11, fontweight="bold")
    ax.set_ylim([0.0, 1.05])
    ax.legend(title="Model Strategy", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=10)
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.5, alpha=0.6)

    _annotate_bars(ax)

    plt.tight_layout()
    save_fig(fig, out_path, dpi=300)
    plt.close(fig)


def plot_feature_importances(
    final_models: Dict[str, RandomForestClassifier], feature_cols: List[str], out_path: Path
) -> None:
    """Plot feature importance heatmaps comparing driver weights across phenotypes.

    Args:
        final_models: Dictionary of trained production models.
        feature_cols: Feature names.
        out_path: Output figure path.
    """
    importance_dict = {}
    for p_name, clf in final_models.items():
        if hasattr(clf, "feature_importances_"):
            importance_dict[p_name] = clf.feature_importances_

    df_imp = pd.DataFrame(importance_dict, index=feature_cols)
    df_imp["Mean_Importance"] = df_imp.mean(axis=1)
    df_imp = df_imp.sort_values(by="Mean_Importance", ascending=False).drop(columns=["Mean_Importance"])
    top_df_imp = df_imp.head(12)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        top_df_imp,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Random Forest Gini Importance"},
        ax=ax,
        linewidths=0.5,
        linecolor="#E5E7EB",
    )

    ax.set_title(
        "Phenotype-Specific Feature Importance Profiles (Top 12 Features)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Melanoma Phenotype Subgroup", fontsize=11, fontweight="bold")
    ax.set_ylabel("Biomarker / Cell Signature Feature", fontsize=11, fontweight="bold")

    plt.tight_layout()
    save_fig(fig, out_path, dpi=300)
    plt.close(fig)


def _print_performance_summary(df_eval: pd.DataFrame) -> None:
    """Print clean summary table of cross-validation performance to stdout.

    Args:
        df_eval: Evaluation metrics DataFrame.
    """
    print("Summary of Cross-Validation Performance (Global Enriched Baseline vs Subgroup Model):")
    for phenotype in df_eval["Phenotype"].unique():
        sub_df = df_eval[df_eval["Phenotype"] == phenotype]
        g_auc = sub_df[sub_df["Model_Scope"] == "Global Enriched Baseline"]["ROC_AUC"].values[0]
        s_auc = sub_df[sub_df["Model_Scope"] != "Global Enriched Baseline"]["ROC_AUC"].values[0]
        g_ppv = sub_df[sub_df["Model_Scope"] == "Global Enriched Baseline"]["PPV"].values[0]
        s_ppv = sub_df[sub_df["Model_Scope"] != "Global Enriched Baseline"]["PPV"].values[0]

        diff_auc = s_auc - g_auc if not np.isnan(s_auc) and not np.isnan(g_auc) else 0.0
        print(
            f"  * {phenotype:<25}: Global AUC = {g_auc:.3f} | Subgroup AUC = {s_auc:.3f} (Delta = {diff_auc:+.3f}) | Subgroup PPV = {s_ppv:.3f}"
        )


# ---------------------------------------------------------------------------
# Main Execution Workflow
# ---------------------------------------------------------------------------


def main() -> None:
    """Main execution entry point for Phase 5 Subgroup Predictive Modelling."""
    print("=" * 80)
    print(f"Starting Phase 5: Subgroup-Specific Predictive Modelling (Project root: {rel_path(PROJECT_ROOT)})")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing patient clusters file at {rel_path(INPUT_FILE)}. Run Phase 3 first.")

    df_clusters = pd.read_csv(INPUT_FILE)
    print(f"Loaded patient dataset: {len(df_clusters)} patients across {df_clusters['COHORT'].nunique()} cohorts")

    feature_cols = get_feature_columns(df_clusters)
    print(f"Selected {len(feature_cols)} biomarker & genomic features for subgroup modelling:")
    print(f"  * {', '.join(feature_cols)}")

    # 1. Evaluate LOCO CV & fit production models
    df_eval, roc_data, final_models = train_and_eval_loco(df_clusters, feature_cols)

    # 2. Save evaluation summary CSV
    out_eval_csv = OUTPUT_DIR / "subgroup_models_evaluation.csv"
    if out_eval_csv.exists():
        try:
            out_eval_csv.unlink()
        except Exception:
            pass
    df_eval.to_csv(out_eval_csv, index=False)
    print(f"\nSaved subgroup models evaluation summary to {rel_path(out_eval_csv)}")

    # 3. Serialise trained models to models/ directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in final_models.items():
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        m_file = MODELS_DIR / f"subgroup_model_{clean_name}.joblib"
        joblib.dump(model, m_file)
        print(f"  * Serialised {name} model -> {rel_path(m_file)}")

    # 4. Generate 300 DPI publication plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    roc_plot_file = PLOTS_DIR / "subgroup_roc_curves.png"
    comp_plot_file = PLOTS_DIR / "subgroup_performance_comparison.png"
    imp_plot_file = PLOTS_DIR / "subgroup_feature_importances.png"

    plot_subgroup_roc_curves(roc_data, df_eval, roc_plot_file)
    plot_performance_comparison(df_eval, comp_plot_file)
    plot_feature_importances(final_models, feature_cols, imp_plot_file)

    print("\n" + "=" * 80)
    print("PHASE 5 SUBGROUP MODELLING COMPLETE")
    print("=" * 80)
    _print_performance_summary(df_eval)

    print(f"\nGenerated Plots:")
    print(f"  1. {rel_path(roc_plot_file)}")
    print(f"  2. {rel_path(comp_plot_file)}")
    print(f"  3. {rel_path(imp_plot_file)}")
    print("=" * 80)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
