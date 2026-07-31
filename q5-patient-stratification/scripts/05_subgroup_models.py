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
from typing import Dict, List, Optional, Tuple

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

if str(SUBPROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_ROOT / "src"))

from q5_constants import CLUSTERING_FEATURES, PHENOTYPE_PROB_COL

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

# Minimum GMM posterior probability for a sample to be included in a
# cluster-specific production model fit.  Samples below this threshold
# contribute negligibly to the loss but cause numerical instability when
# class_weight="balanced_subsample" draws bootstrap samples that are
# effectively all from the same response class in small clusters.
MIN_PROB_WEIGHT: float = 1e-6

set_presentation_style()

# ---------------------------------------------------------------------------
# Feature Selection & Preprocessing Utilities
# ---------------------------------------------------------------------------


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Identify prediction features for subgroup modelling, excluding clustering features.

    Features that were used to define the Phase 3 GMM clusters are explicitly
    excluded to avoid circular evaluation: using the same features both to
    assign cluster membership and to predict response within those clusters
    inflates apparent performance by reducing within-cluster feature variance
    by design.  Only the 10 features orthogonal to the clustering step are
    retained, covering the primary immunotherapy biomarkers (IFN-gamma, PD-L1),
    genomic load (TMB), composite immune scores (IMPRES), and additional
    cell-type deconvolution signatures not used in clustering.

    Args:
        df: Input patient DataFrame.

    Returns:
        List of feature column names present in the DataFrame with valid data,
        excluding any feature that also appears in CLUSTERING_FEATURES.
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
    # Exclude features that defined the Phase 3 clusters to break circularity.
    # CLUSTERING_FEATURES is imported from q5_constants — the single source of
    # truth — so this filter updates automatically if the clustering step changes.
    clustering_feature_set = set(CLUSTERING_FEATURES)
    non_circular = [
        col for col in candidate_features
        if col not in clustering_feature_set
        and col in df.columns
        and df[col].notna().any()
    ]
    return non_circular


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
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    max_depth: int,
    sample_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Impute, scale, train a Random Forest model, and return test predictions.

    Args:
        X_tr: Training features.
        y_tr: Training labels.
        X_te: Test features.
        max_depth: Maximum tree depth for Random Forest.
        sample_weight: Optional per-sample weights (e.g. GMM posterior probabilities).
            When provided, each sample's contribution to the model is proportional
            to its cluster membership probability rather than equal-weighted.

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
    clf.fit(X_tr_proc, y_tr, sample_weight=sample_weight)
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
    cluster_probs: np.ndarray,
    global_oof_prob: np.ndarray,
) -> np.ndarray:
    """Compute soft-weighted subgroup OOF predictions using GMM posterior probabilities.

    For each LOCO fold, K cluster-specific models are trained on the training split with
    each sample weighted by its GMM posterior P(Cluster=k).  The per-patient prediction
    for the held-out cohort is then the posterior-weighted blend across all K models::

        ŷ_patient = Σ_k  P_Cluster_k  ×  ŷ_k(patient)

    This replaces the legacy hard-assignment approach (argmax / subset filtering) and
    respects the probabilistic soft-clustering structure from Phase 3.

    Args:
        X_raw: Full raw feature matrix.
        y_raw: Full target label array.
        cohorts: Cohort identifiers per sample.
        cluster_probs: GMM posterior probability matrix of shape (N, K), where
            cluster_probs[i, k] = P(Cluster=k | patient i).
        global_oof_prob: Global model OOF predictions used as a fallback for
            degenerate folds where a cluster has negligible weight.

    Returns:
        Out-of-fold predicted probabilities for soft-weighted subgroup ensemble.
    """
    n_clusters = cluster_probs.shape[1]
    subgroup_oof_prob = np.zeros(len(y_raw))
    logo = LeaveOneGroupOut()

    for train_idx, test_idx in logo.split(X_raw, y_raw, groups=cohorts):
        X_tr, y_tr = X_raw[train_idx], y_raw[train_idx]
        X_te = X_raw[test_idx]
        w_tr = cluster_probs[train_idx]   # shape (n_train, K)
        w_te = cluster_probs[test_idx]    # shape (n_test,  K)

        fold_blend = np.zeros(len(test_idx))

        for k in range(n_clusters):
            weights_k = w_tr[:, k]
            meaningful = weights_k > MIN_PROB_WEIGHT
            effective_n_tr = meaningful.sum()
            y_sub = y_tr[meaningful]

            # Fall back to global predictions when the training fold has
            # fewer than 5 effective cluster samples or only a single response class
            if effective_n_tr < 5 or len(np.unique(y_sub)) < 2:
                fold_blend += w_te[:, k] * global_oof_prob[test_idx]
                continue
            preds_k = _fit_predict_fold(
                X_tr, y_tr, X_te,
                max_depth=SUBGROUP_MAX_DEPTH,
                sample_weight=weights_k,
            )
            fold_blend += w_te[:, k] * preds_k

        subgroup_oof_prob[test_idx] = fold_blend

    return subgroup_oof_prob


def _fit_production_models(
    df_valid: pd.DataFrame,
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    feature_cols: List[str],
    cluster_probs: np.ndarray,
    phenotype_names: List[str],
) -> Dict[str, RandomForestClassifier]:
    """Train production models on the full dataset using soft GMM cluster weights.

    Each cluster-specific model is trained on ALL samples, with each sample's
    contribution weighted by its GMM posterior probability for that phenotype.
    Column k of cluster_probs corresponds to phenotype_names[k], derived from
    the named P_* columns (e.g. P_Immune_Hot) rather than the arbitrary integer
    P_Cluster_k indices, which shift between GMM runs.

    Args:
        df_valid: Filtered dataset containing valid response labels.
        X_raw: Complete raw feature matrix.
        y_raw: Complete target label array.
        feature_cols: List of feature names.
        cluster_probs: GMM posterior probability matrix, shape (N, K).
        phenotype_names: Ordered list of phenotype display names, one per column of
            cluster_probs.  Used as dictionary keys in the returned models dict.

    Returns:
        Dictionary mapping model scope/phenotype name to trained RandomForestClassifier.
    """
    imp_final = SimpleImputer(strategy="median")
    scaler_final = StandardScaler()
    # Pre-process the full feature matrix once; all cluster models share the same
    # preprocessed X — only their sample weights differ.
    X_full_proc = scaler_final.fit_transform(imp_final.fit_transform(X_raw))

    # Global model: uniform weights across all samples
    global_final_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        max_depth=GLOBAL_MAX_DEPTH,
        class_weight="balanced_subsample",
    )
    global_final_model.fit(X_full_proc, y_raw)

    final_subgroup_models: Dict[str, RandomForestClassifier] = {"Global": global_final_model}

    for k, p_name in enumerate(phenotype_names):
        weights_k = cluster_probs[:, k]

        # Filter to samples with meaningful membership probability.
        # Near-zero-weight samples (~1e-98) contribute nothing to the loss
        # but destabilise class_weight="balanced_subsample" in small clusters:
        # bootstrap samples drawn from 699 samples can end up with only
        # trivially-weighted patients from the target cluster, producing
        # all-one-class bootstraps -> division-by-zero -> NaN importances.
        meaningful = weights_k > MIN_PROB_WEIGHT
        X_sub = X_full_proc[meaningful]
        y_sub = y_raw[meaningful]
        w_sub = weights_k[meaningful]

        if len(np.unique(y_sub)) < 2:
            # Degenerate: only one response class in the filtered subset.
            # Fall back to the global model and log a warning.
            print(
                f"  Warning: {p_name} has only one response class after filtering "
                f"(N={meaningful.sum()}); substituting global model."
            )
            final_subgroup_models[p_name] = global_final_model
            continue

        clf_final = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            max_depth=SUBGROUP_MAX_DEPTH,
            class_weight="balanced_subsample",
        )
        clf_final.fit(X_sub, y_sub, sample_weight=w_sub)
        final_subgroup_models[p_name] = clf_final

    return final_subgroup_models


def _build_cluster_name_map(df: pd.DataFrame) -> Dict[int, str]:
    """Derive a Cluster_ID -> phenotype display name mapping from the data at runtime.

    For each cluster, identifies which named GMM probability column has the highest
    mean value, then resolves that column back to its canonical phenotype name.  This
    avoids relying on hardcoded cluster-index assumptions that break when the GMM
    assigns different integer IDs across runs.

    Args:
        df: Patient DataFrame containing Cluster_ID and named P_* probability columns.

    Returns:
        Dictionary mapping each integer Cluster_ID to its phenotype display name.
    """
    available_named_cols = [col for col in PHENOTYPE_PROB_COL.values() if col in df.columns]
    col_to_name = {col: name for name, col in PHENOTYPE_PROB_COL.items()}
    mapping: Dict[int, str] = {}
    for cid in df["Cluster_ID"].unique():
        cluster_rows = df[df["Cluster_ID"] == cid]
        best_col = cluster_rows[available_named_cols].mean().idxmax()
        mapping[cid] = col_to_name.get(best_col, f"Cluster {cid}")
    return mapping


def _compile_evaluation_metrics(
    df_valid: pd.DataFrame,
    cluster_id_to_name: Dict[int, str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """Compile per-phenotype performance metrics and ROC curve data.

    Args:
        df_valid: Dataset with true labels, predicted probabilities, and cluster IDs.
        cluster_id_to_name: Mapping from integer Cluster_ID to phenotype display name,
            derived at runtime from the GMM posterior columns rather than hardcoded.

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
        p_name = cluster_id_to_name.get(cid, f"Cluster {cid}")
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
    """Perform Leave-One-Cohort-Out CV for Global Enriched Baseline vs Soft-Weighted Subgroup models.

    Cluster membership is taken from the GMM posterior probability columns
    (``P_Cluster_0`` … ``P_Cluster_K``) produced by Phase 3, not from the hard
    ``Cluster_ID`` argmax column.  This ensures the full probabilistic structure
    of the soft-assignment clustering is propagated into the subgroup models.

    Args:
        df: Input patient dataset including GMM posterior columns and response status.
        feature_cols: Selected feature column names.

    Returns:
        Tuple containing evaluation DataFrame, ROC curve data, and trained production models.
    """
    df_valid = df.dropna(subset=["RESPONSE_BINARY"]).copy()
    df_valid["RESPONSE_BINARY"] = df_valid["RESPONSE_BINARY"].astype(int)

    cohorts = df_valid["COHORT"].values
    X_raw = df_valid[feature_cols].values
    y_raw = df_valid["RESPONSE_BINARY"].values

    # Use named phenotype probability columns (P_Immune_Hot, P_Immune_Cold, etc.)
    # rather than the generic P_Cluster_k columns.  Named columns are label-stable:
    # the GMM cluster integer indices are arbitrary and can shift between runs,
    # whereas the named columns always correspond to the same biological phenotype.
    available_prob_cols = {
        name: col for name, col in PHENOTYPE_PROB_COL.items()
        if col in df_valid.columns
    }
    if not available_prob_cols:
        raise ValueError(
            "No named phenotype probability columns (P_Immune_Hot, P_Immune_Cold, etc.) "
            "found in the input dataset. Ensure Phase 3 GMM clustering has been run with "
            "the current phenotyping code and patient_clusters.csv is up to date."
        )
    phenotype_names = list(available_prob_cols.keys())
    prob_col_names = list(available_prob_cols.values())
    cluster_probs = df_valid[prob_col_names].values  # shape (N, K)
    print(f"  Using named GMM phenotype probabilities: {prob_col_names}")

    # Derive Cluster_ID -> phenotype name map from the data (runtime-safe,
    # avoids hardcoded index assumptions)
    cluster_id_to_name = _build_cluster_name_map(df_valid)
    print(f"  Cluster ID -> Phenotype map: {cluster_id_to_name}")

    # 1. Out-of-fold probability predictions
    global_oof_prob = _compute_global_oof_predictions(X_raw, y_raw, cohorts)
    subgroup_oof_prob = _compute_subgroup_oof_predictions(
        X_raw, y_raw, cohorts, cluster_probs, global_oof_prob
    )

    df_valid["Global_OOF_Prob"] = global_oof_prob
    df_valid["Subgroup_OOF_Prob"] = subgroup_oof_prob

    # 2. Fit production models on full dataset using named soft cluster weights
    final_subgroup_models = _fit_production_models(
        df_valid, X_raw, y_raw, feature_cols, cluster_probs, phenotype_names
    )

    # 3. Compile summary statistics & ROC curve points
    # Per-phenotype reporting groups by hard Cluster_ID; names are derived from
    # cluster_id_to_name rather than the stale PHENOTYPE_SHORT_NAMES mapping.
    df_eval, roc_data = _compile_evaluation_metrics(df_valid, cluster_id_to_name)

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
        if m_file.exists():
            try:
                m_file.unlink()
            except Exception:
                pass
        try:
            joblib.dump(model, m_file)
        except OSError:
            tmp_m_file = m_file.with_suffix(".tmp.joblib")
            joblib.dump(model, tmp_m_file)
            try:
                tmp_m_file.replace(m_file)
            except Exception:
                pass
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
