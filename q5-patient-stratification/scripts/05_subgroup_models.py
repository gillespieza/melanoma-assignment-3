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
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
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
from sklearn.model_selection import LeaveOneGroupOut
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
    DARK_SLATE_CHARCOAL,
    STRATEGY_PALETTE,
    get_phenotype_color,
    set_presentation_style,
)
from src.utils.io import safe_save_csv
from src.utils.logging import TeeStream
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, rel_path
from src.utils.plotting import save_fig
from phenotyping import get_cluster_name_map
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
MIN_EFFECTIVE_SAMPLES = 5

# Candidate prediction features prior to excluding clustering features
CANDIDATE_PREDICTION_FEATURES: List[str] = [
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

# Minimum GMM posterior probability for sample inclusion in weighted fits
MIN_PROB_WEIGHT: float = 1e-6

# Probability Calibration & Platt Scaling Constants
CALIBRATION_MIN_SAMPLES: int = 50
CALIBRATION_HOLDOUT_FRAC: float = 0.25
PLATT_C: float = 1.0
PLATT_SOLVER: str = "lbfgs"
PLATT_MAX_ITER: int = 1000

# Visualisation Constants
TOP_N_IMPORTANCES: int = 12
GRID_LINE_COLOR: str = "#E5E7EB"

set_presentation_style()

# ---------------------------------------------------------------------------
# Feature Selection & Preprocessing Utilities
# ---------------------------------------------------------------------------


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Identify prediction features for subgroup modelling, excluding clustering features.

    Args:
        df: Input patient DataFrame.

    Returns:
        List of feature column names present with valid data, excluding clustering features.
    """
    clustering_set = set(CLUSTERING_FEATURES)
    return [
        col
        for col in CANDIDATE_PREDICTION_FEATURES
        if col not in clustering_set and col in df.columns and df[col].notna().any()
    ]


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
    X_tr_proc = np.nan_to_num(scaler.fit_transform(imp.fit_transform(X_train)), nan=0.0)
    X_te_proc = np.nan_to_num(scaler.transform(imp.transform(X_test)), nan=0.0)
    return X_tr_proc, X_te_proc


# ---------------------------------------------------------------------------
# Model Evaluation & Cross-Validation Helpers
# ---------------------------------------------------------------------------


def _compute_confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate PPV and NPV confusion matrix metrics.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Binarised prediction array.

    Returns:
        Dict containing PPV and NPV.
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {"PPV": ppv, "NPV": npv}


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
    prob_is_finite = not np.isnan(y_prob).any()

    auc = roc_auc_score(y_true, y_prob) if (has_two_classes and prob_is_finite) else np.nan
    pr_auc = average_precision_score(y_true, y_prob) if (has_two_classes and prob_is_finite) else np.nan

    cm_metrics = _compute_confusion_metrics(y_true, y_pred)
    return {
        "ROC_AUC": auc,
        "PR_AUC": pr_auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Brier_Score": brier_score_loss(y_true, y_prob),
        **cm_metrics,
    }


class _IdentityPredictor:
    """Pass-through calibrator fallback when calibration fold lacks multiple classes."""

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Return uncalibrated probabilities unchanged.

        Args:
            p: Input probability array.

        Returns:
            Unmodified input array as float.
        """
        return np.asarray(p, dtype=float)


class _LRPredictor:
    """Thin picklable wrapper around LogisticRegression for Platt scaling."""

    def __init__(self, lr: LogisticRegression) -> None:
        self.lr = lr

    def predict(self, p: np.ndarray) -> np.ndarray:
        """Convert 1D raw probabilities to calibrated probabilities via logistic regression.

        Args:
            p: 1D array of raw RF positive-class probabilities.

        Returns:
            1D array of calibrated positive-class probabilities.
        """
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
        """Proxy to the base RF feature importances."""
        return self.base_clf.feature_importances_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probability predictions for input feature matrix.

        Args:
            X: Preprocessed feature matrix.

        Returns:
            Array of shape (N, 2) with [P(class=0), P(class=1)] per sample.
        """
        raw_prob = self.base_clf.predict_proba(X)[:, 1]
        cal_prob = np.clip(self.calibrator.predict(raw_prob), 0.0, 1.0)
        return np.column_stack([1.0 - cal_prob, cal_prob])


def _select_calibration_split(
    X_eff: np.ndarray, y_eff: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Select stratified calibration holdout subset from effective training samples.

    Args:
        X_eff: Effective training feature matrix.
        y_eff: Effective training label array.

    Returns:
        Tuple of (X_cal, y_cal) for probability calibration.
    """
    n_eff = len(y_eff)
    n_cal = max(2, int(np.ceil(n_eff * CALIBRATION_HOLDOUT_FRAC)))

    rng = np.random.default_rng(RANDOM_STATE)
    idx_eff = np.arange(n_eff)
    pos_idx = idx_eff[y_eff == 1]
    neg_idx = idx_eff[y_eff == 0]

    n_cal_pos = max(1, int(round(n_cal * len(pos_idx) / n_eff))) if len(pos_idx) > 0 else 0
    n_cal_neg = max(1, n_cal - n_cal_pos) if len(neg_idx) > 0 else 0
    n_cal_pos = min(n_cal_pos, len(pos_idx))
    n_cal_neg = min(n_cal_neg, len(neg_idx))

    cal_idx = np.concatenate([
        rng.choice(pos_idx, n_cal_pos, replace=False),
        rng.choice(neg_idx, n_cal_neg, replace=False),
    ])
    return X_eff[cal_idx], y_eff[cal_idx]


def _fit_calibrator(base_clf: RandomForestClassifier, X_eff: np.ndarray, y_eff: np.ndarray) -> object:
    """Fit isotonic or Platt scaling calibrator on stratified holdout subset.

    Args:
        base_clf: Trained base Random Forest classifier.
        X_eff: Effective non-zero feature matrix.
        y_eff: Effective label array.

    Returns:
        Fitted calibrator instance exposing predict(p_1d).
    """
    if len(y_eff) < 2 or len(np.unique(y_eff)) < 2:
        return _IdentityPredictor()

    X_cal, y_cal = _select_calibration_split(X_eff, y_eff)
    if len(y_cal) < 2 or len(np.unique(y_cal)) < 2:
        return _IdentityPredictor()

    raw_cal_prob = base_clf.predict_proba(X_cal)[:, 1]
    if np.isnan(raw_cal_prob).any():
        return _IdentityPredictor()

    if len(y_cal) >= CALIBRATION_MIN_SAMPLES:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_cal_prob, y_cal)
        return calibrator

    lr = LogisticRegression(C=PLATT_C, solver=PLATT_SOLVER, max_iter=PLATT_MAX_ITER)
    lr.fit(raw_cal_prob.reshape(-1, 1), y_cal)
    return _LRPredictor(lr)


def _fit_calibrated_rf(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    max_depth: int,
    sample_weight: Optional[np.ndarray] = None,
) -> _CalibratedModel:
    """Fit a base Random Forest and then calibrate its probabilities.

    Args:
        X_tr: Training features.
        y_tr: Training labels.
        max_depth: Maximum tree depth.
        sample_weight: Optional per-sample weights.

    Returns:
        Fitted _CalibratedModel instance.
    """
    base_clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        max_depth=max_depth,
        class_weight="balanced_subsample",
    )
    base_clf.fit(X_tr, y_tr, sample_weight=sample_weight)

    effective_mask = (sample_weight > MIN_PROB_WEIGHT) if sample_weight is not None else np.ones(len(y_tr), dtype=bool)
    X_eff, y_eff = X_tr[effective_mask], y_tr[effective_mask]

    calibrator = _fit_calibrator(base_clf, X_eff, y_eff)
    return _CalibratedModel(base_clf=base_clf, calibrator=calibrator)


def _fit_predict_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    max_depth: int,
    sample_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Impute, scale, train a calibrated Random Forest model, and return test predictions.

    Args:
        X_tr: Training features.
        y_tr: Training labels.
        X_te: Test features.
        max_depth: Maximum tree depth.
        sample_weight: Optional sample weights.

    Returns:
        Test fold probability array.
    """
    X_tr_proc, X_te_proc = _preprocess_features(X_tr, X_te)
    clf = _fit_calibrated_rf(X_tr_proc, y_tr, max_depth=max_depth, sample_weight=sample_weight)
    return clf.predict_proba(X_te_proc)[:, 1]


def _compute_global_oof_predictions(
    X_raw: np.ndarray, y_raw: np.ndarray, cohorts: np.ndarray
) -> np.ndarray:
    """Compute out-of-fold probability predictions using Leave-One-Cohort-Out CV.

    Args:
        X_raw: Full raw feature matrix.
        y_raw: Target labels.
        cohorts: Cohort identifiers per sample.

    Returns:
        Global model out-of-fold predictions.
    """
    global_oof_prob = np.zeros(len(y_raw))
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X_raw, y_raw, groups=cohorts):
        X_tr, y_tr = X_raw[train_idx], y_raw[train_idx]
        X_te = X_raw[test_idx]
        global_oof_prob[test_idx] = _fit_predict_fold(X_tr, y_tr, X_te, max_depth=GLOBAL_MAX_DEPTH)
    return global_oof_prob


def _predict_subgroup_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    w_tr: np.ndarray,
    w_te: np.ndarray,
    global_oof_test: np.ndarray,
) -> np.ndarray:
    """Predict test sample response for a single LOCO fold blended across K cluster models.

    Args:
        X_tr: Training features.
        y_tr: Training labels.
        X_te: Test features.
        w_tr: Training fold cluster weights matrix (N_tr, K).
        w_te: Test fold cluster weights matrix (N_te, K).
        global_oof_test: Fallback global predictions for test set.

    Returns:
        Blended test prediction array.
    """
    n_clusters = w_tr.shape[1]
    fold_blend = np.zeros(X_te.shape[0])
    for k in range(n_clusters):
        weights_k = w_tr[:, k]
        meaningful = weights_k > MIN_PROB_WEIGHT
        effective_n_tr = meaningful.sum()
        y_sub = y_tr[meaningful]

        if effective_n_tr < MIN_EFFECTIVE_SAMPLES or len(np.unique(y_sub)) < 2:
            fold_blend += w_te[:, k] * global_oof_test
            continue

        preds_k = _fit_predict_fold(
            X_tr, y_tr, X_te, max_depth=SUBGROUP_MAX_DEPTH, sample_weight=weights_k
        )
        fold_blend += w_te[:, k] * preds_k
    return fold_blend


def _compute_subgroup_oof_predictions(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    cohorts: np.ndarray,
    cluster_probs: np.ndarray,
    global_oof_prob: np.ndarray,
) -> np.ndarray:
    """Compute soft-weighted subgroup OOF predictions using GMM posterior probabilities.

    Args:
        X_raw: Full raw feature matrix.
        y_raw: Target labels.
        cohorts: Cohort identifiers.
        cluster_probs: GMM posterior matrix (N, K).
        global_oof_prob: Global fallback OOF array.

    Returns:
        Subgroup ensemble OOF probability array.
    """
    subgroup_oof_prob = np.zeros(len(y_raw))
    logo = LeaveOneGroupOut()

    for train_idx, test_idx in logo.split(X_raw, y_raw, groups=cohorts):
        fold_blend = _predict_subgroup_fold(
            X_raw[train_idx],
            y_raw[train_idx],
            X_raw[test_idx],
            cluster_probs[train_idx],
            cluster_probs[test_idx],
            global_oof_prob[test_idx],
        )
        subgroup_oof_prob[test_idx] = fold_blend

    return subgroup_oof_prob


def _fit_single_phenotype_production_model(
    X_full_proc: np.ndarray,
    y_raw: np.ndarray,
    weights_k: np.ndarray,
    p_name: str,
    global_calibrated: _CalibratedModel,
) -> _CalibratedModel:
    """Fit a calibrated production model for a single phenotype subgroup.

    Args:
        X_full_proc: Processed full feature matrix.
        y_raw: Full target label array.
        weights_k: Sample weights for target phenotype.
        p_name: Phenotype display name.
        global_calibrated: Fallback global calibrated model.

    Returns:
        Fitted _CalibratedModel for phenotype subgroup.
    """
    meaningful = weights_k > MIN_PROB_WEIGHT
    X_sub = X_full_proc[meaningful]
    y_sub = y_raw[meaningful]
    w_sub = weights_k[meaningful]
    n_effective = int(meaningful.sum())

    if len(np.unique(y_sub)) < 2:
        print(
            f"  Warning: {p_name} has only one response class after filtering "
            f"(N={n_effective}); substituting global model."
        )
        return global_calibrated

    return _fit_calibrated_rf(X_sub, y_sub, max_depth=SUBGROUP_MAX_DEPTH, sample_weight=w_sub)


def _fit_production_models(
    df_valid: pd.DataFrame,
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    feature_cols: List[str],
    cluster_probs: np.ndarray,
    phenotype_names: List[str],
) -> Dict[str, _CalibratedModel]:
    """Train calibrated production models on full dataset using soft GMM cluster weights.

    Args:
        df_valid: Input dataset with valid labels.
        X_raw: Raw feature matrix.
        y_raw: Label array.
        feature_cols: Selected feature names.
        cluster_probs: GMM probability matrix.
        phenotype_names: Ordered phenotype names.

    Returns:
        Dict mapping phenotype name to fitted _CalibratedModel.
    """
    imp_final = SimpleImputer(strategy="median")
    scaler_final = StandardScaler()
    X_full_proc = np.nan_to_num(scaler_final.fit_transform(imp_final.fit_transform(X_raw)), nan=0.0)

    global_calibrated = _fit_calibrated_rf(X_full_proc, y_raw, max_depth=GLOBAL_MAX_DEPTH)
    final_models: Dict[str, _CalibratedModel] = {"Global": global_calibrated}

    for k, p_name in enumerate(phenotype_names):
        final_models[p_name] = _fit_single_phenotype_production_model(
            X_full_proc, y_raw, cluster_probs[:, k], p_name, global_calibrated
        )

    return final_models


def _evaluate_single_cluster_subgroup(
    c_sub: pd.DataFrame, cid: int, p_name: str
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluate global vs subgroup model metrics and ROC data for a single cluster.

    Args:
        c_sub: Cluster subset DataFrame.
        cid: Cluster integer ID.
        p_name: Phenotype display name.

    Returns:
        Tuple of (global_result_dict, subgroup_result_dict, roc_points_dict).
    """
    y_sub = c_sub["RESPONSE_BINARY"].values
    n_sub = len(c_sub)
    n_resp = int(y_sub.sum())
    resp_rate = y_sub.mean() * 100.0

    g_prob = c_sub["Global_OOF_Prob"].values
    s_prob = c_sub["Subgroup_OOF_Prob"].values

    g_met = evaluate_predictions(y_sub, g_prob)
    s_met = evaluate_predictions(y_sub, s_prob)

    g_res = {
        "Cluster_ID": cid,
        "Phenotype": p_name,
        "Model_Scope": "Global Enriched Baseline",
        "N": n_sub,
        "Responders": n_resp,
        "Response_Rate": resp_rate,
        **g_met,
    }
    s_res = {
        "Cluster_ID": cid,
        "Phenotype": p_name,
        "Model_Scope": "Subgroup Specific",
        "N": n_sub,
        "Responders": n_resp,
        "Response_Rate": resp_rate,
        **s_met,
    }

    fpr_g, tpr_g, _ = roc_curve(y_sub, g_prob)
    fpr_s, tpr_s, _ = roc_curve(y_sub, s_prob)
    r_data = {"fpr_global": fpr_g, "tpr_global": tpr_g, "fpr_subgroup": fpr_s, "tpr_subgroup": tpr_s}

    return g_res, s_res, r_data


def _evaluate_cluster_subgroups(
    df_valid: pd.DataFrame,
    cluster_id_to_name: Dict[int, str],
) -> Tuple[List[Dict[str, float]], Dict[str, Dict[str, np.ndarray]]]:
    """Evaluate predictions per phenotype cluster subgroup.

    Args:
        df_valid: Input dataset with OOF predictions.
        cluster_id_to_name: Map from Cluster_ID to phenotype display name.

    Returns:
        Tuple of (results_list, roc_data_dict).
    """
    results_list = []
    roc_data: Dict[str, Dict[str, np.ndarray]] = {}

    for cid in sorted(df_valid["Cluster_ID"].unique()):
        p_name = cluster_id_to_name.get(cid, f"Cluster {cid}")
        c_sub = df_valid[df_valid["Cluster_ID"] == cid]
        g_res, s_res, r_data = _evaluate_single_cluster_subgroup(c_sub, cid, p_name)
        results_list.extend([g_res, s_res])
        roc_data[p_name] = r_data

    return results_list, roc_data


def _compile_evaluation_metrics(
    df_valid: pd.DataFrame,
    cluster_id_to_name: Dict[int, str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """Compile per-phenotype performance metrics and ROC curve data.

    Args:
        df_valid: Input dataset with OOF predictions.
        cluster_id_to_name: Cluster ID to phenotype mapping.

    Returns:
        Tuple of (evaluation DataFrame, ROC curve data dict).
    """
    g_all = evaluate_predictions(df_valid["RESPONSE_BINARY"].values, df_valid["Global_OOF_Prob"].values)
    s_all = evaluate_predictions(df_valid["RESPONSE_BINARY"].values, df_valid["Subgroup_OOF_Prob"].values)

    n_total = len(df_valid)
    n_resp_total = int(df_valid["RESPONSE_BINARY"].sum())
    resp_rate_total = df_valid["RESPONSE_BINARY"].mean() * 100.0

    results_list = [
        {
            "Cluster_ID": -1,
            "Phenotype": "Overall Cohort",
            "Model_Scope": "Global Enriched Baseline",
            "N": n_total,
            "Responders": n_resp_total,
            "Response_Rate": resp_rate_total,
            **g_all,
        },
        {
            "Cluster_ID": -1,
            "Phenotype": "Overall Cohort",
            "Model_Scope": "Subgroup Ensemble",
            "N": n_total,
            "Responders": n_resp_total,
            "Response_Rate": resp_rate_total,
            **s_all,
        },
    ]

    sub_results, roc_data = _evaluate_cluster_subgroups(df_valid, cluster_id_to_name)
    results_list.extend(sub_results)

    return pd.DataFrame(results_list), roc_data


def _extract_phenotype_probabilities(df_valid: pd.DataFrame) -> Tuple[List[str], List[str], np.ndarray]:
    """Extract named phenotype probability columns and matrix from dataset.

    Args:
        df_valid: Patient dataset.

    Returns:
        Tuple of (phenotype_names, prob_col_names, cluster_probs_matrix).
    """
    available_prob_cols = {
        name: col for name, col in PHENOTYPE_PROB_COL.items()
        if col in df_valid.columns
    }
    if not available_prob_cols:
        raise ValueError(
            "No named phenotype probability columns (P_Immune_Hot, etc.) found in dataset. "
            "Ensure Phase 3 GMM clustering has been run."
        )
    phenotype_names = list(available_prob_cols.keys())
    prob_col_names = list(available_prob_cols.values())
    cluster_probs = df_valid[prob_col_names].values
    return phenotype_names, prob_col_names, cluster_probs


def train_and_eval_loco(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]], Dict[str, _CalibratedModel]]:
    """Perform Leave-One-Cohort-Out CV for Global Enriched Baseline vs Soft-Weighted Subgroup models.

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

    phenotype_names, prob_col_names, cluster_probs = _extract_phenotype_probabilities(df_valid)
    print(f"  Using named GMM phenotype probabilities: {prob_col_names}")

    cluster_id_to_name = get_cluster_name_map(df_valid)
    print(f"  Cluster ID -> Phenotype map: {cluster_id_to_name}")

    global_oof_prob = _compute_global_oof_predictions(X_raw, y_raw, cohorts)
    subgroup_oof_prob = _compute_subgroup_oof_predictions(
        X_raw, y_raw, cohorts, cluster_probs, global_oof_prob
    )

    nan_mask = np.isnan(subgroup_oof_prob)
    if nan_mask.any():
        print(f"  WARNING: {nan_mask.sum()} NaN OOF predictions replaced with global model fallback.")
        subgroup_oof_prob = np.where(nan_mask, global_oof_prob, subgroup_oof_prob)

    df_valid["Global_OOF_Prob"] = global_oof_prob
    df_valid["Subgroup_OOF_Prob"] = subgroup_oof_prob

    final_subgroup_models = _fit_production_models(
        df_valid, X_raw, y_raw, feature_cols, cluster_probs, phenotype_names
    )
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
    """Helper to render a single ROC curve subplot panel for a phenotype.

    Args:
        ax: Subplot Axes.
        p_name: Phenotype name.
        r_dict: Dict with ROC points.
        g_row: Global model metrics row.
        s_row: Subgroup model metrics row.
    """
    ax.plot(
        r_dict["fpr_global"],
        r_dict["tpr_global"],
        color=DARK_SLATE_CHARCOAL,
        linestyle="--",
        linewidth=2,
        label=f"Global Enriched (AUC = {g_row['ROC_AUC']:.3f})",
    )

    phenotype_color = get_phenotype_color(p_name)
    ax.plot(
        r_dict["fpr_subgroup"],
        r_dict["tpr_subgroup"],
        color=phenotype_color,
        linestyle="-",
        linewidth=2.5,
        label=f"Subgroup Model (AUC = {s_row['ROC_AUC']:.3f})",
    )
    ax.plot([0, 1], [0, 1], color=DARK_SLATE_CHARCOAL, linestyle=":", linewidth=1)

    ax.set_title(f"{p_name} (N = {int(g_row['N'])})", fontsize=13, fontweight="bold")
    ax.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=11)
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=GRID_LINE_COLOR, fontsize=10)
    ax.grid(True, color=GRID_LINE_COLOR, linewidth=0.5, alpha=0.6)


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
    """Annotate bar values on top of matplotlib patches.

    Args:
        ax: Subplot Axes containing bars.
    """
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
    ax.legend(title="Model Strategy", frameon=True, facecolor="white", edgecolor=GRID_LINE_COLOR, fontsize=10)
    ax.grid(True, axis="y", color=GRID_LINE_COLOR, linewidth=0.5, alpha=0.6)

    _annotate_bars(ax)
    plt.tight_layout()
    save_fig(fig, out_path, dpi=300)
    plt.close(fig)


def _extract_feature_importances(clf: _CalibratedModel) -> Optional[np.ndarray]:
    """Extract feature importances from a _CalibratedModel wrapper.

    Args:
        clf: A fitted _CalibratedModel instance.

    Returns:
        Feature importances array, or None.
    """
    if hasattr(clf, "feature_importances_"):
        return clf.feature_importances_
    return None


def plot_feature_importances(
    final_models: Dict[str, _CalibratedModel], feature_cols: List[str], out_path: Path
) -> None:
    """Plot feature importance heatmaps comparing driver weights across phenotypes.

    Args:
        final_models: Dictionary of trained calibrated production models.
        feature_cols: Feature names.
        out_path: Output figure path.
    """
    importance_dict = {}
    for p_name, clf in final_models.items():
        importances = _extract_feature_importances(clf)
        if importances is not None:
            importance_dict[p_name] = importances

    df_imp = pd.DataFrame(importance_dict, index=feature_cols)
    df_imp["Mean_Importance"] = df_imp.mean(axis=1)
    df_imp = df_imp.sort_values(by="Mean_Importance", ascending=False).drop(columns=["Mean_Importance"])
    top_df_imp = df_imp.head(TOP_N_IMPORTANCES)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        top_df_imp,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Random Forest Gini Importance"},
        ax=ax,
        linewidths=0.5,
        linecolor=GRID_LINE_COLOR,
    )

    ax.set_title("Phenotype-Specific Feature Importance Profiles (Top 12 Features)", fontsize=13, fontweight="bold", pad=12)
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
        s_ppv = sub_df[sub_df["Model_Scope"] != "Global Enriched Baseline"]["PPV"].values[0]

        diff_auc = s_auc - g_auc if not np.isnan(s_auc) and not np.isnan(g_auc) else 0.0
        print(
            f"  * {phenotype:<25}: Global AUC = {g_auc:.3f} | Subgroup AUC = {s_auc:.3f} (Delta = {diff_auc:+.3f}) | Subgroup PPV = {s_ppv:.3f}"
        )


def _serialize_production_models(final_models: Dict[str, _CalibratedModel]) -> None:
    """Serialise trained production models to models/ directory.

    Args:
        final_models: Map of model name to trained model.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in final_models.items():
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        m_file = MODELS_DIR / f"subgroup_model_{clean_name}.joblib"
        joblib.dump(model, m_file)
        print(f"  * Serialised {name} model -> {rel_path(m_file)}")


def _generate_phase5_plots(
    roc_data: Dict[str, Dict[str, np.ndarray]],
    df_eval: pd.DataFrame,
    final_models: Dict[str, _CalibratedModel],
    feature_cols: List[str],
) -> Tuple[Path, Path, Path]:
    """Generate 300 DPI publication plots for Phase 5.

    Args:
        roc_data: Dict with ROC points.
        df_eval: Metric evaluation DataFrame.
        final_models: Map of model name to trained model.
        feature_cols: Selected feature names.

    Returns:
        Tuple of paths to (roc_plot, comp_plot, imp_plot).
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    roc_plot_file = PLOTS_DIR / "subgroup_roc_curves.png"
    comp_plot_file = PLOTS_DIR / "subgroup_performance_comparison.png"
    imp_plot_file = PLOTS_DIR / "subgroup_feature_importances.png"

    plot_subgroup_roc_curves(roc_data, df_eval, roc_plot_file)
    plot_performance_comparison(df_eval, comp_plot_file)
    plot_feature_importances(final_models, feature_cols, imp_plot_file)

    return roc_plot_file, comp_plot_file, imp_plot_file


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

    df_eval, roc_data, final_models = train_and_eval_loco(df_clusters, feature_cols)

    out_eval_csv = OUTPUT_DIR / "subgroup_models_evaluation.csv"
    safe_save_csv(df_eval, out_eval_csv)
    print(f"\nSaved subgroup models evaluation summary to {rel_path(out_eval_csv)}")

    _serialize_production_models(final_models)
    f1, f2, f3 = _generate_phase5_plots(roc_data, df_eval, final_models, feature_cols)

    print("\n" + "=" * 80)
    print("PHASE 5 SUBGROUP MODELLING COMPLETE")
    print("=" * 80)
    _print_performance_summary(df_eval)
    print(f"\nGenerated Plots:\n  1. {rel_path(f1)}\n  2. {rel_path(f2)}\n  3. {rel_path(f3)}")
    print("=" * 80)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
