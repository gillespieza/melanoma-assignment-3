"""Individual model predictor evaluation modules for each classifier architecture."""

from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

from scripts.models import evaluate_model_tiers


def evaluate_logistic_regression(
    df_features_clean: pd.DataFrame,
    y: np.ndarray,
    sig_features: List[str],
    cv: StratifiedKFold,
    evaluate_auc_cv_func: Any
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Evaluates Logistic Regression (LR) classifier across 6 feature tiers."""
    return evaluate_model_tiers(
        "Logistic Regression (LR)", "lr", df_features_clean, y, sig_features, cv, evaluate_auc_cv_func
    )


def evaluate_random_forest(
    df_features_clean: pd.DataFrame,
    y: np.ndarray,
    sig_features: List[str],
    cv: StratifiedKFold,
    evaluate_auc_cv_func: Any
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Evaluates Random Forest (RF) classifier across 6 feature tiers."""
    return evaluate_model_tiers(
        "Random Forest (RF)", "rf", df_features_clean, y, sig_features, cv, evaluate_auc_cv_func
    )


def evaluate_xgboost(
    df_features_clean: pd.DataFrame,
    y: np.ndarray,
    sig_features: List[str],
    cv: StratifiedKFold,
    evaluate_auc_cv_func: Any
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Evaluates XGBoost (XGB, tuned) classifier across 6 feature tiers."""
    return evaluate_model_tiers(
        "XGBoost (XGB, tuned)", "xgb", df_features_clean, y, sig_features, cv, evaluate_auc_cv_func
    )


def evaluate_svm(
    df_features_clean: pd.DataFrame,
    y: np.ndarray,
    sig_features: List[str],
    cv: StratifiedKFold,
    evaluate_auc_cv_func: Any
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Evaluates Support Vector Machine (SVM) classifier across 6 feature tiers."""
    return evaluate_model_tiers(
        "Support Vector Machine (SVM)", "svm", df_features_clean, y, sig_features, cv, evaluate_auc_cv_func
    )


def evaluate_elasticnet(
    df_features_clean: pd.DataFrame,
    y: np.ndarray,
    sig_features: List[str],
    cv: StratifiedKFold,
    evaluate_auc_cv_func: Any
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Evaluates Elastic-Net classifier across 6 feature tiers."""
    return evaluate_model_tiers(
        "Elastic-Net", "elasticnet", df_features_clean, y, sig_features, cv, evaluate_auc_cv_func
    )
