"""Shared multimodal evaluation utilities and model wrappers."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models import get_model

_COL_TMB = "TMB_NONSYNONYMOUS"
_COL_AGE = "AGE"
_COL_RESPONSE = "response"


class TunedCalibratedModel(BaseEstimator, ClassifierMixin):
    """Scikit-learn model wrapper delegating tuning and calibration to src.models.get_model."""

    _estimator_type = "classifier"

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(self, model_type: str = "rf", calibrate: bool = True) -> None:
        self.model_type = model_type
        self.calibrate = calibrate
        self.fitted_model_: Any = None
        self.classes_: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TunedCalibratedModel":
        self.classes_ = np.unique(y)
        self.fitted_model_ = get_model(
            self.model_type, X, y, calibrate=self.calibrate
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.fitted_model_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.fitted_model_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.fitted_model_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.fitted_model_.predict(X)


def prepare_predictor_features(
    df_clin_merged: pd.DataFrame,
    df_sigs_merged: pd.DataFrame,
    project_root: Path
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Construct clean feature matrix and target array for predictor training."""
    df_sigs_aligned = df_sigs_merged.loc[df_clin_merged.index]
    sig_features = ['IFN_gamma', 'TIS', 'CYT', 'CD8_Tcell', 'IMPRES', 'PD_L1']

    q5_feat_path = project_root / "data" / "processed" / "q5" / "feature_matrix.csv"
    if q5_feat_path.exists():
        df_q5 = pd.read_csv(q5_feat_path).set_index('SAMPLE_ID')
        m1_m2 = df_q5['M1_M2_Ratio'].reindex(df_clin_merged.index)
        mac_stv = df_q5['Macrophage_STV_Score'].reindex(df_clin_merged.index)
    else:
        m1_m2 = pd.Series(np.nan, index=df_clin_merged.index)
        mac_stv = pd.Series(np.nan, index=df_clin_merged.index)

    df_features = pd.concat([
        df_sigs_aligned,
        df_clin_merged[[
            'mut_BRAF', 'mut_NRAS', 'mut_NF1',
            'mut_Antigen_Presentation', 'mut_IFN_gamma_Signaling', 'mut_Survival_Pathways',
            _COL_TMB, _COL_AGE
        ]],
        m1_m2.rename('M1_M2_Ratio'),
        mac_stv.rename('Macrophage_STV')
    ], axis=1)

    df_features[_COL_TMB] = df_features[_COL_TMB].fillna(df_features[_COL_TMB].median())
    df_features[_COL_AGE] = df_features[_COL_AGE].fillna(df_features[_COL_AGE].median())
    df_features['M1_M2_Ratio'] = df_features['M1_M2_Ratio'].fillna(df_features['M1_M2_Ratio'].median())
    df_features['Macrophage_STV'] = df_features['Macrophage_STV'].fillna(df_features['Macrophage_STV'].median())

    clean_idx = df_clin_merged[_COL_RESPONSE].dropna().index
    df_features_clean = df_features.loc[clean_idx]
    y = df_clin_merged.loc[clean_idx, _COL_RESPONSE].values

    return df_features_clean, y, sig_features


def evaluate_model_tiers(
    model_name: str,
    model_key: str,
    df_features_clean: pd.DataFrame,
    y: np.ndarray,
    sig_features: List[str],
    cv: StratifiedKFold,
    evaluate_auc_cv_func: Any
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Evaluates all 6 feature permutation tiers for a given model architecture."""
    model = TunedCalibratedModel(model_key)
    driver_cols = ['mut_BRAF', 'mut_NRAS', 'mut_NF1']

    scores_base = evaluate_auc_cv_func(model, df_features_clean[sig_features].values, y, cv, "6 Signatures Only")
    scores_drivers = evaluate_auc_cv_func(model, df_features_clean[sig_features + driver_cols].values, y, cv, "Signatures + Drivers")
    scores_tmb = evaluate_auc_cv_func(model, df_features_clean[sig_features + [_COL_TMB]].values, y, cv, "Signatures + TMB")
    scores_m1m2 = evaluate_auc_cv_func(model, df_features_clean[sig_features + ['M1_M2_Ratio']].values, y, cv, "Signatures + M1/M2 Ratio")
    scores_mac_stv = evaluate_auc_cv_func(model, df_features_clean[sig_features + ['Macrophage_STV']].values, y, cv, "Signatures + Macrophage STV")

    cols_12 = sig_features + driver_cols + [_COL_TMB, _COL_AGE, 'mut_Antigen_Presentation']
    scores_12 = evaluate_auc_cv_func(model, df_features_clean[cols_12].values, y, cv, "12-Feature Final Model")

    res = {
        'Model': model_name,
        'Base AUROC': f"{scores_base.mean():.3f} (+/-{scores_base.std():.3f})",
        'Sigs+Drivers AUROC': f"{scores_drivers.mean():.3f} (+/-{scores_drivers.std():.3f})",
        'Sigs+TMB AUROC': f"{scores_tmb.mean():.3f} (+/-{scores_tmb.std():.3f})",
        'Sigs+M1/M2 Ratio AUROC': f"{scores_m1m2.mean():.3f} (+/-{scores_m1m2.std():.3f})",
        'Sigs+Macrophage STV AUROC': f"{scores_mac_stv.mean():.3f} (+/-{scores_mac_stv.std():.3f})",
        '12-Feature Model AUROC': f"{scores_12.mean():.3f} (+/-{scores_12.std():.3f})"
    }
    plot_row = {
        'model': model_name,
        'base_mean': scores_base.mean(), 'base_std': scores_base.std(), 'base_folds': scores_base,
        'drivers_mean': scores_drivers.mean(), 'drivers_std': scores_drivers.std(), 'drivers_folds': scores_drivers,
        'tmb_mean': scores_tmb.mean(), 'tmb_std': scores_tmb.std(), 'tmb_folds': scores_tmb,
        'm1m2_mean': scores_m1m2.mean(), 'm1m2_std': scores_m1m2.std(), 'm1m2_folds': scores_m1m2,
        'mac_stv_mean': scores_mac_stv.mean(), 'mac_stv_std': scores_mac_stv.std(), 'mac_stv_folds': scores_mac_stv,
        'model12_mean': scores_12.mean(), 'model12_std': scores_12.std(), 'model12_folds': scores_12,
    }
    return res, plot_row
