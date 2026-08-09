import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def find_optimal_threshold(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """Finds decision threshold maximizing Youden's J statistic (Sensitivity + Specificity - 1).

    Fits strictly on training set predictions to prevent test-set data leakage.
    Clips threshold bounds to [0.1, 0.9] to prevent extreme boundary collapses.

    Args:
        y_true: Ground truth binary labels (0/1).
        y_pred_prob: Predicted probability array.

    Returns:
        Optimal float threshold in range [0.1, 0.9].
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    raw_thresh = float(thresholds[best_idx])
    return float(np.clip(raw_thresh, 0.1, 0.9))


def evaluate_predictions(y_true, y_pred_prob, threshold=0.5):
    """Computes classification performance metrics at a given decision threshold.

    Args:
        y_true: Array of ground-truth binary labels.
        y_pred_prob: Array of predicted response probabilities.
        threshold: Decision threshold for positive classification (default 0.5).

    Returns:
        Dict of metrics: auc, accuracy, precision, recall, f1, threshold.
    """
    y_pred_class = (y_pred_prob >= threshold).astype(int)

    metrics = {
        "auc": (
            roc_auc_score(y_true, y_pred_prob)
            if len(np.unique(y_true)) > 1
            else np.nan
        ),
        "accuracy": accuracy_score(y_true, y_pred_class),
        "precision": precision_score(y_true, y_pred_class, zero_division=0),
        "recall": recall_score(y_true, y_pred_class, zero_division=0),
        "f1": f1_score(y_true, y_pred_class, zero_division=0),
        "threshold": float(threshold),
    }
    return metrics


def run_loco_cv(
    cohort_dfs: dict,
    feature_cols: list,
    model_type: str = "lr",
    optimize_threshold: bool = True,
):
    """Runs Leave-One-Cohort-Out (LOCO) Cross-Validation.

    Args:
        cohort_dfs: Dict mapping cohort_name -> (X, y).
        feature_cols: List of feature column names to use.
        model_type: Classifier architecture key ('lr', 'rf', 'xgb', 'svm', 'elasticnet').
        optimize_threshold: If True, calculates decision threshold on training predictions
            via Youden's J to prevent zero-sensitivity prediction collapses without test leakage.

    Returns:
        Dict of results per test cohort containing y_true, y_pred_prob, metrics, threshold,
        model, and scaler.
    """
    results = {}
    cohort_names = list(cohort_dfs.keys())

    for test_cohort in cohort_names:
        # Prepare training data (combine all cohorts except the test cohort)
        train_cohorts = [c for c in cohort_names if c != test_cohort]

        X_train_list = []
        y_train_list = []

        for c in train_cohorts:
            X_c, y_c = cohort_dfs[c]
            X_train_list.append(X_c[feature_cols])
            y_train_list.append(y_c)

        X_train = pd.concat(X_train_list, axis=0)
        y_train = pd.concat(y_train_list, axis=0)

        # Prepare test data
        X_test, y_test = cohort_dfs[test_cohort]
        X_test = X_test[feature_cols]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(
            X_train_scaled, columns=feature_cols, index=X_train.index
        )
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(
            X_test_scaled, columns=feature_cols, index=X_test.index
        )

        # Tune and train model on scaled training data
        model = get_model(model_type, X_train_scaled, y_train)

        # Determine threshold on training fold predictions to prevent test leakage
        if optimize_threshold:
            y_train_prob = model.predict_proba(X_train_scaled)[:, 1]
            opt_thresh = find_optimal_threshold(y_train.values, y_train_prob)
        else:
            opt_thresh = 0.5

        # Predict on scaled test cohort
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate
        metrics = evaluate_predictions(y_test, y_pred_prob, threshold=opt_thresh)

        results[test_cohort] = {
            "y_true": y_test.values,
            "y_pred_prob": y_pred_prob,
            "threshold": opt_thresh,
            "metrics": metrics,
            "model": model,
            "scaler": scaler,
        }

    return results

import xgboost as xgb


def get_baseline_model(model_type: str, random_state: int = 42) -> Any:
    """Instantiates an unfitted baseline classifier with fixed hyperparameters matching comparison benchmarks.

    Args:
        model_type: One of 'lr', 'rf', 'xgb', 'svm', 'elasticnet'.
        random_state: Seed for pseudo-random number generator.

    Returns:
        Unfitted sklearn-compatible classifier instance.

    Raises:
        ValueError: If model_type is unknown.
    """
    if model_type == "lr":
        return LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)
    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=100, max_depth=5, class_weight='balanced',
            random_state=random_state, n_jobs=-1
        )
    elif model_type == "xgb":
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )
    elif model_type == "svm":
        # sklearn 1.9 deprecation: probability=True on SVC is deprecated — wrap SVC in
        # CalibratedClassifierCV(ensemble=False) for Platt scaling instead.
        return CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, random_state=random_state),
            ensemble=False,
        )
    elif model_type == "elasticnet":
        # sklearn 1.8: penalty='elasticnet' deprecated — l1_ratio=0.5 with
        # solver='saga' implicitly selects ElasticNet (50% L1 / 50% L2 mix).
        return LogisticRegression(
            solver="saga",
            l1_ratio=0.5,
            max_iter=2000,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")


def calibrate_estimator(estimator, X_train, y_train, method='sigmoid', cv=3):
    """
    Fits a CalibratedClassifierCV wrapper around a base estimator using internal cross-validation.
    """
    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=cv,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


class _PlattScaledSVC(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper that applies Platt scaling to a pre-fitted SVC.

    Sklearn 1.9 deprecated ``SVC(probability=True)`` and removed ``cv='prefit'``
    from ``CalibratedClassifierCV``. This class bypasses both by calling
    ``decision_function`` directly and fitting a logistic regression calibrator
    on those scores — which is exactly what Platt scaling is — without ever
    touching the ``probability`` parameter.

    Args:
        fitted_svc: A pre-fitted SVC instance (probability=False).
    """

    def __init__(self, fitted_svc: SVC) -> None:
        self.fitted_svc = fitted_svc

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_PlattScaledSVC":
        """Fit the Platt calibrator on decision function scores."""
        self.classes_ = np.unique(y)
        scores = self.fitted_svc.decision_function(X).reshape(-1, 1)
        self.calibrator_ = LogisticRegression(C=1.0, max_iter=1000).fit(scores, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probability estimates."""
        scores = self.fitted_svc.decision_function(X).reshape(-1, 1)
        return self.calibrator_.predict_proba(scores)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard class predictions."""
        scores = self.fitted_svc.decision_function(X).reshape(-1, 1)
        return self.calibrator_.predict(scores)

def tune_logistic_regression(X_train, y_train, calibrate=True, n_jobs=-1):
    param_grid = {
        'l1_ratio':     [0.0, 1.0],
        'C':            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'class_weight': ['balanced', None],
    }
    lr = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
    n_cv = max(2, min(5, len(y_train) // 10))
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
    grid = GridSearchCV(lr, param_grid, cv=cv, scoring='roc_auc', n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    best_est = grid.best_estimator_
    if calibrate:
        return calibrate_estimator(best_est, X_train, y_train)
    return best_est

def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series, calibrate: bool = True, n_jobs=-1):
    param_dist = {
        'n_estimators':      [100, 200, 300],
        'max_depth':         [3, 5, 10, None],
        'min_samples_leaf':  [1, 2, 4, 8],
        'min_samples_split': [2, 5, 10],
        'max_features':      ['sqrt', 'log2', 0.5],
        'class_weight':      ['balanced', 'balanced_subsample'],
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=n_jobs)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        rf,
        param_dist,
        n_iter=80,
        cv=cv,
        scoring='roc_auc',
        random_state=42,
        n_jobs=n_jobs,
    )
    search.fit(X_train, y_train)
    best_est = search.best_estimator_
    if calibrate:
        return calibrate_estimator(best_est, X_train, y_train)
    return best_est

def tune_xgboost(X_train, y_train, calibrate=True, n_jobs=-1):
    param_dist = {
        'n_estimators':      [50, 100, 150, 200],
        'max_depth':         [3, 5, 7],
        'learning_rate':     [0.01, 0.05, 0.1, 0.2],
        'subsample':         [0.6, 0.7, 0.8, 1.0],
        'colsample_bytree':  [0.6, 0.7, 0.8, 1.0],
        'min_child_weight':  [1, 3, 5],
        'gamma':             [0, 0.1, 0.3, 0.5],
    }
    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

    xgb_clf = xgb.XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=n_jobs,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        xgb_clf,
        param_dist,
        n_iter=60,
        cv=cv,
        scoring='roc_auc',
        random_state=42,
        n_jobs=n_jobs,
    )
    search.fit(X_train, y_train)
    best_est = search.best_estimator_
    if calibrate:
        return calibrate_estimator(best_est, X_train, y_train)
    return best_est

def tune_svc(X_train, y_train, calibrate=True, n_jobs=-1):
    _C_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]
    _cw_grid = ['balanced', None]
    param_grid = [
        {
            'kernel':       ['linear'],
            'C':            _C_grid,
            'class_weight': _cw_grid,
        },
        {
            'kernel':       ['rbf'],
            'C':            _C_grid,
            'gamma':        ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
            'class_weight': _cw_grid,
        },
        {
            'kernel':       ['poly'],
            'C':            _C_grid,
            'degree':       [2, 3],
            'gamma':        ['scale', 'auto', 0.001, 0.01, 0.1],
            'coef0':        [0.0, 0.5, 1.0],
            'class_weight': _cw_grid,
        },
    ]
    svc = SVC(random_state=42, probability=False)
    n_cv = max(2, min(3, len(y_train) // 10))
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        svc, param_grid, n_iter=80, cv=cv, scoring='roc_auc',
        random_state=42, n_jobs=n_jobs,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*probability.*parameter was deprecated.*",
            category=FutureWarning,
            module="sklearn",
        )
        search.fit(X_train, y_train)
    best_est = search.best_estimator_
    if calibrate:
        wrapper = _PlattScaledSVC(best_est)
        wrapper.fit(X_train, y_train)
        return wrapper
    return best_est

def tune_elasticnet(X_train, y_train, calibrate=True, n_jobs=-1):
    param_grid = {
        'C':            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'l1_ratio':     [0.1, 0.3, 0.5, 0.7, 0.9],
        'class_weight': ['balanced', None],
    }
    lr = LogisticRegression(
        solver='saga', random_state=42, max_iter=20000, tol=1e-4
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(lr, param_grid, cv=cv, scoring='roc_auc', n_jobs=n_jobs)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def get_model(model_type, X_train, y_train, calibrate=True, n_jobs=-1):
    """
    Tunes and returns the requested calibrated model.
    """
    if model_type == "lr":
        return tune_logistic_regression(X_train, y_train, calibrate=calibrate, n_jobs=n_jobs)
    elif model_type == "rf":
        return tune_random_forest(X_train, y_train, calibrate=calibrate, n_jobs=n_jobs)
    elif model_type == "xgb":
        return tune_xgboost(X_train, y_train, calibrate=calibrate, n_jobs=n_jobs)
    elif model_type == "svm":
        return tune_svc(X_train, y_train, calibrate=calibrate, n_jobs=n_jobs)
    elif model_type == "elasticnet":
        return tune_elasticnet(X_train, y_train, calibrate=calibrate, n_jobs=n_jobs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


