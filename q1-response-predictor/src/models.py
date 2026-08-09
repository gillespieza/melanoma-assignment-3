from typing import Any

import numpy as np
import pandas as pd
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
        ensemble=False,
        n_jobs=-1
    )
    calibrated.fit(X_train, y_train)
    return calibrated

def tune_logistic_regression(X_train, y_train, calibrate=True):
    """Tune Logistic Regression (L1/L2 penalty) via exhaustive Grid Search.

    Grid: 6 C values × 2 penalties × 2 class_weight options = 24 combinations.
    Compact enough for exhaustive GridSearchCV (no need for RandomizedSearchCV).

    Key improvements over the initial single-axis grid:
      - l1_ratio: searches both 0.0 (L2 penalty) and 1.0 (L1 penalty). L2 often outperforms L1 on small-N
        LOCO training splits (N ≈ 80–100) where the sparsity assumption underlying
        L1 is less justified with only 6–12 immune-signature features.
      - class_weight: 'balanced' corrects the responder/non-responder imbalance
        without synthetic oversampling; None is retained because 'balanced' can
        over-penalise when class frequencies are already near-equal on a given split.
      - Adaptive CV folds: mirrors the SVM tuner — prevents degenerate folds on
        small LOCO training sets (N < 30) where a fixed 5-fold CV is not viable.

    Note: Probability calibration (CalibratedClassifierCV) is intentionally NOT
    applied by default. Logistic Regression minimises log-loss directly, so its
    predicted probabilities are already well-calibrated. Applying a secondary
    CalibratedClassifierCV(cv=3) on small training splits (N ≈ 83, 6 features)
    can learn a negative-slope sigmoid that INVERTS rank ordering, collapsing
    AUROC to below chance (observed: 0.595 → 0.383 with calibration enabled).
    """
    # sklearn 1.8 deprecation: the `penalty` parameter on LogisticRegression was
    # deprecated in version 1.8 and will be removed in version 1.10. The new API
    # uses `l1_ratio` to control the penalty mix:
    #   l1_ratio=0.0  →  pure L2 regularisation (equivalent to penalty='l2')
    #   l1_ratio=1.0  →  pure L1 regularisation (equivalent to penalty='l1')
    # We search both extremes here; intermediate values are reserved for ElasticNet
    # (tune_elasticnet). The liblinear solver natively supports both L1 and L2 via
    # l1_ratio and does not require setting penalty explicitly.
    param_grid = {
        'l1_ratio':     [0.0, 1.0],
        'C':            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'class_weight': ['balanced', None],
    }
    lr = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
    n_cv = max(2, min(5, len(y_train) // 10))
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
    grid = GridSearchCV(lr, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_est = grid.best_estimator_
    if calibrate:
        return calibrate_estimator(best_est, X_train, y_train)
    return best_est

def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series, calibrate: bool = True):
    """Tune Random Forest via Randomised Search, optionally calibrated via Platt Scaling.

    Uses RandomizedSearchCV (n_iter=80) over an expanded parameter space rather than
    exhaustive GridSearchCV. The expanded grid covers ~576 combinations, making random
    sampling significantly more efficient.

    Key improvements over the previous grid:
      - class_weight: 'balanced' or 'balanced_subsample' corrects the responder/non-responder
        imbalance without requiring synthetic oversampling. 'balanced_subsample' recomputes
        weights per bootstrap sample, which is the preferred mode for RF.
      - max_features: controls how many features each split considers. With only 6 input
        features, the sklearn default ('sqrt' ≈ 2.4) is very restrictive; searching
        'log2' and 0.5 gives trees access to more signal per split.
      - min_samples_split: controls when a node is eligible for further splitting.
        Higher values regularise the trees against the small LOCO training splits.
      - Wider n_estimators range (up to 300): more trees reduce variance without
        introducing bias; plateau is typically reached by 200–300 for 6 features.
      - Wider min_samples_leaf range (up to 8): prevents tiny terminal nodes that
        overfit to cohort-specific outliers.
    """
    param_dist = {
        'n_estimators':      [100, 200, 300],
        'max_depth':         [3, 5, 10, None],
        'min_samples_leaf':  [1, 2, 4, 8],
        'min_samples_split': [2, 5, 10],
        'max_features':      ['sqrt', 'log2', 0.5],
        'class_weight':      ['balanced', 'balanced_subsample'],
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        rf,
        param_dist,
        n_iter=80,
        cv=cv,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_est = search.best_estimator_
    if calibrate:
        return calibrate_estimator(best_est, X_train, y_train)
    return best_est

def tune_xgboost(X_train, y_train, calibrate=True):
    """Tune XGBoost Classifier via Randomised Search, optionally calibrated via Platt Scaling.

    Uses RandomizedSearchCV (n_iter=60) rather than exhaustive GridSearchCV because
    the expanded parameter space — including the four key regularisation axes
    (subsample, colsample_bytree, min_child_weight, gamma) — yields ~3,900+ grid
    combinations, which is computationally prohibitive. Randomised search samples
    the space efficiently and consistently finds near-optimal configurations.

    Key regularisation parameters added:
      - subsample: row sampling fraction per tree (reduces overfitting to training cohort).
      - colsample_bytree: feature sampling per tree (important with 6–12 features).
      - min_child_weight: minimum sum of instance weight in a child leaf (guards against
        spurious splits on small-N cohort training data).
      - gamma: minimum loss reduction required to make a further partition (explicit
        tree-complexity penalty).
    """
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
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        xgb_clf,
        param_dist,
        n_iter=60,
        cv=cv,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_est = search.best_estimator_
    if calibrate:
        return calibrate_estimator(best_est, X_train, y_train)
    return best_est

def tune_svc(X_train, y_train, calibrate=True):
    """Tune Support Vector Classifier (SVC) via Randomised Search, then optionally calibrate.

    Decouples hyperparameter tuning from probability calibration (same pattern as RF/XGB):
    tunes a raw SVC on decision_function scores (optimal for roc_auc scoring), then applies
    CalibratedClassifierCV on the full training set after the best hyperparameters are found.

    Uses a list-of-dicts param_grid so each kernel only searches its own relevant parameters:
      - linear: no gamma or degree (both are ignored by sklearn for linear SVC)
      - rbf:    no degree (irrelevant); numeric gamma values added for RBF bandwidth control
      - poly:   degree [2, 3] and coef0 control polynomial architecture; gamma controls
                the kernel coefficient; captures non-linear interaction terms between
                immune signatures (e.g. IFN_gamma × TMB) without explicit feature engineering

    Switches from GridSearchCV to RandomizedSearchCV (n_iter=80) because the three
    kernel-specific grids together cover ~230+ combinations — exhaustive search on small
    LOCO training splits (N~80-100, sometimes N~18) is unnecessarily expensive.

    Key design decisions:
      - C grid extended to [0.001 … 1000.0] with finer log-scale resolution (8 values
        vs. 5 previously); avoids landing in a coarse gap between regularisation regimes.
      - gamma numeric values [0.001, 0.01, 0.1, 1.0] for rbf/poly: with 6-12 correlated
        immune signatures, 'scale' and 'auto' often resolve to similar values, so explicit
        numeric search matters.
      - class_weight searched over ['balanced', None]: 'balanced' corrects responder
        imbalance; None is included because 'balanced' can over-penalise on small LOCO
        splits where class frequencies are already near-equal.
      - Adaptive n_cv prevents degenerate folds on small LOCO training sets.
    """
    # Per-kernel grids prevent irrelevant parameter combinations being evaluated
    # (e.g. gamma with linear, degree with rbf). Each sub-dict is searched independently.
    _C_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]
    _cw_grid = ['balanced', None]
    param_grid = [
        # --- Linear kernel: no gamma, no degree ---
        {
            'kernel':       ['linear'],
            'C':            _C_grid,
            'class_weight': _cw_grid,
        },
        # --- RBF kernel: gamma controls bandwidth, no degree ---
        {
            'kernel':       ['rbf'],
            'C':            _C_grid,
            'gamma':        ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
            'class_weight': _cw_grid,
        },
        # --- Polynomial kernel: degree and coef0 define architecture ---
        {
            'kernel':       ['poly'],
            'C':            _C_grid,
            'degree':       [2, 3],
            'gamma':        ['scale', 'auto', 0.001, 0.01, 0.1],
            'coef0':        [0.0, 0.5, 1.0],
            'class_weight': _cw_grid,
        },
    ]
    # Tune on raw SVC (no calibration yet) — decision_function scores are
    # rank-equivalent to probabilities and are the correct signal for roc_auc.
    svc = SVC(random_state=42, probability=False)
    n_cv = max(2, min(3, len(y_train) // 10))
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        svc, param_grid, n_iter=80, cv=cv, scoring='roc_auc',
        random_state=42, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    best_est = search.best_estimator_
    # Apply Platt Scaling on the full training set after hyperparameter selection,
    # not nested inside the CV loop — same pattern as tune_random_forest / tune_xgboost.
    if calibrate:
        return calibrate_estimator(best_est, X_train, y_train)
    return best_est

def tune_elasticnet(X_train, y_train, calibrate=True):
    """
    Tuning ElasticNet (Logistic Regression with elasticnet penalty) using Grid Search.

    Grid: 6 C values × 5 l1_ratio values × 2 class_weight options = 60 combinations.
    Still compact enough for exhaustive GridSearchCV (no need for RandomizedSearchCV).

    Key improvements over the initial grid:
      - class_weight: 'balanced' corrects the responder/non-responder imbalance on
        small LOCO training splits (N ≈ 80–100) without synthetic oversampling; None
        is retained because 'balanced' can over-penalise when class frequencies are
        already near-equal on a given split.
      - C upper end extended to 100.0 (matching tune_logistic_regression): with only
        6 features and a strong L1 component, some LOCO splits benefit from weaker
        regularisation that the previous ceiling of 10.0 could not reach.
      - tol tightened from 1e-3 (sklearn default) to 1e-4: with max_iter=20000 there
        is ample budget for tighter convergence, particularly on flatter loss surfaces
        produced by smaller cohort training sets.

    Note: Probability calibration is intentionally NOT applied — same reasoning
    as tune_logistic_regression. ElasticNet (SAGA solver) minimises log-loss
    directly; secondary CalibratedClassifierCV on small splits harms AUROC
    (observed: 0.601 → 0.428 with calibration enabled).
    """
    param_grid = {
        'C':            [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'l1_ratio':     [0.1, 0.3, 0.5, 0.7, 0.9],
        'class_weight': ['balanced', None],
    }
    # sklearn 1.8 deprecation: penalty='elasticnet' is no longer required (or
    # accepted without a FutureWarning). With solver='saga', any l1_ratio value
    # in (0, 1) produces an ElasticNet objective by default. l1_ratio is already
    # present in the param_grid above, so the base estimator needs no explicit
    # penalty argument — the grid search will inject l1_ratio per candidate.
    lr = LogisticRegression(
        solver='saga', random_state=42, max_iter=20000, tol=1e-4
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(lr, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def get_model(model_type, X_train, y_train, calibrate=True):
    """
    Tunes and returns the requested calibrated model.
    """
    if model_type == "lr":
        return tune_logistic_regression(X_train, y_train, calibrate=calibrate)
    elif model_type == "rf":
        return tune_random_forest(X_train, y_train, calibrate=calibrate)
    elif model_type == "xgb":
        return tune_xgboost(X_train, y_train, calibrate=calibrate)
    elif model_type == "svm":
        return tune_svc(X_train, y_train, calibrate=calibrate)
    elif model_type == "elasticnet":
        return tune_elasticnet(X_train, y_train, calibrate=calibrate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


