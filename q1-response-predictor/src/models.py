from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
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
            n_estimators=100, max_depth=5, random_state=random_state, n_jobs=-1
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
        return SVC(probability=True, kernel="rbf", C=1.0, random_state=random_state)
    elif model_type == "elasticnet":
        return LogisticRegression(
            penalty="elasticnet",
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
    """
    Tuning L1-penalized Logistic Regression using Grid Search.

    Note: Probability calibration (CalibratedClassifierCV) is intentionally NOT
    applied. Logistic Regression minimises log-loss directly, so its predicted
    probabilities are already well-calibrated. Applying a secondary
    CalibratedClassifierCV(cv=3) on small training splits (N ≈ 83, 6 features)
    can learn a negative-slope sigmoid that INVERTS rank ordering, collapsing
    AUROC to below chance (observed: 0.595 → 0.383 with calibration enabled).
    """
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }
    lr = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(lr, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def tune_random_forest(X_train, y_train, calibrate=True):
    """
    Tuning Random Forest, optionally calibrated via Platt Scaling / Sigmoid calibration.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_est = grid.best_estimator_
    if calibrate:
        return calibrate_estimator(best_est, X_train, y_train)
    return best_est

def tune_xgboost(X_train, y_train, calibrate=True):
    """
    Tuning XGBoost Classifier, optionally calibrated via Platt Scaling.
    """
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
    }
    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_weight, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(xgb_clf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_est = grid.best_estimator_
    if calibrate:
        return calibrate_estimator(best_est, X_train, y_train)
    return best_est

def tune_svc(X_train, y_train, calibrate=True):
    """
    Tuning Support Vector Classifier (SVC) using Grid Search.

    Uses class_weight='balanced' to handle responder/non-responder imbalance.
    Adapts CV fold count to training set size to avoid degenerate folds on
    small LOCO splits (e.g. N~18 when Hugo 2016 is held out).
    Searches over a wider C grid and RBF gamma values for better calibration.
    """
    param_grid = {
        'estimator__C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'estimator__kernel': ['linear', 'rbf'],
        'estimator__gamma': ['scale', 'auto'],
        'estimator__class_weight': ['balanced'],
    }
    svc = CalibratedClassifierCV(
        estimator=SVC(random_state=42, probability=False),
        method='sigmoid',
        cv=3,
        ensemble=False,
        n_jobs=-1,
    )
    # Use fewer CV folds when training set is small (avoids folds with <2 samples per class)
    n_cv = max(2, min(3, len(y_train) // 10))
    cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
    grid = GridSearchCV(svc, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def tune_elasticnet(X_train, y_train, calibrate=True):
    """
    Tuning ElasticNet (Logistic Regression with elasticnet penalty) using Grid Search.

    Note: Probability calibration is intentionally NOT applied — same reasoning
    as tune_logistic_regression. ElasticNet (SAGA solver) minimises log-loss
    directly; secondary CalibratedClassifierCV on small splits harms AUROC
    (observed: 0.601 → 0.428 with calibration enabled).
    """
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    lr = LogisticRegression(
        penalty='elasticnet', solver='saga', random_state=42, max_iter=20000, tol=1e-3
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

def evaluate_predictions(y_true, y_pred_prob):
    """
    Computes performance metrics.
    """
    # Threshold at 0.5 for class labels
    y_pred_class = (y_pred_prob >= 0.5).astype(int)
    
    metrics = {
        'auc': roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else np.nan,
        'accuracy': accuracy_score(y_true, y_pred_class),
        'precision': precision_score(y_true, y_pred_class, zero_division=0),
        'recall': recall_score(y_true, y_pred_class, zero_division=0),
        'f1': f1_score(y_true, y_pred_class, zero_division=0)
    }
    return metrics

def run_loco_cv(cohort_dfs, feature_cols, model_type="lr"):
    """
    Runs Leave-One-Cohort-Out (LOCO) Cross-Validation.
    cohort_dfs: dict mapping cohort_name -> (X, y)
    feature_cols: list of feature column names to use
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
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
        
        # Tune and train model on scaled training data
        model = get_model(model_type, X_train_scaled, y_train)
        
        # Predict on scaled test cohort
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate
        metrics = evaluate_predictions(y_test, y_pred_prob)
        
        results[test_cohort] = {
            'y_true': y_test.values,
            'y_pred_prob': y_pred_prob,
            'metrics': metrics,
            'model': model,
            'scaler': scaler
        }
        
    return results
