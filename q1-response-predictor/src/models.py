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

def tune_logistic_regression(X_train, y_train):
    """
    Tuning L1-penalized Logistic Regression using Grid Search.
    """
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }
    lr = LogisticRegression(solver='liblinear', l1_ratio=1.0, random_state=42, max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(lr, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def tune_random_forest(X_train, y_train):
    """
    Tuning Random Forest.
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
    return grid.best_estimator_

def tune_xgboost(X_train, y_train):
    """
    Tuning XGBoost Classifier.
    """
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
    }
    # scale_pos_weight is useful for class imbalance: sum(negative) / sum(positive)
    pos_count = sum(y_train == 1)
    neg_count = sum(y_train == 0)
    scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_weight, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(xgb_clf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def tune_svc(X_train, y_train):
    """
    Tuning Support Vector Classifier (SVC) using Grid Search.
    """
    param_grid = {
        'estimator__C': [0.01, 0.1, 1.0, 10.0],
        'estimator__kernel': ['linear', 'rbf']
    }
    svc = CalibratedClassifierCV(
        estimator=SVC(random_state=42),
        method='sigmoid',
        cv=3,
        ensemble=False,
        n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(svc, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def tune_elasticnet(X_train, y_train):
    """
    Tuning ElasticNet (Logistic Regression with elasticnet penalty) using Grid Search.
    """
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    lr = LogisticRegression(solver='saga', random_state=42, max_iter=20000, tol=1e-3)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(lr, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def get_model(model_type, X_train, y_train):
    """
    Tunes and returns the requested model.
    """
    if model_type == "lr":
        return tune_logistic_regression(X_train, y_train)
    elif model_type == "rf":
        return tune_random_forest(X_train, y_train)
    elif model_type == "xgb":
        return tune_xgboost(X_train, y_train)
    elif model_type == "svm":
        return tune_svc(X_train, y_train)
    elif model_type == "elasticnet":
        return tune_elasticnet(X_train, y_train)
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
