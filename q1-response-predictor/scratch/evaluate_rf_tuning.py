import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.signatures import extract_all_signatures


def zscore_df(df):
    means = df.mean(axis=0)
    stds = df.std(axis=0).replace(0, 1.0).fillna(1.0)
    return (df - means) / stds


def prepare_cohorts():
    data_dir = BASE_DIR / "data"
    expr_liu, clin_liu = load_liu_2019(data_dir)
    expr_hugo, clin_hugo = load_hugo_2016(data_dir)
    expr_riaz, clin_riaz = load_riaz_2017(data_dir)

    common_genes = expr_liu.columns.intersection(expr_hugo.columns).intersection(expr_riaz.columns)
    cohort_inputs = {
        "Liu 2019": (expr_liu[common_genes], clin_liu),
        "Hugo 2016": (expr_hugo[common_genes], clin_hugo),
        "Riaz 2017": (expr_riaz[common_genes], clin_riaz),
    }

    cohorts = {}
    for name, (expr, clin) in cohort_inputs.items():
        sig = extract_all_signatures(expr)
        labelled = clin.loc[sig.index, "response"].dropna().index
        cohorts[name] = (zscore_df(sig.loc[labelled]), clin.loc[labelled, "response"].astype(int))

    return cohorts


def evaluate_predictions(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "ap": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_loco_rf(cohorts, param_grid, label):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    feature_cols = next(iter(cohorts.values()))[0].columns.tolist()

    for test_cohort in cohorts:
        X_train = pd.concat(
            [cohorts[c][0][feature_cols] for c in cohorts if c != test_cohort],
            axis=0,
        )
        y_train = pd.concat([cohorts[c][1] for c in cohorts if c != test_cohort], axis=0)
        X_test, y_test = cohorts[test_cohort]
        X_test = X_test[feature_cols]

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feature_cols,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=feature_cols,
            index=X_test.index,
        )

        search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train_scaled, y_train)
        y_prob = search.best_estimator_.predict_proba(X_test_scaled)[:, 1]
        metrics = evaluate_predictions(y_test, y_prob)
        rows.append(
            {
                "search": label,
                "test_cohort": test_cohort,
                "n": len(y_test),
                "inner_cv_auc": search.best_score_,
                "best_params": search.best_params_,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def main():
    cohorts = prepare_cohorts()

    current_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 10, None],
        "min_samples_leaf": [1, 2, 4],
    }
    expanded_grid = [
        {
            "n_estimators": [200],
            "max_depth": [2, 3, 5, None],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", None],
            "class_weight": [None, "balanced"],
        },
        {
            "n_estimators": [500],
            "max_depth": [3, 5],
            "min_samples_leaf": [2, 4],
            "max_features": ["sqrt"],
            "class_weight": [None, "balanced"],
        },
    ]

    results = pd.concat(
        [
            run_loco_rf(cohorts, current_grid, "current_grid"),
            run_loco_rf(cohorts, expanded_grid, "expanded_grid"),
        ],
        ignore_index=True,
    )

    display_cols = [
        "search",
        "test_cohort",
        "n",
        "auc",
        "ap",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "inner_cv_auc",
        "best_params",
    ]
    print(results[display_cols].to_string(index=False))

    summary = (
        results.groupby("search")
        .agg(
            mean_auc=("auc", "mean"),
            median_auc=("auc", "median"),
            mean_ap=("ap", "mean"),
            mean_f1=("f1", "mean"),
        )
        .round(3)
    )
    print("\nSummary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
