import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from xgboost import XGBClassifier


SCRATCH_DIR = Path(__file__).resolve().parent
if str(SCRATCH_DIR) not in sys.path:
    sys.path.append(str(SCRATCH_DIR))

from evaluate_multimodal_rf_tuning import prepare_multimodal_features


def evaluate_xgb_grid(X, y, param_grid, label):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    model = GridSearchCV(
        XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        param_grid=param_grid,
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )
    result = cross_validate(
        model,
        X,
        y,
        cv=outer_cv,
        scoring="roc_auc",
        return_estimator=True,
        n_jobs=None,
    )
    scores = result["test_score"]
    best_params = [est.best_params_ for est in result["estimator"]]
    return {
        "search": label,
        "mean_auc": scores.mean(),
        "std_auc": scores.std(),
        "fold_scores": scores,
        "best_params": best_params,
    }


def main():
    features, y = prepare_multimodal_features()

    sig_features = ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "IMPRES", "PD_L1"]
    feature_sets = {
        "Signatures Only": sig_features,
        "Sigs + Drivers + Sex": sig_features + ["mut_BRAF", "mut_NRAS", "mut_NF1", "Sex_Male"],
        "Full Extended": features.columns.tolist(),
    }

    current_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
    }
    expanded_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [2, 3],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    rows = []
    for feature_set_name, cols in feature_sets.items():
        print(f"\nEvaluating {feature_set_name} ({len(cols)} features)")
        X = features[cols].values
        for label, grid in [("current_grid", current_grid), ("expanded_grid", expanded_grid)]:
            outcome = evaluate_xgb_grid(X, y, grid, label)
            rows.append(
                {
                    "feature_set": feature_set_name,
                    "search": label,
                    "auc_mean": outcome["mean_auc"],
                    "auc_std": outcome["std_auc"],
                    "fold_scores": ", ".join(f"{score:.3f}" for score in outcome["fold_scores"]),
                    "best_params_by_fold": outcome["best_params"],
                }
            )
            print(
                f"  {label}: {outcome['mean_auc']:.3f} "
                f"(+/-{outcome['std_auc']:.3f}); folds [{rows[-1]['fold_scores']}]"
            )

    results = pd.DataFrame(rows)
    print("\nXGBoost multimodal 5-fold CV comparison:")
    print(
        results.assign(
            auc=lambda df: df.apply(lambda row: f"{row.auc_mean:.3f} (+/-{row.auc_std:.3f})", axis=1)
        )[["feature_set", "search", "auc", "fold_scores"]].to_string(index=False)
    )

    print("\nBest parameters by outer fold:")
    for _, row in results.iterrows():
        print(f"\n{row['feature_set']} / {row['search']}")
        for fold_idx, params in enumerate(row["best_params_by_fold"], start=1):
            print(f"  fold {fold_idx}: {params}")


if __name__ == "__main__":
    main()
