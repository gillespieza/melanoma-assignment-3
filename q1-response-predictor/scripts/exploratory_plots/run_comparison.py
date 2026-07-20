"""
Cross-Cohort Validation: Curated Immune Signatures vs. SelectKBest Feature Selection.

Evaluates predictive models (Logistic Regression, Random Forest, XGBoost) under Leave-One-Cohort-Out (LOCO)
cross-validation, comparing domain-driven curated signatures against data-driven SelectKBest feature selection
(k=20, k=100, k=200), exporting performance summaries and grouped bar charts.
"""

import contextlib
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import seaborn as sns

# Bootstrap project root resolution for top-level import
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.signatures import extract_all_signatures
from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import find_project_root
from src.utils.plotting import save_fig

# Module-level Constants
DATA_DIR = find_project_root(Path(__file__).resolve()) / "data"
PLOT_DIR = find_project_root(Path(__file__).resolve()) / "plots" / "feature_selection"
REPORTS_DIR = find_project_root(Path(__file__).resolve()) / "reports"
LOG_DIR = find_project_root(Path(__file__).resolve()) / "logs"
LOG_PATH = LOG_DIR / "run_comparison.log"


def _run_loco_signatures(cohort_dfs: Dict[str, Tuple[pd.DataFrame, pd.Series]], model_type: str = "rf") -> Dict[str, float]:
    """Runs LOCO cross-validation using the curated 6 immune signatures.

    Args:
        cohort_dfs: Dictionary mapping cohort names to (signatures_df, response_series).
        model_type: Classifier choice ('lr', 'rf', or 'xgb').

    Returns:
        Dictionary mapping held-out cohort name to test ROC-AUC.
    """
    cohorts = list(cohort_dfs.keys())
    results = {}

    for test_cohort in cohorts:
        train_cohorts = [c for c in cohorts if c != test_cohort]

        X_train = pd.concat([cohort_dfs[c][0] for c in train_cohorts], axis=0)
        y_train = pd.concat([cohort_dfs[c][1] for c in train_cohorts], axis=0)

        X_test = cohort_dfs[test_cohort][0]
        y_test = cohort_dfs[test_cohort][1]

        if model_type == "lr":
            model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        elif model_type == "rf":
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        elif model_type == "xgb":
            model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=-1
            )

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = np.nan

        results[test_cohort] = auc

    return results


def _run_loco_feature_selection(
    cohort_expr_dfs: Dict[str, pd.DataFrame],
    cohort_y_dfs: Dict[str, pd.Series],
    k_features: int = 20,
    model_type: str = "rf",
) -> Dict[str, Tuple[float, List[str]]]:
    """Runs LOCO cross-validation with SelectKBest feature selection on raw expression data.

    Args:
        cohort_expr_dfs: Dictionary mapping cohort names to gene expression DataFrames.
        cohort_y_dfs: Dictionary mapping cohort names to response Series.
        k_features: Number of top features to select.
        model_type: Classifier choice ('lr', 'rf', or 'xgb').

    Returns:
        Dictionary mapping test cohort to (ROC-AUC, selected_genes list).
    """
    cohorts = list(cohort_expr_dfs.keys())
    results = {}

    for test_cohort in cohorts:
        train_cohorts = [c for c in cohorts if c != test_cohort]

        X_train_raw = pd.concat([cohort_expr_dfs[c] for c in train_cohorts], axis=0)
        y_train = pd.concat([cohort_y_dfs[c] for c in train_cohorts], axis=0)

        X_test_raw = cohort_expr_dfs[test_cohort]
        y_test = cohort_y_dfs[test_cohort]

        train_vars = X_train_raw.var(axis=0)
        top_var_genes = train_vars.nlargest(min(1000, X_train_raw.shape[1])).index
        X_train_filtered = X_train_raw[top_var_genes]
        X_test_filtered = X_test_raw[top_var_genes]

        k_val = min(k_features, X_train_filtered.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k_val)
        selector.fit(X_train_filtered, y_train)

        selected_genes = top_var_genes[selector.get_support()]

        X_train_sel = X_train_filtered[selected_genes]
        X_test_sel = X_test_filtered[selected_genes]

        if model_type == "lr":
            model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        elif model_type == "rf":
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
        elif model_type == "xgb":
            model = XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=-1
            )

        model.fit(X_train_sel, y_train)
        y_prob = model.predict_proba(X_test_sel)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = np.nan

        results[test_cohort] = (auc, list(selected_genes))

    return results


def _plot_comparison_results(df_results: pd.DataFrame, out_plot_path: Path) -> None:
    """Generates a grouped bar chart comparing ROC-AUC of Domain Signatures vs Raw SelectKBest.

    Args:
        df_results: Results DataFrame containing LOCO AUC scores.
        out_plot_path: Destination path for figure output artifact.
    """
    set_presentation_style()
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    models = ["LR", "RF", "XGB"]
    model_titles = {"LR": "Logistic Regression", "RF": "Random Forest", "XGB": "XGBoost"}
    palette = {
        "Curated Signatures": COHORT_PALETTE["Liu 2019"],
        "SelectKBest (k=20)": COHORT_PALETTE["Hugo 2016"],
        "SelectKBest (k=100)": COHORT_PALETTE["Riaz 2017"],
        "SelectKBest (k=200)": RESPONSE_PALETTE["PD"],
    }

    for ax, m in zip(axes, models):
        df_sub = df_results[df_results["Model"] == m].copy()

        df_melted = pd.melt(
            df_sub,
            id_vars=["Test Cohort"],
            value_vars=[
                "Curated Signatures AUC",
                "SelectKBest (k=20) AUC",
                "SelectKBest (k=100) AUC",
                "SelectKBest (k=200) AUC",
            ],
            var_name="Feature Representation",
            value_name="ROC-AUC",
        )
        df_melted["Feature Representation"] = df_melted["Feature Representation"].str.replace(" AUC", "")

        sns.barplot(
            data=df_melted,
            x="Test Cohort",
            y="ROC-AUC",
            hue="Feature Representation",
            palette=palette,
            ax=ax,
            edgecolor="black",
            linewidth=0.8,
        )

        ax.set_title(f"{model_titles[m]}", fontsize=13, fontweight="bold", pad=10)
        ax.axhline(0.50, color="gray", linestyle="--", linewidth=1.2, label="Chance Baseline (AUC=0.5)")
        ax.set_ylim(0.25, 0.85)
        ax.set_xlabel("Held-out Test Cohort", fontsize=11, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Cross-Validated ROC-AUC", fontsize=11, fontweight="bold")
        else:
            ax.set_ylabel("")

        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(
                    f"{height:.2f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="black",
                    xytext=(0, 2),
                    textcoords="offset points",
                )

        ax.legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=11, frameon=True)

    plt.suptitle(
        "Cross-Cohort Validation: Curated Immune Signatures vs. Data-Driven SelectKBest Feature Selection",
        fontsize=14,
        fontweight="bold",
        y=1.12,
    )
    save_fig(fig, out_plot_path)
    print(f"Saved comparison plot to {out_plot_path.relative_to(BASE_DIR).as_posix()}")


def main() -> None:
    """Executes feature selection comparison pipeline."""
    print("==================================================")
    print("LOCO CV Evaluation: Curated Signatures vs SelectKBest")
    print("==================================================\n")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)

    expr_liu, clin_liu = load_liu_2019(DATA_DIR)
    expr_hugo, clin_hugo = load_hugo_2016(DATA_DIR)
    expr_riaz, clin_riaz = load_riaz_2017(DATA_DIR)

    common_genes = list(set(expr_liu.columns) & set(expr_hugo.columns) & set(expr_riaz.columns))

    sig_liu = extract_all_signatures(expr_liu[common_genes])
    sig_hugo = extract_all_signatures(expr_hugo[common_genes])
    sig_riaz = extract_all_signatures(expr_riaz[common_genes])

    y_liu = clin_liu.loc[sig_liu.index, "response"].dropna()
    y_hugo = clin_hugo.loc[sig_hugo.index, "response"].dropna()
    y_riaz = clin_riaz.loc[sig_riaz.index, "response"].dropna()

    sig_liu = sig_liu.loc[y_liu.index]
    sig_hugo = sig_hugo.loc[y_hugo.index]
    sig_riaz = sig_riaz.loc[y_riaz.index]

    expr_liu_common = expr_liu.loc[y_liu.index, common_genes]
    expr_hugo_common = expr_hugo.loc[y_hugo.index, common_genes]
    expr_riaz_common = expr_riaz.loc[y_riaz.index, common_genes]

    cohort_sig_dfs = {
        "Liu 2019": (sig_liu, y_liu),
        "Hugo 2016": (sig_hugo, y_hugo),
        "Riaz 2017": (sig_riaz, y_riaz),
    }

    cohort_expr_dfs = {
        "Liu 2019": expr_liu_common,
        "Hugo 2016": expr_hugo_common,
        "Riaz 2017": expr_riaz_common,
    }

    cohort_y_dfs = {
        "Liu 2019": y_liu,
        "Hugo 2016": y_hugo,
        "Riaz 2017": y_riaz,
    }

    models = ["LR", "RF", "XGB"]
    records = []

    for m in models:
        print(f"\n--- Running LOCO CV for Model: {m.upper()} ---")

        sig_res = _run_loco_signatures(cohort_sig_dfs, model_type=m.lower())
        k20_res = _run_loco_feature_selection(cohort_expr_dfs, cohort_y_dfs, k_features=20, model_type=m.lower())
        k100_res = _run_loco_feature_selection(cohort_expr_dfs, cohort_y_dfs, k_features=100, model_type=m.lower())
        k200_res = _run_loco_feature_selection(cohort_expr_dfs, cohort_y_dfs, k_features=200, model_type=m.lower())

        for test_cohort in ["Liu 2019", "Hugo 2016", "Riaz 2017"]:
            records.append({
                "Model": m,
                "Test Cohort": test_cohort,
                "Curated Signatures AUC": sig_res[test_cohort],
                "SelectKBest (k=20) AUC": k20_res[test_cohort][0],
                "SelectKBest (k=100) AUC": k100_res[test_cohort][0],
                "SelectKBest (k=200) AUC": k200_res[test_cohort][0],
            })

    df_results = pd.DataFrame(records)
    print("\nSummary of Cross-Cohort LOCO ROC-AUC Performance:")
    print(df_results.to_string(index=False))

    out_plot_path = PLOT_DIR / "signature_vs_raw_selection_auc.png"
    _plot_comparison_results(df_results, out_plot_path)

    print("==================================================")
    print("Done!")
    print("==================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(BASE_DIR).as_posix()}")
            main()
