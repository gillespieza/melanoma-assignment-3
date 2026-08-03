"""
Cross-Cohort Validation: Curated Multimodal Features vs. SelectKBest Feature Selection.

Evaluates 5 predictive model architectures (Logistic Regression, Random Forest, XGBoost, Support Vector Machine,
and Elastic Net) under Leave-One-Cohort-Out (LOCO) cross-validation, comparing 12 domain-driven curated features
(8 transcriptomic immune signatures + 3 driver mutation flags + TMB) against data-driven SelectKBest feature
selection (k=20, k=100, k=200), exporting performance summaries and grouped bar charts.
"""

import warnings
warnings.filterwarnings("ignore")

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
from sklearn.svm import SVC
from xgboost import XGBClassifier
import seaborn as sns

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.signatures import extract_all_signatures
from src.styles import COHORT_PALETTE, FEATURE_SELECTION_PALETTE, RESPONSE_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, PLOTS_DIR, REPORTS_DIR, SUBPROJECT_ROOT
from src.utils.plotting import save_fig

set_presentation_style()

# Module-level Constants
PLOT_DIR = PLOTS_DIR / "feature_selection"
LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "run_comparison.log"


def _run_loco_signatures(cohort_dfs: Dict[str, Tuple[pd.DataFrame, pd.Series]], model_type: str = "rf") -> Dict[str, float]:
    """Runs LOCO cross-validation using the 12 curated multimodal features.

    Features include 8 transcriptomic immune signatures (IFN-gamma, TIS, CYT, CD8 T-cell,
    IMPRES, PD-L1, Macrophage STV Score, M1/M2 Ratio), 3 somatic driver mutation flags
    (BRAF, NRAS, NF1), and TMB (nonsynonymous mutational burden).

    Args:
        cohort_dfs: Dictionary mapping cohort names to (signatures_df, response_series).
        model_type: Classifier choice ('lr', 'rf', 'xgb', 'svm', or 'elasticnet').

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
        elif model_type == "svm":
            model = SVC(probability=True, kernel="rbf", C=1.0, random_state=42)
        elif model_type == "elasticnet":
            model = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=2000, random_state=42)

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
        model_type: Classifier choice ('lr', 'rf', 'xgb', 'svm', or 'elasticnet').

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
        elif model_type == "svm":
            model = SVC(probability=True, kernel="rbf", C=1.0, random_state=42)
        elif model_type == "elasticnet":
            model = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=2000, random_state=42)

        model.fit(X_train_sel, y_train)
        y_prob = model.predict_proba(X_test_sel)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = np.nan

        results[test_cohort] = (auc, list(selected_genes))

    return results


def _plot_summary_curated_wins(df_results: pd.DataFrame, out_plot_path: Path) -> None:
    """Generates a two-panel summary figure emphasising curated signature superiority.

    Panel A — Strip plot: each dot represents one (model × cohort) LOCO AUC observation,
    with a diamond marker showing the cross-model mean per feature representation method.
    The curated column is lightly shaded to guide the eye to the key comparison.

    Panel B — Δ AUC bar chart: mean Δ ROC-AUC (Curated Signatures minus mean SelectKBest
    AUC across k=20/100/200) per held-out cohort, averaged across all 5 model families.
    Individual model deltas are shown as white dot overlays. Positive bars confirm that
    curated domain-driven feature engineering outperforms data-driven raw gene selection.

    Args:
        df_results: Results DataFrame with LOCO AUC scores per model and test cohort.
        out_plot_path: Destination path for figure output artifact.
    """
    from matplotlib.lines import Line2D

    value_cols = [
        "Curated Signatures AUC",
        "SelectKBest (k=20) AUC",
        "SelectKBest (k=100) AUC",
        "SelectKBest (k=200) AUC",
    ]
    method_labels = [c.replace(" AUC", "") for c in value_cols]
    cohort_order = ["Liu 2019", "Hugo 2016", "Riaz 2017"]
    skb_cols = ["SelectKBest (k=20) AUC", "SelectKBest (k=100) AUC", "SelectKBest (k=200) AUC"]

    fig, (ax_strip, ax_delta) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # -----------------------------------------------------------------------
    # Panel A: Strip plot — AUC distribution per feature representation method
    # -----------------------------------------------------------------------
    df_melted = pd.melt(
        df_results,
        id_vars=["Model", "Test Cohort"],
        value_vars=value_cols,
        var_name="Method",
        value_name="ROC-AUC",
    )
    df_melted["Method"] = df_melted["Method"].str.replace(" AUC", "")

    sns.stripplot(
        data=df_melted,
        x="Method",
        y="ROC-AUC",
        order=method_labels,
        palette=FEATURE_SELECTION_PALETTE,
        ax=ax_strip,
        jitter=0.18,
        size=7,
        alpha=0.5,
        zorder=2,
    )

    # Diamond markers showing cross-model mean per method
    df_means = df_melted.groupby("Method")["ROC-AUC"].mean()
    for i, method in enumerate(method_labels):
        mean_val = df_means[method]
        color = FEATURE_SELECTION_PALETTE[method]
        ax_strip.scatter(
            i, mean_val,
            marker="D", s=120, color=color,
            edgecolors="black", linewidths=1.3, zorder=5,
        )
        ax_strip.annotate(
            f"{mean_val:.3f}",
            (i, mean_val),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color=color,
        )

    # Chance baseline and curated-column shading
    ax_strip.axhline(0.50, color="gray", linestyle="--", linewidth=1.3, zorder=1)
    ax_strip.axvspan(-0.48, 0.48, color="#37474F", alpha=0.06, zorder=0)

    ax_strip.set_ylim(0.25, 0.82)
    ax_strip.set_xlabel("Feature Representation", fontsize=11, fontweight="bold", labelpad=8)
    ax_strip.set_ylabel("LOCO Cross-Validated ROC-AUC", fontsize=11, fontweight="bold")
    ax_strip.set_title(
        "A   AUC Distribution Across All Model × Cohort Conditions",
        fontsize=11, fontweight="bold", loc="left", pad=10,
    )
    ax_strip.set_xticks(range(len(method_labels)))
    ax_strip.set_xticklabels(method_labels, rotation=15, ha="right", fontsize=10)

    legend_elements = [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.3, label="Chance (AUC = 0.50)"),
        Line2D(
            [0], [0], marker="D", color="w",
            markerfacecolor="gray", markeredgecolor="black",
            markersize=9, label="Cross-model mean",
        ),
    ]
    ax_strip.legend(handles=legend_elements, fontsize=9, frameon=True, loc="upper right")

    # -----------------------------------------------------------------------
    # Panel B: Δ AUC bar chart — Curated Signatures minus SelectKBest mean
    # -----------------------------------------------------------------------
    df_delta = df_results.copy()
    df_delta["SelectKBest Mean AUC"] = df_delta[skb_cols].mean(axis=1)
    df_delta["Delta AUC"] = df_delta["Curated Signatures AUC"] - df_delta["SelectKBest Mean AUC"]

    df_delta_cohort = (
        df_delta.groupby("Test Cohort")["Delta AUC"]
        .agg(["mean", "std"])
        .loc[cohort_order]
        .reset_index()
    )

    bar_colors = [COHORT_PALETTE.get(c, "#37474F") for c in df_delta_cohort["Test Cohort"]]
    bars = ax_delta.bar(
        range(len(cohort_order)),
        df_delta_cohort["mean"],
        yerr=df_delta_cohort["std"],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.9,
        capsize=5,
        width=0.5,
        error_kw={"linewidth": 1.2, "ecolor": "black"},
        zorder=3,
    )

    # Annotate mean delta values above/below each bar
    for bar, val in zip(bars, df_delta_cohort["mean"]):
        offset = 0.013 if val >= 0 else -0.013
        v_align = "bottom" if val >= 0 else "top"
        sign_str = "+" if val >= 0 else ""
        ax_delta.annotate(
            f"{sign_str}{val:.3f}",
            (bar.get_x() + bar.get_width() / 2, val + offset),
            ha="center", va=v_align,
            fontsize=10.5, fontweight="bold",
        )

    # Individual model Δ values overlaid as white dots
    for cohort in cohort_order:
        cohort_x = cohort_order.index(cohort)
        model_deltas = df_delta[df_delta["Test Cohort"] == cohort]["Delta AUC"].values
        ax_delta.scatter(
            [cohort_x] * len(model_deltas),
            model_deltas,
            color="white", edgecolors="black", linewidths=0.8,
            s=28, zorder=4, alpha=0.85,
        )

    # Positive/negative half-plane shading and zero baseline
    ax_delta.axhspan(0, 0.35, alpha=0.05, color="#009E73", zorder=0)
    ax_delta.axhspan(-0.35, 0, alpha=0.05, color="#D55E00", zorder=0)
    ax_delta.axhline(0.0, color="black", linestyle="-", linewidth=1.0, zorder=2)

    # Zone labels: curated wins vs raw genes win
    ax_delta.text(
        2.47, 0.20, "Curated\nwins ↑", ha="right", va="center",
        fontsize=8.5, color="#009E73", fontweight="bold",
    )
    ax_delta.text(
        2.47, -0.20, "Raw genes\nwin ↓", ha="right", va="center",
        fontsize=8.5, color="#D55E00", fontweight="bold",
    )

    y_abs_max = max(
        (df_delta_cohort["mean"].abs() + df_delta_cohort["std"]).max() + 0.06,
        0.20,
    )
    ax_delta.set_ylim(-y_abs_max, y_abs_max)
    ax_delta.set_xticks(range(len(cohort_order)))
    ax_delta.set_xticklabels(cohort_order, rotation=15, ha="right", fontsize=10)
    ax_delta.set_xlabel("Held-out Test Cohort", fontsize=11, fontweight="bold", labelpad=8)
    ax_delta.set_ylabel("Δ ROC-AUC (Curated − Mean SelectKBest)", fontsize=11, fontweight="bold")
    ax_delta.set_title(
        "B   Domain Signatures vs. Data-Driven Selection",
        fontsize=11, fontweight="bold", loc="left", pad=10,
    )

    plt.suptitle(
        "Feature Engineering vs. Data-Driven Selection: Out-of-Cohort LOCO Validation",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    # Save transparent copy before save_fig closes the figure
    pt_path = out_plot_path.parent / (out_plot_path.stem + "_transparent.png")
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pt_path, transparent=True, bbox_inches="tight", dpi=300)
    print(f"Saved transparent copy to {pt_path.name}")

    save_fig(fig, out_plot_path)
    print(f"Saved two-panel summary plot to {out_plot_path.name}")


def _plot_comparison_heatmap(df_results: pd.DataFrame, out_plot_path: Path) -> None:
    """Generates a divergent annotated heatmap of LOCO ROC-AUC values with model-family
    group dividers and a cross-model mean summary row.

    Rows are grouped into five model families (LR, RF, XGB, SVM, ElasticNet), each
    containing three held-out cohort rows. Thick horizontal lines visually separate
    model families. A summary row at the bottom shows the cross-model mean AUC per
    feature representation. The divergent colourmap centres on the chance baseline of 0.50
    so below-chance cells are visually distinct from above-chance cells.

    Args:
        df_results: Results DataFrame containing LOCO AUC scores.
        out_plot_path: Destination path for figure output artifact.
    """
    value_cols = [
        "Curated Signatures AUC",
        "SelectKBest (k=20) AUC",
        "SelectKBest (k=100) AUC",
        "SelectKBest (k=200) AUC",
    ]
    display_cols = [c.replace(" AUC", "") for c in value_cols]

    # Build row index as "Model · Cohort"
    df_hm = df_results.copy()
    df_hm["Row"] = df_hm["Model"] + "  ·  " + df_hm["Test Cohort"]
    df_hm = df_hm.set_index("Row")[value_cols]
    df_hm.columns = display_cols

    # Best-performing feature method per data row (used for bold outlines)
    best_col_per_row = df_hm.idxmax(axis=1)
    n_data_rows = len(df_hm)

    # Append cross-model mean summary row below all data rows
    mean_row = df_hm.mean(axis=0)
    mean_row.name = "━━  Cross-Model Mean  ━━"
    df_hm_with_mean = pd.concat([df_hm, mean_row.to_frame().T])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Divergent colourmap centred on 0.50 (chance baseline)
    sns.heatmap(
        df_hm_with_mean,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0.50,
        vmin=0.30,
        vmax=0.75,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "ROC-AUC", "shrink": 0.80},
        ax=ax,
        annot_kws={"fontsize": 10},
    )

    # Bold-outline the best cell in each data row
    for row_idx, (row_label, col_label) in enumerate(best_col_per_row.items()):
        col_idx = display_cols.index(col_label)
        ax.add_patch(plt.Rectangle(
            (col_idx, row_idx), 1, 1,
            fill=False, edgecolor="black", linewidth=2.5,
        ))

    # Thick horizontal dividers separating model families (every 3 rows)
    for divider_y in [3, 6, 9, 12]:
        ax.axhline(divider_y, color="#37474F", linewidth=2.5, zorder=6)

    # Dashed divider before the cross-model mean summary row
    ax.axhline(n_data_rows, color="#37474F", linewidth=3.0, linestyle="--", zorder=6)

    ax.set_xlabel("Feature Representation", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Model  ·  Held-out Test Cohort", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(
        "Out-of-Cohort ROC-AUC Heatmap: Feature Selection Method Comparison",
        fontsize=13, fontweight="bold", pad=14,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    save_fig(fig, out_plot_path)
    print(f"Saved comparison heatmap to {out_plot_path.name}")


def main() -> None:
    """Executes feature selection comparison pipeline."""
    print("==================================================")
    print("LOCO CV Evaluation: Curated Signatures vs SelectKBest (5 Model Families)")
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

    models = ["LR", "RF", "XGB", "SVM", "ElasticNet"]
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
    print("\nSummary of Cross-Cohort LOCO ROC-AUC Performance Across All 5 Model Families:")
    print(df_results.to_string(index=False))

    out_plot_path = PLOT_DIR / "signature_vs_raw_selection_auc.png"
    _plot_summary_curated_wins(df_results, out_plot_path)

    out_heatmap_path = PLOT_DIR / "signature_vs_raw_selection_heatmap.png"
    _plot_comparison_heatmap(df_results, out_heatmap_path)

    print("==================================================")
    print("Done!")
    print("==================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.name}")
            main()
