"""
Exploratory Immune Signature Visualisations and Multivariate Modeling.

Generates box+jitter plots of signature distributions by response, Spearman rank correlation matrix
of continuous signatures, and a multivariate Logistic Regression forest plot of signature Odds Ratios.
"""

import contextlib
from pathlib import Path
import sys
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycombat import Combat
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.signatures import extract_all_signatures
from src.styles import RESPONSE_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import find_project_root
from src.utils.plotting import save_fig

# Module-level Constants
DATA_DIR = find_project_root(Path(__file__).resolve()) / "data"
SIG_PLOT_DIR = find_project_root(Path(__file__).resolve()) / "plots" / "signatures"
LOG_DIR = find_project_root(Path(__file__).resolve()) / "logs"
LOG_PATH = LOG_DIR / "run_extra_plots.log"


def _prepare_signatures(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads cohorts, extracts signatures, batch corrects via PyComBat, and returns clean data.

    Args:
        data_dir: Path to project data directory.

    Returns:
        Tuple of (batch-corrected signature matrix, combined response series).
    """
    expr_liu, clin_liu = load_liu_2019(data_dir)
    expr_hugo, clin_hugo = load_hugo_2016(data_dir)
    expr_riaz, clin_riaz = load_riaz_2017(data_dir)

    common_genes = expr_liu.columns.intersection(expr_hugo.columns).intersection(expr_riaz.columns)

    sig_liu = extract_all_signatures(expr_liu[common_genes])
    sig_hugo = extract_all_signatures(expr_hugo[common_genes])
    sig_riaz = extract_all_signatures(expr_riaz[common_genes])

    y_liu = clin_liu.loc[sig_liu.index, "response"]
    y_hugo = clin_hugo.loc[sig_hugo.index, "response"]
    y_riaz = clin_riaz.loc[sig_riaz.index, "response"]

    non_nan_liu = y_liu.dropna().index
    non_nan_hugo = y_hugo.dropna().index
    non_nan_riaz = y_riaz.dropna().index

    sig_liu = sig_liu.loc[non_nan_liu]
    y_liu = y_liu.loc[non_nan_liu]

    sig_hugo = sig_hugo.loc[non_nan_hugo]
    y_hugo = y_hugo.loc[non_nan_hugo]

    sig_riaz = sig_riaz.loc[non_nan_riaz]
    y_riaz = y_riaz.loc[non_nan_riaz]

    sig_all = pd.concat([sig_liu, sig_hugo, sig_riaz], axis=0)
    y_all = pd.concat([y_liu, y_hugo, y_riaz], axis=0)
    batches = (["liu"] * len(sig_liu)) + (["hugo"] * len(sig_hugo)) + (["riaz"] * len(sig_riaz))

    sig_corrected_arr = Combat().fit_transform(sig_all.values, batches)
    sig_corrected = pd.DataFrame(sig_corrected_arr, index=sig_all.index, columns=sig_all.columns)

    return sig_corrected, y_all


def _plot_signature_violins(sig_corrected: pd.DataFrame, y_all: pd.Series, plot_dir: Path) -> None:
    """Plots signature box+jitter distributions stratified by response status.

    Args:
        sig_corrected: Signature DataFrame.
        y_all: Response series.
        plot_dir: Path to export output plot figure artifact.
    """
    df_plot = sig_corrected.copy()
    df_plot["Response"] = y_all.map({1.0: "Responder (CR/PR)", 0.0: "Non-Responder (PD)"})

    df_melt = pd.melt(df_plot, id_vars=["Response"], var_name="Signature", value_name="Score")

    set_presentation_style()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))

    palette = {"Responder (CR/PR)": RESPONSE_PALETTE["CR/PR"], "Non-Responder (PD)": RESPONSE_PALETTE["PD"]}

    sns.boxplot(
        data=df_melt,
        x="Signature",
        y="Score",
        hue="Response",
        palette=palette,
        ax=ax,
        width=0.5,
        boxprops=dict(alpha=0.8),
    )
    sns.stripplot(
        data=df_melt,
        x="Signature",
        y="Score",
        hue="Response",
        palette=palette,
        dodge=True,
        jitter=0.2,
        size=4,
        alpha=0.6,
        ax=ax,
    )

    signatures = sig_corrected.columns
    for i, sig in enumerate(signatures):
        r_vals = df_plot[df_plot["Response"] == "Responder (CR/PR)"][sig]
        nr_vals = df_plot[df_plot["Response"] == "Non-Responder (PD)"][sig]
        _, p_val = mannwhitneyu(r_vals, nr_vals, alternative="two-sided")

        p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else f"p = {p_val:.2e}"
        max_y = df_melt[df_melt["Signature"] == sig]["Score"].max()
        ax.text(i, max_y + 0.5, p_str, ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title(
        "Immune Signature Score Distributions by Immunotherapy Response\n(Pooled & Batch-Corrected Trial Cohorts, N=162)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Immune Signature", fontsize=12, fontweight="bold")
    ax.set_ylabel("Signature Score", fontsize=12, fontweight="bold")
    plt.xticks(rotation=15, ha="right", fontsize=11)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="upper right", framealpha=0.9)

    out_path = plot_dir / "signature_box_jitter_by_response.png"
    save_fig(fig, out_path)
    print(f"Saved signature distribution plot to {out_path.relative_to(BASE_DIR).as_posix()}")


def _plot_correlation_heatmap(sig_corrected: pd.DataFrame, plot_dir: Path) -> None:
    """Plots Spearman rank correlation matrix of continuous immune signatures.

    Args:
        sig_corrected: Signature DataFrame.
        plot_dir: Directory path to export plot artifact.
    """
    set_presentation_style()
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(8, 7))

    corr = sig_corrected.corr(method="spearman")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Spearman r"})

    ax.set_title("Spearman Correlation between Immune Signatures", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")

    out_path = plot_dir / "signature_correlation_heatmap.png"
    save_fig(fig, out_path)
    print(f"Saved correlation heatmap to {out_path.relative_to(BASE_DIR).as_posix()}")


def _plot_multivariate_forest(sig_corrected: pd.DataFrame, y_all: pd.Series, plot_dir: Path) -> None:
    """Fits multivariate logistic regression on Z-scored signatures and renders Forest Plot.

    Args:
        sig_corrected: Signature DataFrame.
        y_all: Response label series.
        plot_dir: Directory path to export plot artifact.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sig_corrected)

    clf = LogisticRegression(C=1.0, random_state=42)
    clf.fit(X_scaled, y_all)

    coefs = clf.coef_[0]
    ors = np.exp(coefs)

    pred_probs = clf.predict_proba(X_scaled)[:, 1]
    V = pred_probs * (1 - pred_probs)
    X_design = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
    cov_mat = np.linalg.inv(np.dot(X_design.T * V, X_design))
    se_coefs = np.sqrt(np.diag(cov_mat))[1:]

    ci_lower = np.exp(coefs - 1.96 * se_coefs)
    ci_upper = np.exp(coefs + 1.96 * se_coefs)

    df_or = pd.DataFrame({
        "Signature": sig_corrected.columns,
        "OR": ors,
        "CI_lower": ci_lower,
        "CI_upper": ci_upper,
    }).sort_values(by="OR", ascending=True)

    set_presentation_style()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    y_pos = np.arange(len(df_or))

    for i, (_, row) in enumerate(df_or.iterrows()):
        color = RESPONSE_PALETTE["CR/PR"] if row["OR"] > 1 else RESPONSE_PALETTE["PD"]
        ax.errorbar(
            row["OR"],
            i,
            xerr=[[row["OR"] - row["CI_lower"]], [row["CI_upper"] - row["OR"]]],
            fmt="o",
            color="black",
            ecolor=color,
            elinewidth=2.5,
            capsize=5,
            capthick=2,
            markersize=8,
        )

    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_or["Signature"], fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("Odds Ratio (95% CI, Log Scale)", fontsize=12, fontweight="bold")
    ax.set_title("Multivariate Logistic Regression: Signature Odds Ratios for Response", fontsize=13, fontweight="bold", pad=15)

    for i, (_, row) in enumerate(df_or.iterrows()):
        ax.text(
            row["CI_upper"] * 1.15,
            i,
            f"OR = {row['OR']:.2f} ({row['CI_lower']:.2f}-{row['CI_upper']:.2f})",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

    out_path = plot_dir / "forest_plot_odds_ratios.png"
    save_fig(fig, out_path)
    print(f"Saved forest plot to {out_path.relative_to(BASE_DIR).as_posix()}")


def main() -> None:
    """Executes the exploratory signature visualisations pipeline."""
    print("==================================================")
    print("Phase 1: Loading & Batch-Correcting Cohort Signatures...")
    print("==================================================")

    SIG_PLOT_DIR.mkdir(exist_ok=True, parents=True)

    sig_corrected, y_all = _prepare_signatures(DATA_DIR)

    print("\n==================================================")
    print("Phase 2: Generating Signature Distribution Violins & Jitter Plots...")
    print("==================================================")

    _plot_signature_violins(sig_corrected, y_all, SIG_PLOT_DIR)

    print("\n==================================================")
    print("Phase 3: Generating Correlation Heatmap...")
    print("==================================================")

    _plot_correlation_heatmap(sig_corrected, SIG_PLOT_DIR)

    print("\n==================================================")
    print("Phase 4: Multivariate Logistic Regression & Forest Plot...")
    print("==================================================")

    _plot_multivariate_forest(sig_corrected, y_all, SIG_PLOT_DIR)

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
