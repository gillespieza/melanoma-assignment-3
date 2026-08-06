"""
Exploratory Immune Signature Visualisations and Multivariate Modeling.

Generates a raincloud plot of Z-scored signature distributions by response (half-violin + box +
jitter), Spearman rank correlation matrix of continuous signatures, and a multivariate Logistic
Regression forest plot of signature Odds Ratios.
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
from scipy.stats import gaussian_kde
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
from src.utils.paths import DATA_DIR, LOG_DIR, PLOTS_DIR
from src.utils.plotting import save_fig

set_presentation_style()

# Module-level Constants
SIG_PLOT_DIR = PLOTS_DIR / "signatures"
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


# ---------------------------------------------------------------------------
# Raincloud plot helpers
# ---------------------------------------------------------------------------

_JITTER_OFFSET = 0.18   # vertical distance of jitter strip from row centre
_VIOLIN_SCALE = 0.38    # maximum half-height of the KDE violin
_BOX_WIDTH = 0.12       # half-height of the IQR box
_KDE_POINTS = 200       # resolution of KDE curve

# Signatures to include in all distribution and model plots — exactly the 6 curated
# signatures from the report table (Section 2.1), in table order.
_REPORT_SIGS = ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "IMPRES", "PD_L1"]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Cohen's d effect size between two 1-D arrays.

    Args:
        a: First sample.
        b: Second sample.

    Returns:
        Signed Cohen's d (positive when mean(a) > mean(b)).
    """
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(a) - np.mean(b)) / pooled_std) if pooled_std > 0 else 0.0


def _draw_half_violin(
    ax: plt.Axes,
    values: np.ndarray,
    y_centre: float,
    color: str,
    side: str,
    scale: float = _VIOLIN_SCALE,
) -> None:
    """Draws a KDE half-violin clipped to one side of y_centre.

    Args:
        ax: Target axes.
        values: 1-D data array.
        y_centre: The row's vertical baseline position.
        color: Fill colour for the violin.
        side: 'left' draws violin above y_centre, 'right' below.
        scale: Maximum half-width of the KDE curve.
    """
    kde = gaussian_kde(values, bw_method="scott")
    x_grid = np.linspace(values.min() - 0.5, values.max() + 0.5, _KDE_POINTS)
    density = kde(x_grid)
    # Normalise so the peak maps to `scale`
    density = density / density.max() * scale

    sign = -1.0 if side == "left" else 1.0
    # Plot is horizontal: scores on X-axis, row index on Y-axis.
    # fill_between(x_grid, y1, y2) fills between two Y curves over shared X values.
    y_upper = y_centre + sign * density
    y_lower = np.full_like(y_upper, y_centre)

    ax.fill_between(x_grid, y_lower, y_upper, color=color, alpha=0.55)
    ax.plot(x_grid, y_upper, color=color, linewidth=1.2, alpha=0.85)


def _draw_box(
    ax: plt.Axes,
    values: np.ndarray,
    y_centre: float,
    color: str,
    half_width: float = _BOX_WIDTH,
) -> None:
    """Draws a compact IQR box with median line and whiskers.

    Args:
        ax: Target axes.
        values: 1-D data array.
        y_centre: The row's vertical baseline position.
        color: Box edge and median line colour.
        half_width: Half-width of the box rectangle.
    """
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lo_whisker = max(values.min(), q1 - 1.5 * iqr)
    hi_whisker = min(values.max(), q3 + 1.5 * iqr)

    rect = plt.Rectangle(
        (q1, y_centre - half_width), iqr, 2 * half_width,
        linewidth=1.5, edgecolor=color, facecolor="white", zorder=3,
    )
    ax.add_patch(rect)
    ax.plot([median, median], [y_centre - half_width, y_centre + half_width],
            color=color, linewidth=2.2, zorder=4)
    ax.plot([lo_whisker, q1], [y_centre, y_centre], color=color, linewidth=1.2, zorder=2)
    ax.plot([q3, hi_whisker], [y_centre, y_centre], color=color, linewidth=1.2, zorder=2)


def _draw_jitter(
    ax: plt.Axes,
    values: np.ndarray,
    y_centre: float,
    color: str,
    side: str,
    rng: np.random.Generator,
    offset: float = _JITTER_OFFSET,
) -> None:
    """Draws a jittered strip of individual patient dots.

    Args:
        ax: Target axes.
        values: 1-D data array.
        y_centre: The row's vertical baseline position.
        color: Dot colour.
        side: 'left' places jitter left of y_centre, 'right' to the right.
        rng: Seeded random number generator for reproducible jitter.
        offset: Horizontal distance from y_centre.
    """
    sign = -1.0 if side == "left" else 1.0
    jitter_y = y_centre + sign * (offset + rng.uniform(0, 0.08, size=len(values)))
    ax.scatter(values, jitter_y, color=color, alpha=0.55, s=16, zorder=5, linewidths=0)


def _plot_signature_raincloud(
    sig_corrected: pd.DataFrame, y_all: pd.Series, plot_dir: Path
) -> None:
    """Plots a horizontal raincloud (half-violin + box + jitter) of Z-scored signature distributions.

    Signatures are Z-score normalised so all six occupy a single shared axis, and rows are
    ranked by Wilcoxon p-value (most significant at top). Each row is annotated with the
    Wilcoxon p-value and Cohen's d effect size.

    Args:
        sig_corrected: Batch-corrected signature DataFrame (patients × signatures).
        y_all: Binary response series (1 = responder, 0 = non-responder).
        plot_dir: Directory path to export plot artifact.
    """
    # Restrict to the 6 curated report signatures in table order
    available_sigs = [s for s in _REPORT_SIGS if s in sig_corrected.columns]
    sig_cont = sig_corrected[available_sigs]

    # Z-score normalise each signature independently so all share one axis
    sig_z = (sig_cont - sig_cont.mean()) / sig_cont.std()

    resp_mask = y_all == 1
    color_r = RESPONSE_PALETTE["CR/PR"]
    color_nr = RESPONSE_PALETTE["PD"]
    rng = np.random.default_rng(42)

    # Use fixed table order — do not sort by p-value
    ordered_sigs = available_sigs

    n_sigs = len(ordered_sigs)
    fig, ax = plt.subplots(figsize=(12, 1.6 * n_sigs + 1.4))

    for row_idx, sig in enumerate(ordered_sigs):
        y_centre = float(row_idx)
        r_vals = sig_z.loc[resp_mask, sig].dropna().values
        nr_vals = sig_z.loc[~resp_mask, sig].dropna().values
        _, p = mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
        p_values_row = {sig: p}

        # Non-responders: violin + jitter on the LEFT; Responders: on the RIGHT
        _draw_half_violin(ax, nr_vals, y_centre, color_nr, side="left")
        _draw_half_violin(ax, r_vals, y_centre, color_r, side="right")
        _draw_box(ax, np.concatenate([r_vals, nr_vals]), y_centre, color="#212B32")
        _draw_jitter(ax, nr_vals, y_centre, color_nr, side="left", rng=rng)
        _draw_jitter(ax, r_vals, y_centre, color_r, side="right", rng=rng)

        # Annotation: p-value + Cohen's d
        _, p = mannwhitneyu(
            sig_z.loc[resp_mask, sig].dropna().values,
            sig_z.loc[~resp_mask, sig].dropna().values,
            alternative="two-sided",
        )
        d = _cohens_d(r_vals, nr_vals)
        p_str = f"p = {p:.3f}" if p >= 0.001 else f"p = {p:.2e}"
        d_str = f"d = {d:+.2f}"
        # Place annotation above the violin body: with invert_yaxis, subtracting from
        # y_centre moves the text visually upward (toward the top of the axes).
        # _VIOLIN_SCALE = 0.38, so y_centre - 0.32 clears the upper violin edge.
        x_annot = sig_z[sig].max() + 0.6   # beyond the KDE grid extent (max + 0.5)
        ax.text(
            x_annot, y_centre - 0.32, f"{p_str}   {d_str}",
            va="bottom", ha="left", fontsize=9.5, fontweight="bold", color="#212B32",
        )

    n_total = len(y_all)
    ax.set_yticks(range(n_sigs))
    ax.set_yticklabels(ordered_sigs, fontsize=11, fontweight="bold")
    ax.set_xlabel("Z-Scored Signature Score", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Immune Signature Distributions by Immunotherapy Response — Raincloud Plot\n"
        f"(Pooled & Batch-Corrected Trial Cohorts, N={n_total}; ranked by Wilcoxon p)",
        fontsize=13, fontweight="bold", pad=16,
    )
    ax.axvline(0, color="#9BAAB3", linewidth=0.8, linestyle="--")
    # Extend x-axis right boundary by an extra unit so right text annotations are spacious and clear
    x_max_val = float(sig_z.values.max())
    x_min_val = float(sig_z.values.min())
    ax.set_xlim(left=x_min_val - 0.5, right=x_max_val + 1.8)

    # Table order: first entry at top row (index 0 = top with invert_yaxis)
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_r, alpha=0.7, label="Responder (CR/PR)"),
        Patch(facecolor=color_nr, alpha=0.7, label="Non-Responder (PD)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9, fontsize=10)

    plt.tight_layout()
    out_path = plot_dir / "signature_raincloud_by_response.png"
    save_fig(fig, out_path)
    print(f"Saved raincloud plot to {out_path.relative_to(BASE_DIR).as_posix()}")


def _plot_univariate_forest(
    sig_corrected: pd.DataFrame, y_all: pd.Series, plot_dir: Path
) -> None:
    """Plots a univariate effect-size forest plot (Cohen's d + 95% bootstrap CI) per signature.

    Each row shows the standardised mean difference between responders and non-responders
    with a bootstrap 95% confidence interval (2 000 resamples). Rows are ranked by effect
    size (most positive at top). A dashed vertical line marks d = 0 (no effect).

    Args:
        sig_corrected: Batch-corrected signature DataFrame (patients x signatures).
        y_all: Binary response series (1 = responder, 0 = non-responder).
        plot_dir: Directory path to export plot artifact.
    """
    _N_BOOTSTRAP = 2_000
    _ALPHA = 0.05

    available_sigs = [s for s in _REPORT_SIGS if s in sig_corrected.columns]
    sig_cont = sig_corrected[available_sigs]

    resp_mask = (y_all == 1).values
    rng = np.random.default_rng(42)

    rows: list[dict] = []
    for sig in sig_cont.columns:
        vals = sig_cont[sig].dropna().values
        mask = resp_mask[sig_cont[sig].notna().values]
        r_vals = vals[mask]
        nr_vals = vals[~mask]

        d_obs = _cohens_d(r_vals, nr_vals)

        # Percentile bootstrap CI
        boot_ds = np.empty(_N_BOOTSTRAP)
        n_r, n_nr = len(r_vals), len(nr_vals)
        for i in range(_N_BOOTSTRAP):
            boot_r = rng.choice(r_vals, size=n_r, replace=True)
            boot_nr = rng.choice(nr_vals, size=n_nr, replace=True)
            boot_ds[i] = _cohens_d(boot_r, boot_nr)

        ci_lo = float(np.percentile(boot_ds, 100 * _ALPHA / 2))
        ci_hi = float(np.percentile(boot_ds, 100 * (1 - _ALPHA / 2)))

        _, p_val = mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
        rows.append({"Signature": sig, "d": d_obs, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": p_val})

    df_forest = pd.DataFrame(rows).sort_values("d", ascending=False).reset_index(drop=True)
    # Re-apply table order after computing stats — sort to match _REPORT_SIGS order, top = first entry
    order_map = {sig: i for i, sig in enumerate(available_sigs)}
    df_forest = df_forest.sort_values(
        "Signature", key=lambda col: col.map(order_map)
    ).reset_index(drop=True)

    color_r = RESPONSE_PALETTE["CR/PR"]
    color_nr = RESPONSE_PALETTE["PD"]

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(df_forest) + 2.2))
    y_pos = np.arange(len(df_forest))

    for i, row in df_forest.iterrows():
        dot_color = color_r if row["d"] > 0 else color_nr
        ax.errorbar(
            row["d"], i,
            xerr=[[row["d"] - row["ci_lo"]], [row["ci_hi"] - row["d"]]],
            fmt="o", color=dot_color, ecolor="#455C68",
            elinewidth=2.0, capsize=4, capthick=1.8, markersize=8, zorder=3,
        )
        p_str = f"p = {row['p']:.3f}" if row["p"] >= 0.001 else f"p = {row['p']:.2e}"
        ax.text(
            row["d"], i - 0.23,
            f"d = {row['d']:+.2f}  {p_str}",
            va="bottom", ha="center", fontsize=9.0, fontweight="bold", color="#212B32",
        )

    ax.axvline(0, color="#9BAAB3", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_forest["Signature"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Cohen's d  (Responder − Non-Responder)", fontsize=12, fontweight="bold")
    n_total = len(y_all)
    ax.set_title(
        f"Univariate Effect Size: Immune Signatures vs Immunotherapy Response\n"
        f"(Pooled & Batch-Corrected Trial Cohorts, N={n_total}; 95% bootstrap CI, ranked by d)",
        fontsize=12, fontweight="bold", pad=14,
    )
    ax.invert_yaxis()  # table order: first entry at top

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_r, alpha=0.8, label="Higher in Responders (d > 0)"),
        Patch(facecolor=color_nr, alpha=0.8, label="Higher in Non-Responders (d < 0)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              bbox_to_anchor=(0.98, 0.08), framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    out_path = plot_dir / "signature_univariate_forest.png"
    save_fig(fig, out_path)
    print(f"Saved univariate forest plot to {out_path.relative_to(BASE_DIR).as_posix()}")


def _plot_correlation_heatmap(sig_corrected: pd.DataFrame, plot_dir: Path) -> None:
    """Plots Spearman rank correlation matrix of continuous immune signatures.

    Args:
        sig_corrected: Signature DataFrame.
        plot_dir: Directory path to export plot artifact.
    """
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
    # Restrict to the 6 curated report signatures in table order
    available_sigs = [s for s in _REPORT_SIGS if s in sig_corrected.columns]
    sig_cont = sig_corrected[available_sigs]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sig_cont)

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
        "Signature": sig_cont.columns,
        "OR": ors,
        "CI_lower": ci_lower,
        "CI_upper": ci_upper,
    })
    # Apply table order — do not sort by OR
    order_map = {sig: i for i, sig in enumerate(available_sigs)}
    df_or = df_or.sort_values(
        "Signature", key=lambda col: col.map(order_map)
    ).reset_index(drop=True)

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
    n_total = int(y_all.notna().sum())
    ax.set_title(
        f"Multivariate Logistic Regression: Signature Odds Ratios for Response\n"
        f"(6 Curated Signatures, Pooled & Batch-Corrected Cohorts, N={n_total})",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.invert_yaxis()  # table order: first entry at top

    for i, (_, row) in enumerate(df_or.iterrows()):
        ax.text(
            row["OR"],
            i - 0.23,
            f"OR = {row['OR']:.2f} ({row['CI_lower']:.2f}-{row['CI_upper']:.2f})",
            va="bottom",
            ha="center",
            fontsize=9.0,
            fontweight="bold",
            color="#212B32",
        )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=RESPONSE_PALETTE["CR/PR"], alpha=0.8, label="OR > 1: Associated with Response (CR/PR)"),
        Patch(facecolor=RESPONSE_PALETTE["PD"], alpha=0.8, label="OR < 1: Associated with Non-Response (PD)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9, fontsize=9.5)

    out_path = plot_dir / "forest_plot_odds_ratios.png"
    save_fig(fig, out_path)
    print(f"Saved forest plot to {out_path.relative_to(BASE_DIR).as_posix()}")


def _plot_combined_forest(sig_corrected: pd.DataFrame, y_all: pd.Series, plot_dir: Path) -> None:
    """Generates a 1x2 grid comparing Univariate Effect Sizes (Left) and Multivariate Odds Ratios (Right).

    Args:
        sig_corrected: Signature DataFrame.
        y_all: Response label series.
        plot_dir: Directory path to export plot artifact.
    """
    available_sigs = [s for s in _REPORT_SIGS if s in sig_corrected.columns]
    sig_cont = sig_corrected[available_sigs]
    resp_mask = (y_all == 1).values
    rng = np.random.default_rng(42)

    # 1. Compute Univariate Stats
    _N_BOOTSTRAP = 2_000
    _ALPHA = 0.05
    u_rows: list[dict] = []
    for sig in sig_cont.columns:
        vals = sig_cont[sig].dropna().values
        mask = resp_mask[sig_cont[sig].notna().values]
        r_vals, nr_vals = vals[mask], vals[~mask]
        d_obs = _cohens_d(r_vals, nr_vals)

        boot_ds = np.empty(_N_BOOTSTRAP)
        n_r, n_nr = len(r_vals), len(nr_vals)
        for i in range(_N_BOOTSTRAP):
            boot_r = rng.choice(r_vals, size=n_r, replace=True)
            boot_nr = rng.choice(nr_vals, size=n_nr, replace=True)
            boot_ds[i] = _cohens_d(boot_r, boot_nr)

        ci_lo = float(np.percentile(boot_ds, 100 * _ALPHA / 2))
        ci_hi = float(np.percentile(boot_ds, 100 * (1 - _ALPHA / 2)))
        _, p_val = mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
        u_rows.append({"Signature": sig, "d": d_obs, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": p_val})

    df_u = pd.DataFrame(u_rows)
    order_map = {sig: i for i, sig in enumerate(available_sigs)}
    df_u = df_u.sort_values("Signature", key=lambda col: col.map(order_map)).reset_index(drop=True)

    # 2. Compute Multivariate Stats
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sig_cont)
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

    df_m = pd.DataFrame({
        "Signature": sig_cont.columns,
        "OR": ors,
        "CI_lower": ci_lower,
        "CI_upper": ci_upper,
    }).sort_values("Signature", key=lambda col: col.map(order_map)).reset_index(drop=True)

    # Plot 1x2 Grid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    color_r = RESPONSE_PALETTE["CR/PR"]
    color_nr = RESPONSE_PALETTE["PD"]
    y_pos = np.arange(len(available_sigs))

    # --- LEFT PANEL: Univariate ---
    for i, row in df_u.iterrows():
        dot_color = color_r if row["d"] > 0 else color_nr
        ax1.errorbar(
            row["d"], i,
            xerr=[[row["d"] - row["ci_lo"]], [row["ci_hi"] - row["d"]]],
            fmt="o", color=dot_color, ecolor="#455C68",
            elinewidth=2.0, capsize=4, capthick=1.8, markersize=8, zorder=3,
        )
        p_str = f"p = {row['p']:.3f}" if row["p"] >= 0.001 else f"p = {row['p']:.2e}"
        # Position text centered above the point estimate (y = i - 0.23 with invert_yaxis)
        ax1.text(
            row["d"], i - 0.23,
            f"d = {row['d']:+.2f}  {p_str}",
            va="bottom", ha="center", fontsize=8.5, fontweight="bold", color="#212B32",
        )

    ax1.axvline(0, color="#9BAAB3", linewidth=1.0, linestyle="--", zorder=1)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_u["Signature"], fontsize=11, fontweight="bold")
    ax1.set_xlabel("Cohen's d  (Responder − Non-Responder)", fontsize=11, fontweight="bold")
    ax1.set_title("A. Univariate Effect Sizes (Cohen's d & 95% Bootstrap CI)", fontsize=12, fontweight="bold", pad=16)
    ax1.set_ylim(bottom=len(available_sigs) - 0.5, top=-0.75)
    ax1.invert_yaxis()

    # --- RIGHT PANEL: Multivariate ---
    for i, row in df_m.iterrows():
        color = color_r if row["OR"] > 1 else color_nr
        ax2.errorbar(
            row["OR"], i,
            xerr=[[row["OR"] - row["CI_lower"]], [row["CI_upper"] - row["OR"]]],
            fmt="o", color="black", ecolor=color,
            elinewidth=2.5, capsize=5, capthick=2, markersize=8,
        )
        # Position text centered above the point estimate / log center
        ax2.text(
            row["OR"], i - 0.23,
            f"OR = {row['OR']:.2f} ({row['CI_lower']:.2f}-{row['CI_upper']:.2f})",
            va="bottom", ha="center", fontsize=8.5, fontweight="bold", color="#212B32",
        )

    ax2.axvline(x=1.0, color="gray", linestyle="--", linewidth=1.2)
    ax2.set_xscale("log")
    ax2.set_xlim(0.2, 16.0)
    ax2.set_xlabel("Odds Ratio (95% CI, Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_title("B. Multivariate Logistic Regression (Odds Ratios)", fontsize=12, fontweight="bold", pad=16)
    ax2.set_ylim(bottom=len(available_sigs) - 0.5, top=-0.75)

    # Legends
    from matplotlib.patches import Patch
    leg1 = [
        Patch(facecolor=color_r, alpha=0.8, label="Higher in Responders"),
        Patch(facecolor=color_nr, alpha=0.8, label="Higher in Non-Responders"),
    ]
    # Position in upper left space (x < 0) above IFN_gamma line
    ax1.legend(handles=leg1, loc="upper left", bbox_to_anchor=(0.02, 0.96), framealpha=0.9, fontsize=8.5)

    leg2 = [
        Patch(facecolor=color_r, alpha=0.8, label="OR > 1 (Response Favoured)"),
        Patch(facecolor=color_nr, alpha=0.8, label="OR < 1 (Non-Response Favoured)"),
    ]
    ax2.legend(handles=leg2, loc="lower right", bbox_to_anchor=(0.98, 0.02), framealpha=0.9, fontsize=8.5)

    fig.suptitle(
        f"Univariate vs Multivariate Forest Comparison across 6 Curated Immune Signatures (N={len(y_all)})",
        fontsize=13, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    out_path = plot_dir / "combined_forest_plots.png"
    save_fig(fig, out_path)
    print(f"Saved combined forest plot to {out_path.relative_to(BASE_DIR).as_posix()}")


def main() -> None:
    """Executes the exploratory signature visualisations pipeline."""
    print("==================================================")
    print("Phase 1: Loading & Batch-Correcting Cohort Signatures...")
    print("==================================================")

    SIG_PLOT_DIR.mkdir(exist_ok=True, parents=True)

    sig_corrected, y_all = _prepare_signatures(DATA_DIR)

    print("\n==================================================")
    print("Phase 2: Generating Signature Raincloud Plot...")
    print("==================================================")

    _plot_signature_raincloud(sig_corrected, y_all, SIG_PLOT_DIR)

    print("\n==================================================")
    print("Phase 3: Generating Correlation Heatmap...")
    print("==================================================")

    _plot_correlation_heatmap(sig_corrected, SIG_PLOT_DIR)

    print("\n==================================================")
    print("Phase 4: Multivariate Logistic Regression & Forest Plot...")
    print("==================================================")

    _plot_multivariate_forest(sig_corrected, y_all, SIG_PLOT_DIR)

    print("\n==================================================")
    print("Phase 5: Univariate Effect-Size Forest Plot...")
    print("==================================================")

    _plot_univariate_forest(sig_corrected, y_all, SIG_PLOT_DIR)

    print("\n==================================================")
    print("Phase 6: Combined Univariate & Multivariate Forest Grid...")
    print("==================================================")

    _plot_combined_forest(sig_corrected, y_all, SIG_PLOT_DIR)

    print("==================================================")
    print("Done!")
    print("==================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(LOG_DIR.parent).as_posix()}")
            main()
