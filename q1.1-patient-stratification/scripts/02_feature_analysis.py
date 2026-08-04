#!/usr/bin/env python3
"""Script 02: Deep feature interpretation and association testing for Q5 Phase 2.

Performs univariate continuous (Mann-Whitney U, Cohen's d) and categorical (Fisher's exact) association testing,
calculates Youden optimal decision cutoffs, fits logistic regression interaction terms (TIS x BRAF mutation),
and exports publication-quality 300 DPI plots and statistical summary CSVs.
"""

import contextlib
import warnings
import matplotlib.patches as mpatches
from pathlib import Path
import sys
from typing import Any, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SUBPROJECT_ROOT = SCRIPT_DIR.parent
for parent in [SCRIPT_DIR] + list(SCRIPT_DIR.parents):
    if (parent / "data").is_dir() and (parent / "src").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

if str(SUBPROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from src.biology_constants import (
    CLINICAL_EXCLUDE_COLUMNS,
    KEY_DRIVER_MUTATIONS,
    KEY_IMMUNE_FEATURES,
)
from src.styles import (
    DARK_SLATE_CHARCOAL,
    OKABE_ITO,
    PHENOTYPE_PALETTE,
    RESPONSE_PALETTE,
    get_okabe_ito_diverging_cmap,
    set_presentation_style,
)
from src.utils.io import safe_save_csv
from src.utils.logging import TeeStream
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, rel_path
from src.utils.plotting import save_fig

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "02_feature_analysis.log"
FEATURE_MATRIX_FILE = PROCESSED_DIR / "q5" / "feature_matrix.csv"
OUTPUT_DIR = PROCESSED_DIR / "q5"
PLOTS_DIR = SUBPROJECT_ROOT / "plots" / "feature_analysis"

TOP_FEATURES_COUNT = 7
EFFECT_SIZE_CUTOFF = 0.20
MIN_MUTANT_SAMPLES = 3
HEATMAP_VMIN = -0.8
HEATMAP_VMAX = 0.8
LOG_P_SIG_THRESHOLD = 1.301  # -log10(0.05)

# Minimum per-group patient count required for Mann-Whitney U validity
MIN_GROUP_SAMPLES: int = 5

# Numerical floor added to standard deviation to prevent division-by-zero during z-scoring
STD_EPSILON: float = 1e-8

# Convergence iteration cap for statsmodels Logit interaction model
LOGIT_MAX_ITER: int = 100

# BT.601 ITU-R standard luminance coefficients (R, G, B channel weights)
BT601_LUMA_R: float = 0.299
BT601_LUMA_G: float = 0.587
BT601_LUMA_B: float = 0.114

# Perceived-brightness threshold below which white annotation text is used (vs black)
LUMA_DARK_THRESHOLD: float = 0.55

# Minimum p-value floor to prevent log10(0) in -log10(p) transforms across all association tests
P_VALUE_FLOOR: float = 1e-15

# Vertical offsets for the two annotation text lines within each heatmap cell (fraction of cell height)
CELL_BETA_Y_OFFSET: float = 0.36   # Top line: β interaction coefficient label
CELL_PVAL_Y_OFFSET: float = 0.68   # Bottom line: p-value label

set_presentation_style()


# ---------------------------------------------------------------------------
# Statistical Association Helpers
# ---------------------------------------------------------------------------

def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cohen's d effect size between two independent groups."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def run_univariate_associations(df: pd.DataFrame) -> pd.DataFrame:
    """Run Mann-Whitney U tests and Cohen's d for continuous features against RESPONSE_BINARY."""
    results = []
    valid_df = df.dropna(subset=["RESPONSE_BINARY"]).copy()
    resp = valid_df[valid_df["RESPONSE_BINARY"] == 1]
    non_resp = valid_df[valid_df["RESPONSE_BINARY"] == 0]

    numeric_cols = valid_df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in CLINICAL_EXCLUDE_COLUMNS]

    for col in feature_cols:
        r_vals = resp[col].dropna().values
        nr_vals = non_resp[col].dropna().values

        if len(r_vals) < MIN_GROUP_SAMPLES or len(nr_vals) < MIN_GROUP_SAMPLES:
            continue

        stat, pval = stats.mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
        d_val = compute_cohens_d(r_vals, nr_vals)

        results.append({
            "Feature": col,
            "Responder_Mean": float(np.mean(r_vals)),
            "Non_Responder_Mean": float(np.mean(nr_vals)),
            "Mann_Whitney_U": float(stat),
            "p_value": float(pval),
            "neg_log10_p": float(-np.log10(max(pval, P_VALUE_FLOOR))),
            "Cohens_d": d_val,
        })

    res_df = pd.DataFrame(results).sort_values("p_value")
    return res_df


def compute_youden_cutoffs(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Calculate Youden's J statistic optimal threshold cutoffs for key continuous biomarkers."""
    cutoffs = []
    valid_df = df.dropna(subset=["RESPONSE_BINARY"]).copy()
    y_true = valid_df["RESPONSE_BINARY"].values

    for col in feature_cols:
        if col not in valid_df.columns:
            continue
        scores = valid_df[col].values
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)

        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
        best_j = j_scores[best_idx]
        best_sens = tpr[best_idx]
        best_spec = 1.0 - fpr[best_idx]

        cutoffs.append({
            "Feature": col,
            "Optimal_Threshold": float(best_thresh),
            "Youden_J": float(best_j),
            "Sensitivity": float(best_sens),
            "Specificity": float(best_spec),
            "AUC_ROC": float(roc_auc),
        })

    return pd.DataFrame(cutoffs)


# ---------------------------------------------------------------------------
# Interaction Matrix Computation
# ---------------------------------------------------------------------------

def compute_interaction_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute logistic regression interaction p-values across all immune feature x driver mutation pairs."""
    valid_df = df.dropna(subset=["RESPONSE_BINARY"]).copy()
    y = valid_df["RESPONSE_BINARY"].values

    immune_feats = [c for c in KEY_IMMUNE_FEATURES if c in valid_df.columns]
    driver_cols = [c for c in KEY_DRIVER_MUTATIONS if c in valid_df.columns]

    results = []

    for imm in immune_feats:
        for drv in driver_cols:
            x_imm = valid_df[imm].values
            x_drv = valid_df[drv].values

            if np.sum(x_drv == 1) < MIN_MUTANT_SAMPLES:
                continue

            std_imm = (x_imm - np.mean(x_imm)) / (np.std(x_imm) + STD_EPSILON)
            interaction = std_imm * x_drv

            X = np.column_stack([np.ones(len(y)), std_imm, x_drv, interaction])
            try:
                model = sm.Logit(y, X).fit(disp=False, maxiter=LOGIT_MAX_ITER)
                beta_inter = float(model.params[3])
                pval_inter = float(model.pvalues[3])
            except (np.linalg.LinAlgError, ValueError) as err:
                warnings.warn(
                    f"Logistic regression failed for {imm} × {drv}: {err}. "
                    "Setting beta=0.0, p=1.0 as fallback.",
                    UserWarning,
                    stacklevel=2,
                )
                beta_inter, pval_inter = 0.0, 1.0

            results.append({
                "Immune_Feature": imm,
                "Driver_Mutation": drv,
                "Beta_Interaction": beta_inter,
                "p_value": pval_inter,
                "neg_log10_p": float(-np.log10(max(pval_inter, P_VALUE_FLOOR))),
            })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Visualization & Plotting Helpers
# ---------------------------------------------------------------------------

def plot_volcano(df_assoc: pd.DataFrame, save_path: Path) -> None:
    """Generate 300 DPI ranked feature effect size horizontal bar plot."""
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    df_sorted = df_assoc.sort_values("Cohens_d", ascending=True).copy()
    top_pos = df_sorted.tail(TOP_FEATURES_COUNT)
    top_neg = df_sorted.head(TOP_FEATURES_COUNT)
    df_plot = pd.concat([top_neg, top_pos]).drop_duplicates()

    df_plot["Colour"] = np.where(df_plot["Cohens_d"] > 0, RESPONSE_PALETTE["CR/PR"], RESPONSE_PALETTE["PD"])
    ax.barh(df_plot["Feature"], df_plot["Cohens_d"], color=df_plot["Colour"], edgecolor=DARK_SLATE_CHARCOAL, linewidth=1.0, alpha=0.85)

    ax.axvline(0, color=DARK_SLATE_CHARCOAL, linestyle="-", linewidth=1.2, alpha=0.7)
    ax.axvline(EFFECT_SIZE_CUTOFF, color=RESPONSE_PALETTE["CR/PR"], linestyle="--", linewidth=1.2, alpha=0.6)
    ax.axvline(-EFFECT_SIZE_CUTOFF, color=RESPONSE_PALETTE["PD"], linestyle="--", linewidth=1.2, alpha=0.6)

    resp_patch = mpatches.Patch(color=RESPONSE_PALETTE["CR/PR"], label="Enriched in Responders (d > 0)")
    non_resp_patch = mpatches.Patch(color=RESPONSE_PALETTE["PD"], label="Enriched in Non-Responders (d < 0)")
    cutoff_line = plt.Line2D([0], [0], color=DARK_SLATE_CHARCOAL, linestyle="--", linewidth=1.2, alpha=0.6, label=f"Small Effect Cutoff (|d| = {EFFECT_SIZE_CUTOFF:.2f})")

    ax.legend(
        handles=[resp_patch, non_resp_patch, cutoff_line],
        title="Biomarker Association",
        loc="lower right",
        frameon=True,
    )

    ax.set_title(
        "Ranked Biomarker Association Effect Sizes (Cohen's d: Responders vs Non-Responders)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Cohen's d Effect Size", fontsize=12, fontweight="bold")
    ax.set_ylabel("Engineered Biomarker Feature", fontsize=12, fontweight="bold")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved ranked feature association bar plot to {rel_path(save_path)}")


def plot_youden_roc(df: pd.DataFrame, df_cutoffs: pd.DataFrame, save_path: Path) -> None:
    """Generate 300 DPI ROC curves with marked Youden optimal cutoffs."""
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    colors = OKABE_ITO
    valid_df = df.dropna(subset=["RESPONSE_BINARY"]).copy()
    y_true = valid_df["RESPONSE_BINARY"].values

    for idx, row in df_cutoffs.iterrows():
        col = row["Feature"]
        if col not in valid_df.columns:
            continue
        scores = valid_df[col].values
        fpr, tpr, _ = roc_curve(y_true, scores)
        color = colors[idx % len(colors)]

        ax.plot(fpr, tpr, label=f"`{col}` (AUC = {row['AUC_ROC']:.3f})", color=color, linewidth=2.0)

        best_fpr = 1.0 - row["Specificity"]
        best_tpr = row["Sensitivity"]
        ax.scatter([best_fpr], [best_tpr], color=RESPONSE_PALETTE["PD"], s=100, zorder=5)
        label_color = "#000000" if color.upper() == "#F0E442" else color
        ax.text(
            best_fpr - 0.02,
            best_tpr + 0.03,
            f"Youden Cutoff = {row['Optimal_Threshold']:.2f}",
            fontsize=8,
            fontweight="bold",
            color=label_color,
            ha="right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=label_color, alpha=0.85),
        )

    ax.plot([0, 1], [0, 1], color=DARK_SLATE_CHARCOAL, linestyle="--", linewidth=1.2, label="Chance Baseline (AUC = 0.50)")

    n_patients = len(valid_df)
    ax.set_title(
        f"Receiver Operating Characteristic (ROC) & Youden Optimal Cutoffs (N={n_patients})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved Youden ROC plot to {rel_path(save_path)}")


def _prepare_genomic_interaction_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Prepare grouped response rate data for the TIS x `BRAF` interaction bar plot.

    Median-splits TIS into High/Low, resolves the `BRAF` mutation column name from
    common aliases, then computes per-group objective response percentages.

    Args:
        df: Feature matrix containing RESPONSE_BINARY, TIS, and optional `BRAF` mutation columns.

    Returns:
        Tuple of (grouped DataFrame with BRAF_Status, TIS_Status, Response_Pct columns; n_patients int).
    """
    valid_df = df.dropna(subset=["RESPONSE_BINARY", "TIS"]).copy()

    tis_med = valid_df["TIS"].median()
    valid_df["TIS_Status"] = np.where(valid_df["TIS"] >= tis_med, "TIS High", "TIS Low")

    braf_col = None
    for c in ["mut_BRAF", "BRAF_mut", "BRAF"]:
        if c in valid_df.columns:
            braf_col = c
            break

    if braf_col is None:
        valid_df["BRAF_Status"] = "BRAF Wild-Type"
    else:
        valid_df["BRAF_Status"] = np.where(valid_df[braf_col] == 1, "BRAF Mutated", "BRAF Wild-Type")

    grouped = (
        valid_df.groupby(["BRAF_Status", "TIS_Status"])["RESPONSE_BINARY"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped["Response_Pct"] = grouped["mean"] * 100
    return grouped, len(valid_df)


def plot_genomic_interaction(df: pd.DataFrame, save_path: Path) -> None:
    """Generate 300 DPI bar plot demonstrating response rates across TIS High/Low and `BRAF` mutation status."""
    grouped, n_patients = _prepare_genomic_interaction_data(df)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    palette = {"TIS High": RESPONSE_PALETTE["CR/PR"], "TIS Low": PHENOTYPE_PALETTE["Immune Cold"]}

    sns.barplot(
        data=grouped,
        x="BRAF_Status",
        y="Response_Pct",
        hue="TIS_Status",
        palette=palette,
        ax=ax,
        edgecolor=DARK_SLATE_CHARCOAL,
        linewidth=1.2,
    )

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1f}%",
                (p.get_x() + p.get_width() / 2.0, height / 2.0),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
            )

    ax.set_title(
        f"Genomic Synergy: Anti-PD-1 Response Rate by `BRAF` Status & TIS Level (N={n_patients})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Genomic Driver Subtype", fontsize=12, fontweight="bold")
    ax.set_ylabel("Objective Response Rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(title="TIS Phenotype", loc="upper right", frameon=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved genomic interaction plot to {rel_path(save_path)}")


def _annotate_heatmap_cells(ax: plt.Axes, piv_beta: pd.DataFrame, piv_raw_p: pd.DataFrame, okabe_cmap: Any) -> None:
    """Render two-line cell text annotations (Beta coefficient and p-value)."""
    for i in range(piv_beta.shape[0]):
        for j in range(piv_beta.shape[1]):
            val_b = piv_beta.iloc[i, j]
            val_pval = piv_raw_p.iloc[i, j]

            norm_val = np.clip((val_b - HEATMAP_VMIN) / (HEATMAP_VMAX - HEATMAP_VMIN), 0, 1)
            rgba = okabe_cmap(norm_val)
            luminance = BT601_LUMA_R * rgba[0] + BT601_LUMA_G * rgba[1] + BT601_LUMA_B * rgba[2]
            text_color = "white" if luminance < LUMA_DARK_THRESHOLD else "black"

            ax.text(j + 0.5, i + CELL_BETA_Y_OFFSET, f"β = {val_b:+.2f}", ha="center", va="center", fontsize=9.5, fontweight="bold", color=text_color)
            ax.text(j + 0.5, i + CELL_PVAL_Y_OFFSET, f"p = {val_pval:.3f}", ha="center", va="center", fontsize=7.5, fontweight="normal", color=text_color, alpha=0.9)


def _add_significance_borders(ax: plt.Axes, piv_beta: pd.DataFrame, piv_p: pd.DataFrame) -> None:
    """Highlight statistically significant interaction cells (p < 0.05) with white borders."""
    for i in range(piv_beta.shape[0]):
        for j in range(piv_beta.shape[1]):
            # piv_p holds -log10(p); >= threshold means the interaction IS significant
            is_significant = piv_p.iloc[i, j] >= LOG_P_SIG_THRESHOLD
            if is_significant:
                rect = mpatches.Rectangle((j, i), 1, 1, fill=False, edgecolor="#FFFFFF", lw=3.0, zorder=10)
                ax.add_patch(rect)

    sig_patch = mpatches.Patch(facecolor="none", edgecolor="#FFFFFF", linewidth=2.5, label="Statistically Significant (p < 0.05)")
    ax.legend(
        handles=[sig_patch],
        loc="upper right",
        bbox_to_anchor=(1.0, -0.16),
        frameon=True,
        facecolor=DARK_SLATE_CHARCOAL,
        labelcolor="white",
        fontsize=9,
    )


def plot_interaction_heatmap(df_inter: pd.DataFrame, save_path: Path) -> None:
    """Generate 300 DPI annotated heatmap of genomic x immune interaction effect sizes."""
    if df_inter.empty:
        return

    piv_beta = df_inter.pivot(index="Immune_Feature", columns="Driver_Mutation", values="Beta_Interaction")
    piv_p = df_inter.pivot(index="Immune_Feature", columns="Driver_Mutation", values="neg_log10_p")
    piv_raw_p = df_inter.pivot(index="Immune_Feature", columns="Driver_Mutation", values="p_value")

    rename_map = {"mut_BRAF": "BRAF Mutated", "mut_NRAS": "NRAS Mutated", "mut_NF1": "NF1 Mutated"}
    piv_beta = piv_beta.rename(columns=rename_map)
    piv_p = piv_p.rename(columns=rename_map)
    piv_raw_p = piv_raw_p.rename(columns=rename_map)

    fig, ax = plt.subplots(figsize=(9, 6.8), dpi=300)
    okabe_cmap = get_okabe_ito_diverging_cmap()

    sns.heatmap(
        piv_beta,
        annot=False,
        cmap=okabe_cmap,
        center=0,
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        cbar_kws={"label": "Interaction Coefficient (\\beta_{\\text{interaction}})"},
        ax=ax,
        linewidths=0.5,
        linecolor="black",
    )

    _annotate_heatmap_cells(ax, piv_beta, piv_raw_p, okabe_cmap)
    _add_significance_borders(ax, piv_beta, piv_p)

    ax.set_title(
        "Genomic x Immune Interaction Matrix (Logistic Regression: $\\beta_{interaction}$)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Driver Mutation Subtype", fontsize=12, fontweight="bold")
    ax.set_ylabel("Immune Microenvironment Feature", fontsize=12, fontweight="bold")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved interaction matrix heatmap to {rel_path(save_path)}")


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Main execution function for Q5 Phase 2 Feature Analysis."""
    print(f"Starting Q5 Phase 2 Feature Analysis (Project root: {rel_path(PROJECT_ROOT)})")

    if not FEATURE_MATRIX_FILE.exists():
        raise FileNotFoundError(f"Feature matrix file not found at {rel_path(FEATURE_MATRIX_FILE)}. Run 01_load_and_prepare.py first.")

    df_feat = pd.read_csv(FEATURE_MATRIX_FILE)
    print(f"Loaded feature matrix: {len(df_feat)} patients x {df_feat.shape[1]} features.")

    df_assoc = run_univariate_associations(df_feat)
    assoc_file = OUTPUT_DIR / "univariate_feature_associations.csv"
    safe_save_csv(df_assoc, assoc_file)
    print(f"Saved univariate association results to {rel_path(assoc_file)}")

    # Use the centralised biology constant to prevent list drift vs KEY_IMMUNE_FEATURES
    target_feats = [c for c in KEY_IMMUNE_FEATURES if c in df_feat.columns]
    df_cutoffs = compute_youden_cutoffs(df_feat, target_feats)
    cutoffs_file = OUTPUT_DIR / "youden_cutoffs.csv"
    safe_save_csv(df_cutoffs, cutoffs_file)
    print(f"Saved Youden cutoffs to {rel_path(cutoffs_file)}")

    df_inter = compute_interaction_matrix(df_feat)
    inter_file = OUTPUT_DIR / "genomic_immune_interactions.csv"
    safe_save_csv(df_inter, inter_file)
    print(f"Saved interaction matrix to {rel_path(inter_file)}")

    plot_volcano(df_assoc, PLOTS_DIR / "biomarker_volcano_plot.png")
    plot_youden_roc(df_feat, df_cutoffs, PLOTS_DIR / "youden_roc_curves.png")
    plot_genomic_interaction(df_feat, PLOTS_DIR / "genomic_interaction_tis_braf.png")
    plot_interaction_heatmap(df_inter, PLOTS_DIR / "genomic_immune_interaction_matrix.png")

    print("=" * 80)
    print("PHASE 2 FEATURE ANALYSIS COMPLETE")
    print(f"Association CSV: {rel_path(assoc_file)}")
    print(f"Youden Cutoffs CSV: {rel_path(cutoffs_file)}")
    print(f"Interaction Matrix CSV: {rel_path(inter_file)}")
    print(f"Plots Directory: {rel_path(PLOTS_DIR)}")
    print("=" * 80)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
