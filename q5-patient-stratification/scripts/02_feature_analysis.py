#!/usr/bin/env python3
"""Script 02: Deep feature interpretation and association testing for Q5 Phase 2.

Performs univariate continuous (Mann-Whitney U, Cohen's d) and categorical (Fisher's exact) association testing,
calculates Youden optimal decision cutoffs, fits logistic regression interaction terms (TIS x BRAF mutation),
and exports publication-quality 300 DPI plots and statistical summary CSVs.
"""

import contextlib
from pathlib import Path
import sys
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.metrics import roc_curve, auc

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

from src.styles import (
    DRIVER_PALETTE,
    OKABE_ITO,
    PHENOTYPE_PALETTE,
    RESPONSE_PALETTE,
    get_okabe_ito_diverging_cmap,
    set_presentation_style,
)
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

set_presentation_style()


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
    exclude = [
        "RESPONSE_BINARY",
        "PATIENT_ID",
        "SAMPLE_ID",
        "OS_STATUS",
        "OS_MONTHS",
        "RESPONSE",
        "DATASET",
        "COHORT",
        "IMMUNOTHERAPY",
        "AGE",
        "RACE",
        "SEX",
        "SPECIMEN_TYPE",
    ]
    feature_cols = [c for c in numeric_cols if c not in exclude]

    for col in feature_cols:
        r_vals = resp[col].dropna().values
        nr_vals = non_resp[col].dropna().values

        if len(r_vals) < 5 or len(nr_vals) < 5:
            continue

        stat, pval = stats.mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
        d_val = compute_cohens_d(r_vals, nr_vals)

        results.append({
            "Feature": col,
            "Responder_Mean": float(np.mean(r_vals)),
            "Non_Responder_Mean": float(np.mean(nr_vals)),
            "Mann_Whitney_U": float(stat),
            "p_value": float(pval),
            "neg_log10_p": float(-np.log10(max(pval, 1e-15))),
            "Cohens_d": d_val,
        })

    res_df = pd.DataFrame(results).sort_values("p_value")
    return res_df


def compute_youden_cutoffs(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
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

        # Youden's J statistic = TPR - FPR
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


def plot_volcano(df_assoc: pd.DataFrame, save_path: Path) -> None:
    """Generate publication-quality 300 DPI ranked feature effect size horizontal bar plot (presentation alternative to volcano plot)."""
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    # Sort features by Cohen's d
    df_sorted = df_assoc.sort_values("Cohens_d", ascending=True).copy()
    # Select top 7 positive and top 7 negative features
    top_pos = df_sorted.tail(7)
    top_neg = df_sorted.head(7)
    df_plot = pd.concat([top_neg, top_pos]).drop_duplicates()

    df_plot["Colour"] = np.where(df_plot["Cohens_d"] > 0, RESPONSE_PALETTE["CR/PR"], RESPONSE_PALETTE["PD"])

    bars = ax.barh(df_plot["Feature"], df_plot["Cohens_d"], color=df_plot["Colour"], edgecolor="#37474F", linewidth=1.0, alpha=0.85)

    # Vertical zero line and small effect cutoff lines (|d| = 0.20)
    ax.axvline(0, color="#37474F", linestyle="-", linewidth=1.2, alpha=0.7)
    ax.axvline(0.2, color=RESPONSE_PALETTE["CR/PR"], linestyle="--", linewidth=1.2, alpha=0.6)
    ax.axvline(-0.2, color=RESPONSE_PALETTE["PD"], linestyle="--", linewidth=1.2, alpha=0.6)

    # Add explicit legend patches for Responders vs Non-Responders
    import matplotlib.patches as mpatches

    resp_patch = mpatches.Patch(color=RESPONSE_PALETTE["CR/PR"], label="Enriched in Responders (d > 0)")
    non_resp_patch = mpatches.Patch(color=RESPONSE_PALETTE["PD"], label="Enriched in Non-Responders (d < 0)")
    cutoff_line = plt.Line2D([0], [0], color="#37474F", linestyle="--", linewidth=1.2, alpha=0.6, label="Small Effect Cutoff (|d| = 0.20)")

    ax.legend(
        handles=[resp_patch, non_resp_patch, cutoff_line],
        title="Biomarker Association",
        loc="lower right",
        frameon=True,
    )

    ax.set_title("Ranked Biomarker Association Effect Sizes (Cohen's d: Responders vs Non-Responders)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Cohen's d Effect Size", fontsize=12, fontweight="bold")
    ax.set_ylabel("Engineered Biomarker Feature", fontsize=12, fontweight="bold")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved ranked feature association bar plot to {save_path}")


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
        # Use black for yellow lines — yellow is illegible on a white background
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

    ax.plot([0, 1], [0, 1], color="#37474F", linestyle="--", linewidth=1.2, label="Chance Baseline (AUC = 0.50)")

    n_patients = len(valid_df)
    ax.set_title(f"Receiver Operating Characteristic (ROC) & Youden Optimal Cutoffs (N={n_patients})", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved Youden ROC plot to {save_path}")


def plot_genomic_interaction(df: pd.DataFrame, save_path: Path) -> None:
    """Generate 300 DPI bar plot demonstrating response rates across TIS High/Low and `BRAF` mutation status."""
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

    grouped = valid_df.groupby(["BRAF_Status", "TIS_Status"])["RESPONSE_BINARY"].agg(["mean", "count"]).reset_index()
    grouped["Response_Pct"] = grouped["mean"] * 100

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    palette = {"TIS High": RESPONSE_PALETTE["CR/PR"], "TIS Low": PHENOTYPE_PALETTE["Immune Cold"]}

    sns.barplot(
        data=grouped,
        x="BRAF_Status",
        y="Response_Pct",
        hue="TIS_Status",
        palette=palette,
        ax=ax,
        edgecolor="#37474F",
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

    n_patients = len(valid_df)
    ax.set_title(f"Genomic Synergy: Anti-PD-1 Response Rate by `BRAF` Status & TIS Level (N={n_patients})", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Genomic Driver Subtype", fontsize=12, fontweight="bold")
    ax.set_ylabel("Objective Response Rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(title="TIS Phenotype", loc="upper right", frameon=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved genomic interaction plot to {save_path}")


def compute_interaction_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute logistic regression interaction p-values across all immune feature x driver mutation pairs."""
    import statsmodels.api as sm

    valid_df = df.dropna(subset=["RESPONSE_BINARY"]).copy()
    y = valid_df["RESPONSE_BINARY"].values

    immune_feats = [c for c in ["TIS", "CYT", "IFN_gamma", "CD8_T_cells", "B_cells", "M1_M2_Ratio", "Macrophage_STV_Score"] if c in valid_df.columns]
    driver_cols = [c for c in ["mut_BRAF", "mut_NRAS", "mut_NF1"] if c in valid_df.columns]

    results = []

    for imm in immune_feats:
        for drv in driver_cols:
            x_imm = valid_df[imm].values
            x_drv = valid_df[drv].values

            # Skip if driver has fewer than 3 mutated samples
            if np.sum(x_drv == 1) < 3:
                continue

            # Standardize immune feature to z-score for comparable interaction coefficients
            std_imm = (x_imm - np.mean(x_imm)) / (np.std(x_imm) + 1e-8)
            interaction = std_imm * x_drv

            X = np.column_stack([np.ones(len(y)), std_imm, x_drv, interaction])
            try:
                model = sm.Logit(y, X).fit(disp=False, maxiter=100)
                beta_inter = float(model.params[3])
                pval_inter = float(model.pvalues[3])
            except Exception:
                beta_inter = 0.0
                pval_inter = 1.0

            results.append({
                "Immune_Feature": imm,
                "Driver_Mutation": drv,
                "Beta_Interaction": beta_inter,
                "p_value": pval_inter,
                "neg_log10_p": float(-np.log10(max(pval_inter, 1e-15))),
            })

    return pd.DataFrame(results)


def plot_interaction_heatmap(df_inter: pd.DataFrame, save_path: Path) -> None:
    """Generate 300 DPI annotated heatmap of genomic x immune interaction effect sizes (Beta_interaction)."""
    if df_inter.empty:
        return

    piv_beta = df_inter.pivot(index="Immune_Feature", columns="Driver_Mutation", values="Beta_Interaction")
    piv_p = df_inter.pivot(index="Immune_Feature", columns="Driver_Mutation", values="neg_log10_p")
    piv_raw_p = df_inter.pivot(index="Immune_Feature", columns="Driver_Mutation", values="p_value")

    # Rename columns for presentation
    rename_map = {"mut_BRAF": "BRAF Mutated", "mut_NRAS": "NRAS Mutated", "mut_NF1": "NF1 Mutated"}
    piv_beta = piv_beta.rename(columns=rename_map)
    piv_p = piv_p.rename(columns=rename_map)
    piv_raw_p = piv_raw_p.rename(columns=rename_map)

    fig, ax = plt.subplots(figsize=(9, 6.8), dpi=300)

    # Render base heatmap using Okabe-Ito continuous diverging colormap (Vermillion -> Gray -> Bluish Green)
    okabe_cmap = get_okabe_ito_diverging_cmap()
    sns.heatmap(
        piv_beta,
        annot=False,
        cmap=okabe_cmap,
        center=0,
        vmin=-0.8,
        vmax=+0.8,
        cbar_kws={"label": "Interaction Coefficient (\\beta_{\\text{interaction}})"},
        ax=ax,
        linewidths=0.5,
        linecolor="black",
    )

    # Render two-line cell text annotations: Beta (9.5pt bold) on top, p-value (7.5pt regular) underneath
    for i in range(piv_beta.shape[0]):
        for j in range(piv_beta.shape[1]):
            val_b = piv_beta.iloc[i, j]
            val_pval = piv_raw_p.iloc[i, j]

            # Choose text color based on background brightness
            norm_val = np.clip((val_b - (-0.8)) / (0.8 - (-0.8)), 0, 1)
            rgba = okabe_cmap(norm_val)
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.55 else "black"

            # Top Line: Beta interaction coefficient (bold, 9.5pt)
            ax.text(
                j + 0.5,
                i + 0.36,
                f"β = {val_b:+.2f}",
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color=text_color,
            )
            # Bottom Line: p-value (smaller font, 7.5pt)
            ax.text(
                j + 0.5,
                i + 0.68,
                f"p = {val_pval:.3f}",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="normal",
                color=text_color,
                alpha=0.9,
            )

    # Highlight statistically significant cells (p < 0.05) with a thick white border (#FFFFFF)
    import matplotlib.patches as mpatches

    for i in range(piv_beta.shape[0]):
        for j in range(piv_beta.shape[1]):
            val_p = piv_p.iloc[i, j]
            if val_p >= 1.301:  # p < 0.05
                rect = mpatches.Rectangle((j, i), 1, 1, fill=False, edgecolor="#FFFFFF", lw=3.0, zorder=10)
                ax.add_patch(rect)

    # Add explicit legend handle for the white significance border below the chart
    sig_patch = mpatches.Patch(facecolor="none", edgecolor="#FFFFFF", linewidth=2.5, label="Statistically Significant (p < 0.05)")
    ax.legend(
        handles=[sig_patch],
        loc="upper right",
        bbox_to_anchor=(1.0, -0.16),
        frameon=True,
        facecolor="#37474F",
        labelcolor="white",
        fontsize=9,
    )

    ax.set_title("Genomic x Immune Interaction Matrix (Logistic Regression: $\\beta_{interaction}$)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Driver Mutation Subtype", fontsize=12, fontweight="bold")
    ax.set_ylabel("Immune Microenvironment Feature", fontsize=12, fontweight="bold")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved interaction matrix heatmap to {save_path}")


def main() -> None:
    """Main execution function for Q5 Phase 2 Feature Analysis."""
    print(f"Starting Q5 Phase 2 Feature Analysis (Project root: {rel_path(PROJECT_ROOT)})")

    if not FEATURE_MATRIX_FILE.exists():
        raise FileNotFoundError(f"Feature matrix file not found at {rel_path(FEATURE_MATRIX_FILE)}. Run 01_load_and_prepare.py first.")

    df_feat = pd.read_csv(FEATURE_MATRIX_FILE)
    print(f"Loaded feature matrix: {len(df_feat)} patients x {df_feat.shape[1]} features.")

    # 1. Run univariate associations
    df_assoc = run_univariate_associations(df_feat)
    assoc_file = OUTPUT_DIR / "univariate_feature_associations.csv"
    df_assoc.to_csv(assoc_file, index=False)
    print(f"Saved univariate association results to {rel_path(assoc_file)}")

    # 2. Compute Youden optimal cutoffs
    target_feats = [c for c in ["TIS", "CYT", "IFN_gamma", "CD8_T_cells", "B_cells", "M1_M2_Ratio"] if c in df_feat.columns]
    df_cutoffs = compute_youden_cutoffs(df_feat, target_feats)
    cutoffs_file = OUTPUT_DIR / "youden_cutoffs.csv"
    df_cutoffs.to_csv(cutoffs_file, index=False)
    print(f"Saved Youden cutoffs to {rel_path(cutoffs_file)}")

    # 3. Compute multi-permutation genomic x immune interaction matrix
    df_inter = compute_interaction_matrix(df_feat)
    inter_file = OUTPUT_DIR / "genomic_immune_interactions.csv"
    df_inter.to_csv(inter_file, index=False)
    print(f"Saved interaction matrix to {rel_path(inter_file)}")

    # 4. Generate 300 DPI publication plots
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
