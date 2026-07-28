"""
Gene Expression Heatmaps: Top Genes by Variance.

Generates gene-level expression heatmaps (Top 50 highly variable genes) before and after
per-cohort Z-score batch correction across Liu 2019, Hugo 2016, and Riaz 2017 cohorts.
"""

import contextlib
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import find_project_root
from src.utils.plotting import save_fig

# Module-level Constants
DATA_DIR = find_project_root(Path(__file__).resolve()) / "data"
PLOT_DIR = BASE_DIR / "plots" / "exploratory"
REPORT_DIR = BASE_DIR  / "reports" / "pillar-1-cohorts-and-preprocessing"
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "run_expression_heatmap.log"


def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardises DataFrame columns to mean=0, std=1.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Z-score standardized DataFrame.
    """
    means = df.mean(axis=0)
    stds = df.std(axis=0).replace(0, 1.0).fillna(1.0)
    return (df - means) / stds


def main() -> None:
    """Executes the gene expression heatmap workflow."""
    print("==================================================")
    print("Gene Expression Heatmaps: Top Genes by Variance")
    print("==================================================\n")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_DIR.mkdir(exist_ok=True, parents=True)

    print("Loading datasets...")
    expr_liu, clin_liu = load_liu_2019(DATA_DIR)
    expr_hugo, clin_hugo = load_hugo_2016(DATA_DIR)
    expr_riaz, clin_riaz = load_riaz_2017(DATA_DIR)

    response_map = {
        "Complete Response": 1,
        "Partial Response": 1,
        "Progressive Disease": 0,
        "Stable Disease": np.nan,
        "Mixed Response": np.nan,
    }

    for name, df_clin, _ in [("Liu 2019", clin_liu, expr_liu), ("Hugo 2016", clin_hugo, expr_hugo), ("Riaz 2017", clin_riaz, expr_riaz)]:
        df_clin["temp_resp"] = df_clin["RESPONSE"].map(response_map)
        df_clin.dropna(subset=["temp_resp"], inplace=True)
        df_clin.drop(columns=["temp_resp"], inplace=True)

        if name == "Liu 2019":
            expr_liu = expr_liu.loc[df_clin.index]
        elif name == "Hugo 2016":
            expr_hugo = expr_hugo.loc[df_clin.index]
        elif name == "Riaz 2017":
            expr_riaz = expr_riaz.loc[df_clin.index]

    common_genes = list(expr_liu.columns.intersection(expr_hugo.columns).intersection(expr_riaz.columns))
    common_genes.sort()
    print(f"Number of common genes: {len(common_genes)}")

    expr_liu = expr_liu[common_genes]
    expr_hugo = expr_hugo[common_genes]
    expr_riaz = expr_riaz[common_genes]

    expr_raw_merged = pd.concat([expr_liu, expr_hugo, expr_riaz], axis=0)

    expr_liu_scaled = zscore_df(expr_liu)
    expr_hugo_scaled = zscore_df(expr_hugo)
    expr_riaz_scaled = zscore_df(expr_riaz)
    expr_scaled_merged = pd.concat([expr_liu_scaled, expr_hugo_scaled, expr_riaz_scaled], axis=0)

    clin_liu["Cohort"] = "Liu 2019"
    clin_hugo["Cohort"] = "Hugo 2016"
    clin_riaz["Cohort"] = "Riaz 2017"

    clin_merged = pd.concat(
        [clin_liu[["Cohort", "response"]], clin_hugo[["Cohort", "response"]], clin_riaz[["Cohort", "response"]]],
        axis=0,
    )

    expr_raw_merged = expr_raw_merged.loc[clin_merged.index]
    expr_scaled_merged = expr_scaled_merged.loc[clin_merged.index]

    clin_merged["Response"] = clin_merged["response"].map({1.0: "Responder (CR/PR)", 0.0: "Non-responder (PD)"})

    set_presentation_style()

    colors_cohort = COHORT_PALETTE
    colors_response = {
        "Responder (CR/PR)": RESPONSE_PALETTE["CR/PR"],
        "Non-responder (PD)": RESPONSE_PALETTE["PD"],
    }

    col_colors = pd.DataFrame(index=clin_merged.index)
    col_colors["Cohort"] = clin_merged["Cohort"].map(colors_cohort)
    col_colors["Response"] = clin_merged["Response"].map(colors_response)

    legend_elements = [
        Patch(facecolor=COHORT_PALETTE["Liu 2019"], label="Liu 2019"),
        Patch(facecolor=COHORT_PALETTE["Hugo 2016"], label="Hugo 2016"),
        Patch(facecolor=COHORT_PALETTE["Riaz 2017"], label="Riaz 2017"),
        Patch(facecolor="white", edgecolor="none", label=""),
        Patch(facecolor=RESPONSE_PALETTE["CR/PR"], label="Responder (CR/PR)"),
        Patch(facecolor=RESPONSE_PALETTE["PD"], label="Non-responder (PD)"),
    ]

    print("Calculating gene variances on raw log2 data...")
    raw_variances = expr_raw_merged.var(axis=0)
    top_50_genes = raw_variances.sort_values(ascending=False).head(50).index.tolist()
    print("Top 5 genes by variance:", top_50_genes[:5])

    df_raw_heatmap = expr_raw_merged[top_50_genes].T
    df_scaled_heatmap = expr_scaled_merged[top_50_genes].T

    print("Generating raw expression heatmap...")
    g_raw = sns.clustermap(
        df_raw_heatmap,
        method="ward",
        metric="euclidean",
        cmap="viridis",
        col_colors=col_colors,
        figsize=(12, 14),
        yticklabels=True,
        xticklabels=False,
        cbar_pos=(0.02, 0.8, 0.035, 0.15),
        cbar_kws={"label": "log2(Expression + 1)"},
    )

    g_raw.ax_col_dendrogram.set_title(
        "Expression Heatmap Before Batch Correction (Top 50 Genes by Variance)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    g_raw.fig.subplots_adjust(
        top=0.90,
        right=0.95,
    )

    g_raw.ax_col_dendrogram.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=4,
        frameon=True,
    )

    raw_heatmap_path = PLOT_DIR / "heatmap_top_variance_genes_raw.png"
    save_fig(g_raw.fig, raw_heatmap_path)
    print(f"Saved raw heatmap to {raw_heatmap_path.relative_to(BASE_DIR).as_posix()}")

    print("Generating standardized expression heatmap...")
    g_scaled = sns.clustermap(
        df_scaled_heatmap,
        method="ward",
        metric="euclidean",
        cmap="RdBu_r",
        col_colors=col_colors,
        figsize=(12, 14),
        yticklabels=True,
        xticklabels=False,
        vmin=-3,
        vmax=3,
        cbar_pos=(0.02, 0.8, 0.035, 0.15),
        cbar_kws={"label": "Z-score Expression"},
    )

    g_scaled.ax_col_dendrogram.set_title(
        "Expression Heatmap After Batch Correction (Top 50 Genes by Variance)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    g_scaled.fig.subplots_adjust(
        top=0.85,
        right=0.95,
    )

    g_scaled.ax_col_dendrogram.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=4,
        frameon=True,
    )

    scaled_heatmap_path = PLOT_DIR / "heatmap_top_variance_genes_standardized.png"
    save_fig(g_scaled.fig, scaled_heatmap_path)
    print(f"Saved standardized heatmap to {scaled_heatmap_path.relative_to(BASE_DIR).as_posix()}")

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
