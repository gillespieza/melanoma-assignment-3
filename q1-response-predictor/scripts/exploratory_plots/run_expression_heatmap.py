"""Gene Expression Heatmaps: Top Genes by Variance.

Generates gene-level expression heatmaps (Top 50 highly variable genes) before and after
per-cohort Z-score batch correction across dynamically loaded trial cohorts.
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

from src.data_loaders import load_merged_immunotherapy
from src.styles import RESPONSE_PALETTE, resolve_cohort_palette, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import find_project_root
from src.utils.plotting import save_fig

set_presentation_style()

# Module-level Constants
DATA_DIR = find_project_root(Path(__file__).resolve()) / "data"
PLOT_DIR = BASE_DIR / "plots" / "exploratory"
REPORT_DIR = BASE_DIR / "reports" / "pillar-1-cohorts-and-preprocessing"
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "run_expression_heatmap.log"
N_TOP_HEATMAP_GENES: int = 50


def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardises DataFrame columns to mean=0, std=1.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Z-score standardised DataFrame.
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

    print(f"Loading pre-merged immunotherapy dataset from data/processed/merged/immunotherapy/...")
    expr_dict, clin_dict, cohort_order, trial_names = load_merged_immunotherapy(DATA_DIR)

    if not trial_names:
        raise ValueError("No valid trial cohorts loaded from configuration.")

    print(f"Active trial cohorts: {trial_names}")

    clean_expr_dict: dict[str, pd.DataFrame] = {}
    clean_clin_dict: dict[str, pd.DataFrame] = {}

    for name in trial_names:
        df_clin = clin_dict[name].copy()
        df_expr = expr_dict[name].copy()

        resp_col = None
        for col in ["RESPONSE_BINARY", "response", "RESPONDER"]:
            if col in df_clin.columns:
                resp_col = col
                break

        if resp_col is not None:
            df_clin["temp_resp"] = pd.to_numeric(df_clin[resp_col], errors="coerce")
            df_clin.dropna(subset=["temp_resp"], inplace=True)
            df_clin["Response"] = df_clin["temp_resp"].map(
                {1.0: "Responder (CR/PR)", 0.0: "Non-responder (PD)"}
            )
            df_clin.drop(columns=["temp_resp"], inplace=True)
        else:
            df_clin["Response"] = np.nan

        df_clin.dropna(subset=["Response"], inplace=True)
        common = df_expr.index.intersection(df_clin.index)
        clean_expr_dict[name] = df_expr.loc[common]
        clean_clin_dict[name] = df_clin.loc[common]

    common_genes = sorted(
        list(set.intersection(*(set(clean_expr_dict[c].columns) for c in trial_names)))
    )
    print(f"Number of common genes across trial cohorts: {len(common_genes)}")

    raw_dfs: list[pd.DataFrame] = []
    scaled_dfs: list[pd.DataFrame] = []
    clin_dfs: list[pd.DataFrame] = []

    for name in trial_names:
        e_raw = clean_expr_dict[name][common_genes]
        e_scaled = zscore_df(e_raw)
        c_df = clean_clin_dict[name][["Cohort", "Response"]]

        raw_dfs.append(e_raw)
        scaled_dfs.append(e_scaled)
        clin_dfs.append(c_df)

    expr_raw_merged = pd.concat(raw_dfs, axis=0)
    expr_scaled_merged = pd.concat(scaled_dfs, axis=0)
    clin_merged = pd.concat(clin_dfs, axis=0)

    # Sort samples: Cohort order (as loaded), then Responder before Non-responder within each cohort
    _cohort_order_map = {name: i for i, name in enumerate(trial_names)}
    _response_order_map = {"Responder (CR/PR)": 0, "Non-responder (PD)": 1}
    clin_merged["_cohort_rank"] = clin_merged["Cohort"].map(_cohort_order_map)
    clin_merged["_response_rank"] = clin_merged["Response"].map(_response_order_map)
    clin_merged.sort_values(["_cohort_rank", "_response_rank"], inplace=True)
    clin_merged.drop(columns=["_cohort_rank", "_response_rank"], inplace=True)

    sort_idx = clin_merged.index
    expr_raw_merged = expr_raw_merged.loc[sort_idx]
    expr_scaled_merged = expr_scaled_merged.loc[sort_idx]

    # Average within-cohort gene variance to select top biological genes
    gene_variances = pd.DataFrame({
        name: df[common_genes].var(axis=0) for name, df in clean_expr_dict.items()
    }).mean(axis=1)

    top_heatmap_genes = gene_variances.sort_values(ascending=False).head(N_TOP_HEATMAP_GENES).index.tolist()
    print("Top 5 genes by average within-cohort variance:", top_heatmap_genes[:5])

    colors_cohort = resolve_cohort_palette(trial_names)
    colors_response = {
        "Responder (CR/PR)": RESPONSE_PALETTE["CR/PR"],
        "Non-responder (PD)": RESPONSE_PALETTE["PD"],
    }

    col_colors = pd.DataFrame(index=clin_merged.index)
    col_colors["Cohort"] = clin_merged["Cohort"].map(colors_cohort)
    col_colors["Response"] = clin_merged["Response"].map(colors_response)

    legend_elements = [
        Patch(facecolor=colors_cohort[name], label=name) for name in trial_names if name in colors_cohort
    ] + [
        Patch(facecolor="white", edgecolor="none", label=""),
        Patch(facecolor=RESPONSE_PALETTE["CR/PR"], label="Responder (CR/PR)"),
        Patch(facecolor=RESPONSE_PALETTE["PD"], label="Non-responder (PD)"),
    ]

    # Raw: plot true log2(TPM+1) values so cross-cohort baseline shifts are genuinely visible
    df_raw_heatmap = expr_raw_merged[top_heatmap_genes].T
    df_scaled_heatmap = expr_scaled_merged[top_heatmap_genes].T

    print("Generating raw expression heatmap (before cohort batch correction)...")
    g_raw = sns.clustermap(
        df_raw_heatmap,
        method="ward",
        metric="euclidean",
        cmap="viridis",
        col_colors=col_colors,
        col_cluster=False,
        figsize=(12, 14),
        yticklabels=True,
        xticklabels=False,
        robust=True,
        cbar_pos=(0.02, 0.8, 0.035, 0.15),
        cbar_kws={"label": "log2(TPM+1)"},
    )

    g_raw.ax_col_dendrogram.set_title(
        f"Expression Heatmap Before Batch Correction (Top {N_TOP_HEATMAP_GENES} Genes by Variance)",
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
        ncol=min(4, len(legend_elements)),
        frameon=True,
    )

    raw_heatmap_path = PLOT_DIR / "heatmap_top_variance_genes_raw.png"
    save_fig(g_raw.fig, raw_heatmap_path)
    print(f"Saved raw heatmap to {raw_heatmap_path.relative_to(BASE_DIR).as_posix()}")

    print("Generating standardised expression heatmap (after per-cohort Z-score correction)...")
    g_scaled = sns.clustermap(
        df_scaled_heatmap,
        method="ward",
        metric="euclidean",
        cmap="viridis",
        col_colors=col_colors,
        col_cluster=False,
        figsize=(12, 14),
        yticklabels=True,
        xticklabels=False,
        vmin=-3,
        vmax=3,
        cbar_pos=(0.02, 0.8, 0.035, 0.15),
        cbar_kws={"label": "Standardised Z-score (Per-Cohort)"},
    )

    g_scaled.ax_col_dendrogram.set_title(
        f"Expression Heatmap After Batch Correction (Top {N_TOP_HEATMAP_GENES} Genes by Variance)",
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
        ncol=min(4, len(legend_elements)),
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
