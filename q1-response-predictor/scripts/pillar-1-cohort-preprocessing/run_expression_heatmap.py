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

try:
    import fastcluster
    HAS_FASTCLUSTER = True
except ImportError:
    HAS_FASTCLUSTER = False

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


def compute_fast_linkage(
    data: pd.DataFrame, method: str = "ward", metric: str = "euclidean"
) -> np.ndarray:
    """Computes hierarchical clustering linkage matrix via fastcluster if available.

    Args:
        data: Input DataFrame or 2D array.
        method: Hierarchical clustering linkage method.
        metric: Distance metric.

    Returns:
        Computed linkage matrix.
    """
    if HAS_FASTCLUSTER:
        return fastcluster.linkage(data, method=method, metric=metric)
    from scipy.cluster import hierarchy
    return hierarchy.linkage(data, method=method, metric=metric)


def _process_single_cohort_data(
    name: str, clin_dict: dict, expr_dict: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Processes clinical response column and subsets common sample indices."""
    df_clin = clin_dict[name].copy()
    df_expr = expr_dict[name].copy()
    resp_cands = ["RESPONSE_BINARY", "response", "RESPONDER"]
    resp_col = next((c for c in resp_cands if c in df_clin.columns), None)

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
    return df_expr.loc[common], df_clin.loc[common]


def _load_clean_cohort_dicts(
    trial_names: list[str], clin_dict: dict, expr_dict: dict
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[str]]:
    """Loads and harmonises clinical and expression data across trial cohorts."""
    clean_expr, clean_clin = {}, {}
    for name in trial_names:
        e_df, c_df = _process_single_cohort_data(name, clin_dict, expr_dict)
        clean_expr[name], clean_clin[name] = e_df, c_df
    gene_sets = [set(clean_expr[c].columns) for c in trial_names]
    common_genes = sorted(list(set.intersection(*gene_sets)))
    return clean_expr, clean_clin, common_genes


def _merge_and_sort_cohort_dfs(
    trial_names: list[str], clean_expr: dict, clean_clin: dict, common_genes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Concatenates and sorts cohort data by trial order and response status."""
    raw_list, scaled_list, clin_list = [], [], []
    for name in trial_names:
        e_raw = clean_expr[name][common_genes]
        raw_list.append(e_raw)
        scaled_list.append(zscore_df(e_raw))
        clin_list.append(clean_clin[name][["Cohort", "Response"]])

    c_merged = pd.concat(clin_list, axis=0)
    c_map = {name: i for i, name in enumerate(trial_names)}
    r_map = {"Responder (CR/PR)": 0, "Non-responder (PD)": 1}
    c_merged["_c_rank"] = c_merged["Cohort"].map(c_map)
    c_merged["_r_rank"] = c_merged["Response"].map(r_map)
    c_merged.sort_values(["_c_rank", "_r_rank"], inplace=True)
    c_merged.drop(columns=["_c_rank", "_r_rank"], inplace=True)

    sort_idx = c_merged.index
    e_raw_merged = pd.concat(raw_list, axis=0).loc[sort_idx]
    e_scaled_merged = pd.concat(scaled_list, axis=0).loc[sort_idx]
    return e_raw_merged, e_scaled_merged, c_merged


def _select_top_variance_genes(
    clean_expr: dict[str, pd.DataFrame], common_genes: list[str], n_top: int
) -> list[str]:
    """Computes average within-cohort variance and returns top gene symbols."""
    gene_vars = pd.DataFrame({
        name: df[common_genes].var(axis=0) for name, df in clean_expr.items()
    }).mean(axis=1)
    return gene_vars.sort_values(ascending=False).head(n_top).index.tolist()


def _build_annotation_colors(
    trial_names: list[str], clin_merged: pd.DataFrame
) -> tuple[pd.DataFrame, list[Patch]]:
    """Generates column color mapping and legend patches for heatmap."""
    colors_cohort = resolve_cohort_palette(trial_names)
    colors_response = {
        "Responder (CR/PR)": RESPONSE_PALETTE["CR/PR"],
        "Non-responder (PD)": RESPONSE_PALETTE["PD"],
    }
    col_colors = pd.DataFrame(index=clin_merged.index)
    col_colors["Cohort"] = clin_merged["Cohort"].map(colors_cohort)
    col_colors["Response"] = clin_merged["Response"].map(colors_response)

    legend_elements = [
        Patch(facecolor=colors_cohort[name], label=name)
        for name in trial_names if name in colors_cohort
    ] + [
        Patch(facecolor="white", edgecolor="none", label=""),
        Patch(facecolor=RESPONSE_PALETTE["CR/PR"], label="Responder (CR/PR)"),
        Patch(facecolor=RESPONSE_PALETTE["PD"], label="Non-responder (PD)"),
    ]
    return col_colors, legend_elements


def _render_and_save_raw_heatmap(
    df_raw: pd.DataFrame, col_colors: pd.DataFrame, legend_el: list[Patch]
) -> None:
    """Renders and saves uncorrected raw gene expression heatmap."""
    print("Generating raw expression heatmap (before cohort batch correction)...")
    raw_linkage = compute_fast_linkage(df_raw, method="ward", metric="euclidean")
    g = sns.clustermap(
        df_raw, row_linkage=raw_linkage, cmap="viridis", col_colors=col_colors,
        col_cluster=False, figsize=(12, 14), yticklabels=True, xticklabels=False,
        robust=True, cbar_pos=(0.02, 0.8, 0.035, 0.15), cbar_kws={"label": "log2(TPM+1)"}
    )
    g.ax_col_dendrogram.set_title(
        f"Expression Heatmap Before Batch Correction (Top {N_TOP_HEATMAP_GENES} Genes by Variance)",
        fontsize=14, fontweight="bold", pad=15
    )
    g.fig.subplots_adjust(top=0.90, right=0.95)
    g.ax_col_dendrogram.legend(
        handles=legend_el, loc="lower center", bbox_to_anchor=(0.5, -0.25),
        ncol=min(4, len(legend_el)), frameon=True
    )
    out_path = PLOT_DIR / "heatmap_top_variance_genes_raw.png"
    save_fig(g.fig, out_path)
    print(f"Saved raw heatmap to {out_path.relative_to(BASE_DIR).as_posix()}")


def _render_and_save_scaled_heatmap(
    df_scaled: pd.DataFrame, col_colors: pd.DataFrame, legend_el: list[Patch]
) -> None:
    """Renders and saves batch-corrected Z-score gene expression heatmap."""
    print("Generating standardised expression heatmap (after per-cohort Z-score correction)...")
    scaled_linkage = compute_fast_linkage(df_scaled, method="ward", metric="euclidean")
    g = sns.clustermap(
        df_scaled, row_linkage=scaled_linkage, cmap="viridis", col_colors=col_colors,
        col_cluster=False, figsize=(12, 14), yticklabels=True, xticklabels=False,
        vmin=-3, vmax=3, cbar_pos=(0.02, 0.8, 0.035, 0.15),
        cbar_kws={"label": "Standardised Z-score (Per-Cohort)"}
    )
    g.ax_col_dendrogram.set_title(
        f"Expression Heatmap After Batch Correction (Top {N_TOP_HEATMAP_GENES} Genes by Variance)",
        fontsize=14, fontweight="bold", pad=15
    )
    g.fig.subplots_adjust(top=0.85, right=0.95)
    g.ax_col_dendrogram.legend(
        handles=legend_el, loc="lower center", bbox_to_anchor=(0.5, -0.25),
        ncol=min(4, len(legend_el)), frameon=True
    )
    out_path = PLOT_DIR / "heatmap_top_variance_genes_standardized.png"
    save_fig(g.fig, out_path)
    print(f"Saved standardized heatmap to {out_path.relative_to(BASE_DIR).as_posix()}")


def main() -> None:
    """Executes the gene expression heatmap workflow."""
    print("==================================================")
    print("Gene Expression Heatmaps: Top Genes by Variance")
    print("==================================================\n")
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_DIR.mkdir(exist_ok=True, parents=True)

    expr_dict, clin_dict, _, trial_names = load_merged_immunotherapy(DATA_DIR)
    if not trial_names:
        raise ValueError("No valid trial cohorts loaded from configuration.")

    print(f"Active trial cohorts: {trial_names}")
    c_expr, c_clin, common_genes = _load_clean_cohort_dicts(trial_names, clin_dict, expr_dict)
    print(f"Number of common genes across trial cohorts: {len(common_genes)}")

    e_raw, e_scaled, clin_merged = _merge_and_sort_cohort_dfs(
        trial_names, c_expr, c_clin, common_genes
    )
    top_genes = _select_top_variance_genes(c_expr, common_genes, N_TOP_HEATMAP_GENES)
    print("Top 5 genes by average within-cohort variance:", top_genes[:5])

    col_colors, legend_el = _build_annotation_colors(trial_names, clin_merged)
    _render_and_save_raw_heatmap(e_raw[top_genes].T, col_colors, legend_el)
    _render_and_save_scaled_heatmap(e_scaled[top_genes].T, col_colors, legend_el)
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
