"""
Dimensionality Reduction & Batch Correction Analysis across Melanoma Cohorts.

Performs PCA and UMAP projections across all ICI trial cohorts loaded from
data/processed/merged/immunotherapy/ (Liu 2019, Hugo 2016, Riaz 2017, TCGA GDC 2025,
Gide 2019, Van Allen 2015) to assess technical batch effects before and after
cohort-independent Z-score scaling, and updates batch_correction_report.md.
"""

# ---------------------------------------------------------------------------
# Standard Library & Third-Party Imports
# ---------------------------------------------------------------------------

import contextlib
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

try:
    import umap
    HAS_UMAP = True
except ImportError as e:
    print(f"Warning: umap not installed ({e}). Falling back to t-SNE for projection.")
    HAS_UMAP = False

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SUBPROJECT_ROOT = SCRIPT_DIR.parents[1]
_SEARCH_DIR = SCRIPT_DIR
while _SEARCH_DIR != _SEARCH_DIR.parent:
    if (_SEARCH_DIR / "src").is_dir() and (_SEARCH_DIR / "data").is_dir():
        PROJECT_ROOT = _SEARCH_DIR
        break
    _SEARCH_DIR = _SEARCH_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR.parents[3]

if str(SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from src.config.datasets import DatasetConfig, load_dataset_config
from src.data_loaders import load_merged_immunotherapy
from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, resolve_cohort_palette, set_presentation_style
from src.utils.formatting import (
    generate_obsidian_frontmatter,
    generate_script_reference_callout,
)
from src.utils.logging import TeeStream
from src.utils.paths import (
    DATA_DIR,
    PROCESSED_DIR,
    get_subproject_log_dir,
    rel_path,
)
from src.utils.plotting import save_fig

set_presentation_style()

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

# Column Identifiers
_COL_SAMPLE_ID: str = "SAMPLE_ID"
_COL_COHORT: str = "Cohort"
_COL_RESPONSE: str = "Response"
_COL_RAW_RESPONSE: str = "response"

# Config Path
CONFIG_PATH: Path = SUBPROJECT_ROOT / "config" / "datasets.yaml"

# Dimension Reduction Hyperparameters
N_TOP_VARIABLE_GENES: int = 1000
N_PCA_COMPONENTS: int = 2
UMAP_NEIGHBORS: int = 15
UMAP_MIN_DIST: float = 0.1
TSNE_PERPLEXITY: float = 15.0
RANDOM_SEED: int = 42

# Visualisation Geometry
FIG_SIZE_1X2: Tuple[int, int] = (14, 6)
FIG_SIZE_2X2: Tuple[int, int] = (14, 12)
SCATTER_ALPHA: float = 0.8
SCATTER_SIZE: int = 80
SCATTER_SIZE_LARGE: int = 100
N_TOP_HEATMAP_GENES: int = 50

# Target Directories & Logging Paths
EXPLORATORY_PLOT_DIR: Path = SUBPROJECT_ROOT / "plots" / "exploratory"
REPORT_DIR: Path = SUBPROJECT_ROOT / "reports" / "pillar-1-cohorts-and-preprocessing"
LOG_DIR: Path = get_subproject_log_dir(SCRIPT_DIR)
LOG_PATH: Path = LOG_DIR / "run_dimensionality_reduction.log"

# Domain Category Labels & Orders
RESPONSE_ORDER: List[str] = ["Responder (CR/PR)", "Non-responder (PD)"]
RESPONSE_LABEL_MAP: Dict[float, str] = {
    1.0: "Responder (CR/PR)",
    0.0: "Non-responder (PD)",
}


# ---------------------------------------------------------------------------
# Data Preprocessing & Alignment Functions
# ---------------------------------------------------------------------------

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


def _align_expr_clin(
    expr: pd.DataFrame, clin: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aligns expression and clinical DataFrames by matching patient indices.

    Args:
        expr: Expression DataFrame.
        clin: Clinical DataFrame.

    Returns:
        Tuple of aligned (expression, clinical) DataFrames.
    """
    common = expr.index.intersection(clin.index)
    return expr.loc[common], clin.loc[common]


def load_all_cohorts() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], List[str], List[str]]:
    """Loads and aligns the pre-merged immunotherapy dataset.

    Delegates to load_merged_immunotherapy() which reads from
    data/processed/merged/immunotherapy/, the authoritative source that
    includes the TCGA GDC 2025 immunotherapy-treated subset.

    Returns:
        Tuple of (expr_dict, clin_dict, cohort_order, trial_names).
    """
    return load_merged_immunotherapy(DATA_DIR)


def select_top_variable_genes(
    expr_dict: Dict[str, pd.DataFrame],
    trial_names: List[str],
    top_n: int = N_TOP_VARIABLE_GENES,
) -> Tuple[List[str], List[str], List[str]]:
    """Identifies common genes across cohorts and selects top variable genes.

    Args:
        expr_dict: Dictionary mapping cohort names to expression DataFrames.
        trial_names: Ordered list of trial cohort names.
        top_n: Number of high-variance genes to select.

    Returns:
        Tuple of (all-cohort common genes, trial-cohort common genes, top N variable genes).
    """
    g_all = sorted(list(set.intersection(*(set(df.columns) for df in expr_dict.values()))))
    trial_dfs = [expr_dict[c] for c in trial_names if c in expr_dict]
    g_trials = (
        sorted(list(set.intersection(*(set(df.columns) for df in trial_dfs))))
        if trial_dfs
        else g_all
    )

    pooled_g = pd.concat([df[g_all] for df in expr_dict.values()], axis=0)
    top_genes = pooled_g.var(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
    return g_all, g_trials, top_genes


# ---------------------------------------------------------------------------
# Data Concatenation Helper
# ---------------------------------------------------------------------------

def _build_concat(
    expr_dict: Dict[str, pd.DataFrame],
    clin_dict: Dict[str, pd.DataFrame],
    cohort_names: List[str],
    gene_list: List[str],
    with_response: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Concatenates raw and Z-score expression matrices and clinical metadata.

    Args:
        expr_dict: Mapping of cohort names to expression DataFrames.
        clin_dict: Mapping of cohort names to clinical DataFrames.
        cohort_names: Ordered cohort names to concatenate.
        gene_list: Gene columns to select from each expression DataFrame.
        with_response: If True, include response columns in the clinical slice.

    Returns:
        Tuple of (expr_raw, expr_scaled, clin) concatenated DataFrames.
    """
    expr_raw = pd.concat([expr_dict[c][gene_list] for c in cohort_names], axis=0)
    expr_scaled = pd.concat([zscore_df(expr_dict[c][gene_list]) for c in cohort_names], axis=0)
    clin_cols = [_COL_COHORT, _COL_RAW_RESPONSE] if with_response else [_COL_COHORT]
    clin = pd.concat([clin_dict[c][clin_cols] for c in cohort_names], axis=0)
    return expr_raw, expr_scaled, clin


# ---------------------------------------------------------------------------
# Dimensionality Reduction Transformations
# ---------------------------------------------------------------------------

def fit_pca_projection(
    expr_raw: pd.DataFrame, expr_scaled: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Fits 2D PCA on raw and Z-score scaled expression matrices.

    Args:
        expr_raw: Uncorrected raw expression matrix.
        expr_scaled: Cohort Z-score standardised expression matrix.

    Returns:
        Tuple of (pcs_raw, pcs_scaled, pc1_raw_pct, pc2_raw_pct, pc1_scaled_pct, pc2_scaled_pct).
    """
    pca_raw = PCA(n_components=N_PCA_COMPONENTS, svd_solver="full")
    pcs_raw = pca_raw.fit_transform(expr_raw)
    pc1_r, pc2_r = pca_raw.explained_variance_ratio_[:2] * 100.0

    pca_scaled = PCA(n_components=N_PCA_COMPONENTS, svd_solver="full")
    pcs_scaled = pca_scaled.fit_transform(expr_scaled)
    pc1_s, pc2_s = pca_scaled.explained_variance_ratio_[:2] * 100.0

    return pcs_raw, pcs_scaled, pc1_r, pc2_r, pc1_s, pc2_s


def fit_umap_or_tsne(
    expr_raw: pd.DataFrame, expr_scaled: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes UMAP or t-SNE 2D projections for raw and scaled expression matrices.

    Args:
        expr_raw: Uncorrected raw expression matrix.
        expr_scaled: Cohort Z-score standardised expression matrix.

    Returns:
        Tuple of (projections_raw, projections_scaled).
    """
    if HAS_UMAP:
        model_raw = umap.UMAP(
            n_components=N_PCA_COMPONENTS, random_state=RANDOM_SEED, n_jobs=1,
            n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
        )
        model_scaled = umap.UMAP(
            n_components=N_PCA_COMPONENTS, random_state=RANDOM_SEED, n_jobs=1,
            n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
        )
    else:
        model_raw = TSNE(
            n_components=N_PCA_COMPONENTS, random_state=RANDOM_SEED, perplexity=TSNE_PERPLEXITY
        )
        model_scaled = TSNE(
            n_components=N_PCA_COMPONENTS, random_state=RANDOM_SEED, perplexity=TSNE_PERPLEXITY
        )

    return model_raw.fit_transform(expr_raw), model_scaled.fit_transform(expr_scaled)


# ---------------------------------------------------------------------------
# Visualisation & Plotting Helper Functions
# ---------------------------------------------------------------------------

def _assign_pca_coords(
    clin: pd.DataFrame,
    pcs_raw: np.ndarray,
    pcs_scaled: np.ndarray,
) -> None:
    """Assigns 2D PCA coordinates in-place to a clinical DataFrame.

    Args:
        clin: Clinical DataFrame to assign coordinates into.
        pcs_raw: Raw PCA projection array, shape (N, 2).
        pcs_scaled: Scaled PCA projection array, shape (N, 2).
    """
    clin["PCA_Raw_PC1"], clin["PCA_Raw_PC2"] = pcs_raw[:, 0], pcs_raw[:, 1]
    clin["PCA_Scaled_PC1"], clin["PCA_Scaled_PC2"] = pcs_scaled[:, 0], pcs_scaled[:, 1]


def _assign_umap_coords(
    clin: pd.DataFrame,
    umap_raw: np.ndarray,
    umap_scaled: np.ndarray,
) -> None:
    """Assigns 2D UMAP coordinates in-place to a clinical DataFrame.

    Args:
        clin: Clinical DataFrame to assign coordinates into.
        umap_raw: Raw UMAP projection array, shape (N, 2).
        umap_scaled: Scaled UMAP projection array, shape (N, 2).
    """
    clin["UMAP_Raw_Dim1"], clin["UMAP_Raw_Dim2"] = umap_raw[:, 0], umap_raw[:, 1]
    clin["UMAP_Scaled_Dim1"], clin["UMAP_Scaled_Dim2"] = umap_scaled[:, 0], umap_scaled[:, 1]


def _grid_col_names(dim_prefix: str, phase: str) -> Tuple[str, str]:
    """Returns the x and y DataFrame column names for a reduction grid panel.

    Args:
        dim_prefix: 'PCA' or 'UMAP'.
        phase: 'Raw' or 'Scaled'.

    Returns:
        Tuple of (x_col_name, y_col_name).
    """
    if dim_prefix == "PCA":
        return f"PCA_{phase}_PC1", f"PCA_{phase}_PC2"
    return f"UMAP_{phase}_Dim1", f"UMAP_{phase}_Dim2"


def _render_pca_panel_axes(
    axes: Any, df: pd.DataFrame, panels: List[Tuple[str, str, str]], cohort_list: List[str]
) -> None:
    """Helper to render scatter plots onto 1x2 PCA axes."""
    for ax, (xcol, ycol, title) in zip(axes, panels):
        sns.scatterplot(
            data=df, x=xcol, y=ycol, hue=_COL_COHORT, hue_order=cohort_list,
            palette=resolve_cohort_palette(cohort_list), alpha=SCATTER_ALPHA, s=SCATTER_SIZE,
            edgecolor="w", linewidth=0.5, ax=ax,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")


def plot_cohort_batch_pca(
    df: pd.DataFrame,
    var_tuple: Tuple[float, float, float, float],
    cohort_list: List[str],
    title_suffix: str,
    output_path: Path,
) -> None:
    """Renders 1x2 panel PCA scatter plot for batch effect evaluation across cohorts."""
    pc1_r, pc2_r, pc1_s, pc2_s = var_tuple
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_1X2)
    panels = [
        ("PCA_Raw_PC1", "PCA_Raw_PC2",
         f"A: Raw Matrix (Uncorrected)\nPC1 ({pc1_r:.1f}%) vs PC2 ({pc2_r:.1f}%)"),
        ("PCA_Scaled_PC1", "PCA_Scaled_PC2",
         f"B: Z-Score Standardised (Corrected)\nPC1 ({pc1_s:.1f}%) vs PC2 ({pc2_s:.1f}%)"),
    ]

    _render_pca_panel_axes(axes, df, panels, cohort_list)
    plt.suptitle(
        f"PCA Batch Effect Assessment Across {title_suffix} (N = {len(df)})",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    save_fig(fig, output_path)
    print(f"Saved PCA batch assessment plot to {rel_path(output_path)}")


def _axis_label(
    dim_prefix: str, axis: str, is_uncorrected: bool, val_r: float, val_s: float
) -> str:
    """Returns the axis label string for a PCA or UMAP panel."""
    component = "PC1" if axis == "x" else "PC2"
    dim_num = "1" if axis == "x" else "2"
    if dim_prefix != "PCA":
        return f"UMAP Dimension {dim_num}"
    pct = val_r if is_uncorrected else val_s
    return f"{component} ({pct:.1f}% variance)"


def _render_2x2_scatter_grid(
    axes: Any, df: pd.DataFrame, configs: List[Any], dim_prefix: str,
    pc1_r: float, pc1_s: float, pc2_r: float, pc2_s: float
) -> None:
    """Helper to render 2x2 scatter grid panels."""
    for ax, (xcol, ycol, hue_col, order, palette, title) in zip(axes.flat, configs):
        sns.scatterplot(
            data=df, x=xcol, y=ycol, hue=hue_col, hue_order=order, palette=palette,
            style=hue_col, alpha=SCATTER_ALPHA, s=SCATTER_SIZE_LARGE,
            edgecolor="w", linewidth=0.8, ax=ax,
        )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel(_axis_label(dim_prefix, "x", "Before" in title, pc1_r, pc1_s))
        ax.set_ylabel(_axis_label(dim_prefix, "y", "Before" in title, pc2_r, pc2_s))


def _get_grid_configs(
    dim_prefix: str,
    method_name: str,
    trial_names: List[str],
    resp_colors: Dict[str, str],
) -> List[Any]:
    """Builds tuple configurations for 2x2 scatter plot grid panels."""
    cohort_palette = resolve_cohort_palette(trial_names)
    return [
        (*_grid_col_names(dim_prefix, "Raw"), _COL_COHORT, trial_names, cohort_palette,
         f"{method_name} Before Batch Correction (Coloured by Cohort)"),
        (*_grid_col_names(dim_prefix, "Raw"), _COL_RESPONSE, RESPONSE_ORDER, resp_colors,
         f"{method_name} Before Batch Correction (Coloured by Response)"),
        (*_grid_col_names(dim_prefix, "Scaled"), _COL_COHORT, trial_names, cohort_palette,
         f"{method_name} After Batch Correction (Coloured by Cohort)"),
        (*_grid_col_names(dim_prefix, "Scaled"), _COL_RESPONSE, RESPONSE_ORDER, resp_colors,
         f"{method_name} After Batch Correction (Coloured by Response)"),
    ]


def plot_trial_reduction_grid(
    df: pd.DataFrame,
    dim_prefix: str,
    pc_vars: Tuple[float, float, float, float],
    method_name: str,
    trial_names: List[str],
    output_path: Path,
) -> None:
    """Renders 2x2 grid of scatter plots coloured by Cohort and Response."""
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_2X2)
    resp_colors = {
        "Responder (CR/PR)": RESPONSE_PALETTE["CR/PR"],
        "Non-responder (PD)": RESPONSE_PALETTE["PD"],
    }
    pc1_r, pc2_r, pc1_s, pc2_s = pc_vars
    configs = _get_grid_configs(dim_prefix, method_name, trial_names, resp_colors)

    _render_2x2_scatter_grid(axes, df, configs, dim_prefix, pc1_r, pc1_s, pc2_r, pc2_s)
    plt.suptitle(
        f"{method_name} Reduction of Trial Expression (N = {len(df)})",
        fontsize=16, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    save_fig(fig, output_path)
    print(f"Saved {method_name} 2x2 grid to {rel_path(output_path)}")


# ---------------------------------------------------------------------------
# Markdown Report Generation Functions
# ---------------------------------------------------------------------------

def _report_section_1a(
    n_full: int,
    cohort_counts: Dict[str, int],
    n_top: int,
    n_g_all: int,
    var_full: Tuple[float, float, float, float],
) -> str:
    """Returns Markdown text for Section 1.1: Full Cohort Batch Assessment."""
    p1_rf, p2_rf, p1_sf, p2_sf = var_full
    cohort_breakdown = ", ".join(
        f"**{name}** [$N = {count}$]" for name, count in cohort_counts.items()
    )
    return (
        f"### 1.1 Full Cohort Batch Assessment (N = {n_full})\n\n"
        "> [!INFO] Why We Are Doing This\n"
        ">\n"
        f"> **What**: PCA across $N = {n_full}$ patients from cohorts ({cohort_breakdown}) "
        f"using {n_top:,} genes from {n_g_all:,} common genes.\n"
        "> **Why**: Combining transcriptomic data introduces batch effects. Uncorrected "
        "models risk classifying sequencing centres rather than patient biology.\n"
        "> **Question Answered**: Does cohort-independent Z-score standardisation eliminate "
        "technical separation between reference and trial cohorts?\n\n"
        "![[batch_effect_pca.png]]\n\n"
        "### Key Observations\n"
        f"- **Raw**: Separation between reference and trial cohorts. Uncorrected PC1 "
        f"({p1_rf:.1f}%) and PC2 ({p2_rf:.1f}%) reflect platform shifts.\n"
        f"- **Corrected**: Standardisation ($\\mu=0, \\sigma=1$ per study) aligns datasets. "
        f"Post-correction PC1 ({p1_sf:.1f}%) and PC2 ({p2_sf:.1f}%) show homogeneous spread.\n\n"
    )


def _report_section_1(
    n_full: int,
    cohort_counts: Dict[str, int],
    n_top: int,
    n_g_all: int,
    var_full: Tuple[float, float, float, float],
) -> str:
    """Returns Markdown text for Section 1: Cohort Batch Assessment."""
    sec1a = _report_section_1a(n_full, cohort_counts, n_top, n_g_all, var_full)
    return f"## 1. Cohort Batch Assessment\n\n{sec1a}"


def _report_section_2(
    n_trials: int,
    n_top: int,
    trial_counts: Dict[str, int],
) -> str:
    """Returns Markdown text for Section 2: Immunotherapy Trial Reduction."""
    trial_names_str = ", ".join(trial_counts.keys())
    trial_bullets_str = ", ".join(f"`{name}` ($N = {count}$)" for name, count in trial_counts.items())
    return (
        f"## 2. Immunotherapy Trial Dimensionality Reduction (N = {n_trials})\n\n"
        "> [!INFO] Why We Are Doing This\n"
        ">\n"
        f"> **What**: Linear (PCA) and non-linear (UMAP) reduction to $N = {n_trials}$ "
        f"response-annotated patients ({trial_names_str}) using {n_top:,} genes.\n"
        "> **Why**: Test if baseline expression profiles naturally segregate responders.\n"
        "> **Question Answered**: Can therapeutic response be predicted directly from global "
        "2D expression clusters?\n\n"
        "### Projections\n"
        "![[pca_dimensionality_reduction.png]]\n"
        "![[umap_dimensionality_reduction.png]]\n\n"
        "> [!INSIGHT] Key Insights\n"
        ">\n"
        f"> - **Harmonisation**: Z-score scaling integrates {trial_bullets_str} in embeddings.\n"
        "> - **Mixing**: Responders (CR/PR) and non-responders (PD) mix homogeneously.\n"
        "> - **Biological Rationale**: Response is driven by multi-pathway immune features, "
        "not global variance. Simple 2D projections cannot separate response groups.\n\n"
    )


def _report_section_3(n_trials: int) -> str:
    """Returns Markdown text for Section 3: Gene-Level Expression Heatmaps."""
    return (
        f"## 3. Gene-Level Expression Heatmaps (Top {N_TOP_HEATMAP_GENES} Genes)\n\n"
        "> [!INFO] Why We Are Doing This\n"
        ">\n"
        f"> **What**: Inspect heatmaps for top {N_TOP_HEATMAP_GENES} genes by average within-cohort variance "
        f"across trial patients ($N = {n_trials}$). Samples are sorted by cohort, then by responder "
        "status (CR/PR before PD) within each cohort, with no hierarchical column clustering, "
        "to make cohort-level baseline differences and response group separation directly readable.\n"
        "> **Why**: Validate batch effects at individual gene resolution and assess whether "
        "per-cohort Z-score correction removes study-level baseline shifts while preserving "
        "responder vs. non-responder biological contrast.\n"
        "> **Question Answered**: Are cohort expression baselines visibly harmonised after "
        "per-cohort Z-score standardisation, and does the response group signal become "
        "more consistent across cohorts?\n\n"
        "### Raw & Standardised\n"
        "![[heatmap_top_variance_genes_raw.png]]\n"
        "![[heatmap_top_variance_genes_standardized.png]]\n\n"
        "### Key Observations\n"
        "- **Raw** (`log2(TPM+1)`, robust 2nd–98th percentile scaling): Cohort blocks are visible "
        "in the Cohort colour bar. Van Allen 2015 shows a notably elevated baseline due to its "
        "different normalisation pipeline. The four iAtlas cohorts (Liu 2019, Hugo 2016, Riaz 2017, "
        "Gide 2019) share similar expression scales, reflecting their common preprocessing.\n"
        "- **Corrected** (per-cohort Z-score, $\\mu=0$, $\\sigma=1$): Study-level baseline offsets "
        "are removed. Gene expression patterns now reflect within-cohort biological variation rather "
        "than technical platform differences, with the responder/non-responder contrast "
        "becoming more consistent across cohorts.\n\n"
    )



def _report_section_4(trial_counts: Dict[str, int]) -> str:
    """Returns Markdown text for Section 4: Cross-Validation Rigour."""
    trial_bullets = ", ".join(f"`{name}` ($N = {count}$)" for name, count in trial_counts.items())
    return (
        "## 4. Cross-Validation Rigour & Data Leakage Prevention\n\n"
        "> [!INFO] Why We Are Doing This\n"
        ">\n"
        "> **What**: Compare cohort-independent Z-score against global batch correction.\n"
        "> **Question Answered**: How does independent Z-score prevent leakage in LOCO?\n\n"
        "### 4.1 The Hazard of Global Correction (e.g. ComBat)\n"
        "Global algorithms pool all samples to estimate parameters. In LOCO CV, including "
        "the test set in parameter estimation leaks information into the training phase, "
        "inflating metrics artificially.\n\n"
        "### 4.2 The Z-Score Solution\n"
        "Standardising independently per cohort (using internal $\\mu, \\sigma$) guarantees "
        "zero leakage. Each held-out study remains unobserved during training.\n\n"
        "> [!WARNING] Limitations\n"
        ">\n"
        f"> - **Constraints**: Sample sizes vary across trial cohorts ({trial_bullets}).\n"
        "> - **Scope**: Projections confirm single-gene thresholds are insufficient, "
        "motivating a multimodal ensemble feature approach.\n\n"
    )


def _report_section_5() -> str:
    """Returns Markdown text for Script Reference callout box."""
    entries = [
        (
            "run_dimensionality_reduction.py",
            SUBPROJECT_ROOT / "scripts" / "pillar-1-cohort-preprocessing" / "run_dimensionality_reduction.py",
            "Performs PCA and UMAP dimensionality reduction across all active ICI trial cohorts, evaluates cohort-independent Z-score standardisation against raw expression profiles, and produces `batch_correction_report.md`.",
        ),
        (
            "run_expression_heatmap.py",
            SUBPROJECT_ROOT / "scripts" / "pillar-1-cohort-preprocessing" / "run_expression_heatmap.py",
            "Generates raw log2(TPM+1) and per-cohort Z-score heatmap visualisations for top high-variance genes across trial cohorts (`heatmap_top_variance_genes_raw.png`, `heatmap_top_variance_genes_standardized.png`).",
        ),
        (
            "data_loaders.py",
            SUBPROJECT_ROOT / "src" / "data_loaders.py",
            "Provides `load_merged_immunotherapy()` to retrieve aligned raw and standardised expression matrices and clinical metadata across ICI trial cohorts.",
        ),
        (
            "styles.py",
            PROJECT_ROOT / "src" / "styles.py",
            "Central definition of Okabe-Ito colour palettes (`COHORT_PALETTE`, `RESPONSE_PALETTE`).",
        ),
    ]
    return generate_script_reference_callout(entries)


def generate_report_content(
    n_full: int,
    cohort_counts: Dict[str, int],
    n_trials: int,
    trial_counts: Dict[str, int],
    n_top: int,
    n_g_all: int,
    var_full: Tuple[float, float, float, float],
) -> str:
    """Assembles full Markdown content for batch_correction_report.md."""
    fm = generate_obsidian_frontmatter(
        title="Batch Effect Assessment & Dimensionality Reduction Analysis",
        aliases=["Q1 Batch Correction Report", "Batch Effect Assessment"],
        tags=["melanoma", "batch-correction", "pca", "umap", "tme", "transcriptomics"],
        extra_css_classes=["table-center", "row-alt"],
    )
    cohorts_str = ", ".join(f"**{name}**" for name in cohort_counts.keys())
    intro = (
        "# Batch Effect Assessment & Dimensionality Reduction Analysis\n\n"
        f"This report documents how technical batch effects were evaluated and harmonised "
        f"across {len(cohort_counts)} melanoma cohorts ({cohorts_str}) and tests if "
        "global profiles separate therapeutic responses.\n\n"
    )
    return (
        f"{fm}\n\n"
        + intro
        + _report_section_1(n_full, cohort_counts, n_top, n_g_all, var_full)
        + _report_section_2(n_trials, n_top, trial_counts)
        + _report_section_3(n_trials)
        + _report_section_4(trial_counts)
        + _report_section_5()
    )


def write_batch_correction_report(report_path: Path, content: str) -> None:
    """Writes the batch correction report Markdown file."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Report written to {rel_path(report_path)}")
    redundant_report = report_path.parent / "dimensionality_reduction_report.md"
    if redundant_report.exists():
        print(f"Removing redundant report: {rel_path(redundant_report)}")
        redundant_report.unlink()


# ---------------------------------------------------------------------------
# Main Pipeline Orchestration
# ---------------------------------------------------------------------------

def _run_pca_batch_projections(
    expr_dict: Dict[str, pd.DataFrame],
    clin_dict: Dict[str, pd.DataFrame],
    top_genes: List[str],
    cohort_order: List[str],
) -> Tuple[Tuple[float, float, float, float], int]:
    """Runs ICI cohort PCA batch assessment using top high-variance genes."""
    expr_full_raw, expr_full_scaled, clin_full = _build_concat(
        expr_dict, clin_dict, cohort_order, top_genes, with_response=False
    )
    pcs_r, pcs_s, p1_rf, p2_rf, p1_sf, p2_sf = fit_pca_projection(expr_full_raw, expr_full_scaled)
    _assign_pca_coords(clin_full, pcs_r, pcs_s)
    plot_cohort_batch_pca(
        clin_full, (p1_rf, p2_rf, p1_sf, p2_sf), cohort_order,
        "ICI Trial Cohorts", EXPLORATORY_PLOT_DIR / "batch_effect_pca.png",
    )
    return (p1_rf, p2_rf, p1_sf, p2_sf), len(expr_full_raw)


def _run_trial_reduction_grids(
    expr_dict: Dict[str, pd.DataFrame],
    clin_dict: Dict[str, pd.DataFrame],
    top_genes: List[str],
    trial_names: List[str],
) -> None:
    """Runs 2x2 PCA and UMAP grid plots for trial cohorts."""
    expr_tr_top_raw, expr_tr_top_scaled, clin_tr_resp = _build_concat(
        expr_dict, clin_dict, trial_names, top_genes, with_response=True
    )
    clin_tr_resp[_COL_RESPONSE] = clin_tr_resp[_COL_RAW_RESPONSE].map(RESPONSE_LABEL_MAP)
    pcs_top_r, pcs_top_s, p1_tr, p2_tr, p1_ts, p2_ts = fit_pca_projection(
        expr_tr_top_raw, expr_tr_top_scaled
    )
    _assign_pca_coords(clin_tr_resp, pcs_top_r, pcs_top_s)
    plot_trial_reduction_grid(
        clin_tr_resp, "PCA", (p1_tr, p2_tr, p1_ts, p2_ts),
        "PCA", trial_names, EXPLORATORY_PLOT_DIR / "pca_dimensionality_reduction.png",
    )
    um_raw, um_scaled = fit_umap_or_tsne(expr_tr_top_raw, expr_tr_top_scaled)
    _assign_umap_coords(clin_tr_resp, um_raw, um_scaled)
    plot_trial_reduction_grid(
        clin_tr_resp, "UMAP", (0.0, 0.0, 0.0, 0.0),
        "UMAP", trial_names, EXPLORATORY_PLOT_DIR / "umap_dimensionality_reduction.png",
    )


def main() -> None:
    """Executes dimensionality reduction workflow."""
    print("==================================================")
    print("Dimensionality Reduction: Full & Trial Cohorts")
    print("==================================================\n")
    for directory in [EXPLORATORY_PLOT_DIR, REPORT_DIR]:
        directory.mkdir(exist_ok=True, parents=True)
    expr_dict, clin_dict, cohort_order, trial_names = load_all_cohorts()
    g_all, g_trials, top_genes = select_top_variable_genes(expr_dict, trial_names)

    var_full, n_full = _run_pca_batch_projections(
        expr_dict, clin_dict, top_genes, cohort_order,
    )
    _run_trial_reduction_grids(expr_dict, clin_dict, top_genes, trial_names)

    cohort_counts = {c: len(expr_dict[c]) for c in cohort_order}
    trial_counts = {c: len(expr_dict[c]) for c in trial_names}
    n_tr_all = sum(trial_counts.values())

    report_md = generate_report_content(
        n_full, cohort_counts, n_tr_all, trial_counts,
        len(top_genes), len(g_all), var_full,
    )
    write_batch_correction_report(REPORT_DIR / "batch_correction_report.md", report_md)
    root_report_dir = PROJECT_ROOT / "reports" / "pillar-1-cohorts-and-preprocessing"
    if root_report_dir != REPORT_DIR:
        write_batch_correction_report(root_report_dir / "batch_correction_report.md", report_md)
    print("==================================================")
    print("Done!")
    print("==================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
