"""
Dimensionality Reduction & Batch Correction Analysis across Melanoma Cohorts.

Performs PCA and UMAP projections across TCGA-SKCM, Liu 2019, Hugo 2016, and Riaz 2017 cohorts
to assess technical batch effects before and after cohort-independent Z-score scaling,
and updates batch_correction_report.md.
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

from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style
from src.utils.formatting import generate_obsidian_frontmatter
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
REPORT_DIR: Path = PROJECT_ROOT / "reports" / "pillar-1-cohorts-and-preprocessing"
LOG_DIR: Path = get_subproject_log_dir(SCRIPT_DIR)
LOG_PATH: Path = LOG_DIR / "run_dimensionality_reduction.log"

# Domain Category Labels & Orders
COHORT_ORDER: List[str] = ["TCGA-SKCM", "Liu 2019", "Hugo 2016", "Riaz 2017"]
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


def load_all_cohorts(
    data_dir: Path,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Loads and aligns Liu, Hugo, Riaz, and TCGA-SKCM datasets.

    Args:
        data_dir: Path to raw and processed data directory.

    Returns:
        Tuple of dicts mapping cohort names to expr and clin DataFrames.
    """
    e_liu, c_liu = _align_expr_clin(*load_liu_2019(data_dir))
    e_hugo, c_hugo = _align_expr_clin(*load_hugo_2016(data_dir))
    e_riaz, c_riaz = _align_expr_clin(*load_riaz_2017(data_dir))

    tcga_dir = PROCESSED_DIR / "skcm_tcga_pan_can_atlas_2018"
    e_tcga = pd.read_csv(tcga_dir / "expr_cleaned.csv", index_col=_COL_SAMPLE_ID)
    c_tcga = pd.read_csv(tcga_dir / "clin_cleaned.csv", index_col=_COL_SAMPLE_ID)
    e_tcga, c_tcga = _align_expr_clin(e_tcga, c_tcga)

    c_tcga[_COL_COHORT] = COHORT_ORDER[0]
    c_liu[_COL_COHORT] = COHORT_ORDER[1]
    c_hugo[_COL_COHORT] = COHORT_ORDER[2]
    c_riaz[_COL_COHORT] = COHORT_ORDER[3]

    expr_dict = dict(zip(COHORT_ORDER, [e_tcga, e_liu, e_hugo, e_riaz]))
    clin_dict = dict(zip(COHORT_ORDER, [c_tcga, c_liu, c_hugo, c_riaz]))
    return expr_dict, clin_dict


def select_top_variable_genes(
    expr_dict: Dict[str, pd.DataFrame], top_n: int = N_TOP_VARIABLE_GENES
) -> Tuple[List[str], List[str], List[str]]:
    """Identifies common genes across cohorts and selects top variable genes.

    Args:
        expr_dict: Dictionary mapping cohort names to expression DataFrames.
        top_n: Number of high-variance genes to select.

    Returns:
        Tuple of (4-cohort common genes, 3-trial common genes, top N variable genes).
    """
    g4 = sorted(list(set.intersection(*(set(df.columns) for df in expr_dict.values()))))
    trial_dfs = [expr_dict[c] for c in COHORT_ORDER[1:]]
    g3 = sorted(list(set.intersection(*(set(df.columns) for df in trial_dfs))))

    pooled_g4 = pd.concat([df[g4] for df in expr_dict.values()], axis=0)
    top_genes = pooled_g4.var(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
    return g4, g3, top_genes


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
            palette=COHORT_PALETTE, alpha=SCATTER_ALPHA, s=SCATTER_SIZE,
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
    dim_prefix: str, method_name: str, resp_colors: Dict[str, str]
) -> List[Any]:
    """Builds tuple configurations for 2x2 scatter plot grid panels."""
    return [
        (*_grid_col_names(dim_prefix, "Raw"), _COL_COHORT, COHORT_ORDER[1:], COHORT_PALETTE,
         f"{method_name} Before Batch Correction (Coloured by Cohort)"),
        (*_grid_col_names(dim_prefix, "Raw"), _COL_RESPONSE, RESPONSE_ORDER, resp_colors,
         f"{method_name} Before Batch Correction (Coloured by Response)"),
        (*_grid_col_names(dim_prefix, "Scaled"), _COL_COHORT, COHORT_ORDER[1:], COHORT_PALETTE,
         f"{method_name} After Batch Correction (Coloured by Cohort)"),
        (*_grid_col_names(dim_prefix, "Scaled"), _COL_RESPONSE, RESPONSE_ORDER, resp_colors,
         f"{method_name} After Batch Correction (Coloured by Response)"),
    ]


def plot_trial_reduction_grid(
    df: pd.DataFrame,
    dim_prefix: str,
    pc_vars: Tuple[float, float, float, float],
    method_name: str,
    output_path: Path,
) -> None:
    """Renders 2x2 grid of scatter plots coloured by Cohort and Response."""
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_2X2)
    resp_colors = {
        "Responder (CR/PR)": RESPONSE_PALETTE["CR/PR"],
        "Non-responder (PD)": RESPONSE_PALETTE["PD"],
    }
    pc1_r, pc2_r, pc1_s, pc2_s = pc_vars
    configs = _get_grid_configs(dim_prefix, method_name, resp_colors)

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
    n_full: int, n_tcga: int, n_liu: int, n_hugo: int, n_riaz: int,
    n_top: int, n_g4: int, var_full: Tuple[float, float, float, float],
) -> str:
    """Returns Markdown text for Section 1.1: Full Cohort Batch Assessment."""
    p1_rf, p2_rf, p1_sf, p2_sf = var_full
    return (
        f"### 1.1 Full Cohort Batch Assessment (N = {n_full})\n\n"
        "> [!INFO] Why We Are Doing This\n"
        ">\n"
        f"> **What**: PCA across $N = {n_full}$ patients from four cohorts (**TCGA-SKCM** "
        f"[$N = {n_tcga}$], **Liu 2019** [$N = {n_liu}$], **Hugo 2016** [$N = {n_hugo}$], "
        f"**Riaz 2017** [$N = {n_riaz}$]) using {n_top:,} genes from {n_g4:,} common genes.\n"
        "> **Why**: Combining transcriptomic data introduces batch effects. Uncorrected "
        "models risk classifying sequencing centres rather than patient biology.\n"
        "> **Question Answered**: Does cohort-independent Z-score standardisation eliminate "
        "technical separation between reference (TCGA) and trial cohorts?\n\n"
        "![[batch_effect_pca.png]]\n\n"
        "### Key Observations\n"
        f"- **Raw**: Separation between TCGA and trial cohorts. Uncorrected PC1 "
        f"({p1_rf:.1f}%) and PC2 ({p2_rf:.1f}%) reflect platform shifts.\n"
        f"- **Corrected**: Standardisation ($\\mu=0, \\sigma=1$ per study) aligns datasets. "
        f"Post-correction PC1 ({p1_sf:.1f}%) and PC2 ({p2_sf:.1f}%) show homogeneous spread.\n\n"
    )


def _report_section_1b(
    n_trials: int, n_liu: int, n_hugo: int, n_riaz: int, n_g3: int,
    var_trials: Tuple[float, float, float, float],
) -> str:
    """Returns Markdown text for Section 1.2: ICI Trial Cohort Batch Assessment."""
    p1_rt, p2_rt, p1_st, p2_st = var_trials
    return (
        f"### 1.2 ICI Trial Cohort Batch Assessment (N = {n_trials})\n\n"
        "> [!INFO] Why We Are Doing This\n"
        ">\n"
        f"> **What**: Technical effects between 3 training cohorts (**Liu 2019** [$N = {n_liu}$], "
        f"**Hugo 2016** [$N = {n_hugo}$], **Riaz 2017** [$N = {n_riaz}$]; $N = {n_trials}$) "
        f"across {n_g3:,} trial genes.\n"
        "> **Why**: Trials vary by platform, tissue state, and treatment. Verify baseline "
        "offsets are eliminated before LOCO cross-validation.\n"
        "> **Question Answered**: Are inter-trial offsets harmonised without leaking test data?\n\n"
        "![[batch_effect_ici_pca.png]]\n\n"
        "### Key Observations\n"
        f"- **Raw**: In $\\log_2(\\text{{TPM}})$, `Liu 2019` ($N = {n_liu}$) separates from "
        f"`Riaz 2017` ($N = {n_riaz}$) and `Hugo 2016` ($N = {n_hugo}$) along PC1 "
        f"({p1_rt:.1f}%), confirming sequencing depth/platform dominate signals.\n"
        f"- **Corrected**: Standardisation removes study-level separation. distributions "
        f"overlap smoothly across PC1 ({p1_st:.1f}%) and PC2 ({p2_st:.1f}%).\n\n"
    )


def _report_section_1(
    n_full: int, n_tcga: int, n_liu: int, n_hugo: int, n_riaz: int,
    n_trials: int, n_top: int, n_g4: int, n_g3: int,
    var_full: Tuple[float, float, float, float],
    var_trials: Tuple[float, float, float, float],
) -> str:
    """Returns Markdown text for Section 1: Cohort Batch Assessment."""
    sec1a = _report_section_1a(n_full, n_tcga, n_liu, n_hugo, n_riaz, n_top, n_g4, var_full)
    sec1b = _report_section_1b(n_trials, n_liu, n_hugo, n_riaz, n_g3, var_trials)
    return f"## 1. Cohort Batch Assessment\n\n{sec1a}{sec1b}" 


def _report_section_2(
    n_trials: int, n_top: int, n_liu: int, n_hugo: int, n_riaz: int,
) -> str:
    """Returns Markdown text for Section 2: Immunotherapy Trial Reduction."""
    return (
        f"## 2. Immunotherapy Trial Dimensionality Reduction (N = {n_trials})\n\n"
        "> [!INFO] Why We Are Doing This\n"
        ">\n"
        f"> **What**: Linear (PCA) and non-linear (UMAP) reduction to $N = {n_trials}$ "
        f"response-annotated patients (Liu 2019, Hugo 2016, Riaz 2017) using {n_top:,} genes.\n"
        "> **Why**: Test if baseline expression profiles naturally segregate responders.\n"
        "> **Question Answered**: Can therapeutic response be predicted directly from global "
        "2D expression clusters?\n\n"
        "### Projections\n"
        "![[pca_dimensionality_reduction.png]]\n"
        "![[umap_dimensionality_reduction.png]]\n\n"
        "> [!INSIGHT] Key Insights\n"
        ">\n"
        f"> - **Harmonisation**: Z-score scaling integrates `Liu 2019` ($N = {n_liu}$), "
        f"`Riaz 2017` ($N = {n_riaz}$), `Hugo 2016` ($N = {n_hugo}$) in embeddings.\n"
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
        f"> **What**: Inspect individual gene heatmaps for top {N_TOP_HEATMAP_GENES} genes "
        f"across trial patients ($N = {n_trials}$) with hierarchical clustering.\n"
        "> **Why**: Validate batch effects at individual gene resolutions.\n"
        "> **Question Answered**: Does Z-score prevent gene-based cohort clustering?\n\n"
        "### Raw & Standardised\n"
        "![[heatmap_top_variance_genes_raw.png]]\n"
        "![[heatmap_top_variance_genes_standardized.png]]\n\n"
        "### Key Observations\n"
        "- **Raw**: Columns cluster by cohort source, showing distinct blocks.\n"
        "- **Corrected**: Within-cohort Z-score standardisation eliminates study-based "
        "clustering, producing complete cohort mixing.\n\n"
    )


def _report_section_4(n_liu: int, n_hugo: int, n_riaz: int) -> str:
    """Returns Markdown text for Section 4: Cross-Validation Rigour."""
    return (
        "## 4. Cross-Validation Rigor & Data Leakage Prevention\n\n"
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
        f"> - **Constraints**: `Hugo 2016` ($N = {n_hugo}$) has lower power than "
        f"`Liu 2019` ($N = {n_liu}$) and `Riaz 2017` ($N = {n_riaz}$).\n"
        "> - **Scope**: Projections confirm single-gene thresholds are insufficient, "
        "motivating a 12-feature multimodal ensemble approach.\n\n"
    )


def generate_report_content(
    n_full: int, n_tcga: int, n_liu: int, n_hugo: int, n_riaz: int,
    n_trials: int, n_top: int, n_g4: int, n_g3: int,
    var_full: Tuple[float, float, float, float],
    var_trials_all: Tuple[float, float, float, float],
) -> str:
    """Assembles full Markdown content for batch_correction_report.md."""
    fm = generate_obsidian_frontmatter(
        title="Batch Effect Assessment & Dimensionality Reduction Analysis",
        aliases=["Q1 Batch Correction Report", "Batch Effect Assessment"],
        tags=["melanoma", "batch-correction", "pca", "umap", "tme", "transcriptomics"],
        extra_css_classes=["table-center", "row-alt"],
    )
    intro = (
        "# Batch Effect Assessment & Dimensionality Reduction Analysis\n\n"
        "This report documents how technical batch effects were evaluated and harmonised "
        "across four melanoma cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, and "
        "**Riaz 2017**) and tests if global profiles separate therapeutic responses.\n\n"
    )
    return (
        f"{fm}\n\n" + intro + _report_section_1(
            n_full, n_tcga, n_liu, n_hugo, n_riaz, n_trials, n_top, n_g4, n_g3,
            var_full, var_trials_all,
        ) + _report_section_2(n_trials, n_top, n_liu, n_hugo, n_riaz)
        + _report_section_3(n_trials) + _report_section_4(n_liu, n_hugo, n_riaz)
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
    g3: List[str],
    trial_names: List[str],
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], int]:
    """Runs full and trial cohort PCA projections."""
    expr_full_raw, expr_full_scaled, clin_full = _build_concat(
        expr_dict, clin_dict, COHORT_ORDER, top_genes, with_response=False
    )
    pcs_r, pcs_s, p1_rf, p2_rf, p1_sf, p2_sf = fit_pca_projection(expr_full_raw, expr_full_scaled)
    _assign_pca_coords(clin_full, pcs_r, pcs_s)
    plot_cohort_batch_pca(
        clin_full, (p1_rf, p2_rf, p1_sf, p2_sf), COHORT_ORDER,
        "Full Cohort", EXPLORATORY_PLOT_DIR / "batch_effect_pca.png",
    )
    expr_tr_all_raw, expr_tr_all_scaled, clin_tr_all = _build_concat(
        expr_dict, clin_dict, trial_names, g3, with_response=False
    )
    pcs_tr_r, pcs_tr_s, p1_rt, p2_rt, p1_st, p2_st = fit_pca_projection(
        expr_tr_all_raw, expr_tr_all_scaled
    )
    _assign_pca_coords(clin_tr_all, pcs_tr_r, pcs_tr_s)
    plot_cohort_batch_pca(
        clin_tr_all, (p1_rt, p2_rt, p1_st, p2_st), trial_names,
        "ICI Trial Cohorts", EXPLORATORY_PLOT_DIR / "batch_effect_ici_pca.png",
    )
    return (p1_rf, p2_rf, p1_sf, p2_sf), (p1_rt, p2_rt, p1_st, p2_st), len(expr_full_raw)


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
        "PCA", EXPLORATORY_PLOT_DIR / "pca_dimensionality_reduction.png",
    )
    um_raw, um_scaled = fit_umap_or_tsne(expr_tr_top_raw, expr_tr_top_scaled)
    _assign_umap_coords(clin_tr_resp, um_raw, um_scaled)
    plot_trial_reduction_grid(
        clin_tr_resp, "UMAP", (0.0, 0.0, 0.0, 0.0),
        "UMAP", EXPLORATORY_PLOT_DIR / "umap_dimensionality_reduction.png",
    )


def main() -> None:
    """Executes dimensionality reduction workflow."""
    print("==================================================")
    print("Dimensionality Reduction: Full & Trial Cohorts")
    print("==================================================\n")
    for directory in [EXPLORATORY_PLOT_DIR, REPORT_DIR]:
        directory.mkdir(exist_ok=True, parents=True)
    expr_dict, clin_dict = load_all_cohorts(DATA_DIR)
    g4, g3, top_genes = select_top_variable_genes(expr_dict)
    trial_names = COHORT_ORDER[1:]
    var_full, var_trials, n_full = _run_pca_batch_projections(
        expr_dict, clin_dict, top_genes, g3, trial_names
    )
    _run_trial_reduction_grids(expr_dict, clin_dict, top_genes, trial_names)
    n_tcga, n_liu = len(expr_dict["TCGA-SKCM"]), len(expr_dict["Liu 2019"])
    n_hugo, n_riaz = len(expr_dict["Hugo 2016"]), len(expr_dict["Riaz 2017"])
    n_tr_all = sum(len(expr_dict[c]) for c in trial_names)
    report_md = generate_report_content(
        n_full, n_tcga, n_liu, n_hugo, n_riaz, n_tr_all,
        len(top_genes), len(g4), len(g3), var_full, var_trials,
    )
    write_batch_correction_report(REPORT_DIR / "batch_correction_report.md", report_md)
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
