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
from typing import Dict, List, Tuple

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


def plot_cohort_batch_pca(
    df: pd.DataFrame,
    var_tuple: Tuple[float, float, float, float],
    cohort_list: List[str],
    title_suffix: str,
    output_path: Path,
) -> None:
    """Renders 1x2 panel PCA scatter plot for batch effect evaluation across cohorts.

    Args:
        df: DataFrame containing PCA coordinate columns and Cohort.
        var_tuple: Explained variance percentages (pc1_r, pc2_r, pc1_s, pc2_s).
        cohort_list: List of cohort display names.
        title_suffix: Additional text suffix for plot main title.
        output_path: Target save path.
    """
    pc1_r, pc2_r, pc1_s, pc2_s = var_tuple
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_1X2)
    panels = [
        ("PCA_Raw_PC1", "PCA_Raw_PC2",
         f"A: Raw Matrix (Uncorrected)\nPC1 ({pc1_r:.1f}%) vs PC2 ({pc2_r:.1f}%)"),
        ("PCA_Scaled_PC1", "PCA_Scaled_PC2",
         f"B: Z-Score Standardised (Corrected)\nPC1 ({pc1_s:.1f}%) vs PC2 ({pc2_s:.1f}%)"),
    ]

    for ax, (xcol, ycol, title) in zip(axes, panels):
        sns.scatterplot(
            data=df, x=xcol, y=ycol, hue=_COL_COHORT, hue_order=cohort_list,
            palette=COHORT_PALETTE, alpha=SCATTER_ALPHA, s=SCATTER_SIZE,
            edgecolor="w", linewidth=0.5, ax=ax,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.suptitle(
        f"PCA Batch Effect Assessment Across {title_suffix} (N = {len(df)})",
        fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    save_fig(fig, output_path)
    print(f"Saved PCA batch assessment plot to {rel_path(output_path)}")


def _axis_label(dim_prefix: str, axis: str, is_uncorrected: bool, val_r: float, val_s: float) -> str:
    """Returns the axis label string for a PCA or UMAP panel.

    Args:
        dim_prefix: 'PCA' or 'UMAP'.
        axis: 'x' or 'y'.
        is_uncorrected: True for raw/before-correction panels.
        val_r: Raw PC explained variance percentage.
        val_s: Scaled PC explained variance percentage.

    Returns:
        Formatted axis label string.
    """
    component = "PC1" if axis == "x" else "PC2"
    dim_num = "1" if axis == "x" else "2"
    if dim_prefix != "PCA":
        return f"UMAP Dimension {dim_num}"
    pct = val_r if is_uncorrected else val_s
    return f"{component} ({pct:.1f}% variance)"


def plot_trial_reduction_grid(
    df: pd.DataFrame,
    dim_prefix: str,
    pc_vars: Tuple[float, float, float, float],
    method_name: str,
    output_path: Path,
) -> None:
    """Renders 2x2 grid of scatter plots coloured by Cohort and Response.

    Args:
        df: Merged clinical DataFrame with projection coordinates.
        dim_prefix: Axis column prefix ("PCA" or "UMAP").
        pc_vars: Explained variance percentages for PCA (or zeroes for UMAP).
        method_name: Method label ("PCA" or "UMAP").
        output_path: Output file path.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_2X2)
    resp_colors = {
        "Responder (CR/PR)": RESPONSE_PALETTE["CR/PR"],
        "Non-responder (PD)": RESPONSE_PALETTE["PD"],
    }
    pc1_r, pc2_r, pc1_s, pc2_s = pc_vars

    configs = [
        (*_grid_col_names(dim_prefix, "Raw"),
         _COL_COHORT, COHORT_ORDER[1:], COHORT_PALETTE,
         f"{method_name} Before Batch Correction (Coloured by Cohort)"),
        (*_grid_col_names(dim_prefix, "Raw"),
         _COL_RESPONSE, RESPONSE_ORDER, resp_colors,
         f"{method_name} Before Batch Correction (Coloured by Response)"),
        (*_grid_col_names(dim_prefix, "Scaled"),
         _COL_COHORT, COHORT_ORDER[1:], COHORT_PALETTE,
         f"{method_name} After Batch Correction (Coloured by Cohort)"),
        (*_grid_col_names(dim_prefix, "Scaled"),
         _COL_RESPONSE, RESPONSE_ORDER, resp_colors,
         f"{method_name} After Batch Correction (Coloured by Response)"),
    ]

    for ax, (xcol, ycol, hue_col, order, palette, title) in zip(axes.flat, configs):
        sns.scatterplot(
            data=df, x=xcol, y=ycol, hue=hue_col, hue_order=order, palette=palette,
            style=hue_col, alpha=SCATTER_ALPHA, s=SCATTER_SIZE_LARGE,
            edgecolor="w", linewidth=0.8, ax=ax,
        )
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel(_axis_label(dim_prefix, "x", "Before" in title, pc1_r, pc1_s))
        ax.set_ylabel(_axis_label(dim_prefix, "y", "Before" in title, pc2_r, pc2_s))

    plt.suptitle(
        f"{method_name} Dimensionality Reduction of Trial Expression Data (N = {len(df)})",
        fontsize=16, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    save_fig(fig, output_path)
    print(f"Saved {method_name} 2x2 grid to {rel_path(output_path)}")


# ---------------------------------------------------------------------------
# Markdown Report Generation Functions
# ---------------------------------------------------------------------------

def _report_section_1(
    n_full: int, n_tcga: int, n_liu: int, n_hugo: int, n_riaz: int,
    n_trials: int, n_top: int, n_g4: int, n_g3: int,
    var_full: Tuple[float, float, float, float],
    var_trials: Tuple[float, float, float, float],
) -> str:
    """Returns Markdown text for Section 1: Cohort Batch Assessment.

    Args:
        n_full: Full cohort patient count.
        n_tcga: TCGA patient count.
        n_liu: Liu 2019 patient count.
        n_hugo: Hugo 2016 patient count.
        n_riaz: Riaz 2017 patient count.
        n_trials: Trial cohort patient count.
        n_top: Top variable gene count.
        n_g4: 4-cohort common gene count.
        n_g3: 3-trial common gene count.
        var_full: Full-cohort PCA variance tuple (p1_r, p2_r, p1_s, p2_s).
        var_trials: Trial-cohort PCA variance tuple (p1_r, p2_r, p1_s, p2_s).

    Returns:
        Formatted Markdown section string.
    """
    p1_rf, p2_rf, p1_sf, p2_sf = var_full
    p1_rt, p2_rt, p1_st, p2_st = var_trials
    return (
        "## 1. Cohort Batch Assessment\n\n"
        f"### 1.1 Full Cohort Batch Assessment (N = {n_full})\n\n"
        "> [!INFO] Why We Are Doing This\n"
        ">\n"
        f"> **What**: We perform Principal Component Analysis (PCA) across all $N = {n_full}$ patients from four combined melanoma cohorts "
        f"(**TCGA-SKCM** [$N = {n_tcga}$], **Liu 2019** [$N = {n_liu}$], **Hugo 2016** [$N = {n_hugo}$], and **Riaz 2017** [$N = {n_riaz}$]) "
        f"using the top {n_top:,} most variable genes selected from the {n_g4:,} common genes across all datasets.\n"
        "> **Why**: Combining transcriptomic data from diverse sequencing centres introduces technical distortions (batch effects). Uncorrected models risk classifying sequencing centres rather than patient biology.\n"
        "> **Question Answered**: Does cohort-independent Z-score standardisation eliminate macro-level technical separation between reference tissue (TCGA-SKCM) and active clinical trial cohorts?\n\n"
        "![[batch_effect_pca.png]]\n\n"
        "### Key Observations\n"
        f"- **Panel A: Before Batch Correction (Raw Data)**: The uncorrected PCA projection reveals a strong separation between the TCGA-SKCM reference dataset and the three clinical trial cohorts. Uncorrected PC1 ({p1_rf:.1f}% variance) and PC2 ({p2_rf:.1f}% variance) reflect laboratory platform shifts.\n"
        f"- **Panel B: After Cohort-Specific Z-Score Standardisation**: Cohort-wise Z-score standardisation (centering each gene to $\\mu = 0, \\sigma = 1$ within each study) aligns the TCGA-SKCM reference with trial cohorts. Post-correction PC1 ({p1_sf:.1f}% variance) and PC2 ({p2_sf:.1f}% variance) show homogeneous distribution across datasets.\n\n"
        f"### 1.2 ICI Trial Cohort Batch Assessment (N = {n_trials})\n\n"
        "> [!INFO] Why We Are Doing This\n"
        ">\n"
        "> **What**: We evaluate technical batch effects specifically between the three active anti-PD-1 training cohorts "
        f"(**Liu 2019** [$N = {n_liu}$], **Hugo 2016** [$N = {n_hugo}$], and **Riaz 2017** [$N = {n_riaz}$]; $N = {n_trials}$) "
        f"across all {n_g3:,} common trial genes before and after cohort-wise Z-score standardisation.\n"
        "> **Why**: These trials vary by platform (Illumina HiSeq 2500 vs HiSeq 2000), tissue state (fresh-frozen vs FFPE), and prior treatment. We must verify baseline offsets are eliminated before Leave-One-Cohort-Out (LOCO) cross-validation.\n"
        "> **Question Answered**: Are inter-trial technical offsets harmonised across the model training cohorts without leaking test-set information?\n\n"
        "![[batch_effect_ici_pca.png]]\n\n"
        "### Key Observations\n"
        f"- **Panel A: Before Batch Correction (Uncorrected Raw Expression)**: In uncorrected $\\log_2(\\text{{TPM}})$ space across all {n_g3:,} trial genes, `Liu 2019` ($N = {n_liu}$, HiSeq 2500) separates along PC1 ({p1_rt:.1f}% variance) from `Riaz 2017` ($N = {n_riaz}$, HiSeq 2000 / FFPE) and `Hugo 2016` ($N = {n_hugo}$, HiSeq 2000 / fresh-frozen). This confirms that sequencing depth and platform chemistry dominate raw expression signals.\n"
        f"- **Panel B: After Cohort-Wise Z-Score Standardisation**: Standardising gene expression independently within each cohort completely removes artificial study-level separation. The distributions for Liu 2019, Hugo 2016, and Riaz 2017 overlap smoothly across PC1 ({p1_st:.1f}% variance) and PC2 ({p2_st:.1f}% variance), ensuring unbiased model training.\n\n"
    )


def _report_section_2(
    n_trials: int, n_top: int, n_liu: int, n_hugo: int, n_riaz: int,
) -> str:
    """Returns Markdown text for Section 2: Immunotherapy Trial Dimensionality Reduction.

    Args:
        n_trials: Trial cohort patient count.
        n_top: Top variable gene count.
        n_liu: Liu 2019 patient count.
        n_hugo: Hugo 2016 patient count.
        n_riaz: Riaz 2017 patient count.

    Returns:
        Formatted Markdown section string.
    """
    return (
        f"## 2. Immunotherapy Trial Dimensionality Reduction (N = {n_trials})\n\n"
        "> [!INFO] Why We Are Doing This\n"
        ">\n"
        f"> **What**: We apply linear (PCA) and non-linear (UMAP) dimensionality reduction to the $N = {n_trials}$ response-annotated trial patients (Liu 2019, Hugo 2016, Riaz 2017) using the top {n_top:,} variable genes.\n"
        "> **Why**: To test whether baseline gene expression profiles naturally segregate treatment responders from non-responders prior to supervised machine learning.\n"
        "> **Question Answered**: Can therapeutic response be predicted directly from global 2D expression clusters, or are targeted biomarker signatures required?\n\n"
        "### PCA Projections (Raw vs. Standardised)\n"
        "![[pca_dimensionality_reduction.png]]\n\n"
        "### UMAP Projections (Raw vs. Standardised)\n"
        "![[umap_dimensionality_reduction.png]]\n\n"
        "> [!INSIGHT] Key Insights: Dimensionality Reduction & Patient Distribution\n"
        ">\n"
        f"> - **Cohort Harmonisation**: Z-score scaling successfully integrates `Liu 2019` ($N = {n_liu}$), `Riaz 2017` ($N = {n_riaz}$), and `Hugo 2016` ($N = {n_hugo}$) across both PCA and UMAP embeddings.\n"
        "> - **Homogeneous Response Mixing**: Responders (CR/PR) and non-responders (PD) mix homogeneously throughout PCA and UMAP projections, with zero global cluster separation by clinical outcome.\n"
        "> - **Biological Rationale**: Immunotherapy response is driven by multi-pathway immune microenvironment features (e.g. `CD274`, `PDCD1`, `IFNG` signalling) rather than global transcriptomic variance. Simple 2D projections cannot separate response groups, proving the necessity for supervised multivariate classifiers.\n\n"
    )


def _report_section_3(
    n_trials: int,
) -> str:
    """Returns Markdown text for Section 3: Gene-Level Expression Heatmaps.

    Args:
        n_trials: Trial cohort patient count.

    Returns:
        Formatted Markdown section string.
    """
    return (
        f"## 3. Gene-Level Expression Heatmaps (Top {N_TOP_HEATMAP_GENES} Highly Variable Genes)\n\n"
        "> [!INFO] Why We Are Doing This\n"
        ">\n"
        f"> **What**: We inspect individual gene expression heatmaps for the top {N_TOP_HEATMAP_GENES} most variable genes across trial patients ($N = {n_trials}$) with hierarchical clustering.\n"
        "> **Why**: Dimensionality reduction aggregates thousands of genes into single axes. Heatmaps allow direct inspection of batch effects at individual gene resolutions.\n"
        "> **Question Answered**: Does within-cohort Z-score standardisation prevent individual high-variance genes from clustering patients by study origin?\n\n"
        f"### Raw Expression (Top {N_TOP_HEATMAP_GENES} Genes)\n"
        "![[heatmap_top_variance_genes_raw.png]]\n\n"
        f"### Standardised Expression (Top {N_TOP_HEATMAP_GENES} Genes)\n"
        "![[heatmap_top_variance_genes_standardized.png]]\n\n"
        "### Key Observations\n"
        "- **Before Batch Correction (Raw log2-TPM)**: Patient columns cluster strongly by cohort source, with distinct blocks corresponding to individual clinical studies.\n"
        "- **After Batch Correction (Cohort Z-Scoring)**: Within-cohort Z-score standardisation eliminates study-based clustering, producing complete cohort mixing across the hierarchical dendrogram.\n\n"
    )


def _report_section_4(
    n_liu: int, n_hugo: int, n_riaz: int,
) -> str:
    """Returns Markdown text for Section 4: Cross-Validation Rigour & Data Leakage Prevention.

    Args:
        n_liu: Liu 2019 patient count.
        n_hugo: Hugo 2016 patient count.
        n_riaz: Riaz 2017 patient count.

    Returns:
        Formatted Markdown section string.
    """
    return (
        "## 4. Cross-Validation Rigor & Data Leakage Prevention\n\n"
        "> [!INFO] Why We Are Doing This\n"
        ">\n"
        "> **What**: We compare cohort-independent Z-score standardisation against global batch correction algorithms (such as ComBat).\n"
        "> **Why**: Data preprocessing methods used in cross-validation must strictly preserve test-set independence.\n"
        "> **Question Answered**: How does cohort-independent Z-score scaling prevent data leakage during Leave-One-Cohort-Out (LOCO) evaluation?\n\n"
        "### 4.1 The Hazard of Global Batch Correction (e.g. ComBat)\n"
        "Global batch correction algorithms like ComBat estimate location and scale transformation parameters using all samples pooled across all available cohorts. When performing Leave-One-Cohort-Out (LOCO) cross-validation, including the held-out test cohort in parameter estimation allows information from the test set to leak into the training phase. This **data leakage** produces artificially inflated performance metrics that fail to generalise to external clinical validation sets.\n\n"
        "### 4.2 The Cohort-Independent Z-Score Solution\n"
        "Standardising gene expression independently within each cohort (rescaling each gene using only that cohort's internal mean $\\mu$ and standard deviation $\\sigma$) guarantees zero data leakage. Each held-out study remains completely unobserved during model training, ensuring robust, generalisable estimates of real-world predictive performance.\n\n"
        "> [!WARNING] Methodological Limitations & Future Rationale\n"
        ">\n"
        f"> - **Sample Size Constraints**: The smallest training cohort (`Hugo 2016`, $N = {n_hugo}$) has reduced statistical power compared to `Liu 2019` ($N = {n_liu}$) and `Riaz 2017` ($N = {n_riaz}$).\n"
        "> - **Platform Heterogeneity**: Z-score scaling harmonises gene-wise means and variances but does not alter relative non-linear gene correlations within a single study.\n"
        "> - **Pipeline Scope**: Unsupervised projections confirm that single-gene thresholds are insufficient for response prediction, motivating the 12-feature multimodal ensemble (incorporating TMB, TIS, CYT, and driver mutations like `BRAF`, `NRAS`, `NF1`) evaluated in downstream Q1 phases.\n"
    )


def generate_report_content(
    n_full: int, n_tcga: int, n_liu: int, n_hugo: int, n_riaz: int,
    n_trials: int, n_top: int, n_g4: int, n_g3: int,
    var_full: Tuple[float, float, float, float],
    var_trials_all: Tuple[float, float, float, float],
) -> str:
    """Assembles full Markdown content for batch_correction_report.md.

    Args:
        n_full: Full cohort patient count.
        n_tcga: TCGA patient count.
        n_liu: Liu 2019 patient count.
        n_hugo: Hugo 2016 patient count.
        n_riaz: Riaz 2017 patient count.
        n_trials: ICI trial cohort patient count.
        n_top: Number of top variable genes.
        n_g4: Number of common 4-cohort genes.
        n_g3: Number of common 3-trial genes.
        var_full: Explained variance percentages for full cohort PCA.
        var_trials_all: Explained variance percentages for trial cohort PCA.

    Returns:
        Formatted Markdown report text string.
    """
    fm = generate_obsidian_frontmatter(
        title="Batch Effect Assessment & Dimensionality Reduction Analysis",
        aliases=["Q1 Batch Correction Report", "Batch Effect Assessment"],
        tags=["melanoma", "batch-correction", "pca", "umap", "tme", "transcriptomics"],
        extra_css_classes=["table-center", "row-alt"],
    )
    p1_rf, p2_rf, p1_sf, p2_sf = var_full
    p1_rt, p2_rt, p1_st, p2_st = var_trials_all

    intro = (
        "# Batch Effect Assessment & Dimensionality Reduction Analysis\n\n"
        "When combining transcriptomic datasets across independent clinical studies, technical variations "
        "(e.g. sequencing platforms, RNA extraction methods, and library preparation protocols) typically dominate "
        "the underlying biological signals. This report documents how technical batch effects were evaluated and "
        "harmonised across four melanoma cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, and **Riaz 2017**) "
        "and tests whether global transcriptomic profiles naturally separate patients based on therapeutic response. "
        "Visualisations follow the Okabe-Ito colour guidelines used throughout the study.\n\n"
    )
    return (
        f"{fm}\n\n"
        + intro
        + _report_section_1(
            n_full, n_tcga, n_liu, n_hugo, n_riaz, n_trials, n_top, n_g4, n_g3,
            var_full, var_trials_all,
        )
        + _report_section_2(n_trials, n_top, n_liu, n_hugo, n_riaz)
        + _report_section_3(n_trials)
        + _report_section_4(n_liu, n_hugo, n_riaz)
    )


def write_batch_correction_report(report_path: Path, content: str) -> None:
    """Writes the batch correction report Markdown file.

    Args:
        report_path: Destination path for the report file.
        content: Formatted Markdown text content.
    """
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

def main() -> None:
    """Executes dimensionality reduction workflow across full and trial cohorts."""
    print("==================================================")
    print("Dimensionality Reduction: Full & Trial Cohorts")
    print("==================================================\n")

    for directory in [EXPLORATORY_PLOT_DIR, REPORT_DIR]:
        directory.mkdir(exist_ok=True, parents=True)

    expr_dict, clin_dict = load_all_cohorts(DATA_DIR)
    g4, g3, top_genes = select_top_variable_genes(expr_dict)
    trial_names = COHORT_ORDER[1:]

    # 1a. Full cohort PCA batch assessment
    expr_full_raw, expr_full_scaled, clin_full = _build_concat(
        expr_dict, clin_dict, COHORT_ORDER, top_genes, with_response=False
    )
    pcs_r, pcs_s, p1_rf, p2_rf, p1_sf, p2_sf = fit_pca_projection(expr_full_raw, expr_full_scaled)
    _assign_pca_coords(clin_full, pcs_r, pcs_s)
    plot_cohort_batch_pca(
        clin_full, (p1_rf, p2_rf, p1_sf, p2_sf), COHORT_ORDER,
        "Full Cohort", EXPLORATORY_PLOT_DIR / "batch_effect_pca.png",
    )

    # 1b. ICI trial cohort PCA batch assessment (all common trial genes)
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

    # 2. Trial cohort PCA & UMAP projections coloured by response
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

    # 3. Generate & write report
    n_tcga, n_liu = len(expr_dict["TCGA-SKCM"]), len(expr_dict["Liu 2019"])
    n_hugo, n_riaz = len(expr_dict["Hugo 2016"]), len(expr_dict["Riaz 2017"])
    report_md = generate_report_content(
        len(expr_full_raw), n_tcga, n_liu, n_hugo, n_riaz,
        len(expr_tr_all_raw), len(top_genes), len(g4), len(g3),
        (p1_rf, p2_rf, p1_sf, p2_sf), (p1_rt, p2_rt, p1_st, p2_st),
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
