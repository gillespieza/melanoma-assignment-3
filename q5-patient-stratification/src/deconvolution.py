"""Transcriptomic cell-type deconvolution and Macrophage STV calculation module for Q5.

Provides helper functions to calculate the M1/M2 Macrophage Signature Transcript Vector (STV)
and estimate relative cell-type fractions for patient stratification with strict coverage validation.
"""

from pathlib import Path
from typing import Dict, Tuple
import warnings
import numpy as np
import pandas as pd

from q5_constants import CELL_TYPE_MARKERS, M1_M2_NEUTRAL_RATIO

# Minimum number of valid marker genes required per cell type panel to avoid single-gene noise
MIN_MARKERS_THRESHOLD: int = 2


# ---------------------------------------------------------------------------
# Private Helpers — Macrophage STV Computation
# ---------------------------------------------------------------------------

def _load_and_partition_stv_weights(
    stv_path: Path, expr_genes: set
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Load the STV weight matrix and partition genes into M1, M2, and combined sets.

    Args:
        stv_path: Path to m1_m2_stv.csv containing 'Gene' and 'both_M1M2' columns.
        expr_genes: Set of gene names present in the patient expression matrix.

    Returns:
        Tuple of (valid_weights, m1_genes, m2_genes) where:
        - valid_weights: All STV genes overlapping with expr_genes → signed weight.
        - m1_genes: Subset with positive weight (pro-inflammatory M1 programme).
        - m2_genes: Subset with absolute negative weight (immunosuppressive M2 programme).
    """
    df_stv = pd.read_csv(stv_path)
    weights = dict(zip(df_stv["Gene"], df_stv["both_M1M2"]))
    valid_weights = {g: w for g, w in weights.items() if g in expr_genes}
    m1_genes = {g: w for g, w in valid_weights.items() if w > 0}
    m2_genes = {g: abs(w) for g, w in valid_weights.items() if w < 0}
    return valid_weights, m1_genes, m2_genes


def _compute_weighted_score(
    df_expr: pd.DataFrame, gene_weights: Dict[str, float]
) -> pd.Series:
    """Compute a per-sample weighted dot-product score for a given gene-weight set.

    If the gene-weight set is empty (no overlapping genes), returns a zero Series
    to prevent downstream NaN propagation.

    Args:
        df_expr: Expression DataFrame (samples x genes).
        gene_weights: Mapping of gene name to numeric weight.

    Returns:
        Per-sample score Series, indexed identically to df_expr.
    """
    if not gene_weights:
        return pd.Series(0.0, index=df_expr.index)
    weights_series = pd.Series(gene_weights)
    return df_expr[weights_series.index].dot(weights_series)


def _compute_m1_m2_ratio(m1_score: pd.Series, m2_score: pd.Series) -> pd.Series:
    """Compute the M1/(M1+M2) polarisation ratio with a biological boundary fallback.

    Biological Assumption (Zero-Infiltration Boundary Condition):
    When both M1 and M2 scores equal 0 (immune desert with unmeasurable macrophage
    infiltration), division yields 0/0. We assign M1_M2_NEUTRAL_RATIO (0.5) to prevent
    NaN propagation and maintain a balanced polarisation baseline that does not falsely
    bias downstream clustering toward either M1-hot or M2-suppressive extremes.

    Args:
        m1_score: Per-sample M1 macrophage activation score.
        m2_score: Per-sample M2 macrophage immunosuppression score.

    Returns:
        Per-sample M1/(M1+M2) ratio, with 0.5 assigned where both scores are zero.
    """
    denom = m1_score + m2_score
    ratio = np.where(denom > 0, m1_score / denom, M1_M2_NEUTRAL_RATIO)
    return pd.Series(ratio, index=m1_score.index)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_macrophage_stv(
    df_expr: pd.DataFrame, stv_path: Path
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate M1 score, M2 score, M1/M2 ratio, and net STV score using Macrophage STV weights.

    Orchestrates weight loading, partitioning, dot-product scoring, and ratio calculation
    via private helper functions. See `_compute_m1_m2_ratio` for the zero-infiltration
    biological assumption governing the ratio fallback value.

    Args:
        df_expr: Expression DataFrame (samples x genes).
        stv_path: Path to m1_m2_stv.csv file.

    Returns:
        Tuple of (m1_score, m2_score, m1_m2_ratio, net_stv_score) as pd.Series,
        all indexed identically to df_expr.

    Raises:
        FileNotFoundError: If stv_path does not exist.
    """
    if not stv_path.exists():
        raise FileNotFoundError(f"Macrophage STV matrix not found at {stv_path}")

    expr_genes = set(df_expr.columns)
    valid_weights, m1_genes, m2_genes = _load_and_partition_stv_weights(stv_path, expr_genes)

    m1_score = _compute_weighted_score(df_expr, m1_genes)
    m2_score = _compute_weighted_score(df_expr, m2_genes)
    m1_m2_ratio = _compute_m1_m2_ratio(m1_score, m2_score)
    net_stv_score = _compute_weighted_score(df_expr, valid_weights)

    return m1_score, m2_score, m1_m2_ratio, net_stv_score


def _score_cell_type_panel(
    cell_type: str, markers: list, df_expr: pd.DataFrame, min_markers: int
) -> pd.Series:
    """Calculate average expression score for a cell-type marker panel with coverage checks.

    Args:
        cell_type: Name of the cell-type panel.
        markers: List of marker gene symbols.
        df_expr: Expression DataFrame (samples x genes).
        min_markers: Minimum required matching markers.

    Returns:
        Per-sample score Series or NaN Series if marker coverage is insufficient.
    """
    found = [m for m in markers if m in df_expr.columns]
    n_found = len(found)
    n_total = len(markers)

    if n_found < min_markers:
        warnings.warn(
            f"Cell type deconvolution panel '{cell_type}' has insufficient marker coverage "
            f"({n_found}/{n_total} markers present: {found}). Minimum required is {min_markers}. "
            "Assigning NaN to prevent single-gene score degradation.",
            UserWarning,
            stacklevel=2,
        )
        return pd.Series(np.nan, index=df_expr.index)

    if n_found < n_total:
        missing = [m for m in markers if m not in df_expr.columns]
        warnings.warn(
            f"Cell type panel '{cell_type}' missing {len(missing)} marker(s): {missing}. "
            f"Computing average score across {n_found}/{n_total} present markers.",
            UserWarning,
            stacklevel=2,
        )
    return df_expr[found].mean(axis=1)


def compute_cell_deconvolution(
    df_expr: pd.DataFrame, min_markers: int = MIN_MARKERS_THRESHOLD
) -> pd.DataFrame:
    """Compute relative cell type signature scores using marker panels with coverage thresholds.

    Enforces a minimum marker threshold (`min_markers`) to prevent single-gene
    noise from causing silent statistical degradation. If fewer than `min_markers`
    are present in `df_expr`, a warning is raised and `NaN` is assigned.

    Args:
        df_expr: Expression DataFrame (samples x genes).
        min_markers: Minimum number of matching markers required per panel (default: 2).

    Returns:
        DataFrame of estimated cell-type fractions/scores (samples x cell types).
    """
    df_deconv = pd.DataFrame(index=df_expr.index)
    for cell_type, markers in CELL_TYPE_MARKERS.items():
        df_deconv[cell_type] = _score_cell_type_panel(cell_type, markers, df_expr, min_markers)
    return df_deconv

