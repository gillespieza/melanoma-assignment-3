"""Transcriptomic cell-type deconvolution and Macrophage STV calculation module for Q5.

Provides helper functions to calculate the M1/M2 Macrophage Signature Transcript Vector (STV)
and estimate relative cell-type fractions for patient stratification.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from q5_constants import CELL_TYPE_MARKERS


def compute_macrophage_stv(
    df_expr: pd.DataFrame, stv_path: Path
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate M1 score, M2 score, M1/M2 ratio, and net STV score using Macrophage STV weights.

    Args:
        df_expr: Expression DataFrame (samples x genes).
        stv_path: Path to m1_m2_stv.csv file.

    Returns:
        Tuple containing (m1_score, m2_score, m1_m2_ratio, net_stv_score).
    """
    if not stv_path.exists():
        raise FileNotFoundError(f"Macrophage STV matrix not found at {stv_path}")

    df_stv = pd.read_csv(stv_path)
    weights = dict(zip(df_stv["Gene"], df_stv["both_M1M2"]))

    # Find overlapping genes
    expr_genes = set(df_expr.columns)
    valid_weights = {g: w for g, w in weights.items() if g in expr_genes}

    m1_genes = {g: w for g, w in valid_weights.items() if w > 0}
    m2_genes = {g: abs(w) for g, w in valid_weights.items() if w < 0}

    # Compute M1 score (weighted sum of positive weights)
    if m1_genes:
        m1_weights_series = pd.Series(m1_genes)
        m1_score = df_expr[m1_weights_series.index].dot(m1_weights_series)
    else:
        m1_score = pd.Series(0.0, index=df_expr.index)

    # Compute M2 score (weighted sum of absolute negative weights)
    if m2_genes:
        m2_weights_series = pd.Series(m2_genes)
        m2_score = df_expr[m2_weights_series.index].dot(m2_weights_series)
    else:
        m2_score = pd.Series(0.0, index=df_expr.index)

    # Compute M1/M2 Ratio
    denom = m1_score + m2_score
    m1_m2_ratio = np.where(denom > 0, m1_score / denom, 0.5)
    m1_m2_ratio_series = pd.Series(m1_m2_ratio, index=df_expr.index)

    # Compute net STV score
    if valid_weights:
        all_weights_series = pd.Series(valid_weights)
        net_stv_score = df_expr[all_weights_series.index].dot(all_weights_series)
    else:
        net_stv_score = pd.Series(0.0, index=df_expr.index)

    return m1_score, m2_score, m1_m2_ratio_series, net_stv_score


def compute_cell_deconvolution(df_expr: pd.DataFrame) -> pd.DataFrame:
    """Compute relative cell type signature scores using marker panels.

    Args:
        df_expr: Expression DataFrame (samples x genes).

    Returns:
        DataFrame of estimated cell-type fractions/scores (samples x cell types).
    """
    df_deconv = pd.DataFrame(index=df_expr.index)

    for cell_type, markers in CELL_TYPE_MARKERS.items():
        found = [m for m in markers if m in df_expr.columns]
        if found:
            df_deconv[cell_type] = df_expr[found].mean(axis=1)
        else:
            df_deconv[cell_type] = np.nan

    return df_deconv
