"""Phenotype characterisation and profiling module for Q5.

Provides helpers for summarizing cluster profiles, assigning biological
phenotype labels, and plotting radar charts and annotated cluster heatmaps.
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd


def profile_clusters(df: pd.DataFrame, cluster_col: str, feature_cols: List[str]) -> pd.DataFrame:
    """Calculate mean feature values and response rates per cluster.

    Args:
        df: Patient dataset containing cluster labels and features.
        cluster_col: Name of column holding cluster IDs.
        feature_cols: List of feature names to summarize.

    Returns:
        DataFrame of per-cluster summary statistics.
    """
    # Stub implementation
    return pd.DataFrame()


def assign_phenotype_labels(cluster_profiles: pd.DataFrame) -> Dict[int, str]:
    """Assign biological phenotype labels to clusters based on profile rules.

    Args:
        cluster_profiles: DataFrame of cluster feature profiles.

    Returns:
        Dictionary mapping cluster ID to phenotype label string.
    """
    # Stub implementation
    return {}


def plot_radar_chart(profiles: pd.DataFrame, save_path: Path) -> None:
    """Generate spider/radar chart comparing cluster feature profiles.

    Args:
        profiles: Per-cluster mean feature values.
        save_path: Output file path for figure.
    """
    # Stub implementation
    pass


def plot_cluster_heatmap(df: pd.DataFrame, cluster_col: str, feature_cols: List[str], save_path: Path) -> None:
    """Generate annotated heatmap of patients x features sorted by cluster.

    Args:
        df: Patient dataset.
        cluster_col: Column name for cluster assignment.
        feature_cols: List of features to visualize in heatmap.
        save_path: Output file path for figure.
    """
    # Stub implementation
    pass
