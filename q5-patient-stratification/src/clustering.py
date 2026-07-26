"""Clustering module for Q5 patient stratification.

Provides functions for running K-Means and Agglomerative clustering,
evaluating optimal cluster count K (silhouette, GAP statistic), and generating
visualisation plots (silhouette plots, dendrograms, UMAP projections).
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def run_kmeans(df_features: pd.DataFrame, k_range: range) -> Dict[int, Tuple[np.ndarray, float]]:
    """Run K-Means clustering across a range of K values.

    Args:
        df_features: Feature matrix (patients x features).
        k_range: Range of K cluster values to evaluate.

    Returns:
        Dictionary mapping K to tuple of (cluster_labels, silhouette_score).
    """
    # Stub implementation
    return {}


def run_agglomerative(df_features: pd.DataFrame, k_range: range) -> Dict[int, Tuple[np.ndarray, float]]:
    """Run Hierarchical Agglomerative clustering across a range of K values.

    Args:
        df_features: Feature matrix (patients x features).
        k_range: Range of K cluster values to evaluate.

    Returns:
        Dictionary mapping K to tuple of (cluster_labels, silhouette_score).
    """
    # Stub implementation
    return {}


def compute_gap_statistic(df_features: pd.DataFrame, k_range: range, n_refs: int = 10) -> Dict[int, float]:
    """Compute the GAP statistic for K-Means clustering.

    Args:
        df_features: Feature matrix (patients x features).
        k_range: Range of K cluster values.
        n_refs: Number of reference datasets for bootstrapping.

    Returns:
        Dictionary mapping K to computed GAP statistic.
    """
    # Stub implementation
    return {}


def plot_silhouette(df_features: pd.DataFrame, labels: np.ndarray, save_path: Path) -> None:
    """Generate and save silhouette plot for a specific clustering.

    Args:
        df_features: Feature matrix.
        labels: Cluster labels.
        save_path: Path to save output figure.
    """
    # Stub implementation
    pass


def plot_elbow(inertias: Dict[int, float], save_path: Path) -> None:
    """Generate and save elbow plot of inertias vs. K.

    Args:
        inertias: Dictionary mapping K to inertia value.
        save_path: Path to save output figure.
    """
    # Stub implementation
    pass
