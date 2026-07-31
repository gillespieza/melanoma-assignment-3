"""Clustering module for Q5 patient stratification.

Provides functions for running K-Means and Agglomerative clustering,
evaluating optimal cluster count K (silhouette, GAP statistic), and generating
visualisation plots (silhouette plots, dendrograms, 2D projections).
"""

from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from q5_constants import CLUSTERING_FEATURES
from src.styles import PHENOTYPE_PALETTE, get_phenotype_color, set_presentation_style
from src.utils.plotting import save_fig

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

set_presentation_style()

# Phenotype Cluster Index Colors matching PHENOTYPE_PALETTE.
# Default mapping; overridden dynamically by c_name in plot_2d_cluster_projection.
CLUSTER_PALETTE = {
    0: PHENOTYPE_PALETTE["Immune Hot"],                 # Crimson Red (#D55E00)
    1: PHENOTYPE_PALETTE["Immune Cold"],                # Okabe-Ito Blue (#0072B2)
    2: PHENOTYPE_PALETTE["Immunosuppressive M2-High"],  # Okabe-Ito Reddish Purple (#CC79A7)
    3: PHENOTYPE_PALETTE["Mutant-Driven"],              # Okabe-Ito Orange (#E69F00)
}

# 2D Cluster Projection Visualization Constants
KDE_BANDWIDTH: float = 0.25
ALPHA_DENSITY_SCALE: float = 0.55
ALPHA_MIN: float = 0.35
ALPHA_MAX: float = 0.95
TSNE_PERPLEXITY: int = 50
TSNE_MAX_ITER: int = 1000
MARKER_SIZE: float = 65.0
SIGMA_FACTOR: float = 2.0
ELLIPSE_ALPHA: float = 0.18
LABEL_OFFSET_FACTOR: float = 0.35


def prepare_clustering_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Select and standardize multi-modal immune microenvironment and driver mutation features for patient clustering."""
    feature_cols = [c for c in CLUSTERING_FEATURES if c in df.columns]
    df_clean = df.dropna(subset=feature_cols).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[feature_cols])
    return df_clean, X_scaled


def run_kmeans(df_features: pd.DataFrame, k_range: range = range(2, 7)) -> Dict[int, Tuple[np.ndarray, float]]:
    """Run K-Means clustering across a range of K values.

    Returns:
        Dictionary mapping K to tuple of (cluster_labels, silhouette_score).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features.select_dtypes(include=[np.number]))
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        results[k] = (labels, score)
    return results


def _compute_2d_embedding(
    X_scaled: np.ndarray,
    n_patients: int,
    method: str,
) -> Tuple[np.ndarray, str, str, str]:
    """Compute 2D projection coordinates and plot axis labels using PCA or t-SNE."""
    method_lower = method.lower()
    if method_lower == "tsne":
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=TSNE_PERPLEXITY,
            max_iter=TSNE_MAX_ITER,
            init="pca",
            learning_rate="auto",
        )
        coords = tsne.fit_transform(X_scaled)
        xlabel = "t-SNE Dimension 1"
        ylabel = "t-SNE Dimension 2"
        title_str = f"Unsupervised Patient Phenotype Manifold (N={n_patients}, t-SNE)"
    else:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        var_explained = np.sum(pca.explained_variance_ratio_) * 100
        xlabel = "Principal Component 1 (Immune Activation & T-cell Density)"
        ylabel = "Principal Component 2 (M1/M2 Macrophage & Stromal Axis)"
        title_str = f"Unsupervised Patient Phenotype Clusters (N={n_patients}, PCA Variance: {var_explained:.1f}%)"

    return coords, xlabel, ylabel, title_str


def _compute_density_alpha(coords: np.ndarray) -> np.ndarray:
    """Compute per-point density-inverse alpha values to reduce visual overlap clutter."""
    kde_global = gaussian_kde(coords.T, bw_method=KDE_BANDWIDTH)
    density_all = kde_global(coords.T)
    d_norm = (density_all - density_all.min()) / (density_all.max() - density_all.min() + 1e-9)
    return np.clip(1.0 - ALPHA_DENSITY_SCALE * d_norm, ALPHA_MIN, ALPHA_MAX)


def _scatter_cluster(
    ax: plt.Axes,
    sub: pd.DataFrame,
    c_name: str,
    c_color: str,
) -> None:
    """Plot single cluster scatter points, confidence ellipse, and text annotation."""
    sub_alpha = sub["_alpha"].values
    rgba_base = mcolors.to_rgba(c_color)
    rgba_arr = np.array([[rgba_base[0], rgba_base[1], rgba_base[2], a] for a in sub_alpha])

    ax.scatter(
        sub["Dim1"],
        sub["Dim2"],
        label=c_name,
        c=rgba_arr,
        s=MARKER_SIZE,
        edgecolor="white",
        linewidth=0.4,
    )

    if len(sub) > 3:
        mean_x, mean_y = sub["Dim1"].mean(), sub["Dim2"].mean()
        cov = np.cov(sub["Dim1"], sub["Dim2"])
        evals, evecs = np.linalg.eigh(cov)
        order = evals.argsort()[::-1]
        evals, evecs = evals[order], evecs[:, order]
        angle = np.degrees(np.arctan2(*evecs[:, 0][::-1]))
        width, height = SIGMA_FACTOR * np.sqrt(evals)

        ellipse = Ellipse(
            xy=(mean_x, mean_y),
            width=width,
            height=height,
            angle=angle,
            color=c_color,
            alpha=ELLIPSE_ALPHA,
        )
        ax.add_patch(ellipse)

        short_name = c_name.split("(")[0].strip()
        std_y = sub["Dim2"].std()
        offset_y = LABEL_OFFSET_FACTOR * (std_y if std_y > 0 else 0.5)

        ax.text(
            mean_x,
            mean_y + offset_y,
            short_name,
            fontsize=9.5,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=c_color, alpha=0.92, linewidth=1.1),
        )


def plot_2d_cluster_projection(
    df_clean: pd.DataFrame,
    labels: np.ndarray,
    X_scaled: np.ndarray,
    save_path: Path,
    cluster_names: Dict[int, str] = None,
    method: str = "pca",
) -> None:
    """Generate publication-ready 2D cluster projection plot (PCA or t-SNE)."""
    coords, xlabel, ylabel, title_str = _compute_2d_embedding(X_scaled, len(df_clean), method)

    df_plot = df_clean.copy()
    df_plot["Dim1"] = coords[:, 0]
    df_plot["Dim2"] = coords[:, 1]
    df_plot["Cluster"] = labels
    df_plot["_alpha"] = _compute_density_alpha(coords)

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    for cluster_id in sorted(np.unique(labels)):
        sub = df_plot[df_plot["Cluster"] == cluster_id]
        c_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}") if cluster_names else f"Cluster {cluster_id}"
        c_color = get_phenotype_color(c_name)
        _scatter_cluster(ax, sub, c_name, c_color)

    ax.set_title(title_str, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

    legend_loc = "lower left" if method.lower() == "pca" else "upper left"
    ax.legend(
        title="Biological Subtype & Microenvironment",
        loc=legend_loc,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.95,
        fontsize=9,
        title_fontsize=10,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved {method.upper()} 2D cluster projection plot to {save_path}")
