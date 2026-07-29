"""Clustering module for Q5 patient stratification.

Provides functions for running K-Means and Agglomerative clustering,
evaluating optimal cluster count K (silhouette, GAP statistic), and generating
visualisation plots (silhouette plots, dendrograms, 2D projections).
"""

from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from src.styles import PHENOTYPE_PALETTE, set_presentation_style

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

set_presentation_style()

# Phenotype Cluster Index Colors matching PHENOTYPE_PALETTE
CLUSTER_PALETTE = {
    0: PHENOTYPE_PALETTE["Mutant-Driven"],              # Okabe-Ito Orange (#E69F00)
    1: PHENOTYPE_PALETTE["Immune Cold"],                # Okabe-Ito Blue (#0072B2)
    2: PHENOTYPE_PALETTE["Immune Hot"],                 # Crimson Red (#D55E00)
    3: PHENOTYPE_PALETTE["Immunosuppressive M2-High"],  # Okabe-Ito Reddish Purple (#CC79A7)
}


def prepare_clustering_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Select and standardize multi-modal immune microenvironment and driver mutation features for patient clustering."""
    feature_cols = [
        c for c in ["TIS", "CYT", "CD8_T_cells", "M1_Macrophages", "M2_Macrophages", "CAFs", "mut_BRAF", "mut_NRAS", "mut_NF1"]
        if c in df.columns
    ]
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


def plot_2d_cluster_projection(
    df_clean: pd.DataFrame,
    labels: np.ndarray,
    X_scaled: np.ndarray,
    save_path: Path,
    cluster_names: Dict[int, str] = None,
) -> None:
    """Generate publication-ready 2D cluster projection plot with clear cluster separation and expanded biological legend.

    Uses PCA 2D embedding to project patients into visually distinct cluster regions.
    Colors imported from PHENOTYPE_PALETTE in src.styles (Crimson Red for Immune Hot, Blue for Immune Cold).
    """
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    df_plot = df_clean.copy()
    df_plot["Dim1"] = coords[:, 0]
    df_plot["Dim2"] = coords[:, 1]
    df_plot["Cluster"] = labels

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    # Plot scatter points per cluster
    for cluster_id in sorted(np.unique(labels)):
        sub = df_plot[df_plot["Cluster"] == cluster_id]
        c_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}") if cluster_names else f"Cluster {cluster_id}"
        c_color = PHENOTYPE_PALETTE.get(c_name, CLUSTER_PALETTE.get(cluster_id, f"C{cluster_id}"))

        ax.scatter(
            sub["Dim1"],
            sub["Dim2"],
            label=c_name,
            c=c_color,
            s=70,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        # Calculate cluster center and draw shaded confidence ellipse
        if len(sub) > 3:
            mean_x, mean_y = sub["Dim1"].mean(), sub["Dim2"].mean()
            cov = np.cov(sub["Dim1"], sub["Dim2"])
            evals, evecs = np.linalg.eigh(cov)
            order = evals.argsort()[::-1]
            evals, evecs = evals[order], evecs[:, order]
            angle = np.degrees(np.arctan2(*evecs[:, 0][::-1]))
            width, height = 2.5 * np.sqrt(evals)

            ellipse = Ellipse(
                xy=(mean_x, mean_y),
                width=width,
                height=height,
                angle=angle,
                color=c_color,
                alpha=0.18,
            )
            ax.add_patch(ellipse)

            # Center short label text annotation
            short_name = c_name.split("(")[0].strip()
            ax.text(
                mean_x,
                mean_y,
                short_name,
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=c_color, alpha=0.9),
            )

    n_patients = len(df_plot)
    var_explained = np.sum(pca.explained_variance_ratio_) * 100
    ax.set_title(
        f"Unsupervised Patient Phenotype Clusters (N={n_patients}, PCA Variance: {var_explained:.1f}%)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Principal Component 1 (Immune Activation & T-cell Density)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Principal Component 2 (M1/M2 Macrophage & Stromal Axis)", fontsize=12, fontweight="bold")

    # Legend with expanded biological descriptions
    ax.legend(
        title="Biological Subtype & Microenvironment",
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.95,
        fontsize=9,
        title_fontsize=10,
    )

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved 2D cluster projection plot to {save_path}")
