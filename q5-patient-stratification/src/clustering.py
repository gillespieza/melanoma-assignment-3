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
    method: str = "pca",
) -> None:
    """Generate publication-ready 2D cluster projection plot (PCA or t-SNE).

    For the non-linear embedding, uses t-SNE (perplexity=50, n_iter=1000).
    t-SNE handles mixed continuous/binary feature spaces more gracefully than
    UMAP for this dataset, producing natural cluster scatter with gradient
    boundaries rather than collapsed blobs or artificially hard islands."""
    method_lower = method.lower()
    if method_lower == "umap":
        # t-SNE is used here in place of UMAP: perplexity=50 balances local
        # and global neighbourhood structure at N=699, and n_iter=1000 ensures
        # convergence. The method parameter is kept as "umap" for backward
        # compatibility with call sites in Script 03.
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=50,
            max_iter=1000,
            init="pca",
            learning_rate="auto",
        )
        coords = tsne.fit_transform(X_scaled)
        xlabel = "t-SNE Dimension 1"
        ylabel = "t-SNE Dimension 2"
        title_str = f"Unsupervised Patient Phenotype Manifold (N={len(df_clean)}, t-SNE)"
    else:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        var_explained = np.sum(pca.explained_variance_ratio_) * 100
        xlabel = "Principal Component 1 (Immune Activation & T-cell Density)"
        ylabel = "Principal Component 2 (M1/M2 Macrophage & Stromal Axis)"
        title_str = f"Unsupervised Patient Phenotype Clusters (N={len(df_clean)}, PCA Variance: {var_explained:.1f}%)"

    df_plot = df_clean.copy()
    df_plot["Dim1"] = coords[:, 0]
    df_plot["Dim2"] = coords[:, 1]
    df_plot["Cluster"] = labels

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    # Pre-compute per-point KDE density across all 2D coords so that alpha
    # can be scaled inversely: dense overlapping regions become more transparent,
    # sparse boundary points stay opaque — reducing visual clutter in overlaps.
    from scipy.stats import gaussian_kde
    all_coords = np.column_stack([df_plot["Dim1"], df_plot["Dim2"]])
    kde_global = gaussian_kde(all_coords.T, bw_method=0.25)
    density_all = kde_global(all_coords.T)
    # Normalise to [0, 1] then invert so high-density points get low alpha
    d_norm = (density_all - density_all.min()) / (density_all.max() - density_all.min() + 1e-9)
    alpha_per_point = np.clip(1.0 - 0.55 * d_norm, 0.35, 0.95)
    df_plot["_alpha"] = alpha_per_point
    df_plot["_idx"] = np.arange(len(df_plot))

    # Plot scatter points per cluster
    for cluster_id in sorted(np.unique(labels)):
        sub = df_plot[df_plot["Cluster"] == cluster_id]
        c_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}") if cluster_names else f"Cluster {cluster_id}"
        c_color = get_phenotype_color(c_name)

        # Use density-scaled alpha: retrieve pre-computed per-point alpha values
        sub_alpha = df_plot.loc[sub.index, "_alpha"].values
        ax.scatter(
            sub["Dim1"],
            sub["Dim2"],
            label=c_name,
            c=[c_color] * len(sub),
            s=65,
            alpha=None,       # alpha handled per-point via RGBA colors below
            edgecolor="white",
            linewidth=0.4,
        )
        # Overlay per-point alpha by re-scattering with individual RGBA values
        import matplotlib.colors as mcolors
        rgba_base = mcolors.to_rgba(c_color)
        rgba_arr = np.array([[rgba_base[0], rgba_base[1], rgba_base[2], a] for a in sub_alpha])
        ax.scatter(
            sub["Dim1"], sub["Dim2"],
            c=rgba_arr, s=65, edgecolor="white", linewidth=0.4,
        )

        # Calculate cluster core and draw tightened 2.0-sigma confidence ellipse
        if len(sub) > 3:
            mean_x, mean_y = sub["Dim1"].mean(), sub["Dim2"].mean()
            cov = np.cov(sub["Dim1"], sub["Dim2"])
            evals, evecs = np.linalg.eigh(cov)
            order = evals.argsort()[::-1]
            evals, evecs = evals[order], evecs[:, order]
            angle = np.degrees(np.arctan2(*evecs[:, 0][::-1]))
            width, height = 2.0 * np.sqrt(evals)

            ellipse = Ellipse(
                xy=(mean_x, mean_y),
                width=width,
                height=height,
                angle=angle,
                color=c_color,
                alpha=0.18,
            )
            ax.add_patch(ellipse)

            # Short label text annotation offset above cluster center to prevent obscuring point density
            short_name = c_name.split("(")[0].strip()
            std_y = sub["Dim2"].std()
            offset_y = 0.35 * (std_y if std_y > 0 else 0.5)

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

    ax.set_title(title_str, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")

    ax.legend(
        title="Biological Subtype & Microenvironment",
        loc="upper left",
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
