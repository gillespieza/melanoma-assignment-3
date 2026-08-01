"""Clustering module for Q5 patient stratification.

Provides functions for running K-Means and Agglomerative clustering,
evaluating optimal cluster count K (silhouette, GAP statistic), and generating
visualisation plots (silhouette plots, dendrograms, 2D projections).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from q5_constants import CLUSTERING_FEATURES
from src.styles import PHENOTYPE_PALETTE, get_phenotype_color, set_presentation_style
from src.utils.paths import rel_path
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
MIN_SAMPLES_FOR_ELLIPSE: int = 3

# Default Model & Consensus Parameters
DEFAULT_N_COMPONENTS: int = 4
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_SPECTRAL_NEIGHBORS: int = 15
CDF_GRID_POINTS: int = 100


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


def run_gmm(
    X_scaled: np.ndarray,
    n_components: int = DEFAULT_N_COMPONENTS,
    covariance_type: str = "full",
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """Fit Gaussian Mixture Model (GMM) with full covariance matrices and compute soft posterior probabilities.

    Returns:
        Tuple of (fitted_gmm_model, hard_cluster_labels, posterior_probabilities_matrix).
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=10,
    )
    gmm.fit(X_scaled)
    probs = gmm.predict_proba(X_scaled)
    labels = np.argmax(probs, axis=1)
    return gmm, labels, probs


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

    if len(sub) > MIN_SAMPLES_FOR_ELLIPSE:
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
    cluster_names: Optional[Dict[int, str]] = None,
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
        legend_label = f"{c_name} (N={len(sub)})"
        _scatter_cluster(ax, sub, legend_label, c_color)

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


def _compute_coclustering_matrix(
    X_scaled: np.ndarray,
    k: int,
    n_bootstraps: int,
    sample_sub_size: int,
    feat_sub_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Compute consensus co-association matrix across n_bootstraps subsamplings."""
    n_samples = X_scaled.shape[0]
    co_counts = np.zeros((n_samples, n_samples), dtype=float)
    pair_counts = np.zeros((n_samples, n_samples), dtype=float)

    for _ in range(n_bootstraps):
        sample_idx = rng.choice(n_samples, size=sample_sub_size, replace=False)
        feat_idx = rng.choice(X_scaled.shape[1], size=feat_sub_size, replace=False)

        sub_X = X_scaled[np.ix_(sample_idx, feat_idx)]
        km = KMeans(n_clusters=k, random_state=rng.randint(0, 100000), n_init=1)
        lbls = km.fit_predict(sub_X)

        same_cluster = (lbls[:, None] == lbls[None, :])
        idx_grid = np.ix_(sample_idx, sample_idx)
        co_counts[idx_grid] += same_cluster
        pair_counts[idx_grid] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        M_k = np.where(pair_counts > 0, co_counts / pair_counts, 0.0)
    return M_k


def _calculate_cdf_auc_delta(
    consensus_matrices: Dict[int, np.ndarray],
    k_range: range,
    n_samples: int,
    grid_c: np.ndarray,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], Dict[int, float], Dict[int, float], pd.DataFrame]:
    """Calculate Cumulative Distribution Functions, AUC, and Delta Area metrics across K values."""
    cdf_curves: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    auc_dict: Dict[int, float] = {}
    delta_area_dict: Dict[int, float] = {}
    triu_indices = np.triu_indices(n_samples, k=1)

    for k in k_range:
        M_k = consensus_matrices[k]
        u_vals = M_k[triu_indices]
        cdf_vals = np.array([np.mean(u_vals <= c) for c in grid_c])
        cdf_curves[k] = (grid_c, cdf_vals)

        try:
            auc_val = float(np.trapezoid(cdf_vals, grid_c))
        except AttributeError:
            auc_val = float(np.trapz(cdf_vals, grid_c))
        auc_dict[k] = auc_val

    sorted_ks = sorted(list(k_range))
    for idx, k in enumerate(sorted_ks):
        if idx == 0:
            delta_area_dict[k] = auc_dict[k]
        else:
            prev_k = sorted_ks[idx - 1]
            prev_auc = auc_dict[prev_k]
            delta_area_dict[k] = (auc_dict[k] - prev_auc) / prev_auc if prev_auc > 0 else 0.0

    records = [
        {
            "K": k,
            "CDF_AUC": round(auc_dict[k], 4),
            "Delta_Area": round(delta_area_dict[k], 4),
            "Mean_Consensus_Score": round(float(np.mean(consensus_matrices[k][triu_indices])), 4),
        }
        for k in sorted_ks
    ]
    metrics_df = pd.DataFrame(records)
    return cdf_curves, auc_dict, delta_area_dict, metrics_df


def run_consensus_bootstrap(
    X_scaled: np.ndarray,
    k_range: range = range(2, 9),
    n_bootstraps: int = 1000,
    sample_ratio: float = 0.8,
    feature_ratio: float = 0.8,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Tuple[np.ndarray, np.ndarray]], Dict[int, float], Dict[int, float], pd.DataFrame]:
    """Execute 1,000-bootstrap Consensus Clustering across patients and features for K in k_range.

    Returns:
        Tuple of (consensus_matrices, cdf_curves, auc_dict, delta_area_dict, metrics_df).
    """
    n_samples, n_features = X_scaled.shape
    rng = np.random.RandomState(random_state)
    sample_sub_size = max(2, int(n_samples * sample_ratio))
    feat_sub_size = max(1, int(n_features * feature_ratio))

    consensus_matrices: Dict[int, np.ndarray] = {}
    grid_c = np.linspace(0.01, 1.0, CDF_GRID_POINTS)

    for k in k_range:
        consensus_matrices[k] = _compute_coclustering_matrix(
            X_scaled, k, n_bootstraps, sample_sub_size, feat_sub_size, rng
        )

    cdf_curves, auc_dict, delta_area_dict, metrics_df = _calculate_cdf_auc_delta(
        consensus_matrices, k_range, n_samples, grid_c
    )

    return consensus_matrices, cdf_curves, auc_dict, delta_area_dict, metrics_df


def plot_consensus_cdf_curves(
    cdf_curves: Dict[int, Tuple[np.ndarray, np.ndarray]],
    out_cdf_path: Path,
) -> None:
    """Generate 300 DPI publication plot for Consensus CDF curves across K in [2, 8]."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(cdf_curves)))

    for idx, (k, (grid_c, cdf_vals)) in enumerate(sorted(cdf_curves.items())):
        ax.plot(grid_c, cdf_vals, label=f"K = {k}", color=colors[idx], linewidth=2.0)

    ax.set_title("Consensus Clustering Cumulative Distribution Functions (CDF across K)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Consensus Index (c)", fontsize=11, fontweight="bold")
    ax.set_ylabel("CDF (F(c))", fontsize=11, fontweight="bold")
    ax.legend(title="Cluster Count K", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=9.5)
    ax.grid(True, color="#E5E7EB", linewidth=0.5, alpha=0.6)

    out_cdf_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_cdf_path, dpi=300)
    print(f"Saved Consensus CDF curves to {rel_path(out_cdf_path)}")


def plot_consensus_delta_area(
    delta_area_dict: Dict[int, float],
    out_delta_path: Path,
) -> None:
    """Generate 300 DPI publication plot for Consensus Delta Area scores across K in [2, 8]."""
    fig2, ax2 = plt.subplots(figsize=(9, 5.5), dpi=300)
    ks = sorted(list(delta_area_dict.keys()))
    deltas = [delta_area_dict[k] for k in ks]

    ax2.plot(ks, deltas, marker="o", color="#0072B2", linewidth=2.2, markersize=8, label="Relative Delta Area $\\Delta(K)$")
    for k, d in zip(ks, deltas):
        ax2.annotate(f"{d:.3f}", (k, d), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, fontweight="bold")

    ax2.set_title("Consensus Clustering Relative Delta Area Score $\\Delta(K)$", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Cluster Count K", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Relative Change in CDF Area $\\Delta(K)$", fontsize=11, fontweight="bold")
    ax2.set_xticks(ks)
    ax2.grid(True, color="#E5E7EB", linewidth=0.5, alpha=0.6)

    out_delta_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig2, out_delta_path, dpi=300)
    print(f"Saved Consensus Delta Area plot to {rel_path(out_delta_path)}")


def plot_consensus_cdf_and_delta_area(
    cdf_curves: Dict[int, Tuple[np.ndarray, np.ndarray]],
    delta_area_dict: Dict[int, float],
    out_cdf_path: Path,
    out_delta_path: Path,
) -> None:
    """Generate 300 DPI publication plots for Consensus CDF curves and Delta Area scores across K in [2, 8]."""
    plot_consensus_cdf_curves(cdf_curves, out_cdf_path)
    plot_consensus_delta_area(delta_area_dict, out_delta_path)


def plot_consensus_heatmap(
    M_k: np.ndarray,
    out_path: Path,
    k: int = 4,
) -> None:
    """Generate publication-ready clustered heatmap of sample-sample co-association matrix."""
    from scipy.cluster.hierarchy import leaves_list, linkage

    row_link = linkage(M_k, method="average")
    order = leaves_list(row_link)
    M_ordered = M_k[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    im = ax.imshow(M_ordered, cmap="PuBu", aspect="auto", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Consensus Co-clustering Index", fontsize=10, fontweight="bold")

    ax.set_title(f"Consensus Co-association Matrix Heatmap (K = {k})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Samples (Ordered by Consensus Linkage)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Samples (Ordered by Consensus Linkage)", fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_path, dpi=300)
    print(f"Saved Consensus Heatmap (K={k}) to {rel_path(out_path)}")


def transform_mahalanobis_space(
    X_scaled: np.ndarray,
    ridge_alpha: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform feature space by regularized inverse square root covariance matrix (X_mahalanobis = X_scaled @ Sigma^{-1/2}).

    Decorrelates collinear feature traits (TIS, CYT, CD8_T_cells, r > 0.70) into an isotropic Mahalanobis space.

    Returns:
        Tuple of (X_mahalanobis, inv_sqrt_cov_matrix).
    """
    cov = np.cov(X_scaled, rowvar=False)
    cov_reg = cov + ridge_alpha * np.eye(cov.shape[0])

    evals, evecs = np.linalg.eigh(cov_reg)
    evals = np.maximum(evals, 1e-8)
    inv_sqrt_cov = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T

    X_mahalanobis = X_scaled @ inv_sqrt_cov
    return X_mahalanobis, inv_sqrt_cov


def run_spectral_manifold(
    X_scaled: np.ndarray,
    n_clusters: int = DEFAULT_N_COMPONENTS,
    n_neighbors: int = DEFAULT_SPECTRAL_NEIGHBORS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[SpectralClustering, np.ndarray]:
    """Fit Spectral Manifold Clustering on Graph Laplacian eigenvectors using nearest-neighbors affinity.

    Returns:
        Tuple of (spectral_model, hard_cluster_labels).
    """
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        random_state=random_state,
        n_init=10,
    )
    labels = spectral.fit_predict(X_scaled)
    return spectral, labels


def plot_spatial_microenvironment_violins(
    df_clean: pd.DataFrame,
    save_path: Path,
) -> None:
    """Generate 300 DPI publication violin plot comparing spatial microenvironment metrics across phenotypes."""
    spatial_cols = [c for c in ["Spatial_CD8_CAF_Distance_Ratio", "Spatial_Tumour_Infiltration_Index"] if c in df_clean.columns]
    if not spatial_cols or "Phenotype_Label" not in df_clean.columns:
        print("Skipping spatial violin plot: missing spatial columns or Phenotype_Label.")
        return

    fig, axes = plt.subplots(1, len(spatial_cols), figsize=(6.5 * len(spatial_cols), 5.5), dpi=300)
    if len(spatial_cols) == 1:
        axes = [axes]

    labels = sorted(df_clean["Phenotype_Label"].unique())
    palette = [get_phenotype_color(l) for l in labels]

    for idx, col in enumerate(spatial_cols):
        ax = axes[idx]
        clean_data = [df_clean[df_clean["Phenotype_Label"] == l][col].dropna() for l in labels]
        short_names = [l.split("(")[0].strip() for l in labels]

        parts = ax.violinplot(clean_data, showmedians=True, showextrema=False)
        for pc, color in zip(parts["bodies"], palette):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor("black")

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(short_names, fontsize=9.5, fontweight="bold", rotation=15)
        ax.set_title(col.replace("_", " "), fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel("Score Ratio", fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path, dpi=300)
    print(f"Saved spatial microenvironment violin plots to {rel_path(save_path)}")
