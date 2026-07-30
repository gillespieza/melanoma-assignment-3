#!/usr/bin/env python3
"""Script 08: Benchmark clustering algorithm comparison for Q5 patient stratification.

Evaluates K-Means, Hierarchical Agglomerative Clustering (HAC), Gaussian Mixture
Models (GMM), Spectral Clustering, DBSCAN, and Consensus Clustering on the Q5 patient
feature matrix (N=326, 9 multi-modal features). Computes Silhouette Score,
Calinski-Harabasz Index, Davies-Bouldin Index, Adjusted Rand Index, and Response Rate
Spread per algorithm, and generates a 6-panel comparison figure for reporting.
"""

import contextlib
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SUBPROJECT_ROOT = SCRIPT_DIR.parent
for parent in [SCRIPT_DIR] + list(SCRIPT_DIR.parents):
    if (parent / "data").is_dir() and (parent / "src").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

if str(SUBPROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from src.styles import OKABE_ITO, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, rel_path
from src.utils.plotting import save_fig

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

set_presentation_style()

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "08_compare_clustering_algorithms.log"

INPUT_FILE = PROCESSED_DIR / "q5" / "feature_matrix.csv"
OUTPUT_PLOT = SUBPROJECT_ROOT / "plots" / "clustering" / "clustering_algorithms_comparison.png"
OUTPUT_METRICS = SUBPROJECT_ROOT / "reports" / "clustering_metrics_comparison.csv"

# The 9 multi-modal features used for clustering
CLUSTERING_FEATURES: List[str] = [
    "TIS", "CYT", "CD8_T_cells", "M1_Macrophages", "M2_Macrophages",
    "CAFs", "mut_BRAF", "mut_NRAS", "mut_NF1",
]

# Okabe-Ito Scientific color palette for cluster panels
CLUSTER_COLORS: List[str] = OKABE_ITO

N_BOOTSTRAPS = 50
SUBSAMPLE_RATIO = 0.8
K_CLUSTERS = 4
DBSCAN_EPS = 1.8
DBSCAN_MIN_SAMPLES = 5


def load_and_prepare_features() -> tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Load the Q5 feature matrix and prepare scaled clustering features.

    Returns:
        Tuple of (df_clean, X_scaled, feature_cols) where df_clean is the patient
        DataFrame with valid features, X_scaled is the StandardScaler-transformed
        feature matrix, and feature_cols lists the active feature column names.

    Raises:
        FileNotFoundError: If the feature matrix CSV is missing.
    """
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing feature matrix at {rel_path(INPUT_FILE)}. Run Script 01 first."
        )

    df_master = pd.read_csv(INPUT_FILE)
    feature_cols = [c for c in CLUSTERING_FEATURES if c in df_master.columns]
    df_clean = df_master.dropna(subset=feature_cols).copy()
    X_scaled = StandardScaler().fit_transform(df_clean[feature_cols])
    print(f"Loaded {len(df_clean)} patients x {len(feature_cols)} features for clustering benchmark.")
    return df_clean, X_scaled, feature_cols


def run_all_algorithms(X_scaled: np.ndarray) -> Dict[str, np.ndarray]:
    """Fit all benchmark clustering algorithms on the scaled feature matrix.

    Args:
        X_scaled: StandardScaler-normalised patient feature matrix (N x 9).

    Returns:
        Dictionary mapping algorithm name to array of cluster label integers.
        DBSCAN noise points are assigned label -1.
    """
    algorithms: Dict[str, np.ndarray] = {}

    # 1. K-Means – current production benchmark
    algorithms["K-Means (Current)"] = KMeans(
        n_clusters=K_CLUSTERS, random_state=42, n_init=10
    ).fit_predict(X_scaled)

    # 2. Hierarchical Agglomerative Clustering (Ward linkage)
    algorithms["HAC (Ward Linkage)"] = AgglomerativeClustering(
        n_clusters=K_CLUSTERS, linkage="ward"
    ).fit_predict(X_scaled)

    # 3. Gaussian Mixture Model (soft probabilistic assignment, returned as hard labels)
    algorithms["Gaussian Mixture (GMM)"] = GaussianMixture(
        n_components=K_CLUSTERS, random_state=42, n_init=5
    ).fit_predict(X_scaled)

    # 4. Spectral Clustering (k-nearest-neighbor affinity graph, Graph Laplacian eigenvectors)
    algorithms["Spectral Clustering"] = SpectralClustering(
        n_clusters=K_CLUSTERS, random_state=42, affinity="nearest_neighbors"
    ).fit_predict(X_scaled)

    # 5. DBSCAN – density-based with noise point detection (label = -1 for outliers)
    algorithms["DBSCAN (Density-Based)"] = DBSCAN(
        eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES
    ).fit_predict(X_scaled)

    # 6. Consensus Clustering – bootstrap-ensemble K-Means co-occurrence matrix
    algorithms["Consensus Clustering"] = _run_consensus_clustering(X_scaled)

    return algorithms


def _run_consensus_clustering(X_scaled: np.ndarray) -> np.ndarray:
    """Build a bootstrap consensus co-occurrence matrix and apply Spectral Clustering.

    Runs N_BOOTSTRAPS iterations of K-Means on SUBSAMPLE_RATIO subsets of patients,
    accumulating a co-occurrence matrix tracking how often pairs of patients cluster
    together. Spectral Clustering is then applied to the normalized consensus matrix.

    Args:
        X_scaled: Normalised patient feature matrix.

    Returns:
        Array of consensus cluster label integers.
    """
    n = len(X_scaled)
    co_matrix = np.zeros((n, n))
    rng = np.random.RandomState(42)

    for _ in range(N_BOOTSTRAPS):
        idx = rng.choice(n, size=int(n * SUBSAMPLE_RATIO), replace=False)
        lbls = KMeans(
            n_clusters=K_CLUSTERS, random_state=rng.randint(0, 10000), n_init=3
        ).fit_predict(X_scaled[idx])
        for i_loc, i_orig in enumerate(idx):
            for j_loc, j_orig in enumerate(idx):
                if lbls[i_loc] == lbls[j_loc]:
                    co_matrix[i_orig, j_orig] += 1

    co_matrix /= N_BOOTSTRAPS
    return SpectralClustering(
        n_clusters=K_CLUSTERS, affinity="precomputed", random_state=42
    ).fit_predict(co_matrix)


def compute_metrics(
    X_scaled: np.ndarray,
    df_clean: pd.DataFrame,
    algorithms: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute performance metrics for each clustering algorithm.

    Metrics:
        - Silhouette Score: Mean intra-cluster cohesion vs inter-cluster separation (higher is better).
        - Calinski-Harabasz Ratio: Between/within cluster variance ratio (higher is better).
        - Davies-Bouldin Index: Average intra/inter cluster distance ratio (lower is better).
        - ARI vs K-Means: Adjusted Rand Index comparing each algorithm to K-Means benchmark.
        - Response Rate Spread: Range of immunotherapy response rates across clusters (higher = more clinically discriminative).

    Args:
        X_scaled: Normalised patient feature matrix.
        df_clean: Patient DataFrame with RESPONSE_BINARY column.
        algorithms: Dictionary of algorithm name to cluster label arrays.

    Returns:
        DataFrame of per-algorithm metrics.
    """
    baseline_labels = algorithms["K-Means (Current)"]
    metrics_list = []

    for name, labels in algorithms.items():
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        n_noise = int(np.sum(labels == -1))
        valid_mask = labels != -1

        if len(np.unique(labels[valid_mask])) > 1:
            sil = round(silhouette_score(X_scaled[valid_mask], labels[valid_mask]), 3)
            ch = round(calinski_harabasz_score(X_scaled[valid_mask], labels[valid_mask]), 1)
            db = round(davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask]), 3)
        else:
            sil, ch, db = float("nan"), float("nan"), float("nan")

        ari = round(adjusted_rand_score(baseline_labels, labels), 3)

        df_temp = df_clean.copy()
        df_temp["Cluster"] = labels
        resp_rates = (
            df_temp[df_temp["Cluster"] != -1].groupby("Cluster")["RESPONSE_BINARY"].mean() * 100
        )
        rr_range = round(resp_rates.max() - resp_rates.min(), 1) if len(resp_rates) > 1 else float("nan")

        metrics_list.append({
            "Algorithm": name,
            "Clusters (K)": n_clusters,
            "Noise Count": n_noise,
            "Silhouette Score (High)": sil,
            "Calinski-Harabasz (High)": ch,
            "Davies-Bouldin (Low)": db,
            "ARI vs K-Means": ari,
            "Response Rate Spread (%)": rr_range,
        })

    return pd.DataFrame(metrics_list)


def plot_comparison(
    df_clean: pd.DataFrame,
    algorithms: Dict[str, np.ndarray],
    df_metrics: pd.DataFrame,
) -> None:
    """Generate and save a 6-panel 2D PCA scatter plot comparing all clustering algorithms.

    All panels project patients into the same 2D PCA coordinate space for a
    directly comparable visual layout. Cluster colors follow Okabe-Ito palette.
    Cluster labels show N and immunotherapy response rate per group.

    Args:
        df_clean: Patient DataFrame with Dim1/Dim2 PCA coordinates and RESPONSE_BINARY.
        algorithms: Dictionary of algorithm name to cluster label arrays.
        df_metrics: Metrics DataFrame for per-panel silhouette score annotations.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes = axes.flatten()

    n_patients = len(df_clean)
    n_features = len(CLUSTERING_FEATURES)

    for idx, (name, labels) in enumerate(algorithms.items()):
        ax = axes[idx]
        df_plot = df_clean.copy()
        df_plot["Cluster"] = labels

        for cid in sorted(np.unique(labels)):
            sub = df_plot[df_plot["Cluster"] == cid]
            if cid == -1:
                color, marker = "#999999", "x"
                label_name = f"Noise (N={len(sub)})"
            else:
                color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
                marker = "o"
                rr = sub["RESPONSE_BINARY"].mean() * 100
                label_name = f"C{cid} (N={len(sub)}, RR={rr:.1f}%)"

            ax.scatter(
                sub["Dim1"], sub["Dim2"],
                c=color, label=label_name, s=40, alpha=0.8,
                marker=marker,
                edgecolor="white" if marker == "o" else color,
                linewidth=0.3,
            )

        sil_val = df_metrics.loc[df_metrics["Algorithm"] == name, "Silhouette Score (High)"].values[0]
        ax.set_title(f"{name}\n(Silhouette: {sil_val})", fontsize=11, fontweight="bold")
        var1 = pca.explained_variance_ratio_[0] * 100
        var2 = pca.explained_variance_ratio_[1] * 100
        ax.set_xlabel(f"PC1 ({var1:.1f}% Var)", fontsize=9)
        ax.set_ylabel(f"PC2 ({var2:.1f}% Var)", fontsize=9)
        ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)

    plt.suptitle(
        f"Clustering Algorithms Benchmark Comparison (N={n_patients} Patients, {n_features} Multi-Modal Features)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, OUTPUT_PLOT)
    print(f"Saved comparison figure to: {rel_path(OUTPUT_PLOT)}")


def main() -> None:
    """Main execution function for clustering algorithm benchmark."""
    print("=" * 80)
    print("CLUSTERING ALGORITHM BENCHMARK COMPARISON")
    print("=" * 80)

    df_clean, X_scaled, feature_cols = load_and_prepare_features()

    # 2D PCA projection for consistent visual comparison across all algorithm panels
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df_clean["Dim1"] = coords[:, 0]
    df_clean["Dim2"] = coords[:, 1]

    algorithms = run_all_algorithms(X_scaled)
    df_metrics = compute_metrics(X_scaled, df_clean, algorithms)

    OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(OUTPUT_METRICS, index=False)

    print("\nEVALUATION METRICS TABLE:")
    print(df_metrics.to_string(index=False))

    plot_comparison(df_clean, algorithms, df_metrics)

    print(f"Saved metrics CSV to:       {rel_path(OUTPUT_METRICS)}")
    print("=" * 80)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
