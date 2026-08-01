#!/usr/bin/env python3
"""Script 03: Unsupervised patient stratification pipeline.

Applies K-Means clustering to the full patient feature matrix (runtime N), computes silhouette
statistics, generates 2D projection visualisations, and exports cluster assignments to
data/processed/q5/patient_clusters.csv.

Also:
  - Saves the fitted KMeans model to data/processed/q5/kmeans_model.pkl for reuse in Phase 7.
  - Saves the clustering feature column list to data/processed/q5/clustering_feature_cols.json.
  - Compares clustering quality metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
    between the ICI-only cohort and the full cohort, and saves a side-by-side bar chart to
    plots/clustering/cohort_size_clustering_comparison.png.
"""

import contextlib
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

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

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from clustering import (
    DEFAULT_N_COMPONENTS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SPECTRAL_NEIGHBORS,
    plot_2d_cluster_projection,
    plot_spatial_microenvironment_violins,
    prepare_clustering_features,
    run_gmm,
    run_spectral_manifold,
    transform_mahalanobis_space,
)
from phenotyping import assign_phenotype_labels, profile_clusters
from q5_constants import (
    CLUSTERING_FEATURES,
    PHENOTYPE_LABEL_SUFFIX,
    PHENOTYPE_PROFILE_FEATURES,
)
from src.styles import OKABE_ITO, set_presentation_style
from src.utils.io import safe_save_csv
from src.utils.logging import TeeStream
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, rel_path
from src.utils.plotting import save_fig

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "03_cluster_patients.log"

INPUT_FILE_FULL = PROCESSED_DIR / "q5" / "feature_matrix_full.csv"
INPUT_FILE_ICI = PROCESSED_DIR / "q5" / "feature_matrix.csv"
OUTPUT_DIR = PROCESSED_DIR / "q5"

set_presentation_style()


def compute_clustering_metrics(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    gmm: Optional[GaussianMixture] = None,
) -> Dict[str, float]:
    """Compute internal clustering quality metrics for a given label assignment and GMM model."""
    valid = labels != -1
    metrics: Dict[str, float] = {}
    if len(np.unique(labels[valid])) >= 2:
        metrics["Silhouette"] = round(float(silhouette_score(X_scaled[valid], labels[valid])), 4)
        metrics["Calinski_Harabasz"] = round(float(calinski_harabasz_score(X_scaled[valid], labels[valid])), 1)
        metrics["Davies_Bouldin"] = round(float(davies_bouldin_score(X_scaled[valid], labels[valid])), 4)
    else:
        metrics = {"Silhouette": float("nan"), "Calinski_Harabasz": float("nan"), "Davies_Bouldin": float("nan")}

    if gmm is not None:
        metrics["Log_Likelihood"] = round(float(gmm.score(X_scaled)), 4)
        metrics["AIC"] = round(float(gmm.aic(X_scaled)), 1)
        metrics["BIC"] = round(float(gmm.bic(X_scaled)), 1)

    return metrics


def plot_cohort_size_comparison(
    metrics_ici: Dict[str, float],
    metrics_full: Dict[str, float],
    n_ici: int,
    n_full: int,
    out_path: Path,
) -> None:
    """Generate a side-by-side bar chart comparing clustering quality across cohort sizes."""
    metric_labels = ["Silhouette\n(higher better)", "Calinski-Harabasz\n(higher better)", "Davies-Bouldin\n(lower better)"]
    metric_keys = ["Silhouette", "Calinski_Harabasz", "Davies_Bouldin"]

    vals_ici = [metrics_ici.get(k, float("nan")) for k in metric_keys]
    vals_full = [metrics_full.get(k, float("nan")) for k in metric_keys]

    x = np.arange(len(metric_labels))
    width = 0.35
    color_ici = OKABE_ITO[0]
    color_full = OKABE_ITO[4]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_ici = ax.bar(x - width / 2, vals_ici, width, label=f"ICI-only (N={n_ici})", color=color_ici, edgecolor="black", linewidth=0.8)
    bars_full = ax.bar(x + width / 2, vals_full, width, label=f"Full cohort (N={n_full})", color=color_full, edgecolor="black", linewidth=0.8)

    for bar in list(bars_ici) + list(bars_full):
        h = bar.get_height()
        if not np.isnan(h):
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.set_title(
        f"GMM Clustering Quality: ICI Cohort (N={n_ici}) vs Full Cohort (N={n_full})",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Internal Clustering Metric", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=10)
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_path, dpi=300)
    print(f"Saved cohort-size clustering comparison to {rel_path(out_path)}")


def _fit_and_save_gmm_model(
    df_clean: pd.DataFrame,
    X_scaled: np.ndarray,
) -> Tuple[GaussianMixture, np.ndarray, np.ndarray, List[str]]:
    """Fit Gaussian Mixture Model (GMM, K=4, full covariance) and persist model artifacts."""
    feature_cols = [c for c in CLUSTERING_FEATURES if c in df_clean.columns]
    gmm, labels, probs = run_gmm(
        X_scaled,
        n_components=DEFAULT_N_COMPONENTS,
        covariance_type="full",
        random_state=DEFAULT_RANDOM_STATE,
    )

    gmm_model_path = OUTPUT_DIR / "gmm_model.pkl"
    legacy_model_path = OUTPUT_DIR / "kmeans_model.pkl"
    features_path = OUTPUT_DIR / "clustering_feature_cols.json"

    joblib.dump(gmm, gmm_model_path)
    joblib.dump(gmm, legacy_model_path)
    with open(features_path, "w", encoding="utf-8") as fp:
        json.dump(feature_cols, fp)

    print(f"Saved GMM model to {rel_path(gmm_model_path)} (and legacy fallback {rel_path(legacy_model_path)})")
    print(f"Saved clustering feature list to {rel_path(features_path)}")
    return gmm, labels, probs, feature_cols


def _assign_labels(df_clean: pd.DataFrame, labels: np.ndarray) -> Dict[int, str]:
    """Derive biological phenotype labels data-driven from cluster profiles."""
    profile_feature_cols = [c for c in PHENOTYPE_PROFILE_FEATURES if c in df_clean.columns]
    cluster_profiles = profile_clusters(df_clean, "Cluster_ID", profile_feature_cols)
    short_labels: Dict[int, str] = assign_phenotype_labels(cluster_profiles)

    phenotype_names: Dict[int, str] = {
        cid: f"{short} {PHENOTYPE_LABEL_SUFFIX.get(short, '')}".strip()
        for cid, short in short_labels.items()
    }
    print("Data-driven phenotype label assignment:")
    for cid, name in sorted(phenotype_names.items()):
        profile_tis = cluster_profiles.loc[cid, "TIS"] if "TIS" in cluster_profiles.columns else float("nan")
        profile_nf1 = cluster_profiles.loc[cid, "mut_NF1"] if "mut_NF1" in cluster_profiles.columns else float("nan")
        print(f"  Cluster {cid}: TIS={profile_tis:+.3f}, `NF1`={profile_nf1:.0%}  ->  {name}")

    return phenotype_names


def _format_prob_col_name(phenotype_name: str) -> str:
    """Format biological phenotype label into a clean probability DataFrame column name."""
    short_label = phenotype_name.split("(")[0].strip()
    clean_label = short_label.replace(" ", "_").replace("-", "_")
    return f"P_{clean_label}"


def _export_cluster_outputs(
    df_clean: pd.DataFrame,
    probs: np.ndarray,
    phenotype_names: Dict[int, str],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Export posterior probabilities CSV and patient clusters CSV to disk."""
    for cid, name in phenotype_names.items():
        clean_name = _format_prob_col_name(name)
        df_clean[clean_name] = probs[:, cid]

    prob_cols = [c for c in df_clean.columns if c.startswith("P_")]
    id_cols = [c for c in ["PATIENT_ID", "sample_id", "patient_id"] if c in df_clean.columns]
    prob_df_cols = id_cols + ["Cluster_ID", "Phenotype_Label"] + prob_cols
    df_probs_export = df_clean[[c for c in prob_df_cols if c in df_clean.columns]].copy()

    out_probs_file = output_dir / "gmm_posterior_probabilities.csv"
    safe_save_csv(df_probs_export, out_probs_file)
    print(f"Saved GMM posterior probabilities matrix to {rel_path(out_probs_file)}")

    out_clusters = output_dir / "patient_clusters.csv"
    safe_save_csv(df_clean, out_clusters)
    return out_probs_file, out_clusters


def _generate_phase3_plots(
    df_clean: pd.DataFrame,
    labels: np.ndarray,
    X_scaled: np.ndarray,
    phenotype_names: Dict[int, str],
    subproject_root: Path,
) -> None:
    """Generate 2D PCA/t-SNE projection maps and spatial microenvironment violin plots."""
    pca_plot_file = subproject_root / "plots" / "clustering" / "pca_clusters.png"
    tsne_plot_file = subproject_root / "plots" / "clustering" / "tsne_clusters.png"
    plot_2d_cluster_projection(df_clean, labels, X_scaled, pca_plot_file, phenotype_names, method="pca")
    plot_2d_cluster_projection(df_clean, labels, X_scaled, tsne_plot_file, phenotype_names, method="tsne")

    spatial_violin_file = subproject_root / "plots" / "clustering" / "spatial_microenvironment_violins.png"
    plot_spatial_microenvironment_violins(df_clean, spatial_violin_file)


def _evaluate_mahalanobis_spectral_comparisons(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    gmm: GaussianMixture,
    output_dir: Path,
) -> Dict[str, float]:
    """Evaluate Mahalanobis GMM transformation and Spectral Manifold clustering benchmarks."""
    X_mahalanobis, _ = transform_mahalanobis_space(X_scaled)
    gmm_mah, labels_mah, _ = run_gmm(
        X_mahalanobis,
        n_components=DEFAULT_N_COMPONENTS,
        covariance_type="full",
        random_state=DEFAULT_RANDOM_STATE,
    )
    _, spectral_labels = run_spectral_manifold(
        X_scaled,
        n_clusters=DEFAULT_N_COMPONENTS,
        n_neighbors=DEFAULT_SPECTRAL_NEIGHBORS,
        random_state=DEFAULT_RANDOM_STATE,
    )

    gmm_metrics = compute_clustering_metrics(X_scaled, labels, gmm)
    mah_metrics = compute_clustering_metrics(X_mahalanobis, labels_mah, gmm_mah)
    spectral_metrics = compute_clustering_metrics(X_scaled, spectral_labels)

    comp_df = pd.DataFrame([
        {"Method": "GMM (Standard Scaled + Spatial)", **gmm_metrics},
        {"Method": "GMM (Mahalanobis Transformed)", **mah_metrics},
        {"Method": "Spectral Manifold (Graph Laplacian)", **spectral_metrics},
    ])
    out_metrics_file = output_dir / "mahalanobis_spectral_metrics.csv"
    safe_save_csv(comp_df, out_metrics_file)
    print(f"Saved Mahalanobis & Spectral metrics comparison to {rel_path(out_metrics_file)}")
    return gmm_metrics


def _compare_cohort_quality(
    gmm: GaussianMixture,
    feature_cols: List[str],
    metrics_full: Dict[str, float],
    len_full: int,
) -> None:
    """Evaluate clustering quality comparison between ICI-only and full cohort."""
    if not INPUT_FILE_ICI.exists():
        print(f"ICI matrix not found at {rel_path(INPUT_FILE_ICI)} — skipping comparison figure.")
        return

    df_ici = pd.read_csv(INPUT_FILE_ICI)
    ici_cols = [c for c in feature_cols if c in df_ici.columns]
    df_ici_clean = df_ici.dropna(subset=ici_cols).copy()
    X_ici = StandardScaler().fit_transform(df_ici_clean[ici_cols])
    probs_ici = gmm.predict_proba(X_ici)
    labels_ici = np.argmax(probs_ici, axis=1)
    metrics_ici = compute_clustering_metrics(X_ici, labels_ici, gmm)

    print(f"\nClustering quality — ICI-only cohort applied to same GMM model (N={len(df_ici_clean)}):")
    for k, v in metrics_ici.items():
        print(f"  {k}: {v}")

    comparison_plot = SUBPROJECT_ROOT / "plots" / "clustering" / "cohort_size_clustering_comparison.png"
    plot_cohort_size_comparison(metrics_ici, metrics_full, len(df_ici_clean), len_full, comparison_plot)


def main() -> None:
    """Main execution function for patient clustering on full cohort using GMM soft clustering & Mahalanobis space."""
    print(
        f"Starting Phase 3 Unsupervised Patient Stratification "
        f"(GMM + Mahalanobis + Spatial, Project root: {rel_path(PROJECT_ROOT)})"
    )
    if not INPUT_FILE_FULL.exists():
        raise FileNotFoundError(
            f"Missing full-cohort feature matrix at {rel_path(INPUT_FILE_FULL)}. "
            "Run 01_load_and_prepare.py first."
        )

    df_matrix = pd.read_csv(INPUT_FILE_FULL)
    print(f"Loaded full-cohort feature matrix: {len(df_matrix)} patients x {df_matrix.shape[1]} features")

    df_clean, X_scaled = prepare_clustering_features(df_matrix)
    gmm, labels, probs, feature_cols = _fit_and_save_gmm_model(df_clean, X_scaled)
    df_clean["Cluster_ID"] = labels

    for k in range(probs.shape[1]):
        df_clean[f"P_Cluster_{k}"] = probs[:, k]

    phenotype_names = _assign_labels(df_clean, labels)
    df_clean["Phenotype_Label"] = df_clean["Cluster_ID"].map(phenotype_names)

    out_probs_file, out_clusters = _export_cluster_outputs(df_clean, probs, phenotype_names, OUTPUT_DIR)
    _generate_phase3_plots(df_clean, labels, X_scaled, phenotype_names, SUBPROJECT_ROOT)
    gmm_metrics = _evaluate_mahalanobis_spectral_comparisons(X_scaled, labels, gmm, OUTPUT_DIR)
    _compare_cohort_quality(gmm, feature_cols, gmm_metrics, len(df_clean))

    print("=" * 80)
    print("PATIENT STRATIFICATION COMPLETE (GMM + Mahalanobis + Spatial Microenvironment)")
    print(f"Output Clusters File : {rel_path(out_clusters)}")
    print(f"Posterior Probs File : {rel_path(out_probs_file)}")
    print(f"Metrics Output File  : {rel_path(OUTPUT_DIR / 'mahalanobis_spectral_metrics.csv')}")
    print(f"Total Stratified Patients: {len(df_clean)}")
    for cid, name in sorted(phenotype_names.items()):
        cnt = int(np.sum(labels == cid))
        is_ici = df_clean[df_clean["Cluster_ID"] == cid]["IMMUNOTHERAPY"] == 1 if "IMMUNOTHERAPY" in df_clean.columns else False
        n_ici = int(is_ici.sum()) if "IMMUNOTHERAPY" in df_clean.columns else 0
        mean_p = np.mean(probs[:, cid])
        print(
            f"  * Cluster {cid} [{name}]: N={cnt} ({cnt / len(df_clean) * 100:.1f}%), "
            f"Mean P={mean_p:.3f}, ICI-treated={n_ici}"
        )
    print("=" * 80)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
