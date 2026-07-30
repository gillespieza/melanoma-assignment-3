#!/usr/bin/env python3
"""Script 03: Unsupervised patient stratification pipeline.

Applies K-Means clustering to the FULL patient feature matrix (N≈699), computes silhouette
statistics, generates 2D projection visualisations, and exports cluster assignments to
data/processed/q5/patient_clusters.csv.

Also:
  - Saves the fitted KMeans model to data/processed/q5/kmeans_model.pkl for reuse in Phase 7.
  - Saves the clustering feature column list to data/processed/q5/clustering_feature_cols.json.
  - Compares clustering quality metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
    between the ICI-only cohort (N≈326) and the full cohort (N≈699), and saves a
    side-by-side bar chart to plots/clustering/cohort_size_clustering_comparison.png.
"""

import contextlib
import json
from pathlib import Path
import sys
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
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from clustering import (
    plot_2d_cluster_projection,
    prepare_clustering_features,
    run_kmeans,
)
from phenotyping import assign_phenotype_labels, profile_clusters
from src.styles import OKABE_ITO, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, rel_path
from src.utils.plotting import save_fig

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "03_cluster_patients.log"

# Full cohort (N≈699) — used to fit the production KMeans model
INPUT_FILE_FULL = PROCESSED_DIR / "q5" / "feature_matrix_full.csv"
# ICI-only cohort (N≈326) — used only for clustering quality comparison
INPUT_FILE_ICI  = PROCESSED_DIR / "q5" / "feature_matrix.csv"

OUTPUT_DIR = PROCESSED_DIR / "q5"

set_presentation_style()

# Descriptive biological suffixes appended to each data-driven short phenotype label.
PHENOTYPE_LABEL_SUFFIX: dict = {
    "Immune Hot":               "(High TIS & CYT, Inflamed Microenvironment)",
    "Immune Cold":              "(Low TIS & Infiltration, Desert)",
    "Mutant-Driven":            "(NF1 Loss & High Response Subtype)",
    "Immunosuppressive M2-High": "(Depleted T-cells & Stromal Exclusion)",
}


def compute_clustering_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> dict:
    """Compute internal clustering quality metrics for a given label assignment.

    Args:
        X_scaled: StandardScaler-normalised feature matrix.
        labels:   Cluster label array (integer per patient).

    Returns:
        Dictionary with Silhouette, Calinski-Harabasz, and Davies-Bouldin scores.
    """
    valid = labels != -1
    if len(np.unique(labels[valid])) < 2:
        return {"Silhouette": float("nan"), "Calinski_Harabasz": float("nan"), "Davies_Bouldin": float("nan")}
    return {
        "Silhouette":         round(silhouette_score(X_scaled[valid], labels[valid]), 4),
        "Calinski_Harabasz":  round(calinski_harabasz_score(X_scaled[valid], labels[valid]), 1),
        "Davies_Bouldin":     round(davies_bouldin_score(X_scaled[valid], labels[valid]), 4),
    }


def plot_cohort_size_comparison(
    metrics_ici: dict,
    metrics_full: dict,
    n_ici: int,
    n_full: int,
    out_path: Path,
) -> None:
    """Generate a side-by-side bar chart comparing clustering quality across cohort sizes.

    Args:
        metrics_ici:  Metric dict for ICI-only (N=n_ici) clustering.
        metrics_full: Metric dict for full-cohort (N=n_full) clustering.
        n_ici:        Number of patients in the ICI cohort.
        n_full:       Number of patients in the full cohort.
        out_path:     File path to save the 300 DPI PNG figure.
    """
    metric_labels = ["Silhouette\n(higher better)", "Calinski-Harabasz\n(higher better)", "Davies-Bouldin\n(lower better)"]
    metric_keys   = ["Silhouette", "Calinski_Harabasz", "Davies_Bouldin"]

    vals_ici  = [metrics_ici.get(k, float("nan"))  for k in metric_keys]
    vals_full = [metrics_full.get(k, float("nan")) for k in metric_keys]

    x     = np.arange(len(metric_labels))
    width = 0.35
    color_ici  = OKABE_ITO[0]   # Okabe-Ito Blue — ICI cohort
    color_full = OKABE_ITO[4]   # Okabe-Ito Bluish Green — full cohort

    fig, ax = plt.subplots(figsize=(11, 6))

    bars_ici  = ax.bar(x - width / 2, vals_ici,  width, label=f"ICI-only (N={n_ici})",  color=color_ici,  edgecolor="black", linewidth=0.8)
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
        f"K-Means Clustering Quality: ICI Cohort (N={n_ici}) vs Full Cohort (N={n_full})",
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


def main() -> None:
    """Main execution function for patient clustering on the full N=699 cohort."""
    print(f"Starting Phase 3 Unsupervised Patient Stratification (Project root: {rel_path(PROJECT_ROOT)})")

    if not INPUT_FILE_FULL.exists():
        raise FileNotFoundError(
            f"Missing full-cohort feature matrix at {rel_path(INPUT_FILE_FULL)}. "
            "Run 01_load_and_prepare.py first."
        )

    df_matrix = pd.read_csv(INPUT_FILE_FULL)
    print(f"Loaded full-cohort feature matrix: {len(df_matrix)} patients x {df_matrix.shape[1]} features")

    # 1. Prepare features & standardise
    df_clean, X_scaled = prepare_clustering_features(df_matrix)
    feature_cols = [c for c in df_clean.columns if c in df_matrix.columns and df_clean[c].dtype != object]

    # Infer the feature columns used by prepare_clustering_features by comparing shapes
    # (clustering module standardises a fixed set; we reconstruct the list for persistence)
    clustering_candidate_cols = [
        "TIS", "CYT", "CD8_T_cells", "M1_Macrophages", "M2_Macrophages",
        "CAFs", "mut_BRAF", "mut_NRAS", "mut_NF1",
    ]
    feature_cols = [c for c in clustering_candidate_cols if c in df_clean.columns]

    # 2. Fit K-Means (K=4) on the full cohort
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df_clean["Cluster_ID"] = labels

    # 3. Persist the fitted model and feature list for Phase 7 reuse
    model_path    = OUTPUT_DIR / "kmeans_model.pkl"
    features_path = OUTPUT_DIR / "clustering_feature_cols.json"
    joblib.dump(kmeans, model_path)
    with open(features_path, "w", encoding="utf-8") as fp:
        json.dump(feature_cols, fp)
    print(f"Saved KMeans model to {rel_path(model_path)}")
    print(f"Saved clustering feature list to {rel_path(features_path)}")

    # 4. Derive phenotype labels data-driven from cluster profiles
    profile_feature_cols = [c for c in ["TIS", "CYT", "CD8_T_cells", "mut_NF1", "M1_M2_Ratio", "M2_Macrophages"] if c in df_clean.columns]
    cluster_profiles = profile_clusters(df_clean, "Cluster_ID", profile_feature_cols)
    short_labels: dict = assign_phenotype_labels(cluster_profiles)

    phenotype_names: dict = {
        cid: f"{short} {PHENOTYPE_LABEL_SUFFIX.get(short, '')}".strip()
        for cid, short in short_labels.items()
    }
    print("Data-driven phenotype label assignment:")
    for cid, name in sorted(phenotype_names.items()):
        profile_tis = cluster_profiles.loc[cid, "TIS"] if "TIS" in cluster_profiles.columns else float("nan")
        profile_nf1 = cluster_profiles.loc[cid, "mut_NF1"] if "mut_NF1" in cluster_profiles.columns else float("nan")
        print(f"  Cluster {cid}: TIS={profile_tis:+.3f}, `NF1`={profile_nf1:.0%}  ->  {name}")

    df_clean["Phenotype_Label"] = df_clean["Cluster_ID"].map(phenotype_names)

    # 5. Save cluster assignments (N=699)
    out_clusters = OUTPUT_DIR / "patient_clusters.csv"
    if out_clusters.exists():
        try:
            out_clusters.unlink()
        except Exception:
            pass
    df_clean.to_csv(out_clusters, index=False)

    # 6. Generate 300 DPI 2D cluster projection figures (PCA and UMAP)
    pca_plot_file  = SUBPROJECT_ROOT / "plots" / "clustering" / "pca_clusters.png"
    umap_plot_file = SUBPROJECT_ROOT / "plots" / "clustering" / "umap_clusters.png"
    plot_2d_cluster_projection(df_clean, labels, X_scaled, pca_plot_file,  phenotype_names, method="pca")
    plot_2d_cluster_projection(df_clean, labels, X_scaled, umap_plot_file, phenotype_names, method="umap")

    # 7. Clustering quality comparison: ICI-only (N≈326) vs Full (N≈699)
    metrics_full = compute_clustering_metrics(X_scaled, labels)
    print(f"\nClustering quality — Full cohort (N={len(df_clean)}):")
    for k, v in metrics_full.items():
        print(f"  {k}: {v}")

    if INPUT_FILE_ICI.exists():
        df_ici = pd.read_csv(INPUT_FILE_ICI)
        ici_cols = [c for c in feature_cols if c in df_ici.columns]
        df_ici_clean = df_ici.dropna(subset=ici_cols).copy()
        X_ici = StandardScaler().fit_transform(df_ici_clean[ici_cols])
        labels_ici = kmeans.predict(X_ici)
        metrics_ici = compute_clustering_metrics(X_ici, labels_ici)
        print(f"\nClustering quality — ICI-only cohort applied to same model (N={len(df_ici_clean)}):")
        for k, v in metrics_ici.items():
            print(f"  {k}: {v}")

        comparison_plot = SUBPROJECT_ROOT / "plots" / "clustering" / "cohort_size_clustering_comparison.png"
        plot_cohort_size_comparison(metrics_ici, metrics_full, len(df_ici_clean), len(df_clean), comparison_plot)
    else:
        print(f"ICI matrix not found at {rel_path(INPUT_FILE_ICI)} — skipping comparison figure.")
        metrics_ici = {}

    print("=" * 80)
    print("PATIENT STRATIFICATION COMPLETE (Full Cohort)")
    print(f"Output Clusters File : {rel_path(out_clusters)}")
    print(f"KMeans Model         : {rel_path(model_path)}")
    print(f"PCA Plot             : {rel_path(pca_plot_file)}")
    print(f"UMAP Plot            : {rel_path(umap_plot_file)}")
    print(f"Total Stratified Patients: {len(df_clean)}")
    for cid, name in sorted(phenotype_names.items()):
        cnt = int(np.sum(labels == cid))
        n_ici_in_cluster = int((df_clean[df_clean["Cluster_ID"] == cid]["IMMUNOTHERAPY"] == 1).sum()) if "IMMUNOTHERAPY" in df_clean.columns else 0
        print(f"  * Cluster {cid} [{name}]: N={cnt} ({cnt/len(df_clean)*100:.1f}%), ICI-treated={n_ici_in_cluster}")
    print("=" * 80)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
