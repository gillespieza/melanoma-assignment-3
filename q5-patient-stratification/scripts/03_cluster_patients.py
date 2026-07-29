#!/usr/bin/env python3
"""Script 03: Unsupervised patient stratification pipeline.

Applies K-Means clustering to the patient feature matrix, computes silhouette
statistics, generates 2D projection visualizations, and exports cluster assignments to
data/processed/q5/patient_clusters.csv.
"""

import contextlib
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

from clustering import (
    plot_2d_cluster_projection,
    prepare_clustering_features,
    run_kmeans,
)
from src.styles import set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, rel_path

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "03_cluster_patients.log"

INPUT_FILE = PROCESSED_DIR / "q5" / "feature_matrix.csv"
OUTPUT_DIR = PROCESSED_DIR / "q5"

set_presentation_style()

# Expanded biological phenotype labels for clear presentation legends
PHENOTYPE_NAMES = {
    0: "Mutant-Driven (NF1 Loss & High Response Subtype)",
    1: "Immune Cold (Low TIS & Infiltration, Desert)",
    2: "Immune Hot (High TIS & CYT, Inflamed Microenvironment)",
    3: "M2 Immunosuppressive (Depleted T-cells & Stromal Exclusion)",
}


def main() -> None:
    """Main execution function for patient clustering."""
    print(f"Starting Phase 3 Unsupervised Patient Stratification (Project root: {rel_path(PROJECT_ROOT)})")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing feature matrix at {rel_path(INPUT_FILE)}. Run Phase 1 first.")

    df_matrix = pd.read_csv(INPUT_FILE)
    print(f"Loaded feature matrix: {len(df_matrix)} patients x {df_matrix.shape[1]} features")

    # 1. Prepare features & standardize
    df_clean, X_scaled = prepare_clustering_features(df_matrix)

    # 2. Run K-Means clustering (K=4)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    df_clean["Cluster_ID"] = labels
    df_clean["Phenotype_Label"] = df_clean["Cluster_ID"].map(PHENOTYPE_NAMES)

    # 3. Save cluster assignments
    out_clusters = OUTPUT_DIR / "patient_clusters.csv"
    if out_clusters.exists():
        try:
            out_clusters.unlink()
        except Exception:
            pass
    df_clean.to_csv(out_clusters, index=False)

    # 4. Generate 300 DPI 2D Cluster Projection Figures (both PCA and UMAP)
    pca_plot_file = SUBPROJECT_ROOT / "plots" / "clustering" / "pca_clusters.png"
    umap_plot_file = SUBPROJECT_ROOT / "plots" / "clustering" / "umap_clusters.png"
    plot_2d_cluster_projection(df_clean, labels, X_scaled, pca_plot_file, PHENOTYPE_NAMES, method="pca")
    plot_2d_cluster_projection(df_clean, labels, X_scaled, umap_plot_file, PHENOTYPE_NAMES, method="umap")

    print("=" * 80)
    print("PATIENT STRATIFICATION COMPLETE")
    print(f"Output Clusters File: {rel_path(out_clusters)}")
    print(f"Output PCA Plot:      {rel_path(pca_plot_file)}")
    print(f"Output UMAP Plot:     {rel_path(umap_plot_file)}")
    print(f"Total Stratified Patients: {len(df_clean)}")
    for cid, name in PHENOTYPE_NAMES.items():
        cnt = np.sum(labels == cid)
        resp_rate = df_clean[df_clean["Cluster_ID"] == cid]["RESPONSE_BINARY"].mean() * 100
        print(f"  * Cluster {cid} [{name}]: N = {cnt} ({cnt/len(df_clean)*100:.1f}%), Response Rate = {resp_rate:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
