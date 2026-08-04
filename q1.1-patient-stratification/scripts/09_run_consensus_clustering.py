#!/usr/bin/env python3
"""Script 09: Consensus Clustering Pipeline for Q5 patient stratification.

Executes a 1,000-bootstrap consensus clustering ensemble across patients (80% ratio)
and features (80% ratio) for K in [2, 8]. Evaluates Consensus Cumulative Distribution
Functions (CDF), CDF Area Under Curve (AUC), relative Delta Area scores Δ(K), and
consensus co-association heatmaps to establish cluster stability.
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
    plot_consensus_cdf_and_delta_area,
    plot_consensus_heatmap,
    prepare_clustering_features,
    run_consensus_bootstrap,
)
from src.styles import set_presentation_style
from src.utils.io import safe_save_csv
from src.utils.logging import TeeStream
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, rel_path

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "09_run_consensus_clustering.log"

INPUT_FILE_FULL = PROCESSED_DIR / "q5" / "feature_matrix_full.csv"
OUTPUT_DIR = PROCESSED_DIR / "q5"
PLOTS_DIR = SUBPROJECT_ROOT / "plots" / "clustering"

N_BOOTSTRAPS = 1000
SAMPLE_RATIO = 0.8
FEATURE_RATIO = 0.8
K_RANGE = range(2, 9)

set_presentation_style()


def main() -> None:
    """Main execution function for 1,000-bootstrap Consensus Clustering across K in [2, 8]."""
    print(f"Starting 1,000-Bootstrap Consensus Clustering Pipeline (Project root: {rel_path(PROJECT_ROOT)})")
    if not INPUT_FILE_FULL.exists():
        raise FileNotFoundError(f"Missing feature matrix at {rel_path(INPUT_FILE_FULL)}. Run Script 01 first.")

    df_matrix = pd.read_csv(INPUT_FILE_FULL)
    print(f"Loaded full cohort matrix: {len(df_matrix)} patients x {df_matrix.shape[1]} features")

    df_clean, X_scaled = prepare_clustering_features(df_matrix)
    print(f"Prepared {len(df_clean)} patients x {X_scaled.shape[1]} features for consensus bootstrap.")

    print(f"\nRunning Consensus Clustering ensemble ({N_BOOTSTRAPS} bootstraps, K in {list(K_RANGE)})...")
    consensus_matrices, cdf_curves, auc_dict, delta_area_dict, metrics_df = run_consensus_bootstrap(
        X_scaled,
        k_range=K_RANGE,
        n_bootstraps=N_BOOTSTRAPS,
        sample_ratio=SAMPLE_RATIO,
        feature_ratio=FEATURE_RATIO,
        random_state=42,
    )

    out_csv = OUTPUT_DIR / "consensus_clustering_k2_k8_metrics.csv"
    safe_save_csv(metrics_df, out_csv)
    print(f"Saved consensus metrics table to {rel_path(out_csv)}")
    print("\nConsensus Clustering Metrics Summary:")
    print(metrics_df.to_string(index=False))

    out_cdf = PLOTS_DIR / "consensus_cdf_curves.png"
    out_delta = PLOTS_DIR / "consensus_delta_area.png"
    plot_consensus_cdf_and_delta_area(cdf_curves, delta_area_dict, out_cdf, out_delta)

    out_heatmap_k4 = PLOTS_DIR / "consensus_heatmap_k4.png"
    if 4 in consensus_matrices:
        plot_consensus_heatmap(consensus_matrices[4], out_heatmap_k4, k=4)

    print("=" * 80)
    print("CONSENSUS CLUSTERING BOOTSTRAP PIPELINE COMPLETE")
    print(f"Metrics Output : {rel_path(out_csv)}")
    print(f"CDF Curves Plot: {rel_path(out_cdf)}")
    print(f"Delta Area Plot: {rel_path(out_delta)}")
    print(f"Heatmap (K=4)  : {rel_path(out_heatmap_k4)}")
    print("=" * 80)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
