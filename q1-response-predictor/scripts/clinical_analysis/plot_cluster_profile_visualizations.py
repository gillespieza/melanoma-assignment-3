"""
Cluster Profile Visualisations for Clinical Phenotyping.

Generates presentation-ready multi-panel dashboards and radar profile fingerprint charts
for patient clusters identified via Ward's hierarchical clustering on TCGA-SKCM clinical data.
"""

import contextlib
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.styles import set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import find_project_root, rel_path
from src.utils.plotting import save_fig

# Module-level Constants
DATA_DIR = find_project_root(Path(__file__).resolve()) / "data"
PLOT_DIR = find_project_root(Path(__file__).resolve()) / "plots" / "clinical"
LOG_DIR = find_project_root(Path(__file__).resolve()) / "logs"
LOG_PATH = LOG_DIR / "plot_cluster_profile_visualizations.log"

CLUSTER_PALETTE = {
    "Cluster 0: Baseline (N=312)": "#37474F",
    "Cluster 1: High TMB/IO (N=112)": "#009E73",
    "Cluster 2: Stage IV (N=24)": "#D55E00",
}


def _plot_cluster_radar(plot_dir: Path) -> None:
    """Generates multi-dimensional polar radar fingerprint chart for clinical clusters.

    Args:
        plot_dir: Directory path to export plot artifact.
    """
    categories = [
        "TMB (scaled)",
        "Aneuploidy",
        "Hypoxia (milder)",
        "Immunotherapy %",
        "Chemotherapy %",
        "Stage IV %",
        "Median OS (scaled)",
    ]
    n_vars = len(categories)

    angles = [n / float(n_vars) * 2 * np.pi for n in range(n_vars)]
    angles += angles[:1]

    values_c0 = [24.9 / 35.0, 13.1 / 15.0, 1 - (2.57 / 3.5), 0.0, 0.0, 0.0, 93.0 / 100.0]
    values_c0 += values_c0[:1]

    values_c1 = [31.4 / 35.0, 12.9 / 15.0, 1 - (1.61 / 3.5), 0.634, 0.58, 0.0, 66.5 / 100.0]
    values_c1 += values_c1[:1]

    values_c2 = [14.0 / 35.0, 11.8 / 15.0, 1 - (2.33 / 3.5), 0.0, 0.125, 1.0, 28.1 / 100.0]
    values_c2 += values_c2[:1]

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True), dpi=300)

    plt.xticks(angles[:-1], categories, color="black", size=10, weight="bold")

    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.05)

    ax.plot(
        angles,
        values_c0,
        linewidth=2.5,
        linestyle="solid",
        label="Cluster 0: Baseline (N=312)",
        color=CLUSTER_PALETTE["Cluster 0: Baseline (N=312)"],
    )
    ax.fill(angles, values_c0, color=CLUSTER_PALETTE["Cluster 0: Baseline (N=312)"], alpha=0.15)

    ax.plot(
        angles,
        values_c1,
        linewidth=2.5,
        linestyle="solid",
        label="Cluster 1: High TMB/IO (N=112)",
        color=CLUSTER_PALETTE["Cluster 1: High TMB/IO (N=112)"],
    )
    ax.fill(angles, values_c1, color=CLUSTER_PALETTE["Cluster 1: High TMB/IO (N=112)"], alpha=0.15)

    ax.plot(
        angles,
        values_c2,
        linewidth=2.5,
        linestyle="solid",
        label="Cluster 2: Stage IV (N=24)",
        color=CLUSTER_PALETTE["Cluster 2: Stage IV (N=24)"],
    )
    ax.fill(angles, values_c2, color=CLUSTER_PALETTE["Cluster 2: Stage IV (N=24)"], alpha=0.15)

    plt.title("Multi-Dimensional Phenotype Fingerprint (Radar Plot)", size=14, weight="bold", pad=25)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True)

    plt.tight_layout()
    fig2_path = plot_dir / "cluster_profile_radar.png"
    save_fig(fig, fig2_path)
    print(f"Saved Cluster Profile Radar Chart to {rel_path(fig2_path)}")


def main() -> None:
    """Executes the cluster profile visualization pipeline."""
    print("==================================================")
    print("Generating Clinical Cluster Profile Visualisations")
    print("==================================================\n")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_cluster_radar(PLOT_DIR)

    print("==================================================")
    print("Done!")
    print("==================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
