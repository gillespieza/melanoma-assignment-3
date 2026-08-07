"""
Cluster Profile Visualisations for Clinical Phenotyping.

Generates a presentation-ready polar radar profile fingerprint chart
for patient clusters identified via Ward's hierarchical clustering,
reading live cluster assignments from the persisted CSV output of
run_clinical_clustering.py.

Prerequisite: run_clinical_clustering.py must be executed first so that
data/processed/merged/clinical_clusters.csv exists.
"""

import contextlib
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.styles import PHENOTYPE_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, PLOTS_DIR, get_subproject_log_dir, rel_path
from src.utils.plotting import build_radar_angles, save_fig

set_presentation_style()

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).resolve()
LOG_DIR = get_subproject_log_dir(_SCRIPT_PATH)
LOG_PATH = LOG_DIR / "plot_cluster_profile_visualizations.log"

PLOT_DIR = PLOTS_DIR / "clinical"

# CSV produced by run_clinical_clustering.py — prerequisite for this script
CLUSTER_CSV_PATH = DATA_DIR / "processed" / "merged" / "clinical_clusters.csv"

# Cluster colour palette: sourced from PHENOTYPE_PALETTE, consistent with
# CLUSTER_COLORS in run_clinical_clustering.py (AGENTS.md Rules 3, 12).
CLUSTER_COLORS: dict[int, str] = {
    0: PHENOTYPE_PALETTE["Immune Hot"],                 # Vermillion Red (#D55E00)
    1: PHENOTYPE_PALETTE["Immune Cold"],                # Okabe-Ito Blue (#0072B2)
    2: PHENOTYPE_PALETTE["Immunosuppressive M2-High"],  # Reddish Purple (#CC79A7)
}

# Short display names aligned with run_clinical_clustering.CLUSTER_PLOT_NAMES
_CLUSTER_LABELS: dict[int, str] = {
    0: "Cluster 0 (Hot)",
    1: "Cluster 1 (Cold)",
    2: "Cluster 2 (High-TMB)",
}

# Radar axis configuration: (display label, source column in cluster CSV)
# These clinical/genomic axes complement (not duplicate) the Z-score axes in
# run_clinical_clustering._plot_cluster_radar.
_RADAR_AXES: list[tuple[str, str]] = [
    ("TMB",         "TMB_NONSYNONYMOUS"),
    ("IFN-γ Score", "IFN_gamma"),
    ("CD8+ T-cell", "CD8_Tcell"),
    ("CYT Score",   "CYT"),
    ("PD-L1 Proxy", "PD_L1"),
    ("IMPRES",      "IMPRES"),
    ("Median OS",   "_os_median"),   # Sentinel: computed from OS_MONTHS
]

_N_CLUSTERS = 3


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_cluster_data() -> pd.DataFrame:
    """Loads the persisted cluster assignment CSV from run_clinical_clustering.py.

    Returns:
        DataFrame with CLINICAL_CLUSTER assignments and feature columns.

    Raises:
        FileNotFoundError: If the cluster CSV does not exist.
    """
    if not CLUSTER_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Cluster data not found at {rel_path(CLUSTER_CSV_PATH)}. "
            "Run run_clinical_clustering.py first to generate cluster assignments."
        )
    return pd.read_csv(CLUSTER_CSV_PATH, encoding="utf-8")


def _compute_radar_values(df: pd.DataFrame) -> dict[int, list[float]]:
    """Computes per-cluster radar axis values, normalised to [0, 1] across clusters.

    Args:
        df: DataFrame with CLINICAL_CLUSTER assignments and feature columns.

    Returns:
        Dict mapping cluster index to a list of normalised axis values.
    """
    raw: dict[int, list[float]] = {}
    for c in range(_N_CLUSTERS):
        sub = df[df["CLINICAL_CLUSTER"] == c]
        row: list[float] = []
        for _, col in _RADAR_AXES:
            if col == "_os_median":
                val = sub["OS_MONTHS"].median() if "OS_MONTHS" in sub.columns else 0.0
            else:
                val = sub[col].mean() if col in sub.columns else 0.0
            row.append(float(np.nan_to_num(val, nan=0.0)))
        raw[c] = row

    # Min-max normalise each axis across clusters so all axes span [0, 1]
    n_axes = len(_RADAR_AXES)
    all_vals = np.array([raw[c] for c in range(_N_CLUSTERS)])
    col_max = all_vals.max(axis=0)
    col_max[col_max == 0] = 1.0  # Prevent zero-division for constant axes

    return {c: (np.array(raw[c]) / col_max).tolist() for c in range(_N_CLUSTERS)}


def _render_radar_chart(
    ax: plt.Axes,
    angles: list[float],
    cluster_values: dict[int, list[float]],
    cluster_ns: dict[int, int],
) -> None:
    """Renders cluster traces onto an existing polar Axes.

    Args:
        ax: Matplotlib polar Axes to draw on.
        angles: Closed-loop angle array from build_radar_angles().
        cluster_values: Normalised [0, 1] radar values per cluster.
        cluster_ns: Patient count per cluster for legend labels.
    """
    categories = [label for label, _ in _RADAR_AXES]
    for c, vals in cluster_values.items():
        closed_vals = vals + vals[:1]
        label = f"{_CLUSTER_LABELS[c]} (N={cluster_ns.get(c, 0)})"
        ax.plot(angles, closed_vals, linewidth=2.5, linestyle="solid",
                label=label, color=CLUSTER_COLORS[c])
        ax.fill(angles, closed_vals, color=CLUSTER_COLORS[c], alpha=0.15)

    plt.xticks(angles[:-1], categories, color="black", size=10, weight="bold")
    ax.set_rlabel_position(0)
    plt.yticks(
        [0.2, 0.4, 0.6, 0.8, 1.0],
        ["0.2", "0.4", "0.6", "0.8", "1.0"],
        color="grey",
        size=8,
    )
    plt.ylim(0, 1.05)


def _plot_cluster_radar(plot_dir: Path) -> None:
    """Generates a multi-dimensional polar radar fingerprint chart for clinical clusters.

    Loads live cluster assignments from CLUSTER_CSV_PATH, computes per-cluster
    axis values dynamically, and saves a PNG to plot_dir.

    Args:
        plot_dir: Directory path to export the plot artifact.
    """
    df = _load_cluster_data()
    cluster_ns: dict[int, int] = df["CLINICAL_CLUSTER"].value_counts().to_dict()
    cluster_values = _compute_radar_values(df)
    angles = build_radar_angles(len(_RADAR_AXES))

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))
    _render_radar_chart(ax, angles, cluster_values, cluster_ns)

    n_total = len(df)
    ax.set_title(
        f"Multi-Dimensional Clinical Profile Fingerprint (N={n_total})",
        size=14,
        weight="bold",
        pad=25,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True)

    out_path = plot_dir / "cluster_profile_radar.png"
    save_fig(fig, out_path)
    print(f"Saved Cluster Profile Radar Chart to {rel_path(out_path)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Executes the cluster profile visualisation pipeline."""
    print("==================================================")
    print("Generating Clinical Cluster Profile Visualisations")
    print("==================================================\n")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_cluster_radar(PLOT_DIR)

    print("\n==================================================")
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
