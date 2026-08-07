"""
Cluster Profile Visualisations for Clinical Phenotyping.

Generates a presentation-ready 2D UMAP profile scatter plot for patient clusters
identified via Ward's hierarchical clustering, reading live cluster assignments from
the persisted CSV output of run_clinical_clustering.py.

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
import seaborn as sns
import umap

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.styles import PHENOTYPE_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, PLOTS_DIR, get_subproject_log_dir, rel_path
from src.utils.plotting import save_fig

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

# Cluster colour palette: sourced from PHENOTYPE_PALETTE
CLUSTER_COLORS: dict[int, str] = {
    0: PHENOTYPE_PALETTE["Immune Hot"],                 # Vermillion Red (#D55E00)
    1: PHENOTYPE_PALETTE["Immune Cold"],                # Okabe-Ito Blue (#0072B2)
    2: PHENOTYPE_PALETTE["Immunosuppressive M2-High"],  # Reddish Purple (#CC79A7)
}

_CLUSTER_LABELS: dict[int, str] = {
    0: "Cluster 0 (Hot)",
    1: "Cluster 1 (Cold)",
    2: "Cluster 2 (High-TMB)",
}

_FEATURE_COLS: list[str] = [
    "IFN_gamma",
    "TIS",
    "CYT",
    "CD8_Tcell",
    "IMPRES",
    "PD_L1",
    "mut_BRAF",
    "mut_NRAS",
    "mut_NF1",
    "TMB_NONSYNONYMOUS",
    "M1_M2_Ratio",
    "Macrophage_STV",
]


def _load_cluster_data() -> pd.DataFrame:
    """Loads the persisted cluster assignment CSV from run_clinical_clustering.py."""
    if not CLUSTER_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Cluster data not found at {rel_path(CLUSTER_CSV_PATH)}. "
            "Run run_clinical_clustering.py first to generate cluster assignments."
        )
    return pd.read_csv(CLUSTER_CSV_PATH, encoding="utf-8")


def _plot_cluster_umap(plot_dir: Path) -> None:
    """Generates a 2D UMAP scatter plot for clinical clusters."""
    df = _load_cluster_data()
    z_cols = [f"Z_{c}" for c in _FEATURE_COLS if f"Z_{c}" in df.columns]
    if not z_cols:
        z_cols = [c for c in _FEATURE_COLS if c in df.columns]

    z_matrix = df[z_cols].values

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_coords = reducer.fit_transform(z_matrix)

    df_umap = pd.DataFrame(umap_coords, columns=["UMAP1", "UMAP2"], index=df.index)
    df_umap["Cluster"] = df["CLINICAL_CLUSTER"]
    df_umap["Cluster_Name"] = df_umap["Cluster"].map(_CLUSTER_LABELS)

    palette_dict = {
        _CLUSTER_LABELS[0]: CLUSTER_COLORS[0],
        _CLUSTER_LABELS[1]: CLUSTER_COLORS[1],
        _CLUSTER_LABELS[2]: CLUSTER_COLORS[2],
    }
    fig, ax_umap = plt.subplots(figsize=(9, 7.5))
    hue_order = [_CLUSTER_LABELS[0], _CLUSTER_LABELS[1], _CLUSTER_LABELS[2]]

    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue="Cluster_Name",
        style="Cluster_Name",
        data=df_umap,
        hue_order=hue_order,
        palette=palette_dict,
        alpha=0.8,
        s=90,
        ax=ax_umap,
        edgecolor="w",
        linewidth=0.6,
    )

    for c in range(3):
        centroid = df_umap[df_umap["Cluster"] == c][["UMAP1", "UMAP2"]].mean()
        ax_umap.scatter(
            centroid["UMAP1"],
            centroid["UMAP2"],
            marker="X",
            s=240,
            color="black",
            edgecolor="white",
            linewidth=1.5,
            zorder=10,
        )

    n_total = len(df)
    ax_umap.set_title(
        f"2D UMAP Projection of Patient Subtypes (N={n_total})",
        size=14,
        weight="bold",
        pad=15,
    )
    ax_umap.set_xlabel("UMAP Dimension 1", fontsize=12)
    ax_umap.set_ylabel("UMAP Dimension 2", fontsize=12)

    handles, labels = ax_umap.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_umap.legend(by_label.values(), by_label.keys(), title="Patient Subtypes", loc="best", fontsize=10)

    out_path = plot_dir / "cluster_profile_umap.png"
    save_fig(fig, out_path)
    print(f"Saved Cluster Profile UMAP Chart to {rel_path(out_path)}")


def main() -> None:
    """Executes the cluster profile visualisation pipeline."""
    print("==================================================")
    print("Generating Clinical Cluster Profile Visualisations")
    print("==================================================\n")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_cluster_umap(PLOT_DIR)

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
