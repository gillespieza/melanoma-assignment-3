"""Phenotype characterisation and profiling module for Q5.

Provides helpers for summarizing cluster profiles, assigning biological
phenotype labels, and plotting radar charts, annotated cluster heatmaps,
and facetted biomarker violin plots.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from src.styles import RESPONSE_PALETTE, set_presentation_style

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

set_presentation_style()


def profile_clusters(df: pd.DataFrame, cluster_col: str, feature_cols: List[str]) -> pd.DataFrame:
    """Calculate mean feature values and response rates per cluster.

    Args:
        df: Patient dataset containing cluster labels and features.
        cluster_col: Name of column holding cluster IDs.
        feature_cols: List of feature names to summarize.

    Returns:
        DataFrame of per-cluster summary statistics.
    """
    profile = df.groupby(cluster_col)[feature_cols].mean()
    if "RESPONSE_BINARY" in df.columns:
        profile["Response_Rate"] = df.groupby(cluster_col)["RESPONSE_BINARY"].mean()
    if "PATIENT_ID" in df.columns:
        profile["Patient_Count"] = df.groupby(cluster_col)["PATIENT_ID"].count()
    return profile


def _claim_cluster_by_rule(
    remaining: set,
    cluster_profiles: pd.DataFrame,
    col: str,
    method: str = "idxmax",
) -> Optional[int]:
    """Return the cluster ID that wins the specified feature ranking rule.

    Selects among the *remaining* (unclaimed) cluster IDs the one with the
    highest (``idxmax``) or lowest (``idxmin``) value of ``col``. Returns
    ``None`` if no valid selection can be made (column absent, remaining set
    empty, or all values <= 0 when using ``idxmax``).

    Args:
        remaining: Set of unclaimed integer cluster IDs.
        cluster_profiles: Summary DataFrame of cluster mean feature values.
        col: Feature column name to rank clusters on.
        method: Ranking method (``'idxmax'`` or ``'idxmin'``).

    Returns:
        Integer cluster ID of the winning cluster, or ``None``.
    """
    if not remaining or col not in cluster_profiles.columns:
        return None
    rem_list = list(remaining)
    if method == "idxmax":
        if cluster_profiles.loc[rem_list, col].max() <= 0:
            return None
        return int(cluster_profiles.loc[rem_list, col].idxmax())
    if method == "idxmin":
        return int(cluster_profiles.loc[rem_list, col].idxmin())
    return None


def assign_phenotype_labels(cluster_profiles: pd.DataFrame) -> Dict[int, str]:
    """Assign biological phenotype labels to clusters based on empirical profiles.

    Labels are assigned using a sequential rank-based claim-and-eliminate approach
    on actual computed cluster means, so the assignment is robust to K-Means/GMM
    producing different integer cluster IDs across runs or datasets. Each cluster
    is claimed at most once, in this priority order:

    1. **Mutant-Driven (NF1 Loss)**: highest ``mut_NF1`` mutation rate. Skipped
       entirely if ``mut_NF1`` is absent or all-zero.
    2. **Immune Cold**: among remaining clusters, the lowest TIS (Tumour
       Inflammation Score) — the immune desert, low activity across all lineages.
    3. **Immunosuppressive M2-High**: among remaining clusters, the lowest
       ``CD8_T_cells``, matching the "depleted T-cells" component of this
       phenotype. This was deliberately chosen over ranking by ``M1_M2_Ratio``:
       recomputing cluster means directly from real patient data showed TIS and
       M1_M2_Ratio can disagree about which cluster is more "Hot" vs. "M2-High"
       (a cluster can have both the highest TIS *and* the most M2-skewed ratio
       at once), whereas CD8 T-cell depletion cleanly and unambiguously
       separates the two remaining clusters in that same data. Re-verify this
       choice if the clustering/feature set changes materially.
    4. **Immune Hot**: the sole remaining cluster — highest TIS/CYT/CD8 by
       construction, since Cold and M2-High were already claimed.

    Args:
        cluster_profiles: DataFrame indexed by cluster ID with columns including
            at minimum ``TIS`` and ``CD8_T_cells``, and optionally ``mut_NF1``.

    Returns:
        Dictionary mapping integer cluster ID to phenotype label string.
    """
    labels: Dict[int, str] = {}
    remaining = set(cluster_profiles.index.tolist())

    # 1. Mutant-Driven (NF1 Loss)
    if "mut_NF1" in cluster_profiles.columns and cluster_profiles["mut_NF1"].max() > 0:
        cid = _claim_cluster_by_rule(remaining, cluster_profiles, "mut_NF1", "idxmax")
        if cid is not None:
            labels[cid] = "Mutant-Driven"
            remaining.discard(cid)

    # 2. Immune Cold (lowest TIS among remaining)
    cid = _claim_cluster_by_rule(remaining, cluster_profiles, "TIS", "idxmin")
    if cid is not None:
        labels[cid] = "Immune Cold"
        remaining.discard(cid)

    # 3. Immunosuppressive M2-High (lowest CD8_T_cells among remaining —
    #    "depleted T-cells"; see docstring for why this beats M1_M2_Ratio here)
    cid = _claim_cluster_by_rule(remaining, cluster_profiles, "CD8_T_cells", "idxmin")
    if cid is not None:
        labels[cid] = "Immunosuppressive M2-High"
        remaining.discard(cid)

    # 4. Immune Hot (sole remainder)
    for cid in remaining:
        labels[cid] = "Immune Hot"

    return labels


def plot_baseline_signature_boxplots(df: pd.DataFrame, save_path: Path) -> None:
    """Generate 300 DPI publication 2x3 facetted violin plots with inner quartiles.

    Eliminates outlier y-axis distortion by providing independent, auto-scaled y-axes
    per biomarker with cut=0 clipping.
    """
    sig_cols = [c for c in ["TIS", "CYT", "M1_M2_Ratio", "CD8_T_cells", "M2_Macrophages", "CAFs"] if c in df.columns]
    if not sig_cols or "RESPONSE_BINARY" not in df.columns:
        return

    valid_df = df.dropna(subset=sig_cols + ["RESPONSE_BINARY"]).copy()
    valid_df["Response"] = valid_df["RESPONSE_BINARY"].map({1: "Responder (CR/PR)", 0: "Non-Responder (PD)"})

    palette = {
        "Responder (CR/PR)": RESPONSE_PALETTE.get("Responder", "#009E73"),
        "Non-Responder (PD)": RESPONSE_PALETTE.get("Non-responder", "#D55E00"),
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=300)
    axes = axes.flatten()

    for idx, feature in enumerate(sig_cols):
        ax = axes[idx]
        sns.violinplot(
            data=valid_df,
            x="Response",
            y=feature,
            hue="Response",
            palette=palette,
            ax=ax,
            inner="quartile",
            cut=0,  # Clip violins at min/max data range to prevent ugly outlier tails
            linewidth=1.2,
            legend=False,
        )
        ax.set_title(f"`{feature}` Distribution", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Score / Ratio", fontsize=10, fontweight="bold")

    n_patients = len(valid_df)
    fig.suptitle(f"Baseline Immune & Microenvironmental Biomarker Distributions (N={n_patients})", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved facetted biomarker violin plot figure to {save_path}")


def plot_radar_chart(profiles: pd.DataFrame, save_path: Path) -> None:
    """Generate spider/radar chart comparing cluster feature profiles."""
    pass


def plot_cluster_heatmap(df: pd.DataFrame, cluster_col: str, feature_cols: List[str], save_path: Path) -> None:
    """Generate annotated heatmap of patients x features sorted by cluster."""
    pass
