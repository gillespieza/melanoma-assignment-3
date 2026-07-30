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


def assign_phenotype_labels(cluster_profiles: pd.DataFrame) -> Dict[int, str]:
    """Assign biological phenotype labels to clusters based on their empirical profiles.

    Labels are assigned using a rank-based approach on the actual computed cluster
    means, making the assignment robust to K-Means producing different integer cluster
    IDs across runs, random seeds, or different datasets. The four rules are applied
    sequentially, with each cluster claimed at most once:

    1. **Mutant-Driven (NF1 Loss)**: Cluster with the highest ``mut_NF1`` rate.
    2. **Immune Hot**: Among the remaining clusters, the one with the highest TIS.
    3. **M2 Immunosuppressive**: Among the remaining clusters, the one with the
       *lowest M1/M2 macrophage ratio*. TIS alone cannot separate M2-suppressed from
       immune desert — both are cold — but the macrophage polarisation balance
       (M2-skewed vs. uniformly depleted) is the discriminating biological signal.
       Falls back to lowest ``CD8_T_cells`` if ``M1_M2_Ratio`` is unavailable.
    4. **Immune Cold**: The single remaining cluster (moderate immune desert).

    Args:
        cluster_profiles: DataFrame indexed by cluster ID with columns including
            at minimum ``TIS`` and optionally ``mut_NF1``, ``M1_M2_Ratio``,
            ``CD8_T_cells``.

    Returns:
        Dictionary mapping integer cluster ID to phenotype label string.
    """
    labels: Dict[int, str] = {}
    remaining = set(cluster_profiles.index.tolist())

    # --- Rule 1: Mutant-Driven (NF1 Loss) --------------------------------
    # The cluster with the highest NF1 mutation rate is the mutant-dominant subtype.
    # If mut_NF1 is absent or all-zero, this rule is skipped (no NF1-dominant cluster).
    if "mut_NF1" in cluster_profiles.columns and cluster_profiles["mut_NF1"].max() > 0:
        nf1_id = int(cluster_profiles.loc[list(remaining), "mut_NF1"].idxmax())
        labels[nf1_id] = "Mutant-Driven"
        remaining.discard(nf1_id)

    # --- Rule 2: Immune Hot (highest TIS among remaining) -----------------
    hot_id = int(cluster_profiles.loc[list(remaining), "TIS"].idxmax())
    labels[hot_id] = "Immune Hot"
    remaining.discard(hot_id)

    # --- Rule 3: M2 Immunosuppressive (lowest M1/M2 ratio among remaining) ---
    # TIS alone cannot separate M2-suppressed from immune desert — both are cold.
    # The M1/M2 macrophage balance is the discriminating signal: M2-skewed clusters
    # have active immunosuppression, whereas the true immune cold desert is uniformly low.
    # Falls back to lowest CD8_T_cells (most T-cell excluded) if ratio column is absent.
    if "M1_M2_Ratio" in cluster_profiles.columns:
        m2_id = int(cluster_profiles.loc[list(remaining), "M1_M2_Ratio"].idxmin())
    elif "CD8_T_cells" in cluster_profiles.columns:
        m2_id = int(cluster_profiles.loc[list(remaining), "CD8_T_cells"].idxmin())
    else:
        m2_id = int(cluster_profiles.loc[list(remaining), "TIS"].idxmin())
    labels[m2_id] = "Immunosuppressive M2-High"
    remaining.discard(m2_id)

    # --- Rule 4: Immune Cold (sole remainder — moderate immune desert) ----
    for cid in remaining:
        labels[cid] = "Immune Cold"

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
