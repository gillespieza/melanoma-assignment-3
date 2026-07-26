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
    """Assign biological phenotype labels to clusters based on profile rules.

    Args:
        cluster_profiles: DataFrame of cluster feature profiles.

    Returns:
        Dictionary mapping cluster ID to phenotype label string.
    """
    labels = {}
    for cluster_id, row in cluster_profiles.iterrows():
        tis = row.get("TIS", 0)
        m1_m2 = row.get("M1_M2_Ratio", 0.5)
        m2_score = row.get("M2_Macrophages", row.get("M2_score", 0))

        if tis > cluster_profiles["TIS"].median() and m1_m2 > cluster_profiles["M1_M2_Ratio"].median():
            labels[cluster_id] = "Immune Hot"
        elif tis < cluster_profiles["TIS"].median() and m1_m2 < cluster_profiles["M1_M2_Ratio"].median():
            labels[cluster_id] = "Immune Cold"
        elif m2_score > cluster_profiles["M2_score"].median() if "M2_score" in cluster_profiles else m1_m2 < 0.4:
            labels[cluster_id] = "Immunosuppressive M2-High"
        else:
            labels[cluster_id] = "Mutant-Driven"
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
