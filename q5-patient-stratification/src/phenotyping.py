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

from src.styles import PHENOTYPE_PALETTE, RESPONSE_PALETTE, set_presentation_style
from src.utils.paths import rel_path
from src.utils.plotting import save_fig

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

set_presentation_style()


# ---------------------------------------------------------------------------
# Profile Calculation & Phenotype Label Assignment
# ---------------------------------------------------------------------------

def profile_clusters(df: pd.DataFrame, cluster_col: str, feature_cols: List[str]) -> pd.DataFrame:
    """Calculate mean feature values, patient counts, and response rates per cluster.

    Args:
        df: Patient dataset containing cluster labels and features.
        cluster_col: Name of column holding cluster IDs.
        feature_cols: List of feature names to summarize.

    Returns:
        DataFrame of per-cluster summary statistics.
    """
    valid_features = [c for c in feature_cols if c in df.columns]
    profile = df.groupby(cluster_col)[valid_features].mean()
    
    if "RESPONSE_BINARY" in df.columns:
        profile["Response_Rate"] = df.groupby(cluster_col)["RESPONSE_BINARY"].mean()
    
    counts = df.groupby(cluster_col).size()
    profile["Patient_Count"] = counts
    profile["Cohort_Percentage"] = (counts / len(df)) * 100.0
    return profile


def _claim_cluster_by_rule(
    remaining: set,
    cluster_profiles: pd.DataFrame,
    col: str,
    target_label: str,
    method: str = "idxmax",
    labels: Optional[Dict[int, str]] = None,
) -> None:
    """Helper to claim a cluster based on feature ranking rule."""
    if labels is None or not remaining or col not in cluster_profiles.columns:
        return
    rem_list = list(remaining)
    if method == "idxmax" and cluster_profiles[col].max() > 0:
        cid = int(cluster_profiles.loc[rem_list, col].idxmax())
    elif method == "idxmin":
        cid = int(cluster_profiles.loc[rem_list, col].idxmin())
    else:
        return
    labels[cid] = target_label
    remaining.discard(cid)


def assign_phenotype_labels(cluster_profiles: pd.DataFrame) -> Dict[int, str]:
    """Assign biological phenotype labels to clusters based on empirical profiles.

    Args:
        cluster_profiles: DataFrame indexed by cluster ID with profiles.

    Returns:
        Dictionary mapping integer cluster ID to phenotype label string.
    """
    labels: Dict[int, str] = {}
    remaining = set(cluster_profiles.index.tolist())

    # 1. Mutant-Driven (NF1 Loss)
    if "mut_NF1" in cluster_profiles.columns and cluster_profiles["mut_NF1"].max() > 0:
        _claim_cluster_by_rule(remaining, cluster_profiles, "mut_NF1", "Mutant-Driven", "idxmax", labels)

    # 2. Immune Hot (highest TIS among remaining)
    _claim_cluster_by_rule(remaining, cluster_profiles, "TIS", "Immune Hot", "idxmax", labels)

    # 3. Immune Cold (lowest TIS among remaining)
    _claim_cluster_by_rule(remaining, cluster_profiles, "TIS", "Immune Cold", "idxmin", labels)

    # 4. Immunosuppressive M2-High (sole remainder)
    for cid in remaining:
        labels[cid] = "Immunosuppressive M2-High"

    return labels


# ---------------------------------------------------------------------------
# Visualisation Helpers
# ---------------------------------------------------------------------------

def _plot_biomarker_violin(ax: plt.Axes, valid_df: pd.DataFrame, feature: str, palette: Dict[str, str]) -> None:
    """Helper to render a single biomarker violin plot."""
    sns.violinplot(
        data=valid_df,
        x="Response",
        y=feature,
        hue="Response",
        palette=palette,
        ax=ax,
        inner="quartile",
        cut=0,
        linewidth=1.2,
        legend=False,
    )
    ax.set_title(f"`{feature}` Distribution", fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Score / Ratio", fontsize=10, fontweight="bold")


def plot_baseline_signature_boxplots(df: pd.DataFrame, save_path: Path) -> None:
    """Generate 300 DPI publication 2x3 facetted violin plots with inner quartiles.

    Args:
        df: Patient DataFrame containing biomarker features and response.
        save_path: File path to save the generated plot.
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
    axes_flat = axes.flatten()

    for idx, feature in enumerate(sig_cols):
        _plot_biomarker_violin(axes_flat[idx], valid_df, feature, palette)

    n_patients = len(valid_df)
    fig.suptitle(f"Baseline Immune & Microenvironmental Biomarker Distributions (N={n_patients})", fontsize=14, fontweight="bold", y=0.98)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved facetted biomarker violin plot figure to {rel_path(save_path)}")


def plot_radar_chart(profiles: pd.DataFrame, save_path: Path) -> None:
    """Generate spider/radar chart comparing cluster feature profiles across phenotypes.

    Args:
        profiles: Per-cluster profile DataFrame indexed by Cluster_ID or Phenotype_Label.
        save_path: Output PNG file path.
    """
    numeric_cols = profiles.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["Patient_Count", "Cohort_Percentage", "Response_Rate"]]
    if not numeric_cols:
        return

    num_vars = len(numeric_cols)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)
    
    for idx, (label, row) in enumerate(profiles.iterrows()):
        values = row[numeric_cols].values.flatten().tolist()
        values += values[:1]
        color = PHENOTYPE_PALETTE.get(str(label), f"C{idx}")
        ax.plot(angles, values, linewidth=2, label=str(label), color=color)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(numeric_cols, fontsize=9, fontweight="bold")
    ax.set_title("Phenotype Feature Profile Radar Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), frameon=True, fontsize=9)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved radar chart comparison to {rel_path(save_path)}")


def plot_cluster_heatmap(df: pd.DataFrame, cluster_col: str, feature_cols: List[str], save_path: Path) -> None:
    """Generate annotated heatmap of patients x features sorted by cluster.

    Args:
        df: Patient DataFrame containing cluster column and features.
        cluster_col: Name of cluster column.
        feature_cols: List of features to include in heatmap.
        save_path: Output PNG path.
    """
    valid_cols = [c for c in feature_cols if c in df.columns]
    if not valid_cols or cluster_col not in df.columns:
        return

    df_sorted = df.sort_values(by=cluster_col).copy()
    mat = df_sorted[valid_cols].apply(lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1.0))

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(mat.T, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "Z-Score"})
    ax.set_title(f"Patient Microenvironment Heatmap Stratified by Cluster (N={len(df_sorted)})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Patients (Sorted by Cluster)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Biomarker Features", fontsize=11, fontweight="bold")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved cluster heatmap to {rel_path(save_path)}")

