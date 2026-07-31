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

RESPONSE_BINARY_COL: str = "RESPONSE_BINARY"

RESPONSE_DISPLAY_LABELS: Dict[int, str] = {
    1: "Responder (CR/PR)",
    0: "Non-Responder (PD)",
}

BASELINE_VIOLIN_FEATURES: List[str] = [
    "TIS",
    "CYT",
    "M1_M2_Ratio",
    "CD8_T_cells",
    "M2_Macrophages",
    "CAFs",
]


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
    
    if RESPONSE_BINARY_COL in df.columns:
        profile["Response_Rate"] = df.groupby(cluster_col)[RESPONSE_BINARY_COL].mean()
    
    counts = df.groupby(cluster_col).size()
    profile["Patient_Count"] = counts
    profile["Cohort_Percentage"] = (counts / len(df)) * 100.0
    return profile


def _claim_cluster_by_rule(
    remaining: set,
    cluster_profiles: pd.DataFrame,
    col: str,
    method: str = "idxmax",
) -> Optional[int]:
    """Return the cluster ID that wins the specified feature ranking rule.

    Selects among the *remaining* (unclaimed) cluster IDs the one with the
    highest (``idxmax``) or lowest (``idxmin``) value of ``col``.  Returns
    ``None`` if no valid selection can be made (column absent, remaining set
    empty, or all values ≤ 0 when using ``idxmax``).

    The return value is the integer cluster ID, not a label — the caller
    is responsible for assigning the phenotype name and discarding the ID
    from *remaining*.  This avoids the mutable-output-parameter anti-pattern
    and makes the side-effect explicit at every call site.

    Args:
        remaining: Set of unclaimed integer cluster IDs.
        cluster_profiles: Summary DataFrame of cluster mean feature values.
        col: Feature column name to rank clusters on.
        method: Ranking method (``'idxmax'`` or ``'idxmin'``). Defaults to
            ``'idxmax'``.

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

    Labels are assigned using a sequential rank-based approach on actual computed
    cluster means, making the assignment robust to K-Means or GMM producing different
    integer cluster IDs across runs, random seeds, or different datasets. The four
    rules are applied in priority order, with each cluster claimed at most once:

    1. **Mutant-Driven (NF1 Loss)**: Cluster with the highest ``mut_NF1`` mutation
       rate. Skipped entirely if ``mut_NF1`` is absent or all-zero.
    2. **Immune Hot**: Among the remaining clusters, the one with the highest TIS
       (Tumour Inflammation Score). These patients have high CD8+ infiltration and
       active IFN-gamma signalling.
    3. **Immune Cold**: Among the remaining clusters, the one with the lowest TIS.
       These are immune deserts — low infiltration and low immune activity across all
       lineages.
    4. **Immunosuppressive M2-High**: The sole remaining cluster. TIS alone cannot
       separate M2-suppressed from immune desert — both are cold — but the macrophage
       polarisation balance (M2-skewed vs. uniformly depleted) is the discriminating
       biological signal at this step.

    Args:
        cluster_profiles: DataFrame indexed by cluster ID with columns including
            at minimum ``TIS`` and optionally ``mut_NF1``, ``M1_M2_Ratio``,
            ``CD8_T_cells``.

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

    # 2. Immune Hot (highest TIS among remaining)
    cid = _claim_cluster_by_rule(remaining, cluster_profiles, "TIS", "idxmax")
    if cid is not None:
        labels[cid] = "Immune Hot"
        remaining.discard(cid)

    # 3. Immune Cold (lowest TIS among remaining)
    cid = _claim_cluster_by_rule(remaining, cluster_profiles, "TIS", "idxmin")
    if cid is not None:
        labels[cid] = "Immune Cold"
        remaining.discard(cid)

    # 4. Immunosuppressive M2-High (sole remainder)
    for cid in remaining:
        labels[cid] = "Immunosuppressive M2-High"

    return labels


# ---------------------------------------------------------------------------
# Visualisation Helpers
# ---------------------------------------------------------------------------

def _plot_biomarker_violin(
    ax: plt.Axes, valid_df: pd.DataFrame, feature: str, palette: Dict[str, str]
) -> None:
    """Render a single biomarker violin plot with inner quartile lines on the given Axes.

    Args:
        ax: Matplotlib Axes to draw on.
        valid_df: Patient DataFrame with a 'Response' string column and the feature column.
        feature: Column name of the biomarker to plot on the y-axis.
        palette: Mapping of response label string to hex colour string.
    """
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


def _zscore_series(s: pd.Series) -> pd.Series:
    """Compute Z-score standardized Series with zero-variance protection."""
    std_val = s.std()
    return (s - s.mean()) / (std_val if std_val > 0 else 1.0)


def plot_baseline_signature_boxplots(df: pd.DataFrame, save_path: Path) -> None:
    """Generate 300 DPI publication 2x3 facetted violin plots with inner quartiles.

    Args:
        df: Patient DataFrame containing biomarker features and response.
        save_path: File path to save the generated plot.
    """
    sig_cols = [c for c in BASELINE_VIOLIN_FEATURES if c in df.columns]
    if not sig_cols or RESPONSE_BINARY_COL not in df.columns:
        return

    valid_df = df.dropna(subset=sig_cols + [RESPONSE_BINARY_COL]).copy()
    valid_df["Response"] = valid_df[RESPONSE_BINARY_COL].map(RESPONSE_DISPLAY_LABELS)

    palette = {
        RESPONSE_DISPLAY_LABELS[1]: RESPONSE_PALETTE.get("Responder", "#009E73"),
        RESPONSE_DISPLAY_LABELS[0]: RESPONSE_PALETTE.get("Non-responder", "#D55E00"),
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=300)
    axes_flat = axes.flatten()

    for idx, feature in enumerate(sig_cols):
        _plot_biomarker_violin(axes_flat[idx], valid_df, feature, palette)

    n_patients = len(valid_df)
    fig.suptitle(
        f"Baseline Immune & Microenvironmental Biomarker Distributions (N={n_patients})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
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
    
    # Build a fallback colour list for phenotype labels absent from PHENOTYPE_PALETTE.
    # Use a curated seaborn palette rather than matplotlib's f"C{idx}" defaults.
    fallback_colours = sns.color_palette("tab10", n_colors=len(profiles))

    for idx, (label, row) in enumerate(profiles.iterrows()):
        values = row[numeric_cols].values.flatten().tolist()
        values += values[:1]
        color = PHENOTYPE_PALETTE.get(str(label), fallback_colours[idx])
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
    mat = df_sorted[valid_cols].apply(_zscore_series)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(mat.T, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "Z-Score"})
    ax.set_title(f"Patient Microenvironment Heatmap Stratified by Cluster (N={len(df_sorted)})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Patients (Sorted by Cluster)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Biomarker Features", fontsize=11, fontweight="bold")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved cluster heatmap to {rel_path(save_path)}")
