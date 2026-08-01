#!/usr/bin/env python3
"""Script 03: Unsupervised patient stratification pipeline (Two-Stage GMM + NF1 Split).

Implements a two-stage biological phenotyping strategy:

  Stage 1 — GMM (K=3, full covariance) on 6 continuous immune/stromal features:
      TIS, CYT, CD8_T_cells, M1_Macrophages, M2_Macrophages, CAFs.
    Binary mutation indicators (mut_BRAF, mut_NRAS, mut_NF1) are intentionally
    excluded from the GMM feature space. Including them caused near-zero
    within-cluster variance on the mutation axis, collapsing all posterior
    probabilities to degenerate 0/1 hard assignments and eliminating the
    probabilistic uncertainty that justifies choosing GMM over K-Means.
    Stage 1 produces genuine continuous posterior probabilities (P_Stage1_*)
    reflecting real immune microenvironment uncertainty at cluster boundaries.

  Stage 2 — Deterministic NF1 Split:
    The Stage 1 cluster with the highest NF1 mutation rate is identified.
    Within that cluster, patients with mut_NF1=1 are assigned the
    'Mutant-Driven' phenotype label. NF1-negative patients within that cluster
    retain their Stage 1 immune label. This mirrors how biological phenotyping
    works in the literature: immune microenvironment archetypes are defined by
    transcriptomics; driver mutation subtypes are a secondary stratification.

Outputs:
  - data/processed/q5/patient_clusters.csv              — final 4-phenotype assignments
  - data/processed/q5/gmm_posterior_probabilities.csv   — Stage 1 continuous posteriors
  - data/processed/q5/gmm_model.pkl                    — fitted Stage 1 GMM (K=3)
  - data/processed/q5/clustering_feature_cols.json      — Stage 1 feature list
  - data/processed/q5/mahalanobis_spectral_metrics.csv  — GMM vs Mahalanobis vs Spectral quality
  - plots/clustering/pca_clusters.png
  - plots/clustering/tsne_clusters.png
  - plots/clustering/umap_clusters.png
  - plots/clustering/spatial_microenvironment_violins.png
  - plots/clustering/tmb_by_phenotype_comparison.png
  - plots/clustering/cohort_size_clustering_comparison.png
"""

import contextlib
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
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

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from clustering import (
    DEFAULT_N_COMPONENTS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SPECTRAL_NEIGHBORS,
    plot_2d_cluster_projection,
    plot_spatial_microenvironment_violins,
    prepare_clustering_features,
    run_gmm,
    run_spectral_manifold,
    transform_mahalanobis_space,
)
from phenotyping import assign_phenotype_labels, profile_clusters
from q5_constants import (
    GMM_CONTINUOUS_FEATURES,
    PHENOTYPE_LABEL_SUFFIX,
    PHENOTYPE_PROFILE_FEATURES,
)
from src.styles import OKABE_ITO, get_phenotype_color, set_presentation_style
from src.utils.io import safe_save_csv
from src.utils.logging import TeeStream
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, rel_path
from src.utils.plotting import save_fig

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "03_cluster_patients.log"

INPUT_FILE_FULL = PROCESSED_DIR / "q5" / "feature_matrix_full.csv"
INPUT_FILE_ICI = PROCESSED_DIR / "q5" / "feature_matrix.csv"
OUTPUT_DIR = PROCESSED_DIR / "q5"

set_presentation_style()


def _compute_clustering_metrics(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    gmm: Optional[GaussianMixture] = None,
) -> Dict[str, float]:
    """Compute internal clustering quality metrics for a given label assignment and GMM model."""
    valid = labels != -1
    metrics: Dict[str, float] = {}
    if len(np.unique(labels[valid])) >= 2:
        metrics["Silhouette"] = round(float(silhouette_score(X_scaled[valid], labels[valid])), 4)
        metrics["Calinski_Harabasz"] = round(float(calinski_harabasz_score(X_scaled[valid], labels[valid])), 1)
        metrics["Davies_Bouldin"] = round(float(davies_bouldin_score(X_scaled[valid], labels[valid])), 4)
    else:
        metrics = {"Silhouette": float("nan"), "Calinski_Harabasz": float("nan"), "Davies_Bouldin": float("nan")}

    if gmm is not None:
        metrics["Log_Likelihood"] = round(float(gmm.score(X_scaled)), 4)
        metrics["AIC"] = round(float(gmm.aic(X_scaled)), 1)
        metrics["BIC"] = round(float(gmm.bic(X_scaled)), 1)

    return metrics


def _plot_cohort_size_comparison(
    metrics_ici: Dict[str, float],
    metrics_full: Dict[str, float],
    n_ici: int,
    n_full: int,
    out_path: Path,
) -> None:
    """Generate a side-by-side bar chart comparing clustering quality across cohort sizes."""
    metric_labels = ["Silhouette\n(higher better)", "Calinski-Harabasz\n(higher better)", "Davies-Bouldin\n(lower better)"]
    metric_keys = ["Silhouette", "Calinski_Harabasz", "Davies_Bouldin"]

    vals_ici = [metrics_ici.get(k, float("nan")) for k in metric_keys]
    vals_full = [metrics_full.get(k, float("nan")) for k in metric_keys]

    x = np.arange(len(metric_labels))
    width = 0.35
    color_ici = OKABE_ITO[0]
    color_full = OKABE_ITO[4]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_ici = ax.bar(x - width / 2, vals_ici, width, label=f"ICI-only (N={n_ici})", color=color_ici, edgecolor="black", linewidth=0.8)
    bars_full = ax.bar(x + width / 2, vals_full, width, label=f"Full cohort (N={n_full})", color=color_full, edgecolor="black", linewidth=0.8)

    for bar in list(bars_ici) + list(bars_full):
        h = bar.get_height()
        if not np.isnan(h):
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.set_title(
        f"GMM Clustering Quality: ICI Cohort (N={n_ici}) vs Full Cohort (N={n_full})",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Internal Clustering Metric", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=10)
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_path, dpi=300)
    plt.close(fig)
    print(f"Saved cohort-size clustering comparison to {rel_path(out_path)}")


def _fit_and_save_gmm_model(
    X_scaled: np.ndarray,
    feature_cols: List[str],
) -> Tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """Fit Stage 1 GMM (K=3, full covariance) on continuous immune features and persist model artefacts.

    K=3 covers the three continuous immune archetypes: Immune Hot, Immune Cold, and
    Immunosuppressive M2-High. The Mutant-Driven (NF1 Loss) phenotype is added in
    Stage 2 via deterministic NF1 split, not by adding a 4th GMM component.

    Args:
        X_scaled: Pre-scaled feature matrix (N patients × M features) from
            ``prepare_clustering_features``.
        feature_cols: List of column names used for clustering, for persistence to JSON.

    Returns:
        Tuple of (gmm, labels, probs) — the fitted model, hard assignments, and
        per-component posterior probability matrix.
    """
    gmm, labels, probs = run_gmm(
        X_scaled,
        n_components=DEFAULT_N_COMPONENTS,
        covariance_type="full",
        random_state=DEFAULT_RANDOM_STATE,
    )

    gmm_model_path = OUTPUT_DIR / "gmm_model.pkl"
    features_path = OUTPUT_DIR / "clustering_feature_cols.json"

    joblib.dump(gmm, gmm_model_path)
    with open(features_path, "w", encoding="utf-8") as fp:
        json.dump(feature_cols, fp)

    print(f"Saved Stage 1 GMM model (K={DEFAULT_N_COMPONENTS}) to {rel_path(gmm_model_path)}")
    print(f"Saved Stage 1 clustering feature list to {rel_path(features_path)}")
    return gmm, labels, probs


def _apply_nf1_deterministic_split(
    df_clean: pd.DataFrame,
    stage1_labels: np.ndarray,
    stage1_short_labels: Dict[int, str],
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Apply Stage 2 deterministic NF1 split to produce the Mutant-Driven phenotype.

    Identifies the Stage 1 cluster with the highest NF1 mutation rate, then
    reassigns NF1-positive patients within that cluster to a new Cluster ID (3)
    labelled 'Mutant-Driven'. NF1-negative patients within that cluster retain
    their Stage 1 immune label. Patients in other clusters are unaffected.

    Args:
        df_clean: Full patient DataFrame including mut_NF1 column.
        stage1_labels: Hard cluster assignments from Stage 1 GMM (values 0, 1, 2).
        stage1_short_labels: Mapping of Stage 1 cluster ID -> immune phenotype short label.

    Returns:
        Tuple of (final_labels, final_short_labels) where final_labels contains
        values 0–3 and final_short_labels maps 0–3 to biological phenotype names.
    """
    if "mut_NF1" not in df_clean.columns:
        print("WARNING: mut_NF1 column not found — skipping Stage 2 NF1 split. Only 3 phenotypes will be exported.")
        return stage1_labels.copy(), dict(stage1_short_labels)

    # Find the Stage 1 cluster with the highest NF1 mutation rate
    nf1_rates = {
        cid: df_clean.loc[stage1_labels == cid, "mut_NF1"].mean()
        for cid in range(DEFAULT_N_COMPONENTS)
    }
    nf1_base_cluster = max(nf1_rates, key=nf1_rates.__getitem__)
    nf1_base_rate = nf1_rates[nf1_base_cluster]
    print(
        f"Stage 2 NF1 split: base cluster = Cluster {nf1_base_cluster} "
        f"('{stage1_short_labels[nf1_base_cluster]}', NF1 rate = {nf1_base_rate:.1%})"
    )

    nf1_positive = df_clean["mut_NF1"].fillna(0).astype(bool).values
    in_base_cluster = (stage1_labels == nf1_base_cluster)

    # Patients in nf1_base_cluster with mut_NF1=1 -> Cluster 3 (Mutant-Driven)
    final_labels = stage1_labels.copy()
    mutant_driven_mask = in_base_cluster & nf1_positive
    final_labels[mutant_driven_mask] = 3

    n_split = int(mutant_driven_mask.sum())
    n_retained = int((in_base_cluster & ~nf1_positive).sum())
    print(
        f"  Reassigned {n_split} NF1+ patients -> Cluster 3 (Mutant-Driven)"
    )
    print(
        f"  Retained {n_retained} NF1- patients in Cluster {nf1_base_cluster} "
        f"('{stage1_short_labels[nf1_base_cluster]}')"
    )

    final_short_labels: Dict[int, str] = dict(stage1_short_labels)
    final_short_labels[3] = "Mutant-Driven"
    return final_labels, final_short_labels


def _assign_labels(
    df_clean: pd.DataFrame,
    stage1_labels: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, str], Dict[int, str]]:
    """Derive biological phenotype labels from Stage 1 cluster profiles, then apply Stage 2 NF1 split.

    Mutation columns (mut_NF1, mut_BRAF, mut_NRAS) are explicitly stripped from the
    cluster profiles passed to assign_phenotype_labels() so that Stage 1 produces only
    the 3 continuous immune archetypes (Immune Hot, Immune Cold, Immunosuppressive M2-High).
    The Mutant-Driven phenotype emerges exclusively from Stage 2 via the deterministic
    NF1-positive split — never from the Stage 1 GMM label assignment step.

    Returns:
        Tuple of (final_labels, phenotype_names, stage1_short_labels) where:
          - final_labels: integer cluster IDs 0–3 per patient.
          - phenotype_names: mapping from cluster ID to full descriptive label.
          - stage1_short_labels: mapping from Stage 1 cluster ID to short immune label,
            used by ``_export_cluster_outputs`` to derive named probability columns without
            relying on brittle hardcoded GMM index assumptions.
    """
    profile_feature_cols = [c for c in PHENOTYPE_PROFILE_FEATURES if c in df_clean.columns]
    cluster_profiles = profile_clusters(df_clean, "Stage1_Cluster_ID", profile_feature_cols)

    # Strip mutation columns so Stage 1 labelling is purely immune-phenotype driven.
    # Without mut_NF1, assign_phenotype_labels() will assign Hot / Cold / M2-High only.
    mutation_cols = [c for c in ["mut_NF1", "mut_BRAF", "mut_NRAS"] if c in cluster_profiles.columns]
    cluster_profiles_immune = cluster_profiles.drop(columns=mutation_cols)

    stage1_short_labels: Dict[int, str] = assign_phenotype_labels(cluster_profiles_immune)

    print("Stage 1 immune phenotype label assignment (mutation columns excluded):")
    for cid, short in sorted(stage1_short_labels.items()):
        profile_tis = cluster_profiles.loc[cid, "TIS"] if "TIS" in cluster_profiles.columns else float("nan")
        profile_nf1 = cluster_profiles.loc[cid, "mut_NF1"] if "mut_NF1" in cluster_profiles.columns else float("nan")
        print(f"  Cluster {cid}: TIS={profile_tis:+.3f}, NF1 rate={profile_nf1:.1%}  ->  {short}")

    final_labels, final_short_labels = _apply_nf1_deterministic_split(
        df_clean, stage1_labels, stage1_short_labels
    )

    phenotype_names: Dict[int, str] = {
        cid: f"{short} {PHENOTYPE_LABEL_SUFFIX.get(short, '')}".strip()
        for cid, short in final_short_labels.items()
    }
    print("Final phenotype assignments (Stage 1 + Stage 2):")
    for cid, name in sorted(phenotype_names.items()):
        cnt = int(np.sum(final_labels == cid))
        print(f"  Cluster {cid} (N={cnt}): {name}")

    return final_labels, phenotype_names, stage1_short_labels

def _export_cluster_outputs(
    df_clean: pd.DataFrame,
    stage1_probs: np.ndarray,
    phenotype_names: Dict[int, str],
    stage1_short_labels: Dict[int, str],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Export Stage 1 continuous posterior probabilities and named phenotype probability columns to disk.

    Derives the mapping from Stage 1 GMM component indices to named phenotype columns
    at runtime from ``stage1_short_labels``, avoiding brittle hardcoded index assumptions.
    GMM component ordering is non-deterministic across initialisations — the only safe
    approach is to resolve the index via the empirical cluster profiles computed in
    ``_assign_labels``.

    Computes stable named columns (P_Immune_Hot, P_Immune_Cold, P_Immunosuppressive_M2_High,
    P_Mutant_Driven) derived from Stage 1 posteriors and Stage 2 NF1 split, ensuring full
    compatibility with downstream predictive models (Phase 5), clinical utility (Phase 6),
    and treatability scoring (Phase 7).
    """
    # Raw Stage 1 posterior columns (K=3 continuous immune features)
    for k in range(stage1_probs.shape[1]):
        df_clean[f"P_Stage1_Cluster_{k}"] = stage1_probs[:, k]

    # Derive Stage 1 index -> named probability column at runtime from the
    # empirical label assignment so we are robust to GMM component reordering.
    short_to_named = {
        "Immune Hot": "P_Immune_Hot",
        "Immune Cold": "P_Immune_Cold",
        "Immunosuppressive M2-High": "P_Immunosuppressive_M2_High",
    }
    is_mutant_driven = (df_clean["Cluster_ID"] == 3).values

    for stage1_idx, short in stage1_short_labels.items():
        col = short_to_named.get(short)
        if col is None:
            continue
        if short == "Immunosuppressive M2-High":
            # NF1+ patients split out of this cluster; zero their M2-High probability
            # and assign it to P_Mutant_Driven instead.
            df_clean[col] = np.where(is_mutant_driven, 0.0, stage1_probs[:, stage1_idx])
            df_clean["P_Mutant_Driven"] = np.where(is_mutant_driven, stage1_probs[:, stage1_idx], 0.0)
        else:
            df_clean[col] = stage1_probs[:, stage1_idx]

    # Guarantee all four named columns exist even if a label is missing
    for col in ["P_Immune_Hot", "P_Immune_Cold", "P_Immunosuppressive_M2_High", "P_Mutant_Driven"]:
        if col not in df_clean.columns:
            df_clean[col] = 0.0

    named_prob_cols = ["P_Immune_Hot", "P_Immune_Cold", "P_Immunosuppressive_M2_High", "P_Mutant_Driven"]
    stage1_prob_cols = [f"P_Stage1_Cluster_{k}" for k in range(stage1_probs.shape[1])]
    all_prob_cols = named_prob_cols + stage1_prob_cols

    id_cols = [c for c in ["PATIENT_ID", "sample_id", "patient_id"] if c in df_clean.columns]
    prob_df_cols = id_cols + ["Stage1_Cluster_ID", "Cluster_ID", "Phenotype_Label"] + all_prob_cols
    df_probs_export = df_clean[[c for c in prob_df_cols if c in df_clean.columns]].copy()

    out_probs_file = output_dir / "gmm_posterior_probabilities.csv"
    safe_save_csv(df_probs_export, out_probs_file)
    print(f"Saved Stage 1 GMM posterior probabilities to {rel_path(out_probs_file)}")

    out_clusters = output_dir / "patient_clusters.csv"
    safe_save_csv(df_clean, out_clusters)
    return out_probs_file, out_clusters


def _generate_phase3_plots(
    df_clean: pd.DataFrame,
    labels: np.ndarray,
    X_scaled: np.ndarray,
    phenotype_names: Dict[int, str],
    subproject_root: Path,
) -> None:
    """Generate 2D PCA/t-SNE/UMAP projection maps, spatial violins, and TMB comparison."""
    clustering_plot_dir = subproject_root / "plots" / "clustering"
    clustering_plot_dir.mkdir(parents=True, exist_ok=True)

    pca_plot_file  = clustering_plot_dir / "pca_clusters.png"
    tsne_plot_file = clustering_plot_dir / "tsne_clusters.png"
    umap_plot_file = clustering_plot_dir / "umap_clusters.png"
    plot_2d_cluster_projection(df_clean, labels, X_scaled, pca_plot_file, phenotype_names, method="pca")
    plot_2d_cluster_projection(df_clean, labels, X_scaled, tsne_plot_file, phenotype_names, method="tsne")
    plot_2d_cluster_projection(df_clean, labels, X_scaled, umap_plot_file, phenotype_names, method="umap")

    spatial_violin_file = clustering_plot_dir / "spatial_microenvironment_violins.png"
    plot_spatial_microenvironment_violins(df_clean, spatial_violin_file)

    tmb_plot_file = clustering_plot_dir / "tmb_by_phenotype_comparison.png"
    _plot_tmb_by_phenotype(df_clean, phenotype_names, tmb_plot_file)


def _plot_tmb_by_phenotype(
    df: pd.DataFrame,
    phenotype_names: Dict[int, str],
    out_path: Path,
) -> None:
    """Generate a violin + strip plot of TMB (log10) stratified by biological phenotype.

    Visualises the distribution of tumour mutational burden across the four
    TME phenotypes.  The Mutant-Driven cluster is expected to show the highest
    median TMB due to `NF1` loss-of-function and associated RAS hyperactivation
    increasing mutational load.

    Args:
        df: Patient DataFrame with ``Cluster_ID``, ``Phenotype_Label``, and
            ``TMB_NONSYNONYMOUS`` columns.
        phenotype_names: Mapping from integer Cluster_ID to phenotype label string.
        out_path: Destination path for the saved PNG.
    """
    if "TMB_NONSYNONYMOUS" not in df.columns:
        print("  WARNING: TMB_NONSYNONYMOUS not in dataset — skipping TMB phenotype plot.")
        return

    df_plot = df[["Cluster_ID", "Phenotype_Label", "TMB_NONSYNONYMOUS"]].copy()
    df_plot = df_plot.dropna(subset=["TMB_NONSYNONYMOUS"])
    df_plot["TMB_log10"] = np.log10(df_plot["TMB_NONSYNONYMOUS"].clip(lower=0.01))

    # Ordered by phenotype ID so panel order is consistent across runs
    ordered_labels = [phenotype_names[cid] for cid in sorted(phenotype_names.keys())]
    palette = {name: get_phenotype_color(name) for name in ordered_labels}

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.violinplot(
        data=df_plot,
        x="Phenotype_Label",
        y="TMB_log10",
        hue="Phenotype_Label",
        order=ordered_labels,
        palette=palette,
        inner=None,
        cut=0,
        linewidth=0.8,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=df_plot,
        x="Phenotype_Label",
        y="TMB_log10",
        hue="Phenotype_Label",
        order=ordered_labels,
        palette=palette,
        size=3,
        alpha=0.45,
        jitter=True,
        legend=False,
        ax=ax,
    )

    # Annotate each violin with median and N
    for i, label in enumerate(ordered_labels):
        subset = df_plot.loc[df_plot["Phenotype_Label"] == label, "TMB_log10"]
        if subset.empty:
            continue
        median_val = subset.median()
        n = len(subset)
        ax.text(
            i, ax.get_ylim()[1] * 0.97,
            f"N={n}\nMed={10**median_val:.0f}",
            ha="center", va="top", fontsize=9,
            color="#37474F",
        )

    ax.set_title(
        "Tumour Mutational Burden (TMB) Distribution by Biological Phenotype",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Biological Phenotype Subgroup", fontsize=11, fontweight="bold")
    ax.set_ylabel("TMB (log10 non-synonymous mutations/Mb)", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(ordered_labels)))
    ax.set_xticklabels(ordered_labels, rotation=15, ha="right", fontsize=10)
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_path, dpi=300)
    plt.close(fig)
    print(f"  Saved TMB by phenotype plot to {rel_path(out_path)}")


def _evaluate_mahalanobis_spectral_comparisons(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    gmm: GaussianMixture,
    output_dir: Path,
) -> Dict[str, float]:
    """Evaluate Mahalanobis GMM and Spectral Manifold clustering as benchmark comparisons.

    Fits two alternative clustering models on the same feature space and computes
    internal quality metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin, AIC, BIC)
    for all three methods. Results are saved to CSV for report use.

    Args:
        X_scaled: Standard-scaled feature matrix (N × M) from ``prepare_clustering_features``.
        labels: Hard cluster assignments from the primary Stage 1 GMM.
        gmm: Fitted Stage 1 GMM model (used to score standard-space metrics).
        output_dir: Directory to write ``mahalanobis_spectral_metrics.csv``.

    Returns:
        Metric dict for the primary GMM (standard scaled), for use in cohort comparison.
    """
    X_mahalanobis, _ = transform_mahalanobis_space(X_scaled)
    gmm_mah, labels_mah, _ = run_gmm(
        X_mahalanobis,
        n_components=DEFAULT_N_COMPONENTS,
        covariance_type="full",
        random_state=DEFAULT_RANDOM_STATE,
    )
    _, spectral_labels = run_spectral_manifold(
        X_scaled,
        n_clusters=DEFAULT_N_COMPONENTS,
        n_neighbors=DEFAULT_SPECTRAL_NEIGHBORS,
        random_state=DEFAULT_RANDOM_STATE,
    )

    gmm_metrics = _compute_clustering_metrics(X_scaled, labels, gmm)
    mah_metrics = _compute_clustering_metrics(X_mahalanobis, labels_mah, gmm_mah)
    spectral_metrics = _compute_clustering_metrics(X_scaled, spectral_labels)

    comp_df = pd.DataFrame([
        {"Method": "GMM (Standard Scaled + Spatial)", **gmm_metrics},
        {"Method": "GMM (Mahalanobis Transformed)", **mah_metrics},
        {"Method": "Spectral Manifold (Graph Laplacian)", **spectral_metrics},
    ])
    out_metrics_file = output_dir / "mahalanobis_spectral_metrics.csv"
    safe_save_csv(comp_df, out_metrics_file)
    print(f"Saved Mahalanobis & Spectral metrics comparison to {rel_path(out_metrics_file)}")
    return gmm_metrics


def _compare_cohort_quality(
    gmm: GaussianMixture,
    feature_cols: List[str],
    metrics_full: Dict[str, float],
    len_full: int,
    subproject_root: Path,
) -> None:
    """Evaluate clustering quality comparison between ICI-only and full cohort."""
    if not INPUT_FILE_ICI.exists():
        print(f"ICI matrix not found at {rel_path(INPUT_FILE_ICI)} — skipping comparison figure.")
        return

    df_ici = pd.read_csv(INPUT_FILE_ICI)
    ici_cols = [c for c in feature_cols if c in df_ici.columns]
    df_ici_clean = df_ici.dropna(subset=ici_cols).copy()
    X_ici = StandardScaler().fit_transform(df_ici_clean[ici_cols])
    probs_ici = gmm.predict_proba(X_ici)
    labels_ici = np.argmax(probs_ici, axis=1)
    metrics_ici = _compute_clustering_metrics(X_ici, labels_ici, gmm)

    print(f"\nClustering quality — ICI-only cohort applied to same GMM model (N={len(df_ici_clean)}):")
    for k, v in metrics_ici.items():
        print(f"  {k}: {v}")

    comparison_plot = subproject_root / "plots" / "clustering" / "cohort_size_clustering_comparison.png"
    _plot_cohort_size_comparison(metrics_ici, metrics_full, len(df_ici_clean), len_full, comparison_plot)


def _print_stratification_summary(
    df_clean: pd.DataFrame,
    final_labels: np.ndarray,
    stage1_probs: np.ndarray,
    phenotype_names: Dict[int, str],
) -> None:
    """Print a concise per-cluster summary after stratification is complete."""
    print("=" * 80)
    print("PATIENT STRATIFICATION COMPLETE (Two-Stage GMM + NF1 Deterministic Split)")
    print(f"Total Stratified Patients: {len(df_clean)}")
    for cid, name in sorted(phenotype_names.items()):
        cnt = int(np.sum(final_labels == cid))
        n_ici = 0
        if "IMMUNOTHERAPY" in df_clean.columns:
            n_ici = int((df_clean.loc[df_clean["Cluster_ID"] == cid, "IMMUNOTHERAPY"] == 1).sum())
        # Mean Stage 1 posterior: use the patient's own Stage1_Cluster_ID index to look up
        # their posterior probability from the Stage 1 probability matrix.
        stage1_rows = df_clean.loc[df_clean["Cluster_ID"] == cid, "Stage1_Cluster_ID"]
        mean_p = (
            stage1_probs[stage1_rows.index, stage1_rows.values].mean()
            if len(stage1_rows) > 0 else float("nan")
        )
        print(
            f"  * Cluster {cid} [{name}]: N={cnt} ({cnt / len(df_clean) * 100:.1f}%), "
            f"Mean Stage1 P={mean_p:.3f}, ICI-treated={n_ici}"
        )
    print("=" * 80)


def main() -> None:
    """Main execution: two-stage patient stratification on the full cohort.

    Stage 1: GMM (K=3) on 6 continuous immune/stromal features.
    Stage 2: Deterministic NF1 split on the NF1-enriched immune cluster.
    """
    print(
        f"Starting Phase 3 Two-Stage Patient Stratification "
        f"(Stage 1: GMM K=3 continuous features | Stage 2: NF1 split, Project root: {rel_path(PROJECT_ROOT)})"
    )
    if not INPUT_FILE_FULL.exists():
        raise FileNotFoundError(
            f"Missing full-cohort feature matrix at {rel_path(INPUT_FILE_FULL)}. "
            "Run 01_load_and_prepare.py first."
        )

    df_matrix = pd.read_csv(INPUT_FILE_FULL)
    print(f"Loaded full-cohort feature matrix: {len(df_matrix)} patients x {df_matrix.shape[1]} features")

    # Stage 1: GMM on 6 continuous immune features
    df_clean, X_scaled = prepare_clustering_features(df_matrix)
    feature_cols = [c for c in GMM_CONTINUOUS_FEATURES if c in df_clean.columns]
    gmm, stage1_labels, stage1_probs = _fit_and_save_gmm_model(X_scaled, feature_cols)
    df_clean["Stage1_Cluster_ID"] = stage1_labels

    # Stage 2: deterministic NF1 split -> final 4-phenotype labels
    final_labels, phenotype_names, stage1_short_labels = _assign_labels(df_clean, stage1_labels)
    df_clean["Cluster_ID"] = final_labels
    df_clean["Phenotype_Label"] = df_clean["Cluster_ID"].map(phenotype_names)

    out_probs_file, out_clusters = _export_cluster_outputs(
        df_clean, stage1_probs, phenotype_names, stage1_short_labels, OUTPUT_DIR
    )
    _generate_phase3_plots(df_clean, final_labels, X_scaled, phenotype_names, SUBPROJECT_ROOT)
    gmm_metrics = _evaluate_mahalanobis_spectral_comparisons(X_scaled, stage1_labels, gmm, OUTPUT_DIR)
    _compare_cohort_quality(gmm, feature_cols, gmm_metrics, len(df_clean), SUBPROJECT_ROOT)

    print(f"Output Clusters File : {rel_path(out_clusters)}")
    print(f"Posterior Probs File : {rel_path(out_probs_file)}")
    print(f"Metrics Output File  : {rel_path(OUTPUT_DIR / 'mahalanobis_spectral_metrics.csv')}")
    print(f"Stage 1 Features     : {feature_cols}")
    _print_stratification_summary(df_clean, final_labels, stage1_probs, phenotype_names)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
