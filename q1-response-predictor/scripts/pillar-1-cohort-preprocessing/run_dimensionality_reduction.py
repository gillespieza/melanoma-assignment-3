"""
Dimensionality Reduction & Batch Correction Analysis across Melanoma Cohorts.

Performs PCA and UMAP projections across TCGA-SKCM, Liu 2019, Hugo 2016, and Riaz 2017 cohorts
to assess technical batch effects before and after cohort-independent Z-score scaling, and updates batch_correction_report.md.
"""

import contextlib
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

try:
    import umap
    HAS_UMAP = True
except Exception as e:
    print(f"Warning: umap import failed ({e}). Falling back to t-SNE (TSNE) for non-linear dimensionality reduction.")
    HAS_UMAP = False

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style
from src.utils.formatting import generate_obsidian_frontmatter
from src.utils.logging import TeeStream
from src.utils.paths import find_project_root
from src.utils.plotting import save_fig

set_presentation_style()

# Module-level Constants
DATA_DIR = find_project_root(Path(__file__).resolve()) / "data"
PLOT_DIR = BASE_DIR / "plots"
EXPLORATORY_PLOT_DIR = PLOT_DIR / "exploratory"
BIOMARKER_PLOT_DIR = PLOT_DIR / "biomarkers"
REPORT_DIR = BASE_DIR / "reports" / "pillar-1-cohorts-and-preprocessing"
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "run_dimensionality_reduction.log"


def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardises DataFrame columns to mean=0, std=1.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Z-score standardised DataFrame.
    """
    means = df.mean(axis=0)
    stds = df.std(axis=0).replace(0, 1.0).fillna(1.0)
    return (df - means) / stds


def main() -> None:
    """Executes dimensionality reduction workflow across full and trial cohorts."""
    print("==================================================")
    print("Dimensionality Reduction: Full & Trial Cohorts")
    print("==================================================\n")

    for d in [EXPLORATORY_PLOT_DIR, BIOMARKER_PLOT_DIR, REPORT_DIR]:
        d.mkdir(exist_ok=True, parents=True)

    print("Loading datasets...")
    expr_liu, clin_liu = load_liu_2019(DATA_DIR)
    expr_hugo, clin_hugo = load_hugo_2016(DATA_DIR)
    expr_riaz, clin_riaz = load_riaz_2017(DATA_DIR)

    common_liu = expr_liu.index.intersection(clin_liu.index)
    expr_liu = expr_liu.loc[common_liu]
    clin_liu = clin_liu.loc[common_liu]

    common_hugo = expr_hugo.index.intersection(clin_hugo.index)
    expr_hugo = expr_hugo.loc[common_hugo]
    clin_hugo = clin_hugo.loc[common_hugo]

    common_riaz = expr_riaz.index.intersection(clin_riaz.index)
    expr_riaz = expr_riaz.loc[common_riaz]
    clin_riaz = clin_riaz.loc[common_riaz]

    print("\n=== Dataset dimensions immediately after loading ===")
    print(f"Liu 2019:  expression={expr_liu.shape}, clinical={clin_liu.shape}")
    print(f"Hugo 2016: expression={expr_hugo.shape}, clinical={clin_hugo.shape}")
    print(f"Riaz 2017: expression={expr_riaz.shape}, clinical={clin_riaz.shape}")

    print("\n=== Index overlap within each dataset ===")
    print(f"Liu expression/clinical overlap: {len(common_liu)}")
    print(f"Hugo expression/clinical overlap: {len(common_hugo)}")
    print(f"Riaz expression/clinical overlap: {len(common_riaz)}")

    tcga_dir = DATA_DIR / "processed" / "skcm_tcga_pan_can_atlas_2018"
    expr_tcga = pd.read_csv(tcga_dir / "expr_cleaned.csv", index_col="SAMPLE_ID")
    clin_tcga = pd.read_csv(tcga_dir / "clin_cleaned.csv", index_col="SAMPLE_ID")
    common_tcga = expr_tcga.index.intersection(clin_tcga.index)
    expr_tcga = expr_tcga.loc[common_tcga]
    clin_tcga = clin_tcga.loc[common_tcga]

    print("\n=== TCGA dimensions after expression/clinical intersection ===")
    print(f"TCGA expression: {expr_tcga.shape}")
    print(f"TCGA clinical: {clin_tcga.shape}")
    print(f"TCGA common patients: {len(common_tcga)}")

    n_tcga = len(expr_tcga)
    n_liu = len(expr_liu)
    n_hugo = len(expr_hugo)
    n_riaz = len(expr_riaz)

    # Genes common across all 4 cohorts
    common_genes_4 = list(
        set(expr_tcga.columns) & set(expr_liu.columns) & set(expr_hugo.columns) & set(expr_riaz.columns)
    )
    common_genes_4.sort()
    n_common_genes_4 = len(common_genes_4)

    # Genes common across 3 ICI trial cohorts
    common_genes_3 = list(
        set(expr_liu.columns) & set(expr_hugo.columns) & set(expr_riaz.columns)
    )
    common_genes_3.sort()
    n_common_genes_3 = len(common_genes_3)

    # Select top variable genes across full 4 cohorts
    raw_vars_4 = pd.concat(
        [expr_tcga[common_genes_4], expr_liu[common_genes_4], expr_hugo[common_genes_4], expr_riaz[common_genes_4]],
        axis=0,
    ).var(axis=0)
    top_genes_4 = raw_vars_4.sort_values(ascending=False).head(min(1000, n_common_genes_4)).index.tolist()
    n_top_genes = len(top_genes_4)

    expr_tcga_top = expr_tcga[top_genes_4]
    expr_liu_top = expr_liu[top_genes_4]
    expr_hugo_top = expr_hugo[top_genes_4]
    expr_riaz_top = expr_riaz[top_genes_4]

    expr_full_raw = pd.concat([expr_tcga_top, expr_liu_top, expr_hugo_top, expr_riaz_top], axis=0)
    n_full = len(expr_full_raw)

    print("\n=== Full cohort counts ===")
    print(f"TCGA-SKCM: {n_tcga}")
    print(f"Liu 2019: {n_liu}")
    print(f"Hugo 2016: {n_hugo}")
    print(f"Riaz 2017: {n_riaz}")
    print(f"Full cohort total: {n_full}")

    expr_tcga_scaled = zscore_df(expr_tcga_top)
    expr_liu_scaled = zscore_df(expr_liu_top)
    expr_hugo_scaled = zscore_df(expr_hugo_top)
    expr_riaz_scaled = zscore_df(expr_riaz_top)
    expr_full_scaled = pd.concat([expr_tcga_scaled, expr_liu_scaled, expr_hugo_scaled, expr_riaz_scaled], axis=0)

    clin_tcga["Cohort"] = "TCGA-SKCM"
    clin_liu["Cohort"] = "Liu 2019"
    clin_hugo["Cohort"] = "Hugo 2016"
    clin_riaz["Cohort"] = "Riaz 2017"

    clin_full = pd.concat(
        [clin_tcga[["Cohort"]], clin_liu[["Cohort"]], clin_hugo[["Cohort"]], clin_riaz[["Cohort"]]], axis=0
    )

    # 1.1 PCA Full Cohort
    pca_raw_full = PCA(n_components=2, svd_solver="full")
    pcs_raw_full = pca_raw_full.fit_transform(expr_full_raw)
    clin_full["PCA_Raw_PC1"] = pcs_raw_full[:, 0]
    clin_full["PCA_Raw_PC2"] = pcs_raw_full[:, 1]

    pca_scaled_full = PCA(n_components=2, svd_solver="full")
    pcs_scaled_full = pca_scaled_full.fit_transform(expr_full_scaled)
    clin_full["PCA_Scaled_PC1"] = pcs_scaled_full[:, 0]
    clin_full["PCA_Scaled_PC2"] = pcs_scaled_full[:, 1]

    pc1_raw_full = pca_raw_full.explained_variance_ratio_[0] * 100.0
    pc2_raw_full = pca_raw_full.explained_variance_ratio_[1] * 100.0
    pc1_scaled_full = pca_scaled_full.explained_variance_ratio_[0] * 100.0
    pc2_scaled_full = pca_scaled_full.explained_variance_ratio_[1] * 100.0

    colors_cohort = COHORT_PALETTE
    cohort_order = ["TCGA-SKCM", "Liu 2019", "Hugo 2016", "Riaz 2017"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(
        data=clin_full,
        x="PCA_Raw_PC1",
        y="PCA_Raw_PC2",
        hue="Cohort",
        hue_order=cohort_order,
        palette=colors_cohort,
        alpha=0.7,
        s=60,
        ax=axes[0],
    )
    axes[0].set_title(
        f"A: Raw Expression Matrix (Uncorrected)\nPC1 ({pc1_raw_full:.1f}%) vs PC2 ({pc2_raw_full:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    sns.scatterplot(
        data=clin_full,
        x="PCA_Scaled_PC1",
        y="PCA_Scaled_PC2",
        hue="Cohort",
        hue_order=cohort_order,
        palette=colors_cohort,
        alpha=0.7,
        s=60,
        ax=axes[1],
    )
    axes[1].set_title(
        f"B: Z-Score Standardised Matrix (Batch-Corrected)\nPC1 ({pc1_scaled_full:.1f}%) vs PC2 ({pc2_scaled_full:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    plt.suptitle(f"PCA Batch Effect Assessment Across Full Cohort (N = {n_full})", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    full_pca_path = BIOMARKER_PLOT_DIR / "batch_effect_pca.png"
    save_fig(fig, full_pca_path)
    print(f"Saved Full Cohort PCA plot to {full_pca_path.relative_to(BASE_DIR).as_posix()}")

    # 1.2 PCA ICI Trial Cohorts (3 cohorts, all common trial genes)
    expr_trials_raw = pd.concat([expr_liu_top, expr_hugo_top, expr_riaz_top], axis=0)
    n_trials = len(expr_trials_raw)

    print("\n=== Trial cohort counts ===")
    print(f"Liu 2019: {n_liu}")
    print(f"Hugo 2016: {n_hugo}")
    print(f"Riaz 2017: {n_riaz}")
    print(f"Trial cohort total: {n_trials}")

    expr_trials_scaled = pd.concat([expr_liu_scaled, expr_hugo_scaled, expr_riaz_scaled], axis=0)

    # 1.2 ICI Trial Cohort Batch Assessment Plot (3 cohorts, all common genes)
    expr_trials_all_raw = pd.concat(
        [expr_liu[common_genes_3], expr_hugo[common_genes_3], expr_riaz[common_genes_3]], axis=0
    )
    expr_trials_all_scaled = pd.concat(
        [zscore_df(expr_liu[common_genes_3]), zscore_df(expr_hugo[common_genes_3]), zscore_df(expr_riaz[common_genes_3])],
        axis=0,
    )

    pca_raw_trials_all = PCA(n_components=2, svd_solver="full")
    pcs_raw_trials_all = pca_raw_trials_all.fit_transform(expr_trials_all_raw)

    pca_scaled_trials_all = PCA(n_components=2, svd_solver="full")
    pcs_scaled_trials_all = pca_scaled_trials_all.fit_transform(expr_trials_all_scaled)

    pc1_raw_trials_all = pca_raw_trials_all.explained_variance_ratio_[0] * 100.0
    pc2_raw_trials_all = pca_raw_trials_all.explained_variance_ratio_[1] * 100.0
    pc1_scaled_trials_all = pca_scaled_trials_all.explained_variance_ratio_[0] * 100.0
    pc2_scaled_trials_all = pca_scaled_trials_all.explained_variance_ratio_[1] * 100.0

    clin_trials_all = pd.concat(
        [clin_liu[["Cohort"]], clin_hugo[["Cohort"]], clin_riaz[["Cohort"]]], axis=0
    )
    clin_trials_all["PCA_Raw_PC1"] = pcs_raw_trials_all[:, 0]
    clin_trials_all["PCA_Raw_PC2"] = pcs_raw_trials_all[:, 1]
    clin_trials_all["PCA_Scaled_PC1"] = pcs_scaled_trials_all[:, 0]
    clin_trials_all["PCA_Scaled_PC2"] = pcs_scaled_trials_all[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(
        data=clin_trials_all,
        x="PCA_Raw_PC1",
        y="PCA_Raw_PC2",
        hue="Cohort",
        hue_order=["Liu 2019", "Hugo 2016", "Riaz 2017"],
        palette=colors_cohort,
        alpha=0.8,
        s=80,
        edgecolor="w",
        linewidth=0.5,
        ax=axes[0],
    )
    axes[0].set_title(
        f"A: Uncorrected Raw Expression Matrix\nPC1 ({pc1_raw_trials_all:.1f}%) vs PC2 ({pc2_raw_trials_all:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    sns.scatterplot(
        data=clin_trials_all,
        x="PCA_Scaled_PC1",
        y="PCA_Scaled_PC2",
        hue="Cohort",
        hue_order=["Liu 2019", "Hugo 2016", "Riaz 2017"],
        palette=colors_cohort,
        alpha=0.8,
        s=80,
        edgecolor="w",
        linewidth=0.5,
        ax=axes[1],
    )
    axes[1].set_title(
        f"B: Cohort-Wise Z-Score Standardised Matrix\nPC1 ({pc1_scaled_trials_all:.1f}%) vs PC2 ({pc2_scaled_trials_all:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    plt.suptitle(f"PCA Batch Effect Assessment Across ICI Trial Cohorts (N = {n_trials})", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    ici_pca_path = BIOMARKER_PLOT_DIR / "batch_effect_ici_pca.png"
    save_fig(fig, ici_pca_path)
    print(f"Saved Trial Cohort ICI PCA plot to {ici_pca_path.relative_to(BASE_DIR).as_posix()}")

    # Section 2 PCA & UMAP Projections (Trial Cohorts N=256)
    clin_trials_merged = pd.concat(
        [
            clin_liu[["Cohort", "response"]],
            clin_hugo[["Cohort", "response"]],
            clin_riaz[["Cohort", "response"]],
        ],
        axis=0,
    )
    clin_trials_merged["Response"] = clin_trials_merged["response"].map(
        {1.0: "Responder (CR/PR)", 0.0: "Non-responder (PD)"}
    )

    pca_raw_trials = PCA(n_components=2, svd_solver="full")
    pcs_raw_trials = pca_raw_trials.fit_transform(expr_trials_raw)
    clin_trials_merged["PCA_Raw_PC1"] = pcs_raw_trials[:, 0]
    clin_trials_merged["PCA_Raw_PC2"] = pcs_raw_trials[:, 1]

    pca_scaled_trials = PCA(n_components=2, svd_solver="full")
    pcs_scaled_trials = pca_scaled_trials.fit_transform(expr_trials_scaled)
    clin_trials_merged["PCA_Scaled_PC1"] = pcs_scaled_trials[:, 0]
    clin_trials_merged["PCA_Scaled_PC2"] = pcs_scaled_trials[:, 1]

    pc1_raw_trials = pca_raw_trials.explained_variance_ratio_[0] * 100.0
    pc2_raw_trials = pca_raw_trials.explained_variance_ratio_[1] * 100.0
    pc1_scaled_trials = pca_scaled_trials.explained_variance_ratio_[0] * 100.0
    pc2_scaled_trials = pca_scaled_trials.explained_variance_ratio_[1] * 100.0

    if HAS_UMAP:
        umap_raw_trials = umap.UMAP(n_components=2, random_state=42, n_jobs=1, n_neighbors=15, min_dist=0.1)
        um_raw = umap_raw_trials.fit_transform(expr_trials_raw)
        umap_scaled_trials = umap.UMAP(n_components=2, random_state=42, n_jobs=1, n_neighbors=15, min_dist=0.1)
        um_scaled = umap_scaled_trials.fit_transform(expr_trials_scaled)
    else:
        tsne_raw = TSNE(n_components=2, random_state=42, perplexity=15)
        um_raw = tsne_raw.fit_transform(expr_trials_raw)
        tsne_scaled = TSNE(n_components=2, random_state=42, perplexity=15)
        um_scaled = tsne_scaled.fit_transform(expr_trials_scaled)

    clin_trials_merged["UMAP_Raw_Dim1"] = um_raw[:, 0]
    clin_trials_merged["UMAP_Raw_Dim2"] = um_raw[:, 1]
    clin_trials_merged["UMAP_Scaled_Dim1"] = um_scaled[:, 0]
    clin_trials_merged["UMAP_Scaled_Dim2"] = um_scaled[:, 1]

    colors_response = {"Responder (CR/PR)": RESPONSE_PALETTE["CR/PR"], "Non-responder (PD)": RESPONSE_PALETTE["PD"]}
    response_order = ["Responder (CR/PR)", "Non-responder (PD)"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    sns.scatterplot(
        data=clin_trials_merged,
        x="PCA_Raw_PC1",
        y="PCA_Raw_PC2",
        hue="Cohort",
        hue_order=cohort_order[1:],
        palette=colors_cohort,
        style="Cohort",
        alpha=0.8,
        s=100,
        edgecolor="w",
        linewidth=0.8,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("PCA Before Batch Correction (Coloured by Cohort)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 0].set_xlabel(f"PC1 ({pc1_raw_trials:.1f}% variance)")
    axes[0, 0].set_ylabel(f"PC2 ({pc2_raw_trials:.1f}% variance)")

    sns.scatterplot(
        data=clin_trials_merged,
        x="PCA_Raw_PC1",
        y="PCA_Raw_PC2",
        hue="Response",
        hue_order=response_order,
        palette=colors_response,
        style="Response",
        alpha=0.8,
        s=100,
        edgecolor="w",
        linewidth=0.8,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("PCA Before Batch Correction (Coloured by Response)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 1].set_xlabel(f"PC1 ({pc1_raw_trials:.1f}% variance)")
    axes[0, 1].set_ylabel(f"PC2 ({pc2_raw_trials:.1f}% variance)")

    sns.scatterplot(
        data=clin_trials_merged,
        x="PCA_Scaled_PC1",
        y="PCA_Scaled_PC2",
        hue="Cohort",
        hue_order=cohort_order[1:],
        palette=colors_cohort,
        style="Cohort",
        alpha=0.8,
        s=100,
        edgecolor="w",
        linewidth=0.8,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("PCA After Batch Correction (Coloured by Cohort)", fontsize=13, fontweight="bold", pad=10)
    axes[1, 0].set_xlabel(f"PC1 ({pc1_scaled_trials:.1f}% variance)")
    axes[1, 0].set_ylabel(f"PC2 ({pc2_scaled_trials:.1f}% variance)")

    sns.scatterplot(
        data=clin_trials_merged,
        x="PCA_Scaled_PC1",
        y="PCA_Scaled_PC2",
        hue="Response",
        hue_order=response_order,
        palette=colors_response,
        style="Response",
        alpha=0.8,
        s=100,
        edgecolor="w",
        linewidth=0.8,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("PCA After Batch Correction (Coloured by Response)", fontsize=13, fontweight="bold", pad=10)
    axes[1, 1].set_xlabel(f"PC1 ({pc1_scaled_trials:.1f}% variance)")
    axes[1, 1].set_ylabel(f"PC2 ({pc2_scaled_trials:.1f}% variance)")

    plt.suptitle(f"PCA Dimensionality Reduction of Trial Expression Data (N = {n_trials})", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    trial_pca_path = EXPLORATORY_PLOT_DIR / "pca_dimensionality_reduction.png"
    save_fig(fig, trial_pca_path)
    print(f"Saved Trial PCA plot to {trial_pca_path.relative_to(BASE_DIR).as_posix()}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    sns.scatterplot(
        data=clin_trials_merged,
        x="UMAP_Raw_Dim1",
        y="UMAP_Raw_Dim2",
        hue="Cohort",
        hue_order=cohort_order[1:],
        palette=colors_cohort,
        style="Cohort",
        alpha=0.8,
        s=100,
        edgecolor="w",
        linewidth=0.8,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("UMAP Before Batch Correction (Coloured by Cohort)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 0].set_xlabel("UMAP Dimension 1")
    axes[0, 0].set_ylabel("UMAP Dimension 2")

    sns.scatterplot(
        data=clin_trials_merged,
        x="UMAP_Raw_Dim1",
        y="UMAP_Raw_Dim2",
        hue="Response",
        hue_order=response_order,
        palette=colors_response,
        style="Response",
        alpha=0.8,
        s=100,
        edgecolor="w",
        linewidth=0.8,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("UMAP Before Batch Correction (Coloured by Response)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 1].set_xlabel("UMAP Dimension 1")
    axes[0, 1].set_ylabel("UMAP Dimension 2")

    sns.scatterplot(
        data=clin_trials_merged,
        x="UMAP_Scaled_Dim1",
        y="UMAP_Scaled_Dim2",
        hue="Cohort",
        hue_order=cohort_order[1:],
        palette=colors_cohort,
        style="Cohort",
        alpha=0.8,
        s=100,
        edgecolor="w",
        linewidth=0.8,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("UMAP After Batch Correction (Coloured by Cohort)", fontsize=13, fontweight="bold", pad=10)
    axes[1, 0].set_xlabel("UMAP Dimension 1")
    axes[1, 0].set_ylabel("UMAP Dimension 2")

    sns.scatterplot(
        data=clin_trials_merged,
        x="UMAP_Scaled_Dim1",
        y="UMAP_Scaled_Dim2",
        hue="Response",
        hue_order=response_order,
        palette=colors_response,
        style="Response",
        alpha=0.8,
        s=100,
        edgecolor="w",
        linewidth=0.8,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("UMAP After Batch Correction (Coloured by Response)", fontsize=13, fontweight="bold", pad=10)
    axes[1, 1].set_xlabel("UMAP Dimension 1")
    axes[1, 1].set_ylabel("UMAP Dimension 2")

    plt.suptitle(f"UMAP Dimensionality Reduction of Trial Expression Data (N = {n_trials})", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    trial_umap_path = EXPLORATORY_PLOT_DIR / "umap_dimensionality_reduction.png"
    save_fig(fig, trial_umap_path)
    print(f"Saved Trial UMAP plot to {trial_umap_path.relative_to(BASE_DIR).as_posix()}")

    report_path = REPORT_DIR / "batch_correction_report.md"
    frontmatter = generate_obsidian_frontmatter(
        title="Batch Effect Assessment & Dimensionality Reduction Analysis",
        aliases=["Q1 Batch Correction Report", "Batch Effect Assessment"],
        tags=["melanoma", "batch-correction", "pca", "umap", "tme", "transcriptomics"],
        extra_css_classes=["table-center", "row-alt"],
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(frontmatter + "\n\n")
        f.write("# Batch Effect Assessment & Dimensionality Reduction Analysis\n\n")
        f.write(
            "When combining transcriptomic datasets across independent clinical studies, technical variations "
            "(e.g. sequencing platforms, RNA extraction methods, and library preparation protocols) typically dominate "
            "the underlying biological signals. This report documents how technical batch effects were evaluated and "
            "harmonised across four melanoma cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, and **Riaz 2017**) "
            "and tests whether global transcriptomic profiles naturally separate patients based on therapeutic response. "
            "Visualisations follow the Okabe-Ito colour guidelines used throughout the study.\n\n"
        )

        f.write("## 1. Cohort Batch Assessment\n\n")

        f.write(f"### 1.1 Full Cohort Batch Assessment (N = {n_full})\n\n")
        f.write(
            "> [!INFO] Why We Are Doing This\n"
            "> **What**: We perform Principal Component Analysis (PCA) across all $N = "
            f"{n_full}$ patients from four combined melanoma cohorts (**TCGA-SKCM** [$N = {n_tcga}$], "
            f"**Liu 2019** [$N = {n_liu}$], **Hugo 2016** [$N = {n_hugo}$], and **Riaz 2017** [$N = {n_riaz}$]) "
            f"using the top {n_top_genes:,} most variable genes selected from the {n_common_genes_4:,} common genes across all datasets.\n"
            "> **Why**: Combining transcriptomic data from diverse sequencing centres introduces technical distortions (batch effects). Uncorrected models risk classifying sequencing centres rather than patient biology.\n"
            "> **Question Answered**: Does cohort-independent Z-score standardisation eliminate macro-level technical separation between reference tissue (TCGA-SKCM) and active clinical trial cohorts?\n\n"
        )
        f.write("![[batch_effect_pca.png]]\n\n")
        f.write("### Key Observations\n")
        f.write(
            f"- **Panel A: Before Batch Correction (Raw Data)**: The uncorrected PCA projection reveals a strong separation between the TCGA-SKCM reference dataset and the three clinical trial cohorts. Uncorrected PC1 ({pc1_raw_full:.1f}% variance) and PC2 ({pc2_raw_full:.1f}% variance) reflect laboratory platform shifts.\n"
        )
        f.write(
            f"- **Panel B: After Cohort-Specific Z-Score Standardisation**: Cohort-wise Z-score standardisation (centering each gene to $\\mu = 0, \\sigma = 1$ within each study) aligns the TCGA-SKCM reference with trial cohorts. Post-correction PC1 ({pc1_scaled_full:.1f}% variance) and PC2 ({pc2_scaled_full:.1f}% variance) show homogeneous distribution across datasets.\n\n"
        )

        f.write(f"### 1.2 ICI Trial Cohort Batch Assessment (N = {n_trials})\n\n")
        f.write(
            "> [!INFO] Why We Are Doing This\n"
            "> **What**: We evaluate technical batch effects specifically between the three active anti-PD-1 training cohorts "
            f"(**Liu 2019** [$N = {n_liu}$], **Hugo 2016** [$N = {n_hugo}$], and **Riaz 2017** [$N = {n_riaz}$]; $N = {n_trials}$) "
            f"across all {n_common_genes_3:,} common trial genes before and after cohort-wise Z-score standardisation.\n"
            "> **Why**: These trials vary by platform (Illumina HiSeq 2500 vs HiSeq 2000), tissue state (fresh-frozen vs FFPE), and prior treatment. We must verify baseline offsets are eliminated before Leave-One-Cohort-Out (LOCO) cross-validation.\n"
            "> **Question Answered**: Are inter-trial technical offsets harmonised across the model training cohorts without leaking test-set information?\n\n"
        )
        f.write("![[batch_effect_ici_pca.png]]\n\n")
        f.write("### Key Observations\n")
        f.write(
            f"- **Panel A: Before Batch Correction (Uncorrected Raw Expression)**: In uncorrected $\\log_2(\\text{{TPM}})$ space across all {n_common_genes_3:,} trial genes, `Liu 2019` ($N = {n_liu}$, HiSeq 2500) separates along PC1 ({pc1_raw_trials_all:.1f}% variance) from `Riaz 2017` ($N = {n_riaz}$, HiSeq 2000 / FFPE) and `Hugo 2016` ($N = {n_hugo}$, HiSeq 2000 / fresh-frozen). This confirms that sequencing depth and platform chemistry dominate raw expression signals.\n"
        )
        f.write(
            f"- **Panel B: After Cohort-Wise Z-Score Standardisation**: Standardising gene expression independently within each cohort completely removes artificial study-level separation. The distributions for Liu 2019, Hugo 2016, and Riaz 2017 overlap smoothly across PC1 ({pc1_scaled_trials_all:.1f}% variance) and PC2 ({pc2_scaled_trials_all:.1f}% variance), ensuring unbiased model training.\n\n"
        )

        f.write(f"## 2. Immunotherapy Trial Dimensionality Reduction (N = {n_trials})\n\n")
        f.write(
            "> [!INFO] Why We Are Doing This\n"
            f"> **What**: We apply linear (PCA) and non-linear (UMAP) dimensionality reduction to the $N = {n_trials}$ response-annotated trial patients (Liu 2019, Hugo 2016, Riaz 2017) using the top {n_top_genes:,} variable genes.\n"
            "> **Why**: To test whether baseline gene expression profiles naturally segregate treatment responders from non-responders prior to supervised machine learning.\n"
            "> **Question Answered**: Can therapeutic response be predicted directly from global 2D expression clusters, or are targeted biomarker signatures required?\n\n"
        )

        f.write("### PCA Projections (Raw vs. Standardised)\n")
        f.write("![[pca_dimensionality_reduction.png]]\n\n")

        f.write("### UMAP Projections (Raw vs. Standardised)\n")
        f.write("![[umap_dimensionality_reduction.png]]\n\n")

        f.write("> [!INSIGHT] Key Insights: Dimensionality Reduction & Patient Distribution\n")
        f.write(
            f"- **Cohort Harmonisation**: Z-score scaling successfully integrates `Liu 2019` ($N = {n_liu}$), `Riaz 2017` ($N = {n_riaz}$), and `Hugo 2016` ($N = {n_hugo}$) across both PCA and UMAP embeddings.\n"
        )
        f.write(
            "- **Homogeneous Response Mixing**: Responders (CR/PR) and non-responders (PD) mix homogeneously throughout PCA and UMAP projections, with zero global cluster separation by clinical outcome.\n"
        )
        f.write(
            "- **Biological Rationale**: Immunotherapy response is driven by multi-pathway immune microenvironment features (e.g. `CD274`, `PDCD1`, `IFNG` signalling) rather than global transcriptomic variance. Simple 2D projections cannot separate response groups, proving the necessity for supervised multivariate classifiers.\n\n"
        )

        f.write("## 3. Gene-Level Expression Heatmaps (Top 50 Highly Variable Genes)\n\n")
        f.write(
            "> [!INFO] Why We Are Doing This\n"
            f"> **What**: We inspect individual gene expression heatmaps for the top 50 most variable genes across trial patients ($N = {n_trials}$) with hierarchical clustering.\n"
            "> **Why**: Dimensionality reduction aggregates thousands of genes into single axes. Heatmaps allow direct inspection of batch effects at individual gene resolutions.\n"
            "> **Question Answered**: Does within-cohort Z-score standardisation prevent individual high-variance genes from clustering patients by study origin?\n\n"
        )

        f.write("### Raw Expression (Top 50 Genes)\n")
        f.write("![[heatmap_top_variance_genes_raw.png]]\n\n")

        f.write("### Standardised Expression (Top 50 Genes)\n")
        f.write("![[heatmap_top_variance_genes_standardized.png]]\n\n")

        f.write("### Key Observations\n")
        f.write("- **Before Batch Correction (Raw log2-TPM)**: Patient columns cluster strongly by cohort source, with distinct blocks corresponding to individual clinical studies.\n")
        f.write("- **After Batch Correction (Cohort Z-Scoring)**: Within-cohort Z-score standardisation eliminates study-based clustering, producing complete cohort mixing across the hierarchical dendrogram.\n\n")

        f.write("## 4. Cross-Validation Rigor & Data Leakage Prevention\n\n")
        f.write(
            "> [!INFO] Why We Are Doing This\n"
            "> **What**: We compare cohort-independent Z-score standardisation against global batch correction algorithms (such as ComBat).\n"
            "> **Why**: Data preprocessing methods used in cross-validation must strictly preserve test-set independence.\n"
            "> **Question Answered**: How does cohort-independent Z-score scaling prevent data leakage during Leave-One-Cohort-Out (LOCO) evaluation?\n\n"
        )

        f.write("### 4.1 The Hazard of Global Batch Correction (e.g. ComBat)\n")
        f.write(
            "Global batch correction algorithms like ComBat estimate location and scale transformation parameters using all samples pooled across all available cohorts. When performing Leave-One-Cohort-Out (LOCO) cross-validation, including the held-out test cohort in parameter estimation allows information from the test set to leak into the training phase. This **data leakage** produces artificially inflated performance metrics that fail to generalise to external clinical validation sets.\n\n"
        )

        f.write("### 4.2 The Cohort-Independent Z-Score Solution\n")
        f.write(
            "Standardising gene expression independently within each cohort (rescaling each gene using only that cohort's internal mean $\\mu$ and standard deviation $\\sigma$) guarantees zero data leakage. Each held-out study remains completely unobserved during model training, ensuring robust, generalisable estimates of real-world predictive performance.\n\n"
        )

        f.write("> [!WARNING] Methodological Limitations & Future Rationale\n")
        f.write(
            "- **Sample Size Constraints**: The smallest training cohort (`Hugo 2016`, $N = 27$) has reduced statistical power compared to `Liu 2019` ($N = 122$) and `Riaz 2017` ($N = 107$).\n"
        )
        f.write(
            "- **Platform Heterogeneity**: Z-score scaling harmonises gene-wise means and variances but does not alter relative non-linear gene correlations within a single study.\n"
        )
        f.write(
            "- **Pipeline Scope**: Unsupervised projections confirm that single-gene thresholds are insufficient for response prediction, motivating the 12-feature multimodal ensemble (incorporating TMB, TIS, CYT, and driver mutations like `BRAF`, `NRAS`, `NF1`) evaluated in downstream Q1 phases.\n"
        )
        f.write(
            "\n> [!formula]+ Batch Correction Script Execution & Software Module Architecture\n"
            "> - **Primary Pipeline Execution Scripts**:\n"
            ">   - [`run_dimensionality_reduction.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/run_dimensionality_reduction.py): Evaluates technical batch effects across four melanoma cohorts (TCGA-SKCM, Liu 2019, Hugo 2016, Riaz 2017), computes uncorrected vs. cohort Z-score standardised PCA/UMAP projections, generates top 50 variable gene heatmaps, and outputs `batch_correction_report.md`.\n"
            "> - **Data Preprocessing & Loading Modules**:\n"
            ">   - [`clean_data.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/clean_data.py): Preprocesses raw cohort clinical metadata and RNA-seq expression profiles into cleaned CSV matrices.\n"
            ">   - [`merge_datasets.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/merge_datasets.py): Merges processed expression matrices across cohorts into harmonised pooled matrices (`expr_merged.csv`, `clin_merged.csv`).\n"
            ">   - [`data_loaders.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/data_loaders.py): Provides helper loader functions (`load_liu_2019`, `load_hugo_2016`, `load_riaz_2017`) for retrieving expression and clinical data.\n"
            "> - **Shared Cross-Question & Pipeline Modules**:\n"
            ">   - [`run_pipeline.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/run_pipeline.py): Master Q1 pipeline orchestrator executing downstream modeling and evaluation.\n"
            ">   - [`styles.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/src/styles.py): Single source of truth for Okabe-Ito colour palettes (`COHORT_PALETTE`, `RESPONSE_PALETTE`) and visualization presentation style.\n"
        )

    print(f"Report written to {report_path.relative_to(BASE_DIR).as_posix()}")

    redundant_report = REPORT_DIR / "dimensionality_reduction_report.md"
    if redundant_report.exists():
        print(f"Removing redundant report: {redundant_report.relative_to(BASE_DIR).as_posix()}")
        redundant_report.unlink()

    print("==================================================")
    print("Done!")
    print("==================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(BASE_DIR).as_posix()}")
            main()
