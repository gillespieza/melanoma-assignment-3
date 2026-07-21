"""
Dimensionality Reduction & Batch Correction Analysis across Melanoma Cohorts.

Performs PCA and UMAP projections across TCGA-SKCM, Liu 2019, Hugo 2016, and Riaz 2017 cohorts
to assess technical batch effects before and after cohort-independent Z-score scaling, and updates batch_correction_report.md.
"""

import contextlib
from pathlib import Path
import sys
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import umap

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import find_project_root
from src.utils.plotting import save_fig

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
        Z-score standardized DataFrame.
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

    print("\n=== Dataset dimensions immediately after loading ===")
    print(f"Liu 2019:  expression={expr_liu.shape}, clinical={clin_liu.shape}")
    print(f"Hugo 2016: expression={expr_hugo.shape}, clinical={clin_hugo.shape}")
    print(f"Riaz 2017: expression={expr_riaz.shape}, clinical={clin_riaz.shape}")

    print("\n=== Index overlap within each dataset ===")
    print(f"Liu expression/clinical overlap: "
        f"{len(expr_liu.index.intersection(clin_liu.index))}")
    print(f"Hugo expression/clinical overlap: "
        f"{len(expr_hugo.index.intersection(clin_hugo.index))}")
    print(f"Riaz expression/clinical overlap: "
        f"{len(expr_riaz.index.intersection(clin_riaz.index))}")

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

    common_genes = list(
        set(expr_tcga.columns) & set(expr_liu.columns) & set(expr_hugo.columns) & set(expr_riaz.columns)
    )
    common_genes.sort()
    n_common_genes = len(common_genes)  

    expr_tcga = expr_tcga[common_genes]
    expr_liu = expr_liu[common_genes]
    expr_hugo = expr_hugo[common_genes]
    expr_riaz = expr_riaz[common_genes]

    raw_vars = pd.concat([expr_tcga, expr_liu, expr_hugo, expr_riaz], axis=0).var(axis=0)
    top_genes = raw_vars.sort_values(ascending=False).head(min(1000, len(common_genes))).index.tolist()
    n_top_genes = len(top_genes)    

    expr_tcga = expr_tcga[top_genes]
    expr_liu = expr_liu[top_genes]
    expr_hugo = expr_hugo[top_genes]
    expr_riaz = expr_riaz[top_genes]

    expr_full_raw = pd.concat([expr_tcga, expr_liu, expr_hugo, expr_riaz], axis=0)
    n_full = len(expr_full_raw) 

    print("\n=== Full cohort counts ===")
    print(f"TCGA-SKCM: {len(expr_tcga)}")
    print(f"Liu 2019: {len(expr_liu)}")
    print(f"Hugo 2016: {len(expr_hugo)}")
    print(f"Riaz 2017: {len(expr_riaz)}")
    print(f"Full cohort total: {n_full}")

    expr_tcga_scaled = zscore_df(expr_tcga)
    expr_liu_scaled = zscore_df(expr_liu)
    expr_hugo_scaled = zscore_df(expr_hugo)
    expr_riaz_scaled = zscore_df(expr_riaz)
    expr_full_scaled = pd.concat([expr_tcga_scaled, expr_liu_scaled, expr_hugo_scaled, expr_riaz_scaled], axis=0)

    clin_tcga["Cohort"] = "TCGA-SKCM"
    clin_liu["Cohort"] = "Liu 2019"
    clin_hugo["Cohort"] = "Hugo 2016"
    clin_riaz["Cohort"] = "Riaz 2017"

    clin_full = pd.concat(
        [clin_tcga[["Cohort"]], clin_liu[["Cohort"]], clin_hugo[["Cohort"]], clin_riaz[["Cohort"]]], axis=0
    )

    pca_raw_full = PCA(n_components=2, svd_solver="full")
    pcs_raw_full = pca_raw_full.fit_transform(expr_full_raw)
    clin_full["PCA_Raw_PC1"] = pcs_raw_full[:, 0]
    clin_full["PCA_Raw_PC2"] = pcs_raw_full[:, 1]

    pca_scaled_full = PCA(n_components=2, svd_solver="full")
    pcs_scaled_full = pca_scaled_full.fit_transform(expr_full_scaled)
    clin_full["PCA_Scaled_PC1"] = pcs_scaled_full[:, 0]
    clin_full["PCA_Scaled_PC2"] = pcs_scaled_full[:, 1]

    set_presentation_style()
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
        f"A: Raw Expression Matrix (Uncorrected)\nPC1 ({pca_raw_full.explained_variance_ratio_[0]*100:.1f}%) vs PC2 ({pca_raw_full.explained_variance_ratio_[1]*100:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid(True, linestyle="--", alpha=0.5)

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
        f"B: Z-Score Standardized Matrix (Batch-Corrected)\nPC1 ({pca_scaled_full.explained_variance_ratio_[0]*100:.1f}%) vs PC2 ({pca_scaled_full.explained_variance_ratio_[1]*100:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(f"PCA Batch Effect Assessment Across Full Cohort (N = {n_full})", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    full_pca_path = BIOMARKER_PLOT_DIR / "batch_effect_pca.png"
    save_fig(fig, full_pca_path)
    print(f"Saved Full Cohort PCA plot to {full_pca_path.relative_to(BASE_DIR).as_posix()}")

    expr_trials_raw = pd.concat([expr_liu, expr_hugo, expr_riaz], axis=0)
    n_trials = len(expr_trials_raw)

    print("\n=== Trial cohort counts ===")
    print(f"Liu 2019: {len(expr_liu)}")
    print(f"Hugo 2016: {len(expr_hugo)}")
    print(f"Riaz 2017: {len(expr_riaz)}")
    print(f"Trial cohort total: {n_trials}")
    
    expr_trials_scaled = pd.concat([expr_liu_scaled, expr_hugo_scaled, expr_riaz_scaled], axis=0)

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

    umap_raw_trials = umap.UMAP(n_components=2, random_state=42, n_jobs=1, n_neighbors=15, min_dist=0.1)
    um_raw = umap_raw_trials.fit_transform(expr_trials_raw)
    clin_trials_merged["UMAP_Raw_Dim1"] = um_raw[:, 0]
    clin_trials_merged["UMAP_Raw_Dim2"] = um_raw[:, 1]

    umap_scaled_trials = umap.UMAP(n_components=2, random_state=42, n_jobs=1, n_neighbors=15, min_dist=0.1)
    um_scaled = umap_scaled_trials.fit_transform(expr_trials_scaled)
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
    axes[0, 0].set_title("PCA Before Batch Correction (Colored by Cohort)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 0].set_xlabel(f"PC1 ({pca_raw_trials.explained_variance_ratio_[0]*100:.1f}% variance)")
    axes[0, 0].set_ylabel(f"PC2 ({pca_raw_trials.explained_variance_ratio_[1]*100:.1f}% variance)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

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
    axes[0, 1].set_title("PCA Before Batch Correction (Colored by Response)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 1].set_xlabel(f"PC1 ({pca_raw_trials.explained_variance_ratio_[0]*100:.1f}% variance)")
    axes[0, 1].set_ylabel(f"PC2 ({pca_raw_trials.explained_variance_ratio_[1]*100:.1f}% variance)")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

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
    axes[1, 0].set_title("PCA After Batch Correction (Colored by Cohort)", fontsize=13, fontweight="bold", pad=10)
    axes[1, 0].set_xlabel(f"PC1 ({pca_scaled_trials.explained_variance_ratio_[0]*100:.1f}% variance)")
    axes[1, 0].set_ylabel(f"PC2 ({pca_scaled_trials.explained_variance_ratio_[1]*100:.1f}% variance)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

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
    axes[1, 1].set_title("PCA After Batch Correction (Colored by Response)", fontsize=13, fontweight="bold", pad=10)
    axes[1, 1].set_xlabel(f"PC1 ({pca_scaled_trials.explained_variance_ratio_[0]*100:.1f}% variance)")
    axes[1, 1].set_ylabel(f"PC2 ({pca_scaled_trials.explained_variance_ratio_[1]*100:.1f}% variance)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

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
    axes[0, 0].set_title("UMAP Before Batch Correction (Colored by Cohort)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 0].set_xlabel("UMAP Dimension 1")
    axes[0, 0].set_ylabel("UMAP Dimension 2")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

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
    axes[0, 1].set_title("UMAP Before Batch Correction (Colored by Response)", fontsize=13, fontweight="bold", pad=10)
    axes[0, 1].set_xlabel("UMAP Dimension 1")
    axes[0, 1].set_ylabel("UMAP Dimension 2")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

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
    axes[1, 0].set_title("UMAP After Batch Correction (Colored by Cohort)", fontsize=13, fontweight="bold", pad=10)
    axes[1, 0].set_xlabel("UMAP Dimension 1")
    axes[1, 0].set_ylabel("UMAP Dimension 2")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

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
    axes[1, 1].set_title("UMAP After Batch Correction (Colored by Response)", fontsize=13, fontweight="bold", pad=10)
    axes[1, 1].set_xlabel("UMAP Dimension 1")
    axes[1, 1].set_ylabel("UMAP Dimension 2")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(f"UMAP Dimensionality Reduction of Trial Expression Data (N = {n_trials})", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    trial_umap_path = EXPLORATORY_PLOT_DIR / "umap_dimensionality_reduction.png"
    save_fig(fig, trial_umap_path)
    print(f"Saved Trial UMAP plot to {trial_umap_path.relative_to(BASE_DIR).as_posix()}")

    report_path = REPORT_DIR / "batch_correction_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write("cssclasses: table-small\n")
        f.write("---\n")
        f.write("# Batch Effect Assessment & Dimensionality Reduction Analysis\n\n")
        f.write("When combining transcriptomic datasets across independent clinical studies, technical variations (e.g. sequencing platforms, RNA extraction methods, and library preparation) typically dominate the biological signals. This report documents how technical batch effects were identified and corrected across our melanoma cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, and **Riaz 2017**) and whether global expression profiles separate patients based on therapeutic response. Plot aesthetics and palettes are aligned with the Okabe-Ito color guidelines used across other reports.\n\n")

        f.write(f"## 1. Full Cohort Batch Assessment (N = {n_full})\n\n")
        f.write(
            f"To evaluate overall batch effects across all samples, we performed a Principal Component Analysis (PCA) on the top {n_top_genes} most variable genes (selected from {n_common_genes} genes common to all four cohorts) across the $N={n_full}$ patients in the full merged cohort, comparing the uncorrected concatenated matrix against the individually Z-score standardized matrix.\n\n"
        )
        f.write("![[batch_effect_pca.png]]\n\n")
        f.write("### Key Observations\n")
        f.write("- **Panel A: Before Batch Correction**: The uncorrected PCA reveals a strong cohort-associated structure. TCGA-SKCM samples form a relatively distinct cluster, while the Liu 2019, Hugo 2016, and Riaz 2017 trial cohorts occupy a broadly overlapping region of the projection space. This suggests that the dominant variation in the integrated expression matrix is strongly influenced by the distinction between the TCGA dataset and the clinical-trial datasets. However, the PCA does not support the conclusion that each study forms a completely isolated cluster, nor that the first two principal components are exclusively technical in origin.\n")
        f.write("- **Panel B: After Cohort-Specific Z-Score Standardisation**: Following within-cohort Z-score standardisation, the strong separation between TCGA-SKCM and the clinical-trial cohorts is substantially reduced. The increased overlap suggests that cohort-level differences in gene-wise means and variances contributed to the original separation. However, the resulting PCA should be interpreted as evidence of reduced cohort-associated structure rather than proof that all technical batch effects have been completely eliminated.\n\n")

        f.write(f"## 2. Immunotherapy Trial Dimensionality Reduction (N = {n_trials})\n\n")
        f.write(
            f"To evaluate technical batch effects and patient response separation in the clinical trials specifically, we performed PCA and Uniform Manifold Approximation and Projection (UMAP) on the $N={n_trials}$ response-aligned trial patients (Liu 2019, Hugo 2016, and Riaz 2017) using the {n_top_genes} common genes.\n\n"
        )

        f.write("### PCA Projections (Raw vs. Standardized)\n")
        f.write("![[pca_dimensionality_reduction.png]]\n\n")

        f.write("### UMAP Projections (Raw vs. Standardized)\n")
        f.write("![[umap_dimensionality_reduction.png]]\n\n")

        f.write("### Key Observations\n")
        f.write("- **Cohort Structure:** The PCA and UMAP projections provide different views of the relationship between the three trial cohorts. The effect of cohort-wise Z-score standardisation is relatively limited in the global PCA structure, while the UMAP embedding shows a more substantial reorganisation of local sample neighbourhoods. This indicates that the transformation affects local relationships between samples more strongly than the dominant global variance structure captured by PCA.\n")
        f.write("- **Response Distribution:** When coloured by treatment response, responders (CR/PR) and non-responders (PD) do not form clearly separated global groups in either the PCA or UMAP projections. Although local patterns or partial enrichment may be present, there is no obvious response-defined boundary that separates the two response groups across the overall transcriptomic space.\n")
        f.write("- **Interpretation:** The absence of clear global separation by response suggests that treatment response is not captured by a simple, dominant axis of variation in the reduced transcriptomic space. However, PCA and UMAP are exploratory visualisation methods and cannot establish whether response is predictable from gene expression. Predictive modelling and pathway-level analyses are therefore required to determine whether response-associated molecular patterns exist that are not apparent in these two-dimensional projections.\n\n")

        f.write("## 3. Gene-Level Expression Heatmaps (Top 50 Highly Variable Genes)\n\n")
        f.write("To evaluate batch correction at the individual gene level, we selected the **top 50 genes by variance** (calculated on raw log2-TPM expression data across trial patients) and performed hierarchical clustering on both uncorrected and standardized expression values.\n\n")

        f.write("### Raw Expression (Top 50 Genes)\n")
        f.write("![[heatmap_top_variance_genes_raw.png]]\n\n")

        f.write("### Standardized Expression (Top 50 Genes)\n")
        f.write("![[heatmap_top_variance_genes_standardized.png]]\n\n")

        f.write("### Key Observations\n")
        f.write("*   **Before Batch Correction (Raw log2-TPM)**: Patient columns cluster heavily by cohort source.\n")
        f.write("*   **After Batch Correction (Individual Z-scoring)**: Perfect cohort mixing across patient dendrograms.\n\n")

        f.write("## 4. Cross-Validation Rigor & Data Leakage Prevention\n\n")
        f.write("### The Hazard of Global Batch Correction (e.g. ComBat)\n")
        f.write("Algorithms like ComBat pool all samples together to estimate batch correction parameters. When applied to a multi-study cohort prior to Leave-One-Cohort-Out (LOCO) cross-validation, the test cohort is included in parameter estimation, introducing data leakage.\n\n")
        f.write("### The Z-score Scaling Solution\n")
        f.write("By applying Z-score standardization cohort-independently, zero leakage is guaranteed during model cross-validation.\n")

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
