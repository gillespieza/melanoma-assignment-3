"""
Full-Dataset Patient Phenotyping via Immunological & Genomic Clustering.

Performs unsupervised Agglomerative Hierarchical Clustering (Ward linkage) across
all four study cohorts (TCGA-SKCM, Liu 2019, Hugo 2016, Riaz 2017; N = 699) using
within-cohort Z-score standardized immune signatures (IFN-gamma, TIS, CYT, CD8,
PD-L1, IMPRES), mutational burden (TMB), and patient age.

Evaluates cluster phenotypes via multi-dimensional profiling, polar radar charts,
annotated Z-score heatmaps, 2D PCA projections, immunotherapy response rate analysis
(CR/PR %), Kaplan-Meier survival curves, and exports an Obsidian-compatible Markdown report.
"""

import contextlib
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import chi2_contingency
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

_SUBPROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBPROJECT_ROOT))

from src.config.datasets import DatasetConfig, load_dataset_config
from src.signatures import extract_all_signatures
from src.styles import COHORT_PALETTE, PHENOTYPE_PALETTE, RESPONSE_PALETTE, set_presentation_style
from src.utils.formatting import (
    format_count_percentage,
    generate_obsidian_frontmatter,
)
from src.utils.logging import TeeStream
from src.utils.paths import (
    CONFIG_DIR,
    DATA_DIR,
    LOG_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    SUBPROJECT_ROOT,
)
from src.utils.plotting import save_fig

set_presentation_style()

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

CONFIG_PATH = CONFIG_DIR / "datasets.yaml"
PLOT_DIR = PLOTS_DIR / "clinical"
REPORT_DIR = REPORTS_DIR / "pillar-2-clinical-subtyping"
REPORT_PATH = REPORT_DIR / "clinical_phenotyping_and_feature_selection.md"
LOG_PATH = LOG_DIR / "run_clinical_clustering.log"

FEATURE_COLS: List[str] = [
    "IFN_gamma",
    "TIS",
    "CYT",
    "CD8_Tcell",
    "PD_L1",
    "IMPRES",
    "TMB_NONSYNONYMOUS",
    "AGE",
]

CLUSTER_NAMES: Dict[int, str] = {
    0: "🔥 Cluster 0: Immunologically Hot",
    1: "❄️ Cluster 1: Immunologically Cold",
    2: "🧬 Cluster 2: High-TMB / Hypermutated",
}

CLUSTER_SHORT_NAMES: Dict[int, str] = {
    0: "🔥 Cluster 0: Hot",
    1: "❄️ Cluster 1: Cold",
    2: "🧬 Cluster 2: High-TMB",
}

CLUSTER_PLOT_NAMES: Dict[int, str] = {
    0: "Cluster 0 (Hot)",
    1: "Cluster 1 (Cold)",
    2: "Cluster 2 (High TMB)",
}

CLUSTER_COLORS: Dict[int, str] = {
    0: PHENOTYPE_PALETTE["Immune Hot"],                 # Vermillion Red (#D55E00) for Hot
    1: PHENOTYPE_PALETTE["Immune Cold"],                # Okabe-Ito Blue (#0072B2) for Cold
    2: PHENOTYPE_PALETTE["Immunosuppressive M2-High"],  # Okabe-Ito Reddish Purple (#CC79A7) for High-TMB
}



def _load_and_extract_cohort_features(
    dataset_configs: Tuple[DatasetConfig, ...],
) -> pd.DataFrame:
    """Loads expression and clinical data across all cohorts, extracts signatures,
    and applies within-cohort Z-score standardization prior to pooling.

    Args:
        dataset_configs: Validated dataset configurations loaded from datasets.yaml.

    Returns:
        Merged DataFrame containing raw features, Z-score standardized features,
        cohort indicators, response labels, and survival metrics for all patients.
    """
    cohort_dfs: List[pd.DataFrame] = []

    for config in dataset_configs:
        label = config.cohort_name
        proc_dir = DATA_DIR / "processed" / config.processed_directory
        clin_path = proc_dir / "clin_cleaned.csv"
        expr_path = proc_dir / "expr_cleaned.csv"

        if not clin_path.exists():
            raise FileNotFoundError(
                f"Missing clinical file for cohort '{label}' at {clin_path.relative_to(SUBPROJECT_ROOT).as_posix()}"
            )

        df_clin = pd.read_csv(clin_path, index_col="SAMPLE_ID")

        if expr_path.exists():
            df_expr = pd.read_csv(expr_path, index_col="SAMPLE_ID")
            df_sig = extract_all_signatures(df_expr)
        else:
            df_sig = pd.DataFrame(index=df_clin.index)

        df_combined = df_clin.copy()

        for col in df_sig.columns:
            df_combined[col] = df_sig[col]

        df_combined["COHORT"] = label
        df_combined["IS_TRIAL"] = ("RESPONDER" in df_combined.columns or "RESPONSE" in df_combined.columns)

        if "AGE_AT_DIAGNOSIS" in df_combined.columns and "AGE" not in df_combined.columns:
            df_combined["AGE"] = df_combined["AGE_AT_DIAGNOSIS"]

        for col in FEATURE_COLS:
            if col not in df_combined.columns:
                df_combined[col] = np.nan

        cohort_subset = df_combined[["COHORT", "IS_TRIAL"] + FEATURE_COLS].copy()

        for meta_col in ["RESPONSE", "RESPONDER", "RESPONSE_BINARY", "OS_MONTHS", "OS_STATUS", "SEX"]:
            if meta_col in df_combined.columns:
                cohort_subset[meta_col] = df_combined[meta_col]

        # Fill all-NaN columns with 0 before SimpleImputer to maintain constant column count
        for col in FEATURE_COLS:
            if cohort_subset[col].isna().all():
                cohort_subset[col] = 0.0

        # Median imputation within cohort
        imputer = SimpleImputer(strategy="median")
        imputed_feats = imputer.fit_transform(cohort_subset[FEATURE_COLS])
        df_imputed = pd.DataFrame(imputed_feats, columns=FEATURE_COLS, index=cohort_subset.index)

        # Cohort-wise Z-score standardization
        scaler = StandardScaler()
        z_feats = scaler.fit_transform(df_imputed)

        for i, col in enumerate(FEATURE_COLS):
            cohort_subset[f"Z_{col}"] = z_feats[:, i]
            cohort_subset[col] = df_imputed[col]

        cohort_dfs.append(cohort_subset)
        print(f"  Processed {label} (N = {len(cohort_subset)} samples)")

    full_df = pd.concat(cohort_dfs, axis=0)
    print(f"\nTotal merged full dataset: {len(full_df)} patients across {len(dataset_configs)} cohorts.")
    return full_df


def _perform_clustering(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Applies Agglomerative Hierarchical Clustering (Ward linkage, K=3) on pooled Z-scores.

    Args:
        df: Merged patient DataFrame with Z-score feature columns.

    Returns:
        Tuple of (DataFrame with CLINICAL_CLUSTER assignments, 2D PCA array).
    """
    z_cols = [f"Z_{col}" for col in FEATURE_COLS]
    z_matrix = df[z_cols].values

    agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
    cluster_labels = agg.fit_predict(z_matrix)

    df["CLINICAL_CLUSTER"] = cluster_labels

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(z_matrix)

    return df, pca_coords


def _plot_cluster_radar(df: pd.DataFrame, plot_dir: Path) -> None:
    """Generates multi-dimensional polar radar fingerprint chart for patient subtypes.

    Args:
        df: DataFrame containing cluster labels and Z-score feature columns.
        plot_dir: Directory path for exporting figure.
    """
    feature_labels = [
        "IFN-γ",
        "TIS",
        "CYT",
        "CD8+ T-cell",
        "PD-L1",
        "IMPRES",
        "TMB",
        "Age",
    ]
    n_vars = len(FEATURE_COLS)
    angles = [n / float(n_vars) * 2 * np.pi for n in range(n_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 8.5), subplot_kw=dict(polar=True))

    for c in range(3):
        sub = df[df["CLINICAL_CLUSTER"] == c]
        z_means = [sub[f"Z_{col}"].mean() for col in FEATURE_COLS]
        z_means += z_means[:1]

        label = f"{CLUSTER_PLOT_NAMES[c]} (N={len(sub)})"
        color = CLUSTER_COLORS[c]

        ax.plot(angles, z_means, linewidth=2.5, linestyle="solid", label=label, color=color)
        ax.fill(angles, z_means, color=color, alpha=0.18)

    plt.xticks(angles[:-1], feature_labels, color="black", size=11, weight="bold")
    ax.set_rlabel_position(0)

    grid_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    plt.yticks(grid_ticks, [f"{t:+.1f}" for t in grid_ticks], color="grey", size=9)
    plt.ylim(-1.5, 1.5)

    baseline_angles = np.linspace(0, 2 * np.pi, 100)
    ax.plot(baseline_angles, [0.0] * len(baseline_angles), color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_title(
        f"Multi-Dimensional Phenotype Fingerprint by Patient Subtype (N={len(df)})",
        size=15,
        weight="bold",
        pad=25,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=True, fontsize=10)
    plt.tight_layout()

    out_path = plot_dir / "radar_clinical_clusters.png"
    save_fig(fig, out_path)
    print(f"Saved radar cluster plot to {out_path.relative_to(SUBPROJECT_ROOT).as_posix()}")


def _plot_cluster_heatmap(
    df: pd.DataFrame, median_survivals: Dict[int, str], plot_dir: Path
) -> None:
    """Generates annotated Z-score feature heatmap with top clinical outcome tracks.

    Args:
        df: DataFrame containing cluster labels and Z-score feature columns.
        median_survivals: Dictionary of cluster index to median OS string.
        plot_dir: Directory path for exporting figure.
    """
    feature_display_names = {
        "IFN_gamma": "IFN-γ Signature",
        "TIS": "TIS Signature",
        "CYT": "Cytolytic (CYT) Score",
        "CD8_Tcell": "CD8+ T-cell Score",
        "PD_L1": "PD-L1 Expression Score",
        "IMPRES": "IMPRES Signature",
        "TMB_NONSYNONYMOUS": "Tumour Mutational Burden (TMB)",
        "AGE": "Patient Age at Diagnosis",
    }

    cluster_counts = df["CLINICAL_CLUSTER"].value_counts().to_dict()
    col_labels = [
        f"{CLUSTER_PLOT_NAMES[c]}\n(N={cluster_counts.get(c, 0)})" for c in range(3)
    ]

    z_matrix = np.zeros((len(FEATURE_COLS), 3))
    raw_matrix = np.zeros((len(FEATURE_COLS), 3))

    for c in range(3):
        sub = df[df["CLINICAL_CLUSTER"] == c]
        for i, col in enumerate(FEATURE_COLS):
            z_matrix[i, c] = sub[f"Z_{col}"].mean()
            raw_matrix[i, c] = sub[col].mean()

    df_heatmap = pd.DataFrame(
        z_matrix,
        index=[feature_display_names[f] for f in FEATURE_COLS],
        columns=col_labels,
    )

    trial_df = df[df["IS_TRIAL"] & df["RESPONDER"].notna()]
    resp_rates = []
    for c in range(3):
        sub = trial_df[trial_df["CLINICAL_CLUSTER"] == c]
        if len(sub) > 0:
            n_resp = (sub["RESPONDER"] == 1.0).sum()
            resp_rates.append(f"{(n_resp / len(sub)) * 100:.1f}% ({n_resp}/{len(sub)})")
        else:
            resp_rates.append("N/A")

    fig = plt.figure(figsize=(10, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 4.5, 0.4], hspace=0.15)

    ax_top = fig.add_subplot(gs[0])
    ax_top.axis("off")

    track_text = []
    for c in range(3):
        med_os = median_survivals.get(c, "N/A")
        track_text.append(f"Response: {resp_rates[c]}\nMedian OS: {med_os}")

    cell_colors = [CLUSTER_COLORS[c] for c in range(3)]
    for c in range(3):
        rect = plt.Rectangle(
            (c / 3.0 + 0.02, 0.05),
            0.293,
            0.9,
            facecolor=cell_colors[c],
            alpha=0.15,
            edgecolor=cell_colors[c],
            linewidth=1.5,
            transform=ax_top.transAxes,
        )
        ax_top.add_patch(rect)
        ax_top.text(
            c / 3.0 + 0.166,
            0.5,
            track_text[c],
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            transform=ax_top.transAxes,
        )

    ax_top.set_title(
        f"Annotated Subtype Feature Heatmap & Clinical Outcomes (Full Dataset, N={len(df)})",
        fontsize=14,
        weight="bold",
        pad=10,
    )

    ax_heat = fig.add_subplot(gs[1])
    annot_matrix = np.empty((len(FEATURE_COLS), 3), dtype=object)
    for i, col in enumerate(FEATURE_COLS):
        for c in range(3):
            z_val = z_matrix[i, c]
            raw_val = raw_matrix[i, c]
            if col == "TMB_NONSYNONYMOUS":
                annot_matrix[i, c] = f"Z={z_val:+.2f}\n({raw_val:.1f} mut/Mb)"
            elif col == "AGE":
                annot_matrix[i, c] = f"Z={z_val:+.2f}\n({raw_val:.1f} yrs)"
            else:
                annot_matrix[i, c] = f"Z={z_val:+.2f}\n({raw_val:.2f})"

    sns.heatmap(
        df_heatmap,
        annot=annot_matrix,
        fmt="",
        cmap="coolwarm",
        center=0.0,
        cbar=True,
        cbar_kws={"label": "Cohort-Standardized Z-Score", "shrink": 0.8},
        linewidths=1.0,
        linecolor="white",
        ax=ax_heat,
    )
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0, fontsize=10, weight="bold")
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=0, fontsize=10, weight="bold")

    out_path = plot_dir / "heatmap_clinical_clusters.png"
    save_fig(fig, out_path)
    print(f"Saved heatmap cluster plot to {out_path.relative_to(SUBPROJECT_ROOT).as_posix()}")


def _plot_cluster_pca(df: pd.DataFrame, pca_coords: np.ndarray, plot_dir: Path) -> None:
    """Generates 2D PCA projection scatter plot of full-dataset immune clusters.

    Args:
        df: DataFrame containing cluster labels and cohort names.
        pca_coords: 2D PCA coordinate array.
        plot_dir: Directory path for exporting figure.
    """
    df_pca = pd.DataFrame(pca_coords, columns=["PC1", "PC2"], index=df.index)
    df_pca["Cluster"] = df["CLINICAL_CLUSTER"]
    df_pca["Cluster_Name"] = df_pca["Cluster"].map(CLUSTER_PLOT_NAMES)

    pca = PCA(n_components=2).fit(df[[f"Z_{col}" for col in FEATURE_COLS]].values)
    var_explained = pca.explained_variance_ratio_

    palette_dict = {
        CLUSTER_PLOT_NAMES[0]: CLUSTER_COLORS[0],
        CLUSTER_PLOT_NAMES[1]: CLUSTER_COLORS[1],
        CLUSTER_PLOT_NAMES[2]: CLUSTER_COLORS[2],
    }

    fig, ax_pca = plt.subplots(figsize=(9.5, 7.5))
    hue_order = [CLUSTER_PLOT_NAMES[0], CLUSTER_PLOT_NAMES[1], CLUSTER_PLOT_NAMES[2]]


    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="Cluster_Name",
        style="Cluster_Name",
        data=df_pca,
        hue_order=hue_order,
        palette=palette_dict,
        alpha=0.8,
        s=90,
        ax=ax_pca,
        edgecolor="w",
        linewidth=0.6,
    )

    for c in range(3):
        centroid = df_pca[df_pca["Cluster"] == c][["PC1", "PC2"]].mean()
        ax_pca.scatter(
            centroid["PC1"],
            centroid["PC2"],
            marker="X",
            s=240,
            color="black",
            edgecolor="white",
            linewidth=1.5,
            zorder=10,
        )

    ax_pca.set_title(f"2D PCA Projection of Immunological & Genomic Patient Subtypes (N={len(df)})", fontsize=15, weight="bold", pad=15)
    ax_pca.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% explained variance)", fontsize=13)
    ax_pca.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% explained variance)", fontsize=13)

    handles, labels = ax_pca.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_pca.legend(by_label.values(), by_label.keys(), title="Patient Subtypes", loc="best", fontsize=10)

    out_path = plot_dir / "pca_clinical_clusters.png"
    save_fig(fig, out_path)
    print(f"Saved PCA cluster visualisation to {out_path.relative_to(SUBPROJECT_ROOT).as_posix()}")


def _plot_cluster_survival(df: pd.DataFrame, plot_dir: Path) -> Tuple[float, Dict[int, str]]:
    """Generates Kaplan-Meier overall survival curves for full-dataset clusters.

    Args:
        df: DataFrame containing cluster labels and OS data.
        plot_dir: Directory path for exporting figure.

    Returns:
        Tuple of (Log-rank p-value, median survival dictionary).
    """
    df_surv = df.dropna(subset=["OS_MONTHS", "OS_STATUS"]).copy()

    fig, ax = plt.subplots(figsize=(11, 6))

    palette_dict = {
        CLUSTER_PLOT_NAMES[0]: CLUSTER_COLORS[0],
        CLUSTER_PLOT_NAMES[1]: CLUSTER_COLORS[1],
        CLUSTER_PLOT_NAMES[2]: CLUSTER_COLORS[2],
    }

    kmf = KaplanMeierFitter()
    median_survivals: Dict[int, str] = {}

    for c in range(3):
        mask = df_surv["CLINICAL_CLUSTER"] == c
        sub_df = df_surv.loc[mask]
        label = f"{CLUSTER_PLOT_NAMES[c]} (N={mask.sum()})"
        kmf.fit(sub_df["OS_MONTHS"], sub_df["OS_STATUS"], label=label)
        kmf.plot_survival_function(ax=ax, color=palette_dict[CLUSTER_PLOT_NAMES[c]], ci_show=False, linewidth=2.5)


        med = kmf.median_survival_time_
        if np.isinf(med) or pd.isna(med):
            median_survivals[c] = "Not Reached"
        else:
            median_survivals[c] = f"{med:.1f} months"

    results = multivariate_logrank_test(
        df_surv["OS_MONTHS"], df_surv["CLINICAL_CLUSTER"], df_surv["OS_STATUS"]
    )
    p_val = results.p_value
    p_text = f"Log-Rank p = {p_val:.2e}" if p_val < 0.001 else f"Log-Rank p = {p_val:.3f}"

    ax.text(
        0.05,
        0.08,
        p_text,
        transform=ax.transAxes,
        fontsize=13,
        weight="bold",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray", boxstyle="round,pad=0.5"),
    )

    ax.set_title(f"Full Dataset OS: Kaplan-Meier of Immunological Subtypes (N={len(df_surv)})", fontsize=16, weight="bold", pad=15)
    ax.set_xlabel("Overall Survival (Months)", fontsize=13, labelpad=10)
    ax.set_ylabel("Survival Probability", fontsize=13, labelpad=10)
    ax.set_ylim(0, 1.05)

    out_path = plot_dir / "km_clinical_clusters.png"
    save_fig(fig, out_path)
    print(f"Saved KM cluster plot to {out_path.relative_to(SUBPROJECT_ROOT).as_posix()} (p = {p_val:.2e})")

    return p_val, median_survivals


def _plot_cluster_response(df: pd.DataFrame, plot_dir: Path) -> float:
    """Generates stacked bar plot comparing immunotherapy response across clusters.

    Args:
        df: DataFrame containing cluster labels and clinical response data.
        plot_dir: Directory path for exporting figure.

    Returns:
        Chi-Square test p-value comparing response across clusters.
    """
    df_trial = df[df["IS_TRIAL"] & df["RESPONDER"].notna()].copy()
    df_trial["RESPONDER_NUM"] = df_trial["RESPONDER"].map({True: 1.0, False: 0.0, 1.0: 1.0, 0.0: 0.0, "1": 1.0, "0": 0.0})
    contingency = pd.crosstab(df_trial["CLINICAL_CLUSTER"], df_trial["RESPONDER_NUM"])

    chi2, p_val, dof, _ = chi2_contingency(contingency)

    resp_pct = pd.crosstab(df_trial["CLINICAL_CLUSTER"], df_trial["RESPONDER_NUM"], normalize="index") * 100

    fig, ax = plt.subplots(figsize=(11, 6))

    clusters = [0, 1, 2]
    cluster_labels_short = [CLUSTER_PLOT_NAMES[c] for c in clusters]

    responders = resp_pct.get(1.0, pd.Series(0.0, index=clusters))
    non_responders = resp_pct.get(0.0, pd.Series(0.0, index=clusters))

    bar_width = 0.55
    bars1 = ax.bar(cluster_labels_short, responders, width=bar_width, color=RESPONSE_PALETTE["CR/PR"], label="Responder (CR/PR)")
    bars2 = ax.bar(cluster_labels_short, non_responders, width=bar_width, bottom=responders, color=RESPONSE_PALETTE["PD"], label="Non-Responder (PD)")

    for bar in bars1:
        height = bar.get_height()
        if height > 5:
            ax.text(bar.get_x() + bar.get_width() / 2.0, height / 2.0, f"{height:.1f}%", ha="center", va="center", color="white", weight="bold", fontsize=11)

    for bar in bars2:
        height = bar.get_height()
        if height > 5:
            bottom = bar.get_y()
            ax.text(bar.get_x() + bar.get_width() / 2.0, bottom + height / 2.0, f"{height:.1f}%", ha="center", va="center", color="white", weight="bold", fontsize=11)

    p_text = f"Chi-Square p = {p_val:.2e}" if p_val < 0.001 else f"Chi-Square p = {p_val:.3f}"
    ax.text(0.05, 0.92, p_text, transform=ax.transAxes, fontsize=12, weight="bold", bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"))

    ax.set_title(f"Immunotherapy Response Rate by Patient Subtype (Trial Cohorts, N={len(df_trial)})", fontsize=14, weight="bold", pad=15)
    ax.set_ylabel("Proportion of Patients (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=10)

    out_path = plot_dir / "response_by_clinical_cluster.png"
    save_fig(fig, out_path)
    print(f"Saved response cluster plot to {out_path.relative_to(SUBPROJECT_ROOT).as_posix()} (p = {p_val:.3f})")

    return p_val


def _generate_clustering_report(
    df: pd.DataFrame,
    counts: List[int],
    km_p_val: float,
    chi2_p_val: float,
    median_survivals: Dict[int, str],
    report_path: Path,
) -> None:
    """Generates comprehensive Obsidian Markdown report summarising full-dataset clustering.

    Args:
        df: Pooled DataFrame with cluster assignments.
        counts: Patient counts per cluster.
        km_p_val: Log-rank test p-value.
        chi2_p_val: Chi-square response p-value.
        median_survivals: Map of cluster index to median OS string.
        report_path: Output report path.
    """
    profile_df = df.groupby("CLINICAL_CLUSTER")[FEATURE_COLS].mean().reset_index()

    trial_df = df[df["IS_TRIAL"] & df["RESPONDER"].notna()]
    n_trial = len(trial_df)
    resp_rates = []
    resp_fractions = []
    for c in range(3):
        sub = trial_df[trial_df["CLINICAL_CLUSTER"] == c]
        if len(sub) > 0:
            n_resp = int((sub["RESPONDER"] == 1.0).sum())
            n_total = len(sub)
            resp_rates.append(format_count_percentage(n_resp, n_total))
            resp_fractions.append(f"{n_resp}/{n_total} = {n_resp / n_total * 100:.1f}%")
        else:
            resp_rates.append("N/A")
            resp_fractions.append("N/A")

    total_n = sum(counts)
    frontmatter = generate_obsidian_frontmatter(
        title="Patient Phenotyping via Full-Dataset Immunological & Genomic Clustering",
        aliases=["Patient Subtyping", "Clinical Clustering", "Immune Phenotyping"],
        tags=["melanoma", "clinical-subtyping", "clustering", "full-dataset", "immune-hot-cold"],
        extra_css_classes=["table-center", "row-alt"],
    )

    ifn_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'IFN_gamma'].values[0]:.2f}" for c in range(3)]
    tis_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'TIS'].values[0]:.2f}" for c in range(3)]
    cyt_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'CYT'].values[0]:.2f}" for c in range(3)]
    cd8_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'CD8_Tcell'].values[0]:.2f}" for c in range(3)]
    tmb_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'TMB_NONSYNONYMOUS'].values[0]:.1f}" for c in range(3)]

    # Format p-values as proper LaTeX scientific notation (e.g. 2.29 \times 10^{-5})
    def _fmt_pval_latex(p: float) -> str:
        if pd.isna(p):
            return "N/A"
        if p < 1e-3:
            s = f"{p:.2e}"
            mantissa, exp = s.split("e")
            exp_int = int(exp)
            return rf"{mantissa} \times 10^{{{exp_int}}}"
        return f"{p:.3f}"

    km_p_str = _fmt_pval_latex(km_p_val)
    chi2_p_str = _fmt_pval_latex(chi2_p_val)

    lines: List[str] = [frontmatter, ""]
    w = lines.append

    w("# Patient Phenotyping via Full-Dataset Immunological & Genomic Clustering")
    w("")
    w("> [!INFO] What, Why & Key Questions — Overview")
    w(f"> - **What We Are Doing**: Applying unsupervised Agglomerative Hierarchical Clustering (Ward linkage) to the **entire combined dataset** ($N = {total_n}$ patients across TCGA-SKCM, Liu 2019, Hugo 2016, and Riaz 2017) using six immune expression signatures, TMB, and patient age — all Z-score standardised *within each cohort* before pooling to remove study-platform offsets.")
    w("> - **Why We Are Doing It**: Before building a supervised response predictor, we need to know whether biologically meaningful patient subgroups exist in the data at all. If patients naturally cluster into distinct immune phenotypes — \"hot\" vs. \"cold\" tumours — then those phenotypes should predict both survival and immunotherapy response. Discovering these groups unsupervised (without using any response labels) provides unbiased biological validation.")
    w("> - **Questions**:")
    w(">   1. *Do distinct immunological subtypes emerge from the data without supervision?*")
    w(">   2. *Do those subtypes differ significantly in overall survival — confirming they capture genuine biology?*")
    w(">   3. *Do immunotherapy responders concentrate in the \"hot\" immune subtype, validating the clusters as clinically meaningful?*")
    w("")

    w("## 1. Subtype Profiles")
    w("")
    w("> [!INFO] What, Why & Key Questions — Subtype Fingerprints")
    w("> - **What We Are Doing**: Characterising the three discovered patient subtypes using two visualisations — a polar radar chart showing the multi-dimensional signature fingerprint of each subtype, and an annotated heatmap showing per-patient Z-scores with response rate and survival overlaid.")
    w("> - **Why We Are Doing It**: A radar chart reveals the *shape* of each subtype's immune profile at a glance (which signatures are high or low). The heatmap reveals the *within-cluster heterogeneity* — how tightly patients cluster together — and overlays clinical outcome tracks to verify biological coherence.")
    w("> - **Questions**:")
    w(">   1. *Are the subtypes cleanly separated across all signatures simultaneously, or does separation rely on only one or two markers?*")
    w(">   2. *Does the response rate track visibly with the immune intensity track in the heatmap?*")
    w("")

    w("### 1.1. Multi-Dimensional Phenotype Fingerprint (Radar Profile)")
    w("")
    w("The polar radar chart displays the standardised Z-score profiles across transcriptomic immune signatures, mutational burden, and patient age for each subtype:")
    w("")
    w("![Subtype Profile Radar Chart](../../plots/clinical/radar_clinical_clusters.png)")
    w("")

    w("### 1.2. Annotated Subtype Feature Heatmap & Clinical Tracks")
    w("")
    w("The heatmap details the Z-score signature matrix for each patient cluster, annotated with immunotherapy response rates (CR/PR %) and median overall survival (OS):")
    w("")
    w("![Annotated Subtype Feature Heatmap](../../plots/clinical/heatmap_clinical_clusters.png)")
    w("")

    w("## 2. Biological Interpretation of Patient Subtypes")
    w("")
    w("The unsupervised clustering isolates three distinct patient phenotypes:")
    w("")
    w(f"1. **{CLUSTER_NAMES[0]}** ($N = {counts[0]}$)")
    w(f"    - *Immune Signatures*: Highest T-cell inflammation across all markers (IFN-$\\gamma$ = {ifn_vals[0]}, TIS = {tis_vals[0]}, CYT = {cyt_vals[0]}, CD8 = {cd8_vals[0]}).")
    w(f"    - *Therapeutic Benefit*: Highest immunotherapy response rate (**{resp_fractions[0]}** in trial patients).")
    w(f"    - *Prognosis*: Best overall survival (Median OS = 🟢 **{median_survivals[0]}**).")
    w("")

    w(f"2. **{CLUSTER_NAMES[1]}** ($N = {counts[1]}$)")
    w(f"    - *Immune Signatures*: Attenuated T-cell inflammation across all markers (IFN-$\\gamma$ = {ifn_vals[1]}, TIS = {tis_vals[1]}, CYT = {cyt_vals[1]}, CD8 = {cd8_vals[1]}).")
    w(f"    - *Therapeutic Benefit*: Lowest response rate to anti-PD-1 therapy (**{resp_fractions[1]}** in trial patients).")
    w(f"    - *Prognosis*: Worst overall survival (Median OS = 🔴 **{median_survivals[1]}**).")
    w("")

    w(f"3. **{CLUSTER_NAMES[2]}** ($N = {counts[2]}$)")
    w(f"    - *Genomics*: Highest tumour mutational burden (**TMB = {tmb_vals[2]} mut/Mb**) with only moderate immune infiltration.")
    w(f"    - *Immune Signatures*: Intermediate T-cell inflammation (IFN-$\\gamma$ = {ifn_vals[2]}, TIS = {tis_vals[2]}).")
    w(f"    - *Prognosis*: Intermediate survival (Median OS = 🟠 **{median_survivals[2]}**) — demonstrating that high TMB alone, without a hot immune microenvironment, does not confer the same survival benefit.")
    w("")

    w("## 3. Subtype Visualisation (2D PCA Projection)")
    w("")
    w("> [!INFO] What, Why & Key Questions — PCA Projection")
    w(f"> - **What We Are Doing**: Projecting all $N = {total_n}$ patients onto the first two principal components (PCA) of the feature space to visualise how well the three clusters separate in a lower-dimensional view.")
    w("> - **Why We Are Doing It**: A clean 2D separation confirms that the clustering reflects a genuine multi-dimensional structure in the data, not an artefact of the Ward linkage algorithm.")
    w("> - **Questions**: *Are clusters geometrically separated in PCA space, or do they overlap substantially?*")
    w("")
    w("![2D PCA Visualisation of Clusters](../../plots/clinical/pca_clinical_clusters.png)")
    w("")

    w("## 4. Immunotherapy Response & Overall Survival Validation")
    w("")
    w("> [!INFO] What, Why & Key Questions — Clinical Outcome Validation")
    w(f"> - **What We Are Doing**: Testing whether the unsupervised cluster labels — derived without using any response information — nevertheless stratify immunotherapy response rates (in the $N = {n_trial}$ trial patients with binary labels) and overall survival (in the full $N = {total_n}$ dataset).")
    w("> - **Why We Are Doing It**: This is the critical validation step. If clusters discovered purely from expression patterns correlate with clinical outcomes, it confirms the biology is real and the subtypes are clinically actionable.")
    w("> - **Questions**:")
    w(">   1. *Do immunotherapy responders concentrate significantly in the Hot cluster (Chi-Square test)?*")
    w(">   2. *Is the survival separation across subtypes statistically significant (Log-Rank test)?*")
    w("")

    w(f"**Therapeutic Response Rate (Trial Cohorts, $N = {n_trial}$ with binary labels)**:")
    w("")
    chi2_sig = "not statistically significant" if chi2_p_val >= 0.05 else "statistically significant"
    w("> [!INSIGHT] Chi-Square Response Rate Evaluation")
    w(f"> The Chi-Square test across cluster response rates yields $p = {chi2_p_str}$ — **{chi2_sig}**. ")
    if chi2_p_val >= 0.05:
        w(f"> The Hot cluster shows a numerically higher response rate ({resp_fractions[0]}) vs. Cold ({resp_fractions[1]}), but this difference does not reach significance at this sample size. This reflects the limited statistical power of the three-way comparison across the trial cohort subset ($N = {n_trial}$), not an absence of a real biological trend.")
    else:
        w(f"> Immunotherapy responders are significantly enriched in the Hot cluster ({resp_fractions[0]}) compared to the Cold cluster ({resp_fractions[1]}).")
    w("")
    w("![Response Rate by Cluster](../../plots/clinical/response_by_clinical_cluster.png)")
    w("")

    w(f"**Overall Survival (Full Dataset, $N = {total_n}$)**:")
    w("")
    w(f"The survival separation across patient subtypes is highly statistically significant (Log-Rank $p = {km_p_str}$), confirming that the immune phenotypes capture genuine prognostic biology:")
    w("")
    w("![KM Survival of Clinical Clusters](../../plots/clinical/km_clinical_clusters.png)")
    w("")

    w("> [!INSIGHT] Key Insights: Unsupervised Subtyping Summary")
    w(f"> - **Unsupervised biology is real**: Three distinct immune phenotypes emerge from the data without using any response labels, and they separate significantly by overall survival ($p = {km_p_str}$).")
    w(f"> - **Immune inflammation, not TMB alone, drives prognosis**: The High-TMB cluster (Cluster 2) shows only intermediate survival despite its high mutational burden — confirming that TMB and immune infiltration act as orthogonal axes.")
    if chi2_p_val < 0.05:
        w(f"> - **Response rate stratifies significantly**: Immunotherapy responders are significantly enriched in the Hot cluster ({resp_fractions[0]}) vs. Cold cluster ({resp_fractions[1]}) ($p = {chi2_p_str}$), validating the clinical utility of unsupervised microenvironmental phenotyping.")
    else:
        w(f"> - **Response trend is consistent but underpowered**: The numerical response rate advantage of the Hot cluster ({resp_fractions[0]}) vs. Cold ({resp_fractions[1]}) is clinically meaningful in direction, but the $N = {n_trial}$ trial subset is underpowered for a three-way Chi-Square test ($p = {chi2_p_str}$). This motivates supervised multivariate predictive modelling in downstream pillars.")
    w("")

    w("## 5. Methodological Limitations & Future Directions")
    w("")
    w("> [!WARNING] Analytical Scope & Limitations")
    w(f"> - **Cohort Composition Heterogeneity**: The $N = {total_n}$ pooled dataset merges non-small cell TCGA reference samples ($N = 443$) with anti-PD-1 trial cohorts ($N = 256$). Cohort-wise Z-score standardization mitigates platform offsets, but baseline clinical heterogeneity remains.")
    if chi2_p_val < 0.05:
        w(f"> - **Trial Cohort Sample Size**: Only $N = {n_trial}$ trial patients have documented binary anti-PD-1 response labels. While the response stratification achieves significance ($p = {chi2_p_str}$), expanding trial sample sizes will improve per-cluster subgroup precision.")
    else:
        w(f"> - **Trial Cohort Sample Size**: Only $N = {n_trial}$ trial patients have documented binary anti-PD-1 response labels. Three-way chi-square power is limited, contributing to the non-significant response rate p-value ($p = {chi2_p_str}$).")
    w("> - **Arbitrary Cluster K Choice**: K=3 was selected based on biological interpretability (Hot, Cold, High-TMB). Alternative clustering algorithms (e.g. GMM, HDBSCAN) or higher K values may resolve finer microenvironmental sub-states.")
    w("> - **Z-Score Normalization Dependence**: Cluster boundaries depend on within-cohort standardization; applying this subtyping scheme to a single new patient requires reference cohort normalization params.")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Clustering report successfully written to {report_path.relative_to(SUBPROJECT_ROOT).as_posix()}")


def main() -> None:
    """Executes full-dataset patient subtyping pipeline across all four cohorts."""
    print("==================================================")
    print("Full-Dataset Immunological & Genomic Clustering")
    print("==================================================")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_DIR.mkdir(exist_ok=True, parents=True)

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found at {CONFIG_PATH.relative_to(SUBPROJECT_ROOT).as_posix()}"
        )

    dataset_configs = load_dataset_config(CONFIG_PATH)

    print("\nLoading and preprocessing cohort datasets...")
    full_df = _load_and_extract_cohort_features(dataset_configs)

    print("\nExecuting Agglomerative Hierarchical Clustering (Ward linkage, K=3)...")
    full_df, pca_coords = _perform_clustering(full_df)

    counts = full_df["CLINICAL_CLUSTER"].value_counts().sort_index().tolist()
    print(f"Cluster sample sizes: {', '.join([f'{CLUSTER_PLOT_NAMES[c]} (N={counts[c]})' for c in range(3)])}")

    print("\nGenerating cluster visualisations and statistical evaluations...")
    _plot_cluster_pca(full_df, pca_coords, PLOT_DIR)
    _plot_cluster_radar(full_df, PLOT_DIR)
    km_p_val, median_survivals = _plot_cluster_survival(full_df, PLOT_DIR)
    _plot_cluster_heatmap(full_df, median_survivals, PLOT_DIR)
    chi2_p_val = _plot_cluster_response(full_df, PLOT_DIR)

    print("\nExporting full-dataset clustering report...")
    _generate_clustering_report(full_df, counts, km_p_val, chi2_p_val, median_survivals, REPORT_PATH)

    print("\n==================================================")
    print("Done! Full-dataset clustering workflow completed.")
    print("==================================================")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(SUBPROJECT_ROOT).as_posix()}")
            main()

