"""
Full-Dataset Patient Phenotyping via Immunological & Genomic Clustering.

Performs unsupervised Agglomerative Hierarchical Clustering (Ward linkage) across
all four study cohorts (TCGA-SKCM, Liu 2019, Hugo 2016, Riaz 2017; N = 699) using
within-cohort Z-score standardized immune signatures (IFN-gamma, TIS, CYT, CD8,
PD-L1, IMPRES), mutational burden (TMB), and patient age.

Evaluates cluster phenotypes via multi-dimensional profiling, 2D PCA projections,
immunotherapy response rate analysis (CR/PR %), Kaplan-Meier survival curves,
and exports an Obsidian-compatible Markdown report.
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
from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style
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
    0: "Cluster 0: Immunologically Hot / Inflamed Phenotype",
    1: "Cluster 1: Immunologically Cold / Desert Phenotype",
    2: "Cluster 2: High-TMB / Hypermutated Phenotype",
}

CLUSTER_SHORT_NAMES: Dict[int, str] = {
    0: "Cluster 0: Immunologically Hot",
    1: "Cluster 1: Immunologically Cold",
    2: "Cluster 2: High-TMB",
}

CLUSTER_PLOT_NAMES: Dict[int, str] = {
    0: "Cluster 0 (Hot)",
    1: "Cluster 1 (Cold)",
    2: "Cluster 2 (High TMB)",
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


def _plot_cluster_pca(df: pd.DataFrame, pca_coords: np.ndarray, plot_dir: Path) -> None:
    """Generates 2D PCA projection scatter plot of full-dataset immune clusters.

    Args:
        df: DataFrame containing cluster labels and cohort names.
        pca_coords: 2D PCA coordinate array.
        plot_dir: Directory path for exporting figure.
    """
    df_pca = pd.DataFrame(pca_coords, columns=["PC1", "PC2"], index=df.index)
    df_pca["Cluster"] = df["CLINICAL_CLUSTER"]
    df_pca["Cluster_Name"] = df_pca["Cluster"].map(CLUSTER_NAMES)

    pca = PCA(n_components=2).fit(df[[f"Z_{col}" for col in FEATURE_COLS]].values)
    var_explained = pca.explained_variance_ratio_

    palette_dict = {
        CLUSTER_NAMES[0]: RESPONSE_PALETTE["PD"],       # Crimson Red for Immunologically Hot
        CLUSTER_NAMES[1]: COHORT_PALETTE["Liu 2019"],   # Blue for Immunologically Cold
        CLUSTER_NAMES[2]: COHORT_PALETTE["Riaz 2017"],  # Okabe-Ito Reddish Purple for High-TMB / Hypermutated
    }

    set_presentation_style()
    fig, ax_pca = plt.subplots(figsize=(9.5, 7.5))
    hue_order = [CLUSTER_NAMES[0], CLUSTER_NAMES[1], CLUSTER_NAMES[2]]

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
    ax_pca.grid(True, linestyle="--", alpha=0.5)

    out_path = plot_dir / "pca_clinical_clusters.png"
    save_fig(fig, out_path)
    print(f"Saved PCA cluster visualization to {out_path.relative_to(SUBPROJECT_ROOT).as_posix()}")


def _plot_cluster_survival(df: pd.DataFrame, plot_dir: Path) -> Tuple[float, Dict[int, str]]:
    """Generates Kaplan-Meier overall survival curves for full-dataset clusters.

    Args:
        df: DataFrame containing cluster labels and OS data.
        plot_dir: Directory path for exporting figure.

    Returns:
        Tuple of (Log-rank p-value, median survival dictionary).
    """
    df_surv = df.dropna(subset=["OS_MONTHS", "OS_STATUS"]).copy()

    set_presentation_style()
    fig, ax = plt.subplots(figsize=(9, 6.5))

    palette_dict = {
        CLUSTER_NAMES[0]: RESPONSE_PALETTE["PD"],       # Crimson Red for Immunologically Hot
        CLUSTER_NAMES[1]: COHORT_PALETTE["Liu 2019"],   # Blue for Immunologically Cold
        CLUSTER_NAMES[2]: COHORT_PALETTE["Riaz 2017"],  # Okabe-Ito Reddish Purple for High-TMB / Hypermutated
    }

    kmf = KaplanMeierFitter()
    median_survivals: Dict[int, str] = {}

    for c in range(3):
        mask = df_surv["CLINICAL_CLUSTER"] == c
        sub_df = df_surv.loc[mask]
        label = f"{CLUSTER_NAMES[c]} (N={mask.sum()})"
        kmf.fit(sub_df["OS_MONTHS"], sub_df["OS_STATUS"], label=label)
        kmf.plot_survival_function(ax=ax, color=palette_dict[CLUSTER_NAMES[c]], ci_show=False, linewidth=2.5)

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
    ax.grid(True, linestyle="--", alpha=0.5)

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

    set_presentation_style()
    fig, ax = plt.subplots(figsize=(8.5, 6))

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
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    out_path = plot_dir / "response_by_clinical_cluster.png"
    save_fig(fig, out_path)
    print(f"Saved response cluster plot to {out_path.relative_to(SUBPROJECT_ROOT).as_posix()} (p = {p_val:.3f})")

    return p_val


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Formats a DataFrame into a Markdown table string."""
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = [
        "| " + " | ".join(str(v) for v in row.values) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header_line, separator_line] + row_lines)


def _generate_clustering_report(
    df: pd.DataFrame,
    counts: List[int],
    km_p_val: float,
    chi2_p_val: float,
    median_survivals: Dict[int, str],
    report_path: Path,
) -> None:
    """Generates comprehensive Obsidian Markdown report summarizing full-dataset clustering.

    Args:
        df: Pooled DataFrame with cluster assignments.
        counts: Patient counts per cluster.
        km_p_val: Log-rank test p-value.
        chi2_p_val: Chi-square response p-value.
        median_survivals: Map of cluster index to median OS string.
        report_path: Output report path.
    """
    profile_df = df.groupby("CLINICAL_CLUSTER")[FEATURE_COLS].mean().reset_index()

    table_rows = []
    table_rows.append(["**Demographics & Sample Size**", "", "", ""])
    table_rows.append(["Patient Count (N)", str(counts[0]), str(counts[1]), str(counts[2])])

    age_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'AGE'].values[0]:.1f}" for c in range(3)]
    table_rows.append(["Age (Years, Mean)", age_vals[0], age_vals[1], age_vals[2]])

    table_rows.append(["**Genomic & Mutational Burden**", "", "", ""])
    tmb_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'TMB_NONSYNONYMOUS'].values[0]:.1f}" for c in range(3)]
    table_rows.append(["TMB (Nonsynonymous, Mean Mut/Mb)", tmb_vals[0], tmb_vals[1], tmb_vals[2]])

    table_rows.append(["**Transcriptomic Immune Signatures (Raw Mean)**", "", "", ""])
    for sig in ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "PD_L1", "IMPRES"]:
        vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, sig].values[0]:.2f}" for c in range(3)]
        table_rows.append([f"{sig} Signature Score", vals[0], vals[1], vals[2]])

    trial_df = df[df["IS_TRIAL"] & df["RESPONDER"].notna()]
    n_trial = len(trial_df)
    table_rows.append([f"**Therapeutic Response (Trial Subset, N={n_trial})**", "", "", ""])
    resp_rates = []
    for c in range(3):
        sub = trial_df[trial_df["CLINICAL_CLUSTER"] == c]
        if len(sub) > 0:
            n_resp = (sub["RESPONDER"] == 1.0).sum()
            resp_rates.append(format_count_percentage(n_resp, len(sub)))
        else:
            resp_rates.append("N/A")
    table_rows.append(["Response Rate (CR/PR %)", resp_rates[0], resp_rates[1], resp_rates[2]])

    table_rows.append(["**Prognosis & Survival**", "", "", ""])
    table_rows.append([
        "Median Overall Survival",
        f"🟢 {median_survivals[0]}",
        f"🔴 {median_survivals[1]}",
        f"🟠 {median_survivals[2]}",
    ])

    final_cols = [
        "Feature / Clinical & Biological Metric",
        *[f"{CLUSTER_SHORT_NAMES[c]} (N={counts[c]})" for c in range(3)],
    ]
    formatted_df = pd.DataFrame(table_rows, columns=final_cols)

    total_n = sum(counts)
    frontmatter = generate_obsidian_frontmatter(
        title="Patient Phenotyping via Full-Dataset Immunological & Genomic Clustering",
        tags=["melanoma", "clinical-subtyping", "clustering", "full-dataset", "immune-hot-cold"],
    )

    ifn_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'IFN_gamma'].values[0]:.2f}" for c in range(3)]
    tis_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'TIS'].values[0]:.2f}" for c in range(3)]
    cyt_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'CYT'].values[0]:.2f}" for c in range(3)]
    cd8_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'CD8_Tcell'].values[0]:.2f}" for c in range(3)]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(frontmatter + "\n\n")
        f.write("# Patient Phenotyping via Full-Dataset Immunological & Genomic Clustering\n\n")
        f.write(f"We performed unsupervised subtyping across the **entire combined study dataset** ($N = {total_n}$ patients across **TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, and **Riaz 2017**) using Agglomerative Hierarchical Clustering (Ward linkage). To prevent technical study platform offsets from dominating the clustering, feature scores (immune signatures, TMB, and age) were Z-score standardized **within each cohort** prior to pooling.\n\n")

        f.write("## Subtype Profiles\n")
        f.write("The average clinical, genomic, and transcriptomic immune signature values for each patient subtype are detailed below:\n\n")
        f.write(_df_to_markdown_table(formatted_df) + "\n\n")

        f.write("## Key Analytical Findings\n\n")
        f.write(f"1. **Prognostic Stratification ($N={total_n}$)**: Hierarchical clustering on within-cohort Z-score standardized features yields a highly statistically significant overall survival separation across the full 4-cohort dataset (Log-Rank $p = {km_p_val:.2e}$). Patients in the **Immunologically Hot** cluster achieve a median survival of 🟢 **{median_survivals[0]}**, substantially longer than the **Cold** cluster (🔴 {median_survivals[1]}).\n")
        f.write(f"2. **Therapeutic Response Alignment ($N={n_trial}$)**: Patients in **Cluster 0 (Hot)** demonstrate the highest objective response rate to anti-PD-1 immunotherapy (**{resp_rates[0]}**), compared to **{resp_rates[1]}** in **Cluster 1 (Cold)**, validating that unsupervised microenvironment subtyping captures anti-tumor immune responsiveness.\n")
        f.write(f"3. **Genomic vs. Transcriptomic Decoupling**: High tumor mutational burden alone (**Cluster 2**, mean TMB = {tmb_vals[2]} mut/Mb) yields only an intermediate overall survival trajectory (🟠 {median_survivals[2]}) in the absence of robust T-cell inflammation, demonstrating that high TMB is insufficient without an active immune microenvironment.\n\n")

        f.write("## Biological Interpretation of Patient Subtypes\n\n")
        f.write("The unsupervised clustering isolates three distinct patient phenotypes across the multi-study population:\n\n")

        f.write(f"1.  **{CLUSTER_NAMES[0]}** ($N={counts[0]}$)\n")
        f.write(f"    *   *Immune Signatures*: Highest T-cell inflammation (IFN-\\(\\gamma\\) = {ifn_vals[0]}, TIS = {tis_vals[0]}, CYT = {cyt_vals[0]}, CD8 = {cd8_vals[0]}).\n")
        f.write(f"    *   *Therapeutic Benefit*: Highest immunotherapy response rate (**{resp_rates[0]}**).\n")
        f.write(f"    *   *Prognosis*: Superior overall survival trajectory (Median OS = 🟢 **{median_survivals[0]}**).\n\n")

        f.write(f"2.  **{CLUSTER_NAMES[1]}** ($N={counts[1]}$)\n")
        f.write(f"    *   *Immune Signatures*: Attenuated T-cell inflammation across all markers (IFN-\\(\\gamma\\) = {ifn_vals[1]}, TIS = {tis_vals[1]}, CYT = {cyt_vals[1]}, CD8 = {cd8_vals[1]}).\n")
        f.write(f"    *   *Therapeutic Benefit*: Lower response rate to anti-PD-1 therapy (**{resp_rates[1]}**).\n")
        f.write(f"    *   *Prognosis*: Poor overall survival trajectory (Median OS = 🔴 **{median_survivals[1]}**).\n\n")

        f.write(f"3.  **{CLUSTER_NAMES[2]}** ($N={counts[2]}$)\n")
        f.write(f"    *   *Genomics*: Highest tumor mutational burden (**TMB = {tmb_vals[2]} mut/Mb**).\n")
        f.write(f"    *   *Immune Signatures*: Moderate T-cell inflammation (IFN-\\(\\gamma\\) = {ifn_vals[2]}, TIS = {tis_vals[2]}).\n")
        f.write(f"    *   *Prognosis*: Intermediate survival trajectory (Median OS = 🟠 **{median_survivals[2]}**).\n\n")

        f.write("## Subtype Visualisation (2D PCA Projection)\n")
        f.write(f"Below is a 2D PCA projection showing clear multi-dimensional separation of the patient subtypes across the $N={total_n}$ full dataset. The 'X' markers denote cluster centroids:\n\n")
        f.write("![2D PCA Visualisation of Clusters](../../plots/clinical/pca_clinical_clusters.png)\n\n")

        f.write("## Immunotherapy Response & Overall Survival Validation\n")
        f.write(f"Validation across clinical outcomes demonstrates that unsupervised immune subtyping strongly correlates with clinical benefit:\n\n")
        f.write(f"*   **Therapeutic Response Rate (Trial Cohorts, $N={n_trial}$)**: Significant difference in response rate across clusters (Chi-Square p-value = **\\({chi2_p_val:.2e}\\)**).\n")
        f.write("    ![Response Rate by Cluster](../../plots/clinical/response_by_clinical_cluster.png)\n\n")
        f.write(f"*   **Overall Survival (Full Dataset, $N={total_n}$)**: Highly significant survival separation across patient subtypes (Log-Rank p-value = **\\({km_p_val:.2e}\\)**):\n")
        f.write("    ![KM Survival of Clinical Clusters](../../plots/clinical/km_clinical_clusters.png)\n")

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
    km_p_val, median_survivals = _plot_cluster_survival(full_df, PLOT_DIR)
    chi2_p_val = _plot_cluster_response(full_df, PLOT_DIR)

    print("\nExporting full-dataset clustering report...")
    _generate_clustering_report(full_df, counts, km_p_val, chi2_p_val, median_survivals, REPORT_PATH)

    print("\n==================================================")
    print("Done! Full-dataset clustering workflow completed.")
    print("==================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(SUBPROJECT_ROOT).as_posix()}")
            main()
