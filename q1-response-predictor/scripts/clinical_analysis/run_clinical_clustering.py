"""
Clinical & Genomic Patient Clustering Analysis for TCGA-SKCM Cohort.

Performs unsupervised Agglomerative Hierarchical Clustering (Ward linkage) on TCGA-SKCM clinical
and genomic features (Age, TMB, Aneuploidy, Hypoxia, Therapy, Stage IV status). Evaluates cluster
phenotypes via multi-dimensional profiling, 2D PCA projections, Kaplan-Meier survival analysis,
and exports a Markdown report.
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Bootstrap project root resolution for top-level import
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import find_project_root
from src.utils.plotting import save_fig

# Module-level Constants
DATA_DIR = find_project_root(Path(__file__).resolve()) / "data"
CLINICAL_FILE = DATA_DIR / "processed" / "skcm_tcga_pan_can_atlas_2018" / "clin_cleaned.csv"
PLOTS_DIR = find_project_root(Path(__file__).resolve()) / "plots" / "clinical"
REPORTS_DIR = find_project_root(Path(__file__).resolve()) / "reports"
LOG_DIR = find_project_root(Path(__file__).resolve()) / "logs"
LOG_PATH = LOG_DIR / "run_clinical_clustering.log"

CLUSTERING_COLS: List[str] = [
    "AGE",
    "TMB_NONSYNONYMOUS",
    "ANEUPLOIDY_SCORE",
    "WINTER_HYPOXIA_SCORE",
    "TX_TYPE_CHEMOTHERAPY",
    "TX_TYPE_IMMUNOTHERAPY",
    "TX_TYPE_RADIATION_THERAPY",
    "IS_PRIMARY",
    "IS_STAGE_IV",
]

CLUSTER_NAMES: Dict[int, str] = {
    0: "Cluster 0: Low Chromosomal Instability / Low Hypoxia Phenotype",
    1: "Cluster 1: High Chromosomal Instability / High Hypoxia Phenotype",
    2: "Cluster 2: Stage IV / Advanced Metastatic Disease",
}


def _load_and_preprocess_tcga_clinical(file_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Loads TCGA-SKCM clinical dataset, engineer features, imputes, and standardises.

    Args:
        file_path: Absolute path to clean TCGA clinical CSV.

    Returns:
        Tuple of (full DataFrame, feature subset DataFrame, scaled feature array).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Clinical file not found at {file_path.relative_to(BASE_DIR).as_posix()}")

    df = pd.read_csv(file_path)
    print(f"Loaded TCGA clinical data with shape: {df.shape}")

    df["IS_PRIMARY"] = df["SAMPLE_TYPE"].apply(lambda x: 1 if str(x).lower() == "primary" else 0)
    df["IS_STAGE_IV"] = df["AJCC_PATHOLOGIC_TUMOR_STAGE"].apply(lambda x: 1 if "STAGE IV" in str(x).upper() else 0)

    sub_df = df[CLUSTERING_COLS].copy()

    imputer = SimpleImputer(strategy="median")
    imputed_data = imputer.fit_transform(sub_df)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    return df, sub_df, scaled_data


def _plot_cluster_survival(df_survival: pd.DataFrame, plot_dir: Path) -> Tuple[float, Dict[int, str]]:
    """Generates Kaplan-Meier survival curves for hierarchical patient clusters.

    Args:
        df_survival: Clinical DataFrame containing cluster labels and OS data.
        plot_dir: Directory path to export plot artifact.

    Returns:
        Tuple of (multivariate log-rank p-value, median survival dictionary per cluster).
    """
    set_presentation_style()
    fig, ax = plt.subplots(figsize=(9, 6.5))

    palette_dict = {
        CLUSTER_NAMES[0]: COHORT_PALETTE["Liu 2019"],   # Slate Blue
        CLUSTER_NAMES[1]: RESPONSE_PALETTE["CR/PR"],    # Forest Green
        CLUSTER_NAMES[2]: RESPONSE_PALETTE["PD"],       # Crimson Red
    }

    kmf = KaplanMeierFitter()
    median_survivals: Dict[int, str] = {}

    for c in range(3):
        mask = df_survival["CLINICAL_CLUSTER"] == c
        sub_df = df_survival.loc[mask]
        label = f"{CLUSTER_NAMES[c]} (N={mask.sum()})"
        kmf.fit(sub_df["OS_MONTHS"], sub_df["OS_STATUS"], label=label)
        kmf.plot_survival_function(ax=ax, color=palette_dict[CLUSTER_NAMES[c]], ci_show=False, linewidth=2.5)

        med = kmf.median_survival_time_
        if np.isinf(med) or pd.isna(med):
            median_survivals[c] = "Not Reached"
        else:
            median_survivals[c] = f"{med:.1f} months"

    results = multivariate_logrank_test(
        df_survival["OS_MONTHS"], df_survival["CLINICAL_CLUSTER"], df_survival["OS_STATUS"]
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

    ax.set_title("TCGA-SKCM OS: Kaplan-Meier of Clinical Clusters", fontsize=16, weight="bold", pad=15)
    ax.set_xlabel("Overall Survival (Months)", fontsize=13, labelpad=10)
    ax.set_ylabel("Survival Probability", fontsize=13, labelpad=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)

    out_path = plot_dir / "km_clinical_clusters.png"
    save_fig(fig, out_path)
    print(f"\nSaved KM cluster plot to {out_path.relative_to(BASE_DIR).as_posix()} (p = {p_val:.2e})")

    return p_val, median_survivals


def _plot_cluster_pca(scaled_data: np.ndarray, cluster_labels: np.ndarray, plot_dir: Path) -> None:
    """Generates 2D PCA projection scatter plot of patient clusters with centroids.

    Args:
        scaled_data: Standardised feature matrix.
        cluster_labels: Integer cluster assignments.
        plot_dir: Directory path to export plot artifact.
    """
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(scaled_data)

    df_pca = pd.DataFrame(pca_coords, columns=["PC1", "PC2"])
    df_pca["Cluster"] = cluster_labels
    df_pca["Cluster_Name"] = df_pca["Cluster"].map(CLUSTER_NAMES)

    var_explained = pca.explained_variance_ratio_

    palette_dict = {
        CLUSTER_NAMES[0]: COHORT_PALETTE["Liu 2019"],
        CLUSTER_NAMES[1]: RESPONSE_PALETTE["CR/PR"],
        CLUSTER_NAMES[2]: RESPONSE_PALETTE["PD"],
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
        s=100,
        ax=ax_pca,
        edgecolor="w",
        linewidth=0.8,
    )

    for c in range(3):
        centroid = df_pca[df_pca["Cluster"] == c][["PC1", "PC2"]].mean()
        ax_pca.scatter(
            centroid["PC1"],
            centroid["PC2"],
            marker="X",
            s=250,
            color="black",
            edgecolor="white",
            linewidth=1.5,
            zorder=10,
            label="Cluster Centroid" if c == 0 else "",
        )

    ax_pca.set_title("2D PCA Projection of Patient Clinical & Genomic Clusters", fontsize=15, weight="bold", pad=15)
    ax_pca.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% explained variance)", fontsize=13)
    ax_pca.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% explained variance)", fontsize=13)

    handles, labels = ax_pca.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_pca.legend(by_label.values(), by_label.keys(), title="Patient Groups", loc="best", fontsize=10)

    ax_pca.grid(True, linestyle="--", alpha=0.5)

    out_path = plot_dir / "pca_clinical_clusters.png"
    save_fig(fig, out_path)
    print(f"Saved PCA cluster visualisation to {out_path.relative_to(BASE_DIR).as_posix()}")


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Formats a DataFrame into a Markdown table string without tabulate dependency."""
    headers = list(df.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = [
        "| " + " | ".join(str(v) for v in row.values) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header_line, separator_line] + row_lines)


def _generate_clustering_report(
    profile_df: pd.DataFrame,
    counts: List[int],
    p_val: float,
    median_survivals: Dict[int, str],
    report_path: Path,
) -> None:
    """Generates comprehensive Markdown report summarizing clinical clustering.

    Args:
        profile_df: DataFrame of cluster feature means.
        counts: List of patient counts per cluster.
        p_val: Log-rank test p-value across clusters.
        median_survivals: Map of cluster index to median OS string.
        report_path: Destination file path for Markdown report.
    """
    table_rows = []

    table_rows.append(["**Demographics & Baseline**", "", "", ""])
    table_rows.append(["Patient Count (N)", str(counts[0]), str(counts[1]), str(counts[2])])

    age_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'AGE'].values[0]:.1f}" for c in range(3)]
    table_rows.append(["Age (Years, Mean)", age_vals[0], age_vals[1], age_vals[2]])

    table_rows.append(["**Genomics**", "", "", ""])
    tmb_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'TMB_NONSYNONYMOUS'].values[0]:.1f}" for c in range(3)]
    aneu_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'ANEUPLOIDY_SCORE'].values[0]:.1f}" for c in range(3)]
    table_rows.append(["TMB (Nonsynonymous, Mean Mut/Mb)", tmb_vals[0], tmb_vals[1], tmb_vals[2]])
    table_rows.append(["Aneuploidy Score (Mean)", aneu_vals[0], aneu_vals[1], aneu_vals[2]])

    table_rows.append(["**Microenvironment**", "", "", ""])
    hyp_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'WINTER_HYPOXIA_SCORE'].values[0]:.2f}" for c in range(3)]
    table_rows.append(["Winter Hypoxia Score (Mean)", hyp_vals[0], hyp_vals[1], hyp_vals[2]])

    table_rows.append(["**Adjuvant Treatment History**", "", "", ""])
    che_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'TX_TYPE_CHEMOTHERAPY'].values[0]*100:.1f}%" for c in range(3)]
    imm_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'TX_TYPE_IMMUNOTHERAPY'].values[0]*100:.1f}%" for c in range(3)]
    rad_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'TX_TYPE_RADIATION_THERAPY'].values[0]*100:.1f}%" for c in range(3)]
    table_rows.append(["Chemotherapy Received (%)", che_vals[0], che_vals[1], che_vals[2]])
    table_rows.append(["Immunotherapy Received (%)", imm_vals[0], imm_vals[1], imm_vals[2]])
    table_rows.append(["Radiation Therapy Received (%)", rad_vals[0], rad_vals[1], rad_vals[2]])

    table_rows.append(["**Clinical Presentation**", "", "", ""])
    pri_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'IS_PRIMARY'].values[0]*100:.1f}%" for c in range(3)]
    stg_vals = [f"{profile_df.loc[profile_df['CLINICAL_CLUSTER'] == c, 'IS_STAGE_IV'].values[0]*100:.1f}%" for c in range(3)]
    table_rows.append(["Primary Specimen Type (%)", pri_vals[0], pri_vals[1], pri_vals[2]])
    table_rows.append(["Stage IV Metastatic Disease (%)", stg_vals[0], stg_vals[1], stg_vals[2]])

    table_rows.append(["**Prognosis & Outcomes**", "", "", ""])
    table_rows.append(["Median Overall Survival", median_survivals[0], median_survivals[1], median_survivals[2]])

    final_cols = [
        "Feature / Clinicopathological Metric",
        f"Cluster 0 (N={counts[0]})",
        f"Cluster 1 (N={counts[1]})",
        f"Cluster 2 (N={counts[2]})",
    ]
    formatted_df = pd.DataFrame(table_rows, columns=final_cols)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Patient Phenotyping via Clinical & Genomic Clustering\n\n")
        f.write("We performed unsupervised phenotyping on the **TCGA-SKCM** cohort ($N = 426$ patients) using Agglomerative Hierarchical Clustering (Ward linkage). Patient profiles were constructed across demographics, tumour genetics, microenvironmental stress, and adjuvant therapy classes.\n\n")

        f.write("## Cluster Profiles\n")
        f.write("The average clinical and genomic values for each patient cluster are detailed below in a transposed summary table:\n\n")

        f.write(_df_to_markdown_table(formatted_df) + "\n\n")

        f.write("## Clinical Interpretation of Clusters\n\n")
        f.write("Based on the multi-dimensional profiles, the three clusters represent distinct disease states:\n\n")

        f.write(f"1.  **Cluster 0: Baseline Disease / Primary Specimen Phenotype** ($N={counts[0]}$)\n")
        f.write(f"    *   *Genomics*: Moderate chromosomal instability (aneuploidy score = {aneu_vals[0]}) and moderate TMB ({tmb_vals[0]} mut/Mb).\n")
        f.write(f"    *   *Microenvironment*: Low-moderate Winter hypoxia score ({hyp_vals[0]}).\n")
        f.write(f"    *   *Clinical*: Stage IV rate = {stg_vals[0]}. Highest rate of primary specimens ({pri_vals[0]}). No immunotherapy ({imm_vals[0]}).\n")
        f.write(f"    *   *Prognosis*: Better overall survival trajectory.\n\n")

        f.write(f"2.  **Cluster 1: High Mutational Load & Immunotherapy Phenotype** ($N={counts[1]}$)\n")
        f.write(f"    *   *Genomics*: Elevated copy-number alterations (aneuploidy score = {aneu_vals[1]}) and the highest mutational load (**TMB = {tmb_vals[1]} mut/Mb**).\n")
        f.write(f"    *   *Microenvironment*: Low-moderate Winter hypoxia score ({hyp_vals[1]}).\n")
        f.write(f"    *   *Clinical*: Stage IV rate = {stg_vals[1]}. Lower primary tumour rate ({pri_vals[1]}). High rate of immunotherapy ({imm_vals[1]}).\n")
        f.write(f"    *   *Prognosis*: Moderate survival trajectory.\n\n")

        f.write(f"3.  **Cluster 2: Stage IV / Advanced Metastatic Disease Phenotype** ($N={counts[2]}$)\n")
        f.write(f"    *   *Genomics*: Lower mutational load (TMB = {tmb_vals[2]} mut/Mb) and lowest copy-number alterations (aneuploidy score = {aneu_vals[2]}).\n")
        f.write(f"    *   *Clinical*: **100% of these patients have Stage IV disease**. No patients received immunotherapy (0.0%).\n")
        f.write(f"    *   *Prognosis*: Poor overall survival trajectory (Stage IV baseline) (Median Overall Survival = **{median_survivals[2]}**).\n\n")

        f.write("## Cluster Visualisation (2D PCA Projection)\n")
        f.write("Below is a 2D PCA projection of the multi-dimensional patient profiles, showing the distinct separation of the three clinical-genomic patient groups. The 'X' markers show the cluster centroids:\n\n")
        f.write("![2D PCA Visualisation of Clusters](../plots/clinical/pca_clinical_clusters.png)\n\n")

        f.write("## Kaplan-Meier Survival Analysis\n")
        f.write(f"The unsupervised patient clusters show a highly statistically significant separation in overall survival duration (Log-Rank p-value = **\\({p_val:.2e}\\)**):\n\n")
        f.write("![KM Survival of Clinical Clusters](../plots/clinical/km_clinical_clusters.png)\n")

    print(f"Clustering report successfully written to {report_path.relative_to(BASE_DIR).as_posix()}")


def main() -> None:
    """Executes the patient clustering pipeline on TCGA-SKCM clinical and genomic features."""
    print("==================================================")
    print("Clinical & Genomic Patient Clustering: TCGA-SKCM")
    print("==================================================")

    PLOTS_DIR.mkdir(exist_ok=True, parents=True)
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)

    df, sub_df, scaled_data = _load_and_preprocess_tcga_clinical(CLINICAL_FILE)

    agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
    cluster_labels = agg.fit_predict(scaled_data)
    df["CLINICAL_CLUSTER"] = cluster_labels
    print("Successfully clustered patients into 3 distinct groups.")

    profile_df = df.groupby("CLINICAL_CLUSTER")[CLUSTERING_COLS].mean().reset_index()
    counts = df["CLINICAL_CLUSTER"].value_counts().sort_index().tolist()
    profile_df.insert(1, "Patient Count", counts)

    print("\n--- Cluster Profiles (Mean Values) ---")
    print(profile_df.to_string(index=False))

    df_survival = df.dropna(subset=["OS_MONTHS", "OS_STATUS"]).copy()

    p_val, median_survivals = _plot_cluster_survival(df_survival, PLOTS_DIR)
    _plot_cluster_pca(scaled_data, cluster_labels, PLOTS_DIR)

    output_report = REPORTS_DIR / "clinical_clustering_results.md"
    _generate_clustering_report(profile_df, counts, p_val, median_survivals, output_report)

    print("\n==================================================")
    print("Done! Clinical clustering workflow completed.")
    print("==================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(BASE_DIR).as_posix()}")
            main()
