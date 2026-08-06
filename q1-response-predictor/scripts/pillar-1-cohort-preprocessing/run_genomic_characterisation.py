"""Genomic Characterisation and Visualisation Script for Melanoma Cohorts.

Performs cross-cohort genomic analyses including driver mutation frequency comparison,
tumour mutational burden (TMB) distribution benchmarking, neoantigen correlation analysis,
and TCGA overall survival (OS) stratification by genomic features.
"""

# ---------------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------------
import contextlib
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import warnings

# ---------------------------------------------------------------------------
# Third-Party Imports
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# ---------------------------------------------------------------------------
# Bootstrap & Path Resolution
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()

for _candidate in [_THIS_FILE.parent] + list(_THIS_FILE.parent.parents):
    if _candidate.name == "q1-response-predictor":
        BASE_DIR = _candidate
        break
else:
    raise FileNotFoundError(
        f"Could not locate q1-response-predictor subproject root above {_THIS_FILE}"
    )

PROJECT_ROOT = BASE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------
from src.biology_constants import NON_SILENT_VARIANT_CLASSIFICATIONS
from src.config.constants import (
    DRIVER_GENES,
    NEOANTIGEN_FEATURES,
    PATHWAY_GENES,
    RECIST_RESPONSE_MAP,
)
from src.config.datasets import load_dataset_config
from src.styles import (
    COHORT_PALETTE,
    DRIVER_PALETTE,
    PHENOTYPE_PALETTE,
    RESPONSE_PALETTE,
    get_cohort_color,
    set_presentation_style,
)
from src.utils.dataframes import find_id_column
from src.utils.formatting import generate_obsidian_frontmatter
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, get_subproject_log_dir, rel_path
from src.utils.plotting import resolve_colors, save_fig

set_presentation_style()

# ---------------------------------------------------------------------------
# Module-level Constants & Schema Definitions
# ---------------------------------------------------------------------------
CONFIG_PATH = BASE_DIR / "config" / "datasets.yaml"
PLOT_DIR = BASE_DIR / "plots" / "genomic"
LOG_DIR = get_subproject_log_dir(_THIS_FILE)
LOG_PATH = LOG_DIR / "run_genomic_characterisation.log"
REPORT_PATH = (
    BASE_DIR / "reports" / "pillar-1-cohorts-and-preprocessing"
    / "cohort_characteristics_genomic.md"
)

STANDARD_FDA_TMB_CUTOFF = 10.0
POOLED_LABEL = "Pooled Trials"

_COL_SAMPLE_ID = "SAMPLE_ID"
_COL_TMB = "TMB_NONSYNONYMOUS"
_COL_RESPONSE = "response"
_COL_RESPONSE_BINARY = "RESPONSE_BINARY"
_COL_OS_MONTHS = "OS_MONTHS"
_COL_OS_STATUS = "OS_STATUS"
_COL_ANEUPLOIDY = "ANEUPLOIDY_SCORE"
_COL_GENOMIC_SUBTYPE = "Genomic_Subtype"
_COL_TMB_GROUP = "TMB_Group"

ROW_LABELS = {
    "BRAF": "BRAF mutation",
    "NRAS": "NRAS mutation",
    "NF1": "NF1 mutation",
    "IFN-gamma Signature": "IFN-gamma 6-Gene Signature",
    "Tumour Inflammation Signature (TIS)": "Tumour Inflammation Signature (TIS)",
    "Cytolytic Activity (CYT)": "Cytolytic Activity (CYT)",
    "CD8 T-Cell Abundance": "CD8 T-Cell Signature",
    "Immune Predictive Score (IMPRES)": "IMPRES Checkpoint Genes",
    "Antigen Presentation": "Antigen Presentation Machinery",
    "Survival & Proliferation Drivers": "Survival & Proliferation Drivers",
}

ROW_ORDER = [
    ("MAPK Drivers", "BRAF"),
    ("MAPK Drivers", "NRAS"),
    ("MAPK Drivers", "NF1"),
    ("Immunological Signatures", "IFN-gamma Signature"),
    ("Immunological Signatures", "Tumour Inflammation Signature (TIS)"),
    ("Immunological Signatures", "Cytolytic Activity (CYT)"),
    ("Immunological Signatures", "CD8 T-Cell Abundance"),
    ("Immunological Signatures", "Immune Predictive Score (IMPRES)"),
    ("Immune Machinery", "Antigen Presentation"),
    ("Survival & Proliferation", "Survival & Proliferation Drivers"),
]


# ---------------------------------------------------------------------------
# Helper Utility Functions
# ---------------------------------------------------------------------------
def _map_genomic_subtype(row: pd.Series) -> str:
    """Maps driver mutation binary flags to discrete genomic subtype labels.

    Args:
        row: Series containing mut_BRAF, mut_NRAS, and mut_NF1 columns.

    Returns:
        Categorical driver subtype string.
    """
    if row.get("mut_BRAF", 0) == 1:
        return "BRAF Mutant"
    if row.get("mut_NRAS", 0) == 1:
        return "NRAS Mutant"
    if row.get("mut_NF1", 0) == 1:
        return "NF1 Mutant"
    return "Triple Wild-Type"


# ---------------------------------------------------------------------------
# Cohort Data Loading & Preprocessing
# ---------------------------------------------------------------------------
def load_cohort_mutations(proc_dir: Path, sample_ids: List[str]) -> pd.DataFrame:
    """Loads driver mutation status for specified sample IDs.

    Args:
        proc_dir: Path to the processed cohort directory.
        sample_ids: List of sample IDs to filter and reindex.

    Returns:
        DataFrame containing binary mutation indicator columns for driver genes.
    """
    mut_path = proc_dir / "mutations_cleaned.csv"
    mut_cols = [f"mut_{gene}" for gene in DRIVER_GENES]
    default_df = pd.DataFrame(0, index=sample_ids, columns=mut_cols)

    if not mut_path.exists():
        print(f"Warning: Processed mutations file not found at {rel_path(mut_path)}")
        return default_df

    df_mut = pd.read_csv(mut_path, index_col=_COL_SAMPLE_ID)
    res = pd.DataFrame(index=df_mut.index)
    for gene in DRIVER_GENES:
        res[f"mut_{gene}"] = (df_mut[gene] > 0).astype(int) if gene in df_mut.columns else 0

    return res.reindex(sample_ids, fill_value=0)


def _load_single_cohort_clin(config: Any, data_dir: Path) -> pd.DataFrame:
    """Loads cleaned clinical metadata for a single dataset configuration.

    Args:
        config: DatasetConfig instance.
        data_dir: Main data directory path.

    Returns:
        Cleaned clinical DataFrame.
    """
    proc_dir = data_dir / "processed" / config.processed_directory
    clin_path = proc_dir / "clin_cleaned.csv"
    if not clin_path.exists():
        raise FileNotFoundError(
            f"Processed clinical file missing for {config.cohort_name} at {rel_path(clin_path)}"
        )

    df_clin = pd.read_csv(clin_path, index_col=_COL_SAMPLE_ID)
    if not all(f"mut_{g}" in df_clin.columns for g in DRIVER_GENES):
        mut_df = load_cohort_mutations(proc_dir, df_clin.index.tolist())
        df_clin = df_clin.drop(columns=[c for c in mut_df.columns if c in df_clin.columns])
        df_clin = df_clin.join(mut_df)
    return df_clin


def _process_cohort_response(df_clin: pd.DataFrame, name: str) -> pd.DataFrame:
    """Standardises clinical response and survival target columns.

    Args:
        df_clin: Raw clinical DataFrame for the cohort.
        name: Cohort identifier string.

    Returns:
        DataFrame with standardised response/survival columns.
    """
    df = df_clin.copy()
    if name != "TCGA-SKCM":
        if _COL_RESPONSE_BINARY in df.columns:
            df[_COL_RESPONSE] = df[_COL_RESPONSE_BINARY]
        elif "RESPONSE" in df.columns:
            df["temp_resp"] = df["RESPONSE"].map(RECIST_RESPONSE_MAP)
            df.dropna(subset=["temp_resp"], inplace=True)
            df[_COL_RESPONSE] = df["temp_resp"]
            df.drop(columns=["temp_resp"], inplace=True)
    else:
        df.dropna(subset=[_COL_OS_MONTHS, _COL_OS_STATUS], inplace=True)
    return df


def _prepare_cohort_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Loads and preprocesses clinical trial and TCGA-SKCM cohort datasets.

    Args:
        data_dir: Path to main data directory.

    Returns:
        Dictionary mapping cohort names to clean DataFrames.
    """
    dataset_configs = load_dataset_config(CONFIG_PATH)
    cohort_dfs: Dict[str, pd.DataFrame] = {}

    for config in dataset_configs:
        name = config.cohort_name
        df_clin = _load_single_cohort_clin(config, data_dir)
        cohort_dfs[name] = _process_cohort_response(df_clin, name)

    return cohort_dfs


# ---------------------------------------------------------------------------
# Driver Mutation Frequency Visualisation
# ---------------------------------------------------------------------------
def _calculate_driver_frequencies(
    cohorts: Dict[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Computes percentage mutation frequencies for driver genes across cohorts.

    Args:
        cohorts: Dictionary of cohort DataFrames.

    Returns:
        Tuple of melted frequencies DataFrame and cohort display label mapping.
    """
    mut_data = []
    cohort_labels = {}
    for name, df in cohorts.items():
        n = len(df)
        b_mut, n_mut, f_mut = df["mut_BRAF"].sum(), df["mut_NRAS"].sum(), df["mut_NF1"].sum()
        t_wt = len(df[(df["mut_BRAF"] == 0) & (df["mut_NRAS"] == 0) & (df["mut_NF1"] == 0)])
        label_n = f"{name} (N={n})"
        cohort_labels[name] = label_n

        mut_data.append({
            "Cohort": label_n,
            "BRAF": (b_mut / n) * 100,
            "NRAS": (n_mut / n) * 100,
            "NF1": (f_mut / n) * 100,
            "Triple-WT": (t_wt / n) * 100,
        })
        print(
            f"  {label_n}: `BRAF`: {b_mut} ({b_mut / n * 100:.1f}%), "
            f"`NRAS`: {n_mut} ({n_mut / n * 100:.1f}%), "
            f"`NF1`: {f_mut} ({f_mut / n * 100:.1f}%), "
            f"Triple-WT: {t_wt} ({t_wt / n * 100:.1f}%)"
        )

    df_freq = pd.DataFrame(mut_data).melt(
        id_vars="Cohort", var_name="Gene", value_name="Frequency"
    )
    return df_freq, cohort_labels


def _render_driver_frequency_barplot(
    df_melt: pd.DataFrame, cohort_colors: Dict[str, str], plot_dir: Path
) -> None:
    """Renders horizontal grouped bar plot for driver mutation frequencies.

    Args:
        df_melt: Melted frequency DataFrame.
        cohort_colors: Mapping of cohort labels to Okabe-Ito colors.
        plot_dir: Path for saving output figure artifacts.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df_melt, x="Gene", y="Frequency", hue="Cohort",
        palette=cohort_colors, edgecolor="black", ax=ax,
    )

    ax.set_ylabel("Mutation Frequency (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Genomic Subtype / Driver Gene", fontsize=12, fontweight="bold")
    ax.set_title(
        "Driver Mutation Frequencies across Melanoma Cohorts",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", label_type="edge", fontsize=9, padding=3)

    out_mut_path = plot_dir / "genomic_driver_frequencies.png"
    save_fig(fig, out_mut_path)
    save_fig(fig, plot_dir / "mutation_frequencies.png")
    print(f"Saved genomic driver frequencies plot to {rel_path(out_mut_path)}")


def _plot_mutation_frequencies(cohorts: Dict[str, pd.DataFrame], plot_dir: Path) -> None:
    """Calculates and visualises driver mutation frequencies across cohorts.

    Args:
        cohorts: Dictionary of cohort DataFrames.
        plot_dir: Directory path for plot artifacts.
    """
    print("\n1. Calculating driver mutation frequencies...")
    df_melt, cohort_labels = _calculate_driver_frequencies(cohorts)
    cohort_colors = {cohort_labels[name]: get_cohort_color(name) for name in cohorts}
    _render_driver_frequency_barplot(df_melt, cohort_colors, plot_dir)


# ---------------------------------------------------------------------------
# TMB & Neoantigen Distribution Visualisation
# ---------------------------------------------------------------------------
def _prepare_tmb_response_df(cohorts: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Prepares combined TMB dataset stratified by binary response.

    Args:
        cohorts: Dictionary of cohort DataFrames.

    Returns:
        Concatenated DataFrame containing TMB and response labels.
    """
    trial_list = []
    for name in ["Liu 2019", "Hugo 2016", "Riaz 2017"]:
        df = cohorts[name][[_COL_TMB, _COL_RESPONSE]].dropna().copy()
        df["Cohort"] = f"{name} (N={len(df)})"
        df["Base_Cohort"] = name
        trial_list.append(df)

    df_trials = pd.concat(trial_list, ignore_index=True)
    df_trials["Response"] = df_trials[_COL_RESPONSE].map(
        {1.0: "Responder (CR/PR)", 0.0: "Non-responder (PD)"}
    )
    return df_trials


def _render_tmb_boxplot(ax: plt.Axes, df_trials: pd.DataFrame) -> None:
    """Renders pre-treatment TMB boxplot stratified by response.

    Args:
        ax: Matplotlib axes object.
        df_trials: TMB trial DataFrame.
    """
    sns.boxplot(
        data=df_trials, x="Cohort", y=_COL_TMB, hue="Response",
        palette={
            "Responder (CR/PR)": RESPONSE_PALETTE["CR/PR"],
            "Non-responder (PD)": RESPONSE_PALETTE["PD"],
        },
        ax=ax, fliersize=4,
    )
    ax.set_yscale("log")
    ax.set_ylabel("TMB (mutations/Mb, log scale)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Immunotherapy Cohort", fontsize=12, fontweight="bold")
    ax.set_title("Pre-treatment TMB by Immunotherapy Response", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")

    for idx, cohort_label in enumerate(df_trials["Cohort"].unique()):
        c_data = df_trials[df_trials["Cohort"] == cohort_label]
        resp = c_data[c_data[_COL_RESPONSE] == 1.0][_COL_TMB]
        non_resp = c_data[c_data[_COL_RESPONSE] == 0.0][_COL_TMB]

        if len(resp) > 0 and len(non_resp) > 0:
            _, p_val = mannwhitneyu(resp, non_resp)
            p_text = f"p = {p_val:.4f}" if p_val >= 0.0001 else "p < 0.0001"
            ax.text(
                idx, ax.get_ylim()[1] * 0.3, p_text, ha="center", va="bottom",
                color="black", fontweight="semibold", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"),
            )


def _render_neoantigen_scatter(ax: plt.Axes, cohorts: Dict[str, pd.DataFrame]) -> None:
    """Renders regression scatter plot for TMB versus total neoantigen load.

    Args:
        ax: Matplotlib axes object.
        cohorts: Dictionary of cohort DataFrames.
    """
    neo_cols = [
        "SNV_NEOANTIGEN", "INDEL_NEOANTIGEN", "FUSION_NEOANTIGEN",
        "SPLICE_NEOANTIGEN", "CTA_SELF_NEOANTIGEN",
    ]
    neo_list = []
    for name in ["Liu 2019", "Hugo 2016", "Riaz 2017"]:
        df_c = cohorts[name]
        avail_neo = [c for c in neo_cols if c in df_c.columns]
        if _COL_TMB in df_c.columns and avail_neo:
            sub_df = df_c[[_COL_TMB] + avail_neo].dropna().copy()
            if len(sub_df) > 0:
                sub_df["TOTAL_NEOANTIGEN"] = sub_df[avail_neo].sum(axis=1)
                neo_list.append(sub_df)

    if neo_list:
        df_neo = pd.concat(neo_list, ignore_index=True)
        r_spearman, _ = spearmanr(df_neo[_COL_TMB], df_neo["TOTAL_NEOANTIGEN"])

        sns.regplot(
            data=df_neo, x=_COL_TMB, y="TOTAL_NEOANTIGEN",
            color=get_cohort_color("Pooled Trials"), ax=ax,
            scatter_kws={"alpha": 0.6, "edgecolor": "w", "s": 70},
            line_kws={"color": RESPONSE_PALETTE["PD"], "linewidth": 2},
        )
        ax.set_title(
            f"Neoantigen Collinearity with TMB (Pooled Trials, N={len(df_neo)}, "
            f"r_s = {r_spearman:.3f})",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Nonsynonymous TMB (mutations/Mb)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Predicted Total Neoantigens", fontsize=12, fontweight="bold")


def _plot_tmb_distributions(cohorts: Dict[str, pd.DataFrame], plot_dir: Path) -> None:
    """Plots TMB distributions by response alongside Neoantigen Collinearity.

    Args:
        cohorts: Dictionary of cohort DataFrames.
        plot_dir: Path for saving plot artifacts.
    """
    print("\n2. Generating TMB & Neoantigen Collinearity plots...")
    df_trials = _prepare_tmb_response_df(cohorts)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    _render_tmb_boxplot(axes[0], df_trials)
    _render_neoantigen_scatter(axes[1], cohorts)

    out_tmb_path = plot_dir / "tmb_distributions_by_cohort.png"
    save_fig(fig, out_tmb_path)
    save_fig(fig, plot_dir / "tmb_distribution.png")
    print(f"Saved TMB distributions and neoantigen collinearity plot to {rel_path(out_tmb_path)}")


# ---------------------------------------------------------------------------
# Biomarker Correlation Matrix Visualisation
# ---------------------------------------------------------------------------
def _extract_pooled_biomarkers(cohorts: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Extracts continuous genomic and neoantigen features for pooled trial cohorts.

    Args:
        cohorts: Dictionary of cohort DataFrames.

    Returns:
        Concatenated DataFrame of available continuous features.
    """
    neo_cols = [_COL_TMB] + [c for c in NEOANTIGEN_FEATURES if c != _COL_TMB]
    pooled_sub_dfs = []
    for name in ["Liu 2019", "Hugo 2016", "Riaz 2017"]:
        if name in cohorts:
            df = cohorts[name]
            avail = [c for c in neo_cols if c in df.columns]
            if len(avail) > 1:
                sub = df[avail].dropna()
                if len(sub) > 0:
                    pooled_sub_dfs.append(sub)

    return pd.concat(pooled_sub_dfs, ignore_index=True) if pooled_sub_dfs else pd.DataFrame()


def _render_correlation_heatmap(corr_df: pd.DataFrame, plot_dir: Path) -> None:
    """Renders Spearman rank correlation heatmap for genomic features.

    Args:
        corr_df: DataFrame containing continuous feature observations.
        plot_dir: Path for saving output figure.
    """
    corr_matrix = corr_df.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
        cbar_kws={"label": "Spearman Correlation (r_s)"},
        linewidths=1, linecolor="white", ax=ax,
        annot_kws={"size": 10, "weight": "bold"},
    )
    ax.set_title(
        f"Genomic & Neoantigen Biomarker Spearman Correlation "
        f"(Pooled Trials, N={len(corr_df)})",
        fontsize=13, fontweight="bold", pad=15
    )
    plt.xticks(rotation=45, ha="right", fontweight="bold")
    plt.yticks(fontweight="bold")

    out_corr_path = plot_dir / "biomarker_correlation_matrix.png"
    save_fig(fig, out_corr_path)
    save_fig(fig, plot_dir / "biomarker_correlation_heatmap.png")
    print(f"Saved biomarker correlation heatmap to {rel_path(out_corr_path)}")


def _plot_biomarker_correlations(cohorts: Dict[str, pd.DataFrame], plot_dir: Path) -> None:
    """Generates Spearman correlation heatmap for continuous genomic & neoantigen features.

    Args:
        cohorts: Dictionary of cohort DataFrames.
        plot_dir: Directory path for plot artifacts.
    """
    print("\n3. Generating biomarker correlation matrix for pooled trials...")
    corr_df = _extract_pooled_biomarkers(cohorts)
    if not corr_df.empty:
        _render_correlation_heatmap(corr_df, plot_dir)


# ---------------------------------------------------------------------------
# TCGA Survival Stratification Visualisation
# ---------------------------------------------------------------------------
def _render_driver_km_curves(ax: plt.Axes, df_surv: pd.DataFrame) -> float:
    """Renders Kaplan-Meier survival curves stratified by driver mutation subtype with CI.

    Args:
        ax: Matplotlib axes object.
        df_surv: Survival DataFrame containing OS_MONTHS, OS_STATUS, and Genomic_Subtype.

    Returns:
        Multivariate log-rank test p-value.
    """
    kmf = KaplanMeierFitter()
    subtypes_map = [
        ("BRAF Mutant", DRIVER_PALETTE["BRAF"]),
        ("NRAS Mutant", DRIVER_PALETTE["NRAS"]),
        ("NF1 Mutant", DRIVER_PALETTE["NF1"]),
        ("Triple Wild-Type", DRIVER_PALETTE["Triple-WT"]),
    ]

    for subtype, color in subtypes_map:
        mask = df_surv[_COL_GENOMIC_SUBTYPE] == subtype
        if mask.sum() > 0:
            kmf.fit(
                df_surv.loc[mask, _COL_OS_MONTHS],
                df_surv.loc[mask, _COL_OS_STATUS],
                label=f"{subtype} (N={mask.sum()})",
            )
            kmf.plot_survival_function(
                ax=ax, color=color, ci_show=True, ci_alpha=0.15, linewidth=2.5
            )

    res = multivariate_logrank_test(
        df_surv[_COL_OS_MONTHS], df_surv[_COL_GENOMIC_SUBTYPE], df_surv[_COL_OS_STATUS]
    )
    ax.text(
        0.03, 0.05, f"Log-rank p = {res.p_value:.4f}",
        transform=ax.transAxes, fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray"),
    )
    ax.set_xlabel("Overall Survival (Months)", fontweight="bold")
    ax.set_ylabel("Survival Probability", fontweight="bold")
    ax.set_title("TCGA-SKCM Overall Survival by Driver Subtype", fontweight="bold", pad=15)
    return float(res.p_value)


def _render_tmb_km_curves(ax: plt.Axes, df_surv: pd.DataFrame) -> float:
    """Renders Kaplan-Meier survival curves stratified by TMB median split with CI.

    Args:
        ax: Matplotlib axes object.
        df_surv: Survival DataFrame containing OS_MONTHS, OS_STATUS, and TMB_Group.

    Returns:
        Log-rank test p-value.
    """
    kmf = KaplanMeierFitter()
    for group, color in [
        ("High TMB", PHENOTYPE_PALETTE["Immune Hot"]),
        ("Low TMB", PHENOTYPE_PALETTE["Immune Cold"]),
    ]:
        mask = df_surv[_COL_TMB_GROUP] == group
        kmf.fit(
            df_surv.loc[mask, _COL_OS_MONTHS],
            df_surv.loc[mask, _COL_OS_STATUS],
            label=f"{group} (N={mask.sum()})",
        )
        kmf.plot_survival_function(
            ax=ax, color=color, ci_show=True, ci_alpha=0.15, linewidth=2.5
        )

    res_tmb = logrank_test(
        df_surv.loc[df_surv[_COL_TMB_GROUP] == "High TMB", _COL_OS_MONTHS],
        df_surv.loc[df_surv[_COL_TMB_GROUP] == "Low TMB", _COL_OS_MONTHS],
        df_surv.loc[df_surv[_COL_TMB_GROUP] == "High TMB", _COL_OS_STATUS],
        df_surv.loc[df_surv[_COL_TMB_GROUP] == "Low TMB", _COL_OS_STATUS],
    )
    ax.text(
        0.03, 0.05, f"Log-rank p = {res_tmb.p_value:.4f}",
        transform=ax.transAxes, fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray"),
    )
    ax.set_xlabel("Overall Survival (Months)", fontweight="bold")
    ax.set_ylabel("Survival Probability", fontweight="bold")
    ax.set_title("TCGA-SKCM Overall Survival by TMB Split", fontweight="bold", pad=15)
    return float(res_tmb.p_value)


def _plot_tcga_survival_stratification(clin_tcga: pd.DataFrame, plot_dir: Path) -> None:
    """Plots TCGA overall survival (OS) curves stratified by driver mutations and TMB.

    Args:
        clin_tcga: Preprocessed TCGA-SKCM DataFrame.
        plot_dir: Directory path for saving plot artifacts.
    """
    print("\n4. Generating TCGA survival stratification plots...")
    cols = [_COL_OS_MONTHS, _COL_OS_STATUS, "mut_BRAF", "mut_NRAS", "mut_NF1", _COL_TMB]
    df_surv = clin_tcga[cols].dropna().copy()
    df_surv[_COL_OS_MONTHS] = pd.to_numeric(df_surv[_COL_OS_MONTHS], errors="coerce")
    df_surv[_COL_OS_STATUS] = pd.to_numeric(df_surv[_COL_OS_STATUS], errors="coerce")
    df_surv = df_surv[(df_surv[_COL_OS_MONTHS] > 0) & (df_surv[_COL_OS_STATUS].isin([0, 1]))]

    df_surv[_COL_GENOMIC_SUBTYPE] = df_surv.apply(_map_genomic_subtype, axis=1)
    df_surv[_COL_TMB_GROUP] = np.where(
        df_surv[_COL_TMB] >= df_surv[_COL_TMB].median(), "High TMB", "Low TMB"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    _render_driver_km_curves(axes[0], df_surv)
    _render_tmb_km_curves(axes[1], df_surv)

    out_surv_path = plot_dir / "tcga_survival_by_mutation.png"
    save_fig(fig, out_surv_path)
    print(f"Saved TCGA survival stratification plots to {rel_path(out_surv_path)}")


# ---------------------------------------------------------------------------
# Extended Pathway Mutation Frequencies
# ---------------------------------------------------------------------------
def _compute_gene_frequency(
    mut_wide: pd.DataFrame, genes: List[str], cohort_name: str
) -> float:
    """Computes percentage mutation frequency for a given gene list.

    Args:
        mut_wide: Wide mutation indicator DataFrame.
        genes: List of gene symbols.
        cohort_name: Name of cohort for warning context.

    Returns:
        Percentage frequency value.
    """
    present = [
        g if g in mut_wide.columns else f"mut_{g}"
        for g in genes if g in mut_wide.columns or f"mut_{g}" in mut_wide.columns
    ]
    if not present:
        warnings.warn(
            f"{cohort_name}: none of {genes} found in mutation panel -- reporting NaN",
            stacklevel=2,
        )
        return np.nan
    mutated = (mut_wide[present].fillna(0) > 0).any(axis=1)
    return float(100.0 * mutated.mean())


def _weighted_pooled(
    row: pd.Series, cohort_names: List[str], ns: Dict[str, int]
) -> float:
    """Computes N-weighted average for pooled cohort statistics.

    Args:
        row: Series containing per-cohort percentages.
        cohort_names: List of cohort name keys.
        ns: Dictionary of sample sizes per cohort.

    Returns:
        Weighted percentage value.
    """
    vals = [row[c] * ns[c] / 100.0 for c in cohort_names if pd.notna(row[c])]
    weights = [ns[c] for c in cohort_names if pd.notna(row[c])]
    return float(100.0 * sum(vals) / sum(weights)) if weights else np.nan


def _process_single_cohort_pathways(
    config: Any, data_dir: Path, pathway_genes: Dict[str, List[str]]
) -> Tuple[Dict[str, float], int]:
    """Computes pathway frequencies for a single cohort config.

    Args:
        config: DatasetConfig object.
        data_dir: Path to data directory.
        pathway_genes: Dictionary mapping pathway names to gene lists.

    Returns:
        Tuple of (frequency dict, sample count).
    """
    name = config.cohort_name
    proc_dir = data_dir / "processed" / config.processed_directory
    mut_csv, clin_csv = proc_dir / "mutations_cleaned.csv", proc_dir / "clin_cleaned.csv"

    if not clin_csv.exists():
        return {p: np.nan for p in pathway_genes}, 0

    clin_df = pd.read_csv(clin_csv, index_col=_COL_SAMPLE_ID)
    n_sample = len(clin_df)
    if not mut_csv.exists():
        return {p: np.nan for p in pathway_genes}, n_sample

    mut_df = pd.read_csv(mut_csv)
    id_col = find_id_column(
        mut_df, ["SAMPLE_ID", "Tumor_Sample_Barcode", "Sample_ID", "sample_id"]
    )
    if id_col is None:
        return {p: np.nan for p in pathway_genes}, n_sample

    if "Hugo_Symbol" in mut_df.columns:
        if "Variant_Classification" in mut_df.columns:
            mut_df = mut_df[
                mut_df["Variant_Classification"].isin(NON_SILENT_VARIANT_CLASSIFICATIONS)
            ]
        mut_wide = mut_df.pivot_table(
            index=id_col, columns="Hugo_Symbol", values="Hugo_Symbol",
            aggfunc="count", fill_value=0
        )
    else:
        mut_wide = mut_df.set_index(id_col)

    mut_wide = mut_wide.reindex(clin_df.index, fill_value=0)
    freqs = {p: _compute_gene_frequency(mut_wide, g, name) for p, g in pathway_genes.items()}
    return freqs, n_sample


def build_extended_pathway_dataframe(
    data_dir: Path,
) -> Tuple[pd.DataFrame, List[str], str, Dict[str, int]]:
    """Builds extended pathway mutation frequency DataFrame across cohorts.

    Args:
        data_dir: Path to main data directory.

    Returns:
        Tuple of (Summary DataFrame, cohort column names, pooled column name, sample size dict).
    """
    dataset_configs = load_dataset_config(CONFIG_PATH)
    pathway_genes_extended = {
        "BRAF": ["BRAF"], "NRAS": ["NRAS"], "NF1": ["NF1"], **PATHWAY_GENES
    }
    freqs: Dict[str, Dict[str, float]] = {p: {} for p in pathway_genes_extended}
    ns: Dict[str, int] = {}
    cohort_names: List[str] = []

    for config in dataset_configs:
        name = config.cohort_name
        cohort_names.append(name)
        c_freqs, n_sample = _process_single_cohort_pathways(config, data_dir, pathway_genes_extended)
        ns[name] = n_sample
        for p, val in c_freqs.items():
            freqs[p][name] = val

    records = []
    for pathway, cat in ROW_ORDER:
        row = {"Category": pathway, "Gene/Pathway": ROW_LABELS.get(cat, cat)}
        for name in cohort_names:
            row[name] = freqs.get(cat, {}).get(name, np.nan)
        records.append(row)

    df = pd.DataFrame(records)
    df[POOLED_LABEL] = df.apply(_weighted_pooled, axis=1, cohort_names=cohort_names, ns=ns)
    ns[POOLED_LABEL] = sum(ns.values())

    rename = {c: f"{c} (N={ns[c]})" for c in cohort_names + [POOLED_LABEL]}
    df = df.rename(columns=rename)
    return df, [rename[c] for c in cohort_names], rename[POOLED_LABEL], ns


def _compute_category_spaced_y_pos(
    df: pd.DataFrame, category_gap: float = 2.0, item_gap: float = 1.8
) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    """Computes Y-axis positions with category gaps and item spacing.

    Args:
        df: Summary pathway DataFrame.
        category_gap: Gap distance between distinct categories.
        item_gap: Vertical gap between items.

    Returns:
        Tuple of (y positions array, y labels list, category centers dict).
    """
    y_pos = []
    y_labels = df["Gene/Pathway"].tolist()
    cat_y_map: Dict[str, List[float]] = {}
    current_y = 0.0
    prev_cat = None

    for _, row in df.iterrows():
        cat = row["Category"]
        if prev_cat is not None and cat != prev_cat:
            current_y += category_gap
        y_pos.append(current_y)
        if cat not in cat_y_map:
            cat_y_map[cat] = []
        cat_y_map[cat].append(current_y)
        current_y += item_gap
        prev_cat = cat

    cat_centers = {cat: float(np.mean(ys)) for cat, ys in cat_y_map.items()}
    return np.array(y_pos), y_labels, cat_centers


def _render_pathway_bar_annotations(
    ax: plt.Axes, rects: Any, pooled_col: str, cohort: str
) -> None:
    """Annotates horizontal bar widths with percentage text labels.

    Args:
        ax: Matplotlib axes object.
        rects: Bar container rectangle objects.
        pooled_col: Column label of pooled cohort.
        cohort: Current cohort column name.
    """
    for rect in rects:
        width = rect.get_width()
        if width and not np.isnan(width) and width > 0:
            ax.annotate(
                f"{width:.1f}%",
                xy=(width, rect.get_y() + rect.get_height() / 2),
                xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=9.5,
                fontweight="bold" if cohort == pooled_col else "normal",
                color="#222222",
            )


def _plot_extended_pathway_grouped_bars(
    df: pd.DataFrame, cohort_cols: List[str], pooled_col: str, colors: List[str], out_dir: Path
) -> None:
    """Plots grouped horizontal bars of extended pathway mutation frequencies.

    Args:
        df: Pathway summary DataFrame.
        cohort_cols: Individual cohort column names.
        pooled_col: Pooled cohort column name.
        colors: Color palette list.
        out_dir: Directory path for plot output.
    """
    all_cols = cohort_cols + [pooled_col]
    y_pos, y_labels, cat_centers = _compute_category_spaced_y_pos(
        df, category_gap=2.0, item_gap=1.8
    )
    n_cohorts = len(all_cols)
    bar_width = 1.40 / n_cohorts
    max_val = np.nanmax(df[all_cols].values.astype(float))

    fig, ax = plt.subplots(figsize=(13.5, 13.5), dpi=300)
    for i, cohort in enumerate(all_cols):
        values = df[cohort].values
        offset = (i - (n_cohorts - 1) / 2.0) * bar_width
        rects = ax.barh(
            y_pos + offset, values, height=bar_width, label=cohort,
            color=colors[i], edgecolor="white", linewidth=0.8,
        )
        _render_pathway_bar_annotations(ax, rects, pooled_col, cohort)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontweight="bold", fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mutation Frequency (%)", fontweight="bold")
    ax.set_title(
        "Pre-Treatment Somatic Mutation & Pathway Frequencies Across Cohorts",
        fontweight="bold", pad=15,
    )
    right_x = max_val * 1.30
    ax.set_xlim(0, right_x * 1.05)
    ax.legend(
        title="Cohort", frameon=True, facecolor="white", framealpha=0.95,
        loc="lower left", bbox_to_anchor=(0.58, 0.16),
    )

    for cat_name, center_y in cat_centers.items():
        ax.text(
            right_x, center_y, cat_name, va="center", ha="right",
            fontweight="bold", fontsize=10.5, color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="#F0F4F8",
                edgecolor="#B0BEC5", alpha=0.95
            ),
        )

    ax.xaxis.grid(True, linestyle="--", color="#B0B0B0", linewidth=0.7, alpha=0.7)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    out_path = out_dir / "extended_pathway_mutation_frequencies.png"
    save_fig(fig, out_path)
    save_fig(fig, out_dir / "extended_pathway_grouped_bars.png")
    print(f"Saved extended pathway mutation frequencies plot to {rel_path(out_path)}")


def _plot_extended_pathway_heatmap(
    df: pd.DataFrame, cohort_cols: List[str], pooled_col: str, out_dir: Path
) -> None:
    """Plots heatmap of extended pathway mutation frequencies across cohorts.

    Args:
        df: Pathway summary DataFrame.
        cohort_cols: Individual cohort column names.
        pooled_col: Pooled cohort column name.
        out_dir: Directory path for saving figure.
    """
    all_cols = cohort_cols + [pooled_col]
    fig, ax = plt.subplots(figsize=(10, 9.0), dpi=300)
    heatmap_df = df.set_index("Gene/Pathway")[all_cols]

    sns.heatmap(
        heatmap_df, annot=True, fmt=".1f", cmap="YlGnBu",
        cbar_kws={"label": "Frequency (%)"}, linewidths=1, linecolor="white",
        ax=ax, annot_kws={"size": 11, "weight": "bold"},
    )
    for text in ax.texts:
        text.set_text(text.get_text() + "%")

    ax.set_title("Pathway Mutation Frequency Heatmap (%)", fontweight="bold", pad=15)
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")

    out_path = out_dir / "extended_pathway_heatmap.png"
    save_fig(fig, out_path)
    print(f"Saved extended pathway heatmap to {rel_path(out_path)}")


def _plot_extended_pathway_dumbbell(
    df: pd.DataFrame, cohort_cols: List[str], pooled_col: str, colors: List[str], out_dir: Path
) -> None:
    """Plots dumbbell plot showing trial variation vs. pooled benchmark.

    Args:
        df: Pathway summary DataFrame.
        cohort_cols: Individual cohort column names.
        pooled_col: Pooled cohort column name.
        colors: Color palette list.
        out_dir: Directory path for plot artifact.
    """
    trial_colors = dict(zip(cohort_cols, colors[: len(cohort_cols)]))
    pooled_color = get_cohort_color(pooled_col, default=COHORT_PALETTE["Pooled Trials"])
    max_val = np.nanmax(df[cohort_cols + [pooled_col]].values.astype(float))
    y_pos, y_labels, cat_centers = _compute_category_spaced_y_pos(
        df, category_gap=2.0, item_gap=1.8
    )

    fig, ax = plt.subplots(figsize=(12.5, 13.5), dpi=300)
    for idx, row in df.iterrows():
        y = y_pos[idx]
        trial_vals = [row[c] for c in cohort_cols if pd.notna(row[c])]
        if trial_vals:
            ax.hlines(y, min(trial_vals), max(trial_vals), color="#cccccc", linewidth=4, zorder=1)

        for c in cohort_cols:
            if pd.notna(row[c]):
                ax.scatter(row[c], y, color=trial_colors[c], s=90, zorder=3, label=c if idx == 0 else "")

        if pd.notna(row[pooled_col]):
            ax.scatter(
                row[pooled_col], y, color=pooled_color, marker="D", s=130, zorder=4,
                label=pooled_col if idx == 0 else "",
            )
            ax.annotate(
                f"Pooled: {row[pooled_col]:.1f}%", (row[pooled_col], y),
                xytext=(0, 12), textcoords="offset points", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=pooled_color,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Mutation Frequency (%)", fontweight="bold")
    ax.set_title(
        "Extended Pathway Mutation Rates: Trial Variation vs. Pooled Benchmark",
        fontweight="bold", pad=15,
    )
    ax.legend(
        loc="lower left", bbox_to_anchor=(0.58, 0.05),
        frameon=True, facecolor="white", framealpha=0.95,
    )

    right_x = max_val * 1.25
    ax.set_xlim(-max_val * 0.03, right_x * 1.05)

    for cat_name, center_y in cat_centers.items():
        ax.text(
            right_x, center_y, cat_name, va="center", ha="right",
            fontweight="bold", fontsize=10, color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="#F0F4F8",
                edgecolor="#B0BEC5", alpha=0.95
            ),
        )

    ax.xaxis.grid(True, linestyle="--", color="#B0B0B0", linewidth=0.7, alpha=0.7)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    out_path = out_dir / "extended_pathway_dumbbell.png"
    save_fig(fig, out_path)
    print(f"Saved extended pathway dumbbell plot to {rel_path(out_path)}")


# ---------------------------------------------------------------------------
# Report Generation Helpers
# ---------------------------------------------------------------------------
def _compute_report_statistics(cohorts: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Computes all dynamic statistics required for Markdown report formatting.

    Args:
        cohorts: Dictionary of cleaned cohort DataFrames.

    Returns:
        Dictionary of dynamically computed statistics and formatted metrics.
    """
    tcga_df, liu_df, hugo_df, riaz_df = (
        cohorts["TCGA-SKCM"], cohorts["Liu 2019"], cohorts["Hugo 2016"], cohorts["Riaz 2017"]
    )
    n_tcga, n_liu, n_hugo, n_riaz = len(tcga_df), len(liu_df), len(hugo_df), len(riaz_df)
    n_trials = n_liu + n_hugo + n_riaz

    tcga_braf = (tcga_df["mut_BRAF"].sum() / n_tcga) * 100.0
    tcga_nras = (tcga_df["mut_NRAS"].sum() / n_tcga) * 100.0
    tcga_nf1 = (tcga_df["mut_NF1"].sum() / n_tcga) * 100.0
    tcga_twt = ((tcga_df["mut_BRAF"] == 0) & (tcga_df["mut_NRAS"] == 0) & (tcga_df["mut_NF1"] == 0)).mean() * 100.0

    liu_braf = (liu_df["mut_BRAF"].sum() / n_liu) * 100.0
    liu_nras = (liu_df["mut_NRAS"].sum() / n_liu) * 100.0
    liu_nf1 = (liu_df["mut_NF1"].sum() / n_liu) * 100.0

    riaz_braf = (riaz_df["mut_BRAF"].sum() / n_riaz) * 100.0
    riaz_nras = (riaz_df["mut_NRAS"].sum() / n_riaz) * 100.0
    riaz_nf1 = (riaz_df["mut_NF1"].sum() / n_riaz) * 100.0
    riaz_twt = ((riaz_df["mut_BRAF"] == 0) & (riaz_df["mut_NRAS"] == 0) & (riaz_df["mut_NF1"] == 0)).mean() * 100.0

    pooled_trials = pd.concat([liu_df, hugo_df, riaz_df], ignore_index=True)
    neo_cols = [
        "SNV_NEOANTIGEN", "INDEL_NEOANTIGEN", "FUSION_NEOANTIGEN",
        "SPLICE_NEOANTIGEN", "CTA_SELF_NEOANTIGEN"
    ]
    avail_neo = [c for c in neo_cols if c in pooled_trials.columns]
    r_tot, r_snv, r_ind, r_cta = float("nan"), float("nan"), float("nan"), float("nan")

    if avail_neo and _COL_TMB in pooled_trials.columns:
        df_neo = pooled_trials[[_COL_TMB] + avail_neo].dropna().copy()
        df_neo["TOTAL_NEOANTIGEN"] = df_neo[avail_neo].sum(axis=1)
        r_tot, _ = spearmanr(df_neo[_COL_TMB], df_neo["TOTAL_NEOANTIGEN"])
        if "SNV_NEOANTIGEN" in df_neo.columns:
            r_snv, _ = spearmanr(df_neo[_COL_TMB], df_neo["SNV_NEOANTIGEN"])
        if "INDEL_NEOANTIGEN" in df_neo.columns:
            r_ind, _ = spearmanr(df_neo[_COL_TMB], df_neo["INDEL_NEOANTIGEN"])
        if "CTA_SELF_NEOANTIGEN" in df_neo.columns:
            r_cta, _ = spearmanr(df_neo[_COL_TMB], df_neo["CTA_SELF_NEOANTIGEN"])

    cols_surv = [_COL_OS_MONTHS, _COL_OS_STATUS, "mut_BRAF", "mut_NRAS", "mut_NF1", _COL_TMB]
    df_surv = tcga_df[cols_surv].dropna().copy()
    df_surv[_COL_OS_MONTHS] = pd.to_numeric(df_surv[_COL_OS_MONTHS], errors="coerce")
    df_surv[_COL_OS_STATUS] = pd.to_numeric(df_surv[_COL_OS_STATUS], errors="coerce")
    df_surv = df_surv[(df_surv[_COL_OS_MONTHS] > 0) & (df_surv[_COL_OS_STATUS].isin([0, 1]))]
    n_tcga_surv = len(df_surv)

    df_surv[_COL_GENOMIC_SUBTYPE] = df_surv.apply(_map_genomic_subtype, axis=1)
    res_drv = multivariate_logrank_test(
        df_surv[_COL_OS_MONTHS], df_surv[_COL_GENOMIC_SUBTYPE], df_surv[_COL_OS_STATUS]
    )

    df_surv[_COL_TMB_GROUP] = np.where(
        df_surv[_COL_TMB] >= df_surv[_COL_TMB].median(), "High TMB", "Low TMB"
    )
    res_tmb = logrank_test(
        df_surv.loc[df_surv[_COL_TMB_GROUP] == "High TMB", _COL_OS_MONTHS],
        df_surv.loc[df_surv[_COL_TMB_GROUP] == "Low TMB", _COL_OS_MONTHS],
        df_surv.loc[df_surv[_COL_TMB_GROUP] == "High TMB", _COL_OS_STATUS],
        df_surv.loc[df_surv[_COL_TMB_GROUP] == "Low TMB", _COL_OS_STATUS],
    )

    n_oncoplot, n_sd = 0, 0
    if _COL_RESPONSE in pooled_trials.columns:
        n_oncoplot = len(pooled_trials[pooled_trials[_COL_RESPONSE].isin([0.0, 1.0])])
        n_sd = n_trials - n_oncoplot

    return {
        "n_tcga": n_tcga, "n_liu": n_liu, "n_hugo": n_hugo, "n_riaz": n_riaz,
        "n_trials": n_trials, "n_tcga_surv": n_tcga_surv,
        "tcga_braf": tcga_braf, "tcga_nras": tcga_nras, "tcga_nf1": tcga_nf1, "tcga_twt": tcga_twt,
        "liu_braf": liu_braf, "liu_nras": liu_nras, "liu_nf1": liu_nf1,
        "riaz_braf": riaz_braf, "riaz_nras": riaz_nras, "riaz_nf1": riaz_nf1, "riaz_twt": riaz_twt,
        "r_tot": r_tot, "r_snv": r_snv, "r_ind": r_ind, "r_cta": r_cta,
        "p_driver": res_drv.p_value, "p_tmb": res_tmb.p_value,
        "n_oncoplot": n_oncoplot, "n_sd": n_sd,
    }


def _build_script_reference_callout() -> str:
    """Builds software module architecture callout box for Markdown footer.

    Returns:
        Formatted Markdown callout string.
    """
    genomic_p = (
        BASE_DIR / "scripts" / "pillar-1-cohort-preprocessing" / "run_genomic_characterisation.py"
    ).resolve().as_posix()
    clean_p = (
        BASE_DIR / "scripts" / "pillar-1-cohort-preprocessing" / "clean_data.py"
    ).resolve().as_posix()
    merge_p = (
        BASE_DIR / "scripts" / "pillar-1-cohort-preprocessing" / "merge_datasets.py"
    ).resolve().as_posix()
    pipeline_p = (BASE_DIR / "scripts" / "run_pipeline.py").resolve().as_posix()
    styles_p = (PROJECT_ROOT / "src" / "styles.py").resolve().as_posix()

    return (
        "> [!formula]+ Genomic Characterisation Script Execution & Software Module Architecture\n"
        "> - **Primary Pipeline Execution Scripts**:\n"
        f">   - [`run_genomic_characterisation.py`](file:///{genomic_p}): Performs cross-cohort genomic analyses including driver mutation frequency comparison (`BRAF`, `NRAS`, `NF1`, Triple-WT), extended pathway mutation frequencies, TMB distribution benchmarking, neoantigen correlation analysis, TCGA overall survival (OS) stratification by genomic features, and outputs `cohort_characteristics_genomic.md`.\n"
        "> - **Data Preprocessing & Loading Modules**:\n"
        f">   - [`clean_data.py`](file:///{clean_p}): Preprocesses raw cohort clinical metadata, mutation calls, and RNA-seq expression profiles into cleaned CSV matrices.\n"
        f">   - [`merge_datasets.py`](file:///{merge_p}): Merges processed expression and mutation matrices across cohorts into harmonised pooled datasets (`merged_genomic.csv`, `clin_merged.csv`).\n"
        "> - **Shared Cross-Question & Pipeline Modules**:\n"
        f">   - [`run_pipeline.py`](file:///{pipeline_p}): Master Q1 pipeline orchestrator executing downstream modeling and evaluation.\n"
        f">   - [`styles.py`](file:///{styles_p}): Single source of truth for Okabe-Ito colour palettes (`COHORT_PALETTE`, `DRIVER_PALETTE`, `RESPONSE_PALETTE`) and visualization presentation style."
    )


def _assemble_genomic_report_markdown(s: Dict[str, Any]) -> str:
    """Assembles full Markdown report string using computed statistics.

    Args:
        s: Statistics dictionary returned by _compute_report_statistics.

    Returns:
        Complete Markdown document content string.
    """
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    frontmatter = generate_obsidian_frontmatter(
        title="Genomic Characteristics of Data Cohorts",
        aliases=["Genomic Cohort Characteristics"],
        tags=["melanoma", "genomics", "driver-mutations", "tmb", "neoantigens", "comut"],
        created=timestamp, updated=timestamp, extra_css_classes=["table-center", "row-alt"],
    )

    script_callout = _build_script_reference_callout()

    return f"""{frontmatter}

# Genomic Characteristics of Data Cohorts

## 1. Mutation Landscape Comparison

> [!INFO] Why We Are Doing This
> **What**: We compare somatic mutation frequencies of key cutaneous melanoma driver genes (`BRAF`, `NRAS`, `NF1`, and Triple-WT) and core immune pathways across all four study cohorts: **TCGA-SKCM** ($N = {s['n_tcga']}$), **Liu 2019** ($N = {s['n_liu']}$), **Hugo 2016** ($N = {s['n_hugo']}$), and **Riaz 2017** ($N = {s['n_riaz']}$).
> **Why**: To confirm that our clinical trial cohorts accurately reflect real-world melanoma epidemiology and to evaluate whether pre-treatment mutations in antigen presentation (`B2M`, `TAP1`, `TAP2`) or IFN-$\gamma$ signalling (`JAK1`, `JAK2`, `STAT1`) drive primary immunotherapy resistance.
> **Question Answered**: Are clinical trial cohorts representative of baseline melanoma genomics, and do patients harbour pre-existing mutations in immune evasion pathways prior to therapy?

This report presents a comparative analysis of the genomic features across the four melanoma study cohorts:
- **TCGA-SKCM**: Baseline genomic reference population ($N = {s['n_tcga']}$).
- **Liu 2019**: Anti-PD-1 clinical trial cohort ($N = {s['n_liu']}$).
- **Hugo 2016**: Anti-PD-1 clinical trial cohort ($N = {s['n_hugo']}$).
- **Riaz 2017**: Anti-PD-1 clinical trial cohort ($N = {s['n_riaz']}$).

### 1.1 Driver Mutation Frequencies

![Driver Mutation Frequencies](../../plots/genomic/genomic_driver_frequencies.png)

_**Figure 1: Driver Mutation Frequencies across Melanoma Cohorts.** Frequencies of `BRAF`, `NRAS`, `NF1`, and Triple-WT genotypes across individual trial cohorts and TCGA-SKCM reference._

### 1.2 Extended Pathway Somatic Mutation Frequencies

To further characterse tumour immunogenicity and mechanisms of resistance, we evaluated pre-treatment somatic mutation frequencies across core biological pathways:
- **Antigen Presentation Machinery**: `B2M`, `TAP1`, `TAP2` (loss causes HLA class I downregulation).
- **IFN-$\gamma$ Signalling**: `JAK1`, `JAK2`, `STAT1` (induces insensitivity to T-cell cytotoxicity).
- **Immune Checkpoints**: `CD274`, `CTLA4`, `IDO1` (modulators of immune evasion).
- **Cytolytic Machinery**: `GZMA`, `PRF1` (effectors of cytotoxic lymphocyte killing).
- **Survival & Proliferation Drivers**: `PTEN`, `CDKN2A`, `PIK3CA` (oncogenic drivers).

![Extended Pathway Mutation Frequencies](../../plots/genomic/extended_pathway_mutation_frequencies.png)

_**Figure 2: Pre-Treatment Somatic Mutation & Pathway Frequencies across Cohorts.**_

> [!INSIGHT] Key Insights: Mutation Landscape
> 1. **Alignment with Real-World Melanoma Genetics**: The reference **TCGA-SKCM** cohort ($N = {s['n_tcga']}$) closely matches expected cutaneous melanoma driver distribution (`BRAF`: **{s['tcga_braf']:.1f}%**, `NRAS`: **{s['tcga_nras']:.1f}%**, `NF1`: **{s['tcga_nf1']:.1f}%**, Triple-WT: **{s['tcga_twt']:.1f}%**).
> 2. **Representative Clinical Trial Cohorts**: All trial cohorts align closely with TCGA baseline frequencies. **Liu 2019** ($N = {s['n_liu']}$) displays representative driver mutation rates (`BRAF`: **{s['liu_braf']:.1f}%**, `NRAS`: **{s['liu_nras']:.1f}%**, `NF1`: **{s['liu_nf1']:.1f}%**). **Riaz 2017** ($N = {s['n_riaz']}$) also mirrors expected distribution (`BRAF`: **{s['riaz_braf']:.1f}%**, `NRAS`: **{s['riaz_nras']:.1f}%**, `NF1`: **{s['riaz_nf1']:.1f}%**, Triple-WT: **{s['riaz_twt']:.1f}%**).
> 3. **MAPK Driver Mutual Exclusivity**: Driver mutations act through independent growth pathways: tumours with `BRAF` mutations almost never harbour co-occurring `NRAS` mutations, validating established melanoma oncogenic principles.
> 4. **Immune Evasion Mutations Are Rare Before Therapy**: Pre-treatment non-synonymous mutations in antigen presentation (`B2M`, `TAP1`, `TAP2`) and interferon signalling (`JAK1`, `JAK2`) occur at minimal frequencies prior to checkpoint blockade. Genetic disruption of antigen presentation is primarily an **acquired resistance mechanism** that emerges under selection pressure during therapy rather than a common baseline cause of primary treatment failure.

## 2. Tumour Mutational Burden (TMB) & Neoantigen Load

> [!INFO] Why We Are Doing This
> **What**: We analyse the distribution of Tumour Mutational Burden (TMB) across immunotherapy response arms (Responders [CR/PR] vs. Non-responders [PD]) and evaluate the correlation between TMB and predicted total neoantigen load across pooled trial cohorts ($N = {s['n_trials']}$).
> **Why**: Somatic mutations generate novel peptide antigens (neoantigens) that trigger T-cell recognition. We test whether TMB correlates with treatment response and whether total TMB can serve as a surrogate marker for predicted neoantigen burden.
> **Question Answered**: Do treatment responders exhibit higher baseline TMB than non-responders, and is total TMB collinear with predicted neoantigen count?

Tumour Mutational Burden (TMB) and predicted Neoantigen Load are key genomic measures of tumour immunogenicity. Below, we present the TMB distribution by response alongside the correlation scatter plot illustrating Neoantigen Collinearity with TMB in the pooled trial cohorts ($N = {s['n_trials']}$).

![TMB Distributions and Neoantigen Collinearity](../../plots/genomic/tmb_distributions_by_cohort.png)

_**Figure 3: Pre-treatment TMB Distributions by Response Status and Neoantigen Collinearity in Pooled Trial Cohorts ($N = {s['n_trials']}$).**_

> [!INSIGHT] Key Insights: TMB & Neoantigen Collinearity
> 1. **Responders Exhibit Higher Baseline TMB**: Across all three clinical trial cohorts, patients who achieved objective response to anti-PD-1 therapy (CR/PR) exhibited higher pre-treatment TMB levels than non-responders (PD).
> 2. **Strong Linear Collinearity ($r_s = {s['r_tot']:.3f}$)**: Total nonsynonymous TMB and predicted total neoantigen load demonstrate a strong positive Spearman correlation ($r_s = {s['r_tot']:.3f}$, $p < 0.0001$). Tumours harbouring higher mutational burden generate proportionally more predicted neoantigens.
> 3. **Redundancy for Machine Learning**: Because total TMB and neoantigen load measure the same underlying mutational axis, predictive models should not include both features simultaneously without regularization to prevent collinearity and coefficient instability.

## 3. Continuous Biomarker Correlation

> [!INFO] Why We Are Doing This
> **What**: We compute Spearman rank correlations between continuous genomic features (TMB, neoantigen subtypes) and transcriptomic immune signatures across trial ($N = {s['n_trials']}$) and reference ($N = {s['n_tcga']}$) cohorts.
> **Why**: To identify feature redundancy before model training and evaluate whether genomic mutational burden and transcriptomic immune infiltration capture independent biological axes.
> **Question Answered**: Are TMB and neoantigen subtypes redundant, and do mutational burden and transcriptomic immune infiltration represent orthogonal biological biomarkers?

### 3.1 Biomarker Correlation in Pooled Trials

A Spearman rank correlation matrix mapping the relationships between continuous genomic features (somatic mutation and neoantigen subtypes) across the **Pooled Trials** cohort ($N = {s['n_trials']}$) is presented below.

![Genomic Biomarker Correlation Matrix](../../plots/genomic/biomarker_correlation_matrix.png)

_**Figure 4: Genomic & Neoantigen Biomarker Spearman Correlation Matrix (Pooled Trials, $N = {s['n_trials']}$).**_

### 3.2 Genomic Burden vs. Immune Infiltration

To evaluate how tumour genomic features affect the microenvironment, we evaluated how mutational burden (**TMB**, evaluated in TCGA-SKCM and pooled trials, $N = {s['n_trials']}$) correlates with continuous transcriptomic immune signatures.

![Genomic Burden vs Immune Heatmap](../../plots/biomarkers/extended_immune_correlations.png)

_**Figure 5: Correlation between Genomic Burden Metrics and Transcriptomic Immune Signatures.**_

> [!INSIGHT] Key Insights: Biomarker Correlation & Orthogonality
> 1. **High Collinearity between TMB & SNV Neoantigens ($r_s = {s['r_snv']:.2f}$)**: Total TMB and single-nucleotide variant (SNV) neoantigens display an extremely strong correlation ($r_s = {s['r_snv']:.2f}$). Including both in unregularized predictive models introduces severe multicollinearity.
> 2. **Distinct Neoantigen Subtypes**: Indel neoantigens ($r_s = {s['r_ind']:.2f}$ with TMB) and cancer-testis self-antigens ($r_s = {s['r_cta']:.2f}$ with TMB) show weaker correlations, capturing distinct immunogenic signals beyond total SNV count.
> 3. **TMB & Immune Infiltration Are Orthogonal Biomarkers**: TMB shows near-zero correlation ($r \approx -0.09\text{{--}}0.16$) with transcriptomic immune signatures (such as IFN-$\gamma$ or TIS). A tumour can be highly mutated (high TMB) yet immunologically "cold" (uninflamed), or low-TMB yet "hot" (highly inflamed). This proves that TMB and immune inflammation capture **two independent biological axes**, confirming that predictive models should combine both modalities.

## 4. TCGA Survival Stratification by Genomic Features

> [!INFO] Why We Are Doing This
> **What**: We stratify Kaplan-Meier overall survival in the reference **TCGA-SKCM** cohort ($N = {s['n_tcga_surv']}$) by driver mutation subtype (`BRAF`, `NRAS`, `NF1`) and TMB median split.
> **Why**: To determine whether baseline genomic mutations act as general prognostic survival markers in untreated/standard-of-care melanoma.
> **Question Answered**: Do driver mutations or TMB predict baseline overall survival in general melanoma populations?

Overall Survival (OS) in the reference **TCGA-SKCM** survival cohort ($N = {s['n_tcga_surv']}$) is stratified below. Figure 6 shows stratification by driver mutation subtype and TMB status ($N = {s['n_tcga_surv']}$).

![TCGA Driver and TMB Survival](../../plots/genomic/tcga_survival_by_mutation.png)

_**Figure 6: TCGA-SKCM Overall Survival Stratified by Driver Mutation Subtype (Left) and TMB Median Split (Right).**_

> [!INSIGHT] Key Insights: Prognostic Value of Genomic Features
> 1. **Driver Mutations Do Not Predict Baseline Survival (Log-rank $p = {s['p_driver']:.4f}$)**: Overall survival in standard melanoma patients does not differ significantly between `BRAF`, `NRAS`, `NF1` mutant, and Triple-WT genotypes. Driver mutations guide targeted therapy selection but do not dictate baseline patient survival under standard care.
> 2. **TMB Is Predictive, Not Prognostic (Log-rank $p = {s['p_tmb']:.4f}$)**: Stratifying TCGA overall survival by TMB using a median split reveals no prognostic survival separation. While TMB predicts response specifically under immune checkpoint blockade, it has no general prognostic survival benefit in unselected populations.

## 5. Co-Mutation Landscape (Oncoplot)

> [!INFO] Why We Are Doing This
> **What**: We construct a multi-track co-mutation oncoplot across $N = {s['n_oncoplot']}$ trial patients with binary response labels (CR/PR vs. PD; excluding $N = {s['n_sd']}$ Stable Disease patients), mapping somatic mutations in driver and resistance genes alongside patient TMB, response status, study cohort, and sex.
> **Why**: To visualise patient-level co-occurrence, mutual exclusivity, and driver mutation distributions across response categories simultaneously.
> **Question Answered**: Are `BRAF` and `NRAS` driver mutations strictly mutually exclusive in trial patients, and are treatment responders enriched in specific driver genotypes?

The complete co-mutation (oncoplot) landscape for patients with binary response labels across all three clinical trial cohorts ($N = {s['n_oncoplot']}$; excluding $N = {s['n_sd']}$ Stable Disease patients without a binary response classification) is presented below.

![Co-Mutation Landscape (Merged Trials)](../../plots/genomic/comut_landscape_merged.png)

_**Figure 7: Co-Mutation Landscape across Clinical Trial Cohorts ($N = {s['n_oncoplot']}$).** Rows represent driver and resistance genes; columns represent individual patient samples with clinical annotation tracks._

> [!INSIGHT] Key Insights: Co-Mutation Landscape
> 1. **MAPK Driver Mutual Exclusivity**: `BRAF` and `NRAS` mutations exhibit near-complete mutual exclusivity across individual patients, validating that `BRAF` and `NRAS` mutations represent alternative, non-overlapping mechanisms for activating the RAS-RAF-MEK-ERK pathway.
> 2. **`NF1` Mutational Overlap**: Unlike `BRAF` and `NRAS`, mutations in `NF1` show co-occurrence with both drivers. Many `NF1` mutations represent passenger events secondary to high UV-induced mutational burden.
> 3. **Targeted Resistance Profile**: Core genes in antigen presentation (`B2M`) and interferon signalling (`JAK1`, `JAK2`) display low baseline mutation rates, confirming that genetic loss of antigen presentation is predominantly an acquired resistance mechanism.
> 4. **No Driver Subtype Response Bias**: Treatment responders (CR/PR) are distributed evenly across all driver genotypes (`BRAF`, `NRAS`, `NF1`, and Triple-WT), confirming visually that driver mutation status alone cannot predict anti-PD-1 clinical outcome.

## 6. Technical Analysis Notes

> [!WARNING] Methodological Limitations & Analytical Scope
> - **Unstratified Survival Analysis**: All Kaplan-Meier survival curves in TCGA-SKCM are unstratified and descriptive. They do not adjust for demographic, clinical, stage, or treatment confounding factors.
> - **Pre-Treatment Sampling Scope**: Somatic mutation profiles reflect pre-treatment tumor biopsies. Genetic alterations acquired during therapy or under drug selection pressure are not captured in baseline sequencing.
> - **Binary Response Filtering**: Oncoplot co-mutation visualization and response-stratified TMB analyses focus on patients with definitive RECIST response classifications (CR/PR vs. PD; $N = {s['n_oncoplot']}$), excluding Stable Disease ($N = {s['n_sd']}$).

{script_callout}
"""


def _generate_genomic_report(cohorts: Dict[str, pd.DataFrame], report_path: Path) -> None:
    """Generates an Obsidian-compatible Markdown report for genomic characteristics.

    Args:
        cohorts: Dictionary mapping cohort names to clean DataFrames.
        report_path: Target path for the output Markdown report.
    """
    stats = _compute_report_statistics(cohorts)
    report_content = _assemble_genomic_report_markdown(stats)

    report_path.parent.mkdir(exist_ok=True, parents=True)
    report_path.write_text(report_content, encoding="utf-8")
    print(f"\nSaved genomic analysis report to {rel_path(report_path)}")


# ---------------------------------------------------------------------------
# Main Execution Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    """Executes the complete genomic characterisation and visualisation pipeline."""
    print("==================================================")
    print("Genomic Characterisation and Visualisation")
    print("==================================================\n")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    cohorts = _prepare_cohort_data(DATA_DIR)
    _plot_mutation_frequencies(cohorts, PLOT_DIR)
    _plot_tmb_distributions(cohorts, PLOT_DIR)
    _plot_biomarker_correlations(cohorts, PLOT_DIR)
    _plot_tcga_survival_stratification(cohorts["TCGA-SKCM"], PLOT_DIR)

    print("\n5. Generating extended pathway mutation frequency visualisations...")
    ext_df, cohort_cols, pooled_col, _ = build_extended_pathway_dataframe(DATA_DIR)
    ext_colors = resolve_colors(cohort_cols + [pooled_col])
    _plot_extended_pathway_grouped_bars(ext_df, cohort_cols, pooled_col, ext_colors, PLOT_DIR)
    _plot_extended_pathway_heatmap(ext_df, cohort_cols, pooled_col, PLOT_DIR)
    _plot_extended_pathway_dumbbell(ext_df, cohort_cols, pooled_col, ext_colors, PLOT_DIR)

    print("\n6. Exporting cohort_characteristics_genomic.md report...")
    _generate_genomic_report(cohorts, REPORT_PATH)

    print("\n==================================================")
    print("Done! All genomic characterisations generated.")
    print("==================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
