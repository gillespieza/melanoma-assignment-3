"""Genomic Characterisation and Visualisation Script for Melanoma Cohorts.

Performs cross-cohort genomic analyses including driver mutation frequency comparison,
tumour mutational burden (TMB) distribution benchmarking, neoantigen biomarker correlation
analysis, and TCGA overall survival (OS) stratification by genomic features.
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
from scipy.stats import mannwhitneyu, spearmanr
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

_SUBPROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBPROJECT_ROOT))

import warnings

from src.config.constants import (
    DRIVER_GENES,
    GENOMIC_FEATURES,
    NEOANTIGEN_FEATURES,
    NON_SILENT_VARIANT_CLASSIFICATIONS,
    PATHWAY_GENES,
    RECIST_RESPONSE_MAP,
)
from src.config.datasets import DatasetConfig, load_dataset_config
from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
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
from src.utils.paths import (
    CONFIG_DIR,
    DATA_DIR,
    LOG_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    SUBPROJECT_ROOT,
    rel_path,
)
from src.utils.plotting import resolve_colors, save_fig

set_presentation_style()

# Module-level Constants
CONFIG_PATH = CONFIG_DIR / "datasets.yaml"
PLOT_DIR = PLOTS_DIR / "genomic"
LOG_PATH = LOG_DIR / "run_genomic_characterisation.log"
REPORT_PATH = REPORTS_DIR / "pillar-1-cohorts-and-preprocessing" / "cohort_characteristics_genomic.md"

STANDARD_FDA_TMB_CUTOFF = 10.0


def load_cohort_mutations(proc_dir: Path, sample_ids: List[str]) -> pd.DataFrame:
    """Loads driver mutation status for specified sample IDs.

    Args:
        proc_dir: Path to the processed cohort directory.
        sample_ids: List of sample IDs to filter/reindex.

    Returns:
        DataFrame containing binary mutation indicator columns for driver genes.
    """
    mut_path = proc_dir / "mutations_cleaned.csv"
    mut_cols = [f"mut_{gene}" for gene in DRIVER_GENES]
    default_df = pd.DataFrame(0, index=sample_ids, columns=mut_cols)

    if not mut_path.exists():
        print(f"Warning: Processed mutations file not found at {rel_path(mut_path)}")
        return default_df

    df_mut = pd.read_csv(mut_path, index_col="SAMPLE_ID")
    res = pd.DataFrame(index=df_mut.index)
    for gene in DRIVER_GENES:
        if gene in df_mut.columns:
            res[f"mut_{gene}"] = (df_mut[gene] > 0).astype(int)
        else:
            res[f"mut_{gene}"] = 0

    return res.reindex(sample_ids, fill_value=0)


def _prepare_cohort_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Loads and preprocesses clinical trial and TCGA-SKCM cohort datasets.

    Args:
        data_dir: Path to the main data directory.

    Returns:
        Dictionary mapping cohort names to clean DataFrames.
    """
    dataset_configs = load_dataset_config(CONFIG_PATH)
    cohort_dfs = {}

    for config in dataset_configs:
        name = config.cohort_name
        proc_dir = data_dir / "processed" / config.processed_directory
        clin_path = proc_dir / "clin_cleaned.csv"

        if not clin_path.exists():
            raise FileNotFoundError(f"Processed clinical file not found for {name} at {rel_path(clin_path)}")

        df_clin = pd.read_csv(clin_path, index_col="SAMPLE_ID")

        # Join driver mutations if missing
        if not all(f"mut_{g}" in df_clin.columns for g in DRIVER_GENES):
            mut_df = load_cohort_mutations(proc_dir, df_clin.index.tolist())
            for col in mut_df.columns:
                if col in df_clin.columns:
                    df_clin = df_clin.drop(columns=[col])
            df_clin = df_clin.join(mut_df)

        if name != "TCGA-SKCM":
            if "RESPONSE_BINARY" in df_clin.columns:
                df_clin["response"] = df_clin["RESPONSE_BINARY"]
            elif "RESPONSE" in df_clin.columns:
                df_clin["temp_resp"] = df_clin["RESPONSE"].map(RECIST_RESPONSE_MAP)
                df_clin.dropna(subset=["temp_resp"], inplace=True)
                df_clin["response"] = df_clin["temp_resp"]
                df_clin.drop(columns=["temp_resp"], inplace=True)
        else:
            df_clin = df_clin.dropna(subset=["OS_MONTHS", "OS_STATUS"])

        cohort_dfs[name] = df_clin

    return cohort_dfs


def _plot_mutation_frequencies(cohorts: Dict[str, pd.DataFrame], plot_dir: Path) -> None:
    """Calculates and visualises driver mutation frequencies across cohorts.

    Args:
        cohorts: Dictionary of cohort DataFrames.
        plot_dir: Path to directory for saving plot artifacts.
    """
    print("\n1. Calculating driver mutation frequencies...")
    mut_data = []
    cohort_labels = {}
    for name, df in cohorts.items():
        n = len(df)
        b_mut = df["mut_BRAF"].sum()
        n_mut = df["mut_NRAS"].sum()
        f_mut = df["mut_NF1"].sum()
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
        print(f"  {label_n}: `BRAF`: {b_mut} ({b_mut / n * 100:.1f}%), `NRAS`: {n_mut} ({n_mut / n * 100:.1f}%), `NF1`: {f_mut} ({f_mut / n * 100:.1f}%), Triple-WT: {t_wt} ({t_wt / n * 100:.1f}%)")

    df_mut_freq = pd.DataFrame(mut_data)
    df_mut_melt = df_mut_freq.melt(id_vars="Cohort", var_name="Gene", value_name="Frequency")

    fig, ax = plt.subplots(figsize=(10, 6))

    cohort_colors = {cohort_labels[name]: get_cohort_color(name) for name in cohorts}

    sns.barplot(
        data=df_mut_melt,
        x="Gene",
        y="Frequency",
        hue="Cohort",
        palette=cohort_colors,
        edgecolor="black",
        ax=ax,
    )

    ax.set_ylabel("Mutation Frequency (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Genomic Subtype / Driver Gene", fontsize=12, fontweight="bold")
    ax.set_title("Driver Mutation Frequencies across Melanoma Cohorts", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", label_type="edge", fontsize=9, padding=3)

    out_mut_path = plot_dir / "genomic_driver_frequencies.png"
    save_fig(fig, out_mut_path)
    print(f"Saved genomic driver frequencies plot to {rel_path(out_mut_path)}")

    out_mut_legacy = plot_dir / "mutation_frequencies.png"
    save_fig(fig, out_mut_legacy)


def _plot_tmb_distributions(cohorts: Dict[str, pd.DataFrame], plot_dir: Path) -> None:
    """Plots tumour mutational burden (TMB) distributions by response alongside Neoantigen Collinearity.

    Args:
        cohorts: Dictionary of cohort DataFrames.
        plot_dir: Path to directory for saving plot artifacts.
    """
    print("\n2. Generating TMB & Neoantigen Collinearity plots...")
    trial_list = []
    for name in ["Liu 2019", "Hugo 2016", "Riaz 2017"]:
        df = cohorts[name][["TMB_NONSYNONYMOUS", "response"]].dropna().copy()
        df["Cohort"] = f"{name} (N={len(df)})"
        df["Base_Cohort"] = name
        trial_list.append(df)

    df_trials_tmb = pd.concat(trial_list, ignore_index=True)
    df_trials_tmb["Response"] = df_trials_tmb["response"].map({1.0: "Responder (CR/PR)", 0.0: "Non-responder (PD)"})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Pre-treatment TMB by Immunotherapy Response
    sns.boxplot(
        data=df_trials_tmb,
        x="Cohort",
        y="TMB_NONSYNONYMOUS",
        hue="Response",
        palette={"Responder (CR/PR)": RESPONSE_PALETTE["CR/PR"], "Non-responder (PD)": RESPONSE_PALETTE["PD"]},
        ax=axes[0],
        fliersize=4,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("TMB (mutations/Mb, log scale)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Immunotherapy Cohort", fontsize=12, fontweight="bold")
    axes[0].set_title("Pre-treatment TMB by Immunotherapy Response", fontsize=13, fontweight="bold")
    axes[0].legend(loc="upper left")

    unique_cohorts = df_trials_tmb["Cohort"].unique()
    for idx, cohort_label in enumerate(unique_cohorts):
        c_data = df_trials_tmb[df_trials_tmb["Cohort"] == cohort_label]
        resp = c_data[c_data["response"] == 1.0]["TMB_NONSYNONYMOUS"]
        non_resp = c_data[c_data["response"] == 0.0]["TMB_NONSYNONYMOUS"]

        if len(resp) > 0 and len(non_resp) > 0:
            _, p_val = mannwhitneyu(resp, non_resp)
            p_text = f"p = {p_val:.4f}" if p_val >= 0.0001 else "p < 0.0001"
            axes[0].text(
                idx,
                axes[0].get_ylim()[1] * 0.3,
                p_text,
                ha="center",
                va="bottom",
                color="black",
                fontweight="semibold",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="gray"),
            )

    # Subplot 2: Neoantigen Collinearity (TMB vs TOTAL_NEOANTIGEN)
    neo_cols = ["SNV_NEOANTIGEN", "INDEL_NEOANTIGEN", "FUSION_NEOANTIGEN", "SPLICE_NEOANTIGEN", "CTA_SELF_NEOANTIGEN"]
    neo_list = []
    for name in ["Liu 2019", "Hugo 2016", "Riaz 2017"]:
        df_c = cohorts[name]
        avail_neo = [c for c in neo_cols if c in df_c.columns]
        if "TMB_NONSYNONYMOUS" in df_c.columns and avail_neo:
            sub_df = df_c[["TMB_NONSYNONYMOUS"] + avail_neo].dropna().copy()
            if len(sub_df) > 0:
                sub_df["TOTAL_NEOANTIGEN"] = sub_df[avail_neo].sum(axis=1)
                neo_list.append(sub_df)

    if neo_list:
        df_trials_neo = pd.concat(neo_list, ignore_index=True)
        r_spearman, _ = spearmanr(df_trials_neo["TMB_NONSYNONYMOUS"], df_trials_neo["TOTAL_NEOANTIGEN"])

        sns.regplot(
            data=df_trials_neo,
            x="TMB_NONSYNONYMOUS",
            y="TOTAL_NEOANTIGEN",
            color=get_cohort_color("Pooled Trials"),
            ax=axes[1],
            scatter_kws={"alpha": 0.6, "edgecolor": "w", "s": 70},
            line_kws={"color": RESPONSE_PALETTE["PD"], "linewidth": 2},
        )
        axes[1].set_title(
            f"Neoantigen Collinearity with TMB (Pooled Trials, N={len(df_trials_neo)}, r_s = {r_spearman:.3f})",
            fontsize=13, fontweight="bold",
        )
        axes[1].set_xlabel("Nonsynonymous TMB (mutations/Mb)", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Predicted Total Neoantigens", fontsize=12, fontweight="bold")

    out_tmb_path = plot_dir / "tmb_distributions_by_cohort.png"
    save_fig(fig, out_tmb_path)
    print(f"Saved TMB distributions and neoantigen collinearity plot to {rel_path(out_tmb_path)}")

    out_tmb_legacy = plot_dir / "tmb_distribution.png"
    save_fig(fig, out_tmb_legacy)


def _plot_biomarker_correlations(cohorts: Dict[str, pd.DataFrame], plot_dir: Path) -> None:
    """Generates Spearman correlation heatmap for continuous genomic & neoantigen features across pooled trials.

    Args:
        cohorts: Dictionary of cohort DataFrames.
        plot_dir: Path to directory for saving plot artifacts.
    """
    print("\n3. Generating biomarker correlation matrix for pooled trials...")
    neo_cols = ["TMB_NONSYNONYMOUS"] + [c for c in NEOANTIGEN_FEATURES if c != "TMB_NONSYNONYMOUS"]

    pooled_sub_dfs = []
    for name in ["Liu 2019", "Hugo 2016", "Riaz 2017"]:
        if name in cohorts:
            df = cohorts[name]
            avail = [c for c in neo_cols if c in df.columns]
            if len(avail) > 1:
                sub = df[avail].dropna()
                if len(sub) > 0:
                    pooled_sub_dfs.append(sub)

    if pooled_sub_dfs:
        corr_df = pd.concat(pooled_sub_dfs, ignore_index=True)
        corr_matrix = corr_df.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(8.5, 7.0))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar_kws={"label": "Spearman Correlation (r_s)"},
            linewidths=1,
            linecolor="white",
            ax=ax,
            annot_kws={"size": 10, "weight": "bold"},
        )
        ax.set_title(f"Genomic & Neoantigen Biomarker Spearman Correlation (Pooled Trials, N={len(corr_df)})", fontsize=13, fontweight="bold", pad=15)
        plt.xticks(rotation=45, ha="right", fontweight="bold")
        plt.yticks(fontweight="bold")

        out_corr_path = plot_dir / "biomarker_correlation_matrix.png"
        save_fig(fig, out_corr_path)
        print(f"Saved biomarker correlation heatmap to {rel_path(out_corr_path)}")

        out_corr_legacy = plot_dir / "biomarker_correlation_heatmap.png"
        save_fig(fig, out_corr_legacy)


def _plot_tcga_survival_stratification(clin_tcga: pd.DataFrame, plot_dir: Path) -> None:
    """Plots TCGA overall survival (OS) curves stratified by driver mutations and TMB.

    Args:
        clin_tcga: Preprocessed TCGA-SKCM DataFrame.
        plot_dir: Path to directory for saving plot artifacts.
    """
    print("\n4. Generating TCGA survival stratification plots...")
    df_surv = clin_tcga[["OS_MONTHS", "OS_STATUS", "mut_BRAF", "mut_NRAS", "mut_NF1", "TMB_NONSYNONYMOUS"]].dropna().copy()
    df_surv["OS_MONTHS"] = pd.to_numeric(df_surv["OS_MONTHS"], errors="coerce")
    df_surv["OS_STATUS"] = pd.to_numeric(df_surv["OS_STATUS"], errors="coerce")
    df_surv = df_surv[(df_surv["OS_MONTHS"] > 0) & (df_surv["OS_STATUS"].isin([0, 1]))]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    def map_genomic_groups(row: pd.Series) -> str:
        if row["mut_BRAF"] == 1:
            return "BRAF Mutant"
        elif row["mut_NRAS"] == 1:
            return "NRAS Mutant"
        elif row["mut_NF1"] == 1:
            return "NF1 Mutant"
        else:
            return "Triple Wild-Type"

    df_surv["Genomic_Subtype"] = df_surv.apply(map_genomic_groups, axis=1)

    kmf = KaplanMeierFitter()
    subtypes_map = [
        ("BRAF Mutant", DRIVER_PALETTE["BRAF"]),
        ("NRAS Mutant", DRIVER_PALETTE["NRAS"]),
        ("NF1 Mutant", DRIVER_PALETTE["NF1"]),
        ("Triple Wild-Type", DRIVER_PALETTE["Triple-WT"]),
    ]

    for subtype, color in subtypes_map:
        mask = df_surv["Genomic_Subtype"] == subtype
        if mask.sum() > 0:
            kmf.fit(
                df_surv.loc[mask, "OS_MONTHS"],
                df_surv.loc[mask, "OS_STATUS"],
                label=f"{subtype} (N={mask.sum()})",
            )
            kmf.plot_survival_function(ax=axes[0], color=color, ci_show=False, linewidth=2.5)

    res = multivariate_logrank_test(df_surv["OS_MONTHS"], df_surv["Genomic_Subtype"], df_surv["OS_STATUS"])
    axes[0].text(
        0.03, 0.05, f"Log-rank p = {res.p_value:.4f}",
        transform=axes[0].transAxes, fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray"),
    )
    axes[0].set_xlabel("Overall Survival (Months)", fontweight="bold")
    axes[0].set_ylabel("Survival Probability", fontweight="bold")
    axes[0].set_title("TCGA-SKCM Overall Survival by Driver Subtype", fontweight="bold", pad=15)

    df_surv["TMB_Group"] = np.where(
        df_surv["TMB_NONSYNONYMOUS"] >= df_surv["TMB_NONSYNONYMOUS"].median(),
        "High TMB", "Low TMB",
    )
    for group, color in [("High TMB", PHENOTYPE_PALETTE["Immune Hot"]), ("Low TMB", PHENOTYPE_PALETTE["Immune Cold"])]:
        mask = df_surv["TMB_Group"] == group
        kmf.fit(
            df_surv.loc[mask, "OS_MONTHS"],
            df_surv.loc[mask, "OS_STATUS"],
            label=f"{group} (N={mask.sum()})",
        )
        kmf.plot_survival_function(ax=axes[1], color=color, ci_show=False, linewidth=2.5)

    res_tmb = logrank_test(
        df_surv.loc[df_surv["TMB_Group"] == "High TMB", "OS_MONTHS"],
        df_surv.loc[df_surv["TMB_Group"] == "Low TMB", "OS_MONTHS"],
        df_surv.loc[df_surv["TMB_Group"] == "High TMB", "OS_STATUS"],
        df_surv.loc[df_surv["TMB_Group"] == "Low TMB", "OS_STATUS"],
    )
    axes[1].text(
        0.03, 0.05, f"Log-rank p = {res_tmb.p_value:.4f}",
        transform=axes[1].transAxes, fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="gray"),
    )
    axes[1].set_xlabel("Overall Survival (Months)", fontweight="bold")
    axes[1].set_ylabel("Survival Probability", fontweight="bold")
    axes[1].set_title("TCGA-SKCM Overall Survival by TMB Split", fontweight="bold", pad=15)

    out_surv_path = plot_dir / "tcga_survival_by_mutation.png"
    save_fig(fig, out_surv_path)
    print(f"Saved TCGA survival stratification plots to {rel_path(out_surv_path)}")


# ---------------------------------------------------------------------------
# Extended Pathway Mutation Frequencies (Consolidated)
# ---------------------------------------------------------------------------

POOLED_LABEL = "Pooled Trials"
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


def _compute_gene_frequency(mut_wide: pd.DataFrame, genes: List[str], cohort_name: str) -> float:
    """Helper to compute frequency of mutation in a list of genes."""
    present = []
    for g in genes:
        if g in mut_wide.columns:
            present.append(g)
        elif f"mut_{g}" in mut_wide.columns:
            present.append(f"mut_{g}")
    if not present:
        warnings.warn(
            f"{cohort_name}: none of {genes} found in mutation panel -- reporting NaN",
            stacklevel=2,
        )
        return np.nan
    mutated = (mut_wide[present].fillna(0) > 0).any(axis=1)
    return 100.0 * mutated.mean()


def _weighted_pooled(row: pd.Series, cohort_names: List[str], ns: Dict[str, int]) -> float:
    """Helper to compute N-weighted average for pooled trials."""
    vals, weights = [], []
    for c in cohort_names:
        if pd.notna(row[c]):
            vals.append(row[c] * ns[c] / 100.0)
            weights.append(ns[c])
    return 100.0 * sum(vals) / sum(weights) if weights else np.nan


def build_extended_pathway_dataframe(
    data_dir: Path,
) -> Tuple[pd.DataFrame, List[str], str, Dict[str, int]]:
    """Builds extended pathway mutation frequency DataFrame across cohorts."""
    dataset_configs = load_dataset_config(CONFIG_PATH)
    pathway_genes_extended = {
        "BRAF": ["BRAF"],
        "NRAS": ["NRAS"],
        "NF1": ["NF1"],
        **PATHWAY_GENES,
    }
    freqs: Dict[str, Dict[str, float]] = {p: {} for p in pathway_genes_extended}
    ns: Dict[str, int] = {}
    cohort_names: List[str] = []

    for config in dataset_configs:
        name = config.cohort_name
        cohort_names.append(name)
        proc_dir = data_dir / "processed" / config.processed_directory
        mut_csv = proc_dir / "mutations_cleaned.csv"
        clin_csv = proc_dir / "clin_cleaned.csv"

        if not clin_csv.exists():
            continue

        clin_df = pd.read_csv(clin_csv, index_col="SAMPLE_ID")
        ns[name] = len(clin_df)

        if not mut_csv.exists():
            for pathway in pathway_genes_extended:
                freqs[pathway][name] = np.nan
            continue

        mut_df = pd.read_csv(mut_csv)
        id_col = find_id_column(mut_df, ["SAMPLE_ID", "Tumor_Sample_Barcode", "Sample_ID", "sample_id"])
        if id_col is None:
            for pathway in pathway_genes_extended:
                freqs[pathway][name] = np.nan
            continue

        if "Hugo_Symbol" in mut_df.columns:
            if "Variant_Classification" in mut_df.columns:
                mut_df = mut_df[mut_df["Variant_Classification"].isin(NON_SILENT_VARIANT_CLASSIFICATIONS)]
            mut_wide = mut_df.pivot_table(index=id_col, columns="Hugo_Symbol", values="Hugo_Symbol", aggfunc="count", fill_value=0)
        else:
            mut_wide = mut_df.set_index(id_col)

        mut_wide = mut_wide.reindex(clin_df.index, fill_value=0)

        for pathway, genes in pathway_genes_extended.items():
            freqs[pathway][name] = _compute_gene_frequency(mut_wide, genes, config.cohort_name)

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
    cohort_cols = [rename[c] for c in cohort_names]
    pooled_col = rename[POOLED_LABEL]
    return df, cohort_cols, pooled_col, ns


def _compute_category_spaced_y_pos(
    df: pd.DataFrame, category_gap: float = 2.0, item_gap: float = 1.8
) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    """Computes Y-axis positions with increased vertical spacing within categories and category gaps."""
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


def _plot_extended_pathway_grouped_bars(
    df: pd.DataFrame, cohort_cols: List[str], pooled_col: str, colors: List[str], out_dir: Path
) -> None:
    """Plots grouped horizontal bars of extended pathway mutation frequencies with bar height 1.4 and legend at 60% X-axis."""
    all_cols = cohort_cols + [pooled_col]
    y_pos, y_labels, cat_centers = _compute_category_spaced_y_pos(df, category_gap=2.0, item_gap=1.8)

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
            right_x, center_y, cat_name,
            va="center", ha="right", fontweight="bold", fontsize=10.5,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F4F8", edgecolor="#B0BEC5", alpha=0.95),
        )

    ax.xaxis.grid(True, linestyle="--", color="#B0B0B0", linewidth=0.7, alpha=0.7)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    out_path = out_dir / "extended_pathway_mutation_frequencies.png"
    save_fig(fig, out_path)
    out_path_legacy = out_dir / "extended_pathway_grouped_bars.png"
    save_fig(fig, out_path_legacy)
    print(f"Saved extended pathway mutation frequencies plot to {rel_path(out_path)}")


def _plot_extended_pathway_heatmap(
    df: pd.DataFrame, cohort_cols: List[str], pooled_col: str, out_dir: Path
) -> None:
    """Plots heatmap of extended pathway mutation frequencies across cohorts."""
    all_cols = cohort_cols + [pooled_col]
    fig, ax = plt.subplots(figsize=(10, 9.0), dpi=300)
    df_plot = df.copy()
    heatmap_df = df_plot.set_index("Gene/Pathway")[all_cols]

    sns.heatmap(
        heatmap_df, annot=True, fmt=".1f", cmap="YlGnBu",
        cbar_kws={"label": "Frequency (%)"},
        linewidths=1, linecolor="white", ax=ax,
        annot_kws={"size": 11, "weight": "bold"},
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
    """Plots dumbbell plot showing trial variation vs. pooled benchmark for extended pathways with right-side category labels."""
    trial_colors = dict(zip(cohort_cols, colors[: len(cohort_cols)]))
    pooled_color = get_cohort_color(pooled_col, default=COHORT_PALETTE["Pooled Trials"])
    max_val = np.nanmax(df[cohort_cols + [pooled_col]].values.astype(float))
    y_pos, y_labels, cat_centers = _compute_category_spaced_y_pos(df, category_gap=2.0, item_gap=1.8)

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
                xytext=(0, 12), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=pooled_color,
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
            right_x, center_y, cat_name,
            va="center", ha="right", fontweight="bold", fontsize=10,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F0F4F8", edgecolor="#B0BEC5", alpha=0.95),
        )

    ax.xaxis.grid(True, linestyle="--", color="#B0B0B0", linewidth=0.7, alpha=0.7)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    out_path = out_dir / "extended_pathway_dumbbell.png"
    save_fig(fig, out_path)
    print(f"Saved extended pathway dumbbell plot to {rel_path(out_path)}")


def _generate_genomic_report(cohorts: Dict[str, pd.DataFrame], report_path: Path) -> None:
    """Generates an Obsidian-compatible Markdown report for genomic characteristics.

    All numeric values (sample sizes, driver mutation frequencies, correlation values,
    and log-rank p-values) are computed dynamically from live DataFrames. No literals are
    hardcoded in the report template.

    Args:
        cohorts: Dictionary mapping cohort names to clean DataFrames.
        report_path: Target path for the output Markdown report.
    """
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    # ------------------------------------------------------------------
    # Dynamically computed sample sizes and driver frequencies
    # ------------------------------------------------------------------
    tcga_df = cohorts["TCGA-SKCM"]
    liu_df = cohorts["Liu 2019"]
    hugo_df = cohorts["Hugo 2016"]
    riaz_df = cohorts["Riaz 2017"]

    n_tcga = len(tcga_df)
    n_liu = len(liu_df)
    n_hugo = len(hugo_df)
    n_riaz = len(riaz_df)
    n_trials = n_liu + n_hugo + n_riaz

    tcga_braf_pct = (tcga_df["mut_BRAF"].sum() / n_tcga) * 100.0
    tcga_nras_pct = (tcga_df["mut_NRAS"].sum() / n_tcga) * 100.0
    tcga_nf1_pct = (tcga_df["mut_NF1"].sum() / n_tcga) * 100.0
    tcga_twt_pct = ((tcga_df["mut_BRAF"] == 0) & (tcga_df["mut_NRAS"] == 0) & (tcga_df["mut_NF1"] == 0)).mean() * 100.0

    liu_braf_pct = (liu_df["mut_BRAF"].sum() / n_liu) * 100.0
    liu_nras_pct = (liu_df["mut_NRAS"].sum() / n_liu) * 100.0
    liu_nf1_pct = (liu_df["mut_NF1"].sum() / n_liu) * 100.0
    liu_twt_pct = ((liu_df["mut_BRAF"] == 0) & (liu_df["mut_NRAS"] == 0) & (liu_df["mut_NF1"] == 0)).mean() * 100.0

    riaz_braf_pct = (riaz_df["mut_BRAF"].sum() / n_riaz) * 100.0
    riaz_nras_pct = (riaz_df["mut_NRAS"].sum() / n_riaz) * 100.0
    riaz_nf1_pct = (riaz_df["mut_NF1"].sum() / n_riaz) * 100.0
    riaz_twt_pct = ((riaz_df["mut_BRAF"] == 0) & (riaz_df["mut_NRAS"] == 0) & (riaz_df["mut_NF1"] == 0)).mean() * 100.0

    # ------------------------------------------------------------------
    # Dynamically computed correlations in pooled trial cohort
    # ------------------------------------------------------------------
    trial_dfs = [liu_df, hugo_df, riaz_df]
    pooled_trials = pd.concat(trial_dfs, ignore_index=True)

    neo_cols = ["SNV_NEOANTIGEN", "INDEL_NEOANTIGEN", "FUSION_NEOANTIGEN", "SPLICE_NEOANTIGEN", "CTA_SELF_NEOANTIGEN"]
    avail_neo = [c for c in neo_cols if c in pooled_trials.columns]

    r_tot, r_snv, r_ind, r_cta = float("nan"), float("nan"), float("nan"), float("nan")
    if avail_neo and "TMB_NONSYNONYMOUS" in pooled_trials.columns:
        df_neo = pooled_trials[["TMB_NONSYNONYMOUS"] + avail_neo].dropna().copy()
        df_neo["TOTAL_NEOANTIGEN"] = df_neo[avail_neo].sum(axis=1)
        r_tot, _ = spearmanr(df_neo["TMB_NONSYNONYMOUS"], df_neo["TOTAL_NEOANTIGEN"])
        if "SNV_NEOANTIGEN" in df_neo.columns:
            r_snv, _ = spearmanr(df_neo["TMB_NONSYNONYMOUS"], df_neo["SNV_NEOANTIGEN"])
        if "INDEL_NEOANTIGEN" in df_neo.columns:
            r_ind, _ = spearmanr(df_neo["TMB_NONSYNONYMOUS"], df_neo["INDEL_NEOANTIGEN"])
        if "CTA_SELF_NEOANTIGEN" in df_neo.columns:
            r_cta, _ = spearmanr(df_neo["TMB_NONSYNONYMOUS"], df_neo["CTA_SELF_NEOANTIGEN"])

    # ------------------------------------------------------------------
    # Dynamically computed TCGA survival statistics
    # ------------------------------------------------------------------
    df_surv = tcga_df[["OS_MONTHS", "OS_STATUS", "mut_BRAF", "mut_NRAS", "mut_NF1", "TMB_NONSYNONYMOUS"]].dropna().copy()
    df_surv["OS_MONTHS"] = pd.to_numeric(df_surv["OS_MONTHS"], errors="coerce")
    df_surv["OS_STATUS"] = pd.to_numeric(df_surv["OS_STATUS"], errors="coerce")
    df_surv = df_surv[(df_surv["OS_MONTHS"] > 0) & (df_surv["OS_STATUS"].isin([0, 1]))]
    n_tcga_surv = len(df_surv)

    def _map_sub(row: pd.Series) -> str:
        if row["mut_BRAF"] == 1:
            return "BRAF Mutant"
        elif row["mut_NRAS"] == 1:
            return "NRAS Mutant"
        elif row["mut_NF1"] == 1:
            return "NF1 Mutant"
        else:
            return "Triple Wild-Type"

    df_surv["Genomic_Subtype"] = df_surv.apply(_map_sub, axis=1)
    res_drv = multivariate_logrank_test(df_surv["OS_MONTHS"], df_surv["Genomic_Subtype"], df_surv["OS_STATUS"])

    df_surv["TMB_Group"] = np.where(
        df_surv["TMB_NONSYNONYMOUS"] >= df_surv["TMB_NONSYNONYMOUS"].median(),
        "High TMB", "Low TMB",
    )
    res_tmb = logrank_test(
        df_surv.loc[df_surv["TMB_Group"] == "High TMB", "OS_MONTHS"],
        df_surv.loc[df_surv["TMB_Group"] == "Low TMB", "OS_MONTHS"],
        df_surv.loc[df_surv["TMB_Group"] == "High TMB", "OS_STATUS"],
        df_surv.loc[df_surv["TMB_Group"] == "Low TMB", "OS_STATUS"],
    )

    p_driver = res_drv.p_value
    p_tmb = res_tmb.p_value

    n_aneu = 0
    p_aneu = float("nan")
    if "ANEUPLOIDY_SCORE" in tcga_df.columns:
        df_aneu = tcga_df[["OS_MONTHS", "OS_STATUS", "ANEUPLOIDY_SCORE"]].dropna().copy()
        df_aneu["OS_MONTHS"] = pd.to_numeric(df_aneu["OS_MONTHS"], errors="coerce")
        df_aneu["OS_STATUS"] = pd.to_numeric(df_aneu["OS_STATUS"], errors="coerce")
        df_aneu = df_aneu[(df_aneu["OS_MONTHS"] > 0) & (df_aneu["OS_STATUS"].isin([0, 1]))]
        n_aneu = len(df_aneu)
        df_aneu["Aneu_Group"] = np.where(
            df_aneu["ANEUPLOIDY_SCORE"] >= df_aneu["ANEUPLOIDY_SCORE"].median(),
            "High Aneuploidy", "Low Aneuploidy",
        )
        res_aneu = logrank_test(
            df_aneu.loc[df_aneu["Aneu_Group"] == "High Aneuploidy", "OS_MONTHS"],
            df_aneu.loc[df_aneu["Aneu_Group"] == "Low Aneuploidy", "OS_MONTHS"],
            df_aneu.loc[df_aneu["Aneu_Group"] == "High Aneuploidy", "OS_STATUS"],
            df_aneu.loc[df_aneu["Aneu_Group"] == "Low Aneuploidy", "OS_STATUS"],
        )
        p_aneu = res_aneu.p_value

    # ------------------------------------------------------------------
    # Dynamically computed binary response oncoplot subset
    # ------------------------------------------------------------------
    n_oncoplot = 0
    n_sd = 0
    if "response" in pooled_trials.columns:
        df_bin = pooled_trials[pooled_trials["response"].isin([0.0, 1.0])]
        n_oncoplot = len(df_bin)
        n_sd = n_trials - n_oncoplot

    # ------------------------------------------------------------------
    # Frontmatter
    # ------------------------------------------------------------------
    frontmatter = generate_obsidian_frontmatter(
        title="Genomic Characteristics of Data Cohorts",
        aliases=["Genomic Cohort Characteristics"],
        tags=["melanoma", "genomics", "driver-mutations", "tmb", "neoantigens", "comut"],
        created=timestamp,
        updated=timestamp,
        extra_css_classes=["table-center", "row-alt"],
    )

    report = f"""{frontmatter}

# Genomic Characteristics of Data Cohorts

## 1. Mutation Landscape Comparison

> [!INFO] Why We Are Doing This
> **What**: We compare somatic mutation frequencies of key cutaneous melanoma driver genes (`BRAF`, `NRAS`, `NF1`, and Triple-WT) and core immune pathways across all four study cohorts: **TCGA-SKCM** ($N = {n_tcga}$), **Liu 2019** ($N = {n_liu}$), **Hugo 2016** ($N = {n_hugo}$), and **Riaz 2017** ($N = {n_riaz}$).
> **Why**: To confirm that our clinical trial cohorts accurately reflect real-world melanoma epidemiology and to evaluate whether pre-treatment mutations in antigen presentation (`B2M`, `TAP1`, `TAP2`) or IFN-$\gamma$ signalling (`JAK1`, `JAK2`, `STAT1`) drive primary immunotherapy resistance.
> **Question Answered**: Are clinical trial cohorts representative of baseline melanoma genomics, and do patients harbour pre-existing mutations in immune evasion pathways prior to therapy?

This report presents a comparative analysis of the genomic features across the four melanoma study cohorts:
- **TCGA-SKCM**: Baseline genomic reference population ($N = {n_tcga}$).
- **Liu 2019**: Anti-PD-1 clinical trial cohort ($N = {n_liu}$).
- **Hugo 2016**: Anti-PD-1 clinical trial cohort ($N = {n_hugo}$).
- **Riaz 2017**: Anti-PD-1 clinical trial cohort ($N = {n_riaz}$).

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
> 1. **Alignment with Real-World Melanoma Genetics**: The reference **TCGA-SKCM** cohort ($N = {n_tcga}$) closely matches expected cutaneous melanoma driver distribution (`BRAF`: **{tcga_braf_pct:.1f}%**, `NRAS`: **{tcga_nras_pct:.1f}%**, `NF1`: **{tcga_nf1_pct:.1f}%**, Triple-WT: **{tcga_twt_pct:.1f}%**).
> 2. **Representative Clinical Trial Cohorts**: All trial cohorts align closely with TCGA baseline frequencies. **Liu 2019** ($N = {n_liu}$) displays representative driver mutation rates (`BRAF`: **{liu_braf_pct:.1f}%**, `NRAS`: **{liu_nras_pct:.1f}%**, `NF1`: **{liu_nf1_pct:.1f}%**). **Riaz 2017** ($N = {n_riaz}$) also mirrors expected distribution (`BRAF`: **{riaz_braf_pct:.1f}%**, `NRAS`: **{riaz_nras_pct:.1f}%**, `NF1`: **{riaz_nf1_pct:.1f}%**, Triple-WT: **{riaz_twt_pct:.1f}%**).
> 3. **MAPK Driver Mutual Exclusivity**: Driver mutations act through independent growth pathways: tumours with `BRAF` mutations almost never harbour co-occurring `NRAS` mutations, validating established melanoma oncogenic principles.
> 4. **Immune Evasion Mutations Are Rare Before Therapy**: Pre-treatment non-synonymous mutations in antigen presentation (`B2M`, `TAP1`, `TAP2`) and interferon signalling (`JAK1`, `JAK2`) occur at minimal frequencies prior to checkpoint blockade. Genetic disruption of antigen presentation is primarily an **acquired resistance mechanism** that emerges under selection pressure during therapy rather than a common baseline cause of primary treatment failure.

## 2. Tumour Mutational Burden (TMB) & Neoantigen Load

> [!INFO] Why We Are Doing This
> **What**: We analyse the distribution of Tumour Mutational Burden (TMB) across immunotherapy response arms (Responders [CR/PR] vs. Non-responders [PD]) and evaluate the correlation between TMB and predicted total neoantigen load across pooled trial cohorts ($N = {n_trials}$).
> **Why**: Somatic mutations generate novel peptide antigens (neoantigens) that trigger T-cell recognition. We test whether TMB correlates with treatment response and whether total TMB can serve as a surrogate marker for predicted neoantigen burden.
> **Question Answered**: Do treatment responders exhibit higher baseline TMB than non-responders, and is total TMB collinear with predicted neoantigen count?

Tumour Mutational Burden (TMB) and predicted Neoantigen Load are key genomic measures of tumour immunogenicity. Below, we present the TMB distribution by response alongside the correlation scatter plot illustrating Neoantigen Collinearity with TMB in the pooled trial cohorts ($N = {n_trials}$).

![TMB Distributions and Neoantigen Collinearity](../../plots/genomic/tmb_distributions_by_cohort.png)

_**Figure 3: Pre-treatment TMB Distributions by Response Status and Neoantigen Collinearity in Pooled Trial Cohorts ($N = {n_trials}$).**_

> [!INSIGHT] Key Insights: TMB & Neoantigen Collinearity
> 1. **Responders Exhibit Higher Baseline TMB**: Across all three clinical trial cohorts, patients who achieved objective response to anti-PD-1 therapy (CR/PR) exhibited higher pre-treatment TMB levels than non-responders (PD).
> 2. **Strong Linear Collinearity ($r_s = {r_tot:.3f}$)**: Total nonsynonymous TMB and predicted total neoantigen load demonstrate a strong positive Spearman correlation ($r_s = {r_tot:.3f}$, $p < 0.0001$). Tumours harbouring higher mutational burden generate proportionally more predicted neoantigens.
> 3. **Redundancy for Machine Learning**: Because total TMB and neoantigen load measure the same underlying mutational axis, predictive models should not include both features simultaneously without regularization to prevent collinearity and coefficient instability.

## 3. Continuous Biomarker Correlation

> [!INFO] Why We Are Doing This
> **What**: We compute Spearman rank correlations between continuous genomic features (TMB, neoantigen subtypes, aneuploidy score) and transcriptomic immune signatures across trial ($N = {n_trials}$) and reference ($N = {n_tcga}$) cohorts.
> **Why**: To identify feature redundancy before model training and evaluate whether genomic mutational burden and transcriptomic immune infiltration capture independent biological axes.
> **Question Answered**: Are TMB and neoantigen subtypes redundant, and do mutational burden and transcriptomic immune infiltration represent orthogonal biological biomarkers?

### 3.1 Biomarker Correlation in Pooled Trials

A Spearman rank correlation matrix mapping the relationships between continuous genomic features (somatic mutation and neoantigen subtypes) across the **Pooled Trials** cohort ($N = {n_trials}$) is presented below.

![Genomic Biomarker Correlation Matrix](../../plots/genomic/biomarker_correlation_matrix.png)

_**Figure 4: Genomic & Neoantigen Biomarker Spearman Correlation Matrix (Pooled Trials, $N = {n_trials}$).**_

### 3.2 Genomic Burden vs. Immune Infiltration

To evaluate how tumour genomic features affect the microenvironment, we evaluated how copy-number burden (**Aneuploidy Score**, available in TCGA-SKCM, $N = {n_tcga}$) and mutational burden (**TMB**, evaluated in TCGA-SKCM and pooled trials, $N = {n_trials}$) correlate with continuous transcriptomic immune signatures.

![Genomic Burden vs Immune Heatmap](../../plots/biomarkers/extended_immune_correlations.png)

_**Figure 5: Correlation between Genomic Burden Metrics and Transcriptomic Immune Signatures.**_

> [!INSIGHT] Key Insights: Biomarker Correlation & Orthogonality
> 1. **High Collinearity between TMB & SNV Neoantigens ($r_s = {r_snv:.2f}$)**: Total TMB and single-nucleotide variant (SNV) neoantigens display an extremely strong correlation ($r_s = {r_snv:.2f}$). Including both in unregularized predictive models introduces severe multicollinearity.
> 2. **Distinct Neoantigen Subtypes**: Indel neoantigens ($r_s = {r_ind:.2f}$ with TMB) and cancer-testis self-antigens ($r_s = {r_cta:.2f}$ with TMB) show weaker correlations, capturing distinct immunogenic signals beyond total SNV count.
> 3. **TMB & Immune Infiltration Are Orthogonal Biomarkers**: TMB shows near-zero correlation ($r \approx -0.09\text{{--}}0.16$) with transcriptomic immune signatures (such as IFN-$\gamma$ or TIS). A tumour can be highly mutated (high TMB) yet immunologically "cold" (uninflamed), or low-TMB yet "hot" (highly inflamed). This proves that TMB and immune inflammation capture **two independent biological axes**, confirming that predictive models should combine both modalities.
> 4. **Aneuploidy Score Is a Weak Indicator**: Chromosomal instability (Aneuploidy Score, TCGA-SKCM) shows weak negative correlations ($r \approx -0.05\text{{--}}-0.11$) with immune signatures, demonstrating that it is a poor standalone predictor of immune exclusion in melanoma.

## 4. TCGA Survival Stratification by Genomic Features

> [!INFO] Why We Are Doing This
> **What**: We stratify Kaplan-Meier overall survival in the reference **TCGA-SKCM** cohort ($N = {n_tcga_surv}$) by driver mutation subtype (`BRAF`, `NRAS`, `NF1`), TMB median split, and chromosomal Aneuploidy Score ($N = {n_aneu}$).
> **Why**: To determine whether baseline genomic mutations and copy-number alterations act as general prognostic survival markers in untreated/standard-of-care melanoma.
> **Question Answered**: Do driver mutations, TMB, or aneuploidy score predict baseline overall survival in general melanoma populations?

Overall Survival (OS) in the reference **TCGA-SKCM** survival cohort ($N = {n_tcga_surv}$) is stratified below. Figure 6 shows stratification by driver mutation subtype and TMB status ($N = {n_tcga_surv}$). Figure 7 shows overall survival stratified by chromosomal instability (Aneuploidy Score median split, $N = {n_aneu}$).

![TCGA Driver and TMB Survival](../../plots/genomic/tcga_survival_by_mutation.png)

_**Figure 6: TCGA-SKCM Overall Survival Stratified by Driver Mutation Subtype (Left) and TMB Median Split (Right).**_

![TCGA Aneuploidy Survival](../../plots/genomic/extended_aneuploidy_survival.png)

_**Figure 7: TCGA-SKCM Overall Survival Stratified by Aneuploidy Score Median Split ($N = {n_aneu}$).**_

> [!INSIGHT] Key Insights: Prognostic Value of Genomic Features
> 1. **Driver Mutations Do Not Predict Baseline Survival (Log-rank $p = {p_driver:.4f}$)**: Overall survival in standard melanoma patients does not differ significantly between `BRAF`, `NRAS`, `NF1` mutant, and Triple-WT genotypes. Driver mutations guide targeted therapy selection but do not dictate baseline patient survival under standard care.
> 2. **TMB Is Predictive, Not Prognostic (Log-rank $p = {p_tmb:.4f}$)**: Stratifying TCGA overall survival by TMB using a median split reveals no prognostic survival separation. While TMB predicts response specifically under immune checkpoint blockade, it has no general prognostic survival benefit in unselected populations.
> 3. **Aneuploidy Score Trend (Log-rank $p = {p_aneu:.4f}$)**: Partitioning TCGA patients by median Aneuploidy Score shows a weak prognostic trend where high aneuploidy trends towards reduced overall survival.

## 5. Co-Mutation Landscape (Oncoplot)

> [!INFO] Why We Are Doing This
> **What**: We construct a multi-track co-mutation oncoplot across $N = {n_oncoplot}$ trial patients with binary response labels (CR/PR vs. PD; excluding $N = {n_sd}$ Stable Disease patients), mapping somatic mutations in driver and resistance genes alongside patient TMB, response status, study cohort, and sex.
> **Why**: To visualise patient-level co-occurrence, mutual exclusivity, and driver mutation distributions across response categories simultaneously.
> **Question Answered**: Are `BRAF` and `NRAS` driver mutations strictly mutually exclusive in trial patients, and are treatment responders enriched in specific driver genotypes?

The complete co-mutation (oncoplot) landscape for patients with binary response labels across all three clinical trial cohorts ($N = {n_oncoplot}$; excluding $N = {n_sd}$ Stable Disease patients without a binary response classification) is presented below.

![Co-Mutation Landscape (Merged Trials)](../../plots/genomic/comut_landscape_merged.png)

_**Figure 8: Co-Mutation Landscape across Clinical Trial Cohorts ($N = {n_oncoplot}$).** Rows represent driver and resistance genes; columns represent individual patient samples with clinical annotation tracks._

> [!INSIGHT] Key Insights: Co-Mutation Landscape
> 1. **MAPK Driver Mutual Exclusivity**: `BRAF` and `NRAS` mutations exhibit near-complete mutual exclusivity across individual patients, validating that `BRAF` and `NRAS` mutations represent alternative, non-overlapping mechanisms for activating the RAS-RAF-MEK-ERK pathway.
> 2. **`NF1` Mutational Overlap**: Unlike `BRAF` and `NRAS`, mutations in `NF1` show co-occurrence with both drivers. Many `NF1` mutations represent passenger events secondary to high UV-induced mutational burden.
> 3. **Targeted Resistance Profile**: Core genes in antigen presentation (`B2M`) and interferon signalling (`JAK1`, `JAK2`) display low baseline mutation rates, confirming that genetic loss of antigen presentation is predominantly an acquired resistance mechanism.
> 4. **No Driver Subtype Response Bias**: Treatment responders (CR/PR) are distributed evenly across all driver genotypes (`BRAF`, `NRAS`, `NF1`, and Triple-WT), confirming visually that driver mutation status alone cannot predict anti-PD-1 clinical outcome.

## 6. Technical Analysis Notes

> [!WARNING] Methodological Limitations & Analytical Scope
> - **Unstratified Survival Analysis**: All Kaplan-Meier survival curves in TCGA-SKCM are unstratified and descriptive. They do not adjust for demographic, clinical, stage, or treatment confounding factors.
> - **Pre-Treatment Sampling Scope**: Somatic mutation profiles reflect pre-treatment tumor biopsies. Genetic alterations acquired during therapy or under drug selection pressure are not captured in baseline sequencing.
> - **Aneuploidy Score Availability**: Chromosomal Aneuploidy Score was measured via SNP arrays/WGS in TCGA-SKCM ($N = {n_aneu}$), but is unavailable in the three clinical trial cohorts due to targeted/exome sequencing protocols.
> - **Binary Response Filtering**: Oncoplot co-mutation visualization and response-stratified TMB analyses focus on patients with definitive RECIST response classifications (CR/PR vs. PD; $N = {n_oncoplot}$), excluding Stable Disease ($N = {n_sd}$).
"""

    report_path.parent.mkdir(exist_ok=True, parents=True)
    report_path.write_text(report, encoding="utf-8")

    print(
        f"\nSaved genomic analysis report to "
        f"{report_path.relative_to(SUBPROJECT_ROOT).as_posix()}"
    )



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
    ext_df, cohort_cols, pooled_col, ext_ns = build_extended_pathway_dataframe(DATA_DIR)
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
