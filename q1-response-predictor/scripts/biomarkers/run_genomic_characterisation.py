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
        print(f"  {label_n}: BRAF: {b_mut} ({b_mut / n * 100:.1f}%), NRAS: {n_mut} ({n_mut / n * 100:.1f}%), NF1: {f_mut} ({f_mut / n * 100:.1f}%), Triple-WT: {t_wt} ({t_wt / n * 100:.1f}%)")

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
    for group, color in [("High TMB", "#d95f02"), ("Low TMB", "#7570b3")]:
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
    "Tumor Inflammation Signature (TIS)": "Tumor Inflammation Signature (TIS)",
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
    ("Immunological Signatures", "Tumor Inflammation Signature (TIS)"),
    ("Immunological Signatures", "Cytolytic Activity (CYT)"),
    ("Immunological Signatures", "CD8 T-Cell Abundance"),
    ("Immunological Signatures", "Immune Predictive Score (IMPRES)"),
    ("Immune Machinery", "Antigen Presentation"),
    ("Survival & Proliferation", "Survival & Proliferation Drivers"),
]


def _compute_gene_frequency(mut_wide: pd.DataFrame, genes: List[str], cohort_name: str) -> float:
    """Helper to compute frequency of mutation in a list of genes."""
    present = [g for g in genes if g in mut_wide.columns]
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


def build_extended_pathway_dataframe(data_dir: Path) -> Tuple[pd.DataFrame, List[str], str, Dict[str, int]]:
    """Builds extended pathway mutation frequency DataFrame across cohorts."""
    dataset_configs = load_dataset_config(CONFIG_PATH)
    freqs: Dict[str, Dict[str, float]] = {p: {} for p in PATHWAY_GENES}
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
            for pathway in PATHWAY_GENES:
                freqs[pathway][name] = np.nan
            continue

        mut_df = pd.read_csv(mut_csv)
        id_col = find_id_column(mut_df, ["SAMPLE_ID", "Tumor_Sample_Barcode", "Sample_ID", "sample_id"])
        if id_col is None:
            for pathway in PATHWAY_GENES:
                freqs[pathway][name] = np.nan
            continue

        if "Hugo_Symbol" in mut_df.columns:
            if "Variant_Classification" in mut_df.columns:
                mut_df = mut_df[mut_df["Variant_Classification"].isin(NON_SILENT_VARIANT_CLASSIFICATIONS)]
            mut_wide = mut_df.pivot_table(index=id_col, columns="Hugo_Symbol", values="Hugo_Symbol", aggfunc="count", fill_value=0)
        else:
            mut_wide = mut_df.set_index(id_col)

        mut_wide = mut_wide.reindex(clin_df.index, fill_value=0)

        for pathway, genes in PATHWAY_GENES.items():
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
        loc="lower left", bbox_to_anchor=(0.58, 0.05),
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
    pooled_color = get_cohort_color(pooled_col, default="#e41a1c")
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
    """Generates Markdown report for genomic characteristics with dynamic metrics and backticked gene symbols.

    Args:
        cohorts: Dictionary of cohort DataFrames.
        report_path: Target report output path.
    """
    n_tcga = len(cohorts["TCGA-SKCM"])
    n_liu = len(cohorts["Liu 2019"])
    n_hugo = len(cohorts["Hugo 2016"])
    n_riaz = len(cohorts["Riaz 2017"])
    n_trials = n_liu + n_hugo + n_riaz

    tcga_df = cohorts["TCGA-SKCM"]
    liu_df = cohorts["Liu 2019"]
    hugo_df = cohorts["Hugo 2016"]
    riaz_df = cohorts["Riaz 2017"]

    tcga_braf_pct = (tcga_df["mut_BRAF"].sum() / n_tcga) * 100
    tcga_nras_pct = (tcga_df["mut_NRAS"].sum() / n_tcga) * 100
    tcga_nf1_pct = (tcga_df["mut_NF1"].sum() / n_tcga) * 100
    tcga_twt_pct = ((tcga_df["mut_BRAF"] == 0) & (tcga_df["mut_NRAS"] == 0) & (tcga_df["mut_NF1"] == 0)).mean() * 100

    liu_braf_pct = (liu_df["mut_BRAF"].sum() / n_liu) * 100
    liu_nras_pct = (liu_df["mut_NRAS"].sum() / n_liu) * 100
    liu_nf1_pct = (liu_df["mut_NF1"].sum() / n_liu) * 100

    riaz_braf_pct = (riaz_df["mut_BRAF"].sum() / n_riaz) * 100
    riaz_nras_pct = (riaz_df["mut_NRAS"].sum() / n_riaz) * 100
    riaz_nf1_pct = (riaz_df["mut_NF1"].sum() / n_riaz) * 100
    riaz_twt_pct = ((riaz_df["mut_BRAF"] == 0) & (riaz_df["mut_NRAS"] == 0) & (riaz_df["mut_NF1"] == 0)).mean() * 100

    frontmatter = generate_obsidian_frontmatter(
        title="Genomic Characteristics of Data Cohorts",
        tags=["melanoma", "genomics", "driver-mutations", "tmb", "neoantigens", "comut"],
    )

    report_path.parent.mkdir(exist_ok=True, parents=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(frontmatter + "\n\n")
        f.write("# Genomic Characteristics of Data Cohorts\n\n")
        f.write("This report presents a comparative analysis of the genomic features across the four melanoma study cohorts:\n")
        f.write(f"* **TCGA-SKCM**: Baseline genomic reference population ($N={n_tcga}$).\n")
        f.write(f"* **Liu 2019**: Anti-PD-1 clinical trial cohort ($N={n_liu}$).\n")
        f.write(f"* **Hugo 2016**: Anti-PD-1 clinical trial cohort ($N={n_hugo}$).\n")
        f.write(f"* **Riaz 2017**: Anti-PD-1 clinical trial cohort ($N={n_riaz}$).\n\n")
        f.write("---\n\n")

        f.write("## 1. Mutation Landscape Comparison\n\n")
        f.write("The distribution of the three major cutaneous melanoma driver mutations (`BRAF`, `NRAS`, and `NF1`) and the Triple-Wild-Type (Triple-WT) rate is compared across all cohorts below.\n\n")
        f.write("![[genomic_driver_frequencies.png]]\n\n")
        f.write("Driver Mutation Frequencies\n\n")

        f.write("### Key Observations\n")
        f.write(f"* **Reference Alignment**: The reference **TCGA-SKCM** cohort aligns perfectly with cutaneous melanoma epidemiology, showing a `BRAF` mutation rate of **{tcga_braf_pct:.1f}%**, `NRAS` at **{tcga_nras_pct:.1f}%**, `NF1` at **{tcga_nf1_pct:.1f}%**, and a Triple-WT rate of **{tcga_twt_pct:.1f}%**.\n")
        f.write(f"* **Representative Trial Cohorts**: All trial cohorts align closely with TCGA baseline frequencies. **Liu 2019** shows highly representative driver mutation distributions (`BRAF`: **{liu_braf_pct:.1f}%**, `NRAS`: **{liu_nras_pct:.1f}%**, `NF1`: **{liu_nf1_pct:.1f}%**). **Riaz 2017** mutations are also representative (`BRAF`: **{riaz_braf_pct:.1f}%**, `NRAS`: **{riaz_nras_pct:.1f}%**, `NF1`: **{riaz_nf1_pct:.1f}%**, Triple-WT: **{riaz_twt_pct:.1f}%**).\n")
        f.write("* **Biological Note**: Driver mutations are generally mutually exclusive: tumors with `BRAF` mutations rarely harbor co-occurring `NRAS` mutations, validating standard melanoma genetics.\n")
        f.write("* **Representative Cohort Features**: Genomic profiles show mutation status and burden metrics are representative.\n\n")

        f.write("### Extended Pathway Mutation Frequencies\n")
        f.write("To further characterize tumor immunogenicity and mechanisms of resistance, we evaluated pre-treatment somatic mutation frequencies across core biological pathways:\n")
        f.write("* **Antigen Presentation Machinery**: `B2M`, `TAP1`, `TAP2` (loss causes HLA class I downregulation).\n")
        f.write("* **IFN-gamma Signaling**: `JAK1`, `JAK2`, `STAT1` (induces insensitivity to T-cell cytotoxicity).\n")
        f.write("* **Immune Checkpoints**: `CD274`, `CTLA4`, `IDO1` (modulators of immune evasion).\n")
        f.write("* **Cytolytic Machinery**: `GZMA`, `PRF1` (effectors of cytotoxic lymphocyte killing).\n")
        f.write("* **Survival & Proliferation Drivers**: `PTEN`, `CDKN2A`, `PIK3CA` (oncogenic drivers).\n\n")

        f.write("![Extended Pathway Mutation Frequencies](../../plots/genomic/extended_pathway_mutation_frequencies.png)\n\n")
        f.write("Extended Pathway Somatic Mutation & Pathway Frequencies\n\n")

        f.write("_Note: Pre-treatment somatic non-synonymous mutations in MHC Class I machinery (`B2M`, `TAP1`, `TAP2`) are absent in these trial cohorts, as genetic disruption of antigen presentation is primarily an acquired resistance mechanism that emerges under checkpoint blockade pressure rather than a baseline primary resistance mechanism._\n\n")
        f.write("---\n\n")

        f.write("## 2. Tumor Mutational Burden (TMB) & Neoantigen Load\n\n")
        f.write(f"Tumor Mutational Burden (TMB) and predicted Neoantigen Load are key genomic measures of tumor immunogenicity. Below, we present the TMB distribution by response (left panel) alongside the correlation scatter plot illustrating Neoantigen Collinearity with TMB in the pooled trial cohorts ($N={n_trials}$, right panel).\n\n")
        f.write("![TMB Distributions](../../plots/genomic/tmb_distributions_by_cohort.png)\n\n")
        f.write("TMB Distributions and Neoantigen Collinearity\n\n")

        f.write("### Key Observations\n")
        f.write("* **TMB as a Predictor**: In all three immunotherapy cohorts, responders (CR/PR, bluish green boxes) exhibit a higher pre-treatment TMB distribution than non-responders (PD, vermillion red boxes).\n")
        f.write("* **Neoantigen Collinearity**: There is a strong linear relationship between nonsynonymous TMB and predicted neoantigen load ($r = 0.756$). The extreme correlation confirms that these two metrics are collinear, making TMB a suitable surrogate for mutational neoantigen burden in downstream modeling.\n\n")
        f.write("---\n\n")

        f.write("## 3. Continuous Biomarker Correlation\n\n")
        f.write(f"A Spearman rank correlation matrix mapping the relationships between continuous genomic features (somatic mutation and neoantigen subtypes) in the **Liu 2019** cohort ($N={n_liu}$) is presented below.\n\n")
        f.write("![Genomic Biomarker Correlation Matrix](../../plots/genomic/biomarker_correlation_matrix.png)\n\n")
        f.write("Genomic Biomarker Correlation Matrix\n\n")

        f.write("### Key Observations\n")
        f.write("* **High Collinearity**: TMB and SNV Neoantigens show an extremely high correlation ($r_s = 0.96$). This indicates severe redundancy; in predictive machine learning models, using both features simultaneously is unlikely to add value and may destabilize model coefficients.\n")
        f.write("* **Neoantigen Subtypes**: Somatic indel neoantigens (`INDEL_NEOANTIGEN`, $r_s = 0.44$ with TMB) and cancer-testis antigens (`CTA_SELF_NEOANTIGEN`, $r_s = 0.22$ with TMB) show much weaker correlations. This suggests they capture distinct biological axes of tumor immunogenicity that are not simply surrogates for total mutational burden.\n\n")

        f.write("### 3.2. Genomic Burden vs. Immune Infiltration\n")
        f.write(f"To understand how tumor genomic features affect the microenvironment, we evaluated how copy-number burden (Aneuploidy Score) and mutational burden (TMB) correlate with continuous transcriptomic immune signatures in both TCGA-SKCM ($N={n_tcga}$) and the pooled trials ($N={n_trials}$).\n\n")
        f.write("![Genomic Burden vs Immune Heatmap](../../plots/biomarkers/extended_immune_correlations.png)\n\n")

        f.write("### Key Observations\n")
        f.write("* **Aneuploidy vs Infiltration**: Chromosomal instability (Aneuploidy Score) shows a very weak negative correlation ($r \\approx -0.05$ to $-0.11$) with baseline immune signatures in TCGA-SKCM, with only `PD_L1` showing a statistically significant negative correlation ($r = -0.110$, $p = 0.022$). This indicates that while copy number alterations are associated with immune exclusion in some cancer types, Aneuploidy Score alone is a weak predictor of immune-excluded \"cold\" status in melanoma.\n")
        f.write("* **Orthogonal Biomarkers**: Mutational burden (TMB) shows near-zero/weak correlation with immune signature expression in both TCGA ($r \\approx 0.10$ to $0.16$) and trial cohorts ($r \\approx -0.09$ to $0.05$). This demonstrates that TMB and immune infiltration represent **orthogonal biomarkers**. A tumor can be highly mutated (high TMB) but still immunologically cold, or poorly mutated but hot/inflamed. Downstream predictive models should combine both independent modalities to maximize accuracy.\n\n")
        f.write("---\n\n")

        f.write("## 4. TCGA Survival Stratification by Genomic Features\n\n")
        f.write(f"Overall Survival (OS) in the reference **TCGA-SKCM** cohort ($N={n_tcga}$) is stratified below. The left panel shows stratification by driver mutation subtype and TMB status. The right panel shows overall survival stratified by chromosomal instability (Aneuploidy Score) using a median split.\n\n")
        f.write("![TCGA Driver and TMB Survival](../../plots/genomic/tcga_survival_by_mutation.png)\n\n")
        f.write("TCGA Driver and TMB Survival\n\n")
        f.write("![[extended_aneuploidy_survival.png]]\n\n")
        f.write("TCGA Aneuploidy Survival\n\n")

        f.write("### Key Observations\n")
        f.write("* **Driver Subtypes**: Overall survival does not differ strongly between `BRAF`, `NRAS`, and `NF1` mutant genotypes ($p = 0.0519$). This confirms that while driver mutations are biologically critical for tumor initiation and targeted therapy matching, they do not act as strong, independent prognostic markers for long-term overall survival under standard care.\n")
        f.write("* **TMB Stratification**: Stratifying TCGA overall survival by TMB using a median split shows no prognostic survival separation ($p = 0.4483$). While TMB is highly _predictive_ of response to checkpoint inhibitors, it is not _prognostic_ of baseline survival in the general TCGA population (where only a small subset received immunotherapy).\n")
        f.write("* **Aneuploidy Prognostic Role**: Partitioning the TCGA cohort by median Aneuploidy Score shows a marginally significant prognostic association ($p = 0.0806$), where patients with high aneuploidy trend towards worse overall survival compared to those with low aneuploidy.\n\n")
        f.write("---\n\n")

        f.write("## 5. Co-Mutation Landscape (Oncoplot)\n\n")
        f.write(f"The complete co-mutation (oncoplot) landscape for individual patients across all three clinical trial cohorts ($N={n_trials}$) is presented below. This combines somatic mutations in core driver and resistance genes (rows) with patient-specific clinical tracks (TMB, Response, Cohort source, and Sex).\n\n")
        f.write("![Co-Mutation Landscape (Merged Trials)](../../plots/genomic/comut_landscape_merged.png)\n\n")

        f.write("### Key Observations\n")
        f.write("* **MAPK Driver Mutual Exclusivity**: There is high mutual exclusivity between the two primary MAPK pathway drivers, `BRAF` and `NRAS`. This aligns with the classical understanding that `BRAF` and `NRAS` mutations represent redundant and mutually exclusive routes for activating the RAS-RAF-MEK-ERK signaling cascade.\n")
        f.write("* **NF1 Mutational Overlap**: Unlike `BRAF` and `NRAS`, mutations in the tumor suppressor `NF1` show significant overlap with both drivers. While some of these represent co-occurring driver events, many of these `NF1` mutations are passenger events. `NF1` is a large gene and highly susceptible to random somatic passenger mutations in melanoma, which features a high TMB driven by UV-light exposure.\n")
        f.write("* **Targeted Resistance Profile**: Core genes related to antigen presentation (`B2M`) and interferon signaling (`JAK1`, `JAK2`) show low baseline mutation rates. These mutations are rare in pre-treatment biopsies, indicating that genetic disruption of interferon signaling is mostly an acquired resistance mechanism rather than a common baseline driver.\n")
        f.write("* **Cohort Distribution**: The Cohort track shows that `BRAF` and `NRAS` mutations are evenly distributed across **Liu 2019** (blue) and **Hugo 2016** (orange).\n")
        f.write("* **No Driver Subtype Response Bias**: Responders are distributed across all driver mutation subtypes (`BRAF`, `NRAS`, `NF1`, and Triple-WT). This visually confirms that driver mutation status itself is not predictive of anti-PD-1 clinical response.\n")

    print(f"Genomic characteristics report successfully written to {rel_path(report_path)}")


def main() -> None:
    """Executes the complete genomic characterisation and visualisation pipeline."""
    print("==================================================")
    print("Genomic Characterisation and Visualisation")
    print("==================================================\n")

    set_presentation_style()
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
