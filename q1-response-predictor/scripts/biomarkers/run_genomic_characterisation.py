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
from scipy.stats import mannwhitneyu
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
from src.utils.logging import TeeStream
from src.utils.paths import (
    CONFIG_DIR,
    DATA_DIR,
    LOG_DIR,
    PLOTS_DIR,
    SUBPROJECT_ROOT,
    rel_path,
)
from src.utils.plotting import resolve_colors, save_fig

# Module-level Constants
CONFIG_PATH = CONFIG_DIR / "datasets.yaml"
PLOT_DIR = PLOTS_DIR / "genomic"
LOG_PATH = LOG_DIR / "run_genomic_characterisation.log"

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
    """Plots tumour mutational burden (TMB) distributions by response and cohort-wide.

    Args:
        cohorts: Dictionary of cohort DataFrames.
        plot_dir: Path to directory for saving plot artifacts.
    """
    print("\n2. Generating TMB distributions...")
    trial_list = []
    for name in ["Liu 2019", "Hugo 2016", "Riaz 2017"]:
        df = cohorts[name][["TMB_NONSYNONYMOUS", "response"]].dropna().copy()
        df["Cohort"] = f"{name} (N={len(df)})"
        df["Base_Cohort"] = name
        trial_list.append(df)

    df_trials_tmb = pd.concat(trial_list, ignore_index=True)
    df_trials_tmb["Response"] = df_trials_tmb["response"].map({1.0: "Responder (CR/PR)", 0.0: "Non-responder (PD)"})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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

    clin_tcga = cohorts["TCGA-SKCM"]
    tcga_tmb = clin_tcga["TMB_NONSYNONYMOUS"].dropna()
    tcga_color = get_cohort_color("TCGA-SKCM")

    sns.histplot(
        tcga_tmb,
        kde=True,
        log_scale=True,
        color=tcga_color,
        ax=axes[1],
        bins=30,
        edgecolor="black",
    )
    axes[1].axvline(tcga_tmb.median(), color=RESPONSE_PALETTE["PD"], linestyle="--", linewidth=1.5, label=f"Median = {tcga_tmb.median():.2f}")
    axes[1].axvline(STANDARD_FDA_TMB_CUTOFF, color="#555555", linestyle=":", linewidth=1.5, label=f"Standard FDA Cutoff = {STANDARD_FDA_TMB_CUTOFF:.1f}")
    axes[1].set_xlabel("TMB (mutations/Mb, log scale)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
    axes[1].set_title(f"TCGA-SKCM Tumour Mutational Burden (TMB) Distribution (N={len(tcga_tmb)})", fontsize=13, fontweight="bold")
    axes[1].legend(loc="upper right")

    out_tmb_path = plot_dir / "tmb_distributions_by_cohort.png"
    save_fig(fig, out_tmb_path)
    print(f"Saved TMB distributions to {rel_path(out_tmb_path)}")

    out_tmb_legacy = plot_dir / "tmb_distribution.png"
    save_fig(fig, out_tmb_legacy)


def _plot_biomarker_correlations(clin_liu: pd.DataFrame, plot_dir: Path) -> None:
    """Generates Spearman correlation heatmap for neoantigen metrics in Liu 2019.

    Args:
        clin_liu: Clinical DataFrame for Liu 2019 cohort.
        plot_dir: Path to directory for saving plot artifacts.
    """
    print("\n3. Generating biomarker correlation matrix...")
    available_cols = [c for c in NEOANTIGEN_FEATURES if c in clin_liu.columns]
    corr_df = clin_liu[available_cols].dropna()

    if len(corr_df) > 0:
        corr_matrix = corr_df.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(8, 6.5))
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
        ax.set_title(f"Genomic & Neoantigen Biomarker Spearman Correlation (Liu 2019, N={len(corr_df)})", fontsize=13, fontweight="bold", pad=15)
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
                event_observed=df_surv.loc[mask, "OS_STATUS"],
                label=f"{subtype} (N={mask.sum()})",
            )
            kmf.plot_survival_function(ax=axes[0], color=color, linewidth=2.5, ci_show=False)

    results_mut = multivariate_logrank_test(df_surv["OS_MONTHS"], df_surv["Genomic_Subtype"], df_surv["OS_STATUS"])
    axes[0].text(
        0.05,
        0.08,
        f"Multivariate Log-rank p = {results_mut.p_value:.4f}",
        transform=axes[0].transAxes,
        fontsize=11,
        fontweight="semibold",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )
    axes[0].set_title(f"TCGA OS: Stratified by Driver Mutation Subtype (N={len(df_surv)})", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Time (months)", fontsize=11)
    axes[0].set_ylabel("Overall Survival Probability", fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    tcga_tmb_med = df_surv["TMB_NONSYNONYMOUS"].median()
    df_surv["TMB_Group"] = df_surv["TMB_NONSYNONYMOUS"].apply(lambda x: "High TMB" if x >= tcga_tmb_med else "Low TMB")

    mask_high = df_surv["TMB_Group"] == "High TMB"
    mask_low = df_surv["TMB_Group"] == "Low TMB"

    kmf_h = KaplanMeierFitter()
    kmf_l = KaplanMeierFitter()

    kmf_h.fit(df_surv.loc[mask_high, "OS_MONTHS"], event_observed=df_surv.loc[mask_high, "OS_STATUS"], label=f"High TMB (N={mask_high.sum()})")
    kmf_h.plot_survival_function(ax=axes[1], color=RESPONSE_PALETTE["CR/PR"], linewidth=2.5, ci_show=False)

    kmf_l.fit(df_surv.loc[mask_low, "OS_MONTHS"], event_observed=df_surv.loc[mask_low, "OS_STATUS"], label=f"Low TMB (N={mask_low.sum()})")
    kmf_l.plot_survival_function(ax=axes[1], color=RESPONSE_PALETTE["PD"], linewidth=2.5, ci_show=False)

    results_tmb = logrank_test(
        df_surv.loc[mask_high, "OS_MONTHS"],
        df_surv.loc[mask_low, "OS_MONTHS"],
        df_surv.loc[mask_high, "OS_STATUS"],
        df_surv.loc[mask_low, "OS_STATUS"],
    )
    axes[1].text(
        0.05,
        0.08,
        f"Log-rank p = {results_tmb.p_value:.4f}",
        transform=axes[1].transAxes,
        fontsize=11,
        fontweight="semibold",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )
    axes[1].set_title(f"TCGA OS: Stratified by TMB (Median Split = {tcga_tmb_med:.2f}, N={len(df_surv)})", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Time (months)", fontsize=11)
    axes[1].set_ylabel("Overall Survival Probability", fontsize=11)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"TCGA-SKCM Overall Survival by Genomic Features (N={len(df_surv)})", fontsize=16, fontweight="bold", y=0.98)

    out_surv_path = plot_dir / "tcga_survival_by_mutation.png"
    save_fig(fig, out_surv_path)
    print(f"Saved TCGA survival stratification plots to {rel_path(out_surv_path)}")

    out_surv_legacy = plot_dir / "km_genomic_features.png"
    save_fig(fig, out_surv_legacy)


# ---------------------------------------------------------------------------
# Extended Pathway Mutation Frequencies (Consolidated)
# ---------------------------------------------------------------------------

POOLED_LABEL = "Pooled Trials"
ROW_LABELS = {"BRAF": "BRAF mutation", "NRAS": "NRAS mutation", "NF1": "NF1 mutation"}
ROW_ORDER = [
    ("MAPK Drivers", "BRAF"),
    ("MAPK Drivers", "NRAS"),
    ("MAPK Drivers", "NF1"),
    ("Immune Resistance", "IFN-gamma Signaling"),
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


def compute_cohort_frequencies(config: DatasetConfig, data_dir: Path) -> Tuple[Dict[str, float], int]:
    """Compute driver/pathway mutation frequencies (%) for one cohort."""
    cohort_dir = data_dir / "processed" / config.processed_directory
    mut_path = cohort_dir / "mutations_cleaned.csv"
    clin_path = cohort_dir / "clin_cleaned.csv"

    if not mut_path.exists():
        raise FileNotFoundError(f"Missing {rel_path(mut_path)}. Run preprocessing script for {config.cohort_name} first.")
    if not clin_path.exists():
        raise FileNotFoundError(f"Missing {rel_path(clin_path)}. Run preprocessing script for {config.cohort_name} first.")

    df_clin = pd.read_csv(clin_path, index_col=0)
    n_patients = len(df_clin)
    all_patients = df_clin.index

    df_mut = pd.read_csv(mut_path)

    if "Hugo_Symbol" in df_mut.columns:
        id_col = find_id_column(df_mut)
        if "Variant_Classification" in df_mut.columns:
            df_mut = df_mut[df_mut["Variant_Classification"].isin(NON_SILENT_VARIANT_CLASSIFICATIONS)]
        mut_wide = df_mut.pivot_table(
            index=id_col, columns="Hugo_Symbol", values="Hugo_Symbol", aggfunc="count",
        )
    else:
        id_col = find_id_column(df_mut)
        mut_wide = df_mut.set_index(id_col)

    mut_wide = mut_wide.reindex(all_patients, fill_value=0)

    freqs = {gene: _compute_gene_frequency(mut_wide, [gene], config.cohort_name) for gene in DRIVER_GENES}
    for pathway, genes in PATHWAY_GENES.items():
        freqs[pathway] = _compute_gene_frequency(mut_wide, genes, config.cohort_name)

    return freqs, n_patients


def build_extended_pathway_dataframe(data_dir: Path) -> Tuple[pd.DataFrame, List[str], str, Dict[str, int]]:
    """Assemble cohort x gene/pathway frequency table with N-weighted Pooled Trials column."""
    dataset_configs = load_dataset_config(CONFIG_PATH)
    per_cohort = {config.cohort_name: compute_cohort_frequencies(config, data_dir) for config in dataset_configs}
    freqs = {name: f for name, (f, n) in per_cohort.items()}
    ns = {name: n for name, (f, n) in per_cohort.items()}
    cohort_names = [config.cohort_name for config in dataset_configs]

    rows = []
    for category, key in ROW_ORDER:
        label = ROW_LABELS.get(key, key)
        row = {"Category": category, "Gene/Pathway": label}
        for c in cohort_names:
            row[c] = freqs[c].get(key)
        rows.append(row)

    df = pd.DataFrame(rows)

    df[POOLED_LABEL] = df.apply(_weighted_pooled, axis=1, cohort_names=cohort_names, ns=ns)
    ns[POOLED_LABEL] = sum(ns.values())

    rename = {c: f"{c} (N={ns[c]})" for c in cohort_names + [POOLED_LABEL]}
    df = df.rename(columns=rename)
    cohort_cols = [rename[c] for c in cohort_names]
    pooled_col = rename[POOLED_LABEL]
    return df, cohort_cols, pooled_col, ns


def _plot_extended_pathway_grouped_bars(
    df: pd.DataFrame, cohort_cols: List[str], pooled_col: str, colors: List[str], out_dir: Path
) -> None:
    """Plots grouped horizontal bars of extended pathway mutation frequencies."""
    all_cols = cohort_cols + [pooled_col]
    y_labels = df["Gene/Pathway"].tolist()
    y_pos = np.arange(len(y_labels))

    cat_sizes = df.groupby("Category", sort=False).size()
    cat_names = list(cat_sizes.index)
    boundaries = np.cumsum(cat_sizes.values)[:-1] - 0.5
    cat_centers, start = [], 0
    for size in cat_sizes.values:
        cat_centers.append(start + (size - 1) / 2.0)
        start += size

    n_cohorts = len(all_cols)
    bar_width = 0.8 / n_cohorts
    max_val = np.nanmax(df[all_cols].values.astype(float))

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

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
                    ha="left", va="center", fontsize=9,
                    fontweight="bold" if cohort == pooled_col else "normal",
                    color="#222222",
                )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Mutation Frequency (%)", fontweight="bold")
    ax.set_title(
        "Pre-Treatment Somatic Mutation & Pathway Frequencies Across Cohorts",
        fontweight="bold", pad=15,
    )
    ax.legend(title="Cohort", frameon=True, facecolor="white", framealpha=0.9, loc="lower right")

    for b in boundaries:
        ax.axhline(b, color="gray", linestyle="--", alpha=0.5)

    label_x = -max_val * 0.31
    ax.set_xlim(label_x * 1.15, max_val * 1.2)
    for cat_name, center in zip(cat_names, cat_centers):
        ax.text(
            label_x, center, cat_name, rotation=90, va="center", ha="center",
            fontweight="bold", color="#333333", fontsize=11,
        )

    out_path = out_dir / "extended_pathway_mutation_frequencies.png"
    save_fig(fig, out_path)
    print(f"Saved extended pathway mutation frequencies plot to {rel_path(out_path)}")

    out_legacy = out_dir / "extended_pathway_grouped_bars.png"
    save_fig(fig, out_legacy)


def _plot_extended_pathway_heatmap(
    df: pd.DataFrame, cohort_cols: List[str], pooled_col: str, out_dir: Path
) -> None:
    """Plots heatmap of extended pathway mutation frequencies across cohorts."""
    all_cols = cohort_cols + [pooled_col]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    heatmap_df = df.set_index("Gene/Pathway")[all_cols]

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
    """Plots dumbbell plot showing trial variation vs. pooled benchmark for extended pathways."""
    trial_colors = dict(zip(cohort_cols, colors[: len(cohort_cols)]))
    pooled_color = get_cohort_color(pooled_col, default="#e41a1c")
    max_val = np.nanmax(df[cohort_cols + [pooled_col]].values.astype(float))

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    for idx, row in df.iterrows():
        y = idx
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

    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["Gene/Pathway"], fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Mutation Frequency (%)", fontweight="bold")
    ax.set_xlim(-max_val * 0.03, max_val * 1.1)
    ax.set_title(
        "Extended Pathway Mutation Rates: Trial Variation vs. Pooled Benchmark",
        fontweight="bold", pad=15,
    )
    ax.legend(loc="lower right", frameon=True, facecolor="white")

    out_path = out_dir / "extended_pathway_dumbbell.png"
    save_fig(fig, out_path)
    print(f"Saved extended pathway dumbbell plot to {rel_path(out_path)}")


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
    _plot_biomarker_correlations(cohorts["Liu 2019"], PLOT_DIR)
    _plot_tcga_survival_stratification(cohorts["TCGA-SKCM"], PLOT_DIR)

    print("\n5. Generating extended pathway mutation frequency visualisations...")
    ext_df, cohort_cols, pooled_col, ext_ns = build_extended_pathway_dataframe(DATA_DIR)
    ext_colors = resolve_colors(cohort_cols + [pooled_col])
    _plot_extended_pathway_grouped_bars(ext_df, cohort_cols, pooled_col, ext_colors, PLOT_DIR)
    _plot_extended_pathway_heatmap(ext_df, cohort_cols, pooled_col, PLOT_DIR)
    _plot_extended_pathway_dumbbell(ext_df, cohort_cols, pooled_col, ext_colors, PLOT_DIR)

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
