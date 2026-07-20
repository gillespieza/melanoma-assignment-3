"""
Univariate Association Analysis Script for Clinical and Genomic Variables.

Evaluates univariate statistical associations between baseline clinical/genomic features
(Sex, Clinical Stage, BRAF/NRAS/NF1 mutations, Age, TMB, Neoantigens) and immunotherapy response (CR/PR vs. PD)
across individual trial cohorts (Liu 2019, Hugo 2016, Riaz 2017) and the pooled trial dataset.
Uses Fisher's Exact test for categorical features and Mann-Whitney U tests for continuous features.
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
from scipy.stats import fisher_exact, mannwhitneyu
import seaborn as sns

# Bootstrap project root resolution for top-level import
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.styles import set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import find_project_root
from src.utils.plotting import resolve_colors, save_fig

# Module-level Constants
DATA_DIR = find_project_root(Path(__file__).resolve()) / "data"
PLOT_DIR = find_project_root(Path(__file__).resolve()) / "plots" / "clinical"
LOG_DIR = find_project_root(Path(__file__).resolve()) / "logs"
LOG_PATH = LOG_DIR / "run_univariate_associations.log"

CATEGORICAL_VARS: Dict[str, str] = {
    "Sex (Male vs Female)": "SEX",
    "Stage (IV vs III)": "CLINICAL_STAGE",
    "BRAF Mutation (Mut vs WT)": "mut_BRAF",
    "NRAS Mutation (Mut vs WT)": "mut_NRAS",
    "NF1 Mutation (Mut vs WT)": "mut_NF1",
}

CONTINUOUS_VARS: Dict[str, List[str]] = {
    "Age": ["AGE", "AGE_AT_DIAGNOSIS", "AGE (YRS)", "age"],
    "TMB": ["TMB_NONSYNONYMOUS"],
    "SNV Neoantigens": ["SNV_NEOANTIGEN"],
    "Indel Neoantigens": ["INDEL_NEOANTIGEN"],
}


def calculate_associations(df: pd.DataFrame, cohort_name: str) -> pd.DataFrame:
    """Computes univariate association metrics for categorical and continuous variables against response.

    Args:
        df: Clinical DataFrame containing feature columns and binary response.
        cohort_name: Name of the cohort being evaluated.

    Returns:
        DataFrame containing test statistics, odds ratios/mean differences, and p-values.
    """
    results = []

    # 1. Categorical variables vs Response (Fisher's Exact Test)
    for label, col in CATEGORICAL_VARS.items():
        if col in df.columns:
            temp = df[[col, "response"]].dropna()
            if len(temp) > 0 and len(temp[col].unique()) == 2:
                contingency = pd.crosstab(temp[col], temp["response"])
                if contingency.shape == (2, 2):
                    odds_ratio, p_val = fisher_exact(contingency)
                    results.append({
                        "Cohort": cohort_name,
                        "Variable": label,
                        "Test": "Fisher's Exact",
                        "Statistic": odds_ratio,
                        "p-value": p_val,
                        "Type": "Categorical",
                    })

    # 2. Continuous variables vs Response (Mann-Whitney U Test)
    for label, cols in CONTINUOUS_VARS.items():
        col = None
        for c in cols:
            if c in df.columns:
                col = c
                break
        if col:
            temp = df[[col, "response"]].dropna()
            temp[col] = pd.to_numeric(temp[col], errors="coerce")
            temp = temp.dropna()

            responders = temp[temp["response"] == 1.0][col]
            non_responders = temp[temp["response"] == 0.0][col]

            if len(responders) > 1 and len(non_responders) > 1:
                _, p_val = mannwhitneyu(responders, non_responders, alternative="two-sided")
                mean_r = responders.mean()
                mean_nr = non_responders.mean()
                diff = mean_r - mean_nr
                results.append({
                    "Cohort": cohort_name,
                    "Variable": label,
                    "Test": "Mann-Whitney U",
                    "Statistic": diff,
                    "p-value": p_val,
                    "Type": "Continuous",
                })

    return pd.DataFrame(results)


def _prepare_clinical_cohorts(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Loads and standardises clinical trial cohort datasets.

    Args:
        data_dir: Path to project data directory.

    Returns:
        Dictionary mapping cohort names to clean DataFrames.
    """
    _, clin_liu = load_liu_2019(data_dir)
    _, clin_hugo = load_hugo_2016(data_dir)
    _, clin_riaz = load_riaz_2017(data_dir)

    for df in [clin_liu, clin_hugo, clin_riaz]:
        if "SEX" in df.columns:
            df["SEX"] = df["SEX"].map({"Male": "Male", "Female": "Female", "M": "Male", "F": "Female"})
        if "CLINICAL_STAGE" in df.columns:
            df["CLINICAL_STAGE"] = df["CLINICAL_STAGE"].apply(
                lambda x: "IV" if str(x).startswith("IV") else ("III" if str(x).startswith("III") else np.nan)
            )

    return {
        "Liu 2019": clin_liu,
        "Hugo 2016": clin_hugo,
        "Riaz 2017": clin_riaz,
    }


def _plot_univariate_associations(all_results: pd.DataFrame, plot_dir: Path) -> None:
    """Plots barplot of -log10(p-values) across clinical and genomic variables.

    Args:
        all_results: DataFrame of combined univariate statistical results.
        plot_dir: Directory path to export plot figure.
    """
    all_results["-log10(p-value)"] = -np.log10(all_results["p-value"])
    cohort_labels = all_results["Cohort"].unique().tolist()
    palette_colors = resolve_colors(cohort_labels)

    set_presentation_style()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    sns.barplot(
        data=all_results,
        x="Variable",
        y="-log10(p-value)",
        hue="Cohort",
        ax=ax,
        palette=palette_colors,
        edgecolor="black",
    )

    ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1.5, label="p = 0.05 (Significant)")
    ax.axhline(-np.log10(0.01), color="darkred", linestyle=":", linewidth=1.5, label="p = 0.01")

    ax.set_title("Statistical Significance of Univariate Associations with Response", fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel("-log10(p-value)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Clinical / Genomic Variable", fontsize=13, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11)

    out_path = plot_dir / "univariate_associations.png"
    save_fig(fig, out_path)
    print(f"\nSaved univariate associations plot to {out_path.relative_to(BASE_DIR).as_posix()}")

    csv_path = plot_dir / "univariate_associations_stats.csv"
    all_results.to_csv(csv_path, index=False)
    print(f"Saved univariate association statistics table to {csv_path.relative_to(BASE_DIR).as_posix()}")


def main() -> None:
    """Executes the univariate association analysis pipeline."""
    print("==================================================")
    print("Computing Univariate Associations with Response")
    print("==================================================\n")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    cohorts = _prepare_clinical_cohorts(DATA_DIR)

    results_liu = calculate_associations(cohorts["Liu 2019"], "Liu 2019")
    results_hugo = calculate_associations(cohorts["Hugo 2016"], "Hugo 2016")
    results_riaz = calculate_associations(cohorts["Riaz 2017"], "Riaz 2017")

    common_cols = ["response", "SEX", "CLINICAL_STAGE", "mut_BRAF", "mut_NRAS", "mut_NF1", "TMB_NONSYNONYMOUS"]
    pooled_df = pd.concat([
        cohorts["Liu 2019"][[c for c in common_cols if c in cohorts["Liu 2019"].columns]],
        cohorts["Hugo 2016"][[c for c in common_cols if c in cohorts["Hugo 2016"].columns]],
        cohorts["Riaz 2017"][[c for c in common_cols if c in cohorts["Riaz 2017"].columns]],
    ], ignore_index=True)
    results_pooled = calculate_associations(pooled_df, "Pooled Trials")

    all_results = pd.concat([results_liu, results_hugo, results_riaz, results_pooled], ignore_index=True)
    print(all_results.to_string())

    _plot_univariate_associations(all_results, PLOT_DIR)

    print("\n==================================================")
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
