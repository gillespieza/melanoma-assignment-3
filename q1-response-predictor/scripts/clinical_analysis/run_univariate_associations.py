"""Univariate Association Analysis Script for Clinical and Genomic Variables.

Evaluates univariate statistical associations between baseline clinical/genomic features
(Sex, Clinical Stage, BRAF/NRAS/NF1 mutations, Age, TMB, Neoantigens) and immunotherapy response (CR/PR vs. PD)
across individual trial cohorts (Liu 2019, Hugo 2016, Riaz 2017) and the pooled trial dataset.
Outputs Odds Ratios (OR) with 95% Confidence Intervals (95% CI) presented in a publication-grade Forest Plot.
"""

import contextlib
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import fisher_exact, norm
import seaborn as sns

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

_SUBPROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBPROJECT_ROOT))

from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.styles import COHORT_PALETTE, get_cohort_color, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import (
    CONFIG_DIR,
    DATA_DIR,
    LOG_DIR,
    PLOTS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    SUBPROJECT_ROOT,
    rel_path,
)
from src.utils.plotting import save_fig

# Module-level Constants
PLOT_DIR = PLOTS_DIR / "clinical"
LOG_PATH = LOG_DIR / "run_univariate_associations.log"

CATEGORICAL_VARS: Dict[str, str] = {
    "Sex (Male vs Female)": "SEX",
    "Stage (IV vs III)": "CLINICAL_STAGE",
    "BRAF Mutation (Mut vs WT)": "mut_BRAF",
    "NRAS Mutation (Mut vs WT)": "mut_NRAS",
    "NF1 Mutation (Mut vs WT)": "mut_NF1",
}

CONTINUOUS_VARS: Dict[str, List[str]] = {
    "Age (per SD)": ["AGE", "AGE_AT_DIAGNOSIS", "AGE (YRS)", "age"],
    "TMB (per SD)": ["TMB_NONSYNONYMOUS"],
    "SNV Neoantigens (per SD)": ["SNV_NEOANTIGEN"],
    "Indel Neoantigens (per SD)": ["INDEL_NEOANTIGEN"],
}


def _calc_categorical_or(contingency: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Calculates Odds Ratio and 95% CI for a 2x2 contingency matrix using Haldane-Anscombe correction if needed.

    Args:
        contingency: 2x2 contingency matrix (variable vs response).

    Returns:
        Tuple of (odds_ratio, ci_lower, ci_upper, p_value).
    """
    a = contingency.loc[1, 1.0] if (1 in contingency.index and 1.0 in contingency.columns) else 0
    b = contingency.loc[1, 0.0] if (1 in contingency.index and 0.0 in contingency.columns) else 0
    c = contingency.loc[0, 1.0] if (0 in contingency.index and 1.0 in contingency.columns) else 0
    d = contingency.loc[0, 0.0] if (0 in contingency.index and 0.0 in contingency.columns) else 0

    if a == 0 or b == 0 or c == 0 or d == 0:
        a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_c, b_c, c_c, d_c = a, b, c, d

    or_val = (a_c * d_c) / (b_c * c_c)
    se_ln_or = np.sqrt(1.0 / a_c + 1.0 / b_c + 1.0 / c_c + 1.0 / d_c)
    ci_lower = np.exp(np.log(or_val) - 1.96 * se_ln_or)
    ci_upper = np.exp(np.log(or_val) + 1.96 * se_ln_or)

    _, p_val = fisher_exact([[a, b], [c, d]])
    return float(or_val), float(ci_lower), float(ci_upper), float(p_val)


def _fit_univariate_logit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """Fits univariate logistic regression on Z-scored continuous predictor.

    Args:
        x: Continuous feature array.
        y: Binary response array.

    Returns:
        Tuple of (odds_ratio, ci_lower, ci_upper, p_value).
    """
    x_clean = np.asarray(x, dtype=float)
    y_clean = np.asarray(y, dtype=float)
    valid = ~np.isnan(x_clean) & ~np.isnan(y_clean)
    x_clean, y_clean = x_clean[valid], y_clean[valid]

    if len(x_clean) < 5 or len(np.unique(y_clean)) < 2 or np.std(x_clean) == 0:
        return np.nan, np.nan, np.nan, 1.0

    z_score = (x_clean - np.mean(x_clean)) / (np.std(x_clean) + 1e-8)
    X = np.column_stack([np.ones_like(z_score), z_score])

    def loss(beta):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        return -np.sum(y_clean * np.log(p + 1e-12) + (1.0 - y_clean) * np.log(1.0 - p + 1e-12))

    res = minimize(loss, [0.0, 0.0], method="L-BFGS-B")
    beta = res.x
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
    w = p * (1.0 - p)
    w = np.maximum(w, 1e-6)

    H = X.T @ (w[:, None] * X)
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(cov[1, 1])
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, 1.0

    slope = beta[1]
    or_val = np.exp(slope)
    ci_lower = np.exp(slope - 1.96 * se)
    ci_upper = np.exp(slope + 1.96 * se)

    z_stat = slope / (se + 1e-12)
    p_val = 2.0 * (1.0 - norm.cdf(abs(z_stat)))

    return float(or_val), float(ci_lower), float(ci_upper), float(p_val)


def calculate_associations(df: pd.DataFrame, cohort_name: str) -> pd.DataFrame:
    """Computes univariate association metrics (Odds Ratios and 95% CIs) against response.

    Args:
        df: Clinical DataFrame containing feature columns and binary response.
        cohort_name: Name of the cohort being evaluated.

    Returns:
        DataFrame containing Odds Ratios, 95% CIs, and p-values.
    """
    results = []

    # 1. Categorical variables vs Response
    for label, col in CATEGORICAL_VARS.items():
        if col in df.columns:
            temp = df[[col, "response"]].dropna().copy()
            if col == "SEX":
                temp[col] = temp[col].map({"Male": 1, "Female": 0})
            elif col == "CLINICAL_STAGE":
                temp[col] = temp[col].map({"IV": 1, "III": 0})

            temp[col] = pd.to_numeric(temp[col], errors="coerce")
            temp = temp.dropna()

            if len(temp) > 0 and len(temp[col].unique()) == 2:
                contingency = pd.crosstab(temp[col], temp["response"])
                or_val, lower, upper, p_val = _calc_categorical_or(contingency)
                results.append({
                    "Cohort": cohort_name,
                    "Variable": label,
                    "Test": "Fisher's Exact / OR",
                    "Odds Ratio (OR)": or_val,
                    "95% CI Lower": lower,
                    "95% CI Upper": upper,
                    "p-value": p_val,
                    "Type": "Categorical",
                })

    # 2. Continuous variables vs Response (Logistic Regression per 1 SD increase)
    for label, cols in CONTINUOUS_VARS.items():
        col = None
        for c in cols:
            if c in df.columns:
                col = c
                break
        if col:
            temp = df[[col, "response"]].dropna().copy()
            temp[col] = pd.to_numeric(temp[col], errors="coerce")
            temp = temp.dropna()

            if len(temp) > 5 and len(temp["response"].unique()) == 2:
                or_val, lower, upper, p_val = _fit_univariate_logit(temp[col].values, temp["response"].values)
                if not np.isnan(or_val):
                    results.append({
                        "Cohort": cohort_name,
                        "Variable": label,
                        "Test": "Logistic Regression (per SD)",
                        "Odds Ratio (OR)": or_val,
                        "95% CI Lower": lower,
                        "95% CI Upper": upper,
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
        if "RESPONSE_BINARY" in df.columns:
            df["response"] = df["RESPONSE_BINARY"]
        elif "RESPONDER" in df.columns:
            df["response"] = df["RESPONDER"].map({True: 1.0, False: 0.0, 1.0: 1.0, 0.0: 0.0, "1": 1.0, "0": 0.0, "CR/PR": 1.0, "PD": 0.0})

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


def _plot_univariate_associations(all_results: pd.DataFrame, cohort_counts: Dict[str, int], plot_dir: Path) -> None:
    """Renders a publication-ready Forest Plot of Odds Ratios with 95% CIs.

    Args:
        all_results: Combined DataFrame of univariate statistical association results.
        cohort_counts: Dictionary mapping cohort names to dynamic patient sample counts.
        plot_dir: Path to export output plot figure and CSV table.
    """
    cohort_order = ["Pooled Trials", "Liu 2019", "Hugo 2016", "Riaz 2017"]

    variables = all_results["Variable"].unique().tolist()

    set_presentation_style()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13, 10.5))
    plt.subplots_adjust(left=0.28, right=0.62, top=0.90, bottom=0.16)

    ax.set_xscale("log")
    ax.axvline(1.0, color="#37474F", linestyle="--", linewidth=1.5, zorder=1)

    y_pos = 0.0
    y_ticks = []
    y_labels = []

    for var in reversed(variables):
        sub_df = all_results[all_results["Variable"] == var].copy()
        sub_df["Cohort"] = pd.Categorical(sub_df["Cohort"], categories=cohort_order, ordered=True)
        sub_df = sub_df.sort_values("Cohort")

        y_ticks.append(y_pos + (len(sub_df) - 1) / 2.0)
        y_labels.append(var)

        for _, row in sub_df.iterrows():
            cohort = row["Cohort"]
            or_val = row["Odds Ratio (OR)"]
            lower = row["95% CI Lower"]
            upper = row["95% CI Upper"]
            p_val = row["p-value"]

            color = get_cohort_color(cohort)
            is_sig = p_val < 0.05
            weight = "bold" if is_sig else "normal"
            text_color = "#222222" if is_sig else "#555555"

            # Bound CIs for visual plotting display
            disp_lower = max(0.1, lower)
            disp_upper = min(20.0, upper)

            ax.errorbar(
                x=or_val,
                y=y_pos,
                xerr=[[max(0.01, or_val - disp_lower)], [max(0.01, disp_upper - or_val)]],
                fmt="o" if cohort == "Pooled Trials" else "s",
                color=color,
                ecolor=color,
                elinewidth=2.0 if cohort == "Pooled Trials" else 1.2,
                capsize=4 if cohort == "Pooled Trials" else 3,
                capthick=1.5 if cohort == "Pooled Trials" else 1.0,
                markersize=8 if cohort == "Pooled Trials" else 6,
                zorder=4 if cohort == "Pooled Trials" else 3,
            )

            lbl_or = f"{or_val:.2f} ({lower:.2f} - {upper:.2f})"
            lbl_p = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.3f}"
            if is_sig:
                lbl_p += " *"

            cohort_label = f"{cohort} (N={cohort_counts.get(cohort, 0)})"
            ax.text(1.10, y_pos, f"{cohort_label}: {lbl_or}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=9.5, color=text_color, weight=weight)
            ax.text(1.72, y_pos, lbl_p, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=9.5, color=text_color, weight=weight)

            y_pos += 1.0

        y_pos += 0.8  # Gap between variable groups

    header_y = y_pos - 0.2
    ax.text(1.10, header_y, "Cohort & OR (95% CI)", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=10.5, color="#111111", weight="bold")
    ax.text(1.72, header_y, "p-value", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=10.5, color="#111111", weight="bold")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11, fontweight="bold", color="#222222")
    ax.set_ylim(-0.8, y_pos)
    ax.set_xlim(0.1, 20.0)

    ax.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="x", which="both", labelsize=10.5)

    n_pooled = cohort_counts.get("Pooled Trials", 0)
    ax.set_xlabel("Odds Ratio for Immunotherapy Response (log scale)", fontsize=12, labelpad=6, weight="bold")
    ax.set_title(f"Univariate Associations with Immunotherapy Response (Forest Plot, N={n_pooled})", fontsize=15, fontweight="bold", pad=15)

    ax.text(0.95, -0.055, "Favours Responder (OR > 1.0) \u2192", transform=ax.transAxes, ha="right", va="top", color="#555555", fontsize=9.5, style="italic")
    ax.text(0.05, -0.055, "\u2190 Favours Non-Responder (OR < 1.0)", transform=ax.transAxes, ha="left", va="top", color="#555555", fontsize=9.5, style="italic")

    legend_elements = [
        mlines.Line2D([0], [0], marker="o", color="none", markerfacecolor=get_cohort_color("Pooled Trials"), markeredgecolor="none", markersize=8, label=f"Pooled Trials Benchmark (N={cohort_counts.get('Pooled Trials', 0)})"),
        mlines.Line2D([0], [0], marker="s", color="none", markerfacecolor=get_cohort_color("Liu 2019"), markeredgecolor="none", markersize=7, label=f"Liu 2019 (N={cohort_counts.get('Liu 2019', 0)})"),
        mlines.Line2D([0], [0], marker="s", color="none", markerfacecolor=get_cohort_color("Hugo 2016"), markeredgecolor="none", markersize=7, label=f"Hugo 2016 (N={cohort_counts.get('Hugo 2016', 0)})"),
        mlines.Line2D([0], [0], marker="s", color="none", markerfacecolor=get_cohort_color("Riaz 2017"), markeredgecolor="none", markersize=7, label=f"Riaz 2017 (N={cohort_counts.get('Riaz 2017', 0)})"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9.5)

    sns.despine(ax=ax, top=True, right=True)
    ax.yaxis.grid(True, linestyle="--", color="#E0E0E0", linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(True, linestyle=":", color="#E0E0E0", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    out_path = plot_dir / "univariate_associations.png"
    save_fig(fig, out_path)
    print(f"\nSaved univariate associations forest plot to {rel_path(out_path)}")

    csv_path = plot_dir / "univariate_associations_stats.csv"
    all_results.to_csv(csv_path, index=False)
    print(f"Saved univariate association statistics table to {rel_path(csv_path)}")


def main() -> None:
    """Executes the univariate association analysis pipeline."""
    print("==================================================")
    print("Computing Univariate Associations with Response (Forest Plot)")
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

    cohort_counts = {
        "Liu 2019": len(cohorts["Liu 2019"][cohorts["Liu 2019"]["response"].notna()]),
        "Hugo 2016": len(cohorts["Hugo 2016"][cohorts["Hugo 2016"]["response"].notna()]),
        "Riaz 2017": len(cohorts["Riaz 2017"][cohorts["Riaz 2017"]["response"].notna()]),
        "Pooled Trials": len(pooled_df[pooled_df["response"].notna()]),
    }

    all_results = pd.concat([results_liu, results_hugo, results_riaz, results_pooled], ignore_index=True)
    print(all_results.to_string())

    _plot_univariate_associations(all_results, cohort_counts, PLOT_DIR)

    print("\n==================================================")
    print("Done!")
    print("==================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
