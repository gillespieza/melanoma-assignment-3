"""Univariate Association Analysis Script for Clinical and Genomic Variables.

Evaluates univariate statistical associations between baseline clinical/genomic features
(Sex, Clinical Stage, BRAF/NRAS/NF1 mutations, Age, TMB, Neoantigens) and immunotherapy response
across individual trial cohorts (Liu 2019, Hugo 2016, Riaz 2017) and the pooled trial dataset.
Outputs Odds Ratios (OR) with 95% Confidence Intervals (95% CI) presented in a Forest Plot.
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
from scipy.stats.contingency import odds_ratio
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
from src.utils.paths import DATA_DIR, PLOTS_DIR, get_subproject_log_dir, rel_path
from src.utils.plotting import save_fig

set_presentation_style()

# Module-level Constants
LOG_DIR = get_subproject_log_dir(Path(__file__))
PLOT_DIR = PLOTS_DIR / "clinical"
LOG_PATH = LOG_DIR / "run_univariate_associations.log"

_SIGNIFICANCE_ALPHA: float = 0.05
_CI_LOWER_DISPLAY: float = 0.1
_CI_UPPER_DISPLAY: float = 20.0
_PLOT_FIGSIZE: Tuple[int, int] = (13, 11)
_POOLED_COLS: List[str] = [
    "response",
    "SEX",
    "CLINICAL_STAGE",
    "mut_BRAF",
    "mut_NRAS",
    "mut_NF1",
    "TMB_NONSYNONYMOUS",
]

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


def _calc_categorical_or(
    contingency: pd.DataFrame,
) -> Tuple[float, float, float, float]:
    """Calculates Odds Ratio and 95% CI for a 2x2 contingency matrix.

    Args:
        contingency: 2x2 contingency matrix (variable vs response).

    Returns:
        Tuple of (odds_ratio, ci_lower, ci_upper, p_value).
    """
    a = contingency.loc[1, 1.0] if (1 in contingency.index and 1.0 in contingency.columns) else 0
    b = contingency.loc[1, 0.0] if (1 in contingency.index and 0.0 in contingency.columns) else 0
    c = contingency.loc[0, 1.0] if (0 in contingency.index and 1.0 in contingency.columns) else 0
    d = contingency.loc[0, 0.0] if (0 in contingency.index and 0.0 in contingency.columns) else 0

    table = [[a, b], [c, d]]
    res_or = odds_ratio(table)
    ci = res_or.confidence_interval(0.95)
    _, p_val = fisher_exact(table)

    if a == 0 or b == 0 or c == 0 or d == 0:
        a_c, b_c, c_c, d_c = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        point_or = (a_c * d_c) / (b_c * c_c)
    else:
        point_or = float(res_or.statistic)

    return float(point_or), float(ci.low), float(ci.high), float(p_val)


def _logit_cov_and_se(X: np.ndarray, beta: np.ndarray) -> float:
    """Computes standard error of the slope coefficient from Fisher Information matrix.

    Args:
        X: Design matrix.
        beta: Fitted parameter vector.

    Returns:
        Standard error of slope coefficient or NaN on singular matrix.
    """
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
    w = np.maximum(p * (1.0 - p), 1e-6)
    H = X.T @ (w[:, None] * X)
    try:
        cov = np.linalg.inv(H)
        return float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        return np.nan


def _run_logit_optimization(X: np.ndarray, y_clean: np.ndarray) -> np.ndarray:
    """Solves MLE negative log-likelihood for logistic regression."""
    def loss(beta: np.ndarray) -> float:
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        neg_ll = y_clean * np.log(p + 1e-12) + (1.0 - y_clean) * np.log(1.0 - p + 1e-12)
        return float(-np.sum(neg_ll))

    res = minimize(loss, [0.0, 0.0], method="L-BFGS-B")
    return res.x


def _fit_univariate_logit(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float, float]:
    """Fits univariate logistic regression on Z-scored continuous predictor."""
    x_clean = np.asarray(x, dtype=float)
    y_clean = np.asarray(y, dtype=float)
    valid = ~np.isnan(x_clean) & ~np.isnan(y_clean)
    x_clean, y_clean = x_clean[valid], y_clean[valid]

    if len(x_clean) < 5 or len(np.unique(y_clean)) < 2 or np.std(x_clean) == 0:
        return np.nan, np.nan, np.nan, 1.0

    z_score = (x_clean - np.mean(x_clean)) / (np.std(x_clean) + 1e-8)
    X = np.column_stack([np.ones_like(z_score), z_score])

    beta = _run_logit_optimization(X, y_clean)
    slope = beta[1]
    se = _logit_cov_and_se(X, beta)
    if np.isnan(se):
        return np.nan, np.nan, np.nan, 1.0

    or_val = np.exp(slope)
    ci_lower = np.exp(slope - 1.96 * se)
    ci_upper = np.exp(slope + 1.96 * se)
    z_stat = slope / (se + 1e-12)
    p_val = 2.0 * (1.0 - norm.cdf(abs(z_stat)))

    return float(or_val), float(ci_lower), float(ci_upper), float(p_val)


def _eval_single_cat_var(
    df: pd.DataFrame, col: str, label: str, cohort_name: str
) -> Dict[str, str | float] | None:
    """Evaluates a single categorical feature against binary response."""
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
        return {
            "Cohort": cohort_name,
            "Variable": label,
            "Test": "Fisher's Exact / OR",
            "Odds Ratio (OR)": or_val,
            "95% CI Lower": lower,
            "95% CI Upper": upper,
            "p-value": p_val,
            "Type": "Categorical",
        }
    return None


def _compute_categorical_associations(
    df: pd.DataFrame, cohort_name: str
) -> List[Dict[str, str | float]]:
    """Evaluates categorical feature associations against response."""
    results: List[Dict[str, str | float]] = []
    for label, col in CATEGORICAL_VARS.items():
        if col in df.columns:
            res = _eval_single_cat_var(df, col, label, cohort_name)
            if res is not None:
                results.append(res)
    return results


def _eval_single_cont_var(
    df: pd.DataFrame, cols: List[str], label: str, cohort_name: str
) -> Dict[str, str | float] | None:
    """Evaluates a single continuous feature against response via logit."""
    col = next((c for c in cols if c in df.columns), None)
    if not col:
        return None
    temp = df[[col, "response"]].dropna().copy()
    temp[col] = pd.to_numeric(temp[col], errors="coerce")
    temp = temp.dropna()

    if len(temp) > 5 and len(temp["response"].unique()) == 2:
        or_val, lower, upper, p_val = _fit_univariate_logit(
            temp[col].values, temp["response"].values
        )
        if not np.isnan(or_val):
            return {
                "Cohort": cohort_name,
                "Variable": label,
                "Test": "Logistic Regression (per SD)",
                "Odds Ratio (OR)": or_val,
                "95% CI Lower": lower,
                "95% CI Upper": upper,
                "p-value": p_val,
                "Type": "Continuous",
            }
    return None


def _compute_continuous_associations(
    df: pd.DataFrame, cohort_name: str
) -> List[Dict[str, str | float]]:
    """Evaluates continuous feature associations against response via logit."""
    results: List[Dict[str, str | float]] = []
    for label, cols in CONTINUOUS_VARS.items():
        res = _eval_single_cont_var(df, cols, label, cohort_name)
        if res is not None:
            results.append(res)
    return results


def calculate_associations(df: pd.DataFrame, cohort_name: str) -> pd.DataFrame:
    """Computes univariate association metrics (Odds Ratios and 95% CIs) against response.

    Args:
        df: Clinical DataFrame containing feature columns and binary response.
        cohort_name: Name of the cohort being evaluated.

    Returns:
        DataFrame containing Odds Ratios, 95% CIs, and p-values.
    """
    cat_results = _compute_categorical_associations(df, cohort_name)
    cont_results = _compute_continuous_associations(df, cohort_name)
    return pd.DataFrame(cat_results + cont_results)


def _normalise_cohort_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardises clinical variables and binary response column for a cohort.

    Args:
        df: Input clinical DataFrame.

    Returns:
        Standardised DataFrame copy.
    """
    df_clean = df.copy()
    if "RESPONSE_BINARY" in df_clean.columns:
        df_clean["response"] = df_clean["RESPONSE_BINARY"]
    elif "RESPONDER" in df_clean.columns:
        resp_map = {
            True: 1.0, False: 0.0, 1.0: 1.0, 0.0: 0.0,
            "1": 1.0, "0": 0.0, "CR/PR": 1.0, "PD": 0.0,
        }
        df_clean["response"] = df_clean["RESPONDER"].map(resp_map)

    if "SEX" in df_clean.columns:
        df_clean["SEX"] = df_clean["SEX"].map(
            {"Male": "Male", "Female": "Female", "M": "Male", "F": "Female"}
        )
    if "CLINICAL_STAGE" in df_clean.columns:
        df_clean["CLINICAL_STAGE"] = df_clean["CLINICAL_STAGE"].apply(
            lambda x: "IV" if str(x).startswith("IV") else (
                "III" if str(x).startswith("III") else np.nan
            )
        )
    return df_clean


def _enrich_cohort_mutations(
    clin_df: pd.DataFrame, cohort_dir: Path
) -> pd.DataFrame:
    """Enriches clinical DataFrame with driver mutation flags from mutations_cleaned.csv."""
    mut_path = cohort_dir / "mutations_cleaned.csv"
    if not mut_path.exists():
        return clin_df

    mut_df = pd.read_csv(mut_path, index_col="SAMPLE_ID")
    df_enriched = clin_df.copy()
    for gene in ["BRAF", "NRAS", "NF1"]:
        if gene in mut_df.columns:
            df_enriched[f"mut_{gene}"] = df_enriched.index.map(
                lambda sid: (1.0 if mut_df.loc[sid, gene] > 0 else 0.0)
                if sid in mut_df.index else np.nan
            )
    return df_enriched


def _load_single_cohort(
    data_dir: Path, cohort_folder: str, loader_fn
) -> pd.DataFrame:
    """Loads, standardises clinical metadata, and enriches driver mutations."""
    _, clin_df = loader_fn(data_dir)
    norm_df = _normalise_cohort_df(clin_df)
    cohort_dir = data_dir / "processed" / cohort_folder
    return _enrich_cohort_mutations(norm_df, cohort_dir)


def _prepare_clinical_cohorts(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Loads and standardises all configured clinical trial cohort datasets with driver mutations."""
    _SCRIPT_DIR = Path(__file__).resolve().parent
    CONFIG_PATH = _SCRIPT_DIR.parent.parent / "config" / "datasets.yaml"

    if CONFIG_PATH.exists():
        from src.config.datasets import load_dataset_config
        all_configs = load_dataset_config(CONFIG_PATH)
        dataset_configs = [c for c in all_configs if c.cohort_name != "TCGA-SKCM"]
    else:
        dataset_configs = []

    cohorts: Dict[str, pd.DataFrame] = {}
    for config in dataset_configs:
        proc_dir = data_dir / "processed" / config.processed_directory
        clin_path = proc_dir / "clin_cleaned.csv"
        if clin_path.exists():
            clin_df = pd.read_csv(clin_path, index_col="SAMPLE_ID")
            norm_df = _normalise_cohort_df(clin_df)
            cohorts[config.cohort_name] = _enrich_cohort_mutations(norm_df, proc_dir)

    return cohorts


def _draw_forest_errorbar(
    ax: plt.Axes,
    y_pos: float,
    cohort: str,
    or_val: float,
    disp_lower: float,
    disp_upper: float,
) -> None:
    """Renders single error bar on forest plot."""
    color = get_cohort_color(cohort)
    is_pooled = cohort == "Pooled Trials"
    ax.errorbar(
        x=or_val, y=y_pos,
        xerr=[[max(0.01, or_val - disp_lower)], [max(0.01, disp_upper - or_val)]],
        fmt="o" if is_pooled else "s",
        color=color, ecolor=color,
        elinewidth=2.0 if is_pooled else 1.2,
        capsize=4 if is_pooled else 3,
        capthick=1.5 if is_pooled else 1.0,
        markersize=8 if is_pooled else 6,
        zorder=4 if is_pooled else 3,
    )


def _draw_forest_text_annotations(
    ax: plt.Axes,
    y_pos: float,
    cohort: str,
    or_val: float,
    lower: float,
    upper: float,
    p_val: float,
    cohort_counts: Dict[str, int],
) -> None:
    """Draws right-hand text labels for OR (95% CI) and p-value."""
    is_sig = p_val < _SIGNIFICANCE_ALPHA
    weight = "bold" if is_sig else "normal"
    text_color = "#222222" if is_sig else "#555555"

    lbl_or = f"{or_val:.2f} ({lower:.2f} - {upper:.2f})"
    lbl_p = (f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.3f}") + (" *" if is_sig else "")
    c_lbl = f"{cohort} (N={cohort_counts.get(cohort, 0)})"

    ax.text(
        1.10, y_pos, f"{c_lbl}: {lbl_or}",
        transform=ax.get_yaxis_transform(), va="center", ha="left",
        fontsize=9.5, color=text_color, weight=weight,
    )
    ax.text(
        1.72, y_pos, lbl_p,
        transform=ax.get_yaxis_transform(), va="center", ha="left",
        fontsize=9.5, color=text_color, weight=weight,
    )


def _draw_single_forest_entry(
    ax: plt.Axes,
    y_pos: float,
    row: pd.Series,
    cohort_counts: Dict[str, int],
) -> None:
    """Draws a single errorbar and text label pair on the forest plot."""
    cohort = str(row["Cohort"])
    or_val = float(row["Odds Ratio (OR)"])
    lower = float(row["95% CI Lower"])
    upper = float(row["95% CI Upper"])
    p_val = float(row["p-value"])

    disp_lower = max(_CI_LOWER_DISPLAY, lower)
    disp_upper = min(_CI_UPPER_DISPLAY, upper)

    _draw_forest_errorbar(ax, y_pos, cohort, or_val, disp_lower, disp_upper)
    _draw_forest_text_annotations(
        ax, y_pos, cohort, or_val, lower, upper, p_val, cohort_counts
    )


def _draw_variable_group(
    ax: plt.Axes,
    y_pos: float,
    var: str,
    all_results: pd.DataFrame,
    cohort_order: List[str],
    cohort_counts: Dict[str, int],
) -> Tuple[float, float]:
    """Draws errorbars for all cohorts of a given variable."""
    sub_df = all_results[all_results["Variable"] == var].copy()
    sub_df["Cohort"] = pd.Categorical(
        sub_df["Cohort"], categories=cohort_order, ordered=True
    )
    sub_df = sub_df.sort_values("Cohort")

    group_tick = y_pos + (len(sub_df) - 1) / 2.0
    for _, row in sub_df.iterrows():
        _draw_single_forest_entry(ax, y_pos, row, cohort_counts)
        y_pos += 1.0
    y_pos += 0.8
    return y_pos, group_tick


def _draw_forest_variable_rows(
    ax: plt.Axes,
    variables: List[str],
    all_results: pd.DataFrame,
    cohort_order: List[str],
    cohort_counts: Dict[str, int],
) -> Tuple[float, List[float], List[str]]:
    """Draws individual variable error bars and OR text annotations."""
    y_pos = 0.0
    y_ticks: List[float] = []
    y_labels: List[str] = []

    for var in reversed(variables):
        y_pos, group_tick = _draw_variable_group(
            ax, y_pos, var, all_results, cohort_order, cohort_counts
        )
        y_ticks.append(group_tick)
        y_labels.append(var)

    return y_pos, y_ticks, y_labels


def _add_forest_headers_and_ticks(
    ax: plt.Axes, y_pos: float, y_ticks: List[float], y_labels: List[str]
) -> None:
    """Sets y-ticks and column headers on forest plot."""
    header_y = y_pos - 0.2
    ax.text(
        1.10, header_y, "Cohort & OR (95% CI)",
        transform=ax.get_yaxis_transform(), va="bottom", ha="left",
        fontsize=10.5, color="#111111", weight="bold",
    )
    ax.text(
        1.72, header_y, "p-value",
        transform=ax.get_yaxis_transform(), va="bottom", ha="left",
        fontsize=10.5, color="#111111", weight="bold",
    )
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=11, fontweight="bold", color="#222222")
    ax.set_ylim(-0.8, y_pos)
    ax.set_xlim(_CI_LOWER_DISPLAY, _CI_UPPER_DISPLAY)


def _add_forest_titles_and_labels(ax: plt.Axes, n_pooled: int) -> None:
    """Sets x-ticks, title, and directional annotation arrows on forest plot."""
    ax.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="x", which="both", labelsize=10.5)

    ax.set_xlabel(
        "Odds Ratio for Immunotherapy Response (log scale)",
        fontsize=12, labelpad=6, weight="bold",
    )
    ax.set_title(
        f"Univariate Associations with Immunotherapy Response (Forest Plot, N={n_pooled})",
        fontsize=15, fontweight="bold", pad=15,
    )
    ax.text(
        0.95, -0.088, "Favours Responder (OR > 1.0) \u2192",
        transform=ax.transAxes, ha="right", va="top", color="#555555",
        fontsize=9.5, style="italic",
    )
    ax.text(
        0.05, -0.088, "\u2190 Favours Non-Responder (OR < 1.0)",
        transform=ax.transAxes, ha="left", va="top", color="#555555",
        fontsize=9.5, style="italic",
    )


def _configure_forest_axes(
    ax: plt.Axes,
    y_pos: float,
    y_ticks: List[float],
    y_labels: List[str],
    n_pooled: int,
) -> None:
    """Configures axis limits, headers, tick labels, and title formatting."""
    _add_forest_headers_and_ticks(ax, y_pos, y_ticks, y_labels)
    _add_forest_titles_and_labels(ax, n_pooled)


def _create_legend_handle(cohort: str, count: int) -> mlines.Line2D:
    """Creates a Line2D legend handle for a given cohort."""
    is_pooled = cohort == "Pooled Trials"
    lbl = f"{cohort} Benchmark (N={count})" if is_pooled else f"{cohort} (N={count})"
    return mlines.Line2D(
        [0], [0],
        marker="o" if is_pooled else "s",
        color="none",
        markerfacecolor=get_cohort_color(cohort),
        markeredgecolor="none",
        markersize=8 if is_pooled else 7,
        label=lbl,
    )


def _add_forest_legend(ax: plt.Axes, cohort_counts: Dict[str, int]) -> None:
    """Adds cohort markers legend to forest plot."""
    cohort_keys = list(cohort_counts.keys())
    legend_elements = [
        _create_legend_handle(c, cohort_counts.get(c, 0)) for c in cohort_keys
    ]
    ax.legend(
        handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=min(4, len(cohort_keys)), frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.5,
    )


def _save_forest_artifacts(
    fig: plt.Figure, all_results: pd.DataFrame, plot_dir: Path
) -> None:
    """Saves output forest plot PNG figure and statistics CSV."""
    out_path = plot_dir / "univariate_associations.png"
    save_fig(fig, out_path)
    print(f"\nSaved univariate associations forest plot to {rel_path(out_path)}")

    csv_path = plot_dir / "univariate_associations_stats.csv"
    all_results.to_csv(csv_path, index=False)
    print(f"Saved univariate association statistics table to {rel_path(csv_path)}")


def _plot_univariate_associations(
    all_results: pd.DataFrame, cohort_counts: Dict[str, int], plot_dir: Path
) -> None:
    """Renders a publication-ready Forest Plot of Odds Ratios with 95% CIs."""
    cohort_order = ["Pooled Trials"] + [c for c in cohort_counts.keys() if c != "Pooled Trials"]
    variables = all_results["Variable"].unique().tolist()

    n_entries = len(variables) * len(cohort_order)
    fig_height = max(11, int(n_entries * 0.30) + 4)

    fig, ax = plt.subplots(figsize=(14, fig_height))
    plt.subplots_adjust(left=0.28, right=0.62, top=0.92, bottom=0.15)

    ax.set_xscale("log")
    ax.axvline(1.0, color="#37474F", linestyle="--", linewidth=1.5, zorder=1)

    y_pos, y_ticks, y_labels = _draw_forest_variable_rows(
        ax, variables, all_results, cohort_order, cohort_counts
    )
    _configure_forest_axes(
        ax, y_pos, y_ticks, y_labels, cohort_counts.get("Pooled Trials", 0)
    )
    _add_forest_legend(ax, cohort_counts)

    sns.despine(ax=ax, top=True, right=True)
    ax.set_axisbelow(True)
    _save_forest_artifacts(fig, all_results, plot_dir)


def _build_pooled_df(cohorts: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenates available clinical columns across all cohort DataFrames.

    Args:
        cohorts: Dictionary of cohort DataFrames.

    Returns:
        Pooled DataFrame.
    """
    return pd.concat([
        cohorts[c][
            [col for col in _POOLED_COLS if col in cohorts[c].columns]
        ]
        for c in cohorts
    ], ignore_index=True)


def _compute_cohort_counts(
    cohorts: Dict[str, pd.DataFrame], pooled_df: pd.DataFrame
) -> Dict[str, int]:
    """Computes dynamic non-null sample size N for each cohort and pooled dataset.

    Args:
        cohorts: Dictionary of per-cohort DataFrames.
        pooled_df: Combined pooled DataFrame.

    Returns:
        Mapping of cohort name to non-null response count.
    """
    counts: Dict[str, int] = {"Pooled Trials": len(pooled_df[pooled_df["response"].notna()])}
    for c in cohorts:
        counts[c] = len(cohorts[c][cohorts[c]["response"].notna()])
    return counts


def main() -> None:
    """Executes the univariate association analysis pipeline."""
    print("==================================================")
    print("Computing Univariate Associations with Response (Forest Plot)")
    print("==================================================\n")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    cohorts = _prepare_clinical_cohorts(DATA_DIR)

    results_list: List[pd.DataFrame] = []
    for cname, df in cohorts.items():
        results_list.append(calculate_associations(df, cname))

    pooled_df = _build_pooled_df(cohorts)
    results_pooled = calculate_associations(pooled_df, "Pooled Trials")
    results_list.append(results_pooled)
    cohort_counts = _compute_cohort_counts(cohorts, pooled_df)

    all_results = pd.concat(results_list, ignore_index=True)
    print(all_results.to_string())
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
