"""
Univariate Odds Ratio Forest Plot Generation.

Computes Fisher's exact test Odds Ratios and 95% Confidence Intervals for categorical clinical/genomic features
across pooled immunotherapy trial cohorts and exports a forest plot figure and summary CSV.
"""

import contextlib
from pathlib import Path
import sys
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

# Bootstrap project root resolution for top-level import
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.styles import RESPONSE_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import find_project_root
from src.utils.plotting import save_fig

# Module-level Constants
DATA_DIR = find_project_root(Path(__file__).resolve()) / "data"
PLOT_DIR = find_project_root(Path(__file__).resolve()) / "plots" / "clinical"
LOG_DIR = find_project_root(Path(__file__).resolve()) / "logs"
LOG_PATH = LOG_DIR / "run_forest_plot.log"


def calculate_odds_ratio(
    df: pd.DataFrame, col: str, value_exposed: str, value_unexposed: str
) -> Optional[Dict[str, float]]:
    """Calculates Fisher's Exact test Odds Ratio and 95% Confidence Interval.

    Args:
        df: Input DataFrame containing feature column and 'response'.
        col: Feature column name.
        value_exposed: Exposed category label.
        value_unexposed: Unexposed reference category label.

    Returns:
        Dictionary containing OR, CI_lower, CI_upper, p_value, and sample counts.
    """
    temp = df[[col, "response"]].dropna()
    if len(temp) == 0:
        return None

    A = len(temp[(temp[col] == value_exposed) & (temp["response"] == 1.0)])
    B = len(temp[(temp[col] == value_exposed) & (temp["response"] == 0.0)])
    C = len(temp[(temp[col] == value_unexposed) & (temp["response"] == 1.0)])
    D = len(temp[(temp[col] == value_unexposed) & (temp["response"] == 0.0)])

    table = [[A, B], [C, D]]
    _, p_val = fisher_exact(table)

    if A == 0 or B == 0 or C == 0 or D == 0:
        A_c, B_c, C_c, D_c = A + 0.5, B + 0.5, C + 0.5, D + 0.5
    else:
        A_c, B_c, C_c, D_c = A, B, C, D

    or_val = (A_c * D_c) / (B_c * C_c)
    se_ln_or = np.sqrt(1 / A_c + 1 / B_c + 1 / C_c + 1 / D_c)

    ci_lower = np.exp(np.log(or_val) - 1.96 * se_ln_or)
    ci_upper = np.exp(np.log(or_val) + 1.96 * se_ln_or)

    return {
        "OR": or_val,
        "CI_lower": ci_lower,
        "CI_upper": ci_upper,
        "p_value": p_val,
        "Counts": f"Exposed: {A}/{A+B}, Unexposed: {C}/{C+D}",
    }


def main() -> None:
    """Executes the univariate forest plot pipeline."""
    print("==================================================")
    print("Generating Forest Plot of Univariate Odds Ratios")
    print("==================================================\n")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    _, clin_liu = load_liu_2019(DATA_DIR)
    _, clin_hugo = load_hugo_2016(DATA_DIR)
    _, clin_riaz = load_riaz_2017(DATA_DIR)

    for df in [clin_liu, clin_hugo, clin_riaz]:
        if "SEX" in df.columns:
            df["SEX"] = df["SEX"].map({"Male": "Male", "Female": "Female", "M": "Male", "F": "Female"})
        if "CLINICAL_STAGE" in df.columns:
            df["CLINICAL_STAGE"] = df["CLINICAL_STAGE"].apply(
                lambda x: "IV" if str(x).startswith("IV") else ("III" if str(x).startswith("III") else np.nan)
            )

    common_cols = [
        "response",
        "SEX",
        "CLINICAL_STAGE",
        "mut_BRAF",
        "mut_NRAS",
        "mut_NF1",
        "TMB_NONSYNONYMOUS",
        "AGE_AT_DIAGNOSIS",
        "AGE",
    ]

    for df in [clin_liu, clin_hugo, clin_riaz]:
        age_col = [
            c for c in df.columns if c.upper() in ["AGE", "AGE_AT_DIAGNOSIS", "AGE (YRS)", "AGE_AT_DIAGNOSIS"]
        ]
        if age_col:
            df["age_standardized"] = pd.to_numeric(df[age_col[0]], errors="coerce")
        else:
            df["age_standardized"] = np.nan

    pooled_df = pd.concat(
        [
            clin_liu[[c for c in common_cols + ["age_standardized"] if c in clin_liu.columns]],
            clin_hugo[[c for c in common_cols + ["age_standardized"] if c in clin_hugo.columns]],
            clin_riaz[[c for c in common_cols + ["age_standardized"] if c in clin_riaz.columns]],
        ],
        ignore_index=True,
    )

    tmb_median = pooled_df["TMB_NONSYNONYMOUS"].median()
    pooled_df["TMB_High"] = pooled_df["TMB_NONSYNONYMOUS"].apply(lambda x: "High" if x >= tmb_median else "Low")

    age_median = pooled_df["age_standardized"].median()
    pooled_df["Age_High"] = pooled_df["age_standardized"].apply(
        lambda x: "High" if x >= age_median else ("Low" if pd.notna(x) else np.nan)
    )

    features = [
        {"label": "TMB (High vs Low)", "col": "TMB_High", "exp": "High", "unexp": "Low"},
        {"label": "Clinical Stage (IV vs III)", "col": "CLINICAL_STAGE", "exp": "IV", "unexp": "III"},
        {"label": "Sex (Male vs Female)", "col": "SEX", "exp": "Male", "unexp": "Female"},
        {"label": "Age (>=60.5 vs <60.5)", "col": "Age_High", "exp": "High", "unexp": "Low"},
        {"label": "BRAF Mutation (Mut vs WT)", "col": "mut_BRAF", "exp": 1.0, "unexp": 0.0},
        {"label": "NRAS Mutation (Mut vs WT)", "col": "mut_NRAS", "exp": 1.0, "unexp": 0.0},
        {"label": "NF1 Mutation (Mut vs WT)", "col": "mut_NF1", "exp": 1.0, "unexp": 0.0},
    ]

    results = []
    for feat in features:
        res = calculate_odds_ratio(pooled_df, feat["col"], feat["exp"], feat["unexp"])
        if res:
            res["Feature"] = feat["label"]
            results.append(res)

    df_res = pd.DataFrame(results)
    print(df_res[["Feature", "OR", "CI_lower", "CI_upper", "p_value", "Counts"]].to_string(index=False))

    df_res.to_csv(PLOT_DIR / "forest_plot_data.csv", index=False)

    set_presentation_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    df_plot = df_res.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(df_plot))

    for i, row in df_plot.iterrows():
        color = RESPONSE_PALETTE["CR/PR"] if row["p_value"] < 0.05 else "#555555"
        ax.errorbar(
            row["OR"],
            i,
            xerr=[[row["OR"] - row["CI_lower"]], [row["CI_upper"] - row["OR"]]],
            fmt="o",
            color="black",
            ecolor=color,
            elinewidth=2.5,
            capsize=4,
            capthick=1.5,
            markersize=7,
        )

    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["Feature"], fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("Odds Ratio (95% CI, Log Scale)", fontsize=12, fontweight="bold")
    ax.set_title("Univariate Odds Ratios for Response to Anti-PD-1", fontsize=14, fontweight="bold", pad=15)

    for i, row in df_plot.iterrows():
        p_str = f"p = {row['p_value']:.4f}" if row["p_value"] >= 0.001 else f"p = {row['p_value']:.2e}"
        text_label = f"OR: {row['OR']:.2f} ({row['CI_lower']:.2f}-{row['CI_upper']:.2f}), {p_str}"
        color = RESPONSE_PALETTE["CR/PR"] if row["p_value"] < 0.05 else "#555555"
        ax.text(row["CI_upper"] * 1.15, i, text_label, va="center", fontsize=9.5, fontweight="bold", color=color)

    out_plot = PLOT_DIR / "forest_plot_odds_ratios.png"
    save_fig(fig, out_plot)
    print(f"\nSaved forest plot to {out_plot.relative_to(BASE_DIR).as_posix()}")

    print("==================================================")
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
