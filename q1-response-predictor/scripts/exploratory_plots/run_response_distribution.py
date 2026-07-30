"""
RECIST Response Distribution Visualisation for Immunotherapy Cohorts.

Calculates and visualises the response rate distribution (CR/PR vs. PD) across
individual melanoma clinical trial cohorts (Liu 2019, Hugo 2016, Riaz 2017) using stacked bar charts
and evaluates cross-cohort homogeneity via Chi-Square test.
"""

import contextlib
from pathlib import Path
import sys
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.styles import RESPONSE_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, LOG_DIR, PLOTS_DIR
from src.utils.plotting import save_fig

set_presentation_style()

# Module-level Constants
PLOT_DIR = PLOTS_DIR / "clinical"
LOG_PATH = LOG_DIR / "run_response_distribution.log"


def _prepare_response_summary(data_dir: Path) -> Tuple[pd.DataFrame, float]:
    """Extracts response counts and computes Chi-square test of homogeneity.

    Args:
        data_dir: Path to project data directory.

    Returns:
        Tuple of (response summary DataFrame, Chi-square p-value).
    """
    _, clin_liu = load_liu_2019(data_dir)
    _, clin_hugo = load_hugo_2016(data_dir)
    _, clin_riaz = load_riaz_2017(data_dir)

    cohorts = {
        "Liu 2019": clin_liu,
        "Hugo 2016": clin_hugo,
        "Riaz 2017": clin_riaz,
    }

    data = []
    for name, df in cohorts.items():
        if "response" in df.columns:
            resp_counts = df["response"].value_counts()
            r_count = resp_counts.get(1.0, 0)
            nr_count = resp_counts.get(0.0, 0)
            total = r_count + nr_count
            data.append({
                "Cohort": name,
                "Responder (CR/PR)": r_count,
                "Non-responder (PD)": nr_count,
                "Total": total,
                "Response Rate (%)": (r_count / total) * 100 if total > 0 else 0,
            })

    df_resp = pd.DataFrame(data)
    contingency_table = df_resp[["Responder (CR/PR)", "Non-responder (PD)"]].values
    _, p_val, _, _ = chi2_contingency(contingency_table)

    return df_resp, p_val


def _plot_response_distribution(df_resp: pd.DataFrame, p_val: float, plot_dir: Path) -> None:
    """Plots stacked bar chart of RECIST response proportions.

    Args:
        df_resp: Response count and percentage DataFrame.
        p_val: Chi-square test p-value.
        plot_dir: Path to export output figure artifact.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    cohort_names = df_resp["Cohort"].tolist()
    responders = df_resp["Responder (CR/PR)"].values
    non_responders = df_resp["Non-responder (PD)"].values
    totals = df_resp["Total"].values

    r_prop = responders / totals * 100
    nr_prop = non_responders / totals * 100

    c_responder = RESPONSE_PALETTE["CR/PR"]
    c_nonresponder = RESPONSE_PALETTE["PD"]

    ax.bar(cohort_names, r_prop, label="Responder (CR/PR)", color=c_responder, width=0.55, edgecolor="black")
    ax.bar(cohort_names, nr_prop, bottom=r_prop, label="Non-responder (PD)", color=c_nonresponder, width=0.55, edgecolor="black")

    for i in range(len(cohort_names)):
        ax.text(i, r_prop[i] / 2, f"{responders[i]}\n({r_prop[i]:.1f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        ax.text(i, r_prop[i] + nr_prop[i] / 2, f"{non_responders[i]}\n({nr_prop[i]:.1f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=10)

    ax.set_ylabel("Proportion of Patients (%)", fontsize=12, fontweight="bold")
    ax.set_title("RECIST Response Distribution across IO Cohorts", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    ax.text(
        0.5,
        -0.15,
        f"Chi-squared test of homogeneity: p = {p_val:.4f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgrey", edgecolor="none", alpha=0.5),
    )

    out_path = plot_dir / "response_distribution.png"
    save_fig(fig, out_path)
    print(f"\nSaved response distribution plot to {out_path.relative_to(BASE_DIR).as_posix()}")


def main() -> None:
    """Executes the response distribution analysis pipeline."""
    print("==================================================")
    print("Generating Response Distribution Visualisation")
    print("==================================================\n")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    df_resp, p_val = _prepare_response_summary(DATA_DIR)
    print(df_resp.to_string(index=False))
    print(f"\nChi-squared test p-value: {p_val:.4f}")

    _plot_response_distribution(df_resp, p_val, PLOT_DIR)

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
