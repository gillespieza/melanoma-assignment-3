"""
Kaplan-Meier Overall Survival Stratified by Immunotherapy Response.

Renders 3-panel Kaplan-Meier survival curves comparing Responders (CR/PR) vs. Non-responders (PD)
across Liu 2019, Hugo 2016, and Riaz 2017 trial cohorts, performing log-rank tests for statistical separation.
"""

import contextlib
from pathlib import Path
import sys
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import seaborn as sns

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.styles import RESPONSE_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, LOG_DIR, PLOTS_DIR, rel_path
from src.utils.plotting import save_fig

set_presentation_style()

# Module-level Constants
PLOT_DIR = PLOTS_DIR / "clinical"
LOG_PATH = LOG_DIR / "run_response_km_curves.log"


def plot_cohort_km_by_response(ax: plt.Axes, df_clin: pd.DataFrame, cohort_name: str) -> None:
    """Plots Kaplan-Meier survival curves stratified by immunotherapy response for a single cohort.

    Args:
        ax: Matplotlib Axes to draw on.
        df_clin: Clinical DataFrame.
        cohort_name: Title label string for the cohort.
    """
    time_col = "os_months" if "os_months" in df_clin.columns else ("OS_MONTHS" if "OS_MONTHS" in df_clin.columns else None)
    event_col = "os_status" if "os_status" in df_clin.columns else ("OS_STATUS" if "OS_STATUS" in df_clin.columns else None)

    if not time_col or not event_col or "response" not in df_clin.columns:
        print(f"Skipping {cohort_name}: missing survival or response columns")
        return

    df = df_clin[[time_col, event_col, "response"]].copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df = df.dropna().copy()
    df = df[df[time_col] > 0]

    responders = df[df["response"] == 1.0]
    non_responders = df[df["response"] == 0.0]

    if len(responders) == 0 or len(non_responders) == 0:
        print(f"Skipping {cohort_name}: missing response groups")
        return

    kmf_r = KaplanMeierFitter()
    kmf_nr = KaplanMeierFitter()

    kmf_r.fit(responders[time_col], event_observed=responders[event_col])
    kmf_nr.fit(non_responders[time_col], event_observed=non_responders[event_col])

    kmf_r.plot_survival_function(ax=ax, color=RESPONSE_PALETTE["CR/PR"], linewidth=2.5, ci_show=True, alpha=0.15, label=f"Responder (N={len(responders)})")
    kmf_nr.plot_survival_function(ax=ax, color=RESPONSE_PALETTE["PD"], linewidth=2.5, ci_show=True, alpha=0.15, label=f"Non-responder (N={len(non_responders)})")

    results = logrank_test(
        responders[time_col], non_responders[time_col], responders[event_col], non_responders[event_col]
    )
    p_val = results.p_value

    ax.set_title(cohort_name, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (months)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Survival Probability", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", color="#E5E7EB", alpha=0.6, linewidth=0.8)

    p_text = f"Log-rank p = {p_val:.4f}" if p_val >= 0.0001 else "Log-rank p < 0.0001"
    ax.text(
        0.95,
        0.12,
        p_text,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#D1D5DB", alpha=0.85),
    )

    ax.legend(loc="lower left", fontsize=10)


def main() -> None:
    """Executes the response-stratified Kaplan-Meier analysis pipeline."""
    print("==================================================")
    print("Generating Response-Stratified KM Survival Curves")
    print("==================================================\n")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    _, clin_liu = load_liu_2019(DATA_DIR)
    _, clin_hugo = load_hugo_2016(DATA_DIR)
    _, clin_riaz = load_riaz_2017(DATA_DIR)

    fig, axes = plt.subplots(1, 3, figsize=(22, 5.5))

    plot_cohort_km_by_response(axes[0], clin_liu, "Liu 2019")
    plot_cohort_km_by_response(axes[1], clin_hugo, "Hugo 2016")
    plot_cohort_km_by_response(axes[2], clin_riaz, "Riaz 2017")

    fig.suptitle("Overall Survival by Immunotherapy Response (RECIST)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = PLOT_DIR / "km_os_by_response.png"
    save_fig(fig, out_path)
    
    sub_path = BASE_DIR / "plots" / "clinical" / "km_os_by_response.png"
    if sub_path != out_path:
        sub_path.parent.mkdir(parents=True, exist_ok=True)
        save_fig(fig, sub_path)

    # Export transparent copies
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.patch.set_alpha(0.0)
    
    out_trans = PLOT_DIR / "km_os_by_response_transparent.png"
    sub_trans = BASE_DIR / "plots" / "clinical" / "km_os_by_response_transparent.png"
    fig.savefig(out_trans, transparent=True, bbox_inches="tight", dpi=300)
    fig.savefig(sub_trans, transparent=True, bbox_inches="tight", dpi=300)

    print(f"Saved response-stratified KM plots to {rel_path(out_path)} and {rel_path(out_trans)}")

    print("==================================================")
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
