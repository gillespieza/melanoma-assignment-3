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

from src.styles import RESPONSE_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, LOG_DIR, PLOTS_DIR, rel_path
from src.utils.plotting import add_km_risk_table, save_fig

set_presentation_style()

# Module-level Constants
_SUBPROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = _SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "run_response_km_curves.log"
PLOT_DIR = PLOTS_DIR / "clinical"


def _resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def plot_cohort_km_by_response(ax: plt.Axes, df_clin: pd.DataFrame, cohort_name: str) -> None:
    """Plots Kaplan-Meier survival curves stratified by immunotherapy response for a single cohort."""
    time_col = _resolve_col(df_clin, ["OS_MONTHS", "os_months", "OS_DAYS"])
    event_col = _resolve_col(df_clin, ["OS_STATUS", "os_status", "OS_EVENT"])
    resp_col = _resolve_col(df_clin, ["RESPONSE_BINARY", "response", "RESPONSE"])

    if not time_col or not event_col or not resp_col:
        print(f"Skipping {cohort_name}: missing survival or response columns")
        return

    df = df_clin[[time_col, event_col, resp_col]].copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df[resp_col] = pd.to_numeric(df[resp_col], errors="coerce")
    df = df.dropna().copy()
    df = df[df[time_col] > 0]

    responders = df[df[resp_col] == 1.0]
    non_responders = df[df[resp_col] == 0.0]

    if len(responders) == 0 or len(non_responders) == 0:
        print(f"Skipping {cohort_name}: missing response groups")
        return

    kmf_r = KaplanMeierFitter()
    kmf_nr = KaplanMeierFitter()

    kmf_r.fit(responders[time_col], event_observed=responders[event_col])
    kmf_nr.fit(non_responders[time_col], event_observed=non_responders[event_col])

    kmf_r.plot_survival_function(ax=ax, color=RESPONSE_PALETTE["CR/PR"], linewidth=2.5, ci_show=True, alpha=0.15, label=f"Responder (N={len(responders)})")
    kmf_nr.plot_survival_function(ax=ax, color=RESPONSE_PALETTE["PD"], linewidth=2.5, ci_show=True, alpha=0.15, label=f"Non-responder (N={len(non_responders)})")

    add_km_risk_table([kmf_r, kmf_nr], ax)

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

    CONFIG_PATH = BASE_DIR / "config" / "datasets.yaml"
    from src.config.datasets import load_dataset_config
    all_configs = load_dataset_config(CONFIG_PATH)
    dataset_configs = [c for c in all_configs if c.merge_enabled]
    cohort_order = [c.cohort_name for c in dataset_configs]

    cohort_data: dict[str, pd.DataFrame] = {}
    for config in dataset_configs:
        clin_path = DATA_DIR / "processed" / config.processed_directory / "clin_cleaned.csv"
        if clin_path.exists():
            cohort_data[config.cohort_name] = pd.read_csv(clin_path, index_col="SAMPLE_ID")

    n_cohorts = len(cohort_order)
    max_cols = 2
    n_cols = min(max_cols, n_cohorts)
    n_rows = (n_cohorts + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    axes_flat = np.array(axes).flatten() if n_cohorts > 1 else [axes]

    for i, label in enumerate(cohort_order):
        ax = axes_flat[i]
        df_clin = cohort_data.get(label, pd.DataFrame())
        plot_cohort_km_by_response(ax, df_clin, label)

    # Hide any unused subplot axes
    for j in range(n_cohorts, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Overall Survival by Immunotherapy Response (RECIST)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = PLOT_DIR / "km_os_by_response.png"
    save_fig(fig, out_path)

    # Export transparent copies
    fig.patch.set_alpha(0.0)
    for ax in axes_flat:
        ax.patch.set_alpha(0.0)

    out_trans = PLOT_DIR / "km_os_by_response_transparent.png"
    fig.savefig(out_trans, transparent=True, bbox_inches="tight", dpi=300)

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
            print(f"Logging console output to {LOG_PATH.relative_to(_SUBPROJECT_ROOT).as_posix()}")
            main()
