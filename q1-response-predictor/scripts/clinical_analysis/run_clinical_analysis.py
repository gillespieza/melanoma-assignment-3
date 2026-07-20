"""
Phase 1 Clinical Analysis Script for Melanoma Cohorts.

Generates unstratified Kaplan-Meier Overall Survival (OS) curves across all clinical
study cohorts (Liu 2019, Hugo 2016, Riaz 2017, TCGA-SKCM) in a 2x2 grid layout.
"""

import contextlib
from pathlib import Path
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

# Bootstrap project root resolution for top-level import
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.styles import get_cohort_color, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import find_project_root
from src.utils.plotting import save_fig

# Module-level Constants
DATA_DIR = find_project_root(Path(__file__).resolve()) / "data"
PLOT_DIR = find_project_root(Path(__file__).resolve()) / "plots" / "clinical"
LOG_DIR = find_project_root(Path(__file__).resolve()) / "logs"
LOG_PATH = LOG_DIR / "run_clinical_analysis.log"


def _resolve_os_columns(df_clin: pd.DataFrame) -> Tuple[str, str]:
    """Determines overall survival time and event column names.

    Normalises between trial cohorts (lowercase os_months/os_status)
    and TCGA (uppercase OS_MONTHS/OS_STATUS).

    Args:
        df_clin: Clinical DataFrame.

    Returns:
        Tuple of (time_column_name, event_column_name).
    """
    if "os_months" in df_clin.columns:
        return "os_months", "os_status"
    return "OS_MONTHS", "OS_STATUS"


def _clean_os(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    stratify_col: str | None = None,
) -> pd.DataFrame:
    """Drops rows with missing/invalid survival data and casts types for lifelines.

    Args:
        df: Clinical DataFrame.
        time_col: Name of OS time column.
        event_col: Name of OS event indicator column.
        stratify_col: Optional stratification column to preserve.

    Returns:
        Clean DataFrame containing valid survival rows only.
    """
    cols = [time_col, event_col]
    if stratify_col and stratify_col in df.columns:
        cols.append(stratify_col)
    out = df[cols].copy()
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
    out[event_col] = pd.to_numeric(out[event_col], errors="coerce")
    out = out.dropna(subset=[time_col, event_col])
    out = out[out[time_col] > 0]
    return out


def plot_km_os(ax: plt.Axes, df_clin: pd.DataFrame, cohort_label: str) -> None:
    """Plots an unstratified Kaplan-Meier OS curve on the given Matplotlib axes.

    Args:
        ax: Matplotlib Axes to draw on.
        df_clin: Clinical DataFrame.
        cohort_label: Title label string for the cohort.
    """
    time_col, event_col = _resolve_os_columns(df_clin)
    df = _clean_os(df_clin, time_col, event_col)

    cohort_color = get_cohort_color(cohort_label)

    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], event_observed=df[event_col], label=f"N = {len(df)}")
    kmf.plot_survival_function(ax=ax, ci_show=True, color=cohort_color, linewidth=2)

    median_surv = kmf.median_survival_time_
    if np.isfinite(median_surv):
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(median_surv, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(
            0.95,
            0.05,
            f"Median OS = {median_surv:.1f} mo",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.8),
        )

    ax.set_title(cohort_label, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (months)", fontsize=10)
    ax.set_ylabel("Overall Survival Probability", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def main() -> None:
    """Executes the Kaplan-Meier overall survival analysis across cohorts."""
    print("==================================================")
    print("Clinical Analysis — Phase 1: Kaplan-Meier OS Curves")
    print("==================================================\n")

    set_presentation_style()
    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    _, clin_liu = load_liu_2019(DATA_DIR)
    _, clin_hugo = load_hugo_2016(DATA_DIR)
    _, clin_riaz = load_riaz_2017(DATA_DIR)

    tcga_clin_path = DATA_DIR / "processed" / "skcm_tcga_pan_can_atlas_2018" / "clin_cleaned.csv"
    if not tcga_clin_path.exists():
        raise FileNotFoundError(f"Processed TCGA clinical file not found at {tcga_clin_path.relative_to(BASE_DIR).as_posix()}. Run preprocessing first.")
    clin_tcga = pd.read_csv(tcga_clin_path, index_col=0)

    cohorts: List[Tuple[str, pd.DataFrame]] = [
        ("Liu 2019", clin_liu),
        ("Hugo 2016", clin_hugo),
        ("Riaz 2017", clin_riaz),
        ("TCGA-SKCM", clin_tcga),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    flat_axes = axes.ravel()

    for ax, (label, df_clin) in zip(flat_axes, cohorts):
        print(f"  Plotting KM curve for {label} ({len(df_clin)} samples)...")
        plot_km_os(ax, df_clin, label)

    fig.suptitle("Overall Survival — All Cohorts", fontsize=16, fontweight="bold", y=1.01)

    out_path = PLOT_DIR / "km_os_grid.png"
    save_fig(fig, out_path)
    print(f"\nSaved 2×2 KM plot to {out_path.relative_to(BASE_DIR).as_posix()}")

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
