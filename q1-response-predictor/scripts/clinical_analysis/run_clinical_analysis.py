import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Add project root to sys.path for importing src modules
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_liu_2019, load_hugo_2016, load_riaz_2017

# Paths
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots" / "clinical"
PLOT_DIR.mkdir(exist_ok=True, parents=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _resolve_os_columns(df_clin: pd.DataFrame):
    """
    Return (time_col, event_col) after normalising the OS columns
    that differ between IO cohorts (lowercase) and TCGA (uppercase).

    @param df_clin: Clinical DataFrame.
    @return tuple: (time_col, event_col) column name strings.
    """
    if "os_months" in df_clin.columns:
        time_col, event_col = "os_months", "os_status"
    else:
        time_col, event_col = "OS_MONTHS", "OS_STATUS"
    return time_col, event_col


def _clean_os(df: pd.DataFrame, time_col: str, event_col: str, stratify_col: str = None) -> pd.DataFrame:
    """
    Drop rows with missing or non-numeric survival data and ensure
    correct types for KaplanMeierFitter.

    @param df: Clinical DataFrame.
    @param time_col: Name of the OS time column.
    @param event_col: Name of the OS event column.
    @param stratify_col: Optional stratification column to keep.
    @return pd.DataFrame: Cleaned copy with valid OS rows only.
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


def plot_km_os(ax, df_clin: pd.DataFrame, cohort_label: str):
    """
    Plot an unstratified Kaplan-Meier OS curve on the given axes.

    @param ax: Matplotlib Axes to draw on.
    @param df_clin: Cleaned clinical DataFrame.
    @param cohort_label: Title string for this subplot.
    """
    time_col, event_col = _resolve_os_columns(df_clin)
    df = _clean_os(df_clin, time_col, event_col)

    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], event_observed=df[event_col], label=f"n = {len(df)}")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#1f77b4", linewidth=2)

    # Annotate median survival if reached
    median_surv = kmf.median_survival_time_
    if np.isfinite(median_surv):
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(median_surv, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(
            0.95, 0.05, f"Median OS = {median_surv:.1f} mo",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.8),
        )

    ax.set_title(cohort_label, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (months)", fontsize=10)
    ax.set_ylabel("Overall Survival Probability", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("==================================================")
    print("Clinical Analysis — Phase 1: Kaplan-Meier OS Curves")
    print("==================================================\n")

    # Load IO cohorts via data_loaders
    _, clin_liu = load_liu_2019(DATA_DIR)
    _, clin_hugo = load_hugo_2016(DATA_DIR)
    _, clin_riaz = load_riaz_2017(DATA_DIR)

    # Load TCGA-SKCM directly (no expression needed)
    tcga_clin_path = DATA_DIR / "processed" / "skcm_tcga_pan_can_atlas_2018" / "clin_cleaned.csv"
    clin_tcga = pd.read_csv(tcga_clin_path, index_col=0)

    cohorts = [
        ("Liu 2019", clin_liu),
        ("Hugo 2016", clin_hugo),
        ("Riaz 2017", clin_riaz),
        ("TCGA-SKCM", clin_tcga),
    ]

    # ── 2×2 grid ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, (label, df_clin) in zip(axes, cohorts):
        print(f"  Plotting KM curve for {label} ({len(df_clin)} samples)...")
        plot_km_os(ax, df_clin, label)

    fig.suptitle(
        "Overall Survival — All Cohorts",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out_path = PLOT_DIR / "km_os_grid.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved 2×2 KM plot to {out_path}")

    print("\n==================================================")
    print("Done!")
    print("==================================================")


if __name__ == "__main__":
    main()
