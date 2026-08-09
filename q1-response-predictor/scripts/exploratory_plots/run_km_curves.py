"""
Kaplan-Meier Survival Curves for TCGA-SKCM Cohort.

Renders survival curves stratified by Age, Sex, Tumor Stage, and Mutational Burden in TCGA-SKCM,
computing log-rank test p-values and exporting figures.
"""

import contextlib
from pathlib import Path
import sys
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import seaborn as sns

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.styles import (
    COHORT_PALETTE,
    OKABE_ITO,
    PHENOTYPE_PALETTE,
    RESPONSE_PALETTE,
    SEX_PALETTE,
    set_presentation_style,
)
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, LOG_DIR, PLOTS_DIR, rel_path
from src.utils.plotting import add_km_risk_table, save_fig

set_presentation_style()

# Module-level Constants
CLINICAL_FILE = DATA_DIR / "processed" / "skcm_tcga_pan_can_atlas_2018" / "clin_cleaned.csv"
PLOT_DIR = PLOTS_DIR / "clinical"
LOG_PATH = LOG_DIR / "run_km_curves.log"


def _plot_km(
    df: pd.DataFrame,
    group_col: str,
    title: str,
    filename: str,
    palette: List[str],
    labels: Optional[dict] = None,
    split_median: bool = False,
) -> None:
    """Helper function to plot Kaplan-Meier survival curves.

    Args:
        df: Clinical DataFrame with OS_MONTHS and OS_STATUS columns.
        group_col: Name of column to stratify groups.
        title: Plot title.
        filename: Destination filename.
        palette: Color palette list.
        labels: Optional label mapping dictionary.
        split_median: If True, splits group_col into High/Low by median.
    """
    fig, ax = plt.subplots(figsize=(10, 7.5))

    df_sub = df.copy()
    if split_median:
        median_val = df_sub[group_col].median()
        df_sub["temp_group"] = df_sub[group_col].apply(
            lambda x: f"High (>= {median_val:.2f})" if x >= median_val else f"Low (< {median_val:.2f})"
        )
        group_col = "temp_group"

    groups = df_sub[group_col].unique()
    groups = [g for g in groups if pd.notna(g) and str(g).lower() not in ["nan", "unknown"]]
    groups = sorted(groups)

    # Create a fresh KaplanMeierFitter per group so each fitted object can be
    # collected and passed to add_km_risk_table (reusing one instance loses state).
    kmf_list = []
    for i, g in enumerate(groups):
        mask = df_sub[group_col] == g
        label = labels[g] if labels and g in labels else str(g)
        label = f"{label} (N={mask.sum()})"
        kmf = KaplanMeierFitter()
        kmf.fit(df_sub.loc[mask, "OS_MONTHS"], df_sub.loc[mask, "OS_STATUS"], label=label)
        kmf.plot_survival_function(ax=ax, color=palette[i % len(palette)], ci_show=False, linewidth=2.5)
        kmf_list.append(kmf)

    add_km_risk_table(kmf_list, ax)

    if len(groups) == 2:
        g1_mask = df_sub[group_col] == groups[0]
        g2_mask = df_sub[group_col] == groups[1]
        results = logrank_test(
            df_sub.loc[g1_mask, "OS_MONTHS"],
            df_sub.loc[g2_mask, "OS_MONTHS"],
            df_sub.loc[g1_mask, "OS_STATUS"],
            df_sub.loc[g2_mask, "OS_STATUS"],
        )
        p_val = results.p_value
        p_text = f"Log-Rank p = {p_val:.2e}" if p_val < 0.001 else f"Log-Rank p = {p_val:.3f}"
    else:
        results = multivariate_logrank_test(df_sub["OS_MONTHS"], df_sub[group_col], df_sub["OS_STATUS"])
        p_val = results.p_value
        p_text = f"Log-Rank p = {p_val:.2e}" if p_val < 0.001 else f"Log-Rank p = {p_val:.3f}"

    ax.text(
        0.05,
        0.12,
        p_text,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Overall Survival (Months)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Survival Probability", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=True)

    out_path = PLOT_DIR / filename
    save_fig(fig, out_path)
    print(f"Saved plot to {out_path.relative_to(BASE_DIR).as_posix()}")


def main() -> None:
    """Executes the Kaplan-Meier survival curves pipeline for TCGA-SKCM."""
    print("==================================================")
    print("Generating Kaplan-Meier Survival Curves for TCGA-SKCM")
    print("==================================================")

    if not CLINICAL_FILE.exists():
        raise FileNotFoundError(f"Clinical file not found at {CLINICAL_FILE.relative_to(BASE_DIR).as_posix()}")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(CLINICAL_FILE)
    print(f"Loaded clinical data with shape: {df.shape}")

    df = df.dropna(subset=["OS_MONTHS", "OS_STATUS"])
    print(f"Number of samples with valid survival data: {len(df)}")

    palette_sex = [SEX_PALETTE["Male"], SEX_PALETTE["Female"]]
    palette_age = [PHENOTYPE_PALETTE["Immune Cold"], PHENOTYPE_PALETTE["Immune Hot"]]
    palette_stage = OKABE_ITO[:4]

    if "SEX" in df.columns:
        _plot_km(df, "SEX", "TCGA-SKCM Overall Survival by Sex", "km_sex.png", palette_sex)

    if "AGE" in df.columns:
        _plot_km(df, "AGE", "TCGA-SKCM Overall Survival by Age Median Split", "km_age.png", palette_age, split_median=True)

    if "AJCC_PATHOLOGIC_TUMOR_STAGE" in df.columns:
        df["Stage_Group"] = df["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).apply(
            lambda x: "Stage I" if "I" in x and "IV" not in x and "III" not in x and "II" not in x else (
                "Stage II" if "II" in x and "III" not in x else (
                    "Stage III" if "III" in x else (
                        "Stage IV" if "IV" in x else np.nan
                    )
                )
            )
        )
        df_stage = df.dropna(subset=["Stage_Group"])
        if len(df_stage) > 0:
            _plot_km(df_stage, "Stage_Group", "TCGA-SKCM Overall Survival by Pathologic Stage", "km_stage.png", palette_stage)

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
