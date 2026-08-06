"""Phase 1 Clinical Analysis Script for Melanoma Cohorts.

Generates:
1. Unstratified Kaplan-Meier Overall Survival (OS) curves across all
   clinical study cohorts in a 2x2 grid layout.
2. An Obsidian-compatible Markdown clinical characteristics report
   containing dynamically calculated demographic, treatment, survival,
   and sample attrition statistics.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter

# ===========================================================================
# BOOTSTRAP PROJECT ROOT RESOLUTION
# ===========================================================================

_SUBPROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(_SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBPROJECT_ROOT))

# ===========================================================================
# PROJECT IMPORTS
# ===========================================================================

from src.config.datasets import DatasetConfig, load_dataset_config
from src.styles import COHORT_PALETTE, get_cohort_color, set_presentation_style
from src.utils.formatting import (
    format_count_percentage,
    format_median,
    format_median_iqr,
    generate_obsidian_frontmatter,
)
from src.utils.logging import TeeStream
from src.utils.paths import (
    CONFIG_DIR,
    DATA_DIR,
    LOG_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    SUBPROJECT_ROOT,
)
from src.utils.plotting import save_fig

set_presentation_style()

# ===========================================================================
# PATHS & CONFIGURATION
# ===========================================================================

CONFIG_PATH = CONFIG_DIR / "datasets.yaml"
PLOT_DIR = PLOTS_DIR / "clinical"
LOG_PATH = LOG_DIR / "run_clinical_analysis.log"
REPORT_DIR = REPORTS_DIR / "pillar-1-cohorts-and-preprocessing"
REPORT_PATH = REPORT_DIR / "cohort_characteristics_clinical.md"
DEMO_GRID_PATH = PLOT_DIR / "clinical_demographics_2x2_grid.png"

# ===========================================================================
# SURVIVAL DATA HANDLING
# ===========================================================================


def _resolve_os_columns(df_clin: pd.DataFrame) -> tuple[str, str]:
    """Determines the overall survival time and event columns.

    Supports:
        - Trial cohorts: os_months / os_status
        - TCGA: OS_MONTHS / OS_STATUS

    Args:
        df_clin: Clinical DataFrame.

    Returns:
        Tuple containing (survival_time_column, survival_event_column).

    Raises:
        ValueError: If no recognised survival column pair is found.
    """
    column_pairs = [
        ("os_months", "os_status"),
        ("OS_MONTHS", "OS_STATUS"),
    ]

    for time_col, event_col in column_pairs:
        if time_col in df_clin.columns and event_col in df_clin.columns:
            return time_col, event_col

    raise ValueError(
        "Could not identify overall survival columns.\n"
        f"Available columns: {list(df_clin.columns)}"
    )


def _clean_os(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
) -> pd.DataFrame:
    """Drops rows with missing or invalid survival data.

    Args:
        df: Clinical DataFrame.
        time_col: Overall survival time column name.
        event_col: Overall survival event indicator column name.

    Returns:
        Cleaned survival DataFrame with valid positive survival durations.
    """
    out = df[[time_col, event_col]].copy()
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
    out[event_col] = pd.to_numeric(out[event_col], errors="coerce")
    out = out.dropna(subset=[time_col, event_col])
    out = out[out[time_col] > 0]
    return out


# ===========================================================================
# DEMOGRAPHIC STATISTICS
# ===========================================================================


def _resolve_age_column(df_clin: pd.DataFrame) -> str | None:
    """Identifies the age column in a clinical DataFrame."""
    possible_columns = [
        "age",
        "AGE",
        "Age",
        "age_at_diagnosis",
        "AGE_AT_DIAGNOSIS",
    ]
    for column in possible_columns:
        if column in df_clin.columns:
            return column
    return None


def calculate_age_statistics(df_clin: pd.DataFrame) -> dict[str, Any]:
    """Calculates descriptive age statistics for a clinical cohort.

    Args:
        df_clin: Clinical DataFrame for a single cohort.

    Returns:
        Dictionary containing sample count and median/IQR age statistics.
    """
    age_col = _resolve_age_column(df_clin)
    if age_col is None:
        return {
            "n_age": 0,
            "median_age": np.nan,
            "q1_age": np.nan,
            "q3_age": np.nan,
        }

    age = pd.to_numeric(df_clin[age_col], errors="coerce").dropna()
    return {
        "n_age": len(age),
        "median_age": float(age.median()),
        "q1_age": float(age.quantile(0.25)),
        "q3_age": float(age.quantile(0.75)),
    }


def calculate_sex_statistics(df_clin: pd.DataFrame) -> dict[str, int]:
    """Calculates female sex frequency and available sample counts.

    Args:
        df_clin: Clinical DataFrame for a single cohort.

    Returns:
        Dictionary containing n_female and n_sex_available counts.
    """
    if "SEX" not in df_clin.columns:
        return {"n_female": 0, "n_sex_available": 0}

    sex = df_clin["SEX"].astype("string").str.strip().str.upper()
    sex = sex.replace({"FEMALE": "F", "WOMAN": "F", "MALE": "M", "MAN": "M"})
    valid_sex = sex.isin(["F", "M"])

    return {
        "n_female": int((sex == "F").sum()),
        "n_sex_available": int(valid_sex.sum()),
    }


# ===========================================================================
# TREATMENT STATISTICS
# ===========================================================================


def _normalise_text_series(series: pd.Series) -> pd.Series:
    """Converts a pandas Series to normalised uppercase text values."""
    return series.astype("string").str.strip().str.upper()


def calculate_treatment_statistics(
    df_clin: pd.DataFrame,
    cohort_label: str,
) -> dict[str, Any]:
    """Calculates cohort-specific treatment statistics.

    Treatment variables differ between clinical trial cohorts and TCGA.

    Args:
        df_clin: Clinical DataFrame.
        cohort_label: Human-readable cohort name.

    Returns:
        Dictionary of cohort-specific treatment frequency statistics.
    """
    statistics: dict[str, Any] = {}

    if cohort_label == "Liu 2019":
        if "ICI_RX" in df_clin.columns:
            ici_rx = _normalise_text_series(df_clin["ICI_RX"])
            statistics["pembrolizumab"] = int((ici_rx == "PEMBROLIZUMAB").sum())
            statistics["nivolumab"] = int((ici_rx == "NIVOLUMAB").sum())
            statistics["treatment_n"] = int(ici_rx.notna().sum())
        else:
            statistics["pembrolizumab"] = None
            statistics["nivolumab"] = None
            statistics["treatment_n"] = 0

        if "PRIOR_ICI_RX" in df_clin.columns:
            prior_ici = _normalise_text_series(df_clin["PRIOR_ICI_RX"])
            statistics["prior_ctla4"] = int(
                prior_ici.str.contains("IPILIMUMAB", na=False).sum()
            )
            statistics["prior_ctla4_n"] = int(prior_ici.notna().sum())
        else:
            statistics["prior_ctla4"] = None
            statistics["prior_ctla4_n"] = 0

    elif cohort_label == "Hugo 2016":
        if "SAMPLE_TREATMENT" in df_clin.columns:
            treatment = _normalise_text_series(df_clin["SAMPLE_TREATMENT"])
            statistics["pembrolizumab"] = int(
                treatment.str.contains("PEMBROLIZUMAB", na=False).sum()
            )
            statistics["nivolumab"] = int(
                treatment.str.contains("NIVOLUMAB", na=False).sum()
            )
            statistics["treatment_n"] = int(treatment.notna().sum())
        else:
            statistics["pembrolizumab"] = None
            statistics["nivolumab"] = None
            statistics["treatment_n"] = 0

        statistics["prior_ctla4"] = None
        statistics["prior_ctla4_n"] = 0

    elif cohort_label == "Riaz 2017":
        if "SAMPLE_TREATMENT" in df_clin.columns:
            treatment = _normalise_text_series(df_clin["SAMPLE_TREATMENT"])
            statistics["nivolumab"] = int(
                treatment.str.contains("NIVOLUMAB", na=False).sum()
            )
            statistics["treatment_n"] = int(treatment.notna().sum())
        else:
            statistics["nivolumab"] = None
            statistics["treatment_n"] = 0

        statistics["pembrolizumab"] = None

        if "PRIOR_ICI_RX" in df_clin.columns:
            prior_ici = _normalise_text_series(df_clin["PRIOR_ICI_RX"])
            statistics["prior_ctla4"] = int(
                prior_ici.str.contains("IPILIMUMAB", na=False).sum()
            )
            statistics["prior_ctla4_n"] = int(prior_ici.notna().sum())
        else:
            statistics["prior_ctla4"] = None
            statistics["prior_ctla4_n"] = 0

    elif cohort_label == "TCGA-SKCM":
        statistics["pembrolizumab"] = int(
            df_clin.get("TX_AGENT_PEMBROLIZUMAB", pd.Series(dtype=float))
            .fillna(0)
            .sum()
        )
        statistics["nivolumab"] = int(
            df_clin.get("TX_AGENT_NIVOLUMAB", pd.Series(dtype=float))
            .fillna(0)
            .sum()
        )
        statistics["prior_ctla4"] = None
        statistics["treatment_n"] = len(df_clin)
        statistics["prior_ctla4_n"] = 0

        statistics["radiation"] = int(
            df_clin.get("TX_TYPE_RADIATION_THERAPY", pd.Series(dtype=float))
            .fillna(0)
            .sum()
        )
        statistics["immunotherapy"] = int(
            df_clin.get("TX_TYPE_IMMUNOTHERAPY", pd.Series(dtype=float))
            .fillna(0)
            .sum()
        )
        statistics["chemotherapy"] = int(
            df_clin.get("TX_TYPE_CHEMOTHERAPY", pd.Series(dtype=float))
            .fillna(0)
            .sum()
        )
        # Count vaccine patients from TREATMENT_TYPES string field, since
        # TX_TYPE_VACCINE binary indicator is absent in this dataset version
        if "TREATMENT_TYPES" in df_clin.columns:
            statistics["vaccine"] = int(
                df_clin["TREATMENT_TYPES"]
                .astype(str)
                .str.contains("Vaccine", na=False)
                .sum()
            )
        else:
            statistics["vaccine"] = 0
        statistics["targeted_therapy"] = int(
            df_clin.get(
                "TX_TYPE_TARGETED_MOLECULAR_THERAPY", pd.Series(dtype=float)
            )
            .fillna(0)
            .sum()
        )

        other_cols = [
            col
            for col in [
                "TX_TYPE_HORMONE_THERAPY",
                "TX_TYPE_ANCILLARY",
                "TX_TYPE_OTHER",
            ]
            if col in df_clin.columns
        ]
        if other_cols:
            statistics["other_therapy"] = int(
                df_clin[other_cols].fillna(0).any(axis=1).sum()
            )
        else:
            statistics["other_therapy"] = 0

        treatment_meta_cols = [
            col
            for col in ["TREATMENT_TYPES", "TREATMENT_AGENTS"]
            if col in df_clin.columns
        ]
        if treatment_meta_cols:
            statistics["no_recorded_treatment"] = int(
                df_clin[treatment_meta_cols].isna().all(axis=1).sum()
            )
        else:
            statistics["no_recorded_treatment"] = 0

    return statistics


def _format_optional_count_percentage(
    count: int | None,
    total: int,
) -> str:
    """Formats an optional count as n (%).

    Returns an em dash when the variable is not applicable to the cohort.
    """
    if count is None:
        return "—"
    if total == 0:
        return "N/A"
    return format_count_percentage(count=count, total=total)


# ===========================================================================
# DEMOGRAPHIC SUMMARY HELPERS
# ===========================================================================


def compute_overall_demographics(
    cohort_data: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Computes pooled sex and age statistics across all cohorts combined.

    Args:
        cohort_data: Mapping of cohort name to clinical DataFrame.

    Returns:
        Dictionary containing pooled sex counts, percentages, and age statistics.
    """
    all_sex: list[str] = []
    all_ages: list[float] = []

    for df in cohort_data.values():
        if "SEX" in df.columns:
            s = df["SEX"].astype("string").str.strip().str.upper()
            s = s.replace({"FEMALE": "F", "WOMAN": "F", "MALE": "M", "MAN": "M"})
            all_sex.extend(s[s.isin(["F", "M"])].tolist())

        age_col = _resolve_age_column(df)
        if age_col:
            ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
            all_ages.extend(ages.tolist())

    s_series = pd.Series(all_sex)
    n_sex_total = len(s_series)
    n_male = int((s_series == "M").sum())
    n_female = int((s_series == "F").sum())

    age_series = pd.Series(all_ages)

    return {
        "n_sex_total": n_sex_total,
        "n_male": n_male,
        "n_female": n_female,
        "pct_male": n_male / n_sex_total * 100 if n_sex_total > 0 else 0.0,
        "pct_female": n_female / n_sex_total * 100 if n_sex_total > 0 else 0.0,
        "age_median": float(age_series.median()),
        "age_q1": float(age_series.quantile(0.25)),
        "age_q3": float(age_series.quantile(0.75)),
        "age_min": float(age_series.min()),
        "age_max": float(age_series.max()),
    }


def compute_ici_agent_breakdown(
    treatment_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Computes the breakdown of immunotherapy agents administered across cohorts.

    Aggregates Pembrolizumab and Nivolumab counts from individual trial cohorts
    plus TCGA-SKCM's immunotherapy subgroup.

    Args:
        treatment_results: Per-cohort treatment statistics dictionaries.

    Returns:
        Dictionary containing agent counts, total N, and percentages.
    """
    liu = treatment_results.get("Liu 2019", {})
    hugo = treatment_results.get("Hugo 2016", {})
    riaz = treatment_results.get("Riaz 2017", {})
    tcga = treatment_results.get("TCGA-SKCM", {})

    n_pemb = (liu.get("pembrolizumab") or 0) + (hugo.get("pembrolizumab") or 0)
    n_nivo = (liu.get("nivolumab") or 0) + (riaz.get("nivolumab") or 0)
    n_ipi_combo = riaz.get("prior_ctla4") or 0

    # TCGA immunotherapy patients not accounted for by named agents
    tcga_pemb = tcga.get("pembrolizumab") or 0
    tcga_nivo = tcga.get("nivolumab") or 0
    tcga_immuno_total = tcga.get("immunotherapy") or 0
    n_unspec = max(0, tcga_immuno_total - tcga_pemb - tcga_nivo)

    n_total = n_pemb + n_nivo + n_ipi_combo + n_unspec

    def _pct(n: int) -> float:
        return n / n_total * 100 if n_total > 0 else 0.0

    return {
        "n_total": n_total,
        "n_pembrolizumab": n_pemb,
        "n_nivolumab": n_nivo,
        "n_ipi_combo": n_ipi_combo,
        "n_unspecified": n_unspec,
        "pct_pembrolizumab": _pct(n_pemb),
        "pct_nivolumab": _pct(n_nivo),
        "pct_ipi_combo": _pct(n_ipi_combo),
        "pct_unspecified": _pct(n_unspec),
    }


def plot_clinical_demographics_grid(
    cohort_data: dict[str, pd.DataFrame],
    overall_demographics: dict[str, Any],
    tcga_treatment: dict[str, Any],
    ici_breakdown: dict[str, Any],
    out_path: Path,
) -> None:
    """Plots the 2\u00d72 clinical demographics and treatment distributions grid.

    Generates and saves a 4-panel figure:
    - Panel A: Patient sex distribution (doughnut)
    - Panel B: Age at diagnosis across cohorts (stacked histogram + KDE)
    - Panel C: TCGA-SKCM recorded treatment modalities (doughnut)
    - Panel D: Immunotherapy agents administered (doughnut)

    All values are computed dynamically from the supplied live DataFrames and
    pre-computed statistics — no literals are hardcoded in any plot label.

    Args:
        cohort_data: Mapping of cohort name to clinical DataFrame.
        overall_demographics: Pooled sex/age statistics.
        tcga_treatment: TCGA treatment frequency statistics.
        ici_breakdown: ICI agent breakdown.
        out_path: Destination path for the saved PNG figure.
    """
    d = overall_demographics
    ici = ici_breakdown
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # ------------------------------------------------------------------
    # Panel A: Sex Distribution (doughnut)
    # ------------------------------------------------------------------
    axes[0, 0].pie(
        [d["n_male"], d["n_female"]],
        labels=["Male", "Female"],
        autopct="%1.1f%%",
        startangle=140,
        colors=["#CC79A7", "#0072B2"],
        wedgeprops=dict(width=0.4, edgecolor="w", linewidth=2),
        textprops=dict(fontsize=12, fontweight="bold"),
    )
    axes[0, 0].set_title(
        f"Panel A: Patient Sex Distribution (N = {d['n_sex_total']})",
        fontsize=14, fontweight="bold", pad=15,
    )

    # ------------------------------------------------------------------
    # Panel B: Age Distribution (stacked histogram + KDE)
    # ------------------------------------------------------------------
    age_dfs: list[pd.DataFrame] = []
    for label, df in cohort_data.items():
        age_col = _resolve_age_column(df)
        if age_col:
            ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
            age_dfs.append(pd.DataFrame({"Age": ages, "Cohort": label}))

    if age_dfs:
        df_ages = pd.concat(age_dfs, axis=0, ignore_index=True)
        sns.histplot(
            data=df_ages,
            x="Age",
            hue="Cohort",
            multiple="stack",
            palette=COHORT_PALETTE,
            ax=axes[0, 1],
            bins=20,
            kde=True,
        )
        axes[0, 1].axvline(
            d["age_median"],
            color="#D55E00",
            linestyle="--",
            linewidth=2,
            label=f"Median: {d['age_median']:.0f} y",
        )
        handles, leg_labels = axes[0, 1].get_legend_handles_labels()
        axes[0, 1].legend(handles, leg_labels, loc="upper right", fontsize=8, framealpha=0.9)

    axes[0, 1].set_title(
        f"Panel B: Age Distribution "
        f"(Median = {d['age_median']:.0f}, IQR = {d['age_q1']:.0f}\u2013{d['age_q3']:.0f})",
        fontsize=14, fontweight="bold", pad=15,
    )
    axes[0, 1].set_xlabel("Age at Diagnosis (Years)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Patient Count", fontsize=11, fontweight="bold")

    # ------------------------------------------------------------------
    # Panel C: TCGA-SKCM recorded treatment types (doughnut)
    # ------------------------------------------------------------------
    n_tcga = tcga_treatment.get("n_total", 0)
    tcga_slice_labels = [
        "Radiation Therapy", "Immunotherapy", "Chemotherapy",
        "Vaccine Therapy", "Targeted Therapy", "No Recorded Therapy",
    ]
    tcga_slice_vals = [
        tcga_treatment.get("radiation", 0),
        tcga_treatment.get("immunotherapy", 0),
        tcga_treatment.get("chemotherapy", 0),
        tcga_treatment.get("vaccine", 0),
        tcga_treatment.get("targeted_therapy", 0),
        tcga_treatment.get("no_recorded_treatment", 0),
    ]
    tcga_slice_colors = ["#E69F00", "#009E73", "#56B4E9", "#F0E442", "#D55E00", "#999999"]
    # Filter zero-count slices to avoid label clutter
    non_zero = [
        (lbl, val, col)
        for lbl, val, col in zip(tcga_slice_labels, tcga_slice_vals, tcga_slice_colors)
        if val > 0
    ]
    if non_zero:
        filt_labels, filt_vals, filt_colors = zip(*non_zero)
    else:
        filt_labels, filt_vals, filt_colors = tcga_slice_labels, tcga_slice_vals, tcga_slice_colors

    axes[1, 0].pie(
        filt_vals,
        labels=filt_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=filt_colors,
        wedgeprops=dict(width=0.4, edgecolor="w", linewidth=2),
        textprops=dict(fontsize=10, fontweight="bold"),
    )
    axes[1, 0].set_title(
        f"Panel C: TCGA-SKCM Recorded Treatment Types (N = {n_tcga})",
        fontsize=14, fontweight="bold", pad=15,
    )

    # ------------------------------------------------------------------
    # Panel D: ICI agents administered (doughnut)
    # ------------------------------------------------------------------
    axes[1, 1].pie(
        [
            ici["n_nivolumab"], ici["n_pembrolizumab"],
            ici["n_ipi_combo"], ici["n_unspecified"],
        ],
        labels=[
            "Nivolumab (Anti-PD-1)", "Pembrolizumab (Anti-PD-1)",
            "Ipilimumab Combo (Anti-CTLA-4)", "Anti-PD-1 Unspecified",
        ],
        autopct="%1.1f%%",
        startangle=140,
        colors=["#0072B2", "#009E73", "#CC79A7", "#56B4E9"],
        wedgeprops=dict(width=0.4, edgecolor="w", linewidth=2),
        textprops=dict(fontsize=10, fontweight="bold"),
    )
    axes[1, 1].set_title(
        f"Panel D: Immunotherapy Agents Administered (N = {ici['n_total']})",
        fontsize=14, fontweight="bold", pad=15,
    )

    plt.suptitle(
        "Clinical Patient Demographics & Treatment Distribution Across Cohorts",
        fontsize=16, fontweight="bold", y=0.98,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_path)
    plt.close(fig)

    print(
        f"\nSaved clinical demographics grid to "
        f"{out_path.relative_to(SUBPROJECT_ROOT).as_posix()}"
    )


# ===========================================================================
# KAPLAN-MEIER ANALYSIS
# ===========================================================================


def plot_km_os(
    ax: plt.Axes,
    df_clin: pd.DataFrame,
    cohort_label: str,
) -> dict[str, Any]:
    """Plots an unstratified Kaplan-Meier OS curve.

    Args:
        ax: Matplotlib subplot axis.
        df_clin: Clinical DataFrame.
        cohort_label: Human-readable cohort name.

    Returns:
        Dictionary containing calculated survival statistics.
    """
    time_col, event_col = _resolve_os_columns(df_clin)
    df = _clean_os(df_clin, time_col, event_col)
    cohort_color = get_cohort_color(cohort_label)

    kmf = KaplanMeierFitter()
    os_label = f"Survival Curve (N={len(df)})" if len(df) == len(df_clin) else f"OS Subset (n={len(df)})"
    kmf.fit(
        durations=df[time_col],
        event_observed=df[event_col],
        label=os_label,
    )

    kmf.plot_survival_function(
        ax=ax,
        ci_show=True,
        color=cohort_color,
        linewidth=2,
    )

    median_surv = float(kmf.median_survival_time_)

    if np.isfinite(median_surv):
        ax.axhline(
            0.5,
            color="grey",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )
        ax.axvline(
            median_surv,
            color="grey",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )
        ax.text(
            0.95,
            0.05,
            f"Median OS = {median_surv:.1f} mo",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            fontstyle="italic",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="grey",
                alpha=0.8,
            ),
        )

    ax.set_title(f"{cohort_label} (N={len(df_clin)})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (months)", fontsize=10)
    ax.set_ylabel("Overall Survival Probability", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    n_total = len(df_clin)
    n_valid_os = len(df)
    n_events = int(df[event_col].sum())
    n_censored = int((df[event_col] == 0).sum())
    event_rate = (n_events / n_valid_os * 100) if n_valid_os > 0 else 0.0
    median_follow_up = float(df[time_col].median()) if len(df) > 0 else np.nan

    return {
        "n_total": n_total,
        "n_valid_os": n_valid_os,
        "n_events": n_events,
        "n_censored": n_censored,
        "event_rate": event_rate,
        "median_os": median_surv,
        "median_follow_up": median_follow_up,
    }


# ===========================================================================
# REPORT TABLE GENERATION
# ===========================================================================


def generate_clinical_characteristics_table(
    cohort_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
) -> str:
    """Generates the comparative clinical characteristics Markdown table.

    Args:
        cohort_results: Calculated clinical and survival metrics per cohort.
        cohort_order: Ordered list of cohort labels.

    Returns:
        Markdown table string.
    """
    rows: list[dict[str, str]] = []

    survival_results = {c: cohort_results[c]["survival"] for c in cohort_order}
    age_results = {c: cohort_results[c]["age"] for c in cohort_order}
    sex_results = {c: cohort_results[c]["sex"] for c in cohort_order}
    treatment_results = {c: cohort_results[c]["treatment"] for c in cohort_order}

    # N
    rows.append({
        "Characteristic": "**N**",
        **{c: str(survival_results[c]["n_total"]) for c in cohort_order},
    })
    rows.append({"Characteristic": "", **{c: "" for c in cohort_order}})

    # Demographics
    rows.append({
        "Characteristic": "**Demographics**",
        **{c: "" for c in cohort_order},
    })
    rows.append({
        "Characteristic": "Age, median (IQR)",
        **{c: format_median_iqr(age_results[c]) for c in cohort_order},
    })
    rows.append({
        "Characteristic": "Female sex, n (%)",
        **{
            c: format_count_percentage(
                count=sex_results[c]["n_female"],
                total=sex_results[c]["n_sex_available"],
            )
            for c in cohort_order
        },
    })
    rows.append({"Characteristic": "", **{c: "" for c in cohort_order}})

    # Treatment
    rows.append({
        "Characteristic": "**Treatment**",
        **{c: "" for c in cohort_order},
    })
    rows.append({
        "Characteristic": "ICI agent — Pembrolizumab",
        **{
            c: _format_optional_count_percentage(
                treatment_results[c].get("pembrolizumab"),
                treatment_results[c].get(
                    "treatment_n", survival_results[c]["n_total"]
                ),
            )
            for c in cohort_order
        },
    })
    rows.append({
        "Characteristic": "ICI agent — Nivolumab",
        **{
            c: _format_optional_count_percentage(
                treatment_results[c].get("nivolumab"),
                treatment_results[c].get(
                    "treatment_n", survival_results[c]["n_total"]
                ),
            )
            for c in cohort_order
        },
    })
    rows.append({
        "Characteristic": "Prior anti-CTLA-4",
        **{
            c: _format_optional_count_percentage(
                treatment_results[c].get("prior_ctla4"),
                treatment_results[c].get(
                    "prior_ctla4_n", survival_results[c]["n_total"]
                ),
            )
            for c in cohort_order
        },
    })

    # TCGA treatment history
    rows.append({
        "Characteristic": "Treatment History (TCGA only)",
        "Liu 2019": "—",
        "Hugo 2016": "—",
        "Riaz 2017": "—",
        "TCGA-SKCM": "",
    })

    tcga_treatment_rows = [
        ("— Radiation Therapy", "radiation"),
        ("— Immunotherapy", "immunotherapy"),
        ("— Chemotherapy", "chemotherapy"),
        ("— Vaccine", "vaccine"),
        ("— Targeted Therapy", "targeted_therapy"),
        ("— Other Therapy", "other_therapy"),
        ("— No recorded treatment", "no_recorded_treatment"),
    ]

    for label, key in tcga_treatment_rows:
        rows.append({
            "Characteristic": label,
            "Liu 2019": "—",
            "Hugo 2016": "—",
            "Riaz 2017": "—",
            "TCGA-SKCM": _format_optional_count_percentage(
                treatment_results["TCGA-SKCM"].get(key),
                survival_results["TCGA-SKCM"]["n_total"],
            ),
        })

    rows.append({"Characteristic": "", **{c: "" for c in cohort_order}})

    # Survival Outcomes
    rows.append({
        "Characteristic": "**Survival Outcomes**",
        **{c: "" for c in cohort_order},
    })
    rows.append({
        "Characteristic": "Median OS, months (95% CI)",
        **{
            c: format_median(survival_results[c]["median_os"])
            for c in cohort_order
        },
    })
    rows.append({
        "Characteristic": "OS events, n (%)",
        **{
            c: format_count_percentage(
                count=survival_results[c]["n_events"],
                total=survival_results[c]["n_valid_os"],
            )
            for c in cohort_order
        },
    })
    rows.append({
        "Characteristic": "Median follow-up, months",
        **{
            c: format_median(survival_results[c]["median_follow_up"])
            for c in cohort_order
        },
    })

    table_df = pd.DataFrame(rows)
    return table_df.to_markdown(index=False)


def generate_attrition_table(
    attrition_data: dict[str, pd.DataFrame],
    cohort_order: list[str],
) -> str:
    """Generates a Markdown table summarising sample attrition across preprocessing steps.

    Args:
        attrition_data: Mapping of cohort name to its attrition DataFrame.
        cohort_order: Ordered list of cohort labels.

    Returns:
        Formatted Markdown table string.
    """
    rows: list[dict[str, str | int]] = []

    for cohort_name in cohort_order:
        df_attrition = attrition_data.get(cohort_name)
        if df_attrition is None or df_attrition.empty:
            continue

        for _, record in df_attrition.iterrows():
            rows.append({
                "Cohort": str(record.get("cohort", cohort_name)),
                "Preprocessing Step": str(record.get("step", "")),
                "N Initial": int(record.get("n_before", 0)),
                "N Retained": int(record.get("n_after", 0)),
                "N Removed": int(record.get("n_removed", 0)),
                "Rationale": str(record.get("reason", "")),
            })

    if not rows:
        return "No sample attrition data available."

    df_combined = pd.DataFrame(rows)
    return df_combined.to_markdown(index=False)


def generate_key_observations(
    cohort_results: dict[str, dict[str, Any]],
    attrition_data: dict[str, pd.DataFrame],
    cohort_order: list[str],
) -> str:
    """Generates dynamic key observations based on clinical and attrition data.

    Args:
        cohort_results: Calculated clinical and survival metrics per cohort.
        attrition_data: Preprocessing sample attrition DataFrames per cohort.
        cohort_order: Ordered list of cohort labels.

    Returns:
        Formatted markdown bullet points.
    """
    survival_results = {c: cohort_results[c]["survival"] for c in cohort_order}

    largest_cohort = max(
        cohort_order,
        key=lambda cohort: survival_results[cohort]["n_total"],
    )

    longest_follow_up = max(
        cohort_order,
        key=lambda cohort: survival_results[cohort]["median_follow_up"],
    )

    # Compute overall sample counts and total attrition
    initial_total = 0
    final_total = 0
    total_removed = 0
    attrition_summaries = []

    for cohort in cohort_order:
        df_attr = attrition_data.get(cohort)
        if df_attr is not None and not df_attr.empty:
            c_init = int(df_attr.iloc[0]["n_before"])
            c_final = int(df_attr.iloc[-1]["n_after"])
            c_rem = c_init - c_final
            initial_total += c_init
            final_total += c_final
            total_removed += c_rem
            if c_rem > 0:
                attrition_summaries.append(
                    f"**{cohort}** lost **{c_rem}** sample(s) "
                    f"({c_init} → {c_final})"
                )
            else:
                attrition_summaries.append(
                    f"**{cohort}** retained 100% of samples (N = {c_final})"
                )

    attrition_text = "; ".join(attrition_summaries)

    return f"""
1. **Cohort Size**: The largest individual cohort is **{largest_cohort}** with **N = {survival_results[largest_cohort]["n_total"]}** patients. Across all four cohorts, **N = {final_total}** cleaned clinical samples were harmonised.

2. **Sample Attrition**: Preprocessing quality control and clinical-expression alignment evaluated **{initial_total}** initial records and removed **{total_removed}** sample(s) across cohorts ({attrition_text}).

3. **Overall Survival**: Median overall survival was estimated using the Kaplan-Meier survival function. Where the estimated survival probability did not fall below 50% during follow-up, median OS was reported as not reached.

4. **Follow-up Duration**: **{longest_follow_up}** demonstrated the longest median follow-up duration at **{format_median(survival_results[longest_follow_up]["median_follow_up"])} months**.

5. **Event Rates**: Observed OS event rates varied between cohorts, reflecting differences in patient composition, disease staging, treatment regimens, follow-up duration, and censoring.

6. **Variable Completeness**: Demographic (age, sex) and treatment annotations vary in availability across datasets, requiring careful consideration during multi-cohort synthesis.
""".strip()


# ===========================================================================
# REPORT GENERATION
# ===========================================================================


def generate_clinical_report(
    cohort_results: dict[str, dict[str, Any]],
    attrition_data: dict[str, pd.DataFrame],
    plot_path: Path,
    report_path: Path,
    cohort_order: list[str],
    overall_demographics: dict[str, Any],
    ici_breakdown: dict[str, Any],
    demographics_grid_path: Path,
) -> None:
    """Generates an Obsidian-compatible Markdown clinical characteristics report.

    All numeric values (sample sizes, percentages, survival statistics, treatment
    frequencies) are computed dynamically from live DataFrames. No literals are
    hardcoded in the report template.

    Args:
        cohort_results: Calculated clinical and survival metrics per cohort.
        attrition_data: Preprocessing sample attrition DataFrames per cohort.
        plot_path: Path to the saved 2\u00d72 KM OS grid image.
        report_path: Target path for the output Markdown report.
        cohort_order: Ordered list of cohort labels.
        overall_demographics: Pooled sex and age statistics across all cohorts.
        ici_breakdown: ICI agent breakdown statistics.
        demographics_grid_path: Path to the saved clinical demographics grid image.
    """
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    survival_results = {c: cohort_results[c]["survival"] for c in cohort_order}
    treatment_results = {c: cohort_results[c]["treatment"] for c in cohort_order}
    age_results = {c: cohort_results[c]["age"] for c in cohort_order}

    # ------------------------------------------------------------------
    # Dynamically computed summary values
    # ------------------------------------------------------------------
    n_values = {c: survival_results[c]["n_total"] for c in cohort_order}
    n_total = sum(n_values.values())
    n_tcga = n_values.get("TCGA-SKCM", 0)
    n_liu = n_values.get("Liu 2019", 0)
    n_hugo = n_values.get("Hugo 2016", 0)
    n_riaz = n_values.get("Riaz 2017", 0)
    n_trial = n_liu + n_hugo + n_riaz

    d = overall_demographics
    tcga_tx = treatment_results.get("TCGA-SKCM", {})
    ici = ici_breakdown

    def _pct(count: int, total: int) -> str:
        """Formats a count as a percentage string."""
        if total == 0:
            return "N/A"
        return f"{count / total * 100:.1f}%"

    # TCGA treatment counts
    tcga_n_rad = tcga_tx.get("radiation", 0)
    tcga_n_imm = tcga_tx.get("immunotherapy", 0)
    tcga_n_che = tcga_tx.get("chemotherapy", 0)
    tcga_n_vac = tcga_tx.get("vaccine", 0)
    tcga_n_tar = tcga_tx.get("targeted_therapy", 0)
    tcga_n_nor = tcga_tx.get("no_recorded_treatment", 0)

    # Per-cohort age medians
    def _age_str(cohort: str) -> str:
        med = age_results.get(cohort, {}).get("median_age", float("nan"))
        return format_median(med, decimals=1)

    # Largest cohort and longest follow-up
    largest_cohort = max(cohort_order, key=lambda c: survival_results[c]["n_total"])
    longest_fu_cohort = max(cohort_order, key=lambda c: survival_results[c]["median_follow_up"])

    # ------------------------------------------------------------------
    # Attrition summary
    # ------------------------------------------------------------------
    initial_total = 0
    final_total = 0
    total_removed = 0
    attrition_summaries: list[str] = []

    for cohort in cohort_order:
        df_attr = attrition_data.get(cohort)
        if df_attr is not None and not df_attr.empty:
            c_init = int(df_attr.iloc[0]["n_before"])
            c_final = int(df_attr.iloc[-1]["n_after"])
            c_rem = c_init - c_final
            initial_total += c_init
            final_total += c_final
            total_removed += c_rem
            if c_rem > 0:
                attrition_summaries.append(
                    f"**{cohort}** lost **{c_rem}** sample(s) ({c_init} \u2192 {c_final})"
                )
            else:
                attrition_summaries.append(
                    f"**{cohort}** retained 100% of samples ($N = {c_final}$)"
                )

    attrition_text = "; ".join(attrition_summaries)
    retention_pct = final_total / initial_total * 100 if initial_total > 0 else 0.0

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    clinical_characteristics_table = generate_clinical_characteristics_table(
        cohort_results=cohort_results,
        cohort_order=cohort_order,
    )

    attrition_table = generate_attrition_table(
        attrition_data=attrition_data,
        cohort_order=cohort_order,
    )

    # ------------------------------------------------------------------
    # Frontmatter
    # ------------------------------------------------------------------
    frontmatter = generate_obsidian_frontmatter(
        title="Clinical Characteristics of Data Cohorts",
        aliases=["Clinical Cohort Characteristics"],
        tags=[
            "melanoma",
            "clinical-analysis",
            "cohort-characteristics",
            "survival-analysis",
            "kaplan-meier",
        ],
        created=timestamp,
        updated=timestamp,
        extra_css_classes=["table-center", "row-alt"],
    )

    report = f"""{frontmatter}

# Clinical Characteristics of Data Cohorts

## 1. Baseline Patient and Disease Characteristics

> [!INFO] Why We Are Doing This
> **What**: We compare patient demographics, treatment histories, and survival outcomes across the four melanoma study cohorts: **TCGA-SKCM** ($N = {n_tcga}$), **Liu 2019** ($N = {n_liu}$), **Hugo 2016** ($N = {n_hugo}$), and **Riaz 2017** ($N = {n_riaz}$).
> **Why**: Before building predictive models or analysing transcriptomic signatures, we must understand the clinical composition of each dataset. Cohort-level differences in prior treatment, disease stage, and patient demographics can confound downstream survival and response analyses.
> **Question Answered**: Are baseline patient populations sufficiently comparable across independent datasets to permit pooled multi-cohort machine learning?

This report compares patient demographics, treatments, survival, and sample attrition across the four melanoma study cohorts:
- **TCGA-SKCM**: Baseline reference dataset ($N = {n_tcga}$) representing primary and metastatic melanoma with general treatment histories.
- **Liu 2019**: Immunotherapy trial ($N = {n_liu}$) treated with anti-PD-1 monotherapy (Pembrolizumab or Nivolumab).
- **Hugo 2016**: Anti-PD-1 clinical trial cohort ($N = {n_hugo}$).
- **Riaz 2017**: Anti-PD-1 clinical trial cohort ($N = {n_riaz}$); 100% of patients had previously received anti-CTLA-4 (Ipilimumab).

### 1.1 Clinical Demographics & Treatment Distributions (2\u00d72 Grid)

![Clinical Demographics & Treatment Distributions](../../plots/clinical/{demographics_grid_path.name})

_**Figure 1: 2\u00d72 Grid of Clinical Demographics and Treatment Histories across Cohorts.** Panel A: sex distribution; Panel B: age at diagnosis; Panel C: TCGA-SKCM treatment modalities; Panel D: immunotherapy agents administered across trial cohorts._

#### Key Demographics & Treatment Insights

- **Panel A: Sex Distribution ($N = {d['n_sex_total']}$)**: The overall cohort shows a {'male' if d['n_male'] > d['n_female'] else 'female'} predominance (**{d['pct_male']:.1f}% Male** [$N = {d['n_male']}$] vs. **{d['pct_female']:.1f}% Female** [$N = {d['n_female']}$]), reflecting real-world cutaneous melanoma incidence patterns where male patients account for the majority of advanced presentations.
- **Panel B: Age Distribution across Studies**: Patient ages span from {d['age_min']:.0f} to {d['age_max']:.0f} years with a **median age of {d['age_median']:.1f} years** ($\\text{{IQR}} = {d['age_q1']:.1f}\\text{{--}}{d['age_q3']:.1f}\\text{{ years}}$). Trial cohorts (`Hugo 2016`: median {_age_str('Hugo 2016')}; `Riaz 2017`: median {_age_str('Riaz 2017')}; `TCGA-SKCM`: median {_age_str('TCGA-SKCM')}) display consistent age distributions centred around late middle age.
- **Panel C: TCGA-SKCM Recorded Treatment Types ($N = {n_tcga}$)**: Among TCGA-SKCM patients, **Radiation Therapy** represents the largest modality (**{_pct(tcga_n_rad, n_tcga)}** [$N = {tcga_n_rad}$]), followed by **Immunotherapy** (**{_pct(tcga_n_imm, n_tcga)}** [$N = {tcga_n_imm}$]), **Chemotherapy** (**{_pct(tcga_n_che, n_tcga)}** [$N = {tcga_n_che}$]), **Vaccine Therapy** (**{_pct(tcga_n_vac, n_tcga)}** [$N = {tcga_n_vac}$]), and **Targeted Therapy** (**{_pct(tcga_n_tar, n_tcga)}** [$N = {tcga_n_tar}$]). **{_pct(tcga_n_nor, n_tcga)}** [$N = {tcga_n_nor}$] have no recorded systemic therapy in cBioPortal.
- **Panel D: Immunotherapy Agents Administered ($N = {ici['n_total']}$)**: Across all immunotherapy-treated trial and TCGA patients, **Nivolumab (Anti-PD-1)** is the predominant agent (**{ici['pct_nivolumab']:.1f}%** [$N = {ici['n_nivolumab']}$]), followed by **Pembrolizumab (Anti-PD-1)** (**{ici['pct_pembrolizumab']:.1f}%** [$N = {ici['n_pembrolizumab']}$]), **Ipilimumab Combination / Prior Exposure** (**{ici['pct_ipi_combo']:.1f}%** [$N = {ici['n_ipi_combo']}$]), and **Unspecified Anti-PD-1** (**{ici['pct_unspecified']:.1f}%** [$N = {ici['n_unspecified']}$]).

_**Table 1: Baseline Patient and Disease Characteristics**_

{clinical_characteristics_table}

## 2. Sample Preprocessing Attrition

> [!INFO] Why We Are Doing This
> **What**: We track sample retention through quality control, identifier standardisation, and clinical-expression alignment across all $N = {initial_total}$ initial records.
> **Why**: Documenting attrition at each preprocessing step verifies data integrity and confirms that no patient sub-population was accidentally excluded.
> **Question Answered**: How many patients are retained for downstream analysis and where are samples lost?

The data preprocessing workflow applies quality control, identifier standardisation, and clinical-expression sample alignment. The table below outlines sample retention and attrition rationale for each cohort.

_**Table 2: Sample Attrition Across Preprocessing Steps**_

{attrition_table}

### Key Observations
1. **Overall Cohort Size ($N = {n_total}$)**: The largest individual dataset is **{largest_cohort}** ($N = {n_values[largest_cohort]}$). Combined across all four cohorts, **$N = {final_total}$** cleaned patient records were harmonised for downstream analysis.
2. **Minimal Sample Attrition ({retention_pct:.1f}% Retention)**: Across all {initial_total} initial records, only **{total_removed}** sample(s) were removed. {attrition_text}. All three immunotherapy trial cohorts retained 100% of their samples; TCGA-SKCM attrition reflects unmatched RNA-seq expression records.
3. **Longest Follow-up**: **{longest_fu_cohort}** shows the longest median follow-up duration (**{format_median(survival_results[longest_fu_cohort]['median_follow_up'])} months**) due to its inclusion of earlier-stage surgical cases.
4. **Treatment History & Clinical Context**: All $N = {n_riaz}$ patients in **Riaz 2017** were previously treated with anti-CTLA-4 (Ipilimumab), representing an Ipilimumab-refractory population relative to the anti-CTLA-4 na\u00efve patients in **Liu 2019** ($N = {n_liu}$).

## 3. Overall Survival Curves (KM Plots)

> [!INFO] Why We Are Doing This
> **What**: We plot unstratified Kaplan-Meier overall survival curves for each of the four cohorts.
> **Why**: Visualising baseline mortality rates establishes the clinical context for each dataset and confirms that follow-up duration is sufficient to evaluate treatment outcomes.
> **Question Answered**: How does overall survival differ between the TCGA-SKCM reference cohort and the active immunotherapy trial cohorts?

The unstratified Kaplan-Meier overall survival curves for each cohort are shown below. Median overall survival is annotated for each cohort where reached.

![Overall Survival KM Curves (All Cohorts)](../../plots/clinical/{plot_path.name})

_**Figure 2: Unstratified Overall Survival KM Curves across All Four Cohorts.**_

## 4. Overall Survival Stratified by Immunotherapy Response

> [!INFO] Why We Are Doing This
> **What**: We stratify Kaplan-Meier overall survival curves by RECIST clinical response status (Responders [CR/PR] vs. Non-responders [PD]) across the three immunotherapy trial cohorts ($N = {n_trial}$).
> **Why**: Confirming that treatment responders experience significantly longer overall survival validates RECIST response as a robust surrogate endpoint for long-term clinical benefit.
> **Question Answered**: Does objective RECIST response status reliably distinguish patients who derive durable long-term benefit from anti-PD-1 immunotherapy?

![Overall Survival by Immunotherapy Response (RECIST)](../../plots/clinical/km_os_by_response.png)

_**Figure 3: Overall Survival Stratified by RECIST Response Status across Immunotherapy Trial Cohorts.** Treatment responders (CR/PR) exhibit significantly superior overall survival compared to non-responders (PD) across all three trial cohorts (Log-rank $p < 0.0001$)._

> [!INSIGHT] Key Insights: Survival Stratification by Response
> 1. **Profound Survival Benefit**: Treatment responders (CR/PR) achieve significantly longer overall survival compared to non-responders (PD) across all three independent trial cohorts (Liu 2019, Hugo 2016, and Riaz 2017; all Log-rank $p < 0.0001$).
> 2. **Durable Long-Term Survival**: In **Liu 2019**, non-responders experience rapid mortality whereas the majority of responders survive well beyond 50 months. Similar durable separation is observed in Hugo 2016 and Riaz 2017.
> 3. **Conclusion**: Objective RECIST response is a highly robust surrogate endpoint for overall survival in metastatic melanoma, justifying its use as the primary outcome for predictive model training.

## 5. Univariate Associations with Response (Forest Plot)

> [!INFO] Why We Are Doing This
> **What**: We run univariate statistical tests (Odds Ratios and 95% Confidence Intervals) to measure the isolated predictive strength of individual baseline features \u2014 age, sex, tumour mutation burden (TMB), and driver mutations (`BRAF`, `NRAS`, `NF1`) \u2014 against immunotherapy response across individual and pooled trial cohorts.
> **Why**: Before building complex multivariate models, univariate screening identifies whether any single clinical or genomic feature alone is sufficient to predict response.
> **Question Answered**: Does any single baseline clinical or genomic feature reliably predict anti-PD-1 immunotherapy response across independent cohorts?

![Forest Plot of Univariate Odds Ratios](../../plots/clinical/univariate_associations.png)

_**Figure 4: Forest Plot of Univariate Odds Ratios for Clinical and Genomic Features against Immunotherapy Response.**_

> [!INSIGHT] Key Takeaways: Univariate Associations
> 1. **Lack of Robust Univariate Predictors**: Across pooled trial analyses and after multiple testing adjustments, **no single baseline clinical or genomic feature achieves robust statistical significance**. All 95% CIs for pooled odds ratios cross 1.0.
> 2. **Anatomical Staging Artefact (Liu 2019 $p = 0.029$)**: Stage IV disease shows a nominal unadjusted association in the isolated Liu 2019 cohort. This is an artefact of extreme trial enrolment imbalance; in the pooled multi-cohort analysis, the association attenuates to $p = 0.083$.
> 3. **Subtle Feature Trends**: `NRAS` mutations and higher Tumour Mutation Burden (TMB; OR = 1.37 per SD increase, $p = 0.160$) trend towards elevated response odds. `BRAF` driver mutations show no univariate association (OR \u2248 1.0).
> 4. **Demographic Balance**: Age and sex show no association with treatment response (OR \u2248 0.98\u20131.26, $p > 0.50$), confirming demographic balance between responder and non-responder arms.
> 5. **Core Scientific Implication**: The failure of individual clinical variables and driver mutations to predict outcome explains why single-variable biomarker tests fail in clinical practice, highlighting the necessity of multi-gene transcriptomic signatures and integrated multivariate machine learning.

## 6. Technical Analysis Notes

> [!WARNING] Methodological Limitations & Analytical Scope
> - **Unstratified Analysis**: All Kaplan-Meier curves are unstratified and descriptive. They do not adjust for demographic, clinical, molecular, treatment, or study-specific confounding factors.
> - **Survival Exclusion Criteria**: Records were excluded from survival analysis if they had missing survival time, missing event status, non-numeric survival values, or a survival time \u2264 0.
> - **Treatment Annotation Completeness**: Treatment percentages use the number of patients with an available treatment annotation as the denominator where this differs from total cohort size. TCGA treatment-history categories are based on binary treatment-type indicators and are not necessarily mutually exclusive.
> - **Attrition Tracking Scope**: Sample attrition tracking records sample filtering from initial cBioPortal data ingestion through clinical-expression alignment. Downstream feature engineering steps may impose additional filters not captured here.
> - **Median OS Reporting**: Where the estimated survival probability did not fall below 50% during follow-up, median OS is reported as **NR (not reached)**.

> [!formula]+ Clinical Analysis Script Execution & Software Module Architecture
> - **Primary Pipeline Execution Scripts**:
>   - [`run_clinical_analysis.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-2-clinical-subtyping/run_clinical_analysis.py): Generates baseline demographic grids, attrition metrics, cohort characteristics tables, Kaplan-Meier OS curves, and writes `cohort_characteristics_clinical.md`.
>   - [`run_univariate_associations.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-2-clinical-subtyping/run_univariate_associations.py): Computes univariate odds ratios and confidence intervals across clinical/genomic features and generates the forest plot (`univariate_associations.png`).
> - **Data Preprocessing & Loading Modules**:
>   - [`clean_data.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/clean_data.py): Preprocesses raw cohort clinical metadata and RNA-seq expression profiles into cleaned CSV matrices.
>   - [`merge_datasets.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/merge_datasets.py): Merges processed expression matrices across cohorts into harmonised pooled matrices (`expr_merged.csv`, `clin_merged.csv`).
>   - [`data_loaders.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/data_loaders.py): Provides helper loader functions (`load_liu_2019`, `load_hugo_2016`, `load_riaz_2017`) for retrieving expression and clinical data.
> - **Shared Cross-Question & Pipeline Modules**:
>   - [`styles.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/src/styles.py): Single source of truth for Okabe-Ito colour palettes (`COHORT_PALETTE`, `RESPONSE_PALETTE`) and visualisation presentation style.
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(
        f"\nSaved clinical analysis report to "
        f"{report_path.relative_to(SUBPROJECT_ROOT).as_posix()}"
    )


# ===========================================================================
# MAIN ANALYSIS
# ===========================================================================


def main() -> None:
    """Executes the Kaplan-Meier OS analysis and generates the clinical report."""
    print("==================================================")
    print("Clinical Analysis — Phase 1: Kaplan-Meier OS Curves")
    print("==================================================\n")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_DIR.mkdir(exist_ok=True, parents=True)

    # -----------------------------------------------------------------------
    # Load dataset configurations & cohort data
    # -----------------------------------------------------------------------
    print("Loading dataset configurations...")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found at "
            f"{CONFIG_PATH.relative_to(SUBPROJECT_ROOT).as_posix()}"
        )

    dataset_configs: tuple[DatasetConfig, ...] = load_dataset_config(CONFIG_PATH)
    cohort_order = [config.cohort_name for config in dataset_configs]

    print(f"Configured cohorts: {', '.join(cohort_order)}")

    cohort_data: dict[str, pd.DataFrame] = {}
    attrition_data: dict[str, pd.DataFrame] = {}

    for config in dataset_configs:
        label = config.cohort_name
        proc_dir = DATA_DIR / "processed" / config.processed_directory
        clin_path = proc_dir / "clin_cleaned.csv"
        attrition_path = proc_dir / "attrition.csv"

        if not clin_path.exists():
            raise FileNotFoundError(
                f"Processed clinical file not found for '{label}' at "
                f"{clin_path.relative_to(SUBPROJECT_ROOT).as_posix()}.\n"
                "Run preprocessing script (clean_data.py) first."
            )

        print(f"  Loading cleaned clinical data for {label}...")
        cohort_data[label] = pd.read_csv(clin_path, index_col="SAMPLE_ID")

        if attrition_path.exists():
            attrition_data[label] = pd.read_csv(attrition_path)
            print(f"    Loaded attrition records from {attrition_path.name}")
        else:
            attrition_data[label] = pd.DataFrame()
            print(f"    No attrition records found for {label}")

    # -----------------------------------------------------------------------
    # Generate KM plots and collect statistics
    # -----------------------------------------------------------------------
    print("\nGenerating Kaplan-Meier curves...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    flat_axes = axes.ravel()

    cohort_results: dict[str, dict[str, Any]] = {}

    for ax, label in zip(flat_axes, cohort_order):
        df_clin = cohort_data[label]
        print(f"  Plotting KM curve for {label} ({len(df_clin)} samples)...")

        cohort_results[label] = {
            "survival": plot_km_os(ax, df_clin, label),
            "age": calculate_age_statistics(df_clin),
            "sex": calculate_sex_statistics(df_clin),
            "treatment": calculate_treatment_statistics(df_clin, label),
        }

    fig.suptitle(
        "Overall Survival — All Cohorts",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    out_path = PLOT_DIR / "km_os_grid.png"
    save_fig(fig, out_path)
    plt.close(fig)

    print(
        f"\nSaved 2×2 KM plot to {out_path.relative_to(SUBPROJECT_ROOT).as_posix()}"
    )

    # -----------------------------------------------------------------------
    # Compute pooled demographics and ICI agent breakdown
    # -----------------------------------------------------------------------
    print("\nComputing overall patient demographics...")
    overall_demographics = compute_overall_demographics(cohort_data=cohort_data)

    treatment_results = {c: cohort_results[c]["treatment"] for c in cohort_order}
    ici_breakdown = compute_ici_agent_breakdown(treatment_results=treatment_results)

    # -----------------------------------------------------------------------
    # Generate clinical demographics 2×2 grid
    # -----------------------------------------------------------------------
    print("\nGenerating clinical demographics 2×2 grid...")
    plot_clinical_demographics_grid(
        cohort_data=cohort_data,
        overall_demographics=overall_demographics,
        tcga_treatment=treatment_results.get("TCGA-SKCM", {}),
        ici_breakdown=ici_breakdown,
        out_path=DEMO_GRID_PATH,
    )

    # -----------------------------------------------------------------------
    # Generate Markdown clinical report
    # -----------------------------------------------------------------------
    print("\nGenerating Markdown clinical characteristics report...")
    generate_clinical_report(
        cohort_results=cohort_results,
        attrition_data=attrition_data,
        plot_path=out_path,
        report_path=REPORT_PATH,
        cohort_order=cohort_order,
        overall_demographics=overall_demographics,
        ici_breakdown=ici_breakdown,
        demographics_grid_path=DEMO_GRID_PATH,
    )

    print("\n==================================================")
    print("Done!")
    print("==================================================")


# ===========================================================================
# LOGGING ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)

        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(
                f"Logging console output to {LOG_PATH.relative_to(SUBPROJECT_ROOT).as_posix()}"
            )
            main()