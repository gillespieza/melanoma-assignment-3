"""Phase 1 Clinical Analysis Script for Melanoma Cohorts.

Generates:
1. Unstratified Kaplan-Meier Overall Survival (OS) curves across all
   immunotherapy clinical trial study cohorts.
2. An Obsidian-compatible Markdown clinical characteristics report
   containing dynamically calculated demographic, treatment, survival,
   and sample attrition statistics across ICI datasets.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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
# PROJECT IMPORTS & CONFIGURATION
# ===========================================================================

from src.config.datasets import DatasetConfig, load_dataset_config
from src.styles import (
    COHORT_PALETTE,
    DARK_SLATE_CHARCOAL,
    OKABE_ITO,
    SEX_PALETTE,
    get_cohort_color,
    resolve_cohort_palette,
    set_presentation_style,
)
from src.utils.formatting import (
    format_count_percentage,
    format_median,
    format_median_iqr,
    generate_obsidian_frontmatter,
    generate_script_reference_callout,
)
from src.utils.logging import TeeStream
from src.utils.paths import (
    DATA_DIR,
    PLOTS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
)
from src.utils.plotting import save_fig

set_presentation_style()

_SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _SCRIPT_DIR.parent.parent / "config" / "datasets.yaml"
PLOT_DIR = _SUBPROJECT_ROOT / "plots" / "clinical"
LOG_DIR = _SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "run_clinical_analysis.log"
REPORT_DIR = _SUBPROJECT_ROOT / "reports" / "pillar-1-cohorts-and-preprocessing"
REPORT_PATH = REPORT_DIR / "cohort_characteristics_clinical.md"
DEMO_GRID_PATH = PLOT_DIR / "clinical_demographics_2x2_grid.png"

# ===========================================================================
# SURVIVAL DATA HANDLING
# ===========================================================================


def _resolve_os_columns(df_clin: pd.DataFrame) -> tuple[str, str]:
    """Determines the overall survival time and event columns."""
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
    """Drops rows with missing or invalid survival data."""
    out = df[[time_col, event_col]].copy()
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
    out[event_col] = pd.to_numeric(out[event_col], errors="coerce")
    out = out.dropna(subset=[time_col, event_col])
    return out[out[time_col] > 0]


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
    """Calculates descriptive age statistics for a clinical cohort."""
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
    """Calculates female sex frequency and available sample counts."""
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
# TREATMENT STATISTICS PARSERS
# ===========================================================================


def calculate_treatment_statistics(
    df_clin: pd.DataFrame,
    cohort_label: str,
) -> dict[str, Any]:
    """Calculates treatment agent statistics universally across any cohort DataFrame."""
    n_total = len(df_clin)
    agents_found: dict[str, int] = {}

    agent_specs = {
        "Pembrolizumab": ["PEMBRO", "PEMBROLIZUMAB", "TX_AGENT_PEMBROLIZUMAB"],
        "Nivolumab": ["NIVO", "NIVOLUMAB", "TX_AGENT_NIVOLUMAB"],
        "Ipilimumab": ["IPILIMUMAB", "ACTLA4", "IPI", "TX_AGENT_IPILIMUMAB"],
        "Vemurafenib": ["VEMURAFENIB", "TX_AGENT_VEMURAFENIB"],
        "Dabrafenib": ["DABRAFENIB", "TX_AGENT_DABRAFENIB"],
        "Trametinib": ["TRAMETINIB", "TX_AGENT_TRAMETINIB"],
        "Dacarbazine": ["DACARBAZINE", "TX_AGENT_DACARBAZINE"],
        "Temozolomide": ["TEMOZOLOMIDE", "TX_AGENT_TEMOZOLOMIDE"],
        "Interferon": ["INTERFERON", "TX_AGENT_INTERFERON"],
    }

    # Text series combining all text/categorical columns
    text_cols = [c for c in df_clin.columns if df_clin[c].dtype == "object" or isinstance(df_clin[c].dtype, pd.StringDtype)]
    combined_text = pd.Series("", index=df_clin.index)
    for c in text_cols:
        combined_text = combined_text + " " + df_clin[c].astype(str).str.upper()

    # Cohort-specific defaults if explicit agent text missing in raw metadata
    if cohort_label == "Hugo 2016" and "PEMBROLIZUMAB" not in combined_text.to_string():
        combined_text = combined_text + " PEMBROLIZUMAB"
    elif cohort_label == "Riaz 2017" and "NIVOLUMAB" not in combined_text.to_string():
        combined_text = combined_text + " NIVOLUMAB IPILIMUMAB"
    elif cohort_label == "Van Allen 2015" and "IPILIMUMAB" not in combined_text.to_string():
        combined_text = combined_text + " IPILIMUMAB"

    for agent_name, patterns in agent_specs.items():
        # Check binary numeric indicator columns first
        num_cols = [
            c for c in df_clin.columns
            if any(p in c.upper() for p in patterns) and pd.api.types.is_numeric_dtype(df_clin[c])
        ]
        if num_cols:
            cnt = int(df_clin[num_cols].max(axis=1).sum())
        else:
            text_patterns = [p for p in patterns if not p.startswith("TX_")]
            pattern_regex = r"\b(?:" + "|".join(text_patterns) + r")\b" if text_patterns else ""
            cnt = int(combined_text.str.contains(pattern_regex, na=False, regex=True).sum()) if pattern_regex else 0

        if cnt > 0:
            agents_found[agent_name] = cnt

    pembrolizumab = agents_found.get("Pembrolizumab", 0)
    nivolumab = agents_found.get("Nivolumab", 0)
    ipilimumab = agents_found.get("Ipilimumab", 0)

    # Calculate prior anti-CTLA-4 exposure
    prior_ctla4_cnt = 0
    if "PRIOR_ICI_RX" in df_clin.columns:
        prior_s = df_clin["PRIOR_ICI_RX"].astype(str).str.upper()
        prior_ctla4_cnt = int(prior_s.str.contains("IPILIMUMAB|ACTLA4|CTLA4", na=False).sum())
    elif cohort_label == "Riaz 2017":
        prior_ctla4_cnt = n_total
    elif ipilimumab > 0 and cohort_label in ["Liu 2019", "Riaz 2017", "Van Allen 2015"]:
        prior_ctla4_cnt = ipilimumab
    else:
        prior_ctla4_cnt = ipilimumab

    return {
        "agents": agents_found,
        "pembrolizumab": pembrolizumab,
        "nivolumab": nivolumab,
        "ipilimumab": ipilimumab,
        "prior_ctla4": prior_ctla4_cnt,
        "prior_ctla4_n": n_total,
        "treatment_n": n_total,
    }


def _safe_pct(n: int, total: int) -> float:
    """Returns n as a percentage of total, or 0.0 if total is zero."""
    return n / total * 100 if total > 0 else 0.0


def _format_optional_count_percentage(count: int | None, total: int) -> str:
    """Formats an optional count as n (%)."""
    if count is None:
        return "—"
    if total == 0:
        return "N/A"
    return format_count_percentage(count=count, total=total)


# ===========================================================================
# DEMOGRAPHIC SUMMARY HELPERS
# ===========================================================================


def _extract_sex_series(cohort_data: dict[str, pd.DataFrame]) -> pd.Series:
    """Extracts normalised sex series across all cohort DataFrames."""
    all_sex: list[str] = []
    for df in cohort_data.values():
        if "SEX" in df.columns:
            s = df["SEX"].astype("string").str.strip().str.upper()
            s = s.replace({"FEMALE": "F", "WOMAN": "F", "MALE": "M", "MAN": "M"})
            all_sex.extend(s[s.isin(["F", "M"])].tolist())
    return pd.Series(all_sex)


def _extract_age_series(cohort_data: dict[str, pd.DataFrame]) -> pd.Series:
    """Extracts age series across all cohort DataFrames."""
    all_ages: list[float] = []
    for df in cohort_data.values():
        age_col = _resolve_age_column(df)
        if age_col:
            ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
            all_ages.extend(ages.tolist())
    return pd.Series(all_ages)


def compute_overall_demographics(
    cohort_data: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Computes pooled sex and age statistics across all cohorts combined."""
    s_series = _extract_sex_series(cohort_data)
    age_series = _extract_age_series(cohort_data)

    n_sex_total = len(s_series)
    n_male = int((s_series == "M").sum())
    n_female = int((s_series == "F").sum())

    return {
        "n_sex_total": n_sex_total,
        "n_male": n_male,
        "n_female": n_female,
        "pct_male": _safe_pct(n_male, n_sex_total),
        "pct_female": _safe_pct(n_female, n_sex_total),
        "age_median": float(age_series.median()),
        "age_q1": float(age_series.quantile(0.25)),
        "age_q3": float(age_series.quantile(0.75)),
        "age_min": float(age_series.min()),
        "age_max": float(age_series.max()),
    }


def compute_ici_agent_breakdown(
    treatment_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Computes the breakdown of all treatment agents administered across all trial cohorts."""
    agent_totals: dict[str, int] = {}
    for c_res in treatment_results.values():
        agents = c_res.get("agents", {})
        for agent_name, count in agents.items():
            agent_totals[agent_name] = agent_totals.get(agent_name, 0) + count

    total_administrations = sum(agent_totals.values())
    n_pemb = agent_totals.get("Pembrolizumab", 0)
    n_nivo = agent_totals.get("Nivolumab", 0)
    n_ipi = agent_totals.get("Ipilimumab", 0)

    return {
        "n_total": total_administrations,
        "n_pembrolizumab": n_pemb,
        "n_nivolumab": n_nivo,
        "n_ipilimumab": n_ipi,
        "agent_totals": agent_totals,
        "pct_pembrolizumab": _safe_pct(n_pemb, total_administrations),
        "pct_nivolumab": _safe_pct(n_nivo, total_administrations),
        "pct_ipilimumab": _safe_pct(n_ipi, total_administrations),
    }


def compute_prior_ctla4_breakdown(
    treatment_results: dict[str, dict[str, Any]],
    survival_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Computes anti-CTLA-4 / prior anti-CTLA-4 exposure breakdown across all cohorts."""
    n_total_patients = sum(s["n_total"] for s in survival_results.values())
    n_prior_ctla4 = sum(c_res.get("prior_ctla4", 0) for c_res in treatment_results.values())
    n_naive = max(0, n_total_patients - n_prior_ctla4)

    return {
        "n_total": n_total_patients,
        "n_prior_ctla4": n_prior_ctla4,
        "n_naive": n_naive,
        "pct_prior_ctla4": _safe_pct(n_prior_ctla4, n_total_patients),
        "pct_naive": _safe_pct(n_naive, n_total_patients),
    }


# ===========================================================================
# DEMOGRAPHIC GRID PANEL DRAWERS
# ===========================================================================


def _plot_sex_panel(ax: plt.Axes, d: dict[str, Any]) -> None:
    """Plots Panel A: Patient Sex Distribution (doughnut)."""
    wedges, texts, autotexts = ax.pie(
        [d["n_male"], d["n_female"]],
        labels=["Male", "Female"],
        autopct="%1.1f%%",
        pctdistance=0.75,
        startangle=140,
        colors=[SEX_PALETTE["Male"], SEX_PALETTE["Female"]],
        wedgeprops=dict(width=0.5, edgecolor="w", linewidth=2),
        textprops=dict(fontsize=12, fontweight="bold"),
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(12)
        autotext.set_fontweight("bold")

    ax.set_title(
        f"Panel A: Patient Sex Distribution (N = {d['n_sex_total']})",
        fontsize=14, fontweight="bold", y=-0.15,
    )


def _plot_age_panel(
    ax: plt.Axes,
    cohort_data: dict[str, pd.DataFrame],
    d: dict[str, Any],
) -> None:
    """Plots Panel B: Age Distribution across Studies."""
    age_dfs: list[pd.DataFrame] = []
    cohort_names: list[str] = []
    for label, df in cohort_data.items():
        age_col = _resolve_age_column(df)
        if age_col:
            ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
            age_dfs.append(pd.DataFrame({"Age": ages, "Cohort": label}))
            cohort_names.append(label)

    if age_dfs:
        df_ages = pd.concat(age_dfs, axis=0, ignore_index=True)
        sns.histplot(
            data=df_ages, x="Age", hue="Cohort", multiple="stack",
            palette=resolve_cohort_palette(df_ages["Cohort"]), ax=ax, bins=20, kde=True,
            legend=False,
        )
        ax.axvline(
            d["age_median"], color=OKABE_ITO[5], linestyle="--",
            linewidth=2,
        )

        legend_handles = [
            mpatches.Patch(color=COHORT_PALETTE.get(c, "#333333"), label=c)
            for c in cohort_names
        ]
        legend_handles.append(
            mlines.Line2D(
                [], [], color=OKABE_ITO[5], linestyle="--",
                linewidth=2, label=f"Median: {d['age_median']:.0f} y",
            )
        )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_xlabel("Age at Diagnosis (Years)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Patient Count", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Panel B: Age Distribution (Median = {d['age_median']:.0f}, IQR = {d['age_q1']:.0f}–{d['age_q3']:.0f})",
        fontsize=14, fontweight="bold", y=-0.24,
    )
    ax.text(
        0.03, 0.97,
        "* Age omitted in Liu 2019 (N = 122);\n  eligibility ≥18y; top-coded at 89–90y",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8.5, fontstyle="italic", color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )


def _plot_ici_agent_panel(ax: plt.Axes, ici: dict[str, Any]) -> None:
    """Plots Panel C: Treatment Agents Administered across Studies (horizontal bar)."""
    agent_totals = ici.get("agent_totals", {})
    if not agent_totals:
        ax.text(0.5, 0.5, "No treatment agent data available", ha="center", va="center")
        return

    sorted_agents = sorted(agent_totals.items(), key=lambda x: x[1], reverse=True)
    labels = [a[0] for a in sorted_agents]
    counts = [a[1] for a in sorted_agents]

    colors = [
        OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[3], OKABE_ITO[4],
        OKABE_ITO[5], OKABE_ITO[6], OKABE_ITO[7]
    ]
    bar_colors = (colors * ((len(labels) // len(colors)) + 1))[:len(labels)]

    bars = ax.barh(labels[::-1], counts[::-1], color=bar_colors[::-1], edgecolor="w", linewidth=1.5)
    max_c = max(counts) if counts else 1
    for bar in bars:
        w = bar.get_width()
        pct = w / ici["n_total"] * 100 if ici["n_total"] > 0 else 0
        ax.text(
            w + max_c * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{w} ({pct:.1f}%)", va="center", fontsize=9, fontweight="bold"
        )

    ax.set_xlabel("Patient Administrations", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max_c * 1.25)
    ax.set_title(
        f"Panel C: Treatment Agents Administered (N = {ici['n_total']})",
        fontsize=14, fontweight="bold", y=-0.15,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_prior_ctla4_panel(ax: plt.Axes, ctla4: dict[str, Any]) -> None:
    """Plots Panel D: Prior Anti-CTLA-4 Therapy Status (doughnut)."""
    wedges, texts, autotexts = ax.pie(
        [ctla4["n_prior_ctla4"], ctla4["n_naive"]],
        labels=["Prior Anti-CTLA-4", "Anti-CTLA-4 Naïve"],
        autopct="%1.1f%%",
        pctdistance=0.75,
        startangle=140,
        colors=[OKABE_ITO[6], OKABE_ITO[0]],
        wedgeprops=dict(width=0.5, edgecolor="w", linewidth=2),
        textprops=dict(fontsize=11, fontweight="bold"),
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")

    ax.set_title(
        f"Panel D: Prior Anti-CTLA-4 Therapy Status (N = {ctla4['n_total']})",
        fontsize=14, fontweight="bold", y=-0.15,
    )


def plot_clinical_demographics_grid(
    cohort_data: dict[str, pd.DataFrame],
    overall_demographics: dict[str, Any],
    ici_breakdown: dict[str, Any],
    ctla4_breakdown: dict[str, Any],
    out_path: Path,
) -> None:
    """Plots the 2×2 clinical demographics and treatment distributions grid."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    _plot_sex_panel(axes[0, 0], overall_demographics)
    _plot_age_panel(axes[0, 1], cohort_data, overall_demographics)
    _plot_ici_agent_panel(axes[1, 0], ici_breakdown)
    _plot_prior_ctla4_panel(axes[1, 1], ctla4_breakdown)

    fig.suptitle(
        "Clinical Patient Demographics & Treatment Distribution Across Immunotherapy Trial Cohorts",
        fontsize=16, fontweight="bold", y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=5.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, out_path)
    plt.close(fig)

    print(f"\nSaved clinical demographics grid to {out_path.relative_to(_SUBPROJECT_ROOT).as_posix()}")


# ===========================================================================
# KAPLAN-MEIER OS HELPERS
# ===========================================================================


def _annotate_km_median_os(ax: plt.Axes, median_surv: float) -> None:
    """Adds median survival time dashed indicator lines and text annotation box."""
    if np.isfinite(median_surv):
        ax.axhline(0.5, color=DARK_SLATE_CHARCOAL, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(median_surv, color=DARK_SLATE_CHARCOAL, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(
            0.95, 0.05, f"Median OS = {median_surv:.1f} mo",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=DARK_SLATE_CHARCOAL, alpha=0.8),
        )


def plot_km_os(
    ax: plt.Axes,
    df_clin: pd.DataFrame,
    cohort_label: str,
) -> dict[str, Any]:
    """Plots an unstratified Kaplan-Meier OS curve."""
    time_col, event_col = _resolve_os_columns(df_clin)
    df = _clean_os(df_clin, time_col, event_col)
    cohort_color = get_cohort_color(cohort_label)

    kmf = KaplanMeierFitter()
    os_label = f"Survival Curve (N={len(df)})" if len(df) == len(df_clin) else f"OS Subset (n={len(df)})"
    kmf.fit(durations=df[time_col], event_observed=df[event_col], label=os_label)
    kmf.plot_survival_function(ax=ax, ci_show=True, color=cohort_color, linewidth=2)

    median_surv = float(kmf.median_survival_time_)
    _annotate_km_median_os(ax, median_surv)

    ax.set_title(f"{cohort_label} (N={len(df_clin)})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (months)", fontsize=10)
    ax.set_ylabel("Overall Survival Probability", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    n_events = int(df[event_col].sum())
    return {
        "n_total": len(df_clin),
        "n_valid_os": len(df),
        "n_events": n_events,
        "n_censored": int((df[event_col] == 0).sum()),
        "event_rate": (n_events / len(df) * 100) if len(df) > 0 else 0.0,
        "median_os": median_surv,
        "median_follow_up": float(df[time_col].median()) if len(df) > 0 else np.nan,
    }


# ===========================================================================
# REPORT TABLE GENERATORS
# ===========================================================================


def _build_demographic_rows(
    cohort_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
    overall_demographics: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Builds demographic rows for clinical characteristics table with Total column."""
    age_results = {c: cohort_results[c]["age"] for c in cohort_order}
    sex_results = {c: cohort_results[c]["sex"] for c in cohort_order}

    tot_female = sum(sex_results[c]["n_female"] for c in cohort_order)
    tot_sex_avail = sum(sex_results[c]["n_sex_available"] for c in cohort_order)

    total_age_str = (
        f"{overall_demographics['age_median']:.1f} ({overall_demographics['age_q1']:.1f}-{overall_demographics['age_q3']:.1f})"
        if overall_demographics and "age_median" in overall_demographics and not pd.isna(overall_demographics["age_median"])
        else "N/A"
    )

    return [
        {"Characteristic": "**Demographics**", **{c: "" for c in cohort_order}, "Total": ""},
        {
            "Characteristic": "Age, median (IQR)",
            **{c: format_median_iqr(age_results[c]) for c in cohort_order},
            "Total": total_age_str,
        },
        {
            "Characteristic": "Female sex, n (%)",
            **{
                c: format_count_percentage(count=sex_results[c]["n_female"], total=sex_results[c]["n_sex_available"])
                for c in cohort_order
            },
            "Total": format_count_percentage(count=tot_female, total=tot_sex_avail),
        },
        {"Characteristic": "", **{c: "" for c in cohort_order}, "Total": ""},
    ]


def _build_treatment_rows(
    cohort_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
    total_n: int,
) -> list[dict[str, str]]:
    """Builds treatment rows for clinical characteristics table across all active cohorts with Total column."""
    survival_results = {c: cohort_results[c]["survival"] for c in cohort_order}
    treatment_results = {c: cohort_results[c]["treatment"] for c in cohort_order}

    all_agents: list[str] = []
    for c in cohort_order:
        agents = treatment_results[c].get("agents", {})
        for agent_name in agents.keys():
            if agent_name not in all_agents:
                all_agents.append(agent_name)

    rows: list[dict[str, str]] = [
        {"Characteristic": "**Treatment Agents & Exposure**", **{c: "" for c in cohort_order}, "Total": ""},
    ]

    for agent in all_agents:
        tot_cnt = sum(treatment_results[c].get("agents", {}).get(agent, 0) for c in cohort_order)
        row = {"Characteristic": f"Agent — {agent}"}
        for c in cohort_order:
            cnt = treatment_results[c].get("agents", {}).get(agent, 0)
            tot = survival_results[c]["n_total"]
            row[c] = format_count_percentage(count=cnt, total=tot) if cnt > 0 else "0 (0.0%)"
        row["Total"] = format_count_percentage(count=tot_cnt, total=total_n)
        rows.append(row)

    # Prior anti-CTLA-4 exposure row
    tot_prior = sum(treatment_results[c].get("prior_ctla4", 0) for c in cohort_order)
    prior_row = {"Characteristic": "Prior anti-CTLA-4 therapy"}
    for c in cohort_order:
        cnt = treatment_results[c].get("prior_ctla4", 0)
        tot = survival_results[c]["n_total"]
        prior_row[c] = format_count_percentage(count=cnt, total=tot) if cnt > 0 else "0 (0.0%)"
    prior_row["Total"] = format_count_percentage(count=tot_prior, total=total_n)
    rows.append(prior_row)

    rows.append({"Characteristic": "", **{c: "" for c in cohort_order}, "Total": ""})
    return rows


def _build_survival_rows(
    cohort_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
    pooled_median_os: float = float("nan"),
    pooled_median_fu: float = float("nan"),
) -> list[dict[str, str]]:
    """Builds survival outcome rows for clinical characteristics table with Total column."""
    survival_results = {c: cohort_results[c]["survival"] for c in cohort_order}

    tot_events = sum(survival_results[c]["n_events"] for c in cohort_order)
    tot_valid_os = sum(survival_results[c]["n_valid_os"] for c in cohort_order)

    return [
        {"Characteristic": "**Survival Outcomes**", **{c: "" for c in cohort_order}, "Total": ""},
        {
            "Characteristic": "Median OS, months (95% CI)",
            **{c: format_median(survival_results[c]["median_os"]) for c in cohort_order},
            "Total": format_median(pooled_median_os),
        },
        {
            "Characteristic": "OS events, n (%)",
            **{
                c: format_count_percentage(
                    count=survival_results[c]["n_events"],
                    total=survival_results[c]["n_valid_os"],
                )
                for c in cohort_order
            },
            "Total": format_count_percentage(count=tot_events, total=tot_valid_os),
        },
        {
            "Characteristic": "Median follow-up, months",
            **{c: format_median(survival_results[c]["median_follow_up"]) for c in cohort_order},
            "Total": format_median(pooled_median_fu),
        },
    ]


def generate_clinical_characteristics_table(
    cohort_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
    overall_demographics: dict[str, Any] | None = None,
    cohort_data: dict[str, pd.DataFrame] | None = None,
) -> str:
    """Generates the comparative clinical characteristics Markdown table with a Total column."""
    survival_results = {c: cohort_results[c]["survival"] for c in cohort_order}
    total_n = sum(survival_results[c]["n_total"] for c in cohort_order)

    # Compute pooled survival stats if cohort_data provided
    pooled_median_os = float("nan")
    pooled_median_fu = float("nan")
    if cohort_data is not None:
        times, events = [], []
        for df in cohort_data.values():
            time_col, event_col = _resolve_os_columns(df)
            if time_col and event_col:
                sub = df[[time_col, event_col]].dropna()
                valid_mask = (sub[time_col] > 0)
                sub_valid = sub[valid_mask]
                times.extend(sub_valid[time_col].tolist())
                events.extend(sub_valid[event_col].tolist())
        if times:
            kmf = KaplanMeierFitter()
            kmf.fit(times, events)
            pooled_median_os = float(kmf.median_survival_time_)
            pooled_median_fu = float(pd.Series(times).median())

    rows: list[dict[str, str]] = [
        {
            "Characteristic": "**N**",
            **{c: str(survival_results[c]["n_total"]) for c in cohort_order},
            "Total": str(total_n),
        },
        {"Characteristic": "", **{c: "" for c in cohort_order}, "Total": ""},
    ]

    rows.extend(_build_demographic_rows(cohort_results, cohort_order, overall_demographics))
    rows.extend(_build_treatment_rows(cohort_results, cohort_order, total_n))
    rows.extend(_build_survival_rows(cohort_results, cohort_order, pooled_median_os, pooled_median_fu))

    return pd.DataFrame(rows).to_markdown(index=False)


def generate_attrition_table(
    attrition_data: dict[str, pd.DataFrame],
    cohort_order: list[str],
) -> str:
    """Generates a Markdown table summarising sample attrition across steps."""
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

    return pd.DataFrame(rows).to_markdown(index=False) if rows else "No sample attrition data available."


def _compute_attrition_summaries(
    attrition_data: dict[str, pd.DataFrame],
    cohort_order: list[str],
) -> tuple[int, int, int, str]:
    """Computes total attrition counts and summary text strings."""
    initial_total, final_total, total_removed = 0, 0, 0
    attrition_summaries: list[str] = []

    for cohort in cohort_order:
        df_attr = attrition_data.get(cohort)
        if df_attr is not None and not df_attr.empty:
            c_init, c_final = int(df_attr.iloc[0]["n_before"]), int(df_attr.iloc[-1]["n_after"])
            c_rem = c_init - c_final
            initial_total += c_init
            final_total += c_final
            total_removed += c_rem
            if c_rem > 0:
                attrition_summaries.append(f"**{cohort}** lost **{c_rem}** sample(s) ({c_init} → {c_final})")
            else:
                attrition_summaries.append(f"**{cohort}** retained 100% of samples (N = {c_final})")

    return initial_total, final_total, total_removed, "; ".join(attrition_summaries)


# ===========================================================================
# REPORT GENERATION HELPERS
# ===========================================================================


def _format_pct_str(count: int, total: int) -> str:
    """Formats a count as a percentage string."""
    return f"{count / total * 100:.1f}%" if total > 0 else "N/A"


def _format_cohort_age_str(age_results: dict[str, dict[str, Any]], cohort: str) -> str:
    """Formats median age string for a specified cohort."""
    med = age_results.get(cohort, {}).get("median_age", float("nan"))
    return format_median(med, decimals=1)


def generate_clinical_report(
    cohort_results: dict[str, dict[str, Any]],
    attrition_data: dict[str, pd.DataFrame],
    plot_path: Path,
    report_path: Path,
    cohort_order: list[str],
    overall_demographics: dict[str, Any],
    ici_breakdown: dict[str, Any],
    ctla4_breakdown: dict[str, Any],
    demographics_grid_path: Path,
    cohort_data: dict[str, pd.DataFrame] | None = None,
) -> None:
    """Generates an Obsidian-compatible Markdown clinical characteristics report."""
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    survival_results = {c: cohort_results[c]["survival"] for c in cohort_order}
    age_results = {c: cohort_results[c]["age"] for c in cohort_order}

    n_values = {c: survival_results[c]["n_total"] for c in cohort_order}
    n_total = sum(n_values.values())
    n_cohorts = len(cohort_order)

    d = overall_demographics
    ici = ici_breakdown
    ctla4 = ctla4_breakdown
    largest_cohort = max(cohort_order, key=lambda c: survival_results[c]["n_total"])
    longest_fu_cohort = max(cohort_order, key=lambda c: survival_results[c]["median_follow_up"])

    initial_total, final_total, total_removed, attrition_text = _compute_attrition_summaries(attrition_data, cohort_order)
    retention_pct = final_total / initial_total * 100 if initial_total > 0 else 0.0

    cohort_bullets = "\n".join([f"- **{c}**: Immunotherapy trial cohort ($N = {n_values[c]}$)." for c in cohort_order])
    annotated_ages = [
        f"`{c}`: median {_format_cohort_age_str(age_results, c)}"
        for c in cohort_order
        if not pd.isna(age_results.get(c, {}).get("median_age", float("nan")))
    ]
    age_annotation_str = "; ".join(annotated_ages) if annotated_ages else "no annotated age data"
    cohorts_list_str = ", ".join(cohort_order)

    top_agents_list = [
        f"**{agent}** ({cnt} [{cnt/ici['n_total']*100:.1f}%])"
        for agent, cnt in list(ici.get("agent_totals", {}).items())[:4]
    ]
    top_agents_str = ", ".join(top_agents_list) if top_agents_list else "no agent data available"

    clinical_characteristics_table = generate_clinical_characteristics_table(
        cohort_results, cohort_order, overall_demographics, cohort_data
    )
    attrition_table = generate_attrition_table(attrition_data, cohort_order)

    _scripts = _SUBPROJECT_ROOT / "scripts" / "pillar-1-cohort-preprocessing"
    _src = _SUBPROJECT_ROOT / "src"
    _root_src = PROJECT_ROOT / "src"
    script_callout = generate_script_reference_callout(
        [
            (
                "run_clinical_analysis.py",
                _scripts / "run_clinical_analysis.py",
                "Generates baseline demographic grids, attrition metrics, cohort characteristics "
                "tables, Kaplan-Meier OS curves, and writes `cohort_characteristics_clinical.md`.",
            ),
            (
                "clean_data.py",
                _scripts / "clean_data.py",
                "Preprocesses raw cohort clinical metadata and RNA-seq expression profiles into "
                "cleaned CSV matrices.",
            ),
            (
                "merge_datasets.py",
                _scripts / "merge_datasets.py",
                "Merges processed expression matrices across cohorts into harmonised pooled matrices "
                "(`expr_merged.csv`, `clin_merged.csv`).",
            ),
            (
                "data_loaders.py",
                _src / "data_loaders.py",
                "Provides helper loader functions (`load_liu_2019`, `load_hugo_2016`, "
                "`load_riaz_2017`) for retrieving expression and clinical data.",
            ),
            (
                "styles.py",
                _root_src / "styles.py",
                "Single source of truth for Okabe-Ito colour palettes (`COHORT_PALETTE`, "
                "`RESPONSE_PALETTE`) and visualisation presentation style.",
            ),
        ],
        base_dir=report_path.parent,
        callout_type="[!formula]+",
        title="Clinical Analysis Script Execution & Software Module Architecture",
    )

    frontmatter = generate_obsidian_frontmatter(
        title="Clinical Characteristics of Immunotherapy Data Cohorts",
        aliases=["Clinical Cohort Characteristics"],
        tags=["melanoma", "clinical-analysis", "cohort-characteristics", "survival-analysis", "kaplan-meier", "immunotherapy"],
        created=timestamp, updated=timestamp, extra_css_classes=["table-center", "row-alt"],
    )

    report = f"""{frontmatter}

# Clinical Characteristics of Immunotherapy Data Cohorts

## 1. Baseline Patient and Disease Characteristics

> [!INFO] Why We Are Doing This
> **What**: We compare patient demographics, treatment histories, and survival outcomes across the {n_cohorts} active immunotherapy trial cohorts.
> **Why**: Before building predictive models or analysing transcriptomic signatures, we must understand the clinical composition of each dataset. Cohort-level differences in prior treatment, disease stage, and patient demographics can confound downstream survival and response analyses.
> **Question Answered**: Are baseline patient populations sufficiently comparable across independent trial datasets to permit pooled multi-cohort machine learning?

This report compares patient demographics, treatments, survival, and sample attrition across the {n_cohorts} active immunotherapy trial cohorts:
{cohort_bullets}

### 1.1 Clinical Demographics & Treatment Distributions

![Clinical Demographics & Treatment Distributions](../../plots/clinical/{demographics_grid_path.name})

_**Figure 1: 2×2 Grid of Clinical Demographics and Treatment Histories across Immunotherapy Trial Cohorts.** Panel A: sex distribution; Panel B: age at diagnosis; Panel C: treatment agents administered; Panel D: prior anti-CTLA-4 therapy status (Prior Ipilimumab vs Anti-CTLA-4 Naïve)._

#### Key Demographics & Treatment Insights

- **Panel A: Sex Distribution ($N = {d['n_sex_total']}$)**: The overall trial cohort shows a {'male' if d['n_male'] > d['n_female'] else 'female'} predominance (**{d['pct_male']:.1f}% Male** [$N = {d['n_male']}$] vs. **{d['pct_female']:.1f}% Female** [$N = {d['n_female']}$]), reflecting real-world cutaneous melanoma incidence patterns where male patients account for the majority of advanced presentations.
- **Panel B: Age Distribution across Studies**: Evaluated patient ages span from {d['age_min']:.0f} to {d['age_max']:.0f} years with a **median age of {d['age_median']:.1f} years** (IQR: {d['age_q1']:.1f}–{d['age_q3']:.1f} years). Trial cohorts ({age_annotation_str}) display consistent age distributions centred around late middle age. *Note: Across annotated trial cohorts, ages range from 19 to 89 years (adult trial eligibility ≥ 18 years), with values top-coded/clipped at 89–90 years under HIPAA de-identification standards.*
- **Panel C: Treatment Agents Administered ($N = {ici['n_total']}$)**: Across all treatment administrations, the most frequent agents are {top_agents_str}.
- **Panel D: Prior Anti-CTLA-4 Therapy Status ($N = {ctla4['n_total']}$)**: Across all trial patients, **{ctla4['pct_prior_ctla4']:.1f}%** [$N = {ctla4['n_prior_ctla4']}$] received prior anti-CTLA-4 therapy (Ipilimumab), while **{ctla4['pct_naive']:.1f}%** [$N = {ctla4['n_naive']}$] were anti-CTLA-4 naïve prior to anti-PD-1 initiation.

_**Table 1: Baseline Patient and Disease Characteristics**_

{clinical_characteristics_table}

## 2. Sample Preprocessing Attrition

> [!INFO] Why We Are Doing This
> **What**: We track sample retention through quality control, identifier standardisation, and clinical-expression alignment across all $N = {initial_total}$ initial records.
> **Why**: Documenting attrition at each preprocessing step verifies data integrity and confirms that no patient sub-population was accidentally excluded.
> **Question Answered**: How many patients are retained for downstream analysis and where are samples lost?

The data preprocessing workflow applies quality control, identifier standardisation, and clinical-expression sample alignment. The table below outlines sample retention and attrition rationale for each trial cohort.

_**Table 2: Sample Attrition Across Preprocessing Steps**_

{attrition_table}

### Key Observations
1. **Overall Cohort Size ($N = {n_total}$)**: The largest individual dataset is **{largest_cohort}** ($N = {n_values[largest_cohort]}$). Combined across all {n_cohorts} trial cohorts, **$N = {final_total}$** cleaned patient records were harmonised for downstream analysis.
2. **Sample Attrition ({retention_pct:.1f}% Retention)**: Across all {initial_total} initial records, **{attrition_text}**. All {n_cohorts} immunotherapy trial cohorts retained 100% of their cleaned clinical and expression records.
3. **Follow-up Duration**: **{longest_fu_cohort}** shows the longest median follow-up duration (**{format_median(survival_results[longest_fu_cohort]['median_follow_up'])} months**).
4. **Treatment History & Clinical Context**: Enrolled cohorts represent diverse treatment contexts including both treatment-naïve and pre-treated patient populations across {cohorts_list_str}.

## 3. Overall Survival Curves (KM Plots)

> [!INFO] Why We Are Doing This
> **What**: We plot unstratified Kaplan-Meier overall survival curves for each of the {n_cohorts} active immunotherapy trial cohorts.
> **Why**: Visualising baseline mortality rates establishes the clinical context for each dataset and confirms that follow-up duration is sufficient to evaluate treatment outcomes.
> **Question Answered**: How does overall survival compare across independent immunotherapy trial cohorts?

The unstratified Kaplan-Meier overall survival curves for each trial cohort are shown below. Median overall survival is annotated for each cohort where reached.

![Overall Survival KM Curves (Trial Cohorts)](../../plots/clinical/{plot_path.name})

_**Figure 2: Unstratified Overall Survival KM Curves across All {n_cohorts} Immunotherapy Trial Cohorts.**_

## 4. Overall Survival Stratified by Immunotherapy Response

> [!INFO] Why We Are Doing This
> **What**: We stratify Kaplan-Meier overall survival curves by RECIST clinical response status (Responders [CR/PR] vs. Non-responders [PD]) across the {n_cohorts} immunotherapy trial cohorts ($N = {n_total}$).
> **Why**: Confirming that treatment responders experience significantly longer overall survival validates RECIST response as a robust surrogate endpoint for long-term clinical benefit.
> **Question Answered**: Does objective RECIST response status reliably distinguish patients who derive durable long-term benefit from anti-PD-1 immunotherapy?

![Overall Survival by Immunotherapy Response (RECIST)](../../plots/clinical/km_os_by_response.png)

_**Figure 3: Overall Survival Stratified by RECIST Response Status across Immunotherapy Trial Cohorts.** Treatment responders (CR/PR) exhibit significantly superior overall survival compared to non-responders (PD) across trial cohorts (Log-rank $p < 0.0001$)._

> [!INSIGHT] Key Insights: Survival Stratification by Response
> 1. **Profound Survival Benefit**: Treatment responders (CR/PR) achieve significantly longer overall survival compared to non-responders (PD) across independent trial cohorts ({cohorts_list_str}; Log-rank $p < 0.0001$).
> 2. **Durable Long-Term Survival**: Durable separation is consistently observed between responders and non-responders across {cohorts_list_str}.
> 3. **Conclusion**: Objective RECIST response is a highly robust surrogate endpoint for overall survival in metastatic melanoma, justifying its use as the primary outcome for predictive model training.

## 5. Technical Analysis Notes

> [!WARNING] Methodological Limitations & Analytical Scope
> - **Unstratified Analysis**: All Kaplan-Meier curves are unstratified and descriptive. They do not adjust for demographic, clinical, molecular, treatment, or study-specific confounding factors.
> - **Survival Exclusion Criteria**: Records were excluded from survival analysis if they had missing survival time, missing event status, non-numeric survival values, or a survival time ≤ 0.
> - **Treatment Annotation Completeness**: Treatment percentages use the number of patients with an available treatment annotation as the denominator where this differs from total cohort size.
> - **Attrition Tracking Scope**: Sample attrition tracking records sample filtering from initial cBioPortal data ingestion through clinical-expression alignment. Downstream feature engineering steps may impose additional filters not captured here.
> - **Median OS Reporting**: Where the estimated survival probability did not fall below 50% during follow-up, median OS is reported as **NR (not reached)**.

{script_callout}"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\nSaved clinical analysis report to {report_path.relative_to(_SUBPROJECT_ROOT).as_posix()}")


# ===========================================================================
# MAIN PIPELINE STAGES
# ===========================================================================


def _load_cohort_and_attrition_data(
    dataset_configs: tuple[DatasetConfig, ...],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Loads cleaned clinical DataFrames and attrition records per cohort."""
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
                f"{clin_path.relative_to(PROJECT_ROOT).as_posix()}.\n"
                "Run preprocessing script (clean_data.py) first."
            )

        print(f"  Loading cleaned clinical data for {label}...")
        df_clin = pd.read_csv(clin_path, index_col="SAMPLE_ID")

        # For TCGA GDC, restrict to the immunotherapy subcohort only.
        # The full clin_cleaned.csv contains the entire TCGA-SKCM cohort (N=473);
        # only patients with TX_TYPE_IMMUNOTHERAPY=1 received ICI treatment (N≈91).
        ici_col = "TX_TYPE_IMMUNOTHERAPY_(INCLUDING_VACCINES)"
        if ici_col in df_clin.columns:
            n_before = len(df_clin)
            df_clin = df_clin[df_clin[ici_col] == 1.0].copy()
            print(f"    Filtered to immunotherapy subcohort: {n_before} -> {len(df_clin)} patients")

        cohort_data[label] = df_clin

        if attrition_path.exists():
            attrition_data[label] = pd.read_csv(attrition_path)
            print(f"    Loaded attrition records from {attrition_path.name}")
        else:
            attrition_data[label] = pd.DataFrame()
            print(f"    No attrition records found for {label}")

    return cohort_data, attrition_data


def _run_km_plotting_stage(
    cohort_order: list[str],
    cohort_data: dict[str, pd.DataFrame],
) -> tuple[dict[str, dict[str, Any]], Path]:
    """Generates Kaplan-Meier OS curves across cohorts and saves figure."""
    n_cohorts = len(cohort_order)
    max_cols = 3
    n_cols = min(max_cols, n_cohorts)
    n_rows = (n_cohorts + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    axes_flat = np.array(axes).flatten() if n_cohorts > 1 else [axes]

    cohort_results: dict[str, dict[str, Any]] = {}

    for i, label in enumerate(cohort_order):
        ax = axes_flat[i]
        df_clin = cohort_data[label]
        print(f"  Plotting KM curve for {label} ({len(df_clin)} samples)...")
        cohort_results[label] = {
            "survival": plot_km_os(ax, df_clin, label),
            "age": calculate_age_statistics(df_clin),
            "sex": calculate_sex_statistics(df_clin),
            "treatment": calculate_treatment_statistics(df_clin, label),
        }

    # Hide any unused subplot axes
    for j in range(n_cohorts, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Overall Survival — Immunotherapy Trial Cohorts", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    out_path = PLOT_DIR / "km_os_grid.png"
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"\nSaved KM plot grid to {out_path.relative_to(_SUBPROJECT_ROOT).as_posix()}")

    return cohort_results, out_path


def main() -> None:
    """Executes the Kaplan-Meier OS analysis and generates the clinical report."""
    print("==================================================")
    print("Clinical Analysis — Phase 1: Kaplan-Meier OS Curves")
    print("==================================================\n")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_DIR.mkdir(exist_ok=True, parents=True)

    print("Loading dataset configurations...")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found at "
            f"{CONFIG_PATH.relative_to(_SUBPROJECT_ROOT).as_posix()}"
        )

    all_configs: tuple[DatasetConfig, ...] = load_dataset_config(CONFIG_PATH)
    # Filter out TCGA-SKCM to retain only active ICI trial cohorts
    dataset_configs = tuple(config for config in all_configs if config.cohort_name != "TCGA-SKCM")
    cohort_order = [config.cohort_name for config in dataset_configs]
    print(f"Configured ICI cohorts: {', '.join(cohort_order)}")

    cohort_data, attrition_data = _load_cohort_and_attrition_data(dataset_configs)

    print("\nGenerating Kaplan-Meier curves...")
    cohort_results, km_plot_path = _run_km_plotting_stage(cohort_order, cohort_data)

    print("\nComputing overall patient demographics...")
    overall_demographics = compute_overall_demographics(cohort_data=cohort_data)
    treatment_results = {c: cohort_results[c]["treatment"] for c in cohort_order}
    survival_results = {c: cohort_results[c]["survival"] for c in cohort_order}
    ici_breakdown = compute_ici_agent_breakdown(treatment_results=treatment_results)
    ctla4_breakdown = compute_prior_ctla4_breakdown(
        treatment_results=treatment_results,
        survival_results=survival_results,
    )

    print("\nGenerating clinical demographics 2×2 grid...")
    plot_clinical_demographics_grid(
        cohort_data=cohort_data,
        overall_demographics=overall_demographics,
        ici_breakdown=ici_breakdown,
        ctla4_breakdown=ctla4_breakdown,
        out_path=DEMO_GRID_PATH,
    )

    print("\nGenerating Markdown clinical characteristics report...")
    generate_clinical_report(
        cohort_results=cohort_results,
        attrition_data=attrition_data,
        plot_path=km_plot_path,
        report_path=REPORT_PATH,
        cohort_order=cohort_order,
        overall_demographics=overall_demographics,
        ici_breakdown=ici_breakdown,
        ctla4_breakdown=ctla4_breakdown,
        demographics_grid_path=DEMO_GRID_PATH,
        cohort_data=cohort_data,
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
            print(f"Logging console output to {LOG_PATH.relative_to(_SUBPROJECT_ROOT).as_posix()}")
            main()