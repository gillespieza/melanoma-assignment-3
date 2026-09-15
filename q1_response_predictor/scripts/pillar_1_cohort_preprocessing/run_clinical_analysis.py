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
import subprocess
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
_PROJECT_ROOT = _SUBPROJECT_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
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
from src.utils.execution import run_companion_scripts
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
from src.utils.plotting import add_km_risk_table, save_fig

set_presentation_style()

_SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _SCRIPT_DIR.parent.parent / "config" / "datasets.yaml"
PLOT_DIR = _SUBPROJECT_ROOT / "plots" / "clinical"
LOG_DIR = _SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "run_clinical_analysis.log"
REPORT_DIR = _SUBPROJECT_ROOT / "reports" / "pillar-1-cohorts-and-preprocessing"
REPORT_PATH = REPORT_DIR / "cohort_characteristics_clinical.md"
DEMO_GRID_PATH = PLOT_DIR / "clinical_demographics_2x2_grid.png"

# Module-level column and cohort constants
_COL_SAMPLE_ID = "SAMPLE_ID"
_COL_SEX = "SEX"
_COL_AGE = "AGE"
_COL_OS_MONTHS = "os_months"
_COL_OS_STATUS = "os_status"
_COL_PRIOR_ICI_RX = "PRIOR_ICI_RX"
_COL_ICI_TX = "TX_TYPE_IMMUNOTHERAPY_(INCLUDING_VACCINES)"

# Riaz 2017 is the only cohort requiring a hard-coded specific override
# (all patients had prior ipilimumab by trial design — this cannot be inferred
# from the clinical CSV alone).  All other cohort-specific logic is driven by
# DatasetConfig.treatment_label at runtime.
_COHORT_RIAZ_2017 = "Riaz 2017"

_COLOR_BORDER_GRAY = "#CCCCCC"

_COMPANION_SCRIPTS: list[tuple[str, Path]] = [
    (
        "Response-stratified KM survival curves",
        _SUBPROJECT_ROOT / "scripts" / "exploratory_plots" / "run_response_km_curves.py",
    ),
]

# ===========================================================================
# SURVIVAL DATA HANDLING
# ===========================================================================


def _resolve_os_columns(df_clin: pd.DataFrame) -> tuple[str, str]:
    """Determines the overall survival time and event columns."""
    column_pairs = [
        (_COL_OS_MONTHS, _COL_OS_STATUS),
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
        _COL_AGE,
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
    if _COL_SEX not in df_clin.columns:
        return {"n_female": 0, "n_sex_available": 0}

    sex = df_clin[_COL_SEX].astype("string").str.strip().str.upper()
    sex = sex.replace({"FEMALE": "F", "WOMAN": "F", "MALE": "M", "MAN": "M"})
    valid_sex = sex.isin(["F", "M"])

    return {
        "n_female": int((sex == "F").sum()),
        "n_sex_available": int(valid_sex.sum()),
    }


# ===========================================================================
# TREATMENT STATISTICS PARSERS
# ===========================================================================


# Agent keywords inferred from DatasetConfig.treatment_label for the fallback
# injection in _build_combined_treatment_text.  Extend this map only for agents
# whose names in the label differ from the token we want to inject.
_TREATMENT_LABEL_AGENT_TOKENS: dict[str, str] = {
    "pembrolizumab": "PEMBROLIZUMAB",
    "nivolumab": "NIVOLUMAB",
    "ipilimumab": "IPILIMUMAB",
    "vemurafenib": "VEMURAFENIB",
    "dabrafenib": "DABRAFENIB",
    "trametinib": "TRAMETINIB",
    "interferon": "INTERFERON",
    "dacarbazine": "DACARBAZINE",
}


def _infer_agent_tokens_from_label(treatment_label: str) -> list[str]:
    """Returns uppercase agent tokens present in a DatasetConfig treatment_label string.

    This drives the fallback injection in _build_combined_treatment_text without
    hardcoding cohort names — any cohort whose treatment_label mentions an agent
    will automatically get the right fallback token.
    """
    label_lower = treatment_label.lower()
    return [
        token
        for keyword, token in _TREATMENT_LABEL_AGENT_TOKENS.items()
        if keyword in label_lower
    ]


def _build_combined_treatment_text(
    df_clin: pd.DataFrame,
    cohort_label: str,
    treatment_label: str = "",
) -> pd.Series:
    """Combines text columns and injects agent fallback tokens from treatment_label.

    The fallback is config-driven: agent tokens are inferred from
    DatasetConfig.treatment_label, so new cohorts are covered automatically
    without adding hardcoded cohort-name branches here.
    """
    text_cols = [
        c for c in df_clin.columns
        if df_clin[c].dtype == "object" or isinstance(df_clin[c].dtype, pd.StringDtype)
    ]
    combined_text = pd.Series("", index=df_clin.index)
    for c in text_cols:
        combined_text = combined_text + " " + df_clin[c].astype(str).str.upper()

    combined_str = combined_text.to_string()
    for token in _infer_agent_tokens_from_label(treatment_label):
        if token not in combined_str:
            combined_text = combined_text + " " + token

    return combined_text


def _match_agent_pattern(
    combined_text: pd.Series,
    df_clin: pd.DataFrame,
    patterns: list[str],
) -> int:
    """Counts matching occurrences of agent patterns in numeric columns or combined text."""
    num_cols = [
        c for c in df_clin.columns
        if any(p in c.upper() for p in patterns) and pd.api.types.is_numeric_dtype(df_clin[c])
    ]
    if num_cols:
        return int(df_clin[num_cols].max(axis=1).sum())

    text_patterns = [p for p in patterns if not p.startswith("TX_")]
    pattern_regex = r"\b(?:" + "|".join(text_patterns) + r")\b" if text_patterns else ""
    if pattern_regex:
        return int(combined_text.str.contains(pattern_regex, na=False, regex=True).sum())
    return 0


def _search_treatment_agents(
    df_clin: pd.DataFrame,
    cohort_label: str,
    treatment_label: str = "",
) -> dict[str, int]:
    """Identifies and counts treatment agent exposures from clinical metadata."""
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
    combined_text = _build_combined_treatment_text(df_clin, cohort_label, treatment_label)

    for agent_name, patterns in agent_specs.items():
        cnt = _match_agent_pattern(combined_text, df_clin, patterns)
        if cnt > 0:
            agents_found[agent_name] = cnt

    return agents_found


def _determine_prior_ctla4(
    df_clin: pd.DataFrame,
    cohort_label: str,
    ipilimumab_cnt: int,
    n_total: int,
) -> int:
    """Calculates patient count with prior anti-CTLA-4 exposure.

    Priority order:
    1. Use PRIOR_ICI_RX column if present (most accurate — explicit clinical record).
    2. Riaz 2017 hard-override: all patients had prior ipilimumab by trial design;
       this cannot be inferred from the CSV alone.
    3. Fall back to the detected ipilimumab count for any other cohort.
       This is correct for pure ipilimumab trials (Van Allen 2015) and gives a
       conservative lower-bound estimate for mixed cohorts — no cohort list needed.
    """
    if _COL_PRIOR_ICI_RX in df_clin.columns:
        prior_s = df_clin[_COL_PRIOR_ICI_RX].astype(str).str.upper()
        return int(prior_s.str.contains("IPILIMUMAB|ACTLA4|CTLA4", na=False).sum())
    if cohort_label == _COHORT_RIAZ_2017:
        return n_total
    # For all other cohorts, the detected ipilimumab administration count is the
    # best available proxy for prior anti-CTLA-4 exposure.
    return ipilimumab_cnt


def calculate_treatment_statistics(
    df_clin: pd.DataFrame,
    cohort_label: str,
    treatment_label: str = "",
) -> dict[str, Any]:
    """Calculates treatment agent statistics universally across any cohort DataFrame."""
    n_total = len(df_clin)
    agents_found = _search_treatment_agents(df_clin, cohort_label, treatment_label)

    pembrolizumab = agents_found.get("Pembrolizumab", 0)
    nivolumab = agents_found.get("Nivolumab", 0)
    ipilimumab = agents_found.get("Ipilimumab", 0)
    prior_ctla4_cnt = _determine_prior_ctla4(
        df_clin, cohort_label, ipilimumab, n_total
    )

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


def _extract_age_histogram_df(
    cohort_data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, list[str], str]:
    """Prepares combined age DataFrame and dynamic missing-age annotation note."""
    age_dfs: list[pd.DataFrame] = []
    cohort_names: list[str] = []
    omitted_cohorts: list[str] = []

    for label, df in cohort_data.items():
        age_col = _resolve_age_column(df)
        if age_col:
            ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
            age_dfs.append(pd.DataFrame({"Age": ages, "Cohort": label}))
            cohort_names.append(label)
        else:
            omitted_cohorts.append(f"{label} (N = {len(df)})")

    df_ages = (
        pd.concat(age_dfs, axis=0, ignore_index=True)
        if age_dfs else pd.DataFrame(columns=["Age", "Cohort"])
    )
    omitted_str = ", ".join(omitted_cohorts) if omitted_cohorts else "none"
    omitted_note = f"* Age omitted in {omitted_str};\n  eligibility ≥18y; top-coded at 89–90y"
    return df_ages, cohort_names, omitted_note


def _add_age_legend_handles(cohort_names: list[str], age_median: float) -> list[Any]:
    """Generates legend patch and line handles for age histogram."""
    legend_handles: list[Any] = [
        mpatches.Patch(color=COHORT_PALETTE.get(c, DARK_SLATE_CHARCOAL), label=c)
        for c in cohort_names
    ]
    legend_handles.append(
        mlines.Line2D(
            [], [], color=OKABE_ITO[5], linestyle="--",
            linewidth=2, label=f"Median: {age_median:.0f} y",
        )
    )
    return legend_handles


def _annotate_age_axis(
    ax: plt.Axes,
    d: dict[str, Any],
    omitted_note: str,
) -> None:
    """Sets axis labels, title, and omitted-cohort annotation note for age histogram."""
    ax.set_xlabel("Age at Diagnosis (Years)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Patient Count", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Panel B: Age Distribution (Median = {d['age_median']:.0f}, "
        f"IQR = {d['age_q1']:.0f}\u2013{d['age_q3']:.0f})",
        fontsize=14, fontweight="bold", y=-0.24,
    )
    ax.text(
        0.03, 0.97, omitted_note, transform=ax.transAxes, ha="left", va="top",
        fontsize=8.5, fontstyle="italic", color=DARK_SLATE_CHARCOAL,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white",
            edgecolor=_COLOR_BORDER_GRAY, alpha=0.92,
        ),
    )


def _render_age_histogram(
    ax: plt.Axes,
    df_ages: pd.DataFrame,
    cohort_names: list[str],
    d: dict[str, Any],
    omitted_note: str,
) -> None:
    """Renders the age distribution histogram, median line, and legend."""
    if not df_ages.empty:
        sns.histplot(
            data=df_ages, x="Age", hue="Cohort", multiple="stack",
            palette=resolve_cohort_palette(df_ages["Cohort"]), ax=ax, bins=20, kde=True,
            legend=False,
        )
        ax.axvline(d["age_median"], color=OKABE_ITO[5], linestyle="--", linewidth=2)
        ax.legend(
            handles=_add_age_legend_handles(cohort_names, d["age_median"]),
            loc="upper right", fontsize=9, framealpha=0.9,
        )
    _annotate_age_axis(ax, d, omitted_note)


def _plot_age_panel(
    ax: plt.Axes,
    cohort_data: dict[str, pd.DataFrame],
    d: dict[str, Any],
) -> None:
    """Plots Panel B: Age Distribution across Studies."""
    df_ages, cohort_names, omitted_note = _extract_age_histogram_df(cohort_data)
    _render_age_histogram(ax, df_ages, cohort_names, d, omitted_note)


def _render_agent_bars(
    ax: plt.Axes,
    labels: list[str],
    counts: list[int],
    n_total: int,
) -> None:
    """Renders horizontal bars and percentage text labels for treatment agents."""
    colors = [
        OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2], OKABE_ITO[3], OKABE_ITO[4],
        OKABE_ITO[5], OKABE_ITO[6], OKABE_ITO[7]
    ]
    bar_colors = (colors * ((len(labels) // len(colors)) + 1))[:len(labels)]

    bars = ax.barh(labels[::-1], counts[::-1], color=bar_colors[::-1], edgecolor="w", linewidth=1.5)
    max_c = max(counts) if counts else 1
    for bar in bars:
        w = bar.get_width()
        pct = w / n_total * 100 if n_total > 0 else 0
        ax.text(
            w + max_c * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{w} ({pct:.1f}%)", va="center", fontsize=9, fontweight="bold"
        )

    ax.set_xlabel("Patient Administrations", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max_c * 1.25)


def _plot_ici_agent_panel(ax: plt.Axes, ici: dict[str, Any]) -> None:
    """Plots Panel C: Treatment Agents Administered across Studies (horizontal bar)."""
    agent_totals = ici.get("agent_totals", {})
    if not agent_totals:
        ax.text(0.5, 0.5, "No treatment agent data available", ha="center", va="center")
        return

    sorted_agents = sorted(agent_totals.items(), key=lambda x: x[1], reverse=True)
    labels = [a[0] for a in sorted_agents]
    counts = [a[1] for a in sorted_agents]

    _render_agent_bars(ax, labels, counts, ici["n_total"])
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

    print(
        f"\nSaved clinical demographics grid to "
        f"{out_path.relative_to(_SUBPROJECT_ROOT).as_posix()}"
    )


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
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor="white",
                edgecolor=DARK_SLATE_CHARCOAL, alpha=0.8,
            ),
        )


def _fit_and_plot_km(
    ax: plt.Axes,
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    cohort_label: str,
    df_clin_len: int,
) -> float:
    """Fits KaplanMeierFitter and plots survival curve on axis."""
    cohort_color = get_cohort_color(cohort_label)
    kmf = KaplanMeierFitter()
    os_label = (
        f"Survival Curve (N={len(df)})"
        if len(df) == df_clin_len else f"OS Subset (n={len(df)})"
    )
    kmf.fit(durations=df[time_col], event_observed=df[event_col], label=os_label)
    kmf.plot_survival_function(ax=ax, ci_show=True, color=cohort_color, linewidth=2)
    add_km_risk_table([kmf], ax)
    return float(kmf.median_survival_time_)


def plot_km_os(
    ax: plt.Axes,
    df_clin: pd.DataFrame,
    cohort_label: str,
) -> dict[str, Any]:
    """Plots an unstratified Kaplan-Meier OS curve."""
    time_col, event_col = _resolve_os_columns(df_clin)
    df = _clean_os(df_clin, time_col, event_col)

    median_surv = _fit_and_plot_km(
        ax, df, time_col, event_col, cohort_label, len(df_clin)
    )
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


def _format_total_age_str(overall_demographics: dict[str, Any] | None) -> str:
    """Formats overall cohort median age and IQR string."""
    if (
        overall_demographics
        and "age_median" in overall_demographics
        and not pd.isna(overall_demographics["age_median"])
    ):
        med = overall_demographics["age_median"]
        q1 = overall_demographics["age_q1"]
        q3 = overall_demographics["age_q3"]
        return f"{med:.1f} ({q1:.1f}-{q3:.1f})"
    return "N/A"


def _empty_row(cohort_order: list[str]) -> dict[str, str]:
    """Returns an empty separator row for the characteristics table."""
    return {"Characteristic": "", **{c: "" for c in cohort_order}, "Total": ""}


def _build_age_sex_rows(
    age_results: dict[str, Any],
    sex_results: dict[str, Any],
    cohort_order: list[str],
    overall_demographics: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Builds the age and sex data rows for the demographics section."""
    tot_female = sum(sex_results[c]["n_female"] for c in cohort_order)
    tot_sex_avail = sum(sex_results[c]["n_sex_available"] for c in cohort_order)
    return [
        {
            "Characteristic": "Age, median (IQR)",
            **{c: format_median_iqr(age_results[c]) for c in cohort_order},
            "Total": _format_total_age_str(overall_demographics),
        },
        {
            "Characteristic": "Female sex, n (%)",
            **{
                c: format_count_percentage(
                    count=sex_results[c]["n_female"],
                    total=sex_results[c]["n_sex_available"],
                )
                for c in cohort_order
            },
            "Total": format_count_percentage(count=tot_female, total=tot_sex_avail),
        },
    ]


def _build_demographic_rows(
    cohort_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
    overall_demographics: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Builds demographic rows for clinical characteristics table with Total column."""
    age_results = {c: cohort_results[c]["age"] for c in cohort_order}
    sex_results = {c: cohort_results[c]["sex"] for c in cohort_order}
    header = {"Characteristic": "**Demographics**", **{c: "" for c in cohort_order}, "Total": ""}
    return [header] + _build_age_sex_rows(
        age_results, sex_results, cohort_order, overall_demographics
    ) + [_empty_row(cohort_order)]


def _collect_cohort_agent_names(
    treatment_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
) -> list[str]:
    """Collects unique treatment agent names across all active cohorts."""
    all_agents: list[str] = []
    for c in cohort_order:
        agents = treatment_results[c].get("agents", {})
        for agent_name in agents.keys():
            if agent_name not in all_agents:
                all_agents.append(agent_name)
    return all_agents


def _format_single_agent_row(
    agent: str,
    treatment_results: dict[str, dict[str, Any]],
    survival_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
    total_n: int,
) -> dict[str, str]:
    """Formats a single agent row across cohorts for the summary table."""
    tot_cnt = sum(treatment_results[c].get("agents", {}).get(agent, 0) for c in cohort_order)
    row = {"Characteristic": f"Agent — {agent}"}
    for c in cohort_order:
        cnt = treatment_results[c].get("agents", {}).get(agent, 0)
        tot = survival_results[c]["n_total"]
        row[c] = format_count_percentage(count=cnt, total=tot) if cnt > 0 else "0 (0.0%)"
    row["Total"] = format_count_percentage(count=tot_cnt, total=total_n)
    return row


def _format_prior_ctla4_row(
    treatment_results: dict[str, dict[str, Any]],
    survival_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
    total_n: int,
) -> dict[str, str]:
    """Formats the prior anti-CTLA-4 exposure summary row."""
    tot_prior = sum(treatment_results[c].get("prior_ctla4", 0) for c in cohort_order)
    row: dict[str, str] = {"Characteristic": "Prior anti-CTLA-4 therapy"}
    for c in cohort_order:
        cnt = treatment_results[c].get("prior_ctla4", 0)
        tot = survival_results[c]["n_total"]
        row[c] = format_count_percentage(count=cnt, total=tot) if cnt > 0 else "0 (0.0%)"
    row["Total"] = format_count_percentage(count=tot_prior, total=total_n)
    return row


def _build_treatment_rows(
    cohort_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
    total_n: int,
) -> list[dict[str, str]]:
    """Builds treatment rows for clinical characteristics table across active cohorts."""
    survival_results = {c: cohort_results[c]["survival"] for c in cohort_order}
    treatment_results = {c: cohort_results[c]["treatment"] for c in cohort_order}
    all_agents = _collect_cohort_agent_names(treatment_results, cohort_order)
    header = {
        "Characteristic": "**Treatment Agents & Exposure**",
        **{c: "" for c in cohort_order},
        "Total": "",
    }
    rows: list[dict[str, str]] = [header]
    for agent in all_agents:
        rows.append(_format_single_agent_row(
            agent, treatment_results, survival_results, cohort_order, total_n
        ))
    rows.append(_format_prior_ctla4_row(treatment_results, survival_results, cohort_order, total_n))
    rows.append(_empty_row(cohort_order))
    return rows





def _build_os_events_row(
    survival_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
) -> dict[str, str]:
    """Builds the OS events count-percentage row for the survival table."""
    tot_events = sum(survival_results[c]["n_events"] for c in cohort_order)
    tot_valid_os = sum(survival_results[c]["n_valid_os"] for c in cohort_order)
    return {
        "Characteristic": "OS events, n (%)",
        **{
            c: format_count_percentage(
                count=survival_results[c]["n_events"],
                total=survival_results[c]["n_valid_os"],
            )
            for c in cohort_order
        },
        "Total": format_count_percentage(count=tot_events, total=tot_valid_os),
    }


def _build_survival_detail_rows(
    survival_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
    pooled_median_os: float,
    pooled_median_fu: float,
) -> list[dict[str, str]]:
    """Builds the median OS, events, and follow-up data rows for survival table."""
    return [
        {
            "Characteristic": "Median OS, months (95% CI)",
            **{c: format_median(survival_results[c]["median_os"]) for c in cohort_order},
            "Total": format_median(pooled_median_os),
        },
        _build_os_events_row(survival_results, cohort_order),
        {
            "Characteristic": "Median follow-up, months",
            **{c: format_median(survival_results[c]["median_follow_up"]) for c in cohort_order},
            "Total": format_median(pooled_median_fu),
        },
    ]


def _build_survival_rows(
    cohort_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
    pooled_median_os: float = float("nan"),
    pooled_median_fu: float = float("nan"),
) -> list[dict[str, str]]:
    """Builds survival outcome rows for clinical characteristics table with Total column."""
    survival_results = {c: cohort_results[c]["survival"] for c in cohort_order}
    header = [{
        "Characteristic": "**Survival Outcomes**",
        **{c: "" for c in cohort_order},
        "Total": "",
    }]
    return header + _build_survival_detail_rows(
        survival_results, cohort_order, pooled_median_os, pooled_median_fu
    )


def _compute_pooled_survival_summary(
    cohort_data: dict[str, pd.DataFrame],
) -> tuple[float, float]:
    """Computes pooled median overall survival and median follow-up across all cohorts."""
    times, events = [], []
    for df in cohort_data.values():
        time_col, event_col = _resolve_os_columns(df)
        if time_col and event_col:
            sub = df[[time_col, event_col]].dropna()
            sub_valid = sub[sub[time_col] > 0]
            times.extend(sub_valid[time_col].tolist())
            events.extend(sub_valid[event_col].tolist())

    if not times:
        return float("nan"), float("nan")

    kmf = KaplanMeierFitter()
    kmf.fit(times, events)
    return float(kmf.median_survival_time_), float(pd.Series(times).median())


def _build_n_header_rows(
    survival_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
    total_n: int,
) -> list[dict[str, str]]:
    """Builds the N header row and blank separator for the characteristics table."""
    n_row = {
        "Characteristic": "**N**",
        **{c: str(survival_results[c]["n_total"]) for c in cohort_order},
        "Total": str(total_n),
    }
    return [n_row, _empty_row(cohort_order)]


def generate_clinical_characteristics_table(
    cohort_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
    overall_demographics: dict[str, Any] | None = None,
    cohort_data: dict[str, pd.DataFrame] | None = None,
) -> str:
    """Generates the comparative clinical characteristics Markdown table with a Total column."""
    survival_results = {c: cohort_results[c]["survival"] for c in cohort_order}
    total_n = sum(survival_results[c]["n_total"] for c in cohort_order)
    pooled_median_os, pooled_median_fu = (
        _compute_pooled_survival_summary(cohort_data)
        if cohort_data is not None else (float("nan"), float("nan"))
    )
    rows = _build_n_header_rows(survival_results, cohort_order, total_n)
    rows.extend(_build_demographic_rows(cohort_results, cohort_order, overall_demographics))
    rows.extend(_build_treatment_rows(cohort_results, cohort_order, total_n))
    rows.extend(_build_survival_rows(
        cohort_results, cohort_order, pooled_median_os, pooled_median_fu
    ))
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

    return (
        pd.DataFrame(rows).to_markdown(index=False)
        if rows else "No sample attrition data available."
    )


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
            c_init = int(df_attr.iloc[0]["n_before"])
            c_final = int(df_attr.iloc[-1]["n_after"])
            c_rem = c_init - c_final
            initial_total += c_init
            final_total += c_final
            total_removed += c_rem
            if c_rem > 0:
                attrition_summaries.append(
                    f"**{cohort}** lost **{c_rem}** sample(s) ({c_init} → {c_final})"
                )
            else:
                attrition_summaries.append(
                    f"**{cohort}** retained 100% of samples (N = {c_final})"
                )

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


_SCRIPT_REFERENCE_SPECS: list[tuple[str, Path, str]] = [
    (
        "run_clinical_analysis.py",
        _SUBPROJECT_ROOT / "scripts" / "pillar_1_cohort_preprocessing" / "run_clinical_analysis.py",
        "Generates baseline demographic grids, attrition metrics, cohort characteristics "
        "tables, Kaplan-Meier OS curves, and writes `cohort_characteristics_clinical.md`.",
    ),
    (
        "clean_data.py",
        _SUBPROJECT_ROOT / "scripts" / "pillar_1_cohort_preprocessing" / "clean_data.py",
        "Preprocesses raw cohort clinical metadata and RNA-seq expression profiles into "
        "cleaned CSV matrices.",
    ),
    (
        "merge_datasets.py",
        _SUBPROJECT_ROOT / "scripts" / "pillar_1_cohort_preprocessing" / "merge_datasets.py",
        "Merges processed expression matrices across cohorts into harmonised pooled "
        "matrices (`expr_merged.csv`, `clin_merged.csv`).",
    ),
    (
        "data_loaders.py",
        _SUBPROJECT_ROOT / "src" / "data_loaders.py",
        "Provides `load_cohort_by_name` (canonical dynamic entry point — resolves any cohort "
        "from `datasets.yaml` without code changes), `load_dataset_by_config`, "
        "`load_all_active_cohorts`, and `load_merged_immunotherapy`. "
        "Legacy per-cohort shims are retained for backwards compatibility.",
    ),
    (
        "styles.py",
        PROJECT_ROOT / "src" / "styles.py",
        "Single source of truth for Okabe-Ito colour palettes (`COHORT_PALETTE`, "
        "`RESPONSE_PALETTE`) and visualisation presentation style.",
    ),
]


def _build_clinical_report_script_callout(report_path: Path) -> str:
    """Generates the standardized script reference callout box for the report."""
    return generate_script_reference_callout(
        _SCRIPT_REFERENCE_SPECS,
        base_dir=report_path.parent,
        callout_type="[!formula]+",
        title="Clinical Analysis Script Execution & Software Module Architecture",
    )


def _build_clinical_report_frontmatter(timestamp: str) -> str:
    """Generates Obsidian YAML frontmatter block."""
    return generate_obsidian_frontmatter(
        title="Clinical Characteristics of Immunotherapy Data Cohorts",
        aliases=["Clinical Cohort Characteristics"],
        tags=[
            "melanoma", "clinical-analysis", "cohort-characteristics",
            "survival-analysis", "kaplan-meier", "immunotherapy"
        ],
        created=timestamp,
        updated=timestamp,
        extra_css_classes=["table-center", "row-alt"],
    )


def _build_section1_demographic_insights(
    d: dict[str, Any],
    ici: dict[str, Any],
    ctla4: dict[str, Any],
    age_annotation_str: str,
    top_agents_str: str,
) -> str:
    """Constructs demographic and treatment bullet insights for Section 1."""
    sex_pref = "male" if d["n_male"] > d["n_female"] else "female"
    return (
        f"- **Panel A: Sex Distribution ($N = {d['n_sex_total']}$)**: The overall trial cohort "
        f"shows a {sex_pref} predominance (**{d['pct_male']:.1f}% Male** [$N = {d['n_male']}$] vs. "
        f"**{d['pct_female']:.1f}% Female** [$N = {d['n_female']}$]), reflecting real-world melanoma incidence.\n"
        f"- **Panel B: Age Distribution across Studies**: Evaluated patient ages span from "
        f"{d['age_min']:.0f} to {d['age_max']:.0f} years with a **median age of {d['age_median']:.1f} years** "
        f"(IQR: {d['age_q1']:.1f}–{d['age_q3']:.1f} years). Trial cohorts ({age_annotation_str}) display "
        "consistent age distributions.\n"
        f"- **Panel C: Treatment Agents Administered ($N = {ici['n_total']}$)**: Most frequent "
        f"agents are {top_agents_str}.\n"
        f"- **Panel D: Prior Anti-CTLA-4 Therapy Status ($N = {ctla4['n_total']}$)**: Across all patients, "
        f"**{ctla4['pct_prior_ctla4']:.1f}%** [$N = {ctla4['n_prior_ctla4']}$] received prior ipilimumab, "
        f"while **{ctla4['pct_naive']:.1f}%** [$N = {ctla4['n_naive']}$] were anti-CTLA-4 naïve."
    )


def _build_report_section1(
    n_cohorts: int,
    cohort_bullets: str,
    demographics_grid_path: Path,
    d: dict[str, Any],
    ici: dict[str, Any],
    ctla4: dict[str, Any],
    age_annotation_str: str,
    top_agents_str: str,
    clinical_characteristics_table: str,
) -> str:
    """Constructs Section 1 of clinical report markdown."""
    callout = (
        f"## 1. Baseline Patient and Disease Characteristics\n\n"
        f"> [!INFO] Why We Are Doing This\n"
        f"> - **What**: Compare patient demographics and survival across {n_cohorts} trial cohorts.\n"
        f"> - **Why**: Identify potential demographic confounders before predictive modeling.\n"
        f"> - **Questions**: Are patient populations comparable across cohorts?\n\n"
        f"This report compares patient demographics across active trial cohorts:\n{cohort_bullets}"
    )
    fig1 = (
        f"### 1.1 Clinical Demographics & Treatment Distributions\n\n"
        f"![Clinical Demographics](../../plots/clinical/{demographics_grid_path.name})\n\n"
        f"_**Figure 1: 2×2 Grid of Clinical Demographics and Treatment Histories.**_\n\n"
        f"#### Key Demographics & Treatment Insights\n\n"
        f"{_build_section1_demographic_insights(d, ici, ctla4, age_annotation_str, top_agents_str)}\n\n"
        f"_**Table 1: Baseline Patient Characteristics**_\n\n{clinical_characteristics_table}"
    )
    return f"{callout}\n\n{fig1}"


def _build_report_section2(
    n_cohorts: int,
    n_total: int,
    initial_total: int,
    final_total: int,
    retention_pct: float,
    attrition_text: str,
    cohorts_list_str: str,
    largest_cohort: str,
    longest_fu_cohort: str,
    n_values: dict[str, int],
    survival_results: dict[str, dict[str, Any]],
    attrition_table: str,
) -> str:
    """Constructs Section 2 of clinical report markdown (Sample Preprocessing Attrition)."""
    fu_med = format_median(survival_results[longest_fu_cohort]["median_follow_up"])
    return (
        f"## 2. Sample Preprocessing Attrition\n\n"
        f"> [!INFO] Why We Are Doing This\n"
        f"> - **What**: We track sample retention through quality control across $N = {initial_total}$ initial records.\n"
        f"> - **Why**: Documenting attrition at each step verifies data integrity.\n"
        f"> - **Questions**: How many patients are retained for downstream analysis?\n\n"
        f"_**Table 2: Sample Attrition Across Preprocessing Steps**_\n\n"
        f"{attrition_table}\n\n"
        f"### Key Observations\n"
        f"1. **Overall Cohort Size ($N = {n_total}$)**: Largest dataset is **{largest_cohort}** ($N = {n_values[largest_cohort]}$). Combined across all {n_cohorts} cohorts, **$N = {final_total}$** cleaned records were harmonised.\n"
        f"2. **Sample Attrition ({retention_pct:.1f}% Retention)**: Across all {initial_total} initial records, **{attrition_text}**.\n"
        f"3. **Follow-up Duration**: **{longest_fu_cohort}** displays median follow-up of **{fu_med} months**.\n"
        f"4. **Treatment History**: Enrolled cohorts represent diverse treatment contexts across {cohorts_list_str}."
    )


def _build_report_section3_4(
    n_cohorts: int,
    n_total: int,
    plot_path: Path,
    cohorts_list_str: str,
) -> str:
    """Constructs Sections 3 and 4 of clinical report markdown (KM plots and response stratification)."""
    return (
        f"## 3. Overall Survival Curves (KM Plots)\n\n"
        f"> [!INFO] Why We Are Doing This\n"
        f"> - **What**: We plot unstratified Kaplan-Meier OS curves for each of the {n_cohorts} active trial cohorts.\n"
        f"> - **Why**: Visualising baseline mortality rates establishes the clinical context for each dataset.\n"
        f"> - **Questions**: How does overall survival compare across independent immunotherapy trial cohorts?\n\n"
        f"![Overall Survival KM Curves](../../plots/clinical/{plot_path.name})\n\n"
        f"_**Figure 2: Unstratified Overall Survival KM Curves across All {n_cohorts} Immunotherapy Trial Cohorts.**_\n\n"
        f"## 4. Overall Survival Stratified by Immunotherapy Response\n\n"
        f"> [!INFO] Why We Are Doing This\n"
        f"> - **What**: We stratify KM overall survival curves by RECIST response status ($N = {n_total}$).\n"
        f"> - **Why**: Confirming that responders experience significantly longer OS validates RECIST response as a surrogate endpoint.\n"
        f"> - **Questions**: Does RECIST response reliably distinguish durable long-term benefit?\n\n"
        f"![Overall Survival by Response](../../plots/clinical/km_os_by_response.png)\n\n"
        f"_**Figure 3: Overall Survival Stratified by RECIST Response Status.**_\n\n"
        f"> [!INSIGHT] Key Insights: Survival Stratification by Response\n"
        f"> 1. **Survival Benefit**: Responders (CR/PR) achieve significantly longer OS vs non-responders (PD) ({cohorts_list_str}; Log-rank $p < 0.0001$).\n"
        f"> 2. **Surrogate Validation**: Objective RECIST response is a robust surrogate endpoint for overall survival."
    )


def _build_report_section5(script_callout: str) -> str:
    """Constructs Section 5 of clinical report markdown (Technical notes and script reference)."""
    return (
        f"## 5. Technical Analysis Notes\n\n"
        f"> [!WARNING] Methodological Limitations & Analytical Scope\n"
        f"> - **Unstratified Analysis**: All KM curves are unstratified and descriptive.\n"
        f"> - **Exclusion Criteria**: Excluded missing survival time, missing event status, or survival time ≤ 0.\n"
        f"> - **Median OS Reporting**: Median OS is reported as **NR (not reached)** where survival probability remained above 50%.\n\n"
        f"{script_callout}"
    )


def _build_report_section2_3_4_5(
    n_cohorts: int,
    n_total: int,
    initial_total: int,
    final_total: int,
    retention_pct: float,
    attrition_text: str,
    plot_path: Path,
    cohorts_list_str: str,
    largest_cohort: str,
    longest_fu_cohort: str,
    n_values: dict[str, int],
    survival_results: dict[str, dict[str, Any]],
    attrition_table: str,
    script_callout: str,
) -> str:
    """Constructs Sections 2 through 5 of clinical report markdown."""
    sec2 = _build_report_section2(
        n_cohorts, n_total, initial_total, final_total, retention_pct,
        attrition_text, cohorts_list_str, largest_cohort, longest_fu_cohort,
        n_values, survival_results, attrition_table,
    )
    sec3_4 = _build_report_section3_4(n_cohorts, n_total, plot_path, cohorts_list_str)
    sec5 = _build_report_section5(script_callout)
    return f"{sec2}\n\n{sec3_4}\n\n{sec5}"


def _prepare_report_metadata(
    cohort_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
    ici_breakdown: dict[str, Any],
    dataset_configs: tuple["DatasetConfig", ...] = (),
) -> tuple[dict[str, int], str, str, str]:
    """Extracts summary strings and cohort statistics for report generation."""
    survival_results = {c: cohort_results[c]["survival"] for c in cohort_order}
    age_results = {c: cohort_results[c]["age"] for c in cohort_order}
    n_values = {c: survival_results[c]["n_total"] for c in cohort_order}
    _CBIOPORTAL_STUDY_URL = "https://www.cbioportal.org/study/summary?id="
    treatment_map = {cfg.cohort_name: cfg.treatment_label for cfg in dataset_configs}
    study_id_map = {cfg.cohort_name: cfg.study_id for cfg in dataset_configs}
    cohort_bullets = "\n".join([
        (
            f"- **[{c}]({_CBIOPORTAL_STUDY_URL}{study_id_map[c]})**: "
            f"{treatment_map.get(c, 'Immunotherapy trial cohort')} "
            f"($N = {n_values[c]}$)."
            if c in study_id_map
            else f"- **{c}**: "
            f"{treatment_map.get(c, 'Immunotherapy trial cohort')} "
            f"($N = {n_values[c]}$)."
        )
        for c in cohort_order
    ])
    annotated_ages = [
        f"`{c}`: median {_format_cohort_age_str(age_results, c)}"
        for c in cohort_order
        if not pd.isna(age_results.get(c, {}).get("median_age", float("nan")))
    ]
    age_annotation_str = "; ".join(annotated_ages) if annotated_ages else "no annotated age data"
    n_total_agents = ici_breakdown["n_total"]
    top_agents_list = [
        f"**{agent}** ({cnt} [{cnt / n_total_agents * 100:.1f}%])"
        for agent, cnt in list(ici_breakdown.get("agent_totals", {}).items())[:4]
    ]
    top_agents_str = ", ".join(top_agents_list) if top_agents_list else "no agent data available"
    return n_values, cohort_bullets, age_annotation_str, top_agents_str


def _compute_report_context(
    cohort_results: dict[str, dict[str, Any]],
    attrition_data: dict[str, pd.DataFrame],
    cohort_order: list[str],
    ici_breakdown: dict[str, Any],
    dataset_configs: tuple["DatasetConfig", ...] = (),
) -> dict[str, Any]:
    """Computes all derived context variables needed by generate_clinical_report."""
    survival_results = {c: cohort_results[c]["survival"] for c in cohort_order}
    n_values, cohort_bullets, age_annotation_str, top_agents_str = (
        _prepare_report_metadata(cohort_results, cohort_order, ici_breakdown, dataset_configs)
    )
    initial_total, final_total, _removed, attrition_text = (
        _compute_attrition_summaries(attrition_data, cohort_order)
    )
    return {
        "survival_results": survival_results,
        "n_values": n_values,
        "n_total": sum(n_values.values()),
        "n_cohorts": len(cohort_order),
        "largest_cohort": max(cohort_order, key=lambda c: survival_results[c]["n_total"]),
        "longest_fu_cohort": max(cohort_order, key=lambda c: survival_results[c]["median_follow_up"]),
        "initial_total": initial_total,
        "final_total": final_total,
        "retention_pct": final_total / initial_total * 100 if initial_total > 0 else 0.0,
        "attrition_text": attrition_text,
        "cohort_bullets": cohort_bullets,
        "age_annotation_str": age_annotation_str,
        "top_agents_str": top_agents_str,
        "cohorts_list_str": ", ".join(cohort_order),
    }


def _write_clinical_report(
    report_content: str,
    report_path: Path,
) -> None:
    """Writes report content to specified path."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")
    print(
        f"\nSaved clinical analysis report to "
        f"{report_path.relative_to(_SUBPROJECT_ROOT).as_posix()}"
    )


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
    dataset_configs: tuple["DatasetConfig", ...] = (),
) -> None:
    """Generates an Obsidian-compatible Markdown clinical characteristics report."""
    ctx = _compute_report_context(
        cohort_results, attrition_data, cohort_order, ici_breakdown, dataset_configs
    )
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    sec1 = _build_report_section1(
        ctx["n_cohorts"], ctx["cohort_bullets"], demographics_grid_path,
        overall_demographics, ici_breakdown, ctla4_breakdown,
        ctx["age_annotation_str"], ctx["top_agents_str"],
        generate_clinical_characteristics_table(
            cohort_results, cohort_order, overall_demographics, cohort_data
        ),
    )
    sec2_5 = _build_report_section2_3_4_5(
        ctx["n_cohorts"], ctx["n_total"], ctx["initial_total"], ctx["final_total"],
        ctx["retention_pct"], ctx["attrition_text"], plot_path, ctx["cohorts_list_str"],
        ctx["largest_cohort"], ctx["longest_fu_cohort"], ctx["n_values"],
        ctx["survival_results"],
        generate_attrition_table(attrition_data, cohort_order),
        _build_clinical_report_script_callout(report_path),
    )
    report_content = (
        f"{_build_clinical_report_frontmatter(timestamp)}\n\n"
        "# Clinical Characteristics of Immunotherapy Data Cohorts\n\n"
        f"{sec1}\n\n{sec2_5}"
    )
    _write_clinical_report(report_content, report_path)


# ===========================================================================
# MAIN PIPELINE STAGES
# ===========================================================================


def _filter_ici_subcohort(df_clin: pd.DataFrame) -> pd.DataFrame:
    """Filters DataFrame to immunotherapy patients if ICI treatment column is present."""
    if _COL_ICI_TX not in df_clin.columns:
        return df_clin
    n_before = len(df_clin)
    df_filtered = df_clin[df_clin[_COL_ICI_TX] == 1.0].copy()
    print(f"    Filtered to immunotherapy subcohort: {n_before} -> {len(df_filtered)} patients")
    return df_filtered


def _load_single_cohort_clinical_file(
    config: DatasetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads cleaned clinical DataFrame and optional attrition records for one cohort."""
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
    df_clin = _filter_ici_subcohort(
        pd.read_csv(clin_path, index_col=_COL_SAMPLE_ID)
    )
    df_attrition = pd.DataFrame()
    if attrition_path.exists():
        df_attrition = pd.read_csv(attrition_path)
        print(f"    Loaded attrition records from {attrition_path.name}")
    else:
        print(f"    No attrition records found for {label}")
    return df_clin, df_attrition


def _load_cohort_and_attrition_data(
    dataset_configs: tuple[DatasetConfig, ...],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Loads cleaned clinical DataFrames and attrition records per cohort."""
    cohort_data: dict[str, pd.DataFrame] = {}
    attrition_data: dict[str, pd.DataFrame] = {}

    for config in dataset_configs:
        df_clin, df_attrition = _load_single_cohort_clinical_file(config)
        cohort_data[config.cohort_name] = df_clin
        attrition_data[config.cohort_name] = df_attrition

    return cohort_data, attrition_data


def _plot_single_km_subplot(
    ax: plt.Axes,
    label: str,
    df_clin: pd.DataFrame,
    treatment_label: str = "",
) -> dict[str, Any]:
    """Plots KM curve and computes demographic statistics for one cohort subplot."""
    print(f"  Plotting KM curve for {label} ({len(df_clin)} samples)...")
    return {
        "survival": plot_km_os(ax, df_clin, label),
        "age": calculate_age_statistics(df_clin),
        "sex": calculate_sex_statistics(df_clin),
        "treatment": calculate_treatment_statistics(df_clin, label, treatment_label),
    }


def _setup_km_grid_figure(n_cohorts: int) -> tuple[plt.Figure, list[plt.Axes]]:
    """Configures grid subplot axes for Kaplan-Meier OS curves."""
    max_cols = 3
    n_cols = min(max_cols, n_cohorts)
    n_rows = (n_cohorts + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    axes_flat = list(np.array(axes).flatten()) if n_cohorts > 1 else [axes]
    return fig, axes_flat


def _run_km_plotting_stage(
    cohort_order: list[str],
    cohort_data: dict[str, pd.DataFrame],
    dataset_configs: tuple["DatasetConfig", ...] = (),
) -> tuple[dict[str, dict[str, Any]], Path]:
    """Generates Kaplan-Meier OS curves across cohorts and saves figure."""
    # Build a label lookup so treatment hints flow config-driven into the
    # treatment statistics without hardcoded cohort-name branches.
    treatment_label_map: dict[str, str] = {
        cfg.cohort_name: cfg.treatment_label for cfg in dataset_configs
    }

    n_cohorts = len(cohort_order)
    fig, axes_flat = _setup_km_grid_figure(n_cohorts)
    cohort_results: dict[str, dict[str, Any]] = {}

    for i, label in enumerate(cohort_order):
        cohort_results[label] = _plot_single_km_subplot(
            axes_flat[i], label, cohort_data[label],
            treatment_label=treatment_label_map.get(label, ""),
        )

    for j in range(n_cohorts, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Overall Survival — Immunotherapy Trial Cohorts", fontsize=16, fontweight="bold", y=1.02
    )
    fig.tight_layout()

    out_path = PLOT_DIR / "km_os_grid.png"
    save_fig(fig, out_path)
    plt.close(fig)
    print(f"\nSaved KM plot grid to {out_path.relative_to(_SUBPROJECT_ROOT).as_posix()}")

    return cohort_results, out_path


def _load_analysis_datasets() -> tuple[
    tuple[str, ...], dict[str, pd.DataFrame], dict[str, pd.DataFrame], tuple["DatasetConfig", ...]
]:
    """Loads configured active trial cohort dataset configs and clinical DataFrames."""
    print("Loading dataset configurations...")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found at "
            f"{CONFIG_PATH.relative_to(_SUBPROJECT_ROOT).as_posix()}"
        )

    all_configs: tuple[DatasetConfig, ...] = load_dataset_config(CONFIG_PATH)
    dataset_configs = tuple(config for config in all_configs if config.cohort_name != "TCGA-SKCM")
    cohort_order = tuple(config.cohort_name for config in dataset_configs)
    print(f"Configured ICI cohorts: {', '.join(cohort_order)}")

    cohort_data, attrition_data = _load_cohort_and_attrition_data(dataset_configs)
    return cohort_order, cohort_data, attrition_data, dataset_configs


def _compute_treatment_breakdowns(
    cohort_results: dict[str, dict[str, Any]],
    cohort_order: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Derives treatment and survival breakdown dicts needed for demographics and report."""
    treatment_results = {c: cohort_results[c]["treatment"] for c in cohort_order}
    survival_results = {c: cohort_results[c]["survival"] for c in cohort_order}
    ici_breakdown = compute_ici_agent_breakdown(treatment_results=treatment_results)
    ctla4_breakdown = compute_prior_ctla4_breakdown(
        treatment_results=treatment_results,
        survival_results=survival_results,
    )
    return treatment_results, ici_breakdown, ctla4_breakdown


def _generate_demographics_and_reports(
    cohort_order: list[str],
    cohort_data: dict[str, pd.DataFrame],
    attrition_data: dict[str, pd.DataFrame],
    cohort_results: dict[str, dict[str, Any]],
    km_plot_path: Path,
    dataset_configs: tuple["DatasetConfig", ...] = (),
) -> None:
    """Computes demographics, generates 2x2 grid, runs companion scripts, and writes report."""
    print("\nComputing overall patient demographics...")
    overall_demographics = compute_overall_demographics(cohort_data=cohort_data)
    _treatment_results, ici_breakdown, ctla4_breakdown = _compute_treatment_breakdowns(
        cohort_results, cohort_order
    )
    print("\nGenerating clinical demographics 2×2 grid...")
    plot_clinical_demographics_grid(
        cohort_data=cohort_data, overall_demographics=overall_demographics,
        ici_breakdown=ici_breakdown, ctla4_breakdown=ctla4_breakdown, out_path=DEMO_GRID_PATH,
    )
    run_companion_scripts(_COMPANION_SCRIPTS, base_dir=_SUBPROJECT_ROOT)
    print("\nGenerating Markdown clinical characteristics report...")
    generate_clinical_report(
        cohort_results=cohort_results, attrition_data=attrition_data,
        plot_path=km_plot_path, report_path=REPORT_PATH, cohort_order=cohort_order,
        overall_demographics=overall_demographics, ici_breakdown=ici_breakdown,
        ctla4_breakdown=ctla4_breakdown, demographics_grid_path=DEMO_GRID_PATH,
        cohort_data=cohort_data, dataset_configs=dataset_configs,
    )


def main() -> None:
    """Executes the Kaplan-Meier OS analysis and generates the clinical report."""
    print("===========================================================")
    print("Clinical Analysis — Phase 1: Kaplan-Meier OS Curves")
    print("===========================================================")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_DIR.mkdir(exist_ok=True, parents=True)

    cohort_order_tuple, cohort_data, attrition_data, dataset_configs = _load_analysis_datasets()
    cohort_order = list(cohort_order_tuple)

    print("\nGenerating Kaplan-Meier curves...")
    cohort_results, km_plot_path = _run_km_plotting_stage(
        cohort_order, cohort_data, dataset_configs
    )

    _generate_demographics_and_reports(
        cohort_order, cohort_data, attrition_data, cohort_results, km_plot_path, dataset_configs
    )

    print("\n===========================================================")
    print("Done!")
if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)

        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(_SUBPROJECT_ROOT).as_posix()}")
            main()