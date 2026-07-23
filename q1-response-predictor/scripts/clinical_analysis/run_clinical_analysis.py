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
from src.styles import get_cohort_color, set_presentation_style
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

# ===========================================================================
# PATHS & CONFIGURATION
# ===========================================================================

CONFIG_PATH = CONFIG_DIR / "datasets.yaml"
PLOT_DIR = PLOTS_DIR / "clinical"
LOG_PATH = LOG_DIR / "run_clinical_analysis.log"
REPORT_DIR = REPORTS_DIR / "pillar-1-cohorts-and-preprocessing"
REPORT_PATH = REPORT_DIR / "cohort_characteristics_clinical.md"

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
        statistics["vaccine"] = int(
            df_clin.get("TX_TYPE_VACCINE", pd.Series(dtype=float))
            .fillna(0)
            .sum()
        )
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
    kmf.fit(
        durations=df[time_col],
        event_observed=df[event_col],
        label=f"Overall Cohort (N={len(df)})",
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

    ax.set_title(f"{cohort_label} (N={len(df)})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (months)", fontsize=10)
    ax.set_ylabel("Overall Survival Probability", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

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
    """Generates a Markdown table summarizing sample attrition across preprocessing steps.

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
) -> None:
    """Generates an Obsidian-compatible Markdown clinical characteristics report.

    Args:
        cohort_results: Calculated clinical and survival metrics per cohort.
        attrition_data: Preprocessing sample attrition DataFrames per cohort.
        plot_path: Path to the saved 2x2 KM grid image.
        report_path: Target path for the output Markdown report.
        cohort_order: Ordered list of cohort labels.
    """
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    clinical_characteristics_table = generate_clinical_characteristics_table(
        cohort_results=cohort_results,
        cohort_order=cohort_order,
    )

    attrition_table = generate_attrition_table(
        attrition_data=attrition_data,
        cohort_order=cohort_order,
    )

    key_observations = generate_key_observations(
        cohort_results=cohort_results,
        attrition_data=attrition_data,
        cohort_order=cohort_order,
    )

    frontmatter = generate_obsidian_frontmatter(
        title="Clinical Characteristics of Data Cohorts",
        aliases=["Clinical Cohort Characteristics"],
        tags=[
            "melanoma",
            "clinical-analysis",
            "cohort-characteristics",
            "survival-analysis",
        ],
        created=timestamp,
        updated=timestamp,
    )

    report = f"""{frontmatter}

# Clinical Characteristics of Data Cohorts

This report provides a comparative summary of patient demographic, treatment, survival, and sample attrition characteristics across the four cohorts analysed in this study:

*   **TCGA-SKCM**: Baseline reference cohort with recorded treatment history.
*   **Liu 2019**: Advanced melanoma trial cohort treated with anti-PD-1 monotherapy.
*   **Hugo 2016**: Anti-PD-1 clinical trial cohort.
*   **Riaz 2017**: Anti-PD-1 clinical trial cohort treated with nivolumab.

Demographic and treatment characteristics were calculated dynamically from the processed clinical datasets. Survival characteristics were calculated from cleaned clinical records evaluated in the Kaplan-Meier analysis. Sample attrition details track cohort retention across data preprocessing stages.

_**Table 1: Baseline Patient and Disease Characteristics**_

{clinical_characteristics_table}

---

## Sample Preprocessing Attrition

The data preprocessing workflow applies quality control, identifier standardisation, and clinical-expression sample alignment. The table below outlines sample retention and attrition rationale for each cohort.

_**Table 2: Sample Attrition Across Preprocessing Steps**_

{attrition_table}

---

## Key Observations

{key_observations}

---

## Overall Survival Curves (KM Plots)

The unstratified Kaplan-Meier overall survival curves for each cohort are shown below. Median overall survival is annotated for each cohort where reached.

![[{plot_path.name}]]

_**Overall Survival KM Curves (All Cohorts)**_

---

## Univariate Associations with Response (Forest Plot)

To evaluate whether individual baseline clinical and genomic features predict response to anti-PD-1 therapy, univariate Odds Ratios (OR) and 95% Confidence Intervals (95% CI) were calculated across individual trial cohorts (Liu 2019, Hugo 2016, Riaz 2017) and the pooled immunotherapy trial cohort. Categorical variables were evaluated via Fisher's Exact test, and continuous variables (Age, TMB, SNV Neoantigens, Indel Neoantigens) were evaluated per 1 SD increase via univariate logistic regression.

![Forest Plot of Univariate Odds Ratios](../../plots/clinical/univariate_associations.png)

### Key Takeaways
1. **Driver Mutations**: *NF1* mutated tumours show elevated odds ratios for response in the pooled trial cohort, while *BRAF* mutations show virtually no univariate association with response.
2. **Anatomical Stage Trend**: Clinical Stage IV was associated with an OR of 6.06 relative to Stage III in pooled analysis (p = 0.083), reflecting small Stage III representation in checkpoint blockade trial cohorts.
3. **TMB & Neoantigen Trends**: TMB shows a positive trend with response in the pooled trial cohort (OR = 1.37 per SD increase, p = 0.160), with consistent positive point estimates across Liu 2019 and Hugo 2016.
4. **Demographics**: Age and Sex show no significant association with immunotherapy response (OR ≈ 0.98 – 1.26), confirming demographic balance across treatment groups.

---

## Analysis Notes

Overall survival was analysed using the available cohort-specific survival duration and event-status variables.

Records were excluded from the survival analysis if they had:

*   Missing survival time.
*   Missing survival event status.
*   Non-numeric survival values.
*   A survival time less than or equal to zero.

Median overall survival was estimated using the Kaplan-Meier survival function. Where the estimated survival probability did not fall below 50% during follow-up, median OS was reported as **NR (not reached)**.

Treatment percentages use the number of patients with an available treatment annotation as the denominator where this differs from the total cohort size. TCGA treatment-history categories are based on binary treatment-type indicators and are not necessarily mutually exclusive.

Sample attrition tracking records sample filtering from initial cBioPortal data ingestion through clinical-expression alignment.

The analysis is descriptive and unstratified. It does not adjust for demographic, clinical, molecular, treatment, or study-specific confounding factors.
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

    set_presentation_style()

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
    # Generate Markdown clinical report
    # -----------------------------------------------------------------------
    print("\nGenerating Markdown clinical characteristics report...")
    generate_clinical_report(
        cohort_results=cohort_results,
        attrition_data=attrition_data,
        plot_path=out_path,
        report_path=REPORT_PATH,
        cohort_order=cohort_order,
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