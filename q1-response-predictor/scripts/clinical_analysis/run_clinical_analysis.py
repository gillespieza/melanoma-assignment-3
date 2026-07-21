"""
Phase 1 Clinical Analysis Script for Melanoma Cohorts.

Generates:

1. Unstratified Kaplan-Meier Overall Survival (OS) curves across all
   clinical study cohorts in a 2x2 grid layout.

2. An Obsidian-compatible Markdown clinical characteristics report
   containing dynamically calculated demographic, treatment and survival
   statistics.

Cohorts:
    - Liu 2019
    - Hugo 2016
    - Riaz 2017
    - TCGA-SKCM
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


# ===========================================================================
# BOOTSTRAP PROJECT ROOT RESOLUTION
# ===========================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


# ===========================================================================
# PROJECT IMPORTS
# ===========================================================================

from src.data_loaders import (
    load_hugo_2016,
    load_liu_2019,
    load_riaz_2017,
)

from src.styles import (
    get_cohort_color,
    set_presentation_style,
)

from src.utils.formatting import (
    format_count_percentage,
    format_median,
    format_median_iqr,
)

from src.utils.logging import TeeStream
from src.utils.paths import find_project_root
from src.utils.plotting import save_fig


# ===========================================================================
# PATHS
# ===========================================================================

DATA_DIR = (
    find_project_root(
        Path(__file__).resolve()
    )
    / "data"
)

PLOT_DIR = (
    BASE_DIR
    / "plots"
    / "clinical"
)

LOG_DIR = (
    BASE_DIR
    / "logs"
)

LOG_PATH = (
    LOG_DIR
    / "run_clinical_analysis.log"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
    / "pillar-1-cohorts-and-preprocessing"
)

REPORT_PATH = (
    REPORT_DIR
    / "cohort_characteristics_clinical.md"
)


# ===========================================================================
# SURVIVAL DATA HANDLING
# ===========================================================================


def _resolve_os_columns(
    df_clin: pd.DataFrame,
) -> Tuple[str, str]:
    """
    Determines the overall survival time and event columns.

    Supports:

        - Trial cohorts: os_months / os_status
        - TCGA: OS_MONTHS / OS_STATUS

    Args:
        df_clin:
            Clinical DataFrame.

    Returns:
        Tuple containing:

            (
                survival_time_column,
                survival_event_column,
            )

    Raises:
        ValueError:
            If no recognised survival column pair is found.
    """

    column_pairs = [
        (
            "os_months",
            "os_status",
        ),
        (
            "OS_MONTHS",
            "OS_STATUS",
        ),
    ]

    for time_col, event_col in column_pairs:

        if (
            time_col in df_clin.columns
            and event_col in df_clin.columns
        ):

            return (
                time_col,
                event_col,
            )

    raise ValueError(
        "Could not identify overall survival columns.\n"
        f"Available columns: {list(df_clin.columns)}"
    )


def _clean_os(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
) -> pd.DataFrame:
    """
    Drops rows with missing or invalid survival data.

    Args:
        df:
            Clinical DataFrame.

        time_col:
            Overall survival time column.

        event_col:
            Overall survival event indicator column.

    Returns:
        Cleaned survival DataFrame.
    """

    out = df[
        [
            time_col,
            event_col,
        ]
    ].copy()

    out[time_col] = pd.to_numeric(
        out[time_col],
        errors="coerce",
    )

    out[event_col] = pd.to_numeric(
        out[event_col],
        errors="coerce",
    )

    out = out.dropna(
        subset=[
            time_col,
            event_col,
        ]
    )

    out = out[
        out[time_col] > 0
    ]

    return out


# ===========================================================================
# DEMOGRAPHIC STATISTICS
# ===========================================================================


def _resolve_age_column(
    df_clin: pd.DataFrame,
) -> str | None:
    """
    Identifies the age column in a clinical DataFrame.
    """

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


def calculate_age_statistics(
    df_clin: pd.DataFrame,
) -> dict:
    """
    Calculates descriptive age statistics for a clinical cohort.
    """

    age_col = _resolve_age_column(
        df_clin
    )

    if age_col is None:

        return {
            "n_age": 0,
            "median_age": np.nan,
            "q1_age": np.nan,
            "q3_age": np.nan,
        }

    age = pd.to_numeric(
        df_clin[age_col],
        errors="coerce",
    ).dropna()

    return {
        "n_age": len(age),
        "median_age": age.median(),
        "q1_age": age.quantile(0.25),
        "q3_age": age.quantile(0.75),
    }


def calculate_sex_statistics(
    df_clin: pd.DataFrame,
) -> dict:
    """
    Calculates female sex frequency and percentage.

    Returns:
        Dictionary containing:

            - n_female
            - n_sex_available
    """

    if "SEX" not in df_clin.columns:

        return {
            "n_female": 0,
            "n_sex_available": 0,
        }

    sex = (
        df_clin["SEX"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    sex = sex.replace(
        {
            "FEMALE": "F",
            "WOMAN": "F",
            "MALE": "M",
            "MAN": "M",
        }
    )

    valid_sex = sex.isin(
        [
            "F",
            "M",
        ]
    )

    return {
        "n_female": int(
            (sex == "F").sum()
        ),
        "n_sex_available": int(
            valid_sex.sum()
        ),
    }


# ===========================================================================
# TREATMENT STATISTICS
# ===========================================================================


def _normalise_text_series(
    series: pd.Series,
) -> pd.Series:
    """
    Converts a Series to normalised uppercase text values.
    """

    return (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )


def calculate_treatment_statistics(
    df_clin: pd.DataFrame,
    cohort_label: str,
) -> dict:
    """
    Calculates cohort-specific treatment statistics.

    Treatment variables differ substantially between the clinical cohorts,
    so the calculation is cohort-specific rather than fully generic.
    """

    statistics = {}

    # -----------------------------------------------------------------------
    # Liu 2019
    # -----------------------------------------------------------------------

    if cohort_label == "Liu 2019":

        if "ICI_RX" in df_clin.columns:

            ici_rx = _normalise_text_series(
                df_clin["ICI_RX"]
            )

            valid_ici_rx = ici_rx.notna()

            statistics["pembrolizumab"] = int(
                (
                    ici_rx
                    == "PEMBROLIZUMAB"
                ).sum()
            )

            statistics["nivolumab"] = int(
                (
                    ici_rx
                    == "NIVOLUMAB"
                ).sum()
            )

            statistics["treatment_n"] = int(
                valid_ici_rx.sum()
            )

        else:

            statistics["pembrolizumab"] = None
            statistics["nivolumab"] = None
            statistics["treatment_n"] = 0

        if "PRIOR_ICI_RX" in df_clin.columns:

            prior_ici = _normalise_text_series(
                df_clin["PRIOR_ICI_RX"]
            )

            valid_prior_ici = prior_ici.notna()

            statistics["prior_ctla4"] = int(
                prior_ici
                .str.contains(
                    "IPILIMUMAB",
                    na=False,
                )
                .sum()
            )

            statistics["prior_ctla4_n"] = int(
                valid_prior_ici.sum()
            )

        else:

            statistics["prior_ctla4"] = None
            statistics["prior_ctla4_n"] = 0

    # -----------------------------------------------------------------------
    # Hugo 2016
    # -----------------------------------------------------------------------

    elif cohort_label == "Hugo 2016":

        if "SAMPLE_TREATMENT" in df_clin.columns:

            treatment = _normalise_text_series(
                df_clin["SAMPLE_TREATMENT"]
            )

            valid_treatment = treatment.notna()

            statistics["pembrolizumab"] = int(
                treatment
                .str.contains(
                    "PEMBROLIZUMAB",
                    na=False,
                )
                .sum()
            )

            statistics["nivolumab"] = int(
                treatment
                .str.contains(
                    "NIVOLUMAB",
                    na=False,
                )
                .sum()
            )

            statistics["treatment_n"] = int(
                valid_treatment.sum()
            )

        else:

            statistics["pembrolizumab"] = None
            statistics["nivolumab"] = None
            statistics["treatment_n"] = 0

        statistics["prior_ctla4"] = None
        statistics["prior_ctla4_n"] = 0

    # -----------------------------------------------------------------------
    # Riaz 2017
    # -----------------------------------------------------------------------

    elif cohort_label == "Riaz 2017":

        if "SAMPLE_TREATMENT" in df_clin.columns:

            treatment = _normalise_text_series(
                df_clin["SAMPLE_TREATMENT"]
            )

            valid_treatment = treatment.notna()

            statistics["nivolumab"] = int(
                treatment
                .str.contains(
                    "NIVOLUMAB",
                    na=False,
                )
                .sum()
            )

            statistics["treatment_n"] = int(
                valid_treatment.sum()
            )

        else:

            statistics["nivolumab"] = None
            statistics["treatment_n"] = 0

        statistics["pembrolizumab"] = None

        if "PRIOR_ICI_RX" in df_clin.columns:

            prior_ici = _normalise_text_series(
                df_clin["PRIOR_ICI_RX"]
            )

            valid_prior_ici = prior_ici.notna()

            statistics["prior_ctla4"] = int(
                prior_ici
                .str.contains(
                    "IPILIMUMAB",
                    na=False,
                )
                .sum()
            )

            statistics["prior_ctla4_n"] = int(
                valid_prior_ici.sum()
            )

        else:

            statistics["prior_ctla4"] = None
            statistics["prior_ctla4_n"] = 0

    # -----------------------------------------------------------------------
    # TCGA-SKCM
    # -----------------------------------------------------------------------

    elif cohort_label == "TCGA-SKCM":

        statistics["pembrolizumab"] = int(
            df_clin[
                "TX_AGENT_PEMBROLIZUMAB"
            ]
            .fillna(0)
            .sum()
        )

        statistics["nivolumab"] = int(
            df_clin[
                "TX_AGENT_NIVOLUMAB"
            ]
            .fillna(0)
            .sum()
        )

        statistics["prior_ctla4"] = None

        statistics["treatment_n"] = len(
            df_clin
        )

        statistics["prior_ctla4_n"] = 0

        statistics["radiation"] = int(
            df_clin[
                "TX_TYPE_RADIATION_THERAPY"
            ]
            .fillna(0)
            .sum()
        )

        statistics["immunotherapy"] = int(
            df_clin[
                "TX_TYPE_IMMUNOTHERAPY"
            ]
            .fillna(0)
            .sum()
        )

        statistics["chemotherapy"] = int(
            df_clin[
                "TX_TYPE_CHEMOTHERAPY"
            ]
            .fillna(0)
            .sum()
        )

        statistics["vaccine"] = int(
            df_clin[
                "TX_TYPE_VACCINE"
            ]
            .fillna(0)
            .sum()
        )

        statistics["targeted_therapy"] = int(
            df_clin[
                "TX_TYPE_TARGETED_MOLECULAR_THERAPY"
            ]
            .fillna(0)
            .sum()
        )

        statistics["other_therapy"] = int(
            df_clin[
                [
                    "TX_TYPE_HORMONE_THERAPY",
                    "TX_TYPE_ANCILLARY",
                    "TX_TYPE_OTHER",
                ]
            ]
            .fillna(0)
            .any(axis=1)
            .sum()
        )

        statistics["no_recorded_treatment"] = int(
            (
                df_clin[
                    [
                        "TREATMENT_TYPES",
                        "TREATMENT_AGENTS",
                    ]
                ]
                .isna()
                .all(axis=1)
            )
            .sum()
        )

    return statistics


def _format_optional_count_percentage(
    count: int | None,
    total: int,
) -> str:
    """
    Formats an optional count as n (%).

    Returns an em dash when the variable is not applicable to the cohort.
    """

    if count is None:

        return "—"

    if total == 0:

        return "N/A"

    return format_count_percentage(
        count=count,
        total=total,
    )


# ===========================================================================
# KAPLAN-MEIER ANALYSIS
# ===========================================================================


def plot_km_os(
    ax: plt.Axes,
    df_clin: pd.DataFrame,
    cohort_label: str,
) -> dict:
    """
    Plots an unstratified Kaplan-Meier OS curve.

    Returns:
        Dictionary containing dynamically calculated survival statistics
        for report generation.
    """

    time_col, event_col = _resolve_os_columns(
        df_clin
    )

    df = _clean_os(
        df_clin,
        time_col,
        event_col,
    )

    cohort_color = get_cohort_color(
        cohort_label
    )

    kmf = KaplanMeierFitter()

    kmf.fit(
        durations=df[time_col],
        event_observed=df[event_col],
        label=f"N = {len(df)}",
    )

    kmf.plot_survival_function(
        ax=ax,
        ci_show=True,
        color=cohort_color,
        linewidth=2,
    )

    median_surv = (
        kmf.median_survival_time_
    )

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
            (
                "Median OS = "
                f"{median_surv:.1f} mo"
            ),
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

    ax.set_title(
        cohort_label,
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Time (months)",
        fontsize=10,
    )

    ax.set_ylabel(
        "Overall Survival Probability",
        fontsize=10,
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.legend(
        loc="lower left",
        fontsize=9,
        framealpha=0.9,
    )

    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.4,
    )

    # -----------------------------------------------------------------------
    # Calculate survival statistics for the report
    # -----------------------------------------------------------------------

    n_total = len(
        df_clin
    )

    n_valid_os = len(
        df
    )

    n_events = int(
        df[event_col].sum()
    )

    n_censored = int(
        (
            df[event_col]
            == 0
        ).sum()
    )

    event_rate = (
        n_events
        / n_valid_os
        * 100
    )

    median_follow_up = (
        df[time_col]
        .median()
    )

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
    survival_results: dict,
    age_results: dict,
    sex_results: dict,
    treatment_results: dict,
) -> str:
    """
    Generates the comparative clinical characteristics table.
    """

    cohort_order = [
        "Liu 2019",
        "Hugo 2016",
        "Riaz 2017",
        "TCGA-SKCM",
    ]

    rows = []

    # -----------------------------------------------------------------------
    # N
    # -----------------------------------------------------------------------

    rows.append(
        {
            "Characteristic": "**N**",
            **{
                cohort: str(
                    survival_results[
                        cohort
                    ][
                        "n_total"
                    ]
                )
                for cohort in cohort_order
            },
        }
    )

    rows.append(
        {
            "Characteristic": "",
            **{
                cohort: ""
                for cohort in cohort_order
            },
        }
    )

    # -----------------------------------------------------------------------
    # Demographics
    # -----------------------------------------------------------------------

    rows.append(
        {
            "Characteristic": "**Demographics**",
            **{
                cohort: ""
                for cohort in cohort_order
            },
        }
    )

    rows.append(
        {
            "Characteristic": (
                "Age, median (IQR)"
            ),
            **{
                cohort: format_median_iqr(
                    age_results[
                        cohort
                    ]
                )
                for cohort in cohort_order
            },
        }
    )

    rows.append(
        {
            "Characteristic": (
                "Female sex, n (%)"
            ),
            **{
                cohort: format_count_percentage(
                    count=sex_results[
                        cohort
                    ][
                        "n_female"
                    ],
                    total=sex_results[
                        cohort
                    ][
                        "n_sex_available"
                    ],
                )
                for cohort in cohort_order
            },
        }
    )

    rows.append(
        {
            "Characteristic": "",
            **{
                cohort: ""
                for cohort in cohort_order
            },
        }
    )

    # -----------------------------------------------------------------------
    # Treatment
    # -----------------------------------------------------------------------

    rows.append(
        {
            "Characteristic": "**Treatment**",
            **{
                cohort: ""
                for cohort in cohort_order
            },
        }
    )

    rows.append(
        {
            "Characteristic": (
                "ICI agent — Pembrolizumab"
            ),
            **{
                cohort: _format_optional_count_percentage(
                    treatment_results[
                        cohort
                    ].get(
                        "pembrolizumab"
                    ),
                    treatment_results[
                        cohort
                    ].get(
                        "treatment_n",
                        survival_results[
                            cohort
                        ][
                            "n_total"
                        ],
                    ),
                )
                for cohort in cohort_order
            },
        }
    )

    rows.append(
        {
            "Characteristic": (
                "ICI agent — Nivolumab"
            ),
            **{
                cohort: _format_optional_count_percentage(
                    treatment_results[
                        cohort
                    ].get(
                        "nivolumab"
                    ),
                    treatment_results[
                        cohort
                    ].get(
                        "treatment_n",
                        survival_results[
                            cohort
                        ][
                            "n_total"
                        ],
                    ),
                )
                for cohort in cohort_order
            },
        }
    )

    rows.append(
        {
            "Characteristic": (
                "Prior anti-CTLA-4"
            ),
            **{
                cohort: _format_optional_count_percentage(
                    treatment_results[
                        cohort
                    ].get(
                        "prior_ctla4"
                    ),
                    treatment_results[
                        cohort
                    ].get(
                        "prior_ctla4_n",
                        survival_results[
                            cohort
                        ][
                            "n_total"
                        ],
                    ),
                )
                for cohort in cohort_order
            },
        }
    )

    # -----------------------------------------------------------------------
    # TCGA treatment history
    # -----------------------------------------------------------------------

    rows.append(
        {
            "Characteristic": (
                "Treatment History (TCGA only)"
            ),
            "Liu 2019": "—",
            "Hugo 2016": "—",
            "Riaz 2017": "—",
            "TCGA-SKCM": "",
        }
    )

    tcga_treatment_rows = [
        (
            "— Radiation Therapy",
            "radiation",
        ),
        (
            "— Immunotherapy",
            "immunotherapy",
        ),
        (
            "— Chemotherapy",
            "chemotherapy",
        ),
        (
            "— Vaccine",
            "vaccine",
        ),
        (
            "— Targeted Therapy",
            "targeted_therapy",
        ),
        (
            "— Other Therapy",
            "other_therapy",
        ),
        (
            "— No recorded treatment",
            "no_recorded_treatment",
        ),
    ]

    for label, key in tcga_treatment_rows:

        rows.append(
            {
                "Characteristic": label,
                "Liu 2019": "—",
                "Hugo 2016": "—",
                "Riaz 2017": "—",
                "TCGA-SKCM": (
                    _format_optional_count_percentage(
                        treatment_results[
                            "TCGA-SKCM"
                        ].get(
                            key
                        ),
                        survival_results[
                            "TCGA-SKCM"
                        ][
                            "n_total"
                        ],
                    )
                ),
            }
        )

    rows.append(
        {
            "Characteristic": "",
            **{
                cohort: ""
                for cohort in cohort_order
            },
        }
    )

    # -----------------------------------------------------------------------
    # Survival Outcomes
    # -----------------------------------------------------------------------

    rows.append(
        {
            "Characteristic": (
                "**Survival Outcomes**"
            ),
            **{
                cohort: ""
                for cohort in cohort_order
            },
        }
    )

    rows.append(
        {
            "Characteristic": (
                "Median OS, months (95% CI)"
            ),
            **{
                cohort: format_median(
                    survival_results[
                        cohort
                    ][
                        "median_os"
                    ]
                )
                for cohort in cohort_order
            },
        }
    )

    rows.append(
        {
            "Characteristic": (
                "OS events, n (%)"
            ),
            **{
                cohort: format_count_percentage(
                    count=survival_results[
                        cohort
                    ][
                        "n_events"
                    ],
                    total=survival_results[
                        cohort
                    ][
                        "n_valid_os"
                    ],
                )
                for cohort in cohort_order
            },
        }
    )

    rows.append(
        {
            "Characteristic": (
                "Median follow-up, months"
            ),
            **{
                cohort: format_median(
                    survival_results[
                        cohort
                    ][
                        "median_follow_up"
                    ]
                )
                for cohort in cohort_order
            },
        }
    )

    table_df = pd.DataFrame(
        rows
    )

    return table_df.to_markdown(
        index=False
    )

def generate_key_observations(
    survival_results: dict,
) -> str:
    """
    Generates dynamic key observations based on the survival analysis.
    """

    cohort_order = [
        "Liu 2019",
        "Hugo 2016",
        "Riaz 2017",
        "TCGA-SKCM",
    ]

    largest_cohort = max(
        cohort_order,
        key=lambda cohort: (
            survival_results[
                cohort
            ][
                "n_total"
            ]
        ),
    )

    longest_follow_up = max(
        cohort_order,
        key=lambda cohort: (
            survival_results[
                cohort
            ][
                "median_follow_up"
            ]
        ),
    )

    return f"""
1. **Cohort Size**: The largest cohort is **{largest_cohort}** with **N = {survival_results[largest_cohort]["n_total"]}** patients. The TCGA-SKCM cohort serves as a genomic reference population with detailed treatment history. The three IO cohorts are anti-PD-1 clinical trial cohorts.

2. **Overall Survival**: Median OS was estimated using the Kaplan-Meier survival function. Where the estimated survival probability did not fall below 50% during follow-up, median OS was not reached.

3. **Follow-up**: **{longest_follow_up}** had the longest median follow-up at **{format_median(survival_results[longest_follow_up]["median_follow_up"])} months**.

4. **Event Rates**: Observed OS event rates varied between cohorts, reflecting differences in cohort composition, disease stage, treatment context, follow-up duration, and censoring.

5. **Missing Data**: Age, sex and treatment variables have different levels of availability across cohorts. These differences should be considered when comparing clinical characteristics across datasets.

6. **Interpretation**: These unstratified survival estimates provide a descriptive baseline for subsequent molecular and predictive analyses. Cross-cohort comparisons should be interpreted cautiously because the cohorts differ in clinical context and study design.
""".strip()


# ===========================================================================
# REPORT GENERATION
# ===========================================================================


def generate_clinical_report(
    survival_results: dict,
    age_results: dict,
    sex_results: dict,
    treatment_results: dict,
    plot_path: Path,
    report_path: Path,
) -> None:
    """
    Generates an Obsidian-compatible Markdown clinical report.
    """

    timestamp = pd.Timestamp.now().strftime(
        "%Y-%m-%d %H:%M"
    )

    clinical_characteristics_table = (
        generate_clinical_characteristics_table(
            survival_results=survival_results,
            age_results=age_results,
            sex_results=sex_results,
            treatment_results=treatment_results,
        )
    )

    key_observations = (
        generate_key_observations(
            survival_results
        )
    )

    report = f"""---
title: Clinical Characteristics of Data Cohorts
aliases:
  - Clinical Cohort Characteristics
tags:
  - melanoma
  - clinical-analysis
  - cohort-characteristics
  - survival-analysis
created: {timestamp}
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: {timestamp}
---

# Clinical Characteristics of Data Cohorts

This report provides a comparative summary of the patient demographic, treatment and survival characteristics across the four cohorts analysed in this study:

*   **TCGA-SKCM**: Baseline reference cohort with recorded treatment history.
*   **Liu 2019**: Advanced melanoma trial cohort treated with anti-PD-1 monotherapy.
*   **Hugo 2016**: Anti-PD-1 clinical trial cohort.
*   **Riaz 2017**: Anti-PD-1 clinical trial cohort treated with nivolumab.

The demographic and treatment characteristics were calculated from the available clinical cohort data. Survival characteristics were calculated from the cleaned clinical data used in the Kaplan-Meier analysis.

_**Table 1: Baseline Patient and Disease Characteristics**_

{clinical_characteristics_table}

## Key Observations

{key_observations}

---

## Overall Survival Curves (KM Plots)

The unstratified Kaplan-Meier overall survival curves for each cohort are shown below. Median overall survival is annotated for each cohort where reached.

![[km_os_grid.png]]

_**Overall Survival KM Curves (All Cohorts)**_

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

The analysis is descriptive and unstratified. It does not adjust for demographic, clinical, molecular, treatment, or study-specific confounding factors.

"""

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print(
        "\nSaved clinical analysis report to "
        f"{report_path.relative_to(BASE_DIR).as_posix()}"
    )


# ===========================================================================
# MAIN ANALYSIS
# ===========================================================================


def main() -> None:
    """
    Executes the Kaplan-Meier overall survival analysis and generates the
    Obsidian clinical report.
    """

    print(
        "=================================================="
    )

    print(
        "Clinical Analysis — Phase 1: Kaplan-Meier OS Curves"
    )

    print(
        "==================================================\n"
    )

    set_presentation_style()

    PLOT_DIR.mkdir(
        exist_ok=True,
        parents=True,
    )

    REPORT_DIR.mkdir(
        exist_ok=True,
        parents=True,
    )

    # -----------------------------------------------------------------------
    # Load clinical cohorts
    # -----------------------------------------------------------------------

    print(
        "Loading clinical cohorts..."
    )

    _, clin_liu = load_liu_2019(
        DATA_DIR
    )

    _, clin_hugo = load_hugo_2016(
        DATA_DIR
    )

    _, clin_riaz = load_riaz_2017(
        DATA_DIR
    )

    tcga_clin_path = (
        DATA_DIR
        / "processed"
        / "skcm_tcga_pan_can_atlas_2018"
        / "clin_cleaned.csv"
    )

    if not tcga_clin_path.exists():

        raise FileNotFoundError(
            "Processed TCGA clinical file not found at "
            f"{tcga_clin_path.relative_to(BASE_DIR).as_posix()}.\n"
            "Run preprocessing first."
        )

    clin_tcga = pd.read_csv(
        tcga_clin_path,
        index_col=0,
    )

    cohorts: List[
        Tuple[
            str,
            pd.DataFrame,
        ]
    ] = [
        (
            "Liu 2019",
            clin_liu,
        ),
        (
            "Hugo 2016",
            clin_hugo,
        ),
        (
            "Riaz 2017",
            clin_riaz,
        ),
        (
            "TCGA-SKCM",
            clin_tcga,
        ),
    ]

    # -----------------------------------------------------------------------
    # Generate KM plots and collect statistics
    # -----------------------------------------------------------------------

    print(
        "\nGenerating Kaplan-Meier curves..."
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            14,
            10,
        ),
    )

    flat_axes = axes.ravel()

    survival_results = {}
    age_results = {}
    sex_results = {}
    treatment_results = {}

    for ax, (
        label,
        df_clin,
    ) in zip(
        flat_axes,
        cohorts,
    ):

        print(
            f"  Plotting KM curve for "
            f"{label} "
            f"({len(df_clin)} samples)..."
        )

        survival_results[
            label
        ] = plot_km_os(
            ax,
            df_clin,
            label,
        )

        age_results[
            label
        ] = calculate_age_statistics(
            df_clin
        )

        sex_results[
            label
        ] = calculate_sex_statistics(
            df_clin
        )

        treatment_results[
            label
        ] = calculate_treatment_statistics(
            df_clin,
            label,
        )

    fig.suptitle(
        "Overall Survival — All Cohorts",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )

    fig.tight_layout()

    out_path = (
        PLOT_DIR
        / "km_os_grid.png"
    )

    save_fig(
        fig,
        out_path,
    )

    plt.close(
        fig
    )

    print(
        "\nSaved 2×2 KM plot to "
        f"{out_path.relative_to(BASE_DIR).as_posix()}"
    )

    # -----------------------------------------------------------------------
    # Generate Obsidian Markdown report
    # -----------------------------------------------------------------------

    print(
        "\nSurvival result keys:",
        list(
            survival_results.keys()
        ),
    )

    print(
        "Age result keys:",
        list(
            age_results.keys()
        ),
    )

    print(
        "Sex result keys:",
        list(
            sex_results.keys()
        ),
    )

    print(
        "Treatment result keys:",
        list(
            treatment_results.keys()
        ),
    )

    generate_clinical_report(
        survival_results=survival_results,
        age_results=age_results,
        sex_results=sex_results,
        treatment_results=treatment_results,
        plot_path=out_path,
        report_path=REPORT_PATH,
    )

    print(
        "\n=================================================="
    )

    print(
        "Done!"
    )

    print(
        "=================================================="
    )


# ===========================================================================
# LOGGING ENTRY POINT
# ===========================================================================


if __name__ == "__main__":

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        LOG_PATH,
        "w",
        encoding="utf-8",
    ) as log_file:

        stdout_tee = TeeStream(
            sys.stdout,
            log_file,
        )

        stderr_tee = TeeStream(
            sys.stderr,
            log_file,
        )

        with contextlib.redirect_stdout(
            stdout_tee
        ), contextlib.redirect_stderr(
            stderr_tee
        ):

            print(
                "Logging console output to "
                f"{LOG_PATH.relative_to(BASE_DIR).as_posix()}"
            )

            main()