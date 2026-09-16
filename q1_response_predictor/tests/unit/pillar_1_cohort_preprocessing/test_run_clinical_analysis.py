"""Unit tests for run_clinical_analysis.py."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb

from scripts.pillar_1_cohort_preprocessing import run_clinical_analysis as ca


def test_resolve_os_columns_prefers_canonical_lowercase_names() -> None:
    clinical = pd.DataFrame({"os_months": [1.0], "os_status": [1]})

    assert ca._resolve_os_columns(clinical) == ("os_months", "os_status")


def test_resolve_os_columns_supports_legacy_uppercase_names() -> None:
    clinical = pd.DataFrame({"OS_MONTHS": [1.0], "OS_STATUS": [0]})

    assert ca._resolve_os_columns(clinical) == ("OS_MONTHS", "OS_STATUS")


def test_resolve_os_columns_raises_when_pair_is_incomplete() -> None:
    clinical = pd.DataFrame({"os_months": [1.0]})

    with pytest.raises(ValueError, match="Could not identify overall survival columns"):
        ca._resolve_os_columns(clinical)


def test_clean_os_coerces_values_and_removes_invalid_survival_times() -> None:
    clinical = pd.DataFrame(
        {
            "os_months": ["12", "invalid", 0, -2, 6, np.nan],
            "os_status": ["1", 0, 1, 0, "0", np.nan],
        }
    )

    result = ca._clean_os(clinical, "os_months", "os_status")

    assert result["os_months"].tolist() == [12.0, 6.0]
    assert result["os_status"].tolist() == [1.0, 0.0]


def test_calculate_age_statistics_uses_age_alias_and_ignores_invalid_values() -> None:
    clinical = pd.DataFrame({"AGE_AT_DIAGNOSIS": ["40", "60", "invalid", np.nan]})

    result = ca.calculate_age_statistics(clinical)

    assert result["n_age"] == 2
    assert result["median_age"] == 50.0
    assert result["q1_age"] == 45.0
    assert result["q3_age"] == 55.0


def test_calculate_age_statistics_returns_empty_summary_without_age_column() -> None:
    result = ca.calculate_age_statistics(pd.DataFrame({"SEX": ["F"]}))

    assert result == {
        "n_age": 0,
        "median_age": np.nan,
        "q1_age": np.nan,
        "q3_age": np.nan,
    }


def test_calculate_sex_statistics_normalises_labels_and_excludes_unknowns() -> None:
    clinical = pd.DataFrame(
        {"SEX": ["female", "Woman", "MALE", "man", "unknown", np.nan]}
    )

    assert ca.calculate_sex_statistics(clinical) == {
        "n_female": 2,
        "n_sex_available": 4,
    }


def test_search_treatment_agents_detects_text_and_numeric_agent_columns() -> None:
    clinical = pd.DataFrame(
        {
            "TREATMENT": ["Pembrolizumab + ipilimumab", "nivolumab", "none"],
            "TX_AGENT_VEMURAFENIB": [0, 1, 0],
        }
    )

    result = ca._search_treatment_agents(clinical, "Test Cohort")

    assert result["Pembrolizumab"] == 1
    assert result["Ipilimumab"] == 1
    assert result["Nivolumab"] == 1
    assert result["Vemurafenib"] == 1


def test_search_treatment_agents_unions_numeric_and_text_matches() -> None:
    clinical = pd.DataFrame(
        {
            "TREATMENT": ["pembrolizumab", "none", "none"],
            "TX_AGENT_PEMBROLIZUMAB": [0, 1, 0],
        }
    )

    result = ca._search_treatment_agents(clinical, "Test Cohort")

    assert result["Pembrolizumab"] == 2


def test_treatment_statistics_uses_prior_ici_rx_when_available() -> None:
    clinical = pd.DataFrame(
        {
            "PRIOR_ICI_RX": ["ipilimumab", "nivolumab", "none"],
            "TREATMENT": ["none", "none", "none"],
        }
    )

    result = ca.calculate_treatment_statistics(clinical, "Test Cohort")

    assert result["prior_ctla4"] == 1
    assert result["prior_ctla4_n"] == 3
    assert result["treatment_n"] == 3


def test_treatment_statistics_applies_riaz_prior_ctla4_trial_override() -> None:
    clinical = pd.DataFrame({"TREATMENT": ["nivolumab", "nivolumab", "nivolumab"]})

    result = ca.calculate_treatment_statistics(clinical, "Riaz 2017")

    assert result["prior_ctla4"] == 3


def test_generate_clinical_characteristics_table_contains_all_sections() -> None:
    cohort_results = {
        "Cohort A": {
            "survival": {
                "n_total": 2,
                "n_valid_os": 2,
                "n_events": 1,
                "median_os": 12.0,
                "median_follow_up": 15.0,
            },
            "age": {
                "median_age": 55.0,
                "q1_age": 50.0,
                "q3_age": 60.0,
            },
            "sex": {"n_female": 1, "n_sex_available": 2},
            "treatment": {
                "agents": {"Pembrolizumab": 2},
                "prior_ctla4": 0,
            },
        }
    }
    cohort_data = {
        "Cohort A": pd.DataFrame(
            {"os_months": [12.0, 15.0], "os_status": [1, 0]}
        )
    }

    result = ca.generate_clinical_characteristics_table(
        cohort_results,
        ["Cohort A"],
        overall_demographics={
            "age_median": 55.0,
            "age_q1": 50.0,
            "age_q3": 60.0,
        },
        cohort_data=cohort_data,
    )

    assert "**N**" in result
    assert "**Demographics**" in result
    assert "**Treatment Agents & Exposure**" in result
    assert "**Survival Outcomes**" in result
    assert "Agent — Pembrolizumab" in result
    assert "Cohort A" in result


def test_pooled_survival_summary_uses_same_cleaning_as_km_analysis() -> None:
    cohort_data = {
        "Cohort A": pd.DataFrame(
            {
                "os_months": ["12", "invalid", 0, "6"],
                "os_status": ["1", 0, 1, "0"],
            }
        )
    }

    median_os, median_follow_up = ca._compute_pooled_survival_summary(cohort_data)

    assert median_os == 12.0
    assert median_follow_up == 9.0


@pytest.mark.parametrize(
    ("values", "expected_count"),
    [
        ([1.0, 0.0, 1.0], 2),
        ([True, False, True], 2),
        (["Yes", "No", "1"], 2),
    ],
)
def test_filter_ici_subcohort_accepts_common_boolean_encodings(
    values: list[object],
    expected_count: int,
) -> None:
    clinical = pd.DataFrame({ca._COL_ICI_TX: values})

    result = ca._filter_ici_subcohort(clinical)

    assert len(result) == expected_count


def test_setup_km_grid_rejects_empty_cohort_list() -> None:
    with pytest.raises(ValueError, match="At least one cohort"):
        ca._setup_km_grid_figure(0)


def test_setup_km_grid_uses_two_by_two_layout_for_four_cohorts() -> None:
    figure, axes = ca._setup_km_grid_figure(4)

    try:
        assert figure.get_size_inches().tolist() == [11.0, 9.0]
        assert len(axes) == 4
    finally:
        ca.plt.close(figure)


def test_response_report_section_avoids_pooled_recist_claim() -> None:
    result = ca._build_report_section3_4(
        n_cohorts=4,
        n_total=272,
        plot_path=ca.Path("km_os_grid.png"),
        cohorts_list_str="Liu 2019, Hugo 2016, Riaz 2017, Gide 2019",
    )

    assert "Reported Immunotherapy Response" in result
    assert "Hugo 2016 does not use RECIST terminology" in result
    assert "pooled RECIST analysis" in result
    assert "stratified by RECIST response status" not in result
    assert "validates RECIST response as a surrogate endpoint" not in result


def test_shared_plot_directory_uses_underscore_subproject_root() -> None:
    assert ca.PLOTS_DIR == ca.PROJECT_ROOT / "q1_response_predictor" / "plots"
    assert "q1-response-predictor" not in ca.PLOTS_DIR.as_posix()


def test_prior_ctla4_panel_uses_okabe_red_for_exposure_and_green_for_naive() -> None:
    figure, axis = ca.plt.subplots()
    try:
        ca._plot_prior_ctla4_panel(
            axis,
            {"n_prior_ctla4": 3, "n_naive": 7, "n_total": 10},
        )

        wedge_colours = [wedge.get_facecolor()[:3] for wedge in axis.patches]
        assert wedge_colours[0] == tuple(to_rgb(ca.OKABE_ITO[5]))
        assert wedge_colours[1] == tuple(to_rgb(ca.OKABE_ITO[2]))
    finally:
        ca.plt.close(figure)


def test_load_analysis_datasets_excludes_configs_with_merge_disabled(
    dummy_dataset_config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    tcga_config = replace(
        dummy_dataset_config,
        cohort_name="TCGA GDC 2025",
        merge_enabled=False,
    )
    config_path = tmp_path / "datasets.yaml"
    config_path.touch()
    monkeypatch.setattr(ca, "CONFIG_PATH", config_path)
    monkeypatch.setattr(
        ca,
        "load_dataset_config",
        lambda _path: (dummy_dataset_config, tcga_config),
    )
    monkeypatch.setattr(
        ca,
        "_load_cohort_and_attrition_data",
        lambda configs: (
            {config.cohort_name: pd.DataFrame() for config in configs},
            {},
        ),
    )

    cohort_order, cohort_data, _attrition, configs = ca._load_analysis_datasets()

    assert cohort_order == (dummy_dataset_config.cohort_name,)
    assert list(cohort_data) == [dummy_dataset_config.cohort_name]
    assert [config.cohort_name for config in configs] == [
        dummy_dataset_config.cohort_name
    ]
