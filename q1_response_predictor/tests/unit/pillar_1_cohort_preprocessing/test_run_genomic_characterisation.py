"""Unit tests for run_genomic_characterisation.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.pillar_1_cohort_preprocessing import (
    run_genomic_characterisation as gc,
)


def test_dataset_config_preserves_established_merged_cohorts() -> None:
    """Fail when config merge tags change the established analysis cohorts."""
    configs = gc.load_dataset_config(gc.CONFIG_PATH)

    merged_cohorts = [
        config.cohort_name for config in configs if config.merge_enabled
    ]

    assert merged_cohorts == [
        "Liu 2019",
        "Hugo 2016",
        "Riaz 2017",
        "Gide 2019",
    ]
    assert "TCGA GDC 2025" not in merged_cohorts


def test_get_trial_cohort_names_excludes_tcga_reference() -> None:
    cohorts = {
        "TCGA-SKCM": pd.DataFrame(),
        "Riaz 2017": pd.DataFrame(),
        "Gide 2019": pd.DataFrame(),
    }

    assert gc._get_trial_cohort_names(cohorts) == ["Riaz 2017", "Gide 2019"]


def test_load_cohort_mutations_returns_binary_driver_columns_and_fills_samples(
    tmp_path,
) -> None:
    pd.DataFrame(
        {"BRAF": [2, 0], "NRAS": [0, 1], "BRAF_V600": [1, 0]},
        index=pd.Index(["S1", "S2"], name="SAMPLE_ID"),
    ).to_csv(tmp_path / "mutations_cleaned.csv")

    result = gc.load_cohort_mutations(tmp_path, ["S2", "S3"])

    assert result.index.tolist() == ["S2", "S3"]
    assert result.loc["S2", "mut_BRAF"] == 0
    assert result.loc["S2", "mut_NRAS"] == 1
    assert result.loc["S3", "mut_BRAF"] == 0
    assert result.loc["S3", "mut_BRAF_V600"] == 0


def test_load_cohort_mutations_returns_zero_defaults_when_file_missing(
    tmp_path,
) -> None:
    result = gc.load_cohort_mutations(tmp_path, ["S1", "S2"])

    assert result.shape[0] == 2
    assert result.index.tolist() == ["S1", "S2"]
    assert (result == 0).all().all()


def test_process_cohort_response_maps_recist_labels() -> None:
    clinical = pd.DataFrame(
        {
            "RESPONSE": ["Complete Response", "Progressive Disease", "Unknown"],
            "value": [1, 2, 3],
        }
    )

    result = gc._process_cohort_response(clinical, "Riaz 2017")

    assert result["response"].tolist() == [1.0, 0.0]
    assert result["value"].tolist() == [1, 2]
    assert "temp_resp" not in result


def test_process_tcga_response_drops_rows_without_survival_targets() -> None:
    clinical = pd.DataFrame(
        {
            gc._COL_OS_MONTHS: [12.0, np.nan, 8.0],
            gc._COL_OS_STATUS: [1.0, 0.0, np.nan],
        }
    )

    result = gc._process_cohort_response(clinical, "TCGA-SKCM")

    assert result[[gc._COL_OS_MONTHS, gc._COL_OS_STATUS]].values.tolist() == [
        [12.0, 1.0]
    ]


def test_compute_single_cohort_driver_stats_reports_mutually_exclusive_percentages() -> None:
    clinical = pd.DataFrame(
        {
            "mut_BRAF": [1, 1, 0, 0],
            "mut_BRAF_V600": [1, 1, 0, 0],
            "mut_BRAF_V600E": [1, 0, 0, 0],
            "mut_NRAS": [0, 0, 1, 0],
            "mut_NF1": [0, 0, 0, 1],
        }
    )

    result, label = gc._compute_single_cohort_driver_stats(clinical, "Riaz 2017")

    assert label == "Riaz 2017 (N=4)"
    assert result["BRAF V600E"] == 25.0
    assert result["BRAF other V600"] == 25.0
    assert result["BRAF non-V600"] == 0.0
    assert result["NRAS"] == 25.0
    assert result["NF1"] == 25.0
    assert result["Triple-WT"] == 0.0


def test_calculate_driver_frequencies_excludes_reference_and_unprofiled_cohorts() -> None:
    profiled = pd.DataFrame(
        {
            "mut_BRAF": [1, 0],
            "mut_NRAS": [0, 1],
            "mut_NF1": [0, 0],
            "mut_BRAF_V600": [1, 0],
            "mut_BRAF_V600E": [1, 0],
        }
    )
    cohorts = {
        "TCGA-SKCM": profiled,
        "Riaz 2017": profiled,
        "Gide 2019": pd.DataFrame(
            {"mut_BRAF": [0], "mut_NRAS": [0], "mut_NF1": [0]}
        ),
    }

    frequencies, labels = gc._calculate_driver_frequencies(cohorts)

    assert labels == {"Riaz 2017": "Riaz 2017 (N=2)"}
    assert set(frequencies["Gene"]) == {
        "BRAF V600E",
        "BRAF other V600",
        "BRAF non-V600",
        "NRAS",
        "NF1",
        "Triple-WT",
    }


def test_prepare_tmb_response_df_adds_cohort_and_response_labels() -> None:
    cohorts = {
        "Riaz 2017": pd.DataFrame(
            {gc._COL_TMB: [5.0, 10.0], gc._COL_RESPONSE: [1.0, 0.0]}
        ),
        "TCGA-SKCM": pd.DataFrame(
            {gc._COL_TMB: [20.0], gc._COL_RESPONSE: [1.0]}
        ),
    }

    result = gc._prepare_tmb_response_df(cohorts)

    assert result["Base_Cohort"].tolist() == ["Riaz 2017", "Riaz 2017"]
    assert result["Response"].tolist() == [
        "Responder (CR/PR)",
        "Non-responder (PD)",
    ]
    assert result["Cohort"].iloc[0] == "Riaz 2017\n(N=2)"


def test_prepare_neoantigen_df_sums_available_subtypes() -> None:
    cohorts = {
        "Riaz 2017": pd.DataFrame(
            {
                gc._COL_TMB: [5.0, 10.0],
                "SNV_NEOANTIGEN": [2.0, 3.0],
                "INDEL_NEOANTIGEN": [1.0, 4.0],
            }
        )
    }

    result = gc._prepare_neoantigen_df(cohorts)

    assert result["TOTAL_NEOANTIGEN"].tolist() == [3.0, 7.0]


def test_extract_pooled_biomarkers_requires_more_than_one_feature() -> None:
    cohorts = {
        "Riaz 2017": pd.DataFrame(
            {
                gc._COL_TMB: [5.0, np.nan],
                "SNV_NEOANTIGEN": [2.0, 3.0],
            }
        ),
        "Gide 2019": pd.DataFrame({gc._COL_TMB: [8.0]}),
    }

    result = gc._extract_pooled_biomarkers(cohorts)

    assert result.to_dict("records") == [
        {gc._COL_TMB: 5.0, "SNV_NEOANTIGEN": 2.0}
    ]


def test_compute_neoantigen_correlations_returns_expected_spearman_values() -> None:
    cohorts = {
        "Riaz 2017": pd.DataFrame(
            {
                gc._COL_TMB: [1.0, 2.0, 3.0],
                "SNV_NEOANTIGEN": [1.0, 2.0, 3.0],
                "INDEL_NEOANTIGEN": [3.0, 2.0, 1.0],
                "CTA_SELF_NEOANTIGEN": [1.0, 1.0, 2.0],
            }
        )
    }

    total, snv, indel, cta = gc._compute_neoantigen_correlations(cohorts)

    assert total == pytest.approx(0.8660254)
    assert snv == pytest.approx(1.0)
    assert indel == pytest.approx(-1.0)
    assert cta == pytest.approx(0.8660254)


def test_compute_neoantigen_correlations_returns_nan_without_usable_data() -> None:
    result = gc._compute_neoantigen_correlations(
        {"Riaz 2017": pd.DataFrame({"value": [1, 2]})}
    )

    assert all(np.isnan(value) for value in result)


def test_extract_report_subset_counts_tracks_each_analysis_population() -> None:
    cohorts = {
        "Riaz 2017": pd.DataFrame(
            {
                gc._COL_TMB: [5.0, 10.0],
                gc._COL_RESPONSE: [1.0, np.nan],
                "SNV_NEOANTIGEN": [2.0, 3.0],
                "INDEL_NEOANTIGEN": [1.0, 4.0],
            }
        )
    }

    assert gc._extract_report_subset_counts(cohorts, ["Riaz 2017"]) == (
        1,
        1,
        2,
        2,
        1,
    )
