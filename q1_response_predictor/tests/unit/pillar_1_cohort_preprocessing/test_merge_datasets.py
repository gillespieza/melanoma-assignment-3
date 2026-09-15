"""Unit tests for merge_datasets.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.pillar_1_cohort_preprocessing import merge_datasets as md
from src.config.datasets import DatasetConfig


def test_normalise_os_status_maps_common_labels() -> None:
    values = pd.Series(["1:DECEASED", "alive", "0", "unknown", np.nan])

    result = md._normalise_os_status(values)

    assert result.iloc[:3].tolist() == [1.0, 0.0, 0.0]
    assert pd.isna(result.iloc[3])
    assert pd.isna(result.iloc[4])


def test_normalise_os_status_rejects_non_binary_numeric_values() -> None:
    with pytest.raises(ValueError, match="Expected only 0, 1"):
        md._normalise_os_status(pd.Series([0, 2]))


def test_normalise_sex_maps_labels_and_unknowns() -> None:
    values = pd.Series(["male", "F", "not recorded", np.nan])

    result = md._normalise_sex(values)

    assert result.tolist() == ["Male", "Female", "N/A", "N/A"]


def test_normalise_specimen_type_classifies_primary_and_metastatic() -> None:
    values = pd.Series(["Primary tumour", "metastasis", "", np.nan, "blood"])

    result = md._normalise_specimen_type(values)

    assert result.tolist() == ["Primary", "Metastatic", "N/A", "N/A", "N/A"]


def test_harmonise_clinical_preserves_response_and_normalises_fields(
    dummy_dataset_config: DatasetConfig,
) -> None:
    clinical = pd.DataFrame(
        {
            "PATIENT_ID": ["P1"],
            "RESPONSE": ["Partial Response"],
            "RESPONSE_BINARY": ["1"],
            "OS_MONTHS": ["12.5"],
            "OS_STATUS": ["DECEASED"],
            "AGE": ["61"],
            "SEX": ["f"],
            "SAMPLE_TYPE": ["metastatic lesion"],
            "TMB_NONSYNONYMOUS": ["8"],
            "TX_TYPE_IMMUNOTHERAPY": [1],
        },
        index=pd.Index(["S1"], name="SAMPLE_ID"),
    )

    result = md._harmonise_clinical(clinical, dummy_dataset_config)

    assert result.loc["S1", "RESPONSE"] == "Partial Response"
    assert result.loc["S1", "RESPONSE_BINARY"] == 1
    assert result.loc["S1", "OS_MONTHS"] == 12.5
    assert result.loc["S1", "OS_STATUS"] == 1.0
    assert result.loc["S1", "SEX"] == "Female"
    assert result.loc["S1", "SPECIMEN_TYPE"] == "Metastatic"
    assert result.loc["S1", "TMB_NONSYNONYMOUS"] == 8
    assert result.loc["S1", "IMMUNOTHERAPY"] == 1


def test_resolve_immunotherapy_column_accepts_binary_values(
    dummy_dataset_config: DatasetConfig,
) -> None:
    clinical = pd.DataFrame({"TX_TYPE_IMMUNOTHERAPY": [0.0, 1.0, None]})

    result = md._resolve_immunotherapy_column(clinical, dummy_dataset_config)

    assert result.iloc[:2].tolist() == [0.0, 1.0]
    assert pd.isna(result.iloc[2])


def test_resolve_immunotherapy_column_rejects_non_binary_values(
    dummy_dataset_config: DatasetConfig,
) -> None:
    clinical = pd.DataFrame({"TX_TYPE_IMMUNOTHERAPY": [0, 2]})

    with pytest.raises(ValueError, match="Expected only 0, 1"):
        md._resolve_immunotherapy_column(clinical, dummy_dataset_config)


def test_resolve_immunotherapy_column_uses_dataset_fallback(
    dummy_dataset_config: DatasetConfig,
) -> None:
    dataset = dummy_dataset_config
    result = md._resolve_immunotherapy_column(pd.DataFrame(index=["S1", "S2"]), dataset)

    assert result.tolist() == [0, 0]


def test_zscore_expression_standardises_each_gene() -> None:
    expression = pd.DataFrame(
        {"G1": [1.0, 2.0, 3.0], "CONSTANT": [4.0, 4.0, 4.0]},
        index=["S1", "S2", "S3"],
    )

    result = md.zscore_expression(expression)

    np.testing.assert_allclose(result["G1"].mean(), 0.0)
    np.testing.assert_allclose(result["G1"].std(), 1.0)
    assert result["CONSTANT"].tolist() == [0.0, 0.0, 0.0]


def test_find_common_genes_returns_sorted_intersection() -> None:
    expression_data = {
        "A": pd.DataFrame([[1, 2, 3]], columns=["G2", "G1", "SHARED"]),
        "B": pd.DataFrame([[4, 5, 6]], columns=["SHARED", "G3", "G1"]),
    }

    assert md.find_common_genes(expression_data) == ["G1", "SHARED"]


def test_find_common_genes_rejects_empty_selected_cohort() -> None:
    expression_data = {
        "A": pd.DataFrame(columns=["SHARED"]),
        "B": pd.DataFrame({"SHARED": [1.0]}),
    }

    assert md.find_common_genes(expression_data) == []


def test_build_genomic_features_maps_mutations_and_fills_missing_samples(
    dummy_dataset_config: DatasetConfig,
) -> None:
    clinical = pd.DataFrame(
        {"PATIENT_ID": ["P1", "P2"], "TMB_NONSYNONYMOUS": [5.0, np.nan]},
        index=pd.Index(["S1", "S2"], name="SAMPLE_ID"),
    )
    mutations = pd.DataFrame({"BRAF": [1]}, index=["S1"])

    result = md._build_genomic_features(
        [dummy_dataset_config],
        {dummy_dataset_config.cohort_name: clinical},
        {dummy_dataset_config.cohort_name: mutations},
    )

    assert result.columns.is_unique
    assert result["mut_BRAF"].tolist() == [1, 0]
    assert pd.isna(result.loc[result["SAMPLE_ID"] == "S2", "TMB_NONSYNONYMOUS"]).all()


def test_filter_cohort_samples_selects_immunotherapy_rows(
    dummy_dataset_config: DatasetConfig,
) -> None:
    name = dummy_dataset_config.cohort_name
    expression = pd.DataFrame({"G1": [1.0, 2.0]}, index=["S1", "S2"])
    clinical = pd.DataFrame({"IMMUNOTHERAPY": [1, 0]}, index=["S1", "S2"])
    mutations = pd.DataFrame({"BRAF": [1, 0]}, index=["S1", "S2"])

    selected_expr, selected_clin, selected_mut = md._filter_cohort_samples(
        [dummy_dataset_config],
        {name: expression},
        {name: clinical},
        {name: mutations},
        immunotherapy_only=True,
    )

    assert selected_expr[name].index.tolist() == ["S1"]
    assert selected_clin[name].index.tolist() == ["S1"]
    assert selected_mut[name].index.tolist() == ["S1"]


def test_scale_and_concat_expression_rejects_empty_selection(
    dummy_dataset_config: DatasetConfig,
) -> None:
    selected_expression = {dummy_dataset_config.cohort_name: pd.DataFrame()}
    selected_clinical = {dummy_dataset_config.cohort_name: pd.DataFrame()}

    with pytest.raises(
        ValueError,
        match="no selected cohorts contain samples",
    ):
        md._scale_and_concat_expression(
            [dummy_dataset_config],
            selected_expression,
            selected_clinical,
            ["GENE1"],
        )


def test_scale_and_concat_expression_keeps_expression_and_clinical_aligned(
    dummy_dataset_config: DatasetConfig,
) -> None:
    name = dummy_dataset_config.cohort_name
    expression = pd.DataFrame({"G1": [1.0, 3.0]}, index=["S1", "S2"])
    clinical = pd.DataFrame({"COHORT": [name, name]}, index=["S1", "S2"])

    merged_expression, merged_clinical = md._scale_and_concat_expression(
        [dummy_dataset_config],
        {name: expression},
        {name: clinical},
        ["G1"],
    )

    assert merged_expression.index.tolist() == ["S1", "S2"]
    assert merged_clinical.index.tolist() == ["S1", "S2"]
    np.testing.assert_allclose(merged_expression["G1"].mean(), 0.0)
