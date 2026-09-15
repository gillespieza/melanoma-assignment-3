"""Unit tests for clean_data.py."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.pillar_1_cohort_preprocessing import clean_data as cd
from src.config.datasets import DatasetConfig


def test_record_attrition_creates_serialisable_record(
    dummy_dataset_config: DatasetConfig,
) -> None:
    records: list[cd.AttritionRecord] = []

    cd._record_attrition(
        records,
        dummy_dataset_config,
        step="Clinical data harmonised",
        n_before=10,
        n_after=7,
        reason="Removed incomplete records.",
    )

    assert records[0].to_dict() == {
        "cohort": "Test Cohort",
        "step": "Clinical data harmonised",
        "n_before": 10,
        "n_after": 7,
        "n_removed": 3,
        "reason": "Removed incomplete records.",
    }


def test_harmonise_clinical_data_prefixes_and_normalises_fields(
    dummy_dataset_config: DatasetConfig,
) -> None:
    dataset = replace(
        dummy_dataset_config,
        patient_prefix="iatlas_",
        sample_prefix="iatlas_",
    )
    clinical = pd.DataFrame(
        {
            "PATIENT_ID": ["p1", "p2"],
            "SAMPLE_ID": ["s1", "s2"],
            "RESPONSE": ["Complete Response", "Progressive Disease"],
            "AGE_AT_DIAGNOSIS": ["61", "not recorded"],
            "SEX": ["m", "female"],
            "EMPTY": [np.nan, np.nan],
        }
    )

    result = cd._harmonise_clinical_data(clinical, dataset)

    assert list(result.index) == ["IATLAS_S1", "IATLAS_S2"]
    assert result["PATIENT_ID"].tolist() == ["IATLAS_P1", "IATLAS_P2"]
    assert result["RESPONSE_BINARY"].tolist() == [1, 0]
    assert result["AGE"].iloc[0] == 61.0
    assert pd.isna(result["AGE"].iloc[1])
    assert result["SEX"].tolist() == ["Male", "Female"]
    assert "EMPTY" not in result.columns


def test_harmonise_clinical_data_filters_baseline_samples(
    dummy_dataset_config: DatasetConfig,
) -> None:
    dataset = replace(dummy_dataset_config, baseline_only=True)
    clinical = pd.DataFrame(
        {
            "PATIENT_ID": ["p1", "p1"],
            "SAMPLE_ID": ["s1", "s2"],
            "SAMPLE_TREATMENT": ["Pre", "On"],
        }
    )

    result = cd._harmonise_clinical_data(clinical, dataset)

    assert list(result.index) == ["S1"]


def test_process_raw_expression_matrix_aggregates_duplicates_and_logs_values(
    dummy_dataset_config: DatasetConfig,
) -> None:
    expression = pd.DataFrame(
        {
            "Hugo_Symbol": ["G1", "G1", "G2"],
            "Entrez_Gene_Id": [1, 1, 2],
            "sample-1": [0, 3, -2],
            "sample-2": [7, 1, 3],
        }
    )

    result = cd._process_raw_expression_matrix(expression, dummy_dataset_config)

    assert result.index.name == "SAMPLE_ID"
    assert list(result.index) == ["SAMPLE-1", "SAMPLE-2"]
    assert list(result.columns) == ["G1", "G2"]
    np.testing.assert_allclose(
        result.loc["SAMPLE-1"].to_numpy(),
        [np.log2(2.5), 0.0],
    )
    np.testing.assert_allclose(
        result.loc["SAMPLE-2"].to_numpy(),
        [np.log2(5.0), 2.0],
    )


def test_process_raw_expression_matrix_rejects_missing_gene_identifier(
    dummy_dataset_config: DatasetConfig,
) -> None:
    expression = pd.DataFrame({"sample-1": [1.0]})

    with pytest.raises(
        ValueError,
        match="does not contain Hugo_Symbol or Entrez_Gene_Id",
    ):
        cd._process_raw_expression_matrix(expression, dummy_dataset_config)


def test_load_iatlas_expression_accepts_entrez_only_file(
    tmp_path: Path,
    dummy_dataset_config: DatasetConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expression_path = tmp_path / "expression.txt"
    expression_path.write_text(
        "Entrez_Gene_Id\tsample-1\n101\t4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cd,
        "map_entrez_to_symbols",
        lambda ids, cache_path: {"101": "BRAF"},
    )

    result = cd._load_iatlas_expression(expression_path, dummy_dataset_config)

    assert list(result.columns) == ["BRAF"]
    assert result.index.tolist() == ["SAMPLE-1"]


def test_load_iatlas_expression_accepts_metadata_prefixed_entrez_file(
    tmp_path: Path,
    dummy_dataset_config: DatasetConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure cBioPortal metadata before the header is handled by the fallback."""
    expression_path = tmp_path / "expression.txt"
    expression_path.write_text(
        "# source=cBioPortal\n"
        "# study=test\n"
        "Entrez_Gene_Id\tsample-1\n"
        "101\t4\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cd,
        "map_entrez_to_symbols",
        lambda ids, cache_path: {"101": "BRAF"},
    )

    result = cd._load_iatlas_expression(expression_path, dummy_dataset_config)

    assert result.loc["SAMPLE-1", "BRAF"] == np.log2(5)


def test_align_and_record_expression_rejects_duplicate_sample_ids(
    dummy_dataset_config: DatasetConfig,
) -> None:
    """Ensure duplicate IDs are rejected by the shared alignment helper."""
    expression = pd.DataFrame(
        {"BRAF": [1.0, 2.0]},
        index=pd.Index(["S1", "S1"], name="SAMPLE_ID"),
    )
    clinical = pd.DataFrame(
        {"PATIENT_ID": ["P1"]},
        index=pd.Index(["S1"], name="SAMPLE_ID"),
    )

    with pytest.raises(ValueError, match="duplicate sample IDs"):
        cd._align_and_record_expression(
            expression,
            clinical,
            dummy_dataset_config,
            [],
        )


def test_add_tcga_hypoxia_features_applies_patient_prefix(
    tmp_path: Path,
    dummy_dataset_config: DatasetConfig,
) -> None:
    """Ensure supplementary hypoxia patient IDs join prefixed clinical IDs."""
    dataset = replace(dummy_dataset_config, patient_prefix="TCGA_")
    (tmp_path / cd._TCGA_HYPOXIA_FILENAME).write_text(
        "PATIENT_ID\tWINTER_HYPOXIA_SCORE\nP1\t2.5\n",
        encoding="utf-8",
    )
    clinical = pd.DataFrame({"PATIENT_ID": ["TCGA_P1"]})

    result = cd._add_tcga_hypoxia_features(clinical, tmp_path, dataset)

    assert result.loc[0, cd._COL_HYPOXIA_SCORE] == 2.5


def test_run_cleaning_pipeline_attempts_all_datasets_then_raises(
    dummy_dataset_config: DatasetConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure one failed cohort does not prevent later cohorts being attempted."""
    second = replace(dummy_dataset_config, cohort_name="Second Cohort")
    attempted: list[str] = []

    def process(dataset: DatasetConfig) -> list[cd.AttritionRecord]:
        attempted.append(dataset.cohort_name)
        if dataset.cohort_name == dummy_dataset_config.cohort_name:
            raise RuntimeError("cohort failed")
        return []

    monkeypatch.setattr(cd, "process_dataset", process)
    monkeypatch.setattr(cd, "_write_attrition_output", lambda *_args: None)

    with pytest.raises(RuntimeError, match="Test Cohort"):
        cd._run_cleaning_pipeline([dummy_dataset_config, second])

    assert attempted == ["Test Cohort", "Second Cohort"]


def test_write_processed_outputs_preserves_expected_csv_indexes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure clinical output flattens its index while matrices retain it."""
    written: dict[str, bool] = {}

    def save_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
        written[path.name] = index

    monkeypatch.setattr(cd, "safe_save_csv", save_csv)
    cd._write_processed_outputs(
        tmp_path,
        pd.DataFrame({"PATIENT_ID": ["P1"]}, index=pd.Index(["S1"], name="SAMPLE_ID")),
        pd.DataFrame({"BRAF": [1.0]}, index=pd.Index(["S1"], name="SAMPLE_ID")),
        pd.DataFrame({"BRAF": [1]}, index=pd.Index(["S1"], name="SAMPLE_ID")),
    )

    assert written == {
        "clin_cleaned.csv": False,
        "expr_cleaned.csv": True,
        "mutations_cleaned.csv": True,
    }


def test_map_tcga_entrez_identifiers_maps_and_retains_unmapped_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expression = pd.DataFrame(
        {
            "Entrez_Gene_Id": [101, 202],
            "S1": [4.0, 2.0],
        }
    )
    monkeypatch.setattr(
        cd,
        "map_entrez_to_symbols",
        lambda ids, cache_path: {"101": "BRAF"},
    )

    result = cd._map_tcga_entrez_identifiers(expression, tmp_path / "cache.json")

    assert list(result.index) == ["BRAF", "202"]
    assert "Entrez_Gene_Id" not in result.columns


def test_process_mutations_returns_clinical_index_when_file_is_missing(
    tmp_path: Path,
    dummy_dataset_config: DatasetConfig,
) -> None:
    clinical = pd.DataFrame(index=pd.Index(["S1"], name="SAMPLE_ID"))

    result = cd._process_mutations(tmp_path, clinical, dummy_dataset_config)

    assert result.empty
    assert result.index.tolist() == ["S1"]
    assert result.index.name == "SAMPLE_ID"


def test_aggregate_mutations_to_patient_level_reexpands_to_samples() -> None:
    clinical = pd.DataFrame(
        {
            "PATIENT_ID": ["P1", "P1", "P2"],
        },
        index=pd.Index(["P1_S1", "P1_S2", "P2_S1"], name="SAMPLE_ID"),
    )
    mutations = pd.DataFrame(
        {"BRAF": [1, 0], "NRAS": [0, 1]},
        index=["P1", "P2"],
    )

    result = cd._aggregate_mutations_to_patient_level(mutations, clinical)

    assert result.index.tolist() == ["P1_S1", "P1_S2", "P2_S1"]
    assert result["BRAF"].tolist() == [1, 1, 0]
    assert result["NRAS"].tolist() == [0, 0, 1]


@pytest.mark.parametrize(
    ("sample_id", "expected"),
    [
        ("TCGA-AB-1234-01", "TCGA-AB-1234"),
        ("IATLAS_PATIENT_SAMPLE", "IATLAS_PATIENT"),
        ("SINGLE_SAMPLE", "SINGLE"),
        ("unknown", "unknown"),
        (None, None),
    ],
)
def test_sample_to_patient_id(sample_id: object, expected: object) -> None:
    assert cd._sample_to_patient_id(sample_id) == expected


def test_build_treatment_features_selects_best_outcome_and_flags_agents() -> None:
    treatment = pd.DataFrame(
        {
            "PATIENT_ID": ["P1", "P1", "P2"],
            "TREATMENT_TYPE": ["Immunotherapy", "Chemotherapy", "Immunotherapy"],
            "AGENT": ["Nivolumab", "Dacarbazine", "Pembrolizumab"],
            "MEASURE_OF_RESPONSE": ["Stable Disease", "Complete Response", "Partial Response"],
        }
    )

    result = cd._build_treatment_summary_features(treatment)
    result = cd._build_treatment_type_indicators(treatment, result)
    result = cd._build_key_agent_indicators(treatment, result)

    assert result.loc["P1", "TREATMENT_OUTCOME"] == "Complete Response"
    assert result.loc["P1", "TX_IMMUNOTHERAPY_OUTCOME"] == "Stable Disease"
    assert result.loc["P1", "TX_TYPE_IMMUNOTHERAPY"] == 1
    assert result.loc["P1", "TX_AGENT_NIVOLUMAB"] == 1
    assert result.loc["P2", "TX_AGENT_PEMBROLIZUMAB"] == 1


def test_map_durable_benefit_to_response_does_not_overwrite_existing_response() -> None:
    clinical = pd.DataFrame(
        {
            "RESPONSE": ["Stable Disease"],
            "DURABLE_CLINICAL_BENEFIT": ["CR"],
        }
    )

    result = cd._map_durable_benefit_to_response(clinical)

    assert result["RESPONSE"].tolist() == ["Stable Disease"]
    assert "RESPONSE_BINARY" in result.columns
