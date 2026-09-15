from pathlib import Path

import pytest

from src.config.datasets import DatasetConfig
from scripts.pillar-1-cohort-preprocessing import download_data as dd


@pytest.fixture
def dataset() -> DatasetConfig:
    return DatasetConfig(
        cohort_name="Test Cohort",
        study_id="study_123",
        raw_directory="test_raw_dir",
        processed_directory="processed",
        expression_file="expr.txt",
        clinical_file="clinical.txt",
        clinical_sample_file="clinical_sample.txt",
        mutation_file="mutations.txt",
        processing_strategy="raw",
    )


def test_is_dataset_present_false_when_missing_target_dir(dataset: DatasetConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
    assert dd.is_dataset_present(dataset) is False


def test_is_dataset_present_false_for_empty_or_ignored_contents(dataset: DatasetConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
    target = tmp_path / dataset.raw_directory
    target.mkdir(parents=True)
    (target / "desktop.ini").write_text("x", encoding="utf-8")
    (target / ".DS_Store").write_text("x", encoding="utf-8")
    (target / ".hidden").write_text("x", encoding="utf-8")
    assert dd.is_dataset_present(dataset) is False


def test_is_dataset_present_true_when_valid_file_exists(dataset: DatasetConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dd, "RAW_DIR", tmp_path)
    target = tmp_path / dataset.raw_directory
    target.mkdir(parents=True)
    (target / "data_clinical_patient.txt").write_text("ok", encoding="utf-8")
    assert dd.is_dataset_present(dataset) is True


def test_validate_dataset_paths_rejects_path_traversal(dataset: DatasetConfig) -> None:
    bad = DatasetConfig(**{**dataset.__dict__, "study_id": "../evil"})
    with pytest.raises(ValueError, match="Invalid path characters"):
        dd._validate_dataset_paths(bad)


def test_download_and_extract_dataset_skips_when_present(dataset: DatasetConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dd, "is_dataset_present", lambda _d: True)
    called = {"flow": False}

    def _flow(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        called["flow"] = True

    monkeypatch.setattr(dd, "_download_and_extract_flow", _flow)
    dd.download_and_extract_dataset(dataset)
    assert called["flow"] is False
