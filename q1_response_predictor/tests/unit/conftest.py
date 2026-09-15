"""Unit test shared fixtures for q1_response_predictor."""

from pathlib import Path
import pytest

from src.config.datasets import DatasetConfig


@pytest.fixture
def dummy_dataset_config() -> DatasetConfig:
    """Provide a standard DatasetConfig instance for unit testing."""
    return DatasetConfig(
        cohort_name="Test Cohort",
        study_id="study_123",
        raw_directory="test_raw_dir",
        processed_directory="processed",
        expression_file="data_mrna_seq_v2_rsem.txt",
        clinical_file="data_clinical_patient.txt",
        clinical_sample_file="data_clinical_sample.txt",
        mutation_file="data_mutations.txt",
        processing_strategy="raw",
    )


@pytest.fixture
def mock_raw_dir(tmp_path: Path) -> Path:
    """Provide an isolated temporary raw data directory."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir
