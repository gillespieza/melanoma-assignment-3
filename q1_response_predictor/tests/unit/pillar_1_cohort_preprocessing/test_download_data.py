"""Unit tests for download_data.py in scripts/pillar_1_cohort_preprocessing/."""

from __future__ import annotations

import tarfile
from dataclasses import replace
import os
import stat
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.pillar_1_cohort_preprocessing import download_data as dd
from src.config.datasets import DatasetConfig


# ---------------------------------------------------------------------------
# Tests for is_dataset_present
# ---------------------------------------------------------------------------


def test_is_dataset_present_false_when_missing_target_dir(
    dummy_dataset_config: DatasetConfig,
    mock_raw_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure returns False when raw directory does not exist."""
    monkeypatch.setattr(dd, "RAW_DIR", mock_raw_dir)
    assert dd.is_dataset_present(dummy_dataset_config) is False


def test_is_dataset_present_false_when_target_is_file_not_dir(
    dummy_dataset_config: DatasetConfig,
    mock_raw_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure returns False when the raw directory path is a file, not a directory."""
    monkeypatch.setattr(dd, "RAW_DIR", mock_raw_dir)
    target = mock_raw_dir / dummy_dataset_config.raw_directory
    target.write_text("not a directory", encoding="utf-8")
    assert dd.is_dataset_present(dummy_dataset_config) is False


def test_is_dataset_present_false_for_empty_directory(
    dummy_dataset_config: DatasetConfig,
    mock_raw_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure returns False when target directory exists but is completely empty."""
    monkeypatch.setattr(dd, "RAW_DIR", mock_raw_dir)
    target = mock_raw_dir / dummy_dataset_config.raw_directory
    target.mkdir(parents=True)
    assert dd.is_dataset_present(dummy_dataset_config) is False


def test_is_dataset_present_false_for_empty_nested_directory(
    dummy_dataset_config: DatasetConfig,
    mock_raw_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure an empty nested directory does not count as dataset data."""
    monkeypatch.setattr(dd, "RAW_DIR", mock_raw_dir)
    target = mock_raw_dir / dummy_dataset_config.raw_directory
    (target / "nested").mkdir(parents=True)
    assert dd.is_dataset_present(dummy_dataset_config) is False


def test_is_dataset_present_false_for_ignored_files(
    dummy_dataset_config: DatasetConfig,
    mock_raw_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure returns False when target directory only contains OS or hidden metadata."""
    monkeypatch.setattr(dd, "RAW_DIR", mock_raw_dir)
    target = mock_raw_dir / dummy_dataset_config.raw_directory
    target.mkdir(parents=True)
    (target / "desktop.ini").write_text("x", encoding="utf-8")
    (target / "thumbs.db").write_text("x", encoding="utf-8")
    (target / ".DS_Store").write_text("x", encoding="utf-8")
    (target / ".dropbox").write_text("x", encoding="utf-8")
    (target / ".dropbox.cache").write_text("x", encoding="utf-8")
    (target / ".hidden_temp").write_text("x", encoding="utf-8")
    (target / "file.dropbox.attr").write_text("x", encoding="utf-8")
    (target / "~$lockfile.tmp").write_text("x", encoding="utf-8")
    (target / "upload.tmp").write_text("x", encoding="utf-8")
    assert dd.is_dataset_present(dummy_dataset_config) is False


def test_is_dataset_present_true_when_valid_data_file_exists(
    dummy_dataset_config: DatasetConfig,
    mock_raw_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure all pipeline-required files make a dataset present."""
    monkeypatch.setattr(dd, "RAW_DIR", mock_raw_dir)
    target = mock_raw_dir / dummy_dataset_config.raw_directory
    target.mkdir(parents=True)
    for filename in (
        dummy_dataset_config.expression_file,
        dummy_dataset_config.clinical_file,
        dummy_dataset_config.clinical_sample_file,
    ):
        (target / filename).write_text("header\nvalue\n", encoding="utf-8")
    assert dd.is_dataset_present(dummy_dataset_config) is True


# ---------------------------------------------------------------------------
# Tests for _validate_dataset_paths
# ---------------------------------------------------------------------------


def test_validate_dataset_paths_accepts_clean_paths(
    dummy_dataset_config: DatasetConfig,
) -> None:
    """Ensure valid standard study IDs and directory names pass without error."""
    dd._validate_dataset_paths(dummy_dataset_config)


@pytest.mark.parametrize("invalid_value", ["../evil", "sub/folder", "sub\\folder", ".."])
def test_validate_dataset_paths_rejects_path_traversal(
    dummy_dataset_config: DatasetConfig,
    invalid_value: str,
) -> None:
    """Ensure directory traversal characters raise ValueError."""
    bad_study = DatasetConfig(**{**dummy_dataset_config.__dict__, "study_id": invalid_value})
    with pytest.raises(ValueError, match="Invalid path component in dataset study_id"):
        dd._validate_dataset_paths(bad_study)

    bad_dir = DatasetConfig(**{**dummy_dataset_config.__dict__, "raw_directory": invalid_value})
    with pytest.raises(ValueError, match="Invalid path component in dataset raw_directory"):
        dd._validate_dataset_paths(bad_dir)


@pytest.mark.parametrize("empty_value", ["", "   "])
def test_validate_dataset_paths_rejects_empty_values(
    dummy_dataset_config: DatasetConfig,
    empty_value: str,
) -> None:
    """Ensure empty path components cannot resolve to a broader raw-data path."""
    with pytest.raises(ValueError, match="Invalid path component in dataset study_id"):
        dd._validate_dataset_paths(replace(dummy_dataset_config, study_id=empty_value))

    with pytest.raises(ValueError, match="Invalid path component in dataset raw_directory"):
        dd._validate_dataset_paths(replace(dummy_dataset_config, raw_directory=empty_value))


# ---------------------------------------------------------------------------
# Tests for path removal and retry logic
# ---------------------------------------------------------------------------


def test_remove_path_with_retry_non_existent(tmp_path: Path) -> None:
    """Ensure non-existent path returns silently without error."""
    non_existent = tmp_path / "does_not_exist"
    dd.remove_path_with_retry(non_existent)


def test_remove_path_with_retry_single_file(tmp_path: Path) -> None:
    """Ensure single existing file is deleted."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("test", encoding="utf-8")
    assert file_path.exists()
    dd.remove_path_with_retry(file_path)
    assert not file_path.exists()


def test_remove_path_with_retry_directory(tmp_path: Path) -> None:
    """Ensure directory and nested files are recursively removed."""
    dir_path = tmp_path / "nested_dir"
    dir_path.mkdir()
    (dir_path / "sub.txt").write_text("child", encoding="utf-8")
    assert dir_path.exists()
    dd.remove_path_with_retry(dir_path)
    assert not dir_path.exists()


def test_remove_path_with_retry_readonly_file(tmp_path: Path) -> None:
    """Ensure read-only files are unlocked and successfully removed on Windows."""
    readonly_file = tmp_path / "readonly.txt"
    readonly_file.write_text("protected", encoding="utf-8")
    os.chmod(readonly_file, stat.S_IREAD)
    dd.remove_path_with_retry(readonly_file)
    assert not readonly_file.exists()


def test_remove_path_with_retry_exhausted_raises_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure OSError is raised when retries are exhausted."""
    target_file = tmp_path / "stubborn.txt"
    target_file.write_text("content", encoding="utf-8")

    def mock_attempt_fail(_path: Path) -> None:
        raise OSError("Permission denied by external process lock")

    monkeypatch.setattr(dd, "_attempt_remove_path", mock_attempt_fail)
    monkeypatch.setattr(dd, "time", MagicMock())

    with pytest.raises(OSError, match="Failed to remove"):
        dd.remove_path_with_retry(target_file, retries=2, delay_seconds=0.01)


def test_remove_existing_dataset_directory(tmp_path: Path) -> None:
    """Ensure remove_existing_dataset_directory removes target directory if it exists."""
    target_dir = tmp_path / "cohort_dir"
    target_dir.mkdir()
    (target_dir / "file.txt").write_text("data", encoding="utf-8")
    dd.remove_existing_dataset_directory(target_dir)
    assert not target_dir.exists()


def test_attempt_remove_path_raises_when_directory_still_exists_due_to_cloud_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure _attempt_remove_path raises PermissionError if directory remains locked by cloud sync."""
    target_dir = tmp_path / "locked_dir"
    target_dir.mkdir()
    (target_dir / "locked_file.txt").write_text("locked", encoding="utf-8")

    # Simulate shutil.rmtree completing without error but failing to actually remove the path
    monkeypatch.setattr(dd.shutil, "rmtree", lambda _path, **_kw: None)

    with pytest.raises(PermissionError, match="locked by Dropbox, OneDrive, or an active process"):
        dd._attempt_remove_path(target_dir)


def test_remove_path_with_retry_recovers_after_transient_cloud_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure remove_path_with_retry successfully retries and recovers from transient cloud sync lock."""
    target_file = tmp_path / "cloud_locked_file.txt"
    target_file.write_text("content", encoding="utf-8")

    attempts = {"count": 0}
    original_attempt_remove = dd._attempt_remove_path

    def mock_attempt_transient(path: Path) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise PermissionError("[WinError 32] The process cannot access the file (Dropbox lock)")
        original_attempt_remove(path)

    monkeypatch.setattr(dd, "_attempt_remove_path", mock_attempt_transient)
    mock_sleep = MagicMock()
    monkeypatch.setattr(dd.time, "sleep", mock_sleep)

    dd.remove_path_with_retry(target_file, retries=3, delay_seconds=0.1)

    assert not target_file.exists()
    assert attempts["count"] == 2
    mock_sleep.assert_called_once_with(0.1)


def test_remove_path_with_retry_uses_incremental_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure each failed removal attempt waits with increasing delay."""
    target_file = tmp_path / "stubborn.txt"
    target_file.write_text("content", encoding="utf-8")
    mock_sleep = MagicMock()
    monkeypatch.setattr(dd.time, "sleep", mock_sleep)
    monkeypatch.setattr(
        dd,
        "_attempt_remove_path",
        MagicMock(side_effect=OSError("locked")),
    )

    with pytest.raises(OSError):
        dd.remove_path_with_retry(target_file, retries=3, delay_seconds=0.2)

    assert mock_sleep.call_args_list == [
        ((0.2,), {}),
        ((0.4,), {}),
    ]


# ---------------------------------------------------------------------------
# Tests for move and reorganise logic
# ---------------------------------------------------------------------------


def test_move_with_retry_success(tmp_path: Path) -> None:
    """Ensure move_with_retry moves file from source to destination."""
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("payload", encoding="utf-8")
    dd.move_with_retry(src, dst)
    assert not src.exists()
    assert dst.exists()
    assert dst.read_text(encoding="utf-8") == "payload"


def test_move_with_retry_exhausted_raises_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure move_with_retry raises OSError when all retries are exhausted."""
    src = tmp_path / "source.txt"
    dst = tmp_path / "destination.txt"
    src.write_text("data", encoding="utf-8")

    def mock_move_fail(_s: Path, _d: Path) -> None:
        raise OSError("Cross-device link or locked file")

    monkeypatch.setattr(dd.shutil, "move", mock_move_fail)
    monkeypatch.setattr(dd, "time", MagicMock())

    with pytest.raises(OSError, match="Failed to move"):
        dd.move_with_retry(src, dst, retries=2, delay_seconds=0.01)


def test_move_with_retry_uses_incremental_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure failed moves retry with increasing delays and the right count."""
    src = tmp_path / "source.txt"
    dst = tmp_path / "destination.txt"
    src.write_text("data", encoding="utf-8")
    mock_sleep = MagicMock()
    monkeypatch.setattr(dd.time, "sleep", mock_sleep)
    mock_move = MagicMock(side_effect=OSError("locked"))
    monkeypatch.setattr(dd.shutil, "move", mock_move)

    with pytest.raises(OSError):
        dd.move_with_retry(src, dst, retries=3, delay_seconds=0.2)

    assert mock_move.call_count == 3
    assert mock_sleep.call_args_list == [
        ((0.2,), {}),
        ((0.4,), {}),
    ]


def test_move_with_retry_recovers_after_transient_cloud_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure move_with_retry successfully retries and recovers from transient cloud sync lock."""
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("payload", encoding="utf-8")

    attempts = {"count": 0}
    original_move = dd.shutil.move

    def mock_move_transient(s: Path, d: Path) -> Path:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise PermissionError("[WinError 32] Sharing violation from cloud synchronisation")
        return original_move(s, d)

    monkeypatch.setattr(dd.shutil, "move", mock_move_transient)
    mock_sleep = MagicMock()
    monkeypatch.setattr(dd.time, "sleep", mock_sleep)

    dd.move_with_retry(src, dst, retries=3, delay_seconds=0.1)

    assert not src.exists()
    assert dst.exists()
    assert dst.read_text(encoding="utf-8") == "payload"
    assert attempts["count"] == 2
    mock_sleep.assert_called_once_with(0.1)


def test_reorganise_extracted_dataset_missing_source_raises_error(tmp_path: Path) -> None:
    """Ensure FileNotFoundError is raised when extracted directory does not exist."""
    missing = tmp_path / "missing_dir"
    target = tmp_path / "target_dir"
    with pytest.raises(FileNotFoundError, match="Extracted dataset dir missing"):
        dd.reorganise_extracted_dataset(missing, target)


def test_reorganise_extracted_dataset_source_is_file_raises_error(tmp_path: Path) -> None:
    """Ensure NotADirectoryError is raised when source is a file."""
    source_file = tmp_path / "extracted_file.txt"
    source_file.write_text("not a dir", encoding="utf-8")
    target = tmp_path / "target_dir"
    with pytest.raises(NotADirectoryError, match="Extracted path not a dir"):
        dd.reorganise_extracted_dataset(source_file, target)


def test_reorganise_extracted_dataset_empty_dir_raises_error(tmp_path: Path) -> None:
    """Ensure FileNotFoundError is raised when extracted directory contains no files."""
    empty_source = tmp_path / "empty_extracted"
    empty_source.mkdir()
    target = tmp_path / "target_dir"
    with pytest.raises(FileNotFoundError, match="Extracted dataset dir is empty"):
        dd.reorganise_extracted_dataset(empty_source, target)


def test_reorganise_extracted_dataset_moves_contents_and_overwrites(tmp_path: Path) -> None:
    """Ensure files are relocated into target directory and extracted staging is removed."""
    extracted_dir = tmp_path / "extracted_staging"
    extracted_dir.mkdir()
    file1 = extracted_dir / "data1.txt"
    file1.write_text("data1", encoding="utf-8")
    file2 = extracted_dir / "data2.txt"
    file2.write_text("data2", encoding="utf-8")

    target_dir = tmp_path / "target_dataset"
    target_dir.mkdir()
    # Pre-existing file with duplicate name to test overwrite behaviour
    (target_dir / "data1.txt").write_text("old_data1", encoding="utf-8")

    dd.reorganise_extracted_dataset(extracted_dir, target_dir)

    assert not extracted_dir.exists()
    assert (target_dir / "data1.txt").exists()
    assert (target_dir / "data1.txt").read_text(encoding="utf-8") == "data1"
    assert (target_dir / "data2.txt").exists()
    assert (target_dir / "data2.txt").read_text(encoding="utf-8") == "data2"


def test_reorganise_extracted_dataset_handles_real_tar_layout(tmp_path: Path) -> None:
    """Ensure a real archive extracts and reorganises into the target layout."""
    archive_root = tmp_path / "archive_root" / "study_123"
    archive_root.mkdir(parents=True)
    (archive_root / "data_clinical_patient.txt").write_text(
        "PATIENT_ID\n1\n", encoding="utf-8"
    )
    archive = tmp_path / "study_123.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(archive_root, arcname="study_123")

    extraction_root = tmp_path / "extracted"
    dd.extract_tar_gz(archive, extraction_root)
    target_dir = tmp_path / "target"
    dd.reorganise_extracted_dataset(extraction_root / "study_123", target_dir)

    assert (target_dir / "data_clinical_patient.txt").read_text(encoding="utf-8") == (
        "PATIENT_ID\n1\n"
    )
    assert not (extraction_root / "study_123").exists()


# ---------------------------------------------------------------------------
# Tests for download_and_extract_dataset and orchestration
# ---------------------------------------------------------------------------


def test_download_and_extract_dataset_skips_when_present(
    dummy_dataset_config: DatasetConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure dataset download is skipped when data files already exist."""
    monkeypatch.setattr(dd, "is_dataset_present", lambda _d: True)
    mock_flow = MagicMock()
    monkeypatch.setattr(dd, "_download_and_extract_flow", mock_flow)

    dd.download_and_extract_dataset(dummy_dataset_config)
    mock_flow.assert_not_called()


def test_download_and_extract_dataset_orchestrates_flow(
    dummy_dataset_config: DatasetConfig,
    mock_raw_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure complete flow is orchestrated when dataset is missing."""
    monkeypatch.setattr(dd, "RAW_DIR", mock_raw_dir)
    monkeypatch.setattr(dd, "is_dataset_present", lambda _d: False)

    mock_download = MagicMock()
    mock_extract = MagicMock()
    mock_reorganise = MagicMock()

    def populate_candidate(*, extracted_dir: Path, target_dir: Path) -> None:
        target_dir.mkdir(parents=True)
        for filename in (
            dummy_dataset_config.expression_file,
            dummy_dataset_config.clinical_file,
            dummy_dataset_config.clinical_sample_file,
        ):
            (target_dir / filename).write_text("header\nvalue\n", encoding="utf-8")

    mock_reorganise.side_effect = populate_candidate
    monkeypatch.setattr(dd, "download_file", mock_download)
    monkeypatch.setattr(dd, "extract_tar_gz", mock_extract)
    monkeypatch.setattr(dd, "reorganise_extracted_dataset", mock_reorganise)

    dd.download_and_extract_dataset(dummy_dataset_config)

    expected_url = f"{dd.CBIOPORTAL_DATAHUB_URL}/{dummy_dataset_config.study_id}.tar.gz"
    expected_tar = mock_raw_dir / f"{dummy_dataset_config.study_id}.tar.gz"
    expected_staging = mock_raw_dir / "_extracted"
    expected_extracted_dir = expected_staging / dummy_dataset_config.study_id
    expected_target = mock_raw_dir / dummy_dataset_config.raw_directory

    mock_download.assert_called_once_with(expected_url, expected_tar)
    mock_extract.assert_called_once_with(tar_path=expected_tar, extract_to=expected_staging)
    mock_reorganise.assert_called_once_with(
        extracted_dir=expected_extracted_dir,
        target_dir=expected_staging / f"{dummy_dataset_config.study_id}_ready",
    )
    # Staging directory must be cleaned up in finally block
    assert not expected_staging.exists()


def test_download_and_extract_dataset_preserves_existing_data_when_candidate_incomplete(
    dummy_dataset_config: DatasetConfig,
    mock_raw_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure an incomplete replacement cannot remove usable existing data."""
    monkeypatch.setattr(dd, "RAW_DIR", mock_raw_dir)
    target = mock_raw_dir / dummy_dataset_config.raw_directory
    target.mkdir()
    existing_file = target / dummy_dataset_config.clinical_file
    existing_file.write_text("existing\n", encoding="utf-8")

    monkeypatch.setattr(dd, "is_dataset_present", lambda _d: False)
    monkeypatch.setattr(dd, "download_file", MagicMock())
    monkeypatch.setattr(dd, "extract_tar_gz", MagicMock())

    def mock_reorganise(*, extracted_dir: Path, target_dir: Path) -> None:
        target_dir.mkdir(parents=True)
        (target_dir / dummy_dataset_config.clinical_file).write_text(
            "partial\n", encoding="utf-8"
        )

    monkeypatch.setattr(dd, "reorganise_extracted_dataset", mock_reorganise)

    with pytest.raises(ValueError, match="dataset .* incomplete"):
        dd.download_and_extract_dataset(dummy_dataset_config)

    assert existing_file.read_text(encoding="utf-8") == "existing\n"


def test_download_and_extract_dataset_cleans_up_on_failure(
    dummy_dataset_config: DatasetConfig,
    mock_raw_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure staging files and partial archive are cleaned up if extraction fails."""
    monkeypatch.setattr(dd, "RAW_DIR", mock_raw_dir)
    monkeypatch.setattr(dd, "is_dataset_present", lambda _d: False)

    tar_path = mock_raw_dir / f"{dummy_dataset_config.study_id}.tar.gz"
    tar_path.write_text("corrupted archive content", encoding="utf-8")

    staging_dir = mock_raw_dir / "_extracted"
    staging_dir.mkdir(parents=True)
    (staging_dir / "partial.txt").write_text("temp", encoding="utf-8")

    def mock_extract_fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Extraction failed due to corrupted gzip stream")

    monkeypatch.setattr(dd, "download_file", MagicMock())
    monkeypatch.setattr(dd, "extract_tar_gz", mock_extract_fail)

    with pytest.raises(RuntimeError, match="corrupted gzip stream"):
        dd.download_and_extract_dataset(dummy_dataset_config)

    assert not tar_path.exists()
    assert not staging_dir.exists()


def test_download_and_extract_dataset_preserves_primary_error_on_cleanup_failure(
    dummy_dataset_config: DatasetConfig,
    mock_raw_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure cleanup errors do not hide the original download failure."""
    monkeypatch.setattr(dd, "RAW_DIR", mock_raw_dir)
    monkeypatch.setattr(dd, "is_dataset_present", lambda _d: False)
    monkeypatch.setattr(
        dd,
        "_download_and_extract_flow",
        MagicMock(side_effect=RuntimeError("primary failure")),
    )
    cleanup = MagicMock(side_effect=OSError("cleanup failure"))
    monkeypatch.setattr(dd, "remove_path_with_retry", cleanup)

    with pytest.raises(RuntimeError, match="primary failure"):
        dd.download_and_extract_dataset(dummy_dataset_config)


def test_download_and_extract_dataset_surfaces_cleanup_failure_without_primary_error(
    dummy_dataset_config: DatasetConfig,
    mock_raw_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure cleanup failures are reported when the download flow succeeds."""
    monkeypatch.setattr(dd, "RAW_DIR", mock_raw_dir)
    monkeypatch.setattr(dd, "is_dataset_present", lambda _d: False)
    monkeypatch.setattr(dd, "_download_and_extract_flow", MagicMock())
    monkeypatch.setattr(
        dd,
        "remove_path_with_retry",
        MagicMock(side_effect=OSError("cleanup failure")),
    )

    with pytest.raises(OSError, match="Failed to clean"):
        dd.download_and_extract_dataset(dummy_dataset_config)


# ---------------------------------------------------------------------------
# Tests for process loop and setup
# ---------------------------------------------------------------------------


def test_setup_directories(mock_raw_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure setup_directories creates the RAW_DIR."""
    new_raw = mock_raw_dir / "new_raw_folder"
    monkeypatch.setattr(dd, "RAW_DIR", new_raw)
    assert not new_raw.exists()
    dd.setup_directories()
    assert new_raw.exists()
    assert new_raw.is_dir()


def test_process_dataset_loop_counts_successes(
    dummy_dataset_config: DatasetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure _process_dataset_loop returns the total count of successfully processed datasets."""
    mock_download = MagicMock()
    monkeypatch.setattr(dd, "download_and_extract_dataset", mock_download)

    datasets = [
        dummy_dataset_config,
        DatasetConfig(**{**dummy_dataset_config.__dict__, "cohort_name": "Cohort 2", "study_id": "study_2"}),
    ]
    success_count = dd._process_dataset_loop(datasets)
    assert success_count == 2
    assert mock_download.call_count == 2


def test_process_dataset_loop_raises_on_error(
    dummy_dataset_config: DatasetConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure _process_dataset_loop halts and re-raises when a dataset download fails."""

    def mock_fail(_d: DatasetConfig) -> None:
        raise ConnectionError("cBioPortal DataHub server unavailable")

    monkeypatch.setattr(dd, "download_and_extract_dataset", mock_fail)

    with pytest.raises(ConnectionError, match="server unavailable"):
        dd._process_dataset_loop([dummy_dataset_config])


def test_main_loads_config_creates_directories_and_processes_datasets(
    dummy_dataset_config: DatasetConfig,
    mock_raw_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure main wires setup, configuration loading, and dataset processing."""
    monkeypatch.setattr(dd, "RAW_DIR", mock_raw_dir / "new_raw")
    monkeypatch.setattr(dd, "CONFIG_PATH", mock_raw_dir / "datasets.yaml")
    monkeypatch.setattr(dd, "load_dataset_config", MagicMock(return_value=(dummy_dataset_config,)))
    process = MagicMock(return_value=1)
    monkeypatch.setattr(dd, "_process_dataset_loop", process)

    dd.main()

    assert (mock_raw_dir / "new_raw").is_dir()
    process.assert_called_once_with((dummy_dataset_config,))


def test_main_raises_when_configuration_has_no_datasets(
    mock_raw_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure main rejects an empty configuration result."""
    monkeypatch.setattr(dd, "RAW_DIR", mock_raw_dir / "new_raw")
    monkeypatch.setattr(dd, "load_dataset_config", MagicMock(return_value=()))

    with pytest.raises(ValueError, match="No datasets found"):
        dd.main()
