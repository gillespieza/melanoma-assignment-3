"""Download and extract datasets configured for the Q1 subproject.

Dataset metadata is defined in:

    config/datasets.yaml

This script deliberately contains no dataset-specific study IDs or dataset
lists. To add another cBioPortal dataset, add a new entry to datasets.yaml.

Expected project structure:

    📂 melanoma-assignment-3/                  <- PROJECT_ROOT
    ├── 📂 data/
    │   └── 📂 raw/                            <- shared raw data
    
    │
    └── 📂 q1-response-predictor/              <- SUBPROJECT_ROOT
        ├── 📂 config/
        │   └── datasets.yaml
        ├── 📂 logs/
        ├── 📂 scripts/
        │   └── download_data.py
        └── 📂 src/
            ├── 📂 config/
            └── 📂 utils/

Adding a new dataset
--------------------

1. Add a new dataset entry to ``config/datasets.yaml``.
2. Define its cBioPortal study ID.
3. Define the target directory name under ``data/raw/``.
4. Run this script again.

For example:

    datasets:
      - cohort_name: "New Cohort"
        study_id: "cbioportal_study_id"
        raw_directory: "new_cohort"

The download URL is generated automatically from the study ID:

    https://datahub.assets.cbioportal.org/{study_id}.tar.gz

No changes to this script should be required when adding a standard
cBioPortal DataHub dataset.
"""

import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Bootstrap imports
#
# This script lives inside the subproject:
#
#     q1-response-predictor/scripts/download_data.py
#
# The source package is:
#
#     q1-response-predictor/src/
#
# Therefore the subproject root must be added to sys.path before importing
# src.* modules.
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SUBPROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from src.config.datasets import (
    DatasetConfig,
    load_dataset_config,
)
from src.utils.io import (
    download_file,
    extract_tar_gz,
)
from src.utils.logging import setup_logging
from src.utils.paths import (
    RAW_DIR,
    get_subproject_log_dir,
    rel_path,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Resolve config relative to script location.
CONFIG_PATH = SUBPROJECT_ROOT / "config" / "datasets.yaml"

LOG_DIR = get_subproject_log_dir(Path(__file__))
LOG_PATH = LOG_DIR / "download_data.log"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CBIOPORTAL_DATAHUB_URL = (
    "https://datahub.assets.cbioportal.org"
)

MAX_REORGANISATION_RETRIES = 5
REORGANISATION_RETRY_DELAY_SECONDS = 2

# Temporary staging directory created inside data/raw/ during archive extraction.
# Prefixed with underscore so it cannot be confused with a real dataset directory.
_EXTRACTION_STAGING_DIR = "_extracted"

# OS metadata and temporary files ignored when checking dataset presence.
_IGNORED_DATASET_FILES = frozenset({
    "desktop.ini",
    "thumbs.db",
    ".ds_store",
    ".dropbox",
    ".dropbox.attr",
    ".dropbox.cache",
})

_REQUIRED_DATASET_ATTRIBUTES = (
    "expression_file",
    "clinical_file",
    "clinical_sample_file",
)


def is_dataset_present(dataset: DatasetConfig) -> bool:
    """Return whether the configured dataset already contains data.

    A dataset is considered present when all files required by the cleaning
    pipeline exist and are non-empty. Mutation data is intentionally not part
    of this check because some configured cohorts do not provide it.
    """
    target_dir = RAW_DIR / dataset.raw_directory

    if not target_dir.exists() or not target_dir.is_dir():
        return False

    return _has_required_dataset_files(target_dir, dataset)


def _has_required_dataset_files(
    dataset_dir: Path,
    dataset: DatasetConfig,
) -> bool:
    """Return whether a staged or installed dataset has required files."""
    for attribute in _REQUIRED_DATASET_ATTRIBUTES:
        filename = getattr(dataset, attribute)
        path = dataset_dir / filename
        if (
            not path.is_file()
            or path.stat().st_size == 0
            or path.name.lower() in _IGNORED_DATASET_FILES
        ):
            return False
    return True


def _handle_remove_readonly(func: object, path: object, exc_info: object) -> None:
    """Clear read-only attribute on file and retry removal (Windows/cloud-sync fix)."""
    del exc_info
    os.chmod(str(path), stat.S_IWRITE)
    time.sleep(0.05)
    func(path)


def _attempt_remove_path(target_path: Path) -> None:
    """Attempt single removal of file or directory."""
    if target_path.is_dir():
        shutil.rmtree(target_path, onerror=_handle_remove_readonly)
    else:
        os.chmod(target_path, stat.S_IWRITE)
        target_path.unlink()

    if target_path.exists():
        raise PermissionError(
            f"Path {target_path} still exists after removal attempt "
            "(file may be locked by Dropbox, OneDrive, or an active process)."
        )


def _run_with_retry(
    operation: Callable[[], object],
    operation_name: str,
    failure_description: str,
    retries: int = MAX_REORGANISATION_RETRIES,
    delay_seconds: float = REORGANISATION_RETRY_DELAY_SECONDS,
) -> None:
    last_error: OSError | None = None
    for attempt in range(1, retries + 1):
        try:
            operation()
            return
        except OSError as exc:
            last_error = exc
            if attempt < retries:
                wait_seconds = delay_seconds * attempt
                print(
                    f"  {operation_name} attempt {attempt}/{retries} failed "
                    f"({failure_description}: {exc}). "
                    f"Retrying in {wait_seconds:.1f}s..."
                )
                time.sleep(wait_seconds)

    raise OSError(
        f"Failed to {operation_name.lower()} after {retries} attempts "
        f"({failure_description})."
    ) from last_error


def remove_path_with_retry(
    target_path: Path,
    retries: int = MAX_REORGANISATION_RETRIES,
    delay_seconds: float = REORGANISATION_RETRY_DELAY_SECONDS,
) -> None:
    """Remove a file or directory with read-only and retry handling."""
    if not target_path.exists():
        return

    _run_with_retry(
        operation=lambda: _attempt_remove_path(target_path),
        operation_name=f"Remove {rel_path(target_path)}",
        failure_description="cloud sync/OS lock",
        retries=retries,
        delay_seconds=delay_seconds,
    )


def remove_existing_dataset_directory(target_dir: Path) -> None:
    """Remove an existing dataset directory before clean extraction."""
    if not target_dir.exists():
        return

    print(
        f"Removing existing directory "
        f"{rel_path(target_dir)} "
        "to ensure a clean extraction..."
    )

    remove_path_with_retry(target_dir)


def move_with_retry(
    source: Path,
    destination: Path,
    retries: int = MAX_REORGANISATION_RETRIES,
    delay_seconds: float = REORGANISATION_RETRY_DELAY_SECONDS,
) -> None:
    """Move a file or directory with retry handling."""
    _run_with_retry(
        operation=lambda: shutil.move(source, destination),
        operation_name=f"Move {source.name}",
        failure_description=f"destination {destination}",
        retries=retries,
        delay_seconds=delay_seconds,
    )


def _move_extracted_items(extracted_items: list[Path], target_dir: Path) -> None:
    """Move all extracted items to destination directory."""
    for item in extracted_items:
        destination = target_dir / item.name
        if destination.exists():
            remove_path_with_retry(destination)
        move_with_retry(item, destination)


def reorganise_extracted_dataset(
    extracted_dir: Path,
    target_dir: Path,
) -> None:
    """Move extracted dataset contents into the configured target directory."""
    if not extracted_dir.exists():
        raise FileNotFoundError(f"Extracted dataset dir missing: {rel_path(extracted_dir)}")

    if not extracted_dir.is_dir():
        raise NotADirectoryError(f"Extracted path not a dir: {rel_path(extracted_dir)}")

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reorganising files from {extracted_dir.name} to {rel_path(target_dir)}...")

    extracted_items = list(extracted_dir.iterdir())
    if not extracted_items:
        raise FileNotFoundError(f"Extracted dataset dir is empty: {rel_path(extracted_dir)}")

    _move_extracted_items(extracted_items, target_dir)
    remove_path_with_retry(extracted_dir)
    print(f"Reorganisation of {extracted_dir.name} complete.")


def _replace_dataset_directory(
    candidate_dir: Path,
    target_dir: Path,
) -> None:
    """Install a validated candidate while preserving rollback capability."""
    backup_dir = target_dir.with_name(f"{target_dir.name}.backup")
    if backup_dir.exists():
        remove_path_with_retry(backup_dir)

    target_was_moved = False
    try:
        if target_dir.exists():
            move_with_retry(target_dir, backup_dir)
            target_was_moved = True
        move_with_retry(candidate_dir, target_dir)
    except OSError:
        if target_was_moved and not target_dir.exists():
            move_with_retry(backup_dir, target_dir)
        raise
    else:
        if backup_dir.exists():
            remove_path_with_retry(backup_dir)


def _validate_dataset_paths(dataset: DatasetConfig) -> None:
    """Validate identifiers remain single path components under raw-data root."""
    for field_name, value in [
        ("study_id", dataset.study_id),
        ("raw_directory", dataset.raw_directory),
    ]:
        component = Path(value)
        if (
            not value.strip()
            or component.name != value
            or component in (Path("."), Path(".."))
        ):
            raise ValueError(f"Invalid path component in dataset {field_name}: {value!r}")

    raw_root = RAW_DIR.resolve()
    target_dir = (raw_root / dataset.raw_directory).resolve()
    if target_dir.parent != raw_root:
        raise ValueError(
            f"Dataset raw_directory must remain under {raw_root}: "
            f"{dataset.raw_directory!r}"
        )


def _download_and_extract_flow(
    dataset: DatasetConfig,
    target_dir: Path,
    tar_path: Path,
    extraction_root: Path,
    extracted_dir: Path,
) -> None:
    """Execute sequence of download, extract, and reorganisation."""
    if extracted_dir.exists():
        print(f"Removing previous incomplete extraction: {rel_path(extracted_dir)}")
        remove_path_with_retry(extracted_dir)

    url = f"{CBIOPORTAL_DATAHUB_URL}/{dataset.study_id}.tar.gz"
    download_file(url, tar_path)
    extraction_root.mkdir(parents=True, exist_ok=True)
    extract_tar_gz(tar_path=tar_path, extract_to=extraction_root)
    candidate_dir = extraction_root / f"{dataset.study_id}_ready"
    reorganise_extracted_dataset(extracted_dir=extracted_dir, target_dir=candidate_dir)
    if not _has_required_dataset_files(candidate_dir, dataset):
        raise ValueError(
            f"Extracted dataset for {dataset.cohort_name} is incomplete; "
            "required expression and clinical files are missing or empty."
        )
    _replace_dataset_directory(candidate_dir, target_dir)
    print(f"Successfully downloaded and prepared {dataset.cohort_name}.")


def download_and_extract_dataset(dataset: DatasetConfig) -> None:
    """Download, extract, and reorganise one configured dataset."""
    _validate_dataset_paths(dataset)
    target_dir = RAW_DIR / dataset.raw_directory
    tar_path = RAW_DIR / f"{dataset.study_id}.tar.gz"
    extraction_root = RAW_DIR / _EXTRACTION_STAGING_DIR
    extracted_dir = extraction_root / dataset.study_id

    if is_dataset_present(dataset):
        print(f"Dataset {dataset.cohort_name} already exists in {rel_path(target_dir)}. Skipping.")
        return

    print(
        f"\n{'=' * 60}\nProcessing dataset: {dataset.cohort_name}\n"
        f"Study ID: {dataset.study_id}\n{'=' * 60}"
    )
    try:
        _download_and_extract_flow(dataset, target_dir, tar_path, extraction_root, extracted_dir)
    finally:
        primary_error_active = sys.exc_info()[0] is not None
        cleanup_errors: list[OSError] = []
        try:
            remove_path_with_retry(extraction_root)
        except OSError as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if tar_path.exists():
            try:
                remove_path_with_retry(tar_path)
            except OSError as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            if primary_error_active:
                for cleanup_error in cleanup_errors:
                    print(f"Warning: failed to clean download artifact: {cleanup_error}")
            else:
                raise OSError(
                    f"Failed to clean {len(cleanup_errors)} download artifact(s)."
                ) from cleanup_errors[0]


def setup_directories() -> None:
    """Create the shared raw-data directory."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _process_dataset_loop(datasets: list[DatasetConfig]) -> int:
    """Iterate through dataset list and download each."""
    successful = 0
    for dataset in datasets:
        try:
            download_and_extract_dataset(dataset)
            successful += 1
        except Exception as exc:
            print(f"\n[ERROR] Dataset '{dataset.cohort_name}' failed: {exc}")
            raise
    return successful


def main() -> None:
    """Load configuration and download all configured datasets."""
    print(f"{'=' * 60}\nDownloading Raw cBioPortal Datasets\n{'=' * 60}")
    setup_directories()

    datasets = load_dataset_config(CONFIG_PATH)
    if not datasets:
        raise ValueError(f"No datasets found in configuration file: {CONFIG_PATH}")

    print(f"Loaded {len(datasets)} dataset configuration(s) from {rel_path(CONFIG_PATH)}.")
    successful = _process_dataset_loop(datasets)

    print(
        f"\n{'=' * 60}\nAll datasets processed successfully: "
        f"{successful}/{len(datasets)}.\n{'=' * 60}"
    )


if __name__ == "__main__":
    with setup_logging(LOG_PATH):
        main()