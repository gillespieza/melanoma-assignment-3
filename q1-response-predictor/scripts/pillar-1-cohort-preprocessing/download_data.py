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

import contextlib
import os
import shutil
import stat
import sys
import time
from pathlib import Path

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
from src.utils.logging import TeeStream
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
})


def is_dataset_present(dataset: DatasetConfig) -> bool:
    """Return whether the configured dataset already contains data.

    A dataset is considered present when its target directory exists and
    contains at least one valid non-hidden data file (ignoring OS metadata
    files such as desktop.ini or .DS_Store).
    """
    target_dir = RAW_DIR / dataset.raw_directory

    if not target_dir.exists() or not target_dir.is_dir():
        return False

    for item in target_dir.iterdir():
        if item.name.startswith(".") and item.name != "_extracted":
            continue
        if item.name.lower() in _IGNORED_DATASET_FILES:
            continue
        return True

    return False


def _handle_remove_readonly(func: object, path: object, exc_info: object) -> None:
    """Clear read-only attribute on file and retry removal (Windows fix)."""
    try:
        os.chmod(str(path), stat.S_IWRITE)
        func(path)
        _ = exc_info
    except OSError:
        pass


def _attempt_remove_path(target_path: Path) -> None:
    """Attempt single removal of file or directory."""
    if target_path.is_dir():
        shutil.rmtree(target_path, onerror=_handle_remove_readonly)
    else:
        try:
            os.chmod(target_path, stat.S_IWRITE)
        except OSError:
            pass
        target_path.unlink()


def remove_path_with_retry(
    target_path: Path,
    retries: int = MAX_REORGANISATION_RETRIES,
    delay_seconds: float = REORGANISATION_RETRY_DELAY_SECONDS,
) -> None:
    """Remove a file or directory with read-only attribute clearing and retry handling."""
    if not target_path.exists():
        return

    last_error: OSError | None = None
    for attempt in range(1, retries + 1):
        try:
            _attempt_remove_path(target_path)
            return
        except OSError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(delay_seconds * attempt)

    raise OSError(
        f"Failed to remove {rel_path(target_path)} after {retries} attempts."
    ) from last_error


def remove_directory_with_retry(
    target_dir: Path,
    retries: int = MAX_REORGANISATION_RETRIES,
    delay_seconds: float = REORGANISATION_RETRY_DELAY_SECONDS,
) -> None:
    """Remove a directory using remove_path_with_retry."""
    remove_path_with_retry(target_dir, retries=retries, delay_seconds=delay_seconds)


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


def _attempt_single_move(
    source: Path,
    destination: Path,
    attempt: int,
    retries: int,
    delay_seconds: float,
) -> None:
    """Perform single move attempt with backoff logging."""
    wait_seconds = delay_seconds * attempt
    print(f"  Move attempt {attempt}/{retries} failed for {source.name}")
    print(f"  Retrying in {wait_seconds:.1f} seconds...")
    time.sleep(wait_seconds)


def move_with_retry(
    source: Path,
    destination: Path,
    retries: int = MAX_REORGANISATION_RETRIES,
    delay_seconds: float = REORGANISATION_RETRY_DELAY_SECONDS,
) -> None:
    """Move a file or directory with retry handling."""
    last_error: OSError | None = None
    for attempt in range(1, retries + 1):
        try:
            shutil.move(source, destination)
            return
        except OSError as exc:
            last_error = exc
            if attempt < retries:
                _attempt_single_move(source, destination, attempt, retries, delay_seconds)

    raise OSError(
        f"Failed to move {source} to {destination} after {retries} attempts."
    ) from last_error


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


def _validate_dataset_paths(dataset: DatasetConfig) -> None:
    """Validate dataset fields for path traversal characters."""
    for field_name, value in [
        ("study_id", dataset.study_id),
        ("raw_directory", dataset.raw_directory),
    ]:
        if "/" in value or "\\" in value or ".." in value:
            raise ValueError(f"Invalid path characters in dataset {field_name}: {value!r}")


def _download_and_extract_flow(
    dataset: DatasetConfig,
    target_dir: Path,
    tar_path: Path,
    extraction_root: Path,
    extracted_dir: Path,
) -> None:
    """Execute sequence of download, extract, and reorganisation."""
    remove_existing_dataset_directory(target_dir)
    if extracted_dir.exists():
        print(f"Removing previous incomplete extraction: {rel_path(extracted_dir)}")
        remove_directory_with_retry(extracted_dir)

    url = f"{CBIOPORTAL_DATAHUB_URL}/{dataset.study_id}.tar.gz"
    download_file(url, tar_path)
    extraction_root.mkdir(parents=True, exist_ok=True)
    extract_tar_gz(tar_path=tar_path, extract_to=extraction_root)
    reorganise_extracted_dataset(extracted_dir=extracted_dir, target_dir=target_dir)
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
    except Exception as exc:
        print(f"Error downloading/extracting {dataset.cohort_name}: {exc}")
        raise
    finally:
        remove_path_with_retry(extraction_root)
        if tar_path.exists():
            with contextlib.suppress(OSError):
                tar_path.unlink(missing_ok=True)


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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)

        with (
            contextlib.redirect_stdout(stdout_tee),
            contextlib.redirect_stderr(stderr_tee),
        ):
            print(
                f"Logging console output to "
                f"{rel_path(LOG_PATH)}"
            )
            main()