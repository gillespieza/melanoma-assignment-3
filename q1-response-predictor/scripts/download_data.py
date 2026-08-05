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
SUBPROJECT_ROOT = SCRIPT_DIR.parent

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

# datasets.yaml lives inside this subproject's own config/ directory, not the
# shared data/config/. CONFIG_DIR is therefore subproject-specific.
CONFIG_DIR = SUBPROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "datasets.yaml"

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


def _handle_remove_readonly(func, path, exc_info):
    """Clear read-only attribute on file and retry removal (Windows fix)."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def remove_directory_with_retry(
    target_dir: Path,
    retries: int = MAX_REORGANISATION_RETRIES,
    delay_seconds: float = REORGANISATION_RETRY_DELAY_SECONDS,
) -> None:
    """Remove a directory with read-only attribute clearing and retry handling.

    Args:
        target_dir:
            Directory to remove.
        retries:
            Maximum number of attempts.
        delay_seconds:
            Base delay between retry attempts.
    """
    if not target_dir.exists():
        return

    last_error = None

    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(target_dir, onerror=_handle_remove_readonly)
            return
        except OSError as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(delay_seconds * attempt)

    raise OSError(
        f"Failed to remove directory {rel_path(target_dir)} "
        f"after {retries} attempts."
    ) from last_error


def remove_existing_dataset_directory(target_dir: Path) -> None:
    """Remove an existing dataset directory before clean extraction."""
    if not target_dir.exists():
        return

    print(
        f"Removing existing directory "
        f"{rel_path(target_dir)} "
        "to ensure a clean extraction..."
    )

    remove_directory_with_retry(target_dir)


def move_with_retry(
    source: Path,
    destination: Path,
    retries: int = MAX_REORGANISATION_RETRIES,
    delay_seconds: float = REORGANISATION_RETRY_DELAY_SECONDS,
) -> None:
    """Move a file or directory with retry handling.

    Retry behaviour is useful when a synchronisation service or another
    process briefly holds a file handle. The implementation is platform
    independent and works on Windows, macOS, and Linux.

    Args:
        source:
            Source file or directory.

        destination:
            Destination file or directory.

        retries:
            Maximum number of attempts.

        delay_seconds:
            Delay between attempts. The delay increases slightly after
            each failed attempt.

    Raises:
        OSError:
            If the move fails after all retry attempts.
    """
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            shutil.move(source, destination)
            return

        except OSError as exc:
            last_error = exc

            if attempt == retries:
                break

            wait_seconds = delay_seconds * attempt

            print(
                f"  Move attempt {attempt}/{retries} failed for "
                f"{source.name}: {exc}"
            )
            print(
                f"  Retrying in {wait_seconds:.1f} seconds..."
            )

            time.sleep(wait_seconds)

    raise OSError(
        f"Failed to move {source} to {destination} "
        f"after {retries} attempts."
    ) from last_error


def reorganise_extracted_dataset(
    extracted_dir: Path,
    target_dir: Path,
) -> None:
    """Move extracted dataset contents into the configured target directory.

    cBioPortal archives typically extract into a directory named after the
    study ID. This function moves the contents of that directory into the
    configured standard directory under data/raw/.

    The source directory is removed only after all contents have been moved
    successfully.

    Args:
        extracted_dir:
            Path to the temporary directory containing extracted files.
        target_dir:
            Destination directory under ``data/raw/``.

    Raises:
        FileNotFoundError:
            If ``extracted_dir`` does not exist or contains no files.
        NotADirectoryError:
            If ``extracted_dir`` is not a directory.
    """
    if not extracted_dir.exists():
        raise FileNotFoundError(
            f"Expected extracted dataset directory not found: "
            f"{rel_path(extracted_dir)}"
        )

    if not extracted_dir.is_dir():
        raise NotADirectoryError(
            f"Expected extracted dataset path to be a directory: "
            f"{rel_path(extracted_dir)}"
        )

    target_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Reorganising files from {extracted_dir.name} to "
        f"{rel_path(target_dir)}..."
    )

    extracted_items = list(extracted_dir.iterdir())

    if not extracted_items:
        raise FileNotFoundError(
            f"Extracted dataset directory is empty: {rel_path(extracted_dir)}"
        )

    for item in extracted_items:
        destination = target_dir / item.name

        if destination.exists():
            if destination.is_dir():
                remove_directory_with_retry(destination)
            else:
                destination.unlink()

        move_with_retry(item, destination)

    extracted_dir.rmdir()

    print(
        f"Reorganisation of {extracted_dir.name} complete."
    )


def download_and_extract_dataset(
    dataset: DatasetConfig,
) -> None:
    """Download, extract, and reorganise one configured dataset.

    Args:
        dataset:
            Dataset configuration loaded from datasets.yaml.

    Raises:
        ValueError:
            If study_id or raw_directory contain path traversal characters.
        Exception:
            Any download, extraction, or reorganisation failure is
            propagated to the caller after cleanup.
    """
    for field_name, value in [
        ("study_id", dataset.study_id),
        ("raw_directory", dataset.raw_directory),
    ]:
        if "/" in value or "\\" in value or ".." in value:
            raise ValueError(
                f"Invalid path characters in dataset {field_name}: {value!r}"
            )

    target_dir = RAW_DIR / dataset.raw_directory
    tar_path = RAW_DIR / f"{dataset.study_id}.tar.gz"
    extraction_root = RAW_DIR / _EXTRACTION_STAGING_DIR
    extracted_dir = extraction_root / dataset.study_id

    if is_dataset_present(dataset):
        print(
            f"Dataset {dataset.cohort_name} already exists in "
            f"{rel_path(target_dir)}. Skipping."
        )
        return

    print()
    print("=" * 60)
    print(f"Processing dataset: {dataset.cohort_name}")
    print(f"Study ID: {dataset.study_id}")
    print("=" * 60)

    url = (
        f"{CBIOPORTAL_DATAHUB_URL}/"
        f"{dataset.study_id}.tar.gz"
    )

    try:
        remove_existing_dataset_directory(target_dir)

        if extracted_dir.exists():
            print(
                f"Removing previous incomplete extraction: "
                f"{rel_path(extracted_dir)}"
            )
            remove_directory_with_retry(extracted_dir)

        download_file(url, tar_path)

        extraction_root.mkdir(parents=True, exist_ok=True)

        extract_tar_gz(
            tar_path=tar_path,
            extract_to=extraction_root,
        )

        reorganise_extracted_dataset(
            extracted_dir=extracted_dir,
            target_dir=target_dir,
        )

        print(
            f"Successfully downloaded and prepared "
            f"{dataset.cohort_name}."
        )

    except Exception as exc:
        print(
            f"Error downloading/extracting "
            f"{dataset.cohort_name}: {exc}"
        )

        raise

    finally:
        remove_directory_with_retry(extraction_root)
        if tar_path.exists():
            try:
                tar_path.unlink(missing_ok=True)
            except (PermissionError, OSError):
                pass


def setup_directories() -> None:
    """Create the shared raw-data directory."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Load configuration and download all configured datasets.

    Raises:
        ValueError:
            If no datasets are defined in the configuration file.
        DatasetConfigError:
            If the datasets configuration file is missing or invalid.
    """
    print("=" * 60)
    print("Downloading Raw cBioPortal Datasets")
    print("=" * 60)

    setup_directories()

    datasets = load_dataset_config(CONFIG_PATH)

    if not datasets:
        raise ValueError(
            f"No datasets found in configuration file: "
            f"{CONFIG_PATH}"
        )

    print(
        f"Loaded {len(datasets)} dataset configuration(s) "
        f"from {rel_path(CONFIG_PATH)}."
    )

    successful = 0

    for dataset in datasets:
        try:
            download_and_extract_dataset(dataset)
            successful += 1

        except Exception as exc:
            print(
                f"\n[ERROR] Dataset '{dataset.cohort_name}' failed: "
                f"{exc}"
            )

            # Fail fast. A partially downloaded dataset should not be
            # treated as a successful pipeline run.
            raise

    print()
    print("=" * 60)
    print(
        f"All datasets processed successfully: "
        f"{successful}/{len(datasets)}."
    )
    print("=" * 60)


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