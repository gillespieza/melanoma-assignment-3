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
import shutil
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
    CONFIG_DIR,
    LOG_DIR,
    PROJECT_ROOT,
    RAW_DIR,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONFIG_PATH = CONFIG_DIR / "datasets.yaml"
LOG_PATH = LOG_DIR / "download_data.log"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CBIOPORTAL_DATAHUB_URL = (
    "https://datahub.assets.cbioportal.org"
)

MAX_REORGANISATION_RETRIES = 5
REORGANISATION_RETRY_DELAY_SECONDS = 2


def format_relative_path(path: Path) -> str:
    """Return a project-relative path for readable console output.

    Paths outside the project root are returned as absolute paths.
    """
    path = Path(path).resolve()

    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def is_dataset_present(dataset: DatasetConfig) -> bool:
    """Return whether the configured dataset already contains data.

    A dataset is considered present when its target directory exists and
    contains at least one file or directory.

    This avoids relying on a particular filename because different
    cBioPortal datasets may contain different file collections.
    """
    target_dir = RAW_DIR / dataset.raw_directory

    if not target_dir.exists():
        return False

    return any(target_dir.iterdir())


def remove_existing_dataset_directory(target_dir: Path) -> None:
    """Remove an existing dataset directory before clean extraction."""
    if not target_dir.exists():
        return

    print(
        f"Removing existing directory "
        f"{format_relative_path(target_dir)} "
        "to ensure a clean extraction..."
    )

    shutil.rmtree(target_dir)


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
            shutil.move(str(source), str(destination))
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
    """
    if not extracted_dir.exists():
        raise FileNotFoundError(
            f"Expected extracted dataset directory not found: "
            f"{extracted_dir}"
        )

    if not extracted_dir.is_dir():
        raise NotADirectoryError(
            f"Expected extracted dataset path to be a directory: "
            f"{extracted_dir}"
        )

    target_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Reorganising files from {extracted_dir.name} to "
        f"{format_relative_path(target_dir)}..."
    )

    extracted_items = list(extracted_dir.iterdir())

    if not extracted_items:
        raise FileNotFoundError(
            f"Extracted dataset directory is empty: {extracted_dir}"
        )

    for item in extracted_items:
        destination = target_dir / item.name

        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
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
        Exception:
            Any download, extraction, or reorganisation failure is
            propagated to the caller after cleanup.
    """
    target_dir = RAW_DIR / dataset.raw_directory
    tar_path = RAW_DIR / f"{dataset.study_id}.tar.gz"
    extraction_root = RAW_DIR / "_extracted"
    extracted_dir = extraction_root / dataset.study_id

    if is_dataset_present(dataset):
        print(
            f"Dataset {dataset.cohort_name} already exists in "
            f"{format_relative_path(target_dir)}. Skipping."
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
                f"{format_relative_path(extracted_dir)}"
            )
            shutil.rmtree(extracted_dir)

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
        shutil.rmtree(extraction_root, ignore_errors=True)
        if tar_path.exists():
            tar_path.unlink(missing_ok=True)


def setup_directories() -> None:
    """Create the shared raw-data and log directories."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Load configuration and download all configured datasets."""
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
        f"from {format_relative_path(CONFIG_PATH)}."
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
                f"{format_relative_path(LOG_PATH)}"
            )
            main()