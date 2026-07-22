"""
Download and Extract Melanoma Immunotherapy & TCGA Datasets from cBioPortal.

Fetches raw cohort archives from the cBioPortal DataHub, extracts compressed
tarballs, reorganises directories into standardised data/raw/ subfolders,
and logs progress with TeeStream.
"""

import contextlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


from src.utils.io import download_file, extract_tar_gz
from src.utils.logging import TeeStream
from src.utils.paths import find_project_root


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = find_project_root(Path(__file__).resolve())

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "download_data.log"


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a cBioPortal dataset."""

    study_id: str
    directory_name: str


# ---------------------------------------------------------------------------
# DATASET REGISTRY
# ---------------------------------------------------------------------------
#
# To add a new dataset:
#
# 1. Find the dataset's cBioPortal study ID.
#
# 2. Add one new DatasetConfig entry to DATASETS below.
#
# 3. Set:
#
#       study_id
#           The exact cBioPortal study ID. This is used to construct the
#           cBioPortal DataHub download URL:
#
#           https://datahub.assets.cbioportal.org/{study_id}.tar.gz
#
#       directory_name
#           The local directory name used under:
#
#           data/raw/{directory_name}/
#
# Example:
#
#     DatasetConfig(
#         study_id="example_cbioportal_study",
#         directory_name="example_dataset",
#     ),
#
# No other code changes are required. The dataset will automatically be:
#
# - included in directory setup
# - downloaded
# - extracted
# - reorganised into data/raw/{directory_name}/
#
# The directory_name should be descriptive, stable, and consistent with the
# naming conventions used by the rest of the project.
#
# ---------------------------------------------------------------------------

DATASETS = (
    DatasetConfig(
        study_id="mel_iatlas_liu_2019",
        directory_name="liu_2019",
    ),
    DatasetConfig(
        study_id="mel_iatlas_hugo_ucla_2016",
        directory_name="hugo_2016",
    ),
    DatasetConfig(
        study_id="mel_iatlas_riaz_nivolumab_2017",
        directory_name="riaz_2017",
    ),
    DatasetConfig(
        study_id="skcm_tcga_pan_can_atlas_2018",
        directory_name="skcm_tcga_pan_can_atlas_2018",
    ),
)


# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

def get_dataset_directory(dataset: DatasetConfig) -> Path:
    """Returns the standardised raw directory for a dataset."""
    return RAW_DIR / dataset.directory_name


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def setup_directories(
    datasets: tuple[DatasetConfig, ...],
) -> None:
    """Initialises target directories for all configured datasets."""
    for dataset in datasets:
        get_dataset_directory(dataset).mkdir(
            exist_ok=True,
            parents=True,
        )


# ---------------------------------------------------------------------------
# Dataset download and extraction
# ---------------------------------------------------------------------------

def download_and_extract_cbioportal_dataset(
    dataset: DatasetConfig,
) -> None:
    """Downloads, extracts, and reorganises a cBioPortal dataset.

    Args:
        dataset: Dataset configuration containing the cBioPortal study ID
            and standardised local directory name.

    Raises:
        Exception: Re-raises any download, extraction, or file-system error
            so that the overall pipeline cannot report false success.
    """
    target_dir = get_dataset_directory(dataset)
    study_id = dataset.study_id

    patient_file = target_dir / "data_clinical_patient.txt"

    if patient_file.exists() and patient_file.stat().st_size > 0:
        print(
            f"Dataset {study_id} already exists in "
            f"{target_dir.relative_to(PROJECT_ROOT).as_posix()} "
            "and is non-empty."
        )
        return

    if target_dir.exists():
        try:
            shutil.rmtree(target_dir)

            print(
                f"Removed existing directory "
                f"{target_dir.relative_to(PROJECT_ROOT).as_posix()} "
                "to ensure a clean extraction."
            )

        except Exception as error:
            print(
                f"Failed to remove existing directory "
                f"{target_dir.relative_to(PROJECT_ROOT).as_posix()}: "
                f"{error}"
            )
            raise

    target_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    tar_path = RAW_DIR / f"{study_id}.tar.gz"
    url = f"https://datahub.assets.cbioportal.org/{study_id}.tar.gz"

    try:
        print(f"Downloading dataset: {study_id}")

        download_file(url, tar_path)

        print(f"Extracting dataset: {study_id}")

        extract_tar_gz(tar_path, RAW_DIR)

        extracted_dir = RAW_DIR / study_id

        if not extracted_dir.exists() or not extracted_dir.is_dir():
            raise FileNotFoundError(
                f"Expected extracted directory was not found: "
                f"{extracted_dir}"
            )

        print(
            f"Reorganising files from {study_id} to "
            f"{target_dir.relative_to(PROJECT_ROOT).as_posix()}..."
        )

        for item in extracted_dir.iterdir():
            destination = target_dir / item.name

            if destination.is_dir():
                shutil.rmtree(destination)

            elif destination.exists():
                destination.unlink()

            shutil.move(
                str(item),
                str(destination),
            )

        extracted_dir.rmdir()

        print(f"Reorganisation of {study_id} complete.")

    except Exception as error:
        print(
            f"Error downloading/extracting {study_id}: "
            f"{error}"
        )

        # Re-raise the original exception so the script exits with a failure
        # status. This prevents the pipeline from incorrectly reporting that
        # all datasets were downloaded successfully.
        raise

    finally:
        tar_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrates dataset directory setup and downloads."""
    print("=" * 50)
    print("Downloading Raw cBioPortal Datasets")
    print("=" * 50)
    print()

    setup_directories(DATASETS)

    for dataset in DATASETS:
        download_and_extract_cbioportal_dataset(dataset)

    print()
    print("=" * 50)
    print("All raw datasets downloaded and extracted successfully!")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        LOG_PATH,
        "w",
        encoding="utf-8",
    ) as log_file:

        stdout_tee = TeeStream(
            sys.stdout,
            log_file,
        )

        stderr_tee = TeeStream(
            sys.stderr,
            log_file,
        )

        with (
            contextlib.redirect_stdout(stdout_tee),
            contextlib.redirect_stderr(stderr_tee),
        ):
            print(
                "Logging console output to "
                f"{LOG_PATH.relative_to(PROJECT_ROOT).as_posix()}"
            )

            main()