"""
Download and Extract Melanoma Immunotherapy & TCGA Datasets from cBioPortal.

Fetches raw cohort archives (Liu 2019, Hugo 2016, Riaz 2017, TCGA-SKCM) from the cBioPortal DataHub,
extracts compressed tarballs, reorganises directories into standardized data/raw/ subfolders,
and logs progress with TeeStream.
"""

import contextlib
from pathlib import Path
import shutil
import sys

# Bootstrap project root resolution for top-level import
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.io import download_file, extract_tar_gz
from src.utils.logging import TeeStream
from src.utils.paths import find_project_root

# Base directories
DATA_DIR = find_project_root(Path(__file__).resolve()) / "data"
RAW_DIR = DATA_DIR / "raw"
LOG_DIR = find_project_root(Path(__file__).resolve()) / "logs"
LOG_PATH = LOG_DIR / "download_data.log"

LIU_DIR = RAW_DIR / "liu_2019"
HUGO_DIR = RAW_DIR / "hugo_2016"
RIAZ_DIR = RAW_DIR / "riaz_2017"
TCGA_DIR = RAW_DIR / "skcm_tcga_pan_can_atlas_2018"

# Study IDs on cBioPortal
LIU_STUDY_ID = "mel_iatlas_liu_2019"
HUGO_STUDY_ID = "mel_iatlas_hugo_ucla_2016"
RIAZ_STUDY_ID = "mel_iatlas_riaz_nivolumab_2017"
TCGA_STUDY_ID = "skcm_tcga_pan_can_atlas_2018"


def setup_directories() -> None:
    """Initialises the target directories for each study dataset."""
    for d in [LIU_DIR, HUGO_DIR, RIAZ_DIR, TCGA_DIR]:
        d.mkdir(exist_ok=True, parents=True)


def download_and_extract_cbioportal_dataset(study_id: str, target_dir: Path) -> None:
    """Downloads, extracts, and reorganises a cBioPortal dataset tarball.

    Args:
        study_id: The study identifier on cBioPortal DataHub.
        target_dir: The destination directory to place the study files.
    """
    patient_file = target_dir / "data_clinical_patient.txt"
    if patient_file.exists() and patient_file.stat().st_size > 0:
        print(f"Dataset {study_id} already exists in {target_dir.relative_to(BASE_DIR).as_posix()} and is non-empty.")
        return

    if target_dir.exists():
        try:
            shutil.rmtree(target_dir, ignore_errors=False)
            print(f"Removed existing directory {target_dir.relative_to(BASE_DIR).as_posix()} to ensure a clean extraction.")
        except Exception as e:
            print(f"Failed to remove existing directory {target_dir.relative_to(BASE_DIR).as_posix()}: {e}")
            raise

    target_dir.mkdir(exist_ok=True, parents=True)
    tar_path = RAW_DIR / f"{study_id}.tar.gz"
    url = f"https://datahub.assets.cbioportal.org/{study_id}.tar.gz"

    try:
        download_file(url, tar_path)
        extract_tar_gz(tar_path, RAW_DIR)

        extracted_dir = RAW_DIR / study_id
        if extracted_dir.exists() and extracted_dir.is_dir():
            print(f"Reorganising files from {study_id} to {target_dir.relative_to(BASE_DIR).as_posix()}...")
            for item in extracted_dir.iterdir():
                dest_item = target_dir / item.name
                if dest_item.is_dir():
                    shutil.rmtree(dest_item, ignore_errors=True)
                elif dest_item.exists():
                    dest_item.unlink()
                shutil.move(str(item), str(dest_item))
            extracted_dir.rmdir()
            print(f"Reorganisation of {study_id} complete.")
    except Exception as e:
        print(f"Error downloading/extracting {study_id}: {e}")
    finally:
        tar_path.unlink(missing_ok=True)


def main() -> None:
    """Orchestrates the setup of directories and downloads all study datasets."""
    print("==================================================")
    print("Downloading Raw cBioPortal Datasets")
    print("==================================================\n")
    setup_directories()

    download_and_extract_cbioportal_dataset(LIU_STUDY_ID, LIU_DIR)
    download_and_extract_cbioportal_dataset(HUGO_STUDY_ID, HUGO_DIR)
    download_and_extract_cbioportal_dataset(RIAZ_STUDY_ID, RIAZ_DIR)
    download_and_extract_cbioportal_dataset(TCGA_STUDY_ID, TCGA_DIR)

    print("\n==================================================")
    print("All raw datasets downloaded and extracted successfully!")
    print("==================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(BASE_DIR).as_posix()}")
            main()
