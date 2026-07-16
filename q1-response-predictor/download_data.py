import shutil
from pathlib import Path

# Import utility helpers
from src.utils.io import download_file, extract_tar_gz

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

LIU_DIR = RAW_DIR / "liu_2019"
HUGO_DIR = RAW_DIR / "hugo_2016"
RIAZ_DIR = RAW_DIR / "riaz_2017"
TCGA_DIR = RAW_DIR / "skcm_tcga_pan_can_atlas_2018"

# Study IDs on cBioPortal
LIU_STUDY_ID = "mel_iatlas_liu_2019"
HUGO_STUDY_ID = "mel_iatlas_hugo_ucla_2016"
RIAZ_STUDY_ID = "mel_iatlas_riaz_nivolumab_2017"
TCGA_STUDY_ID = "skcm_tcga_pan_can_atlas_2018"


def setup_directories():
    """
    Initialises the target directories for each study dataset.

    @return None
    """
    for d in [LIU_DIR, HUGO_DIR, RIAZ_DIR, TCGA_DIR]:
        d.mkdir(exist_ok=True, parents=True)


def download_and_extract_cbioportal_dataset(study_id: str, target_dir: Path) -> None:
    """
    Downloads, extracts, and reorganises a cBioPortal dataset tarball.

    @param str study_id The study identifier on cBioPortal DataHub.
    @param Path target_dir The destination directory to place the study files.
    @return None
    """
    # Use patient file existence as a proxy check to prevent redownload
    patient_file = target_dir / "data_clinical_patient.txt"
    if patient_file.exists() and patient_file.stat().st_size > 0:
        print(f"Dataset {study_id} already exists in {target_dir.name} and is non-empty.")
        return
    # If target directory exists (possibly from a previous incomplete download), remove it to avoid file lock issues
    if target_dir.exists():
        try:
            shutil.rmtree(target_dir, ignore_errors=False)
            print(f"Removed existing directory {target_dir} to ensure a clean extraction.")
        except Exception as e:
            print(f"Failed to remove existing directory {target_dir}: {e}")
            raise
    target_dir.mkdir(exist_ok=True, parents=True)

    tar_path = RAW_DIR / f"{study_id}.tar.gz"
    url = f"https://datahub.assets.cbioportal.org/{study_id}.tar.gz"
    
    try:
        download_file(url, tar_path)
        extract_tar_gz(tar_path, RAW_DIR)
        
        extracted_dir = RAW_DIR / study_id
        if extracted_dir.exists() and extracted_dir.is_dir():
            print(f"Reorganising files from {study_id} to {target_dir.name}...")
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


def main():
    """
    Orchestrates the setup of directories and downloads all study datasets.

    @return None
    """
    setup_directories()

    # 1. Download mel_dfci_2019 (Liu et al. 2019) from cBioPortal
    download_and_extract_cbioportal_dataset(LIU_STUDY_ID, LIU_DIR)

    # 2. Download mel_iatlas_hugo_ucla_2016 (Hugo et al. 2016) from cBioPortal
    download_and_extract_cbioportal_dataset(HUGO_STUDY_ID, HUGO_DIR)

    # 3. Download mel_iatlas_riaz_nivolumab_2017 (Riaz et al. 2017) from cBioPortal
    download_and_extract_cbioportal_dataset(RIAZ_STUDY_ID, RIAZ_DIR)

    # 4. Download skcm_tcga_pan_can_atlas_2018 (TCGA SKCM PanCancer Atlas) from cBioPortal
    download_and_extract_cbioportal_dataset(TCGA_STUDY_ID, TCGA_DIR)


if __name__ == "__main__":
    main()
