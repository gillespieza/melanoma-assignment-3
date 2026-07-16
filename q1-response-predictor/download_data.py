import tarfile
import shutil
import requests
from pathlib import Path

# Import utility helpers
from src.utils import download_file, extract_tar_gz, download_and_decompress_gzip, download_if_missing
from src.tcga_helpers import download_raw_tcga_skcm

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

LIU_DIR = RAW_DIR / "liu_2019"
HUGO_DIR = RAW_DIR / "hugo_2016"
RIAZ_DIR = RAW_DIR / "riaz_2017"

# Study IDs
LIU_STUDY_ID = "mel_dfci_2019"
HUGO_STUDY_ID = "GSE78220"
RIAZ_STUDY_ID = "GSE91061"
TCGA_STUDY_ID = "skcm_tcga_pan_can_atlas_2018"

# Dataset Source URLs
LIU_URL = f"https://datahub.assets.cbioportal.org/{LIU_STUDY_ID}.tar.gz"

HUGO_MATRIX_URL = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE78nnn/{HUGO_STUDY_ID}/matrix/{HUGO_STUDY_ID}_series_matrix.txt.gz"
HUGO_SUPP_URL = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE78nnn/{HUGO_STUDY_ID}/suppl/{HUGO_STUDY_ID}_PatientFPKM.xlsx"

RIAZ_MATRIX_URL = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE91nnn/{RIAZ_STUDY_ID}/matrix/{RIAZ_STUDY_ID}_series_matrix.txt.gz"
RIAZ_SUPP_URL = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE91nnn/{RIAZ_STUDY_ID}/suppl/{RIAZ_STUDY_ID}_BMS038109Sample.hg19KnownGene.fpkm.csv.gz"


def setup_directories():
    """
    Initialises the target directories for each study dataset.

    @return None
    """
    for d in [LIU_DIR, HUGO_DIR, RIAZ_DIR]:
        d.mkdir(exist_ok=True, parents=True)


def ensure_liu_dataset():
    """
    Downloads, extracts, and reorganises the Liu 2019 dataset from cBioPortal.

    @return None
    """
    tpm_file = LIU_DIR / "data_mrna_seq_tpm.txt"
    if tpm_file.exists() and tpm_file.stat().st_size > 0:
        print("Liu 2019 files already exist and are non-empty.")
        return

    dfci_tar = LIU_DIR / f"{LIU_STUDY_ID}.tar.gz"
    try:
        download_file(LIU_URL, dfci_tar)
        extract_tar_gz(dfci_tar, RAW_DIR)
        
        extracted_dir = RAW_DIR / LIU_STUDY_ID
        if extracted_dir.exists() and extracted_dir.is_dir():
            for f in extracted_dir.iterdir():
                dest_f = LIU_DIR / f.name
                if dest_f.exists():
                    dest_f.unlink()
                shutil.move(str(f), str(dest_f))
            extracted_dir.rmdir()
            print(f"Reorganised Liu files from {LIU_STUDY_ID} to liu_2019")
    except (requests.RequestException, OSError, tarfile.TarError) as e:
        print(f"Error downloading/extracting {LIU_STUDY_ID}: {e}")
    finally:
        dfci_tar.unlink(missing_ok=True)


def ensure_tcga_dataset():
    """
    Downloads, extracts, and reorganises the TCGA-SKCM (pancancer) dataset from cBioPortal assets.

    @return None
    """
    raw_tcga_dir = DATA_DIR / "raw" / TCGA_STUDY_ID
    tpm_file = raw_tcga_dir / "data_mrna_seq_v2_rsem.txt"
    if tpm_file.exists() and tpm_file.stat().st_size > 0:
        print("TCGA-SKCM (pancancer) files already exist and are non-empty.")
        return

    tcga_tar = RAW_DIR / f"{TCGA_STUDY_ID}.tar.gz"
    raw_tcga_dir.mkdir(parents=True, exist_ok=True)
    try:
        url = f"https://datahub.assets.cbioportal.org/{TCGA_STUDY_ID}.tar.gz"
        download_file(url, tcga_tar)
        extract_tar_gz(tcga_tar, RAW_DIR)
        print(f"Extracted TCGA-SKCM files to {raw_tcga_dir}")
    except Exception as e:
        print(f"Error downloading/extracting TCGA-SKCM ({TCGA_STUDY_ID}): {e}")
    finally:
        tcga_tar.unlink(missing_ok=True)


def main():
    """
    Orchestrates the setup of directories and downloads all study datasets.

    @return None
    """
    setup_directories()

    # 1. Download mel_dfci_2019 (Liu et al. 2019) from cBioPortal assets
    ensure_liu_dataset()

    # 2. Download GSE78220 (Hugo et al. 2016) series matrix and expression files
    hugo_gz = HUGO_DIR / f"{HUGO_STUDY_ID}_series_matrix.txt.gz"
    hugo_txt = HUGO_DIR / f"{HUGO_STUDY_ID}_series_matrix.txt"
    download_and_decompress_gzip(HUGO_MATRIX_URL, hugo_txt, hugo_gz)

    # Download GSE78220 expression Excel (Hugo 2016 FPKM)
    hugo_xlsx = HUGO_DIR / f"{HUGO_STUDY_ID}_PatientFPKM.xlsx"
    download_if_missing(HUGO_SUPP_URL, hugo_xlsx)

    # 3. Download GSE91061 (Riaz et al. 2017) series matrix and expression files
    riaz_gz = RIAZ_DIR / f"{RIAZ_STUDY_ID}_series_matrix.txt.gz"
    riaz_txt = RIAZ_DIR / f"{RIAZ_STUDY_ID}_series_matrix.txt"
    download_and_decompress_gzip(RIAZ_MATRIX_URL, riaz_txt, riaz_gz)

    # Download GSE91061 expression FPKM
    riaz_expr_gz = RIAZ_DIR / f"{RIAZ_STUDY_ID}_BMS038109Sample.hg19KnownGene.fpkm.csv.gz"
    riaz_expr_csv = RIAZ_DIR / f"{RIAZ_STUDY_ID}_BMS038109Sample.hg19KnownGene.fpkm.csv"
    download_and_decompress_gzip(RIAZ_SUPP_URL, riaz_expr_csv, riaz_expr_gz)

    # 4. Download and process TCGA-SKCM (Baseline overall survival validation)
    ensure_tcga_dataset()


if __name__ == "__main__":
    main()
