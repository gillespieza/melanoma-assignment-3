import numpy as np
import pandas as pd
from pathlib import Path

# Import TCGA cleaning functions
from src.tcga_helpers import clean_clinical_df, clean_rnaseq_df

# Import utility helpers
from src.utils import parse_geo_metadata, get_gene_symbol_mapping, align_expression_and_clinical

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Study IDs
LIU_STUDY_ID = "mel_dfci_2019"
HUGO_STUDY_ID = "GSE78220"
RIAZ_STUDY_ID = "GSE91061"
TCGA_STUDY_ID = "skcm_tcga_pan_can_atlas_2018"

# Raw input filenames
LIU_PATIENT_FILE = "data_clinical_patient.txt"
LIU_SAMPLE_FILE = "data_clinical_sample.txt"
LIU_EXPR_FILE = "data_mrna_seq_tpm.txt"

HUGO_META_FILE = "GSE78220_series_matrix.txt"
HUGO_EXPR_FILE = "GSE78220_PatientFPKM.xlsx"

RIAZ_META_FILE = "GSE91061_series_matrix.txt"
RIAZ_EXPR_FILE = "GSE91061_BMS038109Sample.hg19KnownGene.fpkm.csv"

TCGA_CLIN_FILE = "clinical.csv"
TCGA_EXPR_FILE = "rnaseq.csv"


def clean_liu_2019() -> None:
    """
    Cleans raw Liu 2019 datasets and writes processed matrices to the processed folder.

    @return None
    """
    raw_dir = RAW_DIR / "liu_2019"
    proc_dir = PROCESSED_DIR / "liu_2019"
    proc_dir.mkdir(parents=True, exist_ok=True)

    if not all((raw_dir / f).exists() for f in [LIU_PATIENT_FILE, LIU_SAMPLE_FILE, LIU_EXPR_FILE]):
        raise FileNotFoundError(f"Missing raw input files in {raw_dir}. Please run download_data.py first.")

    print("Cleaning Liu 2019 (mel_dfci_2019)...")

    # Load raw clinical data
    df_patient = pd.read_csv(raw_dir / LIU_PATIENT_FILE, sep="\t", skiprows=4)
    df_sample = pd.read_csv(raw_dir / LIU_SAMPLE_FILE, sep="\t", skiprows=4)
    df_clin = pd.merge(df_sample, df_patient, on="PATIENT_ID")
    
    # Map Response (CR/PR = 1, PD = 0)
    response_map = {
        "Complete Response": 1,
        "Partial Response": 1,
        "Progressive Disease": 0,
        "Stable Disease": np.nan,
        "Mixed Response": np.nan
    }
    df_clin['response'] = df_clin['BR'].map(response_map)
    df_clin = df_clin.dropna(subset=['response'])
    df_clin = df_clin.set_index("SAMPLE_ID")

    # Load raw expression (TPM)
    df_expr = pd.read_csv(raw_dir / LIU_EXPR_FILE, sep="\t")
    df_expr = df_expr.dropna(subset=["Hugo_Symbol"])
    df_expr = df_expr.set_index("Hugo_Symbol")
    if "Entrez_Gene_Id" in df_expr.columns:
        df_expr = df_expr.drop(columns=["Entrez_Gene_Id"])

    df_expr = df_expr.groupby(df_expr.index).mean()
    df_expr = df_expr.T

    # Align samples
    df_expr, df_clin = align_expression_and_clinical(df_expr, df_clin)

    # Log-transform expression
    df_expr = np.log2(df_expr + 1)

    # Save cleaned
    df_expr.to_csv(proc_dir / "expr_cleaned.csv")
    df_clin.to_csv(proc_dir / "clin_cleaned.csv")
    print(f"  Liu 2019: Cleaned {len(df_clin)} samples.")


def clean_hugo_2016() -> None:
    """
    Cleans raw Hugo 2016 datasets and writes processed matrices to the processed folder.

    @return None
    """
    raw_dir = RAW_DIR / "hugo_2016"
    proc_dir = PROCESSED_DIR / "hugo_2016"
    proc_dir.mkdir(parents=True, exist_ok=True)

    if not all((raw_dir / f).exists() for f in [HUGO_META_FILE, HUGO_EXPR_FILE]):
        raise FileNotFoundError(f"Missing raw input files in {raw_dir}. Please run download_data.py first.")

    print("Cleaning Hugo 2016 (GSE78220)...")

    # Load metadata
    df_meta = parse_geo_metadata(raw_dir / HUGO_META_FILE)
    response_col = [c for c in df_meta.columns if 'response' in c][0]
    patient_col = [c for c in df_meta.columns if 'patient' in c][0]
    os_col = [c for c in df_meta.columns if 'survival' in c][0]
    status_col = [c for c in df_meta.columns if 'vital' in c][0]

    # Normalise clinical columns
    response_map = {
        "Progressive Disease": 0,
        "Partial Response": 1,
        "Complete Response": 1,
        "Stable Disease": np.nan
    }
    df_meta['response'] = df_meta[response_col].map(response_map)
    df_meta = df_meta.dropna(subset=['response'])
    df_meta['patient_id'] = df_meta[patient_col]
    df_meta['os_days'] = pd.to_numeric(df_meta[os_col], errors='coerce')
    df_meta['os_status'] = df_meta[status_col].map({"Dead": 1, "Alive": 0})

    # Load expression Excel
    df_expr = pd.read_excel(raw_dir / HUGO_EXPR_FILE)
    df_expr = df_expr.rename(columns={df_expr.columns[0]: "Gene"})
    df_expr = df_expr.dropna(subset=["Gene"])
    df_expr = df_expr.set_index("Gene")
    df_expr.columns = [c.split('.')[0] for c in df_expr.columns]

    pat_to_gsm = {row['patient_id']: gsm for gsm, row in df_meta.iterrows()}
    valid_cols = [c for c in df_expr.columns if c in pat_to_gsm]
    df_expr = df_expr[valid_cols]
    df_expr = df_expr.rename(columns=pat_to_gsm)

    df_expr = df_expr.groupby(df_expr.index).mean()
    df_expr = df_expr.T

    # Align
    df_expr, df_meta = align_expression_and_clinical(df_expr, df_meta)

    # Log-transform
    df_expr = np.log2(df_expr + 1)

    # Save cleaned
    df_expr.to_csv(proc_dir / "expr_cleaned.csv")
    df_meta.to_csv(proc_dir / "clin_cleaned.csv")
    print(f"  Hugo 2016: Cleaned {len(df_meta)} samples.")


def clean_riaz_2017() -> None:
    """
    Cleans raw Riaz 2017 datasets and writes processed matrices to the processed folder.

    @return None
    """
    raw_dir = RAW_DIR / "riaz_2017"
    proc_dir = PROCESSED_DIR / "riaz_2017"
    proc_dir.mkdir(parents=True, exist_ok=True)

    if not all((raw_dir / f).exists() for f in [RIAZ_META_FILE, RIAZ_EXPR_FILE]):
        raise FileNotFoundError(f"Missing raw input files in {raw_dir}. Please run download_data.py first.")

    print("Cleaning Riaz 2017 (GSE91061)...")

    # Load metadata
    df_meta = parse_geo_metadata(raw_dir / RIAZ_META_FILE)
    visit_col = [c for c in df_meta.columns if 'visit' in c][0]
    response_col = [c for c in df_meta.columns if 'response' in c][0]

    # Pre-treatment only
    df_meta = df_meta[df_meta[visit_col] == "Pre"]

    response_map = {
        "PD": 0, "PRCR": 1, "PR": 1, "CR": 1,
        "SD": np.nan, "UNK": np.nan
    }
    df_meta['response'] = df_meta[response_col].map(response_map)
    df_meta = df_meta.dropna(subset=['response'])

    # Load expression FPKM
    df_expr_raw = pd.read_csv(raw_dir / RIAZ_EXPR_FILE, index_col=0)
    df_expr_raw.index = df_expr_raw.index.astype(str)

    # Entrez symbol mapping
    mapping = get_gene_symbol_mapping(df_expr_raw.index.tolist())
    df_expr_raw.index = df_expr_raw.index.map(mapping)

    df_expr_raw = df_expr_raw.loc[df_expr_raw.index.dropna()]
    df_expr_raw = df_expr_raw.groupby(df_expr_raw.index).mean()

    title_to_gsm = {row['title']: gsm for gsm, row in df_meta.iterrows()}
    valid_cols = [c for c in df_expr_raw.columns if c in title_to_gsm]
    df_expr_raw = df_expr_raw[valid_cols]
    df_expr_raw = df_expr_raw.rename(columns=title_to_gsm)

    df_expr = df_expr_raw.T

    # Align
    df_expr, df_meta = align_expression_and_clinical(df_expr, df_meta)

    # Log-transform
    df_expr = np.log2(df_expr + 1)

    # Save cleaned
    df_expr.to_csv(proc_dir / "expr_cleaned.csv")
    df_meta.to_csv(proc_dir / "clin_cleaned.csv")
    print(f"  Riaz 2017: Cleaned {len(df_meta)} samples.")


def clean_tcga_skcm() -> None:
    """
    Cleans raw TCGA-SKCM datasets and writes processed matrices to the processed folder.

    @return None
    """
    raw_dir = RAW_DIR / TCGA_STUDY_ID
    proc_dir = PROCESSED_DIR / TCGA_STUDY_ID
    proc_dir.mkdir(parents=True, exist_ok=True)

    if not all((raw_dir / f).exists() for f in [TCGA_CLIN_FILE, TCGA_EXPR_FILE]):
        raise FileNotFoundError(f"Missing raw input files in {raw_dir}. Please run download_data.py first.")

    print(f"Cleaning TCGA-SKCM ({TCGA_STUDY_ID})...")

    # Clean Clinical
    raw_clin_df = pd.read_csv(raw_dir / TCGA_CLIN_FILE)
    cleaned_clin_df = clean_clinical_df(raw_clin_df)

    # Clean RNA-seq
    raw_rnaseq_df = pd.read_csv(raw_dir / TCGA_EXPR_FILE)
    cleaned_rnaseq_df = clean_rnaseq_df(raw_rnaseq_df)

    # Save cleaned
    cleaned_clin_df.to_csv(proc_dir / "clinical_cleaned.csv", index=False)
    cleaned_rnaseq_df.to_csv(proc_dir / "rnaseq_cleaned.csv", index=False)
    print(f"  TCGA-SKCM: Cleaned {len(cleaned_clin_df)} samples.")


def main() -> None:
    """
    Orchestrates the data cleaning workflow for all studies.

    @return None
    """
    print("==================================================")
    print("Data Cleaning Pipeline: Transforming Raw Data")
    print("==================================================")
    
    stages = [
        ("Liu 2019", clean_liu_2019),
        ("Hugo 2016", clean_hugo_2016),
        ("Riaz 2017", clean_riaz_2017),
        ("TCGA-SKCM", clean_tcga_skcm)
    ]
    
    success_count = 0
    for name, stage_fn in stages:
        try:
            stage_fn()
            success_count += 1
        except Exception as e:
            print(f"  [ERROR] Stage '{name}' failed: {e}")
            
    print("==================================================")
    print(f"Workflow completed: {success_count}/{len(stages)} stages succeeded.")
    print("==================================================")


if __name__ == "__main__":
    main()
