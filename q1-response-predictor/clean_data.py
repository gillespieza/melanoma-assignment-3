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

TCGA_PATIENT_FILE = "data_clinical_patient.txt"
TCGA_SAMPLE_FILE = "data_clinical_sample.txt"
TCGA_EXPR_FILE = "data_mrna_seq_v2_rsem.txt"


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

    if not all((raw_dir / f).exists() for f in [TCGA_PATIENT_FILE, TCGA_SAMPLE_FILE, TCGA_EXPR_FILE]):
        raise FileNotFoundError(f"Missing raw input files in {raw_dir}. Please run download_data.py first.")

    print(f"Cleaning TCGA-SKCM ({TCGA_STUDY_ID})...")

    # Clean Clinical
    df_patient = pd.read_csv(raw_dir / TCGA_PATIENT_FILE, sep="\t", skiprows=4)
    df_sample = pd.read_csv(raw_dir / TCGA_SAMPLE_FILE, sep="\t", skiprows=4)
    df_clin = pd.merge(df_sample, df_patient, on="PATIENT_ID")
    cleaned_clin_df = clean_clinical_df(df_clin)

    # Parse and Merge Treatment Timeline Data if available
    timeline_file = raw_dir / "data_timeline_treatment.txt"
    if timeline_file.exists():
        print("  Found treatment timeline data. Parsing and aggregating features...")
        df_treat = pd.read_csv(timeline_file, sep="\t")
        df_treat["PATIENT_ID"] = df_treat["PATIENT_ID"].astype(str).str.strip().str.upper()

        # Aggregate unique treatment types and agents
        treatment_types_series = df_treat.groupby("PATIENT_ID")["TREATMENT_TYPE"].apply(
            lambda x: ", ".join(sorted(set(x.dropna())))
        )
        agents_series = df_treat.groupby("PATIENT_ID")["AGENT"].apply(
            lambda x: ", ".join(sorted(set(x.dropna())))
        )

        unique_patients = df_treat["PATIENT_ID"].unique()
        treat_features = pd.DataFrame(index=unique_patients)
        treat_features["TREATMENT_TYPES"] = treatment_types_series
        treat_features["TREATMENT_AGENTS"] = agents_series

        # Pivot treatment types (e.g. Radiation Therapy, Chemotherapy, Immunotherapy)
        unique_types = df_treat["TREATMENT_TYPE"].dropna().unique()
        for t_type in unique_types:
            col_name = f"TX_TYPE_{t_type.replace(' ', '_').upper()}"
            patients_with_type = df_treat[df_treat["TREATMENT_TYPE"] == t_type]["PATIENT_ID"].unique()
            treat_features[col_name] = 0
            treat_features.loc[patients_with_type, col_name] = 1

        # Pivot key drugs/agents
        key_agents = {
            "TX_AGENT_IPILIMUMAB": ["Ipilimumab"],
            "TX_AGENT_PEMBROLIZUMAB": ["Pembrolizumab"],
            "TX_AGENT_NIVOLUMAB": ["Nivolumab"],
            "TX_AGENT_VEMURAFENIB": ["Vemurafenib"],
            "TX_AGENT_DABRAFENIB": ["Dabrafenib"],
            "TX_AGENT_TRAMETINIB": ["Trametinib"],
            "TX_AGENT_DACARBAZINE": ["Dacarbazine"],
            "TX_AGENT_TEMOZOLOMIDE": ["Temozolomide"],
            "TX_AGENT_INTERFERON": ["Interferon Alfa", "Interferon Nos", "Interferon"]
        }

        for feat_name, agents_list in key_agents.items():
            matching_rows = df_treat[df_treat["AGENT"].astype(str).str.upper().apply(
                lambda val: any(agent.upper() in val for agent in agents_list)
            )]
            patients_with_agent = matching_rows["PATIENT_ID"].unique()
            treat_features[feat_name] = 0
            treat_features.loc[patients_with_agent, feat_name] = 1

        treat_features = treat_features.reset_index().rename(columns={"index": "PATIENT_ID"})
        cleaned_clin_df = pd.merge(cleaned_clin_df, treat_features, on="PATIENT_ID", how="left")

        # Fill NaNs for patients who did not receive clinical treatments
        cleaned_clin_df["TREATMENT_TYPES"] = cleaned_clin_df["TREATMENT_TYPES"].fillna("None")
        cleaned_clin_df["TREATMENT_AGENTS"] = cleaned_clin_df["TREATMENT_AGENTS"].fillna("None")
        for col in cleaned_clin_df.columns:
            if col.startswith("TX_TYPE_") or col.startswith("TX_AGENT_"):
                cleaned_clin_df[col] = cleaned_clin_df[col].fillna(0).astype(int)

    # Clean RNA-seq
    df_expr = pd.read_csv(raw_dir / TCGA_EXPR_FILE, sep="\t")
    df_expr = df_expr.dropna(subset=["Entrez_Gene_Id"])
    df_expr["Entrez_Gene_Id"] = df_expr["Entrez_Gene_Id"].astype(int).astype(str)
    df_expr = df_expr.set_index("Entrez_Gene_Id")
    if "Hugo_Symbol" in df_expr.columns:
        df_expr = df_expr.drop(columns=["Hugo_Symbol"])
    df_expr = df_expr.groupby(df_expr.index).mean()
    df_expr = df_expr.T
    df_expr.index.name = "SAMPLE_ID"
    df_expr = df_expr.reset_index()
    
    cleaned_rnaseq_df = clean_rnaseq_df(df_expr)

    # Drop constant, redundant, and administrative columns
    cols_to_drop = [
        "CANCER_TYPE", "CANCER_TYPE_DETAILED", "ONCOTREE_CODE", "CANCER_TYPE_ACRONYM",
        "SUBTYPE", "TUMOR_TYPE", "SOMATIC_STATUS", "DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS",
        "INFORMED_CONSENT_VERIFIED", "DAYS_TO_BIRTH", "OTHER_PATIENT_ID", "TISSUE_SOURCE_SITE_CODE",
        "TISSUE_RETROSPECTIVE_COLLECTION_INDICATOR", "FORM_COMPLETION_DATE", "AJCC_STAGING_EDITION",
        "SAMPLE_COUNT", "TISSUE_PROSPECTIVE_COLLECTION_INDICATOR", "IN_PANCANPATHWAYS_FREEZE"
    ]
    cleaned_clin_df = cleaned_clin_df.drop(columns=[c for c in cols_to_drop if c in cleaned_clin_df.columns])

    # Reorder so PATIENT_ID is the first column
    if "PATIENT_ID" in cleaned_clin_df.columns:
        cols = ["PATIENT_ID"] + [c for c in cleaned_clin_df.columns if c != "PATIENT_ID"]
        cleaned_clin_df = cleaned_clin_df[cols]

    # Save cleaned
    cleaned_clin_df.to_csv(proc_dir / "clinical_cleaned.csv", index=False)
    cleaned_rnaseq_df.to_csv(proc_dir / "rnaseq_cleaned.csv", index=False)
    print(f"  TCGA-SKCM: Cleaned {len(cleaned_clin_df)} samples. Integrated and tidied treatment data fields (PATIENT_ID is first).")


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
