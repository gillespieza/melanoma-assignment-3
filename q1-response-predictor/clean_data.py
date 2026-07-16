import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Import TCGA cleaning functions
from src.tcga_helpers import clean_clinical_df, clean_rnaseq_df

# Import utility helpers
from src.utils import align_expression_and_clinical

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Study IDs
LIU_STUDY_ID = "mel_iatlas_liu_2019"
HUGO_STUDY_ID = "mel_iatlas_hugo_ucla_2016"
RIAZ_STUDY_ID = "mel_iatlas_riaz_nivolumab_2017"
TCGA_STUDY_ID = "skcm_tcga_pan_can_atlas_2018"

# Shared cBioPortal raw filenames
CLIN_PATIENT_FILE = "data_clinical_patient.txt"
CLIN_SAMPLE_FILE = "data_clinical_sample.txt"
EXPR_FILE = "data_mrna_seq_tpm.txt"
MUT_FILE = "data_mutations.txt"

# TCGA uses RSEM counts, not TPM
TCGA_EXPR_FILE = "data_mrna_seq_v2_rsem.txt"

# Standard response mapping for all iAtlas / cBioPortal cohorts
RESPONSE_MAP = {
    "Complete Response": 1,
    "Partial Response": 1,
    "Progressive Disease": 0,
    "Stable Disease": np.nan,
    "Mixed Response": np.nan,
}


def parse_cbioportal_expression(expr_file_path: Path) -> pd.DataFrame:
    """
    Loads a raw cBioPortal expression file, handles Entrez_Gene_Id dropping,
    averages duplicates by Hugo_Symbol, and returns the gene x sample dataframe.
    """
    df_expr = pd.read_csv(expr_file_path, sep="\t")
    df_expr = df_expr.dropna(subset=["Hugo_Symbol"])
    df_expr = df_expr.set_index("Hugo_Symbol")
    if "Entrez_Gene_Id" in df_expr.columns:
        df_expr = df_expr.drop(columns=["Entrez_Gene_Id"])
    df_expr = df_expr.groupby(df_expr.index).mean()
    return df_expr


def parse_maf_mutations(raw_dir: Path, sample_ids: Optional[list] = None) -> pd.DataFrame:
    """
    Parses BRAF, NRAS, NF1 binary mutation status from a cBioPortal MAF file.

    @param Path raw_dir Directory containing data_mutations.txt.
    @param list sample_ids Optional list of sample IDs to restrict to.
    @return pd.DataFrame DataFrame indexed by SAMPLE_ID with columns mut_BRAF, mut_NRAS, mut_NF1.
    """
    mut_path = raw_dir / MUT_FILE
    if not mut_path.exists():
        return pd.DataFrame(columns=["mut_BRAF", "mut_NRAS", "mut_NF1"])

    df_mut = pd.read_csv(mut_path, sep="\t", comment="#", low_memory=False)
    df_mut = df_mut[df_mut["Hugo_Symbol"].isin(["BRAF", "NRAS", "NF1"])]
    if sample_ids is not None:
        df_mut = df_mut[df_mut["Tumor_Sample_Barcode"].isin(sample_ids)]

    # Any non-synonymous mutation = mutated
    non_syn = ["Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
               "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
               "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site"]
    df_mut = df_mut[df_mut["Variant_Classification"].isin(non_syn)]

    pivoted = df_mut.groupby(["Tumor_Sample_Barcode", "Hugo_Symbol"]).size().unstack(fill_value=0)
    for gene in ["BRAF", "NRAS", "NF1"]:
        if gene not in pivoted.columns:
            pivoted[gene] = 0
    pivoted = (pivoted[["BRAF", "NRAS", "NF1"]] > 0).astype(int)
    pivoted.columns = ["mut_BRAF", "mut_NRAS", "mut_NF1"]
    pivoted.index.name = None
    return pivoted


def clean_liu_2019() -> None:
    """
    Cleans raw Liu 2019 datasets and writes processed matrices to the processed folder.

    @return None
    """
    raw_dir = RAW_DIR / "liu_2019"
    proc_dir = PROCESSED_DIR / "liu_2019"
    proc_dir.mkdir(parents=True, exist_ok=True)

    if not all((raw_dir / f).exists() for f in [CLIN_PATIENT_FILE, CLIN_SAMPLE_FILE, EXPR_FILE]):
        raise FileNotFoundError(f"Missing raw input files in {raw_dir}. Please run download_data.py first.")

    print("Cleaning Liu 2019 (mel_iatlas_liu_2019)...")

    # Load raw clinical data
    df_patient = pd.read_csv(raw_dir / CLIN_PATIENT_FILE, sep="\t", skiprows=4)
    df_sample = pd.read_csv(raw_dir / CLIN_SAMPLE_FILE, sep="\t", skiprows=4)
    df_clin = clean_clinical_df(pd.merge(df_sample, df_patient, on="PATIENT_ID"))

    # Map response using iAtlas column
    df_clin['response'] = df_clin['RESPONSE'].map(RESPONSE_MAP)
    df_clin = df_clin.dropna(subset=['response'])
    df_clin = df_clin.set_index("SAMPLE_ID")

    # Standardize clinical metadata columns
    df_clin['patient_id'] = df_clin['PATIENT_ID']
    df_clin['os_months'] = df_clin['OS_MONTHS']
    df_clin['os_status'] = df_clin['OS_STATUS']
    if 'AGE_AT_DIAGNOSIS' in df_clin.columns:
        df_clin['age (yrs)'] = df_clin['AGE_AT_DIAGNOSIS']
    if 'SEX' in df_clin.columns:
        df_clin['gender'] = df_clin['SEX']

    # Append mutation status (BRAF, NRAS, NF1) from MAF
    df_mut = parse_maf_mutations(raw_dir, sample_ids=df_clin.index.tolist())
    df_clin = df_clin.join(df_mut, how="left")
    for col in ["mut_BRAF", "mut_NRAS", "mut_NF1"]:
        df_clin[col] = df_clin[col].fillna(0).astype(int)

    # Load raw expression (TPM)
    df_expr = parse_cbioportal_expression(raw_dir / EXPR_FILE).T

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
    Cleans Hugo 2016 (mel_iatlas_hugo_ucla_2016) cBioPortal datasets and writes
    processed matrices to the processed folder.

    Clinical source: data_clinical_patient.txt + data_clinical_sample.txt
    Expression source: data_mrna_seq_tpm.txt (Hugo gene symbols, TPM)
    Mutations source: data_mutations.txt (MAF format)

    @return None
    """
    raw_dir = RAW_DIR / "hugo_2016"
    proc_dir = PROCESSED_DIR / "hugo_2016"
    proc_dir.mkdir(parents=True, exist_ok=True)

    required = [CLIN_PATIENT_FILE, CLIN_SAMPLE_FILE, EXPR_FILE]
    if not all((raw_dir / f).exists() for f in required):
        raise FileNotFoundError(f"Missing raw input files in {raw_dir}. Please run download_data.py first.")

    print(f"Cleaning Hugo 2016 ({HUGO_STUDY_ID})...")

    # Load raw clinical data
    df_patient = pd.read_csv(raw_dir / CLIN_PATIENT_FILE, sep="\t", skiprows=4)
    df_sample = pd.read_csv(raw_dir / CLIN_SAMPLE_FILE, sep="\t", skiprows=4)
    df_clin = clean_clinical_df(pd.merge(df_sample, df_patient, on="PATIENT_ID"))
    
    # Map Response using iAtlas column
    df_clin['response'] = df_clin['RESPONSE'].map(RESPONSE_MAP)
    df_clin = df_clin.dropna(subset=['response'])
    df_clin = df_clin.set_index("SAMPLE_ID")

    # Standardize clinical metadata columns for downstream compatibility
    df_clin['patient_id'] = df_clin['PATIENT_ID']
    df_clin['os_months'] = df_clin['OS_MONTHS']
    df_clin['os_status'] = df_clin['OS_STATUS']
    if 'AGE_AT_DIAGNOSIS' in df_clin.columns:
        df_clin['age (yrs)'] = df_clin['AGE_AT_DIAGNOSIS']
    if 'SEX' in df_clin.columns:
        df_clin['gender'] = df_clin['SEX']

    # Append mutation status (BRAF, NRAS, NF1) from MAF
    df_mut = parse_maf_mutations(raw_dir, sample_ids=df_clin.index.tolist())
    df_clin = df_clin.join(df_mut, how="left")
    for col in ["mut_BRAF", "mut_NRAS", "mut_NF1"]:
        df_clin[col] = df_clin[col].fillna(0).astype(int)

    # --- Expression (TPM, Hugo symbols) ---
    df_expr = parse_cbioportal_expression(raw_dir / EXPR_FILE).T

    # Align samples
    df_expr, df_clin = align_expression_and_clinical(df_expr, df_clin)

    # Log-transform
    df_expr = np.log2(df_expr + 1)

    # Save cleaned
    df_expr.to_csv(proc_dir / "expr_cleaned.csv")
    df_clin.to_csv(proc_dir / "clin_cleaned.csv")
    print(f"  Hugo 2016: Cleaned {len(df_clin)} samples, {df_expr.shape[1]} genes.")


def clean_riaz_2017() -> None:
    """
    Cleans Riaz 2017 (mel_iatlas_riaz_nivolumab_2017) cBioPortal datasets and writes
    processed matrices to the processed folder.

    Filters to pre-treatment baseline samples only (SAMPLE_ID ending in '_pre').
    Clinical source: data_clinical_patient.txt + data_clinical_sample.txt
    Expression source: data_mrna_seq_tpm.txt (Hugo gene symbols, TPM)
    Mutations source: data_mutations.txt (MAF format)

    @return None
    """
    raw_dir = RAW_DIR / "riaz_2017"
    proc_dir = PROCESSED_DIR / "riaz_2017"
    proc_dir.mkdir(parents=True, exist_ok=True)

    required = [CLIN_PATIENT_FILE, CLIN_SAMPLE_FILE, EXPR_FILE]
    if not all((raw_dir / f).exists() for f in required):
        raise FileNotFoundError(f"Missing raw input files in {raw_dir}. Please run download_data.py first.")

    print(f"Cleaning Riaz 2017 ({RIAZ_STUDY_ID})...")

    # --- Clinical ---
    df_patient = pd.read_csv(raw_dir / CLIN_PATIENT_FILE, sep="\t", skiprows=4)
    df_sample = pd.read_csv(raw_dir / CLIN_SAMPLE_FILE, sep="\t", skiprows=4)
    df_clin = clean_clinical_df(pd.merge(df_sample, df_patient, on="PATIENT_ID"))

    # Filter to pre-treatment baseline biopsies only
    df_clin = df_clin[df_clin["SAMPLE_ID"].str.endswith("_pre")]

    # Map response: CR/PR -> 1, PD -> 0, others -> NaN
    df_clin["response"] = df_clin["RESPONSE"].map(RESPONSE_MAP)
    df_clin = df_clin.dropna(subset=["response"])
    df_clin = df_clin.set_index("SAMPLE_ID")

    # Standardise columns for downstream compatibility
    df_clin["patient_id"] = df_clin["PATIENT_ID"]
    df_clin["os_months"] = df_clin["OS_MONTHS"]
    df_clin["os_status"] = df_clin["OS_STATUS"]
    if "AGE_AT_DIAGNOSIS" in df_clin.columns:
        df_clin["age"] = df_clin["AGE_AT_DIAGNOSIS"]
    if "SEX" in df_clin.columns:
        df_clin["sex"] = df_clin["SEX"]

    # Append mutation status (BRAF, NRAS, NF1) from MAF
    # Note: Riaz MAF sample IDs match the PATIENT_ID, not the SAMPLE_ID (_pre/_on suffix)
    # Map by PATIENT_ID to join onto the sample-level clinical df
    df_mut = parse_maf_mutations(raw_dir)
    pat_mut = df_clin["patient_id"].map(df_mut["mut_BRAF"].to_dict()).fillna(0).astype(int)
    df_clin["mut_BRAF"] = pat_mut
    df_clin["mut_NRAS"] = df_clin["patient_id"].map(df_mut["mut_NRAS"].to_dict()).fillna(0).astype(int)
    df_clin["mut_NF1"] = df_clin["patient_id"].map(df_mut["mut_NF1"].to_dict()).fillna(0).astype(int)

    # --- Expression (TPM, Hugo symbols) ---
    df_expr = parse_cbioportal_expression(raw_dir / EXPR_FILE)
    # Filter expression columns to pre-treatment samples only
    pre_cols = [c for c in df_expr.columns if c.endswith("_pre")]
    df_expr = df_expr[pre_cols].T

    # Align samples
    df_expr, df_clin = align_expression_and_clinical(df_expr, df_clin)

    # Log-transform
    df_expr = np.log2(df_expr + 1)

    # Save cleaned
    df_expr.to_csv(proc_dir / "expr_cleaned.csv")
    df_clin.to_csv(proc_dir / "clin_cleaned.csv")
    print(f"  Riaz 2017: Cleaned {len(df_clin)} samples, {df_expr.shape[1]} genes.")


def clean_tcga_skcm() -> None:
    """
    Cleans raw TCGA-SKCM datasets and writes processed matrices to the processed folder.

    @return None
    """
    raw_dir = RAW_DIR / TCGA_STUDY_ID
    proc_dir = PROCESSED_DIR / TCGA_STUDY_ID
    proc_dir.mkdir(parents=True, exist_ok=True)

    if not all((raw_dir / f).exists() for f in [CLIN_PATIENT_FILE, CLIN_SAMPLE_FILE, TCGA_EXPR_FILE]):
        raise FileNotFoundError(f"Missing raw input files in {raw_dir}. Please run download_data.py first.")

    print(f"Cleaning TCGA-SKCM ({TCGA_STUDY_ID})...")

    # Clean Clinical
    df_patient = pd.read_csv(raw_dir / CLIN_PATIENT_FILE, sep="\t", skiprows=4)
    df_sample = pd.read_csv(raw_dir / CLIN_SAMPLE_FILE, sep="\t", skiprows=4)
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
    cleaned_clin_df.to_csv(proc_dir / "clin_cleaned.csv", index=False)
    cleaned_rnaseq_df.to_csv(proc_dir / "expr_cleaned.csv", index=False)
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
