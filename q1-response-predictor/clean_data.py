import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Import cleaning and utility functions from the modular utils package
from src.utils.preprocessing import clean_clinical_df, clean_rnaseq_df, align_expression_and_clinical

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


def parse_maf_mutations(raw_dir: Path, sample_ids: Optional[list] = None, id_prefix: str = "") -> pd.DataFrame:
    """
    Parses a cBioPortal MAF file to extract binary mutation status for all genes.

    @param Path raw_dir Directory containing data_mutations.txt.
    @param list sample_ids Optional list of sample IDs to restrict to.
    @param str id_prefix Optional prefix to prepend to sample/patient IDs.
    @return pd.DataFrame DataFrame indexed by Tumor_Sample_Barcode with gene symbols as columns.
    """
    mut_path = raw_dir / MUT_FILE
    if not mut_path.exists():
        return pd.DataFrame()

    df_mut = pd.read_csv(mut_path, sep="\t", comment="#", low_memory=False)
    # Prepend prefix and convert to uppercase
    df_mut["Tumor_Sample_Barcode"] = (id_prefix + df_mut["Tumor_Sample_Barcode"].astype(str)).str.upper()

    if sample_ids is not None:
        df_mut = df_mut[df_mut["Tumor_Sample_Barcode"].isin(sample_ids)]

    # Any non-synonymous mutation = mutated
    non_syn = ["Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
               "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
               "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site"]
    df_mut = df_mut[df_mut["Variant_Classification"].isin(non_syn)]

    if df_mut.empty:
        return pd.DataFrame()

    # Pivot: group by sample and gene
    pivoted = df_mut.groupby(["Tumor_Sample_Barcode", "Hugo_Symbol"]).size().unstack(fill_value=0)
    pivoted = (pivoted > 0).astype(int)
    pivoted.index.name = None
    return pivoted


def clean_iatlas_cohort(
    cohort_name: str,
    study_id: str,
    raw_dir: Path,
    proc_dir: Path,
    baseline_only: bool = False,
    mut_by_patient: bool = False,
    patient_prefix: str = "",
    sample_prefix: str = ""
) -> None:
    """
    Generic pipeline function to clean and align iAtlas/cBioPortal datasets.
    """
    proc_dir.mkdir(parents=True, exist_ok=True)

    required = [CLIN_PATIENT_FILE, CLIN_SAMPLE_FILE, EXPR_FILE]
    if not all((raw_dir / f).exists() for f in required):
        raise FileNotFoundError(f"Missing raw input files in {raw_dir}. Please run download_data.py first.")

    print(f"Cleaning {cohort_name} ({study_id})...")

    # Load raw clinical data
    df_patient = pd.read_csv(raw_dir / CLIN_PATIENT_FILE, sep="\t", skiprows=4)
    df_sample = pd.read_csv(raw_dir / CLIN_SAMPLE_FILE, sep="\t", skiprows=4)

    # Prepend dataset prefix and convert to uppercase for Patient and Sample IDs
    df_patient["PATIENT_ID"] = (patient_prefix + df_patient["PATIENT_ID"].astype(str)).str.upper()
    df_sample["PATIENT_ID"] = (patient_prefix + df_sample["PATIENT_ID"].astype(str)).str.upper()
    df_sample["SAMPLE_ID"] = (sample_prefix + df_sample["SAMPLE_ID"].astype(str)).str.upper()

    df_clin = clean_clinical_df(pd.merge(df_sample, df_patient, on="PATIENT_ID"))

    # Optional pre-treatment filter (e.g. for Riaz)
    if baseline_only:
        df_clin = df_clin[df_clin["SAMPLE_ID"].str.endswith("_PRE")]

    # Map response using iAtlas column: keep RESPONSE granular, add RESPONSE_BINARY
    if 'RESPONSE' in df_clin.columns:
        df_clin['RESPONSE_BINARY'] = df_clin['RESPONSE'].map(RESPONSE_MAP)
    else:
        df_clin['RESPONSE_BINARY'] = np.nan
    df_clin = df_clin.set_index("SAMPLE_ID")

    # Standardize clinical metadata columns to uppercase
    if 'AGE_AT_DIAGNOSIS' in df_clin.columns:
        df_clin['AGE'] = df_clin['AGE_AT_DIAGNOSIS'].astype(float)
    if 'SEX' in df_clin.columns:
        df_clin['SEX'] = df_clin['SEX'].map({"Male": "Male", "Female": "Female", "M": "Male", "F": "Female"}).fillna("N/A")

    # Extract mutation status from MAF
    if mut_by_patient:
        df_mut = parse_maf_mutations(raw_dir, id_prefix=sample_prefix)
        if not df_mut.empty:
            # Convert index from sample barcode (e.g., RIAZ_PT3_PRE) to patient ID (e.g., RIAZ_PT3)
            df_mut.index = df_mut.index.map(lambda x: x.rsplit("_", 1)[0] if isinstance(x, str) else x)
            # Aggregate by patient, taking max (1 if mutated in any sample, 0 otherwise)
            df_mut = df_mut.groupby(df_mut.index).max()
            # Map using Patient_ID and reindex to align with df_clin
            df_mut_final = df_mut.reindex(df_clin["PATIENT_ID"], fill_value=0)
            df_mut_final.index = df_clin.index
        else:
            df_mut_final = pd.DataFrame(index=df_clin.index)
    else:
        df_mut = parse_maf_mutations(raw_dir, sample_ids=df_clin.index.tolist(), id_prefix=sample_prefix)
        if not df_mut.empty:
            df_mut_final = df_mut.reindex(df_clin.index, fill_value=0)
        else:
            df_mut_final = pd.DataFrame(index=df_clin.index)

    # Load raw expression (TPM)
    df_expr = parse_cbioportal_expression(raw_dir / EXPR_FILE)
    df_expr.columns = (sample_prefix + df_expr.columns.astype(str)).str.upper()
    if baseline_only:
        pre_cols = [c for c in df_expr.columns if c.endswith("_PRE")]
        df_expr = df_expr[pre_cols].T
    else:
        df_expr = df_expr.T

    # Align samples
    df_expr, df_clin = align_expression_and_clinical(df_expr, df_clin)

    # Log-transform expression
    df_expr = np.log2(df_expr + 1)

    # Align mutation status to final sample IDs and save
    df_mut_final = df_mut_final.loc[df_clin.index]
    df_mut_final.index.name = "SAMPLE_ID"
    df_mut_final.to_csv(proc_dir / "mutations_cleaned.csv")

    # Save cleaned
    df_expr.to_csv(proc_dir / "expr_cleaned.csv")
    
    # Reorder so PATIENT_ID is the first column, and RESPONSE_BINARY is next to RESPONSE
    df_clin = df_clin.reset_index()
    if "PATIENT_ID" in df_clin.columns:
        other_cols = []
        for c in df_clin.columns:
            if c == "PATIENT_ID" or c == "RESPONSE_BINARY":
                continue
            other_cols.append(c)
            if c == "RESPONSE":
                other_cols.append("RESPONSE_BINARY")
        if "RESPONSE_BINARY" not in other_cols and "RESPONSE_BINARY" in df_clin.columns:
            other_cols.append("RESPONSE_BINARY")
        cols = ["PATIENT_ID"] + other_cols
        df_clin = df_clin[cols]
        
    df_clin.to_csv(proc_dir / "clin_cleaned.csv", index=False)
    print(f"  {cohort_name}: Cleaned {len(df_clin)} samples.")


def clean_liu_2019() -> None:
    """
    Cleans raw Liu 2019 datasets and writes processed matrices to the processed folder.
    """
    clean_iatlas_cohort(
        cohort_name="Liu 2019",
        study_id=LIU_STUDY_ID,
        raw_dir=RAW_DIR / "liu_2019",
        proc_dir=PROCESSED_DIR / "liu_2019",
        baseline_only=False,
        mut_by_patient=False,
        patient_prefix="liu_",
        sample_prefix=""
    )


def clean_hugo_2016() -> None:
    """
    Cleans Hugo 2016 (mel_iatlas_hugo_ucla_2016) cBioPortal datasets and writes
    processed matrices to the processed folder.
    """
    clean_iatlas_cohort(
        cohort_name="Hugo 2016",
        study_id=HUGO_STUDY_ID,
        raw_dir=RAW_DIR / "hugo_2016",
        proc_dir=PROCESSED_DIR / "hugo_2016",
        baseline_only=False,
        mut_by_patient=False,
        patient_prefix="hugo_",
        sample_prefix="hugo_"
    )


def clean_riaz_2017() -> None:
    """
    Cleans Riaz 2017 (mel_iatlas_riaz_nivolumab_2017) cBioPortal datasets and writes
    processed matrices to the processed folder.
    """
    clean_iatlas_cohort(
        cohort_name="Riaz 2017",
        study_id=RIAZ_STUDY_ID,
        raw_dir=RAW_DIR / "riaz_2017",
        proc_dir=PROCESSED_DIR / "riaz_2017",
        baseline_only=False,
        mut_by_patient=True,
        patient_prefix="riaz_",
        sample_prefix="riaz_"
    )



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
    
    # Map Entrez IDs to Hugo Symbols using modular preprocessing helper
    from src.utils.preprocessing import map_entrez_to_symbols
    entrez_ids = df_expr["Entrez_Gene_Id"].unique().tolist()
    cache_path = proc_dir / "entrez_to_symbol_cache.json"
    gene_map = map_entrez_to_symbols(entrez_ids, cache_path=cache_path)
    
    # Apply mapping
    df_expr["Hugo_Symbol_Mapped"] = df_expr["Entrez_Gene_Id"].map(gene_map)
    # Fallback to Entrez Gene ID if symbol not found
    df_expr["Hugo_Symbol_Mapped"] = df_expr["Hugo_Symbol_Mapped"].fillna(df_expr["Entrez_Gene_Id"])
    
    df_expr = df_expr.set_index("Hugo_Symbol_Mapped")
    if "Hugo_Symbol" in df_expr.columns:
        df_expr = df_expr.drop(columns=["Hugo_Symbol"])
    if "Entrez_Gene_Id" in df_expr.columns:
        df_expr = df_expr.drop(columns=["Entrez_Gene_Id"])
        
    # Average duplicate Hugo Symbols
    df_expr = df_expr.groupby(df_expr.index).mean()
    df_expr = df_expr.T
    df_expr.index.name = "SAMPLE_ID"
    df_expr = df_expr.reset_index()
    
    cleaned_rnaseq_df = clean_rnaseq_df(df_expr)
    
    # Apply log2(x + 1) transformation directly to the gene expression columns
    gene_cols = [c for c in cleaned_rnaseq_df.columns if c != "SAMPLE_ID"]
    cleaned_rnaseq_df[gene_cols] = np.log2(cleaned_rnaseq_df[gene_cols] + 1)

    # Drop constant, redundant, and administrative columns
    cols_to_drop = [
        "CANCER_TYPE", "CANCER_TYPE_DETAILED", "ONCOTREE_CODE", "CANCER_TYPE_ACRONYM",
        "SUBTYPE", "TUMOR_TYPE", "SOMATIC_STATUS", "DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS",
        "INFORMED_CONSENT_VERIFIED", "DAYS_TO_BIRTH", "OTHER_PATIENT_ID", "TISSUE_SOURCE_SITE_CODE",
        "TISSUE_RETROSPECTIVE_COLLECTION_INDICATOR", "FORM_COMPLETION_DATE", "AJCC_STAGING_EDITION",
        "SAMPLE_COUNT", "TISSUE_PROSPECTIVE_COLLECTION_INDICATOR", "IN_PANCANPATHWAYS_FREEZE",
        "GRADE", "TISSUE_SOURCE_SITE"
    ]
    cleaned_clin_df = cleaned_clin_df.drop(columns=[c for c in cols_to_drop if c in cleaned_clin_df.columns])

    # Reorder so PATIENT_ID is the first column
    if "PATIENT_ID" in cleaned_clin_df.columns:
        cols = ["PATIENT_ID"] + [c for c in cleaned_clin_df.columns if c != "PATIENT_ID"]
        cleaned_clin_df = cleaned_clin_df[cols]

    # Save cleaned
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
