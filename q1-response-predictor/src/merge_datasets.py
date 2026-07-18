import pandas as pd
import numpy as np
import json
from pathlib import Path
# pyCombat import removed to support cohort-independent Z-score standardization

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MERGED_DIR = PROCESSED_DIR / "merged"

# Subdirectories for the two merge variants
FULL_DIR = MERGED_DIR / "full"
IMMUNO_DIR = MERGED_DIR / "immunotherapy"

for d in [FULL_DIR, IMMUNO_DIR]:
    d.mkdir(exist_ok=True, parents=True)

def load_tcga():
    print("Loading TCGA-SKCM...")
    tcga_dir = PROCESSED_DIR / "skcm_tcga_pan_can_atlas_2018"
    df_expr = pd.read_csv(tcga_dir / "expr_cleaned.csv", index_col="SAMPLE_ID")
    df_clin = pd.read_csv(tcga_dir / "clin_cleaned.csv")
    df_expr_log = df_expr
    
    # Align sample ids
    common_ids = list(df_expr_log.index.intersection(df_clin['SAMPLE_ID']))
    df_expr_log = df_expr_log.loc[common_ids]
    df_clin = df_clin.set_index('SAMPLE_ID').loc[common_ids]
    
    # Extract clinical details
    df_clin_harm = pd.DataFrame(index=df_clin.index)
    df_clin_harm["PATIENT_ID"] = df_clin["PATIENT_ID"]
    df_clin_harm["COHORT"] = "TCGA"
    df_clin_harm["OS_MONTHS"] = df_clin["OS_MONTHS"]
    df_clin_harm["OS_STATUS"] = df_clin["OS_STATUS"].astype(float)
    df_clin_harm["RESPONSE"] = np.nan
    df_clin_harm["RESPONSE_BINARY"] = np.nan
    df_clin_harm["AGE"] = df_clin["AGE"]
    df_clin_harm["RACE"] = df_clin["RACE"]
    
    # Sex: standardise values
    df_clin_harm["SEX"] = df_clin["SEX"].map({"Male": "Male", "Female": "Female"}).fillna("N/A")
    df_clin_harm["SPECIMEN_TYPE"] = df_clin["SAMPLE_TYPE"].map({"Primary": "Primary", "Metastatic": "Metastatic"}).fillna("N/A")
    
    # Carry forward the immunotherapy flag so we can filter later
    if "TX_TYPE_IMMUNOTHERAPY" in df_clin.columns:
        df_clin_harm["IMMUNOTHERAPY"] = df_clin["TX_TYPE_IMMUNOTHERAPY"].astype(int)
    else:
        df_clin_harm["IMMUNOTHERAPY"] = 0
    
    return df_expr_log, df_clin_harm


def load_liu():
    print("Loading Liu 2019...")
    liu_dir = PROCESSED_DIR / "liu_2019"
    df_expr = pd.read_csv(liu_dir / "expr_cleaned.csv", index_col=0)
    df_clin = pd.read_csv(liu_dir / "clin_cleaned.csv", index_col="SAMPLE_ID")
    
    # Align IDs
    common_ids = list(set(df_expr.index) & set(df_clin.index))
    df_expr = df_expr.loc[common_ids]
    df_clin = df_clin.loc[common_ids]
    
    df_clin_harm = pd.DataFrame(index=df_clin.index)
    df_clin_harm["PATIENT_ID"] = df_clin["PATIENT_ID"]
    df_clin_harm["COHORT"] = "Liu_2019"
    df_clin_harm["OS_MONTHS"] = df_clin["OS_MONTHS"]
    
    # Map status '1:DECEASED' -> 1, '0:LIVING' -> 0
    df_clin_harm["OS_STATUS"] = df_clin["OS_STATUS"].map({"1:DECEASED": 1.0, "0:LIVING": 0.0})
    df_clin_harm["RESPONSE"] = df_clin["RESPONSE"]
    df_clin_harm["RESPONSE_BINARY"] = df_clin["RESPONSE_BINARY"].astype(float)
    df_clin_harm["AGE"] = np.nan  # Not available
    df_clin_harm["RACE"] = np.nan  # Not available
    df_clin_harm["SEX"] = df_clin["SEX"].map({"Male": "Male", "Female": "Female"}).fillna("N/A")
    
    # Specimen type (Liu's cleaned CSV uses 'BIOPSY_SITE', mostly metastatic sites)
    if "BIOPSY_SITE" in df_clin.columns:
        spec_map = lambda s: "Primary" if "primary" in str(s).lower() else ("Metastatic" if pd.notna(s) and str(s).strip() != "" else "N/A")
        df_clin_harm["SPECIMEN_TYPE"] = df_clin["BIOPSY_SITE"].map(spec_map)
    else:
        df_clin_harm["SPECIMEN_TYPE"] = "N/A"
    
    # All Liu patients received anti-PD-1 immunotherapy
    df_clin_harm["IMMUNOTHERAPY"] = 1
    
    return df_expr, df_clin_harm


def load_hugo():
    print("Loading Hugo 2016...")
    hugo_dir = PROCESSED_DIR / "hugo_2016"
    df_expr = pd.read_csv(hugo_dir / "expr_cleaned.csv", index_col=0)
    df_clin = pd.read_csv(hugo_dir / "clin_cleaned.csv", index_col="SAMPLE_ID")
    
    common_ids = list(set(df_expr.index) & set(df_clin.index))
    df_expr = df_expr.loc[common_ids]
    df_clin = df_clin.loc[common_ids]
    
    df_clin_harm = pd.DataFrame(index=df_clin.index)
    df_clin_harm["PATIENT_ID"] = df_clin["PATIENT_ID"]
    df_clin_harm["COHORT"] = "Hugo_2016"
    
    # os_months from cBioPortal iAtlas clinical file
    df_clin_harm["OS_MONTHS"] = df_clin["OS_MONTHS"]
    df_clin_harm["OS_STATUS"] = df_clin["OS_STATUS"].astype(float)
    df_clin_harm["RESPONSE"] = df_clin["RESPONSE"]
    df_clin_harm["RESPONSE_BINARY"] = df_clin["RESPONSE_BINARY"].astype(float)
    df_clin_harm["AGE"] = df_clin["AGE"].astype(float)
    df_clin_harm["RACE"] = np.nan  # Not available
    df_clin_harm["SEX"] = df_clin["SEX"].map({"Male": "Male", "Female": "Female", "M": "Male", "F": "Female"}).fillna("N/A")
    df_clin_harm["SPECIMEN_TYPE"] = "Metastatic"  # All were pre-treatment metastatic
    
    # All Hugo patients received anti-PD-1 immunotherapy
    df_clin_harm["IMMUNOTHERAPY"] = 1
    
    return df_expr, df_clin_harm


def load_riaz():
    print("Loading Riaz 2017...")
    riaz_dir = PROCESSED_DIR / "riaz_2017"
    df_expr = pd.read_csv(riaz_dir / "expr_cleaned.csv", index_col=0)
    df_clin = pd.read_csv(riaz_dir / "clin_cleaned.csv", index_col="SAMPLE_ID")
    
    common_ids = list(set(df_expr.index) & set(df_clin.index))
    df_expr = df_expr.loc[common_ids]
    df_clin = df_clin.loc[common_ids]
    
    df_clin_harm = pd.DataFrame(index=df_clin.index)
    df_clin_harm["PATIENT_ID"] = df_clin["PATIENT_ID"]
    df_clin_harm["COHORT"] = "Riaz_2017"
    df_clin_harm["OS_MONTHS"] = df_clin["OS_MONTHS"]
    df_clin_harm["OS_STATUS"] = df_clin["OS_STATUS"].astype(float)
    df_clin_harm["RESPONSE"] = df_clin["RESPONSE"]
    df_clin_harm["RESPONSE_BINARY"] = df_clin["RESPONSE_BINARY"].astype(float)
    df_clin_harm["AGE"] = df_clin["AGE"].astype(float)
    df_clin_harm["RACE"] = df_clin["RACE"]
    df_clin_harm["SEX"] = df_clin["SEX"].map({"Male": "Male", "Female": "Female"}).fillna("N/A")
    df_clin_harm["SPECIMEN_TYPE"] = "Metastatic"
    
    # All Riaz patients received nivolumab (anti-PD-1)
    df_clin_harm["IMMUNOTHERAPY"] = 1
    
    return df_expr, df_clin_harm


def zscore_expression(df):
    """
    Standardize expression matrix columns individually (Z-score scaling).
    Avoids division by zero if std is zero.
    """
    means = df.mean(axis=0)
    stds = df.std(axis=0)
    stds = stds.replace(0, 1.0).fillna(1.0)
    return (df - means) / stds


def batch_correct_and_save(df_expr_merged, df_clin_merged, output_dir, label=""):
    """
    Saves the pre-standardized expression matrix and clinical metadata
    to the specified output directory.
    """
    prefix = f"[{label}] " if label else ""

    # Check that indices match perfectly
    assert (df_expr_merged.index == df_clin_merged.index).all(), "Inconsistent sample indices!"
    print(f"{prefix}Total cohort size: {len(df_clin_merged)} samples.")

    # Save results
    output_dir.mkdir(exist_ok=True, parents=True)
    expr_out_path = output_dir / "expr_merged.csv"
    clin_out_path = output_dir / "clin_merged.csv"

    df_expr_merged.to_csv(expr_out_path)
    
    # Reorder df_clin_merged to put PATIENT_ID first
    df_clin_merged = df_clin_merged.reset_index()
    if "PATIENT_ID" in df_clin_merged.columns:
        cols = ["PATIENT_ID"] + [c for c in df_clin_merged.columns if c != "PATIENT_ID"]
        df_clin_merged = df_clin_merged[cols]
    df_clin_merged.to_csv(clin_out_path, index=False)

    print(f"{prefix}Saved expression matrix to: {expr_out_path} (shape: {df_expr_merged.shape})")
    print(f"{prefix}Saved clinical metadata to:  {clin_out_path} (shape: {df_clin_merged.shape})")

    # Build and save merged genomic features
    build_merged_genomic(df_clin_merged, output_dir)


def parse_tcga_mutations(raw_dir: Path, sample_ids: list) -> pd.DataFrame:
    mut_path = raw_dir / "data_mutations.txt"
    if not mut_path.exists():
        return pd.DataFrame(0, index=sample_ids, columns=["mut_BRAF", "mut_NRAS", "mut_NF1"])
    
    # Read mutations in chunks to be memory efficient
    chunks = []
    for chunk in pd.read_csv(mut_path, sep="\t", comment="#", low_memory=False, 
                             usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"], 
                             chunksize=100000):
        filtered = chunk[chunk["Hugo_Symbol"].isin(["BRAF", "NRAS", "NF1"])]
        chunks.append(filtered)
    
    df_mut = pd.concat(chunks, ignore_index=True)
    
    non_syn = ["Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
               "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
               "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site"]
    df_mut = df_mut[df_mut["Variant_Classification"].isin(non_syn)]
    
    df_mut["SAMPLE_ID"] = df_mut["Tumor_Sample_Barcode"].apply(lambda x: x[:15] if isinstance(x, str) else "").str.upper()
    df_mut = df_mut[df_mut["SAMPLE_ID"].isin(sample_ids)]
    
    pivoted = df_mut.groupby(["SAMPLE_ID", "Hugo_Symbol"]).size().unstack(fill_value=0)
    for gene in ["BRAF", "NRAS", "NF1"]:
        if gene not in pivoted.columns:
            pivoted[gene] = 0
            
    pivoted = (pivoted[["BRAF", "NRAS", "NF1"]] > 0).astype(int)
    pivoted.columns = ["mut_BRAF", "mut_NRAS", "mut_NF1"]
    pivoted = pivoted.reindex(sample_ids, fill_value=0)
    return pivoted


def build_merged_genomic(df_clin_merged: pd.DataFrame, output_dir: Path) -> None:
    """
    Builds and saves merged_genomic.csv for the samples in df_clin_merged.
    """
    RAW_DIR = DATA_DIR / "raw"
    print(f"Building merged genomic features file for {output_dir.name} cohort...")
    
    cohort_data = {}
    
    # 1. Liu 2019
    liu_dir = PROCESSED_DIR / "liu_2019"
    df_clin_liu = pd.read_csv(liu_dir / "clin_cleaned.csv", index_col="SAMPLE_ID")
    df_mut_liu = pd.read_csv(liu_dir / "mutations_cleaned.csv", index_col="SAMPLE_ID")
    df_mut_liu = df_mut_liu[['BRAF', 'NRAS', 'NF1']].rename(
        columns={'BRAF': 'mut_BRAF', 'NRAS': 'mut_NRAS', 'NF1': 'mut_NF1'}
    )
    cohort_data["Liu_2019"] = df_clin_liu.join(df_mut_liu, how="left")
    
    # 2. Hugo 2016
    hugo_dir = PROCESSED_DIR / "hugo_2016"
    df_clin_hugo = pd.read_csv(hugo_dir / "clin_cleaned.csv", index_col="SAMPLE_ID")
    df_mut_hugo = pd.read_csv(hugo_dir / "mutations_cleaned.csv", index_col="SAMPLE_ID")
    df_mut_hugo = df_mut_hugo[['BRAF', 'NRAS', 'NF1']].rename(
        columns={'BRAF': 'mut_BRAF', 'NRAS': 'mut_NRAS', 'NF1': 'mut_NF1'}
    )
    cohort_data["Hugo_2016"] = df_clin_hugo.join(df_mut_hugo, how="left")
    
    # 3. Riaz 2017
    riaz_dir = PROCESSED_DIR / "riaz_2017"
    df_clin_riaz = pd.read_csv(riaz_dir / "clin_cleaned.csv", index_col="SAMPLE_ID")
    df_mut_riaz = pd.read_csv(riaz_dir / "mutations_cleaned.csv", index_col="SAMPLE_ID")
    df_mut_riaz = df_mut_riaz[['BRAF', 'NRAS', 'NF1']].rename(
        columns={'BRAF': 'mut_BRAF', 'NRAS': 'mut_NRAS', 'NF1': 'mut_NF1'}
    )
    cohort_data["Riaz_2017"] = df_clin_riaz.join(df_mut_riaz, how="left")
    
    # 4. TCGA
    tcga_dir = PROCESSED_DIR / "skcm_tcga_pan_can_atlas_2018"
    df_clin_tcga = pd.read_csv(tcga_dir / "clin_cleaned.csv", index_col="SAMPLE_ID")
    raw_tcga_dir = RAW_DIR / "skcm_tcga_pan_can_atlas_2018"
    df_mut_tcga = parse_tcga_mutations(raw_tcga_dir, df_clin_tcga.index.tolist())
    cohort_data["TCGA"] = df_clin_tcga.join(df_mut_tcga, how="left")
    
    genomic_rows = []
    
    if "SAMPLE_ID" in df_clin_merged.columns:
        samples_df = df_clin_merged
    else:
        samples_df = df_clin_merged.reset_index()
        
    for _, row in samples_df.iterrows():
        sample_id = row["SAMPLE_ID"]
        cohort = row["COHORT"]
        patient_id = row["PATIENT_ID"]
        
        feat_dict = {
            "PATIENT_ID": patient_id,
            "SAMPLE_ID": sample_id,
            "COHORT": cohort,
            "TMB_NONSYNONYMOUS": np.nan,
            "mut_BRAF": 0,
            "mut_NRAS": 0,
            "mut_NF1": 0,
            "SNV_NEOANTIGEN": np.nan,
            "INDEL_NEOANTIGEN": np.nan,
            "FUSION_NEOANTIGEN": np.nan,
            "SPLICE_NEOANTIGEN": np.nan,
            "CTA_SELF_NEOANTIGEN": np.nan
        }
        
        if cohort in cohort_data:
            df_cohort = cohort_data[cohort]
            if sample_id in df_cohort.index:
                cohort_row = df_cohort.loc[sample_id]
                if isinstance(cohort_row, pd.DataFrame):
                    cohort_row = cohort_row.iloc[0]
                    
                if "TMB_NONSYNONYMOUS" in cohort_row:
                    feat_dict["TMB_NONSYNONYMOUS"] = cohort_row["TMB_NONSYNONYMOUS"]
                    
                for m_gene in ["mut_BRAF", "mut_NRAS", "mut_NF1"]:
                    if m_gene in cohort_row:
                        feat_dict[m_gene] = int(cohort_row[m_gene]) if pd.notna(cohort_row[m_gene]) else 0
                        
                for neo_feat in ["SNV_NEOANTIGEN", "INDEL_NEOANTIGEN", "FUSION_NEOANTIGEN", "SPLICE_NEOANTIGEN", "CTA_SELF_NEOANTIGEN"]:
                    if neo_feat in cohort_row:
                        feat_dict[neo_feat] = cohort_row[neo_feat]
                        
        genomic_rows.append(feat_dict)
        
    df_genomic = pd.DataFrame(genomic_rows)
    
    cols_order = [
        "PATIENT_ID", "SAMPLE_ID", "COHORT",
        "TMB_NONSYNONYMOUS",
        "mut_BRAF", "mut_NRAS", "mut_NF1",
        "SNV_NEOANTIGEN", "INDEL_NEOANTIGEN", "FUSION_NEOANTIGEN", "SPLICE_NEOANTIGEN", "CTA_SELF_NEOANTIGEN"
    ]
    df_genomic = df_genomic[cols_order]
    
    out_path = output_dir / "merged_genomic.csv"
    df_genomic.to_csv(out_path, index=False)
    print(f"Saved merged genomic features to: {out_path} (shape: {df_genomic.shape})")


def main():
    # 1. Load each dataset
    expr_tcga, clin_tcga = load_tcga()
    expr_liu, clin_liu = load_liu()
    expr_hugo, clin_hugo = load_hugo()
    expr_riaz, clin_riaz = load_riaz()

    print("\nAligning gene features...")
    # Find intersecting gene symbols
    common_genes = list(
        set(expr_tcga.columns) &
        set(expr_liu.columns) &
        set(expr_hugo.columns) &
        set(expr_riaz.columns)
    )
    common_genes.sort()
    print(f"Number of common genes intersected: {len(common_genes)}")

    # Subset expressions to common genes
    expr_tcga = expr_tcga[common_genes]
    expr_liu = expr_liu[common_genes]
    expr_hugo = expr_hugo[common_genes]
    expr_riaz = expr_riaz[common_genes]

    # Standardize each cohort individually to avoid data leakage
    print("Standardizing expression datasets individually (Z-score)...")
    expr_tcga_scaled = zscore_expression(expr_tcga)
    expr_liu_scaled = zscore_expression(expr_liu)
    expr_hugo_scaled = zscore_expression(expr_hugo)
    expr_riaz_scaled = zscore_expression(expr_riaz)

    # ------------------------------------------------------------------
    # 2a. FULL MERGE: all 4 cohorts, all patients
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Building FULL merged cohort (all 4 datasets)")
    print("=" * 60)

    df_expr_full = pd.concat([expr_tcga_scaled, expr_liu_scaled, expr_hugo_scaled, expr_riaz_scaled], axis=0)
    df_clin_full = pd.concat([clin_tcga, clin_liu, clin_hugo, clin_riaz], axis=0)

    batch_correct_and_save(df_expr_full, df_clin_full, FULL_DIR, label="Full")

    # ------------------------------------------------------------------
    # 2b. IMMUNOTHERAPY-ONLY MERGE: Liu + Hugo + Riaz (all immuno)
    #     plus TCGA patients flagged as immunotherapy-treated
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Building IMMUNOTHERAPY-ONLY merged cohort")
    print("=" * 60)

    # Filter TCGA to immunotherapy-treated patients only
    tcga_immuno_mask = clin_tcga["IMMUNOTHERAPY"] == 1
    expr_tcga_immuno_scaled = expr_tcga_scaled.loc[tcga_immuno_mask]
    clin_tcga_immuno = clin_tcga.loc[tcga_immuno_mask]
    print(f"  TCGA immunotherapy patients: {len(clin_tcga_immuno)} / {len(clin_tcga)}")

    df_expr_immuno = pd.concat(
        [expr_tcga_immuno_scaled, expr_liu_scaled, expr_hugo_scaled, expr_riaz_scaled], axis=0
    )
    df_clin_immuno = pd.concat(
        [clin_tcga_immuno, clin_liu, clin_hugo, clin_riaz], axis=0
    )

    batch_correct_and_save(df_expr_immuno, df_clin_immuno, IMMUNO_DIR, label="Immunotherapy")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Merged Cohort Generation Completed Successfully!")
    print("=" * 60)
    print(f"  Full merge:          {len(df_clin_full)} samples  -> {FULL_DIR}")
    print(f"  Immunotherapy merge: {len(df_clin_immuno)} samples -> {IMMUNO_DIR}")

if __name__ == "__main__":
    main()
