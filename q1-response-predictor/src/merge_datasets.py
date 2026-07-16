import pandas as pd
import numpy as np
import json
from pathlib import Path
from pycombat import Combat

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MERGED_DIR = PROCESSED_DIR / "merged"
MERGED_DIR.mkdir(exist_ok=True, parents=True)

def load_tcga():
    print("Loading TCGA-SKCM...")
    tcga_dir = PROCESSED_DIR / "skcm_tcga_pan_can_atlas_2018"
    df_expr_raw = pd.read_csv(tcga_dir / "expr_cleaned.csv")
    df_clin = pd.read_csv(tcga_dir / "clin_cleaned.csv")
    
    # Map Entrez to Hugo Symbols
    cache_file = tcga_dir / "entrez_to_symbol_cache.json"
    with open(cache_file, "r") as f:
        entrez_mapping = json.load(f)
        
    df_expr = df_expr_raw.set_index('SAMPLE_ID')
    df_expr = df_expr.rename(columns=entrez_mapping)
    # Average duplicates
    df_expr = df_expr.T.groupby(level=0).mean().T
    # Log-transform
    df_expr_log = np.log2(df_expr + 1)
    
    # Align sample ids
    common_ids = list(df_expr_log.index.intersection(df_clin['SAMPLE_ID']))
    df_expr_log = df_expr_log.loc[common_ids]
    df_clin = df_clin.set_index('SAMPLE_ID').loc[common_ids]
    
    # Extract clinical details
    df_clin_harm = pd.DataFrame(index=df_clin.index)
    df_clin_harm["patient_id"] = df_clin["PATIENT_ID"]
    df_clin_harm["cohort"] = "TCGA"
    df_clin_harm["os_months"] = df_clin["OS_MONTHS"]
    df_clin_harm["os_status"] = df_clin["OS_STATUS"].astype(float)
    df_clin_harm["response"] = np.nan
    df_clin_harm["age"] = df_clin["AGE"]
    
    # Sex: standardise values
    df_clin_harm["sex"] = df_clin["SEX"].map({"Male": "Male", "Female": "Female"}).fillna("N/A")
    df_clin_harm["specimen_type"] = df_clin["SAMPLE_TYPE"].map({"Primary": "Primary", "Metastatic": "Metastatic"}).fillna("N/A")
    
    return df_expr_log, df_clin_harm

def load_liu():
    print("Loading Liu 2019...")
    liu_dir = PROCESSED_DIR / "liu_2019"
    df_expr = pd.read_csv(liu_dir / "expr_cleaned.csv", index_col=0)
    df_clin = pd.read_csv(liu_dir / "clin_cleaned.csv", index_col=0)
    
    # Align IDs
    common_ids = list(set(df_expr.index) & set(df_clin.index))
    df_expr = df_expr.loc[common_ids]
    df_clin = df_clin.loc[common_ids]
    
    df_clin_harm = pd.DataFrame(index=df_clin.index)
    df_clin_harm["patient_id"] = df_clin["PATIENT_ID"]
    df_clin_harm["cohort"] = "Liu_2019"
    df_clin_harm["os_months"] = df_clin["OS_MONTHS"]
    
    # Map status '1:DECEASED' -> 1, '0:LIVING' -> 0
    df_clin_harm["os_status"] = df_clin["OS_STATUS"].map({"1:DECEASED": 1.0, "0:LIVING": 0.0})
    df_clin_harm["response"] = df_clin["response"].astype(float)
    df_clin_harm["age"] = np.nan  # Not available
    df_clin_harm["sex"] = df_clin["SEX"].map({"Male": "Male", "Female": "Female"}).fillna("N/A")
    
    # Specimen type
    spec_map = lambda s: "Primary" if "primary" in str(s).lower() else ("Metastatic" if "metast" in str(s).lower() or "lymph" in str(s).lower() or "skin" in str(s).lower() else "N/A")
    df_clin_harm["specimen_type"] = df_clin["BIOPSY_SITE_CATEG"].map(spec_map)
    
    return df_expr, df_clin_harm

def load_hugo():
    print("Loading Hugo 2016...")
    hugo_dir = PROCESSED_DIR / "hugo_2016"
    df_expr = pd.read_csv(hugo_dir / "expr_cleaned.csv", index_col=0)
    df_clin = pd.read_csv(hugo_dir / "clin_cleaned.csv", index_col=0)
    
    common_ids = list(set(df_expr.index) & set(df_clin.index))
    df_expr = df_expr.loc[common_ids]
    df_clin = df_clin.loc[common_ids]
    
    df_clin_harm = pd.DataFrame(index=df_clin.index)
    df_clin_harm["patient_id"] = df_clin["patient_id"]
    df_clin_harm["cohort"] = "Hugo_2016"
    
    # os_months from cBioPortal iAtlas clinical file
    df_clin_harm["os_months"] = df_clin["os_months"]
    df_clin_harm["os_status"] = df_clin["os_status"].astype(float)
    df_clin_harm["response"] = df_clin["response"].astype(float)
    df_clin_harm["age"] = df_clin["age (yrs)"].astype(float)
    df_clin_harm["sex"] = df_clin["gender"].map({"Male": "Male", "Female": "Female", "M": "Male", "F": "Female"}).fillna("N/A")
    df_clin_harm["specimen_type"] = "Metastatic"  # All were pre-treatment metastatic
    
    return df_expr, df_clin_harm

def load_riaz():
    print("Loading Riaz 2017...")
    riaz_dir = PROCESSED_DIR / "riaz_2017"
    df_expr = pd.read_csv(riaz_dir / "expr_cleaned.csv", index_col=0)
    df_clin = pd.read_csv(riaz_dir / "clin_cleaned.csv", index_col=0)
    
    common_ids = list(set(df_expr.index) & set(df_clin.index))
    df_expr = df_expr.loc[common_ids]
    df_clin = df_clin.loc[common_ids]
    
    df_clin_harm = pd.DataFrame(index=df_clin.index)
    df_clin_harm["patient_id"] = df_clin["patient_id"]
    df_clin_harm["cohort"] = "Riaz_2017"
    df_clin_harm["os_months"] = df_clin["os_months"]
    df_clin_harm["os_status"] = df_clin["os_status"].astype(float)
    df_clin_harm["response"] = df_clin["response"].astype(float)
    df_clin_harm["age"] = df_clin["age"].astype(float)
    df_clin_harm["sex"] = df_clin["sex"].map({"Male": "Male", "Female": "Female"}).fillna("N/A")
    df_clin_harm["specimen_type"] = "Metastatic"
    
    return df_expr, df_clin_harm

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
    
    # Subset expressions
    expr_tcga = expr_tcga[common_genes]
    expr_liu = expr_liu[common_genes]
    expr_hugo = expr_hugo[common_genes]
    expr_riaz = expr_riaz[common_genes]
    
    # 2. Merge matrices
    print("Concatenating clinical and expression tables...")
    df_expr_merged = pd.concat([expr_tcga, expr_liu, expr_hugo, expr_riaz], axis=0)
    df_clin_merged = pd.concat([clin_tcga, clin_liu, clin_hugo, clin_riaz], axis=0)
    
    # Check that indices match perfectly
    assert (df_expr_merged.index == df_clin_merged.index).all(), "Inconsistent sample indices!"
    print(f"Total merged cohort size: {len(df_clin_merged)} samples.")
    
    # Remove any genes with zero variance across the whole set to avoid division by zero in ComBat
    print("Checking for zero-variance genes...")
    zero_var_genes = df_expr_merged.columns[df_expr_merged.var(axis=0) == 0].tolist()
    if zero_var_genes:
        print(f"  Removing {len(zero_var_genes)} zero-variance genes from expression matrix.")
        df_expr_merged = df_expr_merged.drop(columns=zero_var_genes)
        common_genes = [g for g in common_genes if g not in zero_var_genes]
        
    # 3. Perform pyCombat batch correction
    print("\nRunning pyCombat batch-effect correction...")
    # Batch vector representing the cohort origin of each sample
    cohorts_list = df_clin_merged["cohort"].tolist()
    
    # Map cohorts to integer batches for pycombat
    cohort_map = {"TCGA": 0, "Liu_2019": 1, "Hugo_2016": 2, "Riaz_2017": 3}
    batches = np.array([cohort_map[c] for c in cohorts_list])
    
    # pyCombat expects samples as rows and genes as columns
    combat_obj = Combat()
    # Y is the expression values array
    Y = df_expr_merged.values
    
    # Run correction
    Y_corrected = combat_obj.fit_transform(Y, batches)
    
    # Re-construct corrected dataframe
    df_expr_corrected = pd.DataFrame(
        Y_corrected, 
        index=df_expr_merged.index, 
        columns=df_expr_merged.columns
    )
    
    # 4. Save results
    print("\nSaving merged and batch-corrected files...")
    expr_out_path = MERGED_DIR / "expr_merged.csv"
    clin_out_path = MERGED_DIR / "clin_merged.csv"
    
    df_expr_corrected.to_csv(expr_out_path)
    df_clin_merged.to_csv(clin_out_path)
    
    print(f"Saved merged expression matrix to: {expr_out_path} (shape: {df_expr_corrected.shape})")
    print(f"Saved merged clinical metadata to: {clin_out_path} (shape: {df_clin_merged.shape})")
    print("\n==================================================")
    print("Merged Cohort Generation Completed Successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
