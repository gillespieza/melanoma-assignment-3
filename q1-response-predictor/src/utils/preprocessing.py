import pandas as pd
from typing import Tuple, Optional, Any

def standardise_sample_id(sample_id: Any) -> str:
    """
    Standardise TCGA sample barcodes to a uniform 15-character hyphenated format.
    """
    if pd.isna(sample_id) or sample_id is None:
        return ""
    s = str(sample_id).strip().replace(".", "-")
    if s.upper().startswith("TCGA-"):
        s = s.upper()
        if len(s) > 15:
            return s[:15]
    return s


def parse_survival_status(val: Any) -> Optional[float]:
    """
    Map clinical survival status to a binary 0.0 (alive) or 1.0 (deceased).
    """
    if pd.isna(val) or val is None:
        return None
    val_str = str(val).strip().upper()
    if val_str.startswith("1:") or val_str == "1":
        return 1.0
    if val_str.startswith("0:") or val_str == "0":
        return 0.0
    if any(word in val_str for word in ["DECEASED", "DEAD WITH TUMOR", "RECURRED", "PROGRESSION"]):
        return 1.0
    if any(word in val_str for word in ["LIVING", "ALIVE OR DEAD TUMOR FREE", "DISEASEFREE", "CENSORED"]):
        return 0.0
    return None


def clean_clinical_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply deduplication and survival data cleaning."""
    df = raw_df.copy()
    df["SAMPLE_ID"] = df["SAMPLE_ID"].apply(standardise_sample_id)
    df = df[df["SAMPLE_ID"] != ""]
    df = df.drop_duplicates(subset=["SAMPLE_ID"])
    if "OS_STATUS" in df.columns and "OS_MONTHS" in df.columns:
        df["OS_STATUS"] = df["OS_STATUS"].apply(parse_survival_status)
        df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
        pass
    if "PFS_STATUS" in df.columns and "PFS_MONTHS" in df.columns:
        df["PFS_STATUS"] = df["PFS_STATUS"].apply(parse_survival_status)
        df["PFS_MONTHS"] = pd.to_numeric(df["PFS_MONTHS"], errors="coerce")
    if "DSS_STATUS" in df.columns and "DSS_MONTHS" in df.columns:
        df["DSS_STATUS"] = df["DSS_STATUS"].apply(parse_survival_status)
        df["DSS_MONTHS"] = pd.to_numeric(df["DSS_MONTHS"], errors="coerce")
    
    # ---- Admin‑column filter ------------------------------------------------
    def _is_admin(col: str) -> bool:
        # Remove columns that are duplicates or cBioPortal administrative metadata
        admin_suffixes = ("_PATIENT", "_SAMPLE")
        if col.endswith(admin_suffixes):
            return True
        # Common admin columns that are not used in modeling
        unwanted = {"BIRTH_YEAR", "CANCER_TYPE_DETAILED", "PROTOCOL_SUBMIT_DATE", "LENS_ID"}
        if col in unwanted:
            return True
        # Keep identifier columns (only one copy)
        if col in {"PATIENT_ID", "SAMPLE_ID"}:
            return False
        return False

    cols_to_drop = [c for c in df.columns if _is_admin(c)]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    
    # Drop columns that are completely empty (all NaN) or have zero variance (at most 1 unique value including NaN)
    zero_var_cols = [
        c for c in df.columns 
        if df[c].nunique(dropna=False) <= 1 and c not in {"PATIENT_ID", "SAMPLE_ID"}
    ]
    df = df.drop(columns=zero_var_cols)
    return df


def clean_rnaseq_df(rnaseq_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate RNA-seq sample records and remove genes with missing values.
    """
    if rnaseq_df.empty or "SAMPLE_ID" not in rnaseq_df.columns:
        return rnaseq_df

    df = rnaseq_df.copy()
    df["SAMPLE_ID"] = df["SAMPLE_ID"].apply(standardise_sample_id)
    df = df[df["SAMPLE_ID"] != ""]
    df = df.drop_duplicates(subset=["SAMPLE_ID"])

    gene_cols = [c for c in df.columns if c != "SAMPLE_ID"]
    missing_counts = df[gene_cols].isna().sum()
    cols_with_nans = missing_counts[missing_counts > 0].index.tolist()
    if cols_with_nans:
        df = df.drop(columns=cols_with_nans)

    return df


def align_expression_and_clinical(df_expr: pd.DataFrame, df_clin: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns sample IDs between the expression matrix and clinical DataFrame.

    @param pd.DataFrame df_expr Expression DataFrame (samples as rows).
    @param pd.DataFrame df_clin Clinical DataFrame (samples as index).
    @return Tuple[pd.DataFrame, pd.DataFrame] Aligned expression and clinical DataFrames.
    """
    common_samples = df_expr.index.intersection(df_clin.index)
    df_expr = df_expr.loc[common_samples]
    df_clin = df_clin.loc[common_samples]
    df_expr.index.name = "SAMPLE_ID"
    df_clin.index.name = "SAMPLE_ID"
    return df_expr, df_clin


def map_entrez_to_symbols(entrez_ids, cache_path=None):
    """
    Maps Entrez IDs to Hugo Symbols using MyGene.info API.
    Uses local cache if available to prevent redundant API hits.
    """
    import json
    import urllib.request
    from pathlib import Path
    
    if cache_path and Path(cache_path).exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
            
    print("Mapping Entrez IDs to Hugo Symbols via MyGene.info...")
    entrez_mapping = {}
    chunk_size = 1000
    entrez_ids = [str(eid) for eid in entrez_ids]
    
    for i in range(0, len(entrez_ids), chunk_size):
        chunk = entrez_ids[i:i+chunk_size]
        url = 'https://mygene.info/v3/query'
        q_str = ','.join(chunk)
        data = f'q={q_str}&scopes=entrezgene&fields=symbol&species=human'.encode('utf-8')
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                res = json.loads(response.read().decode('utf-8'))
                for item in res:
                    q = item.get('query')
                    sym = item.get('symbol')
                    if q and sym:
                        entrez_mapping[q] = sym
        except Exception as e:
            print(f"  [WARNING] Error mapping Entrez chunk {i}: {e}")
            
    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump(entrez_mapping, f, indent=4)
            
    return entrez_mapping

