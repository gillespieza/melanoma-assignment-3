import pandas as pd
import numpy as np
from src.config.constants import IMMUNE_SIGNATURE_GENES, IMPRES_PAIRS

# Aliases for genes that might have different names in different datasets
GENE_ALIASES = {
    "VSIR": ["VSIR", "C10orf54", "VISTA"],
    "CD274": ["CD274", "PD-L1", "PDL1"],
    "PDCD1": ["PDCD1", "PD-1", "PD1"]
}

def find_gene(df_columns, gene_name):
    """
    Finds a gene in df_columns, checking aliases if needed.
    Returns the matching column name or None.
    """
    if gene_name in df_columns:
        return gene_name
    
    # Check aliases
    aliases = GENE_ALIASES.get(gene_name, [])
    for alias in aliases:
        if alias in df_columns:
            return alias
            
    return None

def compute_ifn_gamma(df_expr):
    """
    IFN-gamma 6-gene signature (Ayers et al., 2017)
    """
    genes = IMMUNE_SIGNATURE_GENES["IFN_gamma"]
    found_genes = [find_gene(df_expr.columns, g) for g in genes]
    found_genes = [g for g in found_genes if g is not None]
    
    if len(found_genes) == 0:
        return pd.Series(np.nan, index=df_expr.index)
        
    return df_expr[found_genes].mean(axis=1)

def compute_tis(df_expr):
    """
    Tumor Inflammation Signature (TIS) 18-gene signature (Ayers et al., 2017)
    """
    genes = IMMUNE_SIGNATURE_GENES["TIS"]
    found_genes = [find_gene(df_expr.columns, g) for g in genes]
    found_genes = [g for g in found_genes if g is not None]
    
    if len(found_genes) == 0:
        return pd.Series(np.nan, index=df_expr.index)
        
    return df_expr[found_genes].mean(axis=1)

def compute_cyt(df_expr):
    """
    Cytolytic activity score (Rooney et al., 2015): mean of GZMA and PRF1
    """
    genes = IMMUNE_SIGNATURE_GENES["CYT"]
    found_genes = [find_gene(df_expr.columns, g) for g in genes]
    found_genes = [g for g in found_genes if g is not None]
    
    if len(found_genes) == 0:
        return pd.Series(np.nan, index=df_expr.index)
        
    return df_expr[found_genes].mean(axis=1)

def compute_cd8_tcell(df_expr):
    """
    CD8 T-cell signature (CD8A, CD8B)
    """
    genes = IMMUNE_SIGNATURE_GENES["CD8_Tcell"]
    found_genes = [find_gene(df_expr.columns, g) for g in genes]
    found_genes = [g for g in found_genes if g is not None]
    
    if len(found_genes) == 0:
        return pd.Series(np.nan, index=df_expr.index)
        
    return df_expr[found_genes].mean(axis=1)

def compute_impres(df_expr):
    """
    IMPRES signature (Auslander et al., 2018)
    Based on 15 pairwise relations of checkpoint genes.
    """
    scores = pd.Series(0.0, index=df_expr.index)
    valid_pairs_count = 0
    
    for gene_a, gene_b in IMPRES_PAIRS:
        col_a = find_gene(df_expr.columns, gene_a)
        col_b = find_gene(df_expr.columns, gene_b)
        
        if col_a and col_b:
            scores += (df_expr[col_a] > df_expr[col_b]).astype(int)
            valid_pairs_count += 1
            
    # Scale score if some pairs are missing, to maintain range 0-15
    if valid_pairs_count > 0:
        scores = scores * (15.0 / valid_pairs_count)
    else:
        scores = pd.Series(np.nan, index=df_expr.index)
        
    return scores

def compute_pd_l1(df_expr):
    """
    PD-L1 proxy (CD274 gene expression)
    """
    col = find_gene(df_expr.columns, "CD274")
    if col:
        return df_expr[col]
    return pd.Series(np.nan, index=df_expr.index)

def compute_m1_m2_ratio(df_expr):
    """
    M1/M2 Macrophage Polarization Ratio (NOS2, TNF, IL1B vs CD163, MSR1, MRC1, CSF1R, TGFB1)
    """
    m1_genes = [find_gene(df_expr.columns, g) for g in ["NOS2", "TNF", "IL1B", "CD68", "FCGR3A"]]
    m2_genes = [find_gene(df_expr.columns, g) for g in ["CD163", "MSR1", "MRC1", "CSF1R", "TGFB1"]]
    
    m1_found = [g for g in m1_genes if g is not None]
    m2_found = [g for g in m2_genes if g is not None]
    
    m1_score = df_expr[m1_found].mean(axis=1) if len(m1_found) > 0 else pd.Series(0.0, index=df_expr.index)
    m2_score = df_expr[m2_found].mean(axis=1) if len(m2_found) > 0 else pd.Series(0.0, index=df_expr.index)
    
    return m1_score - m2_score

def compute_macrophage_stv_score(df_expr):
    """
    Macrophage STV Spatial Barrier Score: M2 density weighted by stromal exclusion markers
    """
    m2_genes = [find_gene(df_expr.columns, g) for g in ["CD163", "MSR1", "MRC1", "CSF1R", "TGFB1"]]
    found_genes = [g for g in m2_genes if g is not None]
    
    if len(found_genes) == 0:
        return pd.Series(0.0, index=df_expr.index)
        
    return df_expr[found_genes].mean(axis=1)

def extract_all_signatures(df_expr):
    """
    Extracts all signatures for a given expression matrix.
    Returns a DataFrame: samples x signatures.
    """
    df_sig = pd.DataFrame(index=df_expr.index)
    df_sig['IFN_gamma'] = compute_ifn_gamma(df_expr)
    df_sig['TIS'] = compute_tis(df_expr)
    df_sig['CYT'] = compute_cyt(df_expr)
    df_sig['CD8_Tcell'] = compute_cd8_tcell(df_expr)
    df_sig['IMPRES'] = compute_impres(df_expr)
    df_sig['PD_L1'] = compute_pd_l1(df_expr)
    df_sig['Macrophage_STV_Score'] = compute_macrophage_stv_score(df_expr)
    df_sig['M1_M2_Ratio'] = compute_m1_m2_ratio(df_expr)
    
    # Drop rows that are completely NaN (e.g. if no genes were found)
    df_sig = df_sig.dropna(how='all')
    return df_sig

def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises numeric columns of a DataFrame to zero mean and unit variance per column.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_scaled = df.copy()
    for col in numeric_cols:
        std = df[col].std()
        if std == 0 or pd.isna(std):
            df_scaled[col] = 0.0
        else:
            df_scaled[col] = (df[col] - df[col].mean()) / std
    return df_scaled

