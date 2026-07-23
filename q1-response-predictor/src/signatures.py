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
    
    # Drop rows that are completely NaN (e.g. if no genes were found)
    df_sig = df_sig.dropna(how='all')
    return df_sig
