import pandas as pd
from pathlib import Path

def _add_legacy_aliases(df_clin: pd.DataFrame, mutations_file: Path) -> pd.DataFrame:
    # Expose lowercase aliases for compatibility with legacy downstream scripts
    for up, low in [('PATIENT_ID', 'patient_id'), ('RESPONSE_BINARY', 'response'), ('SEX', 'sex'), ('AGE', 'age'), ('OS_STATUS', 'os_status'), ('OS_MONTHS', 'os_months')]:
        if up in df_clin.columns:
            df_clin[low] = df_clin[up]
            
    # Load and join mutations if the file exists
    if mutations_file.exists():
        df_mut = pd.read_csv(mutations_file, index_col="SAMPLE_ID")
        for up, low in [('BRAF', 'mut_BRAF'), ('NRAS', 'mut_NRAS'), ('NF1', 'mut_NF1')]:
            if up in df_mut.columns:
                df_mut[low] = df_mut[up]
            else:
                df_mut[low] = 0
        df_clin = df_clin.join(df_mut, how="left")
        
    return df_clin

def load_liu_2019(data_dir):
    """
    Loads pre-processed Liu et al. 2019 dataset.

    @param Path data_dir The base data directory containing processed files.
    @return tuple (df_expr, df_clin) of expression and clinical DataFrames.
    """
    data_dir = Path(data_dir)
    expr_file = data_dir / "processed/liu_2019/expr_cleaned.csv"
    clin_file = data_dir / "processed/liu_2019/clin_cleaned.csv"
    mut_file = data_dir / "processed/liu_2019/mutations_cleaned.csv"
    
    df_expr = pd.read_csv(expr_file, index_col="SAMPLE_ID")
    df_clin = pd.read_csv(clin_file, index_col="SAMPLE_ID")
    df_clin = _add_legacy_aliases(df_clin, mut_file)
    
    return df_expr, df_clin


def load_hugo_2016(data_dir):
    """
    Loads pre-processed Hugo et al. 2016 dataset.

    @param Path data_dir The base data directory containing processed files.
    @return tuple (df_expr, df_clin) of expression and clinical DataFrames.
    """
    data_dir = Path(data_dir)
    expr_file = data_dir / "processed/hugo_2016/expr_cleaned.csv"
    clin_file = data_dir / "processed/hugo_2016/clin_cleaned.csv"
    mut_file = data_dir / "processed/hugo_2016/mutations_cleaned.csv"
    
    df_expr = pd.read_csv(expr_file, index_col="SAMPLE_ID")
    df_clin = pd.read_csv(clin_file, index_col="SAMPLE_ID")
    df_clin = _add_legacy_aliases(df_clin, mut_file)
    
    return df_expr, df_clin


def load_riaz_2017(data_dir):
    """
    Loads pre-processed Riaz et al. 2017 dataset.

    @param Path data_dir The base data directory containing processed files.
    @return tuple (df_expr, df_clin) of expression and clinical DataFrames.
    """
    data_dir = Path(data_dir)
    expr_file = data_dir / "processed/riaz_2017/expr_cleaned.csv"
    clin_file = data_dir / "processed/riaz_2017/clin_cleaned.csv"
    mut_file = data_dir / "processed/riaz_2017/mutations_cleaned.csv"
    
    df_expr = pd.read_csv(expr_file, index_col="SAMPLE_ID")
    df_clin = pd.read_csv(clin_file, index_col="SAMPLE_ID")
    df_clin = _add_legacy_aliases(df_clin, mut_file)
    
    return df_expr, df_clin
