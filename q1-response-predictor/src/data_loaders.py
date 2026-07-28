import pandas as pd
from pathlib import Path

def _add_aliases(df):
    mapping = {
        'RESPONSE_BINARY': 'response',
        'OS_MONTHS': 'os_months',
        'OS_STATUS': 'os_status',
        'AGE': 'age',
        'SEX': 'sex'
    }
    for orig, target in mapping.items():
        if orig in df.columns and target not in df.columns:
            df[target] = df[orig]
    return df

def load_liu_2019(data_dir):
    data_dir = Path(data_dir)
    expr_file = data_dir / "processed/liu_2019/expr_cleaned.csv"
    clin_file = data_dir / "processed/liu_2019/clin_cleaned.csv"
    
    df_expr = pd.read_csv(expr_file, index_col="SAMPLE_ID")
    df_clin = pd.read_csv(clin_file, index_col="SAMPLE_ID")
    df_clin = _add_aliases(df_clin)
    return df_expr, df_clin


def load_hugo_2016(data_dir):
    data_dir = Path(data_dir)
    expr_file = data_dir / "processed/hugo_2016/expr_cleaned.csv"
    clin_file = data_dir / "processed/hugo_2016/clin_cleaned.csv"
    
    df_expr = pd.read_csv(expr_file, index_col="SAMPLE_ID")
    df_clin = pd.read_csv(clin_file, index_col="SAMPLE_ID")
    df_clin = _add_aliases(df_clin)
    return df_expr, df_clin


def load_riaz_2017(data_dir):
    data_dir = Path(data_dir)
    expr_file = data_dir / "processed/riaz_2017/expr_cleaned.csv"
    clin_file = data_dir / "processed/riaz_2017/clin_cleaned.csv"
    
    df_expr = pd.read_csv(expr_file, index_col="SAMPLE_ID")
    df_clin = pd.read_csv(clin_file, index_col="SAMPLE_ID")
    df_clin = _add_aliases(df_clin)

    return df_expr, df_clin
