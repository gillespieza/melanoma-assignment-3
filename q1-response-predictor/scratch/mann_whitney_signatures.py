import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_liu_2019, load_hugo_2016, load_riaz_2017
from src.signatures import extract_all_signatures

# Paths
DATA_DIR = BASE_DIR / "data"

def evaluate_cohort(name, df_expr, df_clin):
    # Align response and signatures
    df_sig = extract_all_signatures(df_expr)
    
    # Locate the binary response column
    resp_col = None
    for col in ['response', 'RESPONSE_BINARY', 'RESPONSE']:
        if col in df_clin.columns:
            # Check if it has 1 and 0 values
            vals = df_clin[col].dropna().unique()
            if 1 in vals or 0 in vals:
                resp_col = col
                break
                
    if resp_col is None:
        print(f"Warning: No binary response column found for {name}. Columns: {list(df_clin.columns)}")
        return None
        
    df_clin_clean = df_clin.loc[df_sig.index].dropna(subset=[resp_col])
    df_sig_clean = df_sig.loc[df_clin_clean.index]
    
    responders = df_clin_clean[resp_col] == 1
    non_responders = df_clin_clean[resp_col] == 0
    
    n_resp = responders.sum()
    n_non = non_responders.sum()
    
    results = []
    for col in df_sig_clean.columns:
        resp_vals = df_sig_clean.loc[responders, col]
        non_vals = df_sig_clean.loc[non_responders, col]
        
        # Calculate mean for both
        mean_resp = resp_vals.mean()
        mean_non = non_vals.mean()
        
        # Calculate Mann-Whitney U
        stat, p_val = mannwhitneyu(resp_vals, non_vals, alternative='two-sided')
        
        results.append({
            'Cohort': name,
            'Signature': col,
            'N Responders': n_resp,
            'N Non-Responders': n_non,
            'Mean Responders': mean_resp,
            'Mean Non-Responders': mean_non,
            'U-statistic': stat,
            'p-value': p_val,
            'Significant (p < 0.05)': 'Yes' if p_val < 0.05 else 'No'
        })
    return pd.DataFrame(results)

def main():
    print("==================================================")
    # Load individual processed datasets using the data loaders
    liu_expr, liu_clin = load_liu_2019(DATA_DIR)
    hugo_expr, hugo_clin = load_hugo_2016(DATA_DIR)
    riaz_expr, riaz_clin = load_riaz_2017(DATA_DIR)
    
    # Load merged trial datasets
    expr_merged = pd.read_csv(DATA_DIR / "processed/merged/immunotherapy/expr_merged.csv", index_col=0)
    clin_merged = pd.read_csv(DATA_DIR / "processed/merged/immunotherapy/clin_merged.csv", index_col="SAMPLE_ID")
    
    # Evaluate
    df_liu = evaluate_cohort('Liu 2019', liu_expr, liu_clin)
    df_hugo = evaluate_cohort('Hugo 2016', hugo_expr, hugo_clin)
    df_riaz = evaluate_cohort('Riaz 2017', riaz_expr, riaz_clin)
    df_merged = evaluate_cohort('Merged Trials', expr_merged, clin_merged)
    
    dfs = [df for df in [df_liu, df_hugo, df_riaz, df_merged] if df is not None]
    all_results = pd.concat(dfs, ignore_index=True)
    
    # Display results
    print("\n--- MANN-WHITNEY U TEST RESULTS BY SIGNATURE ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(all_results.to_string(index=False))
    
    # Save to CSV in scratch directory
    all_results.to_csv("scratch/mann_whitney_results.csv", index=False)
    print("\nResults saved to scratch/mann_whitney_results.csv")

if __name__ == "__main__":
    main()
