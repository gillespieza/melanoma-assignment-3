import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Set matplotlib backend to Agg to avoid GUI errors
import matplotlib
matplotlib.use('Agg')

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

def clean_os_status(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        if val in [1, 1.0]:
            return 1
        if val in [0, 0.0]:
            return 0
    if isinstance(val, str):
        val_upper = val.upper()
        if 'DECEASED' in val_upper or '1' in val_upper:
            return 1
        if 'LIVING' in val_upper or '0' in val_upper:
            return 0
    return np.nan

def main():
    print("==================================================")
    print("TCGA-SKCM TMB Prognostic Survival Analysis...")
    print("==================================================")
    
    # Load TCGA clinical data
    # Load from the local data directory
    tcga_dir = DATA_DIR / "processed" / "skcm_tcga_pan_can_atlas_2018"
    df_tcga_clin = pd.read_csv(tcga_dir / "clinical_cleaned.csv", index_col=0)
    
    # Check if TMB and survival columns exist
    tmb_col = 'TMB_NONSYNONYMOUS'
    if tmb_col not in df_tcga_clin.columns:
        # Fallback to MUTATION_COUNT
        if 'MUTATION_COUNT' in df_tcga_clin.columns:
            tmb_col = 'MUTATION_COUNT'
            print("TMB_NONSYNONYMOUS not found, falling back to MUTATION_COUNT")
        else:
            print("Error: No TMB or mutation count columns found in clinical data.")
            return
            
    df = df_tcga_clin.copy()
    
    # Clean columns
    df['os_status_clean'] = df['OS_STATUS'].apply(clean_os_status)
    df['OS_MONTHS'] = pd.to_numeric(df['OS_MONTHS'], errors='coerce')
    df['tmb_clean'] = pd.to_numeric(df[tmb_col], errors='coerce')
    
    df = df.dropna(subset=['OS_MONTHS', 'os_status_clean', 'tmb_clean'])
    df['OS_MONTHS'] = df['OS_MONTHS'].astype(float)
    df['os_status_clean'] = df['os_status_clean'].astype(float)
    df['tmb_clean'] = df['tmb_clean'].astype(float)
    
    print(f"Cleaned TCGA-SKCM samples with TMB and survival data: {len(df)}")
    
    if len(df) == 0:
        print("No samples available for survival analysis.")
        return
        
    # Perform median split on TMB
    tmb_median = df['tmb_clean'].median()
    print(f"Median TMB value: {tmb_median:.2f} mutations/Mb (or counts)")
    
    high_tmb = df[df['tmb_clean'] >= tmb_median]
    low_tmb = df[df['tmb_clean'] < tmb_median]
    
    # Fit KM curves
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    
    plt.figure(figsize=(8, 6))
    
    kmf_high.fit(high_tmb['OS_MONTHS'], event_observed=high_tmb['os_status_clean'], label=f'High TMB (>= {tmb_median:.1f}, N={len(high_tmb)})')
    kmf_high.plot_survival_function(ci_show=True)
    
    kmf_low.fit(low_tmb['OS_MONTHS'], event_observed=low_tmb['os_status_clean'], label=f'Low TMB (< {tmb_median:.1f}, N={len(low_tmb)})')
    kmf_low.plot_survival_function(ci_show=True)
    
    # Calculate log-rank test
    results = logrank_test(
        high_tmb['OS_MONTHS'], low_tmb['OS_MONTHS'], 
        event_observed_A=high_tmb['os_status_clean'], event_observed_B=low_tmb['os_status_clean']
    )
    p_value = results.p_value
    
    plt.title(f'TCGA-SKCM Baseline Overall Survival by TMB (p = {p_value:.2e})')
    plt.xlabel('Survival Time (Months)')
    plt.ylabel('Survival Probability')
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    
    save_path = PLOT_DIR / "survival_tcga_tmb.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"TCGA-SKCM Overall Survival difference by TMB p-value: {p_value:.3e}")
    print(f"Saved TMB survival plot to {save_path}")
    
    # Copy to artifacts directory
    dest_path = Path("C:/Users/Amanda/.gemini/antigravity/brain/4bb83474-71a7-4da3-beda-f3ee3b7eba05/survival_tcga_tmb.png")
    import shutil
    shutil.copy(save_path, dest_path)
    print("Copied survival_tcga_tmb.png to artifacts directory.")
    print("==================================================")

if __name__ == "__main__":
    main()
