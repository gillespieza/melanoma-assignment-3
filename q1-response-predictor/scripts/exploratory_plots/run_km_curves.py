import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from src.styles import COHORT_PALETTE, RESPONSE_PALETTE

# Add project root to sys.path for importing src modules
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Paths
DATA_DIR = BASE_DIR / "data"
CLINICAL_FILE = DATA_DIR / "processed/skcm_tcga_pan_can_atlas_2018/clin_cleaned.csv"
PLOTS_DIR = BASE_DIR / "plots"

def main():
    print("==================================================")
    print("Generating Kaplan-Meier Survival Curves for TCGA-SKCM")
    print("==================================================")

    if not CLINICAL_FILE.exists():
        print(f"Error: Clinical file not found at {CLINICAL_FILE}")
        return

    # Load data
    df = pd.read_csv(CLINICAL_FILE)
    print(f"Loaded clinical data with shape: {df.shape}")

    # Set up plots directory
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)

    # Clean survival data
    df = df.dropna(subset=['OS_MONTHS', 'OS_STATUS'])
    print(f"Number of samples with valid survival data: {len(df)}")

    # Set style
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'figure.titlesize': 18,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })

    # Color palettes
    palette_2 = [COHORT_PALETTE['Liu 2019'], COHORT_PALETTE['Hugo 2016']]
    palette_multi = [RESPONSE_PALETTE['PD'], COHORT_PALETTE['Riaz 2017'], '#7f7f7f', '#17becf', COHORT_PALETTE['Hugo 2016']]

    # Helper function to plot KM
    def plot_km(df, group_col, title, filename, palette, labels=None, split_median=False):
        fig, ax = plt.subplots(figsize=(9, 6.5))
        
        # Prepare groups
        if split_median:
            median_val = df[group_col].median()
            df['temp_group'] = df[group_col].apply(lambda x: f"High (>= {median_val:.2f})" if x >= median_val else f"Low (< {median_val:.2f})")
            group_col = 'temp_group'

        groups = df[group_col].unique()
        groups = [g for g in groups if pd.notna(g) and str(g).lower() != 'nan' and str(g).lower() != 'unknown']
        groups = sorted(groups)

        kmf = KaplanMeierFitter()
        
        # Fit and plot each group
        for i, g in enumerate(groups):
            mask = df[group_col] == g
            # Use label if provided
            label = labels[g] if labels and g in labels else str(g)
            label = f"{label} (N={mask.sum()})"
            
            kmf.fit(df.loc[mask, 'OS_MONTHS'], df.loc[mask, 'OS_STATUS'], label=label)
            kmf.plot_survival_function(ax=ax, color=palette[i % len(palette)], ci_show=False, linewidth=2.5)

        # Log-rank test
        if len(groups) == 2:
            g1_mask = df[group_col] == groups[0]
            g2_mask = df[group_col] == groups[1]
            results = logrank_test(
                df.loc[g1_mask, 'OS_MONTHS'], df.loc[g2_mask, 'OS_MONTHS'],
                df.loc[g1_mask, 'OS_STATUS'], df.loc[g2_mask, 'OS_STATUS']
            )
            p_val = results.p_value
            p_text = f"Log-Rank p = {p_val:.2e}" if p_val < 0.001 else f"Log-Rank p = {p_val:.3f}"
        else:
            # Multivariate log-rank test
            results = multivariate_logrank_test(
                df['OS_MONTHS'], df[group_col], df['OS_STATUS']
            )
            p_val = results.p_value
            p_text = f"Log-Rank p = {p_val:.2e}" if p_val < 0.001 else f"Log-Rank p = {p_val:.3f}"

        # Add p-value to plot
        ax.text(0.05, 0.08, p_text, transform=ax.transAxes, fontsize=13, weight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

        ax.set_title(title, fontsize=16, weight='bold', pad=15)
        ax.set_xlabel("Overall Survival (Months)", fontsize=13, labelpad=10)
        ax.set_ylabel("Survival Probability", fontsize=13, labelpad=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        
        # Save plots
        plot_path = PLOTS_DIR / filename
        
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"  Saved plot: {filename} (p = {p_val:.2e})")

    # ==========================================
    # Plot 1: SAMPLE_TYPE (Primary vs Metastasis)
    # ==========================================
    labels_sample = {
        'Primary': 'Primary Specimen',
        'Metastasis': 'Metastatic Specimen'
    }
    df_sample = df[df['SAMPLE_TYPE'].isin(['Primary', 'Metastasis'])].copy()
    plot_km(df_sample, 'SAMPLE_TYPE', 'TCGA-SKCM OS: Primary vs. Metastatic Specimen', 
            'km_sample_type.png', palette_2, labels=labels_sample)

    # ==========================================
    # Plot 2: PATH_T_STAGE_T4B (T4b vs Other)
    # ==========================================
    df['T4B_STATUS'] = df['PATH_T_STAGE'].apply(lambda x: 'T4b' if str(x).upper() == 'T4B' else 'Other Stage')
    labels_t4b = {
        'T4b': 'T4b Staging (Deep/Ulcerated)',
        'Other Stage': 'Other T-Stage'
    }
    plot_km(df, 'T4B_STATUS', 'TCGA-SKCM OS: Staging T4b vs. Other T-Stages', 
            'km_t4b_stage.png', palette_2, labels=labels_t4b)

    # ==========================================
    # Plot 3: WINTER_HYPOXIA_SCORE (High vs Low)
    # ==========================================
    plot_km(df, 'WINTER_HYPOXIA_SCORE', 'TCGA-SKCM OS: Winter Hypoxia Score Stratification', 
            'km_winter_hypoxia.png', palette_2, split_median=True)

    # ==========================================
    # Plot 4: TX_TYPE_CHEMOTHERAPY (Chemotherapy vs No Chemotherapy)
    # ==========================================
    labels_chemo = {
        0: 'No Chemotherapy',
        1: 'Received Chemotherapy'
    }
    plot_km(df, 'TX_TYPE_CHEMOTHERAPY', 'TCGA-SKCM OS: Traditional Chemotherapy Status', 
            'km_chemotherapy.png', palette_2, labels=labels_chemo)

    # ==========================================
    # Plot 5: Therapy Groups (Chemo vs Immuno vs Targeted vs None)
    # ==========================================
    def map_therapy_groups(row):
        if row['TX_TYPE_TARGETED_MOLECULAR_THERAPY'] == 1:
            return 'Targeted Therapy'
        elif row['TX_TYPE_IMMUNOTHERAPY'] == 1:
            return 'Immunotherapy'
        elif row['TX_TYPE_CHEMOTHERAPY'] == 1:
            return 'Chemotherapy'
        elif row['TX_TYPE_RADIATION_THERAPY'] == 1:
            return 'Radiation Therapy'
        else:
            return 'None (Observation/Surgery Only)'
            
    df['THERAPY_GROUP'] = df.apply(map_therapy_groups, axis=1)
    
    plot_km(df, 'THERAPY_GROUP', 'TCGA-SKCM OS: Systemic Therapy Classes Comparison', 
            'km_therapy_comparison.png', palette_multi)

    print("Done! All Kaplan-Meier curves generated and saved to plots/ and artifacts directory.")

if __name__ == "__main__":
    main()
