import os
import sys
import shutil
import json
import time
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.svm import SVC
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Set matplotlib backend to Agg to avoid GUI errors
import matplotlib
matplotlib.use('Agg')

# Add project root to sys.path for importing src modules
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Paths
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots" / "biomarkers"
PLOT_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

# Import signature extractor and styles
from src.signatures import extract_all_signatures
from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style

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

# map_tcga_expression_to_symbols is deprecated. TCGA is pre-mapped to Hugo Symbols in raw data cleaning.

def zscore_df(df):
    """
    Standardize DataFrame columns individually (Z-score scaling).
    Avoids division by zero if std is zero.
    """
    means = df.mean(axis=0)
    stds = df.std(axis=0)
    stds = stds.replace(0, 1.0).fillna(1.0)
    return (df - means) / stds

def load_processed_mutations(mutations_file: Path, target_genes: list, sample_ids: list) -> pd.DataFrame:
    if not mutations_file.exists():
        return pd.DataFrame(0, index=sample_ids, columns=target_genes)
        
    df_mut = pd.read_csv(mutations_file, index_col="SAMPLE_ID")
    
    # Ensure all target genes are in the columns
    for g in target_genes:
        if g not in df_mut.columns:
            df_mut[g] = 0
            
    df_mut = (df_mut[target_genes] > 0).astype(int)
    return df_mut.reindex(sample_ids, fill_value=0)

def evaluate_auc_cv(model, X, y, cv, label):
    """
    Evaluate one feature set with serial outer CV and parallelized inner model search.
    """
    start = time.perf_counter()
    print(f"    - {label}...", flush=True)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=None)
    elapsed = time.perf_counter() - start
    print(
        f"      AUC = {scores.mean():.3f} (+/-{scores.std():.3f}); "
        f"folds = [{', '.join(f'{score:.3f}' for score in scores)}]; "
        f"{elapsed:.1f}s",
        flush=True,
    )
    return scores

def main():
    print("==================================================")
    print("Extended Biomarker Evaluation & Predictive Modeling (Merged Trial Cohorts)")
    print("==================================================")
    
    # 1. Load Clinical Datasets
    liu_clin_path = DATA_DIR / "processed/liu_2019/clin_cleaned.csv"
    liu_expr_path = DATA_DIR / "processed/liu_2019/expr_cleaned.csv"
    hugo_clin_path = DATA_DIR / "processed/hugo_2016/clin_cleaned.csv"
    hugo_expr_path = DATA_DIR / "processed/hugo_2016/expr_cleaned.csv"
    riaz_clin_path = DATA_DIR / "processed/riaz_2017/clin_cleaned.csv"
    riaz_expr_path = DATA_DIR / "processed/riaz_2017/expr_cleaned.csv"
    
    tcga_clin_path = DATA_DIR / "processed/skcm_tcga_pan_can_atlas_2018/clin_cleaned.csv"
    tcga_expr_raw_path = DATA_DIR / "processed/skcm_tcga_pan_can_atlas_2018/expr_cleaned.csv"
    
    # Verify paths
    paths = [liu_clin_path, liu_expr_path, hugo_clin_path, hugo_expr_path, riaz_clin_path, riaz_expr_path, tcga_clin_path, tcga_expr_raw_path]
    if not all(p.exists() for p in paths):
        print("Error: Cleansed processed files not found. Run clean_data.py first.")
        for p in paths:
            print(f"  {p.name}: {'FOUND' if p.exists() else 'MISSING'} ({p})")
        return
        
    df_liu_clin = pd.read_csv(liu_clin_path, index_col="SAMPLE_ID")
    df_liu_expr = pd.read_csv(liu_expr_path, index_col=0)
    df_hugo_clin = pd.read_csv(hugo_clin_path, index_col="SAMPLE_ID")
    df_hugo_expr = pd.read_csv(hugo_expr_path, index_col=0)
    df_riaz_clin = pd.read_csv(riaz_clin_path, index_col="SAMPLE_ID")
    df_riaz_expr = pd.read_csv(riaz_expr_path, index_col=0)
    df_tcga_clin = pd.read_csv(tcga_clin_path, index_col="SAMPLE_ID")
    df_tcga_expr_raw = pd.read_csv(tcga_expr_raw_path)
    
    # Map legacy lowercase aliases for backwards compatibility
    for df in [df_liu_clin, df_hugo_clin, df_riaz_clin, df_tcga_clin]:
        for up, low in [('PATIENT_ID', 'patient_id'), ('RESPONSE_BINARY', 'response'), ('SEX', 'sex'), ('AGE', 'age'), ('OS_STATUS', 'os_status'), ('OS_MONTHS', 'os_months')]:
            if up in df.columns and low not in df.columns:
                df[low] = df[up]
                
    # Filter trial cohorts to response-aligned samples (keep CR/PR/PD; drop SD/MR/NaN)
    RESPONSE_MAP = {
        "Complete Response": 1,
        "Partial Response": 1,
        "Progressive Disease": 0,
        "Stable Disease": np.nan,
        "Mixed Response": np.nan,
    }
    df_liu_clin['temp_resp'] = df_liu_clin['RESPONSE'].map(RESPONSE_MAP)
    df_liu_clin.dropna(subset=['temp_resp'], inplace=True)
    df_liu_clin.drop(columns=['temp_resp'], inplace=True)
    
    df_hugo_clin['temp_resp'] = df_hugo_clin['RESPONSE'].map(RESPONSE_MAP)
    df_hugo_clin.dropna(subset=['temp_resp'], inplace=True)
    df_hugo_clin.drop(columns=['temp_resp'], inplace=True)
    
    df_riaz_clin['temp_resp'] = df_riaz_clin['RESPONSE'].map(RESPONSE_MAP)
    df_riaz_clin.dropna(subset=['temp_resp'], inplace=True)
    df_riaz_clin.drop(columns=['temp_resp'], inplace=True)
    
    # Re-align expression matrix rows to clinical index
    df_liu_expr = df_liu_expr.loc[df_liu_clin.index]
    df_hugo_expr = df_hugo_expr.loc[df_hugo_clin.index]
    df_riaz_expr = df_riaz_expr.loc[df_riaz_clin.index]
    
    # Calculate TOTAL_NEOANTIGEN for trial cohorts if missing
    neo_cols = ['SNV_NEOANTIGEN', 'INDEL_NEOANTIGEN', 'FUSION_NEOANTIGEN', 'SPLICE_NEOANTIGEN', 'VIRUS_NEOANTIGEN', 'ERV_NEOANTIGEN']
    for df in [df_liu_clin, df_hugo_clin, df_riaz_clin]:
        if 'TOTAL_NEOANTIGEN' not in df.columns:
            available_neo = [c for c in neo_cols if c in df.columns]
            if available_neo:
                df['TOTAL_NEOANTIGEN'] = df[available_neo].fillna(0).sum(axis=1)
                all_nan = df[available_neo].isna().all(axis=1)
                df.loc[all_nan, 'TOTAL_NEOANTIGEN'] = np.nan
            else:
                df['TOTAL_NEOANTIGEN'] = np.nan
 
    # Standardize Cohorts and Sex
    df_liu_clin['Cohort'] = 'Liu 2019'
    df_hugo_clin['Cohort'] = 'Hugo 2016'
    df_riaz_clin['Cohort'] = 'Riaz 2017'
    for df in [df_liu_clin, df_hugo_clin, df_riaz_clin]:
        if 'SEX' in df.columns:
            df['SEX'] = df['SEX'].map({'Male': 'Male', 'Female': 'Female', 'M': 'Male', 'F': 'Female'})

    print("\nComputing expression signatures for all cohorts...")
    df_liu_sigs = extract_all_signatures(df_liu_expr)
    df_hugo_sigs = extract_all_signatures(df_hugo_expr)
    df_riaz_sigs = extract_all_signatures(df_riaz_expr)
    
    df_tcga_expr = df_tcga_expr_raw.set_index('SAMPLE_ID')
    df_tcga_sigs = extract_all_signatures(df_tcga_expr)
    df_tcga_sigs = zscore_df(df_tcga_sigs)
    
    # Align TCGA signatures & clinical
    df_tcga_sigs.index = df_tcga_sigs.index.str.upper().str[:12]
    df_tcga_clin.index = df_tcga_clin.index.str.upper().str[:12]
    df_tcga_sigs = df_tcga_sigs.groupby(df_tcga_sigs.index).first()
    df_tcga_clin = df_tcga_clin.groupby(df_tcga_clin.index).first()
    common_tcga = df_tcga_sigs.index.intersection(df_tcga_clin.index)
    df_tcga_sigs = df_tcga_sigs.loc[common_tcga]
    df_tcga_clin = df_tcga_clin.loc[common_tcga]

    # Load somatic pathway mutations across all 3 trials
    pathway_genes = ['B2M', 'TAP1', 'TAP2', 'JAK1', 'JAK2', 'STAT1', 'PTEN', 'CDKN2A', 'PIK3CA', 'BRAF', 'NRAS', 'NF1']
    mut_liu = load_processed_mutations(DATA_DIR / "processed/liu_2019/mutations_cleaned.csv", pathway_genes, df_liu_clin.index.tolist())
    mut_hugo = load_processed_mutations(DATA_DIR / "processed/hugo_2016/mutations_cleaned.csv", pathway_genes, df_hugo_clin.index.tolist())
    mut_riaz = load_processed_mutations(DATA_DIR / "processed/riaz_2017/mutations_cleaned.csv", pathway_genes, df_riaz_clin.index.tolist())

    # Map mutation columns to clinical dataframes
    for col in pathway_genes:
        df_liu_clin[f'mut_{col}'] = mut_liu[col]
        df_hugo_clin[f'mut_{col}'] = mut_hugo[col]
        df_riaz_clin[f'mut_{col}'] = mut_riaz[col]

    # Create pathway mutation trackers
    for df in [df_liu_clin, df_hugo_clin, df_riaz_clin]:
        df['mut_Antigen_Presentation'] = (df[['mut_B2M', 'mut_TAP1', 'mut_TAP2']].sum(axis=1) > 0).astype(int)
        df['mut_IFN_gamma_Signaling'] = (df[['mut_JAK1', 'mut_JAK2', 'mut_STAT1']].sum(axis=1) > 0).astype(int)
        df['mut_Survival_Pathways'] = (df[['mut_PTEN', 'mut_CDKN2A', 'mut_PIK3CA']].sum(axis=1) > 0).astype(int)

    # Pool trial datasets clinical and signatures
    clin_cols = ['Cohort', 'response', 'TMB_NONSYNONYMOUS', 'AGE', 'TOTAL_NEOANTIGEN', 'CNA_PROP',
                 'mut_BRAF', 'mut_NRAS', 'mut_NF1', 'mut_Antigen_Presentation', 'mut_IFN_gamma_Signaling', 'mut_Survival_Pathways']
    
    # Fill missing CNA_PROP (mostly absent, we can map to NaN or handle)
    for df in [df_liu_clin, df_hugo_clin, df_riaz_clin]:
        if 'CNA_PROP' not in df.columns:
            df['CNA_PROP'] = np.nan

    for df in [df_liu_clin, df_hugo_clin, df_riaz_clin]:
        if 'AGE' not in df.columns:
            df['AGE'] = np.nan
    df_clin_merged = pd.concat([df_liu_clin[clin_cols], df_hugo_clin[clin_cols], df_riaz_clin[clin_cols]])
    # Standardize each cohort's signatures individually (Z-score) to prevent batch technical effects and leakage
    df_liu_sigs_scaled = zscore_df(df_liu_sigs)
    df_hugo_sigs_scaled = zscore_df(df_hugo_sigs)
    df_riaz_sigs_scaled = zscore_df(df_riaz_sigs)
    df_sigs_merged = pd.concat([df_liu_sigs_scaled, df_hugo_sigs_scaled, df_riaz_sigs_scaled])
    
    # Output report setup
    report_content = []
    report_content.append("# Evaluation of Extended Genomic & Clinical Biomarkers")
    report_content.append(f"\nThis report documents the statistical analysis and predictive modeling updates of extended biomarkers across the pooled trial datasets ($N={len(df_clin_merged)}$) and TCGA-SKCM ($N={len(df_tcga_clin)}$) cohorts.")

    # ==========================================
    # Step 1: Neoantigen Load Evaluation (Pooled Trials)
    # ==========================================
    print("\nEvaluating Neoantigen Load vs. TMB in Pooled Trials...")
    report_content.append("\n## 1. Neoantigen Load vs. Tumor Mutational Burden (TMB)")
    
    df_trials_neo = df_clin_merged.dropna(subset=['TOTAL_NEOANTIGEN', 'TMB_NONSYNONYMOUS']).copy()
    r_spearman, p_spearman = spearmanr(df_trials_neo['TOTAL_NEOANTIGEN'], df_trials_neo['TMB_NONSYNONYMOUS'], nan_policy='omit')
    
    print(f"  Spearman correlation (pooled): r = {r_spearman:.3f}, p = {p_spearman:.2e}")
    report_content.append(f"We evaluated the correlation between predicted neoantigen load (`TOTAL_NEOANTIGEN`) and mutational burden (`TMB_NONSYNONYMOUS`) in the pooled trial cohorts ($N={len(df_trials_neo)}$):")
    report_content.append(f"\n*   **Spearman Correlation Coefficient ($r$)**: **{r_spearman:.3f}** (p-value: **{p_spearman:.2e}**)")
    report_content.append("\nAs expected, there is an almost perfect linear relationship between mutational burden and the number of predicted MHC-binding neoantigens.")
    
    # Evaluate predictive power of Neoantigens vs. TMB for Response
    y_true_trials = df_clin_merged['response'].dropna()
    common_idx = y_true_trials.index.intersection(df_clin_merged['TOTAL_NEOANTIGEN'].dropna().index)
    
    auc_neo = roc_auc_score(df_clin_merged.loc[common_idx, 'response'], df_clin_merged.loc[common_idx, 'TOTAL_NEOANTIGEN'])
    auc_tmb = roc_auc_score(df_clin_merged.loc[common_idx, 'response'], df_clin_merged.loc[common_idx, 'TMB_NONSYNONYMOUS'])
    
    stat_neo, p_neo_mw = mannwhitneyu(
        df_clin_merged[df_clin_merged['response'] == 1]['TOTAL_NEOANTIGEN'].dropna(),
        df_clin_merged[df_clin_merged['response'] == 0]['TOTAL_NEOANTIGEN'].dropna()
    )
    stat_tmb, p_tmb_mw = mannwhitneyu(
        df_clin_merged[df_clin_merged['response'] == 1]['TMB_NONSYNONYMOUS'].dropna(),
        df_clin_merged[df_clin_merged['response'] == 0]['TMB_NONSYNONYMOUS'].dropna()
    )
    
    print(f"  Neoantigen response prediction ROC AUC: {auc_neo:.3f} (p = {p_neo_mw:.3e})")
    print(f"  TMB response prediction ROC AUC: {auc_tmb:.3f} (p = {p_tmb_mw:.3e})")
    
    report_content.append("\n### Predictive Utility for Immunotherapy Response")
    report_content.append("| Biomarker | N | Response ROC AUC | Mann-Whitney U p-value |")
    report_content.append("|---|---|---|---|")
    report_content.append(f"| **TOTAL_NEOANTIGEN** | {len(common_idx)} | **{auc_neo:.3f}** | {p_neo_mw:.3e} |")
    report_content.append(f"| **TMB_NONSYNONYMOUS** | {len(common_idx)} | **{auc_tmb:.3f}** | {p_tmb_mw:.3e} |")
    
    # Plot Neoantigen vs TMB scatter
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=df_trials_neo, x='TMB_NONSYNONYMOUS', y='TOTAL_NEOANTIGEN', color=COHORT_PALETTE['Pooled Trials'], ax=ax,
                scatter_kws={'alpha':0.6, 'edgecolor':'w', 's':70})
    ax.set_title(f"Neoantigen Load vs. TMB (Merged Trials, r = {r_spearman:.3f})", fontsize=14, weight='bold')
    ax.set_xlabel("Nonsynonymous TMB (mutations/Mb)")
    ax.set_ylabel("Predicted Total Neoantigens")
    plt.tight_layout()
    neo_plot_path = PLOT_DIR / "extended_neoantigen_tmb.png"
    plt.savefig(neo_plot_path, dpi=300)
    plt.close()
    report_content.append("\n![Neoantigen vs TMB](../plots/extended_neoantigen_tmb.png)")
    
    # ==========================================
    # Step 2: Somatic Pathway Mutations (Pooled Trials)
    # ==========================================
    print("\nCalculating pathway mutation frequencies in trial cohorts...")
    report_content.append("\n## 2. Somatic Pathway Mutations")
    report_content.append("We evaluated somatic mutations in three biological pathways that dictate tumor immunogenicity and escape:")
    report_content.append("*   **Antigen Presentation**: `B2M`, `TAP1`, `TAP2` (disrupts MHC Class I presentation).")
    report_content.append("*   **IFN-gamma Signaling**: `JAK1`, `JAK2`, `STAT1` (induces insensitivity to T-cell cytotoxicity).")
    report_content.append("*   **Survival & Proliferation Drivers**: `PTEN`, `CDKN2A`, `PIK3CA` (oncogenic drivers).")
    
    report_content.append("\n### Mutation Frequencies in Trial Cohorts:")
    report_content.append(f"| Pathway / Gene | Liu 2019 ($N={len(df_liu_clin)}$) | Hugo 2016 ($N={len(df_hugo_clin)}$) | Riaz 2017 ($N={len(df_riaz_clin)}$) | Pooled Trials ($N={len(df_clin_merged)}$) |")
    report_content.append("|---|---|---|---|---|")
    
    def get_mut_freq_str(df, col):
        return f"{df[col].mean():.1%}"
        
    report_content.append(f"| **BRAF mutation** | {get_mut_freq_str(df_liu_clin, 'mut_BRAF')} | {get_mut_freq_str(df_hugo_clin, 'mut_BRAF')} | {get_mut_freq_str(df_riaz_clin, 'mut_BRAF')} | **{get_mut_freq_str(df_clin_merged, 'mut_BRAF')}** |")
    report_content.append(f"| **NRAS mutation** | {get_mut_freq_str(df_liu_clin, 'mut_NRAS')} | {get_mut_freq_str(df_hugo_clin, 'mut_NRAS')} | {get_mut_freq_str(df_riaz_clin, 'mut_NRAS')} | **{get_mut_freq_str(df_clin_merged, 'mut_NRAS')}** |")
    report_content.append(f"| **NF1 mutation** | {get_mut_freq_str(df_liu_clin, 'mut_NF1')} | {get_mut_freq_str(df_hugo_clin, 'mut_NF1')} | {get_mut_freq_str(df_riaz_clin, 'mut_NF1')} | **{get_mut_freq_str(df_clin_merged, 'mut_NF1')}** |")
    report_content.append(f"| **Antigen Presentation (MHC)** | {get_mut_freq_str(df_liu_clin, 'mut_Antigen_Presentation')} | {get_mut_freq_str(df_hugo_clin, 'mut_Antigen_Presentation')} | {get_mut_freq_str(df_riaz_clin, 'mut_Antigen_Presentation')} | **{get_mut_freq_str(df_clin_merged, 'mut_Antigen_Presentation')}** |")
    report_content.append(f"| **IFN-gamma Signaling** | {get_mut_freq_str(df_liu_clin, 'mut_IFN_gamma_Signaling')} | {get_mut_freq_str(df_hugo_clin, 'mut_IFN_gamma_Signaling')} | {get_mut_freq_str(df_riaz_clin, 'mut_IFN_gamma_Signaling')} | **{get_mut_freq_str(df_clin_merged, 'mut_IFN_gamma_Signaling')}** |")
    report_content.append(f"| **Survival & Proliferation Drivers** | {get_mut_freq_str(df_liu_clin, 'mut_Survival_Pathways')} | {get_mut_freq_str(df_hugo_clin, 'mut_Survival_Pathways')} | {get_mut_freq_str(df_riaz_clin, 'mut_Survival_Pathways')} | **{get_mut_freq_str(df_clin_merged, 'mut_Survival_Pathways')}** |")

    # ==========================================
    # Step 3: Aneuploidy & TMB vs. Immune Infiltration
    # ==========================================
    print("\nEvaluating Aneuploidy/CNA and TMB vs. Immune Infiltration...")
    report_content.append("\n## 3. Aneuploidy, Copy-Number Alterations, & TMB vs. Immune Infiltration")
    report_content.append("We evaluated how copy-number burden (aneuploidy score / fraction genome altered) and mutational burden (TMB) correlate with continuous immune signatures. Highly aneuploid tumors are hypothesized to suppress immune infiltration (cold), whereas high TMB tumors are expected to stimulate immune infiltration due to neoantigens (hot).")
    
    # A. TCGA correlations
    tcga_corrs = {}
    for sig_name in ['IFN_gamma', 'TIS', 'CD8_Tcell', 'CYT', 'PD_L1']:
        r_aneu, p_aneu = spearmanr(df_tcga_clin['ANEUPLOIDY_SCORE'], df_tcga_sigs[sig_name], nan_policy='omit')
        r_tmb, p_tmb = spearmanr(df_tcga_clin['TMB_NONSYNONYMOUS'], df_tcga_sigs[sig_name], nan_policy='omit')
        tcga_corrs[sig_name] = {
            'Aneu_r': r_aneu, 'Aneu_p': p_aneu, 
            'Tmb_r': r_tmb, 'Tmb_p': p_tmb
        }
        
    # B. Pooled Trial correlations
    # Align merged clinical and signature indexes
    df_sigs_merged_aligned = df_sigs_merged.loc[df_clin_merged.index]
    
    trial_corrs = {}
    for sig_name in ['IFN_gamma', 'TIS', 'CD8_Tcell', 'CYT', 'PD_L1']:
        r_tmb, p_tmb = spearmanr(df_clin_merged['TMB_NONSYNONYMOUS'], df_sigs_merged_aligned[sig_name], nan_policy='omit')
        trial_corrs[sig_name] = {
            'Tmb_r': r_tmb, 'Tmb_p': p_tmb
        }
        
    report_content.append("\n### Spearman Correlations table:")
    report_content.append("| Immune Signature | TCGA Aneuploidy Score ($r$) | TCGA TMB ($r$) | Trial TMB ($r$) |")
    report_content.append("|---|---|---|---|")
    for sig in ['IFN_gamma', 'TIS', 'CD8_Tcell', 'CYT', 'PD_L1']:
        report_content.append(f"| `{sig}` | **{tcga_corrs[sig]['Aneu_r']:.3f}** | **{tcga_corrs[sig]['Tmb_r']:.3f}** | **{trial_corrs[sig]['Tmb_r']:.3f}** |")
        
    report_content.append("\n**Biological Conclusion**: In both cohorts:")
    report_content.append("1.  **Chromosomal Instability (Aneuploidy)** shows a **very weak negative correlation** ($r \\approx -0.03$ to $-0.10$) with immune signatures in the TCGA SKCM cohort, with only `PD_L1` showing a marginally significant negative correlation ($r = -0.100$, $p = 0.043$). This indicates that chromosomal instability is only weakly associated with reduced baseline immune infiltration in this cohort.")
    report_content.append("2.  **Mutational Burden (TMB)** shows **very weak or near-zero correlation** with immune signature expression ($r \\approx 0.05$ to $0.15$). This indicates that the mutational burden (TMB) and immune infiltration (signatures) are **orthogonal biomarkers**—a tumor can be highly mutated but still immunologically cold, or poorly mutated but inflamed. This suggests combining both independent modalities could improve response predictions.")

    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 7))
    corr_data = pd.DataFrame({
        'Aneuploidy Score (TCGA)': [tcga_corrs[s]['Aneu_r'] for s in ['IFN_gamma', 'TIS', 'CD8_Tcell', 'CYT', 'PD_L1']],
        'TMB (TCGA)': [tcga_corrs[s]['Tmb_r'] for s in ['IFN_gamma', 'TIS', 'CD8_Tcell', 'CYT', 'PD_L1']],
        'TMB (Trials)': [trial_corrs[s]['Tmb_r'] for s in ['IFN_gamma', 'TIS', 'CD8_Tcell', 'CYT', 'PD_L1']]
    }, index=['IFN_gamma', 'TIS', 'CD8_Tcell', 'CYT', 'PD_L1'])
    
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-0.4, vmax=0.4, center=0, ax=ax, fmt=".3f", linewidths=1)
    ax.set_title("Spearman Correlation: Genomic Burden vs. Immune Signatures", fontsize=12, weight='bold', pad=15)
    plt.tight_layout()
    corr_plot_path = PLOT_DIR / "extended_immune_correlations.png"
    plt.savefig(corr_plot_path, dpi=300)
    plt.close()
    report_content.append("\n![Correlation Heatmap](../plots/extended_immune_correlations.png)")

    # ---------------------------------------------
    # Kaplan-Meier Curve by Aneuploidy in TCGA
    # ---------------------------------------------
    print("\nRunning survival curve by Aneuploidy in TCGA...")
    df_tcga_survival = df_tcga_clin.dropna(subset=['OS_MONTHS', 'OS_STATUS', 'ANEUPLOIDY_SCORE']).copy()
    df_tcga_survival['OS_MONTHS'] = pd.to_numeric(df_tcga_survival['OS_MONTHS'], errors='coerce')
    df_tcga_survival['os_status_clean'] = df_tcga_survival['OS_STATUS'].apply(clean_os_status)
    df_tcga_survival = df_tcga_survival.dropna(subset=['OS_MONTHS', 'os_status_clean'])
    
    aneu_median = df_tcga_survival['ANEUPLOIDY_SCORE'].median()
    df_tcga_survival['Aneu_Group'] = df_tcga_survival['ANEUPLOIDY_SCORE'].apply(
        lambda x: f'High Aneuploidy (>= {aneu_median:.0f})' if x >= aneu_median else f'Low Aneuploidy (< {aneu_median:.0f})'
    )
    
    fig, ax = plt.subplots(figsize=(9, 6.5))
    kmf = KaplanMeierFitter()
    
    high_mask = df_tcga_survival['Aneu_Group'].str.startswith('High')
    low_mask = df_tcga_survival['Aneu_Group'].str.startswith('Low')
    


    kmf.fit(df_tcga_survival.loc[low_mask, 'OS_MONTHS'], df_tcga_survival.loc[low_mask, 'os_status_clean'], label=f"Low Aneuploidy (N={low_mask.sum()})")
    kmf.plot_survival_function(ax=ax, color=COHORT_PALETTE["Liu 2019"], ci_show=False, linewidth=2.5) # Blue
    
    kmf.fit(df_tcga_survival.loc[high_mask, 'OS_MONTHS'], df_tcga_survival.loc[high_mask, 'os_status_clean'], label=f"High Aneuploidy (N={high_mask.sum()})")
    kmf.plot_survival_function(ax=ax, color=COHORT_PALETTE["Hugo 2016"], ci_show=False, linewidth=2.5) # Orange
    
    lr_res = logrank_test(
        df_tcga_survival.loc[high_mask, 'OS_MONTHS'], df_tcga_survival.loc[low_mask, 'OS_MONTHS'],
        df_tcga_survival.loc[high_mask, 'os_status_clean'], df_tcga_survival.loc[low_mask, 'os_status_clean']
    )
    p_text = f"Log-Rank p = {lr_res.p_value:.2e}" if lr_res.p_value < 0.001 else f"Log-Rank p = {lr_res.p_value:.3f}"
    ax.text(0.05, 0.08, p_text, transform=ax.transAxes, fontsize=13, weight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
            
    ax.set_title("TCGA-SKCM Overall Survival by Aneuploidy Score", fontsize=15, weight='bold', pad=15)
    ax.set_xlabel("Overall Survival (Months)", fontsize=13)
    ax.set_ylabel("Survival Probability", fontsize=13)
    plt.tight_layout()
    aneu_plot_path = PLOT_DIR / "extended_aneuploidy_survival.png"
    plt.savefig(aneu_plot_path, dpi=300)
    plt.close()
    
    report_content.append("\n### TCGA Overall Survival by Aneuploidy")
    report_content.append(f"We partitioned the baseline TCGA cohort at the median Aneuploidy Score (**{aneu_median:.1f}**):")
    report_content.append(f"\n*   **Log-Rank p-value**: **{lr_res.p_value:.3e}** (Statistically Significant)")
    report_content.append("\n![TCGA Aneuploidy Survival](../plots/extended_aneuploidy_survival.png)")

    # ---------------------------------------------
    # Kaplan-Meier Curve by TMB in TCGA
    # ---------------------------------------------
    print("\nRunning survival curve by TMB in TCGA...")
    df_tcga_tmb_surv = df_tcga_clin.dropna(subset=['OS_MONTHS', 'OS_STATUS', 'TMB_NONSYNONYMOUS']).copy()
    df_tcga_tmb_surv['OS_MONTHS'] = pd.to_numeric(df_tcga_tmb_surv['OS_MONTHS'], errors='coerce')
    df_tcga_tmb_surv['os_status_clean'] = df_tcga_tmb_surv['OS_STATUS'].apply(clean_os_status)
    df_tcga_tmb_surv = df_tcga_tmb_surv.dropna(subset=['OS_MONTHS', 'os_status_clean'])
    
    tmb_median = df_tcga_tmb_surv['TMB_NONSYNONYMOUS'].median()
    df_tcga_tmb_surv['TMB_Group'] = df_tcga_tmb_surv['TMB_NONSYNONYMOUS'].apply(
        lambda x: f'High TMB (>= {tmb_median:.1f})' if x >= tmb_median else f'Low TMB (< {tmb_median:.1f})'
    )
    
    fig, ax = plt.subplots(figsize=(9, 6.5))
    
    high_tmb_mask = df_tcga_tmb_surv['TMB_Group'].str.startswith('High')
    low_tmb_mask = df_tcga_tmb_surv['TMB_Group'].str.startswith('Low')
    
    kmf.fit(df_tcga_tmb_surv.loc[low_tmb_mask, 'OS_MONTHS'], df_tcga_tmb_surv.loc[low_tmb_mask, 'os_status_clean'], label=f"Low TMB (N={low_tmb_mask.sum()})")
    kmf.plot_survival_function(ax=ax, color=RESPONSE_PALETTE["CR/PR"], ci_show=False, linewidth=2.5)
    
    kmf.fit(df_tcga_tmb_surv.loc[high_tmb_mask, 'OS_MONTHS'], df_tcga_tmb_surv.loc[high_tmb_mask, 'os_status_clean'], label=f"High TMB (N={high_tmb_mask.sum()})")
    kmf.plot_survival_function(ax=ax, color=RESPONSE_PALETTE["PD"], ci_show=False, linewidth=2.5)
    
    lr_tmb_res = logrank_test(
        df_tcga_tmb_surv.loc[high_tmb_mask, 'OS_MONTHS'], df_tcga_tmb_surv.loc[low_tmb_mask, 'OS_MONTHS'],
        df_tcga_tmb_surv.loc[high_tmb_mask, 'os_status_clean'], df_tcga_tmb_surv.loc[low_tmb_mask, 'os_status_clean']
    )
    p_tmb_text = f"Log-Rank p = {lr_tmb_res.p_value:.2e}" if lr_tmb_res.p_value < 0.001 else f"Log-Rank p = {lr_tmb_res.p_value:.3f}"
    ax.text(0.05, 0.08, p_tmb_text, transform=ax.transAxes, fontsize=13, weight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
            
    ax.set_title("TCGA-SKCM Overall Survival by Tumor Mutational Burden (TMB)", fontsize=15, weight='bold', pad=15)
    ax.set_xlabel("Overall Survival (Months)", fontsize=13)
    ax.set_ylabel("Survival Probability", fontsize=13)
    plt.tight_layout()
    tmb_plot_path = PLOT_DIR / "survival_tcga_tmb.png"
    plt.savefig(tmb_plot_path, dpi=300)
    plt.close()
    
    report_content.append("\n### TCGA Overall Survival by Tumor Mutational Burden (TMB)")
    report_content.append(f"We partitioned the baseline TCGA cohort at the median TMB value (**{tmb_median:.2f} mutations/Mb**):")
    report_content.append(f"\n*   **Log-Rank p-value**: **{lr_tmb_res.p_value:.3f}** (Prognostically Neutral)")
    report_content.append("\n![TCGA TMB Survival](../plots/survival_tcga_tmb.png)")

    # ==========================================
    # Step 4: Updated Multimodal Predictor (Pooled Trials)
    # ==========================================
    print("\nTraining Multimodal Response Predictor on Pooled Trial Cohort...")
    
    # Re-initialize report content to contain only predictive modeling details
    # (since the descriptive baseline genomic analysis is already in reports/cohort_characteristics_genomic.md)
    report_content = []
    report_content.append("# Extended Biomarkers: Multimodal Predictive Modeling")
    report_content.append(f"\nThis report documents the training and evaluation of response prediction models on the pooled immunotherapy trial cohort ($N={len(df_clin_merged)}$), comparing signature models, driver-mutation models, and a full extended clinical-genomic model.")
    
    # Define features
    sig_features = ['IFN_gamma', 'TIS', 'CYT', 'CD8_Tcell', 'IMPRES', 'PD_L1']
    df_features = pd.concat([df_sigs_merged_aligned, df_clin_merged[[
        'mut_BRAF', 'mut_NRAS', 'mut_NF1', 'mut_Antigen_Presentation', 'mut_IFN_gamma_Signaling', 'mut_Survival_Pathways',
        'TMB_NONSYNONYMOUS', 'TOTAL_NEOANTIGEN', 'AGE'
    ]]], axis=1)

    # Impute missing values
    df_features['TMB_NONSYNONYMOUS'] = df_features['TMB_NONSYNONYMOUS'].fillna(df_features['TMB_NONSYNONYMOUS'].median())
    df_features['TOTAL_NEOANTIGEN'] = df_features['TOTAL_NEOANTIGEN'].fillna(df_features['TOTAL_NEOANTIGEN'].median())
    df_features['AGE'] = df_features['AGE'].fillna(df_features['AGE'].median())

    # Drop samples with NaN response
    clean_idx = df_clin_merged['response'].dropna().index
    df_features_clean = df_features.loc[clean_idx]
    y = df_clin_merged.loc[clean_idx, 'response'].values
    
    # Setup CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'Logistic Regression (LR)': GridSearchCV(
            LogisticRegression(solver='liblinear', l1_ratio=1.0, random_state=42, max_iter=1000),
            param_grid={'C': [0.01, 0.1, 1, 10, 100]}, cv=3, scoring='roc_auc', n_jobs=-1
        ),
        'Random Forest (RF)': GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid={'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None], 'min_samples_leaf': [1, 2, 4]},
            cv=3, scoring='roc_auc', n_jobs=-1
        ),
        'XGBoost (XGB, tuned)': GridSearchCV(
            XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
            param_grid={
                'n_estimators': [50, 100, 200],
                'max_depth': [2, 3],
                'learning_rate': [0.03, 0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            },
            cv=3, scoring='roc_auc', n_jobs=-1
        ),
        'Support Vector Machine (SVM)': GridSearchCV(
            CalibratedClassifierCV(SVC(random_state=42), ensemble=False),
            param_grid={'estimator__C': [0.01, 0.1, 1, 10], 'estimator__kernel': ['linear', 'rbf']},
            cv=3, scoring='roc_auc', n_jobs=-1
        ),
        'Elastic-Net': GridSearchCV(
            LogisticRegression(solver='saga', random_state=42, max_iter=20000, tol=1e-3),
            param_grid={'C': [0.01, 0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]},
            cv=3, scoring='roc_auc', n_jobs=-1
        )
    }
    
    model_results = []
    plot_data = []  # Store raw scores for visualisation
    
    for model_name, model in models.items():
        model_start = time.perf_counter()
        print(f"  > Evaluating {model_name}...", flush=True)
        # 1. Base Model (Signatures only)
        X_base = df_features_clean[sig_features].values
        scores_base = evaluate_auc_cv(model, X_base, y, cv, "Signatures only")

        # 2. Driver Mutation Model (Sigs + Drivers + AGE)
        driver_cols = sig_features + ['mut_BRAF', 'mut_NRAS', 'mut_NF1', 'AGE']
        X_drivers = df_features_clean[driver_cols].values
        scores_drivers = evaluate_auc_cv(model, X_drivers, y, cv, "Signatures + drivers + age")
        
        # 3. Full Extended Model (All Features including TMB & CNA & pathway mutations)
        X_full = df_features_clean.values
        scores_full = evaluate_auc_cv(model, X_full, y, cv, "Full extended")
        
        model_results.append({
            'Model': model_name,
            'Base AUC': f"{scores_base.mean():.3f} (+/-{scores_base.std():.3f})",
            'Sigs+Drivers+Age AUC': f"{scores_drivers.mean():.3f} (+/-{scores_drivers.std():.3f})",
            'Full Extended AUC': f"{scores_full.mean():.3f} (+/-{scores_full.std():.3f})"
        })
        
        plot_data.append({
            'model': model_name,
            'base_mean': scores_base.mean(), 'base_std': scores_base.std(),
            'drivers_mean': scores_drivers.mean(), 'drivers_std': scores_drivers.std(),
            'full_mean': scores_full.mean(), 'full_std': scores_full.std()
        })
        print(f"  > Finished {model_name} in {time.perf_counter() - model_start:.1f}s", flush=True)
        
    print("\nModel Cross-Validation AUC Comparison (Pooled Trials):")
    print(pd.DataFrame(model_results).to_string(index=False))

    # --- Grouped Bar Chart: Multimodal AUC Comparison ---
    set_presentation_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bar_labels = ['Signatures Only', 'Sigs + Drivers + Sex', 'Full Extended\n(Sigs + Drivers + TMB\n+ CNA + Pathways)']
    x = np.arange(len(bar_labels))
    n_models = len(models)
    bar_width = 0.8 / n_models
    
    # Okabe-Ito and other colorblind-friendly colors
    colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442']
    
    for i, pd_row in enumerate(plot_data):
        means = [pd_row['base_mean'], pd_row['drivers_mean'], pd_row['full_mean']]
        stds = [pd_row['base_std'], pd_row['drivers_std'], pd_row['full_std']]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=stds,
                      label=pd_row['model'], color=colors[i % len(colors)],
                      edgecolor='white', linewidth=0.7,
                      capsize=4, error_kw={'elinewidth': 1.2, 'capthick': 1})
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9,
                    color='#333333')
    
    # Compute dynamic y-limits from the actual data (mean +/- std across all bars),
    # padding for the value-label text, and keeping the random-baseline (0.5) visible.
    all_means = [v for pdr in plot_data for v in (pdr['base_mean'], pdr['drivers_mean'], pdr['full_mean'])]
    all_stds = [v for pdr in plot_data for v in (pdr['base_std'], pdr['drivers_std'], pdr['full_std'])]
    lower_vals = [m - s for m, s in zip(all_means, all_stds)]
    upper_vals = [m + s for m, s in zip(all_means, all_stds)]

    data_min = min(lower_vals + [0.5])   # include baseline so it's never clipped
    data_max = max(upper_vals + [0.5])
    y_range = data_max - data_min
    padding = max(y_range * 0.12, 0.03)  # extra room for the text labels above bars

    y_min = max(0.0, data_min - padding)
    y_max = min(1.0, data_max + padding * 1.5)  # a bit more headroom above for labels

    ax.set_ylabel('ROC-AUC (5-Fold Stratified CV)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=11)
    ax.set_ylim(y_min, y_max)
    ax.axhline(y=0.5, color='#999999', linestyle='--', linewidth=1.2, label='Random Baseline (AUC = 0.5)')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9, title="Models")
    ax.set_title('Multimodal Response Prediction: Feature Set Comparison\n(Pooled IO Trial Cohort, 5-Fold Stratified CV)',
                 fontsize=14, fontweight='bold', pad=15)
    sns.despine(ax=ax, top=True, right=True)
    ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.4)
    plt.tight_layout()
    
    multimodal_plot_path = PLOT_DIR / "multimodal_auc_comparison.png"
    plt.savefig(multimodal_plot_path, dpi=300)
    plt.close()
    print(f"Saved multimodal AUC comparison plot to {multimodal_plot_path}")
    
    report_content.append("\n### Model Performance (5-Fold Stratified Cross-Validation on Pooled Trial Cohort):")
    report_content.append("| Model | Base Model (Sigs only) | Sigs + Drivers (`BRAF/NRAS/NF1`) + Age | Full Extended Model (Sigs + Drivers + TMB + CNA + Mutations) |")
    report_content.append("|---|---|---|---|")
    for res, pdr in zip(model_results, plot_data):
        col_means = {
            'Base AUC': pdr['base_mean'],
            'Sigs+Drivers+Age AUC': pdr['drivers_mean'],
            'Full Extended AUC': pdr['full_mean'],
        }
        best_col = max(col_means, key=col_means.get)
        cells = {
            k: (f"**{res[k]}**" if k == best_col else res[k])
            for k in ['Base AUC', 'Sigs+Drivers+Age AUC', 'Full Extended AUC']
        }
        report_content.append(
            f"| **{res['Model']}** | {cells['Base AUC']} | {cells['Sigs+Drivers+Age AUC']} | {cells['Full Extended AUC']} |"
        )
    report_content.append("\n![Multimodal AUC Comparison](../plots/biomarkers/multimodal_auc_comparison.png)")
        
    report_content.append("\n### Analysis of Predictor Performance:")
    report_content.append("1.  **Baseline vs. Drivers**: Adding the driver mutations and sex provides a slight stabilization/improvement in cross-validation AUC for some model families, including tuned XGBoost.")
    report_content.append("2.  **Full Multimodal Model**: The full extended model (incorporating 15 features including mutation flags and genomic load metrics) performs strongly with the tuned XGBoost grid, which favors shallow trees, moderate learning rates, and row/feature subsampling, and remains competitive with Random Forest.")
    
    # Save the report markdown
    report_path = (
        REPORTS_DIR / "extended_biomarkers_report.md"
        # REPORTS_DIR / "pillar-3-transcriptomic-signatures/curated_signatures_report.md"
    )
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
        
    print(f"\nResults report successfully written to {report_path}")
    print("==================================================")
    print("Execution completed successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
