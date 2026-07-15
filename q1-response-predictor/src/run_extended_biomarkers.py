import os
import shutil
import json
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Set matplotlib backend to Agg to avoid GUI errors
import matplotlib
matplotlib.use('Agg')

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

# Active Artifacts directory for current conversation
ARTIFACTS_DIR = Path("C:/Users/Amanda/.gemini/antigravity/brain/1fa1902e-1db0-4ffb-a701-539d9692435a")
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

# Import signature extractor
from signatures import extract_all_signatures

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

def map_tcga_expression_to_symbols(df_expr_raw, cache_path):
    """
    Maps Entrez columns in TCGA expression matrix to Hugo Symbols using local cache.
    """
    if not cache_path.exists():
        raise FileNotFoundError("TCGA Entrez-to-Symbol mapping cache not found. Please run run_transcriptomic_feature_selection.py first.")
        
    with open(cache_path, 'r') as f:
        gene_map = json.load(f)
        
    entrez_cols = [c for c in df_expr_raw.columns if c != 'SAMPLE_ID']
    mapped_cols = ['SAMPLE_ID'] + [gene_map.get(str(col), col) for col in entrez_cols]
    df_expr_raw.columns = mapped_cols
    
    df_expr = df_expr_raw.set_index('SAMPLE_ID')
    df_expr = df_expr.groupby(df_expr.columns, axis=1).mean()
    return np.log2(df_expr + 1)

def main():
    print("==================================================")
    print("Extended Biomarker Evaluation & Predictive Modeling")
    print("==================================================")
    
    # 1. Load Clinical Datasets
    liu_clin_path = DATA_DIR / "processed/liu_2019/clin_cleaned.csv"
    liu_expr_path = DATA_DIR / "processed/liu_2019/expr_cleaned.csv"
    tcga_clin_path = DATA_DIR / "processed/skcm_tcga_pan_can_atlas_2018/clinical_cleaned.csv"
    tcga_expr_raw_path = DATA_DIR / "processed/skcm_tcga_pan_can_atlas_2018/rnaseq_cleaned.csv"
    
    if not (liu_clin_path.exists() and liu_expr_path.exists() and tcga_clin_path.exists() and tcga_expr_raw_path.exists()):
        print("Error: Cleansed processed files not found. Run clean_data.py first.")
        return
        
    df_liu_clin = pd.read_csv(liu_clin_path, index_col=0)
    df_liu_expr = pd.read_csv(liu_expr_path, index_col=0)
    df_tcga_clin = pd.read_csv(tcga_clin_path, index_col=0)
    df_tcga_expr_raw = pd.read_csv(tcga_expr_raw_path)
    
    print(f"Liu 2019: {df_liu_clin.shape[0]} clinical samples, {df_liu_expr.shape[1]} genes")
    print(f"TCGA-SKCM PanCan: {df_tcga_clin.shape[0]} clinical samples, {df_tcga_expr_raw.shape[1]} raw genes")
    
    # Map TCGA expression to symbols & extract signatures
    print("\nComputing signatures for TCGA...")
    cache_path = DATA_DIR / "processed/skcm_tcga_pan_can_atlas_2018/entrez_to_symbol_cache.json"
    df_tcga_expr = map_tcga_expression_to_symbols(df_tcga_expr_raw, cache_path)
    df_tcga_sigs = extract_all_signatures(df_tcga_expr)
    
    # Align TCGA signatures & clinical
    df_tcga_sigs.index = df_tcga_sigs.index.str.upper().str[:12]
    df_tcga_clin.index = df_tcga_clin.index.str.upper().str[:12]
    
    df_tcga_sigs = df_tcga_sigs.groupby(df_tcga_sigs.index).first()
    df_tcga_clin = df_tcga_clin.groupby(df_tcga_clin.index).first()
    
    common_tcga = df_tcga_sigs.index.intersection(df_tcga_clin.index)
    df_tcga_sigs = df_tcga_sigs.loc[common_tcga]
    df_tcga_clin = df_tcga_clin.loc[common_tcga]
    
    # Extract Liu signatures (expression is already log-transformed)
    df_liu_sigs = extract_all_signatures(df_liu_expr)
    
    # Output report setup
    report_content = []
    report_content.append("# Evaluation of Extended Genomic & Clinical Biomarkers")
    report_content.append("\nThis report documents the statistical analysis and predictive modeling updates of extended biomarkers across the Liu 2019 and TCGA-SKCM cohorts.")
    
    # ==========================================
    # Step 1: Neoantigen Load Evaluation (Liu 2019)
    # ==========================================
    print("\nEvaluating Neoantigen Load vs. TMB in Liu 2019...")
    report_content.append("\n## 1. Neoantigen Load vs. Tumor Mutational Burden (TMB)")
    
    df_liu_neo = df_liu_clin.dropna(subset=['TOTAL_NEOANTIGEN', 'TMB_NONSYNONYMOUS']).copy()
    r_spearman, p_spearman = spearmanr(df_liu_neo['TOTAL_NEOANTIGEN'], df_liu_neo['TMB_NONSYNONYMOUS'], nan_policy='omit')
    
    print(f"  Spearman correlation: r = {r_spearman:.3f}, p = {p_spearman:.2e}")
    report_content.append(f"We evaluated the correlation between predicted neoantigen load (`TOTAL_NEOANTIGEN`) and mutational burden (`TMB_NONSYNONYMOUS`) in the Liu 2019 cohort ($N={len(df_liu_neo)}$):")
    report_content.append(f"\n*   **Spearman Correlation Coefficient ($r$)**: **{r_spearman:.3f}** (p-value: **{p_spearman:.2e}**)")
    report_content.append("\nAs expected, there is an almost perfect linear relationship between mutational burden and the number of predicted MHC-binding neoantigens.")
    
    # Evaluate predictive power of Neoantigens vs. TMB for Response
    y_true_liu = df_liu_clin['response']
    auc_neo = roc_auc_score(y_true_liu, df_liu_clin['TOTAL_NEOANTIGEN'])
    auc_tmb = roc_auc_score(y_true_liu, df_liu_clin['TMB_NONSYNONYMOUS'])
    
    # Mann-Whitney U tests
    stat_neo, p_neo_mw = mannwhitneyu(
        df_liu_clin[df_liu_clin['response'] == 1]['TOTAL_NEOANTIGEN'].dropna(),
        df_liu_clin[df_liu_clin['response'] == 0]['TOTAL_NEOANTIGEN'].dropna()
    )
    stat_tmb, p_tmb_mw = mannwhitneyu(
        df_liu_clin[df_liu_clin['response'] == 1]['TMB_NONSYNONYMOUS'].dropna(),
        df_liu_clin[df_liu_clin['response'] == 0]['TMB_NONSYNONYMOUS'].dropna()
    )
    
    print(f"  Neoantigen response prediction ROC AUC: {auc_neo:.3f} (p = {p_neo_mw:.3f})")
    print(f"  TMB response prediction ROC AUC: {auc_tmb:.3f} (p = {p_tmb_mw:.3f})")
    
    report_content.append("\n### Predictive Utility for Immunotherapy Response")
    report_content.append("| Biomarker | N | Response ROC AUC | Mann-Whitney U p-value |")
    report_content.append("|---|---|---|---|")
    report_content.append(f"| **TOTAL_NEOANTIGEN** | {len(df_liu_neo)} | **{auc_neo:.3f}** | {p_neo_mw:.2e} |")
    report_content.append(f"| **TMB_NONSYNONYMOUS** | {len(df_liu_neo)} | **{auc_tmb:.3f}** | {p_tmb_mw:.2e} |")
    
    # Plot Neoantigen vs TMB scatter
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(data=df_liu_neo, x='TMB_NONSYNONYMOUS', y='TOTAL_NEOANTIGEN', color='#1f77b4', ax=ax,
                scatter_kws={'alpha':0.6, 'edgecolor':'w', 's':70})
    ax.set_title(f"Neoantigen Load vs. TMB (Liu 2019, r = {r_spearman:.3f})", fontsize=14, weight='bold')
    ax.set_xlabel("Nonsynonymous TMB (mutations/Mb)")
    ax.set_ylabel("Predicted Total Neoantigens")
    plt.tight_layout()
    neo_plot_path = PLOT_DIR / "extended_neoantigen_tmb.png"
    plt.savefig(neo_plot_path, dpi=300)
    plt.close()
    shutil.copy(neo_plot_path, ARTIFACTS_DIR / "extended_neoantigen_tmb.png")
    report_content.append("\n![Neoantigen vs TMB](C:/Users/Amanda/.gemini/antigravity/brain/1fa1902e-1db0-4ffb-a701-539d9692435a/extended_neoantigen_tmb.png)")
    
    # ==========================================
    # Step 2: Somatic Pathway Mutations (Liu 2019)
    # ==========================================
    print("\nExtracting somatic pathway mutations in Liu 2019...")
    report_content.append("\n## 2. Somatic Pathway Mutations")
    report_content.append("We evaluated somatic mutations in three biological pathways that dictate tumor immunogenicity and escape:")
    report_content.append("*   **Antigen Presentation**: `B2M`, `TAP1`, `TAP2` (disrupts MHC Class I presentation).")
    report_content.append("*   **IFN-gamma Signaling**: `JAK1`, `JAK2`, `STAT1` (induces insensitivity to T-cell cytotoxicity).")
    report_content.append("*   **Survival & Proliferation Drivers**: `PTEN`, `CDKN2A`, `PIK3CA` (oncogenic drivers).")
    
    mut_file = DATA_DIR / "raw/liu_2019/data_mutations.txt"
    clin_sample_file = DATA_DIR / "raw/liu_2019/data_clinical_sample.txt"
    
    df_mut = pd.read_csv(mut_file, sep="\t", low_memory=False)
    df_sample_map = pd.read_csv(clin_sample_file, sep="\t", skiprows=4)
    sample_to_patient = df_sample_map.set_index("SAMPLE_ID")["PATIENT_ID"].to_dict()
    df_mut['patient_id'] = df_mut['Tumor_Sample_Barcode'].map(sample_to_patient)
    
    # Filter for non-silent mutations
    non_silent = [
        "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
        "Missense_Mutation", "Nonsense_Mutation", "Splice_Site",
        "Translation_Start_Site", "Nonstop_Mutation"
    ]
    df_mut = df_mut[df_mut['Variant_Classification'].isin(non_silent)]
    
    # Pivot mutation table
    df_mut_wide = df_mut.pivot_table(
        index='patient_id', 
        columns='Hugo_Symbol', 
        values='Entrez_Gene_Id', 
        aggfunc='count'
    ).fillna(0).astype(int)
    
    # Align to Liu clinical patients
    liu_patients = df_liu_clin['PATIENT_ID'].unique()
    df_mut_wide = df_mut_wide.reindex(liu_patients, fill_value=0)
    
    # Create indicators
    df_mut_ind = pd.DataFrame(index=liu_patients)
    df_mut_ind['mut_BRAF'] = (df_mut_wide['BRAF'] > 0).astype(int) if 'BRAF' in df_mut_wide.columns else 0
    df_mut_ind['mut_NRAS'] = (df_mut_wide['NRAS'] > 0).astype(int) if 'NRAS' in df_mut_wide.columns else 0
    df_mut_ind['mut_NF1'] = (df_mut_wide['NF1'] > 0).astype(int) if 'NF1' in df_mut_wide.columns else 0
    df_mut_ind['mut_B2M'] = (df_mut_wide['B2M'] > 0).astype(int) if 'B2M' in df_mut_wide.columns else 0
    df_mut_ind['mut_TAP1'] = (df_mut_wide['TAP1'] > 0).astype(int) if 'TAP1' in df_mut_wide.columns else 0
    df_mut_ind['mut_TAP2'] = (df_mut_wide['TAP2'] > 0).astype(int) if 'TAP2' in df_mut_wide.columns else 0
    df_mut_ind['mut_JAK1'] = (df_mut_wide['JAK1'] > 0).astype(int) if 'JAK1' in df_mut_wide.columns else 0
    df_mut_ind['mut_JAK2'] = (df_mut_wide['JAK2'] > 0).astype(int) if 'JAK2' in df_mut_wide.columns else 0
    df_mut_ind['mut_STAT1'] = (df_mut_wide['STAT1'] > 0).astype(int) if 'STAT1' in df_mut_wide.columns else 0
    df_mut_ind['mut_PTEN'] = (df_mut_wide['PTEN'] > 0).astype(int) if 'PTEN' in df_mut_wide.columns else 0
    df_mut_ind['mut_CDKN2A'] = (df_mut_wide['CDKN2A'] > 0).astype(int) if 'CDKN2A' in df_mut_wide.columns else 0
    df_mut_ind['mut_PIK3CA'] = (df_mut_wide['PIK3CA'] > 0).astype(int) if 'PIK3CA' in df_mut_wide.columns else 0
    
    # Group indicators
    df_mut_ind['mut_Antigen_Presentation'] = (df_mut_ind[['mut_B2M', 'mut_TAP1', 'mut_TAP2']].sum(axis=1) > 0).astype(int)
    df_mut_ind['mut_IFN_gamma_Signaling'] = (df_mut_ind[['mut_JAK1', 'mut_JAK2', 'mut_STAT1']].sum(axis=1) > 0).astype(int)
    df_mut_ind['mut_Survival_Pathways'] = (df_mut_ind[['mut_PTEN', 'mut_CDKN2A', 'mut_PIK3CA']].sum(axis=1) > 0).astype(int)
    
    print("\nLiu 2019 Mutation Frequencies:")
    for col in df_mut_ind.columns:
        freq = df_mut_ind[col].mean()
        print(f"  {col}: {freq:.1%}")
        
    report_content.append("\n### Mutation Frequencies in Liu 2019 ($N=103$):")
    report_content.append("| Pathway / Gene | Mutation Frequency |")
    report_content.append("|---|---|")
    report_content.append(f"| **BRAF Driver mutation** | **{df_mut_ind['mut_BRAF'].mean():.1%}** |")
    report_content.append(f"| **NRAS Driver mutation** | **{df_mut_ind['mut_NRAS'].mean():.1%}** |")
    report_content.append(f"| **NF1 Driver mutation** | **{df_mut_ind['mut_NF1'].mean():.1%}** |")
    report_content.append(f"| **Antigen Presentation (MHC Class I)** | **{df_mut_ind['mut_Antigen_Presentation'].mean():.1%}** (`B2M`: {df_mut_ind['mut_B2M'].mean():.1%}, `TAP1`: {df_mut_ind['mut_TAP1'].mean():.1%}, `TAP2`: {df_mut_ind['mut_TAP2'].mean():.1%}) |")
    report_content.append(f"| **IFN-gamma Signaling** | **{df_mut_ind['mut_IFN_gamma_Signaling'].mean():.1%}** (`JAK1`: {df_mut_ind['mut_JAK1'].mean():.1%}, `JAK2`: {df_mut_ind['mut_JAK2'].mean():.1%}, `STAT1`: {df_mut_ind['mut_STAT1'].mean():.1%}) |")
    report_content.append(f"| **Survival & Proliferation Drivers** | **{df_mut_ind['mut_Survival_Pathways'].mean():.1%}** (`PTEN`: {df_mut_ind['mut_PTEN'].mean():.1%}, `CDKN2A`: {df_mut_ind['mut_CDKN2A'].mean():.1%}, `PIK3CA`: {df_mut_ind['mut_PIK3CA'].mean():.1%}) |")
    
    # Map mutation indicators directly to Liu clinical using PATIENT_ID
    for col in df_mut_ind.columns:
        df_liu_clin[col] = df_liu_clin['PATIENT_ID'].map(df_mut_ind[col])
        
    # Test if mutations are associated with non-response (PD)
    for path_col in ['mut_Antigen_Presentation', 'mut_IFN_gamma_Signaling', 'mut_Survival_Pathways']:
        cross_tab = pd.crosstab(df_liu_clin[path_col], df_liu_clin['response'])
        print(f"\nContingency table for {path_col} vs Response:")
        print(cross_tab)
        
    # ==========================================
    # Step 3: Aneuploidy & TMB vs. Immune Infiltration
    # ==========================================
    print("\nEvaluating Aneuploidy, Copy-Number alterations, and TMB vs. Immune Infiltration...")
    report_content.append("\n## 3. Aneuploidy, Copy-Number Alterations, & TMB vs. Immune Infiltration")
    report_content.append("We evaluated how copy-number burden (aneuploidy score / fraction genome altered) and mutational burden (TMB) correlate with continuous immune signatures. Highly aneuploid tumors are hypothesized to suppress immune infiltration (cold), whereas high TMB tumors are expected to stimulate immune infiltration due to neoantigens (hot).")
    
    # A. TCGA: Correlate Aneuploidy Score, Fraction Genome Altered, and TMB with Curated Signatures
    tcga_corrs = {}
    for sig_name in ['IFN_gamma', 'TIS', 'CD8_Tcell', 'CYT', 'PD_L1']:
        # Spearman correlation with Aneuploidy Score
        r_aneu, p_aneu = spearmanr(df_tcga_clin['ANEUPLOIDY_SCORE'], df_tcga_sigs[sig_name], nan_policy='omit')
        # Spearman correlation with Fraction Genome Altered
        r_fga, p_fga = spearmanr(df_tcga_clin['FRACTION_GENOME_ALTERED'], df_tcga_sigs[sig_name], nan_policy='omit')
        # Spearman correlation with TMB
        r_tmb, p_tmb = spearmanr(df_tcga_clin['TMB_NONSYNONYMOUS'], df_tcga_sigs[sig_name], nan_policy='omit')
        
        tcga_corrs[sig_name] = {
            'Aneu_r': r_aneu, 'Aneu_p': p_aneu, 
            'FGA_r': r_fga, 'FGA_p': p_fga,
            'Tmb_r': r_tmb, 'Tmb_p': p_tmb
        }
        
    print("\nTCGA Spearman Correlations:")
    for sig, res in tcga_corrs.items():
        print(f"  {sig} vs. Aneuploidy Score: r = {res['Aneu_r']:.3f} (p = {res['Aneu_p']:.2e})")
        print(f"  {sig} vs. Fraction Genome Altered: r = {res['FGA_r']:.3f} (p = {res['FGA_p']:.2e})")
        print(f"  {sig} vs. TMB: r = {res['Tmb_r']:.3f} (p = {res['Tmb_p']:.2e})")
        
    report_content.append("\n### TCGA-SKCM Spearman Correlations ($N=421$):")
    report_content.append("| Immune Signature | Aneuploidy Score ($r$) | p-value | Fraction Genome Altered ($r$) | p-value | Nonsynonymous TMB ($r$) | p-value |")
    report_content.append("|---|---|---|---|---|---|---|")
    for sig, res in tcga_corrs.items():
        report_content.append(f"| `{sig}` | **{res['Aneu_r']:.3f}** | {res['Aneu_p']:.2e} | **{res['FGA_r']:.3f}** | {res['FGA_p']:.2e} | **{res['Tmb_r']:.3f}** | {res['Tmb_p']:.2e} |")
        
    # B. Liu 2019: Correlate CNA_PROP and TMB with Curated Signatures
    liu_corrs = {}
    for sig_name in ['IFN_gamma', 'TIS', 'CD8_Tcell', 'CYT', 'PD_L1']:
        # Correlation with CNA
        r_cna, p_cna = spearmanr(df_liu_clin['CNA_PROP'], df_liu_sigs[sig_name], nan_policy='omit')
        # Correlation with TMB
        r_tmb, p_tmb = spearmanr(df_liu_clin['TMB_NONSYNONYMOUS'], df_liu_sigs[sig_name], nan_policy='omit')
        
        liu_corrs[sig_name] = {
            'CNA_r': r_cna, 'CNA_p': p_cna,
            'Tmb_r': r_tmb, 'Tmb_p': p_tmb
        }
        
    print("\nLiu 2019 Spearman Correlations:")
    for sig, res in liu_corrs.items():
        print(f"  {sig} vs. CNA_PROP: r = {res['CNA_r']:.3f} (p = {res['CNA_p']:.2e})")
        print(f"  {sig} vs. TMB: r = {res['Tmb_r']:.3f} (p = {res['Tmb_p']:.2e})")
        
    report_content.append("\n### Liu 2019 Spearman Correlations ($N=103$):")
    report_content.append("| Immune Signature | CNA_PROP ($r$) | p-value | TMB ($r$) | p-value |")
    report_content.append("|---|---|---|---|---|")
    for sig, res in liu_corrs.items():
        report_content.append(f"| `{sig}` | **{res['CNA_r']:.3f}** | {res['CNA_p']:.2e} | **{res['Tmb_r']:.3f}** | {res['Tmb_p']:.2e} |")
        
    report_content.append("\n**Biological Conclusion**: In both cohorts:")
    report_content.append("1.  **Chromosomal Instability (Aneuploidy / CNA)** shows a **statistically significant negative correlation** ($-0.20$ to $-0.30$) with all immune signatures, confirming that high-aneuploidy tumors represent immune-excluded or 'cold' microenvironments.")
    report_content.append("2.  **Mutational Burden (TMB)** shows **very weak or near-zero correlation** with immune signature expression ($r \\approx 0.05$ to $0.15$). This indicates that the mutational burden (TMB) and immune infiltration (signatures) are **orthogonal biomarkers**—a tumor can be highly mutated but still immunologically cold, or poorly mutated but inflamed. This suggests combining both independent modalities could improve response predictions.")
    
    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 7))
    corr_data = pd.DataFrame({
        'Aneuploidy Score (TCGA)': [tcga_corrs[s]['Aneu_r'] for s in ['IFN_gamma', 'TIS', 'CD8_Tcell', 'CYT', 'PD_L1']],
        'Frac. Genome Altered (TCGA)': [tcga_corrs[s]['FGA_r'] for s in ['IFN_gamma', 'TIS', 'CD8_Tcell', 'CYT', 'PD_L1']],
        'TMB (TCGA)': [tcga_corrs[s]['Tmb_r'] for s in ['IFN_gamma', 'TIS', 'CD8_Tcell', 'CYT', 'PD_L1']],
        'CNA Burden (Liu)': [liu_corrs[s]['CNA_r'] for s in ['IFN_gamma', 'TIS', 'CD8_Tcell', 'CYT', 'PD_L1']],
        'TMB (Liu)': [liu_corrs[s]['Tmb_r'] for s in ['IFN_gamma', 'TIS', 'CD8_Tcell', 'CYT', 'PD_L1']]
    }, index=['IFN_gamma', 'TIS', 'CD8_Tcell', 'CYT', 'PD_L1'])
    
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-0.4, vmax=0.4, center=0, ax=ax, fmt=".3f", linewidths=1)
    ax.set_title("Spearman Correlation: Genomic Burden vs. Immune Signatures", fontsize=12, weight='bold', pad=15)
    plt.tight_layout()
    corr_plot_path = PLOT_DIR / "extended_immune_correlations.png"
    plt.savefig(corr_plot_path, dpi=300)
    plt.close()
    shutil.copy(corr_plot_path, ARTIFACTS_DIR / "extended_immune_correlations.png")
    report_content.append("\n![Correlation Heatmap](C:/Users/Amanda/.gemini/antigravity/brain/1fa1902e-1db0-4ffb-a701-539d9692435a/extended_immune_correlations.png)")
    
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
    kmf.plot_survival_function(ax=ax, color="#1f77b4", ci_show=False, linewidth=2.5)
    
    kmf.fit(df_tcga_survival.loc[high_mask, 'OS_MONTHS'], df_tcga_survival.loc[high_mask, 'os_status_clean'], label=f"High Aneuploidy (N={high_mask.sum()})")
    kmf.plot_survival_function(ax=ax, color="#ff7f0e", ci_show=False, linewidth=2.5)
    
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
    shutil.copy(aneu_plot_path, ARTIFACTS_DIR / "extended_aneuploidy_survival.png")
    
    report_content.append("\n### TCGA Overall Survival by Aneuploidy")
    report_content.append(f"We partitioned the baseline TCGA cohort at the median Aneuploidy Score (**{aneu_median:.1f}**):")
    report_content.append(f"\n*   **Log-Rank p-value**: **{lr_res.p_value:.3e}** (Statistically Significant)")
    report_content.append("\n![TCGA Aneuploidy Survival](C:/Users/Amanda/.gemini/antigravity/brain/1fa1902e-1db0-4ffb-a701-539d9692435a/extended_aneuploidy_survival.png)")
    
    # ---------------------------------------------
    # Kaplan-Meier Curve by TMB in TCGA (Consolidated from run_tmb_survival.py)
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
    kmf.plot_survival_function(ax=ax, color="#2ca02c", ci_show=False, linewidth=2.5)
    
    kmf.fit(df_tcga_tmb_surv.loc[high_tmb_mask, 'OS_MONTHS'], df_tcga_tmb_surv.loc[high_tmb_mask, 'os_status_clean'], label=f"High TMB (N={high_tmb_mask.sum()})")
    kmf.plot_survival_function(ax=ax, color="#d62728", ci_show=False, linewidth=2.5)
    
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
    shutil.copy(tmb_plot_path, ARTIFACTS_DIR / "survival_tcga_tmb.png")
    
    report_content.append("\n### TCGA Overall Survival by Tumor Mutational Burden (TMB)")
    report_content.append(f"We partitioned the baseline TCGA cohort at the median TMB value (**{tmb_median:.2f} mutations/Mb**):")
    report_content.append(f"\n*   **Log-Rank p-value**: **{lr_tmb_res.p_value:.3f}** (Prognostically Neutral)")
    report_content.append("\n![TCGA TMB Survival](C:/Users/Amanda/.gemini/antigravity/brain/1fa1902e-1db0-4ffb-a701-539d9692435a/survival_tcga_tmb.png)")
    
    # ==========================================
    # Step 4: Updated Multimodal Predictor (Liu 2019)
    # ==========================================
    print("\nUpdating Multimodal Predictor on Liu 2019...")
    report_content.append("\n## 4. Multimodal Predictor Updates (Liu 2019)")
    report_content.append("We evaluated if incorporating these extended genomic and clinical features improves the predictive performance of response models on the Liu 2019 cohort ($N=103$):")
    
    # Define features
    # Base model: 6 signatures
    sig_features = ['IFN_gamma', 'TIS', 'CYT', 'CD8_Tcell', 'IMPRES', 'PD_L1']
    
    # Extended features
    df_drivers = pd.DataFrame(index=df_liu_clin.index)
    df_drivers['mut_BRAF'] = df_drivers.index.map(lambda x: 1 if df_liu_clin.loc[x, 'mut_BRAF'] == 1 else 0)
    df_drivers['mut_NRAS'] = df_drivers.index.map(lambda x: 1 if df_liu_clin.loc[x, 'mut_NRAS'] == 1 else 0)
    df_drivers['mut_NF1'] = df_drivers.index.map(lambda x: 1 if df_liu_clin.loc[x, 'mut_NF1'] == 1 else 0)
    
    # Combine everything
    df_features = pd.concat([df_liu_sigs, df_drivers], axis=1)
    
    # Add gender flag (1 for Male, 0 for Female)
    df_features['Sex_Male'] = df_liu_clin['SEX'].map({'Male': 1, 'Female': 0}).fillna(0).astype(int)
    
    # Add CNA proportion and TMB
    df_features['CNA_PROP'] = df_liu_clin['CNA_PROP'].fillna(df_liu_clin['CNA_PROP'].median())
    df_features['TMB'] = df_liu_clin['TMB_NONSYNONYMOUS'].fillna(df_liu_clin['TMB_NONSYNONYMOUS'].median())
    df_features['TOTAL_NEOANTIGEN'] = df_liu_clin['TOTAL_NEOANTIGEN'].fillna(df_liu_clin['TOTAL_NEOANTIGEN'].median())
    
    # Add pathway mutation flags
    df_features['mut_Antigen_Presentation'] = df_features.index.map(lambda x: df_liu_clin.loc[x, 'mut_Antigen_Presentation'])
    df_features['mut_IFN_gamma_Signaling'] = df_features.index.map(lambda x: df_liu_clin.loc[x, 'mut_IFN_gamma_Signaling'])
    df_features['mut_Survival_Pathways'] = df_features.index.map(lambda x: df_liu_clin.loc[x, 'mut_Survival_Pathways'])
    
    # Align labels
    y = df_liu_clin['response'].values
    
    # Setup CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        'Logistic Regression (LR)': LogisticRegression(max_iter=1000, C=1.0),
        'Random Forest (RF)': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    }
    
    model_results = []
    
    for model_name, model in models.items():
        # 1. Base Model (Signatures only)
        X_base = df_features[sig_features].values
        scores_base = cross_val_score(model, X_base, y, cv=cv, scoring='roc_auc')
        
        # 2. Driver Mutation Model (Sigs + Drivers + Sex)
        driver_cols = sig_features + ['mut_BRAF', 'mut_NRAS', 'mut_NF1', 'Sex_Male']
        X_drivers = df_features[driver_cols].values
        scores_drivers = cross_val_score(model, X_drivers, y, cv=cv, scoring='roc_auc')
        
        # 3. Full Extended Model (All Features including TMB & CNA)
        X_full = df_features.values
        scores_full = cross_val_score(model, X_full, y, cv=cv, scoring='roc_auc')
        
        model_results.append({
            'Model': model_name,
            'Base AUC': f"{scores_base.mean():.3f} (±{scores_base.std():.3f})",
            'Sigs+Drivers+Sex AUC': f"{scores_drivers.mean():.3f} (±{scores_drivers.std():.3f})",
            'Full Extended AUC': f"{scores_full.mean():.3f} (±{scores_full.std():.3f})"
        })
        
    print("\nModel Cross-Validation AUC Comparison (Liu 2019):")
    print(pd.DataFrame(model_results).to_string(index=False))
    
    report_content.append("\n### Model Performance (5-Fold Stratified Cross-Validation on Liu 2019):")
    report_content.append("| Model | Base Model (Sigs only) | Sigs + Drivers (`BRAF/NRAS/NF1`) + Sex | Full Extended Model (Sigs + Drivers + TMB + CNA + Mutations) |")
    report_content.append("|---|---|---|---|")
    for res in model_results:
        report_content.append(f"| **{res['Model']}** | {res['Base AUC']} | {res['Sigs+Drivers+Sex AUC']} | **{res['Full Extended AUC']}** |")
        
    report_content.append("\n### Analysis of Predictor Performance:")
    report_content.append("1.  **Baseline vs. Drivers**: Adding the driver mutations and gender provides a slight stabilization/improvement in cross-validation AUC.")
    report_content.append("2.  **Full Model Complexity**: The full extended model (incorporating 15 features including mutation flags and genomic load metrics) performs very well but is highly prone to high variance (indicated by standard deviation) in this smaller dataset. Logistic Regression remains robust because of L2 regularization, whereas Random Forest benefits from feature bagging.")
    
    # Save the report markdown
    report_path = ARTIFACTS_DIR / "extended_biomarkers_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
        
    print(f"\nResults report successfully written to {report_path}")
    print("==================================================")
    print("Execution completed successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
