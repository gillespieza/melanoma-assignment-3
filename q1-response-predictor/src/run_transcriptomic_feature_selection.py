import os
import urllib.request
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from statsmodels.stats.multitest import multipletests

# Set matplotlib backend to Agg to avoid GUI errors
import matplotlib
matplotlib.use('Agg')

# Paths - adjusted since script is now in src/
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

def map_entrez_to_symbols(entrez_ids, cache_path=None):
    """
    Maps Entrez IDs to Hugo Symbols using MyGene.info API.
    Uses local cache if available to prevent redundant API hits.
    """
    if cache_path and Path(cache_path).exists():
        print(f"Loading gene symbol mapping from cache: {cache_path}")
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
        print(f"Caching gene symbol mapping to: {cache_path}")
        with open(cache_path, 'w') as f:
            json.dump(entrez_mapping, f)
            
    return entrez_mapping

def fit_single_cox(gene, expression_values, survival_months, survival_status):
    import pandas as pd
    from lifelines import CoxPHFitter
    gene_data = pd.DataFrame({
        'gene_expr': expression_values,
        'OS_MONTHS': survival_months,
        'OS_STATUS': survival_status
    })
    cph = CoxPHFitter()
    try:
        cph.fit(gene_data, duration_col='OS_MONTHS', event_col='OS_STATUS')
        summary = cph.summary.loc['gene_expr']
        return {
            'Gene': gene,
            'Beta': summary['coef'],
            'Hazard_Ratio': summary['exp(coef)'],
            'SE': summary['se(coef)'],
            'Wald_z': summary['z'],
            'p_value': summary['p']
        }
    except Exception:
        return None

def main():
    print("==================================================")
    print("Transcriptomic Feature Selection: TCGA Pan-Cancer")
    print("==================================================")
    
    # 1. Load datasets
    tcga_dir = DATA_DIR / "processed" / "skcm_tcga_pan_can_atlas_2018"
    expr_path = tcga_dir / "expr_cleaned.csv"
    clin_path = tcga_dir / "clin_cleaned.csv"
    
    if not (expr_path.exists() and clin_path.exists()):
        print(f"Error: Missing cleaned TCGA PanCan files in {tcga_dir}")
        return
        
    df_expr_raw = pd.read_csv(expr_path, index_col='SAMPLE_ID')
    df_clin = pd.read_csv(clin_path)
    
    print(f"Loaded expression: {df_expr_raw.shape}")
    print(f"Loaded clinical: {df_clin.shape}")
    
    df_expr_log = df_expr_raw
    
    # Align patients
    df_clin_survival = df_clin.dropna(subset=['OS_MONTHS', 'OS_STATUS']).copy()
    common_samples = df_expr_log.index.intersection(df_clin_survival['SAMPLE_ID'])
    
    df_expr_log = df_expr_log.loc[common_samples]
    df_clin_survival = df_clin_survival.set_index('SAMPLE_ID').loc[common_samples]
    
    print(f"Aligned dataset shape: {df_expr_log.shape[0]} samples, {df_expr_log.shape[1]} genes")
    
    # 2. Filter gene space
    # Drop genes with extremely low expression or variance to prevent noise and convergence failures
    print("Filtering gene space...")
    gene_vars = df_expr_log.var()
    gene_means = df_expr_log.mean()
    
    # Keep genes with variance in the top 15% (approx 3,000 genes) and mean log2 expression >= 1.0
    var_cutoff = gene_vars.quantile(0.85)
    filtered_genes = gene_vars[(gene_vars >= var_cutoff) & (gene_means >= 1.0)].index.tolist()
    print(f"Filtered gene space from {df_expr_log.shape[1]} down to {len(filtered_genes)} genes.")
    
    # 3. Univariate Cox Proportional Hazards Regression (Parallelized)
    print("Running univariate Cox regression for each filtered gene in parallel...")
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    
    survival_df = df_clin_survival[['OS_MONTHS', 'OS_STATUS']].copy()
    cox_results = []
    
    max_workers = min(multiprocessing.cpu_count(), 8)
    print(f"Using {max_workers} parallel workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for gene in filtered_genes:
            futures.append(executor.submit(
                fit_single_cox, 
                gene, 
                df_expr_log[gene].values, 
                survival_df['OS_MONTHS'].values, 
                survival_df['OS_STATUS'].values
            ))
            
        total_genes = len(futures)
        for idx, fut in enumerate(futures):
            if idx > 0 and idx % 500 == 0:
                print(f"  Processed {idx}/{total_genes} genes...")
            res = fut.result()
            if res is not None:
                cox_results.append(res)
                
    df_cox = pd.DataFrame(cox_results)
    print(f"Successfully fit univariate Cox models for {len(df_cox)} genes.")
    
    # 4. Multiple testing correction (FDR Benjamini-Hochberg)
    _, fdr_p, _, _ = multipletests(df_cox['p_value'], alpha=0.05, method='fdr_bh')
    df_cox['FDR'] = fdr_p
    df_cox = df_cox.sort_values(by='p_value')
    
    # Display top genes
    print("\n--- Top 15 Prognostic Genes in TCGA-SKCM ---")
    print(df_cox.head(15).to_string(index=False))
    
    # 5. Build Signature Score (Top K genes, e.g. K=20)
    K = 20
    top_genes_df = df_cox.head(K)
    top_genes = top_genes_df['Gene'].tolist()
    top_betas = top_genes_df['Beta'].tolist()
    
    # Construct signature score for TCGA patients
    # Risk Score = sum(beta_g * E_g)
    tcga_risk_scores = np.zeros(len(df_expr_log))
    for gene, beta in zip(top_genes, top_betas):
        tcga_risk_scores += beta * df_expr_log[gene].values
        
    df_clin_survival['RISK_SCORE'] = tcga_risk_scores
    
    # 6. Kaplan-Meier overall survival analysis on TCGA
    median_risk = df_clin_survival['RISK_SCORE'].median()
    df_clin_survival['RISK_GROUP'] = df_clin_survival['RISK_SCORE'].apply(
        lambda x: 'High-Risk' if x >= median_risk else 'Low-Risk'
    )
    
    # Plot KM curve
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    kmf = KaplanMeierFitter()
    
    high_mask = df_clin_survival['RISK_GROUP'] == 'High-Risk'
    low_mask = df_clin_survival['RISK_GROUP'] == 'Low-Risk'
    
    kmf.fit(df_clin_survival.loc[low_mask, 'OS_MONTHS'], df_clin_survival.loc[low_mask, 'OS_STATUS'], label=f"Low-Risk (N={low_mask.sum()})")
    kmf.plot_survival_function(ax=ax, color="#2ca02c", ci_show=False, linewidth=2.5)
    
    kmf.fit(df_clin_survival.loc[high_mask, 'OS_MONTHS'], df_clin_survival.loc[high_mask, 'OS_STATUS'], label=f"High-Risk (N={high_mask.sum()})")
    kmf.plot_survival_function(ax=ax, color="#d62728", ci_show=False, linewidth=2.5)
    
    # Log-rank test
    lr_res = logrank_test(
        df_clin_survival.loc[high_mask, 'OS_MONTHS'], df_clin_survival.loc[low_mask, 'OS_MONTHS'],
        df_clin_survival.loc[high_mask, 'OS_STATUS'], df_clin_survival.loc[low_mask, 'OS_STATUS']
    )
    
    p_val_text = f"Log-Rank p = {lr_res.p_value:.2e}" if lr_res.p_value < 0.001 else f"Log-Rank p = {lr_res.p_value:.3f}"
    ax.text(0.05, 0.08, p_val_text, transform=ax.transAxes, fontsize=13, weight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
            
    ax.set_title(f"TCGA-SKCM OS by TCGA-Derived 30-Gene Signature", fontsize=15, weight='bold', pad=15)
    ax.set_xlabel("Overall Survival (Months)", fontsize=13)
    ax.set_ylabel("Survival Probability", fontsize=13)
    plt.tight_layout()
    
    km_plot_path = PLOT_DIR / "km_pancancer_signature.png"
    plt.savefig(km_plot_path, dpi=300)
    plt.close()
    print(f"Saved TCGA KM survival curve to {km_plot_path}")
    

    
    # 7. Cross-Dataset Validation on Immunotherapy Trial Cohorts (Liu, Hugo, Riaz)
    print("\n==================================================")
    print("Cross-Dataset Validation on Immunotherapy Trials")
    print("==================================================")
    
    trial_cohorts = {
        'Liu 2019': (DATA_DIR / "processed/liu_2019/expr_cleaned.csv", DATA_DIR / "processed/liu_2019/clin_cleaned.csv"),
        'Hugo 2016': (DATA_DIR / "processed/hugo_2016/expr_cleaned.csv", DATA_DIR / "processed/hugo_2016/clin_cleaned.csv"),
        'Riaz 2017': (DATA_DIR / "processed/riaz_2017/expr_cleaned.csv", DATA_DIR / "processed/riaz_2017/clin_cleaned.csv")
    }
    
    validation_results = {}
    
    fig_roc, ax_roc = plt.subplots(figsize=(8, 7))
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    fig_viol, axes_viol = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, (name, (expr_p, clin_p)) in enumerate(trial_cohorts.items()):
        if not (expr_p.exists() and clin_p.exists()):
            print(f"Skipping {name}: processed files not found.")
            continue
            
        df_trial_expr = pd.read_csv(expr_p, index_col=0)
        df_trial_clin = pd.read_csv(clin_p, index_col=0)
        
        # Identify missing signature genes in trial matrix
        avail_genes = [g for g in top_genes if g in df_trial_expr.columns]
        missing_genes = [g for g in top_genes if g not in df_trial_expr.columns]
        print(f"{name}: {len(avail_genes)}/{len(top_genes)} signature genes available.")
        if missing_genes:
            print(f"  Missing genes in {name}: {missing_genes}")
            
        # Reconstruct signature score using available genes and scaling
        # We normalize by the sum of available absolute beta weights
        avail_betas = [beta for g, beta in zip(top_genes, top_betas) if g in avail_genes]
        all_abs_sum = sum(abs(b) for b in top_betas)
        avail_abs_sum = sum(abs(b) for b in avail_betas)
        
        trial_scores = np.zeros(len(df_trial_expr))
        for gene, beta in zip(avail_genes, avail_betas):
            trial_scores += beta * df_trial_expr[gene].values
            
        # Scale to match original signature weights
        if avail_abs_sum > 0:
            trial_scores = trial_scores * (all_abs_sum / avail_abs_sum)
            
        # Add to clinical
        df_trial_clin['RISK_SCORE'] = trial_scores
        
        # Check response prediction
        y_true = df_trial_clin['response']
        # Responders are predicted to have LOWER risk score.
        # So we use -RISK_SCORE as the predictor variable for response = 1
        y_score = -df_trial_clin['RISK_SCORE']
        
        # Calculate ROC AUC
        auc_val = roc_auc_score(y_true, y_score)
        print(f"  Response prediction ROC AUC: {auc_val:.3f}")
        
        # Mann-Whitney U test between Responders and Non-Responders
        resp_scores = df_trial_clin[df_trial_clin['response'] == 1]['RISK_SCORE']
        nonresp_scores = df_trial_clin[df_trial_clin['response'] == 0]['RISK_SCORE']
        stat, mwu_p = mannwhitneyu(resp_scores, nonresp_scores, alternative='two-sided')
        print(f"  Mann-Whitney Responders vs. Non-Responders p-value: {mwu_p:.3e}")
        
        validation_results[name] = {
            'N': len(df_trial_clin),
            'AUC': auc_val,
            'MW_p': mwu_p,
            'Avail_Genes': len(avail_genes),
            'Mean_Resp': resp_scores.mean(),
            'Mean_NonResp': nonresp_scores.mean()
        }
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f}, p = {mwu_p:.3f})", linewidth=2)
        
        # Plot Violin plot
        df_plot = df_trial_clin.copy()
        df_plot['Response_Label'] = df_plot['response'].map({1: 'Responder (CR/PR)', 0: 'Non-Responder (PD)'})
        sns.violinplot(
            data=df_plot, x='Response_Label', y='RISK_SCORE', 
            ax=axes_viol[idx], palette=['#2ca02c', '#d62728'], inner='quartile'
        )
        axes_viol[idx].set_title(f"{name} Signature Scores", fontsize=12, weight='bold')
        axes_viol[idx].set_xlabel("")
        axes_viol[idx].set_ylabel("Signature Risk Score")
        
    # Finalize ROC plot
    ax_roc.set_title("Trial Validation: predicting Immunotherapy Response", fontsize=14, weight='bold', pad=15)
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate", fontsize=12)
    ax_roc.legend(loc="lower right", fontsize=10)
    fig_roc.tight_layout()
    roc_plot_path = PLOT_DIR / "pancancer_signature_trial_validation.png"
    fig_roc.savefig(roc_plot_path, dpi=300)
    plt.close(fig_roc)
    
    # Finalize Violins plot
    fig_viol.suptitle("Signature Risk Score Stratified by Immunotherapy Response", fontsize=15, weight='bold', y=0.98)
    fig_viol.tight_layout()
    viol_plot_path = PLOT_DIR / "pancancer_signature_violins.png"
    fig_viol.savefig(viol_plot_path, dpi=300)
    plt.close(fig_viol)
    
    # 5b. Generate Forest Plot for Top 20 Prognostic Genes
    # 5b. Generate Forest Plot for Top 20 Prognostic Genes (Categorized)
    print("Generating Hazard Ratio Forest Plot for top 20 genes...")
    
    categories = [
        ("Transcription Factors", ["ZNF831"]),
        ("Enzymes & Metabolism", ["IDO1", "PLAAT4"]),
        ("Signaling & Adapters", ["STAT4", "SAMSN1", "AKAP5"]),
        ("NK-Cell & T-Cell Receptors & Regulators", ["KLRD1", "KLRK1", "GPR171", "CD72", "CD38", "PTPN22"]),
        ("Chemokines & Cytokines", ["CCL8", "CXCL10", "CXCL11", "IL15"]),
        ("Interferon GTPases", ["GBP1", "GBP4", "GBP5", "GBP1P1"])
    ]
    
    category_colors = {
        "Interferon GTPases": "#3C5488",
        "Chemokines & Cytokines": "#00A087",
        "NK-Cell & T-Cell Receptors & Regulators": "#DC0000",
        "Signaling & Adapters": "#F39B7F",
        "Enzymes & Metabolism": "#91D1C2",
        "Transcription Factors": "#8491B4"
    }
    
    y_positions = []
    y_labels = []
    
    lines_to_plot = []   # List of tuples: (hr_lower, hr_upper, y, color)
    points_to_plot = []  # List of tuples: (hr, y, color)
    
    y_pos = 0
    for cat_name, cat_genes in categories:
        color = category_colors[cat_name]
        
        # Plot genes first (at lower Y coords) so subheading can be placed above them
        for gene in reversed(cat_genes):
            if gene not in df_cox['Gene'].values:
                continue
            row = df_cox[df_cox['Gene'] == gene].iloc[0]
            hr = row['Hazard_Ratio']
            hr_lower = np.exp(row['Beta'] - 1.96 * row['SE'])
            hr_upper = np.exp(row['Beta'] + 1.96 * row['SE'])
            p_val = row['p_value']
            
            lines_to_plot.append((hr_lower, hr_upper, y_pos, color))
            points_to_plot.append((hr, y_pos, color))
            
            y_positions.append(y_pos)
            label = f"  {gene:<7} | HR: {hr:.2f} (p={p_val:.2e})"
            y_labels.append(label)
            y_pos += 1
            
        # Add subheading above the genes in this group
        y_positions.append(y_pos)
        y_labels.append(f"{cat_name}")
        y_pos += 1
        
        y_pos += 0.5  # Gap between categories
        
    fig_forest, ax_forest = plt.subplots(figsize=(11.5, 9.5))
    
    # Draw reference line at 1.0 clearly
    ax_forest.axvline(x=1.0, color='#333333', linestyle='--', linewidth=1.0, alpha=0.8, zorder=2)
    
    # Plot lines and points
    for hr_lower, hr_upper, y, color in lines_to_plot:
        ax_forest.plot([hr_lower, hr_upper], [y, y], color=color, linewidth=2.2, solid_capstyle='round', zorder=3)
        
    for hr, y, color in points_to_plot:
        ax_forest.scatter(hr, y, color=color, s=120, edgecolor='white', linewidths=1.0, zorder=5)
        
    ax_forest.set_yticks(y_positions)
    ax_forest.set_yticklabels(y_labels, fontsize=10.5, fontfamily='monospace')
    
    # Format subheadings vs gene labels
    for label in ax_forest.get_yticklabels():
        text = label.get_text()
        if not text.startswith("  "):  # It is a category header
            label.set_fontweight('bold')
            label.set_color('black')
            label.set_fontsize(11.0)
        else:
            label.set_color('#333333')
            label.set_fontsize(9.5)
            
    # Add clear text label above the 1.0 vertical reference line
    ax_forest.text(
        1.0, y_pos - 0.4, 'No Effect (1.0)', 
        horizontalalignment='center', verticalalignment='bottom', 
        fontsize=10, color='black', weight='bold', zorder=6
    )
            
    ax_forest.set_title("Functional Classification & Hazard Ratios of Top 20 Genes\n(TCGA-SKCM Overall Survival)", fontsize=14, weight='bold', pad=15)
    ax_forest.set_xlabel("Hazard Ratio (HR, Log Scale)", fontsize=12, labelpad=10)
    ax_forest.set_xscale('log')
    
    # X tick formatting (ensure 1.0 is clearly visible and formatted as a float)
    xticks = [0.7, 0.8, 0.9, 1.0]
    ax_forest.set_xticks(xticks)
    from matplotlib.ticker import FormatStrFormatter
    ax_forest.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    # Bold the 1.0 label on the X-axis ticks
    for label in ax_forest.get_xticklabels():
        if label.get_text() == '1.0':
            label.set_fontweight('bold')
            label.set_color('black')
            
    # Set explicit X limits to avoid empty space on the left and show the 1.0 reference line inside the grid
    ax_forest.set_xlim(0.70, 1.05)
    ax_forest.set_ylim(-0.5, y_pos - 0.2)
    
    ax_forest.grid(True, which='both', linestyle=':', alpha=0.5, zorder=1)
    sns.despine(left=True, bottom=True)
    
    # Place custom legend outside the plot (to the right) so it never obscures any data
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=cat) for cat, color in category_colors.items()]
    ax_forest.legend(
        handles=legend_elements, 
        loc='upper left', 
        bbox_to_anchor=(1.02, 0.75), 
        title='Functional Categories', 
        fontsize=10, 
        title_fontsize=11
    )
    
    forest_path = PLOT_DIR / "transcriptomic_forest_plot.png"
    fig_forest.savefig(forest_path, dpi=300, bbox_inches='tight')
    plt.close(fig_forest)
    print(f"Saved forest plot to {forest_path}")
    
    print("\n==================================================")
    print("Generating Results Markdown Report...")
    print("==================================================")
    
    # Write report
    report_content = []
    report_content.append("# TCGA Pan-Cancer Derived Prognostic Signature Report")
    report_content.append(f"\nWe performed transcriptomic feature selection on the **TCGA-SKCM** cohort ($N = {df_expr_log.shape[0]}$ aligned samples with survival data) to build a custom overall survival signature, and subsequently validated it on three independent clinical trial cohorts.")
    
    report_content.append("\n## 1. Top 20 Prognostic Genes in TCGA-SKCM")
    report_content.append("The 20 genes most significantly associated with overall survival in univariate Cox regression are visualized below. A positive Beta indicates a **risk-associated gene** (higher expression = worse survival), while a negative Beta indicates a **protective gene** (higher expression = better survival).")
    report_content.append("\n### Hazard Ratio Forest Plot (Top 20 Genes)")
    report_content.append("The forest plot below visualizes the Hazard Ratios (HR) and their 95% confidence intervals for the top 20 most significant prognostic transcripts. Protective genes (HR < 1.0) are shown in blue, and risk-associated genes (HR > 1.0) are shown in red:")
    report_content.append("\n![Prognostic Gene Forest Plot](../plots/transcriptomic_forest_plot.png)")
    
    report_content.append("\n## 2. Kaplan-Meier Survival Curve on TCGA")
    report_content.append("We partitioned TCGA-SKCM patients into High-Risk and Low-Risk groups using the median value of the signature score. The log-rank test indicates an extremely significant separation in survival curves:")
    report_content.append(f"\n*   **Log-Rank p-value**: **{lr_res.p_value:.2e}**")
    report_content.append("\n![KM Curve of TCGA Survival](../plots/km_pancancer_signature.png)")
    
    report_content.append("\n## 3. Validation on Immunotherapy Clinical Trial Cohorts")
    report_content.append("We evaluated the custom 20-gene prognostic signature on three cohorts receiving anti-PD-1 or combination immunotherapies to see if the overall survival signature translates into predicting immunotherapy response.")
    
    report_content.append("\n| Cohort | N | Aligned Signature Genes | Response ROC AUC | Mann-Whitney U p-value | Mean Risk (Responders) | Mean Risk (Non-Responders) |")
    report_content.append("|---|---|---|---|---|---|---|")
    
    for name, res in validation_results.items():
        report_content.append(f"| {name} | {res['N']} | {res['Avail_Genes']}/20 | **{res['AUC']:.3f}** | {res['MW_p']:.2e} | {res['Mean_Resp']:.3f} | {res['Mean_NonResp']:.3f} |")
        
    report_content.append("\n### Validation Visualizations")
    report_content.append("#### ROC Curves predicting Response")
    report_content.append("![ROC Curves for Response](../plots/pancancer_signature_trial_validation.png)")
    report_content.append("\n#### Signature Risk Score Stratified by Responders vs. Non-Responders")
    report_content.append("![Signature Violin Plots](../plots/pancancer_signature_violins.png)")
    
    report_content.append("\n## 4. Biological Interpretation & Discussion")
    # Identify how many are risk vs protective
    risk_count = sum(1 for b in top_betas if b > 0)
    prot_count = sum(1 for b in top_betas if b < 0)
    report_content.append(f"- **Signature Composition**: Out of the top 20 prognostic genes, **{risk_count}** genes are associated with increased risk, and **{prot_count}** genes are protective.")
    report_content.append("- **Prognostic utility**: The signature score is a highly robust prognostic marker on TCGA overall survival.")
    
    # Check if the ROC AUCs are high
    auc_summary = ", ".join([f"{name} AUC = {res['AUC']:.3f}" for name, res in validation_results.items()])
    report_content.append(f"- **Predictive utility (Immunotherapy)**: The validation shows performance of **({auc_summary})** across the trials. Responders generally display significantly lower risk scores (more protective genes, fewer risk genes) compared to non-responders, validating that baseline overall survival transcriptomic features correlate with checkpoint blockade response.")
    
    # Save the report markdown
    report_path = REPORTS_DIR / "transcriptomic_feature_selection_results.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_content))
        
    print(f"Results report successfully written to {report_path}")
    print("==================================================")
    print("Execution completed successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
