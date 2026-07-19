import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Set matplotlib backend to Agg to avoid GUI errors
import matplotlib
matplotlib.use('Agg')

# Add project root to sys.path for importing src modules
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_liu_2019, load_hugo_2016, load_riaz_2017
from src.signatures import extract_all_signatures
from pycombat import Combat

from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style

# Set style
set_presentation_style()

# Paths
DATA_DIR = BASE_DIR / "data"
SIG_PLOT_DIR = BASE_DIR / "plots" / "signatures"
SIG_PLOT_DIR.mkdir(exist_ok=True, parents=True)
CLIN_PLOT_DIR = BASE_DIR / "plots" / "clinical"
CLIN_PLOT_DIR.mkdir(exist_ok=True, parents=True)

def main():
    print("==================================================")
    print("Phase 1: Loading & Batch-Correcting Cohort Signatures...")
    print("==================================================")
    
    expr_liu, clin_liu = load_liu_2019(DATA_DIR)
    expr_hugo, clin_hugo = load_hugo_2016(DATA_DIR)
    expr_riaz, clin_riaz = load_riaz_2017(DATA_DIR)
    
    common_genes = expr_liu.columns.intersection(expr_hugo.columns).intersection(expr_riaz.columns)
    
    sig_liu = extract_all_signatures(expr_liu[common_genes])
    sig_hugo = extract_all_signatures(expr_hugo[common_genes])
    sig_riaz = extract_all_signatures(expr_riaz[common_genes])
    
    y_liu = clin_liu.loc[sig_liu.index, 'response']
    y_hugo = clin_hugo.loc[sig_hugo.index, 'response']
    y_riaz = clin_riaz.loc[sig_riaz.index, 'response']
    
    # Filter out samples with missing response (NaN)
    non_nan_liu = y_liu.dropna().index
    non_nan_hugo = y_hugo.dropna().index
    non_nan_riaz = y_riaz.dropna().index
    
    sig_liu = sig_liu.loc[non_nan_liu]
    y_liu = y_liu.loc[non_nan_liu]
    sig_hugo = sig_hugo.loc[non_nan_hugo]
    y_hugo = y_hugo.loc[non_nan_hugo]
    sig_riaz = sig_riaz.loc[non_nan_riaz]
    y_riaz = y_riaz.loc[non_nan_riaz]
    
    # Pool and batch correct
    sig_all = pd.concat([sig_liu, sig_hugo, sig_riaz], axis=0)
    y_all = pd.concat([y_liu, y_hugo, y_riaz], axis=0)
    batches = (['liu'] * len(sig_liu)) + (['hugo'] * len(sig_hugo)) + (['riaz'] * len(sig_riaz))
    
    sig_corrected_arr = Combat().fit_transform(sig_all.values, batches)
    sig_corrected = pd.DataFrame(sig_corrected_arr, index=sig_all.index, columns=sig_all.columns)
    
    print(f"Corrected signatures matrix shape: {sig_corrected.shape}")
    
    print("\n==================================================")
    print("Phase 2: Generating Signature Correlation Heatmap...")
    print("==================================================")
    
    plt.figure(figsize=(8, 6))
    corr_matrix = sig_corrected.corr(method='spearman')
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
    plt.title("Spearman Correlation Heatmap of Batch-Corrected Immune Signatures", fontsize=12, fontweight='bold', pad=15)
    
    corr_path = SIG_PLOT_DIR / "signature_correlation_heatmap.png"
    plt.savefig(corr_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {corr_path}")
    
    print("\n==================================================")
    print("Phase 3: Generating Univariate Response Plots (Box+Jitter)...")
    print("==================================================")
    
    # Box plots + Jitter (stripplot)
    box_palette = {'Responder': '#99D8C9', 'Non-Responder': '#FCAE91'}       # Muted green / vermillion fill
    strip_palette = {'Responder': RESPONSE_PALETTE['CR/PR'], 'Non-Responder': RESPONSE_PALETTE['PD']} # Okabe-Ito Bluish Green & Vermillion Red

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    for i, col in enumerate(sig_corrected.columns):
        ax = axes[i]
        df_plot = pd.DataFrame({
            'Score': sig_corrected[col],
            'Response': y_all.map({1.0: 'Responder', 0.0: 'Non-Responder'})
        })
        resp_scores = df_plot[df_plot['Response'] == 'Responder']['Score']
        non_resp_scores = df_plot[df_plot['Response'] == 'Non-Responder']['Score']
        stat, p_val = mannwhitneyu(resp_scores, non_resp_scores, alternative='two-sided')
        
        # Draw Boxplot with light Okabe-Ito tints
        sns.boxplot(x='Response', y='Score', data=df_plot, ax=ax, 
                    palette=box_palette, 
                    showfliers=False, width=0.5, hue='Response', legend=False)
        # Overlay Jittered Stripplot with full Okabe-Ito colors
        sns.stripplot(x='Response', y='Score', data=df_plot, ax=ax, 
                      palette=strip_palette, 
                      jitter=True, size=4, alpha=0.7, dodge=False, hue='Response', legend=False)
        
        ax.set_title(f"{col}\n(Wilcoxon p = {p_val:.2e})", fontsize=11, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("Signature Score")
    plt.suptitle("Signature Distributions (Box + Jitter) by Immunotherapy Response", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    box_jitter_path = SIG_PLOT_DIR / "signature_box_jitter_by_response.png"
    plt.savefig(box_jitter_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved box + jitter plots to {box_jitter_path}")
    
    print("\n==================================================")
    print("Phase 4: Generating Forest Plot of Odds Ratios...")
    print("==================================================")
    
    # Standardise features so coefficients/odds ratios are directly comparable (per 1 SD increase)
    scaler = StandardScaler()
    sig_scaled = pd.DataFrame(scaler.fit_transform(sig_corrected), columns=sig_corrected.columns, index=sig_corrected.index)
    
    # Fit standard Logistic Regression
    lr = LogisticRegression(C=1e12, max_iter=1000)
    lr.fit(sig_scaled, y_all)
    
    beta = lr.coef_[0]
    odds_ratios = np.exp(beta)
    
    # Calculate confidence intervals via asymptotic Wald standard errors
    X_design = np.hstack([np.ones((sig_scaled.shape[0], 1)), sig_scaled.values])
    p = lr.predict_proba(sig_scaled)[:, 1]
    W = np.diag(p * (1 - p))
    cov_matrix = np.linalg.inv(X_design.T @ W @ X_design)
    standard_errors = np.sqrt(np.diag(cov_matrix))[1:] # drop intercept standard error
    
    z = 1.96
    ci_lower_beta = beta - z * standard_errors
    ci_upper_beta = beta + z * standard_errors
    
    ci_lower_or = np.exp(ci_lower_beta)
    ci_upper_or = np.exp(ci_upper_beta)
    
    df_forest = pd.DataFrame({
        'Feature': sig_corrected.columns,
        'OddsRatio': odds_ratios,
        'CILower': ci_lower_or,
        'CIUpper': ci_upper_or
    }).sort_values(by='OddsRatio', ascending=True)
    
    print("\nOdds Ratios and 95% Confidence Intervals (per 1 SD increase in signature):")
    for _, row in df_forest.iterrows():
        print(f"  {row['Feature']:<12} : OR = {row['OddsRatio']:.3f} (95% CI: {row['CILower']:.3f} - {row['CIUpper']:.3f})")
        
    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(df_forest))
    
    plt.errorbar(
        df_forest['OddsRatio'], y_pos, 
        xerr=[df_forest['OddsRatio'] - df_forest['CILower'], df_forest['CIUpper'] - df_forest['OddsRatio']],
        fmt='o', color=COHORT_PALETTE['Liu 2019'], ecolor='#555555', elinewidth=2, capsize=6, markersize=8,
        label='Odds Ratio (95% CI)'
    )
    
    plt.axvline(x=1.0, color='#D55E00', linestyle='--', linewidth=1.5, label='Null Effect (OR = 1.0)')
    
    plt.yticks(y_pos, df_forest['Feature'], fontsize=11, fontweight='bold')
    plt.xlabel("Odds Ratio of Response (per 1 Standard Deviation Increase)", fontsize=11, fontweight='bold')
    plt.title("Forest Plot of Immune Signature Odds Ratios\n(Multivariable Logistic Regression Model)", fontsize=12, fontweight='bold', pad=15)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    
    plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xticks([0.2, 0.5, 1.0, 2.0, 5.0])
    
    forest_path = SIG_PLOT_DIR / "forest_plot_odds_ratios.png"
    plt.savefig(forest_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\nSaved Forest Plot to {forest_path}")
    
    print("==================================================")

if __name__ == "__main__":
    main()
