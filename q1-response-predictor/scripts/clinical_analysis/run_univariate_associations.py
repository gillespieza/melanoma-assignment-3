import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import fisher_exact, mannwhitneyu

# Add project root to sys.path for importing src modules
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_liu_2019, load_hugo_2016, load_riaz_2017

# Paths
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots" / "clinical"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

def calculate_associations(df, cohort_name):
    results = []
    
    # 1. Categorical variables vs Response
    categorical_vars = {
        'Sex (Male vs Female)': 'SEX',
        'Stage (IV vs III)': 'CLINICAL_STAGE',
        'BRAF Mutation (Mut vs WT)': 'mut_BRAF',
        'NRAS Mutation (Mut vs WT)': 'mut_NRAS',
        'NF1 Mutation (Mut vs WT)': 'mut_NF1'
    }
    
    for label, col in categorical_vars.items():
        if col in df.columns:
            temp = df[[col, 'response']].dropna()
            if len(temp) > 0 and len(temp[col].unique()) == 2:
                contingency = pd.crosstab(temp[col], temp['response'])
                if contingency.shape == (2, 2):
                    odds_ratio, p_val = fisher_exact(contingency)
                    results.append({
                        "Cohort": cohort_name,
                        "Variable": label,
                        "Test": "Fisher's Exact",
                        "Statistic": odds_ratio,
                        "p-value": p_val,
                        "Type": "Categorical"
                    })
    
    # 2. Continuous variables vs Response
    continuous_vars = {
        'Age': ['AGE', 'AGE_AT_DIAGNOSIS', 'AGE (YRS)', 'age'],
        'TMB': ['TMB_NONSYNONYMOUS'],
        'SNV Neoantigens': ['SNV_NEOANTIGEN'],
        'Indel Neoantigens': ['INDEL_NEOANTIGEN']
    }
    
    for label, cols in continuous_vars.items():
        col = None
        for c in cols:
            if c in df.columns:
                col = c
                break
        if col:
            temp = df[[col, 'response']].dropna()
            temp[col] = pd.to_numeric(temp[col], errors='coerce')
            temp = temp.dropna()
            
            responders = temp[temp['response'] == 1.0][col]
            non_responders = temp[temp['response'] == 0.0][col]
            
            if len(responders) > 1 and len(non_responders) > 1:
                stat, p_val = mannwhitneyu(responders, non_responders, alternative='two-sided')
                mean_r = responders.mean()
                mean_nr = non_responders.mean()
                diff = mean_r - mean_nr
                results.append({
                    "Cohort": cohort_name,
                    "Variable": label,
                    "Test": "Mann-Whitney U",
                    "Statistic": diff,
                    "p-value": p_val,
                    "Type": "Continuous"
                })
                
    return pd.DataFrame(results)

def main():
    print("==================================================")
    print("Computing Univariate Associations with Response")
    print("==================================================\n")

    # Load cohorts
    _, clin_liu = load_liu_2019(DATA_DIR)
    _, clin_hugo = load_hugo_2016(DATA_DIR)
    _, clin_riaz = load_riaz_2017(DATA_DIR)

    # Standardize stage names and columns for consistency before comparison
    for df in [clin_liu, clin_hugo, clin_riaz]:
        if 'SEX' in df.columns:
            df['SEX'] = df['SEX'].map({'Male': 'Male', 'Female': 'Female', 'M': 'Male', 'F': 'Female'})
        if 'CLINICAL_STAGE' in df.columns:
            df['CLINICAL_STAGE'] = df['CLINICAL_STAGE'].apply(lambda x: 'IV' if str(x).startswith('IV') else ('III' if str(x).startswith('III') else np.nan))

    results_liu = calculate_associations(clin_liu, "Liu 2019")
    results_hugo = calculate_associations(clin_hugo, "Hugo 2016")
    results_riaz = calculate_associations(clin_riaz, "Riaz 2017")

    # Pool data
    common_cols = ['response', 'SEX', 'CLINICAL_STAGE', 'mut_BRAF', 'mut_NRAS', 'mut_NF1', 'TMB_NONSYNONYMOUS']
    pooled_df = pd.concat([
        clin_liu[[c for c in common_cols if c in clin_liu.columns]],
        clin_hugo[[c for c in common_cols if c in clin_hugo.columns]],
        clin_riaz[[c for c in common_cols if c in clin_riaz.columns]]
    ], ignore_index=True)
    results_pooled = calculate_associations(pooled_df, "Pooled IO")

    all_results = pd.concat([results_liu, results_hugo, results_riaz, results_pooled], ignore_index=True)
    print(all_results.to_string())

    # Plot log10 p-values for visual comparison
    all_results['-log10(p-value)'] = -np.log10(all_results['p-value'])

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.barplot(
        data=all_results,
        x='Variable',
        y='-log10(p-value)',
        hue='Cohort',
        ax=ax,
        palette="viridis",
        edgecolor="black"
    )

    ax.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=1.5, label='p = 0.05 (Significant)')
    ax.axhline(-np.log10(0.01), color='darkred', linestyle=':', linewidth=1.5, label='p = 0.01')
    
    ax.set_title("Statistical Significance of Univariate Associations with Response", fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel("-log10(p-value)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Clinical / Genomic Variable", fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=11)

    plt.tight_layout()
    out_path = PLOT_DIR / "univariate_associations.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot to {out_path}")

    # Write a summary table CSV
    all_results.to_csv(PLOT_DIR / "univariate_associations_stats.csv", index=False)

if __name__ == "__main__":
    main()
