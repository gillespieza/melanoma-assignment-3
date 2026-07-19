import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import fisher_exact

# Add project root to sys.path for importing src modules
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_liu_2019, load_hugo_2016, load_riaz_2017

# Paths
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots" / "clinical"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

def calculate_odds_ratio(df, col, value_exposed, value_unexposed):
    temp = df[[col, 'response']].dropna()
    if len(temp) == 0:
        return None
    
    A = len(temp[(temp[col] == value_exposed) & (temp['response'] == 1.0)])
    B = len(temp[(temp[col] == value_exposed) & (temp['response'] == 0.0)])
    C = len(temp[(temp[col] == value_unexposed) & (temp['response'] == 1.0)])
    D = len(temp[(temp[col] == value_unexposed) & (temp['response'] == 0.0)])
    
    table = [[A, B], [C, D]]
    _, p_val = fisher_exact(table)
    
    if A == 0 or B == 0 or C == 0 or D == 0:
        A_c, B_c, C_c, D_c = A + 0.5, B + 0.5, C + 0.5, D + 0.5
    else:
        A_c, B_c, C_c, D_c = A, B, C, D
        
    or_val = (A_c * D_c) / (B_c * C_c)
    se_ln_or = np.sqrt(1/A_c + 1/B_c + 1/C_c + 1/D_c)
    
    ci_lower = np.exp(np.log(or_val) - 1.96 * se_ln_or)
    ci_upper = np.exp(np.log(or_val) + 1.96 * se_ln_or)
    
    return {
        "OR": or_val,
        "CI_lower": ci_lower,
        "CI_upper": ci_upper,
        "p_value": p_val,
        "Counts": f"Exposed: {A}/{A+B}, Unexposed: {C}/{C+D}"
    }

def main():
    print("==================================================")
    print("Generating Forest Plot of Univariate Odds Ratios")
    print("==================================================\n")

    # Load cohorts
    _, clin_liu = load_liu_2019(DATA_DIR)
    _, clin_hugo = load_hugo_2016(DATA_DIR)
    _, clin_riaz = load_riaz_2017(DATA_DIR)

    # Standardize variables for consistency before pooling
    for df in [clin_liu, clin_hugo, clin_riaz]:
        if 'SEX' in df.columns:
            df['SEX'] = df['SEX'].map({'Male': 'Male', 'Female': 'Female', 'M': 'Male', 'F': 'Female'})
        if 'CLINICAL_STAGE' in df.columns:
            df['CLINICAL_STAGE'] = df['CLINICAL_STAGE'].apply(lambda x: 'IV' if str(x).startswith('IV') else ('III' if str(x).startswith('III') else np.nan))

    # Pool data
    common_cols = ['response', 'SEX', 'CLINICAL_STAGE', 'mut_BRAF', 'mut_NRAS', 'mut_NF1', 'TMB_NONSYNONYMOUS', 'AGE_AT_DIAGNOSIS', 'AGE']
    
    # Clean/merge age column
    for df in [clin_liu, clin_hugo, clin_riaz]:
        age_col = [c for c in df.columns if c.upper() in ['AGE', 'AGE_AT_DIAGNOSIS', 'AGE (YRS)', 'AGE_AT_DIAGNOSIS']][0] if any(c.upper() in ['AGE', 'AGE_AT_DIAGNOSIS', 'AGE (YRS)'] for c in df.columns) else None
        if age_col:
            df['age_standardized'] = pd.to_numeric(df[age_col], errors='coerce')
        else:
            df['age_standardized'] = np.nan

    pooled_df = pd.concat([
        clin_liu[[c for c in common_cols + ['age_standardized'] if c in clin_liu.columns]],
        clin_hugo[[c for c in common_cols + ['age_standardized'] if c in clin_hugo.columns]],
        clin_riaz[[c for c in common_cols + ['age_standardized'] if c in clin_riaz.columns]]
    ], ignore_index=True)

    # Dichotomize continuous variables based on median split
    tmb_median = pooled_df['TMB_NONSYNONYMOUS'].median()
    pooled_df['TMB_High'] = pooled_df['TMB_NONSYNONYMOUS'].apply(lambda x: 'High' if x >= tmb_median else 'Low')
    
    age_median = pooled_df['age_standardized'].median()
    pooled_df['Age_High'] = pooled_df['age_standardized'].apply(lambda x: 'High' if x >= age_median else ('Low' if pd.notna(x) else np.nan))

    # Features to analyze
    features = [
        {"label": "TMB (High vs Low)", "col": "TMB_High", "exp": "High", "unexp": "Low"},
        {"label": "Clinical Stage (IV vs III)", "col": "CLINICAL_STAGE", "exp": "IV", "unexp": "III"},
        {"label": "Sex (Male vs Female)", "col": "SEX", "exp": "Male", "unexp": "Female"},
        {"label": "BRAF Mutation (Mut vs WT)", "col": "mut_BRAF", "exp": 1, "unexp": 0},
        {"label": "NRAS Mutation (Mut vs WT)", "col": "mut_NRAS", "exp": 1, "unexp": 0},
        {"label": "NF1 Mutation (Mut vs WT)", "col": "mut_NF1", "exp": 1, "unexp": 0},
        {"label": "Age (High vs Low)", "col": "Age_High", "exp": "High", "unexp": "Low"}
    ]

    plot_data = []
    for f in features:
        res = calculate_odds_ratio(pooled_df, f["col"], f["exp"], f["unexp"])
        if res:
            plot_data.append({
                "Variable": f["label"],
                "OR": res["OR"],
                "CI_lower": res["CI_lower"],
                "CI_upper": res["CI_upper"],
                "p_value": res["p_value"],
                "Counts": res["Counts"]
            })

    df_plot = pd.DataFrame(plot_data)
    print(df_plot)

    from src.styles import RESPONSE_PALETTE, set_presentation_style

    set_presentation_style()
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Reverse order for plotting from top to bottom
    df_plot = df_plot.iloc[::-1].reset_index(drop=True)
    
    y_pos = np.arange(len(df_plot))
    
    # Plot vertical line at OR = 1.0 (No Effect)
    ax.axvline(1.0, color='#555555', linestyle='--', linewidth=1.5, zorder=1)
    
    # Plot Odds Ratios and Confidence Intervals
    for i in range(len(df_plot)):
        row = df_plot.iloc[i]
        is_sig = row['p_value'] < 0.05
        or_val = row['OR']
        
        if is_sig:
            color = RESPONSE_PALETTE['CR/PR'] if or_val > 1.0 else RESPONSE_PALETTE['PD']
            weight = 'bold'
        else:
            color = '#777777'
            weight = 'normal'
            
        ax.errorbar(
            x=or_val, y=i, 
            xerr=[[max(0.01, or_val - row['CI_lower'])], [max(0.01, row['CI_upper'] - or_val)]], 
            fmt='s', 
            color=color, 
            ecolor=color, 
            elinewidth=2.0, 
            capsize=4, 
            capthick=1.5,
            markersize=7,
            zorder=3
        )
        
        # Add labels with OR, CI and p-values on the right side
        label_text = f"OR: {or_val:.2f} (95% CI: {row['CI_lower']:.2f}–{row['CI_upper']:.2f}), p={row['p_value']:.4f}"
        ax.text(row['CI_upper'] * 1.1 if row['CI_upper'] < 10 else 10.5, i, label_text, 
                va='center', ha='left', fontsize=11, color='#222222', weight=weight)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['Variable'], fontsize=12, fontweight='bold', color='#222222')
    ax.set_xlabel("Odds Ratio (OR) of Response (CR/PR)", fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title("Forest Plot: Univariate Odds Ratios for Response (Pooled IO Cohort)", fontsize=15, fontweight='bold', pad=15)
    
    # Set logarithmic scale for Odds Ratio
    ax.set_xscale('log')
    ax.set_xlim(0.05, 15)
    # Custom ticks
    ax.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Labels to guide reader on protective vs risk direction
    ax.text(0.95, -0.15, 'Higher Response Odds (OR > 1.0) \u2192', transform=ax.transAxes, ha='right', va='top', color='#555555', fontsize=9.5, style='italic')
    ax.text(0.05, -0.15, '\u2190 Lower Response Odds (OR < 1.0)', transform=ax.transAxes, ha='left', va='top', color='#555555', fontsize=9.5, style='italic')

    # Add a custom legend explaining the colors
    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([0], [0], marker='s', color='none', markerfacecolor=RESPONSE_PALETTE['PD'], markeredgecolor='none', markersize=8, label='Significant Harmful / Lower Odds (p < 0.05, OR < 1.0)'),
        mlines.Line2D([0], [0], marker='s', color='none', markerfacecolor=RESPONSE_PALETTE['CR/PR'], markeredgecolor='none', markersize=8, label='Significant Beneficial / Higher Odds (p < 0.05, OR > 1.0)'),
        mlines.Line2D([0], [0], marker='s', color='none', markerfacecolor='#777777', markeredgecolor='none', markersize=8, label='Non-Significant (p \u2265 0.05)')
    ]
    ax.legend(
        handles=legend_elements, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.22), 
        ncol=3, 
        frameon=False, 
        fontsize=9.5
    )

    # Clean aesthetics: remove top and right spines
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    out_path = PLOT_DIR / "forest_plot_odds_ratios.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved Forest Plot to {out_path}")

    # Save data to CSV
    df_plot.to_csv(PLOT_DIR / "forest_plot_data.csv", index=False)

if __name__ == "__main__":
    main()
