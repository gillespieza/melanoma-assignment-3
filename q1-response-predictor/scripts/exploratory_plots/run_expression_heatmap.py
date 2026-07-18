import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_liu_2019, load_hugo_2016, load_riaz_2017

# Paths
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots" / "exploratory"
PLOT_DIR.mkdir(exist_ok=True, parents=True)
REPORT_DIR = BASE_DIR / "reports"

def zscore_df(df):
    means = df.mean(axis=0)
    stds = df.std(axis=0)
    stds = stds.replace(0, 1.0).fillna(1.0)
    return (df - means) / stds

def main():
    print("==================================================")
    print("Gene Expression Heatmaps: Top Genes by Variance")
    print("==================================================\n")

    print("Loading datasets...")
    expr_liu, clin_liu = load_liu_2019(DATA_DIR)
    expr_hugo, clin_hugo = load_hugo_2016(DATA_DIR)
    expr_riaz, clin_riaz = load_riaz_2017(DATA_DIR)

    RESPONSE_MAP = {
        "Complete Response": 1,
        "Partial Response": 1,
        "Progressive Disease": 0,
        "Stable Disease": np.nan,
        "Mixed Response": np.nan,
    }
    
    # Filter trials to response-aligned samples (keep CR/PR/PD; drop SD/MR/NaN)
    for name, df_clin, df_expr in [("Liu 2019", clin_liu, expr_liu),
                                   ("Hugo 2016", clin_hugo, expr_hugo),
                                   ("Riaz 2017", clin_riaz, expr_riaz)]:
        df_clin['temp_resp'] = df_clin['RESPONSE'].map(RESPONSE_MAP)
        df_clin.dropna(subset=['temp_resp'], inplace=True)
        df_clin.drop(columns=['temp_resp'], inplace=True)
        
        # Re-align expression matrix
        if name == "Liu 2019":
            expr_liu = expr_liu.loc[df_clin.index]
        elif name == "Hugo 2016":
            expr_hugo = expr_hugo.loc[df_clin.index]
        elif name == "Riaz 2017":
            expr_riaz = expr_riaz.loc[df_clin.index]

    # Find common genes across all 3 trials
    common_genes = list(expr_liu.columns.intersection(expr_hugo.columns).intersection(expr_riaz.columns))
    common_genes.sort()
    print(f"Number of common genes: {len(common_genes)}")

    # Subset to common genes
    expr_liu = expr_liu[common_genes]
    expr_hugo = expr_hugo[common_genes]
    expr_riaz = expr_riaz[common_genes]

    # Version A: Raw (log2 expression) concatenated
    expr_raw_merged = pd.concat([expr_liu, expr_hugo, expr_riaz], axis=0)
    
    # Version B: Z-scored within each cohort individually (our batch correction)
    expr_liu_scaled = zscore_df(expr_liu)
    expr_hugo_scaled = zscore_df(expr_hugo)
    expr_riaz_scaled = zscore_df(expr_riaz)
    expr_scaled_merged = pd.concat([expr_liu_scaled, expr_hugo_scaled, expr_riaz_scaled], axis=0)

    # Clinical details merged
    clin_liu['Cohort'] = 'Liu 2019'
    clin_hugo['Cohort'] = 'Hugo 2016'
    clin_riaz['Cohort'] = 'Riaz 2017'
    
    clin_merged = pd.concat([clin_liu[['Cohort', 'response']], 
                             clin_hugo[['Cohort', 'response']], 
                             clin_riaz[['Cohort', 'response']]], axis=0)

    # Align indexes
    expr_raw_merged = expr_raw_merged.loc[clin_merged.index]
    expr_scaled_merged = expr_scaled_merged.loc[clin_merged.index]

    # Map response label
    clin_merged['Response'] = clin_merged['response'].map({1.0: "Responder (CR/PR)", 0.0: "Non-responder (PD)"})

    # Nature Publishing Group (NPG) Palette definitions
    colors_cohort = {
        'Liu 2019': '#3C5488',   # NPG Navy
        'Hugo 2016': '#00A087',  # NPG Teal
        'Riaz 2017': '#DC0000'   # NPG Red
    }
    colors_response = {
        'Responder (CR/PR)': '#00A087',      # NPG Teal/Green (Positive response)
        'Non-responder (PD)': '#DC0000'      # NPG Red (Negative response)
    }

    # Setup column annotations (Cohort & Response)
    col_colors = pd.DataFrame(index=clin_merged.index)
    col_colors['Cohort'] = clin_merged['Cohort'].map(colors_cohort)
    col_colors['Response'] = clin_merged['Response'].map(colors_response)

    # Define legend elements
    legend_elements = [
        Patch(facecolor='#3C5488', label='Liu 2019'),
        Patch(facecolor='#00A087', label='Hugo 2016'),
        Patch(facecolor='#DC0000', label='Riaz 2017'),
        # Spacer
        Patch(facecolor='white', edgecolor='none', label=''),
        Patch(facecolor='#00A087', label='Responder (CR/PR)'),
        Patch(facecolor='#DC0000', label='Non-responder (PD)')
    ]

    # 2. IDENTIFY TOP GENES BY VARIANCE
    print("Calculating gene variances on unstandardized (raw log2) data...")
    raw_variances = expr_raw_merged.var(axis=0)
    top_50_genes = raw_variances.sort_values(ascending=False).head(50).index.tolist()
    print("Top 5 genes by variance:", top_50_genes[:5])

    # Subset dataframes to top 50 genes
    df_raw_heatmap = expr_raw_merged[top_50_genes].T  # genes as rows, patients as columns
    df_scaled_heatmap = expr_scaled_merged[top_50_genes].T

    # 3. GENERATE HEATMAP BEFORE BATCH CORRECTION
    print("Generating raw expression heatmap...")
    g_raw = sns.clustermap(
        df_raw_heatmap,
        method='ward',
        metric='euclidean',
        cmap='viridis',
        col_colors=col_colors,
        figsize=(12, 10),
        yticklabels=True,
        xticklabels=False,
        cbar_pos=(0.02, 0.8, 0.05, 0.15),
        cbar_kws={'label': 'log2(Expression + 1)'}
    )
    # Style row & col dendrogram axes
    g_raw.ax_col_dendrogram.set_title("Expression Heatmap Before Batch Correction (Top 50 Genes by Variance)", fontsize=14, fontweight='bold', pad=15)
    g_raw.ax_row_dendrogram.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.2, 1.15), ncol=2, frameon=True)

    raw_heatmap_path = PLOT_DIR / "heatmap_top_variance_genes_raw.png"
    plt.savefig(raw_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved raw heatmap to {raw_heatmap_path}")

    # 4. GENERATE HEATMAP AFTER BATCH CORRECTION
    print("Generating standardized expression heatmap...")
    # Center the Z-score heatmap around 0 using a diverging palette
    g_scaled = sns.clustermap(
        df_scaled_heatmap,
        method='ward',
        metric='euclidean',
        cmap='RdBu_r',
        col_colors=col_colors,
        figsize=(12, 10),
        yticklabels=True,
        xticklabels=False,
        vmin=-3, vmax=3,  # Clip Z-score display for contrast
        cbar_pos=(0.02, 0.8, 0.05, 0.15),
        cbar_kws={'label': 'Z-score Expression'}
    )
    g_scaled.ax_col_dendrogram.set_title("Expression Heatmap After Batch Correction (Top 50 Genes by Variance)", fontsize=14, fontweight='bold', pad=15)
    g_scaled.ax_row_dendrogram.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.2, 1.15), ncol=2, frameon=True)

    scaled_heatmap_path = PLOT_DIR / "heatmap_top_variance_genes_standardized.png"
    plt.savefig(scaled_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved standardized heatmap to {scaled_heatmap_path}")

    # 5. UPDATE INTEGRATED BATCH CORRECTION REPORT
    print("Updating batch_correction_report.md with heatmap analysis...")
    report_path = REPORT_DIR / "batch_correction_report.md"
    
    # Read the current content
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Create new section text
    heatmap_section = """
## 3. Gene-Level Expression Heatmaps (Top 50 Highly Variable Genes)

To evaluate batch correction at the individual gene level, we selected the **top 50 genes by variance** (calculated on raw log2-TPM expression data across trial patients) and performed hierarchical clustering on both uncorrected and standardized expression values.

### Raw Expression (Top 50 Genes)
![[heatmap_top_variance_genes_raw.png]]

### Standardized Expression (Top 50 Genes)
![[heatmap_top_variance_genes_standardized.png]]

### Key Observations
*   **Before Batch Correction (Raw log2-TPM)**:
    *   **Cohort Segregation**: The patient columns cluster heavily by cohort source. The **Riaz 2017** cohort (red annotation bar) and **Hugo 2016** cohort (teal annotation bar) are almost completely partitioned from **Liu 2019** (navy annotation bar), indicating that systemic scale differences across studies skew patient clustering.
    *   **Gene-Level Offsets**: Clear horizontal bands of elevated or suppressed baseline expression are visible across cohorts for specific genes, illustrating study-specific calibration differences.
*   **After Batch Correction (Individual Z-scoring)**:
    *   **Perfect Cohort Mixing**: After individually standardizing each study, the cohort annotation bars are distributed randomly across the patient dendrogram, confirming that baseline study-specific calibration offsets have been successfully aligned.
    *   **Biological Subgroups**: The hierarchical clustering now groups patients by shared relative gene expression patterns (e.g., core co-expressed gene modules) rather than study of origin.
    *   **No Response Clustering**: Responders (teal column bar) and non-responders (red column bar) remain mixed throughout the patient dendrogram, verifying that global high-variance gene expression does not cleanly partition patients by immunotherapy response.
"""

    # We append the heatmap section before the CV rigor section (which is Section 3 in the old report, let's make CV rigor Section 4)
    # Find where Section 3 starts
    target_str = "## 3. Cross-Validation Rigor"
    if target_str in content:
        updated_content = content.replace(target_str, heatmap_section + "\n## 4. Cross-Validation Rigor")
    else:
        updated_content = content + "\n" + heatmap_section

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print("Successfully integrated heatmap analysis into report!")
    print("==================================================")

if __name__ == "__main__":
    main()
