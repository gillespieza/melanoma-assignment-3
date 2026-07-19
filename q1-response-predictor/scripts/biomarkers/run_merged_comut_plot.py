import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from pathlib import Path

# Add project root to sys.path for importing src modules
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_liu_2019, load_hugo_2016, load_riaz_2017

# Paths
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots" / "genomic"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

def load_processed_mutations(mutations_file: Path, target_genes: list, sample_ids: list) -> pd.DataFrame:
    if not mutations_file.exists():
        print(f"Warning: mutation file not found in {mutations_file}")
        return pd.DataFrame(0, index=sample_ids, columns=target_genes)
        
    df_mut = pd.read_csv(mutations_file, index_col="SAMPLE_ID")
    for g in target_genes:
        if g not in df_mut.columns:
            df_mut[g] = 0
            
    df_mut = (df_mut[target_genes] > 0).astype(int)
    return df_mut.reindex(sample_ids, fill_value=0)

def main():
    print("==================================================")
    print("Generating Co-Mutation (Oncoplot) for Merged Trial Cohorts")
    print("==================================================\n")

    # Load Clinical Trial Data
    _, clin_liu = load_liu_2019(DATA_DIR)
    _, clin_hugo = load_hugo_2016(DATA_DIR)
    _, clin_riaz = load_riaz_2017(DATA_DIR)

    # Standardize columns and track data before merging
    clin_liu['Cohort'] = 'Liu 2019'
    clin_hugo['Cohort'] = 'Hugo 2016'
    clin_riaz['Cohort'] = 'Riaz 2017'

    for df in [clin_liu, clin_hugo, clin_riaz]:
        if 'SEX' in df.columns:
            df['SEX'] = df['SEX'].map({'Male': 'Male', 'Female': 'Female', 'M': 'Male', 'F': 'Female'})

    # Target Melanoma Driver Genes
    target_genes = ['BRAF', 'NRAS', 'NF1', 'CDKN2A', 'PTEN', 'KIT', 'TP53', 'JAK1', 'JAK2', 'B2M']

    # Load mutations for each cohort
    print("Loading somatic mutation data per cohort...")
    mut_liu = load_processed_mutations(DATA_DIR / "processed/liu_2019/mutations_cleaned.csv", target_genes, clin_liu.index.tolist())
    mut_hugo = load_processed_mutations(DATA_DIR / "processed/hugo_2016/mutations_cleaned.csv", target_genes, clin_hugo.index.tolist())
    mut_riaz = load_processed_mutations(DATA_DIR / "processed/riaz_2017/mutations_cleaned.csv", target_genes, clin_riaz.index.tolist())

    # Combine clinical metadata
    clin_cols = ['Cohort', 'response', 'TMB_NONSYNONYMOUS', 'SEX', 'patient_id']
    df_clin_merged = pd.concat([
        clin_liu[clin_cols],
        clin_hugo[clin_cols],
        clin_riaz[clin_cols]
    ])

    # Combine mutation data
    df_mut_merged = pd.concat([mut_liu, mut_hugo, mut_riaz])

    # Join clinical and mutation data
    df_merged = df_clin_merged.join(df_mut_merged)
    df_merged = df_merged.dropna(subset=['response']) # Align response
    
    print(f"Total merged clinical trial samples aligned for CoMut plot: {len(df_merged)}")

    # Sort Patients to produce cascading mutation patterns
    # Sort order: BRAF -> NRAS -> NF1 -> CDKN2A -> PTEN -> other genes -> Response -> Cohort
    sort_cols = target_genes + ['response', 'Cohort']
    df_sorted = df_merged.sort_values(by=sort_cols, ascending=False)
    sorted_sample_ids = df_sorted.index.tolist()

    # Prepare Plot Arrays
    n_samples = len(sorted_sample_ids)
    
    # Mutation grid (rows: genes, cols: samples)
    mut_grid = np.zeros((len(target_genes), n_samples))
    for i, g in enumerate(target_genes):
        mut_grid[i, :] = df_sorted[g].values

    tmb_vals = df_sorted['TMB_NONSYNONYMOUS'].fillna(0).values
    response_vals = df_sorted['response'].values
    cohort_vals = df_sorted['Cohort'].map({'Liu 2019': 0, 'Hugo 2016': 1, 'Riaz 2017': 2}).values
    # Map Female -> 0, Male -> 1, NaN -> 2
    sex_vals = df_sorted['SEX'].map({'Female': 0, 'Male': 1}).fillna(2).values

    # Calculate Mutation Frequency for the right panel
    gene_freqs = [(df_merged[g] > 0).mean() * 100 for g in target_genes]

    # Draw Co-Mutation Plot
    sns.set_theme(style="white")
    fig = plt.figure(figsize=(16, 11))

    # GridSpec layout:
    # Row 0: TMB barplot (height = 1.5)
    # Row 1: Central mutation grid & right frequencies (height = 7.0)
    # Row 2: Spacer (height = 0.1)
    # Row 3: Response track (height = 0.35)
    # Row 4: Cohort track (height = 0.35)
    # Row 5: Sex track (height = 0.35)
    gs = gridspec.GridSpec(
        nrows=6, ncols=2,
        width_ratios=[13, 2],
        height_ratios=[1.5, 7.0, 0.1, 0.35, 0.35, 0.35],
        wspace=0.08, hspace=0.12
    )

    color_mut = "#1f77b4" # Blue for mutant
    color_wt = "#f0f0f0"  # Light grey for Wild-type

    # A. Top Subplot: TMB Barplot
    ax_tmb = fig.add_subplot(gs[0, 0])
    ax_tmb.bar(range(n_samples), tmb_vals, color="#737373", width=0.8, edgecolor="none")
    ax_tmb.set_ylabel("TMB (mut/Mb)", fontsize=11, weight='bold')
    ax_tmb.xaxis.set_visible(False)
    ax_tmb.spines['top'].set_visible(False)
    ax_tmb.spines['right'].set_visible(False)
    ax_tmb.spines['bottom'].set_visible(False)

    # B. Central Subplot: Mutation Heatmap Grid
    ax_mut = fig.add_subplot(gs[1, 0])
    cmap_mut = ListedColormap([color_wt, color_mut])
    bounds = [0, 0.5, 5]
    norm = BoundaryNorm(bounds, cmap_mut.N)

    sns.heatmap(
        mut_grid, cmap=cmap_mut, norm=norm, cbar=False,
        linewidths=0.5, linecolor='white', ax=ax_mut,
        yticklabels=target_genes, xticklabels=False
    )
    for label in ax_mut.get_yticklabels():
        label.set_color('black')
        label.set_fontsize(11)
        label.set_fontweight('bold')

    # Draw solid horizontal separator after the top 3 driver genes (BRAF, NRAS, NF1)
    ax_mut.axhline(y=3, color='black', linewidth=1.5, linestyle='-', zorder=10)

    # C. Right Subplot: Gene Frequencies (%)
    ax_freq = fig.add_subplot(gs[1, 1])
    ax_freq.barh(range(len(gene_freqs)), gene_freqs, color=color_mut, height=0.7, edgecolor="none")
    ax_freq.set_ylim(-0.5, len(gene_freqs) - 0.5)
    ax_freq.invert_yaxis()
    ax_freq.set_xlabel("% Mutated", fontsize=10, weight='bold')
    ax_freq.set_xlim(0, max(gene_freqs) + 5)
    ax_freq.yaxis.set_visible(False)
    ax_freq.spines['top'].set_visible(False)
    ax_freq.spines['right'].set_visible(False)
    ax_freq.spines['left'].set_visible(False)

    for i, freq in enumerate(gene_freqs):
        ax_freq.text(freq + 1, i, f"{freq:.1f}%", va='center', fontsize=9, weight='bold')
    ax_freq.axhline(y=2.5, color='black', linewidth=1.5, linestyle='-', zorder=10)

    from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style

    set_presentation_style()

    # D. Bottom Track 1: Response Status
    ax_resp = fig.add_subplot(gs[3, 0])
    cmap_resp = ListedColormap([RESPONSE_PALETTE["PD"], RESPONSE_PALETTE["CR/PR"]]) # Non-responder, Responder
    bounds_resp = [-0.5, 0.5, 1.5]
    norm_resp = BoundaryNorm(bounds_resp, cmap_resp.N)
    sns.heatmap(
        response_vals.reshape(1, -1), cmap=cmap_resp, norm=norm_resp, cbar=False,
        ax=ax_resp, xticklabels=False, yticklabels=["Response"]
    )
    ax_resp.set_yticklabels(["Response"], rotation=0, fontsize=11, weight='bold', color='black')

    # E. Bottom Track 2: Cohort Source
    ax_cohort = fig.add_subplot(gs[4, 0])
    # Colormap: 0 = Liu 2019, 1 = Hugo 2016, 2 = Riaz 2017
    cmap_cohort = ListedColormap([COHORT_PALETTE["Liu 2019"], COHORT_PALETTE["Hugo 2016"], COHORT_PALETTE["Riaz 2017"]])
    bounds_cohort = [-0.5, 0.5, 1.5, 2.5]
    norm_cohort = BoundaryNorm(bounds_cohort, cmap_cohort.N)
    sns.heatmap(
        cohort_vals.reshape(1, -1), cmap=cmap_cohort, norm=norm_cohort, cbar=False,
        ax=ax_cohort, xticklabels=False, yticklabels=["Cohort"]
    )
    ax_cohort.set_yticklabels(["Cohort"], rotation=0, fontsize=11, weight='bold', color='black')

    # F. Bottom Track 3: Sex
    ax_sex = fig.add_subplot(gs[5, 0])
    cmap_sex = ListedColormap(["#f768a1", "#252525", "#e0e0e0"]) # Pink (Female), Dark grey (Male), Light grey (NaN)
    bounds_sex = [-0.5, 0.5, 1.5, 2.5]
    norm_sex = BoundaryNorm(bounds_sex, cmap_sex.N)
    sns.heatmap(
        sex_vals.reshape(1, -1), cmap=cmap_sex, norm=norm_sex, cbar=False,
        ax=ax_sex, xticklabels=False, yticklabels=["Sex"]
    )
    ax_sex.set_yticklabels(["Sex"], rotation=0, fontsize=11, weight='bold', color='black')

    # Align X limits
    for ax in [ax_tmb, ax_mut, ax_resp, ax_cohort, ax_sex]:
        ax.set_xlim(-0.5, n_samples - 0.5)

    # Main Title
    fig.suptitle("Co-Mutation and Clinical Landscape of Checkpoint Blockade Trials (Merged Cohorts)", 
                 fontsize=16, weight='bold', y=0.96)

    # Custom Legend
    ax_legend = fig.add_subplot(gs[3:, 1])
    ax_legend.axis('off')
    
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=color_mut, label='Mutated'),
        mpatches.Patch(color=color_wt, label='Wild-Type'),
        mpatches.Patch(color=RESPONSE_PALETTE["CR/PR"], label='Responder (CR/PR)'),
        mpatches.Patch(color=RESPONSE_PALETTE["PD"], label='Non-responder (PD)'),
        mpatches.Patch(color=COHORT_PALETTE["Liu 2019"], label='Liu 2019'),
        mpatches.Patch(color=COHORT_PALETTE["Hugo 2016"], label='Hugo 2016'),
        mpatches.Patch(color=COHORT_PALETTE["Riaz 2017"], label='Riaz 2017'),
        mpatches.Patch(color="#252525", label='Sex: Male'),
        mpatches.Patch(color="#f768a1", label='Sex: Female'),
        mpatches.Patch(color="#e0e0e0", label='Sex: Unknown')
    ]
    ax_legend.legend(handles=patches, loc='center left', frameon=True, fontsize=9.5)

    out_path = PLOT_DIR / "comut_landscape_merged.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nSaved Merged CoMut plot to {out_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
