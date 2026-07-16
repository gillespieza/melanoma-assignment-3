import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from pathlib import Path

# Set matplotlib backend to Agg to avoid GUI errors
import matplotlib
matplotlib.use('Agg')

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True, parents=True)



def main():
    print("==================================================")
    print("Generating Co-Mutation (Oncoplot) for Liu 2019...")
    print("==================================================")
    
    # 1. Load Clinical & Mutation data
    clin_path = DATA_DIR / "processed/liu_2019/clin_cleaned.csv"
    mut_path = DATA_DIR / "raw/liu_2019/data_mutations.txt"
    clin_sample_file = DATA_DIR / "raw/liu_2019/data_clinical_sample.txt"
    
    if not (clin_path.exists() and mut_path.exists() and clin_sample_file.exists()):
        print("Error: Required raw/processed files for Liu 2019 not found.")
        return
        
    df_clin = pd.read_csv(clin_path, index_col=0)
    df_mut = pd.read_csv(mut_path, sep="\t", low_memory=False)
    df_sample_map = pd.read_csv(clin_sample_file, sep="\t", skiprows=4)
    
    # Map mutations to Patient ID
    sample_to_patient = df_sample_map.set_index("SAMPLE_ID")["PATIENT_ID"].to_dict()
    df_mut['patient_id'] = df_mut['Tumor_Sample_Barcode'].map(sample_to_patient)
    
    # Filter for non-silent somatic mutations
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
    
    # Select target genes
    target_genes = ['BRAF', 'NRAS', 'NF1', 'CDKN2A', 'PTEN', 'JAK1', 'JAK2', 'B2M', 'TAP1', 'TAP2']
    # Ensure all target genes are in the columns
    for g in target_genes:
        if g not in df_mut_wide.columns:
            df_mut_wide[g] = 0
            
    df_mut_target = df_mut_wide[target_genes].copy()
    
    # Map back to sample IDs in df_clin
    df_clin_mut = df_clin.copy()
    for g in target_genes:
        df_clin_mut[f'mut_{g}'] = df_clin_mut['PATIENT_ID'].map(df_mut_target[g]).fillna(0).astype(int)
        
    # Filter/align patients with clinical data (excluding any missing response if necessary)
    df_clin_mut = df_clin_mut.dropna(subset=['response'])
    print(f"Aligned patients for CoMut plot: {len(df_clin_mut)}")
    
    # 2. Sort Patients for Cascading Mutation Pattern
    # Primary sort: BRAF -> NRAS -> NF1 -> CDKN2A -> PTEN -> others -> Response
    sort_cols = [f'mut_{g}' for g in target_genes] + ['response']
    df_clin_mut_sorted = df_clin_mut.sort_values(by=sort_cols, ascending=False)
    
    sorted_sample_ids = df_clin_mut_sorted.index.tolist()
    
    # 3. Prepare Plot Arrays
    # Central mutation grid: shape (n_genes, n_patients)
    mut_grid = np.zeros((len(target_genes), len(sorted_sample_ids)))
    for i, g in enumerate(target_genes):
        mut_grid[i, :] = df_clin_mut_sorted[f'mut_{g}'].values
        
    # Top panel: TMB values
    tmb_vals = df_clin_mut_sorted['TMB_NONSYNONYMOUS'].fillna(0).values
    
    # Right panel: Gene mutation frequencies (percentages)
    gene_freqs = [(df_clin_mut[f'mut_{g}'] > 0).mean() * 100 for g in target_genes]
    
    # Bottom tracks: Response, CNA proportion, Sex
    response_vals = df_clin_mut_sorted['response'].values
    cna_vals = df_clin_mut_sorted['CNA_PROP'].fillna(0).values
    sex_vals = df_clin_mut_sorted['SEX'].map({'Male': 1, 'Female': 0}).fillna(-1).values
    
    # ==========================================
    # 4. Draw the Co-Mutation Plot
    # ==========================================
    sns.set_theme(style="white")
    
    fig = plt.figure(figsize=(15, 12))
    # GridSpec layout: 
    # Row 0: TMB barplot (height ratio = 2)
    # Row 1: Central mutation grid & right frequencies (height ratio = 6)
    # Row 2: Space / Margin (height ratio = 0.2)
    # Row 3: Response track (height ratio = 0.4)
    # Row 4: CNA track (height ratio = 0.4)
    # Row 5: Sex track (height ratio = 0.4)
    
    gs = gridspec.GridSpec(
        nrows=6, ncols=2, 
        width_ratios=[12, 2], 
        height_ratios=[2, 6, 0.2, 0.4, 0.4, 0.4],
        wspace=0.08, hspace=0.12
    )
    
    # Color palette
    color_mut = "#3182bd" # Blue for mutated
    color_wt = "#f0f0f0"  # Grey for wild-type
    
    # A. Top Subplot: TMB Barplot
    ax_tmb = fig.add_subplot(gs[0, 0])
    ax_tmb.bar(range(len(tmb_vals)), tmb_vals, color="#737373", width=0.8, edgecolor="none")
    ax_tmb.set_ylabel("TMB (mut/Mb)", fontsize=11, weight='bold')
    ax_tmb.set_xlim(-0.5, len(tmb_vals) - 0.5)
    ax_tmb.xaxis.set_visible(False)
    ax_tmb.spines['top'].set_visible(False)
    ax_tmb.spines['right'].set_visible(False)
    ax_tmb.spines['bottom'].set_visible(False)
    
    # B. Central Subplot: Mutation Heatmap Grid
    ax_mut = fig.add_subplot(gs[1, 0])
    # Custom colormap: 0 = WT (grey), >=1 = Mutated (blue)
    cmap_mut = ListedColormap([color_wt, color_mut])
    bounds = [0, 0.5, 5]
    norm = BoundaryNorm(bounds, cmap_mut.N)
    
    sns.heatmap(
        mut_grid, cmap=cmap_mut, norm=norm, cbar=False, 
        linewidths=0.5, linecolor='white', ax=ax_mut, 
        yticklabels=target_genes, xticklabels=False
    )
    ax_mut.set_yticklabels(target_genes, rotation=0, fontsize=12, weight='bold')
    
    # C. Right Subplot: Gene Frequencies (Percentage Barplot)
    ax_freq = fig.add_subplot(gs[1, 1])
    ax_freq.barh(range(len(gene_freqs)), gene_freqs, color=color_mut, height=0.7, edgecolor="none")
    ax_freq.set_ylim(-0.5, len(gene_freqs) - 0.5)
    ax_freq.invert_yaxis()  # Align with heatmap rows
    ax_freq.set_xlabel("% Mutated", fontsize=10, weight='bold')
    ax_freq.set_xlim(0, max(gene_freqs) + 5)
    ax_freq.yaxis.set_visible(False)
    ax_freq.spines['top'].set_visible(False)
    ax_freq.spines['right'].set_visible(False)
    ax_freq.spines['left'].set_visible(False)
    
    # Add percentage labels to the bar plot
    for i, freq in enumerate(gene_freqs):
        ax_freq.text(freq + 1, i, f"{freq:.1f}%", va='center', fontsize=10, weight='bold')
        
    # D. Bottom Track 1: Response Status
    ax_resp = fig.add_subplot(gs[3, 0])
    # Colormap: 0 = Non-Responder (PD, Red), 1 = Responder (CR/PR, Green)
    cmap_resp = ListedColormap(["#d62728", "#2ca02c"])
    sns.heatmap(
        response_vals.reshape(1, -1), cmap=cmap_resp, cbar=False, 
        ax=ax_resp, xticklabels=False, yticklabels=["Response"]
    )
    ax_resp.set_yticklabels(["Response"], rotation=0, fontsize=11, weight='bold')
    
    # E. Bottom Track 2: CNA Proportion
    ax_cna = fig.add_subplot(gs[4, 0])
    # Continuous blue-to-red scale representing CNA burden
    sns.heatmap(
        cna_vals.reshape(1, -1), cmap="Reds", cbar=False, 
        ax=ax_cna, xticklabels=False, yticklabels=["CNA Prop"]
    )
    ax_cna.set_yticklabels(["CNA Prop"], rotation=0, fontsize=11, weight='bold')
    
    # F. Bottom Track 3: Sex
    ax_sex = fig.add_subplot(gs[5, 0])
    # Colormap: 0 = Female (light pink/grey), 1 = Male (dark grey)
    cmap_sex = ListedColormap(["#f768a1", "#252525"])
    sns.heatmap(
        sex_vals.reshape(1, -1), cmap=cmap_sex, cbar=False, 
        ax=ax_sex, xticklabels=False, yticklabels=["Sex"]
    )
    ax_sex.set_yticklabels(["Sex"], rotation=0, fontsize=11, weight='bold')
    
    # Align the X-axes of clinical tracks with the mutation grid
    for ax in [ax_tmb, ax_mut, ax_resp, ax_cna, ax_sex]:
        ax.set_xlim(-0.5, len(sorted_sample_ids) - 0.5)
        
    # Add Title
    fig.suptitle("Co-Mutation & Clinical Landscape of immunotherapy Response (Liu 2019)", 
                 fontsize=16, weight='bold', y=0.96)
                 
    # 5. Add Custom Legend on the right of Clinical Tracks
    ax_legend = fig.add_subplot(gs[3:, 1])
    ax_legend.axis('off')
    
    # Custom legend patches
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=color_mut, label='Mutated'),
        mpatches.Patch(color=color_wt, label='Wild-Type'),
        mpatches.Patch(color="#2ca02c", label='Responder (CR/PR)'),
        mpatches.Patch(color="#d62728", label='Non-Responder (PD)'),
        mpatches.Patch(color="#252525", label='Sex: Male'),
        mpatches.Patch(color="#f768a1", label='Sex: Female'),
        mpatches.Patch(color="#fc9272", label='High CNA Burden')
    ]
    ax_legend.legend(handles=patches, loc='center left', frameon=True, fontsize=10)
    
    # Save co-mutation landscape
    comut_path = PLOT_DIR / "comut_landscape_liu_2019.png"
    plt.savefig(comut_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved CoMut plot to {comut_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
