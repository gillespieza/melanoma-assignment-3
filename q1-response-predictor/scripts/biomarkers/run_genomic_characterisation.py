import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

# Add project root to sys.path for importing src modules
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_liu_2019, load_hugo_2016, load_riaz_2017

# Paths
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots" / "genomic"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

def parse_tcga_mutations(raw_dir: Path, sample_ids: list) -> pd.DataFrame:
    mut_path = raw_dir / "data_mutations.txt"
    if not mut_path.exists():
        print(f"Warning: TCGA raw mutations file not found at {mut_path}")
        return pd.DataFrame(index=sample_ids, columns=["mut_BRAF", "mut_NRAS", "mut_NF1"]).fillna(0).astype(int)
    
    print("Parsing raw TCGA mutations data (this may take a few seconds)...")
    chunks = []
    # Read only needed columns to save memory and time
    for chunk in pd.read_csv(mut_path, sep="\t", comment="#", low_memory=False, 
                             usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"], 
                             chunksize=100000):
        filtered = chunk[chunk["Hugo_Symbol"].isin(["BRAF", "NRAS", "NF1"])]
        chunks.append(filtered)
    
    df_mut = pd.concat(chunks, ignore_index=True)
    
    # Filter for non-synonymous variant classifications
    non_syn = ["Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
               "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
               "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site"]
    df_mut = df_mut[df_mut["Variant_Classification"].isin(non_syn)]
    
    # Map barcodes to standard 15-char sample IDs
    df_mut["SAMPLE_ID"] = df_mut["Tumor_Sample_Barcode"].apply(lambda x: x[:15] if isinstance(x, str) else "")
    df_mut = df_mut[df_mut["SAMPLE_ID"].isin(sample_ids)]
    
    pivoted = df_mut.groupby(["SAMPLE_ID", "Hugo_Symbol"]).size().unstack(fill_value=0)
    for gene in ["BRAF", "NRAS", "NF1"]:
        if gene not in pivoted.columns:
            pivoted[gene] = 0
            
    pivoted = (pivoted[["BRAF", "NRAS", "NF1"]] > 0).astype(int)
    pivoted.columns = ["mut_BRAF", "mut_NRAS", "mut_NF1"]
    
    # Reindex to include all sample_ids
    pivoted = pivoted.reindex(sample_ids, fill_value=0)
    return pivoted

def main():
    print("==================================================")
    print("Genomic Characterisation and Visualisation")
    print("==================================================\n")

    # Load clinical trial data (which already has mutations cleaned)
    _, clin_liu = load_liu_2019(DATA_DIR)
    _, clin_hugo = load_hugo_2016(DATA_DIR)
    _, clin_riaz = load_riaz_2017(DATA_DIR)

    # Filter trials to response-aligned cohorts (keep CR/PR/PD; drop SD/MR/NaN)
    RESPONSE_MAP = {
        "Complete Response": 1,
        "Partial Response": 1,
        "Progressive Disease": 0,
        "Stable Disease": np.nan,
        "Mixed Response": np.nan,
    }
    for df in [clin_liu, clin_hugo, clin_riaz]:
        df['temp_resp'] = df['RESPONSE'].map(RESPONSE_MAP)
        df.dropna(subset=['temp_resp'], inplace=True)
        df.drop(columns=['temp_resp'], inplace=True)

    # Load TCGA processed clinical data
    tcga_clin_path = DATA_DIR / "processed" / "skcm_tcga_pan_can_atlas_2018" / "clin_cleaned.csv"
    if not tcga_clin_path.exists():
        print(f"Error: Processed TCGA clinical file not found at {tcga_clin_path}")
        return
    clin_tcga = pd.read_csv(tcga_clin_path)
    
    # Restore standard uppercase SAMPLE_ID index for TCGA-SKCM
    if "sample_id" in clin_tcga.columns:
        clin_tcga = clin_tcga.rename(columns={"sample_id": "SAMPLE_ID"})
    clin_tcga = clin_tcga.set_index("SAMPLE_ID")
    sample_ids_tcga = clin_tcga.index.tolist()

    # Parse TCGA mutations
    raw_tcga_dir = DATA_DIR / "raw" / "skcm_tcga_pan_can_atlas_2018"
    tcga_mut = parse_tcga_mutations(raw_tcga_dir, sample_ids_tcga)
    
    # Join mutations to TCGA clinical df (avoiding duplicates)
    for col in ["mut_BRAF", "mut_NRAS", "mut_NF1"]:
        if col in clin_tcga.columns:
            clin_tcga = clin_tcga.drop(columns=[col])
    clin_tcga = clin_tcga.join(tcga_mut)
    
    # Save the updated TCGA clinical file to include mutations
    clin_tcga.to_csv(DATA_DIR / "processed" / "skcm_tcga_pan_can_atlas_2018" / "clin_cleaned.csv")
    print("Saved updated TCGA clinical file with mutations added.")

    # Align TCGA-SKCM to patient level and common expression samples
    tcga_expr_path = DATA_DIR / "processed" / "skcm_tcga_pan_can_atlas_2018" / "expr_cleaned.csv"
    df_tcga_expr = pd.read_csv(tcga_expr_path, index_col="SAMPLE_ID")
    
    clin_tcga.index = clin_tcga.index.str.upper().str[:12]
    df_tcga_expr.index = df_tcga_expr.index.str.upper().str[:12]
    
    clin_tcga = clin_tcga.groupby(clin_tcga.index).first()
    df_tcga_expr = df_tcga_expr.groupby(df_tcga_expr.index).first()
    
    common_tcga = clin_tcga.index.intersection(df_tcga_expr.index)
    clin_tcga = clin_tcga.loc[common_tcga]
    # Filter to patients with valid survival data (N=427) to align with baseline reference cohort
    clin_tcga = clin_tcga.dropna(subset=["OS_MONTHS", "OS_STATUS"])
    print(f"Aligned TCGA-SKCM cohort: N = {len(clin_tcga)} unique patients with expression and survival data.")

    cohorts = {
        "Liu 2019": clin_liu,
        "Hugo 2016": clin_hugo,
        "Riaz 2017": clin_riaz,
        "TCGA-SKCM": clin_tcga
    }

    # =========================================================================
    # 1. Mutation Frequency Comparison Plot
    # =========================================================================
    print("\n1. Calculating driver mutation frequencies...")
    mut_data = []
    for name, df in cohorts.items():
        n = len(df)
        b_mut = df["mut_BRAF"].sum()
        n_mut = df["mut_NRAS"].sum()
        f_mut = df["mut_NF1"].sum()
        
        # Triple Wild-Type
        t_wt = len(df[(df["mut_BRAF"] == 0) & (df["mut_NRAS"] == 0) & (df["mut_NF1"] == 0)])
        
        mut_data.append({
            "Cohort": name,
            "BRAF": (b_mut / n) * 100,
            "NRAS": (n_mut / n) * 100,
            "NF1": (f_mut / n) * 100,
            "Triple-WT": (t_wt / n) * 100
        })
        print(f"  {name}: BRAF: {b_mut} ({b_mut/n*100:.1f}%), NRAS: {n_mut} ({n_mut/n*100:.1f}%), NF1: {f_mut} ({f_mut/n*100:.1f}%), Triple-WT: {t_wt} ({t_wt/n*100:.1f}%)")

    df_mut_freq = pd.DataFrame(mut_data)
    df_mut_melt = df_mut_freq.melt(id_vars="Cohort", var_name="Gene", value_name="Frequency")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(
        data=df_mut_melt,
        x="Gene",
        y="Frequency",
        hue="Cohort",
        palette="viridis",
        edgecolor="black",
        ax=ax
    )
    
    ax.set_ylabel("Mutation Frequency (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Genomic Subtype / Driver Gene", fontsize=12, fontweight="bold")
    ax.set_title("Driver Mutation Frequencies across Melanoma Cohorts", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    
    # Add labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", label_type="edge", fontsize=9, padding=3)

    plt.tight_layout()
    out_mut_path = PLOT_DIR / "mutation_frequencies.png"
    plt.savefig(out_mut_path, dpi=300)
    plt.close()
    print(f"Saved mutation frequency comparison to {out_mut_path}")

    # =========================================================================
    # 2. TMB Distribution Plot (by Response for trials, and Cohort-wide)
    # =========================================================================
    print("\n2. Generating TMB distributions...")
    # Gather trials into a single df for response comparison
    trial_list = []
    for name in ["Liu 2019", "Hugo 2016", "Riaz 2017"]:
        df = cohorts[name][["TMB_NONSYNONYMOUS", "response"]].dropna().copy()
        df["Cohort"] = name
        trial_list.append(df)
        
    df_trials_tmb = pd.concat(trial_list, ignore_index=True)
    df_trials_tmb["Response"] = df_trials_tmb["response"].map({1.0: "Responder (CR/PR)", 0.0: "Non-responder (PD)"})

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left Panel: Boxplot of TMB by Response in Trials
    sns.boxplot(
        data=df_trials_tmb,
        x="Cohort",
        y="TMB_NONSYNONYMOUS",
        hue="Response",
        palette=["#d62728", "#2ca02c"],
        ax=axes[0],
        fliersize=4
    )
    # Put y-scale on log to handle outliers
    axes[0].set_yscale("log")
    axes[0].set_ylabel("TMB (mutations/Mb, log scale)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Immunotherapy Cohort", fontsize=12, fontweight="bold")
    axes[0].set_title("Pre-treatment TMB by Immunotherapy Response", fontsize=13, fontweight="bold")
    axes[0].legend(loc="upper left")

    # Add Mann-Whitney annotation
    for idx, cohort in enumerate(["Liu 2019", "Hugo 2016", "Riaz 2017"]):
        c_data = df_trials_tmb[df_trials_tmb["Cohort"] == cohort]
        resp = c_data[c_data["response"] == 1.0]["TMB_NONSYNONYMOUS"]
        non_resp = c_data[c_data["response"] == 0.0]["TMB_NONSYNONYMOUS"]
        
        from scipy.stats import mannwhitneyu
        if len(resp) > 0 and len(non_resp) > 0:
            _, p_val = mannwhitneyu(resp, non_resp)
            p_text = f"p = {p_val:.4f}" if p_val >= 0.0001 else "p < 0.0001"
            axes[0].text(idx, axes[0].get_ylim()[1] * 0.3, p_text, ha='center', va='bottom', 
                          color='black', fontweight='semibold', fontsize=10, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Right Panel: TCGA-SKCM TMB Continuous Distribution
    tcga_tmb = clin_tcga["TMB_NONSYNONYMOUS"].dropna()
    sns.histplot(
        tcga_tmb,
        kde=True,
        log_scale=True,
        color="#8c564b",
        ax=axes[1],
        bins=30,
        edgecolor="black"
    )
    axes[1].axvline(tcga_tmb.median(), color="red", linestyle="--", linewidth=1.5, label=f"Median = {tcga_tmb.median():.2f}")
    axes[1].axvline(10.0, color="darkred", linestyle=":", linewidth=1.5, label="Standard FDA Cutoff = 10.0")
    axes[1].set_xlabel("TMB (mutations/Mb, log scale)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Number of Samples", fontsize=12, fontweight="bold")
    axes[1].set_title("TCGA-SKCM Tumor Mutational Burden (TMB) Distribution (N=426)", fontsize=13, fontweight="bold")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    out_tmb_path = PLOT_DIR / "tmb_distribution.png"
    plt.savefig(out_tmb_path, dpi=300)
    plt.close()
    print(f"Saved TMB distributions to {out_tmb_path}")

    # =========================================================================
    # 3. Biomarker Correlation Clustermap / Heatmap
    # =========================================================================
    print("\n3. Generating biomarker correlation matrix...")
    # Select continuous genomic variables from Liu and Hugo
    # (these have detailed neoantigen counts)
    neo_cols = ['TMB_NONSYNONYMOUS', 'SNV_NEOANTIGEN', 'INDEL_NEOANTIGEN', 
                'FUSION_NEOANTIGEN', 'SPLICE_NEOANTIGEN', 'CTA_SELF_NEOANTIGEN']
    
    # Filter available columns
    available_cols = [c for c in neo_cols if c in clin_liu.columns]
    corr_df = clin_liu[available_cols].dropna()
    
    # Calculate Spearman correlation
    corr_matrix = corr_df.corr(method="spearman")
    
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1, vmax=1,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Spearman Correlation of Continuous Genomic Biomarkers (Liu 2019)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    
    out_corr_path = PLOT_DIR / "biomarker_correlation_heatmap.png"
    plt.savefig(out_corr_path, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {out_corr_path}")

    # =========================================================================
    # 4. TCGA Survival Stratification by Genomic Features
    # =========================================================================
    print("\n4. Generating TCGA survival curves stratified by genomic features...")
    # Clean TCGA survival
    df_surv = clin_tcga.dropna(subset=["OS_MONTHS", "OS_STATUS"]).copy()
    df_surv["OS_MONTHS"] = pd.to_numeric(df_surv["OS_MONTHS"], errors="coerce")
    df_surv["OS_STATUS"] = pd.to_numeric(df_surv["OS_STATUS"], errors="coerce")
    df_surv = df_surv[(df_surv["OS_MONTHS"] > 0) & (df_surv["OS_STATUS"].isin([0, 1]))]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Left panel: KM by Mutation Status (BRAF vs NRAS vs NF1 vs Triple-WT)
    # Define mutually exclusive genomic groups
    def map_genomic_groups(row):
        if row["mut_BRAF"] == 1:
            return "BRAF Mutant"
        elif row["mut_NRAS"] == 1:
            return "NRAS Mutant"
        elif row["mut_NF1"] == 1:
            return "NF1 Mutant"
        else:
            return "Triple Wild-Type"
            
    df_surv["Genomic_Subtype"] = df_surv.apply(map_genomic_groups, axis=1)
    
    kmf = KaplanMeierFitter()
    subtypes = ["BRAF Mutant", "NRAS Mutant", "NF1 Mutant", "Triple Wild-Type"]
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f"]
    
    for subtype, color in zip(subtypes, colors):
        mask = df_surv["Genomic_Subtype"] == subtype
        if mask.sum() > 0:
            kmf.fit(df_surv.loc[mask, "OS_MONTHS"], event_observed=df_surv.loc[mask, "OS_STATUS"], 
                    label=f"{subtype} (N={mask.sum()})")
            kmf.plot_survival_function(ax=axes[0], color=color, linewidth=2.5, ci_show=False)
            
    # Multivariate logrank test
    results_mut = multivariate_logrank_test(df_surv["OS_MONTHS"], df_surv["Genomic_Subtype"], df_surv["OS_STATUS"])
    axes[0].text(0.05, 0.08, f"Multivariate Log-rank p = {results_mut.p_value:.4f}", transform=axes[0].transAxes,
                 fontsize=11, fontweight="semibold", bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
    axes[0].set_title("TCGA OS: Stratified by Driver Mutation Subtype", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Time (months)", fontsize=11)
    axes[0].set_ylabel("Overall Survival Probability", fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Right panel: KM by TMB (High vs Low using standard median split)
    tcga_tmb_med = df_surv["TMB_NONSYNONYMOUS"].median()
    df_surv["TMB_Group"] = df_surv["TMB_NONSYNONYMOUS"].apply(lambda x: "High TMB" if x >= tcga_tmb_med else "Low TMB")
    
    mask_high = df_surv["TMB_Group"] == "High TMB"
    mask_low = df_surv["TMB_Group"] == "Low TMB"
    
    kmf_h = KaplanMeierFitter()
    kmf_l = KaplanMeierFitter()
    
    kmf_h.fit(df_surv.loc[mask_high, "OS_MONTHS"], event_observed=df_surv.loc[mask_high, "OS_STATUS"], label=f"High TMB (N={mask_high.sum()})")
    kmf_h.plot_survival_function(ax=axes[1], color="#2ca02c", linewidth=2.5, ci_show=False)
    
    kmf_l.fit(df_surv.loc[mask_low, "OS_MONTHS"], event_observed=df_surv.loc[mask_low, "OS_STATUS"], label=f"Low TMB (N={mask_low.sum()})")
    kmf_l.plot_survival_function(ax=axes[1], color="#d62728", linewidth=2.5, ci_show=False)
    
    results_tmb = logrank_test(df_surv.loc[mask_high, "OS_MONTHS"], df_surv.loc[mask_low, "OS_MONTHS"],
                               df_surv.loc[mask_high, "OS_STATUS"], df_surv.loc[mask_low, "OS_STATUS"])
    axes[1].text(0.05, 0.08, f"Log-rank p = {results_tmb.p_value:.4f}", transform=axes[1].transAxes,
                 fontsize=11, fontweight="semibold", bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
    axes[1].set_title(f"TCGA OS: Stratified by TMB (Median Split = {tcga_tmb_med:.2f})", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Time (months)", fontsize=11)
    axes[1].set_ylabel("Overall Survival Probability", fontsize=11)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="upper right", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("TCGA-SKCM Overall Survival by Genomic Features", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    out_surv_path = PLOT_DIR / "km_genomic_features.png"
    plt.savefig(out_surv_path, dpi=300)
    plt.close()
    print(f"Saved TCGA survival stratification plots to {out_surv_path}")

    print("\n==================================================")
    print("Done! All genomic characterisations generated.")
    print("==================================================")

if __name__ == "__main__":
    main()
