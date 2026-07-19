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

def load_tcga_mutations(proc_dir: Path, sample_ids: list) -> pd.DataFrame:
    mut_path = proc_dir / "mutations_cleaned.csv"
    default_df = pd.DataFrame(0, index=sample_ids, columns=["mut_BRAF", "mut_NRAS", "mut_NF1"])
    if not mut_path.exists():
        print(f"Warning: Processed TCGA mutations file not found at {mut_path}")
        return default_df
        
    df_mut = pd.read_csv(mut_path, index_col="SAMPLE_ID")
    res = pd.DataFrame(index=df_mut.index)
    for gene in ["BRAF", "NRAS", "NF1"]:
        if gene in df_mut.columns:
            res[f"mut_{gene}"] = (df_mut[gene] > 0).astype(int)
        else:
            res[f"mut_{gene}"] = 0
            
    return res.reindex(sample_ids, fill_value=0)

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
    proc_tcga_dir = DATA_DIR / "processed" / "skcm_tcga_pan_can_atlas_2018"
    tcga_mut = load_tcga_mutations(proc_tcga_dir, sample_ids_tcga)
    
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

    from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, DRIVER_PALETTE, set_presentation_style

    set_presentation_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(
        data=df_mut_melt,
        x="Gene",
        y="Frequency",
        hue="Cohort",
        palette=COHORT_PALETTE,
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
    print(f"Saved mutation frequencies plot to {out_mut_path}")

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
        palette={"Responder (CR/PR)": RESPONSE_PALETTE["CR/PR"], "Non-responder (PD)": RESPONSE_PALETTE["PD"]},
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
        color=COHORT_PALETTE["TCGA-SKCM"],
        ax=axes[1],
        bins=30,
        edgecolor="black"
    )
    axes[1].axvline(tcga_tmb.median(), color=RESPONSE_PALETTE["PD"], linestyle="--", linewidth=1.5, label=f"Median = {tcga_tmb.median():.2f}")
    axes[1].axvline(10.0, color="#555555", linestyle=":", linewidth=1.5, label="Standard FDA Cutoff = 10.0")
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
    neo_cols = ['TMB_NONSYNONYMOUS', 'SNV_NEOANTIGEN', 'INDEL_NEOANTIGEN', 
                'FUSION_NEOANTIGEN', 'SPLICE_NEOANTIGEN', 'CTA_SELF_NEOANTIGEN']
    
    available_cols = [c for c in neo_cols if c in clin_liu.columns]
    corr_df = clin_liu[available_cols].dropna()

    if len(corr_df) > 0:
        corr_matrix = corr_df.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Spearman Correlation (r_s)'},
                    linewidths=1, linecolor='white', ax=ax, annot_kws={"size": 10, "weight": "bold"})
        ax.set_title("Genomic & Neoantigen Biomarker Spearman Correlation (Liu 2019)", fontsize=13, fontweight="bold", pad=15)
        plt.xticks(rotation=45, ha='right', fontweight='bold')
        plt.yticks(fontweight='bold')
        plt.tight_layout()
        out_corr_path = PLOT_DIR / "biomarker_correlation_heatmap.png"
        plt.savefig(out_corr_path, dpi=300)
        plt.close()
        print(f"Saved biomarker correlation heatmap to {out_corr_path}")

    # =========================================================================
    # 4. TCGA Overall Survival Stratification by Genomic Features
    # =========================================================================
    print("\n4. Generating TCGA survival stratification plots...")
    df_surv = clin_tcga[["OS_MONTHS", "OS_STATUS", "mut_BRAF", "mut_NRAS", "mut_NF1", "TMB_NONSYNONYMOUS"]].dropna().copy()
    df_surv["OS_MONTHS"] = pd.to_numeric(df_surv["OS_MONTHS"], errors="coerce")
    df_surv["OS_STATUS"] = pd.to_numeric(df_surv["OS_STATUS"], errors="coerce")
    df_surv = df_surv[(df_surv["OS_MONTHS"] > 0) & (df_surv["OS_STATUS"].isin([0, 1]))]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

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
    subtypes_map = [
        ("BRAF Mutant", DRIVER_PALETTE["BRAF"]),
        ("NRAS Mutant", DRIVER_PALETTE["NRAS"]),
        ("NF1 Mutant", DRIVER_PALETTE["NF1"]),
        ("Triple Wild-Type", DRIVER_PALETTE["Triple-WT"])
    ]
    
    for subtype, color in subtypes_map:
        mask = df_surv["Genomic_Subtype"] == subtype
        if mask.sum() > 0:
            kmf.fit(df_surv.loc[mask, "OS_MONTHS"], event_observed=df_surv.loc[mask, "OS_STATUS"], 
                    label=f"{subtype} (N={mask.sum()})")
            kmf.plot_survival_function(ax=axes[0], color=color, linewidth=2.5, ci_show=False)
            
    results_mut = multivariate_logrank_test(df_surv["OS_MONTHS"], df_surv["Genomic_Subtype"], df_surv["OS_STATUS"])
    axes[0].text(0.05, 0.08, f"Multivariate Log-rank p = {results_mut.p_value:.4f}", transform=axes[0].transAxes,
                 fontsize=11, fontweight="semibold", bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"))
    axes[0].set_title("TCGA OS: Stratified by Driver Mutation Subtype", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Time (months)", fontsize=11)
    axes[0].set_ylabel("Overall Survival Probability", fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    tcga_tmb_med = df_surv["TMB_NONSYNONYMOUS"].median()
    df_surv["TMB_Group"] = df_surv["TMB_NONSYNONYMOUS"].apply(lambda x: "High TMB" if x >= tcga_tmb_med else "Low TMB")
    
    mask_high = df_surv["TMB_Group"] == "High TMB"
    mask_low = df_surv["TMB_Group"] == "Low TMB"
    
    kmf_h = KaplanMeierFitter()
    kmf_l = KaplanMeierFitter()
    
    kmf_h.fit(df_surv.loc[mask_high, "OS_MONTHS"], event_observed=df_surv.loc[mask_high, "OS_STATUS"], label=f"High TMB (N={mask_high.sum()})")
    kmf_h.plot_survival_function(ax=axes[1], color=RESPONSE_PALETTE["CR/PR"], linewidth=2.5, ci_show=False)
    
    kmf_l.fit(df_surv.loc[mask_low, "OS_MONTHS"], event_observed=df_surv.loc[mask_low, "OS_STATUS"], label=f"Low TMB (N={mask_low.sum()})")
    kmf_l.plot_survival_function(ax=axes[1], color=RESPONSE_PALETTE["PD"], linewidth=2.5, ci_show=False)
    
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
