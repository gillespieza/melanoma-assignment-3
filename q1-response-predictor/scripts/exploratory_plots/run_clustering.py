import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency
from sklearn.cluster import AgglomerativeClustering

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

# Paths
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

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
    
    # Pool and batch correct
    sig_all = pd.concat([sig_liu, sig_hugo, sig_riaz], axis=0)
    y_all = pd.concat([y_liu, y_hugo, y_riaz], axis=0)
    batches = (['liu'] * len(sig_liu)) + (['hugo'] * len(sig_hugo)) + (['riaz'] * len(sig_riaz))
    
    sig_corrected_arr = Combat().fit_transform(sig_all.values, batches)
    sig_corrected = pd.DataFrame(sig_corrected_arr, index=sig_all.index, columns=sig_all.columns)
    
    print(f"Corrected signatures matrix shape: {sig_corrected.shape}")
    
    print("\n==================================================")
    print("Phase 2: Unsupervised Hierarchical Clustering...")
    print("==================================================")
    
    # 1. Define color maps for column metadata
    cohort_colors_map = {'liu': '#1f77b4', 'hugo': '#ff7f0e', 'riaz': '#2ca02c'}
    col_cohort_colors = pd.Series(batches, index=sig_corrected.index).map(cohort_colors_map)
    
    response_colors_map = {1.0: '#d62728', 0.0: '#9467bd'}
    col_response_colors = y_all.map(response_colors_map)
    
    col_colors = pd.DataFrame({
        'Cohort': col_cohort_colors,
        'Response': col_response_colors
    })
    
    # 2. Plot Clustermap
    sns.set_theme(style="white")
    g = sns.clustermap(
        sig_corrected.T,
        cmap="RdBu_r",
        z_score=0,
        metric="euclidean",
        method="ward",
        col_colors=col_colors,
        figsize=(12, 8),
        cbar_kws={'label': 'Z-score expression'},
        xticklabels=False
    )
    
    g.ax_heatmap.set_xlabel("Patients (N=162)")
    g.ax_heatmap.set_ylabel("Immune Signatures")
    plt.suptitle("Unsupervised Hierarchical Clustering of Patient Immune Signatures\n(Pooled & Batch-Corrected Clinical Cohorts)", y=1.02, fontsize=14, fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Liu 2019'),
        Patch(facecolor='#ff7f0e', label='Hugo 2016'),
        Patch(facecolor='#2ca02c', label='Riaz 2017'),
        Patch(facecolor='#d62728', label='Responder (CR/PR)'),
        Patch(facecolor='#9467bd', label='Non-Responder (PD)')
    ]
    g.ax_col_dendrogram.legend(handles=legend_elements, bbox_to_anchor=(1.45, 1), loc="upper right", title="Metadata Legends")
    
    clustermap_path = PLOT_DIR / "signature_clustermap.png"
    plt.savefig(clustermap_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Clustermap to {clustermap_path}")
    
    print("\n==================================================")
    print("Phase 3: Association of Clusters with Response...")
    print("==================================================")
    
    cluster_model = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
    patient_clusters = cluster_model.fit_predict(sig_corrected)
    
    df_cluster_assoc = pd.DataFrame({
        'Cluster': patient_clusters,
        'Response': y_all
    })
    
    contingency_table = pd.crosstab(df_cluster_assoc['Cluster'], df_cluster_assoc['Response'])
    print("\nPatient Cluster vs. Immunotherapy Response Contingency Table:")
    print(contingency_table)
    
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-Square Test statistics:")
    print(f"  Chi-Square: {chi2:.3f}")
    print(f"  p-value: {p_val:.3e}")
    
    for cluster in [0, 1]:
        cluster_data = df_cluster_assoc[df_cluster_assoc['Cluster'] == cluster]
        resp_rate = cluster_data['Response'].mean() * 100
        print(f"  Cluster {cluster} Response Rate: {resp_rate:.1f}% (N={len(cluster_data)})")
        
    print("==================================================")

if __name__ == "__main__":
    main()
