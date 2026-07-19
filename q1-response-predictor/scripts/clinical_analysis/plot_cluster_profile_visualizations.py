import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style

set_presentation_style()

# Data from clinical phenotyping Ward's hierarchical clustering
# Cluster 0: Baseline / Low Hypoxia (N=312)
# Cluster 1: High TMB / Immunotherapy (N=112)
# Cluster 2: Stage IV Metastatic (N=24)

CLUSTER_PALETTE = {
    "Cluster 0: Baseline (N=312)": "#0072B2",     # Okabe-Ito Blue
    "Cluster 1: High TMB/IO (N=112)": "#009E73",  # Okabe-Ito Bluish Green
    "Cluster 2: Stage IV (N=24)": "#D55E00"       # Okabe-Ito Vermillion Red
}

out_dir = BASE_DIR / "plots" / "clinical"
out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Figure 1: 2x2 Multi-Panel Bar Dashboard (Presentation Ready)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=300)

clusters = ["Cluster 0: Baseline (N=312)", "Cluster 1: High TMB/IO (N=112)", "Cluster 2: Stage IV (N=24)"]
cluster_colors = [CLUSTER_PALETTE[c] for c in clusters]

# Panel A: Adjuvant Treatment & Stage Profile (%)
df_treat = pd.DataFrame({
    "Cluster": ["Cluster 0\n(Baseline)", "Cluster 1\n(High TMB/IO)", "Cluster 2\n(Stage IV)"],
    "Immunotherapy": [0.0, 63.4, 0.0],
    "Chemotherapy": [0.0, 58.0, 12.5],
    "Stage IV Rate": [0.0, 0.0, 100.0]
})

df_treat_melt = df_treat.melt(id_vars="Cluster", var_name="Metric", value_name="Percentage")
sns.barplot(data=df_treat_melt, x="Metric", y="Percentage", hue="Cluster", 
            palette=[CLUSTER_PALETTE[c] for c in clusters], ax=axes[0, 0], edgecolor="white", linewidth=1.0)
axes[0, 0].set_title("A. Treatment History & Stage IV Rate (%)", fontsize=13, fontweight="bold", pad=10)
axes[0, 0].set_ylabel("Proportion (%)", fontsize=11, fontweight="bold")
axes[0, 0].set_xlabel("")
axes[0, 0].set_ylim(0, 115)
for container in axes[0, 0].containers:
    axes[0, 0].bar_label(container, fmt="%.1f%%", fontsize=9, padding=3)

# Panel B: Genomic Burden (TMB & Aneuploidy)
df_gen = pd.DataFrame({
    "Cluster": ["Cluster 0\n(Baseline)", "Cluster 1\n(High TMB/IO)", "Cluster 2\n(Stage IV)"],
    "TMB (mut/Mb)": [24.9, 31.4, 14.0],
    "Aneuploidy Score": [13.1, 12.9, 11.8]
})
df_gen_melt = df_gen.melt(id_vars="Cluster", var_name="Metric", value_name="Value")
sns.barplot(data=df_gen_melt, x="Metric", y="Value", hue="Cluster", 
            palette=[CLUSTER_PALETTE[c] for c in clusters], ax=axes[0, 1], edgecolor="white", linewidth=1.0)
axes[0, 1].set_title("B. Tumor Genomic Burden Metrics", fontsize=13, fontweight="bold", pad=10)
axes[0, 1].set_ylabel("Score / Mean Mut/Mb", fontsize=11, fontweight="bold")
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylim(0, 40)
for container in axes[0, 1].containers:
    axes[0, 1].bar_label(container, fmt="%.1f", fontsize=9.5, padding=3)

# Panel C: Microenvironmental Hypoxia (Winter Score)
df_hyp = pd.DataFrame({
    "Cluster": ["Cluster 0\n(Baseline)", "Cluster 1\n(High TMB/IO)", "Cluster 2\n(Stage IV)"],
    "Hypoxia Score": [-2.57, -1.61, -2.33]
})
bars_c = axes[1, 0].bar(df_hyp["Cluster"], df_hyp["Hypoxia Score"], color=cluster_colors, width=0.55, edgecolor="white", linewidth=1.0)
axes[1, 0].set_title("C. Microenvironmental Hypoxia (Winter Score)", fontsize=13, fontweight="bold", pad=10)
axes[1, 0].set_ylabel("Winter Hypoxia Score (Mean)", fontsize=11, fontweight="bold")
axes[1, 0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
axes[1, 0].set_ylim(-3.2, 0.5)
for bar in bars_c:
    height = bar.get_height()
    axes[1, 0].annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, -14), textcoords="offset points",
                        ha='center', va='top', fontsize=10, fontweight='bold', color='white')

# Panel D: Median Overall Survival (Months)
df_os = pd.DataFrame({
    "Cluster": ["Cluster 0\n(Baseline)", "Cluster 1\n(High TMB/IO)", "Cluster 2\n(Stage IV)"],
    "Median OS": [93.0, 66.5, 28.1]
})
bars_d = axes[1, 1].bar(df_os["Cluster"], df_os["Median OS"], color=cluster_colors, width=0.55, edgecolor="white", linewidth=1.0)
axes[1, 1].set_title("D. Median Overall Survival (Months)", fontsize=13, fontweight="bold", pad=10)
axes[1, 1].set_ylabel("Months", fontsize=11, fontweight="bold")
axes[1, 1].set_ylim(0, 110)
for bar in bars_d:
    height = bar.get_height()
    axes[1, 1].annotate(f'{height:.1f} mo',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#222222')

fig.suptitle("Patient Phenotype Profiles: Hierarchical Clusters (TCGA-SKCM, N=441)", 
             fontsize=16, fontweight="bold", y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig1_path = out_dir / "cluster_profile_dashboard.png"
plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved Cluster Profile Dashboard to {fig1_path}")

# -------------------------------------------------------------------------
# Figure 2: Radar / Spider Polar Profile Chart (Fingerprint)
# -------------------------------------------------------------------------
categories = ['TMB (scaled)', 'Aneuploidy', 'Hypoxia (milder)', 'Immunotherapy %', 'Chemotherapy %', 'Stage IV %', 'Median OS (scaled)']
N_vars = len(categories)

# Standardize / Min-Max scale values for radar comparison
# Raw data:
# Cluster 0: TMB=24.9, Aneu=13.1, Hypo=-2.57 (milder=-(-2.57)=2.57), IO=0%, Chemo=0%, StgIV=0%, OS=93.0
# Cluster 1: TMB=31.4, Aneu=12.9, Hypo=-1.61 (milder=-(-1.61)=1.61), IO=63.4%, Chemo=58.0%, StgIV=0%, OS=66.5
# Cluster 2: TMB=14.0, Aneu=11.8, Hypo=-2.33 (milder=-(-2.33)=2.33), IO=0%, Chemo=12.5%, StgIV=100.0%, OS=28.1

values_c0 = [24.9/35, 13.1/15, 2.57/3.0, 0.0/100, 0.0/100, 0.0/100, 93.0/100]
values_c1 = [31.4/35, 12.9/15, 1.61/3.0, 63.4/100, 58.0/100, 0.0/100, 66.5/100]
values_c2 = [14.0/35, 11.8/15, 2.33/3.0, 0.0/100, 12.5/100, 100.0/100, 28.1/100]

# Complete circular polygon
angles = [n / float(N_vars) * 2 * np.pi for n in range(N_vars)]
angles += angles[:1]
values_c0 += values_c0[:1]
values_c1 += values_c1[:1]
values_c2 += values_c2[:1]

fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(polar=True), dpi=300)

plt.xticks(angles[:-1], categories, size=10, weight='bold')
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
plt.ylim(0, 1.1)

# Plot each cluster
ax.plot(angles, values_c0, linewidth=2.5, linestyle='solid', label="Cluster 0: Baseline (N=312)", color=CLUSTER_PALETTE["Cluster 0: Baseline (N=312)"])
ax.fill(angles, values_c0, color=CLUSTER_PALETTE["Cluster 0: Baseline (N=312)"], alpha=0.15)

ax.plot(angles, values_c1, linewidth=2.5, linestyle='solid', label="Cluster 1: High TMB/IO (N=112)", color=CLUSTER_PALETTE["Cluster 1: High TMB/IO (N=112)"])
ax.fill(angles, values_c1, color=CLUSTER_PALETTE["Cluster 1: High TMB/IO (N=112)"], alpha=0.15)

ax.plot(angles, values_c2, linewidth=2.5, linestyle='solid', label="Cluster 2: Stage IV (N=24)", color=CLUSTER_PALETTE["Cluster 2: Stage IV (N=24)"])
ax.fill(angles, values_c2, color=CLUSTER_PALETTE["Cluster 2: Stage IV (N=24)"], alpha=0.15)

plt.title("Multi-Dimensional Phenotype Fingerprint (Radar Plot)", size=14, weight='bold', pad=25)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True)

plt.tight_layout()
fig2_path = out_dir / "cluster_profile_radar.png"
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved Cluster Profile Radar Chart to {fig2_path}")

print("==================================================")
