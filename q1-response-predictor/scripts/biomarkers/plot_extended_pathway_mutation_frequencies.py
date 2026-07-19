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

from src.styles import COHORT_PALETTE, set_presentation_style

set_presentation_style()

# Data setup
data = [
    {"Category": "MAPK Drivers", "Gene/Pathway": "BRAF mutation", "Liu 2019 (N=104)": 40.4, "Hugo 2016 (N=27)": 59.3, "Riaz 2017 (N=64)": 40.6, "Pooled Trials (N=195)": 43.1},
    {"Category": "MAPK Drivers", "Gene/Pathway": "NRAS mutation", "Liu 2019 (N=104)": 30.8, "Hugo 2016 (N=27)": 18.5, "Riaz 2017 (N=64)": 17.2, "Pooled Trials (N=195)": 24.6},
    {"Category": "MAPK Drivers", "Gene/Pathway": "NF1 mutation", "Liu 2019 (N=104)": 18.3, "Hugo 2016 (N=27)": 25.9, "Riaz 2017 (N=64)": 3.1, "Pooled Trials (N=195)": 14.4},
    {"Category": "Immune Resistance", "Gene/Pathway": "Antigen Presentation (MHC)", "Liu 2019 (N=104)": 0.0, "Hugo 2016 (N=27)": 0.0, "Riaz 2017 (N=64)": 0.0, "Pooled Trials (N=195)": 0.0},
    {"Category": "Immune Resistance", "Gene/Pathway": "IFN-gamma Signaling", "Liu 2019 (N=104)": 10.6, "Hugo 2016 (N=27)": 18.5, "Riaz 2017 (N=64)": 3.1, "Pooled Trials (N=195)": 9.2},
    {"Category": "Survival & Proliferation", "Gene/Pathway": "Survival & Proliferation Drivers", "Liu 2019 (N=104)": 23.1, "Hugo 2016 (N=27)": 29.6, "Riaz 2017 (N=64)": 10.9, "Pooled Trials (N=195)": 20.0},
]

df = pd.DataFrame(data)

out_dir = BASE_DIR / "plots" / "genomic"
out_dir.mkdir(parents=True, exist_ok=True)

# --- Option 1: Horizontal Grouped Bar Chart ---
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

y_labels = df["Gene/Pathway"].tolist()
y_pos = np.arange(len(y_labels))
bar_width = 0.18

cohorts = ["Liu 2019 (N=104)", "Hugo 2016 (N=27)", "Riaz 2017 (N=64)", "Pooled Trials (N=195)"]
colors = [COHORT_PALETTE[c] for c in cohorts]

for i, cohort in enumerate(cohorts):
    values = df[cohort].values
    offset = (i - 1.5) * bar_width
    rects = ax.barh(y_pos + offset, values, height=bar_width, label=cohort, color=colors[i], edgecolor='white', linewidth=0.8)
    
    # Add direct percentage data labels
    for rect in rects:
        width = rect.get_width()
        if width > 0:
            ax.annotate(f'{width:.1f}%',
                        xy=(width, rect.get_y() + rect.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points",
                        ha='left', va='center', fontsize=9, fontweight='bold' if "Pooled" in cohort else 'normal',
                        color='#222222')
        else:
            ax.annotate('0.0%',
                        xy=(0.5, rect.get_y() + rect.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points",
                        ha='left', va='center', fontsize=9, color='#888888', fontstyle='italic')

ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontweight='bold')
ax.invert_yaxis()  # top-down order
ax.set_xlabel("Mutation Frequency (%)", fontweight='bold')
ax.set_xlim(0, 70)
ax.set_title("Pre-Treatment Somatic Mutation & Pathway Frequencies Across Cohorts", fontweight='bold', pad=15)
ax.legend(title="Cohort", frameon=True, facecolor='white', framealpha=0.9, loc='lower right')

# Add subtle category separation lines & background shading
ax.axhline(2.5, color='gray', linestyle='--', alpha=0.5)
ax.axhline(4.5, color='gray', linestyle='--', alpha=0.5)

# Annotate Categories on left
ax.text(-18, 1.0, "MAPK Drivers", rotation=90, va='center', ha='center', fontweight='bold', color='#333333', fontsize=11)
ax.text(-18, 3.5, "Immune Resistance", rotation=90, va='center', ha='center', fontweight='bold', color='#333333', fontsize=11)
ax.text(-18, 5.0, "Survival Drivers", rotation=90, va='center', ha='center', fontweight='bold', color='#333333', fontsize=11)

plt.tight_layout()
fig_path1 = out_dir / "extended_pathway_grouped_bars.png"
plt.savefig(fig_path1, bbox_inches='tight')
plt.close()

# --- Option 2: Annotated Heatmap Matrix ---
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
heatmap_df = df.set_index("Gene/Pathway")[cohorts]

sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Frequency (%)'}, 
            linewidths=1, linecolor='white', ax=ax, annot_kws={"size": 11, "weight": "bold"})

# Add percentage sign to annotations
for text in ax.texts:
    text.set_text(text.get_text() + "%")

ax.set_title("Pathway Mutation Frequency Heatmap (%)", fontweight='bold', pad=15)
ax.set_ylabel("")
plt.xticks(rotation=15, ha='right', fontweight='bold')
plt.yticks(fontweight='bold')

plt.tight_layout()
fig_path2 = out_dir / "extended_pathway_heatmap.png"
plt.savefig(fig_path2, bbox_inches='tight')
plt.close()

# --- Option 3: Dumbbell / Range Plot ---
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

for idx, row in df.iterrows():
    y = idx
    val_liu = row["Liu 2019 (N=104)"]
    val_hugo = row["Hugo 2016 (N=27)"]
    val_riaz = row["Riaz 2017 (N=64)"]
    val_pooled = row["Pooled Trials (N=195)"]
    
    vals = [val_liu, val_hugo, val_riaz]
    min_v, max_v = min(vals), max(vals)
    
    # Draw horizontal range line across trial cohorts
    ax.hlines(y, min_v, max_v, color='#cccccc', linewidth=4, zorder=1)
    
    # Plot individual cohorts
    ax.scatter(val_liu, y, color='#2b5c8f', s=90, zorder=3, label='Liu 2019' if idx == 0 else "")
    ax.scatter(val_hugo, y, color='#d95f02', s=90, zorder=3, label='Hugo 2016' if idx == 0 else "")
    ax.scatter(val_riaz, y, color='#7570b3', s=90, zorder=3, label='Riaz 2017' if idx == 0 else "")
    
    # Highlight Pooled Trials with a large diamond marker
    ax.scatter(val_pooled, y, color='#e41a1c', marker='D', s=130, zorder=4, label='Pooled Trials (N=195)' if idx == 0 else "")
    
    # Label pooled value
    ax.annotate(f'Pooled: {val_pooled:.1f}%', (val_pooled, y), xytext=(0, 12), textcoords='offset points',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#e41a1c')

ax.set_yticks(np.arange(len(df)))
ax.set_yticklabels(df["Gene/Pathway"], fontweight='bold')
ax.invert_yaxis()
ax.set_xlabel("Mutation Frequency (%)", fontweight='bold')
ax.set_xlim(-2, 65)
ax.set_title("Extended Pathway Mutation Rates: Trial Variation vs. Pooled Benchmark", fontweight='bold', pad=15)
ax.legend(loc='lower right', frameon=True, facecolor='white')

plt.tight_layout()
fig_path3 = out_dir / "extended_pathway_dumbbell.png"
plt.savefig(fig_path3, bbox_inches='tight')
plt.close()

print(f"Successfully generated 3 visualization figures in {out_dir}")
