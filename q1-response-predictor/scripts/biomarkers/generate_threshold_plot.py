import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PLOT_DIR = BASE_DIR / "plots" / "models"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Data for Default (0.5) vs Optimal (Youden's J) Threshold Performance
data = [
    {"Model": "Random Forest", "Cohort": "Liu 2019 (N=104)", "Strategy": "Default (0.50)", "Accuracy": 0.596, "Sensitivity": 0.188},
    {"Model": "Random Forest", "Cohort": "Liu 2019 (N=104)", "Strategy": "Optimal (Youden's J)", "Accuracy": 0.635, "Sensitivity": 0.625},

    {"Model": "Random Forest", "Cohort": "Hugo 2016 (N=27)", "Strategy": "Default (0.50)", "Accuracy": 0.444, "Sensitivity": 0.286},
    {"Model": "Random Forest", "Cohort": "Hugo 2016 (N=27)", "Strategy": "Optimal (Youden's J)", "Accuracy": 0.556, "Sensitivity": 0.571},

    {"Model": "Random Forest", "Cohort": "Riaz 2017 (N=64)", "Strategy": "Default (0.50)", "Accuracy": 0.609, "Sensitivity": 0.800},
    {"Model": "Random Forest", "Cohort": "Riaz 2017 (N=64)", "Strategy": "Optimal (Youden's J)", "Accuracy": 0.672, "Sensitivity": 0.750},
]

df = pd.DataFrame(data)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
sns.set_theme(style="whitegrid")

# Plot 1: Accuracy Gain
sns.barplot(
    data=df,
    x="Cohort",
    y="Accuracy",
    hue="Strategy",
    palette=["#56B4E9", "#009E73"],
    ax=axes[0],
    edgecolor="white",
    linewidth=1.2
)
axes[0].set_title("Classification Accuracy: Default (0.50) vs. Optimal Threshold", fontsize=12, fontweight="bold", pad=12)
axes[0].set_ylabel("Accuracy", fontsize=11, fontweight="bold")
axes[0].set_ylim(0, 0.85)
axes[0].legend(title="Threshold Strategy", frameon=True)

for p in axes[0].patches:
    h = p.get_height()
    if h > 0:
        axes[0].annotate(f"{h:.1%}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 3), textcoords='offset points')

# Plot 2: Sensitivity Recovery
sns.barplot(
    data=df,
    x="Cohort",
    y="Sensitivity",
    hue="Strategy",
    palette=["#E69F00", "#D55E00"],
    ax=axes[1],
    edgecolor="white",
    linewidth=1.2
)
axes[1].set_title("Sensitivity Recovery: Default (0.50) vs. Optimal Threshold", fontsize=12, fontweight="bold", pad=12)
axes[1].set_ylabel("Sensitivity (True Positive Rate)", fontsize=11, fontweight="bold")
axes[1].set_ylim(0, 1.05)
axes[1].legend(title="Threshold Strategy", frameon=True)

for p in axes[1].patches:
    h = p.get_height()
    if h > 0:
        axes[1].annotate(f"{h:.1%}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 3), textcoords='offset points')

sns.despine(top=True, right=True)
plt.tight_layout()

out_path = PLOT_DIR / "threshold_tuning_impact.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved threshold tuning comparison plot to {out_path}")
