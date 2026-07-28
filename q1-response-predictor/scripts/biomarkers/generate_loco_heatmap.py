import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PLOT_DIR = BASE_DIR / "plots" / "models"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Data parsed from Table 3 in Section 7 of curated_signatures_report.md
data = [
    {"Model": "Logistic Regression", "Liu 2019 (N=104)": 0.609, "Hugo 2016 (N=27)": 0.415, "Riaz 2017 (N=64)": 0.500},
    {"Model": "Random Forest", "Liu 2019 (N=104)": 0.580, "Hugo 2016 (N=27)": 0.423, "Riaz 2017 (N=64)": 0.678},
    {"Model": "XGBoost", "Liu 2019 (N=104)": 0.581, "Hugo 2016 (N=27)": 0.319, "Riaz 2017 (N=64)": 0.618},
    {"Model": "Support Vector Machine", "Liu 2019 (N=104)": 0.617, "Hugo 2016 (N=27)": 0.434, "Riaz 2017 (N=64)": 0.277},
    {"Model": "Elastic-Net", "Liu 2019 (N=104)": 0.616, "Hugo 2016 (N=27)": 0.415, "Riaz 2017 (N=64)": 0.500},
]

df = pd.DataFrame(data).set_index("Model")

# Set up styling
plt.figure(figsize=(10, 5.5), dpi=300)
sns.set_theme(style="white")

ax = sns.heatmap(
    df,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    cbar_kws={"label": "LOCO ROC-AUC Score"},
    linewidths=1.5,
    linecolor="white",
    annot_kws={"size": 13, "weight": "bold"},
    vmin=0.25,
    vmax=0.70,
)

plt.title("Leave-One-Cohort-Out (LOCO) ROC-AUC Performance Across Models & Held-Out Cohorts", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Held-Out Test Cohort", fontsize=11, fontweight="bold", labelpad=10)
plt.ylabel("Model Architecture", fontsize=11, fontweight="bold", labelpad=10)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11, rotation=0)

plt.tight_layout()
out_path = PLOT_DIR / "loco_performance_heatmap.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved LOCO performance heatmap to {out_path}")
