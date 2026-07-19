import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency

# Add project root to sys.path for importing src modules
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_liu_2019, load_hugo_2016, load_riaz_2017

# Paths
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots" / "clinical"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

def main():
    print("==================================================")
    print("Generating Response Distribution Visualisation")
    print("==================================================\n")

    # Load cohorts
    _, clin_liu = load_liu_2019(DATA_DIR)
    _, clin_hugo = load_hugo_2016(DATA_DIR)
    _, clin_riaz = load_riaz_2017(DATA_DIR)

    cohorts = {
        "Liu 2019": clin_liu,
        "Hugo 2016": clin_hugo,
        "Riaz 2017": clin_riaz
    }

    data = []
    for name, df in cohorts.items():
        if 'response' in df.columns:
            resp_counts = df['response'].value_counts()
            r_count = resp_counts.get(1.0, 0)
            nr_count = resp_counts.get(0.0, 0)
            total = r_count + nr_count
            data.append({
                "Cohort": name,
                "Responder (CR/PR)": r_count,
                "Non-responder (PD)": nr_count,
                "Total": total,
                "Response Rate (%)": (r_count / total) * 100 if total > 0 else 0
            })

    df_resp = pd.DataFrame(data)
    print(df_resp)

    # Perform Chi-squared test of homogeneity
    contingency_table = df_resp[["Responder (CR/PR)", "Non-responder (PD)"]].values
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-squared test p-value: {p_val:.4f}")

    from src.styles import RESPONSE_PALETTE, set_presentation_style

    set_presentation_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    cohort_names = df_resp["Cohort"].tolist()
    responders = df_resp["Responder (CR/PR)"].values
    non_responders = df_resp["Non-responder (PD)"].values
    totals = df_resp["Total"].values

    r_prop = responders / totals * 100
    nr_prop = non_responders / totals * 100

    c_responder = RESPONSE_PALETTE["CR/PR"]
    c_nonresponder = RESPONSE_PALETTE["PD"]

    bars1 = ax.bar(cohort_names, r_prop, label="Responder (CR/PR)", color=c_responder, width=0.55, edgecolor="black")
    bars2 = ax.bar(cohort_names, nr_prop, bottom=r_prop, label="Non-responder (PD)", color=c_nonresponder, width=0.55, edgecolor="black")

    for i in range(len(cohort_names)):
        ax.text(i, r_prop[i]/2, f"{responders[i]}\n({r_prop[i]:.1f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        ax.text(i, r_prop[i] + nr_prop[i]/2, f"{non_responders[i]}\n({nr_prop[i]:.1f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=10)

    ax.set_ylabel("Proportion of Patients (%)", fontsize=12, fontweight="bold")
    ax.set_title("RECIST Response Distribution across IO Cohorts", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    
    ax.text(0.5, -0.15, f"Chi-squared test of homogeneity: p = {p_val:.4f}", transform=ax.transAxes,
            ha="center", va="center", fontsize=11, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgrey", edgecolor="none", alpha=0.5))

    plt.tight_layout()
    out_path = PLOT_DIR / "response_distribution.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot to {out_path}")

if __name__ == "__main__":
    main()
