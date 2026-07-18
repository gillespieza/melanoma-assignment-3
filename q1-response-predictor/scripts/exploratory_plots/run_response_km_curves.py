import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Add project root to sys.path for importing src modules
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_liu_2019, load_hugo_2016, load_riaz_2017

# Paths
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots" / "clinical"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

def plot_cohort_km_by_response(ax, df_clin, cohort_name):
    # normalise survival column names
    time_col = 'os_months' if 'os_months' in df_clin.columns else ('OS_MONTHS' if 'OS_MONTHS' in df_clin.columns else None)
    event_col = 'os_status' if 'os_status' in df_clin.columns else ('OS_STATUS' if 'OS_STATUS' in df_clin.columns else None)
    
    if not time_col or not event_col or 'response' not in df_clin.columns:
        print(f"Skipping {cohort_name}: missing survival or response columns")
        return

    # Clean
    df = df_clin[[time_col, event_col, 'response']].copy()
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
    df[event_col] = pd.to_numeric(df[event_col], errors='coerce')
    df = df.dropna().copy()
    df = df[df[time_col] > 0]

    responders = df[df['response'] == 1.0]
    non_responders = df[df['response'] == 0.0]

    if len(responders) == 0 or len(non_responders) == 0:
        print(f"Skipping {cohort_name}: missing response groups")
        return

    # Fit KM
    kmf_r = KaplanMeierFitter()
    kmf_nr = KaplanMeierFitter()

    kmf_r.fit(responders[time_col], event_observed=responders[event_col])
    kmf_nr.fit(non_responders[time_col], event_observed=non_responders[event_col])

    # Plot
    kmf_r.plot_survival_function(ax=ax, color="#2ca02c", linewidth=2.5, ci_show=True, alpha=0.15, label=f"Responder (N={len(responders)})")
    kmf_nr.plot_survival_function(ax=ax, color="#d62728", linewidth=2.5, ci_show=True, alpha=0.15, label=f"Non-responder (N={len(non_responders)})")

    # Log-rank test
    results = logrank_test(responders[time_col], non_responders[time_col],
                           responders[event_col], non_responders[event_col])
    p_val = results.p_value

    title_suffix = f"\nLog-rank p = {p_val:.4f}" if p_val >= 0.0001 else f"\nLog-rank p < 0.0001"
    ax.set_title(f"{cohort_name}{title_suffix}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (months)", fontsize=11)
    ax.set_ylabel("Survival Probability", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

def main():
    print("==================================================")
    print("Generating Response-Stratified KM Survival Curves")
    print("==================================================\n")

    # Load cohorts
    _, clin_liu = load_liu_2019(DATA_DIR)
    _, clin_hugo = load_hugo_2016(DATA_DIR)
    _, clin_riaz = load_riaz_2017(DATA_DIR)

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    plot_cohort_km_by_response(axes[0], clin_liu, "Liu 2019")
    plot_cohort_km_by_response(axes[1], clin_hugo, "Hugo 2016")
    plot_cohort_km_by_response(axes[2], clin_riaz, "Riaz 2017")

    fig.suptitle("Overall Survival by Immunotherapy Response (RECIST)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = PLOT_DIR / "km_os_by_response.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved response-stratified KM plots to {out_path}")

if __name__ == "__main__":
    main()
