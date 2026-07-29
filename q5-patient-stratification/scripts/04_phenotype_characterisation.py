#!/usr/bin/env python3
"""Script 04: Annotate and characterise discovered patient phenotypes (Integrated with Q3 ODE Dynamics).

Profiles discovered clusters across immune signatures, cell deconvolution (M1/M2 ratio), genomic alterations,
and clinical response rates. Integrates Q3 ODE dynamic tumour growth/regression simulations (T(t) trajectories)
parameterised per phenotype.
"""

import contextlib
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
for parent in [SCRIPT_DIR] + list(SCRIPT_DIR.parents):
    if (parent / "src").is_dir() and (parent / "data").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import solve_ivp

from src.styles import PHENOTYPE_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, SUBPROJECT_ROOT, rel_path

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "04_phenotype_characterisation.log"
INPUT_FILE = PROCESSED_DIR / "q5" / "patient_clusters.csv"
OUTPUT_DIR = PROCESSED_DIR / "q5"
PLOTS_DIR = SUBPROJECT_ROOT / "plots" / "phenotypes"

set_presentation_style()

PHENOTYPE_COLORS = {
    0: "#E69F00",  # Mutant-Driven (Okabe-Ito Orange)
    1: "#0072B2",  # Immune Cold Desert (Okabe-Ito Blue)
    2: "#D55E00",  # Immune Hot Inflamed (Crimson Red)
    3: "#CC79A7",  # M2 Immunosuppressive (Okabe-Ito Reddish Purple)
}


def compute_phenotype_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean biomarker features, cell fractions, and clinical response rates per cluster."""
    numeric_cols = [
        c for c in ["TIS", "CYT", "CD8_T_cells", "M1_Macrophages", "M2_Macrophages", "CAFs", "mut_BRAF", "mut_NRAS", "mut_NF1", "RESPONSE_BINARY"]
        if c in df.columns
    ]
    summary = df.groupby("Cluster_ID")[numeric_cols].mean().reset_index()
    summary["Patient_Count"] = df.groupby("Cluster_ID").size().values
    summary["Cohort_Percentage"] = (summary["Patient_Count"] / len(df)) * 100
    return summary


def plot_baseline_boxplots(df: pd.DataFrame, save_path: Path) -> None:
    """Generate 300 DPI publication boxplots comparing core biomarker Z-scores across clusters."""
    features = [c for c in ["TIS", "CYT", "CD8_T_cells", "M1_Macrophages", "M2_Macrophages", "CAFs"] if c in df.columns]
    
    # Standardize features for comparable boxplot Z-scores
    df_plot = df.copy()
    for col in features:
        mean_val = df_plot[col].mean()
        std_val = df_plot[col].std()
        df_plot[col] = (df_plot[col] - mean_val) / (std_val if std_val > 0 else 1.0)
        
    df_melt = pd.melt(
        df_plot,
        id_vars=["Cluster_ID", "Phenotype_Label"],
        value_vars=features,
        var_name="Biomarker",
        value_name="Z_Score"
    )

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)
    palette = [PHENOTYPE_COLORS[i] for i in sorted(PHENOTYPE_COLORS.keys())]

    sns.boxplot(
        data=df_melt,
        x="Biomarker",
        y="Z_Score",
        hue="Phenotype_Label",
        palette=palette,
        ax=ax,
        fliersize=2,
        linewidth=1.2,
    )

    ax.set_title("Biomarker Profile Z-Scores Across Stratified Phenotype Clusters", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Immune Microenvironment & Biomarker Signature", fontsize=12, fontweight="bold")
    ax.set_ylabel("Standardized Z-Score", fontsize=12, fontweight="bold")
    ax.axhline(0, color="#37474F", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.legend(title="Phenotype Subtype", loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved baseline signature boxplot to {save_path}")


def simulate_q3_ode_trajectories(df: pd.DataFrame, save_path: Path) -> None:
    """Simulate Q3 ODE tumor volume trajectories T(t) per Q5 patient phenotype over 180 days."""
    def tumor_immune_ode(t, y, r, K, c, s, p, g, d_E, mu):
        T, E = y
        dTdt = r * T * (1.0 - T / K) - c * E * T
        dEdt = s + (p * E * T) / (g + T) - d_E * E - mu * E * T
        return [dTdt, dEdt]

    t_span = (0, 180)
    t_eval = np.linspace(0, 180, 181)

    # Phenotype ODE parameters [r, K, c, s, p, g, d_E, mu], initial [T0, E0]
    phenotype_params = {
        0: {"name": "Mutant-Driven (NF1/BRAF)", "params": [0.18, 1.0, 0.32, 0.08, 0.12, 0.30, 0.05, 0.03], "y0": [0.8, 0.40], "color": "#E69F00", "ls": "-"},
        1: {"name": "Immune Cold Desert", "params": [0.18, 1.0, 0.15, 0.02, 0.05, 0.30, 0.05, 0.04], "y0": [0.8, 0.15], "color": "#0072B2", "ls": "-"},
        2: {"name": "Immune Hot Inflamed", "params": [0.18, 1.0, 0.45, 0.10, 0.15, 0.30, 0.05, 0.02], "y0": [0.8, 0.80], "color": "#D55E00", "ls": "-"},
        3: {"name": "M2 Immunosuppressive (Anti-PD-1 Monotherapy)", "params": [0.18, 1.0, 0.08, 0.01, 0.03, 0.30, 0.05, 0.05], "y0": [0.8, 0.08], "color": "#CC79A7", "ls": "-"},
        "rescue": {"name": "M2 Immunosuppressive + Combination Rescue (M2-Depleting)", "params": [0.18, 1.0, 0.40, 0.12, 0.15, 0.30, 0.05, 0.02], "y0": [0.8, 0.08], "color": "#CC79A7", "ls": "--"},
    }

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    for key, cfg in phenotype_params.items():
        sol = solve_ivp(tumor_immune_ode, t_span, cfg["y0"], args=tuple(cfg["params"]), t_eval=t_eval)
        lw = 2.5 if key == "rescue" else 2.2
        ax.plot(
            sol.t,
            sol.y[0],
            label=f"{cfg['name']} (T_180 = {sol.y[0][-1]:.2f})",
            color=cfg["color"],
            linestyle=cfg["ls"],
            linewidth=lw,
        )

    ax.set_title("Q3 ODE Dynamic Tumour Burden Trajectories T(t) Across Patient Phenotypes (t = 180 Days)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Time Post-Treatment Initiation (Days)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Relative Tumour Volume T(t) / K", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="#37474F", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.legend(title="Phenotype & Therapy Arm", loc="center right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Q3 ODE trajectory plot to {save_path}")


def main() -> None:
    """Main execution function for Phase 4 phenotype characterisation."""
    print(f"Starting Q5 Phase 4 Phenotype Characterisation (Project root: {rel_path(PROJECT_ROOT)})")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing patient clusters file at {rel_path(INPUT_FILE)}. Run 03_cluster_patients.py first.")

    df_clusters = pd.read_csv(INPUT_FILE)
    print(f"Loaded stratified patient dataset: {len(df_clusters)} patients.")

    # 1. Compute phenotype characterisation profiles
    df_profiles = compute_phenotype_profiles(df_clusters)
    out_profile = OUTPUT_DIR / "phenotype_characterisation.csv"
    df_profiles.to_csv(out_profile, index=False)
    print(f"Saved phenotype characterisation summary to {rel_path(out_profile)}")

    # 2. Generate publication baseline boxplots and Q3 ODE trajectories
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_baseline_boxplots(df_clusters, PLOTS_DIR / "baseline_signature_boxplots.png")
    simulate_q3_ode_trajectories(df_clusters, PLOTS_DIR / "ode_trajectories.png")

    print("=" * 80)
    print("PHASE 4 PHENOTYPE CHARACTERISATION COMPLETE")
    print(f"Summary CSV: {rel_path(out_profile)}")
    print(f"Plots Directory: {rel_path(PLOTS_DIR)}")
    print("=" * 80)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
