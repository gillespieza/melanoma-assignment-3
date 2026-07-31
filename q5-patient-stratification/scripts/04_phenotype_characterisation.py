#!/usr/bin/env python3
"""Script 04: Annotate and characterise discovered patient phenotypes (Integrated with Q3 ODE Dynamics).

Profiles discovered clusters across immune signatures, cell deconvolution (M1/M2 ratio), genomic alterations,
and clinical response rates. Integrates Q3 ODE dynamic tumour growth/regression simulations (T(t) trajectories)
parameterised per phenotype.
"""

import contextlib
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SUBPROJECT_ROOT = SCRIPT_DIR.parent
for parent in [SCRIPT_DIR] + list(SCRIPT_DIR.parents):
    if (parent / "src").is_dir() and (parent / "data").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

if str(SUBPROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.integrate import solve_ivp

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    _LIFELINES_AVAILABLE = True
except ImportError:
    _LIFELINES_AVAILABLE = False

from src.styles import PHENOTYPE_PALETTE, get_phenotype_color, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, rel_path
from src.utils.plotting import save_fig
from phenotyping import profile_clusters
from q5_constants import CLUSTERING_FEATURES

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "04_phenotype_characterisation.log"
INPUT_FILE = PROCESSED_DIR / "q5" / "patient_clusters.csv"
OUTPUT_DIR = PROCESSED_DIR / "q5"
PLOTS_DIR = SUBPROJECT_ROOT / "plots" / "phenotypes"

set_presentation_style()


# ---------------------------------------------------------------------------
# Profile Calculation & Palette Helpers
# ---------------------------------------------------------------------------

def compute_phenotype_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean biomarker features, cell fractions, and clinical response rates per cluster."""
    feature_cols = CLUSTERING_FEATURES + ["RESPONSE_BINARY"]
    summary = profile_clusters(df, "Cluster_ID", feature_cols).reset_index()
    return summary


def _build_label_palette(df: pd.DataFrame) -> Dict[str, str]:
    """Build a Phenotype_Label -> hex colour mapping using get_phenotype_color.

    Ensures boxplots and KM survival curves use the exact Okabe-Ito colours.
    """
    return {
        label: get_phenotype_color(label)
        for label in df["Phenotype_Label"].dropna().unique()
    }


# ---------------------------------------------------------------------------
# Plotting & ODE Trajectory Simulation Functions
# ---------------------------------------------------------------------------

def _prepare_boxplot_data(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Helper to standardize features for comparable Z-score boxplots."""
    df_plot = df.copy()
    for col in features:
        mean_val = df_plot[col].mean()
        std_val = df_plot[col].std()
        df_plot[col] = (df_plot[col] - mean_val) / (std_val if std_val > 0 else 1.0)

    return pd.melt(
        df_plot,
        id_vars=["Cluster_ID", "Phenotype_Label"],
        value_vars=features,
        var_name="Biomarker",
        value_name="Z_Score",
    )


def plot_baseline_boxplots(df: pd.DataFrame, save_path: Path) -> None:
    """Generate 300 DPI publication boxplots comparing core biomarker Z-scores across clusters."""
    features = [c for c in ["TIS", "CYT", "CD8_T_cells", "M1_Macrophages", "M2_Macrophages", "CAFs"] if c in df.columns]
    df_melt = _prepare_boxplot_data(df, features)
    label_palette = _build_label_palette(df)

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    sns.boxplot(
        data=df_melt,
        x="Biomarker",
        y="Z_Score",
        hue="Phenotype_Label",
        palette=label_palette,
        ax=ax,
        fliersize=2,
        linewidth=1.2,
    )

    ax.set_title("Biomarker Profile Z-Scores Across Stratified Phenotype Clusters", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Immune Microenvironment & Biomarker Signature", fontsize=12, fontweight="bold")
    ax.set_ylabel("Standardized Z-Score", fontsize=12, fontweight="bold")
    ax.axhline(0, color="#37474F", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.legend(title="Phenotype Subtype", loc="lower right", frameon=True, fontsize=9)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved baseline signature boxplot to {rel_path(save_path)}")


def _tumor_immune_ode(t: float, y: List[float], r: float, K: float, c: float, s: float, p: float, g: float, d_E: float, mu: float) -> List[float]:
    """Q3 Tumour-immune 2-state ODE differential system."""
    T, E = y
    dTdt = r * T * (1.0 - T / K) - c * E * T
    dEdt = s + (p * E * T) / (g + T) - d_E * E - mu * E * T
    return [dTdt, dEdt]


def _get_ode_configs() -> Dict[str, Dict[str, Dict]]:
    """Return phenotype ODE parameter configurations for Immunotherapy and Targeted Therapy arms."""
    immuno = {
        "hot": {"name": "Immune Hot", "params": [0.18, 1.0, 0.45, 0.10, 0.15, 0.30, 0.05, 0.02], "y0": [0.8, 0.80], "color": PHENOTYPE_PALETTE["Immune Hot"], "ls": "-"},
        "cold": {"name": "Immune Cold", "params": [0.18, 1.0, 0.15, 0.02, 0.05, 0.30, 0.05, 0.04], "y0": [0.8, 0.15], "color": PHENOTYPE_PALETTE["Immune Cold"], "ls": "-"},
        "m2": {"name": "M2 Immunosuppressive (Anti-PD-1)", "params": [0.18, 1.0, 0.08, 0.01, 0.03, 0.30, 0.05, 0.05], "y0": [0.8, 0.08], "color": PHENOTYPE_PALETTE["Immunosuppressive M2-High"], "ls": ":"},
        "mut": {"name": "Mutant-Driven", "params": [0.18, 1.0, 0.32, 0.08, 0.12, 0.30, 0.05, 0.03], "y0": [0.8, 0.40], "color": PHENOTYPE_PALETTE["Mutant-Driven"], "ls": "-"},
        "m2_rescue": {"name": "M2 + Combination Rescue", "params": [0.18, 1.0, 0.40, 0.12, 0.15, 0.30, 0.05, 0.02], "y0": [0.8, 0.08], "color": PHENOTYPE_PALETTE["Immunosuppressive M2-High"], "ls": "--"},
    }
    targeted = {
        "mut": {"name": "Mutant-Driven (BRAFi Sensitive)", "params": [0.04, 1.0, 0.35, 0.08, 0.12, 0.30, 0.05, 0.03], "y0": [0.8, 0.40], "color": PHENOTYPE_PALETTE["Mutant-Driven"], "ls": "-"},
        "hot": {"name": "Immune Hot (BRAF-mut Subset)", "params": [0.07, 1.0, 0.40, 0.10, 0.15, 0.30, 0.05, 0.02], "y0": [0.8, 0.80], "color": PHENOTYPE_PALETTE["Immune Hot"], "ls": "-"},
        "cold": {"name": "Immune Cold (WT / Primary Resistant)", "params": [0.18, 1.0, 0.10, 0.02, 0.05, 0.30, 0.05, 0.04], "y0": [0.8, 0.15], "color": PHENOTYPE_PALETTE["Immune Cold"], "ls": ":"},
        "m2": {"name": "M2 Immunosuppressive (WT / Stromal)", "params": [0.15, 1.0, 0.08, 0.01, 0.03, 0.30, 0.05, 0.05], "y0": [0.8, 0.08], "color": PHENOTYPE_PALETTE["Immunosuppressive M2-High"], "ls": ":"},
    }
    return {"Immunotherapy": immuno, "Targeted Therapy": targeted}


def _plot_ode_panel(ax: plt.Axes, configs: Dict[str, Dict], title: str, t_span: Tuple[int, int], t_eval: np.ndarray) -> None:
    """Helper to simulate and plot ODE trajectory curves for a single therapy arm."""
    for key, cfg in configs.items():
        sol = solve_ivp(_tumor_immune_ode, t_span, cfg["y0"], args=tuple(cfg["params"]), t_eval=t_eval)
        lw = 2.5 if "rescue" in key else 2.2
        ax.plot(
            sol.t,
            sol.y[0],
            label=rf"{cfg['name']} ($T_{{180}} = {sol.y[0][-1]:.2f}$)",
            color=cfg["color"],
            linestyle=cfg["ls"],
            linewidth=lw,
        )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time Post-Treatment Initiation (Days)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Relative Tumour Volume T(t) / K", fontsize=10, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color="#37474F", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)


def simulate_q3_ode_trajectories(df: pd.DataFrame, save_path: Path) -> None:
    """Simulate Q3 ODE tumour volume trajectories T(t) per phenotype across Immunotherapy and Targeted Therapy arms."""
    t_span = (0, 180)
    t_eval = np.linspace(0, 180, 181)
    configs = _get_ode_configs()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    _plot_ode_panel(ax1, configs["Immunotherapy"], "A. Immunotherapy (Anti-PD-1 & Combination Rescue)", t_span, t_eval)
    _plot_ode_panel(ax2, configs["Targeted Therapy"], "B. Targeted Therapy (BRAF/MEK Inhibitor Monotherapy)", t_span, t_eval)

    fig.suptitle("Q3 ODE Dynamic Tumour Burden Trajectories T(t) Across Patient Phenotypes & Treatment Arms (t = 180 Days)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved Q3 ODE trajectory plot to {rel_path(save_path)}")


def _plot_single_cluster_km(ax: plt.Axes, grp: pd.DataFrame, pheno_label: str, color: str) -> None:
    """Helper to fit and plot a single cluster Kaplan-Meier curve."""
    median_os = grp["OS_MONTHS"].median()
    kmf = KaplanMeierFitter()
    kmf.fit(
        grp["OS_MONTHS"],
        event_observed=grp["OS_STATUS"],
        label=f"{pheno_label}  (N={len(grp)}, median OS={median_os:.1f} mo)",
    )
    kmf.plot_survival_function(
        ax=ax,
        color=color,
        ci_show=True,
        ci_alpha=0.12,
        linewidth=2.2,
    )


def plot_kaplan_meier_by_phenotype(df: pd.DataFrame, save_path: Path) -> None:
    """Plot Kaplan-Meier overall survival curves stratified by phenotype cluster."""
    if not _LIFELINES_AVAILABLE:
        raise ImportError("lifelines is required for Kaplan-Meier curves. Install with: pip install lifelines")

    required_cols = ["OS_MONTHS", "OS_STATUS", "Cluster_ID", "Phenotype_Label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Skipping KM plot — missing columns: {missing}")
        return

    df_km = df.dropna(subset=["OS_MONTHS", "OS_STATUS"]).copy()
    df_km["OS_STATUS"] = pd.to_numeric(df_km["OS_STATUS"], errors="coerce").fillna(0).astype(int)
    n_valid = len(df_km)

    if n_valid < 10:
        print(f"Skipping KM plot — only {n_valid} patients with OS data (minimum 10 required).")
        return

    lr_result = multivariate_logrank_test(df_km["OS_MONTHS"], df_km["Cluster_ID"], df_km["OS_STATUS"])
    p_val = lr_result.p_value
    p_label = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    label_palette = _build_label_palette(df_km)

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    for cluster_id in sorted(df_km["Cluster_ID"].unique()):
        grp = df_km[df_km["Cluster_ID"] == cluster_id]
        pheno_label = grp["Phenotype_Label"].iloc[0]
        color = label_palette.get(pheno_label, f"C{cluster_id}")
        _plot_single_cluster_km(ax, grp, pheno_label, color)

    ax.set_title(
        f"Kaplan-Meier Overall Survival by Phenotype Cluster\n"
        f"(N={n_valid} patients with OS data, log-rank {p_label})",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Overall Survival (Months)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Survival Probability", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="#37474F", linestyle=":", linewidth=1.0, alpha=0.6, label="Median OS (50%)")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=9)
    ax.grid(True, color="#E5E7EB", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path, dpi=300)
    print(f"Saved Kaplan-Meier survival plot to {rel_path(save_path)}")


# ---------------------------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------------------------

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

    # 3. Kaplan-Meier survival curves stratified by phenotype
    plot_kaplan_meier_by_phenotype(df_clusters, PLOTS_DIR / "km_survival_by_phenotype.png")

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

