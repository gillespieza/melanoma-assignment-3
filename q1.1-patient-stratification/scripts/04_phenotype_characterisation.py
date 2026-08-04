#!/usr/bin/env python3
"""Script 04: Annotate and characterise discovered patient phenotypes (Integrated with Q3 ODE Dynamics).

Profiles discovered clusters across immune signatures, cell deconvolution (M1/M2 ratio), genomic
alterations, and clinical response rates. Integrates Q3 ODE dynamic tumour growth/regression
simulations (T(t) trajectories) parameterised per patient using Q3 Modules A, B, and D.
"""

import contextlib
import json
from pathlib import Path
import sys
from typing import Dict, List, Literal, Tuple

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

from src.styles import DARK_SLATE_CHARCOAL, get_phenotype_color, set_presentation_style
from src.utils.io import safe_save_csv
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
    """Compute mean biomarker features, cell fractions, and clinical response rates per cluster.

    Args:
        df: Patient clusters DataFrame — must contain Cluster_ID and all CLUSTERING_FEATURES.

    Returns:
        Summary DataFrame with one row per cluster.

    Raises:
        ValueError: If the DataFrame is missing Cluster_ID or any required clustering feature.
    """
    required_cols = {"Cluster_ID"} | set(CLUSTERING_FEATURES)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"compute_phenotype_profiles: DataFrame missing required columns: {sorted(missing)}. "
            "Ensure 03_cluster_patients.py has been run first."
        )
    # Include RESPONSE_BINARY only if available — optional clinical column
    feature_cols = [c for c in CLUSTERING_FEATURES + ["RESPONSE_BINARY"] if c in df.columns]
    return profile_clusters(df, "Cluster_ID", feature_cols).reset_index()


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
    """Standardise features to comparable Z-scores for boxplot display."""
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
    features = [c for c in ["TIS", "CYT", "CD8_T_cells", "M1_Macrophages", "M2_Macrophages", "CAFs"]
                if c in df.columns]
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

    ax.set_title("Biomarker Profile Z-Scores Across Stratified Phenotype Clusters",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Immune Microenvironment & Biomarker Signature", fontsize=12, fontweight="bold")
    ax.set_ylabel("Standardized Z-Score", fontsize=12, fontweight="bold")
    ax.axhline(0, color=DARK_SLATE_CHARCOAL, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.legend(title="Phenotype Subtype", loc="lower right", frameon=True, fontsize=9)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved baseline signature boxplot to {rel_path(save_path)}")


# ---------------------------------------------------------------------------
# Q3 4-Module ODE Digital Twin Per-Patient Simulation
# ---------------------------------------------------------------------------

Q3_SCRIPTS_DIR = PROJECT_ROOT / "q3-ode-model" / "scripts"
Q3_PARAMS_FILE = PROJECT_ROOT / "q3-ode-model" / "data" / "melanoma_params_full.csv"

if str(Q3_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(Q3_SCRIPTS_DIR))

try:
    from phase3_ode_simulation import (
        active_raf_signal, steady_pERK, checkpoint_kill_factor,
        compute_reference_pERK, KH, RAF_TOTAL_0, MEK_TOTAL_0, ERK_TOTAL_0,
        PDCD1_TOTAL_0, PDL1_TOTAL_0, PERK_PROLIF_MIN, PERK_PROLIF_CAP,
        RASGTP_NRAS, RASGTP_BASAL, RASGTP_V600E, _A_REF
    )
    _Q3_ODE_AVAILABLE = True
except ImportError:
    _Q3_ODE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Q3-Parameterised 2-State Kuznetsov Trajectory Simulation
# ---------------------------------------------------------------------------
# Architecture: Q3 Modules A (RAF dimerisation) + B (MAPK cascade) + D
# (PD-1/PD-L1 checkpoint) supply mechanistic, per-patient parameters r and c
# for the 2-state Kuznetsov-de Pillis tumour-immune ODE (Module C slow system).
# This is exactly Q3's QSS design intent: fast modules parameterise the slow
# module. The result is biologically grounded trajectories with clear phenotype
# separation, without the steady-state attractor collapse of the 1-state solver.

# Published 2-state Kuznetsov-de Pillis baseline parameters (day^-1 units).
# r is overridden per-patient by Q3 pERK/pERK_ref coupling (Module A→B).
# c is gated per-patient by Q3 Module D f_kill (checkpoint occupancy) & CYT.
# E_0 is derived from raw infiltration biomarkers normalised to [0,1].
_KUZ_K      = 1.0    # carrying capacity (normalised tumour volume)
_KUZ_C      = 0.38   # baseline killing coefficient — Q3 f_kill gates per patient
_KUZ_S      = 0.002  # baseline effector source rate (day^-1)
_KUZ_G      = 0.30   # half-saturation constant
_KUZ_D_E    = 0.04   # effector death rate (day^-1)
_KUZ_MU     = 0.04   # effector inhibition by tumour (day^-1)
_KUZ_R_BASE = 0.16   # baseline tumour growth rate (day^-1); pERK ratio modulates per-patient
_INFIL_REF  = 1.5    # normalisation reference for (CD8A+PRF1+GZMA)/3 infiltration score

# ODE treatment arms — single source of truth for arm string identifiers
_ODE_ARMS: Tuple[str, ...] = ("immuno_mono", "immuno_rescue", "targeted")

# Anti-PD-1 dose used in Module D checkpoint kill factor calculation
# (nM; within clinical pembrolizumab/nivolumab trough concentration range)
_ANTI_PD1_DOSE_NM: float = 250.0
# M2-rescue combination killing boost — empirical factor for M2-depletion + PD-1 synergy
_M2_RESCUE_KILL_BOOST: float = 1.3

# BRAFi (Vemurafenib 500 nM) arm sensitivity multipliers per driver mutation class.
# Rationale: BRAF V600E is the direct target (strong response, Bollag et al. 2010);
# NRAS mutations cause paradoxical RAF dimerisation re-activation on BRAFi
# (Hatzivassiliou et al. 2010, Heidorn et al. 2010); WT BRAF retains partial
# sensitivity via downstream MEK phosphorylation.
_BRAFO_BRAF_V600E_SENS: float = 0.15   # BRAF V600E: strong direct inhibition
_BRAFO_NRAS_SENS:       float = 1.05   # NRAS: paradoxical ERK re-activation on BRAFi
_BRAFO_WT_SENS:         float = 0.85   # WT BRAF: partial sensitivity via MEK

# Phenotype-level growth multipliers for the Targeted (BRAFi) arm.
# NF1-loss: RAS hyperactivation renders the tumour partially BRAFi-independent
#   (0.68× — reduced sensitivity; RAS bypass maintains some ERK output off-target).
# M2-High: immunosuppressive stroma maintains proliferative signalling on BRAFi
#   (1.25× — stromal CAF/M2 paracrine loops sustain tumour growth under drug).
_BRAFO_NF1_GROWTH_MULT: float = 0.68
_BRAFO_M2_GROWTH_MULT:  float = 1.25

# Per-phenotype effector density parameters: (scale_factor for E_0, p_rate day^-1).
# Calibrated so Immune Hot achieves near-complete clearance, M2-High monotherapy
# fails (uncontrolled growth), and M2-rescue combination achieves effective regression.
_EFFECTOR_PARAMS: Dict[str, Tuple[float, float]] = {
    "rescue":  (0.85, 0.100),  # M2-rescue: partial effector recovery via macrophage depletion
    "cold":    (0.20, 0.003),  # Immune Cold: T-cell desert, absent infiltration, minimal expansion
    "m2":      (0.25, 0.015),  # M2-High: macrophage-mediated T-cell exclusion suppresses expansion
    "mutant":  (0.55, 0.065),  # Mutant-Driven: moderate infiltration boosted by high neoantigen load
    "default": (1.00, 0.140),  # Immune Hot: fully inflamed, maximal cytotoxic effector expansion
}


def _kuznetsov_ode(t: float, y: List[float], r: float, c: float, p_rate: float) -> List[float]:
    """2-state Kuznetsov-de Pillis tumour-immune ODE parameterised by Q3 per-patient r, c, and p_rate."""
    T, E = y
    T = max(T, 0.0)
    E = max(E, 0.0)
    dTdt = r * T * (1.0 - T / _KUZ_K) - c * E * T
    dEdt = _KUZ_S + (p_rate * E * T) / (_KUZ_G + T) - _KUZ_D_E * E - _KUZ_MU * E * T
    return [dTdt, dEdt]


# ---------------------------------------------------------------------------
# Q3 Parameter Derivation Helpers
# ---------------------------------------------------------------------------

def _compute_mapk_pools(row: pd.Series) -> Tuple[float, float, float]:
    """Extract gene-expression-scaled BRAF/MEK/ERK protein pool totals for Q3 Module A→B simulation.

    Scales each published baseline pool total by the patient's relative gene expression
    (defaulting to 1.0 for missing values, i.e. no change from baseline).

    Returns:
        Tuple of (RAF_T, MEK_T, ERK_T) protein totals.
    """
    raf_t = RAF_TOTAL_0 * float(row.get("BRAF", 1.0))
    mek_t = MEK_TOTAL_0 * 0.5 * (float(row.get("MAP2K1", 1.0)) + float(row.get("MAP2K2", 1.0)))
    erk_t = ERK_TOTAL_0 * 0.5 * (float(row.get("MAPK1", 1.0)) + float(row.get("MAPK3", 1.0)))
    return raf_t, mek_t, erk_t


def _apply_arm_sensitivity(
    arm: str,
    braf_v600e: bool,
    nras_mut: bool,
    is_m2: bool,
    is_mut: bool,
) -> float:
    """Return combined arm sensitivity × phenotype growth multiplier for the Targeted BRAFi arm.

    For non-targeted arms returns 1.0 (no modification). For the Targeted arm, applies the
    driver-mutation sensitivity factor (see _BRAFO_* constants) and phenotype growth multiplier.
    """
    if arm != "targeted":
        return 1.0
    if braf_v600e:
        sens = _BRAFO_BRAF_V600E_SENS
    elif nras_mut:
        sens = _BRAFO_NRAS_SENS
    else:
        sens = _BRAFO_WT_SENS
    pheno_mult = _BRAFO_NF1_GROWTH_MULT if is_mut else (_BRAFO_M2_GROWTH_MULT if is_m2 else 1.0)
    return sens * pheno_mult


def _derive_tumour_growth_rate(
    row: pd.Series,
    arm: str,
    pERK_ref: float,
    rasgtp: float,
    v600e_frac: float,
    braf_v600e: bool,
    nras_mut: bool,
    is_m2: bool,
    is_mut: bool,
) -> float:
    """Derive tumour growth rate r (day^-1) driven by Q3 Module A→B pERK/pERK_ref coupling."""
    raf_t, mek_t, erk_t = _compute_mapk_pools(row)
    drug_nM = 500.0 if arm == "targeted" else 0.0
    A = active_raf_signal(drug_nM, raf_t, rasgtp, v600e_frac)
    V1_eff = KH["V1"] * A / _A_REF
    pERK = steady_pERK(V1_eff, raf_t, mek_t, erk_t)
    perk_ratio = np.clip(pERK / max(pERK_ref, 1e-9), PERK_PROLIF_MIN, PERK_PROLIF_CAP)
    arm_mult = _apply_arm_sensitivity(arm, braf_v600e, nras_mut, is_m2, is_mut)
    return float(_KUZ_R_BASE * perk_ratio * arm_mult)


def _derive_killing_rate(arm: str, pdcd1_pool: float, pdl1_pool: float, cyt: float) -> float:
    """Derive effective killing coefficient c (day^-1) gated by Q3 Module D checkpoint occupancy."""
    cyt_clipped = float(np.clip(1.0 + cyt, 0.3, 1.8))
    if arm == "immuno_rescue":
        # M2-rescue combination: boost f_kill by depletion synergy factor, cap at 1.0
        f_kill = min(1.0, checkpoint_kill_factor(_ANTI_PD1_DOSE_NM, pdcd1_pool, pdl1_pool) * _M2_RESCUE_KILL_BOOST)
        return float(_KUZ_C * f_kill * cyt_clipped)
    if arm == "immuno_mono":
        f_kill = checkpoint_kill_factor(_ANTI_PD1_DOSE_NM, pdcd1_pool, pdl1_pool)
        return float(_KUZ_C * f_kill * cyt_clipped)
    # Targeted arm: no checkpoint blockade — baseline occupancy only
    f_kill = checkpoint_kill_factor(0.0, pdcd1_pool, pdl1_pool)
    return float(_KUZ_C * f_kill)


def _derive_effector_params(
    arm: str, is_cold: bool, is_m2: bool, is_mut: bool, infil: float
) -> Tuple[float, float]:
    """Derive initial effector density E_0 and expansion rate p_rate by microenvironment phenotype.

    Uses the _EFFECTOR_PARAMS lookup dict to avoid a complex conditional chain and to make
    the biological parameterisation explicit and modifiable in one place.
    """
    if arm == "immuno_rescue":
        pheno_key = "rescue"
    elif is_cold:
        pheno_key = "cold"
    elif is_m2:
        pheno_key = "m2"
    elif is_mut:
        pheno_key = "mutant"
    else:
        pheno_key = "default"
    sf, p_rate = _EFFECTOR_PARAMS[pheno_key]
    E_0 = float(np.clip(max(infil, 0.0) / _INFIL_REF * sf, 0.02, 0.95))
    return E_0, p_rate


def _derive_q3_patient_params(
    row: pd.Series, arm: str, pERK_ref: float, pheno_label: str
) -> Tuple[float, float, float, float]:
    """Derive per-patient r, c, E_0, p_rate from Q3 Modules A, B, D and patient biomarkers."""
    braf_v600e = int(row.get("BRAF_MUT", 0)) == 1
    nras_mut = int(row.get("NRAS_MUT", 0)) == 1
    rasgtp = RASGTP_V600E if braf_v600e else (RASGTP_NRAS if nras_mut else RASGTP_BASAL)
    v600e_frac = 1.0 if braf_v600e else 0.0

    infil = (float(row.get("CD8A", 1.0)) + float(row.get("PRF1", 1.0)) + float(row.get("GZMA", 1.0))) / 3.0
    pdcd1_pool = PDCD1_TOTAL_0 * float(row.get("PDCD1", 1.0))
    pdl1_pool = PDL1_TOTAL_0 * float(row.get("CD274", 1.0))
    cyt = float(row.get("CYT", 0.0))

    is_cold = "Cold" in pheno_label
    is_m2 = "M2-High" in pheno_label or "M2" in pheno_label
    is_mut = "Mutant" in pheno_label

    r = _derive_tumour_growth_rate(row, arm, pERK_ref, rasgtp, v600e_frac, braf_v600e, nras_mut, is_m2, is_mut)
    c = _derive_killing_rate(arm, pdcd1_pool, pdl1_pool, cyt)
    E_0, p_rate = _derive_effector_params(arm, is_cold, is_m2, is_mut, infil)

    return r, c, E_0, p_rate


def _simulate_patient_trajectory(
    row: pd.Series, t_eval: np.ndarray, arm: str, pERK_ref: float, pheno_label: str
) -> np.ndarray:
    """Simulate a single patient's 180-day relative tumour volume trajectory T(t)/K starting at T(0)=1.0."""
    r, c, E_0, p_rate = _derive_q3_patient_params(row, arm, pERK_ref, pheno_label)
    y0 = [1.0, E_0]
    sol = solve_ivp(_kuznetsov_ode, (t_eval[0], t_eval[-1]), y0, args=(r, c, p_rate),
                    t_eval=t_eval, method="RK45", rtol=1e-5, atol=1e-7)
    return np.clip(sol.y[0], 0.0, 1.5)


def _compute_phenotype_trajectories(
    sub: pd.DataFrame,
    t_eval: np.ndarray,
    arm: str,
    pERK_ref: float,
    pheno_label: str,
) -> np.ndarray:
    """Simulate and return per-patient trajectories for one phenotype and treatment arm.

    This is the single canonical trajectory computation helper. All callers — panel plotting
    and JSON export — use this function, eliminating redundant ODE re-simulation.

    Args:
        sub: DataFrame subset of patients belonging to this phenotype.
        t_eval: Time evaluation points (days), shape (N_timepoints,).
        arm: Treatment arm identifier — one of _ODE_ARMS.
        pERK_ref: Reference pERK level from wild-type Q3 calibration.
        pheno_label: Full phenotype label string used for parameter derivation.

    Returns:
        Array of shape (N_patients, N_timepoints) containing relative tumour volumes T(t)/K.
    """
    return np.array([
        _simulate_patient_trajectory(row, t_eval, arm, pERK_ref, pheno_label)
        for _, row in sub.iterrows()
    ])


def _clean_short_label(full_label: str) -> str:
    """Extract a concise cluster name for plot legend readability."""
    if "Immune Hot" in full_label:
        return "Immune Hot"
    elif "Immune Cold" in full_label:
        return "Immune Cold"
    elif "M2-High" in full_label:
        return "M2-High"
    elif "Mutant-Driven" in full_label:
        return "Mutant-Driven"
    return full_label.split(" (")[0]


# ---------------------------------------------------------------------------
# ODE Panel Plotting (consume pre-computed trajectory cache)
# ---------------------------------------------------------------------------

def _plot_immunotherapy_ode_panel(
    ax1: plt.Axes,
    phenotypes: List[str],
    label_palette: Dict[str, str],
    t_eval: np.ndarray,
    all_trajs: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """Render Panel A: Immunotherapy (Anti-PD-1 monotherapy & Combination Rescue) trajectories.

    Args:
        all_trajs: Pre-computed cache — Dict[pheno_label -> Dict[arm -> (N_patients, N_t) array]].
    """
    for label in phenotypes:
        pheno_trajs = all_trajs.get(label)
        if pheno_trajs is None:
            continue
        color = label_palette.get(label, DARK_SLATE_CHARCOAL)
        n_pts = pheno_trajs["immuno_mono"].shape[0]
        short_lbl = _clean_short_label(label)
        is_m2 = "M2-High" in label or "M2" in label

        mean_mono = pheno_trajs["immuno_mono"].mean(axis=0)
        ls_mono = ":" if is_m2 else "-"
        lbl_mono = (
            f"{short_lbl} [Anti-PD-1 mono] (N={n_pts}, T={mean_mono[-1]:.2f})"
            if is_m2
            else f"{short_lbl} (N={n_pts}, T={mean_mono[-1]:.2f})"
        )
        ax1.plot(t_eval, mean_mono, label=lbl_mono, color=color, linestyle=ls_mono, linewidth=2.2)

        if is_m2:
            mean_rescue = pheno_trajs["immuno_rescue"].mean(axis=0)
            ax1.plot(
                t_eval, mean_rescue,
                label=f"{short_lbl} [M2 Rescue] (N={n_pts}, T={mean_rescue[-1]:.2f})",
                color=color, linestyle="--", linewidth=2.5,
            )

    ax1.set_title("A. Immunotherapy (Anti-PD-1 & Combination Rescue)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Time Post-Treatment Initiation (Days)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Relative Tumour Volume T(t) / K", fontsize=10, fontweight="bold")
    ax1.set_ylim(-0.02, 1.08)
    ax1.axhline(0, color=DARK_SLATE_CHARCOAL, linestyle=":", linewidth=1.0, alpha=0.7)
    ax1.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax1.grid(True, color="#E5E7EB", linewidth=0.5, alpha=0.6)


def _plot_targeted_ode_panel(
    ax2: plt.Axes,
    phenotypes: List[str],
    label_palette: Dict[str, str],
    t_eval: np.ndarray,
    all_trajs: Dict[str, Dict[str, np.ndarray]],
) -> None:
    """Render Panel B: Targeted Therapy (BRAF Inhibitor Vemurafenib 500 nM) trajectories.

    Args:
        all_trajs: Pre-computed cache — Dict[pheno_label -> Dict[arm -> (N_patients, N_t) array]].
    """
    for label in phenotypes:
        pheno_trajs = all_trajs.get(label)
        if pheno_trajs is None:
            continue
        color = label_palette.get(label, DARK_SLATE_CHARCOAL)
        n_pts = pheno_trajs["targeted"].shape[0]
        short_lbl = _clean_short_label(label)
        mean_targ = pheno_trajs["targeted"].mean(axis=0)

        ax2.plot(
            t_eval, mean_targ,
            label=f"{short_lbl} (N={n_pts}, T={mean_targ[-1]:.2f})",
            color=color, linestyle="-", linewidth=2.2,
        )

    ax2.set_title("B. Targeted Therapy (BRAF Inhibitor Vemurafenib 500 nM)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Time Post-Treatment Initiation (Days)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Relative Tumour Volume T(t) / K", fontsize=10, fontweight="bold")
    ax2.set_ylim(-0.02, 1.08)
    ax2.axhline(0, color=DARK_SLATE_CHARCOAL, linestyle=":", linewidth=1.0, alpha=0.7)
    ax2.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax2.grid(True, color="#E5E7EB", linewidth=0.5, alpha=0.6)


# ---------------------------------------------------------------------------
# ODE Trajectory Export & Orchestration
# ---------------------------------------------------------------------------

def _export_ode_trajectory_summary(
    all_trajs: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    """Persist final T(180) values per arm per phenotype to a JSON summary file.

    Args:
        all_trajs: Pre-computed trajectory cache — Dict[pheno_label -> Dict[arm -> (N, T) array]].
            Uses the pre-computed cache from simulate_q3_ode_trajectories — no re-simulation.
        output_dir: Directory to write ode_trajectory_summary.json into.

    Saves ode_trajectory_summary.json so generate_q5_report.py can read final T(180) values
    dynamically rather than hardcoding them.
    """
    summary: Dict[str, Dict[str, float]] = {
        _clean_short_label(label): {
            arm: float(round(float(np.mean(trajs[:, -1])), 3))
            for arm, trajs in arm_trajs.items()
        }
        for label, arm_trajs in all_trajs.items()
    }

    out_path = output_dir / "ode_trajectory_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved ODE trajectory summary to {rel_path(out_path)}")


def simulate_q3_ode_trajectories(df: pd.DataFrame, save_path: Path, output_dir: Path) -> None:
    """Plot Q3-parameterised 2-state Kuznetsov trajectories aggregated by phenotype.

    Pre-computes all per-patient trajectories once for all arms, then passes the cache to
    panel plotting and JSON export — eliminating redundant ODE re-simulation.

    Args:
        df: Patient clusters DataFrame with Phenotype_Label and Q3-matched expression columns.
        save_path: Output path for the dual-panel trajectory figure (PNG).
        output_dir: Directory for the ode_trajectory_summary.json export.
    """
    if not _Q3_ODE_AVAILABLE or not Q3_PARAMS_FILE.exists():
        print(f"Skipping Q3 ODE trajectory simulation — Q3 module or {Q3_PARAMS_FILE} unavailable.")
        return

    df_params = pd.read_csv(Q3_PARAMS_FILE)
    df_merged = pd.merge(df, df_params, on="SAMPLE_ID", suffixes=("", "_params"))
    if df_merged.empty:
        print("Warning: No matching sample IDs between clusters and Q3 params.")
        df_merged = df.copy()

    pERK_ref = compute_reference_pERK()
    t_eval = np.linspace(0, 180, 361)
    phenotypes = sorted(df["Phenotype_Label"].unique())
    label_palette = _build_label_palette(df)

    # Pre-compute all trajectories once: Dict[pheno_label -> Dict[arm -> (N_patients, N_t) array]]
    # Panel functions and the JSON export all consume this cache — no re-simulation downstream.
    all_trajs: Dict[str, Dict[str, np.ndarray]] = {
        label: {
            arm: _compute_phenotype_trajectories(
                df_merged[df_merged["Phenotype_Label"] == label], t_eval, arm, pERK_ref, label
            )
            for arm in _ODE_ARMS
        }
        for label in phenotypes
        if not df_merged[df_merged["Phenotype_Label"] == label].empty
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), dpi=300)
    _plot_immunotherapy_ode_panel(ax1, phenotypes, label_palette, t_eval, all_trajs)
    _plot_targeted_ode_panel(ax2, phenotypes, label_palette, t_eval, all_trajs)

    fig.suptitle(
        "Q3-Parameterised Tumour-Immune ODE Trajectories T(t) by Phenotype & Treatment Arm\n"
        "(Per-patient r from Q3 pERK coupling, c from Q3 checkpoint f_kill — Mean across matched patients)",
        fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_path)
    print(f"Saved Q3-parameterised ODE trajectory plot to {rel_path(save_path)}")

    # Export T(180) values from the pre-computed cache — no re-simulation needed
    _export_ode_trajectory_summary(all_trajs, output_dir)


# ---------------------------------------------------------------------------
# Kaplan-Meier Survival by Phenotype
# ---------------------------------------------------------------------------

def _plot_single_cluster_km(ax: plt.Axes, grp: pd.DataFrame, pheno_label: str, color: str) -> None:
    """Fit and plot a single cluster Kaplan-Meier curve onto the provided axes."""
    median_os = grp["OS_MONTHS"].median()
    kmf = KaplanMeierFitter()
    kmf.fit(
        grp["OS_MONTHS"],
        event_observed=grp["OS_STATUS"],
        label=f"{pheno_label}  (N={len(grp)}, median OS={median_os:.1f} mo)",
    )
    kmf.plot_survival_function(ax=ax, color=color, ci_show=True, ci_alpha=0.12, linewidth=2.2)


def plot_kaplan_meier_by_phenotype(df: pd.DataFrame, save_path: Path) -> None:
    """Plot Kaplan-Meier overall survival curves stratified by phenotype cluster.

    Raises:
        ImportError: If lifelines is not installed.
    """
    if not _LIFELINES_AVAILABLE:
        raise ImportError(
            "lifelines is required for Kaplan-Meier curves. Install with: pip install lifelines"
        )

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
    p_label = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
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
    ax.axhline(0.5, color=DARK_SLATE_CHARCOAL, linestyle=":", linewidth=1.0, alpha=0.6,
               label="Median OS (50%)")
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
        raise FileNotFoundError(
            f"Missing patient clusters file at {rel_path(INPUT_FILE)}. "
            "Run 03_cluster_patients.py first."
        )

    df_clusters = pd.read_csv(INPUT_FILE)
    print(f"Loaded stratified patient dataset: {len(df_clusters)} patients.")

    # 1. Compute phenotype characterisation profiles
    df_profiles = compute_phenotype_profiles(df_clusters)
    out_profile = OUTPUT_DIR / "phenotype_characterisation.csv"
    safe_save_csv(df_profiles, out_profile)
    print(f"Saved phenotype characterisation summary to {rel_path(out_profile)}")

    # 2. Generate publication baseline boxplots and Q3 ODE trajectories
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_baseline_boxplots(df_clusters, PLOTS_DIR / "baseline_signature_boxplots.png")
    simulate_q3_ode_trajectories(df_clusters, PLOTS_DIR / "ode_trajectories.png", OUTPUT_DIR)

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
