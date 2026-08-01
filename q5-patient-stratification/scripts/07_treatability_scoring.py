#!/usr/bin/env python3
"""Script 07: 3-Arm Clinical Decision Support & Q2/Q4 Integrated Treatability Scoring.

Constructs a 3-arm decision framework routing patients to optimal therapeutic arms:
  - Arm A: Immunotherapy Monotherapy (high predicted response / Immune Hot)
  - Arm B: Targeted Therapy (BRAF V600 / NRAS mutants failing Arm A, integrating Q2 Dabrafenib sensitivity weights)
  - Arm C: Combination / Reversal Therapy (non-responders with Q4 DepMap essentiality targets CSF1R, MDM2, AXL)

Calculates a patient-level Treatability Index (0-100) quantifying the biological feasibility of converting
predicted non-responders to immunotherapy sensitivity.
"""

import contextlib
from pathlib import Path
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
    sys.path.append(str(SUBPROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from phenotyping import get_cluster_name_map
from src.styles import ARM_PALETTE, DARK_SLATE_CHARCOAL, PHENOTYPE_PALETTE, set_presentation_style
from src.utils.io import safe_save_csv
from src.utils.logging import TeeStream
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, rel_path
from src.utils.plotting import save_fig

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "07_treatability_scoring.log"

INPUT_PATIENTS = PROCESSED_DIR / "q5" / "patient_clusters.csv"
INPUT_EXPR = PROCESSED_DIR / "merged" / "full" / "expr_merged.csv"
INPUT_CLIN_FULL = PROCESSED_DIR / "merged" / "full" / "clin_merged.csv"
Q2_DABRAFENIB_CSV = PROJECT_ROOT / "q2-viability-predictor" / "outputs" / "important_genes_Dabrafenib.csv"

OUTPUT_DIR = PROCESSED_DIR / "q5"
PLOTS_DIR = SUBPROJECT_ROOT / "plots" / "treatability"

# Clinical Decision Thresholds & Scoring Weights
DEFAULT_DABRAFENIB_INTERCEPT: float = 1.727
TIS_HIGH_QUANTILE: float = 0.60
TREATABILITY_CONVERSION_THRESHOLD: float = 40.0

# Treatability Index component weights (sum = 1.0)
TREAT_WEIGHT_AG_PRES: float = 0.35
TREAT_WEIGHT_IFN: float = 0.35
TREAT_WEIGHT_EFFECTOR: float = 0.15
TREAT_WEIGHT_M2_BARRIER: float = 0.15

# Antigen presentation sub-component weights (CYT vs TIS contribution)
AG_PRES_CYT_WEIGHT: float = 0.50
AG_PRES_TIS_WEIGHT: float = 0.50

# Confidence band boundaries
CONF_HIGH_THRESHOLD: float = 0.70
CONF_MOD_THRESHOLD: float = 0.45

# Arm A confidence scoring weights (TIS distance + IFN + CD8)
CONF_A_TIS_WEIGHT: float = 0.50
CONF_A_IFN_WEIGHT: float = 0.30
CONF_A_CD8_WEIGHT: float = 0.20

# Arm B confidence scoring weights (mutation strength + Dabrafenib sensitivity)
CONF_B_MUT_WEIGHT: float = 0.40
CONF_B_DAB_WEIGHT: float = 0.60

# Arm C confidence scoring weights (phenotype alignment + treatability + inverse M2)
CONF_C_ALIGN_WEIGHT: float = 0.50
CONF_C_TREAT_WEIGHT: float = 0.30
CONF_C_M2_WEIGHT: float = 0.20

# Mutation strength scores for Arm B confidence weighting
MUT_STRENGTH_BRAF: float = 1.00
MUT_STRENGTH_NRAS: float = 0.70

# Phenotype alignment scores for Arm C confidence weighting
ALIGN_SCORE_M2: float = 1.00
ALIGN_SCORE_COLD_HIGH: float = 0.75
ALIGN_SCORE_COLD_LOW: float = 0.50
ALIGN_SCORE_DEFAULT: float = 0.50

# Numerical safety and plot thresholds
STD_EPSILON: float = 1e-9
BAR_ANNOTATION_MIN_HEIGHT: float = 5.0
KDE_MIN_UNIQUE_VALUES: int = 3
KDE_MIN_VARIANCE: float = 1e-3

set_presentation_style()


# ---------------------------------------------------------------------------
# Safe Column-Access Helpers (DRY wrappers)
# ---------------------------------------------------------------------------

def _safe_z_score(df: pd.DataFrame, col: str) -> pd.Series:
    """Compute z-score for a column if it exists; returns 0.0 Series otherwise."""
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return _z_score(df[col], df.index)


def _safe_minmax(df: pd.DataFrame, col: str) -> pd.Series:
    """Min-max normalise a column to [0, 1] if it exists; returns 0.5 Series otherwise."""
    if col not in df.columns:
        return pd.Series(0.5, index=df.index)
    return _minmax(df[col])


def _add_phenotype_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'Phenotype' display-name column mapped from Cluster_ID."""
    cluster_id_to_name = get_cluster_name_map(df)
    df["Phenotype"] = df["Cluster_ID"].map(cluster_id_to_name)
    return df


# ---------------------------------------------------------------------------
# Q2 Dabrafenib Sensitivity Scoring
# ---------------------------------------------------------------------------

def load_q2_dabrafenib_weights() -> Dict[str, float]:
    """Load Q2 LASSO regression gene weights for Dabrafenib sensitivity prediction."""
    if not Q2_DABRAFENIB_CSV.exists():
        print(f"Warning: Q2 Dabrafenib CSV not found at {rel_path(Q2_DABRAFENIB_CSV)}. Using default empty dict.")
        return {}

    df_q2 = pd.read_csv(Q2_DABRAFENIB_CSV)
    weights = {}
    for _, row in df_q2.iterrows():
        gene_raw = str(row["gene"])
        coef = float(row["coefficient"])
        if gene_raw == "(Intercept)":
            weights["(Intercept)"] = coef
        else:
            gene_symbol = gene_raw.split(" ")[0].strip()
            weights[gene_symbol] = coef
    print(f"Loaded Q2 Dabrafenib sensitivity model: {len(weights)-1} gene features + intercept from {rel_path(Q2_DABRAFENIB_CSV)}")
    return weights


def calculate_q2_dabrafenib_score(df_merged: pd.DataFrame, q2_weights: Dict[str, float]) -> np.ndarray:
    """Calculate Q2 predicted Dabrafenib response score (AUC linear predictor) for each patient."""
    intercept = q2_weights.get("(Intercept)", DEFAULT_DABRAFENIB_INTERCEPT)
    scores = np.full(len(df_merged), intercept, dtype=float)

    for gene, coef in q2_weights.items():
        if gene != "(Intercept)" and gene in df_merged.columns:
            vals = df_merged[gene].fillna(df_merged[gene].median()).values
            scores += coef * vals

    min_val, max_val = np.min(scores), np.max(scores)
    if max_val > min_val:
        return 100.0 * (1.0 - (scores - min_val) / (max_val - min_val))
    return np.full(len(df_merged), 50.0)


# ---------------------------------------------------------------------------
# Treatability Index Computation
# ---------------------------------------------------------------------------

def _z_score(series: pd.Series, df_index: pd.Index) -> pd.Series:
    """Compute z-score for a feature Series; returns 0.0 for empty or constant Series."""
    if series is None or series.count() == 0:
        return pd.Series(0.0, index=df_index)
    s = series.fillna(series.median())
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=df_index)
    return (s - s.mean()) / std


def _compute_treatability_raw_components(
    df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute normalized z-scores and raw weighted treatability component series."""
    z_tis = _safe_z_score(df, "TIS")
    z_ifn = _safe_z_score(df, "IFN_gamma")
    z_cyt = _safe_z_score(df, "CYT")
    z_cd8 = _safe_z_score(df, "CD8_Tcell")
    z_m1m2 = _safe_z_score(df, "M1_M2_Ratio")
    z_m2 = _safe_z_score(df, "M2_score")

    ag_pres_score = AG_PRES_CYT_WEIGHT * z_cyt + AG_PRES_TIS_WEIGHT * z_tis
    ifn_pathway_score = z_ifn
    immuno_effector = z_cd8 + z_m1m2
    immunosuppressive_barrier = z_m2

    raw_treatability = (
        TREAT_WEIGHT_AG_PRES * ag_pres_score +
        TREAT_WEIGHT_IFN * ifn_pathway_score +
        TREAT_WEIGHT_EFFECTOR * immuno_effector -
        TREAT_WEIGHT_M2_BARRIER * immunosuppressive_barrier
    )
    return ag_pres_score, ifn_pathway_score, immuno_effector, immunosuppressive_barrier, raw_treatability


def _scale_treatability_index(raw_treatability: pd.Series, df_len: int) -> np.ndarray:
    """Rescale raw treatability scores into a 0-100 index."""
    min_t, max_t = raw_treatability.min(), raw_treatability.max()
    if max_t > min_t and not np.isnan(max_t):
        return 100.0 * (raw_treatability - min_t) / (max_t - min_t)
    return np.full(df_len, 50.0)


def calculate_treatability_index(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Calculate composite Treatability Index (0-100) and component sub-scores."""
    ag_pres, ifn_path, effector, barrier, raw_t = _compute_treatability_raw_components(df)
    treatability_index = _scale_treatability_index(raw_t, len(df))

    df_components = pd.DataFrame(
        {
            "Ag_Presentation_Score": ag_pres,
            "IFN_Pathway_Score": ifn_path,
            "Immuno_Effector_Score": effector,
            "M2_Suppression_Barrier": barrier,
            "Treatability_Index": treatability_index,
        },
        index=df.index,
    )
    return treatability_index, df_components


# ---------------------------------------------------------------------------
# Treatment Arm Allocation Rules
# ---------------------------------------------------------------------------

def _evaluate_arm_c_therapy(pheno_name: str, treat_idx: float) -> Tuple[str, str]:
    """Determine recommended therapy and Q4 target nomination for Arm C combination patients."""
    if pheno_name in ("M2 Immunosuppressive", "Immunosuppressive M2-High"):
        rx = "Anti-PD-1 + CSF1R Inhibitor (Pexidartinib) [Macrophage Reprogramming]"
        q4_target = "CSF1R (M2 TAM Depletion)"
    elif pheno_name == "Immune Cold":
        if treat_idx > TREATABILITY_CONVERSION_THRESHOLD:
            rx = "Anti-PD-1 + AXL Inhibitor (Bemcentinib) [STING / Type-I IFN Priming]"
            q4_target = "AXL / STING Pathway"
        else:
            rx = "Chemotherapy (Dacarbazine) / HDAC Inhibitor + Anti-PD-1"
            q4_target = "HDAC / Epigenetic Remodeling"
    else:
        rx = "Anti-PD-1 + MDM2 Antagonist (Idasanutlin) [p53 Reactivation]"
        q4_target = "MDM2 (p53 Activation)"
    return rx, q4_target


def _evaluate_patient_arm(
    row: pd.Series, pheno_name: str, tis_q60: float
) -> Tuple[str, str, str]:
    """Evaluate decision tree rules for a single patient to assign treatment arm and target."""
    mut_braf = row.get("mut_BRAF", 0) == 1
    mut_nras = row.get("mut_NRAS", 0) == 1
    response = row.get("RESPONSE_BINARY", np.nan)
    tis_val = row.get("TIS", 0.0)

    if pheno_name == "Immune Hot" or (tis_val > tis_q60 and response != 0):
        return (
            "Arm A: Immunotherapy",
            "Anti-PD-1 Monotherapy (Pembrolizumab / Nivolumab)",
            "N/A (Arm A Candidate)",
        )
    if mut_braf or mut_nras:
        if mut_braf:
            dab_sens = row["Dabrafenib_Sensitivity_Index"]
            rx = f"BRAF + MEK Inhibitor (Dabrafenib + Trametinib; Q2 Sens: {dab_sens:.1f}/100)"
            target = "BRAF V600E / MAPK Pathway"
        else:
            rx = "MEK + CDK4/6 Inhibitor (Binimetinib + Ribociclib)"
            target = "NRAS / MAPK / Cell Cycle"
        return "Arm B: Targeted Therapy", rx, target

    rx, q4_target = _evaluate_arm_c_therapy(pheno_name, row["Treatability_Index"])
    return "Arm C: Combination/Reversal", rx, q4_target


def assign_treatment_arms(
    df: pd.DataFrame, df_components: pd.DataFrame, dabrafenib_scores: np.ndarray
) -> pd.DataFrame:
    """Assign patients to Arm A (Immunotherapy), Arm B (Targeted Therapy), or Arm C (Combination/Reversal)."""
    df_out = df.copy()
    df_out["Dabrafenib_Sensitivity_Index"] = dabrafenib_scores
    df_out["Treatability_Index"] = df_components["Treatability_Index"].values

    arm_assignments, recommended_therapies, q4_target_nominations = [], [], []
    tis_q60 = df["TIS"].quantile(TIS_HIGH_QUANTILE) if "TIS" in df.columns else 0.0
    cluster_id_to_name = get_cluster_name_map(df)

    for _, row in df_out.iterrows():
        cluster_id = row.get("Cluster_ID", 0)
        pheno_name = cluster_id_to_name.get(int(cluster_id), "Unknown")
        arm, rx, q4_target = _evaluate_patient_arm(row, pheno_name, tis_q60)

        arm_assignments.append(arm)
        recommended_therapies.append(rx)
        q4_target_nominations.append(q4_target)

    df_out["Treatment_Arm"] = arm_assignments
    df_out["Recommended_Therapy"] = recommended_therapies
    df_out["Q4_Nominated_Target"] = q4_target_nominations
    return df_out


# ---------------------------------------------------------------------------
# Confidence Index Computation
# ---------------------------------------------------------------------------

def _minmax(series: pd.Series) -> pd.Series:
    """Normalise a Series to [0, 1]; returns 0.5 everywhere if range is zero."""
    lo, hi = series.min(), series.max()
    return (series - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=series.index)


def _assign_confidence_band(raw_score: float, is_nras_only: bool) -> str:
    """Return 'High' / 'Moderate' / 'Low'; NRAS-only Arm B patients are capped at Moderate."""
    if raw_score >= CONF_HIGH_THRESHOLD:
        return "Moderate" if is_nras_only else "High"
    return "Moderate" if raw_score >= CONF_MOD_THRESHOLD else "Low"


def _compute_arm_a_confidence(
    tis_dist_norm: pd.Series, ifn_norm: pd.Series, cd8_norm: pd.Series, mask_a: pd.Series
) -> pd.Series:
    """Compute raw confidence score for Arm A (Immunotherapy)."""
    return (
        CONF_A_TIS_WEIGHT * tis_dist_norm[mask_a] +
        CONF_A_IFN_WEIGHT * ifn_norm[mask_a] +
        CONF_A_CD8_WEIGHT * cd8_norm[mask_a]
    )


def _compute_arm_b_confidence(
    df: pd.DataFrame, dab_norm: pd.Series, mask_b: pd.Series
) -> pd.Series:
    """Compute raw confidence score for Arm B (Targeted Therapy)."""
    mut_strength = pd.Series(0.0, index=df.index)
    mut_strength[df["mut_BRAF"] == 1] = MUT_STRENGTH_BRAF
    mut_strength[(df["mut_BRAF"] != 1) & (df["mut_NRAS"] == 1)] = MUT_STRENGTH_NRAS
    return CONF_B_MUT_WEIGHT * mut_strength[mask_b] + CONF_B_DAB_WEIGHT * dab_norm[mask_b]


def _compute_arm_c_confidence(
    alignment_score: pd.Series, treat_norm: pd.Series, m2_norm_inv: pd.Series, mask_c: pd.Series
) -> pd.Series:
    """Compute raw confidence score for Arm C (Combination/Reversal)."""
    return (
        CONF_C_ALIGN_WEIGHT * alignment_score[mask_c] +
        CONF_C_TREAT_WEIGHT * treat_norm[mask_c] +
        CONF_C_M2_WEIGHT * m2_norm_inv[mask_c]
    )


def _build_alignment_scores(df: pd.DataFrame) -> pd.Series:
    """Build phenotype-specific alignment scores for Arm C confidence weighting."""
    cluster_id_to_name = get_cluster_name_map(df)
    pheno_short = df["Cluster_ID"].map(cluster_id_to_name)
    alignment_score = pd.Series(ALIGN_SCORE_DEFAULT, index=df.index)
    alignment_score[pheno_short.isin(["M2 Immunosuppressive", "Immunosuppressive M2-High"])] = ALIGN_SCORE_M2
    alignment_score[(pheno_short == "Immune Cold") & (df["Treatability_Index"] > TREATABILITY_CONVERSION_THRESHOLD)] = ALIGN_SCORE_COLD_HIGH
    alignment_score[(pheno_short == "Immune Cold") & (df["Treatability_Index"] <= TREATABILITY_CONVERSION_THRESHOLD)] = ALIGN_SCORE_COLD_LOW
    return alignment_score


def _compute_arm_confidence_scores(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Compute raw confidence scores and NRAS-only masks per arm."""
    tis_q60 = df["TIS"].quantile(TIS_HIGH_QUANTILE) if "TIS" in df.columns else 0.0
    tis_norm = _safe_minmax(df, "TIS")
    ifn_norm = _safe_minmax(df, "IFN_gamma")
    cd8_norm = _safe_minmax(df, "CD8_Tcell")
    dab_norm = _minmax(df["Dabrafenib_Sensitivity_Index"])
    treat_norm = _minmax(df["Treatability_Index"])
    m2_norm_inv = 1.0 - _safe_minmax(df, "M2_score")

    tis_above_boundary = ((df["TIS"] - tis_q60) / (df["TIS"].std() + STD_EPSILON)).clip(lower=0.0)
    tis_dist_norm = _minmax(tis_above_boundary)
    alignment_score = _build_alignment_scores(df)

    arm_col = df["Treatment_Arm"]
    mask_a = arm_col.str.startswith("Arm A")
    mask_b = arm_col.str.startswith("Arm B")
    mask_c = arm_col.str.startswith("Arm C")

    conf_raw = pd.Series(0.0, index=df.index)
    conf_raw[mask_a] = _compute_arm_a_confidence(tis_dist_norm, ifn_norm, cd8_norm, mask_a)
    conf_raw[mask_b] = _compute_arm_b_confidence(df, dab_norm, mask_b)
    conf_raw[mask_c] = _compute_arm_c_confidence(alignment_score, treat_norm, m2_norm_inv, mask_c)

    nras_only = (df["mut_BRAF"] != 1) & (df["mut_NRAS"] == 1) & mask_b
    return conf_raw, nras_only


def compute_recommendation_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Compute arm-specific Recommendation_Confidence_Index (0-100) and Confidence_Band."""
    df_out = df.copy()
    conf_raw, nras_only = _compute_arm_confidence_scores(df)

    df_out["Recommendation_Confidence_Index"] = (conf_raw * 100.0).clip(0.0, 100.0).round(2)
    df_out["Confidence_Band"] = [
        _assign_confidence_band(raw, nras) for raw, nras in zip(conf_raw, nras_only)
    ]
    return df_out


# ---------------------------------------------------------------------------
# Visualisation & Plot Generation
# ---------------------------------------------------------------------------

def _annotate_bar_segments(ax: plt.Axes) -> None:
    """Annotate stacked bar segments with percentage labels."""
    for container in ax.containers:
        for p in container:
            height = p.get_height()
            if height > BAR_ANNOTATION_MIN_HEIGHT:
                ax.annotate(
                    f"{height:.1f}%",
                    (p.get_x() + p.get_width() / 2.0, p.get_y() + height / 2.0),
                    ha="center", va="center", fontsize=8.5,
                    color="white" if p.get_facecolor()[0] < 0.5 else "black",
                    fontweight="bold",
                )


def _style_arm_breakdown_axes(ax: plt.Axes) -> None:
    """Apply presentation styling to the arm allocation breakdown axes."""
    ax.set_title("3-Arm Clinical Decision System Allocation across Biological Phenotypes", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Biological Phenotype Subgroup", fontsize=11, fontweight="bold")
    ax.set_ylabel("Proportion of Patients (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(title="Clinical Decision Arm", loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=9.5)
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.5, alpha=0.6)
    plt.xticks(rotation=0)


def plot_arm_assignment_breakdown(df_assigned: pd.DataFrame, out_path: Path) -> None:
    """Plot distribution of patient allocation across Arm A, B, and C by biological phenotype."""
    df_plot = _add_phenotype_column(df_assigned.copy())
    ct = pd.crosstab(df_plot["Phenotype"], df_plot["Treatment_Arm"], normalize="index") * 100.0

    fig, ax = plt.subplots(figsize=(11, 6))
    ct.plot(
        kind="bar", stacked=True,
        color=[ARM_PALETTE.get(col, DARK_SLATE_CHARCOAL) for col in ct.columns],
        ax=ax, edgecolor="black", linewidth=0.8, width=0.55,
    )
    _style_arm_breakdown_axes(ax)
    _annotate_bar_segments(ax)
    plt.tight_layout()
    save_fig(fig, out_path, dpi=300)
    plt.close(fig)


def _plot_treatability_boxplot_panel(ax: plt.Axes, df_plot: pd.DataFrame) -> None:
    """Render Treatability Index boxplot panel."""
    sns.boxplot(
        data=df_plot, x="Phenotype", y="Treatability_Index", hue="Phenotype", legend=False,
        palette=PHENOTYPE_PALETTE, ax=ax,
        boxprops=dict(alpha=0.85, edgecolor="black", linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.5), width=0.5,
    )
    sns.stripplot(
        data=df_plot, x="Phenotype", y="Treatability_Index", color="black",
        alpha=0.35, size=4, jitter=0.2, ax=ax,
    )
    ax.set_title("Treatability Index (0-100) across Biological Phenotypes", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Biological Phenotype", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Treatability Index (Higher = Convertible)", fontsize=10.5, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.5, alpha=0.6)
    ax.tick_params(axis="x", rotation=15)


def _plot_dabrafenib_hist_panel(ax: plt.Axes, df_plot: pd.DataFrame) -> None:
    """Render Q2 Dabrafenib Sensitivity Index histogram panel."""
    df_arm_b = df_plot[df_plot["Treatment_Arm"] == "Arm B: Targeted Therapy"]
    if len(df_arm_b) > 0:
        has_var = df_arm_b["Dabrafenib_Sensitivity_Index"].nunique() > KDE_MIN_UNIQUE_VALUES and df_arm_b["Dabrafenib_Sensitivity_Index"].std() > KDE_MIN_VARIANCE
        sns.histplot(
            data=df_arm_b, x="Dabrafenib_Sensitivity_Index", hue="Phenotype",
            palette=PHENOTYPE_PALETTE, kde=has_var, ax=ax, element="step", alpha=0.5,
        )
        ax.set_title("Q2 Dabrafenib Sensitivity Index in Arm B (Targeted Therapy)", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Q2 Dabrafenib Sensitivity Index (0-100)", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Patient Count", fontsize=10.5, fontweight="bold")
        ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.5, alpha=0.6)
    else:
        ax.text(0.5, 0.5, "No Arm B Patients Identified", ha="center", va="center", fontsize=12)


def plot_treatability_distributions(df_assigned: pd.DataFrame, out_path: Path) -> None:
    """Plot distribution of Treatability Index and Q2 Dabrafenib Sensitivity Scores."""
    df_plot = _add_phenotype_column(df_assigned.copy())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    _plot_treatability_boxplot_panel(ax1, df_plot)
    _plot_dabrafenib_hist_panel(ax2, df_plot)

    fig.suptitle("Treatability Index & Q2 Targeted Sensitivity Distribution", fontsize=13.5, fontweight="bold", y=0.98)
    plt.tight_layout()
    save_fig(fig, out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline Orchestration Helpers
# ---------------------------------------------------------------------------

def _load_phase7_patient_data() -> pd.DataFrame:
    """Load patient dataset and join clinical IMMUNOTHERAPY flag."""
    if not INPUT_PATIENTS.exists():
        raise FileNotFoundError(f"Missing patient dataset at {rel_path(INPUT_PATIENTS)}. Run Phase 3/4 first.")

    df_patients = pd.read_csv(INPUT_PATIENTS)
    print(f"Loaded patient dataset: {len(df_patients)} patients")

    if INPUT_CLIN_FULL.exists() and "IMMUNOTHERAPY" not in df_patients.columns:
        df_clin_full = pd.read_csv(INPUT_CLIN_FULL, usecols=["SAMPLE_ID", "IMMUNOTHERAPY"])
        df_patients = df_patients.merge(df_clin_full, on="SAMPLE_ID", how="left")
        print(f"Joined IMMUNOTHERAPY flag from {rel_path(INPUT_CLIN_FULL)}")

    df_patients["ICI_Treated"] = df_patients["IMMUNOTHERAPY"].fillna(0).astype(int) if "IMMUNOTHERAPY" in df_patients.columns else 0
    n_ici = int(df_patients["ICI_Treated"].sum())
    print(f"  ICI-treated (known response): {n_ici}")
    print(f"  Prospective (no ICI label):   {len(df_patients) - n_ici}")
    return df_patients


def _merge_q2_expression(df_patients: pd.DataFrame) -> pd.DataFrame:
    """Merge expression matrix for Q2 Dabrafenib scoring if available."""
    if INPUT_EXPR.exists():
        df_expr = pd.read_csv(INPUT_EXPR)
        df_merged = pd.merge(df_patients, df_expr, on="SAMPLE_ID", how="left", suffixes=("", "_expr"))
        print(f"Merged expression matrix: {df_merged.shape[1]} total columns")
        return df_merged
    print(f"Warning: Expression matrix not found at {rel_path(INPUT_EXPR)}. Using patient features only.")
    return df_patients.copy()


def _print_arm_allocation_summary(df_assigned: pd.DataFrame) -> None:
    """Print arm allocation metrics split by ICI-treated vs prospective cohorts."""
    conf_counts = df_assigned["Confidence_Band"].value_counts()
    print(f"Recommendation Confidence — High: {conf_counts.get('High', 0)}, "
          f"Moderate: {conf_counts.get('Moderate', 0)}, Low: {conf_counts.get('Low', 0)}")

    df_ici = df_assigned[df_assigned["ICI_Treated"] == 1]
    df_pros = df_assigned[df_assigned["ICI_Treated"] == 0]

    for label, sub in [("ICI-treated ", df_ici), ("Prospective ", df_pros)]:
        cA = int((sub["Treatment_Arm"].str.startswith("Arm A")).sum())
        cB = int((sub["Treatment_Arm"].str.startswith("Arm B")).sum())
        cC = int((sub["Treatment_Arm"].str.startswith("Arm C")).sum())
        print(f"  {label} (N={len(sub)}):   Arm A={cA}, Arm B={cB}, Arm C={cC}")


def _save_phase7_csv_outputs(df_assigned: pd.DataFrame) -> None:
    """Save treatability scores and arm summary metrics CSVs."""
    out_csv = OUTPUT_DIR / "treatability_scores.csv"
    safe_save_csv(df_assigned, out_csv)
    print(f"\nSaved treatability scores and arm assignments to {rel_path(out_csv)}")

    df_summary = _add_phenotype_column(df_assigned.copy())
    df_summary = df_summary.groupby(["Phenotype", "Treatment_Arm"]).size().reset_index(name="Patient_Count")
    out_sum_csv = OUTPUT_DIR / "arm_summary_metrics.csv"
    safe_save_csv(df_summary, out_sum_csv)
    print(f"Saved arm summary metrics to {rel_path(out_sum_csv)}")


def _generate_phase7_plots(df_assigned: pd.DataFrame) -> Tuple[Path, Path]:
    """Generate and save Phase 7 visualization figures."""
    plot_arm_file = PLOTS_DIR / "arm_assignment_breakdown.png"
    plot_arm_assignment_breakdown(df_assigned, plot_arm_file)

    plot_dist_file = PLOTS_DIR / "treatability_index_distribution.png"
    plot_treatability_distributions(df_assigned, plot_dist_file)

    return plot_arm_file, plot_dist_file


def main() -> None:
    """Main execution function for Phase 7 treatability scoring."""
    print("=" * 80)
    print(f"Starting Phase 7: 3-Arm Decision Support & Treatability Scoring (Project root: {rel_path(PROJECT_ROOT)})")
    print("=" * 80)

    df_patients = _load_phase7_patient_data()
    q2_weights = load_q2_dabrafenib_weights()
    df_merged = _merge_q2_expression(df_patients)

    dabrafenib_scores = calculate_q2_dabrafenib_score(df_merged, q2_weights)
    _, df_components = calculate_treatability_index(df_patients)

    df_assigned = assign_treatment_arms(df_patients, df_components, dabrafenib_scores)
    df_assigned["ICI_Treated"] = df_patients["ICI_Treated"].values
    df_assigned = compute_recommendation_confidence(df_assigned)

    _print_arm_allocation_summary(df_assigned)
    _save_phase7_csv_outputs(df_assigned)

    p_arm, p_dist = _generate_phase7_plots(df_assigned)
    print(f"\nGenerated Plots:\n  1. {rel_path(p_arm)}\n  2. {rel_path(p_dist)}")

    print("\n" + "=" * 80)
    print("PHASE 7 TREATABILITY SCORING & DECISION SUPPORT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
