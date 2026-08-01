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
from typing import Dict, List, Tuple

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

from q5_constants import PHENOTYPE_PROB_COL
from phenotyping import get_cluster_name_map
from src.styles import ARM_PALETTE, DARK_SLATE_CHARCOAL, PHENOTYPE_PALETTE, set_presentation_style
from src.utils.formatting import generate_obsidian_frontmatter
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
# Full cohort expression matrix (N=699) — needed for Q2 Dabrafenib score on all patients
INPUT_EXPR     = PROCESSED_DIR / "merged" / "full" / "expr_merged.csv"
# Full cohort clinical file — provides IMMUNOTHERAPY flag for ICI vs prospective split
INPUT_CLIN_FULL = PROCESSED_DIR / "merged" / "full" / "clin_merged.csv"
Q2_DABRAFENIB_CSV = PROJECT_ROOT / "q2-viability-predictor" / "outputs" / "important_genes_Dabrafenib.csv"

OUTPUT_DIR = PROCESSED_DIR / "q5"
PLOTS_DIR = SUBPROJECT_ROOT / "plots" / "treatability"
REPORTS_DIR = SUBPROJECT_ROOT / "reports" / "q5_phases"

set_presentation_style()





def load_q2_dabrafenib_weights() -> Dict[str, float]:
    """Load Q2 LASSO regression gene weights for Dabrafenib sensitivity prediction."""
    q2_file = Q2_DABRAFENIB_CSV
    if not q2_file.exists():
        q2_file = PROJECT_ROOT / "q2-viability-predictor" / "important_genes_Dabrafenib.csv"
    if not q2_file.exists():
        print(f"Warning: Q2 Dabrafenib CSV not found at {rel_path(Q2_DABRAFENIB_CSV)}. Using default empty dict.")
        return {}

    df_q2 = pd.read_csv(q2_file)
    weights = {}
    for _, row in df_q2.iterrows():
        gene_raw = str(row["gene"])
        coef = float(row["coefficient"])
        if gene_raw == "(Intercept)":
            weights["(Intercept)"] = coef
        else:
            gene_symbol = gene_raw.split(" ")[0].strip()
            weights[gene_symbol] = coef
    print(f"Loaded Q2 Dabrafenib sensitivity model: {len(weights)-1} gene features + intercept from {rel_path(q2_file)}")
    return weights


def calculate_q2_dabrafenib_score(df_merged: pd.DataFrame, q2_weights: Dict[str, float]) -> np.ndarray:
    """Calculate Q2 predicted Dabrafenib response score (AUC linear predictor) for each patient."""
    intercept = q2_weights.get("(Intercept)", 1.727)
    scores = np.full(len(df_merged), intercept, dtype=float)

    for gene, coef in q2_weights.items():
        if gene == "(Intercept)":
            continue
        if gene in df_merged.columns:
            vals = df_merged[gene].fillna(df_merged[gene].median()).values
            scores += coef * vals

    # Rescale raw predicted AUC to 0-100 Dabrafenib Sensitivity Index (lower predicted AUC = higher sensitivity)
    min_val, max_val = np.min(scores), np.max(scores)
    if max_val > min_val:
        norm_sensitivity = 100.0 * (1.0 - (scores - min_val) / (max_val - min_val))
    else:
        norm_sensitivity = np.full(len(df_merged), 50.0)

    return norm_sensitivity


def _z_score(series: pd.Series, df_index: pd.Index) -> pd.Series:
    """Compute z-score for a feature Series; returns 0.0 for empty or constant Series."""
    if series is None or series.count() == 0:
        return pd.Series(0.0, index=df_index)
    s = series.fillna(series.median())
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=df_index)
    return (s - s.mean()) / std


def calculate_treatability_index(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Calculate composite Treatability Index (0-100) and component sub-scores for non-responders.

    Treatability Index measures biological convertibility based on:
      1. Antigen Presentation Integrity (B2M, TAP1, HLA-A or proxy TIS/CYT)
      2. Interferon-Gamma Pathway Intactness (IFN_gamma, TIS)
      3. Tumor Mutational Burden (TMB_NONSYNONYMOUS)
      4. Low Immunosuppressive Microenvironment Barrier (1 - normalized M2_score)
    """
    z_tis = _z_score(df["TIS"], df.index) if "TIS" in df.columns else pd.Series(0.0, index=df.index)
    z_ifn = _z_score(df["IFN_gamma"], df.index) if "IFN_gamma" in df.columns else pd.Series(0.0, index=df.index)
    z_cyt = _z_score(df["CYT"], df.index) if "CYT" in df.columns else pd.Series(0.0, index=df.index)
    z_cd8 = _z_score(df["CD8_Tcell"], df.index) if "CD8_Tcell" in df.columns else pd.Series(0.0, index=df.index)
    z_m1m2 = _z_score(df["M1_M2_Ratio"], df.index) if "M1_M2_Ratio" in df.columns else pd.Series(0.0, index=df.index)
    z_m2 = _z_score(df["M2_score"], df.index) if "M2_score" in df.columns else pd.Series(0.0, index=df.index)

    ag_pres_score = 0.5 * z_cyt + 0.5 * z_tis
    ifn_pathway_score = z_ifn
    immuno_effector = z_cd8 + z_m1m2
    immunosuppressive_barrier = z_m2

    raw_treatability = (
        0.35 * ag_pres_score +
        0.35 * ifn_pathway_score +
        0.15 * immuno_effector -
        0.15 * immunosuppressive_barrier
    )

    min_t, max_t = raw_treatability.min(), raw_treatability.max()
    if max_t > min_t and not np.isnan(max_t):
        treatability_index = 100.0 * (raw_treatability - min_t) / (max_t - min_t)
    else:
        treatability_index = np.full(len(df), 50.0)

    df_components = pd.DataFrame(
        {
            "Ag_Presentation_Score": ag_pres_score,
            "IFN_Pathway_Score": ifn_pathway_score,
            "Immuno_Effector_Score": immuno_effector,
            "M2_Suppression_Barrier": immunosuppressive_barrier,
            "Treatability_Index": treatability_index,
        },
        index=df.index,
    )

    return treatability_index, df_components


def _evaluate_patient_arm(
    row: pd.Series, pheno_name: str, tis_q60: float
) -> Tuple[str, str, str]:
    """Evaluate decision tree rules for a single patient to assign treatment arm and target."""
    mut_braf = row.get("mut_BRAF", 0) == 1
    mut_nras = row.get("mut_NRAS", 0) == 1
    response = row.get("RESPONSE_BINARY", np.nan)
    tis_val = row.get("TIS", 0.0)

    is_immune_hot = pheno_name == "Immune Hot"
    is_high_tis = tis_val > tis_q60

    if is_immune_hot or (is_high_tis and response != 0):
        arm = "Arm A: Immunotherapy"
        rx = "Anti-PD-1 Monotherapy (Pembrolizumab / Nivolumab)"
        q4_target = "N/A (Arm A Candidate)"
    elif mut_braf or mut_nras:
        arm = "Arm B: Targeted Therapy"
        if mut_braf:
            dab_sens = row["Dabrafenib_Sensitivity_Index"]
            rx = f"BRAF + MEK Inhibitor (Dabrafenib + Trametinib; Q2 Sens: {dab_sens:.1f}/100)"
            q4_target = "BRAF V600E / MAPK Pathway"
        else:
            rx = "MEK + CDK4/6 Inhibitor (Binimetinib + Ribociclib)"
            q4_target = "NRAS / MAPK / Cell Cycle"
    else:
        arm = "Arm C: Combination/Reversal"
        treat_idx = row["Treatability_Index"]

        if pheno_name in ("M2 Immunosuppressive", "Immunosuppressive M2-High"):
            rx = "Anti-PD-1 + CSF1R Inhibitor (Pexidartinib) [Macrophage Reprogramming]"
            q4_target = "CSF1R (M2 TAM Depletion)"
        elif pheno_name == "Immune Cold":
            if treat_idx > 40.0:
                rx = "Anti-PD-1 + AXL Inhibitor (Bemcentinib) [STING / Type-I IFN Priming]"
                q4_target = "AXL / STING Pathway"
            else:
                rx = "Chemotherapy (Dacarbazine) / HDAC Inhibitor + Anti-PD-1"
                q4_target = "HDAC / Epigenetic Remodeling"
        else:
            rx = "Anti-PD-1 + MDM2 Antagonist (Idasanutlin) [p53 Reactivation]"
            q4_target = "MDM2 (p53 Activation)"

    return arm, rx, q4_target


def assign_treatment_arms(
    df: pd.DataFrame, df_components: pd.DataFrame, dabrafenib_scores: np.ndarray
) -> pd.DataFrame:
    """Assign patients to Arm A (Immunotherapy), Arm B (Targeted Therapy), or Arm C (Combination/Reversal)."""
    df_out = df.copy()
    df_out["Dabrafenib_Sensitivity_Index"] = dabrafenib_scores
    df_out["Treatability_Index"] = df_components["Treatability_Index"].values

    arm_assignments = []
    recommended_therapies = []
    q4_target_nominations = []

    tis_q60 = df["TIS"].quantile(0.60) if "TIS" in df.columns else 0.0
    cluster_id_to_name = get_cluster_name_map(df)

    for idx, row in df_out.iterrows():
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

# Thresholds that map a raw 0-1 composite score to a labelled confidence band.
# High ≥ 0.70 | Moderate ≥ 0.45 | Low < 0.45
CONF_HIGH_THRESHOLD: float = 0.70
CONF_MOD_THRESHOLD: float  = 0.45


def _minmax(series: pd.Series) -> pd.Series:
    """Normalise a Series to [0, 1]; returns 0.5 everywhere if range is zero."""
    lo, hi = series.min(), series.max()
    if hi > lo:
        return (series - lo) / (hi - lo)
    return pd.Series(0.5, index=series.index)


def _assign_confidence_band(raw_score: float, is_nras_only: bool) -> str:
    """Return 'High' / 'Moderate' / 'Low'; NRAS-only Arm B patients are capped at Moderate."""
    if raw_score >= CONF_HIGH_THRESHOLD:
        if is_nras_only:
            return "Moderate"
        return "High"
    if raw_score >= CONF_MOD_THRESHOLD:
        return "Moderate"
    return "Low"


def _compute_arm_confidence_scores(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Compute raw confidence scores and NRAS-only masks per arm."""
    tis_q60 = df["TIS"].quantile(0.60) if "TIS" in df.columns else 0.0
    tis_norm = _minmax(df["TIS"]) if "TIS" in df.columns else pd.Series(0.5, index=df.index)
    ifn_norm = _minmax(df["IFN_gamma"]) if "IFN_gamma" in df.columns else pd.Series(0.5, index=df.index)
    cd8_norm = _minmax(df["CD8_Tcell"]) if "CD8_Tcell" in df.columns else pd.Series(0.5, index=df.index)
    dab_norm = _minmax(df["Dabrafenib_Sensitivity_Index"])
    treat_norm = _minmax(df["Treatability_Index"])
    m2_norm_inv = 1.0 - _minmax(df["M2_score"]) if "M2_score" in df.columns else pd.Series(0.5, index=df.index)

    tis_above_boundary = ((df["TIS"] - tis_q60) / (df["TIS"].std() + 1e-9)).clip(lower=0.0)
    tis_dist_norm = _minmax(tis_above_boundary)

    cluster_id_to_name = get_cluster_name_map(df)
    pheno_short = df["Cluster_ID"].map(cluster_id_to_name)
    alignment_score = pd.Series(0.5, index=df.index)
    is_m2 = pheno_short.isin(["M2 Immunosuppressive", "Immunosuppressive M2-High"])
    is_cold_high = (pheno_short == "Immune Cold") & (df["Treatability_Index"] > 40.0)
    is_cold_low = (pheno_short == "Immune Cold") & (df["Treatability_Index"] <= 40.0)
    alignment_score[is_m2] = 1.00
    alignment_score[is_cold_high] = 0.75
    alignment_score[is_cold_low] = 0.50

    conf_raw = pd.Series(0.0, index=df.index)
    arm_col = df["Treatment_Arm"]
    mask_a = arm_col.str.startswith("Arm A")
    mask_b = arm_col.str.startswith("Arm B")
    mask_c = arm_col.str.startswith("Arm C")

    conf_raw[mask_a] = 0.50 * tis_dist_norm[mask_a] + 0.30 * ifn_norm[mask_a] + 0.20 * cd8_norm[mask_a]

    mut_strength = pd.Series(0.0, index=df.index)
    mut_strength[df["mut_BRAF"] == 1] = 1.00
    mut_strength[(df["mut_BRAF"] != 1) & (df["mut_NRAS"] == 1)] = 0.70
    conf_raw[mask_b] = 0.40 * mut_strength[mask_b] + 0.60 * dab_norm[mask_b]

    conf_raw[mask_c] = 0.50 * alignment_score[mask_c] + 0.30 * treat_norm[mask_c] + 0.20 * m2_norm_inv[mask_c]

    nras_only = (df["mut_BRAF"] != 1) & (df["mut_NRAS"] == 1) & mask_b
    return conf_raw, nras_only


def compute_recommendation_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Compute arm-specific Recommendation_Confidence_Index (0–100) and Confidence_Band."""
    df_out = df.copy()
    conf_raw, nras_only = _compute_arm_confidence_scores(df)

    conf_index = (conf_raw * 100.0).clip(0.0, 100.0)
    df_out["Recommendation_Confidence_Index"] = conf_index.round(2)
    df_out["Confidence_Band"] = [
        _assign_confidence_band(raw, nras) for raw, nras in zip(conf_raw, nras_only)
    ]
    return df_out


def plot_arm_assignment_breakdown(df_assigned: pd.DataFrame, out_path: Path) -> None:
    """Plot distribution of patient allocation across Arm A, B, and C by biological phenotype."""
    df_plot = df_assigned.copy()
    cluster_id_to_name = get_cluster_name_map(df_plot)
    df_plot["Phenotype"] = df_plot["Cluster_ID"].map(cluster_id_to_name)

    ct = pd.crosstab(df_plot["Phenotype"], df_plot["Treatment_Arm"], normalize="index") * 100.0

    fig, ax = plt.subplots(figsize=(11, 6))
    ct.plot(
        kind="bar",
        stacked=True,
        color=[ARM_PALETTE.get(col, DARK_SLATE_CHARCOAL) for col in ct.columns],
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
        width=0.55,
    )

    ax.set_title("3-Arm Clinical Decision System Allocation across Biological Phenotypes", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Biological Phenotype Subgroup", fontsize=11, fontweight="bold")
    ax.set_ylabel("Proportion of Patients (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(title="Clinical Decision Arm", loc="upper right", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=9.5)
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.5, alpha=0.6)
    plt.xticks(rotation=0)

    # Annotate bar segments
    for container in ax.containers:
        for p in container:
            height = p.get_height()
            if height > 5.0:
                ax.annotate(
                    f"{height:.1f}%",
                    (p.get_x() + p.get_width() / 2.0, p.get_y() + height / 2.0),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if p.get_facecolor()[0] < 0.5 else "black",
                    fontweight="bold",
                )

    plt.tight_layout()
    save_fig(fig, out_path, dpi=300)
    plt.close(fig)


def _plot_treatability_boxplot_panel(ax1: plt.Axes, df_plot: pd.DataFrame) -> None:
    """Render Treatability Index boxplot panel."""
    sns.boxplot(
        data=df_plot,
        x="Phenotype",
        y="Treatability_Index",
        hue="Phenotype",
        legend=False,
        palette=PHENOTYPE_PALETTE,
        ax=ax1,
        boxprops=dict(alpha=0.85, edgecolor="black", linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.5),
        width=0.5,
    )
    sns.stripplot(
        data=df_plot,
        x="Phenotype",
        y="Treatability_Index",
        color="black",
        alpha=0.35,
        size=4,
        jitter=0.2,
        ax=ax1,
    )
    ax1.set_title("Treatability Index (0-100) across Biological Phenotypes", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Biological Phenotype", fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Treatability Index (Higher = Convertible)", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, axis="y", color="#E5E7EB", linewidth=0.5, alpha=0.6)
    ax1.tick_params(axis="x", rotation=15)


def _plot_dabrafenib_hist_panel(ax2: plt.Axes, df_plot: pd.DataFrame) -> None:
    """Render Q2 Dabrafenib Sensitivity Index histogram panel."""
    df_arm_b = df_plot[df_plot["Treatment_Arm"] == "Arm B: Targeted Therapy"]
    if len(df_arm_b) > 0:
        has_var = df_arm_b["Dabrafenib_Sensitivity_Index"].nunique() > 3 and df_arm_b["Dabrafenib_Sensitivity_Index"].std() > 1e-3
        sns.histplot(
            data=df_arm_b,
            x="Dabrafenib_Sensitivity_Index",
            hue="Phenotype",
            palette=PHENOTYPE_PALETTE,
            kde=has_var,
            ax=ax2,
            element="step",
            alpha=0.5,
        )
        ax2.set_title("Q2 Dabrafenib Sensitivity Index in Arm B (Targeted Therapy)", fontsize=11.5, fontweight="bold")
        ax2.set_xlabel("Q2 Dabrafenib Sensitivity Index (0-100)", fontsize=10.5, fontweight="bold")
        ax2.set_ylabel("Patient Count", fontsize=10.5, fontweight="bold")
        ax2.grid(True, axis="y", color="#E5E7EB", linewidth=0.5, alpha=0.6)
    else:
        ax2.text(0.5, 0.5, "No Arm B Patients Identified", ha="center", va="center", fontsize=12)


def plot_treatability_distributions(df_assigned: pd.DataFrame, out_path: Path) -> None:
    """Plot distribution of Treatability Index and Q2 Dabrafenib Sensitivity Scores."""
    df_plot = df_assigned.copy()
    cluster_id_to_name = get_cluster_name_map(df_plot)
    df_plot["Phenotype"] = df_plot["Cluster_ID"].map(cluster_id_to_name)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def main() -> None:
    """Main execution function for Phase 7 treatability scoring."""
    print("=" * 80)
    print(f"Starting Phase 7: 3-Arm Decision Support & Treatability Scoring (Project root: {rel_path(PROJECT_ROOT)})")
    print("=" * 80)

    if not INPUT_PATIENTS.exists():
        raise FileNotFoundError(f"Missing patient dataset at {rel_path(INPUT_PATIENTS)}. Run Phase 3/4 first.")

    df_patients = pd.read_csv(INPUT_PATIENTS)
    print(f"Loaded patient dataset: {len(df_patients)} patients")

    # Join IMMUNOTHERAPY flag from full clinical file so we can distinguish
    # ICI-treated patients (known response) from prospective TCGA patients
    if INPUT_CLIN_FULL.exists() and "IMMUNOTHERAPY" not in df_patients.columns:
        df_clin_full = pd.read_csv(INPUT_CLIN_FULL, usecols=["SAMPLE_ID", "IMMUNOTHERAPY"])
        df_patients = df_patients.merge(df_clin_full, on="SAMPLE_ID", how="left")
        print(f"Joined IMMUNOTHERAPY flag from {rel_path(INPUT_CLIN_FULL)}")

    # Ensure ICI_Treated column is present (1 = known ICI-treated, 0 = prospective)
    if "IMMUNOTHERAPY" in df_patients.columns:
        df_patients["ICI_Treated"] = df_patients["IMMUNOTHERAPY"].fillna(0).astype(int)
    else:
        df_patients["ICI_Treated"] = 0
        print("Warning: IMMUNOTHERAPY column not found — ICI_Treated defaulted to 0 for all patients.")

    n_ici  = int(df_patients["ICI_Treated"].sum())
    n_pros = len(df_patients) - n_ici
    print(f"  ICI-treated (known response): {n_ici}")
    print(f"  Prospective (no ICI label):   {n_pros}")

    q2_weights = load_q2_dabrafenib_weights()

    if INPUT_EXPR.exists():
        df_expr = pd.read_csv(INPUT_EXPR)
        df_merged = pd.merge(df_patients, df_expr, on="SAMPLE_ID", how="left", suffixes=("", "_expr"))
        print(f"Merged expression matrix: {df_merged.shape[1]} total columns")
    else:
        print(f"Warning: Expression matrix not found at {rel_path(INPUT_EXPR)}. Using patient features only.")
        df_merged = df_patients.copy()

    # 1. Calculate Q2 Dabrafenib Sensitivity Score
    dabrafenib_scores = calculate_q2_dabrafenib_score(df_merged, q2_weights)

    # 2. Calculate Treatability Index & Components
    treatability_index, df_components = calculate_treatability_index(df_patients)

    # 3. Assign Treatment Arms (Arm A, Arm B, Arm C)
    df_assigned = assign_treatment_arms(df_patients, df_components, dabrafenib_scores)

    # Tag each patient as ICI-treated (retrospective validation) or prospective
    df_assigned["ICI_Treated"] = df_patients["ICI_Treated"].values

    # 4. Compute Recommendation Confidence Index and band
    df_assigned = compute_recommendation_confidence(df_assigned)
    conf_counts = df_assigned["Confidence_Band"].value_counts()
    print(f"Recommendation Confidence — High: {conf_counts.get('High', 0)}, "
          f"Moderate: {conf_counts.get('Moderate', 0)}, Low: {conf_counts.get('Low', 0)}")

    # 5. Save output CSVs
    out_csv = OUTPUT_DIR / "treatability_scores.csv"
    safe_save_csv(df_assigned, out_csv)
    print(f"\nSaved treatability scores and arm assignments to {rel_path(out_csv)}")

    # Arm summary split by ICI-treated vs prospective
    df_ici_sub  = df_assigned[df_assigned["ICI_Treated"] == 1]
    df_pros_sub = df_assigned[df_assigned["ICI_Treated"] == 0]
    print(f"  ICI-treated  (N={len(df_ici_sub)}):   "
          f"Arm A={int((df_ici_sub['Treatment_Arm'].str.startswith('Arm A')).sum())}, "
          f"Arm B={int((df_ici_sub['Treatment_Arm'].str.startswith('Arm B')).sum())}, "
          f"Arm C={int((df_ici_sub['Treatment_Arm'].str.startswith('Arm C')).sum())}")
    print(f"  Prospective  (N={len(df_pros_sub)}): "
          f"Arm A={int((df_pros_sub['Treatment_Arm'].str.startswith('Arm A')).sum())}, "
          f"Arm B={int((df_pros_sub['Treatment_Arm'].str.startswith('Arm B')).sum())}, "
          f"Arm C={int((df_pros_sub['Treatment_Arm'].str.startswith('Arm C')).sum())}")

    df_summary = df_assigned.copy()
    cluster_id_to_name = get_cluster_name_map(df_summary)
    df_summary["Phenotype"] = df_summary["Cluster_ID"].map(cluster_id_to_name)
    df_summary = df_summary.groupby(["Phenotype", "Treatment_Arm"]).size().reset_index(name="Patient_Count")
    out_sum_csv = OUTPUT_DIR / "arm_summary_metrics.csv"
    safe_save_csv(df_summary, out_sum_csv)
    print(f"Saved arm summary metrics to {rel_path(out_sum_csv)}")

    # 6. Generate plots
    plot_arm_file = PLOTS_DIR / "arm_assignment_breakdown.png"
    plot_arm_assignment_breakdown(df_assigned, plot_arm_file)

    plot_dist_file = PLOTS_DIR / "treatability_index_distribution.png"
    plot_treatability_distributions(df_assigned, plot_dist_file)

    print(f"\nGenerated Plots:\n  1. {rel_path(plot_arm_file)}\n  2. {rel_path(plot_dist_file)}")

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
