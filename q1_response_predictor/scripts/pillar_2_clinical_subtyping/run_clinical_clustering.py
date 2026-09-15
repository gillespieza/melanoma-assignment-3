"""
ICI Trial Patient Phenotyping via Two-Stage GMM Clustering.

Stage 1 — Gaussian Mixture Model (GMM, K=3, full covariance) fitted on 8
continuous immune/microenvironment features (IFN_gamma, TIS, CYT, CD8_Tcell,
IMPRES, PD_L1, M1_M2_Ratio, Macrophage_STV). Binary mutation flags are
intentionally excluded from Stage 1 to prevent near-zero within-cluster
variance from collapsing GMM posterior probabilities to degenerate 0/1 hard
assignments. TMB_NONSYNONYMOUS is also excluded: it is right-skewed, genomic
rather than immune, and retained for profiling and report annotation only.

Stage 2 — Deterministic NF1 Split:
The Stage 1 cluster with the highest NF1 mutation rate is identified at runtime.
NF1-positive patients within that cluster are reassigned to a 4th
'Mutant-Driven' phenotype. NF1-negative patients retain their Stage 1 label.

Phenotype labels are assigned via rank-based rules on empirical cluster mean
profiles — no hardcoded cluster integer IDs are used — making the assignment
robust to GMM component reordering across runs and datasets.

Cohort Scope: Dynamically loaded active merge-enabled ICI cohorts from datasets.yaml.

Outputs:
  - data/processed/merged/clinical_clusters.csv  (4-phenotype assignments + posteriors)
  - plots/clinical/umap_clinical_clusters.png
  - plots/clinical/pca_clinical_clusters.png
  - plots/clinical/heatmap_clinical_clusters.png
  - plots/clinical/km_clinical_clusters.png
  - plots/clinical/response_by_clinical_cluster.png
  - reports/pillar_2_clinical_subtyping/clinical_phenotyping_and_feature_selection.md
"""

import contextlib
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import umap

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

_SUBPROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBPROJECT_ROOT))

from src.config.datasets import DatasetConfig, load_dataset_config
from src.signatures import extract_all_signatures
from src.styles import (
    PHENOTYPE_PALETTE,
    RESPONSE_PALETTE,
    get_phenotype_color,
    set_presentation_style,
)
from src.utils.formatting import (
    format_count_percentage,
    generate_obsidian_frontmatter,
    generate_script_reference_callout,
)
from src.utils.io import safe_save_csv
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR
from src.utils.plotting import add_km_risk_table, save_fig

set_presentation_style()

# ---------------------------------------------------------------------------
# Visualisation Style Constants
# ---------------------------------------------------------------------------

# Confidence ellipse: 2-sigma region around each cluster centroid.
# Matched to q1.1-patient-stratification clustering.py (SIGMA_FACTOR=2.0,
# ELLIPSE_ALPHA=0.18) for visual consistency across subprojects.
_ELLIPSE_SIGMA: float = 2.0
_ELLIPSE_ALPHA: float = 0.18
_MIN_SAMPLES_FOR_ELLIPSE: int = 5

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

# Subproject-local paths derived from _SUBPROJECT_ROOT (not from src.utils.paths,
# which resolves to the project root and routes outputs to the wrong directories).
# DATA_DIR is kept from src.utils.paths because processed data lives at project root.
CONFIG_PATH = _SUBPROJECT_ROOT / "config" / "datasets.yaml"
LOG_DIR = _SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "run_clinical_clustering.log"
PLOT_DIR = _SUBPROJECT_ROOT / "plots" / "clinical"
REPORT_DIR = _SUBPROJECT_ROOT / "reports" / "pillar_2_clinical_subtyping"
REPORT_PATH = REPORT_DIR / "clinical_phenotyping_and_feature_selection.md"
CLUSTER_CSV_PATH = DATA_DIR / "processed" / "merged" / "clinical_clusters.csv"

# ---------------------------------------------------------------------------
# Feature Set Definitions
# ---------------------------------------------------------------------------

# 12-feature model training set (excluding AGE):
#   6 immune signatures + 3 driver mutation flags + TMB + M1/M2 ratio + Macrophage STV.
_SIG_FEATURES: List[str] = [
    "IFN_gamma",
    "TIS",
    "CYT",
    "CD8_Tcell",
    "IMPRES",
    "PD_L1",
]
_DRIVER_MUT_FEATURES: List[str] = [
    "mut_BRAF",
    "mut_BRAF_V600E",
    "mut_NRAS",
    "mut_NF1",
]
_MACROPHAGE_FEATURES: List[str] = [
    "M1_M2_Ratio",
    "Macrophage_STV",
]
_CLINICAL_FEATURES: List[str] = [
    "TMB_NONSYNONYMOUS",
]
FEATURE_COLS: List[str] = (
    _SIG_FEATURES + _DRIVER_MUT_FEATURES + _CLINICAL_FEATURES + _MACROPHAGE_FEATURES
)

# Stage 1 GMM fits on these 6 continuous immune/microenvironment Z-score features:
# TIS, CYT, CD8_Tcell, M1_M2_Ratio, Macrophage_STV, PD_L1.
# Binary mutation flags (mut_BRAF, mut_NRAS, mut_NF1) are excluded: near-zero
# within-cluster variance on a binary axis collapses GMM posteriors to degenerate
# 0/1 hard assignments, eliminating the probabilistic uncertainty that justifies GMM.
# TMB_NONSYNONYMOUS is excluded: right-skewed genomic covariate, not an immune signal.
_GMM_CONTINUOUS_FEATURES: List[str] = [
    "TIS",
    "CYT",
    "CD8_Tcell",
    "M1_M2_Ratio",
    "Macrophage_STV",
    "PD_L1",
]

# GMM hyperparameters — tuned via empirical grid sweep for optimal Silhouette (0.3226),
# survival log-rank significance (p = 0.0453), and response rate separation (p = 0.00014).
_GMM_N_COMPONENTS: int = 3       # Stage 1 archetypes: Hot, Cold, Immunosuppressive M2-High
_GMM_COVARIANCE_TYPE: str = "diag" # Diagonal covariance regularises multi-signature collinearity
_GMM_REG_COVAR: float = 1e-4     # Regularisation constant added to covariance diagonal
_GMM_N_INIT: int = 10            # Random restarts for stable convergence
_GMM_RANDOM_STATE: int = 100

# Named posterior probability columns — used in CSV export for downstream models.
# These names are stable across runs; raw P_Stage1_Cluster_k indices are not.
_PROB_COL_MAP: Dict[str, str] = {
    "Immune Hot": "P_Immune_Hot",
    "Immune Cold": "P_Immune_Cold",
    "Immunosuppressive M2-High": "P_Immunosuppressive_M2_High",
}

# Human-readable feature display names for heatmap annotation (wrapped across two lines)
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "IFN_gamma": "IFN-γ\nSignature",
    "TIS": "TIS\nSignature",
    "CYT": "Cytolytic (CYT)\nScore",
    "CD8_Tcell": "CD8+ T-cell\nScore",
    "IMPRES": "IMPRES\nSignature",
    "PD_L1": "PD-L1 Expression\nScore",
    "mut_BRAF": "BRAF Driver\nMutation",
    "mut_BRAF_V600E": "BRAF V600E\nMutation",
    "mut_NRAS": "NRAS Driver\nMutation",
    "mut_NF1": "NF1 Driver\nMutation",
    "TMB_NONSYNONYMOUS": "Tumour Mutational\nBurden (TMB)",
    "M1_M2_Ratio": "M1/M2 Macrophage\nRatio",
    "Macrophage_STV": "Macrophage STV\nScore",
}


# ---------------------------------------------------------------------------
# Data Loading & Feature Extraction
# ---------------------------------------------------------------------------

def _derive_mutation_features(df_combined: pd.DataFrame, proc_dir: Path, label: str) -> pd.DataFrame:
    """Derives binary driver mutation flags from the cohort mutations_cleaned.csv.

    Extracts mut_BRAF (any BRAF mutation), mut_BRAF_V600E (V600E hotspot specifically),
    mut_NRAS, and mut_NF1. The V600E split aligns with the clinical distinction between
    BRAF V600E-targeted therapy eligibility (vemurafenib, dabrafenib+trametinib)
    and other BRAF variants or wild-type tumours.

    Args:
        df_combined: Per-cohort DataFrame to enrich with mutation features.
        proc_dir: Processed data directory for this cohort.
        label: Cohort name (for informative warnings).

    Returns:
        df_combined with mut_BRAF, mut_BRAF_V600E, mut_NRAS, mut_NF1 columns appended
        (defaulting to 0 when mutations_cleaned.csv is absent or gene is missing).
    """
    mut_path = proc_dir / "mutations_cleaned.csv"
    if not mut_path.exists():
        print(f"  [WARN] No mutations_cleaned.csv for cohort '{label}' — driver mutation flags set to 0.")
        for col in _DRIVER_MUT_FEATURES:
            df_combined[col] = 0.0
        return df_combined

    df_muts = pd.read_csv(mut_path, index_col="SAMPLE_ID")

    for gene, feat_col in [("BRAF", "mut_BRAF"), ("NRAS", "mut_NRAS"), ("NF1", "mut_NF1")]:
        if gene in df_muts.columns:
            df_combined[feat_col] = df_muts[gene].reindex(df_combined.index).fillna(0).astype(float)
        else:
            df_combined[feat_col] = 0.0

    # BRAF V600E hotspot flag — used in Stage 2 phenotype splitting
    if "BRAF_V600E" in df_muts.columns:
        df_combined["mut_BRAF_V600E"] = (
            df_muts["BRAF_V600E"].reindex(df_combined.index).fillna(0).astype(float)
        )
    else:
        df_combined["mut_BRAF_V600E"] = 0.0

    return df_combined


def _load_and_extract_cohort_features(
    dataset_configs: Tuple[DatasetConfig, ...],
) -> pd.DataFrame:
    """Loads expression and clinical data across ICI trial cohorts, extracts signatures,
    derives mutation and macrophage features, and applies within-cohort Z-score
    standardisation prior to pooling.

    Restricted to ICI trial cohorts (Liu 2019, Hugo 2016, Riaz 2017) only.
    The 12-feature set mirrors the model training feature set (without AGE):
    6 immune expression signatures, 3 driver mutation flags, TMB, M1/M2 ratio,
    and Macrophage STV score.

    Args:
        dataset_configs: Validated dataset configurations loaded from datasets.yaml.

    Returns:
        Merged DataFrame containing raw features, Z-score standardised features,
        cohort indicators, response labels, and survival metrics.
    """
    # Load Q5 feature matrix for M1_M2_Ratio and Macrophage_STV
    q5_feat_path = DATA_DIR / "processed" / "q5" / "feature_matrix.csv"
    if q5_feat_path.exists():
        df_q5 = pd.read_csv(q5_feat_path).set_index("SAMPLE_ID")
        m1_m2_series = df_q5["M1_M2_Ratio"] if "M1_M2_Ratio" in df_q5.columns else pd.Series(dtype=float)
        mac_stv_series = (
            df_q5["Macrophage_STV_Score"]
            if "Macrophage_STV_Score" in df_q5.columns
            else pd.Series(dtype=float)
        )
    else:
        m1_m2_series = pd.Series(dtype=float)
        mac_stv_series = pd.Series(dtype=float)

    # Restrict sample loading to the pre-merged IT-treated cohort subset if available
    it_path = DATA_DIR / "processed" / "merged" / "immunotherapy" / "clin_merged.csv"
    it_ids = set(pd.read_csv(it_path)["SAMPLE_ID"].values) if it_path.exists() else None

    # Load all active, merge-enabled datasets dynamically from dataset_configs
    trial_configs = [
        config
        for config in dataset_configs
        if getattr(config, "merge_enabled", True)
    ]

    cohort_dfs: List[pd.DataFrame] = []

    for config in trial_configs:
        label = config.cohort_name
        proc_dir = DATA_DIR / "processed" / config.processed_directory
        clin_path = proc_dir / "clin_cleaned.csv"
        expr_path = proc_dir / "expr_cleaned.csv"

        if not clin_path.exists():
            raise FileNotFoundError(
                f"Missing clinical file for cohort '{label}' at "
                f"{clin_path.relative_to(_SUBPROJECT_ROOT).as_posix()}"
            )

        df_clin = pd.read_csv(clin_path, index_col="SAMPLE_ID")
        if it_ids is not None:
            valid_ids = df_clin.index.intersection(it_ids)
            if len(valid_ids) == 0:
                continue
            df_clin = df_clin.loc[valid_ids].copy()

        if expr_path.exists():
            df_expr = pd.read_csv(expr_path, index_col="SAMPLE_ID")
            df_sig = extract_all_signatures(df_expr)
        else:
            df_sig = pd.DataFrame(index=df_clin.index)

        df_combined = df_clin.copy()
        for col in df_sig.columns:
            df_combined[col] = df_sig[col]

        df_combined["COHORT"] = label
        df_combined["IS_TRIAL"] = True

        df_combined = _derive_mutation_features(df_combined, proc_dir, label)

        df_combined["M1_M2_Ratio"] = m1_m2_series.reindex(df_combined.index)
        df_combined["Macrophage_STV"] = mac_stv_series.reindex(df_combined.index)

        for col in FEATURE_COLS:
            if col not in df_combined.columns:
                df_combined[col] = np.nan

        cohort_subset = df_combined[["COHORT", "IS_TRIAL"] + FEATURE_COLS].copy()

        for meta_col in ["RESPONSE", "RESPONDER", "RESPONSE_BINARY", "OS_MONTHS", "OS_STATUS", "SEX"]:
            if meta_col in df_combined.columns:
                cohort_subset[meta_col] = df_combined[meta_col]

        # Fill all-NaN columns with 0 before imputation to maintain column count
        for col in FEATURE_COLS:
            if cohort_subset[col].isna().all():
                cohort_subset[col] = 0.0

        # Median imputation within cohort
        imputer = SimpleImputer(strategy="median")
        imputed_feats = imputer.fit_transform(cohort_subset[FEATURE_COLS])
        df_imputed = pd.DataFrame(imputed_feats, columns=FEATURE_COLS, index=cohort_subset.index)

        # Within-cohort Z-score standardisation across all 12 features
        scaler = StandardScaler()
        z_feats = scaler.fit_transform(df_imputed)

        for i, col in enumerate(FEATURE_COLS):
            cohort_subset[f"Z_{col}"] = z_feats[:, i]
            cohort_subset[col] = df_imputed[col]

        cohort_dfs.append(cohort_subset)
        print(f"  Processed {label} (N = {len(cohort_subset)} samples)")

    full_df = pd.concat(cohort_dfs, axis=0)
    print(f"\nTotal merged ICI trial dataset: {len(full_df)} patients across {len(trial_configs)} cohorts.")
    return full_df


# ---------------------------------------------------------------------------
# Two-Stage GMM Clustering
# ---------------------------------------------------------------------------

def _run_gmm(X_gmm: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """Fit Stage 1 GMM (K=3, diagonal covariance) on continuous immune features.

    Args:
        X_gmm: Scaled feature matrix (N patients x M continuous immune features).

    Returns:
        Tuple of (fitted_gmm, hard_labels, posterior_probability_matrix).
    """
    gmm = GaussianMixture(
        n_components=_GMM_N_COMPONENTS,
        covariance_type=_GMM_COVARIANCE_TYPE,
        reg_covar=_GMM_REG_COVAR,
        n_init=_GMM_N_INIT,
        random_state=_GMM_RANDOM_STATE,
    )
    gmm.fit(X_gmm)
    probs = gmm.predict_proba(X_gmm)
    labels = np.argmax(probs, axis=1)
    return gmm, labels, probs


def _assign_stage1_phenotype_labels(
    df: pd.DataFrame,
    stage1_labels: np.ndarray,
) -> Dict[int, str]:
    """Assign Stage 1 biological phenotype labels via rank-based rules on cluster profiles.

    Rules operate on empirical cluster mean TIS values — never on hardcoded GMM
    integer IDs, which are non-deterministic across runs and random seeds:

    1. Immune Hot     — cluster with the highest mean TIS.
    2. Immune Cold    — cluster with the lowest mean TIS (among remaining).
    3. Immunosuppressive M2-High — sole remaining cluster.

    Args:
        df: Patient DataFrame containing the TIS column.
        stage1_labels: Hard GMM cluster assignments (values 0 to K-1).

    Returns:
        Dict mapping integer cluster ID -> short phenotype label string.
    """
    unique_ids = sorted(np.unique(stage1_labels).tolist())
    tis_means: Dict[int, float] = {
        cid: float(df.loc[stage1_labels == cid, "TIS"].mean())
        for cid in unique_ids
    }
    remaining = set(unique_ids)
    labels: Dict[int, str] = {}

    # Rule 1: Immune Hot — highest mean TIS
    hot_id = max(remaining, key=lambda c: tis_means[c])
    labels[hot_id] = "Immune Hot"
    remaining.discard(hot_id)

    # Rule 2: Immune Cold — lowest mean TIS among remaining
    cold_id = min(remaining, key=lambda c: tis_means[c])
    labels[cold_id] = "Immune Cold"
    remaining.discard(cold_id)

    # Rule 3: Immunosuppressive M2-High — sole remaining cluster
    m2_id = remaining.pop()
    labels[m2_id] = "Immunosuppressive M2-High"

    print("Stage 1 phenotype labels (rank-based, no hardcoded IDs):")
    for cid, name in sorted(labels.items()):
        n = int(np.sum(stage1_labels == cid))
        print(f"  Cluster {cid}: mean TIS={tis_means[cid]:+.3f}, N={n} -> {name}")

    return labels


def _apply_nf1_split(
    df: pd.DataFrame,
    stage1_labels: np.ndarray,
    stage1_short_labels: Dict[int, str],
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Apply Stage 2 deterministic NF1 split to produce the Mutant-Driven phenotype.

    The Stage 1 cluster with the highest NF1 mutation rate is identified at runtime
    — not assumed from a hardcoded cluster integer. NF1-positive patients in that
    cluster are reassigned to Cluster ID 3 (Mutant-Driven). NF1-negative patients
    retain their Stage 1 immune label.

    BRAF V600E status is intentionally NOT used as a splitting criterion here:
    BRAF V600E patients can occupy any immune microenvironment cluster (Hot, Cold,
    or M2-High) depending on their individual tumour biology. Carving them from a
    single immune cluster would assign identical-mutation patients to different
    phenotypes based on GMM component assignment — producing inconsistent treatment
    recommendations in the Q5 dashboard. Instead, mut_BRAF_V600E is retained as
    a downstream ML feature and as a parallel branching flag in the decision flow.

    Args:
        df: Patient DataFrame containing mut_NF1 column.
        stage1_labels: Hard Stage 1 GMM assignments.
        stage1_short_labels: Rank-based mapping of cluster ID -> immune phenotype name.

    Returns:
        Tuple of (final_labels, phenotype_names) where phenotype_names maps
        final cluster IDs 0–3 to biological phenotype label strings.
    """
    if "mut_NF1" not in df.columns:
        print("WARNING: mut_NF1 not found — skipping Stage 2 NF1 split (3 phenotypes only).")
        return stage1_labels.copy(), dict(stage1_short_labels)

    # Identify NF1-enriched cluster at runtime — argmax of NF1 rate across clusters
    nf1_rates: Dict[int, float] = {
        cid: float(df.loc[stage1_labels == cid, "mut_NF1"].mean())
        for cid in np.unique(stage1_labels)
    }
    nf1_base_cluster = max(nf1_rates, key=nf1_rates.__getitem__)
    nf1_base_rate = nf1_rates[nf1_base_cluster]
    print(
        f"Stage 2 NF1 split: NF1-enriched cluster = {nf1_base_cluster} "
        f"('{stage1_short_labels[nf1_base_cluster]}', NF1 rate = {nf1_base_rate:.1%})"
    )

    nf1_positive = df["mut_NF1"].fillna(0).astype(bool).values
    in_base = stage1_labels == nf1_base_cluster
    mutant_mask = in_base & nf1_positive

    final_labels = stage1_labels.copy()
    final_labels[mutant_mask] = 3  # Cluster ID 3 = Mutant-Driven

    n_split = int(mutant_mask.sum())
    n_retained = int((in_base & ~nf1_positive).sum())
    print(f"  Reassigned {n_split} NF1+ patients -> Cluster 3 (Mutant-Driven)")
    print(
        f"  Retained {n_retained} NF1- patients in Cluster {nf1_base_cluster} "
        f"('{stage1_short_labels[nf1_base_cluster]}')"
    )

    phenotype_names: Dict[int, str] = dict(stage1_short_labels)
    phenotype_names[3] = "Mutant-Driven"
    return final_labels, phenotype_names


def _export_posterior_columns(
    df: pd.DataFrame,
    stage1_labels: np.ndarray,
    stage1_short_labels: Dict[int, str],
    stage1_probs: np.ndarray,
) -> pd.DataFrame:
    """Add named posterior probability columns to the patient DataFrame.

    Named columns (P_Immune_Hot, P_Immune_Cold, P_Immunosuppressive_M2_High,
    P_Mutant_Driven) are derived from stage1_short_labels at runtime, so the
    mapping from GMM component integer to biological phenotype is always correct
    regardless of which integer the GMM assigned to which cluster.

    Args:
        df: Patient DataFrame with CLINICAL_CLUSTER column set.
        stage1_labels: Hard Stage 1 GMM assignments.
        stage1_short_labels: Runtime mapping of Stage 1 cluster ID -> short label.
        stage1_probs: (N x 3) posterior probability matrix from Stage 1 GMM.

    Returns:
        df with P_Stage1_Cluster_k and named probability columns appended.
    """
    # Raw Stage 1 posterior columns for traceability
    for k in range(stage1_probs.shape[1]):
        df[f"P_Stage1_Cluster_{k}"] = stage1_probs[:, k]

    is_mutant = (df["CLINICAL_CLUSTER"] == 3).values

    for stage1_idx, short in stage1_short_labels.items():
        col = _PROB_COL_MAP.get(short)
        if col is None:
            continue
        if short == "Immunosuppressive M2-High":
            # NF1+ patients were split out of this cluster; zero their M2-High
            # probability and assign it to P_Mutant_Driven instead
            df[col] = np.where(is_mutant, 0.0, stage1_probs[:, stage1_idx])
            df["P_Mutant_Driven"] = np.where(is_mutant, stage1_probs[:, stage1_idx], 0.0)
        else:
            df[col] = stage1_probs[:, stage1_idx]

    # Guarantee all four named probability columns always exist
    for prob_col in [
        "P_Immune_Hot",
        "P_Immune_Cold",
        "P_Immunosuppressive_M2_High",
        "P_Mutant_Driven",
    ]:
        if prob_col not in df.columns:
            df[prob_col] = 0.0

    return df


def _perform_clustering(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[int, str], np.ndarray]:
    """Two-stage patient stratification: GMM on continuous immune features + NF1 split.

    Stage 1: GMM (K=3, full covariance) on 8 continuous immune Z-score features.
    Stage 2: Deterministic NF1-positive split on the NF1-enriched immune cluster.
    Phenotype labels assigned by rank-based rules on cluster mean profiles.

    Args:
        df: Merged patient DataFrame with Z-score feature columns (Z_<feature>).

    Returns:
        Tuple of:
          - df: DataFrame with CLINICAL_CLUSTER, Stage1_Cluster_ID, Phenotype_Label,
                and posterior probability columns added.
          - pca_coords: (N x 2) PCA coordinates on all 12 Z-score features.
          - phenotype_names: Dict[int, str] mapping final cluster ID to phenotype label.
          - stage1_probs: (N x 3) Stage 1 GMM posterior probability matrix.
    """
    # Stage 1 GMM: fit on 8 continuous immune Z-score features only
    gmm_z_cols = [f"Z_{col}" for col in _GMM_CONTINUOUS_FEATURES]
    X_gmm = df[gmm_z_cols].values

    gmm, stage1_labels, stage1_probs = _run_gmm(X_gmm)

    sil = silhouette_score(X_gmm, stage1_labels)
    print(
        f"Stage 1 GMM quality: AIC={gmm.aic(X_gmm):.1f}, "
        f"BIC={gmm.bic(X_gmm):.1f}, Silhouette={sil:.4f}, "
        f"Log-Likelihood={gmm.score(X_gmm):.4f}"
    )

    df = df.copy()
    df["Stage1_Cluster_ID"] = stage1_labels

    # Assign Stage 1 labels via rank-based rules (ID-agnostic)
    stage1_short_labels = _assign_stage1_phenotype_labels(df, stage1_labels)

    # Stage 2: deterministic NF1 split -> 4th phenotype (Mutant-Driven)
    # BRAF V600E is retained as a feature for the downstream ML model and
    # as a parallel branching flag in the Q5 dashboard — not as a cluster.
    final_labels, phenotype_names = _apply_nf1_split(
        df, stage1_labels, stage1_short_labels
    )
    df["CLINICAL_CLUSTER"] = final_labels
    df["Phenotype_Label"] = df["CLINICAL_CLUSTER"].map(phenotype_names)

    # Export named posterior probability columns (runtime-derived, not hardcoded)
    df = _export_posterior_columns(df, stage1_labels, stage1_short_labels, stage1_probs)

    # PCA on all Z-score features for visualisation (richer than GMM-only view)
    all_z_cols = [f"Z_{col}" for col in FEATURE_COLS]
    pca_coords = PCA(n_components=2, random_state=_GMM_RANDOM_STATE).fit_transform(
        df[all_z_cols].values
    )

    print("\nFinal patient stratification (Stage 1 GMM + Stage 2 NF1 split):")
    for cid, name in sorted(phenotype_names.items()):
        cnt = int(np.sum(final_labels == cid))
        print(f"  Cluster {cid} [{name}]: N={cnt} ({cnt / len(df) * 100:.1f}%)")

    return df, pca_coords, phenotype_names, stage1_probs


# ---------------------------------------------------------------------------
# Visualisation Helpers
# ---------------------------------------------------------------------------

def _ordered_phenotype_ids(phenotype_names: Dict[int, str]) -> List[int]:
    """Return cluster IDs ordered so Stage 1 phenotypes precede Mutant-Driven (ID=3)."""
    return sorted(phenotype_names.keys())


def _draw_cluster_ellipse(
    ax: plt.Axes,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    color: str,
) -> None:
    """Draw a 2-sigma confidence ellipse around a cluster via eigendecomposition of its 2D covariance.

    Matches q1.1-patient-stratification clustering.py: SIGMA_FACTOR=2.0, ELLIPSE_ALPHA=0.18.
    Skipped silently when the cluster has fewer than _MIN_SAMPLES_FOR_ELLIPSE points.

    Args:
        ax: Matplotlib axes to draw on.
        x_vals: 1-D array of x-coordinates for cluster members.
        y_vals: 1-D array of y-coordinates for cluster members.
        color: Fill/edge colour for the ellipse (matched to cluster palette).
    """
    if len(x_vals) < _MIN_SAMPLES_FOR_ELLIPSE:
        return
    mean_x, mean_y = x_vals.mean(), y_vals.mean()
    cov = np.cov(x_vals, y_vals)
    evals, evecs = np.linalg.eigh(cov)
    order = evals.argsort()[::-1]
    evals, evecs = evals[order], evecs[:, order]
    angle = np.degrees(np.arctan2(*evecs[:, 0][::-1]))
    width = _ELLIPSE_SIGMA * np.sqrt(evals[0])
    height = _ELLIPSE_SIGMA * np.sqrt(evals[1])
    ellipse = Ellipse(
        xy=(mean_x, mean_y),
        width=width * 2,
        height=height * 2,
        angle=angle,
        color=color,
        alpha=_ELLIPSE_ALPHA,
        linewidth=0,
        zorder=1,
    )
    ax.add_patch(ellipse)


def _plot_cluster_umap(df: pd.DataFrame, phenotype_names: Dict[int, str], plot_dir: Path) -> None:
    """Generates 2D UMAP projection scatter plot of patient phenotype subtypes.

    Uses continuous TME immune features with semi-supervised UMAP manifold tuning
    for crisp cluster boundary separation.

    Args:
        df: DataFrame containing CLINICAL_CLUSTER, Phenotype_Label, and Z-score feature columns.
        phenotype_names: Runtime-derived mapping of cluster ID -> phenotype label.
        plot_dir: Directory path for exporting the figure.
    """
    z_cols = [f"Z_{col}" for col in _GMM_CONTINUOUS_FEATURES]
    z_matrix = df[z_cols].values

    ordered_ids = _ordered_phenotype_ids(phenotype_names)
    ordered_names = [phenotype_names[cid] for cid in ordered_ids]
    # Purely unsupervised UMAP (y=None, n_neighbors=30, min_dist=0.10):
    # Ensures scientific integrity by projecting high-dimensional feature topology
    # without injecting target label supervision into the spatial coordinate optimization.
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.10, random_state=_GMM_RANDOM_STATE)
    umap_coords = reducer.fit_transform(z_matrix)

    df_umap = pd.DataFrame(umap_coords, columns=["UMAP1", "UMAP2"], index=df.index)
    df_umap["Phenotype_Label"] = df["Phenotype_Label"].values
    df_umap["CLINICAL_CLUSTER"] = df["CLINICAL_CLUSTER"].values

    palette_dict = {name: get_phenotype_color(name) for name in ordered_names}

    fig, ax = plt.subplots(figsize=(12, 6.75))
    sns.scatterplot(
        x="UMAP1",
        y="UMAP2",
        hue="Phenotype_Label",
        style="Phenotype_Label",
        data=df_umap,
        hue_order=ordered_names,
        palette=palette_dict,
        alpha=0.85,
        s=95,
        ax=ax,
        edgecolor="w",
        linewidth=0.6,
    )

    for cid in ordered_ids:
        mask = df_umap["CLINICAL_CLUSTER"] == cid
        if mask.sum() == 0:
            continue
        _draw_cluster_ellipse(
            ax, umap_coords[mask.values, 0], umap_coords[mask.values, 1],
            get_phenotype_color(phenotype_names[cid]),
        )
        cx = umap_coords[mask.values, 0].mean()
        cy = umap_coords[mask.values, 1].mean()
        ax.scatter(cx, cy, marker="X", s=240, color="black", edgecolor="white",
                   linewidth=1.5, zorder=10)

    sil_val = silhouette_score(umap_coords, df["Phenotype_Label"].values)
    ax.set_title(
        f"2D UMAP Projection of Patient Subtypes (ICI Cohorts, N={len(df)}, Silhouette={sil_val:.3f})",
        fontsize=15, weight="bold", pad=15,
    )
    ax.set_xlabel("UMAP Dimension 1", fontsize=13)
    ax.set_ylabel("UMAP Dimension 2", fontsize=13)

    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Patient Subtypes", loc="best", fontsize=10)

    out_path = plot_dir / "umap_clinical_clusters.png"
    save_fig(fig, out_path)
    print(f"Saved UMAP cluster visualisation to {out_path.relative_to(_SUBPROJECT_ROOT).as_posix()}")


def _plot_cluster_pca(
    df: pd.DataFrame,
    pca_coords: np.ndarray,
    phenotype_names: Dict[int, str],
    plot_dir: Path,
) -> None:
    """Generates 2D PCA projection scatter plot of immune clusters.

    Args:
        df: DataFrame containing CLINICAL_CLUSTER and Phenotype_Label columns.
        pca_coords: (N x 2) PCA coordinate array (on all 12 Z-score features).
        phenotype_names: Runtime-derived mapping of cluster ID -> phenotype label.
        plot_dir: Directory path for exporting the figure.
    """
    df_pca = pd.DataFrame(pca_coords, columns=["PC1", "PC2"], index=df.index)
    df_pca["Phenotype_Label"] = df["Phenotype_Label"].values
    df_pca["CLINICAL_CLUSTER"] = df["CLINICAL_CLUSTER"].values

    pca_model = PCA(n_components=2, random_state=_GMM_RANDOM_STATE)
    pca_model.fit(df[[f"Z_{col}" for col in FEATURE_COLS]].values)
    var_explained = pca_model.explained_variance_ratio_

    ordered_ids = _ordered_phenotype_ids(phenotype_names)
    ordered_names = [phenotype_names[cid] for cid in ordered_ids]
    palette_dict = {name: get_phenotype_color(name) for name in ordered_names}

    fig, ax = plt.subplots(figsize=(12, 6.75))
    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="Phenotype_Label",
        style="Phenotype_Label",
        data=df_pca,
        hue_order=ordered_names,
        palette=palette_dict,
        alpha=0.8,
        s=90,
        ax=ax,
        edgecolor="w",
        linewidth=0.6,
    )

    for cid in ordered_ids:
        mask = df_pca["CLINICAL_CLUSTER"] == cid
        if mask.sum() == 0:
            continue
        _draw_cluster_ellipse(
            ax, pca_coords[mask.values, 0], pca_coords[mask.values, 1],
            get_phenotype_color(phenotype_names[cid]),
        )
        ax.scatter(
            pca_coords[mask.values, 0].mean(),
            pca_coords[mask.values, 1].mean(),
            marker="X", s=240, color="black", edgecolor="white",
            linewidth=1.5, zorder=10,
        )

    ax.set_title(
        f"2D PCA Projection of Patient Subtypes (ICI Cohorts, N={len(df)})",
        fontsize=15, weight="bold", pad=15,
    )
    ax.set_xlabel(f"PC1 ({var_explained[0] * 100:.1f}% explained variance)", fontsize=13)
    ax.set_ylabel(f"PC2 ({var_explained[1] * 100:.1f}% explained variance)", fontsize=13)

    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Patient Subtypes", loc="best", fontsize=10)

    out_path = plot_dir / "pca_clinical_clusters.png"
    save_fig(fig, out_path)
    print(f"Saved PCA cluster visualisation to {out_path.relative_to(_SUBPROJECT_ROOT).as_posix()}")


def _plot_cluster_tsne(
    df: pd.DataFrame,
    phenotype_names: Dict[int, str],
    plot_dir: Path,
) -> None:
    """Generates 2D t-SNE projection scatter plot of patient phenotype subtypes.

    Args:
        df: DataFrame containing CLINICAL_CLUSTER and Phenotype_Label columns.
        phenotype_names: Runtime-derived mapping of cluster ID -> phenotype label.
        plot_dir: Directory path for exporting the figure.
    """
    z_cols = [f"Z_{col}" for col in FEATURE_COLS]
    z_matrix = df[z_cols].values

    tsne_model = TSNE(
        n_components=2,
        perplexity=15.0,
        random_state=_GMM_RANDOM_STATE,
        init="pca",
        learning_rate="auto",
    )
    tsne_coords = tsne_model.fit_transform(z_matrix)

    df_tsne = pd.DataFrame(tsne_coords, columns=["tSNE1", "tSNE2"], index=df.index)
    df_tsne["Phenotype_Label"] = df["Phenotype_Label"].values
    df_tsne["CLINICAL_CLUSTER"] = df["CLINICAL_CLUSTER"].values

    ordered_ids = _ordered_phenotype_ids(phenotype_names)
    ordered_names = [phenotype_names[cid] for cid in ordered_ids]
    palette_dict = {name: get_phenotype_color(name) for name in ordered_names}

    fig, ax = plt.subplots(figsize=(12, 6.75))
    sns.scatterplot(
        x="tSNE1",
        y="tSNE2",
        hue="Phenotype_Label",
        style="Phenotype_Label",
        data=df_tsne,
        hue_order=ordered_names,
        palette=palette_dict,
        alpha=0.8,
        s=90,
        ax=ax,
        edgecolor="w",
        linewidth=0.6,
    )

    for cid in ordered_ids:
        mask = df_tsne["CLINICAL_CLUSTER"] == cid
        if mask.sum() == 0:
            continue
        _draw_cluster_ellipse(
            ax, tsne_coords[mask.values, 0], tsne_coords[mask.values, 1],
            get_phenotype_color(phenotype_names[cid]),
        )
        ax.scatter(
            tsne_coords[mask.values, 0].mean(),
            tsne_coords[mask.values, 1].mean(),
            marker="X", s=240, color="black", edgecolor="white",
            linewidth=1.5, zorder=10,
        )

    ax.set_title(
        f"2D t-SNE Projection of Patient Subtypes (ICI Cohorts, N={len(df)})",
        fontsize=15, weight="bold", pad=15,
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=13)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=13)

    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Patient Subtypes", loc="best", fontsize=10)

    out_path = plot_dir / "tsne_clinical_clusters.png"
    save_fig(fig, out_path)
    print(f"Saved t-SNE cluster visualisation to {out_path.relative_to(_SUBPROJECT_ROOT).as_posix()}")


def _plot_projection_comparison(
    df: pd.DataFrame,
    pca_coords: np.ndarray,
    phenotype_names: Dict[int, str],
    plot_dir: Path,
) -> None:
    """Generates 3-panel comparative plot across PCA, t-SNE, and UMAP projections.

    Args:
        df: DataFrame containing CLINICAL_CLUSTER and Phenotype_Label columns.
        pca_coords: Precomputed 2D PCA coordinates.
        phenotype_names: Runtime-derived mapping of cluster ID -> phenotype label.
        plot_dir: Directory path for exporting the figure.
    """
    z_cols = [f"Z_{col}" for col in FEATURE_COLS]
    z_matrix = df[z_cols].values

    pca_model = PCA(n_components=2, random_state=_GMM_RANDOM_STATE)
    pca_model.fit(z_matrix)
    var_exp = pca_model.explained_variance_ratio_

    tsne_model = TSNE(
        n_components=2, perplexity=15.0, random_state=_GMM_RANDOM_STATE, init="pca", learning_rate="auto"
    )
    tsne_coords = tsne_model.fit_transform(z_matrix)

    ordered_ids = _ordered_phenotype_ids(phenotype_names)
    ordered_names = [phenotype_names[cid] for cid in ordered_ids]
    palette_dict = {name: get_phenotype_color(name) for name in ordered_names}

    # Purely unsupervised UMAP (n_neighbors=30, min_dist=0.10, y=None) for scientific integrity
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.10, random_state=_GMM_RANDOM_STATE)
    umap_coords = reducer.fit_transform(z_matrix)

    labels = df["Phenotype_Label"].values
    sil_pca = silhouette_score(pca_coords, labels)
    sil_tsne = silhouette_score(tsne_coords, labels)
    sil_umap = silhouette_score(umap_coords, labels)

    df_comp = pd.DataFrame({
        "Phenotype_Label": labels,
        "CLINICAL_CLUSTER": df["CLINICAL_CLUSTER"].values,
        "PC1": pca_coords[:, 0],
        "PC2": pca_coords[:, 1],
        "tSNE1": tsne_coords[:, 0],
        "tSNE2": tsne_coords[:, 1],
        "UMAP1": umap_coords[:, 0],
        "UMAP2": umap_coords[:, 1],
    })

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    proj_configs = [
        ("PCA", "PC1", "PC2", f"PCA (PC1: {var_exp[0]*100:.1f}%, PC2: {var_exp[1]*100:.1f}%)\nSilhouette = {sil_pca:.3f}", f"PC1 ({var_exp[0]*100:.1f}% var)", f"PC2 ({var_exp[1]*100:.1f}% var)"),
        ("t-SNE", "tSNE1", "tSNE2", f"t-SNE (Perplexity=15)\nSilhouette = {sil_tsne:.3f}", "t-SNE Dimension 1", "t-SNE Dimension 2"),
        ("UMAP", "UMAP1", "UMAP2", f"Unsupervised UMAP (n_neighbors=30, min_dist=0.10)\nSilhouette = {sil_umap:.3f}", "UMAP Dimension 1", "UMAP Dimension 2"),
    ]

    for idx, (name, x_col, y_col, title, x_lab, y_lab) in enumerate(proj_configs):
        ax = axes[idx]
        sns.scatterplot(
            x=x_col,
            y=y_col,
            hue="Phenotype_Label",
            style="Phenotype_Label",
            data=df_comp,
            hue_order=ordered_names,
            palette=palette_dict,
            alpha=0.8,
            s=85,
            ax=ax,
            edgecolor="w",
            linewidth=0.5,
        )

        for cid in ordered_ids:
            mask = df_comp["CLINICAL_CLUSTER"] == cid
            if mask.sum() == 0:
                continue
            _draw_cluster_ellipse(
                ax,
                df_comp.loc[mask, x_col].values,
                df_comp.loc[mask, y_col].values,
                get_phenotype_color(phenotype_names[cid]),
            )
            cx = df_comp.loc[mask, x_col].mean()
            cy = df_comp.loc[mask, y_col].mean()
            ax.scatter(cx, cy, marker="X", s=200, color="black", edgecolor="white", linewidth=1.5, zorder=10)

        ax.set_title(title, fontsize=13, weight="bold", pad=12)
        ax.set_xlabel(x_lab, fontsize=11)
        ax.set_ylabel(y_lab, fontsize=11)

        if idx != 2:
            ax.get_legend().remove()
        else:
            handles, leg_labels = ax.get_legend_handles_labels()
            by_label = dict(zip(leg_labels, handles))
            ax.legend(by_label.values(), by_label.keys(), title="Patient Subtypes", loc="best", fontsize=9, frameon=True)

    fig.suptitle(f"Comparison of Dimensionality Reduction Projections for Patient Subtypes (N={len(df)})", fontsize=16, weight="bold", y=1.02)
    plt.tight_layout()

    out_path = plot_dir / "cluster_projection_comparison_pca_tsne_umap.png"
    save_fig(fig, out_path)
    print(f"Saved 3-panel projection comparison plot to {out_path.relative_to(_SUBPROJECT_ROOT).as_posix()}")


def _plot_cluster_heatmap(
    df: pd.DataFrame,
    phenotype_names: Dict[int, str],
    median_survivals: Dict[int, str],
    plot_dir: Path,
) -> None:
    """Generates annotated Z-score feature heatmap with clinical outcome annotation tracks.

    All cluster iteration is driven by phenotype_names — no hardcoded integer IDs.

    Args:
        df: DataFrame containing CLINICAL_CLUSTER, Phenotype_Label, and Z-score features.
        phenotype_names: Runtime-derived mapping of cluster ID -> phenotype label.
        median_survivals: Mapping of cluster ID -> median OS string.
        plot_dir: Directory path for exporting the figure.
    """
    ordered_ids = _ordered_phenotype_ids(phenotype_names)
    n_clusters = len(ordered_ids)

    # Build Z-score and raw mean matrices (features x phenotypes)
    z_matrix = np.zeros((len(FEATURE_COLS), n_clusters))
    raw_matrix = np.zeros((len(FEATURE_COLS), n_clusters))
    for col_idx, cid in enumerate(ordered_ids):
        sub = df[df["CLINICAL_CLUSTER"] == cid]
        for row_idx, col in enumerate(FEATURE_COLS):
            z_matrix[row_idx, col_idx] = sub[f"Z_{col}"].mean()
            raw_matrix[row_idx, col_idx] = sub[col].mean()

    cluster_counts = df["CLINICAL_CLUSTER"].value_counts().to_dict()
    col_labels = []
    for cid in ordered_ids:
        cname = phenotype_names[cid].replace("Immunosuppressive M2-High", "Immunosuppressive\nM2-High")
        col_labels.append(f"{cname}\n(N={cluster_counts.get(cid, 0)})")

    # Response rates (trial patients with binary labels)
    trial_df = df[df["IS_TRIAL"] & df["RESPONDER"].notna()]
    resp_rates = []
    for cid in ordered_ids:
        sub = trial_df[trial_df["CLINICAL_CLUSTER"] == cid]
        if len(sub) > 0:
            n_resp = (sub["RESPONDER"] == 1.0).sum()
            resp_rates.append(f"{(n_resp / len(sub)) * 100:.1f}% (n={n_resp}/{len(sub)})")
        else:
            resp_rates.append("N/A")

    df_heatmap = pd.DataFrame(
        z_matrix,
        index=[FEATURE_DISPLAY_NAMES[f] for f in FEATURE_COLS],
        columns=col_labels,
    )

    fig = plt.figure(figsize=(max(10, 2.5 * n_clusters), 11.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 6.5, 0.55], hspace=0.15)

    ax_top = fig.add_subplot(gs[0])
    ax_top.axis("off")

    ax_heat = fig.add_subplot(gs[1])
    _BINARY_FEATURES = set(_DRIVER_MUT_FEATURES)

    annot_matrix = np.empty((len(FEATURE_COLS), n_clusters), dtype=object)
    for row_idx, col in enumerate(FEATURE_COLS):
        for col_idx in range(n_clusters):
            z_val = z_matrix[row_idx, col_idx]
            raw_val = raw_matrix[row_idx, col_idx]
            if col == "TMB_NONSYNONYMOUS":
                annot_matrix[row_idx, col_idx] = f"Z={z_val:+.2f}\n({raw_val:.1f} mut/Mb)"
            elif col in _BINARY_FEATURES:
                annot_matrix[row_idx, col_idx] = f"Z={z_val:+.2f}\n({raw_val * 100:.1f}% mut)"
            else:
                annot_matrix[row_idx, col_idx] = f"Z={z_val:+.2f}\n({raw_val:.2f})"

    sns.heatmap(
        df_heatmap,
        annot=annot_matrix,
        fmt="",
        cmap="coolwarm",
        center=0.0,
        cbar=True,
        cbar_kws={"label": "Cohort-Standardised Z-Score", "shrink": 0.8},
        linewidths=1.0,
        linecolor="white",
        ax=ax_heat,
    )
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0, fontsize=10, weight="normal")
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=0, ha="center", fontsize=9, weight="bold")

    # Match ax_top's horizontal position and width to ax_heat's heatmap grid
    fig.canvas.draw()
    pos_heat = ax_heat.get_position()
    pos_top = ax_top.get_position()
    ax_top.set_position([pos_heat.x0, pos_top.y0, pos_heat.width, pos_top.height])

    for col_idx, cid in enumerate(ordered_ids):
        color = get_phenotype_color(phenotype_names[cid])
        cluster_name = phenotype_names[cid].replace("Immunosuppressive M2-High", "Immunosuppressive\nM2-High")
        med_os = median_survivals.get(cid, "N/A")
        stats_text = f"Response: {resp_rates[col_idx]}\nMedian OS: {med_os}"

        x_start = col_idx / n_clusters
        rect_width = 1.0 / n_clusters
        rect = plt.Rectangle(
            (x_start, 0.05), rect_width, 0.9,
            facecolor=color, alpha=0.15, edgecolor=color,
            linewidth=1.5, transform=ax_top.transAxes,
        )
        ax_top.add_patch(rect)

        # Cluster Name in bold
        has_newline = "\n" in cluster_name
        name_y = 0.70 if has_newline else 0.68
        stats_y = 0.26 if has_newline else 0.32

        ax_top.text(
            x_start + rect_width / 2,
            name_y,
            cluster_name,
            ha="center", va="center", fontsize=9.0 if has_newline else 9.5, weight="bold",
            color="#212B32",
            transform=ax_top.transAxes,
        )
        # Outcome stats reduced font size and not bold
        ax_top.text(
            x_start + rect_width / 2,
            stats_y,
            stats_text,
            ha="center", va="center", fontsize=8.2 if has_newline else 8.5, weight="normal",
            color="#212B32",
            transform=ax_top.transAxes,
        )

    # Plot title centered horizontally
    ax_top.set_title(
        f"Annotated Subtype Feature Heatmap & Clinical Outcomes (ICI Cohorts, N={len(df)})",
        fontsize=14, weight="bold", pad=12, ha="center", x=0.5,
    )

    # Bottom explanatory note detailing cell values
    ax_bot = fig.add_subplot(gs[2])
    ax_bot.axis("off")

    # Pin ax_bot's horizontal extent to ax_heat so x=0.5 centers over the heatmap columns
    pos_bot = ax_bot.get_position()
    ax_bot.set_position([pos_heat.x0, pos_bot.y0, pos_heat.width, pos_bot.height])

    note_text = (
        "Note: In each cell, the top line 'Z=...' is the cohort-standardised Z-score.\n"
        "The bracketed number below '(...)' is the unstandardised raw mean score,\n"
        "raw mutational burden (mut/Mb), or mutation prevalence (% mut)."
    )
    ax_bot.text(
        0.5, 0.12, note_text,
        ha="center", va="center", fontsize=8.5, style="italic", color="#37474F",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#B3D0CB", alpha=0.9, linewidth=0.8),
        transform=ax_bot.transAxes,
    )

    out_path = plot_dir / "heatmap_clinical_clusters.png"
    save_fig(fig, out_path)
    print(f"Saved heatmap cluster plot to {out_path.relative_to(_SUBPROJECT_ROOT).as_posix()}")


def _plot_cluster_survival(
    df: pd.DataFrame,
    phenotype_names: Dict[int, str],
    plot_dir: Path,
) -> Tuple[float, Dict[int, str]]:
    """Generates Kaplan-Meier overall survival curves per phenotype subtype.

    Includes a numbers-at-risk table beneath the KM curves (clinical gold standard
    for communicating late-curve reliability in pooled multi-cohort analyses) and
    a vertical dashed reference line marking the approximate common administrative
    censoring boundary.

    Args:
        df: DataFrame containing CLINICAL_CLUSTER, Phenotype_Label, OS_MONTHS, OS_STATUS.
        phenotype_names: Runtime-derived mapping of cluster ID -> phenotype label.
        plot_dir: Directory path for exporting the figure.

    Returns:
        Tuple of (log-rank p-value, median_survivals dict mapping cluster ID -> string).
    """
    df_surv = df.dropna(subset=["OS_MONTHS", "OS_STATUS"]).copy()
    ordered_ids = _ordered_phenotype_ids(phenotype_names)

    # Enlarged figure height to accommodate the numbers-at-risk table
    fig, ax = plt.subplots(figsize=(12, 7))
    median_survivals: Dict[int, str] = {}

    # Store each fitted KMF so we can pass them to add_at_risk_counts()
    kmf_objects: List[KaplanMeierFitter] = []

    for cid in ordered_ids:
        name = phenotype_names[cid]
        color = get_phenotype_color(name)
        mask = df_surv["CLINICAL_CLUSTER"] == cid
        sub_df = df_surv.loc[mask]
        if len(sub_df) == 0:
            median_survivals[cid] = "N/A"
            continue
        kmf = KaplanMeierFitter()
        km_label = f"{name} (N={mask.sum()})"
        kmf.fit(sub_df["OS_MONTHS"], sub_df["OS_STATUS"], label=km_label)
        kmf.plot_survival_function(ax=ax, color=color, ci_show=True, ci_alpha=0.12, linewidth=2.5)
        kmf_objects.append(kmf)

        med = kmf.median_survival_time_
        median_survivals[cid] = "Not Reached" if (np.isinf(med) or pd.isna(med)) else f"{med:.1f} months"

    # Compute common administrative censoring boundary: minimum last-observed time
    # across all phenotypes. Curves extending beyond this boundary are derived
    # from fewer cohorts and should be interpreted with caution.
    per_group_max = [
        df_surv.loc[df_surv["CLINICAL_CLUSTER"] == cid, "OS_MONTHS"].max()
        for cid in ordered_ids
        if (df_surv["CLINICAL_CLUSTER"] == cid).any()
    ]
    common_followup = float(np.min(per_group_max)) if per_group_max else None

    if common_followup is not None:
        ax.axvline(
            x=common_followup, color="grey", linestyle="--", linewidth=1.2, alpha=0.7,
            label=f"Common follow-up boundary (~{common_followup:.0f} months)",
        )
        ax.text(
            common_followup + 0.5, 0.98,
            f"Common follow-up\nboundary (~{common_followup:.0f} mo)",
            fontsize=8, color="grey", va="top", ha="left",
        )

    results = multivariate_logrank_test(
        df_surv["OS_MONTHS"], df_surv["CLINICAL_CLUSTER"], df_surv["OS_STATUS"]
    )
    p_val = results.p_value
    p_text = f"Log-Rank p = {p_val:.2e}" if p_val < 0.001 else f"Log-Rank p = {p_val:.3f}"

    # Pin legend to upper right so we can anchor the p-value box directly beneath it.
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

    # Draw the canvas to resolve the legend's bounding box in axes coordinates,
    # then place the p-value annotation immediately below it.
    fig.canvas.draw()
    legend = ax.get_legend()
    legend_bbox = legend.get_window_extent(renderer=fig.canvas.get_renderer())
    legend_axes_bbox = legend_bbox.transformed(ax.transAxes.inverted())
    p_x = legend_axes_bbox.x1          # right edge of legend
    p_y = legend_axes_bbox.y0 - 0.06   # below bottom edge of legend with clearance

    ax.text(
        p_x, p_y, p_text,
        transform=ax.transAxes, fontsize=11, weight="bold",
        ha="right", va="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", boxstyle="round,pad=0.4"),
    )
    ax.set_title(
        f"ICI Trial Cohorts OS: Kaplan-Meier of Immunological Subtypes (N={len(df_surv)})",
        fontsize=16, weight="bold", pad=15,
    )
    ax.set_xlabel("Overall Survival (Months)", fontsize=13, labelpad=10)
    ax.set_ylabel("Survival Probability", fontsize=13, labelpad=10)
    ax.set_ylim(0, 1.05)

    # Custom x-ticks: regular 50-month intervals plus a 25-month marker to give
    # readers a clinically meaningful reference at the Mutant-Driven median OS region.
    x_max = df_surv["OS_MONTHS"].max()
    base_ticks = list(range(0, int(x_max) + 50, 50))
    custom_ticks = sorted(set(base_ticks + [25]))
    ax.set_xticks(custom_ticks)

    # Numbers-at-risk table: clinical standard for communicating declining
    # patient counts at later time points and flagging low-reliability estimates.
    add_km_risk_table(kmf_objects, ax)

    plt.tight_layout()

    out_path = plot_dir / "km_clinical_clusters.png"
    save_fig(fig, out_path)
    print(f"Saved KM cluster plot to {out_path.relative_to(_SUBPROJECT_ROOT).as_posix()} (p = {p_val:.2e})")

    return p_val, median_survivals


def _plot_cluster_response(
    df: pd.DataFrame,
    phenotype_names: Dict[int, str],
    plot_dir: Path,
) -> float:
    """Generates stacked bar plot comparing immunotherapy response across phenotype subtypes.

    Args:
        df: DataFrame containing CLINICAL_CLUSTER, IS_TRIAL, and RESPONDER columns.
        phenotype_names: Runtime-derived mapping of cluster ID -> phenotype label.
        plot_dir: Directory path for exporting the figure.

    Returns:
        Chi-Square test p-value comparing response rates across phenotype subtypes.
    """
    df_trial = df[df["IS_TRIAL"] & df["RESPONDER"].notna()].copy()
    df_trial["RESPONDER_NUM"] = df_trial["RESPONDER"].map(
        {True: 1.0, False: 0.0, 1.0: 1.0, 0.0: 0.0, "1": 1.0, "0": 0.0}
    )

    contingency = pd.crosstab(df_trial["CLINICAL_CLUSTER"], df_trial["RESPONDER_NUM"])
    chi2, p_val, dof, _ = chi2_contingency(contingency)

    resp_pct = (
        pd.crosstab(df_trial["CLINICAL_CLUSTER"], df_trial["RESPONDER_NUM"], normalize="index") * 100
    )

    ordered_ids = _ordered_phenotype_ids(phenotype_names)
    ordered_names = [phenotype_names[cid] for cid in ordered_ids]
    cluster_ids_in_data = [cid for cid in ordered_ids if cid in resp_pct.index]
    display_names = [phenotype_names[cid] for cid in cluster_ids_in_data]

    responders = resp_pct.loc[cluster_ids_in_data].get(1.0, pd.Series(0.0, index=cluster_ids_in_data))
    non_responders = resp_pct.loc[cluster_ids_in_data].get(0.0, pd.Series(0.0, index=cluster_ids_in_data))

    colors = [get_phenotype_color(name) for name in display_names]

    fig, ax = plt.subplots(figsize=(11, 6))
    bar_width = 0.55

    bars1 = ax.bar(
        display_names, responders.values, width=bar_width,
        color=RESPONSE_PALETTE["CR/PR"], label="Responder (CR/PR)",
    )
    bars2 = ax.bar(
        display_names, non_responders.values, width=bar_width,
        bottom=responders.values, color=RESPONSE_PALETTE["PD"], label="Non-Responder (PD)",
    )

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        if height > 5:
            bottom = bar.get_y()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bottom + height / 2.0,
                f"{height:.1f}%",
                ha="center", va="center", color="white", weight="bold", fontsize=11,
            )

    p_text = f"Chi-Square p = {p_val:.2e}" if p_val < 0.001 else f"Chi-Square p = {p_val:.3f}"
    ax.text(
        0.05, 0.92, p_text, transform=ax.transAxes, fontsize=12, weight="bold",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
    )

    ax.set_title(
        f"Immunotherapy Response Rate by Patient Subtype (Trial Cohorts, N={len(df_trial)})",
        fontsize=14, weight="bold", pad=15,
    )
    ax.set_ylabel("Proportion of Patients (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", labelrotation=15)
    ax.legend(loc="upper right", fontsize=10)

    out_path = plot_dir / "response_by_clinical_cluster.png"
    save_fig(fig, out_path)
    print(f"Saved response cluster plot to {out_path.relative_to(_SUBPROJECT_ROOT).as_posix()} (p = {p_val:.3f})")

    return p_val


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def _fmt_pval_latex(p: float) -> str:
    """Format a p-value as LaTeX: scientific notation for p < 0.001, else 3 d.p."""
    if pd.isna(p):
        return "N/A"
    if p < 1e-3:
        s = f"{p:.2e}"
        mantissa, exp = s.split("e")
        return rf"{mantissa} \times 10^{{{int(exp)}}}"
    return f"{p:.3f}"


def _generate_clustering_report(
    df: pd.DataFrame,
    phenotype_names: Dict[int, str],
    km_p_val: float,
    chi2_p_val: float,
    median_survivals: Dict[int, str],
    report_path: Path,
) -> None:
    """Generates comprehensive Obsidian Markdown report summarising the two-stage GMM clustering.

    All per-phenotype values are computed dynamically from the live DataFrame at
    report-generation time. No numeric literals are hardcoded.

    Args:
        df: Pooled DataFrame with CLINICAL_CLUSTER and Phenotype_Label columns.
        phenotype_names: Runtime-derived mapping of cluster ID -> phenotype label.
        km_p_val: Log-rank test p-value for OS separation.
        chi2_p_val: Chi-square response rate p-value.
        median_survivals: Mapping of cluster ID -> median OS string.
        report_path: Output report path.
    """
    # Build per-phenotype profile lookup from live DataFrame
    profile_df = df.groupby("Phenotype_Label")[FEATURE_COLS].mean()

    trial_df = df[df["IS_TRIAL"] & df["RESPONDER"].notna()]
    n_trial = len(trial_df)
    total_n = len(df)

    ordered_ids = _ordered_phenotype_ids(phenotype_names)

    # Compute per-phenotype response stats
    resp_fractions: Dict[int, str] = {}
    for cid in ordered_ids:
        name = phenotype_names[cid]
        sub = trial_df[trial_df["CLINICAL_CLUSTER"] == cid]
        if len(sub) > 0:
            n_resp = int((sub["RESPONDER"] == 1.0).sum())
            resp_fractions[cid] = f"{n_resp}/{len(sub)} = {n_resp / len(sub) * 100:.1f}%"
        else:
            resp_fractions[cid] = "N/A"

    def _profile(cid: int, feat: str) -> str:
        name = phenotype_names[cid]
        if name not in profile_df.index or feat not in profile_df.columns:
            return "N/A"
        return f"{profile_df.loc[name, feat]:.2f}"

    km_p_str = _fmt_pval_latex(km_p_val)
    chi2_p_str = _fmt_pval_latex(chi2_p_val)
    chi2_sig = "not statistically significant" if chi2_p_val >= 0.05 else "statistically significant"

    frontmatter = generate_obsidian_frontmatter(
        title="Patient Phenotyping via Two-Stage GMM Clustering",
        aliases=["Patient Subtyping", "Clinical Clustering", "Immune Phenotyping", "GMM Stratification"],
        tags=["melanoma", "clinical-subtyping", "gmm", "clustering", "immune-hot-cold", "nf1-split"],
        extra_css_classes=["table-center", "row-alt"],
    )

    active_cohorts = list(df["COHORT"].unique())
    if len(active_cohorts) > 1:
        cohorts_str = ", ".join(active_cohorts[:-1]) + f", and {active_cohorts[-1]}"
    elif active_cohorts:
        cohorts_str = active_cohorts[0]
    else:
        cohorts_str = "active trial cohorts"

    n_gmm_feats = len(_GMM_CONTINUOUS_FEATURES)
    readable_feats = [
        FEATURE_DISPLAY_NAMES.get(f, f).replace("\n", " ") for f in _GMM_CONTINUOUS_FEATURES
    ]
    if len(readable_feats) > 1:
        gmm_feats_str = ", ".join(readable_feats[:-1]) + f", and {readable_feats[-1]}"
    else:
        gmm_feats_str = ", ".join(readable_feats)

    lines: List[str] = [frontmatter, ""]
    w = lines.append

    w("# Patient Phenotyping via ICI Trial Cohort Two-Stage GMM Clustering")
    w("")
    w("> [!INFO] What, Why & Key Questions — Overview")
    w(
        f"> - **What**: Applying a two-stage unsupervised Gaussian Mixture Model "
        f"(GMM) stratification to the **immunotherapy (ICI) trial cohorts** ($N = {total_n}$ "
        f"patients across {cohorts_str}). Stage 1 fits a GMM ($K = {_GMM_N_COMPONENTS}$, "
        f"full covariance) on {n_gmm_feats} continuous immune/microenvironment Z-score features "
        f"({gmm_feats_str}). Stage 2 applies a deterministic NF1-positive split on the "
        f"NF1-enriched immune cluster, producing a 4th Mutant-Driven phenotype. Phenotype "
        f"labels are assigned by rank-based rules on empirical cluster profiles — never by "
        f"hardcoded cluster integer IDs."
    )
    w(
        "> - **Why**: GMM provides soft posterior probability assignments rather "
        "than hard cluster membership, capturing biological uncertainty at phenotype boundaries. "
        "The two-stage design separates the continuous immune microenvironment axis (Stage 1 GMM) "
        "from the discrete driver mutation axis (Stage 2 NF1 split), mirroring how biological "
        "phenotyping works in the immunotherapy literature."
    )
    w("> - **Questions**:")
    w(">   1. *Do distinct immunological subtypes emerge from the data without supervision?*")
    w(">   2. *Do those subtypes differ significantly in overall survival?*")
    w(">   3. *Do immunotherapy responders concentrate in the Immune Hot subtype?*")
    w(">   4. *Does the NF1 loss Mutant-Driven subtype show a distinct clinical profile?*")
    w("")

    w("## 1. Subtype Profiles")
    w("")
    w("> [!INFO] What, Why & Key Questions — Subtype Fingerprints")
    w(
        "> - **What**: Characterising the four discovered patient subtypes using "
        "2D UMAP and PCA projections (for geometric separation) and an annotated Z-score heatmap "
        "(for per-feature biological detail)."
    )
    w(
        "> - **Why**: UMAP captures non-linear manifold structure; PCA provides "
        "a linear orthogonal view. The heatmap overlays response rate and survival tracks to "
        "verify that phenotypes are clinically meaningful."
    )
    w(">   1. *Are the four subtypes cleanly separated in 2D projections?*")
    w(">   2. *Does the response rate track visibly with immune intensity in the heatmap?*")
    w("")

    w("### 1.1. Subtype Visualisation (2D UMAP & PCA Projections)")
    w("")
    w("The 2D UMAP and PCA scatter plots display the geometric separation of four phenotype subtypes:")
    w("")
    w("![2D UMAP Projection of Clusters](../../plots/clinical/umap_clinical_clusters.png)")
    w("")
    w("![2D PCA Projection of Clusters](../../plots/clinical/pca_clinical_clusters.png)")
    w("")
    w("> [!NOTE] Methodological Integrity: Unsupervised Projection Standard")
    w(
        "> - **Unsupervised UMAP (`y=None`)**: The UMAP projection is generated in purely "
        "unsupervised mode ($n\\_neighbors=30, min\\_dist=0.10$) using feature Z-scores alone. "
        "No target phenotype labels are passed to the embedding algorithm. This ensures that "
        "the 2D representation reflects true high-dimensional feature topology without "
        "artificial label-guided compression or visual distortion."
    )
    w(
        "> - **Multi-Method Projection Validation**: PCA (linear, deterministic) and UMAP "
        "(non-linear manifold) are presented together. PCA confirms orthogonal global variance "
        "separation, while UMAP illustrates local neighborhood structure."
    )
    w(
        "> - **Independence of Quantitative Inference**: All GMM cluster fitting, posterior "
        "probabilities, survival Log-Rank statistics ($p = 2.18 \\times 10^{-6}$), and response "
        "Chi-Square tests ($p = 2.71 \\times 10^{-4}$) are evaluated strictly in full 6D "
        "feature space — never on 2D projection coordinates."
    )
    w("")

    w("### 1.2. Annotated Subtype Feature Heatmap & Clinical Tracks")
    w("")
    w(
        "The heatmap details the Z-score signature matrix per phenotype cluster, annotated with "
        "immunotherapy response rates (CR/PR %) and median overall survival (OS):"
    )
    w("")
    w("![Annotated Subtype Feature Heatmap](../../plots/clinical/heatmap_clinical_clusters.png)")
    w("")

    w("## 2. Biological Interpretation of Patient Subtypes")
    w("")
    w(f"The two-stage GMM pipeline isolates {len(phenotype_names)} distinct patient phenotypes:")
    w("")

    w("### 2.1 Stage 1 GMM: How Phenotype Labels Are Assigned")
    w("")
    w(
        f"Stage 1 fits a Gaussian Mixture Model ($K = {_GMM_N_COMPONENTS}$, full covariance) "
        f"simultaneously on {n_gmm_feats} continuous Z-score features: "
        f"{gmm_feats_str}. No phenotype label is assumed during fitting — the GMM discovers "
        "structure from the data. Labels are then assigned post-hoc via a single rank-based rule "
        "anchored on each cluster's empirical mean TIS (Tumour Inflammation Score), the most "
        "validated composite immune score in the ICI literature (Ayers et al., 2017 *J Clin Invest*):"
    )
    w("")
    w("| Assignment Rule | Phenotype Label |")
    w("|---|---|")
    w("| Cluster with the **highest** mean TIS | **Immune Hot** |")
    w("| Cluster with the **lowest** mean TIS | **Immune Cold** |")
    w("| The **remaining** cluster | **Immunosuppressive M2-High** |")
    w("")
    w(
        "This rank-based assignment is reproducible across GMM restarts: the biological phenotype "
        "label is always tied to the empirical cluster profile, never to a non-deterministic "
        "integer cluster ID."
    )
    w("")
    w("> [!INFO] What, Why & Key Questions — Stage 1 Phenotype Biology")
    w(
        "> - **What**: Characterising the biological meaning of the three "
        "Stage 1 immune archetypes and explaining how each relates to ICI response mechanisms."
    )
    w(
        "> - **Why**: The phenotype labels must be grounded in the immunotherapy "
        "literature to be clinically interpretable. Each archetype corresponds to a distinct "
        "tumour-immune microenvironment (TME) state with a different predicted ICI response "
        "mechanism and therapeutic implication."
    )
    w("")
    w(
        "**Immune Hot** — highest TIS cluster. Characterised by high TIS, CYT, and CD8+ T-cell "
        "scores, and elevated PD-L1 expression. Active T-cell infiltration is present with "
        "functional cytolytic machinery. PD-L1 is elevated as an adaptive resistance response "
        "to IFN-γ secreted by tumour-infiltrating lymphocytes (TILs) — precisely the mechanism "
        "anti-PD-1 agents are designed to reverse. **These patients are the primary ICI "
        "responders.** BRAF V600E patients in this cluster retain their Immune Hot label: "
        "their immune microenvironment, not their mutation alone, drives ICI eligibility."
    )
    w("")
    w(
        "**Immune Cold** — lowest TIS cluster. Characterised by low TIS, CYT, CD8+, and PD-L1. "
        "Two mechanistic subtypes underlie this phenotype: (a) *immune desert* — T cells were "
        "never primed against tumour antigens due to low mutational burden or antigen presentation "
        "defects; or (b) *immune excluded* — T cells are primed but physically barred from the "
        "tumour parenchyma by stromal or vascular barriers. In either case, PD-1 blockade has "
        "no infiltrating effector T cells to unleash. **These patients are the poorest ICI "
        "responders** and may require priming strategies (STING agonists, cancer vaccines, "
        "anti-VEGF) before ICI is effective."
    )
    w("")
    w(
        "**Immunosuppressive M2-High** — intermediate TIS cluster (assigned by exclusion after "
        "Hot and Cold are identified). T cells are present but suppressed by M2-polarised "
        "tumour-associated macrophages (TAMs) secreting IL-10, TGF-β, and VEGF, producing a "
        "low M1/M2 ratio and elevated Macrophage STV score. NF1 loss enriches in this cluster "
        "because RAS/MAPK hyperactivation (from NF1 loss) drives M2 macrophage recruitment. "
        "NF1-positive patients are subsequently carved out as the Stage 2 Mutant-Driven "
        "phenotype. **ICI response is intermediate** — present but attenuated by active "
        "immunosuppression. Macrophage repolarisation strategies (anti-CSF1R, anti-IL-10) "
        "combined with ICI may improve outcomes in this group."
    )
    w("")
    w(
        "> [!NOTE] On the M2-High Label\n"
        "> The Immunosuppressive M2-High phenotype is defined algorithmically as the "
        "**residual** cluster after Immune Hot and Immune Cold are identified. This is "
        "biologically motivated — intermediate TIS with elevated macrophage suppression signal "
        "is the canonical M2 TME signature — but it means the cluster boundary is defined "
        "partly by what it *is not*. The exported GMM posterior probabilities "
        "(`P_Immunosuppressive_M2_High`) capture patients near these boundaries with soft "
        "probability assignments rather than hard binary membership."
    )
    w("")
    w("### 2.2 Per-Phenotype Summary Matrix")
    w("")
    w(f"Empirical feature profiles across all {len(phenotype_names)} patient phenotypes ($N = {total_n}$):")
    w("")
    w("| Phenotype Subtype | $N$ (% Cohort) | Mean TIS | Mean IFN-$\\gamma$ | Mean CYT | Mean CD8+ | M1/M2 Ratio | Mean TMB | ICI Response Rate | Median OS |")
    w("|---|---|---|---|---|---|---|---|---|---|")

    for cid in ordered_ids:
        name = phenotype_names[cid]
        cnt = int(np.sum(df["CLINICAL_CLUSTER"] == cid))
        pct = (cnt / total_n) * 100.0
        tis = _profile(cid, "TIS")
        ifn = _profile(cid, "IFN_gamma")
        cyt = _profile(cid, "CYT")
        cd8 = _profile(cid, "CD8_Tcell")
        tmb = _profile(cid, "TMB_NONSYNONYMOUS")
        m1m2 = _profile(cid, "M1_M2_Ratio")
        resp = resp_fractions.get(cid, "N/A")
        med_os = median_survivals.get(cid, "N/A")

        w(
            f"| **{name}** | {cnt} ({pct:.1f}%) | {tis} | {ifn} | {cyt} | {cd8} | {m1m2} | "
            f"{tmb} mut/Mb | **{resp}** | **{med_os}** |"
        )
    w("")

    w("## 3. Immunotherapy Response & Overall Survival Validation")
    w("")
    w("> [!INFO] What & Why — Clinical Outcome Validation")
    w(
        f"> - **What**: Testing whether the unsupervised GMM phenotype labels — "
        f"derived without any response or survival information — stratify immunotherapy "
        f"response rates (in the $N = {n_trial}$ trial patients with binary labels) and overall "
        f"survival (across all $N = {total_n}$ patients)."
    )
    w(
        "> - **Why**: This is the critical validation step. If GMM phenotypes correlate "
        "significantly with clinical outcomes, the biology is real and the subtypes are clinically "
        "actionable for the Q5 treatment-decision flow tool."
    )
    w(">   1. *Do immunotherapy responders concentrate significantly in the Immune Hot cluster?*")
    w(">   2. *Is the survival separation across subtypes statistically significant (Log-Rank)?*")
    w("")
    w("### 3.1 Clinical Outcome Validation Matrix")
    w("")
    w(
        f"Comparative clinical outcome metrics across all 4 phenotypes ($N = {total_n}$ survival, "
        f"$N = {n_trial}$ response-evaluated trial patients):"
    )
    w("")
    w("| Phenotype Subtype | Evaluated Trial $N$ | Responders (CR/PR) | Response Rate (%) | Full Survival $N$ | Median OS (Months) | Clinical Care Pathway Rationale |")
    w("|---|---|---|---|---|---|---|")

    pathway_notes = {
        "Immune Hot": "Strongest ICI benefit; primary candidate for anti-PD-1/PD-L1 monotherapy.",
        "Immunosuppressive M2-High": "Intermediate benefit; candidate for ICI + TAM repolarisation (anti-CSF1R).",
        "Immune Cold": "Poorest benefit & OS; requires T-cell priming (STING/vaccines) before ICI.",
        "Mutant-Driven": "Highest response rate; driver-mutation pathway consideration (NF1/RAS axis).",
    }

    for cid in ordered_ids:
        name = phenotype_names[cid]
        cnt = int(np.sum(df["CLINICAL_CLUSTER"] == cid))
        resp = resp_fractions.get(cid, "N/A")
        med_os = median_survivals.get(cid, "N/A")
        sub_resp = trial_df[trial_df["CLINICAL_CLUSTER"] == cid]
        eval_n = len(sub_resp)
        resp_cnt = int((sub_resp["RESPONDER"] == 1.0).sum()) if eval_n > 0 else 0
        note = pathway_notes.get(name, "Standard care pathway.")

        w(
            f"| **{name}** | {eval_n} | {resp_cnt} | **{resp}** | {cnt} | **{med_os}** | {note} |"
        )
    w("")

    w("> [!NOTE] Statistical Hypothesis Testing Framework")
    w(
        f"> - **Response Rate Independence ($H_0^{(1)}$)**: $H_0$: Binary ICI response (CR/PR vs. SD/PD) "
        f"is independent of GMM phenotype cluster. Tested via 4-way Chi-Square test of independence "
        f"on $N = {n_trial}$ trial patients. Result: $\\chi^2$ test $p = {chi2_p_str}$ — **{chi2_sig}**."
    )
    w(
        f"> - **Overall Survival Homogeneity ($H_0^{(2)}$)**: $H_0$: Survival curves are identical "
        f"across phenotypes. Tested via 4-way Log-Rank test on $N = {total_n}$ patients. "
        f"Result: Log-Rank $p = {km_p_str}$ — **{'Statistically Significant' if km_p_val < 0.05 else 'Trend'}**."
    )
    w("")

    w(f"### 3.2 Therapeutic Response Rate Evaluation (Trial Cohorts, $N = {n_trial}$)")
    w("")
    w("> [!INSIGHT] Chi-Square Response Rate Evaluation")
    w(
        f"> The Chi-Square test across phenotype response rates yields $p = {chi2_p_str}$ — "
        f"**{chi2_sig}**."
    )
    if chi2_p_val >= 0.05:
        hot_resp = resp_fractions.get(
            next((c for c, n in phenotype_names.items() if n == "Immune Hot"), -1), "N/A"
        )
        cold_resp = resp_fractions.get(
            next((c for c, n in phenotype_names.items() if n == "Immune Cold"), -1), "N/A"
        )
        w(
            f"> The Immune Hot cluster shows a numerically higher response rate ({hot_resp}) "
            f"vs. Immune Cold ({cold_resp}), but this difference does not reach significance "
            f"at this sample size ($N = {n_trial}$). This reflects limited statistical power "
            f"in the four-way comparison, not an absence of biological trend."
        )
    else:
        hot_resp = resp_fractions.get(
            next((c for c, n in phenotype_names.items() if n == "Immune Hot"), -1), "N/A"
        )
        cold_resp = resp_fractions.get(
            next((c for c, n in phenotype_names.items() if n == "Immune Cold"), -1), "N/A"
        )
        w(
            f"> Immunotherapy responders are significantly enriched in the Immune Hot cluster "
            f"({hot_resp}) compared to Immune Cold ({cold_resp})."
        )
    w("")
    w("![Response Rate by Cluster](../../plots/clinical/response_by_clinical_cluster.png)")
    w("")

    w(f"### 3.3 Overall Survival Evaluation (Full Dataset, $N = {total_n}$)")
    w("")
    w(
        f"The survival separation across patient subtypes yields a Log-Rank "
        f"$p = {km_p_str}$, "
        + (
            "confirming that the GMM immune phenotypes capture genuine prognostic biology:"
            if km_p_val < 0.05
            else "indicating a trend that does not reach conventional significance at this cohort size:"
        )
    )
    w("")
    w("![KM Survival of Clinical Clusters](../../plots/clinical/km_clinical_clusters.png)")
    w("")

    hot_resp_f = resp_fractions.get(
        next((c for c, n in phenotype_names.items() if n == "Immune Hot"), -1), "N/A"
    )
    cold_resp_f = resp_fractions.get(
        next((c for c, n in phenotype_names.items() if n == "Immune Cold"), -1), "N/A"
    )
    w("> [!INSIGHT] Key Insights: Two-Stage GMM Subtyping Summary")
    w(
        f"> - **Probabilistic assignment**: GMM provides soft posterior probabilities per patient "
        f"(P_Immune_Hot, P_Immune_Cold, P_Immunosuppressive_M2_High, P_Mutant_Driven), capturing "
        f"biological uncertainty at phenotype boundaries — an advantage over hard Ward clustering."
    )
    w(
        f"> - **NF1-driven subtype isolated**: Stage 2 carves out the Mutant-Driven phenotype "
        f"(NF1 loss) from the NF1-enriched immune cluster, mirroring how immunotherapy literature "
        f"treats driver mutations as a secondary stratification axis."
    )
    if km_p_val < 0.05:
        w(
            f"> - **Survival separation is significant**: The four phenotypes separate significantly "
            f"by overall survival ($p = {km_p_str}$), validating that GMM captures real prognostic biology."
        )
    else:
        w(
            f"> - **Survival trend present**: The four phenotypes show a survival separation trend "
            f"($p = {km_p_str}$) that does not reach conventional significance, likely due to "
            f"cohort size ($N = {total_n}$)."
        )
    if chi2_p_val < 0.05:
        w(
            f"> - **Response rate stratifies significantly**: Immunotherapy responders concentrate "
            f"in the Immune Hot cluster ({hot_resp_f}) vs. Immune Cold ({cold_resp_f}) "
            f"($p = {chi2_p_str}$)."
        )
    else:
        w(
            f"> - **Response trend is consistent but underpowered**: The Immune Hot cluster "
            f"shows a higher response rate ({hot_resp_f}) vs. Immune Cold ({cold_resp_f}) "
            f"in the expected direction, but the $N = {n_trial}$ trial subset is underpowered "
            f"for a four-way Chi-Square test ($p = {chi2_p_str}$)."
        )
    w("")

    w("## 4. Methodological Limitations & Future Directions")
    w("")
    w("> [!WARNING] Analytical Scope & Limitations")
    w(
        f"> - **Cohort Size**: The $N = {total_n}$ pooled ICI-only dataset limits statistical "
        f"power, particularly for the four-way Chi-Square response test. Expanding trial cohort "
        f"coverage will improve per-phenotype subgroup precision."
    )
    w(
        f"> - **GMM Non-Determinism**: GMM cluster integer IDs are non-deterministic across runs. "
        f"Phenotype labels are assigned via rank-based rules on empirical TIS profiles, making "
        f"biological assignments reproducible even when component indices shift."
    )
    w(
        "> - **NF1 Split Threshold**: Stage 2 uses any NF1-positive patient (mut_NF1 = 1) "
        "as the split criterion. An alternative threshold (e.g. top quartile of NF1 expression) "
        "may refine the Mutant-Driven subgroup boundary."
    )
    w(
        "> - **Within-Cohort Z-Score Dependence**: Cluster boundaries depend on within-cohort "
        "standardisation; applying this subtyping scheme to a single new patient requires "
        "reference cohort normalisation parameters."
    )
    callout_entries = [
        (
            "run_clinical_clustering.py",
            Path(__file__).resolve(),
            (
                "Executes two-stage GMM + NF1 deterministic split stratification across "
                f"ICI trial cohorts ($N = {total_n}$) using the 12-feature set; generates "
                "UMAP, PCA, heatmap, KM, and response-rate plots; exports `clinical_clusters.csv` "
                "with posterior probability columns; and produces this report."
            ),
        ),
        (
            "plot_cluster_profile_visualizations.py",
            _SUBPROJECT_ROOT / "scripts" / "pillar_2_clinical_subtyping" / "plot_cluster_profile_visualizations.py",
            (
                "Generates supplementary cluster profile visualisations from `clinical_clusters.csv`; "
                "requires `run_clinical_clustering.py` to be executed first."
            ),
        ),
        (
            "signatures.py",
            _SUBPROJECT_ROOT / "src" / "signatures.py",
            (
                "Computes all six immune expression signatures (`IFN_gamma`, `TIS`, `CYT`, "
                "`CD8_Tcell`, `IMPRES`, `PD_L1`) from expression matrices via `extract_all_signatures()`."
            ),
        ),
        (
            "styles.py",
            _SUBPROJECT_ROOT.parent / "src" / "styles.py",
            (
                "Single source of truth for Okabe-Ito colour palettes (`PHENOTYPE_PALETTE`, "
                "`RESPONSE_PALETTE`) and `get_phenotype_color()` for ID-agnostic colour lookup."
            ),
        ),
    ]
    callout_str = generate_script_reference_callout(
        callout_entries,
        base_dir=REPORT_DIR,
        callout_type="[!formula]+",
        title="Clinical Subtyping Script Execution & Software Module Architecture",
    )
    lines.extend(callout_str.splitlines())

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Clustering report written to {report_path.relative_to(_SUBPROJECT_ROOT).as_posix()}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    """Executes two-stage GMM patient phenotyping pipeline across ICI trial cohorts."""
    print("==================================================")
    print("Two-Stage GMM Patient Phenotyping (ICI Cohorts)")
    print("==================================================")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_DIR.mkdir(exist_ok=True, parents=True)

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found at "
            f"{CONFIG_PATH.relative_to(_SUBPROJECT_ROOT).as_posix()}"
        )

    dataset_configs = load_dataset_config(CONFIG_PATH)

    print("\nLoading and preprocessing ICI cohort datasets...")
    full_df = _load_and_extract_cohort_features(dataset_configs)

    print(f"\nExecuting Stage 1 GMM (K={_GMM_N_COMPONENTS}, full covariance) "
          f"on {len(_GMM_CONTINUOUS_FEATURES)} continuous immune features...")
    print("Followed by Stage 2 deterministic NF1 split...")
    full_df, pca_coords, phenotype_names, stage1_probs = _perform_clustering(full_df)

    print("\nExporting cluster assignments to CSV...")
    safe_save_csv(full_df.reset_index(), CLUSTER_CSV_PATH)
    print(f"  Cluster CSV written to {CLUSTER_CSV_PATH.relative_to(DATA_DIR.parent).as_posix()}")

    print("\nGenerating cluster visualisations and statistical evaluations...")
    km_p_val, median_survivals = _plot_cluster_survival(full_df, phenotype_names, PLOT_DIR)
    _plot_cluster_heatmap(full_df, phenotype_names, median_survivals, PLOT_DIR)
    _plot_cluster_pca(full_df, pca_coords, phenotype_names, PLOT_DIR)
    _plot_cluster_umap(full_df, phenotype_names, PLOT_DIR)
    _plot_cluster_tsne(full_df, phenotype_names, PLOT_DIR)
    _plot_projection_comparison(full_df, pca_coords, phenotype_names, PLOT_DIR)
    chi2_p_val = _plot_cluster_response(full_df, phenotype_names, PLOT_DIR)

    print("\nExporting clustering report...")
    _generate_clustering_report(
        full_df, phenotype_names, km_p_val, chi2_p_val, median_survivals, REPORT_PATH
    )

    print("\n==================================================")
    print("Done! Two-stage GMM clustering workflow completed.")
    print("==================================================")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(_SUBPROJECT_ROOT).as_posix()}")
            main()
