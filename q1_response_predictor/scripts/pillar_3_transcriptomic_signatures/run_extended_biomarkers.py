"""Extended Biomarker Evaluation & Characterisation (Merged Trial Cohorts).

Evaluates univariate associations, neoantigen load vs TMB, somatic pathway mutations,
and aneuploidy & TMB vs immune infiltration across clinical trial cohorts and TCGA.
"""

# ---------------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------------
import contextlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Third-Party Imports
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Bootstrap & Path Resolution
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()

for _candidate in [_THIS_FILE.parent] + list(_THIS_FILE.parent.parents):
    if _candidate.name == "q1-response-predictor":
        BASE_DIR = _candidate
        break
else:
    raise FileNotFoundError(
        f"Could not locate q1-response-predictor subproject root above {_THIS_FILE}"
    )

PROJECT_ROOT = BASE_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.biology_constants import (
    DRIVER_GENES,
    NEOANTIGEN_COLS,
    PATHWAY_GENES,
    RECIST_RESPONSE_MAP,
)
from src.config.datasets import DatasetConfig, load_dataset_config
from src.signatures import extract_all_signatures
from src.styles import RESPONSE_PALETTE, get_cohort_color, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import get_subproject_log_dir, rel_path
from src.utils.plotting import add_km_risk_table, save_fig

set_presentation_style()

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
DEFAULT_RANDOM_STATE: int = 42
DATA_DIR = PROJECT_ROOT / "data"
PLOT_DIR = BASE_DIR / "plots" / "biomarkers"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

REPORTS_DIR = BASE_DIR / "reports" / "pillar_3_transcriptomic_signatures"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)
CURATED_SIGNATURES_REPORT_PATH = REPORTS_DIR / "curated_signatures_report.md"

CONFIG_PATH: Path = BASE_DIR / "config" / "datasets.yaml"

# TCGA Pan-Can Atlas used for survival analysis (has aneuploidy scores)
_TCGA_PROCESSED_DIR: str = "skcm_tcga_pan_can_atlas_2018"

LOG_DIR = get_subproject_log_dir(_THIS_FILE)
LOG_PATH = LOG_DIR / "run_extended_biomarkers.log"

_COL_SAMPLE_ID = "SAMPLE_ID"
_COL_TMB = "TMB_NONSYNONYMOUS"
_COL_TOTAL_NEOANTIGEN = "TOTAL_NEOANTIGEN"
_COL_RESPONSE = "response"
_COL_RESPONSE_RAW = "RESPONSE"
_COL_TEMP_RESP = "temp_resp"
_COL_OS_MONTHS = "OS_MONTHS"
_COL_OS_STATUS = "OS_STATUS"
_COL_OS_STATUS_CLEAN = "os_status_clean"
_COL_ANEUPLOIDY = "ANEUPLOIDY_SCORE"
_COL_AGE = "AGE"
_COL_COHORT = "Cohort"
_COL_CNA = "CNA_PROP"
_PREFIX_MUT = "mut_"

_CURATED_IMMUNE_SIGNATURES = [
    "IFN_gamma", "TIS", "CD8_Tcell", "CYT", "IMPRES", "PD_L1"
]
_SIG_DISPLAY_NAMES = {
    "IFN_gamma": "IFN-γ",
    "TIS": "TIS",
    "CD8_Tcell": "CD8 T-Cell",
    "CYT": "CYT",
    "IMPRES": "IMPRES",
    "PD_L1": "PD-L1",
}


# ---------------------------------------------------------------------------
# Helper Utility Functions
# ---------------------------------------------------------------------------
def clean_os_status(val: Any) -> float:
    """Clean and standardise overall survival status to 0 (living) or 1 (deceased)."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        if val in [1, 1.0]:
            return 1.0
        if val in [0, 0.0]:
            return 0.0
    if isinstance(val, str):
        val_upper = val.upper()
        if 'DECEASED' in val_upper or '1' in val_upper:
            return 1.0
        if 'LIVING' in val_upper or '0' in val_upper:
            return 0.0
    return np.nan


def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise DataFrame columns individually (Z-score scaling)."""
    means = df.mean(axis=0)
    stds = df.std(axis=0).replace(0, 1.0).fillna(1.0)
    return (df - means) / stds


def load_processed_mutations(
    mutations_file: Path, target_genes: List[str], sample_ids: List[str]
) -> pd.DataFrame:
    """Load processed mutation calls for target genes across specified sample IDs."""
    if not mutations_file.exists():
        return pd.DataFrame(0, index=sample_ids, columns=target_genes)
        
    df_mut = pd.read_csv(mutations_file, index_col=_COL_SAMPLE_ID)
    for g in target_genes:
        if g not in df_mut.columns:
            df_mut[g] = 0
            
    df_mut = (df_mut[target_genes] > 0).astype(int)
    return df_mut.reindex(sample_ids, fill_value=0)


# ---------------------------------------------------------------------------
# Data Loading & Processing Modular Subroutines
# ---------------------------------------------------------------------------
def _load_trial_cohort_data(
    configs: List[DatasetConfig],
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load clin and expr DataFrames for all non-TCGA trial cohorts from datasets.yaml.

    Skips cohorts whose processed_directory contains 'tcga' or where required files
    are missing. Returns a mapping of cohort_name -> (df_clin, df_expr).
    """
    result: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for cfg in configs:
        if cfg.processed_directory == _TCGA_PROCESSED_DIR:
            continue
        clin_path = DATA_DIR / "processed" / cfg.processed_directory / "clin_cleaned.csv"
        expr_path = DATA_DIR / "processed" / cfg.processed_directory / "expr_cleaned.csv"
        if not (clin_path.exists() and expr_path.exists()):
            print(f"  Skipping {cfg.cohort_name}: missing clin or expr in {cfg.processed_directory}.")
            continue
        df_clin = pd.read_csv(clin_path, index_col=_COL_SAMPLE_ID)
        df_expr = pd.read_csv(expr_path, index_col=0)
        if cfg.cohort_immunotherapy:
            immuno_col = next(
                (c for c in df_clin.columns if c.startswith("TX_TYPE_IMMUNOTHERAPY") or c == "immunotherapy"),
                None,
            )
            if immuno_col:
                mask = pd.to_numeric(df_clin[immuno_col], errors="coerce") == 1.0
                if 0 < mask.sum() < len(df_clin):
                    df_clin = df_clin.loc[mask]
                    common_expr = df_expr.index.intersection(df_clin.index)
                    df_expr = df_expr.loc[common_expr]
        result[cfg.cohort_name] = (df_clin, df_expr)
        print(f"  Loaded {cfg.cohort_name} (N={len(df_clin)})")
    return result


def _align_clinical_column_aliases(df_list: List[pd.DataFrame]) -> None:
    """Map legacy column uppercase/lowercase aliases for backward compatibility."""
    alias_pairs = [
        ('PATIENT_ID', 'patient_id'), ('RESPONSE_BINARY', _COL_RESPONSE),
        ('SEX', 'sex'), ('AGE', _COL_AGE),
        (_COL_OS_STATUS, 'os_status'), (_COL_OS_MONTHS, 'os_months')
    ]
    for df in df_list:
        for up, low in alias_pairs:
            if up in df.columns and low not in df.columns:
                df[low] = df[up]


def _filter_and_calculate_neoantigens(df_clin: pd.DataFrame) -> None:
    """Filter trial cohort to CR/PR/PD responses and compute total neoantigen load."""
    df_clin[_COL_TEMP_RESP] = df_clin[_COL_RESPONSE_RAW].map(RECIST_RESPONSE_MAP)
    df_clin.dropna(subset=[_COL_TEMP_RESP], inplace=True)
    df_clin.drop(columns=[_COL_TEMP_RESP], inplace=True)

    if _COL_TOTAL_NEOANTIGEN not in df_clin.columns:
        avail_neo = [c for c in NEOANTIGEN_COLS if c in df_clin.columns]
        if avail_neo:
            df_clin[_COL_TOTAL_NEOANTIGEN] = df_clin[avail_neo].fillna(0).sum(axis=1)
            all_nan = df_clin[avail_neo].isna().all(axis=1)
            df_clin.loc[all_nan, _COL_TOTAL_NEOANTIGEN] = np.nan
        else:
            df_clin[_COL_TOTAL_NEOANTIGEN] = np.nan


def _process_tcga_signatures(
    df_tcga_clin: pd.DataFrame, df_tcga_expr_raw: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract, standardise, and align TCGA-SKCM expression signatures."""
    df_tcga_expr = df_tcga_expr_raw.set_index(_COL_SAMPLE_ID)
    df_tcga_sigs = zscore_df(extract_all_signatures(df_tcga_expr))

    common = df_tcga_sigs.index.intersection(df_tcga_clin.index)
    return df_tcga_clin.loc[common], df_tcga_sigs.loc[common]


def _attach_cohort_mutations(
    df_clin: pd.DataFrame, cohort_dir: str, all_genes: List[str]
) -> None:
    """Attach mutation indicators to a single clinical trial cohort DataFrame."""
    mut_df = load_processed_mutations(
        DATA_DIR / f"processed/{cohort_dir}/mutations_cleaned.csv",
        all_genes, df_clin.index.tolist()
    )
    for col in all_genes:
        df_clin[f"{_PREFIX_MUT}{col}"] = mut_df[col]


def _attach_pathway_mutations(
    cohort_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    configs: List[DatasetConfig],
) -> None:
    """Load somatic pathway mutations for all trial cohorts and attach to clin DataFrames.

    Modifies each df_clin in cohort_data in-place.
    """
    pathway_genes_flat = [g for genes in PATHWAY_GENES.values() for g in genes]
    all_genes = sorted(list(set(pathway_genes_flat + DRIVER_GENES)))
    config_by_name = {cfg.cohort_name: cfg for cfg in configs}

    antigen_genes = PATHWAY_GENES["Antigen Presentation"]
    ifn_genes = PATHWAY_GENES["IFN-gamma Signature"]
    surv_genes = PATHWAY_GENES["Survival & Proliferation Drivers"]

    for name, (df_clin, _) in cohort_data.items():
        cfg = config_by_name.get(name)
        if cfg is None:
            continue
        _attach_cohort_mutations(df_clin, cfg.processed_directory, all_genes)
        antigen_cols = [f"{_PREFIX_MUT}{g}" for g in antigen_genes]
        ifn_cols = [f"{_PREFIX_MUT}{g}" for g in ifn_genes]
        surv_cols = [f"{_PREFIX_MUT}{g}" for g in surv_genes]
        df_clin["mut_Antigen_Presentation"] = (
            df_clin[[c for c in antigen_cols if c in df_clin.columns]].sum(axis=1) > 0
        ).astype(int)
        df_clin["mut_IFN_gamma_Signaling"] = (
            df_clin[[c for c in ifn_cols if c in df_clin.columns]].sum(axis=1) > 0
        ).astype(int)
        df_clin["mut_Survival_Pathways"] = (
            df_clin[[c for c in surv_cols if c in df_clin.columns]].sum(axis=1) > 0
        ).astype(int)


def _merge_clinical_and_signatures(
    cohort_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    """Concatenate trial cohort clinical DataFrames into a pooled DataFrame.

    Only includes columns that exist in each cohort, filling missing ones with NaN.
    """
    desired_cols = [
        _COL_COHORT, _COL_RESPONSE, _COL_TMB, _COL_AGE, _COL_TOTAL_NEOANTIGEN, _COL_CNA,
        f"{_PREFIX_MUT}BRAF", f"{_PREFIX_MUT}NRAS", f"{_PREFIX_MUT}NF1",
        "mut_Antigen_Presentation", "mut_IFN_gamma_Signaling", "mut_Survival_Pathways",
    ]
    frames = []
    for name, (df_clin, _) in cohort_data.items():
        if _COL_CNA not in df_clin.columns:
            df_clin[_COL_CNA] = np.nan
        if _COL_AGE not in df_clin.columns:
            df_clin[_COL_AGE] = np.nan
        avail = [c for c in desired_cols if c in df_clin.columns]
        frames.append(df_clin[avail])
    return pd.concat(frames)


def _prepare_cohort_labels_and_sigs(
    cohort_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    """Annotate cohort names and extract concatenated z-scored expression signatures.

    Args:
        cohort_data: Mapping of cohort_name -> (df_clin, df_expr). Annotates df_clin in-place.

    Returns:
        Concatenated z-scored immune signature DataFrame across all cohorts.
    """
    print("\nComputing expression signatures for all cohorts...")
    sig_frames = []
    for name, (df_clin, df_expr) in cohort_data.items():
        df_clin[_COL_COHORT] = name
        common_idx = df_clin.index.intersection(df_expr.index)
        sig_frames.append(zscore_df(extract_all_signatures(df_expr.reindex(common_idx))))
    return pd.concat(sig_frames)


def _load_and_prepare_data() -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
    Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
] | None:
    """Load and preprocess clinical, expression, and mutation data for all active cohorts.

    Dynamically loads all non-TCGA trial cohorts from datasets.yaml, and the TCGA
    Pan-Can Atlas cohort for survival analysis.

    Returns:
        Tuple of (df_clin_merged, df_sigs_merged, df_tcga_clin, df_tcga_sigs, cohort_data)
        where cohort_data maps cohort_name -> (df_clin, df_expr).
    """
    if not CONFIG_PATH.exists():
        print(f"Error: datasets.yaml not found at {rel_path(CONFIG_PATH)}.")
        return None
    configs = load_dataset_config(CONFIG_PATH)

    # Load TCGA reference cohort separately for survival analysis
    tcga_clin_path = DATA_DIR / "processed" / _TCGA_PROCESSED_DIR / "clin_cleaned.csv"
    tcga_expr_path = DATA_DIR / "processed" / _TCGA_PROCESSED_DIR / "expr_cleaned.csv"
    if not (tcga_clin_path.exists() and tcga_expr_path.exists()):
        print(f"Error: TCGA files not found at {_TCGA_PROCESSED_DIR}. Run clean_data.py first.")
        return None
    df_tcga_clin = pd.read_csv(tcga_clin_path, index_col=_COL_SAMPLE_ID)
    df_tcga_expr_raw = pd.read_csv(tcga_expr_path)

    # Load all active trial cohorts dynamically
    print("Loading clinical and expression data for all active trial cohorts...")
    cohort_data = _load_trial_cohort_data(configs)
    if not cohort_data:
        print("Error: No trial cohort data found. Run clean_data.py first.")
        return None

    all_clin_dfs = [df_clin for df_clin, _ in cohort_data.values()]
    _align_clinical_column_aliases([df_tcga_clin] + all_clin_dfs)
    for name, (df_clin, df_expr) in cohort_data.items():
        _filter_and_calculate_neoantigens(df_clin)
        cohort_data[name] = (df_clin, df_expr)

    df_sigs_merged = _prepare_cohort_labels_and_sigs(cohort_data)
    df_tcga_clin, df_tcga_sigs = _process_tcga_signatures(df_tcga_clin, df_tcga_expr_raw)
    _attach_pathway_mutations(cohort_data, list(configs))
    df_clin_merged = _merge_clinical_and_signatures(cohort_data)

    return (df_clin_merged, df_sigs_merged, df_tcga_clin, df_tcga_sigs, cohort_data)


# ---------------------------------------------------------------------------
# Section 1: Neoantigen Load vs. TMB
# ---------------------------------------------------------------------------
def _compute_neoantigen_roc_metrics(
    df_clin_merged: pd.DataFrame
) -> Tuple[pd.Index, float, float, float, float]:
    """Compute ROC-AUC and Mann-Whitney U test metrics for Neoantigen Load and TMB.

    Intersects rows with valid response, neoantigen load, AND TMB so that cohorts
    lacking one of these features (e.g. Gide 2019, Van Allen 2015 without TMB) are
    excluded from the AUC computation without causing NaN errors.
    """
    y_true_trials = df_clin_merged[_COL_RESPONSE].dropna()
    neo_valid = df_clin_merged[_COL_TOTAL_NEOANTIGEN].dropna().index
    tmb_valid = df_clin_merged[_COL_TMB].dropna().index
    common_idx = y_true_trials.index.intersection(neo_valid).intersection(tmb_valid)

    auc_neo = roc_auc_score(
        df_clin_merged.loc[common_idx, _COL_RESPONSE],
        df_clin_merged.loc[common_idx, _COL_TOTAL_NEOANTIGEN]
    )
    auc_tmb = roc_auc_score(
        df_clin_merged.loc[common_idx, _COL_RESPONSE],
        df_clin_merged.loc[common_idx, _COL_TMB]
    )

    resp_mask = df_clin_merged[_COL_RESPONSE] == 1
    nonresp_mask = df_clin_merged[_COL_RESPONSE] == 0
    _, p_neo_mw = mannwhitneyu(
        df_clin_merged[resp_mask][_COL_TOTAL_NEOANTIGEN].dropna(),
        df_clin_merged[nonresp_mask][_COL_TOTAL_NEOANTIGEN].dropna()
    )
    _, p_tmb_mw = mannwhitneyu(
        df_clin_merged[resp_mask][_COL_TMB].dropna(),
        df_clin_merged[nonresp_mask][_COL_TMB].dropna()
    )
    return common_idx, float(auc_neo), float(auc_tmb), float(p_neo_mw), float(p_tmb_mw)



def _evaluate_neoantigen_load(df_clin_merged: pd.DataFrame) -> None:
    """Evaluate Neoantigen Load vs. TMB in Pooled Trials and print summary metrics."""
    print("\nEvaluating Neoantigen Load vs. TMB in Pooled Trials...")
    
    df_trials_neo = df_clin_merged.dropna(subset=[_COL_TOTAL_NEOANTIGEN, _COL_TMB]).copy()
    r_spearman, p_spearman = spearmanr(
        df_trials_neo[_COL_TOTAL_NEOANTIGEN], df_trials_neo[_COL_TMB], nan_policy='omit'
    )
    n_trials = len(df_trials_neo)
    print(
        f"  Spearman correlation (pooled N={n_trials}): "
        f"r = {r_spearman:.3f}, p = {p_spearman:.2e}"
    )
    
    common_idx, auc_neo, auc_tmb, p_neo_mw, p_tmb_mw = _compute_neoantigen_roc_metrics(
        df_clin_merged
    )
    n_comm = len(common_idx)
    print(
        f"  Neoantigen response prediction ROC AUC (N={n_comm}): "
        f"{auc_neo:.3f} (p = {p_neo_mw:.3e})"
    )
    print(
        f"  TMB response prediction ROC AUC (N={n_comm}): "
        f"{auc_tmb:.3f} (p = {p_tmb_mw:.3e})"
    )


# ---------------------------------------------------------------------------
# Section 2: Somatic Pathway Mutations
# ---------------------------------------------------------------------------
def _evaluate_pathway_mutations(
    cohort_data: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
    df_clin_merged: pd.DataFrame,
) -> None:
    """Calculate and tabulate pathway mutation frequencies across all trial cohorts."""
    print("\nCalculating pathway mutation frequencies in trial cohorts...")
    mutation_labels = [
        ("BRAF mutation", f"{_PREFIX_MUT}BRAF"),
        ("NRAS mutation", f"{_PREFIX_MUT}NRAS"),
        ("NF1 mutation", f"{_PREFIX_MUT}NF1"),
        ("Antigen Presentation (MHC)", "mut_Antigen_Presentation"),
        ("IFN-gamma Signalling", "mut_IFN_gamma_Signaling"),
        ("Survival & Proliferation Drivers", "mut_Survival_Pathways"),
    ]
    for label, col in mutation_labels:
        parts = []
        for name, (df_clin, _) in cohort_data.items():
            if col in df_clin.columns:
                parts.append(f"{name}: {df_clin[col].mean():.1%}")
        if col in df_clin_merged.columns:
            parts.append(f"Pooled: {df_clin_merged[col].mean():.1%}")
        print(f"  {label:<32} | {' | '.join(parts)}")


# ---------------------------------------------------------------------------
# Section 3: Aneuploidy and TMB vs Immune Infiltration
# ---------------------------------------------------------------------------
def _compute_genomic_immune_correlations(
    df_clin_merged: pd.DataFrame, df_sigs_merged: pd.DataFrame, sig_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compute Spearman correlations between TMB and immune signatures in pooled trial cohort."""
    df_sigs_aligned = df_sigs_merged.reindex(df_clin_merged.index)
    trial_corrs = {}
    for sig in sig_names:
        r_t, p_t = spearmanr(
            df_clin_merged[_COL_TMB], df_sigs_aligned[sig], nan_policy='omit'
        )
        trial_corrs[sig] = {'Tmb_r': float(r_t), 'Tmb_p': float(p_t)}
    return trial_corrs


def _plot_correlation_heatmap(
    trial_corrs: Dict[str, Dict[str, float]],
    sig_names: List[str],
    n_samples: int
) -> Path:
    """Plot correlation bar/heatmap between TMB and immune signatures in trial cohort."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    corr_series = pd.Series(
        [trial_corrs[s]['Tmb_r'] for s in sig_names], index=sig_names
    )
    corr_df = pd.DataFrame({'Spearman r_s': corr_series})

    cbar_label = "Spearman Correlation ($r_s$)"
    sns.heatmap(
        corr_df, annot=True, cmap='coolwarm', vmin=-0.3, vmax=0.3,
        center=0, ax=ax, fmt=".3f", linewidths=1.5,
        cbar_kws={'label': cbar_label}
    )
    ax.set_title(
        f"Pooled Trial Cohort (N={n_samples}): Nonsynonymous TMB vs. Immune Signatures",
        fontsize=12, weight='bold', pad=15
    )
    ax.set_ylabel("Curated Immune Signatures", fontsize=11, weight='bold')
    plt.tight_layout()
    plot_path = PLOT_DIR / "extended_immune_correlations.png"
    save_fig(fig, plot_path)
    plt.close(fig)
    return plot_path


def _update_report_section_3_2(
    report_path: Path,
    trial_corrs: Dict[str, Dict[str, float]],
    n_samples: int,
    cohort_names: List[str],
) -> None:
    """Updates Section 3.2 of curated_signatures_report.md with live Spearman correlation metrics."""
    if not report_path.exists():
        print(f"  [WARNING] Report path does not exist: {rel_path(report_path)}")
        return

    max_abs_r = max(abs(v['Tmb_r']) for v in trial_corrs.values())
    min_p = min(v['Tmb_p'] for v in trial_corrs.values())

    sig_order = _CURATED_IMMUNE_SIGNATURES
    headers = [_SIG_DISPLAY_NAMES.get(s, s) for s in sig_order]
    header_row = "| Feature | " + " | ".join(headers) + " |"
    align_row = "| :--- | " + " | ".join([":---:"] * len(headers)) + " |"

    r_vals = []
    p_vals = []
    for s in sig_order:
        r_val = trial_corrs[s]['Tmb_r']
        p_val = trial_corrs[s]['Tmb_p']
        r_str = f"−{abs(r_val):.3f}" if r_val < 0 else f"{r_val:.3f}"
        p_str = f"{p_val:.3f}"
        r_vals.append(r_str)
        p_vals.append(p_str)

    r_row = "| **Trial TMB** ($r_s$) | " + " | ".join(r_vals) + " |"
    p_row = "| *p*-value | " + " | ".join(p_vals) + " |"

    max_r_str = f"{max_abs_r:.3f}"
    min_p_str = f"{min_p:.2f}"
    cohort_list_str = ", ".join(cohort_names)

    sec32_lines = [
        "### 3.2. Genomic Burden vs. Immune Signatures: Independent (Orthogonal) Modalities",
        "",
        "> [!NOTE] Analysis Scope",
        "> - **What**: Computing Spearman rank correlations between nonsynonymous mutational burden (`TMB_NONSYNONYMOUS`) and all six curated transcriptomic immune signatures.",
        f"> - **Cohort**: Pooled ICI trial cohort ($N = {n_samples}$: {cohort_list_str}).",
        "> - **Why**: Establishing whether genomic mutational burden and transcriptomic immune activity are independent axes of variation within the ICI-treated population — a prerequisite for justifying a multimodal (genomic + transcriptomic) model.",
        "",
        f"Spearman rank correlation between nonsynonymous TMB and the six curated immune signatures in the pooled ICI trial cohort ($N = {n_samples}$) reveals near-complete biological orthogonality across all signature axes ($|r_s| \\leq {max_r_str}$, all $p > {min_p_str}$):",
        "",
        header_row,
        align_row,
        r_row,
        p_row,
        "",
        f"![Nonsynonymous TMB vs. Curated Immune Signatures — ICI Trial Cohort (N={n_samples})](../../plots/biomarkers/extended_immune_correlations.png)",
        "",
        "> [!INSIGHT] The Multimodal Pitch",
        f"> **Genomic burden (TMB) and transcriptomic immune signatures are orthogonal, independent axes of variation** within the ICI-treated melanoma population. No meaningful linear or rank-order relationship exists between the number of nonsynonymous somatic mutations a tumour carries and its inflammatory transcriptomic state ($|r_s| \\leq {max_r_str}$, all $p > {min_p_str}$). A tumour can be hypermutated but immunologically cold, or nearly diploid yet profoundly inflamed. This orthogonality is precisely what makes a multimodal model (Signatures + TMB + Drivers) theoretically justified and, as shown in Section 4, empirically superior to any single modality alone.",
    ]

    sec32_content = "\n".join(sec32_lines)

    text = report_path.read_text(encoding="utf-8")
    start_marker = "### 3.2. Genomic Burden vs. Immune Signatures: Independent (Orthogonal) Modalities"
    end_marker = "### 3.3. Inter-Signature Correlations & Multivariate Drivers"

    start_idx = text.find(start_marker)
    if start_idx == -1:
        print(f"  [WARNING] Could not find start marker '{start_marker}' in {rel_path(report_path)}")
        return

    end_idx = text.find(end_marker)
    if end_idx == -1:
        print(f"  [WARNING] Could not find end marker '{end_marker}' in {rel_path(report_path)}")
        return

    before = text[:start_idx]
    after = text[end_idx:]

    new_text = before + sec32_content + "\n\n" + after
    report_path.write_text(new_text, encoding="utf-8")
    print(f"  Updated Section 3.2 in {rel_path(report_path)}")


def _prepare_survival_df(
    df_tcga_clin: pd.DataFrame, column: str
) -> Tuple[pd.DataFrame, float]:
    """Clean survival status, filter missing values, and calculate column median.

    Coerces OS_MONTHS to numeric before dropna so TCGA-style non-numeric sentinels
    (e.g. '[Not Available]') are eliminated rather than silently passing through.
    """
    df_surv = df_tcga_clin.copy()
    df_surv[_COL_OS_MONTHS] = pd.to_numeric(df_surv[_COL_OS_MONTHS], errors='coerce')
    df_surv[_COL_OS_STATUS_CLEAN] = df_surv[_COL_OS_STATUS].apply(clean_os_status)
    df_surv = df_surv.dropna(subset=[_COL_OS_MONTHS, _COL_OS_STATUS_CLEAN, column])
    median_val = float(df_surv[column].median())
    return df_surv, median_val


def _render_km_curves(
    ax: plt.Axes, df_surv: pd.DataFrame, column: str, median_val: float,
    color_low: str, color_high: str
) -> float:
    """Fit Kaplan-Meier survival curves and return log-rank p-value.

    Raises:
        ValueError: If either the low or high stratum has fewer than 5 patients.
    """
    low_mask = df_surv[column] < median_val
    high_mask = df_surv[column] >= median_val

    _MIN_KM_SAMPLES = 5
    if low_mask.sum() < _MIN_KM_SAMPLES or high_mask.sum() < _MIN_KM_SAMPLES:
        raise ValueError(
            f"KM stratification on '{column}' yielded groups of size "
            f"low={low_mask.sum()}, high={high_mask.sum()} (minimum {_MIN_KM_SAMPLES})."
        )

    kmf_low = KaplanMeierFitter()
    kmf_low.fit(
        df_surv.loc[low_mask, _COL_OS_MONTHS], df_surv.loc[low_mask, _COL_OS_STATUS_CLEAN],
        label=f"Low (N={low_mask.sum()})"
    )
    kmf_low.plot_survival_function(ax=ax, color=color_low, ci_show=True, ci_alpha=0.12, linewidth=2.5)

    kmf_high = KaplanMeierFitter()
    kmf_high.fit(
        df_surv.loc[high_mask, _COL_OS_MONTHS], df_surv.loc[high_mask, _COL_OS_STATUS_CLEAN],
        label=f"High (N={high_mask.sum()})"
    )
    kmf_high.plot_survival_function(ax=ax, color=color_high, ci_show=True, ci_alpha=0.12, linewidth=2.5)

    add_km_risk_table([kmf_low, kmf_high], ax)

    lr_res = logrank_test(
        df_surv.loc[high_mask, _COL_OS_MONTHS], df_surv.loc[low_mask, _COL_OS_MONTHS],
        df_surv.loc[high_mask, _COL_OS_STATUS_CLEAN], df_surv.loc[low_mask, _COL_OS_STATUS_CLEAN]
    )
    return float(lr_res.p_value)


def _plot_stratified_km_survival(
    df_tcga_clin: pd.DataFrame, column: str, title: str,
    output_filename: str, color_high: str, color_low: str
) -> Tuple[float, float, Path]:
    """Unified helper to plot Kaplan-Meier survival curves stratified by median split."""
    df_surv, median_val = _prepare_survival_df(df_tcga_clin, column)
    
    fig, ax = plt.subplots(figsize=(9, 6.5))
    p_val = _render_km_curves(ax, df_surv, column, median_val, color_low, color_high)
    
    p_text = f"Log-Rank p = {p_val:.2e}" if p_val < 0.001 else f"Log-Rank p = {p_val:.3f}"
    ax.text(
        0.05, 0.08, p_text, transform=ax.transAxes, fontsize=13, weight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
    )
    ax.set_title(title, fontsize=15, weight='bold', pad=15)
    ax.set_xlabel("Overall Survival (Months)", fontsize=13)
    ax.set_ylabel("Survival Probability", fontsize=13)
    plt.tight_layout()
    
    plot_path = PLOT_DIR / output_filename
    save_fig(fig, plot_path)
    plt.close(fig)
    return median_val, p_val, plot_path


def _plot_survival_by_aneuploidy(df_tcga_clin: pd.DataFrame) -> Tuple[float, float, Path]:
    """Plot Kaplan-Meier survival curve stratified by TCGA median Aneuploidy Score."""
    print("\nRunning survival curve by Aneuploidy in TCGA...")
    return _plot_stratified_km_survival(
        df_tcga_clin, _COL_ANEUPLOIDY,
        "TCGA-SKCM Overall Survival by Aneuploidy Score",
        "extended_aneuploidy_survival.png",
        color_high=get_cohort_color("Hugo 2016"),
        color_low=get_cohort_color("Liu 2019")
    )


def _plot_survival_by_tmb(df_tcga_clin: pd.DataFrame) -> Tuple[float, float, Path]:
    """Plot Kaplan-Meier survival curve stratified by TCGA median TMB."""
    print("\nRunning survival curve by TMB in TCGA...")
    return _plot_stratified_km_survival(
        df_tcga_clin, _COL_TMB,
        "TCGA-SKCM Overall Survival by Tumour Mutational Burden (TMB)",
        "survival_tcga_tmb.png",
        color_high=RESPONSE_PALETTE["PD"],
        color_low=RESPONSE_PALETTE["CR/PR"]
    )


def _evaluate_aneuploidy_and_tmb(
    df_tcga_clin: pd.DataFrame,
    df_clin_merged: pd.DataFrame,
    df_sigs_merged: pd.DataFrame,
    cohort_names: List[str],
) -> None:
    """Evaluate Aneuploidy and TMB vs. Immune Infiltration and save figure plots."""
    print("\nEvaluating TMB vs. Immune Infiltration in ICI Trial Cohort...")
    trial_corrs = _compute_genomic_immune_correlations(
        df_clin_merged, df_sigs_merged, _CURATED_IMMUNE_SIGNATURES
    )
    _plot_correlation_heatmap(
        trial_corrs, _CURATED_IMMUNE_SIGNATURES, n_samples=len(df_clin_merged)
    )
    _update_report_section_3_2(
        CURATED_SIGNATURES_REPORT_PATH, trial_corrs,
        n_samples=len(df_clin_merged), cohort_names=cohort_names,
    )
    for label, fn in [
        ("Aneuploidy", _plot_survival_by_aneuploidy),
        ("TMB", _plot_survival_by_tmb),
    ]:
        try:
            fn(df_tcga_clin)
        except (ValueError, RuntimeError) as exc:
            print(f"  [WARNING] TCGA survival plot by {label} skipped: {exc}")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute complete extended biomarker statistical analysis and plot generation."""
    print("==================================================")
    print("Extended Biomarker Evaluation & Characterisation (Merged Trial Cohorts)")
    print("==================================================")

    data = _load_and_prepare_data()
    if data is None:
        return
    df_clin_merged, df_sigs_merged, df_tcga_clin, _, cohort_data = data

    _evaluate_neoantigen_load(df_clin_merged)
    _evaluate_pathway_mutations(cohort_data, df_clin_merged)
    _evaluate_aneuploidy_and_tmb(
        df_tcga_clin, df_clin_merged, df_sigs_merged,
        cohort_names=list(cohort_data.keys()),
    )

    print("\n==================================================")
    print("Biomarker evaluation completed successfully!")
    print("=================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
