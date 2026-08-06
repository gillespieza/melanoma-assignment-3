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
from src.signatures import extract_all_signatures
from src.styles import RESPONSE_PALETTE, get_cohort_color, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import get_subproject_log_dir, rel_path
from src.utils.plotting import save_fig

set_presentation_style()

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
PLOT_DIR = BASE_DIR / "plots" / "biomarkers"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

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
def _get_raw_cohort_paths() -> List[Path]:
    """Return mandatory input file paths for raw clinical and expression data."""
    return [
        DATA_DIR / "processed/liu_2019/clin_cleaned.csv",
        DATA_DIR / "processed/liu_2019/expr_cleaned.csv",
        DATA_DIR / "processed/hugo_2016/clin_cleaned.csv",
        DATA_DIR / "processed/hugo_2016/expr_cleaned.csv",
        DATA_DIR / "processed/riaz_2017/clin_cleaned.csv",
        DATA_DIR / "processed/riaz_2017/expr_cleaned.csv",
        DATA_DIR / "processed/skcm_tcga_pan_can_atlas_2018/clin_cleaned.csv",
        DATA_DIR / "processed/skcm_tcga_pan_can_atlas_2018/expr_cleaned.csv",
    ]


def _load_raw_cohort_files() -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
] | None:
    """Load raw clinical and expression DataFrames for trial cohorts and TCGA."""
    paths = _get_raw_cohort_paths()
    if not all(p.exists() for p in paths):
        print("Error: Cleansed processed files not found. Run clean_data.py first.")
        for p in paths:
            print(f"  {p.name}: {'FOUND' if p.exists() else 'MISSING'} ({rel_path(p)})")
        return None

    return (
        pd.read_csv(paths[0], index_col=_COL_SAMPLE_ID),
        pd.read_csv(paths[1], index_col=0),
        pd.read_csv(paths[2], index_col=_COL_SAMPLE_ID),
        pd.read_csv(paths[3], index_col=0),
        pd.read_csv(paths[4], index_col=_COL_SAMPLE_ID),
        pd.read_csv(paths[5], index_col=0),
        pd.read_csv(paths[6], index_col=_COL_SAMPLE_ID),
        pd.read_csv(paths[7]),
    )


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

    df_tcga_sigs.index = df_tcga_sigs.index.str.upper().str[:12]
    df_tcga_clin.index = df_tcga_clin.index.str.upper().str[:12]
    
    df_tcga_sigs = df_tcga_sigs.groupby(df_tcga_sigs.index).first()
    df_tcga_clin_aligned = df_tcga_clin.groupby(df_tcga_clin.index).first()
    
    common = df_tcga_sigs.index.intersection(df_tcga_clin_aligned.index)
    return df_tcga_clin_aligned.loc[common], df_tcga_sigs.loc[common]


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
    df_liu_clin: pd.DataFrame, df_hugo_clin: pd.DataFrame, df_riaz_clin: pd.DataFrame
) -> None:
    """Load somatic pathway mutations across all 3 trial cohorts."""
    pathway_genes_flat = [g for genes in PATHWAY_GENES.values() for g in genes]
    all_genes = sorted(list(set(pathway_genes_flat + DRIVER_GENES)))
    
    _attach_cohort_mutations(df_liu_clin, "liu_2019", all_genes)
    _attach_cohort_mutations(df_hugo_clin, "hugo_2016", all_genes)
    _attach_cohort_mutations(df_riaz_clin, "riaz_2017", all_genes)

    for df in [df_liu_clin, df_hugo_clin, df_riaz_clin]:
        antigen_cols = [f"{_PREFIX_MUT}{g}" for g in PATHWAY_GENES["Antigen Presentation"]]
        ifn_cols = [f"{_PREFIX_MUT}{g}" for g in PATHWAY_GENES["IFN-gamma Signature"]]
        surv_cols = [f"{_PREFIX_MUT}{g}" for g in PATHWAY_GENES["Survival & Proliferation Drivers"]]
        df['mut_Antigen_Presentation'] = (df[antigen_cols].sum(axis=1) > 0).astype(int)
        df['mut_IFN_gamma_Signaling'] = (df[ifn_cols].sum(axis=1) > 0).astype(int)
        df['mut_Survival_Pathways'] = (df[surv_cols].sum(axis=1) > 0).astype(int)


def _merge_clinical_and_signatures(
    df_liu_clin: pd.DataFrame, df_hugo_clin: pd.DataFrame, df_riaz_clin: pd.DataFrame
) -> pd.DataFrame:
    """Concatenate cleaned trial cohort clinical DataFrames."""
    clin_cols = [
        _COL_COHORT, _COL_RESPONSE, _COL_TMB, _COL_AGE, _COL_TOTAL_NEOANTIGEN, _COL_CNA,
        f"{_PREFIX_MUT}BRAF", f"{_PREFIX_MUT}NRAS", f"{_PREFIX_MUT}NF1",
        'mut_Antigen_Presentation', 'mut_IFN_gamma_Signaling', 'mut_Survival_Pathways'
    ]
    for df in [df_liu_clin, df_hugo_clin, df_riaz_clin]:
        if _COL_CNA not in df.columns:
            df[_COL_CNA] = np.nan
        if _COL_AGE not in df.columns:
            df[_COL_AGE] = np.nan

    return pd.concat([df_liu_clin[clin_cols], df_hugo_clin[clin_cols], df_riaz_clin[clin_cols]])


def _prepare_cohort_labels_and_sigs(
    df_liu_clin: pd.DataFrame, df_liu_expr: pd.DataFrame,
    df_hugo_clin: pd.DataFrame, df_hugo_expr: pd.DataFrame,
    df_riaz_clin: pd.DataFrame, df_riaz_expr: pd.DataFrame
) -> pd.DataFrame:
    """Annotate cohort names and extract concatenated z-scored expression signatures."""
    df_liu_clin[_COL_COHORT] = 'Liu 2019'
    df_hugo_clin[_COL_COHORT] = 'Hugo 2016'
    df_riaz_clin[_COL_COHORT] = 'Riaz 2017'

    print("\nComputing expression signatures for all cohorts...")
    return pd.concat([
        zscore_df(extract_all_signatures(df_liu_expr.loc[df_liu_clin.index])),
        zscore_df(extract_all_signatures(df_hugo_expr.loc[df_hugo_clin.index])),
        zscore_df(extract_all_signatures(df_riaz_expr.loc[df_riaz_clin.index]))
    ])


def _load_and_prepare_data() -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame
] | None:
    """Load and preprocess clinical, expression, and mutation data for trial cohorts and TCGA."""
    raw_data = _load_raw_cohort_files()
    if raw_data is None:
        return None
        
    (
        df_liu_clin, df_liu_expr, df_hugo_clin, df_hugo_expr,
        df_riaz_clin, df_riaz_expr, df_tcga_clin, df_tcga_expr_raw
    ) = raw_data
    
    _align_clinical_column_aliases([df_liu_clin, df_hugo_clin, df_riaz_clin, df_tcga_clin])
    for df in [df_liu_clin, df_hugo_clin, df_riaz_clin]:
        _filter_and_calculate_neoantigens(df)
    
    df_sigs_merged = _prepare_cohort_labels_and_sigs(
        df_liu_clin, df_liu_expr, df_hugo_clin, df_hugo_expr, df_riaz_clin, df_riaz_expr
    )

    df_tcga_clin, df_tcga_sigs = _process_tcga_signatures(df_tcga_clin, df_tcga_expr_raw)
    _attach_pathway_mutations(df_liu_clin, df_hugo_clin, df_riaz_clin)
    df_clin_merged = _merge_clinical_and_signatures(df_liu_clin, df_hugo_clin, df_riaz_clin)
    
    return (
        df_clin_merged, df_sigs_merged, df_tcga_clin, df_tcga_sigs,
        df_liu_clin, df_hugo_clin, df_riaz_clin
    )


# ---------------------------------------------------------------------------
# Section 1: Neoantigen Load vs. TMB
# ---------------------------------------------------------------------------
def _compute_neoantigen_roc_metrics(
    df_clin_merged: pd.DataFrame
) -> Tuple[pd.Index, float, float, float, float]:
    """Compute ROC-AUC and Mann-Whitney U test metrics for Neoantigen Load and TMB."""
    y_true_trials = df_clin_merged[_COL_RESPONSE].dropna()
    neo_valid = df_clin_merged[_COL_TOTAL_NEOANTIGEN].dropna().index
    common_idx = y_true_trials.index.intersection(neo_valid)
    
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
def _format_pathway_mutation_row(
    label: str, col: str,
    df_liu_clin: pd.DataFrame, df_hugo_clin: pd.DataFrame,
    df_riaz_clin: pd.DataFrame, df_clin_merged: pd.DataFrame
) -> str:
    """Format single pathway mutation frequency table row across cohorts."""
    p_liu = df_liu_clin[col].mean()
    p_hugo = df_hugo_clin[col].mean()
    p_riaz = df_riaz_clin[col].mean()
    p_pool = df_clin_merged[col].mean()
    return (
        f"  {label:<32} | Liu: {p_liu:.1%} | Hugo: {p_hugo:.1%} | "
        f"Riaz: {p_riaz:.1%} | Pooled: {p_pool:.1%}"
    )


def _evaluate_pathway_mutations(
    df_liu_clin: pd.DataFrame, df_hugo_clin: pd.DataFrame,
    df_riaz_clin: pd.DataFrame, df_clin_merged: pd.DataFrame
) -> None:
    """Calculate and tabulate pathway mutation frequencies across trial cohorts."""
    print("\nCalculating pathway mutation frequencies in trial cohorts...")
    mutation_labels = [
        ('BRAF mutation', f"{_PREFIX_MUT}BRAF"),
        ('NRAS mutation', f"{_PREFIX_MUT}NRAS"),
        ('NF1 mutation', f"{_PREFIX_MUT}NF1"),
        ('Antigen Presentation (MHC)', 'mut_Antigen_Presentation'),
        ('IFN-gamma Signalling', 'mut_IFN_gamma_Signaling'),
        ('Survival & Proliferation Drivers', 'mut_Survival_Pathways'),
    ]
    for label, col in mutation_labels:
        row_str = _format_pathway_mutation_row(
            label, col, df_liu_clin, df_hugo_clin, df_riaz_clin, df_clin_merged
        )
        print(row_str)


# ---------------------------------------------------------------------------
# Section 3: Aneuploidy and TMB vs Immune Infiltration
# ---------------------------------------------------------------------------
def _compute_genomic_immune_correlations(
    df_clin_merged: pd.DataFrame, df_sigs_merged: pd.DataFrame, sig_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compute Spearman correlations between TMB and immune signatures in pooled trial cohort."""
    df_sigs_aligned = df_sigs_merged.loc[df_clin_merged.index]
    trial_corrs = {}
    for sig in sig_names:
        r_t, p_t = spearmanr(
            df_clin_merged[_COL_TMB], df_sigs_aligned[sig], nan_policy='omit'
        )
        trial_corrs[sig] = {'Tmb_r': float(r_t), 'Tmb_p': float(p_t)}
    return trial_corrs


def _plot_correlation_heatmap(
    trial_corrs: Dict[str, Dict[str, float]],
    sig_names: List[str]
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
        f"Pooled Trial Cohort (N={len(corr_series)}): Nonsynonymous TMB vs. Immune Signatures",
        fontsize=12, weight='bold', pad=15
    )
    ax.set_ylabel("Curated Immune Signatures", fontsize=11, weight='bold')
    plt.tight_layout()
    plot_path = PLOT_DIR / "extended_immune_correlations.png"
    save_fig(fig, plot_path)
    return plot_path


def _prepare_survival_df(
    df_tcga_clin: pd.DataFrame, column: str
) -> Tuple[pd.DataFrame, float]:
    """Clean survival status, filter missing values, and calculate column median."""
    df_surv = df_tcga_clin.dropna(subset=[_COL_OS_MONTHS, _COL_OS_STATUS, column]).copy()
    df_surv[_COL_OS_MONTHS] = pd.to_numeric(df_surv[_COL_OS_MONTHS], errors='coerce')
    df_surv[_COL_OS_STATUS_CLEAN] = df_surv[_COL_OS_STATUS].apply(clean_os_status)
    df_surv = df_surv.dropna(subset=[_COL_OS_MONTHS, _COL_OS_STATUS_CLEAN])
    median_val = float(df_surv[column].median())
    return df_surv, median_val


def _render_km_curves(
    ax: plt.Axes, df_surv: pd.DataFrame, column: str, median_val: float,
    color_low: str, color_high: str
) -> float:
    """Fit Kaplan-Meier survival curves and return log-rank p-value."""
    kmf = KaplanMeierFitter()
    low_mask = df_surv[column] < median_val
    high_mask = df_surv[column] >= median_val
    
    kmf.fit(
        df_surv.loc[low_mask, _COL_OS_MONTHS], df_surv.loc[low_mask, _COL_OS_STATUS_CLEAN],
        label=f"Low (N={low_mask.sum()})"
    )
    kmf.plot_survival_function(ax=ax, color=color_low, ci_show=False, linewidth=2.5)
    
    kmf.fit(
        df_surv.loc[high_mask, _COL_OS_MONTHS], df_surv.loc[high_mask, _COL_OS_STATUS_CLEAN],
        label=f"High (N={high_mask.sum()})"
    )
    kmf.plot_survival_function(ax=ax, color=color_high, ci_show=False, linewidth=2.5)
    
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
    df_tcga_clin: pd.DataFrame, df_tcga_sigs: pd.DataFrame,
    df_clin_merged: pd.DataFrame, df_sigs_merged: pd.DataFrame
) -> None:
    """Evaluate Aneuploidy and TMB vs. Immune Infiltration and save figure plots."""
    print("\nEvaluating TMB vs. Immune Infiltration in ICI Trial Cohort...")
    trial_corrs = _compute_genomic_immune_correlations(
        df_clin_merged, df_sigs_merged, _CURATED_IMMUNE_SIGNATURES
    )
    _plot_correlation_heatmap(trial_corrs, _CURATED_IMMUNE_SIGNATURES)
    _plot_survival_by_aneuploidy(df_tcga_clin)
    _plot_survival_by_tmb(df_tcga_clin)


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
    (
        df_clin_merged, df_sigs_merged, df_tcga_clin, df_tcga_sigs,
        df_liu_clin, df_hugo_clin, df_riaz_clin
    ) = data

    _evaluate_neoantigen_load(df_clin_merged)
    _evaluate_pathway_mutations(df_liu_clin, df_hugo_clin, df_riaz_clin, df_clin_merged)
    _evaluate_aneuploidy_and_tmb(df_tcga_clin, df_tcga_sigs, df_clin_merged, df_sigs_merged)

    print("\n==================================================")
    print("Biomarker evaluation completed successfully!")
    print("==================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
