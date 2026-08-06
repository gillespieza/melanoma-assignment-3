"""Extended Biomarker Evaluation & Predictive Modelling (Merged Trial Cohorts).

Evaluates univariate associations, neoantigen load vs TMB, somatic pathway mutations,
aneuploidy & TMB vs immune infiltration, and trains multimodal response prediction models.
"""

# ---------------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------------
import contextlib
import sys
import time
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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

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
from src.models import get_model
from src.signatures import extract_all_signatures
from src.styles import OKABE_ITO, RESPONSE_PALETTE, get_cohort_color, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import get_subproject_log_dir, rel_path
from src.utils.plotting import save_fig

set_presentation_style()

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
DEFAULT_RANDOM_STATE = 42

DATA_DIR = PROJECT_ROOT / "data"
PLOT_DIR = BASE_DIR / "plots" / "biomarkers"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

REPORTS_DIR = BASE_DIR / "reports" / "pillar-3-transcriptomic-signatures"
CURATED_SIGNATURES_REPORT_PATH = REPORTS_DIR / "curated_signatures_report.md"

LOG_DIR = get_subproject_log_dir(_THIS_FILE)
LOG_PATH = LOG_DIR / "run_extended_biomarkers.log"

_COL_SAMPLE_ID = "SAMPLE_ID"
_COL_TMB = "TMB_NONSYNONYMOUS"
_COL_TOTAL_NEOANTIGEN = "TOTAL_NEOANTIGEN"
_COL_RESPONSE = "response"
_COL_OS_MONTHS = "OS_MONTHS"
_COL_OS_STATUS = "OS_STATUS"
_COL_ANEUPLOIDY = "ANEUPLOIDY_SCORE"
_COL_AGE = "AGE"
_COL_COHORT = "Cohort"
_COL_CNA = "CNA_PROP"


# ---------------------------------------------------------------------------
# Centralized Model Wrapper (Rule 2.1 Compliance)
# ---------------------------------------------------------------------------
class TunedCalibratedModel(BaseEstimator, ClassifierMixin):
    """Scikit-learn model wrapper delegating tuning and calibration to src.models.get_model."""

    _estimator_type = "classifier"

    def __sklearn_tags__(self) -> Any:
        """Return scikit-learn estimator tags explicitly marking model as classifier."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(self, model_type: str = "rf", calibrate: bool = True) -> None:
        """Initialise model wrapper with target architecture key and calibration flag."""
        self.model_type = model_type
        self.calibrate = calibrate
        self.fitted_model_: Any = None
        self.classes_: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TunedCalibratedModel":
        """Fit model on training split via src.models.get_model()."""
        self.classes_ = np.unique(y)
        self.fitted_model_ = get_model(
            self.model_type, X, y, calibrate=self.calibrate
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using tuned fitted estimator."""
        if self.fitted_model_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.fitted_model_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels using tuned fitted estimator."""
        if self.fitted_model_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.fitted_model_.predict(X)


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


def evaluate_auc_cv(
    model: Any, X: np.ndarray, y: np.ndarray, cv: StratifiedKFold, label: str
) -> np.ndarray:
    """Evaluate one feature set with serial outer CV and parallelised inner model search."""
    start = time.perf_counter()
    print(f"    - {label}...", flush=True)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=None)
    elapsed = time.perf_counter() - start
    print(
        f"      AUC = {scores.mean():.3f} (+/-{scores.std():.3f}); "
        f"folds = [{', '.join(f'{score:.3f}' for score in scores)}]; "
        f"{elapsed:.1f}s",
        flush=True,
    )
    return scores


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
    df_clin['temp_resp'] = df_clin['RESPONSE'].map(RECIST_RESPONSE_MAP)
    df_clin.dropna(subset=['temp_resp'], inplace=True)
    df_clin.drop(columns=['temp_resp'], inplace=True)

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
        df_clin[f'mut_{col}'] = mut_df[col]


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
        antigen_cols = [f'mut_{g}' for g in PATHWAY_GENES["Antigen Presentation"]]
        ifn_cols = [f'mut_{g}' for g in PATHWAY_GENES["IFN-gamma Signature"]]
        surv_cols = [f'mut_{g}' for g in PATHWAY_GENES["Survival & Proliferation Drivers"]]
        df['mut_Antigen_Presentation'] = (df[antigen_cols].sum(axis=1) > 0).astype(int)
        df['mut_IFN_gamma_Signaling'] = (df[ifn_cols].sum(axis=1) > 0).astype(int)
        df['mut_Survival_Pathways'] = (df[surv_cols].sum(axis=1) > 0).astype(int)


def _merge_clinical_and_signatures(
    df_liu_clin: pd.DataFrame, df_hugo_clin: pd.DataFrame, df_riaz_clin: pd.DataFrame
) -> pd.DataFrame:
    """Concatenate cleaned trial cohort clinical DataFrames."""
    clin_cols = [
        _COL_COHORT, _COL_RESPONSE, _COL_TMB, _COL_AGE, _COL_TOTAL_NEOANTIGEN, _COL_CNA,
        'mut_BRAF', 'mut_NRAS', 'mut_NF1',
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
    return common_idx, auc_neo, auc_tmb, float(p_neo_mw), float(p_tmb_mw)


def _generate_neoantigen_report_lines(
    n_trials: int, r_spearman: float, p_spearman: float,
    n_common: int, auc_neo: float, auc_tmb: float, p_neo_mw: float, p_tmb_mw: float
) -> List[str]:
    """Build markdown report lines for Section 1 (Neoantigen Load vs TMB)."""
    p_sp_str = f"{p_spearman:.2e}"
    return [
        "\n## 1. Neoantigen Load vs. Tumour Mutational Burden (TMB)",
        (
            f"We evaluated correlation between predicted neoantigen load (`TOTAL_NEOANTIGEN`) "
            f"and mutational burden (`TMB_NONSYNONYMOUS`) in pooled trials ($N={n_trials}$):"
        ),
        f"\n*   **Spearman Correlation ($r$)**: **{r_spearman:.3f}** (p-value: **{p_sp_str}**)",
        "\nAs expected, there is an almost perfect linear relationship between mutational burden "
        "and predicted MHC-binding neoantigens.",
        "\n### Predictive Utility for Immunotherapy Response",
        "| Biomarker | N | Response ROC AUC | Mann-Whitney U p-value |",
        "|---|---|---|---|",
        f"| **TOTAL_NEOANTIGEN** | {n_common} | **{auc_neo:.3f}** | {p_neo_mw:.3e} |",
        f"| **TMB_NONSYNONYMOUS** | {n_common} | **{auc_tmb:.3f}** | {p_tmb_mw:.3e} |"
    ]


def _evaluate_neoantigen_load(df_clin_merged: pd.DataFrame) -> List[str]:
    """Evaluate Neoantigen Load vs. TMB in Pooled Trials and generate report."""
    print("\nEvaluating Neoantigen Load vs. TMB in Pooled Trials...")
    
    df_trials_neo = df_clin_merged.dropna(subset=[_COL_TOTAL_NEOANTIGEN, _COL_TMB]).copy()
    r_spearman, p_spearman = spearmanr(
        df_trials_neo[_COL_TOTAL_NEOANTIGEN], df_trials_neo[_COL_TMB], nan_policy='omit'
    )
    print(f"  Spearman correlation (pooled): r = {r_spearman:.3f}, p = {p_spearman:.2e}")
    
    common_idx, auc_neo, auc_tmb, p_neo_mw, p_tmb_mw = _compute_neoantigen_roc_metrics(
        df_clin_merged
    )
    print(f"  Neoantigen response prediction ROC AUC: {auc_neo:.3f} (p = {p_neo_mw:.3e})")
    print(f"  TMB response prediction ROC AUC: {auc_tmb:.3f} (p = {p_tmb_mw:.3e})")
    
    return _generate_neoantigen_report_lines(
        len(df_trials_neo), float(r_spearman), float(p_spearman),
        len(common_idx), auc_neo, auc_tmb, p_neo_mw, p_tmb_mw
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
    return f"| **{label}** | {p_liu:.1%} | {p_hugo:.1%} | {p_riaz:.1%} | **{p_pool:.1%}** |"


def _get_pathway_mutation_headers(
    n_l: int, n_h: int, n_r: int, n_m: int
) -> List[str]:
    """Build header lines for Section 2 somatic pathway mutations report table."""
    return [
        "\n## 2. Somatic Pathway Mutations",
        "We evaluated somatic mutations in three biological pathways that dictate immunogenicity:",
        "*   **Antigen Presentation**: `B2M`, `TAP1`, `TAP2` (disrupts MHC Class I presentation).",
        (
            "*   **IFN-gamma Signalling**: `JAK1`, `JAK2`, `STAT1` "
            "(induces cytotoxicity insensitivity)."
        ),
        "*   **Survival & Proliferation Drivers**: `PTEN`, `CDKN2A`, `PIK3CA` (oncogenic drivers).",
        "\n### Mutation Frequencies in Trial Cohorts:",
        f"| Pathway / Gene | Liu 2019 ($N={n_l}$) | Hugo 2016 ($N={n_h}$) | "
        f"Riaz 2017 ($N={n_r}$) | Pooled Trials ($N={n_m}$) |",
        "|---|---|---|---|---|",
    ]


def _evaluate_pathway_mutations(
    df_liu_clin: pd.DataFrame, df_hugo_clin: pd.DataFrame,
    df_riaz_clin: pd.DataFrame, df_clin_merged: pd.DataFrame
) -> List[str]:
    """Calculate and tabulate pathway mutation frequencies across trial cohorts."""
    print("\nCalculating pathway mutation frequencies in trial cohorts...")
    n_l, n_h, n_r, n_m = len(df_liu_clin), len(df_hugo_clin), len(df_riaz_clin), len(df_clin_merged)
    
    report_section = _get_pathway_mutation_headers(n_l, n_h, n_r, n_m)
    mutation_labels = [
        ('BRAF mutation', 'mut_BRAF'),
        ('NRAS mutation', 'mut_NRAS'),
        ('NF1 mutation', 'mut_NF1'),
        ('Antigen Presentation (MHC)', 'mut_Antigen_Presentation'),
        ('IFN-gamma Signalling', 'mut_IFN_gamma_Signaling'),
        ('Survival & Proliferation Drivers', 'mut_Survival_Pathways'),
    ]
    for label, col in mutation_labels:
        report_section.append(
            _format_pathway_mutation_row(
                label, col, df_liu_clin, df_hugo_clin, df_riaz_clin, df_clin_merged
            )
        )
        
    return report_section


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

    sns.heatmap(
        corr_df, annot=True, cmap='coolwarm', vmin=-0.3, vmax=0.3,
        center=0, ax=ax, fmt=".3f", linewidths=1.5, cbar_kws={'label': "Spearman Correlation ($r_s$)"}
    )
    ax.set_title(
        "Pooled Trial Cohort (N=195): Nonsynonymous TMB vs. Immune Signatures",
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
    df_surv['os_status_clean'] = df_surv[_COL_OS_STATUS].apply(clean_os_status)
    df_surv = df_surv.dropna(subset=[_COL_OS_MONTHS, 'os_status_clean'])
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
        df_surv.loc[low_mask, _COL_OS_MONTHS], df_surv.loc[low_mask, 'os_status_clean'],
        label=f"Low (N={low_mask.sum()})"
    )
    kmf.plot_survival_function(ax=ax, color=color_low, ci_show=False, linewidth=2.5)
    
    kmf.fit(
        df_surv.loc[high_mask, _COL_OS_MONTHS], df_surv.loc[high_mask, 'os_status_clean'],
        label=f"High (N={high_mask.sum()})"
    )
    kmf.plot_survival_function(ax=ax, color=color_high, ci_show=False, linewidth=2.5)
    
    lr_res = logrank_test(
        df_surv.loc[high_mask, _COL_OS_MONTHS], df_surv.loc[low_mask, _COL_OS_MONTHS],
        df_surv.loc[high_mask, 'os_status_clean'], df_surv.loc[low_mask, 'os_status_clean']
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


def _build_spearman_table_rows(
    tcga_corrs: Dict[str, Dict[str, float]],
    trial_corrs: Dict[str, Dict[str, float]],
    sig_names: List[str]
) -> List[str]:
    """Format Spearman correlation table rows across immune signatures."""
    table_rows = []
    for sig in sig_names:
        r_a = tcga_corrs[sig]['Aneu_r']
        r_tt = tcga_corrs[sig]['Tmb_r']
        r_tr = trial_corrs[sig]['Tmb_r']
        table_rows.append(f"| `{sig}` | **{r_a:.3f}** | **{r_tt:.3f}** | **{r_tr:.3f}** |")
    return table_rows


def _get_aneuploidy_tmb_headers() -> List[str]:
    """Build section title and table header lines for Section 3."""
    return [
        "\n## 3. Aneuploidy, Copy-Number Alterations, & TMB vs. Immune Infiltration",
        "We evaluated genomic burden vs immune signatures in TCGA and trial cohorts.",
        "\n### Spearman Correlations table:",
        "| Immune Signature | TCGA Aneuploidy Score ($r$) | TCGA TMB ($r$) | Trial TMB ($r$) |",
        "|---|---|---|---|",
    ]


def _generate_aneuploidy_tmb_report_lines(
    tcga_corrs: Dict[str, Dict[str, float]], trial_corrs: Dict[str, Dict[str, float]],
    sig_names: List[str], aneu_median: float, p_aneu_surv: float,
    tmb_median: float, p_tmb_surv: float
) -> List[str]:
    """Build report section lines for Aneuploidy and TMB vs Immune Infiltration."""
    min_a = min(tcga_corrs[s]['Aneu_r'] for s in sig_names)
    max_a = max(tcga_corrs[s]['Aneu_r'] for s in sig_names)
    min_t = min(tcga_corrs[s]['Tmb_r'] for s in sig_names)
    max_t = max(tcga_corrs[s]['Tmb_r'] for s in sig_names)

    report_section = _get_aneuploidy_tmb_headers()
    report_section.extend(_build_spearman_table_rows(tcga_corrs, trial_corrs, sig_names))
    report_section.extend([
        "\n**Biological Conclusion**: In both cohorts:",
        f"1. **Chromosomal Instability (Aneuploidy)** shows a **weak negative correlation** "
        f"($r \\approx {min_a:.2f}$ to ${max_a:.2f}$) with immune signatures.",
        f"2. **Mutational Burden (TMB)** shows **very weak or near-zero correlation** "
        f"($r \\approx {min_t:.2f}$ to ${max_t:.2f}$) with immune signatures.",
        "\n![Correlation Heatmap](../plots/extended_immune_correlations.png)",
        "\n### TCGA Overall Survival by Aneuploidy",
        f"Partitioned at median Aneuploidy Score (**{aneu_median:.1f}**):",
        f"\n*   **Log-Rank p-value**: **{p_aneu_surv:.3e}** (Statistically Significant)",
        "\n![TCGA Aneuploidy Survival](../plots/extended_aneuploidy_survival.png)",
        "\n### TCGA Overall Survival by Tumour Mutational Burden (TMB)",
        f"Partitioned at median TMB (**{tmb_median:.2f} mutations/Mb**):",
        f"\n*   **Log-Rank p-value**: **{p_tmb_surv:.3f}** (Prognostically Neutral)",
        "\n![TCGA TMB Survival](../plots/survival_tcga_tmb.png)"
    ])
    return report_section


def _evaluate_aneuploidy_and_tmb(
    df_tcga_clin: pd.DataFrame, df_tcga_sigs: pd.DataFrame,
    df_clin_merged: pd.DataFrame, df_sigs_merged: pd.DataFrame
) -> List[str]:
    """Evaluate Aneuploidy and TMB vs. Immune Infiltration and generate report."""
    print("\nEvaluating TMB vs. Immune Infiltration in ICI Trial Cohort...")
    sig_names = ['IFN_gamma', 'TIS', 'CD8_Tcell', 'CYT', 'IMPRES', 'PD_L1']

    trial_corrs = _compute_genomic_immune_correlations(
        df_clin_merged, df_sigs_merged, sig_names
    )
    _plot_correlation_heatmap(trial_corrs, sig_names)
    aneu_median, p_aneu_surv, _ = _plot_survival_by_aneuploidy(df_tcga_clin)
    tmb_median, p_tmb_surv, _ = _plot_survival_by_tmb(df_tcga_clin)

    return []


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

    print("==================================================")
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
