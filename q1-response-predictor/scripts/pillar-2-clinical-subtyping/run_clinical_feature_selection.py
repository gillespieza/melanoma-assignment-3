"""
Two-Tiered Clinical & Transcriptomic Feature Selection Pipeline.

Tier 1 (ICI Cohort Consensus, N≈256 across 3 cohorts):
  Evaluates 6 transcriptomic immune signatures (IFN-gamma, TIS, CYT, CD8 T-cell,
  PD-L1, IMPRES), TMB, Age, and Sex for anti-PD-1 Binary Response using Random
  Forest Gini Importance and Univariate/Multivariate Logistic Regression with
  Benjamini-Hochberg FDR adjustment.

Tier 2 (ICI Granular Clinical Feature Selection, N≈256):
  Evaluates baseline clinical covariates available in the ICI trial cohorts
  (CLINICAL_STAGE, BIOPSY_SITE, TISSUE_SUBTYPE, PRIOR_ICI_RX, PRIOR_RX,
  SAMPLE_TREATMENT, METASTASIZED) against anti-PD-1 Binary Response using
  the same Random Forest + Logistic Regression two-step framework.

All sample sizes and statistical metrics are derived at runtime from live DataFrames;
no hardcoded numeric literals appear in report text. Exports presentation-ready plots
and an Obsidian-compatible Markdown report.
"""

import contextlib
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import seaborn as sns

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

_SUBPROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBPROJECT_ROOT))

from src.biology_constants import KEY_DRIVER_MUTATIONS
from src.config.constants import IMMUNE_SIGNATURE_LABELS
from src.config.datasets import DatasetConfig, load_dataset_config
from src.models import get_model
from src.signatures import extract_all_signatures
from src.styles import (
    DARK_SLATE_CHARCOAL,
    MODEL_TYPE_PALETTE,
    RESPONSE_PALETTE,
    set_presentation_style,
)
from src.utils.execution import run_companion_scripts
from src.utils.formatting import (
    generate_obsidian_frontmatter,
    generate_script_reference_callout,
)
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, PLOTS_DIR, get_subproject_log_dir, rel_path
from src.utils.plotting import save_fig

set_presentation_style()

# ---------------------------------------------------------------------------
# Module-level Constants & Directory Paths
# ---------------------------------------------------------------------------

CONFIG_PATH = _SUBPROJECT_ROOT / "config" / "datasets.yaml"
PLOT_DIR = PLOTS_DIR / "clinical"
REPORT_DIR = _SUBPROJECT_ROOT / "reports" / "pillar-2-clinical-subtyping"
REPORT_PATH = REPORT_DIR / "feature_selection_report.md"

_LOG_DIR = get_subproject_log_dir(Path(__file__))
LOG_PATH = _LOG_DIR / "run_clinical_feature_selection.log"

_COMPANION_SCRIPTS: Tuple[Tuple[str, Path], ...] = (
    (
        "Univariate Clinical & Genomic Associations",
        _SUBPROJECT_ROOT
        / "scripts"
        / "pillar-2-clinical-subtyping"
        / "run_univariate_associations.py",
    ),
)

ICI_COHORT_NAMES: frozenset = frozenset({"Liu 2019", "Hugo 2016", "Riaz 2017"})

TIER1_CONT_COLS: List[str] = [
    "IFN_gamma", "TIS", "CYT", "CD8_Tcell", "IMPRES", "PD_L1",
    "M1_M2_Ratio", "Macrophage_STV_Score",
    "SNV_NEOANTIGEN", "INDEL_NEOANTIGEN",
    "TMB_NONSYNONYMOUS", "AGE",
]

TIER1_BIN_COLS: List[str] = [
    "mut_BRAF_V600", "mut_BRAF_nonV600", "mut_NRAS", "mut_NF1"
]

# Canonical display order for Tier 1 uni vs multivariate forest plot.
# Immune signatures follow the project-mandated order (AGENTS.md), then myeloid,
# genomic burden, driver mutations, and host demographics.
TIER1_PLOT_ORDER: List[str] = [
    "Z_IFN_gamma", "Z_TIS", "Z_CYT", "Z_CD8_Tcell", "Z_IMPRES", "Z_PD_L1",
    "Z_M1_M2_Ratio", "Z_Macrophage_STV_Score",
    "Z_TMB_NONSYNONYMOUS", "Z_SNV_NEOANTIGEN", "Z_INDEL_NEOANTIGEN",
    "mut_BRAF_V600", "mut_BRAF_nonV600", "mut_NRAS", "mut_NF1",
    "SEX_Male", "SEX_Female", "Z_AGE",
]

TIER2_CANDIDATE_COLS: List[str] = [
    "CLINICAL_STAGE", "BIOPSY_SITE", "TISSUE_SUBTYPE", "PRIOR_ICI_RX",
    "PRIOR_RX", "SAMPLE_TREATMENT", "METASTASIZED"
]

_FDR_ALPHA: float = 0.05
_CI_Z_SCORE: float = 1.96
_MAX_SE_THRESHOLD: float = 10.0
_MAX_COEF_THRESHOLD: float = 10.0
_MAX_MULTI_FEATURES: int = 8
_LOGIT_MAX_ITER: int = 300

_COLOR_NULL_LINE: str = DARK_SLATE_CHARCOAL
_COLOR_GRID: str = "#E0E0E0"
_COLOR_TEXT_SIG: str = "#222222"
_COLOR_TEXT_INSIG: str = "#666666"
_COLOR_LEGEND_EDGE: str = "#CCCCCC"

_COL_RESPONSE_BINARY: str = "RESPONSE_BINARY"
_COL_FEATURE: str = "Feature"
_COL_IMPORTANCE: str = "Importance"
_COL_FORMATTED_FEATURE: str = "Formatted_Feature"
_COL_ODDS_RATIO: str = "Odds Ratio (OR)"
_COL_P_VALUE: str = "p-value"
_COL_FDR_ADJ_P: str = "FDR_adj_p"
_COL_OR_LOWER_95: str = "OR lower 95%"
_COL_OR_UPPER_95: str = "OR upper 95%"
_COL_COHORT: str = "COHORT"
_COL_SAMPLE_ID: str = "SAMPLE_ID"
_STR_NA: str = "N/A"

_FONT_TITLE_MAIN: float = 13.5
_FONT_TITLE_SUB: float = 13.0
_FONT_LABEL: float = 11.0
_FONT_TICK: float = 10.5
_FONT_ANNOTATION: float = 8.5
_FONT_ANNOTATION_SM: float = 8.0
_FONT_LEGEND: float = 8.5
_POINT_SIZE_FOREST: int = 75
_POINT_SIZE_COMP: int = 70
_LINEWIDTH_FOREST: float = 2.0
_LINEWIDTH_NULL: float = 1.2

_FEATURE_CLEAN_MAP: Dict[str, str] = {
    **IMMUNE_SIGNATURE_LABELS,
    "M1_M2_Ratio": "M1/M2 Macrophage Ratio",
    "Macrophage_STV_Score": "Macrophage STV Score",
    "SNV_NEOANTIGEN": "SNV Neoantigen Burden",
    "INDEL_NEOANTIGEN": "Indel Neoantigen Burden",
    "TMB_NONSYNONYMOUS": "Tumour Mutational Burden (TMB)",
    "AGE": "Patient Age",
    "SEX_Male": "Sex: Male",
    "SEX_Female": "Sex: Female",
    "mut_BRAF_V600": "BRAF V600 Mutation (V600E/K)",
    "mut_BRAF_nonV600": "BRAF Non-V600 Mutation",
    "mut_NRAS": "NRAS Mutation",
    "mut_NF1": "NF1 Mutation",
    "SAMPLE_TYPE_Primary": "Sample Type: Primary Tumour",
    "SAMPLE_TYPE_Metastatic": "Sample Type: Metastatic Specimen",
    "SAMPLE_TYPE_Metastasis": "Sample Type: Metastasis",
}
_ZSCORE_EXCLUDE_PREFIXES = ("mut_", "SEX_", "SAMPLE_TYPE_")
for _k, _v in list(_FEATURE_CLEAN_MAP.items()):
    if not any(_k.startswith(pfx) for pfx in _ZSCORE_EXCLUDE_PREFIXES):
        _FEATURE_CLEAN_MAP[f"Z_{_k}"] = f"{_v} (Z-Score)"


# ---------------------------------------------------------------------------
# Statistical & String Formatting Utilities
# ---------------------------------------------------------------------------


def _compute_bh_ranks(sorted_p: np.ndarray, n: int) -> np.ndarray:
    """Computes cumulative minimum adjusted p-values for FDR."""
    adjusted = np.zeros(n, dtype=float)
    cummin_val = 1.0
    for i in range(n - 1, -1, -1):
        cummin_val = min(cummin_val, (sorted_p[i] * n) / (i + 1))
        adjusted[i] = min(1.0, cummin_val)
    return adjusted


def _benjamini_hochberg(
    p_values: np.ndarray, alpha: float = _FDR_ALPHA
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies Benjamini-Hochberg false discovery rate adjustment."""
    p_vals = np.asarray(p_values)
    n = len(p_vals)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    idx = np.argsort(p_vals)
    adj = _compute_bh_ranks(p_vals[idx], n)
    p_adj = np.zeros(n, dtype=float)
    p_adj[idx] = adj
    return p_adj < alpha, p_adj


def _format_p_value(p_val: float) -> str:
    """Formats p-value into clean LaTeX scientific notation or decimal string."""
    if pd.isna(p_val):
        return "N/A"
    if p_val < 1e-3:
        exp = int(np.floor(np.log10(p_val)))
        return f"{p_val / (10 ** exp):.2f} \\times 10^{{{exp}}}"
    return f"{p_val:.3f}"


def _format_feature_words(name: str) -> str:
    """Capitalises feature name words respecting domain abbreviations."""
    abbrevs = {"ICD-10", "ICD-O-3", "AJCC", "DNA", "RNA", "TMB", "MSI", "ICI"}
    res = []
    for w in name.split():
        if any(char.isdigit() for char in w) or w in abbrevs:
            res.append(w)
        elif ":" in w:
            res.append(":".join([p.capitalize() for p in w.split(":")]))
        else:
            res.append(w.capitalize())
    return " ".join(res)


def _format_feature_name(name: str) -> str:
    """Formats feature column names into clean display titles."""
    if name in _FEATURE_CLEAN_MAP:
        return _FEATURE_CLEAN_MAP[name]
    replacements = [
        ("CLINICAL_STAGE_", "Clinical Stage: "),
        ("BIOPSY_SITE_", "Biopsy Site: "),
        ("TISSUE_SUBTYPE_", "Tissue Subtype: "),
        ("PRIOR_ICI_RX_", "Prior ICI Therapy: "),
        ("PRIOR_RX_", "Prior Non-ICI Therapy: "),
        ("SAMPLE_TREATMENT_", "Biopsy Timing: "),
        ("METASTASIZED_", "Metastasised: "),
        ("SAMPLE_TYPE_", "Sample Type: "),
        ("_", " ")
    ]
    for old, new in replacements:
        name = name.replace(old, new)
    return _format_feature_words(name)




# ---------------------------------------------------------------------------
# Tier 1 Data Loading & Preprocessing
# ---------------------------------------------------------------------------


def _attach_tier1_binary_mutations(df_comb: pd.DataFrame, mut_path: Path) -> pd.DataFrame:
    """Attaches binary driver mutation indicators from mutations_cleaned.csv.

    Args:
        df_comb: DataFrame containing cohort clinical and signature features.
        mut_path: Path to the mutations_cleaned.csv file.

    Returns:
        DataFrame updated with binary driver mutation indicator columns.
    """
    if mut_path.exists():
        df_mut = pd.read_csv(mut_path, index_col=0)
        v600_s = df_mut["BRAF_V600"] if "BRAF_V600" in df_mut.columns else pd.Series(dtype=float)
        df_comb["mut_BRAF_V600"] = df_comb.index.map(v600_s).fillna(0.0).astype(float)

        if "BRAF" in df_mut.columns and "BRAF_V600" in df_mut.columns:
            non_v600_s = ((df_mut["BRAF"] == 1) & (df_mut["BRAF_V600"] == 0)).astype(float)
        else:
            non_v600_s = pd.Series(dtype=float)
        df_comb["mut_BRAF_nonV600"] = df_comb.index.map(non_v600_s).fillna(0.0).astype(float)

        for gene, col in [("NRAS", "mut_NRAS"), ("NF1", "mut_NF1")]:
            df_comb[col] = df_comb.index.map(
                df_mut[gene] if gene in df_mut.columns else pd.Series(dtype=float)
            ).fillna(0.0).astype(float)
    else:
        for col in TIER1_BIN_COLS:
            df_comb[col] = 0.0
    return df_comb


def _read_and_combine_tier1_cohort(config: DatasetConfig) -> pd.DataFrame:
    """Reads clinical, signature, and driver mutation data for a single cohort.

    Raises FileNotFoundError if clin_cleaned.csv is missing.
    """
    proc_dir = DATA_DIR / "processed" / config.processed_directory
    clin_path = proc_dir / "clin_cleaned.csv"
    expr_path = proc_dir / "expr_cleaned.csv"
    mut_path = proc_dir / "mutations_cleaned.csv"
    if not clin_path.exists():
        raise FileNotFoundError(f"Missing clinical file at {rel_path(clin_path)}")
    df_clin = pd.read_csv(clin_path, index_col=_COL_SAMPLE_ID)
    df_sig = (
        extract_all_signatures(pd.read_csv(expr_path, index_col=_COL_SAMPLE_ID))
        if expr_path.exists()
        else pd.DataFrame(index=df_clin.index)
    )
    df_comb = df_clin.copy()
    for col in df_sig.columns:
        df_comb[col] = df_sig[col]
    df_comb[_COL_COHORT] = config.cohort_name
    if "AGE_AT_DIAGNOSIS" in df_comb.columns and "AGE" not in df_comb.columns:
        df_comb["AGE"] = df_comb["AGE_AT_DIAGNOSIS"]
    return _attach_tier1_binary_mutations(df_comb, mut_path)


def _impute_and_scale_tier1_cohort(df_comb: pd.DataFrame) -> pd.DataFrame:
    """Imputes continuous features, scales Z-scores, and passes binary features through.

    Args:
        df_comb: Raw combined cohort DataFrame.

    Returns:
        Processed DataFrame with median-imputed and Z-score scaled features.
    """
    for col in TIER1_CONT_COLS:
        if col not in df_comb.columns or df_comb[col].isna().all():
            df_comb[col] = 0.0
    for col in TIER1_BIN_COLS:
        if col not in df_comb.columns:
            df_comb[col] = 0.0
    subset = df_comb[[_COL_COHORT] + TIER1_CONT_COLS + TIER1_BIN_COLS].copy()
    for meta in ["RESPONSE", "RESPONDER", _COL_RESPONSE_BINARY, "SEX", "SAMPLE_TYPE"]:
        if meta in df_comb.columns:
            subset[meta] = df_comb[meta]
    imputed = SimpleImputer(strategy="median").fit_transform(subset[TIER1_CONT_COLS])
    z_feats = StandardScaler().fit_transform(imputed)
    for i, col in enumerate(TIER1_CONT_COLS):
        subset[f"Z_{col}"] = z_feats[:, i]
        subset[col] = imputed[:, i]
    # Binary columns: fill any remaining NaN with 0, no scaling
    for col in TIER1_BIN_COLS:
        subset[col] = subset[col].fillna(0.0).astype(float)
    return subset


def _process_single_cohort_tier1(config: DatasetConfig) -> pd.DataFrame:
    """Loads and standardises Tier 1 features for a single ICI cohort.

    Args:
        config: DatasetConfig specifying cohort directory and metadata.

    Returns:
        Standardised DataFrame for the specified cohort.
    """
    return _impute_and_scale_tier1_cohort(_read_and_combine_tier1_cohort(config))


def _load_tier1_dataset(dataset_configs: Tuple[DatasetConfig, ...]) -> pd.DataFrame:
    """Loads clinical and expression data across all active ICI response cohorts (N=473).

    Args:
        dataset_configs: Tuple of active DatasetConfig instances.

    Returns:
        Concatenated DataFrame filtered strictly for response-labelled patients (N=473).
    """
    ici_configs = [c for c in dataset_configs if getattr(c, "merge_enabled", True)]
    full_df = pd.concat([_process_single_cohort_tier1(c) for c in ici_configs], axis=0)
    full_df = full_df.loc[full_df["RESPONSE_BINARY"].notna()].copy()
    cat_cols = [c for c in ["SEX", "SAMPLE_TYPE"] if c in full_df.columns]
    if cat_cols:
        dummies = pd.get_dummies(full_df[cat_cols], drop_first=False, dtype=float)
        for dcol in dummies.columns:
            full_df[dcol] = dummies[dcol]
    return full_df


# ---------------------------------------------------------------------------
# Logistic Regression Core Routines
# ---------------------------------------------------------------------------


def _compute_logit_stats(col_name: str, model: object) -> Dict[str, float]:
    """Extracts OR and CI from fitted Logit model."""
    coef = model.params.get(col_name, np.nan)
    se = model.bse.get(col_name, np.nan)
    p = model.pvalues.get(col_name, np.nan)
    if any(pd.isna([coef, se, p])) or se > _MAX_SE_THRESHOLD or abs(coef) > _MAX_COEF_THRESHOLD:
        return {}
    or_val = np.exp(coef)
    or_low = np.exp(coef - _CI_Z_SCORE * se)
    or_high = np.exp(coef + _CI_Z_SCORE * se)
    if not (np.isfinite(or_val) and np.isfinite(or_low) and np.isfinite(or_high)):
        return {}
    return {
        "Feature": col_name,
        "Odds Ratio (OR)": float(or_val),
        "OR lower 95%": float(or_low),
        "OR upper 95%": float(or_high),
        "p-value": float(p),
        "coef": float(coef),
        "se": float(se)
    }


def _fit_single_logit(X_col: pd.Series, y: pd.Series) -> Dict[str, float]:
    """Fits univariate Logit model for a single column with warning suppression."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            X_const = sm.add_constant(X_col.fillna(0.0), has_constant="add")
            model = sm.Logit(y, X_const).fit(disp=0, maxiter=_LOGIT_MAX_ITER)
            return _compute_logit_stats(X_col.name, model)
    except (np.linalg.LinAlgError, ValueError, RuntimeError):
        return {}


def _run_univariate_logistic(
    df_valid: pd.DataFrame, feature_cols: List[str], response_col: str = "RESPONSE_BINARY"
) -> pd.DataFrame:
    """Fits univariate Logistic Regression per feature and returns OR table."""
    y = df_valid[response_col].astype(int)
    results = [r for c in feature_cols if (r := _fit_single_logit(df_valid[c], y))]
    df_or = pd.DataFrame(results)
    if df_or.empty:
        return df_or
    rej, p_adj = _benjamini_hochberg(df_or["p-value"].values, alpha=_FDR_ALPHA)
    df_or["FDR_adj_p"], df_or["Significant_FDR"] = p_adj, rej
    return df_or.sort_values(by="p-value", ascending=True).reset_index(drop=True)


def _get_rf_importance(model: object) -> np.ndarray:
    """Extracts feature importances from Random Forest or calibrated wrapper."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "estimator") and hasattr(model.estimator, "feature_importances_"):
        return model.estimator.feature_importances_
    if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
        base = model.calibrated_classifiers_[0].estimator
        if hasattr(base, "feature_importances_"):
            return base.feature_importances_
    raise AttributeError("Model does not expose feature_importances_")


def _plot_rf_importance_bar(df_rf: pd.DataFrame, title: str, out_path: Path) -> None:
    """Renders horizontal bar plot for Random Forest feature importances."""
    fig, ax = plt.subplots(figsize=(10, max(6.5, len(df_rf) * 0.4)))
    palette = sns.color_palette("Blues_r", n_colors=len(df_rf))
    sns.barplot(
        data=df_rf, y=_COL_FORMATTED_FEATURE, x=_COL_IMPORTANCE,
        hue=_COL_FORMATTED_FEATURE, palette=palette, ax=ax, edgecolor="black",
        linewidth=0.5, legend=False
    )
    ax.set_title(title, fontsize=_FONT_TITLE_SUB, fontweight="bold", pad=15)
    ax.set_xlabel(
        "Mean Decrease in Impurity (Gini Importance)",
        fontsize=_FONT_LABEL, fontweight="bold"
    )
    ax.set_ylabel("")

    for p in ax.patches:
        w = p.get_width()
        xy = (w, p.get_y() + p.get_height() / 2.0)
        ax.annotate(
            f"{w:.4f}", xy, ha="left", va="center", xytext=(5, 0),
            textcoords="offset points", fontsize=_FONT_ANNOTATION
        )

    ax.set_xlim(0, df_rf[_COL_IMPORTANCE].max() * 1.15)
    save_fig(fig, out_path)


def _evaluate_tier1_rf_response(df: pd.DataFrame, plots_dir: Path) -> pd.DataFrame:
    """Trains Random Forest Classifier predicting response via get_model()."""
    df_v = df.loc[df["RESPONSE_BINARY"].notna()].copy()
    sex_cols = [e for e in ["SEX_Male", "SEX_Female"] if e in df_v.columns]
    z_feats = [f"Z_{c}" for c in TIER1_CONT_COLS]
    bin_feats = [c for c in TIER1_BIN_COLS if c in df_v.columns]
    feats = z_feats + sex_cols + bin_feats
    X_rf, y_rf = df_v[feats].fillna(0.0), df_v["RESPONSE_BINARY"].astype(int)

    rf_model = get_model("rf", X_rf, y_rf, calibrate=False)
    df_rf = pd.DataFrame({"Feature": feats, "Importance": _get_rf_importance(rf_model)})
    df_rf["Formatted_Feature"] = df_rf["Feature"].apply(_format_feature_name)
    df_rf = df_rf.sort_values(by="Importance", ascending=False).reset_index(drop=True)

    n_samp, n_resp = len(df_v), int(y_rf.sum())
    title = (
        f"Tier 1 RF Importance: Anti-PD-1 Response "
        f"(ICI Cohorts, N={n_samp}, Responders={n_resp})"
    )
    out_file = plots_dir / "tier1_rf_response_importance.png"
    _plot_rf_importance_bar(df_rf, title, out_file)
    print(f"Saved Tier 1 RF importance plot to {rel_path(out_file)}")
    return df_rf


def _evaluate_tier1_logistic(df: pd.DataFrame, plots_dir: Path) -> pd.DataFrame:
    """Fits univariate Logistic Regression per feature across Tier 1 ICI dataset."""
    df_v = df.loc[df["RESPONSE_BINARY"].notna()].copy()
    sex_cols = [e] if (e := "SEX_Male") in df_v.columns else []
    z_feats = [f"Z_{c}" for c in TIER1_CONT_COLS]
    bin_feats = [c for c in TIER1_BIN_COLS if c in df_v.columns]
    feats = z_feats + sex_cols + bin_feats
    df_or = _run_univariate_logistic(df_v, feats)
    if df_or.empty:
        print("WARNING: No valid univariate logistic results for Tier 1.")
        return df_or

    out_path = plots_dir / "tier1_univariate_or_forest.png"
    title = (
        f"Tier 1 Univariate Logistic Regression: "
        f"Anti-PD-1 Response (ICI Cohorts, N={len(df_v)})"
    )
    xlabel = "Odds Ratio (OR, Log Scale — 95% CI per +1 SD)"
    _plot_or_forest(df_or, title, xlabel, out_path, {"n_samples": len(df_v)})
    print(f"Saved Tier 1 univariate OR forest plot to {rel_path(out_path)}")
    return df_or


def _extract_multi_logit_results(model: object, feature_cols: List[str]) -> pd.DataFrame:
    """Extracts odds ratio statistics from fitted multivariate Logit model."""
    results = [r for c in feature_cols if (r := _compute_logit_stats(c, model))]
    df_multi = pd.DataFrame(results)
    if df_multi.empty:
        return df_multi
    rej, p_adj = _benjamini_hochberg(df_multi["p-value"].values, alpha=_FDR_ALPHA)
    df_multi["FDR_adj_p"], df_multi["Significant_FDR"] = p_adj, rej
    return df_multi.sort_values(by="p-value", ascending=True).reset_index(drop=True)


def _prepare_tier1_multivariate_design(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepares design matrix X and target y for Tier 1 multivariate Logit."""
    df_v = df.loc[df["RESPONSE_BINARY"].notna()].copy()
    sex_cols = [e for e in ["SEX_Male", "SEX_Female"] if e in df_v.columns]
    z_feats = [f"Z_{c}" for c in TIER1_CONT_COLS]
    bin_feats = [c for c in TIER1_BIN_COLS if c in df_v.columns]
    feats = z_feats + sex_cols + bin_feats
    model_df = df_v[["RESPONSE_BINARY"] + feats].dropna().copy()
    X = sm.add_constant(model_df[feats], has_constant="add")
    y = model_df["RESPONSE_BINARY"].astype(int)
    return X, y, feats


def _evaluate_tier1_multivariate_logistic(
    df: pd.DataFrame, plots_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Fits multivariate Logistic Regression across harmonised Tier 1 immune features."""
    X, y, feats = _prepare_tier1_multivariate_design(df)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = sm.Logit(y, X).fit(disp=0, maxiter=_LOGIT_MAX_ITER)
    except (np.linalg.LinAlgError, ValueError, RuntimeError) as exc:
        raise RuntimeError(f"Tier 1 multivariate logit failed: {exc}") from exc

    df_multi = _extract_multi_logit_results(model, feats)
    metrics = {
        "auc": float(roc_auc_score(y, model.predict(X))),
        "pseudo_r2": float(model.prsquared),
        "n_samples": len(y)
    }
    title = f"Tier 1 Multivariate Logistic Regression: Anti-PD-1 Response (N={len(y)})"
    xlabel = "Adjusted Odds Ratio (aOR, Log Scale — 95% CI per +1 SD)"
    out_path = plots_dir / "tier1_multivariate_or_forest.png"
    _plot_or_forest(df_multi, title, xlabel, out_path, metrics)
    return df_multi, metrics


# ---------------------------------------------------------------------------
# Tier 2 Data Loading & Evaluation
# ---------------------------------------------------------------------------


def _process_single_cohort_tier2(config: DatasetConfig) -> pd.DataFrame:
    """Loads candidate Tier 2 clinical columns for a single cohort."""
    proc_dir = DATA_DIR / "processed" / config.processed_directory
    clin_path = proc_dir / "clin_cleaned.csv"
    if not clin_path.exists():
        raise FileNotFoundError(f"Missing clinical file at {rel_path(clin_path)}")
    df_clin = pd.read_csv(clin_path, index_col="SAMPLE_ID")
    df_cohort = pd.DataFrame(index=df_clin.index)
    df_cohort["COHORT"] = config.cohort_name
    if "RESPONSE_BINARY" in df_clin.columns:
        df_cohort["RESPONSE_BINARY"] = df_clin["RESPONSE_BINARY"]
    else:
        df_cohort["RESPONSE_BINARY"] = np.nan
    for col in TIER2_CANDIDATE_COLS:
        if col in df_clin.columns:
            df_cohort[col] = df_clin[col]
    return df_cohort


def _encode_tier2_features(df_full: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Encodes dummy features and removes zero-variance columns for Tier 2."""
    present = [c for c in TIER2_CANDIDATE_COLS if c in df_full.columns]
    cat_cols = df_full[present].select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df_full[present].select_dtypes(include=np.number).columns.tolist()
    X_enc = pd.get_dummies(df_full[present], columns=cat_cols, drop_first=False, dtype=float)

    for col in num_cols:
        if col in X_enc.columns:
            if X_enc[col].isnull().all():
                X_enc[col] = 0.0
            else:
                X_enc[col] = SimpleImputer(strategy="median").fit_transform(X_enc[[col]]).ravel()

    zero_var = [c for c in X_enc.columns if X_enc[c].nunique() <= 1]
    if zero_var:
        X_enc = X_enc.drop(columns=zero_var)
    return X_enc, df_full["RESPONSE_BINARY"].copy()


def _load_ici_tier2_features(
    dataset_configs: Tuple[DatasetConfig, ...]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Loads ICI cohort clinical data and encodes features for Tier 2 (N=473)."""
    ici_configs = [c for c in dataset_configs if getattr(c, "merge_enabled", True)]
    df_full = pd.concat([_process_single_cohort_tier2(c) for c in ici_configs], axis=0)
    df_full = df_full.loc[df_full["RESPONSE_BINARY"].notna()].copy()
    X_enc, y_resp = _encode_tier2_features(df_full)
    return df_full, X_enc, y_resp


def _evaluate_ici_tier2_rf(
    X_encoded: pd.DataFrame, y_response: pd.Series, plots_dir: Path
) -> pd.DataFrame:
    """Trains Random Forest Classifier on Tier 2 dummy features via get_model()."""
    valid = y_response.notna()
    X_rf, y_rf = X_encoded.loc[valid].copy(), y_response.loc[valid].astype(int)
    rf_model = get_model("rf", X_rf, y_rf, calibrate=False)

    df_rf = pd.DataFrame({"Feature": X_encoded.columns, "Importance": _get_rf_importance(rf_model)})
    df_rf["Formatted_Feature"] = df_rf["Feature"].apply(_format_feature_name)
    df_rf = df_rf.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    df_top20 = df_rf.head(20).reset_index(drop=True)

    n_samp, n_resp = len(X_rf), int(y_rf.sum())
    title = (
        f"Tier 2 RF Importance: Granular ICI Clinical Predictors "
        f"(N={n_samp}, Responders={n_resp})"
    )
    out_file = plots_dir / "ici_tier2_rf_importance.png"
    _plot_rf_importance_bar(df_top20, title, out_file)
    print(f"Saved Tier 2 ICI RF importance plot to {rel_path(out_file)}")
    return df_rf


def _evaluate_ici_tier2_logistic(
    X_encoded: pd.DataFrame, y_response: pd.Series, plots_dir: Path
) -> pd.DataFrame:
    """Fits univariate Logistic Regression per feature across Tier 2 dummy features."""
    valid = y_response.notna()
    df_with_resp = X_encoded.loc[valid].copy()
    df_with_resp["RESPONSE_BINARY"] = y_response.loc[valid].values

    df_or = _run_univariate_logistic(df_with_resp, X_encoded.columns.tolist())
    if df_or.empty:
        print("WARNING: No valid univariate logistic results for Tier 2.")
        return df_or

    df_top20 = df_or.head(20).copy()
    out_path = plots_dir / "ici_tier2_univariate_or_forest.png"
    title = f"Tier 2 Univariate Logistic Regression: Top ICI Clinical Predictors (N={valid.sum()})"
    xlabel = "Odds Ratio (OR, Log Scale — 95% CI)"
    _plot_or_forest(df_top20, title, xlabel, out_path, {"n_samples": valid.sum()})
    print(f"Saved Tier 2 ICI univariate OR forest plot to {rel_path(out_path)}")
    return df_or


def _select_rank_full_cols(X_valid: pd.DataFrame, top_features: List[str]) -> List[str]:
    """Greedily selects top features while ensuring design matrix is full rank."""
    selected: List[str] = []
    for col in top_features:
        if len(selected) >= _MAX_MULTI_FEATURES:
            break
        if col in X_valid.columns and X_valid[col].nunique() > 1:
            cand = sm.add_constant(X_valid[selected + [col]], has_constant="add")
            if np.linalg.matrix_rank(cand) == cand.shape[1]:
                selected.append(col)
    return selected


def _fit_tier2_multivariate_model(X: pd.DataFrame, y: pd.Series) -> object:
    """Fits unregularized or regularized Logit model with warning suppression."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        try:
            return sm.Logit(y, X).fit(disp=0, maxiter=_LOGIT_MAX_ITER)
        except (np.linalg.LinAlgError, ValueError, RuntimeError) as exc:
            print(f"Warning: Logit fit failed ({exc}); trying regularized fit.")
            try:
                return sm.Logit(y, X).fit_regularized(
                    alpha=0.01, disp=0, maxiter=_LOGIT_MAX_ITER
                )
            except (np.linalg.LinAlgError, ValueError, RuntimeError) as exc2:
                print(f"Warning: Regularized fit also failed ({exc2}).")
                return None


def _prepare_tier2_multivariate_data(
    X_encoded: pd.DataFrame, y_response: pd.Series, tier2_univariate_or: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepares rank-full multivariate data for Tier 2."""
    valid = y_response.notna()
    X_valid = X_encoded.loc[valid].copy()
    y_valid = y_response.loc[valid].astype(int)
    selected = _select_rank_full_cols(X_valid, tier2_univariate_or["Feature"].tolist())
    if not selected:
        return pd.DataFrame(), pd.Series(), []
    model_df = X_valid[selected].copy()
    model_df["RESPONSE_BINARY"] = y_valid.values
    model_df = model_df.dropna()
    return model_df[selected], model_df["RESPONSE_BINARY"].astype(int), selected


def _evaluate_ici_tier2_multivariate_logistic(
    X_encoded: pd.DataFrame, y_response: pd.Series,
    tier2_univariate_or: pd.DataFrame, plots_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Fits multivariate Logistic Regression across top ICI dummy features."""
    X_sub, y, selected = _prepare_tier2_multivariate_data(
        X_encoded, y_response, tier2_univariate_or
    )
    if not selected:
        return pd.DataFrame(), {"auc": 0.5, "pseudo_r2": 0.0, "n_samples": len(X_encoded)}

    X = sm.add_constant(X_sub, has_constant="add")
    model = _fit_tier2_multivariate_model(X, y)
    if model is None:
        return pd.DataFrame(), {"auc": 0.5, "pseudo_r2": 0.0, "n_samples": len(X_sub)}

    df_multi = _extract_multi_logit_results(model, selected)
    metrics = {
        "auc": float(roc_auc_score(y, model.predict(X))),
        "pseudo_r2": float(getattr(model, "prsquared", 0.0)),
        "n_samples": len(X_sub)
    }
    title = f"Tier 2 Multivariate Logistic Regression: Top ICI Clinical Predictors (N={len(X_sub)})"
    xlabel = "Adjusted Odds Ratio (aOR, Log Scale — 95% CI)"
    out_path = plots_dir / "ici_tier2_multivariate_or_forest.png"
    _plot_or_forest(df_multi, title, xlabel, out_path, metrics)
    return df_multi, metrics


# ---------------------------------------------------------------------------
# Tier 3: Unified Multimodal Feature Selection Pipeline
# ---------------------------------------------------------------------------


def _build_tier3_unified_matrix(
    t1_df: pd.DataFrame, t2_enc: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """Combines Tier 1 signatures and Tier 2 clinical features into X_unified."""
    z_cols = [f"Z_{c}" for c in TIER1_CONT_COLS if f"Z_{c}" in t1_df.columns]
    b_cols = [c for c in TIER1_BIN_COLS + ["SEX_Male", "SEX_Female"] if c in t1_df.columns]
    t1_sub = t1_df[z_cols + b_cols].copy()

    common_idx = t1_sub.index.intersection(t2_enc.index)
    X_unified = pd.concat([t1_sub.loc[common_idx], t2_enc.loc[common_idx]], axis=1)

    # Drop duplicate columns if any
    X_unified = X_unified.loc[:, ~X_unified.columns.duplicated()]
    y_resp = t1_df.loc[common_idx, "RESPONSE_BINARY"].copy()

    valid = y_resp.notna()
    return X_unified.loc[valid].copy(), y_resp.loc[valid].astype(int)


def _evaluate_tier3_rf_response(
    X_rf: pd.DataFrame, y_rf: pd.Series, plots_dir: Path
) -> pd.DataFrame:
    """Trains Random Forest on unified multimodal feature matrix X_unified."""
    rf_model = get_model("rf", X_rf, y_rf, calibrate=False)
    df_rf = pd.DataFrame({"Feature": X_rf.columns, "Importance": _get_rf_importance(rf_model)})
    df_rf["Formatted_Feature"] = df_rf["Feature"].apply(_format_feature_name)
    df_rf = df_rf.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    df_top25 = df_rf.head(25).reset_index(drop=True)

    n_samp, n_resp = len(X_rf), int(y_rf.sum())
    title = (
        f"Tier 3 Unified Multimodal RF Importance: Molecular & Clinical "
        f"(N={n_samp}, Responders={n_resp})"
    )
    out_file = plots_dir / "unified_tier3_rf_importance.png"
    _plot_rf_importance_bar(df_top25, title, out_file)
    print(f"Saved Tier 3 Unified RF importance plot to {rel_path(out_file)}")
    return df_rf


def _evaluate_tier3_multivariate_logistic(
    X_unified: pd.DataFrame, y_unified: pd.Series,
    t3_rf: pd.DataFrame, plots_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Fits multivariate Logistic Regression on top Tier 3 unified features."""
    top_feats = t3_rf.head(15)["Feature"].tolist()
    selected = _select_rank_full_cols(X_unified, top_feats)
    if not selected:
        return pd.DataFrame(), {"auc": 0.5, "pseudo_r2": 0.0, "n_samples": len(X_unified)}

    X = sm.add_constant(X_unified[selected], has_constant="add")
    model = _fit_tier2_multivariate_model(X, y_unified)
    if model is None:
        return pd.DataFrame(), {"auc": 0.5, "pseudo_r2": 0.0, "n_samples": len(X_unified)}

    df_multi = _extract_multi_logit_results(model, selected)
    metrics = {
        "auc": float(roc_auc_score(y_unified, model.predict(X))),
        "pseudo_r2": float(getattr(model, "prsquared", 0.0)),
        "n_samples": len(X_unified)
    }
    title = f"Tier 3 Unified Multivariate Logit: Anti-PD-1 Response (N={len(X_unified)})"
    xlabel = "Adjusted Odds Ratio (aOR, Log Scale — 95% CI)"
    out_path = plots_dir / "unified_tier3_multivariate_or_forest.png"
    _plot_or_forest(df_multi, title, xlabel, out_path, metrics)
    return df_multi, metrics


def _run_tier3_pipeline(
    t1_df: pd.DataFrame, t2_enc: pd.DataFrame, plots_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Executes Tier 3 Unified Multimodal Feature Selection pipeline."""
    print("\n--- Tier 3: Unified Multimodal Feature Selection Leaderboard ---")
    X_u, y_u = _build_tier3_unified_matrix(t1_df, t2_enc)
    print(f"Loaded Unified Multimodal matrix: {len(X_u)} patients; {len(X_u.columns)} features.")

    t3_rf = _evaluate_tier3_rf_response(X_u, y_u, plots_dir)
    t3_multi_or, t3_multi_met = _evaluate_tier3_multivariate_logistic(
        X_u, y_u, t3_rf, plots_dir
    )
    return t3_rf, t3_multi_or, t3_multi_met


# ---------------------------------------------------------------------------
# Shared Forest Plotting Routines
# ---------------------------------------------------------------------------


def _setup_forest_axis(ax: plt.Axes, df_plot: pd.DataFrame, title: str, xlabel: str) -> None:
    """Configures common axis ticks, gridlines, and formatting for forest plots."""
    ax.set_yticks(np.arange(len(df_plot)))
    ax.set_yticklabels(df_plot[_COL_FORMATTED_FEATURE], fontsize=_FONT_TICK, fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.grid(True, linestyle="--", color=_COLOR_GRID, linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(True, linestyle=":", color=_COLOR_GRID, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=_FONT_TITLE_MAIN, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=_FONT_LABEL, fontweight="bold")


def _draw_or_forest_points(ax: plt.Axes, df_plot: pd.DataFrame) -> None:
    """Plots confidence interval lines and scatter points for forest plot."""
    y_pos = np.arange(len(df_plot))
    ors = df_plot[_COL_ODDS_RATIO].values
    lowers = df_plot[_COL_OR_LOWER_95].values
    uppers = df_plot[_COL_OR_UPPER_95].values
    colors = [RESPONSE_PALETTE["CR/PR"] if o >= 1.0 else RESPONSE_PALETTE["PD"] for o in ors]

    ax.axvline(x=1.0, color=_COLOR_NULL_LINE, linestyle="--", linewidth=_LINEWIDTH_NULL, alpha=0.7)
    for i in range(len(df_plot)):
        ax.plot(
            [lowers[i], uppers[i]], [y_pos[i], y_pos[i]],
            color=colors[i], linewidth=_LINEWIDTH_FOREST, alpha=0.85
        )
        ax.scatter(
            ors[i], y_pos[i], color=colors[i], s=_POINT_SIZE_FOREST, zorder=5,
            edgecolor="black", linewidth=0.7
        )


def _annotate_or_forest_bars(ax: plt.Axes, df_plot: pd.DataFrame, label_prefix: str) -> None:
    """Annotates OR and p-values beside forest plot markers."""
    y_pos = np.arange(len(df_plot))
    ors = df_plot[_COL_ODDS_RATIO].values
    uppers = df_plot[_COL_OR_UPPER_95].values
    p_vals = df_plot[_COL_P_VALUE].values
    fdrs = df_plot[_COL_FDR_ADJ_P].values

    for i in range(len(df_plot)):
        fdr = fdrs[i]
        txt = f"{label_prefix}={ors[i]:.2f} (p={p_vals[i]:.1e}, FDR={fdr:.1e})"
        is_sig = fdr < _FDR_ALPHA
        ax.annotate(
            txt, (uppers[i], y_pos[i]), xytext=(8, -3), textcoords="offset points",
            fontsize=_FONT_ANNOTATION, fontweight="bold" if is_sig else "normal",
            color=_COLOR_TEXT_SIG if is_sig else _COLOR_TEXT_INSIG
        )


def _set_forest_legend(ax: plt.Axes, prefix: str, metrics: Dict[str, float]) -> None:
    """Sets up forest plot legend box."""
    auc_str = f"AUROC = {metrics['auc']:.3f}" if "auc" in metrics else ""
    r2_str = f"McFadden R² = {metrics['pseudo_r2']:.3f}" if "pseudo_r2" in metrics else ""
    lbl = " | ".join([p for p in [auc_str, r2_str] if p])
    handles = [
        mpatches.Patch(
            color=RESPONSE_PALETTE["CR/PR"],
            label=f"{prefix} ≥ 1.0 (response-favourable)"
        ),
        mpatches.Patch(
            color=RESPONSE_PALETTE["PD"],
            label=f"{prefix} < 1.0 (response-unfavourable)"
        ),
        mlines.Line2D(
            [], [], color=_COLOR_NULL_LINE, linestyle="--", linewidth=_LINEWIDTH_NULL,
            label="Null Effect (OR = 1.0)"
        )
    ]
    if lbl:
        handles.append(mlines.Line2D([], [], color="none", label=lbl))
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.01, 0.0),
        frameon=True, facecolor="white", edgecolor=_COLOR_LEGEND_EDGE,
        fontsize=_FONT_LEGEND, borderaxespad=0.0,
    )


def _plot_or_forest(
    df_or: pd.DataFrame, title: str, xlabel: str, out_path: Path, metrics: Dict[str, float]
) -> None:
    """Renders a presentation-ready Odds Ratio Forest Plot."""
    df_plot = df_or.copy()
    df_plot["Formatted_Feature"] = df_plot["Feature"].apply(_format_feature_name)
    df_plot = df_plot.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(6.5, len(df_plot) * 0.55)))
    _draw_or_forest_points(ax, df_plot)
    _setup_forest_axis(ax, df_plot, title, xlabel)

    fin_low = [l for l in df_plot["OR lower 95%"].values if np.isfinite(l) and l > 0]
    fin_high = [u for u in df_plot["OR upper 95%"].values if np.isfinite(u) and u > 0]
    min_x = max(0.05, (min(fin_low) if fin_low else 0.1) * 0.8)
    max_x = min(100.0, (max(fin_high) if fin_high else 2.0) * 1.6)
    ax.set_xlim(min_x, max_x)
    prefix = "aOR" if "Multivariate" in title else "OR"
    _annotate_or_forest_bars(ax, df_plot, prefix)
    _set_forest_legend(ax, prefix, metrics)
    sns.despine(top=True, right=True)
    save_fig(fig, out_path)


def _plot_point_ci(
    ax: plt.Axes, y: float, estimate: float, low: float, high: float,
    model_type: str, marker: str
) -> None:
    """Plots confidence interval line and point estimate marker."""
    c = MODEL_TYPE_PALETTE[model_type]
    ax.plot([low, high], [y, y], color=c, linewidth=_LINEWIDTH_FOREST, alpha=0.85)
    ax.scatter(
        estimate, y, color=c, marker=marker, s=_POINT_SIZE_COMP, zorder=5,
        edgecolor="black", linewidth=0.6
    )


def _draw_single_comparison_row(ax: plt.Axes, row: pd.Series, y_pos: float) -> None:
    """Draws univariate vs multivariate errorbars for a single feature row."""
    yu, ym = y_pos + 0.15, y_pos - 0.15
    or_col, low_col = _COL_ODDS_RATIO, _COL_OR_LOWER_95
    high_col, p_col = _COL_OR_UPPER_95, _COL_P_VALUE
    _plot_point_ci(
        ax, yu, row[f"{or_col}_uni"], row[f"{low_col}_uni"],
        row[f"{high_col}_uni"], "Univariate", "o"
    )
    _plot_point_ci(
        ax, ym, row[f"{or_col}_multi"], row[f"{low_col}_multi"],
        row[f"{high_col}_multi"], "Multivariate", "s"
    )
    mr = max(row[f"{high_col}_uni"], row[f"{high_col}_multi"])
    u_p, m_p = row[f"{p_col}_uni"], row[f"{p_col}_multi"]
    u_txt = f"Uni: {row[f'{or_col}_uni']:.2f} (p={u_p:.1e})"
    m_txt = f"Multi: {row[f'{or_col}_multi']:.2f} (p={m_p:.1e})"
    ax.annotate(
        u_txt, (mr, yu), xytext=(8, -3), textcoords="offset points", fontsize=_FONT_ANNOTATION_SM,
        color=MODEL_TYPE_PALETTE["Univariate"],
        fontweight="bold" if u_p < _FDR_ALPHA else "normal"
    )
    ax.annotate(
        m_txt, (mr, ym), xytext=(8, -3), textcoords="offset points", fontsize=_FONT_ANNOTATION_SM,
        color=MODEL_TYPE_PALETTE["Multivariate"],
        fontweight="bold" if m_p < _FDR_ALPHA else "normal"
    )


def _draw_comparison_rows(ax: plt.Axes, merged: pd.DataFrame) -> None:
    """Draws univariate vs multivariate errorbars for comparison plot."""
    y_idx = np.arange(len(merged))
    for i, row in merged.iterrows():
        _draw_single_comparison_row(ax, row, y_idx[i])


def _configure_comparison_xlim(ax: plt.Axes, merged: pd.DataFrame) -> None:
    """Sets log-scale x-axis limits for comparison forest plot."""
    all_low = [
        l for l in (
            list(merged["OR lower 95%_uni"].values)
            + list(merged["OR lower 95%_multi"].values)
        )
        if np.isfinite(l) and l > 0
    ]
    all_high = [
        u for u in (
            list(merged["OR upper 95%_uni"].values)
            + list(merged["OR upper 95%_multi"].values)
        )
        if np.isfinite(u) and u > 0
    ]
    min_x = max(0.05, (min(all_low) if all_low else 0.1) * 0.8)
    max_x = min(100.0, (max(all_high) if all_high else 2.0) * 1.8)
    ax.set_xlim(min_x, max_x)


def _add_comparison_legend(ax: plt.Axes) -> None:
    """Adds legend for comparison forest plot."""
    handles = [
        mlines.Line2D(
            [], [], color=MODEL_TYPE_PALETTE["Univariate"], marker="o",
            linestyle="-", linewidth=2.0, markersize=8, label="Univariate OR (95% CI)"
        ),
        mlines.Line2D(
            [], [], color=MODEL_TYPE_PALETTE["Multivariate"], marker="s",
            linestyle="-", linewidth=2.0, markersize=8, label="Multivariate Adjusted aOR (95% CI)"
        ),
    ]
    ax.legend(
        handles=handles, loc="upper right", frameon=True, facecolor="white",
        edgecolor=_COLOR_LEGEND_EDGE, fontsize=9.0
    )


def _apply_canonical_feature_order(
    merged: pd.DataFrame, canonical_order: List[str]
) -> pd.DataFrame:
    """Reindexes merged OR DataFrame by canonical feature order.

    Features present in canonical_order appear first in the specified sequence;
    any remaining features (not in the order list) are appended at the end.
    """
    present_ordered = [f for f in canonical_order if f in merged["Feature"].values]
    remainder = [f for f in merged["Feature"].values if f not in canonical_order]
    final_order = present_ordered + remainder
    order_map = {feat: idx for idx, feat in enumerate(final_order)}
    merged = merged.copy()
    merged["_sort_key"] = merged["Feature"].map(order_map)
    return merged.sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)


def _plot_univariate_vs_multivariate_comparison(
    df_uni: pd.DataFrame, df_multi: pd.DataFrame, title: str, xlabel: str, out_path: Path,
    canonical_order: Optional[List[str]] = None,
) -> None:
    """Renders a paired side-by-side forest plot.

    If canonical_order is provided, features are sorted by that order before
    the y-axis reversal so the plot reads top-to-bottom in canonical sequence.
    """
    merged = pd.merge(
        df_uni, df_multi, on="Feature", suffixes=("_uni", "_multi")
    )
    if canonical_order is not None:
        merged = _apply_canonical_feature_order(merged, canonical_order)
    merged = merged.iloc[::-1].reset_index(drop=True)
    merged["Formatted_Feature"] = merged["Feature"].apply(_format_feature_name)

    fig, ax = plt.subplots(figsize=(13, max(7.0, len(merged) * 0.65)))
    ax.axvline(x=1.0, color=_COLOR_NULL_LINE, linestyle="--", linewidth=1.2, alpha=0.7)
    _draw_comparison_rows(ax, merged)
    _setup_forest_axis(ax, merged, title, xlabel)
    _configure_comparison_xlim(ax, merged)
    _add_comparison_legend(ax)
    sns.despine(top=True, right=True)
    save_fig(fig, out_path)
    print(f"Saved Univariate vs Multivariate comparison plot to {rel_path(out_path)}")


def _format_ici_cohort_names(
    dataset_configs: Tuple[DatasetConfig, ...]
) -> Tuple[str, str, int]:
    """Generates dynamic string representations of active ICI trial cohorts."""
    names = [c.cohort_name for c in dataset_configs if getattr(c, "merge_enabled", True)]
    n = len(names)
    if n == 0:
        return "N/A", "N/A", 0
    if n == 1:
        return names[0], f"**{names[0]}**", 1
    if n == 2:
        return f"{names[0]} and {names[1]}", f"**{names[0]}** and **{names[1]}**", 2
    plain = f"{', '.join(names[:-1])}, and {names[-1]}"
    formatted = f"{', '.join([f'**{x}**' for x in names[:-1]])}, and **{names[-1]}**"
    return plain, formatted, n


def _build_report_intro_paragraph(ici_count: int, ici_plain_str: str) -> str:
    """Builds introductory paragraph for feature selection report."""
    return (
        "This report presents a **two-tiered feature selection architecture** evaluating "
        "clinical and transcriptomic predictors of **immunotherapy binary response (RECIST)** across "
        f"the {ici_count} active immunotherapy datasets ({ici_plain_str}). "
        "The dataset pool utilizes the updated NCI Genomic Data Commons TCGA-SKCM cohort "
        "(`skcm_tcga_gdc`), which supersedes the legacy 2018 TCGA Pan-Cancer Atlas dataset due "
        "to updated GDC data harmonization. Feature selection is conducted strictly on "
        "the immunotherapy-treated (IT-treated) subcohort with RECIST response labels "
        "(anti-PD-1, anti-CTLA-4, and vaccine arms where applicable), excluding non-IT-treated "
        "patients such as surgery-only or chemotherapy-only cases."
    )


def _build_tier1_summary_bullet(t1_n: int, ici_fmt_str: str) -> str:
    """Builds Tier 1 summary bullet detailing all 18 features across 5 domains."""
    return (
        f"- **Tier 1 (ICI Multi-Cohort Immune & Genomic Signatures, $N = {t1_n}$)**: "
        "Evaluates 18 features across 5 biological domains — 6 transcriptomic immune "
        "signatures (`IFN_gamma`, `TIS`, `CYT`, `CD8_Tcell`, `IMPRES`, `PD_L1`), 2 myeloid "
        "signatures (`M1_M2_Ratio`, `Macrophage_STV`), 3 genomic burden metrics (`TMB`, "
        "`SNV_NEOANTIGEN`, `INDEL_NEOANTIGEN`), 4 driver mutation subtypes (`mut_BRAF_V600`, "
        "`mut_BRAF_nonV600`, `mut_NRAS`, `mut_NF1`), and 3 host demographics (`AGE`, `SEX_Male`, "
        f"`SEX_Female`) — pooled across {ici_fmt_str}."
    )


def _build_report_frontmatter_header(
    tier1_total_n: int, ici2_n: int, ici2_feat_count: int,
    ici_plain_str: str, ici_fmt_str: str, ici_count: int
) -> List[str]:
    """Generates YAML frontmatter and title header for report."""
    fm = generate_obsidian_frontmatter(
        title="Two-Tiered Clinical & Transcriptomic Feature Selection Report",
        aliases=["Feature Selection Report", "Clinical Feature Selection", "ICI Feature Selection"],
        tags=[
            "melanoma", "clinical-subtyping", "feature-selection", "logistic-regression",
            "random-forest", "two-tiered", "ici-cohorts", "anti-pd1", "binary-response"
        ],
        extra_css_classes=["table-center", "row-alt"],
    )
    intro_text = _build_report_intro_paragraph(ici_count, ici_plain_str)
    t1_bullet = _build_tier1_summary_bullet(tier1_total_n, ici_fmt_str)
    t2_bullet = (
        f"- **Tier 2 (ICI Granular Clinical Features, $N = {ici2_n}$)**: "
        f"Evaluates {ici2_feat_count} granular categorical baseline clinical "
        "covariates available within the ICI trial cohorts."
    )
    return [
        fm, "", "# Two-Tiered Clinical & Transcriptomic Feature Selection Report", "",
        intro_text, "",
        t1_bullet,
        t2_bullet, "",
    ]


def _build_tier1_preamble_what_text(
    t1_n_resp: int, t1_n_nonresp: int, ici_count: int, t1_resp_n: int
) -> str:
    """Builds What text for Tier 1 info callout box."""
    return (
        f"> - **What**: Evaluating 18 features across 5 biological domains (6 transcriptomic "
        "immune signatures, 2 myeloid signatures, 3 genomic burden metrics, 4 driver mutation "
        f"subtypes, and 3 host demographics) against **immunotherapy binary response** "
        f"($N_{{\\text{{resp}}}} = {t1_n_resp}$ / $N_{{\\text{{non-resp}}}} = {t1_n_nonresp}$) "
        f"across all {ici_count} ICI cohorts pooled ($N = {t1_resp_n}$). Two complementary "
        "models are applied: **Random Forest** importance and **Logistic Regression**."
    )


def _build_report_tier1_preamble(
    t1_total_n: int, t1_resp_n: int, t1_n_resp: int, t1_n_nonresp: int,
    t1_n_sig: int, t1_m_sig: int, ici_plain_str: str, ici_count: int
) -> List[str]:
    """Builds Tier 1 preamble lines."""
    s_tot = f"- **Total ICI Sample Size**: {t1_total_n} patients across {ici_count} cohorts "
    s_tot += f"({ici_plain_str})"
    s_eval = f"- **Response Evaluation Cohort**: {t1_resp_n} patients "
    s_eval += f"({t1_n_resp} responders / {t1_n_nonresp} non-responders)"
    w_text = _build_tier1_preamble_what_text(t1_n_resp, t1_n_nonresp, ici_count, t1_resp_n)
    return [
        fr"## 1. Tier 1: ICI Multi-Cohort _Immune_ Feature Selection ($N = {t1_total_n}$)", "",
        "> [!INFO] What, Why & Questions — Tier 1",
        w_text,
        "> - **Why**: These features directly reflect the tumour "
        "immune microenvironment hypothesised to drive anti-PD-1 response.",
        "> - **Questions**:",
        ">   1. *Which transcriptomic immune features are individually associated "
        "with anti-PD-1 response?*",
        ">   2. *After mutual adjustment, which features retain independent "
        "predictive significance?*",
        ">   3. *Do the 6 immune expression signatures exhibit collinearity?*", "",
        s_tot, s_eval,
        f"- **Univariate FDR-Significant Predictors (FDR < 0.05)**: {t1_n_sig} features",
        f"- **Multivariate FDR-Significant Predictors (FDR < 0.05)**: {t1_m_sig} feature(s)", "",
    ]


def _build_rf_mutation_bullet(top: pd.DataFrame, total_imp: float) -> List[str]:
    """Builds driver mutation signal bullet for RF importance callout."""
    mut_in_top = [row for _, row in top.iterrows() if row["Feature"] in TIER1_BIN_COLS]
    if not mut_in_top:
        return []
    mut_label = _format_feature_name(mut_in_top[0]["Feature"])
    mut_pct = (
        mut_in_top[0]["Importance"] / total_imp * 100
        if total_imp > 0 else 0.0
    )
    return [
        f"> - **Driver Mutation Signal**: **`{mut_label}`** "
        f"({mut_pct:.1f}% importance) appears among the top predictors, "
        "suggesting driver mutation subtype modulates ICI response."
    ]


def _build_rf_key_takeaways(df_rf: pd.DataFrame, n_top: int = 5) -> List[str]:
    """Builds a dynamic INSIGHT callout from the top-N RF feature importances."""
    if df_rf.empty:
        return []
    top = df_rf.head(n_top).reset_index(drop=True)
    total_imp = df_rf["Importance"].sum()
    top1_name = _format_feature_name(top.iloc[0]["Feature"])
    top1_pct = (top.iloc[0]["Importance"] / total_imp * 100) if total_imp > 0 else 0.0
    lines: List[str] = [
        "> [!INSIGHT] Key Takeaways: Tier 1 RF Importance",
        f"> - **Top Predictor**: **`{top1_name}`** accounts for "
        f"{top1_pct:.1f}% of total RF Gini importance across all features.",
        f"> - **Top {n_top} Features by Gini Importance**:",
    ]
    for _, row in top.iterrows():
        feat_label = _format_feature_name(row["Feature"])
        pct = (row["Importance"] / total_imp * 100) if total_imp > 0 else 0.0
        lines.append(f">   - `{feat_label}` — {pct:.1f}%")
    lines.extend(_build_rf_mutation_bullet(top, total_imp))
    lines.append("")
    return lines


def _build_tier1_rationale_rows_immune() -> Tuple[str, str]:
    """Builds rationale table rows for immune signatures and myeloid features."""
    c1 = (
        "| **Transcriptomic Immune Signatures** | `IFN_gamma`, `TIS`, `CYT`, "
        "`CD8_Tcell`, `IMPRES`, `PD_L1` | **T-Cell Inflammation & Exhaustion**: "
        "Pre-existing CD8+ T-cell infiltration (`CD8_Tcell`) and cytolytic killing "
        "activity (`CYT`). IFN-$\gamma$ signaling upregulates adaptive PD-L1 (`CD274`). "
        "`TIS` and `IMPRES` capture suppressed adaptive microenvironmental immunity. |"
    )
    c2 = (
        "| **Myeloid & TAM Polarization** | `M1_M2_Ratio`, `Macrophage_STV_Score` | "
        "**Innate Immunosuppression**: Pro-inflammatory M1 (antitumour) vs "
        "immunosuppressive M2 (protumour) macrophage balance. M2 TAMs secrete "
        "IL-10/TGF-$\\beta$, impairing T-cell trafficking. |"
    )
    return c1, c2


def _build_tier1_rationale_rows_genomic() -> Tuple[str, str, str]:
    """Builds rationale table rows for genomic burden, drivers, and demographics."""
    c3 = (
        "| **Genomic Burden & Neoantigens** | `TMB`, `SNV_NEOANTIGEN`, `INDEL_NEOANTIGEN` | "
        "**Somatic Antigenicity**: Somatic mutation density (`TMB`) generates HLA-bound "
        "neoepitopes. Frameshift indels generate novel non-self peptides with high "
        "immunogenicity. |"
    )
    c4 = (
        "| **Genomic Driver Subtypes** | `mut_BRAF_V600`, `mut_BRAF_nonV600`, `mut_NRAS`, "
        "`mut_NF1` | **Oncogenic Secretome & Kinase Activation**: `BRAF V600` (Class 1 "
        "hotspots V600E/K) drives monomeric MAPK hyperactivation and MHC-I downregulation; "
        "`BRAF Non-V600` (Class 2/3) relies on RAS dimerization. `NRAS`/`NF1` mutations "
        "associate with high UV mutational burden. |"
    )
    c5 = (
        "| **Host Baseline Demographics** | `AGE`, `SEX_Male`, `SEX_Female` | "
        "**Host Immunosenescence & Dimorphism**: Age-related decline in naive T-cell "
        "repertoire diversity and sex-specific hormonal immunomodulation. |"
    )
    return c3, c4, c5


def _build_tier1_rationale_table() -> List[str]:
    """Builds markdown table describing Tier 1 feature selection rationale."""
    h1 = (
        "| Feature Category | Features Included | "
        "Primary Biological Rationale & Mechanistic Role |"
    )
    h2 = "| :--- | :--- | :--- |"
    c1, c2 = _build_tier1_rationale_rows_immune()
    c3, c4, c5 = _build_tier1_rationale_rows_genomic()
    return [h1, h2, c1, c2, c3, c4, c5, ""]


def _build_report_tier1_rationale_block() -> List[str]:
    """Builds Tier 1 feature selection biological rationale section."""
    lines = [
        "### 1.1. Biological & Clinical Rationale for Tier 1 Feature Selection", "",
        "Tier 1 features were selected to evaluate five primary biological domains hypothesised ",
        "to drive response or primary resistance to anti-PD-1 immune checkpoint blockade:", "",
    ]
    lines.extend(_build_tier1_rationale_table())
    return lines


def _build_report_tier1_rf_block(t1_resp_n: int, tier1_rf: pd.DataFrame) -> List[str]:
    """Builds Tier 1 RF section header and key takeaways."""
    lines = [
        fr"### 1.2. Random Forest Importance for Anti-PD-1 Response ($N = {t1_resp_n}$)",
        f"Random Forest feature importance trained on {t1_resp_n} ICI patients:", "",
        "![Tier 1 RF Response Importance]"
        "(../../plots/clinical/tier1_rf_response_importance.png)", "",
    ]
    lines.extend(_build_rf_key_takeaways(tier1_rf))
    return lines


def _build_report_tier1_collinearity_text(
    sig_mask: pd.Series, or_min: float, or_max: float, sig_max_p: float
) -> List[str]:
    """Builds transcriptomic collinearity insight bullet."""
    if sig_mask.any() and not pd.isna(or_min):
        p_str = _format_p_value(sig_max_p)
        return [
            fr"> - **Transcriptomic Collinearity**: All 8 transcriptomic/myeloid signatures "
            fr"(IFN-$\gamma$, TIS, CYT, CD8 T-cell, IMPRES, `PD-L1`, M1/M2 Ratio, "
            fr"Macrophage STV) show association with response in univariate logistic regression "
            fr"($\text{{OR}} \approx {or_min:.2f}\text{{--}}{or_max:.2f}$, "
            fr"leading $p \le {p_str}$). In multivariate "
            fr"modelling, individual signatures attenuate towards "
            fr"$\text{{OR}} = 1.0$ due to shared variance."
        ]
    return [
        "> - **Transcriptomic Collinearity**: Immune and myeloid signatures show "
        "univariate association with response; multivariate modelling reveals collinearity."
    ]


def _build_tier1_or_candidate_bullet(
    feat_title: str, top_multi_aor: float, p_top: str, fdr_top: str, fdr_met: bool
) -> str:
    """Builds candidate predictor bullet for Tier 1 OR comparison."""
    if fdr_met:
        return (
            fr"> - **Top Independent Predictor**: **`{feat_title}`** "
            fr"($\text{{aOR}} = {top_multi_aor:.2f}$, $p = {p_top}$, FDR $= {fdr_top}$) "
            "retains statistically significant independent association with response."
        )
    return (
        fr"> - **Top Nominal Candidate**: **`{feat_title}`** demonstrates the strongest "
        fr"unadjusted trend ($\text{{aOR}} = {top_multi_aor:.2f}$, nominal $p = {p_top}$), "
        fr"though it does not clear FDR correction (FDR $= {fdr_top}$). High-dimensional "
        "modelling (Pillar 3/4) is required to aggregate these weak, correlated signals."
    )


def _build_tier1_or_insight_bullets(
    t1_n_sig: int,
    t1_m_sig: int,
    feat_title: str,
    top_multi_aor: float,
    p_top: str,
    fdr_top: str,
    fdr_met: bool,
) -> List[str]:
    """Generates insight bullet lines for Tier 1 Odds Ratio comparison."""
    bullets: List[str] = []
    if t1_n_sig == 0 and t1_m_sig == 0:
        bullets.append(
            "> - **Statistical Significance & Power Limitations**: Neither univariate "
            "nor multivariate logistic models yield features passing FDR correction "
            fr"($q < {_FDR_ALPHA}$). This is driven by modest sample size ($N = 195$), "
            "multiple testing burden across 18 candidate features, and substantial "
            "multicollinearity among transcriptomic signatures."
        )
    else:
        bullets.append(
            f"> - **FDR Significance**: {t1_n_sig} feature(s) achieve univariate FDR < 0.05, "
            f"and {t1_m_sig} feature(s) retain FDR < 0.05 after multivariate adjustment."
        )
    bullets.append(_build_tier1_or_candidate_bullet(
        feat_title, top_multi_aor, p_top, fdr_top, fdr_met
    ))
    return bullets


def _build_tier1_or_comparison_block(
    t1_resp_n: int, auc_val: float, r2_val: float
) -> List[str]:
    """Builds the OR comparison section header and plot embed lines."""
    return [
        fr"### 1.3. Univariate vs. Multivariate Odds Ratio Comparison "
        fr"for ICI Immune Signatures ($N = {t1_resp_n}$)",
        fr"Contrasting unadjusted Univariate $\text{{OR}}$ (blue circles) against "
        fr"multivariable-adjusted $\text{{aOR}}$ (orange squares) across "
        fr"$N = {t1_resp_n}$ ICI patients "
        fr"(Model AUROC = **{auc_val:.3f}**, McFadden $R^2 = {r2_val:.3f}$):", "",
        "![Tier 1 ICI Univariate vs Multivariate OR Comparison]"
        "(../../plots/clinical/tier1_uni_vs_multi_or_forest.png)", "",
        "> [!INSIGHT] Key Insights: ICI Immune Signature Predictors (Tier 1)",
    ]


def _build_report_tier1_plots_and_insights(
    t1_resp_n: int, t1_n_sig: int, t1_m_sig: int, tier1_rf: pd.DataFrame,
    metrics: Dict[str, float], sig_mask: pd.Series, or_min: float, or_max: float,
    sig_max_p: float, top_multi_feat: str, top_multi_aor: float,
    top_multi_p: float, top_multi_fdr: float
) -> List[str]:
    """Builds Tier 1 rationale, plot embeds, OR comparison section, and insight callout."""
    lines = _build_report_tier1_rationale_block()
    lines.extend(_build_report_tier1_rf_block(t1_resp_n, tier1_rf))
    auc_val = metrics.get("auc", float("nan"))
    r2_val = metrics.get("pseudo_r2", float("nan"))
    feat_title = _format_feature_name(top_multi_feat)
    p_top = f"{top_multi_p:.3f}" if not pd.isna(top_multi_p) else "N/A"
    fdr_top = f"{top_multi_fdr:.3f}" if not pd.isna(top_multi_fdr) else "N/A"
    fdr_met = (not pd.isna(top_multi_fdr)) and (top_multi_fdr < _FDR_ALPHA)
    lines.extend(_build_tier1_or_comparison_block(t1_resp_n, auc_val, r2_val))
    lines.extend(_build_report_tier1_collinearity_text(sig_mask, or_min, or_max, sig_max_p))
    lines.extend(_build_tier1_or_insight_bullets(
        t1_n_sig, t1_m_sig, feat_title, top_multi_aor, p_top, fdr_top, fdr_met
    ))
    lines.append("")
    return lines


def _build_report_tier1_section(
    t1_total_n: int,
    t1_resp_n: int,
    t1_n_resp: int,
    t1_n_nonresp: int,
    t1_n_sig: int,
    t1_m_sig: int,
    tier1_rf: pd.DataFrame,
    metrics: Dict[str, float],
    sig_mask: pd.Series,
    or_min: float,
    or_max: float,
    sig_max_p: float,
    top_multi_feat: str, top_multi_aor: float, top_multi_p: float, top_multi_fdr: float,
    ici_plain_str: str, ici_count: int
) -> List[str]:
    """Builds Tier 1 markdown report lines."""
    lines = _build_report_tier1_preamble(
        t1_total_n, t1_resp_n, t1_n_resp, t1_n_nonresp, t1_n_sig, t1_m_sig,
        ici_plain_str, ici_count
    )
    lines.extend(_build_report_tier1_plots_and_insights(
        t1_resp_n, t1_n_sig, t1_m_sig, tier1_rf, metrics, sig_mask,
        or_min, or_max, sig_max_p,
        top_multi_feat, top_multi_aor, top_multi_p, top_multi_fdr
    ))
    return lines


def _build_report_tier2_preamble(
    ici2_n: int, ici2_resp_n: int, ici2_n_resp: int, ici2_n_nonresp: int, ici2_feat_count: int,
    ici2_n_sig: int, ici2_m_sig: int, ici_plain_str: str
) -> List[str]:
    """Builds Tier 2 preamble lines."""
    return [
        f"## 2. Tier 2: ICI Granular Clinical Feature Selection ($N = {ici2_n}$)", "",
        "> [!INFO] What, Why & Questions — Tier 2",
        f"> **What We Are Doing**: Evaluating {ici2_feat_count} encoded dummy variables "
        f"derived from granular baseline categorical clinical covariates available in the "
        f"ICI trial cohorts ($N = {ici2_n}$) — including clinical staging (`CLINICAL_STAGE`), "
        "anatomical biopsy site (`BIOPSY_SITE`), histological subtype (`TISSUE_SUBTYPE`), "
        "prior ICI therapy (`PRIOR_ICI_RX`), prior non-ICI therapy (`PRIOR_RX`), "
        "biopsy timing (`SAMPLE_TREATMENT`), and metastasis status (`METASTASIZED`) — against "
        "**immunotherapy binary response** using the same RF + Logistic Regression framework.",
        "> **Why We Are Doing It**: Granular clinical covariates may independently predict "
        "ICI response beyond immune expression signatures.",
        "> - **Questions**:",
        ">   1. *Which clinical staging or treatment covariates "
        "carry independent response signal?*",
        ">   2. *Does prior ICI therapy confound response classification?*",
        ">   3. *Do anatomical biopsy sites carry differential response rates?*", "",
        f"- **ICI Cohort Sample Size**: {ici2_n} patients ({ici_plain_str})",
        f"- **Response Evaluation Cohort**: {ici2_resp_n} patients "
        f"({ici2_n_resp} responders / {ici2_n_nonresp} non-responders)",
        f"- **Encoded Dummy Features**: {ici2_feat_count} dummy variables",
        f"- **Univariate FDR-Significant Predictors (FDR < 0.05)**: {ici2_n_sig} features",
        f"- **Multivariate FDR-Significant Predictors (FDR < 0.05)**: {ici2_m_sig} feature(s)", "",
    ]


def _build_tier2_rf_key_takeaways(df_rf: pd.DataFrame, n_top: int = 5) -> List[str]:
    """Builds a dynamic INSIGHT callout for Tier 2 RF clinical feature importances."""
    if df_rf.empty:
        return []
    top = df_rf.head(n_top).reset_index(drop=True)
    total_imp = df_rf["Importance"].sum()
    lines: List[str] = ["> [!INSIGHT] Key Takeaways: Tier 2 Clinical RF Importance"]
    top1_name = _format_feature_name(top.iloc[0]["Feature"])
    top1_imp = top.iloc[0]["Importance"]
    top1_pct = (top1_imp / total_imp * 100) if total_imp > 0 else 0.0
    lines.append(
        f"> - **Top Clinical Predictor**: **`{top1_name}`** accounts for "
        f"{top1_pct:.1f}% of total RF Gini importance among granular clinical attributes."
    )
    lines.append(f"> - **Top {n_top} Clinical Features by Gini Importance**:")
    for _, row in top.iterrows():
        feat_label = _format_feature_name(row["Feature"])
        pct = (row["Importance"] / total_imp * 100) if total_imp > 0 else 0.0
        lines.append(f">   - `{feat_label}` — {pct:.1f}%")
    lines.append("")
    return lines


def _build_tier2_nominal_bullets(
    top1_feat: str, top1_aor: float, top1_p: float, top2_feat: str, top2_aor: float, top2_p: float
) -> List[str]:
    """Builds nominal predictor bullets for Tier 2 Odds Ratio comparison."""
    bullets: List[str] = []
    if top1_feat != "N/A":
        t1_title = _format_feature_name(top1_feat)
        t1_p_str = _format_p_value(top1_p)
        bullets.append(
            fr"> - **Top Nominal Factor**: **`{t1_title}`** "
            fr"($\text{{aOR}} = {top1_aor:.2f}$, nominal $p = {t1_p_str}$) "
            "demonstrates the strongest unadjusted clinical association, "
            "though it does not clear FDR correction."
        )
    if top2_feat != "N/A":
        t2_title = _format_feature_name(top2_feat)
        t2_p_str = _format_p_value(top2_p)
        bullets.append(
            fr"> - **Second Nominal Factor**: **`{t2_title}`** "
            fr"($\text{{aOR}} = {top2_aor:.2f}$, nominal $p = {t2_p_str}$) "
            "shows weak secondary trend."
        )
    bullets.append(
        "> - **Multivariate Attenuation**: All clinical covariates attenuate toward "
        fr"$\text{{aOR}} = 1.0$ in mutual adjustment, underscoring that routine "
        "clinical attributes cannot substitute for multi-omic biomarker panels."
    )
    return bullets


def _build_tier2_or_insight_bullets(
    ici2_n_sig: int,
    ici2_m_sig: int,
    top1_feat: str,
    top1_aor: float,
    top1_p: float,
    top2_feat: str,
    top2_aor: float,
    top2_p: float,
) -> List[str]:
    """Builds insight bullets for Tier 2 Odds Ratio comparison."""
    bullets: List[str] = []
    if ici2_n_sig == 0 and ici2_m_sig == 0:
        bullets.append(
            "> - **Lack of FDR Significance**: Neither univariate nor multivariate "
            "logistic regression models yield granular clinical covariates passing "
            fr"FDR correction ($q < {_FDR_ALPHA}$). Baseline clinical attributes alone "
            "(biopsy site, staging, prior therapy) provide insufficient predictive power "
            "for anti-PD-1 response without molecular TME profiling."
        )
    else:
        bullets.append(
            f"> - **FDR Significance**: {ici2_n_sig} clinical feature(s) achieve "
            f"univariate FDR < 0.05, and {ici2_m_sig} feature(s) retain FDR < 0.05 "
            "after multivariate adjustment."
        )
    bullets.extend(_build_tier2_nominal_bullets(
        top1_feat, top1_aor, top1_p, top2_feat, top2_aor, top2_p
    ))
    return bullets


def _build_report_tier2_rf_block(ici2_resp_n: int, ici2_rf: pd.DataFrame) -> List[str]:
    """Builds Tier 2 RF section header and key takeaways."""
    lines = [
        fr"### 2.1. Random Forest Importance for Granular ICI Clinical Attributes "
        fr"($N = {ici2_resp_n}$)",
        fr"Top-20 RF clinical predictors of anti-PD-1 response trained on "
        fr"$N = {ici2_resp_n}$ ICI patients:", "",
        "![Tier 2 ICI RF Importance]"
        "(../../plots/clinical/ici_tier2_rf_importance.png)", "",
    ]
    lines.extend(_build_tier2_rf_key_takeaways(ici2_rf))
    return lines


def _build_report_tier2_plots_and_insights(
    ici2_resp_n: int, ici2_n_sig: int, ici2_m_sig: int, ici2_rf: pd.DataFrame,
    metrics: Dict[str, float], top1_feat: str, top1_aor: float, top1_p: float,
    top2_feat: str, top2_aor: float, top2_p: float
) -> List[str]:
    """Builds Tier 2 plots and insight lines."""
    lines = _build_report_tier2_rf_block(ici2_resp_n, ici2_rf)
    auc_val = metrics.get('auc', float('nan'))
    r2_val = metrics.get('pseudo_r2', float('nan'))
    lines.extend([
        fr"### 2.2. Univariate vs. Multivariate Odds Ratio Comparison "
        fr"for ICI Clinical Attributes ($N = {ici2_resp_n}$)",
        fr"Contrasting unadjusted Univariate $\text{{OR}}$ (blue circles) "
        fr"against multivariable-adjusted $\text{{aOR}}$ "
        fr"(orange squares) across $N = {ici2_resp_n}$ ICI patients "
        fr"(Model AUROC = **{auc_val:.3f}**, "
        fr"McFadden $R^2 = {r2_val:.3f}$):", "",
        "![Tier 2 ICI Univariate vs Multivariate OR Comparison]"
        "(../../plots/clinical/ici_tier2_uni_vs_multi_or_forest.png)", "",
        "> [!INSIGHT] Key Insights: ICI Granular Clinical Predictors (Tier 2)",
    ])
    lines.extend(_build_tier2_or_insight_bullets(
        ici2_n_sig, ici2_m_sig, top1_feat, top1_aor, top1_p, top2_feat, top2_aor, top2_p
    ))
    lines.append("")
    return lines


def _build_report_tier2_section(
    ici2_n: int, ici2_resp_n: int, ici2_n_resp: int, ici2_n_nonresp: int, ici2_feat_count: int,
    ici2_n_sig: int, ici2_m_sig: int, ici2_rf: pd.DataFrame, metrics: Dict[str, float],
    top1_feat: str, top1_aor: float, top1_p: float, top2_feat: str, top2_aor: float,
    top2_p: float, ici_plain_str: str
) -> List[str]:
    """Builds Tier 2 markdown report lines."""
    lines = _build_report_tier2_preamble(
        ici2_n, ici2_resp_n, ici2_n_resp, ici2_n_nonresp, ici2_feat_count,
        ici2_n_sig, ici2_m_sig, ici_plain_str
    )
    lines.extend(_build_report_tier2_plots_and_insights(
        ici2_resp_n, ici2_n_sig, ici2_m_sig, ici2_rf, metrics,
        top1_feat, top1_aor, top1_p, top2_feat, top2_aor, top2_p
    ))
    return lines


def _build_overall_insights_callout(
    f1_title: str, top_t1_or: float, p1_str: str, fdr1_str: str,
    f1m_title: str, top_t1_m_aor: float, p1m_str: str, fdr1m_str: str,
    t1_auc: float, f2_title: str, top_t2_or: float, p2_str: str, fdr2_str: str,
    t1_total_n: int, t1_resp_n: int
) -> List[str]:
    """Builds overall findings INSIGHT callout box."""
    return [
        "> [!INSIGHT] Key Insights: Overall Findings",
        fr"> 1. **Tier 1 Top Response Biomarker**: **`{f1_title}`** is the single "
        fr"strongest univariate predictor of response across all {t1_total_n} ICI patients "
        fr"($\text{{OR}} = {top_t1_or:.2f}$, $p = {p1_str}$, FDR $= {fdr1_str}$)." ,
        fr"> 2. **Tier 1 Independent Biomarker**: In joint multivariate modelling "
        fr"($N = {t1_resp_n}$), **`{f1m_title}`** remains an independent predictor "
        fr"($\text{{aOR}} = {top_t1_m_aor:.2f}$, $p = {p1m_str}$, FDR $= {fdr1m_str}$; "
        fr"Model AUC = **{t1_auc:.3f}**)." ,
        fr"> 3. **Tier 2 Strongest Clinical Predictor**: **`{f2_title}`** is the strongest "
        fr"univariate ICI clinical predictor "
        fr"($\text{{OR}} = {top_t2_or:.2f}$, $p = {p2_str}$, FDR $= {fdr2_str}$)." ,
        fr"> 4. **Biological Alignment**: Microenvironmental T-cell inflammation signatures "
        fr"consistently associate with response ($\text{{OR}} > 1.0$) "
        fr"across all {t1_total_n} ICI patients.", "",
    ]


def _build_report_summary_section(
    t1_total_n: int, t1_resp_n: int, top_t1_feat: str, top_t1_or: float, top_t1_p: float,
    top_t1_fdr: float, top_t1_m_feat: str, top_t1_m_aor: float, top_t1_m_p: float,
    top_t1_m_fdr: float, t1_auc: float, top_t2_feat: str, top_t2_or: float, top_t2_p: float,
    top_t2_fdr: float
) -> List[str]:
    """Builds biological summary section lines."""
    f1_title = _format_feature_name(top_t1_feat)
    f1m_title = _format_feature_name(top_t1_m_feat)
    f2_title = _format_feature_name(top_t2_feat)

    p1_str, fdr1_str = _format_p_value(top_t1_p), _format_p_value(top_t1_fdr)
    p1m_str, fdr1m_str = _format_p_value(top_t1_m_p), _format_p_value(top_t1_m_fdr)
    p2_str, fdr2_str = _format_p_value(top_t2_p), _format_p_value(top_t2_fdr)

    lines = ["## 3. Key Analytical & Biological Summary", ""]
    lines.extend(_build_overall_insights_callout(
        f1_title, top_t1_or, p1_str, fdr1_str,
        f1m_title, top_t1_m_aor, p1m_str, fdr1m_str,
        t1_auc, f2_title, top_t2_or, p2_str, fdr2_str,
        t1_total_n, t1_resp_n
    ))
    return lines


def _build_limitations_callout(
    t1_resp_n: int, t1_n_resp: int, ici2_feat_count: int, ici_plain_str: str
) -> List[str]:
    """Builds analytical scope & limitations callout box."""
    return [
        "> [!WARNING] Analytical Scope & Limitations",
        fr"> - **Small Response-Labelled Cohort**: The pooled ICI response dataset "
        fr"($N = {t1_resp_n}$, {t1_n_resp} responders) is small relative to the feature "
        f"space in Tier 2 ({ici2_feat_count} encoded variables), increasing risk of overfitting.",
        f"> - **Cohort Heterogeneity**: {ici_plain_str} differ in "
        "treatment agent, response definition, and biopsy timing.",
        "> - **Collinearity Among Immune Signatures**: The six transcriptomic immune signatures "
        "share overlapping gene sets (`CD274`, `STAT1`, `IDO1`), inducing collinearity.",
        f"> - **High-Dimensional Tier 2 Feature Space**: One-hot encoding of "
        f"{ici2_feat_count} clinical variables against $N = {t1_resp_n}$ patients "
        f"creates a sparse matrix.",
        "> - **No TCGA Pathological Staging**: Granular TNM staging and AJCC categories "
        "are not available in the ICI cohorts.", "",
    ]


def _build_script_entries_primary(p2: Path) -> List[Tuple[str, Path, str]]:
    """Builds primary script entries for architecture callout box."""
    e1 = (
        "run_clinical_feature_selection.py",
        p2 / "run_clinical_feature_selection.py",
        "Evaluates immune signatures (Tier 1) & clinical covariates (Tier 2) via RF/Logit.",
    )
    e2 = (
        "run_univariate_associations.py",
        p2 / "run_univariate_associations.py",
        "Evaluates per-cohort and pooled univariate statistical associations & forest plots.",
    )
    return [e1, e2]


def _build_script_entries_support(
    p1: Path, q1_src: Path, root_src: Path
) -> List[Tuple[str, Path, str]]:
    """Builds supporting module entries for architecture callout box."""
    e3 = (
        "clean_data.py",
        p1 / "clean_data.py",
        "Preprocesses raw clinical metadata and expression into cleaned CSV matrices.",
    )
    e4 = (
        "signatures.py",
        q1_src / "signatures.py",
        "Computes transcriptomic immune signatures across cohort expression matrices.",
    )
    e5 = (
        "styles.py",
        root_src / "styles.py",
        "Single source of truth for Okabe-Ito colour palettes and Matplotlib styling.",
    )
    return [e3, e4, e5]


def _build_architecture_entries() -> List[Tuple[str, Path, str]]:
    """Builds entry list for software architecture callout box."""
    p2 = _SUBPROJECT_ROOT / "scripts" / "pillar-2-clinical-subtyping"
    p1 = _SUBPROJECT_ROOT / "scripts" / "pillar-1-cohort-preprocessing"
    q1_src = _SUBPROJECT_ROOT / "src"
    root_src = _SUBPROJECT_ROOT.parent / "src"
    entries = _build_script_entries_primary(p2)
    entries.extend(_build_script_entries_support(p1, q1_src, root_src))
    return entries


def _build_architecture_callout() -> List[str]:
    """Builds software architecture callout box for report footer using shared generator."""
    callout_str = generate_script_reference_callout(
        _build_architecture_entries(),
        base_dir=REPORT_DIR,
        callout_type="[!formula]+",
        title="Clinical Feature Selection Script Execution & Software Module Architecture",
    )
    return callout_str.splitlines()


def _build_report_limitations_section(
    t1_resp_n: int, t1_n_resp: int, ici2_feat_count: int, t1_total_n: int, ici_plain_str: str
) -> List[str]:
    """Builds limitations & software architecture section lines."""
    lines = ["## 4. Methodological Limitations & Future Directions", ""]
    lines.extend(_build_limitations_callout(t1_resp_n, t1_n_resp, ici2_feat_count, ici_plain_str))
    lines.extend(_build_architecture_callout())
    return lines


def _compute_tier1_stats(
    tier1_df: pd.DataFrame, tier1_uni_or: pd.DataFrame, tier1_multi_or: pd.DataFrame
) -> Tuple[int, int, int, int, int, int]:
    """Computes sample size and significance counts for Tier 1."""
    t1_tot = len(tier1_df)
    t1_resp = int(tier1_df["RESPONSE_BINARY"].notna().sum())
    has_resp = tier1_df["RESPONSE_BINARY"].notna().any()
    t1_r = int(tier1_df["RESPONSE_BINARY"].sum()) if has_resp else 0
    t1_nr = t1_resp - t1_r
    t1_sig = int((tier1_uni_or["FDR_adj_p"] < _FDR_ALPHA).sum()) if not tier1_uni_or.empty else 0
    t1_m_sig = (
        int((tier1_multi_or["FDR_adj_p"] < _FDR_ALPHA).sum())
        if not tier1_multi_or.empty else 0
    )
    return t1_tot, t1_resp, t1_r, t1_nr, t1_sig, t1_m_sig


def _extract_report_tier1_vars(
    tier1_df: pd.DataFrame, tier1_uni_or: pd.DataFrame, tier1_multi_or: pd.DataFrame
) -> Tuple[int, int, int, int, int, int, pd.Series, float, float, float, pd.Series, pd.Series]:
    """Extracts summary variables for Tier 1 report section."""
    t1_tot, t1_resp, t1_r, t1_nr, t1_sig, t1_m_sig = _compute_tier1_stats(
        tier1_df, tier1_uni_or, tier1_multi_or
    )
    null_s = pd.Series({
        "Feature": "N/A", "Odds Ratio (OR)": float("nan"),
        "p-value": float("nan"), "FDR_adj_p": float("nan"),
    })
    top1 = tier1_uni_or.iloc[0] if not tier1_uni_or.empty else null_s
    top1_m = tier1_multi_or.iloc[0] if not tier1_multi_or.empty else null_s

    sig_mask = tier1_uni_or["Feature"].isin([
        f"Z_{s}" for s in [
            "IFN_gamma", "TIS", "CYT", "CD8_Tcell", "IMPRES", "PD_L1",
            "M1_M2_Ratio", "Macrophage_STV_Score",
        ]
    ])
    or_min = tier1_uni_or.loc[sig_mask, "Odds Ratio (OR)"].min() if sig_mask.any() else float("nan")
    or_max = tier1_uni_or.loc[sig_mask, "Odds Ratio (OR)"].max() if sig_mask.any() else float("nan")
    sig_max_p = tier1_uni_or.loc[sig_mask, "p-value"].max() if sig_mask.any() else float("nan")

    return (
        t1_tot, t1_resp, t1_r, t1_nr, t1_sig, t1_m_sig,
        sig_mask, or_min, or_max, sig_max_p, top1, top1_m
    )


def _compute_tier2_stats(
    ici2_df: pd.DataFrame, ici2_encoded: pd.DataFrame, ici2_y: pd.Series,
    ici2_uni_or: pd.DataFrame, ici2_multi_or: pd.DataFrame
) -> Tuple[int, int, int, int, int, int, int]:
    """Computes sample size and significance counts for Tier 2."""
    i2_tot, i2_resp = len(ici2_df), int(ici2_y.notna().sum())
    i2_r = int(ici2_y.sum()) if ici2_y.notna().any() else 0
    i2_nr = i2_resp - i2_r
    i2_cnt = len(ici2_encoded.columns)
    i2_sig = int((ici2_uni_or["FDR_adj_p"] < _FDR_ALPHA).sum()) if not ici2_uni_or.empty else 0
    i2_m_sig = (
        int((ici2_multi_or["FDR_adj_p"] < _FDR_ALPHA).sum())
        if not ici2_multi_or.empty else 0
    )
    return i2_tot, i2_resp, i2_r, i2_nr, i2_cnt, i2_sig, i2_m_sig


def _extract_report_tier2_vars(
    ici2_df: pd.DataFrame, ici2_encoded: pd.DataFrame, ici2_y: pd.Series,
    ici2_uni_or: pd.DataFrame, ici2_multi_or: pd.DataFrame
) -> Tuple[int, int, int, int, int, int, int, pd.Series, str, float, float, str, float, float]:
    """Extracts summary variables for Tier 2 report section."""
    i2_tot, i2_resp, i2_r, i2_nr, i2_cnt, i2_sig, i2_m_sig = _compute_tier2_stats(
        ici2_df, ici2_encoded, ici2_y, ici2_uni_or, ici2_multi_or
    )
    null_s = pd.Series({
        "Feature": "N/A", "Odds Ratio (OR)": float("nan"),
        "p-value": float("nan"), "FDR_adj_p": float("nan"),
    })
    top2 = ici2_uni_or.iloc[0] if not ici2_uni_or.empty else null_s
    fav = (
        ici2_multi_or[ici2_multi_or["Odds Ratio (OR)"] > 1.0]
        .sort_values("p-value")
        .reset_index(drop=True)
    ) if not ici2_multi_or.empty else ici2_uni_or
    t1_f = fav.iloc[0]["Feature"] if len(fav) >= 1 else "N/A"
    t1_aor = fav.iloc[0]["Odds Ratio (OR)"] if len(fav) >= 1 else float("nan")
    t1_p = fav.iloc[0]["p-value"] if len(fav) >= 1 else float("nan")
    t2_f = fav.iloc[1]["Feature"] if len(fav) >= 2 else "N/A"
    t2_aor = fav.iloc[1]["Odds Ratio (OR)"] if len(fav) >= 2 else float("nan")
    t2_p = fav.iloc[1]["p-value"] if len(fav) >= 2 else float("nan")

    return (
        i2_tot, i2_resp, i2_r, i2_nr, i2_cnt,
        i2_sig, i2_m_sig, top2, t1_f, t1_aor, t1_p, t2_f, t2_aor, t2_p
    )


def _write_report_to_disk(lines: List[str], report_path: Path) -> None:
    """Writes report text lines to target file and mirrors to root reports."""
    report_path.parent.mkdir(exist_ok=True, parents=True)
    txt = "\n".join(lines)
    report_path.write_text(txt, encoding="utf-8")
    print(
        f"Two-tiered ICI feature selection report successfully written "
        f"to {rel_path(report_path)}"
    )
    root_dir = _SUBPROJECT_ROOT.parent / "reports" / "pillar-2-clinical-subtyping"
    root_report = root_dir / report_path.name
    if root_dir.exists():
        root_report.write_text(txt, encoding="utf-8")


def _unpack_t1_vars(t1_vars: tuple) -> tuple:
    """Unpacks Tier 1 variable tuple for report assembly."""
    (
        t1_tot, t1_resp, t1_r, t1_nr, t1_sig, t1_m_sig,
        sig_mask, or_min, or_max, sig_max_p, top1, top1_m
    ) = t1_vars
    return (
        t1_tot, t1_resp, t1_r, t1_nr, t1_sig, t1_m_sig,
        sig_mask, or_min, or_max, sig_max_p, top1, top1_m
    )


def _unpack_i2_vars(i2_vars: tuple) -> tuple:
    """Unpacks Tier 2 variable tuple for report assembly."""
    (
        i2_tot, i2_resp, i2_r, i2_nr, i2_cnt, i2_sig, i2_m_sig,
        top2, t1_f, t1_aor, t1_p, t2_f, t2_aor, t2_p
    ) = i2_vars
    return (
        i2_tot, i2_resp, i2_r, i2_nr, i2_cnt, i2_sig, i2_m_sig,
        top2, t1_f, t1_aor, t1_p, t2_f, t2_aor, t2_p
    )


def _assemble_report_tail(
    lines: List[str], top1: pd.Series, top1_m: pd.Series, top2: pd.Series,
    t1_tot: int, t1_resp: int, tier1_multi_metrics: Dict[str, float],
    i2_cnt: int, t1_r: int, ici_plain_str: str
) -> List[str]:
    """Appends summary and limitations sections to the report line list."""
    lines.extend(_build_report_summary_section(
        t1_tot, t1_resp, top1["Feature"], top1["Odds Ratio (OR)"], top1["p-value"],
        top1["FDR_adj_p"], top1_m["Feature"], top1_m["Odds Ratio (OR)"],
        top1_m["p-value"], top1_m["FDR_adj_p"],
        tier1_multi_metrics.get("auc", float("nan")), top2["Feature"], top2["Odds Ratio (OR)"],
        top2["p-value"], top2["FDR_adj_p"]
    ))
    lines.extend(_build_report_limitations_section(t1_resp, t1_r, i2_cnt, t1_tot, ici_plain_str))
    return lines


def _build_report_header_and_tiers(
    t1_vars: tuple, i2_vars: tuple, tier1_rf: pd.DataFrame,
    tier1_multi_metrics: Dict[str, float], ici2_rf: pd.DataFrame,
    ici2_multi_metrics: Dict[str, float], t3_rf: pd.DataFrame,
    t3_multi_metrics: Dict[str, float], dataset_configs: Tuple[DatasetConfig, ...]
) -> List[str]:
    """Builds frontmatter and Tier 1, 2, and 3 report section lines."""
    ici_plain, ici_fmt, ici_cnt = _format_ici_cohort_names(dataset_configs)
    v1 = _unpack_t1_vars(t1_vars)
    v2 = _unpack_i2_vars(i2_vars)
    lines: List[str] = _build_report_frontmatter_header(
        v1[0], v2[0], v2[4], ici_plain, ici_fmt, ici_cnt
    )
    lines.extend(_build_report_tier1_section(
        v1[0], v1[1], v1[2], v1[3], v1[4], v1[5], tier1_rf, tier1_multi_metrics,
        v1[6], v1[7], v1[8], v1[9], v1[11]["Feature"], v1[11]["Odds Ratio (OR)"],
        v1[11]["p-value"], v1[11]["FDR_adj_p"], ici_plain, ici_cnt
    ))
    lines.extend(_build_report_tier2_section(
        v2[0], v2[1], v2[2], v2[3], v2[4], v2[5], v2[6], ici2_rf, ici2_multi_metrics,
        v2[8], v2[9], v2[10], v2[11], v2[12], v2[13], ici_plain
    ))
    lines.extend(_build_report_tier3_section(t3_rf, t3_multi_metrics, v1[1]))
    return lines


def _assemble_report_body(
    t1_vars: tuple, i2_vars: tuple, tier1_rf: pd.DataFrame,
    tier1_multi_metrics: Dict[str, float], ici2_rf: pd.DataFrame,
    ici2_multi_metrics: Dict[str, float], t3_rf: pd.DataFrame,
    t3_multi_metrics: Dict[str, float], dataset_configs: Tuple[DatasetConfig, ...]
) -> List[str]:
    """Assembles all markdown report sections into a single line list."""
    lines = _build_report_header_and_tiers(
        t1_vars, i2_vars, tier1_rf, tier1_multi_metrics, ici2_rf, ici2_multi_metrics,
        t3_rf, t3_multi_metrics, dataset_configs
    )
    (t1_tot, t1_resp, t1_r, _, _, _, _, _, _, _, top1, top1_m) = _unpack_t1_vars(t1_vars)
    (i2_tot, _, _, _, i2_cnt, _, _, top2, _, _, _, _, _, _) = _unpack_i2_vars(i2_vars)
    ici_plain, _, _ = _format_ici_cohort_names(dataset_configs)
    return _assemble_report_tail(
        lines, top1, top1_m, top2, t1_tot, t1_resp, tier1_multi_metrics, i2_cnt, t1_r, ici_plain
    )


def _generate_two_tiered_report(
    tier1_df: pd.DataFrame, tier1_rf: pd.DataFrame, tier1_uni_or: pd.DataFrame,
    tier1_multi_or: pd.DataFrame, tier1_multi_metrics: Dict[str, float],
    ici2_df: pd.DataFrame, ici2_encoded: pd.DataFrame, ici2_y: pd.Series,
    ici2_rf: pd.DataFrame, ici2_uni_or: pd.DataFrame, ici2_multi_or: pd.DataFrame,
    ici2_multi_metrics: Dict[str, float], t3_rf: pd.DataFrame,
    t3_multi_or: pd.DataFrame, t3_multi_metrics: Dict[str, float],
    dataset_configs: Tuple[DatasetConfig, ...], report_path: Path
) -> None:
    """Generates a multi-tiered Markdown report with dynamic sample metrics."""
    t1_vars = _extract_report_tier1_vars(tier1_df, tier1_uni_or, tier1_multi_or)
    i2_vars = _extract_report_tier2_vars(ici2_df, ici2_encoded, ici2_y, ici2_uni_or, ici2_multi_or)
    lines = _assemble_report_body(
        t1_vars, i2_vars, tier1_rf, tier1_multi_metrics, ici2_rf, ici2_multi_metrics,
        t3_rf, t3_multi_metrics, dataset_configs
    )
    _write_report_to_disk(lines, report_path)


# ---------------------------------------------------------------------------
# Main Orchestration Pipelines
# ---------------------------------------------------------------------------


def _run_tier1_pipeline(
    dataset_configs: Tuple[DatasetConfig, ...]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Executes Tier 1 immune feature selection pipeline."""
    print("\n--- Tier 1: ICI Multi-Cohort Immune Feature Selection ---")
    tier1_df = _load_tier1_dataset(dataset_configs)
    ici_n = len([c for c in dataset_configs if getattr(c, "merge_enabled", True)])
    print(
        f"Loaded {len(tier1_df)} response-labelled ICI patients "
        f"across {ici_n} active trial cohorts."
    )

    tier1_rf = _evaluate_tier1_rf_response(tier1_df, PLOT_DIR)
    tier1_uni_or = _evaluate_tier1_logistic(tier1_df, PLOT_DIR)
    tier1_multi_or, tier1_multi_metrics = _evaluate_tier1_multivariate_logistic(tier1_df, PLOT_DIR)

    title = "Tier 1 Univariate vs. Multivariate Odds Ratios (ICI Cohorts)"
    xlabel = "Odds Ratio (Log Scale — 95% CI per +1 SD)"
    out_path = PLOT_DIR / "tier1_uni_vs_multi_or_forest.png"
    _plot_univariate_vs_multivariate_comparison(
        tier1_uni_or, tier1_multi_or, title, xlabel, out_path,
        canonical_order=TIER1_PLOT_ORDER,
    )
    return tier1_df, tier1_rf, tier1_uni_or, tier1_multi_or, tier1_multi_metrics


def _run_tier2_pipeline(
    dataset_configs: Tuple[DatasetConfig, ...]
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    Dict[str, float]
]:
    """Executes Tier 2 granular clinical feature selection pipeline."""
    print("\n--- Tier 2: ICI Granular Clinical Feature Selection ---")
    ici2_df, ici2_encoded, ici2_y = _load_ici_tier2_features(dataset_configs)
    n_feats = len(ici2_encoded.columns)
    print(
        f"Loaded {len(ici2_df)} response-labelled ICI patients; "
        f"{n_feats} encoded dummy features."
    )

    ici2_rf = _evaluate_ici_tier2_rf(ici2_encoded, ici2_y, PLOT_DIR)
    ici2_uni_or = _evaluate_ici_tier2_logistic(ici2_encoded, ici2_y, PLOT_DIR)
    ici2_multi_or, ici2_multi_metrics = _evaluate_ici_tier2_multivariate_logistic(
        ici2_encoded, ici2_y, ici2_uni_or, PLOT_DIR
    )

    title = "Tier 2 ICI Univariate vs. Multivariate Odds Ratios"
    xlabel = "Odds Ratio (Log Scale — 95% CI)"
    out_path = PLOT_DIR / "ici_tier2_uni_vs_multi_or_forest.png"
    _plot_univariate_vs_multivariate_comparison(
        ici2_uni_or.head(10), ici2_multi_or, title, xlabel, out_path
    )
    return ici2_df, ici2_encoded, ici2_y, ici2_rf, ici2_uni_or, ici2_multi_or, ici2_multi_metrics


def _build_tier3_insights(top_df: pd.DataFrame, n_samp: int) -> List[str]:
    """Builds key insights callout box for Tier 3 Unified Leaderboard."""
    top1_feat = _format_feature_name(top_df.iloc[0]["Feature"])
    top1_imp = top_df.iloc[0]["Importance"] * 100.0
    l1 = (
        f"> - **Top Overall Biomarker**: **`{top1_feat}`** ranks #1 across all "
        f"molecular and clinical variables, accounting for {top1_imp:.1f}% of total "
        f"Gini importance ($N = {n_samp}$)."
    )
    lines = [
        "> [!INSIGHT] Key Takeaways: Unified Multimodal Leaderboard",
        l1,
        "> - **Top 5 Features Across Domains**:",
    ]
    for idx in range(min(5, len(top_df))):
        f_name = _format_feature_name(top_df.iloc[idx]["Feature"])
        f_imp = top_df.iloc[idx]["Importance"] * 100.0
        lines.append(f">   - `{f_name}` — {f_imp:.1f}%")
    lines.append("")
    return lines


def _build_tier3_head_lines(n_samp: int) -> List[str]:
    """Builds Tier 3 header lines and info callout box."""
    w_line = (
        f"> **What We Are Doing**: Evaluating all Tier 1 molecular immune signatures and "
        f"Tier 2 granular clinical covariates head-to-head in a single unified Random Forest "
        f"model ($N = {n_samp}$)."
    )
    y_line = (
        "> **Why We Are Doing It**: Evaluates whether molecular signatures outrank clinical "
        "attributes when competing in the same model space."
    )
    img1 = (
        "![Tier 3 Unified Multimodal RF Importance]"
        "(../../plots/clinical/unified_tier3_rf_importance.png)"
    )
    return [
        f"## 3. Tier 3: Unified Multimodal Feature Selection Leaderboard ($N = {n_samp}$)",
        "",
        "> [!INFO] What & Why — Tier 3 Unified Multimodal Leaderboard",
        w_line,
        y_line,
        "",
        f"### 3.1. Unified Multimodal Random Forest Importance ($N = {n_samp}$)",
        "",
        img1,
        "",
    ]


def _build_report_tier3_section(
    t3_rf: pd.DataFrame, t3_multi_metrics: Dict[str, float], n_samp: int
) -> List[str]:
    """Builds Tier 3 Unified Multimodal Feature Selection report section lines."""
    auc_val = t3_multi_metrics.get("auc", 0.5)
    r2_val = t3_multi_metrics.get("pseudo_r2", 0.0)
    img2 = (
        "![Tier 3 Unified Multivariate OR Forest]"
        "(../../plots/clinical/unified_tier3_multivariate_or_forest.png)"
    )
    p2_line = (
        f"Multivariate-adjusted Odds Ratios (\\text{{aOR}}) across top molecular and clinical "
        f"features (Model AUC = **{auc_val:.3f}**, McFadden $R^2 = {r2_val:.3f}$):"
    )
    lines = _build_tier3_head_lines(n_samp)
    lines.extend(_build_tier3_insights(t3_rf, n_samp))
    lines.extend([
        f"### 3.2. Unified Multivariate Odds Ratio Comparison ($N = {n_samp}$)",
        p2_line,
        "",
        img2,
        "",
    ])
    return lines


def _run_feature_selection_pipelines(
    dataset_configs: Tuple[DatasetConfig, ...]
) -> Tuple[tuple, tuple, tuple]:
    """Executes Tier 1, Tier 2, and Tier 3 feature selection pipelines."""
    t1_res = _run_tier1_pipeline(dataset_configs)
    t2_res = _run_tier2_pipeline(dataset_configs)
    t3_res = _run_tier3_pipeline(t1_res[0], t2_res[1], PLOT_DIR)
    return t1_res, t2_res, t3_res


def _print_header_and_prep_dirs(dataset_configs: Tuple[DatasetConfig, ...]) -> None:
    """Prints execution header banner and creates target plot/report directories."""
    ici_plain, _, _ = _format_ici_cohort_names(dataset_configs)
    print("==================================================")
    print("Two-Tiered ICI Clinical & Transcriptomic Feature Selection")
    print(f"(Evaluating immunotherapy Binary Response — {ici_plain})")
    print("==================================================")
    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_DIR.mkdir(exist_ok=True, parents=True)


def main() -> None:
    """Executes two-tiered ICI clinical feature selection pipeline."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Dataset configuration file not found at {rel_path(CONFIG_PATH)}")

    dataset_configs = load_dataset_config(CONFIG_PATH)
    _print_header_and_prep_dirs(dataset_configs)

    t1_res, t2_res, t3_res = _run_feature_selection_pipelines(dataset_configs)
    t1_df, t1_rf, t1_u_or, t1_m_or, t1_m_met = t1_res
    t2_df, t2_enc, t2_y, t2_rf, t2_u_or, t2_m_or, t2_m_met = t2_res
    t3_rf, t3_m_or, t3_m_met = t3_res

    print("\nExporting multi-tiered ICI feature selection report...")
    _generate_two_tiered_report(
        t1_df, t1_rf, t1_u_or, t1_m_or, t1_m_met,
        t2_df, t2_enc, t2_y, t2_rf, t2_u_or, t2_m_or, t2_m_met,
        t3_rf, t3_m_or, t3_m_met,
        dataset_configs, REPORT_PATH
    )

    run_companion_scripts(_COMPANION_SCRIPTS, base_dir=_SUBPROJECT_ROOT)

    print("\n==================================================")
    print("Done! Multi-tiered ICI clinical feature selection completed.")
    print("==================================================")


if __name__ == "__main__":
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(_SUBPROJECT_ROOT).as_posix()}")
            main()
