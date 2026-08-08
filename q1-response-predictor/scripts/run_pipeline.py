"""Master LOCO Cross-Validation & Model Evaluation Pipeline for Q1.

Trains, evaluates, and benchmarks machine learning models across melanoma
immunotherapy cohorts (Liu 2019, Hugo 2016, Riaz 2017) and validates downstream
overall survival stratification on TCGA-SKCM.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import pickle
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Set matplotlib backend to Agg to avoid GUI display errors.
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SUBPROJECT_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SUBPROJECT_ROOT.parent

if str(SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from scripts.exploratory_plots.generate_loco_heatmap import (
    plot_loco_heatmap_from_results,
)

from src.data_loaders import (
    load_hugo_2016,
    load_liu_2019,
    load_riaz_2017,
)
from src.evaluation import (
    calculate_cindex,
    calculate_extended_metrics,
    find_optimal_threshold,
    plot_calibration_curves,
    plot_confusion_matrices,
    plot_pr_curves,
    plot_roc_curves,
    plot_survival_2x2_grid,
    run_survival_analysis,
)
from src.models import get_model, run_loco_cv
from src.signatures import extract_all_signatures
from src.styles import set_presentation_style
from src.utils.formatting import generate_obsidian_frontmatter
from src.utils.logging import TeeStream, display_path
from src.utils.paths import (
    DATA_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    get_subproject_log_dir,
)

# Apply central presentation aesthetics at module load.
set_presentation_style()

# ---------------------------------------------------------------------------
# Paths and Configuration
# ---------------------------------------------------------------------------

MODEL_PLOTS_DIR = PLOTS_DIR / "models"
MODEL_PLOTS_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

LOG_DIR = get_subproject_log_dir(SCRIPT_DIR)
LOG_PATH = LOG_DIR / "q1_pipeline.log"

# ---------------------------------------------------------------------------
# Cohort name constants
# ---------------------------------------------------------------------------

_COHORT_LIU: str = "Liu 2019"
_COHORT_HUGO: str = "Hugo 2016"
_COHORT_RIAZ: str = "Riaz 2017"
_COHORT_TCGA: str = "TCGA-SKCM"

# ---------------------------------------------------------------------------
# File / directory path constants
# ---------------------------------------------------------------------------

_TCGA_DATASET_DIR: str = "skcm_tcga_pan_can_atlas_2018"
_EXPR_FILENAME: str = "expr_cleaned.csv"
_CLIN_FILENAME: str = "clin_cleaned.csv"
_REPORT_SUBDIR: str = "pillar-4-out-of-cohort-benchmarks"
_REPORT_FILENAME: str = "model_evaluation_report.md"
_MODELS_DIR_NAME: str = "models"
_SCALER_FILENAME: str = "final_scaler.pkl"

# ---------------------------------------------------------------------------
# Numeric constants
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLD: float = 0.5
_MIN_PRED_STD: float = 1e-6
_TCGA_PATIENT_ID_LEN: int = 12
_MAX_SURVIVAL_GRID_PANELS: int = 4

# ---------------------------------------------------------------------------
# Result-dict key constants
# ---------------------------------------------------------------------------

_KEY_Y_TRUE: str = "y_true"
_KEY_Y_PRED: str = "y_pred_prob"
_KEY_METRICS_EXT: str = "metrics_extended"
_KEY_METRICS_OPT: str = "metrics_optimal"
_KEY_OS_TIME: str = "os_time"
_KEY_OS_STATUS: str = "os_status"

_COL_RESPONSE: str = "response"

# Per-cohort OS column names (Liu uses uppercase; Hugo/Riaz use lowercase).
_COL_OS_TIME_LIU:   str = "OS_MONTHS"
_COL_OS_TIME:       str = "os_months"
_COL_OS_STATUS_LIU: str = "OS_STATUS"
_COL_OS_STATUS:     str = "os_status"
_COL_OS_STATUS_CLN: str = "os_status_clean"

# Lookup table: cohort -> (time_col, status_col) for survival analysis post-cleaning.
_COHORT_OS_COLS: dict[str, tuple[str, str]] = {
    _COHORT_LIU:  (_COL_OS_TIME_LIU, _COL_OS_STATUS_CLN),
    _COHORT_HUGO: (_COL_OS_TIME,     _COL_OS_STATUS),
    _COHORT_RIAZ: (_COL_OS_TIME,     _COL_OS_STATUS),
    _COHORT_TCGA: (_COL_OS_TIME_LIU, _COL_OS_STATUS_CLN),
}

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

_MODEL_TYPES: list[str] = ["lr", "rf", "xgb", "svm", "elasticnet"]

_MODEL_NAMES: dict[str, str] = {
    "lr": "Logistic Regression (L1-Penalised)",
    "rf": "Random Forest Classifier",
    "xgb": "XGBoost Gradient Boosting",
    "svm": "Support Vector Machine (SVM)",
    "elasticnet": "ElasticNet Logistic Regression",
}

_MODEL_SHORTS: dict[str, str] = {
    "lr": "LR",
    "rf": "RF",
    "xgb": "XGB",
    "svm": "SVM",
    "elasticnet": "ElasticNet",
}

# ---------------------------------------------------------------------------
# Cohort Data Bundle
# ---------------------------------------------------------------------------


class CohortBundle(NamedTuple):
    """Immutable bundle holding per-cohort data produced by Phases 1–3.

    Replaces the 12-item positional return tuple from `_run_expression_part1`,
    allowing callers to access fields by name rather than destructure by position.
    """

    sig_liu: pd.DataFrame
    y_liu: pd.Series
    sig_hugo: pd.DataFrame
    y_hugo: pd.Series
    sig_riaz: pd.DataFrame
    y_riaz: pd.Series
    clin_liu: pd.DataFrame
    clin_hugo: pd.DataFrame
    clin_riaz: pd.DataFrame
    sig_corrected: pd.DataFrame
    cohort_survival: dict[str, dict[str, np.ndarray]]
    common_genes: pd.Index


# ---------------------------------------------------------------------------
# Utility & Transformation Functions
# ---------------------------------------------------------------------------


def zscore_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise DataFrame columns individually (Z-score scaling).

    Constant or missing features receive std=1 to avoid division by zero.
    """
    means = df.mean(axis=0)
    stds = df.std(axis=0).replace(0, 1.0).fillna(1.0)
    return (df - means) / stds


def _drop_duplicate_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate columns, keeping the first occurrence."""
    return df.loc[:, ~df.columns.duplicated()]


def extract_driver_mutations(df_meta: pd.DataFrame) -> pd.DataFrame:
    """Extract BRAF, NRAS, and NF1 mutation indicators from clinical metadata."""
    df = df_meta.copy()
    for col in ("mut_BRAF", "mut_NRAS", "mut_NF1"):
        if col not in df.columns:
            df[col] = 0
    return df[["mut_BRAF", "mut_NRAS", "mut_NF1"]]


def get_survival_data(
    clin_df: pd.DataFrame,
    sig_index: pd.Index,
    os_time_col: str,
    os_status_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract aligned overall survival times and binary event statuses."""
    clin_aligned = clin_df.loc[sig_index].copy()
    os_time = pd.to_numeric(clin_aligned[os_time_col], errors="coerce").values
    os_status = pd.to_numeric(clin_aligned[os_status_col], errors="coerce").values
    return os_time, os_status


def clean_os_status(val: object) -> float:
    """Convert heterogeneous survival status values to binary 1/0 or NaN."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        if val in (1, 1.0):
            return 1.0
        if val in (0, 0.0):
            return 0.0
    if isinstance(val, str):
        val_upper = val.upper()
        if "DECEASED" in val_upper or "1" in val_upper:
            return 1.0
        if "LIVING" in val_upper or "0" in val_upper:
            return 0.0
    return np.nan


# ---------------------------------------------------------------------------
# Report Generation Subroutines
# ---------------------------------------------------------------------------


def _build_report_overview() -> list[str]:
    """Generate YAML frontmatter, header, and methodology overview lines."""
    frontmatter = generate_obsidian_frontmatter(
        title="Model Evaluation Report: Leave-One-Cohort-Out (LOCO) Cross-Validation",
        aliases=["LOCO Model Evaluation Report", "Q1 Response Predictor Evaluation"],
        tags=["q1", "model-evaluation", "loco-cv", "immunotherapy-response", "calibration"],
        extra_css_classes=["table-center", "row-alt"],
    )
    return [
        frontmatter + "\n\n",
        "# Model Evaluation Report: Leave-One-Cohort-Out (LOCO) Cross-Validation\n\n",
        "> [!summary] What, Why & Key Questions\n",
        "> **What**: Tested 5 machine learning models (LR, RF, XGBoost, SVM, ElasticNet) "
        "predicting anti-PD-1 response.\n",
        "> **Why**: Testing on held-out hospital trial cohorts (LOCO) evaluates real-world "
        "generalisability.\n",
        "> **Key Questions**: Which model generalises best? Does multimodal integration improve "
        "accuracy?\n\n",
        "## Overview & Methodology\n\n",
        "1. **Evaluation Framework (LOCO)**: Trained on 2 trial cohorts, tested on 1 unseen "
        "cohort.\n",
        "2. **Test Cohorts**: Liu 2019, Hugo 2016, Riaz 2017 with cohort-independent Z-scoring.\n",
        "3. **Features**: 8 transcriptomic signatures plus somatic driver mutations "
        "(`mut_BRAF`, `mut_NRAS`, `mut_NF1`).\n\n",
    ]


def _get_or_compute_metrics(res: dict[str, object]) -> dict[str, object]:
    """Return cached extended metrics, or compute them from raw predictions."""
    return res.get(_KEY_METRICS_EXT) or calculate_extended_metrics(
        res[_KEY_Y_TRUE], res[_KEY_Y_PRED]
    )


def _format_auc_row(
    mkey: str,
    all_cohorts: list[str],
    loco: dict[str, dict[str, object]],
) -> tuple[str, float]:
    """Format single table row for model AUC cross-cohort summary."""
    row_aucs = []
    for cohort in all_cohorts:
        res = loco.get(cohort, {})
        m = _get_or_compute_metrics(res)
        row_aucs.append(m["auc"] if m else float("nan"))
    mean_auc = float(np.nanmean(row_aucs))
    auc_cells = " | ".join(f"{v:.3f}" if not np.isnan(v) else "N/A" for v in row_aucs)
    row_str = f"| **{_MODEL_SHORTS[mkey]}** | {auc_cells} | **{mean_auc:.3f}** |\n"
    return row_str, mean_auc


def _build_cross_model_auc_table(
    all_loco_results: dict[str, dict[str, dict[str, object]]],
) -> list[str]:
    """Build summary AUC markdown table comparing performance across models."""
    all_cohorts = sorted({c for res in all_loco_results.values() for c in res})
    lines = [
        "## Cross-Model AUC Summary\n\n",
        "| Model | " + " | ".join(all_cohorts) + " | **Mean AUC** |\n",
        "|:---|" + ":".join(["---:"] * len(all_cohorts)) + "---:|\n",
    ]

    summary_rows: list[tuple[str, float]] = []
    for mkey in _MODEL_TYPES:
        if mkey not in all_loco_results:
            continue
        row_str, mean_auc = _format_auc_row(mkey, all_cohorts, all_loco_results[mkey])
        lines.append(row_str)
        summary_rows.append((mkey, mean_auc))

    if not summary_rows:
        return lines

    best_key, best_mean = max(summary_rows, key=lambda x: x[1])
    lines.append(
        f"\n> [!INSIGHT] Best Generalising Model: {_MODEL_SHORTS[best_key]}\n"
        f"> **{_MODEL_NAMES[best_key]}** achieves the highest mean cross-cohort AUC of "
        f"**{best_mean:.3f}** across held-out test cohorts.\n\n"
    )
    return lines


def _format_single_model_row(
    cohort: str,
    res: dict[str, object],
) -> str:
    """Format metric table row for single model on one test cohort."""
    m = _get_or_compute_metrics(res)
    n_samples = len(res[_KEY_Y_TRUE])
    cidx = f"{m['cindex']:.3f}" if "cindex" in m and not np.isnan(m["cindex"]) else "N/A"
    ece = f"{m['ece']:.3f}" if "ece" in m and not np.isnan(m["ece"]) else "N/A"
    brier = (
        f"{m['brier_score']:.3f}"
        if "brier_score" in m and not np.isnan(m["brier_score"])
        else "N/A"
    )
    return (
        f"| {cohort} | {n_samples} | {m['auc']:.3f} | {ece} | {brier} | "
        f"{m['accuracy']:.3f} | {m['sensitivity']:.3f} | {m['specificity']:.3f} | "
        f"{m['precision']:.3f} | {m['f1']:.3f} | {cidx} |\n"
    )


def _build_single_model_report(
    model_key: str,
    loco_results: dict[str, dict[str, object]],
) -> list[str]:
    """Build metric tables and diagnostic figure markdown for one model architecture."""
    model_label = _MODEL_NAMES.get(model_key, model_key.upper())
    lines = [
        f"## {model_label}\n\n",
        "### Performance Metrics (Default Threshold = 0.5)\n\n",
        "| Test Cohort | N | AUC | ECE | Brier | Accuracy | Sens | Spec | Prec | F1 | C-Index |\n",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n",
    ]

    for cohort, res in sorted(loco_results.items()):
        lines.append(_format_single_model_row(cohort, res))

    lines.extend([
        "\n#### Visualisations & Diagnostics\n\n",
        f"![Confusion Matrices](../../plots/models/confusion_matrices_{model_key}.png)\n\n",
        f"![Calibration Curves](../../plots/models/calibration_curves_{model_key}.png)\n\n",
        f"![ROC Curves](../../plots/models/roc_curves_{model_key}.png)\n\n",
        f"![PR Curves](../../plots/models/pr_curves_{model_key}.png)\n\n",
    ])
    return lines


def _build_multimodal_report(
    combined_loco_results: dict[str, dict[str, dict[str, object]]],
) -> list[str]:
    """Build multimodal integration section with mutation features."""
    lines = ["## Multimodal Integration: Immune Signatures + Driver Mutations\n\n"]
    for model_key, loco_results in combined_loco_results.items():
        model_label = _MODEL_NAMES.get(model_key, model_key.upper())
        lines.append(f"### {model_label} (Multimodal)\n\n")
        lines.append(
            "| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1 |\n"
            "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
        )
        for cohort, res in sorted(loco_results.items()):
            m = calculate_extended_metrics(res[_KEY_Y_TRUE], res[_KEY_Y_PRED])
            lines.append(
                f"| {cohort} | {len(res[_KEY_Y_TRUE])} | {m['auc']:.3f} | {m['accuracy']:.3f} | "
                f"{m['sensitivity']:.3f} | {m['specificity']:.3f} | {m['precision']:.3f} | "
                f"{m['f1']:.3f} |\n"
            )
        lines.append(
            f"\n![Multimodal ROC](../../plots/models/roc_curves_combined_{model_key}.png)\n\n"
        )
    return lines


def _build_survival_report(survival_results: list[dict[str, object]]) -> list[str]:
    """Build overall survival stratification report table and figures."""
    lines = [
        "## Downstream Overall Survival Stratification\n\n",
        "| Cohort | Model Selected | Best LOCO AUC | Log-Rank p-value | Significant? |\n",
        "|:---|:---:|:---:|:---:|:---:|\n",
    ]
    for sr in survival_results:
        p = sr["p_value"]
        p_str = f"{p:.3e}" if p is not None else "N/A"
        sig_str = "N/A" if p is None else ("Yes" if p < 0.05 else "No")
        auc_str = f"{sr['auc']:.3f}" if isinstance(sr["auc"], (int, float)) else str(sr["auc"])
        lines.append(
            f"| {sr['cohort']} | {str(sr['model']).upper()} | {auc_str} | {p_str} | {sig_str} |\n"
        )

    lines.extend([
        "\n### Kaplan-Meier Survival Curves\n\n",
        "![Kaplan-Meier 2x2 Grid](../../plots/models/survival_2x2_grid.png)\n\n",
    ])
    return lines


def generate_model_evaluation_report(
    all_loco_results: dict[str, dict[str, dict[str, object]]],
    output_dir: Path,
    survival_results: list[dict[str, object]] | None = None,
    combined_loco_results: dict[str, dict[str, dict[str, object]]] | None = None,
) -> Path:
    """Generate comprehensive markdown report documenting model evaluation metrics."""
    report_lines = []
    report_lines.extend(_build_report_overview())
    report_lines.extend(_build_cross_model_auc_table(all_loco_results))

    for model_key, loco_results in all_loco_results.items():
        report_lines.extend(_build_single_model_report(model_key, loco_results))

    if combined_loco_results:
        report_lines.extend(_build_multimodal_report(combined_loco_results))

    if survival_results:
        report_lines.extend(_build_survival_report(survival_results))

    report_path = output_dir / _REPORT_SUBDIR / _REPORT_FILENAME
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(report_lines), encoding="utf-8")

    print(f"\n[SUCCESS] Model evaluation report saved to: {display_path(report_path)}")
    return report_path


# ---------------------------------------------------------------------------
# Main Execution Phases
# ---------------------------------------------------------------------------


def _phase1_load_and_align_cohorts(
    data_dir: Path,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
]:
    """Phase 1: Load raw expression and clinical tables and align common genes."""
    print("=" * 50 + "\nPhase 1: Loading Melanoma IO Cohorts...\n" + "=" * 50)
    expr_liu, clin_liu = load_liu_2019(data_dir)
    expr_hugo, clin_hugo = load_hugo_2016(data_dir)
    expr_riaz, clin_riaz = load_riaz_2017(data_dir)

    common_genes = (
        expr_liu.columns.intersection(expr_hugo.columns).intersection(expr_riaz.columns)
    )
    print(f"Common genes across all 3 cohorts: {len(common_genes):,}")

    return (
        expr_liu[common_genes], clin_liu,
        expr_hugo[common_genes], clin_hugo,
        expr_riaz[common_genes], clin_riaz,
    )


def _phase2_compute_signatures(
    expr_liu: pd.DataFrame, clin_liu: pd.DataFrame,
    expr_hugo: pd.DataFrame, clin_hugo: pd.DataFrame,
    expr_riaz: pd.DataFrame, clin_riaz: pd.DataFrame,
) -> tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
]:
    """Phase 2: Extract transcriptomic signatures and align valid response labels."""
    print("\n" + "=" * 50 + "\nPhase 2: Computing Immune Signatures...\n" + "=" * 50)
    sig_liu = extract_all_signatures(expr_liu)
    sig_hugo = extract_all_signatures(expr_hugo)
    sig_riaz = extract_all_signatures(expr_riaz)

    idx_liu = clin_liu.loc[sig_liu.index, _COL_RESPONSE].dropna().index
    idx_hugo = clin_hugo.loc[sig_hugo.index, _COL_RESPONSE].dropna().index
    idx_riaz = clin_riaz.loc[sig_riaz.index, _COL_RESPONSE].dropna().index

    return (
        sig_liu.loc[idx_liu], clin_liu.loc[idx_liu, _COL_RESPONSE],
        sig_hugo.loc[idx_hugo], clin_hugo.loc[idx_hugo, _COL_RESPONSE],
        sig_riaz.loc[idx_riaz], clin_riaz.loc[idx_riaz, _COL_RESPONSE],
    )


def _extract_cohort_survival_data(
    clin_liu: pd.DataFrame, idx_liu: pd.Index,
    clin_hugo: pd.DataFrame, idx_hugo: pd.Index,
    clin_riaz: pd.DataFrame, idx_riaz: pd.Index,
) -> dict[str, dict[str, np.ndarray]]:
    """Extract per-cohort OS time and status arrays aligned to signature indices."""
    t_liu, s_liu = get_survival_data(clin_liu, idx_liu, _COL_OS_TIME_LIU, _COL_OS_STATUS_LIU)
    t_hugo, s_hugo = get_survival_data(clin_hugo, idx_hugo, _COL_OS_TIME, _COL_OS_STATUS)
    t_riaz, s_riaz = get_survival_data(clin_riaz, idx_riaz, _COL_OS_TIME, _COL_OS_STATUS)
    return {
        _COHORT_LIU:  {_KEY_OS_TIME: t_liu,  _KEY_OS_STATUS: s_liu},
        _COHORT_HUGO: {_KEY_OS_TIME: t_hugo, _KEY_OS_STATUS: s_hugo},
        _COHORT_RIAZ: {_KEY_OS_TIME: t_riaz, _KEY_OS_STATUS: s_riaz},
    }


def _phase3_standardise_and_align_survival(
    sig_liu: pd.DataFrame, clin_liu: pd.DataFrame,
    sig_hugo: pd.DataFrame, clin_hugo: pd.DataFrame,
    sig_riaz: pd.DataFrame, clin_riaz: pd.DataFrame,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
    dict[str, dict[str, np.ndarray]],
]:
    """Phase 3: Standardise signatures per cohort and extract survival arrays."""
    print(
        "\n" + "=" * 50 + "\nPhase 3: Cohort Standardisation (Z-score)...\n" + "=" * 50
    )
    sig_c_liu, sig_c_hugo, sig_c_riaz = zscore_df(sig_liu), zscore_df(sig_hugo), zscore_df(sig_riaz)
    sig_corrected = pd.concat([sig_c_liu, sig_c_hugo, sig_c_riaz], axis=0)
    cohort_survival = _extract_cohort_survival_data(
        clin_liu, sig_c_liu.index, clin_hugo, sig_c_hugo.index, clin_riaz, sig_c_riaz.index,
    )
    return sig_c_liu, sig_c_hugo, sig_c_riaz, sig_corrected, cohort_survival


def _plot_single_model_diagnostics(
    model_type: str,
    loco_results: dict[str, dict[str, object]],
) -> None:
    """Generate diagnostic plots for single model evaluation."""
    m_str = model_type.upper()
    plot_roc_curves(
        loco_results, m_str, MODEL_PLOTS_DIR / f"roc_curves_{model_type}.png"
    )
    plot_pr_curves(
        loco_results, m_str, MODEL_PLOTS_DIR / f"pr_curves_{model_type}.png"
    )
    plot_calibration_curves(
        loco_results, m_str, MODEL_PLOTS_DIR / f"calibration_curves_{model_type}.png"
    )
    plot_confusion_matrices(
        loco_results, m_str, MODEL_PLOTS_DIR / f"confusion_matrices_{model_type}.png"
    )


def _evaluate_single_model_loco(
    model_type: str,
    cohort_dfs: dict[str, tuple[pd.DataFrame, pd.Series]],
    signature_cols: list[str],
    cohort_survival: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, object]]:
    """Evaluate one model architecture across LOCO CV folds."""
    print(f"\nTraining and testing model: {model_type.upper()}")
    loco_results = run_loco_cv(cohort_dfs, signature_cols, model_type=model_type)

    for cohort, res in loco_results.items():
        m_def = calculate_extended_metrics(
            res[_KEY_Y_TRUE], res[_KEY_Y_PRED], threshold=_DEFAULT_THRESHOLD
        )
        opt_t = find_optimal_threshold(res[_KEY_Y_TRUE], res[_KEY_Y_PRED])
        m_opt = calculate_extended_metrics(res[_KEY_Y_TRUE], res[_KEY_Y_PRED], threshold=opt_t)
        surv = cohort_survival[cohort]
        cidx = calculate_cindex(res[_KEY_Y_PRED], surv[_KEY_OS_TIME], surv[_KEY_OS_STATUS])
        m_def["cindex"], m_opt["cindex"] = cidx, cidx
        res[_KEY_METRICS_EXT], res[_KEY_METRICS_OPT] = m_def, m_opt

    _plot_single_model_diagnostics(model_type, loco_results)
    return loco_results


def _phase4_run_loco_eval(
    cohort_dfs: dict[str, tuple[pd.DataFrame, pd.Series]],
    signature_cols: list[str],
    cohort_survival: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, dict[str, object]]]:
    """Phase 4: Run LOCO cross-validation for all model architectures."""
    print("\n" + "=" * 50 + "\nPhase 4: LOCO Cross-Validation...\n" + "=" * 50)
    all_results = {}
    for model_type in _MODEL_TYPES:
        all_results[model_type] = _evaluate_single_model_loco(
            model_type, cohort_dfs, signature_cols, cohort_survival
        )
    plot_loco_heatmap_from_results(all_results, MODEL_PLOTS_DIR / "loco_performance_heatmap.png")
    return all_results


def _select_best_model_for_cohort(
    cohort_name: str,
    all_loco_results: dict[str, dict[str, dict[str, object]]],
) -> tuple[str | None, float]:
    """Identify highest-performing non-degenerate model for a given cohort."""
    best_key, best_auc = None, -1.0
    for mkey, mresults in all_loco_results.items():
        auc_val = mresults[cohort_name][_KEY_METRICS_EXT]["auc"]
        if not np.isnan(auc_val) and auc_val > best_auc:
            if np.std(mresults[cohort_name][_KEY_Y_PRED]) > _MIN_PRED_STD:
                best_auc = auc_val
                best_key = mkey
    return best_key, best_auc


def _run_single_cohort_survival(
    cohort_name: str,
    best_model_key: str,
    best_auc: float,
    all_loco_results: dict[str, dict[str, dict[str, object]]],
    clin_map: dict[str, pd.DataFrame],
    sig_map: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """Execute log-rank survival stratification for one clinical cohort."""
    print(f"{cohort_name}: Using {best_model_key.upper()} (best LOCO AUC = {best_auc:.3f}).")
    cohort_pred = all_loco_results[best_model_key][cohort_name][_KEY_Y_PRED]
    clin_valid = clin_map[cohort_name].loc[sig_map[cohort_name].index].copy()

    time_col, status_col = _COHORT_OS_COLS[cohort_name]
    if cohort_name == _COHORT_LIU:
        clin_valid[_COL_OS_STATUS_CLN] = clin_valid[_COL_OS_STATUS_LIU].apply(clean_os_status)

    plot_file = f"survival_{cohort_name.split()[0].lower()}_{best_model_key}.png"
    p_val = run_survival_analysis(
        clin_valid, cohort_pred, time_col=time_col, status_col=status_col,
        save_path=MODEL_PLOTS_DIR / plot_file,
    )
    return {
        "cohort": cohort_name, "model": best_model_key, "auc": best_auc, "p_value": p_val,
        "plot_filename": plot_file, "df_clin": clin_valid, _KEY_Y_PRED: cohort_pred,
        "time_col": time_col, "status_col": status_col,
    }


def _phase5_run_survival_analysis(
    all_loco_results: dict[str, dict[str, dict[str, object]]],
    clin_map: dict[str, pd.DataFrame],
    sig_map: dict[str, pd.DataFrame],
) -> list[dict[str, object]]:
    """Phase 5: Perform log-rank survival analysis across all trial cohorts."""
    print("\n" + "=" * 50 + "\nPhase 5: Survival Analysis (Log-rank test)...\n" + "=" * 50)
    survival_results: list[dict[str, object]] = []

    for cohort_name in (_COHORT_HUGO, _COHORT_LIU, _COHORT_RIAZ):
        best_key, best_auc = _select_best_model_for_cohort(cohort_name, all_loco_results)
        if best_key is None:
            print(f"{cohort_name}: All models constant, skipping survival.")
            continue
        res = _run_single_cohort_survival(
            cohort_name, best_key, best_auc, all_loco_results, clin_map, sig_map
        )
        survival_results.append(res)
    return survival_results


def _scale_train_and_tcga(
    sig_corrected: pd.DataFrame,
    sig_tcga_scaled: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit StandardScaler on pooled train data and transform TCGA signatures."""
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(sig_corrected),
        columns=sig_corrected.columns, index=sig_corrected.index,
    )
    X_tcga = pd.DataFrame(
        scaler.transform(sig_tcga_scaled),
        columns=sig_tcga_scaled.columns, index=sig_tcga_scaled.index,
    )
    return X_train, X_tcga


def _prepare_tcga_matrices(
    common_genes: pd.Index,
    sig_corrected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, index-normalise, and scale TCGA expression and clinical data."""
    tcga_dir = DATA_DIR / "processed" / _TCGA_DATASET_DIR
    df_expr = pd.read_csv(tcga_dir / _EXPR_FILENAME, index_col=0).reindex(
        columns=common_genes, fill_value=0
    )
    df_clin = pd.read_csv(tcga_dir / _CLIN_FILENAME, index_col=0)
    sig_tcga = extract_all_signatures(df_expr)
    sig_tcga.index = sig_tcga.index.str.upper().str[:_TCGA_PATIENT_ID_LEN]
    df_clin.index = df_clin.index.str.upper().str[:_TCGA_PATIENT_ID_LEN]
    sig_grp = sig_tcga.groupby(sig_tcga.index).first()
    clin_grp = df_clin.groupby(df_clin.index).first()
    common_tcga = sig_grp.index.intersection(clin_grp.index)
    sig_tcga_scaled = zscore_df(sig_grp.loc[common_tcga])
    df_tcga_clin_clean = clin_grp.loc[common_tcga].copy()
    X_train_scaled, X_tcga_scaled = _scale_train_and_tcga(sig_corrected, sig_tcga_scaled)
    return X_train_scaled, X_tcga_scaled, df_tcga_clin_clean


def _phase5b_tcga_validation(
    common_genes: pd.Index,
    sig_corrected: pd.DataFrame,
    y_train_full: pd.Series,
) -> dict[str, object] | None:
    """Validate model survival stratification on TCGA-SKCM benchmark cohort using get_model()."""
    try:
        X_train_scaled, X_tcga_scaled, df_tcga_clin_clean = _prepare_tcga_matrices(
            common_genes, sig_corrected
        )
        final_model = get_model("lr", X_train_scaled, y_train_full)
        tcga_pred = final_model.predict_proba(X_tcga_scaled)[:, 1]

        df_tcga_clin_clean[_COL_OS_STATUS_CLN] = (
            df_tcga_clin_clean[_COL_OS_STATUS_LIU].apply(clean_os_status)
        )
        p_tcga = run_survival_analysis(
            df_tcga_clin_clean, tcga_pred,
            time_col=_COL_OS_TIME_LIU, status_col=_COL_OS_STATUS_CLN,
            save_path=MODEL_PLOTS_DIR / "survival_tcga_lr.png"
        )
        return {
            "cohort": _COHORT_TCGA, "model": "lr", "auc": "N/A (external)", "p_value": p_tcga,
            "plot_filename": "survival_tcga_lr.png", "df_clin": df_tcga_clin_clean,
            _KEY_Y_PRED: tcga_pred,
            "time_col": _COL_OS_TIME_LIU, "status_col": _COL_OS_STATUS_CLN,
        }
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Skipping TCGA survival validation: {exc}")
        return None


def _render_survival_grid(survival_results: list[dict[str, object]]) -> None:
    """Render 2x2 Kaplan-Meier overall survival grid if at least 2 results exist."""
    if len(survival_results) < 2:
        return
    print("\nGenerating 2x2 Kaplan-Meier survival grid across cohorts...")
    grid_data = [
        {
            "cohort": sr["cohort"], "df_clin": sr["df_clin"],
            _KEY_Y_PRED: sr[_KEY_Y_PRED],
            "time_col": sr["time_col"], "status_col": sr["status_col"],
            "model_name": sr["model"],
        }
        for sr in survival_results if "df_clin" in sr and sr.get(_KEY_Y_PRED) is not None
    ]
    plot_survival_2x2_grid(
        grid_data[:_MAX_SURVIVAL_GRID_PANELS],
        save_path=MODEL_PLOTS_DIR / "survival_2x2_grid.png",
    )
    print(f"  2x2 KM grid saved to: {display_path(MODEL_PLOTS_DIR / 'survival_2x2_grid.png')}")


def _phase6_run_multimodal_loco(
    comb_liu: pd.DataFrame, y_liu: pd.Series,
    comb_hugo: pd.DataFrame, y_hugo: pd.Series,
    comb_riaz: pd.DataFrame, y_riaz: pd.Series,
    comb_features: list[str],
) -> dict[str, dict[str, dict[str, object]]]:
    """Phase 6: Benchmark multimodal models integrating expression and driver mutations."""
    print("\n" + "=" * 50 + "\nPhase 6: Combined Features Model (Multimodal)...\n" + "=" * 50)
    cohort_dfs_comb = {
        _COHORT_LIU: (comb_liu, y_liu),
        _COHORT_HUGO: (comb_hugo, y_hugo),
        _COHORT_RIAZ: (comb_riaz, y_riaz),
    }
    all_combined_results = {}
    for model_type in _MODEL_TYPES:
        loco = run_loco_cv(cohort_dfs_comb, comb_features, model_type=model_type)
        all_combined_results[model_type] = loco
        plot_roc_curves(
            loco, f"COMBINED_{model_type.upper()}",
            MODEL_PLOTS_DIR / f"roc_curves_combined_{model_type}.png"
        )
        plot_pr_curves(
            loco, f"COMBINED_{model_type.upper()}",
            MODEL_PLOTS_DIR / f"pr_curves_combined_{model_type}.png"
        )
    return all_combined_results


def _phase7_save_tuned_models(
    sig_corrected: pd.DataFrame,
    y_train_final: pd.Series,
) -> None:
    """Phase 7: Tune and save final pipeline models via get_model() for Q5 integration."""
    print("\n" + "=" * 50 + "\nPhase 7: Saving Best Models for Q5 Integration...\n" + "=" * 50)
    models_dir = PROJECT_ROOT / _MODELS_DIR_NAME
    models_dir.mkdir(exist_ok=True)

    scaler_final = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler_final.fit_transform(sig_corrected),
        columns=sig_corrected.columns, index=sig_corrected.index,
    )

    for model_type in _MODEL_TYPES:
        print(f"  Tuning {model_type.upper()} via GridSearchCV on pooled data...")
        tuned_model = get_model(model_type, X_scaled, y_train_final)
        pkl_path = models_dir / f"final_{model_type}_model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(tuned_model, f)
        print(f"    Saved to {display_path(pkl_path)}")

    with open(models_dir / _SCALER_FILENAME, "wb") as f:
        pickle.dump(scaler_final, f)
    print(f"  Saved scaler and {len(_MODEL_TYPES)} models to {display_path(models_dir)}")


def _run_expression_part1(data_dir: Path) -> CohortBundle:
    """Execute expression loading, signature computation, and cohort Z-scoring."""
    expr_l, clin_l, expr_h, clin_h, expr_r, clin_r = _phase1_load_and_align_cohorts(data_dir)
    sig_l, y_l, sig_h, y_h, sig_r, y_r = _phase2_compute_signatures(
        expr_l, clin_l, expr_h, clin_h, expr_r, clin_r
    )
    sig_c_l, sig_c_h, sig_c_r, sig_corrected, cohort_survival = (
        _phase3_standardise_and_align_survival(sig_l, clin_l, sig_h, clin_h, sig_r, clin_r)
    )
    return CohortBundle(
        sig_liu=sig_c_l, y_liu=y_l,
        sig_hugo=sig_c_h, y_hugo=y_h,
        sig_riaz=sig_c_r, y_riaz=y_r,
        clin_liu=clin_l, clin_hugo=clin_h, clin_riaz=clin_r,
        sig_corrected=sig_corrected,
        cohort_survival=cohort_survival,
        common_genes=expr_l.columns,
    )


def _run_loco_and_survival(
    b: CohortBundle,
) -> tuple[dict[str, dict[str, dict[str, object]]], list[dict[str, object]], pd.Series]:
    """Execute Phase 4 LOCO evaluation and Phase 5 survival analysis."""
    c_dfs = {
        _COHORT_LIU:  (b.sig_liu,  b.y_liu),
        _COHORT_HUGO: (b.sig_hugo, b.y_hugo),
        _COHORT_RIAZ: (b.sig_riaz, b.y_riaz),
    }
    loco_res = _phase4_run_loco_eval(c_dfs, b.sig_corrected.columns.tolist(), b.cohort_survival)

    c_map = {_COHORT_HUGO: b.clin_hugo, _COHORT_LIU: b.clin_liu, _COHORT_RIAZ: b.clin_riaz}
    s_map = {_COHORT_HUGO: b.sig_hugo,  _COHORT_LIU: b.sig_liu,  _COHORT_RIAZ: b.sig_riaz}
    surv_res = _phase5_run_survival_analysis(loco_res, c_map, s_map)

    y_full = pd.concat([b.y_liu, b.y_hugo, b.y_riaz], axis=0)
    tcga = _phase5b_tcga_validation(b.common_genes, b.sig_corrected, y_full)
    if tcga:
        surv_res.append(tcga)
    _render_survival_grid(surv_res)
    return loco_res, surv_res, y_full


def _run_expression_pipeline(
    data_dir: Path,
) -> tuple[
    dict[str, dict[str, dict[str, object]]],
    list[dict[str, object]],
    pd.DataFrame,
    pd.Series,
    tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
]:
    """Execute expression loading, signature extraction, LOCO evaluation, and survival."""
    b = _run_expression_part1(data_dir)
    loco_res, surv_res, y_full = _run_loco_and_survival(b)
    sigs = (b.sig_liu, b.y_liu, b.sig_hugo, b.y_hugo, b.sig_riaz, b.y_riaz)
    clins = (b.clin_liu, b.clin_hugo, b.clin_riaz)
    return loco_res, surv_res, b.sig_corrected, y_full, sigs, clins


def _run_multimodal_and_save(
    sigs: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    sig_corrected: pd.DataFrame,
    y_train_full: pd.Series,
    clin_liu: pd.DataFrame,
    clin_hugo: pd.DataFrame,
    clin_riaz: pd.DataFrame,
) -> dict[str, dict[str, dict[str, object]]]:
    """Execute multimodal LOCO cross-validation and save tuned final models."""
    sig_c_liu, y_liu, sig_c_hugo, y_hugo, sig_c_riaz, y_riaz = sigs

    mut_hugo, mut_liu, mut_riaz = (
        extract_driver_mutations(clin_hugo.loc[sig_c_hugo.index]),
        extract_driver_mutations(clin_liu.loc[sig_c_liu.index]),
        extract_driver_mutations(clin_riaz.loc[sig_c_riaz.index]),
    )
    comb_liu  = _drop_duplicate_cols(pd.concat([sig_c_liu, mut_liu], axis=1))
    comb_hugo = _drop_duplicate_cols(pd.concat([sig_c_hugo, mut_hugo], axis=1))
    comb_riaz = _drop_duplicate_cols(pd.concat([sig_c_riaz, mut_riaz], axis=1))
    comb_features = list(
        dict.fromkeys(sig_corrected.columns.tolist() + ["mut_BRAF", "mut_NRAS", "mut_NF1"])
    )

    all_combined_results = _phase6_run_multimodal_loco(
        comb_liu, y_liu, comb_hugo, y_hugo, comb_riaz, y_riaz, comb_features
    )
    _phase7_save_tuned_models(sig_corrected, y_train_full)
    return all_combined_results


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for master pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Master Preprocessing, LOCO Cross-Validation & Model Evaluation Pipeline."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download raw datasets from cBioPortal / sources before running pipeline.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean and preprocess raw datasets before running pipeline.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge cleaned datasets into unified cohorts before running pipeline.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full end-to-end workflow: download -> clean -> merge -> LOCO evaluation.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Run specified preprocessing steps (download/clean/merge) but skip model evaluation.",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    """Run master LOCO cross-validation and evaluation pipeline."""
    parsed_args = parse_args(args)

    run_download_step = parsed_args.download or parsed_args.all
    run_clean_step = parsed_args.clean or parsed_args.all
    run_merge_step = parsed_args.merge or parsed_args.all

    if run_download_step:
        print("\n" + "=" * 50)
        print("PIPELINE STEP: Downloading Raw Datasets")
        print("=" * 50)
        download_module = importlib.import_module(
            "scripts.pillar-1-cohort-preprocessing.download_data"
        )
        download_module.main()

    if run_clean_step:
        print("\n" + "=" * 50)
        print("PIPELINE STEP: Cleaning & Preprocessing Datasets")
        print("=" * 50)
        clean_module = importlib.import_module(
            "scripts.pillar-1-cohort-preprocessing.clean_data"
        )
        clean_module.main()

    if run_merge_step:
        print("\n" + "=" * 50)
        print("PIPELINE STEP: Merging Cohort Datasets")
        print("=" * 50)
        merge_module = importlib.import_module(
            "scripts.pillar-1-cohort-preprocessing.merge_datasets"
        )
        merge_module.main()

    if parsed_args.skip_eval:
        print("\n" + "=" * 50)
        print("Preprocessing steps completed. Skipping model evaluation (--skip-eval requested).")
        print("=" * 50)
        return

    print("\n" + "=" * 50)
    print("PIPELINE STEP: LOCO CV, Survival Analysis & Model Evaluation")
    print("=" * 50)

    (
        all_loco_results, survival_results, sig_corrected, y_train_full,
        sigs, clins,
    ) = _run_expression_pipeline(DATA_DIR)

    all_combined_results = _run_multimodal_and_save(
        sigs, sig_corrected, y_train_full, *clins
    )

    generate_model_evaluation_report(
        all_loco_results, REPORTS_DIR,
        survival_results=survival_results, combined_loco_results=all_combined_results,
    )
    print("\n" + "=" * 50 + "\nDone! All analysis runs completed successfully.\n" + "=" * 50)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {display_path(LOG_PATH)}")
            main()
