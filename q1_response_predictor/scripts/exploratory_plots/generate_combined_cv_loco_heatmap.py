"""Generates 1x2 Side-by-Side Heatmap Grid: 5-Fold CV Comparison + LOCO Performance (Option 1).

Left panel:  5-Fold Stratified CV AUROC (Mean ± SD) — Curated Signatures vs. SelectKBest (k=20, 100, 200).
Right panel: Leave-One-Cohort-Out (LOCO) AUROC across held-out cohorts with significance asterisks (* p < 0.05 vs chance),
             and Mean ± SD for summary row/col.

Model names on the y-axis are wrapped onto two lines for readability.
Transparent background PNGs exported for presentation decks.

Parallelisation notes
---------------------
- 5-Fold CV panel : 20 (model × feature-set) tasks are submitted concurrently via
  ``ProcessPoolExecutor``.  Each task is a full 5-fold CV run for one
  (model_type, feature_set) combination, returning the five per-fold AUROCs.
- LOCO panel      : 5 model LOCO runs are submitted concurrently via the same pool.
- Bootstrap SD    : All ``n_boot`` index arrays are generated in a single NumPy call
  (replacing the original Python-level ``for _ in range(n_boot)`` loop),
  substantially reducing per-resample overhead.

Windows / pickling constraint
------------------------------
``ProcessPoolExecutor`` on Windows uses *spawn* (not fork).  Worker callables must
therefore be defined at **module scope** — nested functions and lambdas are not
picklable and will raise ``AttributeError`` at runtime.  ``_cv_worker`` and
``_loco_worker`` satisfy this requirement.
"""

# Set single-threading for OpenMP / BLAS inside workers to prevent Windows process join deadlocks
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

SUBPROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = SUBPROJECT_ROOT.parent

if str(SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data_loaders import load_all_active_cohorts
from src.models import get_model, run_loco_cv
from src.signatures import extract_all_signatures, zscore_df
from src.styles import set_presentation_style
from src.utils.logging import setup_logging
from src.utils.paths import DATA_DIR, PLOTS_DIR, get_subproject_log_dir, rel_path
from src.utils.plotting import save_fig

set_presentation_style()

# ---------------------------------------------------------------------------
# Module-level Constants
# ---------------------------------------------------------------------------
MODEL_ORDER: List[str] = ["xgb", "rf", "svm", "elasticnet", "lr"]
MODEL_LABELS: Dict[str, str] = {
    "xgb": "XGBoost",
    "rf": "Random Forest",
    "svm": "Support Vector Machine",
    "elasticnet": "ElasticNet",
    "lr": "Logistic Regression",
}
MODEL_LABELS_WRAPPED: Dict[str, str] = {
    "xgb": "XGBoost",
    "rf": "Random\nForest",
    "svm": "Support\nVector\nMachine",
    "elasticnet": "ElasticNet",
    "lr": "Logistic\nRegression",
}

K_VALUES: List[int] = [20, 100, 200]
TOP_VAR_PREFILTER: int = 1000
FEATURE_COLS_SIGS: List[str] = ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "IMPRES", "PD_L1"]
FEATURE_COL_LABELS: List[str] = ["Curated\nSignatures"] + [f"SelectKBest\n(k={k})" for k in K_VALUES]

N_FOLDS: int = 5
RANDOM_STATE: int = 42
# Cap at 8 to avoid over-subscribing on hyperthreaded machines; each worker itself
# uses n_jobs=-1 inside sklearn, so it already spawns threads internally.
N_WORKERS: int = min(os.cpu_count() or 4, 8)
N_BOOTSTRAP: int = 1000  # Resamples for LOCO bootstrap SD estimation

HEATMAP_VMIN: float = 0.30
HEATMAP_VMAX: float = 0.70
HEATMAP_CMAP: str = "YlGnBu"
HEATMAP_ANNOT_SIZE: int = 11
HEATMAP_LABEL_SIZE: int = 11

OUTPUT_PLOT_DIR: Path = PLOTS_DIR / "models"
OUTPUT_PATH: Path = OUTPUT_PLOT_DIR / "cv_loco_1x2_heatmap.png"
OUTPUT_TRANS_PATH: Path = OUTPUT_PLOT_DIR / "cv_loco_1x2_heatmap_transparent.png"
LOG_DIR: Path = get_subproject_log_dir(Path(__file__))
LOG_PATH: Path = LOG_DIR / "generate_1x2_cv_loco_heatmap.log"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def _load_pooled_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Loads and pools all active trial cohorts, intersecting genes to avoid NaNs."""
    config_path = SUBPROJECT_ROOT / "config" / "datasets.yaml"
    expr_dict, clin_dict, _, trial_names = load_all_active_cohorts(
        config_path, data_dir, merge_only=True
    )

    print(f"\n--- Loading {len(trial_names)} Active Trial Cohorts for 5-Fold CV ---")
    sys.stdout.flush()

    common_genes = expr_dict[trial_names[0]].columns
    for name in trial_names[1:]:
        common_genes = common_genes.intersection(expr_dict[name].columns)
    common_genes = list(common_genes)
    print(f"  Intersected gene expression panel: {len(common_genes):,} common genes across {len(trial_names)} cohorts")
    sys.stdout.flush()

    def _align(expr: pd.DataFrame, clin: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        resp_col = "response" if "response" in clin.columns else "RESPONDER"
        sig = extract_all_signatures(expr[common_genes])
        y = clin.loc[sig.index, resp_col].dropna()
        return sig.loc[y.index], y.astype(int)

    sig_parts, expr_parts, y_parts = [], [], []
    for idx, name in enumerate(trial_names, start=1):
        print(f"  [{idx}/{len(trial_names)}] Prepping pooled data for '{name}'...")
        sys.stdout.flush()
        sig, y = _align(expr_dict[name], clin_dict[name])
        sig_parts.append(sig)
        expr_parts.append(expr_dict[name].loc[y.index, common_genes])
        y_parts.append(y)
        n_pats = len(y)
        n_resp = int(y.sum())
        resp_pct = 100.0 * n_resp / n_pats if n_pats > 0 else 0.0
        print(f"      -> {name}: N={n_pats} patients, {n_resp} responders ({resp_pct:.1f}%)")
        sys.stdout.flush()

    X_expr = pd.concat(expr_parts).reset_index(drop=True)
    X_sigs = pd.concat(sig_parts).reset_index(drop=True)
    y_pooled = pd.concat(y_parts).reset_index(drop=True)

    n_null_sigs = X_sigs.isna().sum().sum()
    n_null_expr = X_expr.isna().sum().sum()
    print(f"\n--- Pooled Feature Matrices Summary ---")
    print(f"  Total pooled sample size : N={len(y_pooled)} patients")
    print(f"  Total responders count   : {y_pooled.sum()}/{len(y_pooled)} ({100.0 * y_pooled.mean():.1f}%)")
    print(f"  Curated signatures matrix: {X_sigs.shape[0]} samples x {X_sigs.shape[1]} columns | {n_null_sigs} NaNs")
    print(f"  Full expression matrix   : {X_expr.shape[0]} samples x {X_expr.shape[1]} genes   | {n_null_expr} NaNs")
    sys.stdout.flush()

    return X_expr, X_sigs, y_pooled


def _load_cohort_dfs(data_dir: Path) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Loads individual cohort DataFrames for LOCO evaluation."""
    config_path = SUBPROJECT_ROOT / "config" / "datasets.yaml"
    expr_dict, clin_dict, _, trial_names = load_all_active_cohorts(
        config_path, data_dir, merge_only=True
    )
    print(f"\n--- Loading Individual Cohorts for LOCO Cross-Validation ---")
    sys.stdout.flush()

    cohort_dfs: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    for idx, name in enumerate(trial_names, start=1):
        print(f"  [{idx}/{len(trial_names)}] Building LOCO signatures for '{name}'...")
        sys.stdout.flush()
        expr, clin = expr_dict[name], clin_dict[name]
        resp_col = "response" if "response" in clin.columns else "RESPONDER"
        mask = clin[resp_col].notna()
        sigs = zscore_df(extract_all_signatures(expr[mask]))
        y = clin[mask][resp_col].astype(int)
        cohort_dfs[name] = (sigs, y)
        n_pats = len(y)
        n_resp = int(y.sum())
        resp_pct = 100.0 * n_resp / n_pats if n_pats > 0 else 0.0
        print(f"      -> {name}: N={n_pats} patients, {n_resp} responders ({resp_pct:.1f}%) | 6 signatures z-scored")
        sys.stdout.flush()

    return cohort_dfs


# ---------------------------------------------------------------------------
# 5-Fold CV Routines
# ---------------------------------------------------------------------------
def _cv_fold_aucs_signatures(X_sigs: pd.DataFrame, y: pd.Series, model_type: str) -> List[float]:
    """Computes per-fold AUC list for curated signatures (5-fold stratified CV)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    X_arr, y_arr = X_sigs.values, y.values
    fold_aucs: List[float] = []
    for train_idx, val_idx in skf.split(X_arr, y_arr):
        scaler = StandardScaler()
        X_tr = pd.DataFrame(scaler.fit_transform(X_arr[train_idx]), columns=X_sigs.columns)
        X_val = pd.DataFrame(scaler.transform(X_arr[val_idx]), columns=X_sigs.columns)
        y_tr = pd.Series(y_arr[train_idx])
        model = get_model(model_type, X_tr, y_tr)
        y_prob = model.predict_proba(X_val)[:, 1]
        try:
            fold_aucs.append(roc_auc_score(y_arr[val_idx], y_prob))
        except ValueError:
            fold_aucs.append(np.nan)
    return fold_aucs


def _cv_fold_aucs_selectkbest(
    X_expr: pd.DataFrame, y: pd.Series, k: int, model_type: str
) -> List[float]:
    """Computes per-fold AUC list for SelectKBest (5-fold stratified CV)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_arr = y.values
    X_vals = X_expr.values
    fold_aucs: List[float] = []
    for train_idx, val_idx in skf.split(X_vals, y_arr):
        X_tr, X_val = X_vals[train_idx], X_vals[val_idx]
        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
        train_var = X_tr.var(axis=0)
        top_idx = np.argsort(train_var)[::-1][:min(TOP_VAR_PREFILTER, X_tr.shape[1])]
        X_tr_filt, X_val_filt = X_tr[:, top_idx], X_val[:, top_idx]
        k_val = min(k, X_tr_filt.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k_val)
        X_tr_sel = selector.fit_transform(X_tr_filt, y_tr)
        X_val_sel = selector.transform(X_val_filt)
        cols = [f"gene_{i}" for i in range(X_tr_sel.shape[1])]
        X_tr_df = pd.DataFrame(X_tr_sel, columns=cols)
        X_val_df = pd.DataFrame(X_val_sel, columns=cols)
        model = get_model(model_type, X_tr_df, pd.Series(y_tr))
        y_prob = model.predict_proba(X_val_df)[:, 1]
        try:
            fold_aucs.append(roc_auc_score(y_val, y_prob))
        except ValueError:
            fold_aucs.append(np.nan)
    return fold_aucs


# ---------------------------------------------------------------------------
# Module-level parallel worker functions
#
# IMPORTANT — Windows pickling constraint:
#   ProcessPoolExecutor on Windows uses *spawn* (not fork).  Worker callables
#   are pickled by name and re-imported in the child process, so they MUST be
#   defined at module scope.  Nested functions and lambdas are NOT picklable and
#   will raise ``AttributeError: Can't pickle local object`` at runtime.
# ---------------------------------------------------------------------------
def _cv_worker(task: Dict[str, Any]) -> Tuple[str, str, List[float]]:
    """Executes a single (model_type, feature_set) 5-fold CV task in a worker process.

    Receives plain numpy arrays to minimise pickle overhead across process
    boundaries.  DataFrames are reconstructed inside the worker.

    Args:
        task: Dict with keys:
            ``mtype``       — model type string (e.g. ``"xgb"``).
            ``feature_set`` — ``"sigs"`` for curated signatures, or an ``int`` k
                              for SelectKBest.
            ``X``           — numpy feature matrix (float64).
            ``y``           — numpy label vector (int).
            ``sig_cols``    — list of column names (signatures only; ``None`` for
                              SelectKBest tasks).

    Returns:
        Tuple of ``(mtype, column_key, fold_aucs)`` where ``column_key`` matches
        an entry in ``FEATURE_COL_LABELS``.
    """
    mtype: str = task["mtype"]
    feature_set = task["feature_set"]
    y = pd.Series(task["y"])

    if feature_set == "sigs":
        X = pd.DataFrame(task["X"], columns=task["sig_cols"])
        fold_aucs = _cv_fold_aucs_signatures(X, y, mtype)
        col_key = "Curated\nSignatures"
    else:
        k: int = feature_set
        X = pd.DataFrame(task["X"])
        fold_aucs = _cv_fold_aucs_selectkbest(X, y, k, mtype)
        col_key = f"SelectKBest\n(k={k})"

    return mtype, col_key, fold_aucs


def _loco_worker(
    args: Tuple[str, Dict[str, Tuple[pd.DataFrame, pd.Series]]]
) -> Tuple[str, Dict]:
    """Executes a single model's full LOCO cross-validation in a worker process.

    Args:
        args: Tuple of ``(model_type_key, cohort_dfs)``.

    Returns:
        Tuple of ``(model_type_key, loco_results_dict)``.
    """
    mtype, cohort_dfs = args
    return mtype, run_loco_cv(cohort_dfs, FEATURE_COLS_SIGS, model_type=mtype)


# ---------------------------------------------------------------------------
# Significance Testing
# ---------------------------------------------------------------------------
def _ttest_asterisk(fold_aucs: List[float]) -> str:
    """One-sample one-tailed t-test of fold AUCs against chance (AUROC = 0.50).

    Uses df = N_FOLDS - 1. Returns significance asterisk at standard thresholds.
    This is the appropriate method when per-fold AUC scores are available,
    as they provide a direct empirical distribution to test against 0.50.

    Args:
        fold_aucs: List of per-fold AUC values.

    Returns:
        Significance string: ``'***'``, ``'**'``, ``'*'``, or ``''``.
    """
    valid = [a for a in fold_aucs if not np.isnan(a)]
    if len(valid) < 2:
        return ""
    arr = np.array(valid)
    t_stat = (arr.mean() - 0.5) / (arr.std(ddof=1) / np.sqrt(len(arr)))
    p_val = stats.t.sf(t_stat, df=len(arr) - 1)  # one-tailed: AUC > 0.5
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return ""


# ---------------------------------------------------------------------------
# 5-Fold CV: Build Results (Parallelised)
# ---------------------------------------------------------------------------
def _build_5f_cv_results(
    X_expr: pd.DataFrame, X_sigs: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds numeric and annotated text DataFrames for the 5-fold CV panel.

    All 20 (model_type × feature_set) tasks are submitted concurrently to a
    ``ProcessPoolExecutor``; the main process collects results via
    ``as_completed`` and assembles the DataFrames in canonical order after all
    futures resolve.

    Cell annotations follow the format: ``AUROC\\n±SD`` + significance asterisk.
    Significance is assessed via one-sample one-tailed t-test of fold AUCs
    against H₀: AUROC = 0.50 (df = N_FOLDS − 1 = 4).
    The Cross-Model Mean summary row shows Mean ± SD across models (no asterisk).

    Args:
        X_expr: Pooled gene-expression matrix (samples × genes).
        X_sigs: Pre-computed immune-signature scores (samples × 6 signatures).
        y:      Binary response labels aligned to ``X_expr`` / ``X_sigs``.

    Returns:
        ``(df_num, df_ann)`` — numeric AUROC DataFrame and annotation string DataFrame.
    """
    # Pass numpy arrays to minimise per-task pickle overhead
    X_sigs_arr = X_sigs.values
    X_expr_arr = X_expr.values
    y_arr = y.values
    sig_cols = list(X_sigs.columns)

    # Build one task dict per (model, feature_set) combination — 20 tasks total
    tasks: List[Dict[str, Any]] = []
    for mtype in MODEL_ORDER:
        tasks.append({
            "mtype": mtype, "feature_set": "sigs",
            "X": X_sigs_arr, "y": y_arr, "sig_cols": sig_cols,
        })
        for k in K_VALUES:
            tasks.append({
                "mtype": mtype, "feature_set": k,
                "X": X_expr_arr, "y": y_arr, "sig_cols": None,
            })

    # Collect per-fold AUC lists keyed by (mtype, col_key)
    fold_results: Dict[Tuple[str, str], List[float]] = {}
    print(f"  Submitting {len(tasks)} CV tasks to {N_WORKERS} workers...")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        future_map = {pool.submit(_cv_worker, t): t for t in tasks}
        for fut in as_completed(future_map):
            mtype, col_key, fold_aucs = fut.result()
            fold_results[(mtype, col_key)] = fold_aucs
            mean_auc = float(np.nanmean(fold_aucs))
            print(f"  ✓ {MODEL_LABELS[mtype]:25s}  {col_key!r:25s}  AUC = {mean_auc:.3f}")

    # Reassemble DataFrames in canonical MODEL_ORDER / FEATURE_COL_LABELS order
    rows_num: List[Dict[str, Any]] = []
    rows_annot: List[Dict[str, Any]] = []
    for mtype in MODEL_ORDER:
        label = MODEL_LABELS_WRAPPED[mtype]
        row_num: Dict[str, Any] = {"Model": label}
        row_ann: Dict[str, Any] = {"Model": label}
        for col_key in FEATURE_COL_LABELS:
            fold_aucs = fold_results[(mtype, col_key)]
            m = float(np.nanmean(fold_aucs))
            s = float(np.nanstd(fold_aucs))
            ast = _ttest_asterisk(fold_aucs)
            row_num[col_key] = m
            row_ann[col_key] = f"{m:.3f}\n±{s:.3f}{ast}"
        rows_num.append(row_num)
        rows_annot.append(row_ann)

    df_num = pd.DataFrame(rows_num).set_index("Model")
    df_ann = pd.DataFrame(rows_annot).set_index("Model")

    # Cross-Model Mean summary row (no t-test asterisk — aggregated across models)
    mean_vals = df_num.mean(axis=0)
    std_vals = df_num.std(axis=0)
    mean_row_num = mean_vals.to_frame().T
    mean_row_num.index = ["Cross-Model\nMean"]
    mean_row_ann = {col: f"{mean_vals[col]:.3f}\n±{std_vals[col]:.3f}" for col in df_num.columns}
    df_mean_ann = pd.DataFrame([mean_row_ann], index=["Cross-Model\nMean"])

    df_num = pd.concat([df_num, mean_row_num])
    df_ann = pd.concat([df_ann, df_mean_ann])
    return df_num[FEATURE_COL_LABELS], df_ann[FEATURE_COL_LABELS]


# ---------------------------------------------------------------------------
# LOCO Evaluation with Mann-Whitney Significance & Bootstrap SD
# ---------------------------------------------------------------------------
def _get_significance_asterisk(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    """Computes Mann-Whitney U test p-value for AUC > 0.5 and returns significance asterisk."""
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return ""
    _, p_val = stats.mannwhitneyu(pos, neg, alternative="greater")
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return ""


def _bootstrap_auc_sd(
    y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = N_BOOTSTRAP
) -> float:
    """Estimates AUROC standard deviation via vectorised non-parametric bootstrap resampling.

    All ``n_boot`` index arrays are generated in a **single** NumPy call — shape
    ``(n_boot, n)`` — replacing the original Python-level ``for _ in range(n_boot)``
    loop.  Only the final ``roc_auc_score`` call per resample remains in Python,
    which is unavoidable without a fully custom AUC implementation.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.
        n_boot: Number of bootstrap resamples (default 1,000).

    Returns:
        Bootstrap standard deviation of AUROC, or ``np.nan`` if no valid resamples.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    # Generate all bootstrap index arrays in one vectorised call: (n_boot, n)
    idx = rng.integers(0, n, size=(n_boot, n))
    y_b = y_true[idx]   # (n_boot, n) — integer array, cheap fancy-index
    p_b = y_prob[idx]   # (n_boot, n)
    # Retain only resamples where both classes are represented
    class_sums = y_b.sum(axis=1)
    valid_rows = np.where((class_sums > 0) & (class_sums < n))[0]
    boot_aucs: List[float] = []
    for i in valid_rows:
        try:
            boot_aucs.append(roc_auc_score(y_b[i], p_b[i]))
        except ValueError:
            continue
    return float(np.std(boot_aucs)) if boot_aucs else np.nan


# ---------------------------------------------------------------------------
# LOCO: Build Results (Parallelised)
# ---------------------------------------------------------------------------
def _build_loco_results(
    cohort_dfs: Dict[str, Tuple[pd.DataFrame, pd.Series]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds numeric and annotated text DataFrames for the LOCO panel.

    All 5 model LOCO runs are submitted concurrently to a ``ProcessPoolExecutor``.
    Bootstrap SD and significance testing run in the main process after all
    futures resolve (each call is fast; parallelising them would add more overhead
    than it saves).

    Per-cohort cells display:  AUROC ± bootstrap SD + significance asterisk.
    Summary Mean LOCO column:  Mean ± SD across the held-out cohorts.
    Cross-Model Mean row:       Mean ± SD across models.

    Args:
        cohort_dfs: Dict mapping cohort name → (signature DataFrame, response Series).

    Returns:
        ``(df_num, df_ann)`` — numeric AUROC DataFrame and annotation string DataFrame.
    """
    # Submit one LOCO task per model — all five are fully independent
    all_loco_results: Dict[str, Dict] = {}
    print(f"  Submitting {len(MODEL_ORDER)} LOCO tasks to {N_WORKERS} workers...")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        future_map = {
            pool.submit(_loco_worker, (mtype, cohort_dfs)): mtype
            for mtype in MODEL_ORDER
        }
        for fut in as_completed(future_map):
            mtype, loco_res = fut.result()
            all_loco_results[mtype] = loco_res
            print(f"  ✓ LOCO complete: {MODEL_LABELS[mtype]}")

    all_cohorts = sorted({c for res in all_loco_results.values() for c in res})
    rows_num: List[Dict[str, Any]] = []
    rows_ann: List[Dict[str, Any]] = []

    for mtype in MODEL_ORDER:
        lbl = MODEL_LABELS_WRAPPED[mtype]
        row_n: Dict[str, Any] = {"Model": lbl}
        row_a: Dict[str, Any] = {"Model": lbl}
        loco = all_loco_results[mtype]

        cohort_aucs: List[float] = []
        for c in all_cohorts:
            res = loco.get(c, {})
            n_s = len(res["y_true"]) if "y_true" in res else 0
            col_name = f"{c}\n(N={n_s})"

            y_true = np.array(res["y_true"])
            y_prob = np.array(res["y_pred_prob"])
            auc = roc_auc_score(y_true, y_prob) if len(y_true) > 0 else np.nan
            boot_sd = _bootstrap_auc_sd(y_true, y_prob) if len(y_true) > 0 else np.nan
            asterisk = _get_significance_asterisk(y_true, y_prob) if len(y_true) > 0 else ""

            row_n[col_name] = auc
            # Format: AUROC\n±SD* — bootstrap SD estimates within-cohort variability
            row_a[col_name] = f"{auc:.3f}\n±{boot_sd:.3f}{asterisk}"
            cohort_aucs.append(auc)

        mean_loco = float(np.mean(cohort_aucs))
        std_loco = float(np.std(cohort_aucs))
        row_n["Mean\nLOCO"] = mean_loco
        row_a["Mean\nLOCO"] = f"{mean_loco:.3f}\n±{std_loco:.3f}"

        rows_num.append(row_n)
        rows_ann.append(row_a)

    df_num = pd.DataFrame(rows_num).set_index("Model")
    df_ann = pd.DataFrame(rows_ann).set_index("Model")

    # Cross-Model Mean row
    cross_mean = df_num.mean(axis=0)
    cross_std = df_num.std(axis=0)
    mean_row_num = cross_mean.to_frame().T
    mean_row_num.index = ["Cross-Model\nMean"]
    row_cross_ann = {
        col: f"{cross_mean[col]:.3f}\n±{cross_std[col]:.3f}"
        for col in df_num.columns
    }
    df_cross_ann = pd.DataFrame([row_cross_ann], index=["Cross-Model\nMean"])

    df_num = pd.concat([df_num, mean_row_num])
    df_ann = pd.concat([df_ann, df_cross_ann])
    return df_num, df_ann


# ---------------------------------------------------------------------------
# 1×2 Grid Plot Generation
# ---------------------------------------------------------------------------
def generate_1x2_grid_heatmap() -> None:
    """Generates the combined 1×2 side-by-side grid heatmap figure."""
    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    X_expr, X_sigs, y_pooled = _load_pooled_data(DATA_DIR)
    cohort_dfs = _load_cohort_dfs(DATA_DIR)

    print(f"\nRunning 5-Fold Stratified CV (Curated vs. SelectKBest) — {N_WORKERS} workers...")
    df_cv_num, df_cv_ann = _build_5f_cv_results(X_expr, X_sigs, y_pooled)

    print(f"\nRunning LOCO Cross-Validation — {N_WORKERS} workers...")
    df_loco_num, df_loco_ann = _build_loco_results(cohort_dfs)

    # --- Render 1×2 figure ---
    print("\nRendering 1×2 Grid Heatmap (Option 1 with ±SD & Asterisks)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(23, 8.2))

    n_data_cv = df_cv_num.shape[0] - 1
    n_data_loco = df_loco_num.shape[0] - 1

    # --- Left Panel: 5-Fold CV Comparison ---
    sns.heatmap(
        df_cv_num,
        annot=df_cv_ann,
        fmt="",
        cmap=HEATMAP_CMAP,
        cbar_kws={"label": "AUROC Score"},
        linewidths=1.0,
        linecolor="white",
        annot_kws={"size": HEATMAP_ANNOT_SIZE, "weight": "bold"},
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        ax=ax1,
    )
    # Bold outline on the best feature column per model row (excl. mean row)
    for row_idx in range(n_data_cv):
        row_data = df_cv_num.iloc[row_idx]
        best_col_idx = int(row_data.values.argmax())
        ax1.add_patch(plt.Rectangle(
            (best_col_idx, row_idx), 1, 1,
            fill=False, edgecolor="red", linewidth=2.2,
        ))
    # Thick white vertical separator between Curated Signatures and SelectKBest
    ax1.axvline(1, color="white", linewidth=7.0, zorder=6)
    # Thick white horizontal separator before Cross-Model Mean row
    ax1.axhline(n_data_cv, color="white", linewidth=7.0, zorder=6)

    ax1.set_title(
        f"A) 5-Fold Stratified CV AUROC (Mean ± SD):\n"
        f"Curated Signatures vs. SelectKBest (N={len(y_pooled)};  * p < 0.05, ** p < 0.01 vs. chance)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax1.set_xlabel("Feature Representation", fontsize=HEATMAP_LABEL_SIZE, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Model Architecture", fontsize=HEATMAP_LABEL_SIZE, fontweight="bold", labelpad=8)
    ax1.tick_params(axis="y", labelrotation=0, labelsize=10)
    ax1.tick_params(axis="x", labelsize=10)

    # --- Right Panel: LOCO ---
    sns.heatmap(
        df_loco_num,
        annot=df_loco_ann,
        fmt="",
        cmap=HEATMAP_CMAP,
        cbar_kws={"label": "AUROC Score"},
        linewidths=1.0,
        linecolor="white",
        annot_kws={"size": HEATMAP_ANNOT_SIZE, "weight": "bold"},
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        ax=ax2,
    )
    n_cols_loco = df_loco_num.shape[1]
    # Bold outline on best model per column (excl. mean row)
    for c_idx in range(n_cols_loco):
        col_data = df_loco_num.iloc[:n_data_loco, c_idx]
        best_r = int(col_data.values.argmax())
        ax2.add_patch(plt.Rectangle(
            (c_idx, best_r), 1, 1,
            fill=False, edgecolor="red", linewidth=2.2,
        ))
    ax2.axvline(n_cols_loco - 1, color="white", linewidth=7.0, zorder=6)
    ax2.axhline(n_data_loco, color="white", linewidth=7.0, zorder=6)

    ax2.set_title(
        "B) Leave-One-Cohort-Out (LOCO) AUROC Across Held-Out Cohorts\n"
        "(* p < 0.05, ** p < 0.01 vs. chance; ±SD from 1,000-sample bootstrap)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax2.set_xlabel("Held-Out Test Cohort", fontsize=HEATMAP_LABEL_SIZE, fontweight="bold", labelpad=8)
    ax2.set_ylabel("")  # Shared y-axis labels from left panel
    ax2.tick_params(axis="y", labelrotation=0, labelsize=10)
    ax2.tick_params(axis="x", labelsize=10)

    plt.tight_layout(w_pad=3.0)

    # Save standard version
    save_fig(fig, OUTPUT_PATH)

    # Save transparent version
    fig.patch.set_alpha(0.0)
    ax1.patch.set_alpha(0.0)
    ax2.patch.set_alpha(0.0)
    fig.savefig(OUTPUT_TRANS_PATH, transparent=True, bbox_inches="tight", dpi=300)

    plt.close(fig)
    print(f"\nSaved Option 1 1×2 grid heatmap to {rel_path(OUTPUT_PATH)}")
    print(f"Saved transparent copy to {rel_path(OUTPUT_TRANS_PATH)}")


if __name__ == "__main__":
    with setup_logging(LOG_PATH, relative_to=SUBPROJECT_ROOT):
        generate_1x2_grid_heatmap()
