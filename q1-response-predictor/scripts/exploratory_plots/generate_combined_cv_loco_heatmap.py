"""Generates 1x2 Side-by-Side Heatmap Grid: 5-Fold CV Comparison + LOCO Performance (Option 1).

Left panel:  5-Fold Stratified CV AUROC (Mean ± SD) — Curated Signatures vs. SelectKBest (k=20, 100, 200).
Right panel: Leave-One-Cohort-Out (LOCO) AUROC across held-out cohorts with significance asterisks (* p < 0.05 vs chance),
             and Mean ± SD for summary row/col.

Model names on the y-axis are wrapped onto two lines for readability.
Transparent background PNGs exported for presentation decks.
"""

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
import contextlib
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.evaluation import calculate_extended_metrics
from src.models import get_model, run_loco_cv
from src.signatures import extract_all_signatures, zscore_df
from src.styles import set_presentation_style
from src.utils.logging import TeeStream
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
    """Loads and pools all three cohorts, intersecting genes to avoid NaNs."""
    expr_liu, clin_liu = load_liu_2019(data_dir)
    expr_hugo, clin_hugo = load_hugo_2016(data_dir)
    expr_riaz, clin_riaz = load_riaz_2017(data_dir)

    common_genes = list(
        set(expr_liu.columns) & set(expr_hugo.columns) & set(expr_riaz.columns)
    )
    print(f"  Common genes across all cohorts: {len(common_genes)}")

    sig_liu = extract_all_signatures(expr_liu[common_genes])
    sig_hugo = extract_all_signatures(expr_hugo[common_genes])
    sig_riaz = extract_all_signatures(expr_riaz[common_genes])

    def _align(sig, clin):
        resp_col = "response" if "response" in clin.columns else "RESPONDER"
        y = clin.loc[sig.index, resp_col].dropna()
        return sig.loc[y.index], y.astype(int)

    sig_liu, y_liu = _align(sig_liu, clin_liu)
    sig_hugo, y_hugo = _align(sig_hugo, clin_hugo)
    sig_riaz, y_riaz = _align(sig_riaz, clin_riaz)

    expr_liu_c = expr_liu.loc[y_liu.index, common_genes]
    expr_hugo_c = expr_hugo.loc[y_hugo.index, common_genes]
    expr_riaz_c = expr_riaz.loc[y_riaz.index, common_genes]

    X_expr = pd.concat([expr_liu_c, expr_hugo_c, expr_riaz_c]).reset_index(drop=True)
    X_sigs = pd.concat([sig_liu, sig_hugo, sig_riaz]).reset_index(drop=True)
    y_pooled = pd.concat([y_liu, y_hugo, y_riaz]).reset_index(drop=True)

    return X_expr, X_sigs, y_pooled


def _load_cohort_dfs(data_dir: Path) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Loads individual cohort DataFrames for LOCO evaluation."""
    from src.signatures import zscore_df as _zscore
    loaders = {"Liu 2019": load_liu_2019, "Hugo 2016": load_hugo_2016, "Riaz 2017": load_riaz_2017}
    cohort_dfs = {}
    for name, loader in loaders.items():
        expr, clin = loader(data_dir)
        mask = clin["RESPONDER"].notna()
        sigs = _zscore(extract_all_signatures(expr[mask]))
        y = clin[mask]["RESPONDER"].astype(int)
        cohort_dfs[name] = (sigs, y)
    return cohort_dfs


# ---------------------------------------------------------------------------
# 5-Fold CV Routines with Mean, SD, and Per-Fold AUCs (using get_model tuning)
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


def _cv_fold_aucs_selectkbest(X_expr: pd.DataFrame, y: pd.Series, k: int, model_type: str) -> List[float]:
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


def _ttest_asterisk(fold_aucs: List[float]) -> str:
    """One-sample one-tailed t-test of fold AUCs against chance (AUROC = 0.50).

    Uses df = N_FOLDS - 1. Returns significance asterisk at standard thresholds.
    This is the appropriate method when per-fold AUC scores are available,
    as they provide a direct empirical distribution to test against 0.50.

    Args:
        fold_aucs: List of per-fold AUC values.

    Returns:
        Significance string: '***', '**', '*', or '' (not significant).
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


def _build_5f_cv_results(X_expr: pd.DataFrame, X_sigs: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds numeric values and annotated text DataFrames for 5-fold CV panel.

    Cell annotations follow the format: AUROC\n±SD + significance asterisk.
    Significance is assessed via one-sample one-tailed t-test of fold AUCs
    against H0: AUROC = 0.50 (df = N_FOLDS - 1 = 4).
    The Cross-Model Mean summary row shows Mean ± SD across models (no asterisk).
    """
    rows_num, rows_annot = [], []
    for mtype in MODEL_ORDER:
        label = MODEL_LABELS_WRAPPED[mtype]
        print(f"\n--- {MODEL_LABELS[mtype]} ---")

        folds_sig = _cv_fold_aucs_signatures(X_sigs, y, mtype)
        m_sig = float(np.nanmean(folds_sig))
        s_sig = float(np.nanstd(folds_sig))
        ast_sig = _ttest_asterisk(folds_sig)
        row_num = {"Model": label, "Curated\nSignatures": m_sig}
        row_ann = {"Model": label, "Curated\nSignatures": f"{m_sig:.3f}\n±{s_sig:.3f}{ast_sig}"}
        print(f"  Curated Signatures:    AUC = {m_sig:.3f} +/- {s_sig:.3f} {ast_sig}")

        for k in K_VALUES:
            col = f"SelectKBest\n(k={k})"
            folds_skb = _cv_fold_aucs_selectkbest(X_expr, y, k, mtype)
            m_skb = float(np.nanmean(folds_skb))
            s_skb = float(np.nanstd(folds_skb))
            ast_skb = _ttest_asterisk(folds_skb)
            row_num[col] = m_skb
            row_ann[col] = f"{m_skb:.3f}\n±{s_skb:.3f}{ast_skb}"
            print(f"  SelectKBest (k={k:>3}): AUC = {m_skb:.3f} +/- {s_skb:.3f} {ast_skb}")

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
N_BOOTSTRAP: int = 1000  # Number of bootstrap resamples for LOCO AUROC SD estimation


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


def _bootstrap_auc_sd(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = N_BOOTSTRAP) -> float:
    """Estimates AUROC standard deviation via non-parametric bootstrap resampling.

    Draws n_boot samples with replacement from the held-out test set and computes
    AUROC on each resample. Returns the standard deviation across bootstrap AUROCs
    as an empirical estimate of variability for a single LOCO evaluation.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive-class probabilities.
        n_boot: Number of bootstrap resamples (default 1,000).

    Returns:
        Bootstrap standard deviation of AUROC.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    boot_aucs: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = y_true[idx], y_prob[idx]
        # Skip degenerate resamples with only one class present
        if len(np.unique(y_b)) < 2:
            continue
        try:
            boot_aucs.append(roc_auc_score(y_b, p_b))
        except ValueError:
            continue
    return float(np.std(boot_aucs)) if boot_aucs else np.nan


def _build_loco_results(cohort_dfs: Dict[str, Tuple[pd.DataFrame, pd.Series]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Builds numeric values and annotated text DataFrames for LOCO panel.

    Per-cohort cells display: AUROC ± bootstrap SD + significance asterisk.
    Summary Mean LOCO column displays: Mean ± SD across the held-out cohorts.
    Cross-Model Mean row displays: Mean ± SD across models.
    """
    all_loco_results = {
        mtype: run_loco_cv(cohort_dfs, FEATURE_COLS_SIGS, model_type=mtype)
        for mtype in MODEL_ORDER
    }

    all_cohorts = sorted({c for res in all_loco_results.values() for c in res})
    rows_num, rows_ann = [], []

    for mtype in MODEL_ORDER:
        lbl = MODEL_LABELS_WRAPPED[mtype]
        row_n = {"Model": lbl}
        row_a = {"Model": lbl}
        loco = all_loco_results[mtype]

        cohort_aucs = []
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
            # Format: AUROC\n±SD* — bootstrap SD for within-cohort variability estimate
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

    row_cross_ann = {}
    for col in df_num.columns:
        row_cross_ann[col] = f"{cross_mean[col]:.3f}\n±{cross_std[col]:.3f}"
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

    print("\nRunning 5-Fold Stratified CV (Curated vs. SelectKBest)...")
    df_cv_num, df_cv_ann = _build_5f_cv_results(X_expr, X_sigs, y_pooled)

    print("\nRunning LOCO Cross-Validation...")
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
        best_col_idx = row_data.values.argmax()
        ax1.add_patch(plt.Rectangle(
            (best_col_idx, row_idx), 1, 1,
            fill=False, edgecolor="black", linewidth=2.2,
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
        best_r = col_data.values.argmax()
        ax2.add_patch(plt.Rectangle(
            (c_idx, best_r), 1, 1,
            fill=False, edgecolor="black", linewidth=2.2,
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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            generate_1x2_grid_heatmap()
