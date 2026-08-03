"""Generates 5-Fold Stratified CV Comparison Heatmap: Curated Signatures vs SelectKBest.

Evaluates the same five model architectures (Logistic Regression, Random Forest,
XGBoost, SVM, ElasticNet) as the LOCO comparison in run_comparison.py, but using
5-fold stratified CV on the pooled immunotherapy cohort instead of leave-one-cohort-out.

Compares four feature representations:
  - Curated Signatures (transcriptomic immune signatures extracted from common genes)
  - SelectKBest k=20  (top-20 by F-score from raw expression, common gene set)
  - SelectKBest k=100 (top-100)
  - SelectKBest k=200 (top-200)

Data loading mirrors run_comparison.py exactly: genes are intersected across all three
cohorts before pooling to avoid NaN values from missing gene coverage.

Rows = model families. Columns = feature representations. Colour scheme matches
loco_performance_heatmap.png (YlGnBu, vmin=0.25, vmax=0.75). The best-performing
representation per model row is highlighted with a bold black outline. A cross-model
mean summary row is appended at the bottom.
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
# Standard Library & Third-Party Imports
# ---------------------------------------------------------------------------
import contextlib
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------
from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.signatures import extract_all_signatures
from src.styles import set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, PLOTS_DIR, SUBPROJECT_ROOT, rel_path
from src.utils.plotting import save_fig

set_presentation_style()

# ---------------------------------------------------------------------------
# Module-level Constants & Configuration
# ---------------------------------------------------------------------------

MODEL_ORDER: List[str] = ["xgb", "rf", "svm", "elasticnet", "lr"]
MODEL_LABELS: Dict[str, str] = {
    "xgb": "XGBoost",
    "rf": "Random Forest",
    "svm": "Support Vector Machine",
    "elasticnet": "ElasticNet",
    "lr": "Logistic Regression",
}

# SelectKBest k values to evaluate (mirroring run_comparison.py)
K_VALUES: List[int] = [20, 100, 200]

# Pre-filter to top-N most variable genes before SelectKBest (same as run_comparison.py)
TOP_VAR_PREFILTER: int = 1000

N_FOLDS: int = 5
RANDOM_STATE: int = 42

# Standardised shared colour threshold (0.30 to 0.70) across both 5-Fold CV and LOCO heatmaps for direct visual comparability
HEATMAP_VMIN: float = 0.30
HEATMAP_VMAX: float = 0.70
HEATMAP_CMAP: str = "YlGnBu"
HEATMAP_FIGSIZE: Tuple[int, int] = (16, 9)  # 16:9 aspect ratio
HEATMAP_ANNOT_SIZE: int = 14
HEATMAP_LABEL_SIZE: int = 12

FEATURE_COL_LABELS: List[str] = (
    ["Curated Signatures"] + [f"SelectKBest (k={k})" for k in K_VALUES]
)

OUTPUT_PLOT_DIR: Path = PLOTS_DIR / "feature_selection"
OUTPUT_PATH: Path = OUTPUT_PLOT_DIR / "5f_cv_feature_comparison_heatmap.png"
LOG_DIR: Path = SUBPROJECT_ROOT / "logs"
LOG_PATH: Path = LOG_DIR / "generate_5f_cv_comparison_heatmap.log"


# ---------------------------------------------------------------------------
# Data Loading — mirrors run_comparison.py exactly
# ---------------------------------------------------------------------------
def _load_pooled_data(
    data_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Loads and pools all three immunotherapy cohorts, intersecting genes to avoid NaNs.

    Replicates the data loading logic from run_comparison.py:
      1. Load raw expression and clinical data for each cohort.
      2. Intersect gene columns across all three cohorts.
      3. Extract transcriptomic signatures from the common gene set.
      4. Align samples with non-null response labels.
      5. Concatenate into pooled expression, signature, and label arrays.

    Args:
        data_dir: Root data directory.

    Returns:
        Tuple of (pooled_expr_common_genes, pooled_signatures, pooled_response_labels),
        all with reset integer indices.
    """
    expr_liu, clin_liu = load_liu_2019(data_dir)
    expr_hugo, clin_hugo = load_hugo_2016(data_dir)
    expr_riaz, clin_riaz = load_riaz_2017(data_dir)

    # Intersect genes across all three cohorts to guarantee no NaN columns
    common_genes = list(
        set(expr_liu.columns) & set(expr_hugo.columns) & set(expr_riaz.columns)
    )
    print(f"  Common genes across all cohorts: {len(common_genes)}")

    # Extract signatures from the common gene set (same as run_comparison.py)
    sig_liu = extract_all_signatures(expr_liu[common_genes])
    sig_hugo = extract_all_signatures(expr_hugo[common_genes])
    sig_riaz = extract_all_signatures(expr_riaz[common_genes])

    # Align with non-null response labels — run_comparison.py uses "response" column
    def _align(sig, clin):
        resp_col = "response" if "response" in clin.columns else "RESPONDER"
        y = clin.loc[sig.index, resp_col].dropna()
        return sig.loc[y.index], y.astype(int)

    sig_liu, y_liu = _align(sig_liu, clin_liu)
    sig_hugo, y_hugo = _align(sig_hugo, clin_hugo)
    sig_riaz, y_riaz = _align(sig_riaz, clin_riaz)

    # Pool expression (common genes only) aligned to same samples as signatures
    expr_liu_c = expr_liu.loc[y_liu.index, common_genes]
    expr_hugo_c = expr_hugo.loc[y_hugo.index, common_genes]
    expr_riaz_c = expr_riaz.loc[y_riaz.index, common_genes]

    X_expr = pd.concat([expr_liu_c, expr_hugo_c, expr_riaz_c]).reset_index(drop=True)
    X_sigs = pd.concat([sig_liu, sig_hugo, sig_riaz]).reset_index(drop=True)
    y_pooled = pd.concat([y_liu, y_hugo, y_riaz]).reset_index(drop=True)

    return X_expr, X_sigs, y_pooled


# ---------------------------------------------------------------------------
# Model Factory — fixed hyperparameters matching run_comparison.py
# ---------------------------------------------------------------------------
def _build_model(model_type: str):
    """Instantiates a classifier with fixed hyperparameters matching run_comparison.py.

    Args:
        model_type: One of 'lr', 'rf', 'xgb', 'svm', 'elasticnet'.

    Returns:
        Unfitted sklearn-compatible classifier.

    Raises:
        ValueError: If model_type is not recognised.
    """
    if model_type == "lr":
        return LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)
    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1
        )
    elif model_type == "xgb":
        return XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1,
        )
    elif model_type == "svm":
        return SVC(probability=True, kernel="rbf", C=1.0, random_state=RANDOM_STATE)
    elif model_type == "elasticnet":
        return LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5,
            max_iter=2000, random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")


# ---------------------------------------------------------------------------
# 5-Fold CV Routines
# ---------------------------------------------------------------------------
def _cv_mean_auc_signatures(
    X_sigs: pd.DataFrame,
    y: pd.Series,
    model_type: str,
) -> float:
    """Computes 5-fold CV mean AUC using the curated transcriptomic signature panel.

    Features are standardized within-fold to ensure scale-sensitive models (SVM, LR)
    are properly calibrated across features with different ranges (e.g. TMB vs z-scores).

    Args:
        X_sigs: Pooled signature DataFrame (patients x signature features).
        y: Binary response labels.
        model_type: Classifier key.

    Returns:
        Mean ROC-AUC across 5 stratified folds.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    X_arr = X_sigs.values
    y_arr = y.values
    fold_aucs: List[float] = []

    for train_idx, val_idx in skf.split(X_arr, y_arr):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_arr[train_idx])
        X_val = scaler.transform(X_arr[val_idx])

        model = _build_model(model_type)
        model.fit(X_tr, y_arr[train_idx])
        y_prob = model.predict_proba(X_val)[:, 1]
        try:
            fold_aucs.append(roc_auc_score(y_arr[val_idx], y_prob))
        except ValueError:
            fold_aucs.append(np.nan)

    return float(np.nanmean(fold_aucs))


def _cv_mean_auc_selectkbest(
    X_expr: pd.DataFrame,
    y: pd.Series,
    k: int,
    model_type: str,
) -> float:
    """Computes 5-fold CV mean AUC using SelectKBest on raw expression data.

    Mirrors the run_comparison.py SelectKBest pipeline within each fold:
      1. Pre-filter to top-1000 most variable genes (train split only).
      2. Select k features by F-score (train split only).
      3. Transform val split with the same selector.
      4. Fit model and evaluate AUC.

    Args:
        X_expr: Pooled raw expression DataFrame (patients x common genes).
        y: Binary response labels.
        k: Number of features to select via SelectKBest.
        model_type: Classifier key.

    Returns:
        Mean ROC-AUC across 5 stratified folds.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_arr = y.values
    X_vals = X_expr.values
    col_names = np.array(X_expr.columns)
    fold_aucs: List[float] = []

    for train_idx, val_idx in skf.split(X_vals, y_arr):
        X_tr = X_vals[train_idx]
        X_val = X_vals[val_idx]
        y_tr = y_arr[train_idx]
        y_val = y_arr[val_idx]

        # Pre-filter to top-variance genes on train split only (avoids leakage)
        train_var = X_tr.var(axis=0)
        top_idx = np.argsort(train_var)[::-1][:min(TOP_VAR_PREFILTER, X_tr.shape[1])]
        X_tr_filt = X_tr[:, top_idx]
        X_val_filt = X_val[:, top_idx]

        # SelectKBest within-fold
        k_val = min(k, X_tr_filt.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k_val)
        X_tr_sel = selector.fit_transform(X_tr_filt, y_tr)
        X_val_sel = selector.transform(X_val_filt)

        model = _build_model(model_type)
        model.fit(X_tr_sel, y_tr)
        y_prob = model.predict_proba(X_val_sel)[:, 1]
        try:
            fold_aucs.append(roc_auc_score(y_val, y_prob))
        except ValueError:
            fold_aucs.append(np.nan)

    return float(np.nanmean(fold_aucs))


# ---------------------------------------------------------------------------
# Results Aggregation
# ---------------------------------------------------------------------------
def _build_results_dataframe(
    X_expr: pd.DataFrame,
    X_sigs: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Runs all (model x feature representation) combinations and assembles results.

    Args:
        X_expr: Pooled raw expression DataFrame (common genes, no NaNs).
        X_sigs: Pooled signature DataFrame.
        y: Binary response labels.

    Returns:
        DataFrame with rows = model labels, columns = FEATURE_COL_LABELS, values = mean AUC.
    """
    rows = []
    for mtype in MODEL_ORDER:
        label = MODEL_LABELS[mtype]
        print(f"\n--- {label} ---")

        sig_auc = _cv_mean_auc_signatures(X_sigs, y, mtype)
        print(f"  Curated Signatures:    AUC = {sig_auc:.3f}")

        row = {"Model": label, "Curated Signatures": sig_auc}

        for k in K_VALUES:
            col = f"SelectKBest (k={k})"
            skb_auc = _cv_mean_auc_selectkbest(X_expr, y, k, mtype)
            row[col] = skb_auc
            print(f"  SelectKBest (k={k:>3}): AUC = {skb_auc:.3f}")

        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    return df[FEATURE_COL_LABELS]


# ---------------------------------------------------------------------------
# Heatmap Plotting
# ---------------------------------------------------------------------------
def _plot_comparison_heatmap(
    df: pd.DataFrame,
    n_pooled: int,
    output_path: Path,
) -> None:
    """Renders and saves the 5-fold CV feature comparison heatmap.

    Uses the same YlGnBu colour scheme, vmin/vmax, font sizes, and line weights as
    loco_performance_heatmap.png. The best-performing feature representation per
    model row is highlighted with a bold black outline. A cross-model mean summary
    row is appended at the bottom, separated by a dashed horizontal line.

    Args:
        df: Results DataFrame (rows = model labels, columns = feature representations).
        n_pooled: Total patient count for the figure title N= annotation.
        output_path: Destination path for the saved PNG.
    """
    # Append cross-model mean summary row
    mean_row = df.mean(axis=0)
    mean_row.name = "Cross-Model Mean"
    df_with_mean = pd.concat([df, mean_row.to_frame().T])

    n_data_rows = len(df)
    best_col_per_row = df.idxmax(axis=1)

    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)

    sns.heatmap(
        df_with_mean,
        annot=True,
        fmt=".3f",
        cmap=HEATMAP_CMAP,
        cbar_kws={"label": "5-Fold CV ROC-AUC Score"},
        linewidths=0.4,
        linecolor="white",
        annot_kws={"size": HEATMAP_ANNOT_SIZE, "weight": "bold"},
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        ax=ax,
    )

    # Bold outline on the best-performing feature column per model row
    for row_idx, (_, col_label) in enumerate(best_col_per_row.items()):
        col_idx = FEATURE_COL_LABELS.index(col_label)
        ax.add_patch(plt.Rectangle(
            (col_idx, row_idx), 1, 1,
            fill=False, edgecolor="black", linewidth=2.5,
        ))

    # Thick white horizontal separator before the Cross-Model Mean row — matches LOCO heatmap style
    ax.axhline(n_data_rows, color="white", linewidth=4.0, zorder=6)

    ax.set_title(
        f"5-Fold Stratified CV ROC-AUC: Curated Signatures vs. SelectKBest\n"
        f"(Pooled Immunotherapy Cohorts, N={n_pooled}  |  bold outline = best per row)",
        fontsize=HEATMAP_ANNOT_SIZE,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Feature Representation", fontsize=HEATMAP_LABEL_SIZE, fontweight="bold", labelpad=10)
    ax.set_ylabel("Model Architecture", fontsize=HEATMAP_LABEL_SIZE, fontweight="bold", labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=HEATMAP_LABEL_SIZE)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=HEATMAP_LABEL_SIZE)

    # Explicit margins force the heatmap to fill the 16:9 canvas rather than
    # shrinking to fit cell aspect ratios (which tight_layout would do).
    fig.subplots_adjust(left=0.18, right=0.86, top=0.88, bottom=0.14)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save at exact figsize (16:9) — no bbox_inches="tight" so canvas ratio is preserved
    fig.savefig(output_path, dpi=300)
    print(f"Saved 5-fold CV comparison heatmap to {rel_path(output_path)}")

    transparent_path = output_path.parent / (output_path.stem + "_transparent.png")
    fig.savefig(transparent_path, transparent=True, dpi=300)
    print(f"Saved transparent copy to {rel_path(transparent_path)}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    """Runs the full 5-fold CV comparison pipeline and saves the heatmap."""
    print("=" * 60)
    print("5-Fold CV: Curated Signatures vs. SelectKBest (5 Models)")
    print("=" * 60)

    print("\nLoading and pooling cohorts (Liu 2019 + Hugo 2016 + Riaz 2017)...")
    X_expr, X_sigs, y = _load_pooled_data(DATA_DIR)
    n_pooled = len(y)
    print(f"  Pooled dataset: N={n_pooled} patients, {int(y.sum())} responders "
          f"({100 * y.mean():.1f}%)")
    print(f"  Expression matrix: {X_expr.shape[1]} common genes")
    print(f"  Signature features: {list(X_sigs.columns)}")

    df_results = _build_results_dataframe(X_expr, X_sigs, y)

    print("\n\nSummary of 5-Fold CV Mean AUC:")
    print(df_results.to_string())

    _plot_comparison_heatmap(df_results, n_pooled, OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
