"""Generates 5-Fold Stratified Cross-Validation ROC-AUC Performance Heatmap.

Evaluates the same five model architectures (Logistic Regression, Random Forest,
XGBoost, SVM, ElasticNet) and the same six curated transcriptomic signature features
used in the LOCO heatmap, but using 5-fold stratified CV on the pooled immunotherapy
cohort (pooled across active trial cohorts) rather than leave-one-cohort-out splits.

Each fold reports per-fold AUC; the heatmap cells show mean AUC +/- std across folds,
rendered in the same YlGnBu colour scheme as loco_performance_heatmap.png for
direct visual comparison.
"""

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------
from src.data_loaders import load_all_active_cohorts, load_cohort_by_name
from src.models import get_model
from src.signatures import extract_all_signatures, zscore_df
from src.styles import set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, PLOTS_DIR, rel_path
from src.utils.plotting import save_fig

set_presentation_style()

# ---------------------------------------------------------------------------
# Module-level Constants & Configuration
# ---------------------------------------------------------------------------

# Same model ordering and labels as generate_loco_heatmap.py
MODEL_ORDER: List[str] = ["lr", "rf", "xgb", "svm", "elasticnet"]
MODEL_LABELS: Dict[str, str] = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "svm": "Support Vector Machine",
    "elasticnet": "ElasticNet",
}

# Same 6-feature panel as the LOCO heatmap (z-scored transcriptomic signatures)
FEATURE_COLS: List[str] = ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "IMPRES", "PD_L1"]

N_FOLDS: int = 5
RANDOM_STATE: int = 42

# Identical colour scheme to loco_performance_heatmap for direct visual comparison
HEATMAP_VMIN: float = 0.25
HEATMAP_VMAX: float = 0.75
HEATMAP_FMT: str = ".3f"
HEATMAP_CMAP: str = "YlGnBu"
HEATMAP_FIGSIZE: Tuple[int, int] = (9, 6)
HEATMAP_ANNOT_SIZE: int = 13
HEATMAP_LABEL_SIZE: int = 11

OUTPUT_PLOT_DIR: Path = PLOTS_DIR / "models"
OUTPUT_PATH: Path = OUTPUT_PLOT_DIR / "5f_cv_performance_heatmap.png"
LOG_DIR: Path = SUBPROJECT_ROOT / "logs"
LOG_PATH: Path = LOG_DIR / "generate_5f_cv_heatmap.log"


# ---------------------------------------------------------------------------
# Data Loading Helpers
# ---------------------------------------------------------------------------
def _load_and_prep_cohort(cohort_name: str, data_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads one cohort by name, drops NaN responders, and returns z-scored signatures and labels.

    Args:
        cohort_name: Cohort name as in datasets.yaml (e.g. 'Liu 2019').
        data_dir: Root data directory.

    Returns:
        Tuple of (signatures DataFrame, binary response Series).
    """
    expr, clin = load_cohort_by_name(cohort_name, data_dir)
    resp_col = "response" if "response" in clin.columns else "RESPONDER"
    mask = clin[resp_col].notna()
    sigs = zscore_df(extract_all_signatures(expr[mask]))
    y = clin[mask][resp_col].astype(int)
    return sigs, y


def _pool_cohorts(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Pools all active trial cohorts into one feature matrix and label vector.

    Args:
        data_dir: Root data directory.

    Returns:
        Tuple of (pooled signatures DataFrame, pooled response Series, trial_names) with reset indices.
    """
    config_path = SUBPROJECT_ROOT / "config" / "datasets.yaml"
    _, _, _, trial_names = load_all_active_cohorts(config_path, data_dir, merge_only=True)
    sigs_list, y_list = [], []
    for name in trial_names:
        sigs, y = _load_and_prep_cohort(name, data_dir)
        sigs_list.append(sigs)
        y_list.append(y)
    X_pooled = pd.concat(sigs_list, axis=0).reset_index(drop=True)
    y_pooled = pd.concat(y_list, axis=0).reset_index(drop=True)
    return X_pooled, y_pooled, trial_names


# ---------------------------------------------------------------------------
# 5-Fold CV Evaluation
# ---------------------------------------------------------------------------
def _run_5f_cv_for_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
) -> Dict:
    """Runs 5-fold stratified CV for a single model family and returns per-fold AUC scores.

    Features are scaled independently per fold (fit on train split, transform on val
    split) to prevent data leakage. Model tuning uses the same get_model() pipeline
    as LOCO evaluation for consistency.

    Args:
        X: Full pooled feature DataFrame (signature columns only).
        y: Full pooled binary response labels.
        model_type: One of 'lr', 'rf', 'xgb', 'svm', 'elasticnet'.

    Returns:
        Dict with keys 'fold_aucs' (list[float]), 'mean_auc' (float), 'std_auc' (float).
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    X_feat = X[FEATURE_COLS].values
    y_arr = y.values
    fold_aucs: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_feat, y_arr), start=1):
        X_train_raw, X_val_raw = X_feat[train_idx], X_feat[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        # Per-fold standardisation to prevent leakage across splits
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_raw), columns=FEATURE_COLS
        )
        X_val_scaled = scaler.transform(X_val_raw)

        model = get_model(model_type, X_train_scaled, pd.Series(y_train), calibrate=True)
        y_prob = model.predict_proba(X_val_scaled)[:, 1]

        try:
            auc = roc_auc_score(y_val, y_prob)
        except ValueError:
            auc = np.nan

        fold_aucs.append(auc)
        print(f"    Fold {fold_idx}/{N_FOLDS}: AUC = {auc:.3f}")

    mean_auc = float(np.nanmean(fold_aucs))
    std_auc = float(np.nanstd(fold_aucs))
    return {"fold_aucs": fold_aucs, "mean_auc": mean_auc, "std_auc": std_auc}


def _run_all_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
    """Runs 5-fold CV for all five model families and returns a results dictionary.

    Args:
        X: Pooled feature DataFrame.
        y: Pooled binary response labels.

    Returns:
        Dict mapping model key to cv result dict (fold_aucs, mean_auc, std_auc).
    """
    results: Dict[str, Dict] = {}
    for mtype in MODEL_ORDER:
        print(f"\n--- Running 5-Fold CV for Model: {MODEL_LABELS[mtype]} ---")
        results[mtype] = _run_5f_cv_for_model(X, y, mtype)
        r = results[mtype]
        print(f"  Mean AUC = {r['mean_auc']:.3f} +/- {r['std_auc']:.3f}")
    return results


# ---------------------------------------------------------------------------
# Heatmap Construction & Plotting
# ---------------------------------------------------------------------------
def _build_heatmap_dataframes(
    cv_results: Dict[str, Dict],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Constructs numeric and annotation DataFrames for seaborn heatmap rendering.

    Columns represent individual folds (Fold 1 to Fold 5) plus a summary Mean column.
    Rows represent model architectures in MODEL_ORDER. The mean column annotation
    includes +/- std for uncertainty display.

    Args:
        cv_results: Output of _run_all_models().

    Returns:
        Tuple of (df_values, df_annot):
            df_values: float AUC values for colour mapping.
            df_annot: pre-formatted string annotations per cell.
    """
    fold_cols = [f"Fold {i + 1}" for i in range(N_FOLDS)]
    mean_col = f"Mean ({N_FOLDS}-Fold)"

    rows_values, rows_annot = [], []

    for mtype in MODEL_ORDER:
        res = cv_results[mtype]
        fold_aucs = res["fold_aucs"]
        mean_auc = res["mean_auc"]
        std_auc = res["std_auc"]

        row_values = {col: auc for col, auc in zip(fold_cols, fold_aucs)}
        row_values[mean_col] = mean_auc

        row_annot = {col: f"{auc:.3f}" for col, auc in zip(fold_cols, fold_aucs)}
        row_annot[mean_col] = f"{mean_auc:.3f}\n+/-{std_auc:.3f}"

        rows_values.append({"Model": MODEL_LABELS[mtype], **row_values})
        rows_annot.append({"Model": MODEL_LABELS[mtype], **row_annot})

    df_values = pd.DataFrame(rows_values).set_index("Model")
    df_annot = pd.DataFrame(rows_annot).set_index("Model")
    return df_values, df_annot


def _plot_heatmap(
    cv_results: Dict[str, Dict],
    n_pooled: int,
    output_path: Path,
) -> None:
    """Renders and saves the 5-fold CV heatmap matching the loco_performance_heatmap style.

    Individual fold columns are coloured by AUC value. The final mean column is
    visually separated by a thicker white vertical line and annotated with mean +/- std.

    Args:
        cv_results: Output of _run_all_models().
        n_pooled: Total patient count in the pooled dataset (for the figure title).
        output_path: Destination path for the saved PNG.
    """
    df_values, df_annot = _build_heatmap_dataframes(cv_results)

    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)

    sns.heatmap(
        df_values,
        annot=df_annot,
        fmt="",
        cmap=HEATMAP_CMAP,
        cbar_kws={"label": "ROC-AUC Score"},
        linewidths=1.5,
        linecolor="white",
        annot_kws={"size": HEATMAP_ANNOT_SIZE, "weight": "bold"},
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        ax=ax,
    )

    # Thick separator before the mean summary column
    n_cols = df_values.shape[1]
    ax.axvline(n_cols - 1, color="white", linewidth=4.5, zorder=6)

    ax.set_title(
        f"5-Fold Stratified CV ROC-AUC Performance Across Model Architectures"
        f"\n(Pooled Immunotherapy Cohorts, N={n_pooled})",
        fontsize=HEATMAP_ANNOT_SIZE,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Cross-Validation Fold", fontsize=HEATMAP_LABEL_SIZE, fontweight="bold", labelpad=10)
    ax.set_ylabel("Model Architecture", fontsize=HEATMAP_LABEL_SIZE, fontweight="bold", labelpad=10)
    plt.xticks(fontsize=HEATMAP_LABEL_SIZE)
    plt.yticks(fontsize=HEATMAP_LABEL_SIZE, rotation=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    print(f"\nSaved 5-fold CV heatmap to {rel_path(output_path)}")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    """Runs the full 5-fold CV pipeline: load data, evaluate models, plot heatmap."""
    print("=" * 56)
    print("5-Fold Stratified CV: Curated Signatures Feature Panel")
    print("=" * 56)

    X_pooled, y_pooled, trial_names = _pool_cohorts(DATA_DIR)
    print(f"\nLoading and pooling active trial cohorts ({' + '.join(trial_names)})...")
    n_pooled = len(y_pooled)
    n_resp = int(y_pooled.sum())
    print(f"  Pooled dataset: N={n_pooled} patients, {n_resp} responders "
          f"({100 * y_pooled.mean():.1f}%)")

    cv_results = _run_all_models(X_pooled, y_pooled)
    _plot_heatmap(cv_results, n_pooled, OUTPUT_PATH)

    print("\n" + "=" * 56)
    print("Done!")
    print("=" * 56)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
