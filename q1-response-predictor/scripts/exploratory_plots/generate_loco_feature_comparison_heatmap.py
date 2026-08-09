"""Generates 1x2 Dual LOCO Heatmap Panel (Curated vs. SelectKBest per Cohort).

Left Panel (Panel A):
  Leave-One-Cohort-Out (LOCO) AUROC per held-out cohort for Curated Signatures.
  Columns: Active trial cohorts (N per cohort) + Mean LOCO.
  Rows: 5 model architectures + Cross-Model Mean.
  Cells: AUROC ± 1,000-sample bootstrap SD (* p < 0.05, ** p < 0.01 vs. chance).

Right Panel (Panel B):
  LOCO AUROC for SelectKBest (k=20, 100, 200) broken down per held-out cohort.
  Columns: N cohorts x 3 k-values (per-cohort columns) + 3 Mean LOCO columns (k=20, 100, 200).
  Rows: 5 model architectures + Cross-Model Mean.
  Cells: AUROC ± bootstrap SD (* p < 0.05, ** p < 0.01 vs. chance).

Uses get_model() for hyperparameter tuning & calibration consistency across all panels.

Outputs:
  - q1-response-predictor/plots/models/loco_dual_1x2_heatmap.png
  - q1-response-predictor/plots/models/loco_dual_1x2_heatmap_transparent.png
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
import scipy.stats as stats
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------
from src.data_loaders import load_all_active_cohorts
from src.models import get_model
from src.signatures import extract_all_signatures, zscore_df
from src.styles import set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, get_subproject_log_dir, PLOTS_DIR, rel_path
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

# Populated at runtime from datasets.yaml by _load_all_cohort_data()
COHORT_ORDER: List[str] = []

FEATURE_COLS_SIGS: List[str] = ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "IMPRES", "PD_L1"]
K_VALUES: List[int] = [20, 100, 200]
TOP_VAR_PREFILTER: int = 1000

RANDOM_STATE: int = 42
N_BOOTSTRAP: int = 1000

HEATMAP_VMIN: float = 0.30
HEATMAP_VMAX: float = 0.70
HEATMAP_CMAP: str = "YlGnBu"
HEATMAP_ANNOT_SIZE: int = 10
HEATMAP_LABEL_SIZE: int = 11

OUTPUT_PLOT_DIR: Path = PLOTS_DIR / "models"
OUTPUT_PATH: Path = OUTPUT_PLOT_DIR / "loco_dual_1x2_heatmap.png"
OUTPUT_TRANS_PATH: Path = OUTPUT_PLOT_DIR / "loco_dual_1x2_heatmap_transparent.png"
LOG_DIR: Path = get_subproject_log_dir(Path(__file__))
LOG_PATH: Path = LOG_DIR / "generate_loco_dual_1x2_heatmap.log"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def _load_all_cohort_data(data_dir: Path) -> Tuple[
    Dict[str, Tuple[pd.DataFrame, pd.Series]],
    Dict[str, Tuple[pd.DataFrame, pd.Series]],
    Dict[str, int],
]:
    """Loads per-cohort signature and expression DataFrames.

    Signatures are computed as in run_loco_cv to ensure AUC match.
    Expression datasets intersect common genes across cohorts for SelectKBest.
    Sets the module-level COHORT_ORDER from datasets.yaml trial names.
    """
    global COHORT_ORDER
    config_path = SUBPROJECT_ROOT / "config" / "datasets.yaml"
    expr_dict, clin_dict, _, trial_names = load_all_active_cohorts(
        config_path, data_dir, merge_only=True
    )
    COHORT_ORDER = trial_names

    cohort_sigs, cohort_expr, cohort_n = {}, {}, {}
    raw_expr = {}

    for name in trial_names:
        expr, clin = expr_dict[name], clin_dict[name]
        resp_col = "response" if "response" in clin.columns else "RESPONDER"
        y = clin[resp_col].dropna().astype(int)
        expr_m = expr.loc[y.index]

        sigs = zscore_df(extract_all_signatures(expr_m))
        cohort_sigs[name] = (sigs.reset_index(drop=True), y.reset_index(drop=True))
        cohort_n[name] = len(y)
        raw_expr[name] = (expr_m, y)

    common_genes = raw_expr[trial_names[0]][0].columns
    for name in trial_names[1:]:
        common_genes = common_genes.intersection(raw_expr[name][0].columns)
    common_genes = list(common_genes)
    print(f"  Common genes across all cohorts: {len(common_genes)}")

    for name in trial_names:
        expr_c = raw_expr[name][0][common_genes].reset_index(drop=True)
        y_r = raw_expr[name][1].reset_index(drop=True)
        cohort_expr[name] = (expr_c, y_r)

    return cohort_sigs, cohort_expr, cohort_n


# ---------------------------------------------------------------------------
# Statistical Helpers
# ---------------------------------------------------------------------------
def _bootstrap_auc_sd(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Estimates AUROC SD via 1,000-sample non-parametric bootstrap resampling."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    boot_aucs: List[float] = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = y_true[idx], y_prob[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            boot_aucs.append(roc_auc_score(y_b, p_b))
        except ValueError:
            continue
    return float(np.std(boot_aucs)) if boot_aucs else np.nan


def _mannwhitney_asterisk(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    """One-tailed Mann-Whitney U test for AUROC > 0.5. Returns significance asterisk."""
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
    return ""


# ---------------------------------------------------------------------------
# LOCO Evaluation: Curated Signatures (using get_model tuning)
# ---------------------------------------------------------------------------
def _run_loco_curated(
    cohort_sigs: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    model_type: str,
) -> Dict[str, Tuple[float, float, str]]:
    """LOCO evaluation using curated signatures with hyperparameter tuning."""
    results = {}
    for held_out in COHORT_ORDER:
        train_names = [c for c in COHORT_ORDER if c != held_out]

        X_train = pd.concat([cohort_sigs[c][0][FEATURE_COLS_SIGS] for c in train_names]).values
        y_train = pd.concat([cohort_sigs[c][1] for c in train_names]).values
        X_test = cohort_sigs[held_out][0][FEATURE_COLS_SIGS].values
        y_test = cohort_sigs[held_out][1].values

        scaler = StandardScaler()
        X_tr_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS_SIGS)
        X_te_sc = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS_SIGS)

        model = get_model(model_type, X_tr_sc, pd.Series(y_train))
        y_prob = model.predict_proba(X_te_sc)[:, 1]

        results[held_out] = (
            roc_auc_score(y_test, y_prob),
            _bootstrap_auc_sd(y_test, y_prob),
            _mannwhitney_asterisk(y_test, y_prob),
        )
    return results


# ---------------------------------------------------------------------------
# LOCO Evaluation: SelectKBest (using get_model tuning)
# ---------------------------------------------------------------------------
def _run_loco_selectkbest(
    cohort_expr: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    k: int,
    model_type: str,
) -> Dict[str, Tuple[float, float, str]]:
    """LOCO evaluation using SelectKBest with hyperparameter tuning."""
    results = {}
    for held_out in COHORT_ORDER:
        train_names = [c for c in COHORT_ORDER if c != held_out]

        X_train = np.vstack([cohort_expr[c][0].values for c in train_names])
        y_train = np.concatenate([cohort_expr[c][1].values for c in train_names])
        X_test = cohort_expr[held_out][0].values
        y_test = cohort_expr[held_out][1].values

        # Pre-filter top-variance genes on train split
        top_idx = np.argsort(X_train.var(axis=0))[::-1][:min(TOP_VAR_PREFILTER, X_train.shape[1])]
        X_tr_filt = X_train[:, top_idx]
        X_te_filt = X_test[:, top_idx]

        # SelectKBest on train split
        k_val = min(k, X_tr_filt.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k_val)
        X_tr_sel = selector.fit_transform(X_tr_filt, y_train)
        X_te_sel = selector.transform(X_te_filt)

        cols = [f"gene_{i}" for i in range(X_tr_sel.shape[1])]
        X_tr_df = pd.DataFrame(X_tr_sel, columns=cols)
        X_te_df = pd.DataFrame(X_te_sel, columns=cols)

        model = get_model(model_type, X_tr_df, pd.Series(y_train))
        y_prob = model.predict_proba(X_te_df)[:, 1]

        results[held_out] = (
            roc_auc_score(y_test, y_prob),
            _bootstrap_auc_sd(y_test, y_prob),
            _mannwhitney_asterisk(y_test, y_prob),
        )
    return results


# ---------------------------------------------------------------------------
# Build DataFrames for Each Panel
# ---------------------------------------------------------------------------
def _build_left_panel(
    cohort_sigs: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    cohort_n: Dict[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Panel A: Curated Signatures LOCO per cohort (4 columns)."""
    rows_num, rows_ann = [], []

    for mtype in MODEL_ORDER:
        lbl = MODEL_LABELS_WRAPPED[mtype]
        res = _run_loco_curated(cohort_sigs, mtype)

        row_n, row_a = {"Model": lbl}, {"Model": lbl}
        cohort_aucs = []
        for c in COHORT_ORDER:
            auc, sd, ast = res[c]
            col = f"{c}\n(N={cohort_n[c]})"
            row_n[col] = auc
            row_a[col] = f"{auc:.3f}\n±{sd:.3f}{ast}"
            cohort_aucs.append(auc)

        mean_l = float(np.mean(cohort_aucs))
        std_l = float(np.std(cohort_aucs))
        row_n["Mean\nLOCO"] = mean_l
        row_a["Mean\nLOCO"] = f"{mean_l:.3f}\n±{std_l:.3f}"

        rows_num.append(row_n)
        rows_ann.append(row_a)

    df_num = pd.DataFrame(rows_num).set_index("Model")
    df_ann = pd.DataFrame(rows_ann).set_index("Model")

    cm = df_num.mean(axis=0)
    cs = df_num.std(axis=0)
    cm_num = cm.to_frame().T
    cm_num.index = ["Cross-Model\nMean"]
    cm_ann = pd.DataFrame(
        [{col: f"{cm[col]:.3f}\n±{cs[col]:.3f}" for col in df_num.columns}],
        index=["Cross-Model\nMean"],
    )
    df_num = pd.concat([df_num, cm_num])
    df_ann = pd.concat([df_ann, cm_ann])

    return df_num, df_ann


def _build_right_panel(
    cohort_expr: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    cohort_n: Dict[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Panel B: SelectKBest LOCO per cohort for k=20, 100, 200 (12 columns)."""
    rows_num, rows_ann = [], []

    for mtype in MODEL_ORDER:
        lbl = MODEL_LABELS_WRAPPED[mtype]
        print(f"\n  --- {MODEL_LABELS[mtype]} ---")

        row_n, row_a = {"Model": lbl}, {"Model": lbl}

        # Run LOCO for each k value
        k_results = {k: _run_loco_selectkbest(cohort_expr, k, mtype) for k in K_VALUES}

        # Per-cohort columns for each k
        for c in COHORT_ORDER:
            for k in K_VALUES:
                auc, sd, ast = k_results[k][c]
                col = f"{c}\nk={k}"
                row_n[col] = auc
                row_a[col] = f"{auc:.3f}\n±{sd:.3f}{ast}"

        # Summary Mean LOCO columns for each k
        for k in K_VALUES:
            k_aucs = [k_results[k][c][0] for c in COHORT_ORDER]
            mean_k = float(np.mean(k_aucs))
            std_k = float(np.std(k_aucs))
            col = f"Mean LOCO\nk={k}"
            row_n[col] = mean_k
            row_a[col] = f"{mean_k:.3f}\n±{std_k:.3f}"
            print(f"    SelectKBest k={k:>3}: Mean LOCO = {mean_k:.3f} ± {std_k:.3f}")

        rows_num.append(row_n)
        rows_ann.append(row_a)

    df_num = pd.DataFrame(rows_num).set_index("Model")
    df_ann = pd.DataFrame(rows_ann).set_index("Model")

    cm = df_num.mean(axis=0)
    cs = df_num.std(axis=0)
    cm_num = cm.to_frame().T
    cm_num.index = ["Cross-Model\nMean"]
    cm_ann = pd.DataFrame(
        [{col: f"{cm[col]:.3f}\n±{cs[col]:.3f}" for col in df_num.columns}],
        index=["Cross-Model\nMean"],
    )
    df_num = pd.concat([df_num, cm_num])
    df_ann = pd.concat([df_ann, cm_ann])

    return df_num, df_ann


# ---------------------------------------------------------------------------
# Render 1x2 Dual LOCO Panel (width ratio 1 : 3)
# ---------------------------------------------------------------------------
def _render_dual_loco_panel(
    df_left_num: pd.DataFrame,
    df_left_ann: pd.DataFrame,
    df_right_num: pd.DataFrame,
    df_right_ann: pd.DataFrame,
) -> None:
    """Renders 1x2 dual LOCO heatmap with 1:3 width ratio."""
    n_data = df_left_num.shape[0] - 1

    # 1x2 grid with 1:3 width ratio (Left panel = 4 cols, Right panel = 12 cols)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(28, 8.5), gridspec_kw={"width_ratios": [1, 3]}
    )

    # --- Left Panel: Curated Signatures LOCO ---
    sns.heatmap(
        df_left_num,
        annot=df_left_ann,
        fmt="",
        cmap=HEATMAP_CMAP,
        cbar=False,
        linewidths=1.0,
        linecolor="white",
        annot_kws={"size": HEATMAP_ANNOT_SIZE, "weight": "bold"},
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        ax=ax1,
    )
    n_cols_left = df_left_num.shape[1]
    for c_idx in range(n_cols_left - 1):
        col_data = df_left_num.iloc[:n_data, c_idx]
        ax1.add_patch(plt.Rectangle(
            (c_idx, col_data.values.argmax()), 1, 1,
            fill=False, edgecolor="black", linewidth=2.2,
        ))
    ax1.axvline(n_cols_left - 1, color="white", linewidth=7.0, zorder=6)
    ax1.axhline(n_data, color="white", linewidth=7.0, zorder=6)

    ax1.set_title(
        "A) Curated Signatures LOCO AUROC per Cohort\n"
        "(* p < 0.05, ** p < 0.01 vs. chance; ±SD from 1,000-sample bootstrap)",
        fontsize=11, fontweight="bold", pad=12,
    )
    ax1.set_xlabel("Held out test cohort and Curated Signatures", fontsize=HEATMAP_LABEL_SIZE, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Model Architecture", fontsize=HEATMAP_LABEL_SIZE, fontweight="bold", labelpad=8)
    ax1.tick_params(axis="y", labelrotation=0, labelsize=10)
    ax1.tick_params(axis="x", labelsize=9)

    # --- Right Panel: SelectKBest LOCO per Cohort (3x wider) ---
    sns.heatmap(
        df_right_num,
        annot=df_right_ann,
        fmt="",
        cmap=HEATMAP_CMAP,
        cbar_kws={"label": "AUROC Score"},
        linewidths=1.0,
        linecolor="white",
        annot_kws={"size": 9, "weight": "bold"},
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        ax=ax2,
    )
    n_cols_right = df_right_num.shape[1]
    # Bold outline best k per model for each cohort section
    for r_idx in range(n_data):
        row_data = df_right_num.iloc[r_idx]
        best_col = row_data.values.argmax()
        ax2.add_patch(plt.Rectangle(
            (best_col, r_idx), 1, 1,
            fill=False, edgecolor="black", linewidth=2.2,
        ))

    # Thick vertical lines separating cohort groups (every 3 columns)
    for v_line in range(3, n_cols_right + 1, 3):
        if v_line <= n_cols_right:
            ax2.axvline(v_line, color="white", linewidth=5.0, zorder=6)
    ax2.axvline(n_cols_right - 3, color="white", linewidth=7.0, zorder=6)
    ax2.axhline(n_data, color="white", linewidth=7.0, zorder=6)

    ax2.set_title(
        "B) SelectKBest LOCO AUROC per Cohort (k=20, 100, 200)\n"
        "(* p < 0.05, ** p < 0.01 vs. chance; ±SD from 1,000-sample bootstrap)",
        fontsize=11, fontweight="bold", pad=12,
    )
    ax2.set_xlabel("Held-Out Cohort & SelectKBest Features (k)", fontsize=HEATMAP_LABEL_SIZE, fontweight="bold", labelpad=8)
    ax2.set_ylabel("")
    ax2.tick_params(axis="y", labelrotation=0, labelsize=10)
    ax2.tick_params(axis="x", labelrotation=45, labelsize=8.5)

    plt.tight_layout(w_pad=2.5)

    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    save_fig(fig, OUTPUT_PATH)

    # Transparent version
    fig.patch.set_alpha(0.0)
    ax1.patch.set_alpha(0.0)
    ax2.patch.set_alpha(0.0)
    fig.savefig(OUTPUT_TRANS_PATH, transparent=True, bbox_inches="tight", dpi=300)

    plt.close(fig)
    print(f"\nSaved dual LOCO panel to {rel_path(OUTPUT_PATH)}")
    print(f"Saved transparent copy to {rel_path(OUTPUT_TRANS_PATH)}")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    """Runs the full 1x2 dual LOCO panel pipeline."""
    print("=" * 60)
    print("Dual LOCO Panel: Curated vs. SelectKBest per Cohort (1:3 Width Ratio)")
    print("=" * 60)

    print("\nLoading per-cohort datasets...")
    cohort_sigs, cohort_expr, cohort_n = _load_all_cohort_data(DATA_DIR)

    for name in COHORT_ORDER:
        n = cohort_n[name]
        n_pos = int(cohort_sigs[name][1].sum())
        print(f"  {name}: N={n}, responders={n_pos} ({100 * n_pos / n:.1f}%)")

    print("\n--- Building Left Panel (Curated Signatures LOCO) ---")
    df_left_num, df_left_ann = _build_left_panel(cohort_sigs, cohort_n)

    print("\n--- Building Right Panel (SelectKBest LOCO per Cohort) ---")
    df_right_num, df_right_ann = _build_right_panel(cohort_expr, cohort_n)

    print("\n\nRendering 1x2 Dual LOCO Heatmap Panel (Width Ratio 1:3)...")
    _render_dual_loco_panel(df_left_num, df_left_ann, df_right_num, df_right_ann)

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
