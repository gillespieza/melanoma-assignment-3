"""Generates Leave-One-Cohort-Out (LOCO) ROC-AUC Performance Heatmap.

Evaluates multi-modal machine learning model architectures (Logistic Regression, Random Forest,
XGBoost, SVM, ElasticNet) across cross-cohort validation splits and plots the LOCO performance matrix.
"""

# ---------------------------------------------------------------------------
# Bootstrap & Root Search Path Setup
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
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.evaluation import calculate_extended_metrics
from src.models import run_loco_cv
from src.signatures import extract_all_signatures, zscore_df
from src.styles import set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, PLOTS_DIR, SUBPROJECT_ROOT, rel_path
from src.utils.plotting import save_fig

set_presentation_style()

# ---------------------------------------------------------------------------
# Module-level Constants & Configuration
# ---------------------------------------------------------------------------
MODEL_ORDER: List[str] = ["lr", "rf", "xgb", "svm", "elasticnet"]
MODEL_LABELS: Dict[str, str] = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
    "svm": "Support Vector Machine",
    "elasticnet": "ElasticNet",
}
DEFAULT_FEATURE_COLS: List[str] = ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "IMPRES", "PD_L1"]

HEATMAP_VMIN: float = 0.25
HEATMAP_VMAX: float = 0.75
HEATMAP_FMT: str = ".3f"
HEATMAP_CMAP: str = "YlGnBu"
HEATMAP_FIGSIZE: Tuple[int, int] = (11, 6)
HEATMAP_ANNOT_SIZE: int = 13
HEATMAP_LABEL_SIZE: int = 11

OUTPUT_PLOT_DIR: Path = PLOTS_DIR / "models"
DEFAULT_OUTPUT_PATH: Path = OUTPUT_PLOT_DIR / "loco_performance_heatmap.png"
LOG_DIR: Path = SUBPROJECT_ROOT / "logs"
LOG_PATH: Path = LOG_DIR / "generate_loco_heatmap.log"


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _load_and_prep_cohort(
    loader_func: Any,
    data_dir: Path,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads a cohort dataset, filters non-null responder targets, and computes z-scored signatures."""
    expr, clin = loader_func(data_dir)
    mask = clin["RESPONDER"].notna()
    expr_clean, clin_clean = expr[mask], clin[mask]
    sigs = zscore_df(extract_all_signatures(expr_clean))
    y = clin_clean["RESPONDER"].astype(int)
    return sigs, y


def _build_loco_summary_dataframe(
    all_loco_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Transforms raw LOCO evaluation results into a structured summary DataFrame for heatmaps."""
    all_cohorts = sorted({c for res in all_loco_results.values() for c in res})
    rows = []

    for mkey in MODEL_ORDER:
        if mkey not in all_loco_results:
            continue
        row_dict = {"Model": MODEL_LABELS[mkey]}
        loco = all_loco_results[mkey]
        for c in all_cohorts:
            res = loco.get(c, {})
            n_samples = len(res["y_true"]) if "y_true" in res else 0
            col_name = f"{c} (N={n_samples})"
            metrics = res.get("metrics_extended") or (
                calculate_extended_metrics(res["y_true"], res["y_pred_prob"]) if res else None
            )
            row_dict[col_name] = metrics["auc"] if metrics else np.nan
        rows.append(row_dict)

    return pd.DataFrame(rows).set_index("Model")


# ---------------------------------------------------------------------------
# Main Plotting & Execution Pipelines
# ---------------------------------------------------------------------------
def plot_loco_heatmap_from_results(
    all_loco_results: Dict[str, Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    """Generates a LOCO ROC-AUC performance heatmap dynamically from live evaluation results."""
    target_path = output_path or DEFAULT_OUTPUT_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df_heatmap = _build_loco_summary_dataframe(all_loco_results)

    fig, ax_heatmap = plt.subplots(figsize=HEATMAP_FIGSIZE)
    sns.heatmap(
        df_heatmap,
        annot=True,
        fmt=HEATMAP_FMT,
        cmap=HEATMAP_CMAP,
        cbar_kws={"label": "LOCO ROC-AUC Score"},
        linewidths=1.5,
        linecolor="white",
        annot_kws={"size": HEATMAP_ANNOT_SIZE, "weight": "bold"},
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        ax=ax_heatmap,
    )

    plt.title("Leave-One-Cohort-Out (LOCO) ROC-AUC Performance Across Models & Held-Out Cohorts", fontsize=HEATMAP_ANNOT_SIZE, fontweight="bold", pad=15)
    plt.xlabel("Held-Out Test Cohort", fontsize=HEATMAP_LABEL_SIZE, fontweight="bold", labelpad=10)
    plt.ylabel("Model Architecture", fontsize=HEATMAP_LABEL_SIZE, fontweight="bold", labelpad=10)
    plt.xticks(fontsize=HEATMAP_LABEL_SIZE)
    plt.yticks(fontsize=HEATMAP_LABEL_SIZE, rotation=0)

    save_fig(fig, target_path)
    print(f"Saved LOCO performance heatmap dynamically to {rel_path(target_path)}")
    return target_path


def generate_loco_heatmap_standalone() -> Path:
    """Standalone runner that computes LOCO cross-validation on live datasets and plots heatmap."""
    sigs_liu, y_liu = _load_and_prep_cohort(load_liu_2019, DATA_DIR)
    sigs_hugo, y_hugo = _load_and_prep_cohort(load_hugo_2016, DATA_DIR)
    sigs_riaz, y_riaz = _load_and_prep_cohort(load_riaz_2017, DATA_DIR)

    cohort_dfs = {
        "Liu 2019": (sigs_liu, y_liu),
        "Hugo 2016": (sigs_hugo, y_hugo),
        "Riaz 2017": (sigs_riaz, y_riaz),
    }

    all_loco_results = {
        mtype: run_loco_cv(cohort_dfs, DEFAULT_FEATURE_COLS, model_type=mtype)
        for mtype in MODEL_ORDER
    }

    return plot_loco_heatmap_from_results(all_loco_results)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            generate_loco_heatmap_standalone()
