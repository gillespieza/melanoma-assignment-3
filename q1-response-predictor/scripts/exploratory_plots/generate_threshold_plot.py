"""Generates Default vs. Optimal (Youden's J) Threshold Tuning Impact Plot.

Runs LOCO cross-validation for the Random Forest model across all active trial
cohorts and computes classification accuracy and sensitivity under both the
default 0.50 threshold and the Youden's J-optimal threshold. All metrics and
cohort sample sizes are computed at runtime from live data — no values are
hardcoded.
"""

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_SUBPROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROJECT_ROOT = _SUBPROJECT_ROOT.parent

if str(_SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBPROJECT_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
from sklearn.metrics import accuracy_score, recall_score

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------
from src.data_loaders import load_all_active_cohorts, load_cohort_by_name
from src.evaluation import find_optimal_threshold
from src.models import get_model, run_loco_cv
from src.signatures import extract_all_signatures, zscore_df
from src.styles import THRESHOLD_STRATEGY_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, PLOTS_DIR, rel_path
from src.utils.plotting import save_fig

set_presentation_style()

# ---------------------------------------------------------------------------
# Module-level Constants & Configuration
# ---------------------------------------------------------------------------
_CONFIG_PATH: Path = _SUBPROJECT_ROOT / "config" / "datasets.yaml"

# Random Forest is the focus of this threshold tuning comparison plot
_MODEL_TYPE: str = "rf"
_MODEL_LABEL: str = "Random Forest"

# Transcriptomic signature features (canonical order from AGENTS.md)
_FEATURE_COLS: List[str] = ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "IMPRES", "PD_L1"]

_STRATEGY_DEFAULT: str = "Default (0.50)"
_STRATEGY_OPTIMAL: str = "Optimal (Youden's J)"

LOG_DIR: Path = _SUBPROJECT_ROOT / "logs"
LOG_PATH: Path = LOG_DIR / "generate_threshold_plot.log"
_OUTPUT_PATH: Path = PLOTS_DIR / "models" / "threshold_tuning_impact.png"


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------
def _load_and_prep_cohort(cohort_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads a cohort, filters valid responders, and computes z-scored signatures.

    Args:
        cohort_name: Human-readable cohort name.

    Returns:
        Tuple of (signature DataFrame, binary response Series).
    """
    expr, clin = load_cohort_by_name(cohort_name, DATA_DIR, _CONFIG_PATH)
    resp_col = "response" if "response" in clin.columns else "RESPONDER"
    mask = clin[resp_col].notna()
    expr_clean, clin_clean = expr[mask], clin[mask]
    sigs = zscore_df(extract_all_signatures(expr_clean))
    y = clin_clean[resp_col].astype(int)
    return sigs, y


# ---------------------------------------------------------------------------
# Threshold Metric Computation
# ---------------------------------------------------------------------------
def _compute_threshold_comparison(
    cohort_dfs: Dict[str, Tuple[pd.DataFrame, pd.Series]],
) -> pd.DataFrame:
    """Runs LOCO CV for RF and computes per-cohort accuracy and sensitivity
    under both default (0.50) and Youden's J-optimal thresholds.

    Args:
        cohort_dfs: Mapping of cohort name to (X_signatures, y_response) tuples.

    Returns:
        DataFrame with columns: Cohort, Strategy, Accuracy, Sensitivity.
    """
    loco_results = run_loco_cv(cohort_dfs, _FEATURE_COLS, model_type=_MODEL_TYPE)
    rows: List[Dict] = []

    for cohort_name, res in loco_results.items():
        y_true: np.ndarray = res["y_true"]
        y_pred_prob: np.ndarray = res["y_pred_prob"]
        n_samples: int = len(y_true)
        cohort_label: str = f"{cohort_name}\n(N={n_samples})"

        for strategy, threshold in [
            (_STRATEGY_DEFAULT, 0.5),
            (_STRATEGY_OPTIMAL, find_optimal_threshold(y_true, y_pred_prob)),
        ]:
            y_pred = (y_pred_prob >= threshold).astype(int)
            rows.append({
                "Cohort": cohort_label,
                "Strategy": strategy,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot Generation
# ---------------------------------------------------------------------------
def _annotate_bars(ax: plt.Axes) -> None:
    """Adds percentage value labels above each bar in a barplot axes.

    Args:
        ax: Matplotlib axes containing bar patches to annotate.
    """
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isnan(height):
            y_pos = max(height, 0.01)
            ax.annotate(
                f"{height:.1%}",
                (patch.get_x() + patch.get_width() / 2.0, y_pos),
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                xytext=(0, 2),
                textcoords="offset points",
            )


def _build_palette() -> List[str]:
    """Returns the two-colour palette list for Default vs Optimal strategies.

    Returns:
        List of two hex colour strings from THRESHOLD_STRATEGY_PALETTE.
    """
    return [
        THRESHOLD_STRATEGY_PALETTE[_STRATEGY_DEFAULT],
        THRESHOLD_STRATEGY_PALETTE[_STRATEGY_OPTIMAL],
    ]


def plot_threshold_comparison(df: pd.DataFrame, output_path: Path) -> Path:
    """Renders the dual-panel threshold tuning impact bar chart and saves it.

    Panel 1 — Classification Accuracy: default 0.50 vs Youden's J threshold.
    Panel 2 — Sensitivity Recovery: default 0.50 vs Youden's J threshold.

    Args:
        df: DataFrame returned by _compute_threshold_comparison.
        output_path: Destination path for the saved PNG.

    Returns:
        The resolved output path.
    """
    palette = _build_palette()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # Panel 1: Accuracy
    sns.barplot(
        data=df,
        x="Cohort",
        y="Accuracy",
        hue="Strategy",
        palette=palette,
        ax=axes[0],
        width=0.75,
        edgecolor="white",
        linewidth=1.2,
    )
    axes[0].set_title(
        f"Classification Accuracy: Default (0.50) vs. Optimal Threshold\n({_MODEL_LABEL})",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    axes[0].set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("", fontsize=11)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(title="Threshold Strategy", frameon=True)
    _annotate_bars(axes[0])

    # Panel 2: Sensitivity
    sns.barplot(
        data=df,
        x="Cohort",
        y="Sensitivity",
        hue="Strategy",
        palette=palette,
        ax=axes[1],
        width=0.75,
        edgecolor="white",
        linewidth=1.2,
    )
    axes[1].set_title(
        f"Sensitivity Recovery: Default (0.50) vs. Optimal Threshold\n({_MODEL_LABEL})",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    axes[1].set_ylabel("Sensitivity (True Positive Rate)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("", fontsize=11)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(title="Threshold Strategy", frameon=True)
    _annotate_bars(axes[1])

    sns.despine(top=True, right=True)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_fig(fig, output_path)
    print(f"Saved threshold tuning comparison plot to {rel_path(output_path)}")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    """Orchestrates data loading, LOCO evaluation, and plot generation."""
    _, _, _, trial_names = load_all_active_cohorts(_CONFIG_PATH, DATA_DIR, merge_only=True)

    if not trial_names:
        raise RuntimeError(
            "No active immunotherapy trial cohorts found in config/datasets.yaml. "
            "Ensure at least one cohort has merge_enabled=true and processed files exist."
        )

    print(f"Active trial cohorts for threshold analysis: {trial_names}")

    cohort_dfs: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {
        name: _load_and_prep_cohort(name)
        for name in trial_names
    }

    df_comparison = _compute_threshold_comparison(cohort_dfs)
    plot_threshold_comparison(df_comparison, _OUTPUT_PATH)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(_SUBPROJECT_ROOT).as_posix()}")
            main()
