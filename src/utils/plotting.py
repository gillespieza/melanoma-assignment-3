"""Small reusable matplotlib/seaborn helpers for report-figure scripts."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

from src.styles import get_cohort_color


def save_fig(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    """Standard tight-layout + save + close, saving exclusively to q1-response-predictor/plots/."""
    fig.tight_layout()
    path = Path(path).resolve()
    
    from src.utils.paths import PROJECT_ROOT
    q1_plots = PROJECT_ROOT / "q1-response-predictor" / "plots"
    
    path_str = path.as_posix()
    if "/plots/" in path_str:
        rel_plots_part = path_str.split("/plots/")[-1]
        target_path = q1_plots / rel_plots_part
    else:
        target_path = path
    
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def resolve_colors(labels: List[str]) -> List[str]:
    """Resolve a color per label via get_cohort_color's substring match,

    falling back to a generated palette for any label not defined in styles.py.
    """
    fallback = sns.color_palette("Set2", n_colors=len(labels)).as_hex()
    return [get_cohort_color(label, default=fallback[i]) for i, label in enumerate(labels)]


def build_radar_angles(n_vars: int) -> List[float]:
    """Computes closed-loop angle values (in radians) for polar radar charts.

    Args:
        n_vars: Number of variables/axes on the radar chart.

    Returns:
        List of angle values with the starting angle repeated at the end to close the loop.
    """
    import numpy as np

    angles = [n / float(n_vars) * 2 * np.pi for n in range(n_vars)]
    angles += angles[:1]
    return angles


def add_km_risk_table(
    kmf_objects: list,
    ax: "plt.Axes",
    fontsize: int = 9,
) -> None:
    """Adds a numbers-at-risk table beneath a Kaplan-Meier survival plot axis.

    This is the clinical gold standard for communicating declining patient counts
    at later time points in pooled multi-cohort KM plots, making it immediately
    visible when late-curve estimates are based on very few patients.

    Must be called BEFORE save_fig() / tight_layout() so the sub-axis is
    positioned correctly relative to the KM panel.

    Args:
        kmf_objects: List of fitted KaplanMeierFitter instances, one per group.
                     Each must already have been fitted (kmf.fit(...) called).
        ax: The matplotlib Axes object the KM curves were plotted on.
        fontsize: Font size for the risk table row numbers. Defaults to 9.
    """
    from lifelines.plotting import add_at_risk_counts

    if not kmf_objects:
        return
    add_at_risk_counts(
        *kmf_objects,
        ax=ax,
        fontsize=fontsize,
        rows_to_show=["At risk"],
    )
