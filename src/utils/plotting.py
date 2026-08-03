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
