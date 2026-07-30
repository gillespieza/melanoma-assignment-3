"""Small reusable matplotlib/seaborn helpers for report-figure scripts."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns

from src.styles import get_cohort_color


def save_fig(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    """Standard tight-layout + save + close, so this isn't repeated per chart."""
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def resolve_colors(labels: List[str]) -> List[str]:
    """Resolve a color per label via get_cohort_color's substring match,

    falling back to a generated palette for any label not defined in styles.py.
    """
    fallback = sns.color_palette("Set2", n_colors=len(labels)).as_hex()
    return [get_cohort_color(label, default=fallback[i]) for i, label in enumerate(labels)]
