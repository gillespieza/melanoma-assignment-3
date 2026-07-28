"""Small reusable matplotlib/seaborn helpers for report-figure scripts."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.styles import get_cohort_color


def save_fig(fig, path: Path) -> None:
    """Standard tight-layout + save + close, so this isn't repeated per chart."""
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def resolve_colors(labels: list[str]) -> list[str]:
    """
    Resolve a color per label via get_cohort_color's substring match
    (robust to N's baked into the label, e.g. "Liu 2019 (N=103)"),
    falling back to a generated palette for any label not defined
    in styles.py.
    """
    fallback = sns.color_palette("Set2", n_colors=len(labels)).as_hex()
    return [get_cohort_color(label, default=fallback[i]) for i, label in enumerate(labels)]