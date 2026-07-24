"""
Waffle Chart Visualisation for Patient Response Distribution across Cohorts.

Renders a 3-panel waffle chart displaying individual patient response counts
(1 square = 1 patient) for Liu 2019, Hugo 2016, and Riaz 2017 cohorts using Okabe-Ito colors.
"""

import contextlib
from pathlib import Path
import sys
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.styles import RESPONSE_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import find_project_root
from src.utils.plotting import save_fig

# Module-level Constants
PLOT_DIR = find_project_root(Path(__file__).resolve()) / "plots" / "clinical"
LOG_DIR = find_project_root(Path(__file__).resolve()) / "logs"
LOG_PATH = LOG_DIR / "run_waffle_chart.log"


def draw_waffle_cohort(ax: plt.Axes, n_resp: int, n_nr: int, title: str, n_cols: int = 10) -> None:
    """Draws a waffle chart subplot for one cohort.

    Args:
        ax: Matplotlib Axes to draw on.
        n_resp: Number of responder patients.
        n_nr: Number of non-responder patients.
        title: Subplot title string.
        n_cols: Number of columns in waffle grid.
    """
    total = n_resp + n_nr
    c_resp = RESPONSE_PALETTE["CR/PR"]
    c_nr = RESPONSE_PALETTE["PD"]

    block_colors = [c_resp] * n_resp + [c_nr] * n_nr
    n_rows = int(np.ceil(total / n_cols))

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)

    for idx in range(total):
        row = idx // n_cols
        col = idx % n_cols
        color = block_colors[idx]

        rect = plt.Rectangle(
            (col - 0.4, row - 0.4),
            0.8,
            0.8,
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
        )
        ax.add_patch(rect)

    ax.set_aspect("equal")
    ax.axis("off")

    pct_resp = (n_resp / total) * 100
    subtext = f"Responders: {n_resp} ({pct_resp:.1f}%)\nNon-responders: {n_nr}"
    ax.text(
        n_cols / 2 - 0.5,
        n_rows - 0.1,
        f"{title} (N={total})",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        n_cols / 2 - 0.5,
        -0.6,
        subtext,
        ha="center",
        va="top",
        fontsize=11,
        fontstyle="italic",
        linespacing=1.3,
    )


def main() -> None:
    """Executes the waffle chart generation pipeline."""
    print("==================================================")
    print("Generating Waffle Chart for Presentation")
    print("==================================================\n")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    set_presentation_style()

    cohorts: Dict[str, Dict[str, int]] = {
        "Liu 2019": {"resp": 48, "nr": 56, "cols": 10},
        "Hugo 2016": {"resp": 13, "nr": 13, "cols": 10},
        "Riaz 2017": {"resp": 5, "nr": 15, "cols": 10},
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 7))

    draw_waffle_cohort(axes[0], cohorts["Liu 2019"]["resp"], cohorts["Liu 2019"]["nr"], "Liu 2019", n_cols=10)
    draw_waffle_cohort(axes[1], cohorts["Hugo 2016"]["resp"], cohorts["Hugo 2016"]["nr"], "Hugo 2016", n_cols=5)
    draw_waffle_cohort(axes[2], cohorts["Riaz 2017"]["resp"], cohorts["Riaz 2017"]["nr"], "Riaz 2017", n_cols=5)

    fig.suptitle("Patient Response Distribution (1 block = 1 patient)", fontsize=18, fontweight="bold", y=0.95)

    legend_patches = [
        mpatches.Patch(color=RESPONSE_PALETTE["CR/PR"], label="Responder (CR/PR)"),
        mpatches.Patch(color=RESPONSE_PALETTE["PD"], label="Non-responder (PD)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.05))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.8)

    out_path = PLOT_DIR / "response_waffle_chart.png"
    save_fig(fig, out_path)
    print(f"Saved waffle chart to {out_path.relative_to(BASE_DIR).as_posix()}")

    print("==================================================")
    print("Done!")
    print("==================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(BASE_DIR).as_posix()}")
            main()
