#!/usr/bin/env python3
"""
Create a cleaner, presentation-ready SOX10 proxy-target scatter plot.

Reads:
    outputs_proxy/results/all_sox10_codependencies.csv
    outputs_proxy/results/sox10_proxy_candidates_ranked.csv

Writes:
    outputs_proxy/figures/01_sox10_proxy_presentation_clean.png
    outputs_proxy/figures/01_sox10_proxy_presentation_clean.pdf

Design choices:
- Labels only the three prioritised druggable candidates.
- Keeps BRAF and MAPK1 as small positive-control labels.
- Uses numbers 1–3 on the plot and a compact evidence box beside it.
- Removes long overlapping callouts.
- Shades the candidate region.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parent
RES_DIR = ROOT / "outputs_proxy" / "results"
FIG_DIR = ROOT / "outputs_proxy" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALL_PATH = RES_DIR / "all_sox10_codependencies.csv"
CANDIDATE_PATH = RES_DIR / "sox10_proxy_candidates_ranked.csv"

X_COL = "pancancer_pearson_r"
Y_COL = "selectivity"

X_THRESHOLD = 0.20
Y_THRESHOLD = -0.10

# Only these genes receive large presentation annotations.
FOCUS = {
    "AMD1": {
        "number": "1",
        "subtitle": "Druggable metabolic enzyme",
        "marker": "*",
    },
    "PGM3": {
        "number": "2",
        "subtitle": "Druggable metabolic enzyme",
        "marker": "*",
    },
    "LCMT1": {
        "number": "3",
        "subtitle": "Druggable PP2A regulator",
        "marker": "*",
    },
}

# Small labels only; these demonstrate that the method recovers known biology.
CONTROLS = ["BRAF", "MAPK1"]

# Mechanistic support can be shown but without long annotations.
MECHANISTIC = ["DUSP4", "CRTC3"]

# Other genes that are still candidates but not selected as final druggable hits.
POORLY_DRUGGABLE = ["MITF", "ZEB2", "TFAP2A"]
EXPLORATORY = ["FERMT2", "PPM1G", "PSME3"]
DEPRIORITISED = ["PPP2R2A", "ELOA", "MAFF", "CRTC2"]


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    require(ALL_PATH)
    require(CANDIDATE_PATH)

    all_genes = pd.read_csv(ALL_PATH, index_col=0)
    candidates = pd.read_csv(CANDIDATE_PATH, index_col=0)

    for name, table in [("all genes", all_genes), ("candidates", candidates)]:
        missing = {X_COL, Y_COL} - set(table.columns)
        if missing:
            raise ValueError(
                f"{name} table is missing columns: {', '.join(sorted(missing))}"
            )

    return all_genes, candidates


def plot_group(
    ax: plt.Axes,
    table: pd.DataFrame,
    genes: list[str],
    *,
    marker: str,
    size: float,
    label: str,
    zorder: int,
    edge: bool = True,
) -> None:
    present = [gene for gene in genes if gene in table.index]
    if not present:
        return

    ax.scatter(
        table.loc[present, X_COL],
        table.loc[present, Y_COL],
        marker=marker,
        s=size,
        linewidths=0.8 if edge else 0,
        edgecolors="black" if edge else "none",
        label=label,
        zorder=zorder,
    )


def main() -> None:
    all_genes, candidates = load_tables()

    fig, ax = plt.subplots(figsize=(12.8, 7.2))

    # Shade the region that satisfies both primary proxy thresholds.
    ax.axvspan(
        X_THRESHOLD,
        max(0.66, all_genes[X_COL].max()),
        ymin=0,
        ymax=1,
        alpha=0.035,
        zorder=0,
    )
    ax.axhspan(
        min(-0.84, all_genes[Y_COL].min()),
        Y_THRESHOLD,
        xmin=0,
        xmax=1,
        alpha=0.035,
        zorder=0,
    )

    # Full gene background.
    background = all_genes.drop(
        index=candidates.index,
        errors="ignore",
    ).dropna(subset=[X_COL, Y_COL])

    ax.scatter(
        background[X_COL],
        background[Y_COL],
        s=8,
        alpha=0.075,
        linewidths=0,
        label="Other genes",
        zorder=1,
    )

    # Unselected candidate genes are visible but visually quiet.
    unselected = candidates.drop(
        index=(
            list(FOCUS)
            + CONTROLS
            + MECHANISTIC
            + POORLY_DRUGGABLE
            + EXPLORATORY
            + DEPRIORITISED
        ),
        errors="ignore",
    )

    if not unselected.empty:
        ax.scatter(
            unselected[X_COL],
            unselected[Y_COL],
            s=48,
            alpha=0.65,
            linewidths=0.7,
            edgecolors="black",
            label="Other proxy candidates",
            zorder=3,
        )

    plot_group(
        ax,
        candidates,
        CONTROLS,
        marker="s",
        size=80,
        label="Established target / control",
        zorder=5,
    )
    plot_group(
        ax,
        candidates,
        MECHANISTIC,
        marker="D",
        size=88,
        label="Mechanistic support",
        zorder=5,
    )
    plot_group(
        ax,
        candidates,
        POORLY_DRUGGABLE,
        marker="^",
        size=82,
        label="Poorly druggable lineage regulator",
        zorder=4,
    )
    plot_group(
        ax,
        candidates,
        EXPLORATORY,
        marker="o",
        size=78,
        label="Exploratory candidate",
        zorder=4,
    )
    plot_group(
        ax,
        candidates,
        DEPRIORITISED,
        marker="x",
        size=72,
        label="Deprioritised",
        zorder=4,
        edge=False,
    )

    # Plot and number only the final three druggable candidates.
    for gene, meta in FOCUS.items():
        if gene not in candidates.index:
            continue

        row = candidates.loc[gene]
        x = row[X_COL]
        y = row[Y_COL]

        ax.scatter(
            [x],
            [y],
            marker=meta["marker"],
            s=180,
            linewidths=1.0,
            edgecolors="black",
            zorder=8,
        )

        # Compact numbered circle near the point.
        ax.annotate(
            meta["number"],
            xy=(x, y),
            xytext=(8, 8),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox={
                "boxstyle": "circle,pad=0.24",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.8,
            },
            zorder=9,
        )

    # Small labels only for important controls/mechanistic validation.
    label_offsets = {
        "BRAF": (5, 5),
        "MAPK1": (5, 5),
        "DUSP4": (5, -11),
        "CRTC3": (5, 5),
    }

    for gene, offset in label_offsets.items():
        if gene not in candidates.index:
            continue
        row = candidates.loc[gene]
        ax.annotate(
            gene,
            xy=(row[X_COL], row[Y_COL]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            fontweight="bold" if gene in CONTROLS else "normal",
            zorder=7,
        )

    # Thresholds.
    ax.axvline(
        X_THRESHOLD,
        linestyle="--",
        linewidth=1.1,
        alpha=0.8,
    )
    ax.axhline(
        Y_THRESHOLD,
        linestyle="--",
        linewidth=1.1,
        alpha=0.8,
    )

    # Focus the displayed range on the biologically relevant candidate region,
    # while retaining enough background to show separation.
    ax.set_xlim(-0.10, 0.66)
    ax.set_ylim(-0.84, 0.08)

    ax.set_xlabel(
        "Pan-cancer co-dependency with SOX10\n"
        "(Pearson correlation of Chronos scores)"
    )
    ax.set_ylabel(
        "Melanoma selectivity\n"
        "(more negative = more melanoma-selective)"
    )
    ax.set_title(
        "SOX10 co-dependency identifies candidate therapeutic proxies",
        fontweight="bold",
        pad=12,
    )

    # Candidate-region label.
    ax.text(
        0.215,
        -0.805,
        "Candidate region",
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # Compact evidence panel placed outside the axes.
    evidence_lines = [
        r"$\bf{Prioritised\ candidates}$",
        "1  AMD1   — existing inhibitor",
        "2  PGM3   — preclinical inhibitor",
        "3  LCMT1  — early melanoma evidence",
        "",
        r"$\it{BRAF/MAPK1}$ recovered as controls",
    ]

    ax.text(
        1.025,
        0.44,
        "\n".join(evidence_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "white",
            "edgecolor": "0.65",
            "linewidth": 0.8,
        },
    )

    # Short legend: no long category names competing with the data.
    handles = [
        Line2D(
            [0], [0],
            marker="*",
            linestyle="",
            markersize=12,
            markeredgecolor="black",
            label="Prioritised druggable candidate",
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="",
            markersize=7,
            markeredgecolor="black",
            label="Established target / control",
        ),
        Line2D(
            [0], [0],
            marker="D",
            linestyle="",
            markersize=7,
            markeredgecolor="black",
            label="Mechanistic support",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=6,
            markeredgecolor="black",
            label="Other candidate",
        ),
    ]

    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=8.5,
    )

    # Reserve space on the right for the evidence box.
    fig.subplots_adjust(
        left=0.10,
        right=0.76,
        bottom=0.15,
        top=0.89,
    )

    png_path = FIG_DIR / "01_sox10_proxy_presentation_clean.png"
    pdf_path = FIG_DIR / "01_sox10_proxy_presentation_clean.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
