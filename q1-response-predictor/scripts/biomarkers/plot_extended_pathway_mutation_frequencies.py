import contextlib
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Add project root to sys.path for importing src modules (search upward for
# markers instead of assuming a fixed parent.parent.parent depth)
_THIS_FILE = Path(__file__).resolve()
for _candidate in [_THIS_FILE.parent] + list(_THIS_FILE.parent.parents):
    if (_candidate / "src").exists() and (_candidate / "data").exists():
        BASE_DIR = _candidate
        break
else:
    raise FileNotFoundError(f"Could not locate project root above {_THIS_FILE}")

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.biology_constants import COHORT_DIRS, NON_SILENT, DRIVER_GENES, PATHWAY_GENES
from src.styles import get_cohort_color, set_presentation_style
from src.utils.dataframes import find_id_column
from src.utils.logging import TeeStream
from src.utils.plotting import resolve_colors, save_fig

set_presentation_style()

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "plots" / "genomic"
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "extended_pathway_mutation_frequencies.log"

POOLED_LABEL = "Pooled Trials"

ROW_LABELS = {"BRAF": "BRAF mutation", "NRAS": "NRAS mutation", "NF1": "NF1 mutation"}
ROW_ORDER = [
    ("MAPK Drivers", "BRAF"),
    ("MAPK Drivers", "NRAS"),
    ("MAPK Drivers", "NF1"),
    ("Immune Resistance", "IFN-gamma Signaling"),
    ("Survival & Proliferation", "Survival & Proliferation Drivers"),
]


# --------------------------------------------------------------------------
# Compute pathway/driver mutation frequencies live from each cohort's
# processed/mutations_cleaned.csv, with N taken from clin_cleaned.csv
# (source of truth for patient count).
#
# Handles two possible cleaned-file shapes so this doesn't silently
# mis-parse if the cleaning step's output format differs slightly between
# cohorts:
#   (a) long format -- one row per mutation call, with a sample/patient id
#       column + Hugo_Symbol (+ optionally Variant_Classification)
#   (b) wide format -- already a patient x gene indicator/count matrix
# --------------------------------------------------------------------------
def _compute_gene_frequency(mut_wide: pd.DataFrame, genes: list[str], cohort_name: str) -> float:
    """Helper to compute frequency of mutation in a list of genes."""
    present = [g for g in genes if g in mut_wide.columns]
    if not present:
        warnings.warn(
            f"{cohort_name}: none of {genes} found in mutation panel -- reporting NaN",
            stacklevel=2,
        )
        return np.nan
    mutated = (mut_wide[present].fillna(0) > 0).any(axis=1)
    return 100.0 * mutated.mean()


def _weighted_pooled(row: pd.Series, cohort_names: list[str], ns: dict[str, int]) -> float:
    """Helper to compute N-weighted average for pooled trials."""
    vals, weights = [], []
    for c in cohort_names:
        if pd.notna(row[c]):
            vals.append(row[c] * ns[c] / 100.0)
            weights.append(ns[c])
    return 100.0 * sum(vals) / sum(weights) if weights else np.nan
def compute_cohort_frequencies(cohort_name: str) -> tuple[dict, int]:
    """
    Compute driver/pathway mutation frequencies (%) for one cohort.
    Returns (freqs, n_patients). A gene/pathway not covered by the cohort's
    panel is reported as NaN (not 0%), since "not tested" and "wild-type"
    are different things.
    """
    cohort_dir = PROCESSED_DIR / COHORT_DIRS[cohort_name]
    mut_path = cohort_dir / "mutations_cleaned.csv"
    clin_path = cohort_dir / "clin_cleaned.csv"

    if not mut_path.exists():
        raise FileNotFoundError(f"Missing {mut_path}. Run clean_data.py for {cohort_name} first.")
    if not clin_path.exists():
        raise FileNotFoundError(f"Missing {clin_path}. Run clean_data.py for {cohort_name} first.")

    df_clin = pd.read_csv(clin_path, index_col=0)
    n_patients = len(df_clin)
    all_patients = df_clin.index

    df_mut = pd.read_csv(mut_path)

    if "Hugo_Symbol" in df_mut.columns:
        id_col = find_id_column(df_mut)
        if "Variant_Classification" in df_mut.columns:
            df_mut = df_mut[df_mut["Variant_Classification"].isin(NON_SILENT)]
        mut_wide = df_mut.pivot_table(
            index=id_col, columns="Hugo_Symbol", values="Hugo_Symbol", aggfunc="count",
        )
    else:
        id_col = find_id_column(df_mut)
        mut_wide = df_mut.set_index(id_col)

    mut_wide = mut_wide.reindex(all_patients, fill_value=0)

    freqs = {gene: _compute_gene_frequency(mut_wide, [gene], cohort_name) for gene in DRIVER_GENES}
    for pathway, genes in PATHWAY_GENES.items():
        freqs[pathway] = _compute_gene_frequency(mut_wide, genes, cohort_name)

    return freqs, n_patients


def build_dataframe() -> tuple[pd.DataFrame, list[str], str, dict]:
    """
    Assemble the cohort x gene/pathway frequency table, plus an N-weighted
    Pooled Trials column, from live-computed per-cohort frequencies.
    Returns (df, cohort_columns, pooled_column, patient_counts).
    """
    per_cohort = {name: compute_cohort_frequencies(name) for name in COHORT_DIRS}
    freqs = {name: f for name, (f, n) in per_cohort.items()}
    ns = {name: n for name, (f, n) in per_cohort.items()}
    cohort_names = list(COHORT_DIRS.keys())

    rows = []
    for category, key in ROW_ORDER:
        label = ROW_LABELS.get(key, key)
        row = {"Category": category, "Gene/Pathway": label}
        for c in cohort_names:
            row[c] = freqs[c].get(key)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Pooled = N-weighted average across cohorts that have a value for that
    # row, using each cohort's real patient count (not a fixed literal).
    df[POOLED_LABEL] = df.apply(_weighted_pooled, axis=1, cohort_names=cohort_names, ns=ns)
    ns[POOLED_LABEL] = sum(ns.values())

    rename = {c: f"{c} (N={ns[c]})" for c in cohort_names + [POOLED_LABEL]}
    df = df.rename(columns=rename)
    cohort_cols = [rename[c] for c in cohort_names]
    pooled_col = rename[POOLED_LABEL]
    return df, cohort_cols, pooled_col, ns


# --------------------------------------------------------------------------
# Chart builders -- one function per figure
# --------------------------------------------------------------------------
def plot_grouped_bars(df: pd.DataFrame, cohort_cols: list[str], pooled_col: str, colors: list[str], out_path: Path) -> None:
    all_cols = cohort_cols + [pooled_col]
    y_labels = df["Gene/Pathway"].tolist()
    y_pos = np.arange(len(y_labels))

    cat_sizes = df.groupby("Category", sort=False).size()
    cat_names = list(cat_sizes.index)
    boundaries = np.cumsum(cat_sizes.values)[:-1] - 0.5
    cat_centers, start = [], 0
    for size in cat_sizes.values:
        cat_centers.append(start + (size - 1) / 2.0)
        start += size

    n_cohorts = len(all_cols)
    bar_width = 0.8 / n_cohorts  # 0.8 leaves a visible gap between row groups
    max_val = np.nanmax(df[all_cols].values.astype(float))

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    for i, cohort in enumerate(all_cols):
        values = df[cohort].values
        offset = (i - (n_cohorts - 1) / 2.0) * bar_width
        rects = ax.barh(
            y_pos + offset, values, height=bar_width, label=cohort,
            color=colors[i], edgecolor="white", linewidth=0.8,
        )
        for rect in rects:
            width = rect.get_width()
            if width and not np.isnan(width) and width > 0:
                ax.annotate(
                    f"{width:.1f}%",
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9,
                    fontweight="bold" if cohort == pooled_col else "normal",
                    color="#222222",
                )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Mutation Frequency (%)", fontweight="bold")
    ax.set_title(
        "Pre-Treatment Somatic Mutation & Pathway Frequencies Across Cohorts",
        fontweight="bold", pad=15,
    )
    ax.legend(title="Cohort", frameon=True, facecolor="white", framealpha=0.9, loc="lower right")

    for b in boundaries:
        ax.axhline(b, color="gray", linestyle="--", alpha=0.5)

    # 31% left margin fits the rotated category label; 20% right headroom
    # fits the widest "xx.x%" value annotation.
    label_x = -max_val * 0.31
    ax.set_xlim(label_x * 1.15, max_val * 1.2)
    for cat_name, center in zip(cat_names, cat_centers):
        ax.text(
            label_x, center, cat_name, rotation=90, va="center", ha="center",
            fontweight="bold", color="#333333", fontsize=11,
        )

    save_fig(fig, out_path)


def plot_heatmap(df: pd.DataFrame, cohort_cols: list[str], pooled_col: str, out_path: Path) -> None:
    all_cols = cohort_cols + [pooled_col]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    heatmap_df = df.set_index("Gene/Pathway")[all_cols]

    sns.heatmap(
        heatmap_df, annot=True, fmt=".1f", cmap="YlGnBu",
        cbar_kws={"label": "Frequency (%)"},
        linewidths=1, linecolor="white", ax=ax,
        annot_kws={"size": 11, "weight": "bold"},
    )
    for text in ax.texts:
        text.set_text(text.get_text() + "%")

    ax.set_title("Pathway Mutation Frequency Heatmap (%)", fontweight="bold", pad=15)
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")

    save_fig(fig, out_path)


def plot_dumbbell(df: pd.DataFrame, cohort_cols: list[str], pooled_col: str, colors: list[str], out_path: Path) -> None:
    trial_colors = dict(zip(cohort_cols, colors[: len(cohort_cols)]))
    pooled_color = get_cohort_color(pooled_col, default="#e41a1c")
    max_val = np.nanmax(df[cohort_cols + [pooled_col]].values.astype(float))

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    for idx, row in df.iterrows():
        y = idx
        trial_vals = [row[c] for c in cohort_cols if pd.notna(row[c])]
        if trial_vals:
            ax.hlines(y, min(trial_vals), max(trial_vals), color="#cccccc", linewidth=4, zorder=1)

        for c in cohort_cols:
            if pd.notna(row[c]):
                ax.scatter(row[c], y, color=trial_colors[c], s=90, zorder=3, label=c if idx == 0 else "")

        if pd.notna(row[pooled_col]):
            ax.scatter(
                row[pooled_col], y, color=pooled_color, marker="D", s=130, zorder=4,
                label=pooled_col if idx == 0 else "",
            )
            ax.annotate(
                f"Pooled: {row[pooled_col]:.1f}%", (row[pooled_col], y),
                xytext=(0, 12), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=pooled_color,
            )

    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["Gene/Pathway"], fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Mutation Frequency (%)", fontweight="bold")
    ax.set_xlim(-max_val * 0.03, max_val * 1.1)  # small left pad, 10% right headroom
    ax.set_title(
        "Extended Pathway Mutation Rates: Trial Variation vs. Pooled Benchmark",
        fontweight="bold", pad=15,
    )
    ax.legend(loc="lower right", frameon=True, facecolor="white")

    save_fig(fig, out_path)


def main() -> None:
    print("==================================================")
    print("Extended Pathway Mutation Frequencies")
    print("==================================================")

    df, cohort_cols, pooled_col, ns = build_dataframe()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    colors = resolve_colors(cohort_cols + [pooled_col])

    plot_grouped_bars(df, cohort_cols, pooled_col, colors, OUT_DIR / "extended_pathway_grouped_bars.png")
    plot_heatmap(df, cohort_cols, pooled_col, OUT_DIR / "extended_pathway_heatmap.png")
    plot_dumbbell(df, cohort_cols, pooled_col, colors, OUT_DIR / "extended_pathway_dumbbell.png")

    print(f"Successfully generated 3 visualisation figures in {OUT_DIR.relative_to(BASE_DIR).as_posix()}")
    print(f"Patient counts used (from clin_cleaned.csv): {ns}")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(BASE_DIR).as_posix()}")
            main()