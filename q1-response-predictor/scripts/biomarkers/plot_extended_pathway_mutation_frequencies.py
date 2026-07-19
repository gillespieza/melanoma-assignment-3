import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. Robust project root resolution (was: fixed parent.parent.parent depth)
# --------------------------------------------------------------------------
def find_project_root(start: Path, markers=("src", "data")) -> Path:
    for candidate in [start] + list(start.parents):
        if all((candidate / m).exists() for m in markers):
            return candidate
    raise FileNotFoundError(f"Could not locate project root (looked for {markers}) above {start}")


THIS_FILE = Path(__file__).resolve()
BASE_DIR = find_project_root(THIS_FILE.parent)
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.styles import COHORT_PALETTE, set_presentation_style  # noqa: E402

set_presentation_style()

PROCESSED_DIR = BASE_DIR / "data" / "processed"
COHORT_DIRS = {
    "Liu 2019": "liu_2019",
    "Hugo 2016": "hugo_2016",
    "Riaz 2017": "riaz_2017",
}

NON_SILENT = [
    "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
    "Missense_Mutation", "Nonsense_Mutation", "Splice_Site",
    "Translation_Start_Site", "Nonstop_Mutation",
]

PATHWAY_GENES = {
    "IFN-gamma Signaling": ["JAK1", "JAK2", "STAT1"],
    "Survival & Proliferation Drivers": ["PTEN", "CDKN2A", "PIK3CA"],
}
DRIVER_GENES = ["BRAF", "NRAS", "NF1"]
ROW_LABELS = {"BRAF": "BRAF mutation", "NRAS": "NRAS mutation", "NF1": "NF1 mutation"}
ROW_ORDER = [
    ("MAPK Drivers", "BRAF"),
    ("MAPK Drivers", "NRAS"),
    ("MAPK Drivers", "NF1"),
    ("Immune Resistance", "IFN-gamma Signaling"),
    ("Survival & Proliferation", "Survival & Proliferation Drivers"),
]


# --------------------------------------------------------------------------
# 2. Compute pathway/driver mutation frequencies live from each cohort's
#    processed/mutations_cleaned.csv, with N taken from clin_cleaned.csv
#    (source of truth for patient count). Was: all values + all N's typed
#    in by hand, with no traceability back to the underlying data.
#
#    Handles two possible cleaned-file shapes so this doesn't silently
#    mis-parse if the cleaning step's output format differs slightly
#    between cohorts:
#      (a) long format -- one row per mutation call, with a sample/patient
#          id column + Hugo_Symbol (+ optionally Variant_Classification)
#      (b) wide format -- already a patient x gene indicator/count matrix
# --------------------------------------------------------------------------
def _id_col(df):
    for c in ["PATIENT_ID", "patient_id", "SAMPLE_ID", "sample_id"]:
        if c in df.columns:
            return c
    raise ValueError(f"No recognizable patient/sample id column found in: {list(df.columns)}")


def compute_cohort_frequencies(cohort_name):
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
        id_col = _id_col(df_mut)
        if "Variant_Classification" in df_mut.columns:
            df_mut = df_mut[df_mut["Variant_Classification"].isin(NON_SILENT)]
        mut_wide = df_mut.pivot_table(
            index=id_col, columns="Hugo_Symbol", values="Hugo_Symbol", aggfunc="count",
        )
    else:
        id_col = _id_col(df_mut)
        mut_wide = df_mut.set_index(id_col)

    mut_wide = mut_wide.reindex(all_patients, fill_value=0)

    def freq(genes):
        present = [g for g in genes if g in mut_wide.columns]
        if not present:
            print(f"  [WARNING] {cohort_name}: none of {genes} found in mutation panel -- reporting NaN")
            return np.nan
        mutated = (mut_wide[present].fillna(0) > 0).any(axis=1)
        return 100.0 * mutated.mean()

    freqs = {gene: freq([gene]) for gene in DRIVER_GENES}
    for pathway, genes in PATHWAY_GENES.items():
        freqs[pathway] = freq(genes)

    return freqs, n_patients


def build_dataframe():
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
    def weighted_pooled(row):
        vals, weights = [], []
        for c in cohort_names:
            if pd.notna(row[c]):
                vals.append(row[c] * ns[c] / 100.0)
                weights.append(ns[c])
        return 100.0 * sum(vals) / sum(weights) if weights else np.nan

    df["Pooled Trials"] = df.apply(weighted_pooled, axis=1)
    ns["Pooled Trials"] = sum(ns.values())

    rename = {c: f"{c} (N={ns[c]})" for c in cohort_names + ["Pooled Trials"]}
    df = df.rename(columns=rename)
    cohorts = [rename[c] for c in cohort_names + ["Pooled Trials"]]
    return df, cohorts, ns


def main():
    df, cohorts, ns = build_dataframe()

    out_dir = BASE_DIR / "plots" / "genomic"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve colors from the shared palette; generate a fallback for any
    # cohort not already defined there instead of hardcoding hex values.
    fallback_colors = sns.color_palette("Set2", n_colors=len(cohorts)).as_hex()
    colors = [COHORT_PALETTE.get(c, fallback_colors[i]) for i, c in enumerate(cohorts)]

    y_labels = df["Gene/Pathway"].tolist()
    y_pos = np.arange(len(y_labels))

    # Derive category boundaries and label positions from the data itself
    # (was: hardcoded axhline(2.5)/(3.5) and text() coordinates).
    cat_sizes = df.groupby("Category", sort=False).size()
    cat_names = list(cat_sizes.index)
    boundaries = np.cumsum(cat_sizes.values)[:-1] - 0.5
    cat_centers, start = [], 0
    for size in cat_sizes.values:
        cat_centers.append(start + (size - 1) / 2.0)
        start += size

    n_cohorts = len(cohorts)
    bar_width = 0.8 / n_cohorts
    all_vals = df[cohorts].values.astype(float)
    max_val = np.nanmax(all_vals)

    # ---------------- Option 1: Horizontal Grouped Bar Chart ----------------
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    for i, cohort in enumerate(cohorts):
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
                    fontweight="bold" if "Pooled" in cohort else "normal",
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

    label_x = -max_val * 0.31
    ax.set_xlim(label_x * 1.15, max_val * 1.2)
    for cat_name, center in zip(cat_names, cat_centers):
        ax.text(
            label_x, center, cat_name, rotation=90, va="center", ha="center",
            fontweight="bold", color="#333333", fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(out_dir / "extended_pathway_grouped_bars.png", bbox_inches="tight")
    plt.close()

    # ---------------- Option 2: Annotated Heatmap Matrix ----------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    heatmap_df = df.set_index("Gene/Pathway")[cohorts]

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
    plt.xticks(rotation=15, ha="right", fontweight="bold")
    plt.yticks(fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "extended_pathway_heatmap.png", bbox_inches="tight")
    plt.close()

    # ---------------- Option 3: Dumbbell / Range Plot ----------------
    trial_cohorts = [c for c in cohorts if not c.startswith("Pooled")]
    pooled_col = next(c for c in cohorts if c.startswith("Pooled"))
    trial_colors = dict(zip(trial_cohorts, colors[: len(trial_cohorts)]))
    pooled_color = COHORT_PALETTE.get(pooled_col, "#e41a1c")

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    for idx, row in df.iterrows():
        y = idx
        trial_vals = [row[c] for c in trial_cohorts if pd.notna(row[c])]
        if trial_vals:
            ax.hlines(y, min(trial_vals), max(trial_vals), color="#cccccc", linewidth=4, zorder=1)

        for c in trial_cohorts:
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
    ax.set_xlim(-max_val * 0.03, max_val * 1.1)
    ax.set_title(
        "Extended Pathway Mutation Rates: Trial Variation vs. Pooled Benchmark",
        fontweight="bold", pad=15,
    )
    ax.legend(loc="lower right", frameon=True, facecolor="white")

    plt.tight_layout()
    plt.savefig(out_dir / "extended_pathway_dumbbell.png", bbox_inches="tight")
    plt.close()

    print(f"Successfully generated 3 visualization figures in {out_dir}")
    print(f"Patient counts used (from clin_cleaned.csv): {ns}")


if __name__ == "__main__":
    main()