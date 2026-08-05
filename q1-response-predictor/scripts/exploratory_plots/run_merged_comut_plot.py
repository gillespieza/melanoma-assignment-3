"""
Co-Mutation (Oncoplot) Visualisation for Merged Checkpoint Blockade Trial Cohorts.

Generates a multi-track co-mutation landscape plot across merged melanoma clinical
trial cohorts (Liu 2019, Hugo 2016, Riaz 2017), illustrating somatic mutation frequencies
in key melanoma driver genes alongside TMB, response status, cohort source, and sex annotations.
"""

import contextlib
from pathlib import Path
import sys
from typing import List, Optional, Tuple
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BASE_DIR.parent if (BASE_DIR.parent / "src").exists() else BASE_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config.constants import MERGED_COMUT_DRIVER_GENES
from src.data_loaders import load_hugo_2016, load_liu_2019, load_riaz_2017
from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, SEX_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, LOG_DIR, PLOTS_DIR
from src.utils.plotting import save_fig

set_presentation_style()

# Module-level Constants
PLOT_DIR = PLOTS_DIR / "genomic"
LOG_PATH = LOG_DIR / "run_merged_comut_plot.log"




def load_processed_mutations(mutations_file: Path, target_genes: List[str], sample_ids: List[str]) -> pd.DataFrame:
    """Loads processed somatic mutations for specified genes and sample IDs.

    Args:
        mutations_file: Path to the mutations_cleaned.csv file.
        target_genes: List of gene symbols to extract.
        sample_ids: List of sample IDs for indexing.

    Returns:
        DataFrame containing binary mutation indicator columns.
    """
    if not mutations_file.exists():
        print(f"Warning: mutation file not found at {mutations_file.relative_to(PROJECT_ROOT).as_posix()}")
        return pd.DataFrame(0, index=sample_ids, columns=target_genes)

    df_mut = pd.read_csv(mutations_file, index_col="SAMPLE_ID")
    for g in target_genes:
        if g not in df_mut.columns:
            df_mut[g] = 0

    df_mut = (df_mut[target_genes] > 0).astype(int)
    return df_mut.reindex(sample_ids, fill_value=0)


def _load_and_align_merged_data(data_dir: Path) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    """Loads clinical and somatic mutation data across all three trial cohorts and aligns them.

    Args:
        data_dir: Path to project data directory.

    Returns:
        Tuple of (sorted merged DataFrame, sorted sample IDs list) or None if error.
    """
    _, clin_liu = load_liu_2019(data_dir)
    _, clin_hugo = load_hugo_2016(data_dir)
    _, clin_riaz = load_riaz_2017(data_dir)

    clin_liu['Cohort'] = 'Liu 2019'
    clin_hugo['Cohort'] = 'Hugo 2016'
    clin_riaz['Cohort'] = 'Riaz 2017'

    for df in [clin_liu, clin_hugo, clin_riaz]:
        if 'SEX' in df.columns:
            df['SEX'] = df['SEX'].map({'Male': 'Male', 'Female': 'Female', 'M': 'Male', 'F': 'Female'})

    print("Loading somatic mutation data per cohort...")
    mut_liu = load_processed_mutations(data_dir / "processed/liu_2019/mutations_cleaned.csv", MERGED_COMUT_DRIVER_GENES, clin_liu.index.tolist())
    mut_hugo = load_processed_mutations(data_dir / "processed/hugo_2016/mutations_cleaned.csv", MERGED_COMUT_DRIVER_GENES, clin_hugo.index.tolist())
    mut_riaz = load_processed_mutations(data_dir / "processed/riaz_2017/mutations_cleaned.csv", MERGED_COMUT_DRIVER_GENES, clin_riaz.index.tolist())

    clin_cols = ['Cohort', 'RESPONSE_BINARY', 'TMB_NONSYNONYMOUS', 'SEX', 'PATIENT_ID']
    df_clin_merged = pd.concat([
        clin_liu[clin_cols],
        clin_hugo[clin_cols],
        clin_riaz[clin_cols],
    ])

    df_mut_merged = pd.concat([mut_liu, mut_hugo, mut_riaz])
    df_merged = df_clin_merged.join(df_mut_merged)
    df_merged = df_merged.dropna(subset=['RESPONSE_BINARY'])

    print(f"Total merged clinical trial samples aligned for CoMut plot: {len(df_merged)}")

    sort_cols = MERGED_COMUT_DRIVER_GENES + ['RESPONSE_BINARY', 'Cohort']
    df_sorted = df_merged.sort_values(by=sort_cols, ascending=False)
    sorted_sample_ids = df_sorted.index.tolist()

    return df_sorted, sorted_sample_ids


def _draw_merged_comut_plot(df_sorted: pd.DataFrame, sorted_sample_ids: List[str], plot_dir: Path) -> None:
    """Renders and saves the multi-track co-mutation landscape figure.

    Args:
        df_sorted: Merged DataFrame sorted by mutation and clinical tracks.
        sorted_sample_ids: List of sample IDs in display order.
        plot_dir: Path to directory for saving plot artifact.
    """
    n_samples = len(sorted_sample_ids)

    mut_grid = np.zeros((len(MERGED_COMUT_DRIVER_GENES), n_samples))
    for i, g in enumerate(MERGED_COMUT_DRIVER_GENES):
        mut_grid[i, :] = df_sorted[g].values

    tmb_vals = df_sorted['TMB_NONSYNONYMOUS'].fillna(0).values
    response_vals = df_sorted['RESPONSE_BINARY'].values
    cohort_vals = df_sorted['Cohort'].map({'Liu 2019': 0, 'Hugo 2016': 1, 'Riaz 2017': 2}).values
    sex_vals = df_sorted['SEX'].map({'Female': 0, 'Male': 1}).fillna(2).values

    gene_freqs = [(df_sorted[g] > 0).mean() * 100 for g in MERGED_COMUT_DRIVER_GENES]

    fig = plt.figure(figsize=(16, 11))

    gs = gridspec.GridSpec(
        nrows=6, ncols=2,
        width_ratios=[13, 2],
        height_ratios=[1.5, 7.0, 0.1, 0.35, 0.35, 0.35],
        wspace=0.08, hspace=0.12,
    )

    color_mut = COHORT_PALETTE["Liu 2019"]
    color_wt = "#f0f0f0"

    # A. Top Subplot: TMB Barplot
    ax_tmb = fig.add_subplot(gs[0, 0])
    ax_tmb.bar(range(n_samples), tmb_vals, color="#737373", width=0.8, edgecolor="none")
    ax_tmb.set_ylabel("TMB (mut/Mb)", fontsize=11, weight="bold")
    ax_tmb.xaxis.set_visible(False)
    ax_tmb.spines["top"].set_visible(False)
    ax_tmb.spines["right"].set_visible(False)
    ax_tmb.spines["bottom"].set_visible(False)

    # B. Central Subplot: Mutation Heatmap Grid
    ax_mut = fig.add_subplot(gs[1, 0])
    cmap_mut = ListedColormap([color_wt, color_mut])
    bounds = [0, 0.5, 5]
    norm = BoundaryNorm(bounds, cmap_mut.N)

    sns.heatmap(
        mut_grid, cmap=cmap_mut, norm=norm, cbar=False,
        linewidths=0.5, linecolor="white", ax=ax_mut,
        yticklabels=MERGED_COMUT_DRIVER_GENES, xticklabels=False,
    )
    for label in ax_mut.get_yticklabels():
        label.set_color("black")
        label.set_fontsize(11)
        label.set_fontweight("bold")

    ax_mut.axhline(y=3, color="black", linewidth=1.5, linestyle="-", zorder=10)

    # C. Right Subplot: Gene Frequencies (%)
    ax_freq = fig.add_subplot(gs[1, 1])
    ax_freq.barh(range(len(gene_freqs)), gene_freqs, color=color_mut, height=0.7, edgecolor="none")
    ax_freq.set_ylim(-0.5, len(gene_freqs) - 0.5)
    ax_freq.invert_yaxis()
    ax_freq.set_xlabel("% Mutated", fontsize=10, weight="bold")
    ax_freq.set_xlim(0, max(gene_freqs) + 5)
    ax_freq.yaxis.set_visible(False)
    ax_freq.spines["top"].set_visible(False)
    ax_freq.spines["right"].set_visible(False)
    ax_freq.spines["left"].set_visible(False)

    for i, freq in enumerate(gene_freqs):
        ax_freq.text(freq + 1, i, f"{freq:.1f}%", va="center", fontsize=9, weight="bold")
    ax_freq.axhline(y=2.5, color="black", linewidth=1.5, linestyle="-", zorder=10)

    # D. Bottom Track 1: Response Status
    ax_resp = fig.add_subplot(gs[3, 0])
    cmap_resp = ListedColormap([RESPONSE_PALETTE["PD"], RESPONSE_PALETTE["CR/PR"]])
    bounds_resp = [-0.5, 0.5, 1.5]
    norm_resp = BoundaryNorm(bounds_resp, cmap_resp.N)
    sns.heatmap(
        response_vals.reshape(1, -1), cmap=cmap_resp, norm=norm_resp, cbar=False,
        ax=ax_resp, xticklabels=False, yticklabels=["Response"],
    )
    ax_resp.set_yticklabels(["Response"], rotation=0, fontsize=11, weight="bold", color="black")

    # E. Bottom Track 2: Cohort Source
    ax_cohort = fig.add_subplot(gs[4, 0])
    cmap_cohort = ListedColormap([COHORT_PALETTE["Liu 2019"], COHORT_PALETTE["Hugo 2016"], COHORT_PALETTE["Riaz 2017"]])
    bounds_cohort = [-0.5, 0.5, 1.5, 2.5]
    norm_cohort = BoundaryNorm(bounds_cohort, cmap_cohort.N)
    sns.heatmap(
        cohort_vals.reshape(1, -1), cmap=cmap_cohort, norm=norm_cohort, cbar=False,
        ax=ax_cohort, xticklabels=False, yticklabels=["Cohort"],
    )
    ax_cohort.set_yticklabels(["Cohort"], rotation=0, fontsize=11, weight="bold", color="black")

    # F. Bottom Track 3: Sex
    ax_sex = fig.add_subplot(gs[5, 0])
    cmap_sex = ListedColormap([SEX_PALETTE["Female"], SEX_PALETTE["Male"], SEX_PALETTE["Unknown"]])
    bounds_sex = [-0.5, 0.5, 1.5, 2.5]
    norm_sex = BoundaryNorm(bounds_sex, cmap_sex.N)
    sns.heatmap(
        sex_vals.reshape(1, -1), cmap=cmap_sex, norm=norm_sex, cbar=False,
        ax=ax_sex, xticklabels=False, yticklabels=["Sex"],
    )
    ax_sex.set_yticklabels(["Sex"], rotation=0, fontsize=11, weight="bold", color="black")

    for ax in [ax_tmb, ax_mut, ax_resp, ax_cohort, ax_sex]:
        ax.set_xlim(-0.5, n_samples - 0.5)

    fig.suptitle(f"Co-Mutation and Clinical Landscape of Checkpoint Blockade Trials (Merged Cohorts, N={n_samples})",
                 fontsize=16, weight="bold", y=0.96)

    ax_legend = fig.add_subplot(gs[3:, 1])
    ax_legend.axis("off")

    patches = [
        mpatches.Patch(color=color_mut, label="Mutated"),
        mpatches.Patch(color=color_wt, label="Wild-Type"),
        mpatches.Patch(color=RESPONSE_PALETTE["CR/PR"], label="Responder (CR/PR)"),
        mpatches.Patch(color=RESPONSE_PALETTE["PD"], label="Non-responder (PD)"),
        mpatches.Patch(color=COHORT_PALETTE["Liu 2019"], label="Liu 2019"),
        mpatches.Patch(color=COHORT_PALETTE["Hugo 2016"], label="Hugo 2016"),
        mpatches.Patch(color=COHORT_PALETTE["Riaz 2017"], label="Riaz 2017"),
        mpatches.Patch(color=SEX_PALETTE["Male"], label="Sex: Male"),
        mpatches.Patch(color=SEX_PALETTE["Female"], label="Sex: Female"),
        mpatches.Patch(color=SEX_PALETTE["Unknown"], label="Sex: Unknown"),
    ]
    ax_legend.legend(handles=patches, loc="center left", frameon=True, fontsize=9.5)

    out_path = plot_dir / "comut_landscape_merged.png"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        save_fig(fig, out_path)

    print(f"\nSaved Merged CoMut plot to {out_path.relative_to(PROJECT_ROOT).as_posix()}")


def main() -> None:
    """Executes the merged co-mutation landscape generation pipeline."""
    print("==================================================")
    print("Generating Co-Mutation (Oncoplot) for Merged Trial Cohorts")
    print("==================================================\n")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    result = _load_and_align_merged_data(DATA_DIR)
    if result is None:
        return

    df_sorted, sorted_sample_ids = result
    _draw_merged_comut_plot(df_sorted, sorted_sample_ids, PLOT_DIR)

    print("==================================================")
    print("Done! Merged Co-Mutation plot generated.")
    print("==================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(PROJECT_ROOT).as_posix()}")
            main()
