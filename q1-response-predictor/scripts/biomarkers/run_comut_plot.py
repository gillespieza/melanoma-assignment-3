import sys
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from pathlib import Path

# Add project root to sys.path for importing src modules
_THIS_FILE = Path(__file__).resolve()
for _candidate in [_THIS_FILE.parent] + list(_THIS_FILE.parent.parents):
    if (_candidate / "src").exists() and (_candidate / "data").exists():
        BASE_DIR = _candidate
        break
else:
    raise FileNotFoundError(f"Could not locate project root above {_THIS_FILE}")

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style
from src.biology_constants import COMUT_DRIVER_GENES, SIGNATURE_GENES
from src.utils.logging import TeeStream
from src.utils.plotting import save_fig
from src.data_loaders import load_liu_2019

# Paths
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots" / "genomic"
PLOT_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "run_comut_plot.log"

# --------------------------------------------------------------------------
# Colors that are specific to THIS plot's semantics (not tied to cohort
# identity, so they won't silently change meaning if COHORT_PALETTE is
# edited for an unrelated reason).
# --------------------------------------------------------------------------
SEX_PALETTE = {
    "Male": "#4C72B0",
    "Female": "#f768a1",
}
MISSING_DATA_COLOR = "#bdbdbd"  # neutral grey for genuinely missing annotations


def _load_and_align_data() -> tuple[pd.DataFrame, list[str]] | None:
    # 1. Load Clinical & Mutation data
    mut_path = DATA_DIR / "processed/liu_2019/mutations_cleaned.csv"
    try:
        _, df_clin = load_liu_2019(DATA_DIR)

        print(df_clin.columns.tolist())
            
    except Exception as e:
        print(f"Error loading Liu 2019 clinical data: {e}")
        return None

    if not mut_path.exists():
        print("Error: Required processed mutations file for Liu 2019 not found.")
        return None

    df_mut = pd.read_csv(mut_path, index_col=0)

    # --- Sanity check: verify the mutation table index actually overlaps
    # the clinical index before mapping. A silent all-zero mapping from a
    # sample-ID vs. patient-ID mismatch is worse than a loud failure.
    overlap = df_clin.index.intersection(df_mut.index)
    overlap_frac = len(overlap) / max(len(df_clin.index), 1)
    if overlap_frac < 0.5:
        print(
            f"  [WARNING] Only {overlap_frac:.1%} of clinical IDs are present in "
            f"the mutations table index. This likely indicates a sample-ID vs. "
            f"patient-ID mismatch between '{mut_path.name}' and the clinical data. "
            f"Mutation calls below may be silently wrong (all zeros) for "
            f"unmatched patients."
        )
    else:
        print(f"  Mutation/clinical index overlap: {overlap_frac:.1%} ({len(overlap)} patients)")

    # Select target genes (from biology_constants)
    target_genes = COMUT_DRIVER_GENES + SIGNATURE_GENES

    # Map back to sample IDs in df_clin
    df_clin_mut = df_clin.copy()
    for g in target_genes:
        if g in df_mut.columns:
            df_clin_mut[f'mut_{g}'] = df_clin_mut.index.map(df_mut[g]).fillna(0).astype(int)
        else:
            df_clin_mut[f'mut_{g}'] = 0

    # Filter/align patients with clinical data (excluding any missing response if necessary)
    df_clin_mut = df_clin_mut.dropna(subset=['response'])
    print(f"Aligned patients for CoMut plot: {len(df_clin_mut)}")

    # 2. Sort Patients for Cascading Mutation Pattern
    # Primary sort: BRAF -> NRAS -> NF1 -> CDKN2A -> PTEN -> others -> Response
    sort_cols = [f'mut_{g}' for g in target_genes] + ['response']
    df_clin_mut_sorted = df_clin_mut.sort_values(by=sort_cols, ascending=False)

    sorted_sample_ids = df_clin_mut_sorted.index.tolist()
    
    return df_clin_mut_sorted, sorted_sample_ids


def _draw_comut_plot(df_clin_mut_sorted: pd.DataFrame, sorted_sample_ids: list[str]) -> None:
    target_genes = COMUT_DRIVER_GENES + SIGNATURE_GENES
    
    # 3. Prepare Plot Arrays
    # Central mutation grid: shape (n_genes, n_patients)
    mut_grid = np.zeros((len(target_genes), len(sorted_sample_ids)))
    for i, g in enumerate(target_genes):
        mut_grid[i, :] = df_clin_mut_sorted[f'mut_{g}'].values

    # Top panel: TMB values
    tmb_vals = df_clin_mut_sorted['TMB_NONSYNONYMOUS'].fillna(0).values

    # Right panel: Gene mutation frequencies (percentages)
    gene_freqs = [(df_clin_mut_sorted[f'mut_{g}'] > 0).mean() * 100 for g in target_genes]

    # Bottom tracks: Response, Sex
    response_vals = df_clin_mut_sorted['response'].values

    # --- Sex: keep genuinely missing values distinguishable from "Female"
    # rather than silently defaulting unmapped/absent values to Female (0).
    if 'SEX' in df_clin_mut_sorted.columns:
        sex_raw = df_clin_mut_sorted['SEX'].map({'Male': 1.0, 'Female': 0.0})
        sex_vals = sex_raw.values
    else:
        sex_vals = np.full(len(df_clin_mut_sorted), np.nan)
    sex_missing_mask = np.isnan(sex_vals)

    # ==========================================
    # 4. Draw the Co-Mutation Plot
    # ==========================================
    sns.set_theme(style="white")

    fig = plt.figure(figsize=(15, 17))
    # GridSpec layout:
    # Row 0: TMB barplot (height ratio = 2)
    # Row 1: Central mutation grid & right frequencies (height ratio = 14)
    # Row 2: Space / Margin (height ratio = 0.2)
    # Row 3: Response track (height ratio = 0.4)
    # Row 4: Sex track (height ratio = 0.4)

    gs = gridspec.GridSpec(
        nrows=5, ncols=2,
        width_ratios=[12, 2],
        height_ratios=[2, 14, 0.2, 0.4, 0.4],
        wspace=0.08, hspace=0.12
    )

    set_presentation_style()

    # Color palette
    color_mut = COHORT_PALETTE["Liu 2019"]
    color_wt = "#f0f0f0"  # Grey for wild-type

    # A. Top Subplot: TMB Barplot
    ax_tmb = fig.add_subplot(gs[0, 0])
    ax_tmb.bar(range(len(tmb_vals)), tmb_vals, color="#737373", width=0.8, edgecolor="none")
    ax_tmb.set_ylabel("TMB (mut/Mb)", fontsize=11, weight='bold')
    ax_tmb.set_xlim(-0.5, len(tmb_vals) - 0.5)
    ax_tmb.xaxis.set_visible(False)
    ax_tmb.spines['top'].set_visible(False)
    ax_tmb.spines['right'].set_visible(False)
    ax_tmb.spines['bottom'].set_visible(False)

    # B. Central Subplot: Mutation Heatmap Grid
    ax_mut = fig.add_subplot(gs[1, 0])
    # Custom colormap: 0 = WT (grey), >=1 = Mutated (blue)
    cmap_mut = ListedColormap([color_wt, color_mut])
    bounds = [0, 0.5, 5]
    norm = BoundaryNorm(bounds, cmap_mut.N)

    sns.heatmap(
        mut_grid, cmap=cmap_mut, norm=norm, cbar=False,
        linewidths=0.5, linecolor='white', ax=ax_mut,
        yticklabels=target_genes, xticklabels=False
    )

    # Format driver vs signature gene labels
    for label in ax_mut.get_yticklabels():
        gene_name = label.get_text()
        if gene_name in SIGNATURE_GENES:
            label.set_color('#3C5488')  # NPG Blue for signature genes
            label.set_fontsize(10.0)
            label.set_fontweight('bold')
        else:
            label.set_color('black')    # Black for driver genes
            label.set_fontsize(11.0)
            label.set_fontweight('bold')

    # Draw a solid horizontal black line separating the driver panel from the signature panel
    ax_mut.axhline(y=len(COMUT_DRIVER_GENES), color='black', linewidth=2.0, linestyle='-', zorder=10)

    # C. Right Subplot: Gene Frequencies (Percentage Barplot)
    ax_freq = fig.add_subplot(gs[1, 1])
    ax_freq.barh(range(len(gene_freqs)), gene_freqs, color=color_mut, height=0.7, edgecolor="none")
    ax_freq.set_ylim(-0.5, len(gene_freqs) - 0.5)
    ax_freq.invert_yaxis()  # Align with heatmap rows
    ax_freq.set_xlabel("% Mutated", fontsize=10, weight='bold')
    ax_freq.set_xlim(0, max(gene_freqs) + 5)
    ax_freq.yaxis.set_visible(False)
    ax_freq.spines['top'].set_visible(False)
    ax_freq.spines['right'].set_visible(False)
    ax_freq.spines['left'].set_visible(False)

    # Add percentage labels to the bar plot
    for i, freq in enumerate(gene_freqs):
        ax_freq.text(freq + 1, i, f"{freq:.1f}%", va='center', fontsize=8.5, weight='bold')

    # Draw horizontal separator in frequencies plot to match the central grid
    ax_freq.axhline(y=len(COMUT_DRIVER_GENES) - 0.5, color='black', linewidth=1.5, linestyle='-', zorder=10)

    # D. Bottom Track 1: Response Status
    ax_resp = fig.add_subplot(gs[3, 0])
    # Colormap: 0 = Non-Responder (PD, Red), 1 = Responder (CR/PR, Green)
    cmap_resp = ListedColormap([RESPONSE_PALETTE["PD"], RESPONSE_PALETTE["CR/PR"]])
    sns.heatmap(
        response_vals.reshape(1, -1), cmap=cmap_resp, cbar=False,
        ax=ax_resp, xticklabels=False, yticklabels=["Response"]
    )
    ax_resp.set_yticklabels(["Response"], rotation=0, fontsize=11, weight='bold')

    # E. Bottom Track 2: Sex
    # Missing/unmapped sex values are masked out and painted the same
    # "no data" grey rather than silently defaulting to Female.
    ax_sex = fig.add_subplot(gs[4, 0])
    cmap_sex = ListedColormap([SEX_PALETTE["Female"], SEX_PALETTE["Male"]])
    sex_display = np.where(sex_missing_mask, 0, sex_vals).reshape(1, -1)
    sns.heatmap(
        sex_display, cmap=cmap_sex, cbar=False,
        ax=ax_sex, xticklabels=False, yticklabels=["Sex"],
        mask=sex_missing_mask.reshape(1, -1)
    )
    ax_sex.set_yticklabels(["Sex"], rotation=0, fontsize=11, weight='bold')
    for j, is_missing in enumerate(sex_missing_mask):
        if is_missing:
            ax_sex.add_patch(plt.Rectangle((j, 0), 1, 1, facecolor=MISSING_DATA_COLOR, edgecolor='none'))

    # Align the X-axes of clinical tracks with the mutation grid
    for ax in [ax_tmb, ax_mut, ax_resp, ax_sex]:
        ax.set_xlim(-0.5, len(sorted_sample_ids) - 0.5)

    # Add Title
    fig.suptitle("Co-Mutation & Clinical Landscape of immunotherapy Response (Liu 2019)",
                 fontsize=16, weight='bold', y=0.96)

    # 5. Add Custom Legend on the right of Clinical Tracks
    ax_legend = fig.add_subplot(gs[3:, 1])
    ax_legend.axis('off')

    # Custom legend patches
    patches = [
        mpatches.Patch(color=color_mut, label='Mutated'),
        mpatches.Patch(color=color_wt, label='Wild-Type'),
        mpatches.Patch(color=RESPONSE_PALETTE["CR/PR"], label='Responder (CR/PR)'),
        mpatches.Patch(color=RESPONSE_PALETTE["PD"], label='Non-Responder (PD)'),
        mpatches.Patch(color=SEX_PALETTE["Male"], label='Sex: Male'),
        mpatches.Patch(color=SEX_PALETTE["Female"], label='Sex: Female'),
        mpatches.Patch(color=MISSING_DATA_COLOR, label='No Data'),
    ]
    ax_legend.legend(handles=patches, loc='center left', frameon=True, fontsize=10)

    # Save co-mutation landscape
    comut_path = PLOT_DIR / "comut_landscape_liu_2019.png"
    save_fig(fig, comut_path)

    print(f"Saved CoMut plot to {comut_path.relative_to(BASE_DIR).as_posix()}")


def main() -> None:
    print("==================================================")
    print("Generating Co-Mutation (Oncoplot) for Liu 2019...")
    print("==================================================")

    data = _load_and_align_data()
    if data is None:
        return
    df_clin_mut_sorted, sorted_sample_ids = data
    
    _draw_comut_plot(df_clin_mut_sorted, sorted_sample_ids)

    print("==================================================")

if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(BASE_DIR).as_posix()}")
            main()