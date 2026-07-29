"""
Q4: Identifying Novel Drug Targets in Melanoma Using DepMap Data
================================================================

Uses CRISPR knockout screens (DepMap 24Q2) to identify genes that melanoma
cell lines depend on for survival — candidates for new drug targets.

Workflow:
    1. Load and filter data for melanoma cell lines
    2. Identify top dependencies (most essential genes)
    3. Compute melanoma-selective dependencies (vs all other cancers)
    4. Validate targets are expressed in melanoma lines
    5. Flag known vs novel targets
    6. Export ranked target list and figures

Outputs (written to outputs/):
    figures/01_top_dependencies.png
    figures/02_selectivity_volcano.png
    figures/03_dependency_vs_expression.png
    figures/04_dependency_probability_heatmap.png
    results/melanoma_drug_targets_ranked.csv
    results/summary_stats.txt
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Paths
# =============================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "outputs" / "figures"
RES_DIR = ROOT / "outputs" / "results"

FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Known melanoma drug targets (sanity-check / annotation)
# These are established targets with approved drugs in melanoma.
# We use them to validate our method recovers known biology, then highlight
# novel targets beyond them.
# =============================================================================

KNOWN_MELANOMA_TARGETS = {
    "BRAF",   # Vemurafenib, Dabrafenib
    "MAP2K1", # MEK1 — Trametinib
    "MAP2K2", # MEK2 — Cobimetinib
    "PDCD1",  # PD-1 — Pembrolizumab, Nivolumab
    "CD274",  # PD-L1 — Atezolizumab
    "CTLA4",  # Ipilimumab
    "KIT",    # Imatinib (rare KIT-mutant melanoma)
    "NRAS",   # No approved drug yet, but well-known driver
    "CDK4",   # Palbociclib trials
    "CDK6",   # Palbociclib trials
    "EGFR",   # Context-specific
    "MET",    # Crizotinib trials
}

# Chronos dependency score threshold:
# < -0.5 is commonly used as "dependent"; < -1.0 is strongly dependent.
DEPENDENCY_THRESHOLD = -0.5
STRONG_DEPENDENCY_THRESHOLD = -1.0

# Selectivity: how much MORE dependent melanoma is vs other cancers
SELECTIVITY_THRESHOLD = -0.2

# Dependency probability threshold (from CRISPRGeneDependency)
PROB_THRESHOLD = 0.5


# =============================================================================
# Plotting style
# =============================================================================

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

MELANOMA_COLOR = "#C0392B"
OTHER_COLOR = "#ADB5BD"
KNOWN_COLOR = "#F39C12"
NOVEL_COLOR = "#1A6B3C"


# =============================================================================
# 1. Data loading
# =============================================================================

def load_model_metadata(data_dir: Path) -> pd.DataFrame:
    """Load cell line metadata and return melanoma model IDs."""
    print("Loading Model.csv ...")
    model = pd.read_csv(data_dir / "Model.csv")
    print(f"  Total cell lines: {len(model):,}")
    return model


def get_melanoma_ids(model: pd.DataFrame) -> list[str]:
    """Return ModelIDs for melanoma cell lines."""
    mask = model["OncotreeLineage"].str.contains(
    "Skin", case=False, na=False
    )
    mel_ids = model.loc[mask, "ModelID"].tolist()
    print(f"  Melanoma cell lines: {len(mel_ids)}")
    return mel_ids


def load_gene_effect(data_dir: Path) -> pd.DataFrame:
    """Load CRISPR Chronos gene effect matrix (ModelID x Gene)."""
    print("Loading CRISPRGeneEffect.csv (this may take a moment) ...")
    df = pd.read_csv(data_dir / "CRISPRGeneEffect.csv", index_col=0)
    # Column names are formatted as "GENENAME (EntrezID)" — strip to symbol
    df.columns = [c.split(" (")[0] for c in df.columns]
    print(f"  Shape: {df.shape[0]:,} cell lines × {df.shape[1]:,} genes")
    return df


def load_gene_dependency(data_dir: Path) -> pd.DataFrame:
    """Load CRISPR gene dependency probability matrix (ModelID x Gene)."""
    print("Loading CRISPRGeneDependency.csv ...")
    df = pd.read_csv(data_dir / "CRISPRGeneDependency.csv", index_col=0)
    df.columns = [c.split(" (")[0] for c in df.columns]
    print(f"  Shape: {df.shape[0]:,} cell lines × {df.shape[1]:,} genes")
    return df


def load_expression(data_dir: Path) -> pd.DataFrame:
    """Load RNA-seq expression matrix (ModelID x Gene), log2(TPM+1)."""
    print("Loading OmicsExpressionProteinCodingGenesTPMLogp1.csv ...")
    df = pd.read_csv(
        data_dir / "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        index_col=0,
    )
    df.columns = [c.split(" (")[0] for c in df.columns]
    print(f"  Shape: {df.shape[0]:,} cell lines × {df.shape[1]:,} genes")
    return df


# =============================================================================
# 2. Core analysis
# =============================================================================

def compute_dependency_stats(
    gene_effect: pd.DataFrame,
    melanoma_ids: list[str],
) -> pd.DataFrame:
    """
    For each gene, compute:
      - mean effect in melanoma lines
      - mean effect in all other cancer lines
      - selectivity (melanoma mean − other mean); more negative = more selective
      - fraction of melanoma lines that are dependent (score < threshold)
    """
    mel_ids_present = [m for m in melanoma_ids if m in gene_effect.index]
    other_ids = [m for m in gene_effect.index if m not in set(melanoma_ids)]

    mel_effect = gene_effect.loc[mel_ids_present]
    other_effect = gene_effect.loc[other_ids]

    stats = pd.DataFrame({
        "mean_effect_melanoma": mel_effect.mean(),
        "mean_effect_other": other_effect.mean(),
        "std_effect_melanoma": mel_effect.std(),
        "n_melanoma_lines": mel_effect.notna().sum(),
        "frac_dependent_melanoma": (
            (mel_effect < DEPENDENCY_THRESHOLD).sum() / mel_effect.notna().sum()
        ),
        "frac_dependent_other": (
            (other_effect < DEPENDENCY_THRESHOLD).sum()
            / other_effect.notna().sum()
        ),
    })

    # Selectivity: negative value = melanoma more dependent than other cancers
    stats["selectivity"] = (
        stats["mean_effect_melanoma"] - stats["mean_effect_other"]
    )

    return stats


def add_dependency_probability(
    stats: pd.DataFrame,
    gene_dependency: pd.DataFrame,
    melanoma_ids: list[str],
) -> pd.DataFrame:
    """Add mean dependency probability for melanoma lines."""
    mel_ids_present = [m for m in melanoma_ids if m in gene_dependency.index]
    mel_dep = gene_dependency.loc[mel_ids_present]

    common_genes = stats.index.intersection(mel_dep.columns)
    stats.loc[common_genes, "mean_dep_prob_melanoma"] = (
        mel_dep[common_genes].mean()
    )
    return stats


def add_expression(
    stats: pd.DataFrame,
    expression: pd.DataFrame,
    melanoma_ids: list[str],
) -> pd.DataFrame:
    """Add mean expression (log2 TPM+1) in melanoma lines."""
    mel_ids_present = [m for m in melanoma_ids if m in expression.index]
    mel_expr = expression.loc[mel_ids_present]

    common_genes = stats.index.intersection(mel_expr.columns)
    stats.loc[common_genes, "mean_expr_melanoma"] = (
        mel_expr[common_genes].mean()
    )
    return stats


def rank_targets(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Apply filters and rank genes as drug target candidates.

    Filters:
      - Must be expressed (mean log2 TPM+1 > 1, i.e., TPM > 1)
      - Must be dependent in melanoma (mean effect < DEPENDENCY_THRESHOLD)
      - Must show some melanoma selectivity (selectivity < SELECTIVITY_THRESHOLD)

    Ranking: composite score = mean_effect_melanoma + selectivity
    (lower = better candidate).
    """
    candidates = stats[
        (stats["mean_effect_melanoma"] < DEPENDENCY_THRESHOLD)
        & (stats["selectivity"] < SELECTIVITY_THRESHOLD)
        & (stats["mean_expr_melanoma"] > 1.0)
    ].copy()

    candidates["composite_score"] = (
        candidates["mean_effect_melanoma"] + candidates["selectivity"]
    )

    candidates["is_known_target"] = candidates.index.isin(
        KNOWN_MELANOMA_TARGETS
    )

    candidates = candidates.sort_values("composite_score")
    candidates["rank"] = range(1, len(candidates) + 1)

    print(f"\n  Total candidate drug targets: {len(candidates):,}")
    print(
        f"  Known targets recovered: "
        f"{candidates['is_known_target'].sum()} / {len(KNOWN_MELANOMA_TARGETS)}"
    )
    print(
        f"  Novel targets (not in known list): "
        f"{(~candidates['is_known_target']).sum()}"
    )

    return candidates


# =============================================================================
# 3. Figures
# =============================================================================

def plot_top_dependencies(
    candidates: pd.DataFrame,
    n: int = 25,
) -> None:
    """Bar chart of top N melanoma dependencies by mean Chronos effect."""
    top = candidates.head(n).copy()

    colors = [
        KNOWN_COLOR if known else NOVEL_COLOR
        for known in top["is_known_target"]
    ]

    fig, ax = plt.subplots(figsize=(9, 7))

    bars = ax.barh(
        top.index[::-1],
        top["mean_effect_melanoma"][::-1],
        color=colors[::-1],
        edgecolor="none",
        height=0.7,
    )

    ax.axvline(
        DEPENDENCY_THRESHOLD,
        color="#E74C3C",
        linestyle="--",
        linewidth=1.2,
        label=f"Dependency threshold ({DEPENDENCY_THRESHOLD})",
    )
    ax.axvline(
        STRONG_DEPENDENCY_THRESHOLD,
        color="#922B21",
        linestyle=":",
        linewidth=1.2,
        label=f"Strong dependency ({STRONG_DEPENDENCY_THRESHOLD})",
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=KNOWN_COLOR, label="Known melanoma target"),
        Patch(facecolor=NOVEL_COLOR, label="Novel candidate"),
        plt.Line2D([0], [0], color="#E74C3C", linestyle="--",
                   label=f"Threshold ({DEPENDENCY_THRESHOLD})"),
        plt.Line2D([0], [0], color="#922B21", linestyle=":",
                   label=f"Strong ({STRONG_DEPENDENCY_THRESHOLD})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=False)

    ax.set_xlabel("Mean Chronos Gene Effect Score\n(more negative = more essential)")
    ax.set_title(
        f"Top {n} Melanoma Dependencies (DepMap 24Q2)",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(
        min(top["mean_effect_melanoma"].min() * 1.1, -1.5),
        0.1,
    )

    plt.tight_layout()
    path = FIG_DIR / "01_top_dependencies.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(ROOT)}")


def plot_selectivity_volcano(stats: pd.DataFrame) -> None:
    """
    Scatter plot: mean melanoma effect (y) vs selectivity (x).
    Top-left quadrant = most selective and most essential = best targets.
    """
    plot_df = stats.dropna(
        subset=["mean_effect_melanoma", "selectivity", "mean_expr_melanoma"]
    ).copy()

    plot_df["is_known"] = plot_df.index.isin(KNOWN_MELANOMA_TARGETS)
    plot_df["is_candidate"] = (
        (plot_df["mean_effect_melanoma"] < DEPENDENCY_THRESHOLD)
        & (plot_df["selectivity"] < SELECTIVITY_THRESHOLD)
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    # Background: all genes
    ax.scatter(
        plot_df.loc[~plot_df["is_candidate"], "selectivity"],
        plot_df.loc[~plot_df["is_candidate"], "mean_effect_melanoma"],
        c=OTHER_COLOR,
        alpha=0.3,
        s=8,
        linewidths=0,
        rasterized=True,
        label="Non-candidate genes",
    )

    # Novel candidates
    novel = plot_df[plot_df["is_candidate"] & ~plot_df["is_known"]]
    ax.scatter(
        novel["selectivity"],
        novel["mean_effect_melanoma"],
        c=NOVEL_COLOR,
        alpha=0.7,
        s=20,
        linewidths=0,
        label=f"Novel candidates (n={len(novel):,})",
    )

    # Known targets
    known = plot_df[plot_df["is_known"]]
    ax.scatter(
        known["selectivity"],
        known["mean_effect_melanoma"],
        c=KNOWN_COLOR,
        s=60,
        linewidths=0.5,
        edgecolors="black",
        zorder=5,
        label=f"Known targets (n={len(known)})",
    )

    # Label top novel candidates
    top_novel = novel.nsmallest(10, "mean_effect_melanoma")
    for gene, row in top_novel.iterrows():
        ax.annotate(
            gene,
            xy=(row["selectivity"], row["mean_effect_melanoma"]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=7,
            color=NOVEL_COLOR,
            va="center",
        )

    # Label known targets
    for gene, row in known.iterrows():
        ax.annotate(
            gene,
            xy=(row["selectivity"], row["mean_effect_melanoma"]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=7,
            color=KNOWN_COLOR,
            va="center",
            fontweight="bold",
        )

    ax.axhline(DEPENDENCY_THRESHOLD, color="#E74C3C", linestyle="--",
               linewidth=1, alpha=0.7)
    ax.axvline(SELECTIVITY_THRESHOLD, color="#E74C3C", linestyle="--",
               linewidth=1, alpha=0.7)

    ax.set_xlabel("Selectivity Score\n(melanoma − other cancers; more negative = more melanoma-selective)")
    ax.set_ylabel("Mean Chronos Effect in Melanoma\n(more negative = more essential)")
    ax.set_title(
        "Melanoma Dependency Selectivity\n"
        "Candidates in lower-left quadrant",
        fontweight="bold",
        pad=12,
    )
    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    path = FIG_DIR / "02_selectivity_volcano.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path.relative_to(ROOT)}")


def plot_dependency_vs_expression(candidates: pd.DataFrame) -> None:
    """
    Scatter of dependency score vs expression for top candidates.
    Targets should be both essential AND expressed.
    """
    plot_df = candidates.head(50).dropna(
        subset=["mean_effect_melanoma", "mean_expr_melanoma"]
    ).copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        plot_df["mean_expr_melanoma"],
        plot_df["mean_effect_melanoma"],
        c=plot_df["selectivity"],
        cmap="RdYlGn_r",
        s=60,
        alpha=0.85,
        edgecolors="none",
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Selectivity score", fontsize=9)

    for gene, row in plot_df.iterrows():
        ax.annotate(
            gene,
            xy=(row["mean_expr_melanoma"], row["mean_effect_melanoma"]),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=6.5,
            alpha=0.85,
        )

    ax.axhline(DEPENDENCY_THRESHOLD, color="#E74C3C", linestyle="--",
               linewidth=1, alpha=0.6, label=f"Dependency threshold")
    ax.axvline(1.0, color="#3498DB", linestyle="--",
               linewidth=1, alpha=0.6, label="Expression threshold (log2 TPM+1 = 1)")

    ax.set_xlabel("Mean Expression in Melanoma [log2(TPM+1)]")
    ax.set_ylabel("Mean Chronos Effect in Melanoma")
    ax.set_title(
        "Dependency vs Expression for Top 50 Candidates\n"
        "Colour = melanoma selectivity",
        fontweight="bold",
        pad=12,
    )
    ax.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    path = FIG_DIR / "03_dependency_vs_expression.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(ROOT)}")


def plot_dependency_probability_heatmap(
    gene_dependency: pd.DataFrame,
    candidates: pd.DataFrame,
    melanoma_ids: list[str],
    n_genes: int = 30,
    n_lines: int = 40,
) -> None:
    """
    Heatmap of dependency probability (0-1) across melanoma cell lines
    for the top candidate genes. Shows heterogeneity across lines.
    """
    top_genes = candidates.head(n_genes).index.tolist()
    mel_ids_present = [m for m in melanoma_ids if m in gene_dependency.index]

    # Limit cell lines for readability
    mel_ids_plot = mel_ids_present[:n_lines]

    genes_present = [g for g in top_genes if g in gene_dependency.columns]
    plot_df = gene_dependency.loc[mel_ids_plot, genes_present]

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        plot_df.T,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=0,
        cbar_kws={"label": "Dependency Probability", "shrink": 0.6},
        xticklabels=False,
    )

    ax.set_xlabel(f"Melanoma Cell Lines (n={len(mel_ids_plot)})")
    ax.set_ylabel("Gene")
    ax.set_title(
        f"Dependency Probability Heatmap\nTop {n_genes} Melanoma Candidates",
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    path = FIG_DIR / "04_dependency_probability_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.relative_to(ROOT)}")


# =============================================================================
# 4. Export results
# =============================================================================

def export_results(candidates: pd.DataFrame) -> None:
    """Write ranked target list to CSV."""
    out = candidates[[
        "rank",
        "mean_effect_melanoma",
        "mean_effect_other",
        "selectivity",
        "frac_dependent_melanoma",
        "frac_dependent_other",
        "mean_dep_prob_melanoma",
        "mean_expr_melanoma",
        "composite_score",
        "is_known_target",
        "n_melanoma_lines",
    ]].copy()

    out.columns = [
        "Rank",
        "Mean_Chronos_Effect_Melanoma",
        "Mean_Chronos_Effect_Other_Cancers",
        "Selectivity_Score",
        "Fraction_Dependent_Melanoma",
        "Fraction_Dependent_Other",
        "Mean_Dependency_Probability_Melanoma",
        "Mean_Expression_Melanoma_log2TPM",
        "Composite_Score",
        "Is_Known_Target",
        "N_Melanoma_Lines_Screened",
    ]

    out.index.name = "Gene"
    path = RES_DIR / "melanoma_drug_targets_ranked.csv"
    out.to_csv(path)
    print(f"  Saved: {path.relative_to(ROOT)}")


def export_summary(candidates: pd.DataFrame, melanoma_ids: list[str]) -> None:
    """Write a plain-text summary."""
    novel = candidates[~candidates["is_known_target"]]
    known_recovered = candidates[candidates["is_known_target"]]

    lines = [
        "Q4 DepMap Analysis — Melanoma Drug Target Discovery",
        "=" * 55,
        f"DepMap release: 24Q2",
        f"Melanoma cell lines: {len(melanoma_ids)}",
        "",
        "--- Filtering criteria ---",
        f"  Dependency threshold (Chronos): < {DEPENDENCY_THRESHOLD}",
        f"  Selectivity threshold: < {SELECTIVITY_THRESHOLD}",
        f"  Expression threshold: log2(TPM+1) > 1.0",
        "",
        "--- Results ---",
        f"  Total candidate drug targets:  {len(candidates)}",
        f"  Known targets recovered:       {len(known_recovered)} / {len(KNOWN_MELANOMA_TARGETS)}",
        f"  Novel candidates:              {len(novel)}",
        "",
        "--- Top 10 novel candidates ---",
    ]

    top_novel = novel.head(10)
    for gene, row in top_novel.iterrows():
        lines.append(
            f"  {gene:<12}  "
            f"effect={row['mean_effect_melanoma']:.3f}  "
            f"selectivity={row['selectivity']:.3f}  "
            f"expr={row['mean_expr_melanoma']:.2f}"
        )

    lines += [
        "",
        "--- Known targets recovered ---",
    ]
    for gene, row in known_recovered.iterrows():
        lines.append(
            f"  {gene:<12}  rank={int(row['rank'])}"
        )

    path = RES_DIR / "summary_stats.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {path.relative_to(ROOT)}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("Q4: Melanoma Drug Target Discovery via DepMap")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data...")
    model = load_model_metadata(DATA_DIR)
    melanoma_ids = get_melanoma_ids(model)

    gene_effect = load_gene_effect(DATA_DIR)
    gene_dependency = load_gene_dependency(DATA_DIR)
    expression = load_expression(DATA_DIR)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    print("\n[DIAG] Checking data alignment...")
    mel_in_effect = [m for m in melanoma_ids if m in gene_effect.index]
    mel_in_expr   = [m for m in melanoma_ids if m in expression.index]
    print(f"  Melanoma IDs in gene effect matrix: {len(mel_in_effect)}")
    print(f"  Melanoma IDs in expression matrix:  {len(mel_in_expr)}")
    print(f"  Gene effect index sample: {gene_effect.index[:3].tolist()}")
    print(f"  Model melanoma ID sample: {melanoma_ids[:3]}")
    print(f"  Gene effect col sample:   {gene_effect.columns[:5].tolist()}")
    print(f"  Expression col sample:    {expression.columns[:5].tolist()}")

    if len(mel_in_effect) == 0:
        raise ValueError(
            "No melanoma cell lines matched in CRISPRGeneEffect.csv. "
            "Check that Model.csv and CRISPRGeneEffect.csv are from the same release."
        )

    # ------------------------------------------------------------------
    # Compute statistics
    # ------------------------------------------------------------------
    print("\n[2/5] Computing dependency statistics...")
    stats = compute_dependency_stats(gene_effect, melanoma_ids)
    stats = add_dependency_probability(stats, gene_dependency, melanoma_ids)
    stats = add_expression(stats, expression, melanoma_ids)

    # Diagnostics on stats
    print(f"\n[DIAG] Stats summary:")
    print(f"  Genes with mean_effect_melanoma < {DEPENDENCY_THRESHOLD}: "
          f"{(stats['mean_effect_melanoma'] < DEPENDENCY_THRESHOLD).sum()}")
    print(f"  Genes with selectivity < {SELECTIVITY_THRESHOLD}: "
          f"{(stats['selectivity'] < SELECTIVITY_THRESHOLD).sum()}")
    print(f"  Genes with mean_expr_melanoma > 1.0: "
          f"{(stats['mean_expr_melanoma'] > 1.0).sum()}")
    print(f"  Effect range: {stats['mean_effect_melanoma'].min():.3f} to "
          f"{stats['mean_effect_melanoma'].max():.3f}")
    print(f"  Selectivity range: {stats['selectivity'].min():.3f} to "
          f"{stats['selectivity'].max():.3f}")
    print(f"  Expression range: {stats['mean_expr_melanoma'].min():.3f} to "
          f"{stats['mean_expr_melanoma'].max():.3f}")

    # ------------------------------------------------------------------
    # Rank targets
    # ------------------------------------------------------------------
    print("\n[3/5] Ranking drug target candidates...")
    candidates = rank_targets(stats)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\n[4/5] Generating figures...")
    plot_top_dependencies(candidates)
    plot_selectivity_volcano(stats)
    plot_dependency_vs_expression(candidates)
    plot_dependency_probability_heatmap(gene_dependency, candidates, melanoma_ids)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    print("\n[5/5] Exporting results...")
    export_results(candidates)
    export_summary(candidates, melanoma_ids)

    print("\n" + "=" * 60)
    print("Done. Outputs written to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()

    #gene annotation related to common pathways. 