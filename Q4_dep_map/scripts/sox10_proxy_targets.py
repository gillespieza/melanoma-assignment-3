#!/usr/bin/env python3
"""
Focused SOX10 proxy-target discovery extension.

Run this after melanoma_depmap_v2.py from the same Q4_DepMap folder.
It finds genes whose CRISPR dependencies co-vary with SOX10 and which are
also melanoma-dependent, melanoma-selective, expressed, and not common
essentials. It also creates a top-10 druggability audit template.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
V2_RESULTS_DIR = ROOT / "outputs_v2" / "results"
OUT_DIR = ROOT / "outputs_proxy"
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"
for directory in (FIG_DIR, RES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

TARGET_GENE = "SOX10"
MIN_PANCANCER_CORRELATION = 0.20
MAX_PANCANCER_FDR = 0.05
MAX_MEAN_EFFECT_MELANOMA = -0.30
MAX_SELECTIVITY = -0.10
MIN_MEAN_EXPRESSION = 1.00
MIN_EXPRESSION_PREVALENCE = 0.50
MIN_PAIRED_MODELS = 100
N_FOR_SPEARMAN = 500

REFERENCE_MELANOMA_TARGETS = {
    "BRAF", "MAP2K1", "MAP2K2", "MAPK1", "KIT", "CDK4", "CDK6",
    "NRAS", "RAF1",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run melanoma_depmap_v2.py first and check the folder structure."
        )


def clean_gene_symbol(value: object) -> str:
    text = str(value).strip()
    return text.split(" (", 1)[0] if " (" in text else text


def collapse_duplicate_gene_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_gene_symbol(c) for c in df.columns]
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0, sort=False).mean().T
    return df


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return pd.Series(adjusted, index=p_values.index)

    valid_positions = np.flatnonzero(valid)
    ordered_positions = valid_positions[np.argsort(p[valid])]
    ordered_p = p[ordered_positions]
    m = len(ordered_p)
    ordered_adjusted = ordered_p * m / np.arange(1, m + 1)
    ordered_adjusted = np.minimum.accumulate(ordered_adjusted[::-1])[::-1]
    ordered_adjusted = np.clip(ordered_adjusted, 0, 1)
    adjusted[ordered_positions] = ordered_adjusted
    return pd.Series(adjusted, index=p_values.index)


def correlation_p_values(r: pd.Series, paired_n: pd.Series) -> pd.Series:
    r = r.clip(-0.999999, 0.999999)
    degrees_freedom = paired_n - 2
    statistic = r * np.sqrt(degrees_freedom / (1 - r.pow(2)))
    p = pd.Series(
        2 * student_t.sf(np.abs(statistic), df=degrees_freedom),
        index=r.index,
    )
    p.loc[degrees_freedom <= 0] = np.nan
    return p


def load_gene_effect() -> pd.DataFrame:
    path = DATA_DIR / "CRISPRGeneEffect.csv"
    require_file(path)
    print("Loading CRISPRGeneEffect.csv ...")
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df = collapse_duplicate_gene_columns(df)
    print(f"  {df.shape[0]:,} models x {df.shape[1]:,} genes")
    return df


def load_model_metadata() -> pd.DataFrame:
    path = DATA_DIR / "Model.csv"
    require_file(path)
    model = pd.read_csv(path)
    if "ModelID" not in model.columns:
        raise ValueError("Model.csv does not contain ModelID.")
    return model


def select_melanoma_ids(model: pd.DataFrame) -> list[str]:
    masks = []
    if "OncotreePrimaryDisease" in model.columns:
        masks.append(
            model["OncotreePrimaryDisease"]
            .astype("string")
            .str.fullmatch("Melanoma", case=False, na=False)
        )
    if "OncotreeSubtype" in model.columns:
        masks.append(
            model["OncotreeSubtype"]
            .astype("string")
            .str.contains("Melanoma", case=False, na=False)
        )
    if not masks:
        raise ValueError(
            "Model.csv has no OncotreePrimaryDisease or OncotreeSubtype."
        )
    mask = masks[0].copy()
    for additional in masks[1:]:
        mask = mask | additional
    ids = (
        model.loc[mask, "ModelID"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    print(f"Explicit melanoma models: {len(ids):,}")
    return ids


def load_v2_statistics() -> pd.DataFrame:
    path = V2_RESULTS_DIR / "all_gene_statistics.csv"
    require_file(path)
    stats = pd.read_csv(path, index_col=0)
    stats.index = stats.index.map(clean_gene_symbol)
    stats.index.name = "Gene"
    required = {
        "mean_effect_melanoma", "mean_effect_other", "selectivity", "fdr_bh",
        "frac_dependent_melanoma", "frac_dependent_other",
        "mean_dep_prob_melanoma", "mean_expr_melanoma",
        "frac_expressed_melanoma", "is_common_essential",
    }
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(
            "all_gene_statistics.csv is missing: " + ", ".join(sorted(missing))
        )
    return stats


def compute_codependency(
    gene_effect: pd.DataFrame,
    melanoma_ids: list[str],
) -> pd.DataFrame:
    if TARGET_GENE not in gene_effect.columns:
        raise KeyError(f"{TARGET_GENE} is absent from CRISPRGeneEffect.csv.")

    target_all = gene_effect[TARGET_GENE]
    pancancer_r = gene_effect.corrwith(target_all, axis=0)
    paired_n = gene_effect.notna().mul(target_all.notna(), axis=0).sum(axis=0)
    pancancer_p = correlation_p_values(pancancer_r, paired_n)

    results = pd.DataFrame({
        "pancancer_pearson_r": pancancer_r,
        "pancancer_p_value": pancancer_p,
        "pancancer_fdr": bh_fdr(pancancer_p),
        "pancancer_paired_n": paired_n,
    })
    results.index.name = "Gene"

    mel_ids_present = [m for m in melanoma_ids if m in gene_effect.index]
    melanoma_effect = gene_effect.loc[mel_ids_present]
    target_melanoma = melanoma_effect[TARGET_GENE]

    results["melanoma_pearson_r"] = melanoma_effect.corrwith(
        target_melanoma, axis=0
    )
    results["melanoma_paired_n"] = melanoma_effect.notna().mul(
        target_melanoma.notna(), axis=0
    ).sum(axis=0)

    top_for_spearman = (
        results.drop(index=TARGET_GENE, errors="ignore")
        .sort_values("pancancer_pearson_r", ascending=False)
        .head(N_FOR_SPEARMAN)
        .index
    )
    results["melanoma_spearman_rho"] = np.nan
    results["melanoma_spearman_p"] = np.nan

    for gene in top_for_spearman:
        pair = melanoma_effect[[TARGET_GENE, gene]].dropna()
        if len(pair) < 5:
            continue
        rho, p_value = spearmanr(pair[TARGET_GENE], pair[gene])
        results.loc[gene, "melanoma_spearman_rho"] = rho
        results.loc[gene, "melanoma_spearman_p"] = p_value

    return results


def merge_and_filter(
    codependency: pd.DataFrame,
    v2_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = codependency.join(v2_stats, how="left")
    merged["is_reference_melanoma_target"] = merged.index.isin(
        REFERENCE_MELANOMA_TARGETS
    )
    merged = merged.drop(index=TARGET_GENE, errors="ignore")

    common_essential = merged["is_common_essential"].fillna(False).astype(bool)
    mask = (
        (merged["pancancer_pearson_r"] >= MIN_PANCANCER_CORRELATION)
        & (merged["pancancer_fdr"] <= MAX_PANCANCER_FDR)
        & (merged["pancancer_paired_n"] >= MIN_PAIRED_MODELS)
        & (merged["mean_effect_melanoma"] <= MAX_MEAN_EFFECT_MELANOMA)
        & (merged["selectivity"] <= MAX_SELECTIVITY)
        & (merged["mean_expr_melanoma"] >= MIN_MEAN_EXPRESSION)
        & (merged["frac_expressed_melanoma"] >= MIN_EXPRESSION_PREVALENCE)
        & (~common_essential)
    )
    candidates = merged.loc[mask].copy()
    if candidates.empty:
        return merged, candidates

    candidates["melanoma_spearman_for_rank"] = (
        candidates["melanoma_spearman_rho"].fillna(-1)
    )
    candidates["prevalence_advantage"] = (
        candidates["frac_dependent_melanoma"]
        - candidates["frac_dependent_other"]
    )
    candidates["rank_codependency"] = candidates[
        "pancancer_pearson_r"
    ].rank(pct=True, ascending=False)
    candidates["rank_within_melanoma"] = candidates[
        "melanoma_spearman_for_rank"
    ].rank(pct=True, ascending=False)
    candidates["rank_melanoma_effect"] = candidates[
        "mean_effect_melanoma"
    ].rank(pct=True, ascending=True)
    candidates["rank_selectivity"] = candidates[
        "selectivity"
    ].rank(pct=True, ascending=True)
    candidates["rank_prevalence"] = candidates[
        "prevalence_advantage"
    ].rank(pct=True, ascending=False)

    candidates["proxy_priority_score"] = (
        0.35 * candidates["rank_codependency"]
        + 0.15 * candidates["rank_within_melanoma"]
        + 0.20 * candidates["rank_melanoma_effect"]
        + 0.20 * candidates["rank_selectivity"]
        + 0.10 * candidates["rank_prevalence"]
    )
    candidates = candidates.sort_values(
        ["proxy_priority_score", "pancancer_pearson_r", "selectivity"],
        ascending=[True, False, True],
    )
    candidates["proxy_rank"] = np.arange(1, len(candidates) + 1)
    return merged, candidates


def make_druggability_template(candidates: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    columns = [
        "proxy_rank", "pancancer_pearson_r", "melanoma_pearson_r",
        "melanoma_spearman_rho", "mean_effect_melanoma", "selectivity",
        "frac_dependent_melanoma", "frac_dependent_other",
        "mean_expr_melanoma", "fdr_bh", "is_reference_melanoma_target",
    ]
    available = [c for c in columns if c in candidates.columns]
    audit = candidates.head(n)[available].copy()
    audit.insert(0, "Gene", audit.index)
    audit["Target_class"] = ""
    audit["Existing_ligand_or_drug"] = ""
    audit["Highest_development_stage"] = ""
    audit["Directly_druggable"] = ""
    audit["Potential_indirect_strategy"] = ""
    audit["Prior_melanoma_target_evidence"] = ""
    audit["Novelty_category"] = ""
    audit["Safety_or_normal_tissue_concern"] = ""
    audit["Final_keep_or_reject"] = ""
    audit["Evidence_notes"] = ""
    return audit.reset_index(drop=True)


def plot_codependency_selectivity(
    merged: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    plot_df = merged.dropna(subset=["pancancer_pearson_r", "selectivity"])
    fig, ax = plt.subplots(figsize=(9, 7))
    background = plot_df.loc[~plot_df.index.isin(candidates.index)]
    ax.scatter(
        background["pancancer_pearson_r"],
        background["selectivity"],
        s=8,
        alpha=0.12,
        label="Other genes",
    )
    if not candidates.empty:
        ax.scatter(
            candidates["pancancer_pearson_r"],
            candidates["selectivity"],
            s=35,
            alpha=0.8,
            label=f"Proxy candidates (n={len(candidates)})",
        )
        for gene, row in candidates.head(15).iterrows():
            ax.annotate(
                gene,
                (row["pancancer_pearson_r"], row["selectivity"]),
                xytext=(4, 2),
                textcoords="offset points",
                fontsize=7,
            )
    ax.axvline(MIN_PANCANCER_CORRELATION, linestyle="--", linewidth=1)
    ax.axhline(MAX_SELECTIVITY, linestyle="--", linewidth=1)
    ax.set_xlabel(
        f"Pan-cancer co-dependency with {TARGET_GENE}\n"
        "(Pearson correlation of Chronos scores)"
    )
    ax.set_ylabel(
        "Melanoma selectivity\n"
        "(more negative = more melanoma-selective)"
    )
    ax.set_title(f"{TARGET_GENE} co-dependency and melanoma selectivity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "01_sox10_codependency_selectivity.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_top_proxy_candidates(candidates: pd.DataFrame, n: int = 20) -> None:
    if candidates.empty:
        return
    top = candidates.head(n).sort_values("proxy_priority_score", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.index, top["proxy_priority_score"])
    ax.set_xlabel("Proxy-priority score (lower = stronger combined evidence)")
    ax.set_title(f"Top candidate therapeutic proxies for {TARGET_GENE}")
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "02_top_proxy_candidates.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_summary(candidates: pd.DataFrame) -> None:
    lines = [
        "SOX10 druggable-proxy discovery summary",
        "=" * 41,
        "",
        "Question:",
        (
            "Which genes show SOX10 co-dependency together with melanoma "
            "dependency, melanoma selectivity, and melanoma expression?"
        ),
        "",
        f"Candidates retained: {len(candidates)}",
        "",
        "Top candidates:",
    ]
    if candidates.empty:
        lines.append("  None passed all screening rules.")
    else:
        for gene, row in candidates.head(15).iterrows():
            lines.append(
                f"  {int(row['proxy_rank']):>2}. {gene:<12} "
                f"co-dep={row['pancancer_pearson_r']:.3f}, "
                f"mel-effect={row['mean_effect_melanoma']:.3f}, "
                f"selectivity={row['selectivity']:.3f}, "
                f"mel-prev={row['frac_dependent_melanoma']:.2f}"
            )
    lines += [
        "",
        "Interpretation:",
        (
            "These genes are candidate therapeutic entry points into the "
            "SOX10-dependent melanoma state. Co-dependency does not prove "
            "direct regulation, physical interaction, or druggability."
        ),
        "",
        "Next action:",
        (
            "Complete top10_druggability_audit_template.csv using Open "
            "Targets, ChEMBL, and a focused melanoma literature search."
        ),
    ]
    (RES_DIR / "sox10_proxy_summary.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    print("=" * 68)
    print("SOX10 druggable-proxy discovery")
    print("=" * 68)

    gene_effect = load_gene_effect()
    model = load_model_metadata()
    melanoma_ids = select_melanoma_ids(model)
    v2_stats = load_v2_statistics()

    print("\nComputing SOX10 co-dependencies ...")
    codependency = compute_codependency(gene_effect, melanoma_ids)
    merged, candidates = merge_and_filter(codependency, v2_stats)

    merged.sort_values("pancancer_pearson_r", ascending=False).to_csv(
        RES_DIR / "all_sox10_codependencies.csv"
    )
    candidates.to_csv(RES_DIR / "sox10_proxy_candidates_ranked.csv")
    make_druggability_template(candidates, n=10).to_csv(
        RES_DIR / "top10_druggability_audit_template.csv", index=False
    )

    plot_codependency_selectivity(merged, candidates)
    plot_top_proxy_candidates(candidates)
    write_summary(candidates)

    print(f"\nCandidates retained: {len(candidates)}")
    if not candidates.empty:
        columns = [
            "proxy_rank", "pancancer_pearson_r", "melanoma_spearman_rho",
            "mean_effect_melanoma", "selectivity",
            "frac_dependent_melanoma",
        ]
        print(candidates.head(15)[columns].round(3).to_string())
    print(f"\nOutputs written to: {OUT_DIR}")
    print("=" * 68)


if __name__ == "__main__":
    main()
