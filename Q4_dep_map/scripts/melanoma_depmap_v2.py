#!/usr/bin/env python3
"""
Melanoma DepMap target-prioritisation workflow: v2

Main improvements over v1
-------------------------
1. Selects melanoma using OncotreePrimaryDisease / OncotreeSubtype rather
   than treating every "Skin" model as melanoma.
2. Removes inferred common-essential genes before candidate ranking.
3. Tests melanoma-versus-other-cancer dependency statistically and applies
   Benjamini-Hochberg FDR correction.
4. Uses dependency prevalence, dependency probability, expression prevalence,
   and lineage selectivity rather than a simple sum of two mean scores.
5. Adds SOX10-specific bootstrap, leave-one-out, threshold-sensitivity,
   expression/dependency, and per-cell-line heterogeneity analyses.
6. Uses careful terminology: "candidate melanoma-selective dependency", not
   automatically "novel drug target".
7. Optionally prepares up/down gene lists for a LINCS query when an independent
   SOX10-knockdown differential-expression file is supplied.

Expected input files in ./data
------------------------------
Required:
    Model.csv
    CRISPRGeneEffect.csv
    CRISPRGeneDependency.csv
    OmicsExpressionProteinCodingGenesTPMLogp1.csv

Strongly recommended:
    CRISPRInferredCommonEssentials.csv

Optional LINCS-signature input:
    SOX10_KD_DEG.csv

The optional differential-expression file should contain:
    gene / Gene / symbol
    log2FoldChange / log2FC
    padj / FDR / qvalue

Outputs are written to ./outputs_v2.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs_v2"
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"

for directory in (OUT_DIR, FIG_DIR, RES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

TARGET_GENE = "SOX10"
RANDOM_SEED = 42

# Set True only as a sensitivity analysis if Model.csv contains PatientID.
# It retains one model per patient and can remove non-independent derivatives.
DEDUPLICATE_BY_PATIENT = False

# Candidate thresholds. Treat these as transparent screening rules and assess
# robustness with the threshold-sensitivity output.
DEPENDENCY_THRESHOLD = -0.50
SELECTIVITY_THRESHOLD = -0.20
FDR_THRESHOLD = 0.05
MIN_FRAC_DEPENDENT_MELANOMA = 0.40
MAX_FRAC_DEPENDENT_OTHER = 0.20
MIN_MEAN_DEPENDENCY_PROBABILITY = 0.50
MIN_MEAN_EXPRESSION = 1.00
MIN_EXPRESSION_PREVALENCE = 0.50
MIN_MELANOMA_LINES = 10

# SOX10 robustness settings
N_BOOTSTRAP = 1000

# Optional independent SOX10 perturbation signature
SOX10_DEG_FILE = DATA_DIR / "SOX10_KD_DEG.csv"
LINCS_SIGNATURE_SIZE = 150

# Tumour-cell intrinsic reference genes for annotation only.
# These are not used as a formal gold-standard set.
REFERENCE_GENES = {
    "SOX10", "MITF", "MLANA", "PMEL", "TYR", "DCT",
    "BRAF", "MAP2K1", "MAP2K2", "KIT", "CDK4", "CDK6",
}


# =============================================================================
# Utility functions
# =============================================================================

def clean_gene_symbol(value: object) -> str:
    """Convert 'GENE (EntrezID)' or similar labels to a gene symbol."""
    text = str(value).strip()
    if " (" in text:
        text = text.split(" (", 1)[0]
    return text


def collapse_duplicate_gene_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip Entrez IDs and average columns that collapse to the same gene symbol.
    """
    df = df.copy()
    df.columns = [clean_gene_symbol(c) for c in df.columns]

    if df.columns.duplicated().any():
        duplicated = int(df.columns.duplicated().sum())
        print(f"  Averaging {duplicated:,} duplicate gene-symbol columns.")
        df = df.T.groupby(level=0, sort=False).mean().T

    return df


def bh_fdr(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR correction that preserves the input index."""
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(p), np.nan, dtype=float)

    valid = np.isfinite(p)
    if not valid.any():
        return pd.Series(result, index=p_values.index, name="fdr_bh")

    valid_positions = np.flatnonzero(valid)
    ordered_positions = valid_positions[np.argsort(p[valid])]
    ordered_p = p[ordered_positions]
    m = len(ordered_p)

    adjusted = ordered_p * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result[ordered_positions] = adjusted
    return pd.Series(result, index=p_values.index, name="fdr_bh")


def hedges_g(
    mean_1: pd.Series,
    mean_2: pd.Series,
    std_1: pd.Series,
    std_2: pd.Series,
    n_1: pd.Series,
    n_2: pd.Series,
) -> pd.Series:
    """
    Vectorised Hedges' g.
    Negative values mean stronger dependency in melanoma.
    """
    denominator_df = n_1 + n_2 - 2
    pooled_variance = (
        ((n_1 - 1) * std_1.pow(2) + (n_2 - 1) * std_2.pow(2))
        / denominator_df.replace(0, np.nan)
    )
    pooled_sd = np.sqrt(pooled_variance)
    cohen_d = (mean_1 - mean_2) / pooled_sd.replace(0, np.nan)

    correction = 1 - (3 / (4 * (n_1 + n_2) - 9))
    return cohen_d * correction


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Place the file in the script's data/ directory."
        )


# =============================================================================
# Data loading and melanoma definition
# =============================================================================

def load_model_metadata() -> pd.DataFrame:
    path = DATA_DIR / "Model.csv"
    require_file(path)

    model = pd.read_csv(path)
    if "ModelID" not in model.columns:
        raise ValueError("Model.csv must contain a ModelID column.")

    print(f"Loaded Model.csv: {len(model):,} models")
    return model


def select_melanoma_models(model: pd.DataFrame) -> pd.DataFrame:
    """
    Select melanoma explicitly, avoiding the broad Skin-lineage definition.
    """
    masks: list[pd.Series] = []

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
            "Model.csv contains neither OncotreePrimaryDisease nor "
            "OncotreeSubtype. Do not silently fall back to all Skin models."
        )

    mask = masks[0].copy()
    for additional_mask in masks[1:]:
        mask = mask | additional_mask

    melanoma = model.loc[mask].copy()
    melanoma = melanoma.dropna(subset=["ModelID"]).drop_duplicates("ModelID")

    if melanoma.empty:
        raise ValueError(
            "No melanoma models were selected. Inspect the Oncotree columns "
            "and update select_melanoma_models() for this DepMap release."
        )

    if "PatientID" in melanoma.columns:
        repeated = melanoma["PatientID"].dropna().duplicated(keep=False)
        n_related_models = int(repeated.sum())
        n_repeated_patients = int(
            melanoma.loc[repeated, "PatientID"].nunique()
        )
        print(
            "Related-model diagnostic: "
            f"{n_related_models} models from {n_repeated_patients} repeated patients."
        )

        if DEDUPLICATE_BY_PATIENT:
            before = len(melanoma)
            melanoma = (
                melanoma.sort_values("ModelID")
                .drop_duplicates("PatientID", keep="first")
            )
            print(
                f"Patient-level sensitivity mode: retained "
                f"{len(melanoma):,}/{before:,} melanoma models."
            )

    print(f"Explicit melanoma models selected: {len(melanoma):,}")

    diagnostic_columns = [
        c for c in [
            "ModelID",
            "PatientID",
            "OncotreeLineage",
            "OncotreePrimaryDisease",
            "OncotreeSubtype",
        ]
        if c in melanoma.columns
    ]
    melanoma[diagnostic_columns].to_csv(
        RES_DIR / "selected_melanoma_models.csv",
        index=False,
    )
    return melanoma


def load_matrix(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    require_file(path)

    print(f"Loading {filename} ...")
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df = collapse_duplicate_gene_columns(df)
    print(f"  Shape: {df.shape[0]:,} models × {df.shape[1]:,} genes")
    return df


def load_common_essentials() -> set[str]:
    """
    Load the DepMap inferred-common-essential list.

    The function handles a one-column gene list, a named Gene column, or a
    file where genes were written to the index.
    """
    path = DATA_DIR / "CRISPRInferredCommonEssentials.csv"

    if not path.exists():
        print(
            "WARNING: CRISPRInferredCommonEssentials.csv is missing. "
            "Candidate ranking will run, but pan-essential genes will not be "
            "formally removed."
        )
        return set()

    raw = pd.read_csv(path)

    candidate_columns = [
        c for c in raw.columns
        if str(c).lower() in {"gene", "genes", "symbol", "gene_symbol"}
    ]

    if candidate_columns:
        values = raw[candidate_columns[0]]
    elif raw.shape[1] == 1:
        values = raw.iloc[:, 0]
    else:
        # Retry with the first column as index, which covers some releases.
        indexed = pd.read_csv(path, index_col=0)
        values = pd.Series(indexed.index)

    genes = {
        clean_gene_symbol(x)
        for x in values.dropna().astype(str)
        if clean_gene_symbol(x)
    }

    print(f"Loaded {len(genes):,} inferred common-essential genes.")
    return genes


# =============================================================================
# Core dependency statistics
# =============================================================================

def align_groups(
    matrix: pd.DataFrame,
    melanoma_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    melanoma_set = set(melanoma_ids)
    mel_present = [m for m in melanoma_ids if m in matrix.index]
    other_present = [m for m in matrix.index if m not in melanoma_set]

    if len(mel_present) < MIN_MELANOMA_LINES:
        raise ValueError(
            f"Only {len(mel_present)} melanoma models overlap this matrix; "
            f"at least {MIN_MELANOMA_LINES} are required."
        )

    return matrix.loc[mel_present], matrix.loc[other_present]


def compute_dependency_statistics(
    gene_effect: pd.DataFrame,
    melanoma_ids: list[str],
) -> pd.DataFrame:
    mel, other = align_groups(gene_effect, melanoma_ids)

    common_genes = mel.columns.intersection(other.columns)
    mel = mel[common_genes]
    other = other[common_genes]

    stats = pd.DataFrame(index=common_genes)
    stats.index.name = "Gene"

    stats["mean_effect_melanoma"] = mel.mean(axis=0)
    stats["median_effect_melanoma"] = mel.median(axis=0)
    stats["std_effect_melanoma"] = mel.std(axis=0)
    stats["n_melanoma"] = mel.notna().sum(axis=0)

    stats["mean_effect_other"] = other.mean(axis=0)
    stats["median_effect_other"] = other.median(axis=0)
    stats["std_effect_other"] = other.std(axis=0)
    stats["n_other"] = other.notna().sum(axis=0)

    stats["selectivity"] = (
        stats["mean_effect_melanoma"] - stats["mean_effect_other"]
    )

    stats["frac_dependent_melanoma"] = (
        mel.lt(DEPENDENCY_THRESHOLD).sum(axis=0)
        / mel.notna().sum(axis=0).replace(0, np.nan)
    )
    stats["frac_dependent_other"] = (
        other.lt(DEPENDENCY_THRESHOLD).sum(axis=0)
        / other.notna().sum(axis=0).replace(0, np.nan)
    )
    stats["dependency_prevalence_difference"] = (
        stats["frac_dependent_melanoma"]
        - stats["frac_dependent_other"]
    )

    # Vectorised Welch test across all genes.
    test = ttest_ind(
        mel.to_numpy(dtype=float),
        other.to_numpy(dtype=float),
        axis=0,
        equal_var=False,
        nan_policy="omit",
    )
    stats["welch_t"] = test.statistic
    stats["p_value"] = test.pvalue
    stats["fdr_bh"] = bh_fdr(stats["p_value"])

    stats["hedges_g"] = hedges_g(
        stats["mean_effect_melanoma"],
        stats["mean_effect_other"],
        stats["std_effect_melanoma"],
        stats["std_effect_other"],
        stats["n_melanoma"],
        stats["n_other"],
    )

    return stats


def add_dependency_probability(
    stats: pd.DataFrame,
    gene_dependency: pd.DataFrame,
    melanoma_ids: list[str],
) -> pd.DataFrame:
    mel, other = align_groups(gene_dependency, melanoma_ids)
    common = stats.index.intersection(mel.columns).intersection(other.columns)

    stats.loc[common, "mean_dep_prob_melanoma"] = mel[common].mean(axis=0)
    stats.loc[common, "mean_dep_prob_other"] = other[common].mean(axis=0)
    stats.loc[common, "frac_prob_gt_05_melanoma"] = (
        mel[common].gt(0.5).sum(axis=0)
        / mel[common].notna().sum(axis=0).replace(0, np.nan)
    )
    stats.loc[common, "frac_prob_gt_05_other"] = (
        other[common].gt(0.5).sum(axis=0)
        / other[common].notna().sum(axis=0).replace(0, np.nan)
    )
    return stats


def add_expression_statistics(
    stats: pd.DataFrame,
    expression: pd.DataFrame,
    melanoma_ids: list[str],
) -> pd.DataFrame:
    mel, _ = align_groups(expression, melanoma_ids)
    common = stats.index.intersection(mel.columns)

    stats.loc[common, "mean_expr_melanoma"] = mel[common].mean(axis=0)
    stats.loc[common, "median_expr_melanoma"] = mel[common].median(axis=0)
    stats.loc[common, "frac_expressed_melanoma"] = (
        mel[common].gt(MIN_MEAN_EXPRESSION).sum(axis=0)
        / mel[common].notna().sum(axis=0).replace(0, np.nan)
    )
    return stats


def annotate_and_rank(
    stats: pd.DataFrame,
    common_essentials: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = stats.copy()
    stats["is_common_essential"] = stats.index.isin(common_essentials)
    stats["is_reference_gene"] = stats.index.isin(REFERENCE_GENES)

    required = [
        "mean_effect_melanoma",
        "selectivity",
        "fdr_bh",
        "frac_dependent_melanoma",
        "frac_dependent_other",
        "mean_dep_prob_melanoma",
        "mean_expr_melanoma",
        "frac_expressed_melanoma",
    ]
    analysis_ready = stats.dropna(subset=required).copy()

    candidate_mask = (
        (~analysis_ready["is_common_essential"])
        & (analysis_ready["n_melanoma"] >= MIN_MELANOMA_LINES)
        & (
            analysis_ready["mean_effect_melanoma"]
            < DEPENDENCY_THRESHOLD
        )
        & (analysis_ready["selectivity"] < SELECTIVITY_THRESHOLD)
        & (analysis_ready["fdr_bh"] < FDR_THRESHOLD)
        & (
            analysis_ready["frac_dependent_melanoma"]
            >= MIN_FRAC_DEPENDENT_MELANOMA
        )
        & (
            analysis_ready["frac_dependent_other"]
            <= MAX_FRAC_DEPENDENT_OTHER
        )
        & (
            analysis_ready["mean_dep_prob_melanoma"]
            >= MIN_MEAN_DEPENDENCY_PROBABILITY
        )
        & (
            analysis_ready["mean_expr_melanoma"]
            >= MIN_MEAN_EXPRESSION
        )
        & (
            analysis_ready["frac_expressed_melanoma"]
            >= MIN_EXPRESSION_PREVALENCE
        )
    )

    candidates = analysis_ready.loc[candidate_mask].copy()

    if candidates.empty:
        print(
            "WARNING: No candidates passed every threshold. "
            "Inspect threshold_sensitivity.csv and relax rules transparently."
        )
        stats["is_candidate"] = False
        return stats, candidates

    # Lower score = higher priority. Rank aggregation prevents one extreme
    # metric from dominating the entire prioritisation.
    candidates["rank_effect"] = candidates[
        "mean_effect_melanoma"
    ].rank(method="average", pct=True, ascending=True)

    candidates["rank_selectivity"] = candidates[
        "selectivity"
    ].rank(method="average", pct=True, ascending=True)

    candidates["rank_prevalence"] = candidates[
        "dependency_prevalence_difference"
    ].rank(method="average", pct=True, ascending=False)

    candidates["rank_probability"] = candidates[
        "mean_dep_prob_melanoma"
    ].rank(method="average", pct=True, ascending=False)

    candidates["rank_fdr"] = candidates[
        "fdr_bh"
    ].rank(method="average", pct=True, ascending=True)

    candidates["priority_score"] = (
        0.30 * candidates["rank_effect"]
        + 0.30 * candidates["rank_selectivity"]
        + 0.20 * candidates["rank_prevalence"]
        + 0.10 * candidates["rank_probability"]
        + 0.10 * candidates["rank_fdr"]
    )

    candidates = candidates.sort_values(
        ["priority_score", "selectivity", "mean_effect_melanoma"]
    )
    candidates["candidate_rank"] = np.arange(1, len(candidates) + 1)

    stats["is_candidate"] = stats.index.isin(candidates.index)
    stats.loc[candidates.index, "candidate_rank"] = candidates[
        "candidate_rank"
    ]
    stats.loc[candidates.index, "priority_score"] = candidates[
        "priority_score"
    ]

    return stats, candidates


# =============================================================================
# SOX10-specific validation
# =============================================================================

def target_cell_line_table(
    gene_effect: pd.DataFrame,
    gene_dependency: pd.DataFrame,
    expression: pd.DataFrame,
    melanoma_metadata: pd.DataFrame,
    gene: str,
) -> pd.DataFrame:
    for matrix_name, matrix in [
        ("gene effect", gene_effect),
        ("gene dependency", gene_dependency),
        ("expression", expression),
    ]:
        if gene not in matrix.columns:
            raise KeyError(f"{gene} is missing from the {matrix_name} matrix.")

    ids = (
        set(melanoma_metadata["ModelID"])
        & set(gene_effect.index)
        & set(gene_dependency.index)
        & set(expression.index)
    )
    ids = sorted(ids)

    table = melanoma_metadata[
        melanoma_metadata["ModelID"].isin(ids)
    ].copy()
    table = table.drop_duplicates("ModelID").set_index("ModelID")

    table[f"{gene}_gene_effect"] = gene_effect.loc[
        table.index, gene
    ]
    table[f"{gene}_dependency_probability"] = gene_dependency.loc[
        table.index, gene
    ]
    table[f"{gene}_expression_log2TPM1"] = expression.loc[
        table.index, gene
    ]

    return table.reset_index()


def bootstrap_target_metrics(
    gene_effect: pd.DataFrame,
    melanoma_ids: list[str],
    gene: str,
    n_bootstrap: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    if gene not in gene_effect.columns:
        raise KeyError(f"{gene} not found in CRISPRGeneEffect.csv.")

    mel, other = align_groups(gene_effect[[gene]], melanoma_ids)
    mel_values = mel[gene].dropna().to_numpy(dtype=float)
    other_values = other[gene].dropna().to_numpy(dtype=float)

    if len(mel_values) < 2 or len(other_values) < 2:
        raise ValueError("Insufficient non-missing values for bootstrap.")

    rng = np.random.default_rng(RANDOM_SEED)
    records = []

    for iteration in range(n_bootstrap):
        mel_sample = rng.choice(
            mel_values, size=len(mel_values), replace=True
        )
        other_sample = rng.choice(
            other_values, size=len(other_values), replace=True
        )

        mel_mean = float(np.mean(mel_sample))
        other_mean = float(np.mean(other_sample))

        records.append({
            "iteration": iteration + 1,
            "mean_effect_melanoma": mel_mean,
            "mean_effect_other": other_mean,
            "selectivity": mel_mean - other_mean,
            "frac_dependent_melanoma": float(
                np.mean(mel_sample < DEPENDENCY_THRESHOLD)
            ),
            "frac_dependent_other": float(
                np.mean(other_sample < DEPENDENCY_THRESHOLD)
            ),
        })

    return pd.DataFrame.from_records(records)


def leave_one_out_target(
    gene_effect: pd.DataFrame,
    melanoma_ids: list[str],
    gene: str,
) -> pd.DataFrame:
    if gene not in gene_effect.columns:
        raise KeyError(f"{gene} not found in CRISPRGeneEffect.csv.")

    mel, other = align_groups(gene_effect[[gene]], melanoma_ids)
    other_mean = float(other[gene].mean())

    rows = []
    for omitted_id in mel.index:
        retained = mel.drop(index=omitted_id)[gene]
        mel_mean = float(retained.mean())

        rows.append({
            "omitted_model_id": omitted_id,
            "retained_n": int(retained.notna().sum()),
            "mean_effect_melanoma": mel_mean,
            "mean_effect_other": other_mean,
            "selectivity": mel_mean - other_mean,
            "frac_dependent_melanoma": float(
                retained.lt(DEPENDENCY_THRESHOLD).mean()
            ),
        })

    return pd.DataFrame(rows)


def threshold_sensitivity(
    stats: pd.DataFrame,
    common_essentials: set[str],
    gene: str,
) -> pd.DataFrame:
    dependency_thresholds = [-0.40, -0.50, -0.60, -0.75]
    selectivity_thresholds = [-0.10, -0.20, -0.30, -0.40]

    rows = []

    for dep_threshold in dependency_thresholds:
        for sel_threshold in selectivity_thresholds:
            mask = (
                (~stats.index.isin(common_essentials))
                & (stats["mean_effect_melanoma"] < dep_threshold)
                & (stats["selectivity"] < sel_threshold)
                & (stats["fdr_bh"] < FDR_THRESHOLD)
                & (
                    stats["frac_dependent_melanoma"]
                    >= MIN_FRAC_DEPENDENT_MELANOMA
                )
                & (
                    stats["frac_dependent_other"]
                    <= MAX_FRAC_DEPENDENT_OTHER
                )
                & (
                    stats["mean_expr_melanoma"]
                    >= MIN_MEAN_EXPRESSION
                )
                & (
                    stats["frac_expressed_melanoma"]
                    >= MIN_EXPRESSION_PREVALENCE
                )
            )

            rows.append({
                "dependency_threshold": dep_threshold,
                "selectivity_threshold": sel_threshold,
                "n_candidates": int(mask.sum()),
                f"{gene}_passes": bool(
                    gene in stats.index and mask.get(gene, False)
                ),
            })

    return pd.DataFrame(rows)


def summarise_target(
    stats: pd.DataFrame,
    candidates: pd.DataFrame,
    bootstrap: pd.DataFrame,
    loo: pd.DataFrame,
    cell_line_table: pd.DataFrame,
    gene: str,
) -> str:
    if gene not in stats.index:
        return f"{gene} was not found in the integrated statistics table."

    row = stats.loc[gene]
    expression_col = f"{gene}_expression_log2TPM1"
    effect_col = f"{gene}_gene_effect"

    correlation_data = cell_line_table[
        [expression_col, effect_col]
    ].dropna()

    if len(correlation_data) >= 3:
        rho, p_corr = spearmanr(
            correlation_data[expression_col],
            correlation_data[effect_col],
        )
    else:
        rho, p_corr = np.nan, np.nan

    boot_ci_effect = np.quantile(
        bootstrap["mean_effect_melanoma"], [0.025, 0.975]
    )
    boot_ci_selectivity = np.quantile(
        bootstrap["selectivity"], [0.025, 0.975]
    )

    candidate_rank = (
        int(candidates.loc[gene, "candidate_rank"])
        if gene in candidates.index
        else None
    )

    lines = [
        f"{gene} validation summary",
        "=" * (len(gene) + 19),
        f"Candidate under primary thresholds: {gene in candidates.index}",
        f"Candidate rank: {candidate_rank if candidate_rank is not None else 'NA'}",
        f"Common essential: {bool(row.get('is_common_essential', False))}",
        "",
        "Primary DepMap statistics",
        f"  Mean melanoma Chronos effect: {row['mean_effect_melanoma']:.4f}",
        f"  Mean other-cancer effect:     {row['mean_effect_other']:.4f}",
        f"  Selectivity:                  {row['selectivity']:.4f}",
        f"  Welch-test FDR:               {row['fdr_bh']:.4g}",
        f"  Hedges g:                     {row['hedges_g']:.4f}",
        f"  Melanoma dependent fraction:  {row['frac_dependent_melanoma']:.3f}",
        f"  Other-cancer dependent frac.: {row['frac_dependent_other']:.3f}",
        f"  Mean melanoma dep. prob.:     {row['mean_dep_prob_melanoma']:.3f}",
        f"  Mean melanoma expression:     {row['mean_expr_melanoma']:.3f}",
        f"  Melanoma expression fraction: {row['frac_expressed_melanoma']:.3f}",
        "",
        f"Bootstrap ({len(bootstrap)} iterations)",
        (
            "  Mean melanoma effect 95% CI: "
            f"[{boot_ci_effect[0]:.4f}, {boot_ci_effect[1]:.4f}]"
        ),
        (
            "  Selectivity 95% CI:         "
            f"[{boot_ci_selectivity[0]:.4f}, "
            f"{boot_ci_selectivity[1]:.4f}]"
        ),
        "",
        "Leave-one-out stability",
        (
            "  Melanoma mean-effect range: "
            f"[{loo['mean_effect_melanoma'].min():.4f}, "
            f"{loo['mean_effect_melanoma'].max():.4f}]"
        ),
        (
            "  Selectivity range:          "
            f"[{loo['selectivity'].min():.4f}, "
            f"{loo['selectivity'].max():.4f}]"
        ),
        "",
        "Cell-line-level expression/dependency association",
        f"  Spearman rho: {rho:.4f}",
        f"  Spearman p:   {p_corr:.4g}",
        "",
        "Interpretation note",
        (
            "  These analyses support or weaken a melanoma-selective genetic "
            "dependency claim. They do not alone demonstrate direct "
            "pharmacological tractability or clinical efficacy."
        ),
    ]

    return "\n".join(lines)


# =============================================================================
# Optional LINCS-signature preparation
# =============================================================================

def find_column(
    columns: pd.Index,
    accepted_names: set[str],
) -> str | None:
    lower_map = {str(c).lower(): str(c) for c in columns}
    for name in accepted_names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def export_lincs_signature_from_deg(path: Path) -> None:
    """
    Prepare two-sided LINCS query lists from an independent SOX10 perturbation.

    This intentionally does not derive a pseudo-knockdown signature from
    baseline DepMap expression. A real perturbation DEG table is required.
    """
    if not path.exists():
        print(
            "Optional LINCS step skipped: "
            f"{path.name} was not found."
        )
        return

    deg = pd.read_csv(path)

    gene_col = find_column(
        deg.columns, {"gene", "genes", "symbol", "gene_symbol"}
    )
    lfc_col = find_column(
        deg.columns,
        {"log2foldchange", "log2fc", "logfc", "lfc"},
    )
    fdr_col = find_column(
        deg.columns, {"padj", "fdr", "qvalue", "adj_p_value"}
    )

    missing = [
        label
        for label, value in [
            ("gene", gene_col),
            ("log2 fold-change", lfc_col),
            ("adjusted p/FDR", fdr_col),
        ]
        if value is None
    ]
    if missing:
        raise ValueError(
            "SOX10_KD_DEG.csv is missing required column types: "
            + ", ".join(missing)
        )

    working = deg[[gene_col, lfc_col, fdr_col]].copy()
    working.columns = ["Gene", "log2FC", "FDR"]
    working["Gene"] = working["Gene"].map(clean_gene_symbol)
    working["log2FC"] = pd.to_numeric(
        working["log2FC"], errors="coerce"
    )
    working["FDR"] = pd.to_numeric(
        working["FDR"], errors="coerce"
    )
    working = working.dropna().drop_duplicates("Gene")

    significant = working.loc[working["FDR"] < 0.05].copy()

    up = (
        significant.loc[significant["log2FC"] > 0]
        .sort_values(["log2FC", "FDR"], ascending=[False, True])
        .head(LINCS_SIGNATURE_SIZE)
    )
    down = (
        significant.loc[significant["log2FC"] < 0]
        .sort_values(["log2FC", "FDR"], ascending=[True, True])
        .head(LINCS_SIGNATURE_SIZE)
    )

    up.to_csv(RES_DIR / "lincs_sox10_kd_up.csv", index=False)
    down.to_csv(RES_DIR / "lincs_sox10_kd_down.csv", index=False)

    (RES_DIR / "lincs_sox10_kd_up.txt").write_text(
        "\n".join(up["Gene"]), encoding="utf-8"
    )
    (RES_DIR / "lincs_sox10_kd_down.txt").write_text(
        "\n".join(down["Gene"]), encoding="utf-8"
    )

    print(
        "Prepared optional LINCS signature: "
        f"{len(up)} upregulated and {len(down)} downregulated genes."
    )


# =============================================================================
# Figures
# =============================================================================

def plot_top_candidates(
    candidates: pd.DataFrame,
    n: int = 25,
) -> None:
    if candidates.empty:
        return

    top = candidates.head(n).sort_values(
        "mean_effect_melanoma", ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(
        top.index,
        top["mean_effect_melanoma"],
    )

    if TARGET_GENE in top.index:
        target_position = list(top.index).index(TARGET_GENE)
        bars[target_position].set_linewidth(2.0)
        bars[target_position].set_edgecolor("black")

    ax.axvline(
        DEPENDENCY_THRESHOLD,
        linestyle="--",
        linewidth=1,
        label=f"Dependency threshold ({DEPENDENCY_THRESHOLD})",
    )
    ax.set_xlabel(
        "Mean Chronos gene-effect score in melanoma\n"
        "(more negative = stronger dependency)"
    )
    ax.set_title(
        "Top candidate melanoma-selective dependencies\n"
        "(common essentials excluded)"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "01_top_candidate_dependencies.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_dependency_selectivity(
    stats: pd.DataFrame,
    candidates: pd.DataFrame,
) -> None:
    plot_df = stats.dropna(
        subset=["mean_effect_melanoma", "selectivity"]
    ).copy()

    fig, ax = plt.subplots(figsize=(9, 7))

    background = plot_df.loc[
        ~plot_df["is_candidate"] & ~plot_df["is_common_essential"]
    ]
    ax.scatter(
        background["selectivity"],
        background["mean_effect_melanoma"],
        s=9,
        alpha=0.15,
        label="Other non-common-essential genes",
    )

    if not candidates.empty:
        ax.scatter(
            candidates["selectivity"],
            candidates["mean_effect_melanoma"],
            s=30,
            alpha=0.8,
            label=f"Candidates (n={len(candidates)})",
        )

        label_genes = list(candidates.head(10).index)
    else:
        label_genes = []

    for gene in sorted(REFERENCE_GENES):
        if gene in plot_df.index and gene not in label_genes:
            label_genes.append(gene)

    for gene in label_genes:
        if gene not in plot_df.index:
            continue
        row = plot_df.loc[gene]
        ax.annotate(
            gene,
            (row["selectivity"], row["mean_effect_melanoma"]),
            xytext=(4, 2),
            textcoords="offset points",
            fontsize=7,
        )

    ax.axhline(DEPENDENCY_THRESHOLD, linestyle="--", linewidth=1)
    ax.axvline(SELECTIVITY_THRESHOLD, linestyle="--", linewidth=1)

    ax.set_xlabel(
        "Selectivity: melanoma mean − other-cancer mean\n"
        "(more negative = more melanoma-selective)"
    )
    ax.set_ylabel(
        "Mean Chronos effect in melanoma\n"
        "(more negative = stronger dependency)"
    )
    ax.set_title("Melanoma dependency and lineage selectivity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "02_dependency_selectivity.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_target_expression_dependency(
    cell_line_table: pd.DataFrame,
    gene: str,
) -> None:
    expression_col = f"{gene}_expression_log2TPM1"
    effect_col = f"{gene}_gene_effect"

    plot_df = cell_line_table[
        ["ModelID", expression_col, effect_col]
    ].dropna()

    if len(plot_df) < 3:
        return

    rho, p_value = spearmanr(
        plot_df[expression_col],
        plot_df[effect_col],
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        plot_df[expression_col],
        plot_df[effect_col],
        s=35,
        alpha=0.8,
    )

    # Add a simple least-squares trend for visual guidance only.
    x = plot_df[expression_col].to_numpy(dtype=float)
    y = plot_df[effect_col].to_numpy(dtype=float)
    if np.unique(x).size > 1:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, linewidth=1)

    ax.axhline(DEPENDENCY_THRESHOLD, linestyle="--", linewidth=1)
    ax.set_xlabel(f"{gene} expression [log2(TPM+1)]")
    ax.set_ylabel(f"{gene} Chronos gene-effect score")
    ax.set_title(
        f"{gene} expression versus dependency in melanoma\n"
        f"Spearman ρ={rho:.2f}, p={p_value:.3g}"
    )
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / f"03_{gene.lower()}_expression_dependency.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_dependency_probability_heatmap(
    gene_dependency: pd.DataFrame,
    candidates: pd.DataFrame,
    melanoma_ids: list[str],
    n_genes: int = 25,
) -> None:
    if candidates.empty:
        return

    genes = list(candidates.head(n_genes).index)
    if TARGET_GENE in gene_dependency.columns and TARGET_GENE not in genes:
        genes = [TARGET_GENE] + genes[:-1]

    genes = [g for g in genes if g in gene_dependency.columns]
    ids = [m for m in melanoma_ids if m in gene_dependency.index]

    if not genes or not ids:
        return

    matrix = gene_dependency.loc[ids, genes].T

    if TARGET_GENE in matrix.index:
        order = matrix.loc[TARGET_GENE].sort_values(
            ascending=False, na_position="last"
        ).index
        matrix = matrix.loc[:, order]

    masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(14, 8))
    image = ax.imshow(
        masked,
        aspect="auto",
        vmin=0,
        vmax=1,
    )
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xticks([])
    ax.set_xlabel(
        f"Melanoma cell lines (n={matrix.shape[1]}), "
        f"ordered by {TARGET_GENE} dependency probability"
    )
    ax.set_ylabel("Candidate gene")
    ax.set_title("Dependency-probability heterogeneity across melanoma models")
    fig.colorbar(image, ax=ax, label="Dependency probability")
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "04_dependency_probability_heatmap.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_threshold_sensitivity(
    sensitivity: pd.DataFrame,
    gene: str,
) -> None:
    count_matrix = sensitivity.pivot(
        index="dependency_threshold",
        columns="selectivity_threshold",
        values="n_candidates",
    ).sort_index(ascending=False)

    pass_matrix = sensitivity.pivot(
        index="dependency_threshold",
        columns="selectivity_threshold",
        values=f"{gene}_passes",
    ).reindex(
        index=count_matrix.index,
        columns=count_matrix.columns,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(
        count_matrix.to_numpy(dtype=float),
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(count_matrix.columns)))
    ax.set_xticklabels(
        [f"{x:.2f}" for x in count_matrix.columns]
    )
    ax.set_yticks(np.arange(len(count_matrix.index)))
    ax.set_yticklabels(
        [f"{x:.2f}" for x in count_matrix.index]
    )

    for i in range(count_matrix.shape[0]):
        for j in range(count_matrix.shape[1]):
            count = int(count_matrix.iloc[i, j])
            passes = bool(pass_matrix.iloc[i, j])
            marker = "*" if passes else ""
            ax.text(
                j,
                i,
                f"{count}{marker}",
                ha="center",
                va="center",
                fontsize=9,
            )

    ax.set_xlabel("Selectivity threshold")
    ax.set_ylabel("Dependency threshold")
    ax.set_title(
        f"Candidate-count sensitivity\n"
        f"* indicates {gene} passes"
    )
    fig.colorbar(image, ax=ax, label="Number of candidates")
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "05_threshold_sensitivity.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("Melanoma DepMap candidate-dependency analysis — v2")
    print("=" * 72)

    model = load_model_metadata()
    melanoma_metadata = select_melanoma_models(model)
    melanoma_ids = melanoma_metadata["ModelID"].astype(str).tolist()

    gene_effect = load_matrix("CRISPRGeneEffect.csv")
    gene_dependency = load_matrix("CRISPRGeneDependency.csv")
    expression = load_matrix(
        "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    )
    common_essentials = load_common_essentials()

    print("\nComputing differential-dependency statistics ...")
    stats = compute_dependency_statistics(gene_effect, melanoma_ids)
    stats = add_dependency_probability(
        stats, gene_dependency, melanoma_ids
    )
    stats = add_expression_statistics(
        stats, expression, melanoma_ids
    )
    stats, candidates = annotate_and_rank(
        stats, common_essentials
    )

    print(f"Candidates passing primary thresholds: {len(candidates):,}")
    if TARGET_GENE in stats.index:
        print(
            f"{TARGET_GENE} passes primary thresholds: "
            f"{TARGET_GENE in candidates.index}"
        )

    stats.sort_values(
        ["is_candidate", "priority_score", "fdr_bh"],
        ascending=[False, True, True],
        na_position="last",
    ).to_csv(RES_DIR / "all_gene_statistics.csv")

    candidates.to_csv(
        RES_DIR / "candidate_dependencies_ranked.csv"
    )

    print(f"\nRunning {TARGET_GENE}-specific validation ...")
    cell_line_table = target_cell_line_table(
        gene_effect,
        gene_dependency,
        expression,
        melanoma_metadata,
        TARGET_GENE,
    )
    cell_line_table.to_csv(
        RES_DIR / f"{TARGET_GENE.lower()}_cell_line_data.csv",
        index=False,
    )

    bootstrap = bootstrap_target_metrics(
        gene_effect,
        melanoma_ids,
        TARGET_GENE,
        n_bootstrap=N_BOOTSTRAP,
    )
    bootstrap.to_csv(
        RES_DIR / f"{TARGET_GENE.lower()}_bootstrap.csv",
        index=False,
    )

    loo = leave_one_out_target(
        gene_effect,
        melanoma_ids,
        TARGET_GENE,
    )
    loo.to_csv(
        RES_DIR / f"{TARGET_GENE.lower()}_leave_one_out.csv",
        index=False,
    )

    sensitivity = threshold_sensitivity(
        stats,
        common_essentials,
        TARGET_GENE,
    )
    sensitivity.to_csv(
        RES_DIR / "threshold_sensitivity.csv",
        index=False,
    )

    summary = summarise_target(
        stats,
        candidates,
        bootstrap,
        loo,
        cell_line_table,
        TARGET_GENE,
    )
    (RES_DIR / f"{TARGET_GENE.lower()}_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )
    print("\n" + summary)

    print("\nGenerating figures ...")
    plot_top_candidates(candidates)
    plot_dependency_selectivity(stats, candidates)
    plot_target_expression_dependency(
        cell_line_table,
        TARGET_GENE,
    )
    plot_dependency_probability_heatmap(
        gene_dependency,
        candidates,
        melanoma_ids,
    )
    plot_threshold_sensitivity(
        sensitivity,
        TARGET_GENE,
    )

    export_lincs_signature_from_deg(SOX10_DEG_FILE)

    print("\n" + "=" * 72)
    print(f"Done. Results written to: {OUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
