"""
Merge Cleaned Melanoma Cohorts.

Loads configuration-driven cleaned datasets, harmonises clinical metadata,
intersects common expression genes, independently Z-score standardises
expression within each cohort, and produces:

1. Full merged cohort:
   All configured datasets.

2. Immunotherapy-only merged cohort:
   Immunotherapy-treated samples from configured immunotherapy cohorts,
   plus the TCGA immunotherapy-treated subset.

The merge stage consumes only the outputs of clean_data.py:

    clin_cleaned.csv
    expr_cleaned.csv
    mutations_cleaned.csv

Dataset-specific paths and processing strategies are defined in:

    config/datasets.yaml

Domain-level constants are defined in:

    src/config/constants.py
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# ============================================================================
# Bootstrap project root resolution
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SUBPROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_ROOT))


# ============================================================================
# Project imports
# ============================================================================

from src.config.constants import (
    DRIVER_GENES,
    GENOMIC_FEATURES,
)
from src.config.datasets import (
    DatasetConfig,
    load_dataset_config,
)
from src.utils.logging import (
    TeeStream,
    display_path,
)
from src.utils.paths import (
    PROCESSED_DIR,
    get_subproject_log_dir,
)


# ============================================================================
# Configuration
# ============================================================================

# Resolve config relative to script location.
CONFIG_PATH = SUBPROJECT_ROOT / "config" / "datasets.yaml"
LOG_DIR = get_subproject_log_dir(SCRIPT_DIR)
LOG_PATH = LOG_DIR / "merge_datasets.log"

MERGED_DIR = PROCESSED_DIR / "merged"
FULL_DIR = MERGED_DIR / "full"
IMMUNOTHERAPY_DIR = MERGED_DIR / "immunotherapy"

# Sentinel value for missing strings.
_VAL_NA: str = "N/A"

# Protocol column names.
_COL_PATIENT_ID: str = "PATIENT_ID"
_COL_SAMPLE_ID: str = "SAMPLE_ID"
_COL_COHORT: str = "COHORT"
_COL_IMMUNOTHERAPY: str = "IMMUNOTHERAPY"
_COL_SPECIMEN_TYPE: str = "SPECIMEN_TYPE"
_COL_OS_MONTHS: str = "OS_MONTHS"
_COL_OS_STATUS: str = "OS_STATUS"
_COL_RESPONSE: str = "RESPONSE"
_COL_RESPONSE_BINARY: str = "RESPONSE_BINARY"
_COL_RACE: str = "RACE"
_COL_SEX: str = "SEX"
_COL_AGE: str = "AGE"

# Filenames.
_EXPR_CLEANED_FILENAME: str = "expr_cleaned.csv"
_CLIN_CLEANED_FILENAME: str = "clin_cleaned.csv"
_MUTATIONS_CLEANED_FILENAME: str = "mutations_cleaned.csv"
_EXPR_MERGED_FILENAME: str = "expr_merged.csv"
_CLIN_MERGED_FILENAME: str = "clin_merged.csv"
_GENOMIC_MERGED_FILENAME: str = "merged_genomic.csv"

# Quantitative genomic burden features in clinical metadata.
_GENOMIC_BURDEN_COLS: list[str] = [
    "TMB_NONSYNONYMOUS",
    "SNV_NEOANTIGEN",
    "INDEL_NEOANTIGEN",
    "FUSION_NEOANTIGEN",
    "SPLICE_NEOANTIGEN",
    "CTA_SELF_NEOANTIGEN",
    "VIRUS_NEOANTIGEN",
    "ERV_NEOANTIGEN",
    "TOTAL_NEOANTIGEN",
    "CNA_PROP",
    "ANEUPLOIDY_SCORE",
    "MSI_SCORE_MANTIS",
    "MSI_SENSOR_SCORE",
]

# Processing strategy identifiers.
_STRATEGY_IATLAS: str = "iatlas"
_STRATEGY_TCGA: str = "tcga"


# ============================================================================
# Type aliases
# ============================================================================

DatasetFrames = Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]


# ============================================================================
# Dataset loading
# ============================================================================


def _get_dataset_directory(dataset: DatasetConfig) -> Path:
    """Return the processed directory for a configured dataset."""
    return PROCESSED_DIR / dataset.processed_directory


def _load_cleaned_expression(dataset: DatasetConfig) -> pd.DataFrame:
    """Load a cleaned expression matrix indexed by SAMPLE_ID."""
    expression_path = _get_dataset_directory(dataset) / _EXPR_CLEANED_FILENAME
    if not expression_path.exists():
        raise FileNotFoundError(
            f"Cleaned expression file not found: {display_path(expression_path)}"
        )

    df_expr = pd.read_csv(expression_path, index_col=0)
    df_expr.index = df_expr.index.astype(str).str.strip().str.upper()
    df_expr.index.name = _COL_SAMPLE_ID
    return df_expr


def _load_cleaned_clinical(dataset: DatasetConfig) -> pd.DataFrame:
    """Load a cleaned clinical dataset indexed by SAMPLE_ID."""
    clinical_path = _get_dataset_directory(dataset) / _CLIN_CLEANED_FILENAME
    if not clinical_path.exists():
        raise FileNotFoundError(f"Cleaned clinical file not found: {display_path(clinical_path)}")

    df_clin = pd.read_csv(clinical_path)
    if _COL_SAMPLE_ID not in df_clin.columns:
        raise ValueError(
            f"Cleaned clinical file does not contain"
            f" {_COL_SAMPLE_ID}: {display_path(clinical_path)}"
        )

    df_clin[_COL_SAMPLE_ID] = df_clin[_COL_SAMPLE_ID].astype(str).str.strip().str.upper()
    df_clin = df_clin.set_index(_COL_SAMPLE_ID)
    df_clin.index.name = _COL_SAMPLE_ID
    return df_clin


def _load_cleaned_mutations(dataset: DatasetConfig) -> pd.DataFrame:
    """Load a cleaned binary mutation matrix indexed by SAMPLE_ID."""
    mutation_path = _get_dataset_directory(dataset) / _MUTATIONS_CLEANED_FILENAME
    if not mutation_path.exists():
        print(
            f"  No cleaned mutation file found for {dataset.cohort_name}."
            " Using an empty mutation matrix."
        )
        return pd.DataFrame()

    df_mut = pd.read_csv(mutation_path, index_col=0)
    df_mut.index = df_mut.index.astype(str).str.strip().str.upper()
    df_mut.index.name = _COL_SAMPLE_ID
    return df_mut


def load_dataset(dataset: DatasetConfig) -> DatasetFrames:
    """Load all cleaned outputs for one configured dataset."""
    print(f"Loading {dataset.cohort_name} ({dataset.study_id})...")
    df_expr = _load_cleaned_expression(dataset)
    df_clin = _load_cleaned_clinical(dataset)
    df_mut = _load_cleaned_mutations(dataset)

    print(f"  Expression: {df_expr.shape[0]:,} samples \u00d7 {df_expr.shape[1]:,} genes")
    print(f"  Clinical:   {df_clin.shape[0]:,} samples \u00d7 {df_clin.shape[1]:,} features")
    print(f"  Mutations:  {df_mut.shape[0]:,} samples \u00d7 {df_mut.shape[1]:,} genes")

    return df_expr, df_clin, df_mut


# ============================================================================
# Clinical harmonisation
# ============================================================================


def _normalise_os_status(
    series: pd.Series,
) -> pd.Series:
    """
    Convert common survival-status representations to binary values.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors="coerce",
        )

    status_map = {
        "1:DECEASED": 1.0,
        "0:LIVING": 0.0,
        "DECEASED": 1.0,
        "LIVING": 0.0,
        "DEAD": 1.0,
        "ALIVE": 0.0,
        "1": 1.0,
        "0": 0.0,
    }

    return (
        series
        .astype(str)
        .str.strip()
        .str.upper()
        .map(status_map)
    )


def _normalise_sex(
    series: pd.Series,
) -> pd.Series:
    """
    Harmonise sex values.
    """
    sex_map = {
        "MALE": "Male",
        "M": "Male",
        "FEMALE": "Female",
        "F": "Female",
    }

    return (
        series
        .astype(str)
        .str.strip()
        .str.upper()
        .map(sex_map)
        .fillna(_VAL_NA)
    )


def _normalise_specimen_type(series: pd.Series) -> pd.Series:
    """Harmonise specimen-type descriptions to Primary, Metastatic, or N/A."""
    def classify(value: object) -> str:
        """Classify a single specimen type string into Primary, Metastatic, or N/A."""
        if pd.isna(value):
            return _VAL_NA

        val_str = str(value).strip().lower()
        if not val_str:
            return _VAL_NA

        if "primary" in val_str:
            return "Primary"

        if any(term in val_str for term in ("metastatic", "metastasis", "metastase")):
            return "Metastatic"

        return _VAL_NA

    return series.map(classify)


def _copy_clinical_survival(df_clin: pd.DataFrame, df: pd.DataFrame) -> None:
    """Copy and normalise overall survival metrics."""
    if _COL_OS_MONTHS in df_clin.columns:
        df[_COL_OS_MONTHS] = pd.to_numeric(df_clin[_COL_OS_MONTHS], errors="coerce")
    else:
        df[_COL_OS_MONTHS] = np.nan

    if _COL_OS_STATUS in df_clin.columns:
        df[_COL_OS_STATUS] = _normalise_os_status(df_clin[_COL_OS_STATUS])
    else:
        df[_COL_OS_STATUS] = np.nan


def _copy_clinical_demographics(df_clin: pd.DataFrame, df: pd.DataFrame) -> None:
    """Copy and normalise demographic variables (age, race, sex)."""
    df[_COL_AGE] = (
        pd.to_numeric(df_clin[_COL_AGE], errors="coerce")
        if _COL_AGE in df_clin.columns else np.nan
    )
    df[_COL_RACE] = df_clin[_COL_RACE] if _COL_RACE in df_clin.columns else np.nan
    df[_COL_SEX] = _normalise_sex(df_clin[_COL_SEX]) if _COL_SEX in df_clin.columns else _VAL_NA


def _copy_clinical_specimen_and_tx(
    df_clin: pd.DataFrame,
    df: pd.DataFrame,
    dataset: DatasetConfig,
) -> None:
    """Copy and normalise specimen type, response, and immunotherapy flags."""
    df[_COL_RESPONSE] = df_clin[_COL_RESPONSE] if _COL_RESPONSE in df_clin.columns else np.nan
    df[_COL_RESPONSE_BINARY] = (
        pd.to_numeric(df_clin[_COL_RESPONSE_BINARY], errors="coerce")
        if _COL_RESPONSE_BINARY in df_clin.columns else np.nan
    )

    spec_cols = (_COL_SPECIMEN_TYPE, "SAMPLE_TYPE", "BIOPSY_SITE")
    spec_col = next((c for c in spec_cols if c in df_clin.columns), None)
    df[_COL_SPECIMEN_TYPE] = _normalise_specimen_type(df_clin[spec_col]) if spec_col else _VAL_NA

    tx_cols = (_COL_IMMUNOTHERAPY, "TX_TYPE_IMMUNOTHERAPY")
    tx_col = next((c for c in tx_cols if c in df_clin.columns), None)
    if tx_col:
        df[_COL_IMMUNOTHERAPY] = (
            pd.to_numeric(df_clin[tx_col], errors="coerce")
            .fillna(0).astype(int)
        )
    else:
        df[_COL_IMMUNOTHERAPY] = 1 if dataset.processing_strategy == _STRATEGY_IATLAS else 0


def _copy_clinical_genomic_burdens(df_clin: pd.DataFrame, df: pd.DataFrame) -> None:
    """Copy quantitative genomic burden features from clinical metadata."""
    for col in _GENOMIC_BURDEN_COLS:
        if col in df_clin.columns:
            df[col] = pd.to_numeric(df_clin[col], errors="coerce")


def _harmonise_clinical(
    df_clin: pd.DataFrame,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """Convert cohort-specific clinical data to a common schema."""
    df = pd.DataFrame(index=df_clin.index)
    df[_COL_PATIENT_ID] = df_clin[_COL_PATIENT_ID] if _COL_PATIENT_ID in df_clin.columns else np.nan
    df[_COL_COHORT] = dataset.cohort_name

    _copy_clinical_survival(df_clin, df)
    _copy_clinical_demographics(df_clin, df)
    _copy_clinical_specimen_and_tx(df_clin, df, dataset)
    _copy_clinical_genomic_burdens(df_clin, df)

    df.index.name = _COL_SAMPLE_ID
    return df


# ============================================================================
# Expression processing
# ============================================================================


def zscore_expression(df_expr: pd.DataFrame) -> pd.DataFrame:
    """Z-score standardise each gene independently within one cohort.

    Constant or missing genes receive std=1 to avoid division by zero.
    """
    df = df_expr.copy()
    means = df.mean(axis=0, skipna=True)
    stds = df.std(axis=0, skipna=True).replace(0, 1.0).fillna(1.0)
    return (df - means) / stds


def find_common_genes(
    expression_data: dict[str, pd.DataFrame],
) -> list[str]:
    """
    Find the intersection of genes present in all datasets.
    """
    gene_sets = [
        set(df.columns)
        for df in expression_data.values()
    ]

    if not gene_sets:
        return []

    common_genes = set.intersection(
        *gene_sets
    )

    return sorted(
        common_genes
    )


# ============================================================================
# Genomic feature processing
# ============================================================================


def _mutation_feature_columns() -> list[str]:
    """
    Return mutation feature names derived from DRIVER_GENES.
    """
    return [
        f"mut_{gene}"
        for gene in DRIVER_GENES
    ]


def _load_driver_mutation_features(
    df_mut: pd.DataFrame,
) -> pd.DataFrame:
    """Extract configured driver mutation features (mut_BRAF, mut_NRAS, mut_NF1)."""
    result = pd.DataFrame(index=df_mut.index)
    for gene in DRIVER_GENES:
        col = f"mut_{gene}"
        if gene in df_mut.columns:
            result[col] = pd.to_numeric(df_mut[gene], errors="coerce").fillna(0).astype(int)
        else:
            result[col] = 0
    return result


def _build_sample_genomic_row(
    sample_id: str,
    clinical_row: pd.Series,
    cohort_name: str,
    df_mut: pd.DataFrame,
    df_driver_mut: pd.DataFrame,
    mutation_features: list[str],
) -> dict[str, object]:
    """Assemble a single genomic feature row for one sample."""
    row: dict[str, object] = {
        _COL_PATIENT_ID: clinical_row.get(_COL_PATIENT_ID, np.nan),
        _COL_SAMPLE_ID: sample_id,
        _COL_COHORT: cohort_name,
    }
    for feature in GENOMIC_FEATURES:
        if feature in clinical_row.index:
            row[feature] = clinical_row[feature]
        elif feature in df_mut.columns:
            row[feature] = df_mut.loc[sample_id, feature]
        else:
            row[feature] = np.nan

    for col in mutation_features:
        row[col] = df_driver_mut.loc[sample_id, col] if sample_id in df_driver_mut.index else 0
    return row


def _collect_cohort_genomic_rows(
    dataset: DatasetConfig,
    clinical_data: dict[str, pd.DataFrame],
    mutation_data: dict[str, pd.DataFrame],
    mutation_features: list[str],
) -> list[dict[str, object]]:
    """Assemble genomic feature rows for all samples in one cohort."""
    cohort_name = dataset.cohort_name
    df_clin = clinical_data[cohort_name]
    df_mut = mutation_data.get(cohort_name, pd.DataFrame())
    df_driver_mut = (
        _load_driver_mutation_features(df_mut) if not df_mut.empty
        else pd.DataFrame(index=df_clin.index, columns=mutation_features).fillna(0)
    )
    return [
        _build_sample_genomic_row(
            str(sid), row, cohort_name, df_mut, df_driver_mut, mutation_features
        )
        for sid, row in df_clin.iterrows()
    ]


def _build_genomic_features(
    datasets: list[DatasetConfig],
    clinical_data: dict[str, pd.DataFrame],
    mutation_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a unified genomic feature matrix for all merged samples."""
    mutation_features = _mutation_feature_columns()
    all_cols = [_COL_PATIENT_ID, _COL_SAMPLE_ID, _COL_COHORT] + GENOMIC_FEATURES + mutation_features
    genomic_rows: list[dict[str, object]] = []

    for dataset in datasets:
        genomic_rows.extend(
            _collect_cohort_genomic_rows(dataset, clinical_data, mutation_data, mutation_features)
        )

    if not genomic_rows:
        return pd.DataFrame(columns=all_cols)

    df_genomic = pd.DataFrame(genomic_rows)
    for col in all_cols:
        if col not in df_genomic.columns:
            df_genomic[col] = np.nan
    return df_genomic[all_cols]


# ============================================================================
# Merge output
# ============================================================================


def _save_merged_outputs(
    df_expr: pd.DataFrame,
    df_clin: pd.DataFrame,
    df_genomic: pd.DataFrame,
    output_dir: Path,
    label: str,
) -> None:
    """Save merged expression, clinical, and genomic outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    expr_path = output_dir / _EXPR_MERGED_FILENAME
    clinical_path = output_dir / _CLIN_MERGED_FILENAME
    genomic_path = output_dir / _GENOMIC_MERGED_FILENAME

    df_expr.to_csv(expr_path)
    df_clin.reset_index().to_csv(clinical_path, index=False)
    df_genomic.to_csv(genomic_path, index=False)

    print(f"\n[{label}] Saved merged outputs:")
    print(f"  Expression: {display_path(expr_path)} {df_expr.shape}")
    print(f"  Clinical:   {display_path(clinical_path)} {df_clin.shape}")
    print(f"  Genomic:    {display_path(genomic_path)} {df_genomic.shape}")


# ============================================================================
# Merge construction
# ============================================================================


def _align_single_cohort_data(
    dataset: DatasetConfig,
) -> DatasetFrames:
    """Load and align expression, clinical, and mutation data for one cohort."""
    df_expr, df_clin_raw, df_mut = load_dataset(dataset)
    common_ids = df_expr.index.intersection(df_clin_raw.index)
    if len(common_ids) == 0:
        raise ValueError(f"No overlapping samples for {dataset.cohort_name}.")

    df_expr = df_expr.loc[common_ids]
    df_clin_raw = df_clin_raw.loc[common_ids]

    if not df_mut.empty:
        df_mut = df_mut.loc[df_mut.index.intersection(common_ids)]

    df_clin = _harmonise_clinical(df_clin_raw, dataset).loc[df_expr.index]
    print(f"  Aligned samples: {len(common_ids):,}")
    return df_expr, df_clin, df_mut


def _prepare_dataset_data(
    datasets: list[DatasetConfig],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load and harmonise all configured datasets."""
    expression_data, clinical_data, mutation_data = {}, {}, {}
    for dataset in datasets:
        expr, clin, mut = _align_single_cohort_data(dataset)
        expression_data[dataset.cohort_name] = expr
        clinical_data[dataset.cohort_name] = clin
        mutation_data[dataset.cohort_name] = mut
    return expression_data, clinical_data, mutation_data


def _filter_cohort_samples(
    datasets: list[DatasetConfig],
    expression_data: dict[str, pd.DataFrame],
    clinical_data: dict[str, pd.DataFrame],
    mutation_data: dict[str, pd.DataFrame],
    immunotherapy_only: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Select cohort subsets based on immunotherapy filtering."""
    sel_expr, sel_clin, sel_mut = {}, {}, {}
    for dataset in datasets:
        name = dataset.cohort_name
        df_expr, df_clin, df_mut = expression_data[name], clinical_data[name], mutation_data[name]

        if immunotherapy_only:
            mask = df_clin[_COL_IMMUNOTHERAPY] == 1
            df_expr = df_expr.loc[mask]
            df_clin = df_clin.loc[mask]
            if not df_mut.empty:
                df_mut = df_mut.loc[df_mut.index.intersection(df_expr.index)]

        sel_expr[name], sel_clin[name], sel_mut[name] = df_expr, df_clin, df_mut
        print(f"  {name}: {len(df_clin):,} samples")
    return sel_expr, sel_clin, sel_mut


def _scale_and_concat_expression(
    datasets: list[DatasetConfig],
    selected_expression: dict[str, pd.DataFrame],
    selected_clinical: dict[str, pd.DataFrame],
    common_genes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Z-score standardise each cohort on common genes and concatenate."""
    scaled_expr, clinical_frames = [], []
    for dataset in datasets:
        name = dataset.cohort_name
        df_expr, df_clin = selected_expression[name], selected_clinical[name]
        if df_expr.empty:
            continue
        df_scaled = zscore_expression(df_expr[common_genes]).loc[df_clin.index]
        scaled_expr.append(df_scaled)
        clinical_frames.append(df_clin)

    df_expr_merged = pd.concat(scaled_expr, axis=0)
    df_clin_merged = pd.concat(clinical_frames, axis=0)
    if not (df_expr_merged.index == df_clin_merged.index).all():
        raise ValueError("Expression and clinical sample indices are not identical after merge.")
    return df_expr_merged, df_clin_merged


def build_merged_cohort(
    datasets: list[DatasetConfig],
    expression_data: dict[str, pd.DataFrame],
    clinical_data: dict[str, pd.DataFrame],
    mutation_data: dict[str, pd.DataFrame],
    output_dir: Path,
    label: str,
    immunotherapy_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build, standardise, and save a merged cohort."""
    print(f"\n{'=' * 60}\nBuilding {label.upper()} merged cohort\n{'=' * 60}")
    sel_expr, sel_clin, sel_mut = _filter_cohort_samples(
        datasets, expression_data, clinical_data, mutation_data, immunotherapy_only
    )

    print("\nFinding common expression genes...")
    common_genes = find_common_genes(sel_expr)
    if not common_genes:
        raise ValueError("No common expression genes found across selected datasets.")
    print(f"  Common genes: {len(common_genes):,}")

    print("\nZ-score standardising expression within each cohort...")
    df_expr_merged, df_clin_merged = _scale_and_concat_expression(
        datasets, sel_expr, sel_clin, common_genes
    )

    df_genomic = _build_genomic_features(datasets, sel_clin, sel_mut)
    df_genomic = df_genomic[df_genomic[_COL_SAMPLE_ID].isin(df_clin_merged.index)]
    _save_merged_outputs(df_expr_merged, df_clin_merged, df_genomic, output_dir, label)
    return df_expr_merged, df_clin_merged


# ============================================================================
# Main workflow
# ============================================================================


def _print_merge_header(datasets: list[DatasetConfig]) -> None:
    """Print initial workflow header and dataset configuration list."""
    print("==================================================")
    print("Merging Cleaned Melanoma Cohorts")
    print("==================================================\n")
    print(f"Loaded {len(datasets)} dataset configuration(s) from {display_path(CONFIG_PATH)}:\n")
    for dataset in datasets:
        print(f"  - {dataset.cohort_name} ({dataset.study_id})")
    print()


def _print_merge_summary(df_clin_full: pd.DataFrame, df_clin_immuno: pd.DataFrame) -> None:
    """Print completion banner and final dataset sample counts."""
    print(f"\n{'=' * 60}\nMerged Cohort Generation Completed Successfully\n{'=' * 60}")
    print(f"  Full merge: {len(df_clin_full):,} samples")
    print(f"  Immunotherapy merge: {len(df_clin_immuno):,} samples")
    print(f"\n  Full output: {display_path(FULL_DIR)}")
    print(f"  Immunotherapy output: {display_path(IMMUNOTHERAPY_DIR)}")


def main() -> None:
    """Run the configuration-driven dataset merge."""
    all_datasets = load_dataset_config(CONFIG_PATH)
    datasets = [d for d in all_datasets if d.merge_enabled]
    _print_merge_header(datasets)

    disabled = [d for d in all_datasets if not d.merge_enabled]
    if disabled:
        print("Datasets excluded from merge (merge_enabled: false):")
        for d in disabled:
            print(f"  - {d.cohort_name} ({d.study_id})")
        print()

    expression_data, clinical_data, mutation_data = _prepare_dataset_data(datasets)

    df_expr_full, df_clin_full = build_merged_cohort(
        datasets=datasets, expression_data=expression_data, clinical_data=clinical_data,
        mutation_data=mutation_data, output_dir=FULL_DIR, label="Full", immunotherapy_only=False,
    )

    df_expr_immuno, df_clin_immuno = build_merged_cohort(
        datasets=datasets,
        expression_data=expression_data,
        clinical_data=clinical_data,
        mutation_data=mutation_data,
        output_dir=IMMUNOTHERAPY_DIR,
        label="Immunotherapy",
        immunotherapy_only=True,
    )

    _print_merge_summary(df_clin_full, df_clin_immuno)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {display_path(LOG_PATH)}")
            main()