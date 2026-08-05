"""
Data Cleaning & Preprocessing Pipeline for Immunotherapy Cohorts & TCGA-SKCM.

Cleans raw clinical, gene expression, and somatic mutation datasets across
the configured melanoma cohorts.

The workflow is configuration-driven. Dataset-specific behaviour is determined
by ``processing_strategy`` in ``config/datasets.yaml``.

Cleaning includes:
- standardising patient and sample identifiers;
- merging patient- and sample-level clinical data;
- harmonising clinical response variables;
- cleaning and aligning expression matrices;
- converting expression values to log2-transformed values;
- mapping TCGA Entrez identifiers to gene symbols;
- parsing non-silent somatic mutations;
- optionally aggregating mutations to patient level;
- recording sample attrition during preprocessing;
- exporting cleaned datasets to ``data/processed/``.

This module deliberately does not perform downstream feature selection.
Variance filtering, dimensionality reduction, and machine-learning feature
selection belong to downstream analysis stages.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SUBPROJECT_ROOT = SCRIPT_DIR.parent

if str(SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_ROOT))


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from src.config.constants import (
    RECIST_RESPONSE_MAP,
    NON_SILENT_VARIANT_CLASSIFICATIONS,
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
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    get_subproject_log_dir,
    rel_path,
)
from src.utils.preprocessing import (
    align_expression_and_clinical,
    clean_clinical_df,
    map_entrez_to_symbols,
    parse_maf_mutations,
    read_cbioportal_table,
    standardise_sample_id,
    drop_empty_columns,
)


# ---------------------------------------------------------------------------
# Module Constants & Configuration
# ---------------------------------------------------------------------------

CONFIG_DIR = SUBPROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "datasets.yaml"
LOG_DIR = get_subproject_log_dir(SCRIPT_DIR)
LOG_PATH = LOG_DIR / "clean_data.log"

CLEANED_CLINICAL_FILENAME = "clin_cleaned.csv"
CLEANED_EXPRESSION_FILENAME = "expr_cleaned.csv"
CLEANED_MUTATION_FILENAME = "mutations_cleaned.csv"
ATTRITION_FILENAME = "attrition.csv"
ENTREZ_CACHE_FILENAME = "entrez_to_symbol_cache.json"

EXPRESSION_MIN_GENES_WARN = 1_000
EXPRESSION_MIN_GENES_INFO = 5_000
EXPRESSION_MAX_VALUE_THRESHOLD = 50.0
EXAMPLE_GENES_DISPLAY_LIMIT = 20

TCGA_KEY_TREATMENT_AGENTS: dict[str, list[str]] = {
    "TX_AGENT_IPILIMUMAB": ["Ipilimumab"],
    "TX_AGENT_PEMBROLIZUMAB": ["Pembrolizumab"],
    "TX_AGENT_NIVOLUMAB": ["Nivolumab"],
    "TX_AGENT_VEMURAFENIB": ["Vemurafenib"],
    "TX_AGENT_DABRAFENIB": ["Dabrafenib"],
    "TX_AGENT_TRAMETINIB": ["Trametinib"],
    "TX_AGENT_DACARBAZINE": ["Dacarbazine"],
    "TX_AGENT_TEMOZOLOMIDE": ["Temozolomide"],
    "TX_AGENT_INTERFERON": [
        "Interferon Alfa",
        "Interferon Nos",
        "Interferon",
    ],
}


# ---------------------------------------------------------------------------
# Attrition Data Structures & Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttritionRecord:
    """Record sample attrition at a single cleaning step."""

    cohort: str
    step: str
    n_before: int
    n_after: int
    n_removed: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Convert the record to a serialisable dictionary."""
        return {
            "cohort": self.cohort,
            "step": self.step,
            "n_before": self.n_before,
            "n_after": self.n_after,
            "n_removed": self.n_removed,
            "reason": self.reason,
        }


def _record_attrition(
    records: list[AttritionRecord],
    dataset: DatasetConfig,
    step: str,
    n_before: int,
    n_after: int,
    reason: str,
) -> None:
    """Append a sample attrition record."""
    records.append(
        AttritionRecord(
            cohort=dataset.cohort_name,
            step=step,
            n_before=n_before,
            n_after=n_after,
            n_removed=n_before - n_after,
            reason=reason,
        )
    )


# ---------------------------------------------------------------------------
# Clinical Data Processing Helpers
# ---------------------------------------------------------------------------


def _validate_clinical_filepaths(
    raw_dir: Path, dataset: DatasetConfig
) -> tuple[Path, Path]:
    """Validate existence of patient and sample clinical files."""
    patient_path = raw_dir / dataset.clinical_file
    sample_path = raw_dir / dataset.clinical_sample_file

    missing_files = [p for p in (patient_path, sample_path) if not p.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required clinical file(s): "
            + ", ".join(display_path(p) for p in missing_files)
        )
    return patient_path, sample_path


def _validate_clinical_columns(
    df_patient: pd.DataFrame,
    df_sample: pd.DataFrame,
    patient_path: Path,
    sample_path: Path,
) -> None:
    """Validate required identifier columns in clinical DataFrames."""
    if "PATIENT_ID" not in df_patient.columns:
        raise ValueError(f"{display_path(patient_path)} does not contain PATIENT_ID.")

    required_sample_columns = {"SAMPLE_ID", "PATIENT_ID"}
    missing_sample_columns = required_sample_columns - set(df_sample.columns)
    if missing_sample_columns:
        raise ValueError(
            f"{display_path(sample_path)} is missing required columns: "
            f"{sorted(missing_sample_columns)}"
        )


def _load_clinical_data(
    raw_dir: Path,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """Load and merge patient- and sample-level clinical data."""
    patient_path, sample_path = _validate_clinical_filepaths(raw_dir, dataset)

    df_patient = read_cbioportal_table(patient_path, required_column="PATIENT_ID")
    df_sample = read_cbioportal_table(sample_path, required_column="SAMPLE_ID")

    _validate_clinical_columns(df_patient, df_sample, patient_path, sample_path)

    return pd.merge(df_sample, df_patient, on="PATIENT_ID", how="inner")


def _apply_identifier_prefixes(
    df_clin: pd.DataFrame,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """Apply configured prefixes to patient and sample identifiers."""
    df = df_clin.copy()
    if "PATIENT_ID" in df.columns:
        df["PATIENT_ID"] = (
            dataset.patient_prefix + df["PATIENT_ID"].astype(str)
        ).str.upper()

    if "SAMPLE_ID" in df.columns:
        df["SAMPLE_ID"] = (
            dataset.sample_prefix + df["SAMPLE_ID"].astype(str)
        ).str.upper()

    return df


def _standardise_sex_column(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonise sex categories into standardized string representations."""
    if "SEX" not in df.columns:
        return df

    sex_map = {"MALE": "Male", "FEMALE": "Female", "M": "Male", "F": "Female"}
    df["SEX"] = (
        df["SEX"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(sex_map)
        .fillna("N/A")
    )
    return df


def _derive_binary_response(df: pd.DataFrame) -> pd.DataFrame:
    """Derive binary response label from RECIST response column."""
    if "RESPONSE" in df.columns:
        df["RESPONSE_BINARY"] = df["RESPONSE"].map(RECIST_RESPONSE_MAP)
    else:
        df["RESPONSE_BINARY"] = np.nan
    return df


def _harmonise_clinical_data(
    df_clin: pd.DataFrame,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """Clean and harmonise clinical data for a configured dataset."""
    df = _apply_identifier_prefixes(df_clin, dataset)
    df = clean_clinical_df(df)
    df = drop_empty_columns(df)

    if dataset.baseline_only:
        df = df[df["SAMPLE_ID"].str.endswith("_PRE")]

    df = _derive_binary_response(df)

    if "AGE_AT_DIAGNOSIS" in df.columns:
        df["AGE"] = pd.to_numeric(df["AGE_AT_DIAGNOSIS"], errors="coerce")

    df = _standardise_sex_column(df)
    return df.set_index("SAMPLE_ID")


# ---------------------------------------------------------------------------
# Expression Processing Helpers
# ---------------------------------------------------------------------------


def _transform_iatlas_expression_matrix(
    df_expr: pd.DataFrame, dataset: DatasetConfig
) -> pd.DataFrame:
    """Filter baseline samples, transpose, convert to numeric, and log2-transform iAtlas expression."""
    if dataset.baseline_only:
        df_expr = df_expr[[col for col in df_expr.columns if col.endswith("_PRE")]]

    df_expr = df_expr.T
    df_expr.index.name = "SAMPLE_ID"

    df_numeric = df_expr.apply(pd.to_numeric, errors="coerce")
    return np.log2(df_numeric.clip(lower=0) + 1)


def _load_iatlas_expression(
    expression_path: Path,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """Load and prepare an iAtlas/cBioPortal expression matrix."""
    if not expression_path.exists():
        raise FileNotFoundError(
            f"Expression file not found: {display_path(expression_path)}"
        )

    df_expr = pd.read_csv(expression_path, sep="\t")
    if "Hugo_Symbol" not in df_expr.columns:
        raise ValueError(
            f"Expression file {display_path(expression_path)} does not contain Hugo_Symbol."
        )

    df_expr = df_expr.dropna(subset=["Hugo_Symbol"]).set_index("Hugo_Symbol")
    if "Entrez_Gene_Id" in df_expr.columns:
        df_expr = df_expr.drop(columns=["Entrez_Gene_Id"])

    df_expr = df_expr.groupby(level=0).mean(numeric_only=True)
    df_expr.columns = (dataset.sample_prefix + df_expr.columns.astype(str)).str.upper()

    return _transform_iatlas_expression_matrix(df_expr, dataset)


def _map_tcga_entrez_genes(
    df_expr: pd.DataFrame, cache_path: Path
) -> pd.DataFrame:
    """Map TCGA Entrez gene IDs to Hugo gene symbols and average duplicates."""
    df_expr = df_expr.dropna(subset=["Entrez_Gene_Id"])
    df_expr["Entrez_Gene_Id"] = (
        pd.to_numeric(df_expr["Entrez_Gene_Id"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )

    entrez_ids = df_expr["Entrez_Gene_Id"].dropna().unique().tolist()
    gene_map = map_entrez_to_symbols(entrez_ids, cache_path=cache_path)

    df_expr["Hugo_Symbol_Mapped"] = df_expr["Entrez_Gene_Id"].map(gene_map).fillna(
        df_expr["Entrez_Gene_Id"]
    )
    df_expr = df_expr.set_index("Hugo_Symbol_Mapped")

    columns_to_drop = [c for c in ("Hugo_Symbol", "Entrez_Gene_Id") if c in df_expr.columns]
    df_expr = df_expr.drop(columns=columns_to_drop)

    return df_expr.groupby(level=0).mean(numeric_only=True)


def _load_tcga_expression(
    expression_path: Path,
    dataset: DatasetConfig,
    cache_path: Path,
) -> pd.DataFrame:
    """Load and prepare the TCGA RSEM expression matrix."""
    if not expression_path.exists():
        raise FileNotFoundError(
            f"Expression file not found: {display_path(expression_path)}"
        )

    df_expr = pd.read_csv(expression_path, sep="\t")
    if "Entrez_Gene_Id" not in df_expr.columns:
        raise ValueError("TCGA expression matrix does not contain Entrez_Gene_Id.")

    df_expr = _map_tcga_entrez_genes(df_expr, cache_path)
    df_expr = df_expr.T

    df_expr.index = (
        dataset.sample_prefix + df_expr.index.astype(str)
    ).map(standardise_sample_id)
    df_expr.index.name = "SAMPLE_ID"

    df_numeric = df_expr.apply(pd.to_numeric, errors="coerce")
    return np.log2(df_numeric.clip(lower=0) + 1)


# ---------------------------------------------------------------------------
# Mutation Processing Helpers
# ---------------------------------------------------------------------------


def _sample_to_patient_id(sample_id: str) -> str:
    """Convert a sample identifier to its patient identifier."""
    if not isinstance(sample_id, str):
        return sample_id

    sample_id = sample_id.strip()
    if "_" in sample_id:
        return sample_id.rsplit("_", 1)[0]

    parts = sample_id.split("-")
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])

    return sample_id


def _aggregate_mutations_by_patient(
    df_mut: pd.DataFrame, clinical_df: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate sample-level binary mutation profiles to patient-level."""
    df_mut.index = df_mut.index.map(_sample_to_patient_id)
    df_mut = df_mut.groupby(level=0).max()

    patient_to_sample = (
        clinical_df[["PATIENT_ID"]]
        .reset_index()
        .drop_duplicates(subset=["PATIENT_ID"])
        .set_index("PATIENT_ID")
    )

    df_mut = df_mut.reindex(patient_to_sample.index, fill_value=0)
    df_mut = df_mut.join(patient_to_sample, how="right")
    df_mut = df_mut.set_index(patient_to_sample.loc[df_mut.index].index)

    return (
        clinical_df[["PATIENT_ID"]]
        .join(df_mut, on="PATIENT_ID")
        .drop(columns=["PATIENT_ID"])
        .fillna(0)
    )


def _process_mutations(
    raw_dir: Path,
    clinical_df: pd.DataFrame,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """Parse and align mutation data for a configured dataset."""
    mutation_path = raw_dir / dataset.mutation_file
    if not mutation_path.exists():
        print("  No mutation file found. Writing an empty mutation matrix.")
        return pd.DataFrame(index=clinical_df.index)

    df_mut = parse_maf_mutations(maf_path=mutation_path)
    if df_mut.empty:
        return pd.DataFrame(index=clinical_df.index)

    df_mut.index = (
        dataset.sample_prefix + df_mut.index.astype(str)
    ).map(standardise_sample_id)

    if dataset.mutations_by_patient:
        df_mut = _aggregate_mutations_by_patient(df_mut, clinical_df)
    else:
        df_mut = df_mut.reindex(clinical_df.index, fill_value=0)

    df_mut.index.name = "SAMPLE_ID"
    return df_mut


# ---------------------------------------------------------------------------
# TCGA Feature Engineering Helpers
# ---------------------------------------------------------------------------


def _extract_treatment_type_flags(
    df_treatment: pd.DataFrame, treatment_features: pd.DataFrame
) -> pd.DataFrame:
    """Extract binary treatment type indicator columns."""
    treatment_types = df_treatment["TREATMENT_TYPE"].dropna().unique()
    for treatment_type in treatment_types:
        col_name = "TX_TYPE_" + str(treatment_type).replace(" ", "_").upper()
        patients = df_treatment.loc[
            df_treatment["TREATMENT_TYPE"] == treatment_type, "PATIENT_ID"
        ].unique()
        treatment_features[col_name] = 0
        treatment_features.loc[
            treatment_features.index.isin(patients), col_name
        ] = 1
    return treatment_features


def _extract_agent_flags(
    df_treatment: pd.DataFrame, treatment_features: pd.DataFrame
) -> pd.DataFrame:
    """Extract binary agent indicator columns based on curated agent mapping."""
    for feature_name, agents in TCGA_KEY_TREATMENT_AGENTS.items():
        pattern = "|".join(agent.upper() for agent in agents)
        patients = df_treatment.loc[
            df_treatment["AGENT"]
            .astype(str)
            .str.upper()
            .str.contains(pattern, na=False, regex=True),
            "PATIENT_ID",
        ].unique()
        treatment_features[feature_name] = (
            treatment_features.index.isin(patients).astype(int)
        )
    return treatment_features


def _add_tcga_treatment_features(
    clinical_df: pd.DataFrame,
    raw_dir: Path,
) -> pd.DataFrame:
    """Add aggregated treatment features from the TCGA treatment timeline."""
    timeline_path = raw_dir / "data_timeline_treatment.txt"
    if not timeline_path.exists():
        return clinical_df

    print("  Found treatment timeline data. Aggregating treatment features...")
    df_treatment = pd.read_csv(timeline_path, sep="\t")

    required_columns = {"PATIENT_ID", "TREATMENT_TYPE", "AGENT"}
    if not required_columns.issubset(df_treatment.columns):
        print("  [WARNING] Treatment timeline is missing required columns. Skipping.")
        return clinical_df

    df_treatment["PATIENT_ID"] = df_treatment["PATIENT_ID"].astype(str).str.strip().str.upper()

    treatment_features = df_treatment.groupby("PATIENT_ID").agg(
        TREATMENT_TYPES=(
            "TREATMENT_TYPE",
            lambda vals: ", ".join(sorted(set(vals.dropna().astype(str)))),
        ),
        TREATMENT_AGENTS=(
            "AGENT",
            lambda vals: ", ".join(sorted(set(vals.dropna().astype(str)))),
        ),
    )

    treatment_features = _extract_treatment_type_flags(df_treatment, treatment_features)
    treatment_features = _extract_agent_flags(df_treatment, treatment_features)

    treatment_features = treatment_features.reset_index().rename(columns={"index": "PATIENT_ID"})
    clinical_df = clinical_df.merge(treatment_features, on="PATIENT_ID", how="left")

    clinical_df["TREATMENT_TYPES"] = clinical_df["TREATMENT_TYPES"].fillna("None")
    clinical_df["TREATMENT_AGENTS"] = clinical_df["TREATMENT_AGENTS"].fillna("None")

    tx_cols = [c for c in clinical_df.columns if c.startswith("TX_TYPE_") or c.startswith("TX_AGENT_")]
    clinical_df[tx_cols] = clinical_df[tx_cols].fillna(0).astype(int)
    return clinical_df


def _add_tcga_hypoxia_features(
    clinical_df: pd.DataFrame,
    raw_dir: Path,
) -> pd.DataFrame:
    """Add supplementary TCGA hypoxia data when available."""
    hypoxia_path = raw_dir / "data_clinical_supp_hypoxia.txt"
    if not hypoxia_path.exists():
        return clinical_df

    print("  Found supplementary hypoxia data. Merging hypoxia features...")
    df_hypoxia = pd.read_csv(hypoxia_path, sep="\t", comment="#")

    if not {"PATIENT_ID", "WINTER_HYPOXIA_SCORE"}.issubset(df_hypoxia.columns):
        print("  [WARNING] Hypoxia file is missing required columns. Skipping.")
        return clinical_df

    df_hypoxia["PATIENT_ID"] = df_hypoxia["PATIENT_ID"].astype(str).str.strip().str.upper()
    return clinical_df.merge(
        df_hypoxia[["PATIENT_ID", "WINTER_HYPOXIA_SCORE"]], on="PATIENT_ID", how="left"
    )


# ---------------------------------------------------------------------------
# Output Writing & Persistence Helpers
# ---------------------------------------------------------------------------


def _write_processed_outputs(
    processed_dir: Path,
    clinical_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    mutation_df: pd.DataFrame,
) -> None:
    """Write cleaned clinical, expression, and mutation datasets."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    clinical_output = processed_dir / CLEANED_CLINICAL_FILENAME
    expression_output = processed_dir / CLEANED_EXPRESSION_FILENAME
    mutation_output = processed_dir / CLEANED_MUTATION_FILENAME

    clinical_df.reset_index().to_csv(clinical_output, index=False)
    expression_df.to_csv(expression_output)
    mutation_df.to_csv(mutation_output)

    print("  Wrote processed outputs:")
    for path in (clinical_output, expression_output, mutation_output):
        print(f"    {display_path(path)}")


def _write_attrition_output(
    processed_dir: Path,
    attrition_records: list[AttritionRecord],
) -> None:
    """Write structured preprocessing attrition data."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    attrition_output = processed_dir / ATTRITION_FILENAME

    attrition_df = pd.DataFrame(rec.to_dict() for rec in attrition_records)
    attrition_df.to_csv(attrition_output, index=False)

    print("  Wrote attrition output:")
    print(f"    {display_path(attrition_output)}")


# ---------------------------------------------------------------------------
# Dataset Orchestration Pipeline
# ---------------------------------------------------------------------------


def _execute_dataset_cleaning_pipeline(
    dataset: DatasetConfig,
    raw_dir: Path,
    processed_dir: Path,
    load_expression_fn: Callable[[], pd.DataFrame],
    extra_clinical_transforms_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> list[AttritionRecord]:
    """Shared pipeline orchestrator for cohort data cleaning."""
    print(f"Cleaning {dataset.cohort_name} ({dataset.study_id})...")
    attrition_records: list[AttritionRecord] = []

    # 1. Clinical Data
    clinical_df = _load_clinical_data(raw_dir, dataset)
    n_raw = len(clinical_df)

    _record_attrition(
        attrition_records, dataset, "Clinical data loaded", n_raw, n_raw,
        "Merged patient-level and sample-level clinical records."
    )

    if extra_clinical_transforms_fn is not None:
        clinical_df = _apply_identifier_prefixes(clinical_df, dataset)
        clinical_df = clean_clinical_df(clinical_df)
        _record_attrition(
            attrition_records, dataset, "Clinical data harmonised", n_raw, len(clinical_df),
            "Applied identifier standardisation and clinical data cleaning."
        )
        clinical_df = extra_clinical_transforms_fn(clinical_df)
        clinical_df = clinical_df.set_index("SAMPLE_ID")
    else:
        clinical_df = _harmonise_clinical_data(clinical_df, dataset)
        _record_attrition(
            attrition_records, dataset, "Clinical data harmonised", n_raw, len(clinical_df),
            "Applied identifier standardisation, clinical cleaning, baseline filtering, and variable harmonisation."
        )

    # 2. Expression Data
    expression_df = load_expression_fn()
    n_before_align = len(clinical_df)

    expression_df, clinical_df = align_expression_and_clinical(expression_df, clinical_df)
    _record_attrition(
        attrition_records, dataset, "Clinical-expression alignment", n_before_align, len(clinical_df),
        "Retained samples with matching clinical and expression data."
    )

    # 3. Mutation Data
    mutation_df = _process_mutations(raw_dir, clinical_df, dataset)

    # 4. Outputs & Verification
    _write_processed_outputs(processed_dir, clinical_df, expression_df, mutation_df)
    print(f"  {dataset.cohort_name}: Cleaned {len(clinical_df):,} samples.")

    _run_sanity_checks(raw_dir=raw_dir, processed_dir=processed_dir, dataset=dataset)
    return attrition_records


def process_iatlas_dataset(
    dataset: DatasetConfig,
    raw_dir: Path,
    processed_dir: Path,
) -> list[AttritionRecord]:
    """Process an iAtlas/cBioPortal immunotherapy cohort."""
    return _execute_dataset_cleaning_pipeline(
        dataset=dataset,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        load_expression_fn=lambda: _load_iatlas_expression(
            raw_dir / dataset.expression_file, dataset
        ),
    )


def process_tcga_dataset(
    dataset: DatasetConfig,
    raw_dir: Path,
    processed_dir: Path,
) -> list[AttritionRecord]:
    """Process the TCGA-SKCM dataset."""
    def _tcga_clinical_transforms(df: pd.DataFrame) -> pd.DataFrame:
        df = _add_tcga_treatment_features(df, raw_dir)
        return _add_tcga_hypoxia_features(df, raw_dir)

    return _execute_dataset_cleaning_pipeline(
        dataset=dataset,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        load_expression_fn=lambda: _load_tcga_expression(
            raw_dir / dataset.expression_file,
            dataset,
            processed_dir / ENTREZ_CACHE_FILENAME,
        ),
        extra_clinical_transforms_fn=_tcga_clinical_transforms,
    )


def process_dataset(
    dataset: DatasetConfig,
) -> list[AttritionRecord]:
    """Process one configured dataset using its processing strategy."""
    raw_dir = RAW_DIR / dataset.raw_directory
    processed_dir = PROCESSED_DIR / dataset.processed_directory

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {display_path(raw_dir)}")

    if dataset.processing_strategy == "iatlas":
        return process_iatlas_dataset(dataset, raw_dir, processed_dir)

    if dataset.processing_strategy == "tcga":
        return process_tcga_dataset(dataset, raw_dir, processed_dir)

    raise ValueError(f"Unsupported processing strategy: {dataset.processing_strategy!r}")


# ---------------------------------------------------------------------------
# Post-Processing Sanity Check Sub-Functions
# ---------------------------------------------------------------------------


def _load_raw_non_synonymous_genes(mutation_path: Path) -> set[str] | None:
    """Load raw non-synonymous mutation gene set from MAF file."""
    try:
        df_raw = pd.read_csv(mutation_path, sep="\t", comment="#", low_memory=False)
    except Exception as error:
        print(f"  [WARNING] Could not read raw mutation file: {error}")
        return None

    required_columns = {"Hugo_Symbol", "Variant_Classification"}
    missing = required_columns - set(df_raw.columns)
    if missing:
        print(f"  [WARNING] Raw mutation file missing required columns: {', '.join(sorted(missing))}")
        return None

    df_filtered = df_raw[df_raw["Variant_Classification"].isin(NON_SILENT_VARIANT_CLASSIFICATIONS)]
    return set(df_filtered["Hugo_Symbol"].dropna().astype(str).str.strip())


def _compare_mutation_gene_sets(
    raw_genes: set[str], cleaned_genes: set[str]
) -> None:
    """Compare raw vs cleaned mutation gene sets and log diagnostic results."""
    genes_retained = raw_genes & cleaned_genes
    missing = raw_genes - cleaned_genes
    extra = cleaned_genes - raw_genes

    print(f"    Raw non-synonymous genes: {len(raw_genes):,}")
    print(f"    Cleaned mutation genes:   {len(cleaned_genes):,}")
    print(f"    Genes retained:           {len(genes_retained):,}")
    print(f"    Genes missing from clean: {len(missing):,}")
    print(f"    Extra cleaned genes:      {len(extra):,}")

    if missing:
        print("  [WARNING] Some raw non-synonymous mutation genes were missing from cleaned matrix.")
        print(f"    Example missing genes: {', '.join(sorted(missing)[:EXAMPLE_GENES_DISPLAY_LIMIT])}")
    elif raw_genes == cleaned_genes:
        print("  [PASS] All raw non-synonymous mutation genes are represented in the cleaned matrix.")

    if extra:
        print("  [WARNING] Cleaned mutation matrix contains genes not found in raw set.")
        print(f"    Example extra genes: {', '.join(sorted(extra)[:EXAMPLE_GENES_DISPLAY_LIMIT])}")


def _check_raw_mutation_gene_count(
    raw_dir: Path,
    dataset: DatasetConfig,
    mutation_df: pd.DataFrame,
) -> None:
    """Compare unique non-synonymous genes in raw file with cleaned mutation matrix."""
    mutation_path = raw_dir / dataset.mutation_file
    if not mutation_path.exists():
        print("  [INFO] Raw mutation file not found. Skipping raw-versus-cleaned mutation comparison.")
        return

    print("\n  Mutation gene-count comparison:")
    raw_genes = _load_raw_non_synonymous_genes(mutation_path)
    if raw_genes is None:
        return

    cleaned_genes = set(mutation_df.columns.astype(str).str.strip())
    _compare_mutation_gene_sets(raw_genes, cleaned_genes)


def _verify_output_files_exist(
    processed_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Check existence of expected output CSVs and return loaded DataFrames."""
    clinical_path = processed_dir / CLEANED_CLINICAL_FILENAME
    expression_path = processed_dir / CLEANED_EXPRESSION_FILENAME
    mutation_path = processed_dir / CLEANED_MUTATION_FILENAME

    missing = [p for p in (clinical_path, expression_path, mutation_path) if not p.exists()]
    if missing:
        print(f"  [FAIL] Missing output file(s): {', '.join(display_path(p) for p in missing)}")
        return None

    clin_df = pd.read_csv(clinical_path)
    expr_df = pd.read_csv(expression_path, index_col=0)
    mut_df = pd.read_csv(mutation_path, index_col=0)

    print("\n  Dataset dimensions:")
    print(f"    Clinical:   {len(clin_df):,} samples × {len(clin_df.columns):,} columns")
    print(f"    Expression: {len(expr_df):,} samples × {len(expr_df.columns):,} genes")
    print(f"    Mutations:  {len(mut_df):,} samples × {len(mut_df.columns):,} genes")

    return clin_df, expr_df, mut_df


def _verify_clinical_sanity(clinical_df: pd.DataFrame) -> None:
    """Perform diagnostic sanity checks on cleaned clinical data."""
    if "SAMPLE_ID" not in clinical_df.columns:
        print("  [FAIL] Clinical data has no SAMPLE_ID column.")
        return

    dups = clinical_df["SAMPLE_ID"].duplicated().sum()
    if dups:
        print(f"  [WARNING] Duplicate clinical sample IDs: {dups:,}")
    else:
        print("  [PASS] Clinical sample IDs are unique.")

    empty_cols = clinical_df.columns[clinical_df.isna().all()].tolist()
    if empty_cols:
        print("  [WARNING] Completely empty clinical columns:")
        for col in empty_cols:
            print(f"    - {col}")
    else:
        print("  [PASS] No completely empty clinical columns.")


def _verify_expression_sanity(expression_df: pd.DataFrame) -> None:
    """Perform diagnostic sanity checks on cleaned expression data."""
    dups = expression_df.index.duplicated().sum()
    if dups:
        print(f"  [WARNING] Duplicate expression sample IDs: {dups:,}")
    else:
        print("  [PASS] Expression sample IDs are unique.")

    nan_cnt = expression_df.isna().sum().sum()
    if nan_cnt:
        print(f"  [WARNING] Expression matrix contains {nan_cnt:,} NaN values.")
    else:
        print("  [PASS] Expression matrix contains no NaN values.")

    n_genes = len(expression_df.columns)
    if n_genes < EXPRESSION_MIN_GENES_WARN:
        print(f"  [WARNING] Expression matrix contains only {n_genes:,} genes.")
    elif n_genes < EXPRESSION_MIN_GENES_INFO:
        print(f"  [INFO] Expression matrix contains {n_genes:,} genes.")
    else:
        print(f"  [PASS] Expression matrix contains {n_genes:,} genes.")


def _verify_sample_overlap_sanity(
    clinical_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    mutation_df: pd.DataFrame,
) -> None:
    """Verify sample identifier alignment across clinical, expression, and mutation matrices."""
    if "SAMPLE_ID" not in clinical_df.columns:
        return

    clin_ids = set(clinical_df["SAMPLE_ID"].astype(str))
    expr_ids = set(expression_df.index.astype(str))
    mut_ids = set(mutation_df.index.astype(str))

    print("\n  Sample identifier overlap:")
    print(f"    Clinical <-> Expression: {len(clin_ids & expr_ids):,}")
    print(f"    Clinical <-> Mutation:   {len(clin_ids & mut_ids):,}")


def _verify_expression_range_sanity(expression_df: pd.DataFrame) -> None:
    """Check range and validity of numeric expression values."""
    vals = expression_df.select_dtypes(include="number").to_numpy()
    if not vals.size:
        return

    finite_vals = vals[np.isfinite(vals)]
    if not finite_vals.size:
        return

    print("\n  Expression value range:")
    print(f"    Minimum: {finite_vals.min():.4f}")
    print(f"    Maximum: {finite_vals.max():.4f}")

    if finite_vals.min() < 0:
        print("  [WARNING] Negative expression values detected.")
    if finite_vals.max() > EXPRESSION_MAX_VALUE_THRESHOLD:
        print("  [WARNING] Extremely large expression values detected.")


def _run_sanity_checks(
    raw_dir: Path,
    processed_dir: Path,
    dataset: DatasetConfig,
) -> None:
    """Run post-processing sanity checks on cleaned dataset outputs."""
    print("\n  Running post-processing sanity checks...")

    dfs = _verify_output_files_exist(processed_dir)
    if dfs is None:
        return

    clinical_df, expression_df, mutation_df = dfs

    _verify_clinical_sanity(clinical_df)
    _verify_expression_sanity(expression_df)

    if mutation_df.empty:
        print("  [WARNING] Mutation matrix is empty.")
    else:
        print(f"  [PASS] Mutation matrix contains {len(mutation_df.columns):,} genes.")

    _check_raw_mutation_gene_count(raw_dir, dataset, mutation_df)
    _verify_sample_overlap_sanity(clinical_df, expression_df, mutation_df)
    _verify_expression_range_sanity(expression_df)

    if dataset.processing_strategy == "tcga":
        n_genes = len(expression_df.columns)
        if n_genes < EXPRESSION_MIN_GENES_WARN:
            print(f"\n  [WARNING] TCGA expression matrix has only {n_genes:,} genes.")
        else:
            print(f"\n  [PASS] TCGA expression gene count appears plausible: {n_genes:,}.")

    print("\n  Sanity checks complete.")


# ---------------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------------


def _run_dataset_workflow(datasets: list[DatasetConfig]) -> int:
    """Iterate through configured datasets and execute cleaning pipeline."""
    success_count = 0
    for dataset in datasets:
        try:
            attrition_records = process_dataset(dataset)
            processed_dir = PROCESSED_DIR / dataset.processed_directory
            _write_attrition_output(processed_dir, attrition_records)
            success_count += 1
        except Exception as error:
            print(f"\n[ERROR] Dataset '{dataset.cohort_name}' failed: {error}")
    return success_count


def main() -> None:
    """Run the configured data-cleaning workflow."""
    print("==================================================")
    print("Data Cleaning Pipeline: Transforming Raw Data")
    print("==================================================\n")

    datasets = load_dataset_config(CONFIG_PATH)
    print(f"Loaded {len(datasets)} dataset configuration(s) from {display_path(CONFIG_PATH)}.\n")

    success_count = _run_dataset_workflow(datasets)

    print("\n==================================================")
    print(f"Workflow completed: {success_count}/{len(datasets)} datasets succeeded.")
    print("==================================================")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)

        with (
            contextlib.redirect_stdout(stdout_tee),
            contextlib.redirect_stderr(stderr_tee),
        ):
            print(f"Logging console output to {display_path(LOG_PATH)}")
            main()