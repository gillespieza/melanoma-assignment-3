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
    NON_SILENT_VARIANT_CLASSIFICATIONS
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
    CONFIG_DIR,
    LOG_DIR,
    RAW_DIR,
    PROCESSED_DIR,
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


# ============================================================================
# Configuration
# ============================================================================

CONFIG_PATH = CONFIG_DIR / "datasets.yaml"
LOG_PATH = LOG_DIR / "clean_data.log"


# ============================================================================
# Attrition data
# ============================================================================


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
    """
    Append a sample attrition record.

    Args:
        records:
            List receiving the new record.

        dataset:
            Dataset configuration.

        step:
            Name of the preprocessing step.

        n_before:
            Number of samples before the step.

        n_after:
            Number of samples after the step.

        reason:
            Explanation of the step or exclusion.
    """
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


# ============================================================================
# Clinical data
# ============================================================================


def _load_clinical_data(
    raw_dir: Path,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """
    Load and merge patient- and sample-level clinical data.

    Args:
        raw_dir:
            Raw dataset directory.

        dataset:
            Dataset configuration containing the clinical filenames.

    Returns:
        Merged clinical DataFrame.

    Raises:
        FileNotFoundError:
            If either required clinical file is missing.

        ValueError:
            If either clinical file lacks the required identifier column.
    """
    patient_path = raw_dir / dataset.clinical_file
    sample_path = raw_dir / dataset.clinical_sample_file

    missing_files = [
        path
        for path in (patient_path, sample_path)
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required clinical file(s): "
            + ", ".join(
                display_path(path)
                for path in missing_files
            )
        )

    df_patient = read_cbioportal_table(
        patient_path,
        required_column="PATIENT_ID",
    )

    df_sample = read_cbioportal_table(
        sample_path,
        required_column="SAMPLE_ID",
    )

    if "PATIENT_ID" not in df_patient.columns:
        raise ValueError(
            f"{display_path(patient_path)} "
            "does not contain PATIENT_ID."
        )

    required_sample_columns = {
        "SAMPLE_ID",
        "PATIENT_ID",
    }

    missing_sample_columns = (
        required_sample_columns
        - set(df_sample.columns)
    )

    if missing_sample_columns:
        raise ValueError(
            f"{display_path(sample_path)} "
            "is missing required columns: "
            f"{sorted(missing_sample_columns)}"
        )

    return pd.merge(
        df_sample,
        df_patient,
        on="PATIENT_ID",
        how="inner",
    )


def _apply_identifier_prefixes(
    df_clin: pd.DataFrame,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """
    Apply configured prefixes to patient and sample identifiers.

    Prefixes prevent identifier collisions when multiple cohorts are merged.
    """
    df = df_clin.copy()

    if "PATIENT_ID" in df.columns:
        df["PATIENT_ID"] = (
            dataset.patient_prefix
            + df["PATIENT_ID"].astype(str)
        ).str.upper()

    if "SAMPLE_ID" in df.columns:
        df["SAMPLE_ID"] = (
            dataset.sample_prefix
            + df["SAMPLE_ID"].astype(str)
        ).str.upper()

    return df


def _harmonise_clinical_data(
    df_clin: pd.DataFrame,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """
    Clean and harmonise clinical data for a configured dataset.
    """
    df = _apply_identifier_prefixes(
        df_clin,
        dataset,
    )

    df = clean_clinical_df(df)
    df = drop_empty_columns(df)

    if dataset.baseline_only:
        df = df[
            df["SAMPLE_ID"].str.endswith("_PRE")
        ]

    if "RESPONSE" in df.columns:
        df["RESPONSE_BINARY"] = (
            df["RESPONSE"].map(RECIST_RESPONSE_MAP)
        )
    else:
        df["RESPONSE_BINARY"] = np.nan

    if "AGE_AT_DIAGNOSIS" in df.columns:
        df["AGE"] = pd.to_numeric(
            df["AGE_AT_DIAGNOSIS"],
            errors="coerce",
        )

    if "SEX" in df.columns:
        sex_map = {
            "MALE": "Male",
            "FEMALE": "Female",
            "M": "Male",
            "F": "Female",
        }

        df["SEX"] = (
            df["SEX"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map(sex_map)
            .fillna("N/A")
        )

    return df.set_index("SAMPLE_ID")


# ============================================================================
# Expression processing
# ============================================================================


def _load_iatlas_expression(
    expression_path: Path,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """
    Load and prepare an iAtlas/cBioPortal expression matrix.
    """
    if not expression_path.exists():
        raise FileNotFoundError(
            "Expression file not found: "
            f"{display_path(expression_path)}"
        )

    df_expr = pd.read_csv(
        expression_path,
        sep="\t",
    )

    if "Hugo_Symbol" not in df_expr.columns:
        raise ValueError(
            "Expression file "
            f"{display_path(expression_path)} "
            "does not contain Hugo_Symbol."
        )

    df_expr = df_expr.dropna(
        subset=["Hugo_Symbol"]
    )

    df_expr = df_expr.set_index(
        "Hugo_Symbol"
    )

    if "Entrez_Gene_Id" in df_expr.columns:
        df_expr = df_expr.drop(
            columns=["Entrez_Gene_Id"]
        )

    # Duplicate gene symbols are averaged.
    df_expr = (
        df_expr
        .groupby(level=0)
        .mean(numeric_only=True)
    )

    df_expr.columns = (
        dataset.sample_prefix
        + df_expr.columns.astype(str)
    ).str.upper()

    if dataset.baseline_only:
        df_expr = df_expr[
            [
                column
                for column in df_expr.columns
                if column.endswith("_PRE")
            ]
        ]

    # Samples become rows and genes become columns.
    df_expr = df_expr.T

    df_expr.index.name = "SAMPLE_ID"

    df_expr = df_expr.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # iAtlas expression is TPM. Log2(TPM + 1) establishes the standard
    # processed expression representation.
    df_expr = np.log2(
        df_expr.clip(lower=0) + 1
    )

    return df_expr


def _load_tcga_expression(
    expression_path: Path,
    dataset: DatasetConfig,
    cache_path: Path,
) -> pd.DataFrame:
    """
    Load and prepare the TCGA RSEM expression matrix.

    TCGA expression data may contain Entrez gene identifiers that require
    mapping to Hugo gene symbols.
    """
    if not expression_path.exists():
        raise FileNotFoundError(
            "Expression file not found: "
            f"{display_path(expression_path)}"
        )

    df_expr = pd.read_csv(
        expression_path,
        sep="\t",
    )

    if "Entrez_Gene_Id" not in df_expr.columns:
        raise ValueError(
            "TCGA expression matrix does not contain Entrez_Gene_Id."
        )

    df_expr = df_expr.dropna(
        subset=["Entrez_Gene_Id"]
    )

    df_expr["Entrez_Gene_Id"] = (
        pd.to_numeric(
            df_expr["Entrez_Gene_Id"],
            errors="coerce",
        )
        .astype("Int64")
        .astype(str)
    )

    entrez_ids = (
        df_expr["Entrez_Gene_Id"]
        .dropna()
        .unique()
        .tolist()
    )

    gene_map = map_entrez_to_symbols(
        entrez_ids,
        cache_path=cache_path,
    )

    df_expr["Hugo_Symbol_Mapped"] = (
        df_expr["Entrez_Gene_Id"]
        .map(gene_map)
        .fillna(df_expr["Entrez_Gene_Id"])
    )

    df_expr = df_expr.set_index(
        "Hugo_Symbol_Mapped"
    )

    columns_to_drop = [
        column
        for column in (
            "Hugo_Symbol",
            "Entrez_Gene_Id",
        )
        if column in df_expr.columns
    ]

    df_expr = df_expr.drop(
        columns=columns_to_drop
    )

    # Duplicate mapped gene symbols are averaged.
    df_expr = (
        df_expr
        .groupby(level=0)
        .mean(numeric_only=True)
    )

    # Samples become rows and genes become columns.
    df_expr = df_expr.T

    df_expr.index = (
        dataset.sample_prefix
        + df_expr.index.astype(str)
    ).map(standardise_sample_id)

    df_expr.index.name = "SAMPLE_ID"

    df_expr = df_expr.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # RSEM values are transformed to the standard log2 representation.
    df_expr = np.log2(
        df_expr.clip(lower=0) + 1
    )

    return df_expr


# ============================================================================
# Mutation processing
# ============================================================================


def _process_mutations(
    raw_dir: Path,
    clinical_df: pd.DataFrame,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """
    Parse and align mutation data for a configured dataset.

    Mutation availability does not cause sample attrition. Samples without
    qualifying mutations are retained in the final cohort with an all-zero
    mutation profile.
    """
    mutation_path = raw_dir / dataset.mutation_file

    if not mutation_path.exists():
        print(
            "  No mutation file found. "
            "Writing an empty mutation matrix."
        )

        return pd.DataFrame(
            index=clinical_df.index
        )

    # Parse all qualifying mutations first. The dataset-specific sample
    # prefix is applied here so raw MAF identifiers can be aligned with the
    # already-prefixed clinical identifiers.
    df_mut = parse_maf_mutations(
        maf_path=mutation_path,
    )

    if df_mut.empty:
        return pd.DataFrame(
            index=clinical_df.index
        )

    df_mut.index = (
        dataset.sample_prefix
        + df_mut.index.astype(str)
    ).map(standardise_sample_id)

    if dataset.mutations_by_patient:
        df_mut.index = (
            df_mut.index
            .map(_sample_to_patient_id)
        )

        df_mut = (
            df_mut
            .groupby(level=0)
            .max()
        )

        patient_to_sample = (
            clinical_df[
                ["PATIENT_ID"]
            ]
            .reset_index()
            .drop_duplicates(
                subset=["PATIENT_ID"]
            )
            .set_index("PATIENT_ID")
        )

        df_mut = df_mut.reindex(
            patient_to_sample.index,
            fill_value=0,
        )

        df_mut = df_mut.join(
            patient_to_sample,
            how="right",
        )

        df_mut = df_mut.set_index(
            patient_to_sample.loc[
                df_mut.index
            ]
            .index
        )

        # Reconstruct a sample-level matrix aligned to clinical_df.
        df_mut = (
            clinical_df[
                ["PATIENT_ID"]
            ]
            .join(
                df_mut,
                on="PATIENT_ID",
            )
            .drop(
                columns=["PATIENT_ID"]
            )
            .fillna(0)
        )

    else:
        df_mut = df_mut.reindex(
            clinical_df.index,
            fill_value=0,
        )

    df_mut.index.name = "SAMPLE_ID"

    return df_mut


def _sample_to_patient_id(
    sample_id: str,
) -> str:
    """
    Convert a sample identifier to its patient identifier.

    For standard TCGA-style identifiers, the first three barcode components
    identify the patient.

    For prefixed cohort identifiers, the final sample suffix is removed.
    """
    if not isinstance(sample_id, str):
        return sample_id

    sample_id = sample_id.strip()

    if "_" in sample_id:
        return sample_id.rsplit("_", 1)[0]

    parts = sample_id.split("-")

    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])

    return sample_id


# ============================================================================
# TCGA clinical feature engineering
# ============================================================================


def _add_tcga_treatment_features(
    clinical_df: pd.DataFrame,
    raw_dir: Path,
) -> pd.DataFrame:
    """
    Add aggregated treatment features from the TCGA treatment timeline.
    """
    timeline_path = (
        raw_dir / "data_timeline_treatment.txt"
    )

    if not timeline_path.exists():
        return clinical_df

    print(
        "  Found treatment timeline data. "
        "Aggregating treatment features..."
    )

    df_treatment = pd.read_csv(
        timeline_path,
        sep="\t",
    )

    required_columns = {
        "PATIENT_ID",
        "TREATMENT_TYPE",
        "AGENT",
    }

    if not required_columns.issubset(
        df_treatment.columns
    ):
        print(
            "  [WARNING] Treatment timeline is missing "
            "required columns. Skipping treatment features."
        )

        return clinical_df

    df_treatment["PATIENT_ID"] = (
        df_treatment["PATIENT_ID"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    treatment_features = (
        df_treatment
        .groupby("PATIENT_ID")
        .agg(
            TREATMENT_TYPES=(
                "TREATMENT_TYPE",
                lambda values: ", ".join(
                    sorted(
                        set(
                            values
                            .dropna()
                            .astype(str)
                        )
                    )
                ),
            ),
            TREATMENT_AGENTS=(
                "AGENT",
                lambda values: ", ".join(
                    sorted(
                        set(
                            values
                            .dropna()
                            .astype(str)
                        )
                    )
                ),
            ),
        )
    )

    treatment_types = (
        df_treatment["TREATMENT_TYPE"]
        .dropna()
        .unique()
    )

    for treatment_type in treatment_types:
        column_name = (
            "TX_TYPE_"
            + str(treatment_type)
            .replace(" ", "_")
            .upper()
        )

        patients = (
            df_treatment.loc[
                df_treatment["TREATMENT_TYPE"]
                == treatment_type,
                "PATIENT_ID",
            ]
            .unique()
        )

        treatment_features[column_name] = 0

        treatment_features.loc[
            treatment_features.index.isin(patients),
            column_name,
        ] = 1

    key_agents = {
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

    for feature_name, agents in key_agents.items():
        pattern = "|".join(
            agent.upper()
            for agent in agents
        )

        patients = (
            df_treatment.loc[
                df_treatment["AGENT"]
                .astype(str)
                .str.upper()
                .str.contains(
                    pattern,
                    na=False,
                    regex=True,
                ),
                "PATIENT_ID",
            ]
            .unique()
        )

        treatment_features[feature_name] = (
            treatment_features.index
            .isin(patients)
            .astype(int)
        )

    treatment_features = (
        treatment_features
        .reset_index()
        .rename(
            columns={
                "index": "PATIENT_ID"
            }
        )
    )

    clinical_df = clinical_df.merge(
        treatment_features,
        on="PATIENT_ID",
        how="left",
    )

    clinical_df["TREATMENT_TYPES"] = (
        clinical_df["TREATMENT_TYPES"]
        .fillna("None")
    )

    clinical_df["TREATMENT_AGENTS"] = (
        clinical_df["TREATMENT_AGENTS"]
        .fillna("None")
    )

    treatment_columns = [
        column
        for column in clinical_df.columns
        if column.startswith("TX_TYPE_")
        or column.startswith("TX_AGENT_")
    ]

    clinical_df[treatment_columns] = (
        clinical_df[treatment_columns]
        .fillna(0)
        .astype(int)
    )

    return clinical_df


def _add_tcga_hypoxia_features(
    clinical_df: pd.DataFrame,
    raw_dir: Path,
) -> pd.DataFrame:
    """
    Add supplementary TCGA hypoxia data when available.
    """
    hypoxia_path = (
        raw_dir / "data_clinical_supp_hypoxia.txt"
    )

    if not hypoxia_path.exists():
        return clinical_df

    print(
        "  Found supplementary hypoxia data. "
        "Merging hypoxia features..."
    )

    df_hypoxia = pd.read_csv(
        hypoxia_path,
        sep="\t",
        comment="#",
    )

    required_columns = {
        "PATIENT_ID",
        "WINTER_HYPOXIA_SCORE",
    }

    if not required_columns.issubset(
        df_hypoxia.columns
    ):
        print(
            "  [WARNING] Hypoxia file is missing "
            "required columns. Skipping."
        )

        return clinical_df

    df_hypoxia["PATIENT_ID"] = (
        df_hypoxia["PATIENT_ID"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return clinical_df.merge(
        df_hypoxia[
            [
                "PATIENT_ID",
                "WINTER_HYPOXIA_SCORE",
            ]
        ],
        on="PATIENT_ID",
        how="left",
    )


# ============================================================================
# Dataset processing
# ============================================================================


def process_iatlas_dataset(
    dataset: DatasetConfig,
    raw_dir: Path,
    processed_dir: Path,
) -> list[AttritionRecord]:
    """
    Process an iAtlas/cBioPortal immunotherapy cohort.

    Returns:
        Attrition records generated during preprocessing.
    """
    print(
        f"Cleaning {dataset.cohort_name} "
        f"({dataset.study_id})..."
    )

    attrition_records: list[AttritionRecord] = []

    # ------------------------------------------------------------------
    # Clinical data
    # ------------------------------------------------------------------

    clinical_df = _load_clinical_data(
        raw_dir,
        dataset,
    )

    n_clinical_raw = len(clinical_df)

    _record_attrition(
        attrition_records,
        dataset,
        step="Clinical data loaded",
        n_before=n_clinical_raw,
        n_after=n_clinical_raw,
        reason=(
            "Merged patient-level and sample-level "
            "clinical records."
        ),
    )

    clinical_df = _harmonise_clinical_data(
        clinical_df,
        dataset,
    )

    _record_attrition(
        attrition_records,
        dataset,
        step="Clinical data harmonised",
        n_before=n_clinical_raw,
        n_after=len(clinical_df),
        reason=(
            "Applied identifier standardisation, clinical "
            "cleaning, baseline filtering, and variable "
            "harmonisation."
        ),
    )

    # ------------------------------------------------------------------
    # Expression data
    # ------------------------------------------------------------------

    expression_df = _load_iatlas_expression(
        raw_dir / dataset.expression_file,
        dataset,
    )

    n_before_alignment = len(clinical_df)

    expression_df, clinical_df = (
        align_expression_and_clinical(
            expression_df,
            clinical_df,
        )
    )

    _record_attrition(
        attrition_records,
        dataset,
        step="Clinical-expression alignment",
        n_before=n_before_alignment,
        n_after=len(clinical_df),
        reason=(
            "Retained samples with matching clinical and "
            "expression data."
        ),
    )

    # ------------------------------------------------------------------
    # Mutation data
    # ------------------------------------------------------------------

    mutation_df = _process_mutations(
        raw_dir,
        clinical_df,
        dataset,
    )

    # Lack of a qualifying mutation is not sample attrition.
    # Samples remain in the cohort with zero-valued mutation features.

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    _write_processed_outputs(
        processed_dir=processed_dir,
        clinical_df=clinical_df,
        expression_df=expression_df,
        mutation_df=mutation_df,
    )

    print(
        f"  {dataset.cohort_name}: "
        f"Cleaned {len(clinical_df):,} samples."
    )

    _run_sanity_checks(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        dataset=dataset,
    )

    return attrition_records


def process_tcga_dataset(
    dataset: DatasetConfig,
    raw_dir: Path,
    processed_dir: Path,
) -> list[AttritionRecord]:
    """
    Process the TCGA-SKCM dataset.

    Returns:
        Attrition records generated during preprocessing.
    """
    print(
        f"Cleaning {dataset.cohort_name} "
        f"({dataset.study_id})..."
    )

    attrition_records: list[AttritionRecord] = []

    # ------------------------------------------------------------------
    # Clinical data
    # ------------------------------------------------------------------

    clinical_df = _load_clinical_data(
        raw_dir,
        dataset,
    )

    n_clinical_raw = len(clinical_df)

    _record_attrition(
        attrition_records,
        dataset,
        step="Clinical data loaded",
        n_before=n_clinical_raw,
        n_after=n_clinical_raw,
        reason=(
            "Merged patient-level and sample-level "
            "clinical records."
        ),
    )

    clinical_df = _apply_identifier_prefixes(
        clinical_df,
        dataset,
    )

    clinical_df = clean_clinical_df(
        clinical_df
    )

    _record_attrition(
        attrition_records,
        dataset,
        step="Clinical data harmonised",
        n_before=n_clinical_raw,
        n_after=len(clinical_df),
        reason=(
            "Applied identifier standardisation and "
            "clinical data cleaning."
        ),
    )

    clinical_df = _add_tcga_treatment_features(
        clinical_df,
        raw_dir,
    )

    clinical_df = _add_tcga_hypoxia_features(
        clinical_df,
        raw_dir,
    )

    clinical_df = clinical_df.set_index(
        "SAMPLE_ID"
    )

    # ------------------------------------------------------------------
    # Expression data
    # ------------------------------------------------------------------

    expression_df = _load_tcga_expression(
        raw_dir / dataset.expression_file,
        dataset,
        processed_dir / "entrez_to_symbol_cache.json",
    )

    n_before_alignment = len(clinical_df)

    expression_df, clinical_df = (
        align_expression_and_clinical(
            expression_df,
            clinical_df,
        )
    )

    _record_attrition(
        attrition_records,
        dataset,
        step="Clinical-expression alignment",
        n_before=n_before_alignment,
        n_after=len(clinical_df),
        reason=(
            "Retained samples with matching clinical and "
            "expression data."
        ),
    )

    # ------------------------------------------------------------------
    # Mutation data
    # ------------------------------------------------------------------

    mutation_df = _process_mutations(
        raw_dir,
        clinical_df,
        dataset,
    )

    # Lack of a qualifying mutation is not sample attrition.
    # Samples remain in the cohort with zero-valued mutation features.

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    _write_processed_outputs(
        processed_dir=processed_dir,
        clinical_df=clinical_df,
        expression_df=expression_df,
        mutation_df=mutation_df,
    )

    print(
        f"  {dataset.cohort_name}: "
        f"Cleaned {len(clinical_df):,} samples."
    )

    _run_sanity_checks(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        dataset=dataset,
    )

    return attrition_records


def process_dataset(
    dataset: DatasetConfig,
) -> list[AttritionRecord]:
    """
    Process one configured dataset using its processing strategy.

    Returns:
        Attrition records generated during preprocessing.
    """
    raw_dir = RAW_DIR / dataset.raw_directory

    processed_dir = (
        PROCESSED_DIR
        / dataset.processed_directory
    )

    if not raw_dir.exists():
        raise FileNotFoundError(
            "Raw dataset directory not found: "
            f"{display_path(raw_dir)}"
        )

    if dataset.processing_strategy == "iatlas":
        return process_iatlas_dataset(
            dataset,
            raw_dir,
            processed_dir,
        )

    if dataset.processing_strategy == "tcga":
        return process_tcga_dataset(
            dataset,
            raw_dir,
            processed_dir,
        )

    raise ValueError(
        "Unsupported processing strategy: "
        f"{dataset.processing_strategy!r}"
    )


# ============================================================================
# Output handling
# ============================================================================


def _write_processed_outputs(
    processed_dir: Path,
    clinical_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    mutation_df: pd.DataFrame,
) -> None:
    """
    Write cleaned clinical, expression, and mutation datasets.
    """
    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clinical_output = (
        processed_dir / "clin_cleaned.csv"
    )

    expression_output = (
        processed_dir / "expr_cleaned.csv"
    )

    mutation_output = (
        processed_dir / "mutations_cleaned.csv"
    )

    clinical_df.reset_index().to_csv(
        clinical_output,
        index=False,
    )

    expression_df.to_csv(
        expression_output
    )

    mutation_df.to_csv(
        mutation_output
    )

    print(
        "  Wrote processed outputs:"
    )

    for path in (
        clinical_output,
        expression_output,
        mutation_output,
    ):
        print(
            f"    {display_path(path)}"
        )


def _write_attrition_output(
    processed_dir: Path,
    attrition_records: list[AttritionRecord],
) -> None:
    """
    Write structured preprocessing attrition data.
    """
    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    attrition_output = (
        processed_dir / "attrition.csv"
    )

    attrition_df = pd.DataFrame(
        record.to_dict()
        for record in attrition_records
    )

    attrition_df.to_csv(
        attrition_output,
        index=False,
    )

    print(
        "  Wrote attrition output:"
    )

    print(
        f"    {display_path(attrition_output)}"
    )

# ============================================================================
# Post-processing sanity checks
# ============================================================================
def _check_raw_mutation_gene_count(
    raw_dir: Path,
    dataset: DatasetConfig,
    mutation_df: pd.DataFrame,
) -> None:
    """
    Compare unique non-synonymous genes in the raw mutation file with the
    genes retained in the cleaned mutation matrix.
    """

    mutation_path = (
        raw_dir / dataset.mutation_file
    )

    if not mutation_path.exists():
        print(
            "  [INFO] Raw mutation file not found. "
            "Skipping raw-versus-cleaned mutation comparison."
        )

        return

    print(
        "\n  Mutation gene-count comparison:"
    )

    try:
        df_raw_mutations = pd.read_csv(
            mutation_path,
            sep="\t",
            comment="#",
            low_memory=False,
        )

    except Exception as error:
        print(
            "  [WARNING] Could not read raw mutation file: "
            f"{error}"
        )

        return

    required_columns = {
        "Hugo_Symbol",
        "Variant_Classification",
    }

    missing_columns = (
        required_columns
        - set(df_raw_mutations.columns)
    )

    if missing_columns:
        print(
            "  [WARNING] Raw mutation file is missing "
            "required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

        return

    # Restrict to the same non-synonymous mutation classes used by the
    # production mutation parser.
    df_raw_non_synonymous = (
        df_raw_mutations[
            df_raw_mutations[
                "Variant_Classification"
            ].isin(
                NON_SILENT_VARIANT_CLASSIFICATIONS
            )
        ]
    )

    raw_genes = set(
        df_raw_non_synonymous[
            "Hugo_Symbol"
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )

    cleaned_genes = set(
        mutation_df.columns
        .astype(str)
        .str.strip()
    )

    genes_retained = (
        raw_genes
        & cleaned_genes
    )

    genes_missing_from_cleaned = (
        raw_genes
        - cleaned_genes
    )

    extra_cleaned_genes = (
        cleaned_genes
        - raw_genes
    )

    print(
        f"    Raw non-synonymous genes: "
        f"{len(raw_genes):,}"
    )

    print(
        f"    Cleaned mutation genes:   "
        f"{len(cleaned_genes):,}"
    )

    print(
        f"    Genes retained:           "
        f"{len(genes_retained):,}"
    )

    print(
        f"    Genes missing from clean: "
        f"{len(genes_missing_from_cleaned):,}"
    )

    print(
        f"    Extra cleaned genes:      "
        f"{len(extra_cleaned_genes):,}"
    )

    if genes_missing_from_cleaned:
        print(
            "  [WARNING] Some raw non-synonymous mutation genes "
            "were not retained in the cleaned matrix."
        )

        print(
            "    Example missing genes: "
            + ", ".join(
                sorted(
                    genes_missing_from_cleaned
                )[:20]
            )
        )

    elif raw_genes == cleaned_genes:
        print(
            "  [PASS] All raw non-synonymous mutation genes "
            "are represented in the cleaned matrix."
        )

    if extra_cleaned_genes:
        print(
            "  [WARNING] Cleaned mutation matrix contains "
            "genes not found in the raw non-synonymous mutation set."
        )

        print(
            "    Example extra genes: "
            + ", ".join(
                sorted(
                    extra_cleaned_genes
                )[:20]
            )
        )

def _run_sanity_checks(
    raw_dir: Path,
    processed_dir: Path,
    dataset: DatasetConfig,
) -> None:
    """
    Run post-processing sanity checks on cleaned dataset outputs.

    The checks are diagnostic only and do not modify the cleaned data.
    """

    print(
        "\n  Running post-processing sanity checks..."
    )

    clinical_path = (
        processed_dir / "clin_cleaned.csv"
    )

    expression_path = (
        processed_dir / "expr_cleaned.csv"
    )

    mutation_path = (
        processed_dir / "mutations_cleaned.csv"
    )

    # ------------------------------------------------------------------
    # Check output files
    # ------------------------------------------------------------------

    required_files = {
        "clinical": clinical_path,
        "expression": expression_path,
        "mutation": mutation_path,
    }

    missing_files = [
        path
        for path in required_files.values()
        if not path.exists()
    ]

    if missing_files:
        print(
            "  [FAIL] Missing output file(s): "
            + ", ".join(
                display_path(path)
                for path in missing_files
            )
        )

        return

    # ------------------------------------------------------------------
    # Load outputs
    # ------------------------------------------------------------------

    clinical_df = pd.read_csv(
        clinical_path
    )

    expression_df = pd.read_csv(
        expression_path,
        index_col=0,
    )

    mutation_df = pd.read_csv(
        mutation_path,
        index_col=0,
    )

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    print(
        "\n  Dataset dimensions:"
    )

    print(
        f"    Clinical:   "
        f"{len(clinical_df):,} samples × "
        f"{len(clinical_df.columns):,} columns"
    )

    print(
        f"    Expression: "
        f"{len(expression_df):,} samples × "
        f"{len(expression_df.columns):,} genes"
    )

    print(
        f"    Mutations:  "
        f"{len(mutation_df):,} samples × "
        f"{len(mutation_df.columns):,} genes"
    )

    # ------------------------------------------------------------------
    # Clinical checks
    # ------------------------------------------------------------------

    if "SAMPLE_ID" not in clinical_df.columns:
        print(
            "  [FAIL] Clinical data has no SAMPLE_ID column."
        )

    else:
        duplicate_clinical_ids = (
            clinical_df["SAMPLE_ID"]
            .duplicated()
            .sum()
        )

        if duplicate_clinical_ids:
            print(
                "  [WARNING] Duplicate clinical sample IDs: "
                f"{duplicate_clinical_ids:,}"
            )

        else:
            print(
                "  [PASS] Clinical sample IDs are unique."
            )

    empty_clinical_columns = (
        clinical_df.columns[
            clinical_df.isna().all()
        ]
        .tolist()
    )

    if empty_clinical_columns:
        print(
            "  [WARNING] Completely empty clinical columns:"
        )

        for column in empty_clinical_columns:
            print(
                f"    - {column}"
            )

    else:
        print(
            "  [PASS] No completely empty clinical columns."
        )

    # ------------------------------------------------------------------
    # Expression checks
    # ------------------------------------------------------------------

    duplicate_expression_ids = (
        expression_df.index
        .duplicated()
        .sum()
    )

    if duplicate_expression_ids:
        print(
            "  [WARNING] Duplicate expression sample IDs: "
            f"{duplicate_expression_ids:,}"
        )

    else:
        print(
            "  [PASS] Expression sample IDs are unique."
        )

    expression_nan_count = (
        expression_df.isna()
        .sum()
        .sum()
    )

    if expression_nan_count:
        print(
            "  [WARNING] Expression matrix contains "
            f"{expression_nan_count:,} NaN values."
        )

    else:
        print(
            "  [PASS] Expression matrix contains no NaN values."
        )

    if len(expression_df.columns) < 1_000:
        print(
            "  [WARNING] Expression matrix contains only "
            f"{len(expression_df.columns):,} genes."
        )

    elif len(expression_df.columns) < 5_000:
        print(
            "  [INFO] Expression matrix contains "
            f"{len(expression_df.columns):,} genes."
        )

    else:
        print(
            "  [PASS] Expression matrix contains "
            f"{len(expression_df.columns):,} genes."
        )

    # ------------------------------------------------------------------
    # Mutation checks
    # ------------------------------------------------------------------

    if mutation_df.empty:
        print(
            "  [WARNING] Mutation matrix is empty."
        )

    else:
        print(
            "  [PASS] Mutation matrix contains "
            f"{len(mutation_df.columns):,} genes."
        )

    _check_raw_mutation_gene_count(
        raw_dir=raw_dir,
        dataset=dataset,
        mutation_df=mutation_df,
    )

    # ------------------------------------------------------------------
    # Sample overlap checks
    # ------------------------------------------------------------------

    if "SAMPLE_ID" in clinical_df.columns:

        clinical_ids = set(
            clinical_df["SAMPLE_ID"]
            .astype(str)
        )

        expression_ids = set(
            expression_df.index
            .astype(str)
        )

        mutation_ids = set(
            mutation_df.index
            .astype(str)
        )

        clinical_expression_overlap = (
            clinical_ids
            & expression_ids
        )

        clinical_mutation_overlap = (
            clinical_ids
            & mutation_ids
        )

        print(
            "\n  Sample identifier overlap:"
        )

        print(
            f"    Clinical ↔ Expression: "
            f"{len(clinical_expression_overlap):,}"
        )

        print(
            f"    Clinical ↔ Mutation:   "
            f"{len(clinical_mutation_overlap):,}"
        )

    # ------------------------------------------------------------------
    # Expression value range
    # ------------------------------------------------------------------

    expression_values = (
        expression_df
        .select_dtypes(include="number")
        .to_numpy()
    )

    if expression_values.size:

        finite_values = (
            expression_values[
                np.isfinite(expression_values)
            ]
        )

        if finite_values.size:

            print(
                "\n  Expression value range:"
            )

            print(
                f"    Minimum: "
                f"{finite_values.min():.4f}"
            )

            print(
                f"    Maximum: "
                f"{finite_values.max():.4f}"
            )

            if finite_values.min() < 0:
                print(
                    "  [WARNING] Negative expression values detected."
                )

            if finite_values.max() > 50:
                print(
                    "  [WARNING] Extremely large expression values detected."
                )

    # ------------------------------------------------------------------
    # Dataset-specific TCGA check
    # ------------------------------------------------------------------

    if dataset.processing_strategy == "tcga":

        n_genes = len(
            expression_df.columns
        )

        if n_genes < 1_000:
            print(
                "\n  [WARNING] TCGA expression matrix has "
                f"only {n_genes:,} genes."
            )

            print(
                "  Expected a substantially larger "
                "gene expression matrix."
            )

        else:
            print(
                "\n  [PASS] TCGA expression gene count "
                f"appears plausible: {n_genes:,}."
            )

    print(
        "\n  Sanity checks complete."
    )

# ============================================================================
# Main workflow
# ============================================================================


def main() -> None:
    """
    Run the configured data-cleaning workflow.
    """
    print(
        "=================================================="
    )
    print(
        "Data Cleaning Pipeline: Transforming Raw Data"
    )
    print(
        "==================================================\n"
    )

    datasets = load_dataset_config(
        CONFIG_PATH
    )

    print(
        f"Loaded {len(datasets)} dataset configuration(s) "
        f"from {display_path(CONFIG_PATH)}.\n"
    )

    success_count = 0

    for dataset in datasets:
        try:
            attrition_records = process_dataset(
                dataset
            )

            processed_dir = (
                PROCESSED_DIR
                / dataset.processed_directory
            )

            _write_attrition_output(
                processed_dir,
                attrition_records,
            )

            success_count += 1

        except Exception as error:
            print(
                f"\n[ERROR] Dataset "
                f"'{dataset.cohort_name}' failed: "
                f"{error}"
            )

    print(
        "\n=================================================="
    )
    print(
        f"Workflow completed: "
        f"{success_count}/{len(datasets)} datasets succeeded."
    )
    print(
        "=================================================="
    )


if __name__ == "__main__":
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with LOG_PATH.open(
        "w",
        encoding="utf-8",
    ) as log_file:

        stdout_tee = TeeStream(
            sys.stdout,
            log_file,
        )

        stderr_tee = TeeStream(
            sys.stderr,
            log_file,
        )

        with (
            contextlib.redirect_stdout(stdout_tee),
            contextlib.redirect_stderr(stderr_tee),
        ):
            print(
                "Logging console output to "
                f"{display_path(LOG_PATH)}"
            )

            main()