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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# Bootstrap project root resolution
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_CANDIDATE = SCRIPT_DIR.parent

if str(PROJECT_ROOT_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_CANDIDATE))


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
    CONFIG_DIR,
    LOG_DIR,
    PROCESSED_DIR,
)


# ============================================================================
# Configuration
# ============================================================================

CONFIG_PATH = CONFIG_DIR / "datasets.yaml"
LOG_PATH = LOG_DIR / "merge_datasets.log"

MERGED_DIR = PROCESSED_DIR / "merged"

FULL_DIR = MERGED_DIR / "full"
IMMUNOTHERAPY_DIR = MERGED_DIR / "immunotherapy"


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


def _get_dataset_directory(
    dataset: DatasetConfig,
) -> Path:
    """
    Return the processed directory for a configured dataset.
    """
    return (
        PROCESSED_DIR
        / dataset.processed_directory
    )


def _load_cleaned_expression(
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """
    Load a cleaned expression matrix.

    Returns:
        DataFrame indexed by SAMPLE_ID with genes as columns.
    """
    dataset_dir = _get_dataset_directory(dataset)
    expression_path = dataset_dir / "expr_cleaned.csv"

    if not expression_path.exists():
        raise FileNotFoundError(
            "Cleaned expression file not found: "
            f"{display_path(expression_path)}"
        )

    df_expr = pd.read_csv(
        expression_path,
        index_col=0,
    )

    df_expr.index = (
        df_expr.index
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df_expr.index.name = "SAMPLE_ID"

    return df_expr


def _load_cleaned_clinical(
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """
    Load a cleaned clinical dataset.

    Returns:
        DataFrame indexed by SAMPLE_ID.
    """
    dataset_dir = _get_dataset_directory(dataset)
    clinical_path = dataset_dir / "clin_cleaned.csv"

    if not clinical_path.exists():
        raise FileNotFoundError(
            "Cleaned clinical file not found: "
            f"{display_path(clinical_path)}"
        )

    df_clin = pd.read_csv(
        clinical_path,
    )

    if "SAMPLE_ID" not in df_clin.columns:
        raise ValueError(
            "Cleaned clinical file does not contain "
            f"SAMPLE_ID: {display_path(clinical_path)}"
        )

    df_clin["SAMPLE_ID"] = (
        df_clin["SAMPLE_ID"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df_clin = df_clin.set_index(
        "SAMPLE_ID"
    )

    df_clin.index.name = "SAMPLE_ID"

    return df_clin


def _load_cleaned_mutations(
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """
    Load a cleaned binary mutation matrix.

    Returns:
        DataFrame indexed by SAMPLE_ID or patient identifier,
        with gene symbols as columns.
    """
    dataset_dir = _get_dataset_directory(dataset)
    mutation_path = dataset_dir / "mutations_cleaned.csv"

    if not mutation_path.exists():
        print(
            f"  No cleaned mutation file found for "
            f"{dataset.cohort_name}. "
            "Using an empty mutation matrix."
        )

        return pd.DataFrame()

    df_mut = pd.read_csv(
        mutation_path,
        index_col=0,
    )

    df_mut.index = (
        df_mut.index
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df_mut.index.name = "SAMPLE_ID"

    return df_mut


def load_dataset(
    dataset: DatasetConfig,
) -> DatasetFrames:
    """
    Load all cleaned outputs for one configured dataset.

    Returns:
        Tuple containing:

        - expression DataFrame;
        - clinical DataFrame;
        - mutation DataFrame.
    """
    print(
        f"Loading {dataset.cohort_name} "
        f"({dataset.study_id})..."
    )

    df_expr = _load_cleaned_expression(
        dataset
    )

    df_clin = _load_cleaned_clinical(
        dataset
    )

    df_mut = _load_cleaned_mutations(
        dataset
    )

    print(
        f"  Expression: {df_expr.shape[0]:,} samples × "
        f"{df_expr.shape[1]:,} genes"
    )

    print(
        f"  Clinical:   {df_clin.shape[0]:,} samples × "
        f"{df_clin.shape[1]:,} features"
    )

    print(
        f"  Mutations:  {df_mut.shape[0]:,} samples × "
        f"{df_mut.shape[1]:,} genes"
    )

    return (
        df_expr,
        df_clin,
        df_mut,
    )


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
        .fillna("N/A")
    )


def _normalise_specimen_type(
    series: pd.Series,
) -> pd.Series:
    """
    Harmonise specimen-type descriptions.
    """
    def classify(value: object) -> str:
        if pd.isna(value):
            return "N/A"

        value = str(value).strip().lower()

        if not value:
            return "N/A"

        if "primary" in value:
            return "Primary"

        if (
            "metastatic" in value
            or "metastasis" in value
            or "metastase" in value
        ):
            return "Metastatic"

        return "N/A"

    return series.map(classify)


def _harmonise_clinical(
    df_clin: pd.DataFrame,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """
    Convert cohort-specific clinical data to a common schema.
    """
    df = pd.DataFrame(
        index=df_clin.index
    )

    # ------------------------------------------------------------------
    # Required identifiers
    # ------------------------------------------------------------------

    if "PATIENT_ID" in df_clin.columns:
        pass

    # ------------------------------------------------------------------
    # Patient identifier
    # ------------------------------------------------------------------

    if "PATIENT_ID" in df_clin.columns:
        df["PATIENT_ID"] = df_clin[
            "PATIENT_ID"
        ]
    else:
        df["PATIENT_ID"] = np.nan

    # ------------------------------------------------------------------
    # Cohort
    # ------------------------------------------------------------------

    df["COHORT"] = dataset.cohort_name

    # ------------------------------------------------------------------
    # Overall survival
    # ------------------------------------------------------------------

    if "OS_MONTHS" in df_clin.columns:
        df["OS_MONTHS"] = pd.to_numeric(
            df_clin["OS_MONTHS"],
            errors="coerce",
        )
    else:
        df["OS_MONTHS"] = np.nan

    if "OS_STATUS" in df_clin.columns:
        df["OS_STATUS"] = _normalise_os_status(
            df_clin["OS_STATUS"]
        )
    else:
        df["OS_STATUS"] = np.nan

    # ------------------------------------------------------------------
    # Response
    # ------------------------------------------------------------------

    if "RESPONSE" in df_clin.columns:
        df["RESPONSE"] = df_clin[
            "RESPONSE"
        ]
    else:
        df["RESPONSE"] = np.nan

    if "RESPONSE_BINARY" in df_clin.columns:
        df["RESPONSE_BINARY"] = pd.to_numeric(
            df_clin["RESPONSE_BINARY"],
            errors="coerce",
        )
    else:
        df["RESPONSE_BINARY"] = np.nan

    # ------------------------------------------------------------------
    # Demographics
    # ------------------------------------------------------------------

    if "AGE" in df_clin.columns:
        df["AGE"] = pd.to_numeric(
            df_clin["AGE"],
            errors="coerce",
        )
    else:
        df["AGE"] = np.nan

    if "RACE" in df_clin.columns:
        df["RACE"] = df_clin[
            "RACE"
        ]
    else:
        df["RACE"] = np.nan

    if "SEX" in df_clin.columns:
        df["SEX"] = _normalise_sex(
            df_clin["SEX"]
        )
    else:
        df["SEX"] = "N/A"

    # ------------------------------------------------------------------
    # Specimen type
    # ------------------------------------------------------------------

    if "SPECIMEN_TYPE" in df_clin.columns:
        df["SPECIMEN_TYPE"] = (
            _normalise_specimen_type(
                df_clin["SPECIMEN_TYPE"]
            )
        )

    elif "SAMPLE_TYPE" in df_clin.columns:
        df["SPECIMEN_TYPE"] = (
            _normalise_specimen_type(
                df_clin["SAMPLE_TYPE"]
            )
        )

    elif "BIOPSY_SITE" in df_clin.columns:
        df["SPECIMEN_TYPE"] = (
            _normalise_specimen_type(
                df_clin["BIOPSY_SITE"]
            )
        )

    else:
        df["SPECIMEN_TYPE"] = "N/A"

    # ------------------------------------------------------------------
    # Immunotherapy status
    # ------------------------------------------------------------------

    if "IMMUNOTHERAPY" in df_clin.columns:
        df["IMMUNOTHERAPY"] = pd.to_numeric(
            df_clin["IMMUNOTHERAPY"],
            errors="coerce",
        ).fillna(0).astype(int)

    elif (
        "TX_TYPE_IMMUNOTHERAPY"
        in df_clin.columns
    ):
        df["IMMUNOTHERAPY"] = pd.to_numeric(
            df_clin[
                "TX_TYPE_IMMUNOTHERAPY"
            ],
            errors="coerce",
        ).fillna(0).astype(int)

    else:
        # Configured trial cohorts are assumed to be
        # immunotherapy cohorts.
        if dataset.processing_strategy == "iatlas":
            df["IMMUNOTHERAPY"] = 1
        else:
            df["IMMUNOTHERAPY"] = 0

    df.index.name = "SAMPLE_ID"

    return df


# ============================================================================
# Expression processing
# ============================================================================


def zscore_expression(
    df_expr: pd.DataFrame,
) -> pd.DataFrame:
    """
    Z-score standardise each gene independently within one cohort.

    Standardisation is performed within each cohort before merging to
    prevent cohort-specific expression distributions from dominating
    the merged matrix.

    Constant or entirely missing genes receive a standard deviation of
    one to avoid division by zero.
    """
    df = df_expr.copy()

    means = df.mean(
        axis=0,
        skipna=True,
    )

    stds = (
        df.std(
            axis=0,
            skipna=True,
        )
        .replace(0, 1.0)
        .fillna(1.0)
    )

    return (
        df - means
    ) / stds


def find_common_genes(
    expression_data: Dict[str, pd.DataFrame],
) -> List[str]:
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


def _mutation_feature_columns() -> List[str]:
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
    """
    Extract configured driver mutation features.

    Converts gene-level mutation columns such as:

        BRAF
        NRAS
        NF1

    into:

        mut_BRAF
        mut_NRAS
        mut_NF1
    """
    result = pd.DataFrame(
        index=df_mut.index
    )

    for gene in DRIVER_GENES:
        output_column = (
            f"mut_{gene}"
        )

        if gene in df_mut.columns:
            result[output_column] = (
                pd.to_numeric(
                    df_mut[gene],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
            )
        else:
            result[output_column] = 0

    return result


def _build_genomic_features(
    datasets: List[DatasetConfig],
    clinical_data: Dict[str, pd.DataFrame],
    mutation_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build a unified genomic feature matrix for all merged samples.
    """
    mutation_features = (
        _mutation_feature_columns()
    )

    all_feature_columns = [
        "PATIENT_ID",
        "SAMPLE_ID",
        "COHORT",
    ]

    all_feature_columns.extend(
        GENOMIC_FEATURES
    )

    all_feature_columns.extend(
        mutation_features
    )

    genomic_rows = []

    for dataset in datasets:
        cohort_name = dataset.cohort_name

        df_clin = clinical_data[
            cohort_name
        ]

        df_mut = mutation_data.get(
            cohort_name,
            pd.DataFrame(),
        )

        if not df_mut.empty:
            df_driver_mut = (
                _load_driver_mutation_features(
                    df_mut
                )
            )
        else:
            df_driver_mut = pd.DataFrame(
                index=df_clin.index
            )

            for column in mutation_features:
                df_driver_mut[column] = 0

        for sample_id, clinical_row in df_clin.iterrows():
            row = {
                "PATIENT_ID": clinical_row.get(
                    "PATIENT_ID",
                    np.nan,
                ),
                "SAMPLE_ID": sample_id,
                "COHORT": cohort_name,
            }

            # Add configured genomic features.
            for feature in GENOMIC_FEATURES:
                if feature in df_clin.columns:
                    row[feature] = clinical_row[
                        feature
                    ]
                elif (
                    feature in df_mut.columns
                ):
                    row[feature] = df_mut.loc[
                        sample_id,
                        feature,
                    ]
                else:
                    row[feature] = np.nan

            # Add driver mutation features.
            for column in mutation_features:
                if (
                    sample_id
                    in df_driver_mut.index
                ):
                    row[column] = (
                        df_driver_mut.loc[
                            sample_id,
                            column,
                        ]
                    )
                else:
                    row[column] = 0

            genomic_rows.append(row)

    if not genomic_rows:
        return pd.DataFrame(
            columns=all_feature_columns
        )

    df_genomic = pd.DataFrame(
        genomic_rows
    )

    # Ensure all configured columns exist.
    for column in all_feature_columns:
        if column not in df_genomic.columns:
            df_genomic[column] = np.nan

    return df_genomic[
        all_feature_columns
    ]


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
    """
    Save merged expression, clinical, and genomic outputs.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    expr_path = (
        output_dir
        / "expr_merged.csv"
    )

    clinical_path = (
        output_dir
        / "clin_merged.csv"
    )

    genomic_path = (
        output_dir
        / "merged_genomic.csv"
    )

    df_expr.to_csv(
        expr_path
    )

    df_clin.reset_index().to_csv(
        clinical_path,
        index=False,
    )

    df_genomic.to_csv(
        genomic_path,
        index=False,
    )

    print(
        f"\n[{label}] Saved merged outputs:"
    )

    print(
        f"  Expression: "
        f"{display_path(expr_path)} "
        f"{df_expr.shape}"
    )

    print(
        f"  Clinical:   "
        f"{display_path(clinical_path)} "
        f"{df_clin.shape}"
    )

    print(
        f"  Genomic:    "
        f"{display_path(genomic_path)} "
        f"{df_genomic.shape}"
    )


# ============================================================================
# Merge construction
# ============================================================================


def _prepare_dataset_data(
    datasets: List[DatasetConfig],
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
]:
    """
    Load and harmonise all configured datasets.
    """
    expression_data = {}
    clinical_data = {}
    mutation_data = {}

    for dataset in datasets:
        (
            df_expr,
            df_clin_raw,
            df_mut,
        ) = load_dataset(
            dataset
        )

        # --------------------------------------------------------------
        # Align expression and clinical samples.
        # --------------------------------------------------------------

        common_ids = (
            df_expr.index
            .intersection(
                df_clin_raw.index
            )
        )

        if len(common_ids) == 0:
            raise ValueError(
                f"No overlapping samples between "
                f"expression and clinical data for "
                f"{dataset.cohort_name}."
            )

        df_expr = df_expr.loc[
            common_ids
        ]

        df_clin_raw = df_clin_raw.loc[
            common_ids
        ]

        # --------------------------------------------------------------
        # Align mutation data where possible.
        # --------------------------------------------------------------

        if not df_mut.empty:
            mutation_ids = (
                df_mut.index
                .intersection(
                    common_ids
                )
            )

            df_mut = df_mut.loc[
                mutation_ids
            ]

        # --------------------------------------------------------------
        # Harmonise clinical metadata.
        # --------------------------------------------------------------

        df_clin = _harmonise_clinical(
            df_clin_raw,
            dataset,
        )

        # Ensure exactly the same sample order.
        df_clin = df_clin.loc[
            df_expr.index
        ]

        expression_data[
            dataset.cohort_name
        ] = df_expr

        clinical_data[
            dataset.cohort_name
        ] = df_clin

        mutation_data[
            dataset.cohort_name
        ] = df_mut

        print(
            f"  Aligned samples: "
            f"{len(common_ids):,}"
        )

    return (
        expression_data,
        clinical_data,
        mutation_data,
    )


def build_merged_cohort(
    datasets: List[DatasetConfig],
    expression_data: Dict[str, pd.DataFrame],
    clinical_data: Dict[str, pd.DataFrame],
    mutation_data: Dict[str, pd.DataFrame],
    output_dir: Path,
    label: str,
    immunotherapy_only: bool = False,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Build, standardise, and save a merged cohort.
    """
    print(
        f"\n{'=' * 60}"
    )

    print(
        f"Building {label.upper()} merged cohort"
    )

    print(
        f"{'=' * 60}"
    )

    # ------------------------------------------------------------------
    # Select samples
    # ------------------------------------------------------------------

    selected_expression = {}
    selected_clinical = {}
    selected_mutations = {}

    for dataset in datasets:
        cohort_name = dataset.cohort_name

        df_expr = expression_data[
            cohort_name
        ]

        df_clin = clinical_data[
            cohort_name
        ]

        df_mut = mutation_data[
            cohort_name
        ]

        if immunotherapy_only:
            mask = (
                df_clin[
                    "IMMUNOTHERAPY"
                ]
                == 1
            )

            df_expr = df_expr.loc[
                mask
            ]

            df_clin = df_clin.loc[
                mask
            ]

            if not df_mut.empty:
                df_mut = df_mut.loc[
                    df_mut.index.intersection(
                        df_expr.index
                    )
                ]

        selected_expression[
            cohort_name
        ] = df_expr

        selected_clinical[
            cohort_name
        ] = df_clin

        selected_mutations[
            cohort_name
        ] = df_mut

        print(
            f"  {cohort_name}: "
            f"{len(df_clin):,} samples"
        )

    # ------------------------------------------------------------------
    # Find common expression genes.
    # ------------------------------------------------------------------

    print(
        "\nFinding common expression genes..."
    )

    common_genes = find_common_genes(
        selected_expression
    )

    if not common_genes:
        raise ValueError(
            "No common expression genes found "
            "across selected datasets."
        )

    print(
        f"  Common genes: "
        f"{len(common_genes):,}"
    )

    # ------------------------------------------------------------------
    # Restrict and standardise each cohort.
    # ------------------------------------------------------------------

    print(
        "\nZ-score standardising expression "
        "within each cohort..."
    )

    scaled_expression = []

    clinical_frames = []

    for dataset in datasets:
        cohort_name = (
            dataset.cohort_name
        )

        df_expr = selected_expression[
            cohort_name
        ]

        df_clin = selected_clinical[
            cohort_name
        ]

        if df_expr.empty:
            continue

        df_expr = df_expr[
            common_genes
        ]

        df_expr = zscore_expression(
            df_expr
        )

        # Ensure identical ordering.
        df_expr = df_expr.loc[
            df_clin.index
        ]

        scaled_expression.append(
            df_expr
        )

        clinical_frames.append(
            df_clin
        )

    # ------------------------------------------------------------------
    # Concatenate cohorts.
    # ------------------------------------------------------------------

    df_expr_merged = pd.concat(
        scaled_expression,
        axis=0,
    )

    df_clin_merged = pd.concat(
        clinical_frames,
        axis=0,
    )

    if not (
        df_expr_merged.index
        == df_clin_merged.index
    ).all():
        raise ValueError(
            "Expression and clinical sample "
            "indices are not identical after merge."
        )

    # ------------------------------------------------------------------
    # Build genomic feature output.
    # ------------------------------------------------------------------

    df_genomic = _build_genomic_features(
        datasets=datasets,
        clinical_data=selected_clinical,
        mutation_data=selected_mutations,
    )

    # Only retain genomic rows present in this merged cohort.
    df_genomic = df_genomic[
        df_genomic["SAMPLE_ID"].isin(
            df_clin_merged.index
        )
    ]

    _save_merged_outputs(
        df_expr=df_expr_merged,
        df_clin=df_clin_merged,
        df_genomic=df_genomic,
        output_dir=output_dir,
        label=label,
    )

    return (
        df_expr_merged,
        df_clin_merged,
    )


# ============================================================================
# Main workflow
# ============================================================================


def main() -> None:
    """
    Run the configuration-driven dataset merge.
    """
    print(
        "=================================================="
    )

    print(
        "Merging Cleaned Melanoma Cohorts"
    )

    print(
        "==================================================\n"
    )

    datasets = load_dataset_config(
        CONFIG_PATH
    )

    print(
        f"Loaded {len(datasets)} dataset "
        f"configuration(s) from "
        f"{display_path(CONFIG_PATH)}:\n"
    )

    for dataset in datasets:
        print(
            f"  - {dataset.cohort_name} "
            f"({dataset.study_id})"
        )

    print()

    (
        expression_data,
        clinical_data,
        mutation_data,
    ) = _prepare_dataset_data(
        datasets
    )

    # ------------------------------------------------------------------
    # FULL MERGE
    # ------------------------------------------------------------------

    (
        df_expr_full,
        df_clin_full,
    ) = build_merged_cohort(
        datasets=datasets,
        expression_data=expression_data,
        clinical_data=clinical_data,
        mutation_data=mutation_data,
        output_dir=FULL_DIR,
        label="Full",
        immunotherapy_only=False,
    )

    # ------------------------------------------------------------------
    # IMMUNOTHERAPY-ONLY MERGE
    # ------------------------------------------------------------------

    (
        df_expr_immuno,
        df_clin_immuno,
    ) = build_merged_cohort(
        datasets=datasets,
        expression_data=expression_data,
        clinical_data=clinical_data,
        mutation_data=mutation_data,
        output_dir=IMMUNOTHERAPY_DIR,
        label="Immunotherapy",
        immunotherapy_only=True,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Merged Cohort Generation Completed Successfully"
    )

    print(
        "=" * 60
    )

    print(
        f"  Full merge: "
        f"{len(df_clin_full):,} samples"
    )

    print(
        f"  Immunotherapy merge: "
        f"{len(df_clin_immuno):,} samples"
    )

    print(
        f"\n  Full output: "
        f"{display_path(FULL_DIR)}"
    )

    print(
        f"  Immunotherapy output: "
        f"{display_path(IMMUNOTHERAPY_DIR)}"
    )


# ============================================================================
# Script entry point
# ============================================================================


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
            contextlib.redirect_stdout(
                stdout_tee
            ),
            contextlib.redirect_stderr(
                stderr_tee
            ),
        ):

            print(
                "Logging console output to "
                f"{display_path(LOG_PATH)}"
            )

            main()