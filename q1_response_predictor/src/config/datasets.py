"""Dataset configuration loading utilities for cBioPortal datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import yaml
from yaml import YAMLError

from src.config.constants import SUPPORTED_PROCESSING_STRATEGIES


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a dataset used by the Q1 pipeline.

    Attributes:
        cohort_name:
            Human-readable cohort name.

        study_id:
            Unique cBioPortal study identifier.

        raw_directory:
            Directory name under the shared ``data/raw`` directory.

        processed_directory:
            Directory name under the shared ``data/processed`` directory.

        expression_file:
            Expression matrix filename in the raw dataset directory.

        clinical_file:
            Patient-level clinical data filename.

        clinical_sample_file:
            Sample-level clinical data filename.

        mutation_file:
            Somatic mutation data filename.

        processing_strategy:
            Strategy for processing the dataset.

        baseline_only:
            Whether cleaning should restrict the dataset to baseline samples.

        mutations_by_patient:
            Whether mutation records should be aggregated to patient level
            rather than retained at sample level.

        patient_prefix:
            Prefix applied to patient identifiers to prevent collisions between
            cohorts.

        sample_prefix:
            Prefix applied to sample identifiers to prevent collisions between
            cohorts.

        merge_enabled:
            Whether this dataset should be included in merged cohort outputs.
            Set to False to include in cleaning/QC but exclude from merges
            (e.g. superseded datasets kept for archival reference).

        cohort_immunotherapy:
            Whether all samples in this cohort received immunotherapy.
            Used as the authoritative fallback when no IMMUNOTHERAPY or
            TX_TYPE_IMMUNOTHERAPY column is present in the cleaned clinical
            data. iAtlas cohorts set this implicitly; non-iAtlas cohorts
            that are pure immunotherapy trials (e.g. Van Allen 2015) must
            set this explicitly to True in datasets.yaml.

        treatment_label:
            Short human-readable description of the treatment regimen used
            in this cohort (e.g. "Anti-PD-1 (pembrolizumab/nivolumab)").
            Used in report text to accurately describe cohort treatment context.
            Defaults to an empty string when not specified.
    """

    cohort_name: str
    study_id: str
    raw_directory: str
    processed_directory: str
    expression_file: str
    clinical_file: str
    clinical_sample_file: str
    mutation_file: str
    processing_strategy: str

    baseline_only: bool = False
    mutations_by_patient: bool = False
    merge_enabled: bool = True
    cohort_immunotherapy: bool = False
    treatment_label: str = ""
    patient_prefix: str = ""
    sample_prefix: str = ""

    SUPPORTED_PROCESSING_STRATEGIES: ClassVar[
        frozenset[str]
    ] = frozenset(SUPPORTED_PROCESSING_STRATEGIES)


class DatasetConfigError(Exception):
    """Exception raised for dataset configuration errors."""


def load_dataset_config(
    config_path: Path,
) -> tuple[DatasetConfig, ...]:
    """Load and validate dataset definitions from a YAML file.

    Args:
        config_path:
            Path to the YAML configuration file.

    Returns:
        Tuple of validated ``DatasetConfig`` objects.

    Raises:
        FileNotFoundError:
            If the configuration file does not exist.

        DatasetConfigError:
            If the configuration is malformed or invalid.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

    except YAMLError as error:
        raise DatasetConfigError(
            f"Invalid YAML in dataset configuration file "
            f"'{config_path}': {error}"
        ) from error

    if not config:
        raise DatasetConfigError(
            f"Dataset configuration file '{config_path}' is empty."
        )

    if not isinstance(config, dict):
        raise DatasetConfigError(
            "Dataset configuration must be a dictionary, "
            f"got {type(config).__name__}."
        )

    datasets = config.get("datasets")

    if datasets is None:
        raise DatasetConfigError(
            "Dataset configuration must contain a 'datasets' section."
        )

    if not isinstance(datasets, list):
        raise DatasetConfigError(
            "The 'datasets' configuration must be a list, "
            f"got {type(datasets).__name__}."
        )

    if not datasets:
        raise DatasetConfigError(
            "The 'datasets' list is empty. "
            "At least one dataset must be specified."
        )

    required_fields = {
        "cohort_name",
        "study_id",
        "raw_directory",
        "processed_directory",
        "expression_file",
        "clinical_file",
        "clinical_sample_file",
        "mutation_file",
        "processing_strategy",
    }

    allowed_fields = {
        *required_fields,
        "baseline_only",
        "mutations_by_patient",
        "merge_enabled",
        "cohort_immunotherapy",
        "treatment_label",
        "patient_prefix",
        "sample_prefix",
    }

    configs: list[DatasetConfig] = []
    cohort_names: set[str] = set()

    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise DatasetConfigError(
                f"Dataset at index {index} must be a dictionary, "
                f"got {type(dataset).__name__}."
            )

        missing_fields = required_fields - dataset.keys()

        if missing_fields:
            raise DatasetConfigError(
                f"Dataset at index {index} is missing required field(s): "
                f"{', '.join(sorted(missing_fields))}."
            )

        unexpected_fields = set(dataset) - allowed_fields

        if unexpected_fields:
            raise DatasetConfigError(
                f"Dataset at index {index} has unexpected field(s): "
                f"{', '.join(sorted(unexpected_fields))}."
            )

        _validate_field_types(dataset, index)

        try:
            config = DatasetConfig(**dataset)

        except TypeError as error:
            raise DatasetConfigError(
                f"Dataset at index {index} has invalid fields: {error}"
            ) from error

        if config.cohort_name in cohort_names:
            raise DatasetConfigError(
                f"Duplicate cohort_name in dataset configuration: "
                f"{config.cohort_name!r}."
            )
        cohort_names.add(config.cohort_name)
        configs.append(config)

    return tuple(configs)


def _validate_field_types(
    dataset: dict,
    index: int,
) -> None:
    """Validate dataset configuration field types."""

    string_fields = {
        "cohort_name",
        "study_id",
        "raw_directory",
        "processed_directory",
        "expression_file",
        "clinical_file",
        "clinical_sample_file",
        "mutation_file",
        "patient_prefix",
        "sample_prefix",
        "processing_strategy",
        "treatment_label",
    }

    boolean_fields = {
        "baseline_only",
        "mutations_by_patient",
        "merge_enabled",
        "cohort_immunotherapy",
    }

    for field in string_fields:
        if field in dataset and not isinstance(dataset[field], str):
            raise DatasetConfigError(
                f"Dataset at index {index}: field '{field}' must be a string, "
                f"got {type(dataset[field]).__name__}."
            )

    for field in boolean_fields:
        if field in dataset and not isinstance(dataset[field], bool):
            raise DatasetConfigError(
                f"Dataset at index {index}: field '{field}' must be a boolean, "
                f"got {type(dataset[field]).__name__}."
            )

    processing_strategy = dataset.get("processing_strategy")

    if (
        isinstance(processing_strategy, str)
        and processing_strategy
        not in SUPPORTED_PROCESSING_STRATEGIES
    ):
        raise DatasetConfigError(
            f"Dataset at index {index}: field 'processing_strategy' "
            f"must be one of "
            f"{sorted(SUPPORTED_PROCESSING_STRATEGIES)}, "
            f"got {processing_strategy!r}."
        )


def load_dataset_config_or_empty(
    config_path: Path,
    default: tuple[DatasetConfig, ...] | None = None,
) -> tuple[DatasetConfig, ...]:
    """Load dataset configuration, returning a default if the file is absent.

    Args:
        config_path:
            Path to the YAML configuration file.

        default:
            Default tuple returned when the configuration file does not exist.
            Defaults to an empty tuple.

    Returns:
        Tuple of ``DatasetConfig`` objects, or the supplied default.
    """
    if default is None:
        default = ()

    try:
        return load_dataset_config(config_path)

    except FileNotFoundError:
        return default