"""Dataset configuration loading utilities for cBioPortal DataHub datasets."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from yaml import YAMLError


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a dataset downloaded from cBioPortal DataHub.

    Attributes:
        cohort_name: Human-readable name for the cohort.
        study_id: Unique study identifier used by cBioPortal.
        raw_directory: Local directory path where raw data will be stored.
    """

    cohort_name: str
    study_id: str
    raw_directory: str


class DatasetConfigError(Exception):
    """Exception raised for dataset configuration errors."""


def load_dataset_config(
    config_path: Path,
) -> tuple[DatasetConfig, ...]:
    """Load dataset definitions from a YAML configuration file.

    Expected YAML format:

    ```yaml
    datasets:
      - cohort_name: "Example Cohort"
        study_id: "study_id_001"
        raw_directory: "data/raw/example"
      - cohort_name: "Another Cohort"
        study_id: "study_id_002"
        raw_directory: "data/raw/another"
    ```

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Tuple of DatasetConfig objects.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        DatasetConfigError: If the configuration is malformed or missing
            required fields.
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
            f"got {type(config).__name__}"
        )

    datasets = config.get("datasets")

    if datasets is None:
        raise DatasetConfigError(
            "Dataset configuration must contain a 'datasets' section."
        )

    if not isinstance(datasets, list):
        raise DatasetConfigError(
            "The 'datasets' configuration must be a list, "
            f"got {type(datasets).__name__}"
        )

    if not datasets:
        raise DatasetConfigError(
            "The 'datasets' list is empty. "
            "At least one dataset must be specified."
        )

    configs: list[DatasetConfig] = []

    required_fields = {
        "cohort_name",
        "study_id",
        "raw_directory",
    }

    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise DatasetConfigError(
                f"Dataset at index {index} must be a dictionary, "
                f"got {type(dataset).__name__}"
            )

        missing_fields = required_fields - dataset.keys()

        if missing_fields:
            raise DatasetConfigError(
                f"Dataset at index {index} is missing required field(s): "
                f"{', '.join(sorted(missing_fields))}"
            )

        unexpected_fields = set(dataset) - required_fields

        if unexpected_fields:
            raise DatasetConfigError(
                f"Dataset at index {index} has unexpected field(s): "
                f"{', '.join(sorted(unexpected_fields))}"
            )

        try:
            configs.append(DatasetConfig(**dataset))
        except TypeError as error:
            raise DatasetConfigError(
                f"Dataset at index {index} has invalid fields: {error}"
            ) from error

    return tuple(configs)


def load_dataset_config_or_empty(
    config_path: Path,
    default: tuple[DatasetConfig, ...] | None = None,
) -> tuple[DatasetConfig, ...]:
    """Load dataset configuration, returning a default if the file doesn't exist.

    This is a convenience wrapper around load_dataset_config that handles
    FileNotFoundError gracefully.

    Args:
        config_path: Path to the YAML configuration file.
        default: Default tuple to return if the file does not exist.
            Defaults to an empty tuple.

    Returns:
        Tuple of DatasetConfig objects, or default if the file is not found.

    Raises:
        DatasetConfigError: If the configuration is malformed.
    """
    if default is None:
        default = ()

    try:
        return load_dataset_config(config_path)
    except FileNotFoundError:
        return default