"""Dataset loading utilities for Melanoma Cohorts.

The canonical entry points are:

- ``load_cohort_by_name`` — load any cohort by its human-readable name, resolved
  dynamically from ``config/datasets.yaml``.  New datasets added to the YAML are
  immediately available without touching this file.

- ``load_dataset_by_config`` — load a single cohort from an already-resolved
  ``DatasetConfig`` object.

- ``load_all_active_cohorts`` — batch-load every enabled cohort from the YAML.

- ``load_merged_immunotherapy`` — load the pre-merged immunotherapy file, resolving
  per-cohort expression paths through the YAML config rather than a hardcoded map.

Legacy per-cohort convenience functions (``load_liu_2019``, ``load_hugo_2016``, …)
are thin wrappers around ``load_cohort_by_name`` kept for backwards compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config.datasets import DatasetConfig, load_dataset_config
from src.utils.paths import DATA_DIR

# ---------------------------------------------------------------------------
# Default config path — resolved relative to this file so imports work
# regardless of the calling script's CWD.
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG_PATH: Path = (
    Path(__file__).resolve().parents[1] / "config" / "datasets.yaml"
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _add_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Adds standard lowercase alias columns to clinical DataFrame."""
    mapping = {
        "RESPONSE_BINARY": "response",
        "OS_MONTHS": "os_months",
        "OS_STATUS": "os_status",
        "AGE": "age",
        "SEX": "sex",
    }
    for orig, target in mapping.items():
        if orig in df.columns and target not in df.columns:
            df[target] = df[orig]
    return df


def _find_config_by_name(
    cohort_name: str,
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> DatasetConfig:
    """Returns the DatasetConfig matching *cohort_name* from the YAML file.

    Args:
        cohort_name: Human-readable cohort name (e.g. ``"Gide 2019"``).
        config_path: Path to ``datasets.yaml``.

    Raises:
        KeyError: If no config entry matches *cohort_name*.
    """
    all_configs = load_dataset_config(config_path)
    for cfg in all_configs:
        if cfg.cohort_name == cohort_name:
            return cfg
    available = ", ".join(f"'{c.cohort_name}'" for c in all_configs)
    raise KeyError(
        f"No dataset configuration found for cohort '{cohort_name}'. "
        f"Available cohorts: {available}. "
        "Add the cohort to config/datasets.yaml to enable loading."
    )


def _build_cohort_dir_map(config_path: Path = _DEFAULT_CONFIG_PATH) -> Dict[str, str]:
    """Builds a mapping of cohort_name -> processed_directory from the YAML config.

    This replaces any hardcoded ``cohort_dir_map`` dict and automatically picks
    up new cohorts added to ``datasets.yaml`` without code changes.
    """
    return {
        cfg.cohort_name: cfg.processed_directory
        for cfg in load_dataset_config(config_path)
    }


# ---------------------------------------------------------------------------
# Public core loaders
# ---------------------------------------------------------------------------


def load_dataset_by_config(
    config: DatasetConfig,
    data_dir: Path = DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and aligns expression and clinical DataFrames for a dataset configuration.

    Args:
        config: DatasetConfig instance from load_dataset_config.
        data_dir: Base data directory path.

    Returns:
        Tuple of (aligned expression DataFrame, aligned clinical DataFrame).
    """
    proc_dir = data_dir / "processed" / config.processed_directory
    expr_path = proc_dir / "expr_cleaned.csv"
    clin_path = proc_dir / "clin_cleaned.csv"

    if not expr_path.exists() or not clin_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset files not found for cohort '{config.cohort_name}' "
            f"in {proc_dir}. Run preprocessing script (clean_data.py) first."
        )

    df_expr = pd.read_csv(expr_path, index_col="SAMPLE_ID")
    df_clin = pd.read_csv(clin_path, index_col="SAMPLE_ID")

    common = df_expr.index.intersection(df_clin.index)
    df_expr = df_expr.loc[common]
    df_clin = df_clin.loc[common]

    df_clin["Cohort"] = config.cohort_name
    df_clin = _add_aliases(df_clin)

    return df_expr, df_clin


def load_cohort_by_name(
    cohort_name: str,
    data_dir: Path = DATA_DIR,
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads any cohort by its human-readable name, resolved from ``datasets.yaml``.

    This is the futureproof entry point: adding a new dataset to the YAML makes
    it immediately loadable here without touching this file.

    Args:
        cohort_name: Human-readable cohort name, e.g. ``"Gide 2019"``.
        data_dir: Base data directory path.
        config_path: Path to ``datasets.yaml``.

    Returns:
        Tuple of (expression DataFrame, clinical DataFrame).

    Raises:
        KeyError: If *cohort_name* is not found in the YAML configuration.
        FileNotFoundError: If the processed CSV files do not yet exist on disk.
    """
    config = _find_config_by_name(cohort_name, config_path)
    return load_dataset_by_config(config, data_dir)


def load_all_active_cohorts(
    config_path: Path,
    data_dir: Path = DATA_DIR,
    merge_only: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], List[str], List[str]]:
    """Loads all active cohorts defined in dataset configuration.

    Args:
        config_path: Path to datasets.yaml.
        data_dir: Base data directory.
        merge_only: Restrict to datasets with merge_enabled=True.

    Returns:
        Tuple of (expr_dict, clin_dict, cohort_order, trial_names).
    """
    all_configs = load_dataset_config(config_path)
    configs = (
        [c for c in all_configs if c.merge_enabled] if merge_only else list(all_configs)
    )

    expr_dict: Dict[str, pd.DataFrame] = {}
    clin_dict: Dict[str, pd.DataFrame] = {}
    cohort_order: List[str] = []
    trial_names: List[str] = []

    for config in configs:
        try:
            expr, clin = load_dataset_by_config(config, data_dir)
        except FileNotFoundError:
            continue

        if expr.empty or clin.empty:
            continue

        name = config.cohort_name
        expr_dict[name] = expr
        clin_dict[name] = clin
        cohort_order.append(name)

        is_trial = (
            config.cohort_immunotherapy
            or config.processing_strategy == "iatlas"
        )
        if is_trial:
            trial_names.append(name)

    return expr_dict, clin_dict, cohort_order, trial_names


def load_merged_immunotherapy(
    data_dir: Path = DATA_DIR,
    use_raw: bool = True,
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], List[str], List[str]]:
    """Loads the pre-merged immunotherapy cohort dataset.

    Reads clinical metadata from data/processed/merged/immunotherapy/clin_merged.csv.
    If use_raw=True (default), loads true raw log2 expression matrices from each
    cohort's processed directory (resolved from datasets.yaml) matched to the
    immunotherapy subset sample IDs.
    If use_raw=False, reads from expr_merged.csv (pre-Z-score standardised).

    Args:
        data_dir: Base data directory path.
        use_raw: If True, load un-standardised raw expression data for each cohort.
        config_path: Path to datasets.yaml (used to resolve processed_directory
            for each cohort dynamically — no hardcoded cohort_dir_map required).

    Returns:
        Tuple of (expr_dict, clin_dict, cohort_order, trial_names) keyed by cohort name.
    """
    merged_dir = data_dir / "processed" / "merged" / "immunotherapy"
    clin_path = merged_dir / "clin_merged.csv"

    if not clin_path.exists():
        raise FileNotFoundError(
            f"Pre-merged immunotherapy clinical file not found in {merged_dir}. "
            "Run merge_datasets.py first."
        )

    df_clin = pd.read_csv(clin_path, index_col="SAMPLE_ID")
    df_clin = _add_aliases(df_clin)

    # Resolve cohort -> processed_directory dynamically from YAML so that adding
    # a new cohort to datasets.yaml is sufficient — no code change needed here.
    cohort_dir_map = _build_cohort_dir_map(config_path)

    cohort_col = "COHORT" if "COHORT" in df_clin.columns else "Cohort"
    seen: set = set()
    cohort_order: List[str] = []
    for name in df_clin[cohort_col]:
        if name not in seen:
            seen.add(name)
            cohort_order.append(name)

    expr_dict: Dict[str, pd.DataFrame] = {}
    clin_dict: Dict[str, pd.DataFrame] = {}

    if use_raw:
        for name in cohort_order:
            mask = df_clin[cohort_col] == name
            clin_sub = df_clin.loc[mask].copy()
            clin_sub["Cohort"] = name

            dir_name = cohort_dir_map.get(name)
            if not dir_name:
                continue

            expr_path = data_dir / "processed" / dir_name / "expr_cleaned.csv"
            if not expr_path.exists():
                continue

            df_expr_cohort = pd.read_csv(expr_path, index_col="SAMPLE_ID")
            common = clin_sub.index.intersection(df_expr_cohort.index)

            expr_dict[name] = df_expr_cohort.loc[common]
            clin_dict[name] = clin_sub.loc[common]
    else:
        expr_path = merged_dir / "expr_merged.csv"
        if not expr_path.exists():
            raise FileNotFoundError(f"Pre-merged expr file not found at {expr_path}")

        df_expr = pd.read_csv(expr_path, index_col="SAMPLE_ID")
        common = df_expr.index.intersection(df_clin.index)
        df_expr = df_expr.loc[common]
        df_clin = df_clin.loc[common]

        for name in cohort_order:
            mask = df_clin[cohort_col] == name
            clin_sub = df_clin.loc[mask].copy()
            clin_sub["Cohort"] = name
            expr_dict[name] = df_expr.loc[mask]
            clin_dict[name] = clin_sub

    trial_names = [c for c in cohort_order if c in expr_dict]

    return expr_dict, clin_dict, cohort_order, trial_names


# ---------------------------------------------------------------------------
# Backwards-compatible per-cohort shim functions
#
# These are thin wrappers around load_cohort_by_name().  Do NOT add
# per-cohort shims for future datasets — use load_cohort_by_name() directly.
# ---------------------------------------------------------------------------


def load_liu_2019(
    data_dir: Path = DATA_DIR,
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backwards-compatible loader for Liu 2019. Delegates to load_cohort_by_name."""
    return load_cohort_by_name("Liu 2019", data_dir, config_path)


def load_hugo_2016(
    data_dir: Path = DATA_DIR,
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backwards-compatible loader for Hugo 2016. Delegates to load_cohort_by_name."""
    return load_cohort_by_name("Hugo 2016", data_dir, config_path)


def load_riaz_2017(
    data_dir: Path = DATA_DIR,
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backwards-compatible loader for Riaz 2017. Delegates to load_cohort_by_name."""
    return load_cohort_by_name("Riaz 2017", data_dir, config_path)


def load_gide_2019(
    data_dir: Path = DATA_DIR,
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backwards-compatible loader for Gide 2019. Delegates to load_cohort_by_name."""
    return load_cohort_by_name("Gide 2019", data_dir, config_path)


def load_van_allen_2015(
    data_dir: Path = DATA_DIR,
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backwards-compatible loader for Van Allen 2015. Delegates to load_cohort_by_name."""
    return load_cohort_by_name("Van Allen 2015", data_dir, config_path)


def load_skcm_tcga_gdc(
    data_dir: Path = DATA_DIR,
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backwards-compatible loader for TCGA GDC 2025. Delegates to load_cohort_by_name."""
    return load_cohort_by_name("TCGA GDC 2025", data_dir, config_path)



