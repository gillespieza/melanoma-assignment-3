"""Dataset loading utilities for Melanoma Cohorts."""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.config.datasets import DatasetConfig, load_dataset_config
from src.utils.paths import DATA_DIR


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
            f"Cleaned dataset files not found for cohort '{config.cohort_name}' in {proc_dir}"
        )

    df_expr = pd.read_csv(expr_path, index_col="SAMPLE_ID")
    df_clin = pd.read_csv(clin_path, index_col="SAMPLE_ID")

    common = df_expr.index.intersection(df_clin.index)
    df_expr = df_expr.loc[common]
    df_clin = df_clin.loc[common]

    df_clin["Cohort"] = config.cohort_name
    df_clin = _add_aliases(df_clin)

    return df_expr, df_clin


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
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], List[str], List[str]]:
    """Loads the pre-merged immunotherapy cohort dataset.

    Reads clinical metadata from data/processed/merged/immunotherapy/clin_merged.csv.
    If use_raw=True (default), loads true raw log2 expression matrices from each
    cohort's processed directory (liu_2019, hugo_2016, riaz_2017, gide_2019,
    van_allen_2015, skcm_tcga_gdc) matched to the immunotherapy subset sample IDs.
    If use_raw=False, reads from expr_merged.csv (pre-Z-score standardised).

    Args:
        data_dir: Base data directory path.
        use_raw: If True, load un-standardised raw expression data for each cohort.

    Returns:
        Tuple of (expr_dict, clin_dict, cohort_order, trial_names) keyed by cohort name.
    """
    merged_dir = data_dir / "processed" / "merged" / "immunotherapy"
    clin_path = merged_dir / "clin_merged.csv"

    if not clin_path.exists():
        raise FileNotFoundError(
            f"Pre-merged immunotherapy clinical file not found in {merged_dir}."
        )

    df_clin = pd.read_csv(clin_path, index_col="SAMPLE_ID")
    df_clin = _add_aliases(df_clin)

    cohort_dir_map = {
        "Liu 2019": "liu_2019",
        "Hugo 2016": "hugo_2016",
        "Riaz 2017": "riaz_2017",
        "Gide 2019": "gide_2019",
        "Van Allen 2015": "van_allen_2015",
        "TCGA GDC 2025": "skcm_tcga_gdc",
    }

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




def load_liu_2019(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads Liu 2019 dataset."""
    expr_file = Path(data_dir) / "processed/liu_2019/expr_cleaned.csv"
    clin_file = Path(data_dir) / "processed/liu_2019/clin_cleaned.csv"

    df_expr = pd.read_csv(expr_file, index_col="SAMPLE_ID")
    df_clin = pd.read_csv(clin_file, index_col="SAMPLE_ID")
    df_clin = _add_aliases(df_clin)
    return df_expr, df_clin


def load_hugo_2016(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads Hugo 2016 dataset."""
    expr_file = Path(data_dir) / "processed/hugo_2016/expr_cleaned.csv"
    clin_file = Path(data_dir) / "processed/hugo_2016/clin_cleaned.csv"

    df_expr = pd.read_csv(expr_file, index_col="SAMPLE_ID")
    df_clin = pd.read_csv(clin_file, index_col="SAMPLE_ID")
    df_clin = _add_aliases(df_clin)
    return df_expr, df_clin


def load_riaz_2017(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads Riaz 2017 dataset."""
    expr_file = Path(data_dir) / "processed/riaz_2017/expr_cleaned.csv"
    clin_file = Path(data_dir) / "processed/riaz_2017/clin_cleaned.csv"

    df_expr = pd.read_csv(expr_file, index_col="SAMPLE_ID")
    df_clin = pd.read_csv(clin_file, index_col="SAMPLE_ID")
    df_clin = _add_aliases(df_clin)
    return df_expr, df_clin
