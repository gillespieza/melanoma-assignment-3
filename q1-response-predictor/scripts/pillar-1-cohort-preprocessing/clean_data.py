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
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SUBPROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    PROCESSED_DIR,
    RAW_DIR,
    get_subproject_log_dir,
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

# Resolve config relative to script location.
CONFIG_PATH = SUBPROJECT_ROOT / "config" / "datasets.yaml"
LOG_DIR = get_subproject_log_dir(SCRIPT_DIR)
LOG_PATH = LOG_DIR / "clean_data.log"


# ============================================================================
# Module-level constants
# ============================================================================

# cBioPortal / MAF protocol column names.
_COL_PATIENT_ID: str = "PATIENT_ID"
_COL_SAMPLE_ID: str = "SAMPLE_ID"
_COL_HUGO_SYMBOL: str = "Hugo_Symbol"
_COL_ENTREZ_ID: str = "Entrez_Gene_Id"
_BASELINE_SUFFIX: str = "_PRE"

# Output filenames for processed datasets.
_CLINICAL_OUTPUT_FILENAME: str = "clin_cleaned.csv"
_EXPRESSION_OUTPUT_FILENAME: str = "expr_cleaned.csv"
_MUTATIONS_OUTPUT_FILENAME: str = "mutations_cleaned.csv"
_ATTRITION_OUTPUT_FILENAME: str = "attrition.csv"
_ENTREZ_CACHE_FILENAME: str = "entrez_to_symbol_cache.json"

# TCGA supplementary data filenames.
_TCGA_TIMELINE_FILENAME: str = "data_timeline_treatment.txt"
_TCGA_HYPOXIA_FILENAME: str = "data_clinical_supp_hypoxia.txt"
_COL_HYPOXIA_SCORE: str = "WINTER_HYPOXIA_SCORE"

# MAF / mutation processing column names.
_COL_VARIANT_CLASSIFICATION: str = "Variant_Classification"
_COL_HUGO_SYMBOL_MAPPED: str = "Hugo_Symbol_Mapped"

# TCGA clinical feature engineering column names.
_COL_TREATMENT_TYPE: str = "TREATMENT_TYPE"
_COL_AGENT: str = "AGENT"
_COL_START_DATE: str = "START_DATE"
_COL_STOP_DATE: str = "STOP_DATE"
_COL_AGE_AT_DIAGNOSIS: str = "AGE_AT_DIAGNOSIS"
_COL_RESPONSE: str = "RESPONSE"
_COL_SEX: str = "SEX"

# Processing strategy identifiers.
_STRATEGY_IATLAS: str = "iatlas"
_STRATEGY_TCGA: str = "tcga"

# Post-processing sanity check thresholds.
_EXPR_GENE_COUNT_WARN: int = 1_000
_EXPR_GENE_COUNT_INFO: int = 5_000
_EXPR_VALUE_CEILING: float = 50.0
_MAX_EXAMPLE_GENES: int = 20

# CLI banner line.
_BANNER_LINE: str = "=" * 50

# Reason strings for common attrition recording steps.
_REASON_CLINICAL_HARMONISED: str = (
    "Applied identifier standardisation, clinical cleaning, "
    "baseline filtering, and variable harmonisation."
)

# Key immunotherapy and targeted therapy agents tracked in TCGA treatment data.
_KEY_TREATMENT_AGENTS: dict[str, list[str]] = {
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

_OUTCOME_CANONICAL_MAP: dict[str, str] = {
    "COMPLETE RESPONSE": "Complete Response",
    "PARTIAL RESPONSE": "Partial Response",
    "STABLE DISEASE": "Stable Disease",
    "PROGRESSIVE DISEASE": "Progressive Disease",
    "CLINICAL PROGRESSIVE DISEASE": "Progressive Disease",
    "RADIOGRAPHIC PROGRESSIVE DISEASE": "Progressive Disease",
}

_RESPONSE_RANK: dict[str, int] = {
    "Complete Response": 4,
    "Partial Response": 3,
    "Stable Disease": 2,
    "Progressive Disease": 1,
}


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


@dataclass(frozen=True)
class CleanedDataBundle:
    """Grouped outputs from a dataset cleaning pipeline stage."""

    clinical_df: pd.DataFrame
    expression_df: pd.DataFrame
    mutation_df: pd.DataFrame
    attrition_records: list[AttritionRecord]


def _record_attrition(
    attrition_records: list[AttritionRecord],
    dataset: DatasetConfig,
    step: str,
    n_before: int,
    n_after: int,
    reason: str,
) -> None:
    """Record a sample attrition event for a dataset pipeline step."""
    attrition_records.append(
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


def _load_split_clinical_tables(
    patient_path: Path,
    sample_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load patient-level and sample-level cBioPortal clinical tables."""
    df_patient = read_cbioportal_table(patient_path, required_column=_COL_PATIENT_ID)
    df_sample = read_cbioportal_table(sample_path, required_column=_COL_SAMPLE_ID)

    if _COL_PATIENT_ID not in df_patient.columns:
        raise ValueError(f"{display_path(patient_path)} does not contain {_COL_PATIENT_ID}.")

    missing = {_COL_SAMPLE_ID, _COL_PATIENT_ID} - set(df_sample.columns)
    if missing:
        raise ValueError(f"{display_path(sample_path)} missing required columns: {sorted(missing)}")

    return df_patient, df_sample


def _load_clinical_data(
    raw_dir: Path,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """Load and merge patient- and sample-level clinical data."""
    patient_path = raw_dir / dataset.clinical_file
    sample_path = raw_dir / dataset.clinical_sample_file

    missing = [p for p in (patient_path, sample_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required clinical file(s): " + ", ".join(display_path(p) for p in missing)
        )

    df_patient, df_sample = _load_split_clinical_tables(patient_path, sample_path)
    return pd.merge(df_sample, df_patient, on=_COL_PATIENT_ID, how="inner")


def _apply_identifier_prefixes(
    df_clin: pd.DataFrame,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """
    Apply configured prefixes to patient and sample identifiers.

    Prefixes prevent identifier collisions when multiple cohorts are merged.
    """
    df = df_clin.copy()

    if _COL_PATIENT_ID in df.columns:
        df[_COL_PATIENT_ID] = (
            dataset.patient_prefix
            + df[_COL_PATIENT_ID].astype(str)
        ).str.upper()

    if _COL_SAMPLE_ID in df.columns:
        df[_COL_SAMPLE_ID] = (
            dataset.sample_prefix
            + df[_COL_SAMPLE_ID].astype(str)
        ).str.upper()

    return df


def _harmonise_clinical_data(
    df_clin: pd.DataFrame,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """Clean and harmonise clinical data for a configured dataset."""
    df = _apply_identifier_prefixes(df_clin, dataset)
    df = drop_empty_columns(clean_clinical_df(df))

    if dataset.baseline_only:
        df = df[df[_COL_SAMPLE_ID].str.endswith(_BASELINE_SUFFIX)]

    if _COL_RESPONSE not in df.columns and "DURABLE_CLINICAL_BENEFIT" in df.columns:
        recist_map = {
            "CR": "Complete Response",
            "PR": "Partial Response",
            "SD": "Stable Disease",
            "PD": "Progressive Disease",
        }
        df[_COL_RESPONSE] = df["DURABLE_CLINICAL_BENEFIT"].map(recist_map)

    if _COL_RESPONSE in df.columns:
        df["RESPONSE_BINARY"] = df[_COL_RESPONSE].map(RECIST_RESPONSE_MAP)
    else:
        df["RESPONSE_BINARY"] = np.nan

    if _COL_AGE_AT_DIAGNOSIS in df.columns:
        df["AGE"] = pd.to_numeric(df[_COL_AGE_AT_DIAGNOSIS], errors="coerce")

    if _COL_SEX in df.columns:
        sex_map = {"MALE": "Male", "FEMALE": "Female", "M": "Male", "F": "Female"}
        df[_COL_SEX] = df[_COL_SEX].astype(str).str.strip().str.upper().map(sex_map).fillna("N/A")

    return df.set_index(_COL_SAMPLE_ID)


# ============================================================================
# Expression processing
# ============================================================================


def _process_raw_expression_matrix(
    df_expr: pd.DataFrame,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """Clean, aggregate, transpose, and log2-transform expression matrix."""
    if _COL_HUGO_SYMBOL not in df_expr.columns:
        if _COL_ENTREZ_ID in df_expr.columns:
            cache_path = CONFIG_PATH.parent / _ENTREZ_CACHE_FILENAME
            df_expr = _map_tcga_entrez_identifiers(df_expr, cache_path)
            df_expr = df_expr.groupby(level=0).mean(numeric_only=True)
            df_expr = df_expr.T
            df_expr.index = df_expr.index.astype(str).str.strip().str.upper()
            df_expr.index.name = _COL_SAMPLE_ID
            return np.log2(df_expr + 1.0)
        raise ValueError("Expression file does not contain Hugo_Symbol or Entrez_Gene_Id.")

    df_expr = df_expr.dropna(subset=[_COL_HUGO_SYMBOL]).set_index(_COL_HUGO_SYMBOL)
    if _COL_ENTREZ_ID in df_expr.columns:
        df_expr = df_expr.drop(columns=[_COL_ENTREZ_ID])

    df_expr = df_expr.groupby(level=0).mean(numeric_only=True)
    df_expr.columns = (dataset.sample_prefix + df_expr.columns.astype(str)).str.upper()

    if dataset.baseline_only:
        df_expr = df_expr[[col for col in df_expr.columns if col.endswith(_BASELINE_SUFFIX)]]

    df_expr = df_expr.T
    df_expr.index.name = _COL_SAMPLE_ID
    df_expr = df_expr.apply(pd.to_numeric, errors="coerce")
    return np.log2(df_expr.clip(lower=0) + 1)


def _load_iatlas_expression(
    expression_path: Path,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """Load and prepare an iAtlas/cBioPortal expression matrix."""
    if not expression_path.exists():
        raise FileNotFoundError(f"Expression file not found: {display_path(expression_path)}")

    df_expr = pd.read_csv(expression_path, sep="\t")
    return _process_raw_expression_matrix(df_expr, dataset)


def _map_tcga_entrez_identifiers(
    df_expr: pd.DataFrame,
    cache_path: Path,
) -> pd.DataFrame:
    """Map TCGA Entrez gene IDs to Hugo gene symbols using local cache / Ensembl."""
    if _COL_ENTREZ_ID not in df_expr.columns:
        raise ValueError("TCGA expression matrix does not contain Entrez_Gene_Id.")

    df_expr = df_expr.dropna(subset=[_COL_ENTREZ_ID]).copy()
    df_expr[_COL_ENTREZ_ID] = (
        pd.to_numeric(df_expr[_COL_ENTREZ_ID], errors="coerce").astype("Int64").astype(str)
    )

    entrez_ids = df_expr[_COL_ENTREZ_ID].dropna().unique().tolist()
    gene_map = map_entrez_to_symbols(entrez_ids, cache_path=cache_path)

    df_expr[_COL_HUGO_SYMBOL_MAPPED] = (
        df_expr[_COL_ENTREZ_ID].map(gene_map).fillna(df_expr[_COL_ENTREZ_ID])
    )
    df_expr = df_expr.set_index(_COL_HUGO_SYMBOL_MAPPED)

    cols_to_drop = [c for c in (_COL_HUGO_SYMBOL, _COL_ENTREZ_ID) if c in df_expr.columns]
    return df_expr.drop(columns=cols_to_drop)


def _load_tcga_expression(
    expression_path: Path,
    dataset: DatasetConfig,
    cache_path: Path,
) -> pd.DataFrame:
    """
    Load and prepare the TCGA RSEM expression matrix.
    """
    if not expression_path.exists():
        raise FileNotFoundError(
            "Expression file not found: "
            f"{display_path(expression_path)}"
        )

    df_expr = pd.read_csv(expression_path, sep="\t")
    df_expr = _map_tcga_entrez_identifiers(df_expr, cache_path)

    # Duplicate mapped gene symbols are averaged.
    df_expr = df_expr.groupby(level=0).mean(numeric_only=True).T

    df_expr.index = (
        dataset.sample_prefix + df_expr.index.astype(str)
    ).map(standardise_sample_id)

    df_expr.index.name = _COL_SAMPLE_ID
    df_expr = df_expr.apply(pd.to_numeric, errors="coerce")

    return np.log2(df_expr.clip(lower=0) + 1)


# ============================================================================
# Mutation processing
# ============================================================================


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
        df_mut = _aggregate_mutations_to_patient_level(df_mut, clinical_df)
    else:
        df_mut = df_mut.reindex(clinical_df.index, fill_value=0)

    df_mut.index.name = _COL_SAMPLE_ID
    return df_mut


def _aggregate_mutations_to_patient_level(
    df_mut: pd.DataFrame,
    clinical_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate a sample-level mutation matrix to patient level then re-expand."""
    df_mut.index = df_mut.index.map(_sample_to_patient_id)
    df_mut = df_mut.groupby(level=0).max()

    patient_to_sample = (
        clinical_df[[_COL_PATIENT_ID]]
        .reset_index()
        .drop_duplicates(subset=[_COL_PATIENT_ID])
        .set_index(_COL_PATIENT_ID)
    )

    df_mut = df_mut.reindex(patient_to_sample.index, fill_value=0)
    df_mut = df_mut.join(patient_to_sample, how="right")
    df_mut = df_mut.set_index(patient_to_sample.loc[df_mut.index].index)

    return (
        clinical_df[[_COL_PATIENT_ID]]
        .join(df_mut, on=_COL_PATIENT_ID)
        .drop(columns=[_COL_PATIENT_ID])
        .fillna(0)
    )


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


def _extract_best_outcome(series: pd.Series) -> str | float:
    valid = [x for x in series.dropna() if x in _RESPONSE_RANK]
    if not valid:
        return np.nan
    return max(valid, key=lambda x: _RESPONSE_RANK[x])


def _build_treatment_summary_features(
    df_treatment: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate treatment timeline to patient-level summary features."""
    df_sum = df_treatment.groupby(_COL_PATIENT_ID).agg(
        TREATMENT_TYPES=(
            _COL_TREATMENT_TYPE,
            lambda values: ", ".join(
                sorted(set(values.dropna().astype(str)))
            ),
        ),
        TREATMENT_AGENTS=(
            _COL_AGENT,
            lambda values: ", ".join(
                sorted(set(values.dropna().astype(str)))
            ),
        ),
    )

    outcome_col = next(
        (c for c in ("TREATMENT_OUTCOME", "MEASURE_OF_RESPONSE") if c in df_treatment.columns),
        None,
    )
    if outcome_col:
        df_tx = df_treatment.copy()
        df_tx["_canon_outcome"] = (
            df_tx[outcome_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .map(_OUTCOME_CANONICAL_MAP)
        )
        overall = df_tx.groupby(_COL_PATIENT_ID)["_canon_outcome"].apply(_extract_best_outcome)
        df_sum["TREATMENT_OUTCOME"] = overall

        immuno_mask = df_tx[_COL_TREATMENT_TYPE].astype(str).str.upper().str.contains("IMMUNO", na=False)
        if immuno_mask.any():
            immuno = df_tx[immuno_mask].groupby(_COL_PATIENT_ID)["_canon_outcome"].apply(_extract_best_outcome)
            df_sum["TX_IMMUNOTHERAPY_OUTCOME"] = immuno

    return df_sum


def _build_treatment_type_indicators(
    df_treatment: pd.DataFrame,
    treatment_features: pd.DataFrame,
) -> pd.DataFrame:
    """Add binary indicator column per treatment type to treatment_features."""
    treatment_types = df_treatment[_COL_TREATMENT_TYPE].dropna().unique()

    for treatment_type in treatment_types:
        col_name = "TX_TYPE_" + str(treatment_type).replace(" ", "_").upper()
        patients = df_treatment.loc[
            df_treatment[_COL_TREATMENT_TYPE] == treatment_type, _COL_PATIENT_ID
        ].unique()

        treatment_features[col_name] = 0
        treatment_features.loc[treatment_features.index.isin(patients), col_name] = 1

    return treatment_features


def _build_key_agent_indicators(
    df_treatment: pd.DataFrame,
    treatment_features: pd.DataFrame,
) -> pd.DataFrame:
    """Add binary indicator columns for key immunotherapy agents."""
    for feature_name, agents in _KEY_TREATMENT_AGENTS.items():
        pattern = "|".join(agent.upper() for agent in agents)
        agent_col = df_treatment[_COL_AGENT].astype(str).str.upper()
        match_mask = agent_col.str.contains(pattern, na=False, regex=True)
        patients = df_treatment.loc[match_mask, _COL_PATIENT_ID].unique()
        treatment_features[feature_name] = (
            treatment_features.index.isin(patients).astype(int)
        )

    return treatment_features


def _add_tcga_treatment_features(
    clinical_df: pd.DataFrame,
    raw_dir: Path,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """Add aggregated treatment features from the TCGA treatment timeline."""
    timeline_path = raw_dir / _TCGA_TIMELINE_FILENAME
    if not timeline_path.exists():
        return clinical_df

    print("  Found treatment timeline data. Aggregating treatment features...")
    df_treatment = pd.read_csv(timeline_path, sep="\t", comment="#", low_memory=False)

    if "THERAPEUTIC_AGENT" in df_treatment.columns and _COL_AGENT not in df_treatment.columns:
        df_treatment = df_treatment.rename(columns={"THERAPEUTIC_AGENT": _COL_AGENT})

    required_cols = {_COL_PATIENT_ID, _COL_TREATMENT_TYPE, _COL_AGENT}
    if not required_cols.issubset(df_treatment.columns):
        return clinical_df

    for col in (_COL_START_DATE, _COL_STOP_DATE):
        if col in df_treatment.columns:
            df_treatment[col] = pd.to_numeric(df_treatment[col], errors="coerce")

    n_rows = len(df_treatment)
    n_patients = df_treatment[_COL_PATIENT_ID].nunique()
    n_types = df_treatment[_COL_TREATMENT_TYPE].nunique()
    print(f"    Loaded {n_rows:,} treatment events for {n_patients:,} patients "
          f"({n_types} treatment types).")

    # Apply the same patient prefix used on the clinical DataFrame so that the
    # merge key matches (e.g. TCGA GDC prefixes IDs with "TCGA_GDC_").
    if dataset.patient_prefix:
        df_treatment[_COL_PATIENT_ID] = (
            dataset.patient_prefix + df_treatment[_COL_PATIENT_ID].astype(str)
        ).str.upper()
    else:
        df_treatment[_COL_PATIENT_ID] = (
            df_treatment[_COL_PATIENT_ID].astype(str).str.upper()
        )

    print("    Building patient-level treatment summary...")
    tx_features = _build_treatment_summary_features(df_treatment)
    print(f"    Building treatment type indicator columns ({n_types} types)...")
    tx_features = _build_treatment_type_indicators(df_treatment, tx_features)
    n_agents = len(_KEY_TREATMENT_AGENTS)
    print(f"    Building key agent indicator columns ({n_agents} agents)...")
    tx_features = _build_key_agent_indicators(df_treatment, tx_features)
    tx_features = tx_features.reset_index()
    print(f"    Merging {tx_features.shape[1] - 1} treatment features onto clinical data...")

    merged = clinical_df.merge(tx_features, on=_COL_PATIENT_ID, how="left")
    if _COL_RESPONSE not in merged.columns:
        if "TX_IMMUNOTHERAPY_OUTCOME" in merged.columns and "TREATMENT_OUTCOME" in merged.columns:
            merged[_COL_RESPONSE] = merged["TX_IMMUNOTHERAPY_OUTCOME"].fillna(merged["TREATMENT_OUTCOME"])
        elif "TX_IMMUNOTHERAPY_OUTCOME" in merged.columns:
            merged[_COL_RESPONSE] = merged["TX_IMMUNOTHERAPY_OUTCOME"]
        elif "TREATMENT_OUTCOME" in merged.columns:
            merged[_COL_RESPONSE] = merged["TREATMENT_OUTCOME"]
    return merged


def _add_tcga_hypoxia_features(
    clinical_df: pd.DataFrame,
    raw_dir: Path,
    dataset: DatasetConfig,
) -> pd.DataFrame:
    """Add supplementary TCGA hypoxia data when available."""
    hypoxia_path = raw_dir / _TCGA_HYPOXIA_FILENAME
    if not hypoxia_path.exists():
        return clinical_df

    print("  Found supplementary hypoxia data. Merging hypoxia features...")
    df_hypoxia = pd.read_csv(hypoxia_path, sep="\t", comment="#")

    required_columns = {_COL_PATIENT_ID, _COL_HYPOXIA_SCORE}
    if not required_columns.issubset(df_hypoxia.columns):
        print("  [WARNING] Hypoxia file missing required columns. Skipping.")
        return clinical_df

    # Apply the same patient prefix used on the clinical DataFrame so that
    # the join key matches (mirrors fix applied to treatment timeline join).
    if dataset.patient_prefix:
        df_hypoxia[_COL_PATIENT_ID] = (
            dataset.patient_prefix + df_hypoxia[_COL_PATIENT_ID].astype(str).str.strip()
        ).str.upper()
    else:
        df_hypoxia[_COL_PATIENT_ID] = (
            df_hypoxia[_COL_PATIENT_ID].astype(str).str.strip().str.upper()
        )

    return clinical_df.merge(
        df_hypoxia[[_COL_PATIENT_ID, _COL_HYPOXIA_SCORE]],
        on=_COL_PATIENT_ID,
        how="left",
    )


# ============================================================================
# Dataset processing
# ============================================================================


def _process_iatlas_clinical_stage(
    raw_dir: Path,
    dataset: DatasetConfig,
    attrition: list[AttritionRecord],
) -> pd.DataFrame:
    """Load, record attrition, and harmonise iAtlas clinical data."""
    df_raw = _load_clinical_data(raw_dir, dataset)
    n_raw = len(df_raw)
    _record_attrition(
        attrition, dataset, step="Clinical data loaded",
        n_before=n_raw, n_after=n_raw,
        reason="Merged patient-level and sample-level clinical records.",
    )
    df_clean = _harmonise_clinical_data(df_raw, dataset)
    _record_attrition(
        attrition, dataset, step="Clinical data harmonised",
        n_before=n_raw, n_after=len(df_clean),
        reason=_REASON_CLINICAL_HARMONISED,
    )
    return df_clean


_DURABLE_BENEFIT_RECIST_MAP: dict[str, str] = {
    "CR": "Complete Response",
    "PR": "Partial Response",
    "SD": "Stable Disease",
    "PD": "Progressive Disease",
}


def _map_durable_benefit_to_response(df: pd.DataFrame) -> pd.DataFrame:
    """Map DURABLE_CLINICAL_BENEFIT abbreviations to canonical RESPONSE/RESPONSE_BINARY.

    Used for datasets (e.g. Van Allen 2015) that store response as CR/PR/SD/PD
    rather than the iAtlas full-text RESPONSE column.  Skipped silently when
    RESPONSE is already present or DURABLE_CLINICAL_BENEFIT is absent.
    """
    if _COL_RESPONSE not in df.columns and "DURABLE_CLINICAL_BENEFIT" in df.columns:
        df = df.copy()
        df[_COL_RESPONSE] = df["DURABLE_CLINICAL_BENEFIT"].map(_DURABLE_BENEFIT_RECIST_MAP)

    if _COL_RESPONSE in df.columns and "RESPONSE_BINARY" not in df.columns:
        df = df.copy() if not df.columns.duplicated().any() else df
        df["RESPONSE_BINARY"] = df[_COL_RESPONSE].map(RECIST_RESPONSE_MAP)

    return df


def _process_tcga_clinical_stage(
    raw_dir: Path,
    dataset: DatasetConfig,
    attrition: list[AttritionRecord],
) -> pd.DataFrame:
    """Load, record attrition, clean, and engineer TCGA clinical data."""
    df_raw = _load_clinical_data(raw_dir, dataset)
    n_raw = len(df_raw)
    _record_attrition(
        attrition, dataset, step="Clinical data loaded",
        n_before=n_raw, n_after=n_raw,
        reason="Merged patient-level and sample-level clinical records.",
    )
    df_clean = _apply_identifier_prefixes(df_raw, dataset)
    df_clean = clean_clinical_df(df_clean)
    _record_attrition(
        attrition, dataset, step="Clinical data harmonised",
        n_before=n_raw, n_after=len(df_clean),
        reason="Applied identifier standardisation and clinical data cleaning.",
    )
    df_clean = _add_tcga_treatment_features(df_clean, raw_dir, dataset)
    df_clean = _add_tcga_hypoxia_features(df_clean, raw_dir, dataset)
    df_clean = _map_durable_benefit_to_response(df_clean)
    return df_clean.set_index(_COL_SAMPLE_ID)


def _align_and_record_expression(
    expression_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    dataset: DatasetConfig,
    attrition: list[AttritionRecord],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter expression matrix to clinical samples and record expression availability."""
    common_samples = expression_df.index.intersection(clinical_df.index)
    aligned_expr = expression_df.loc[common_samples].copy()
    aligned_expr.index.name = _COL_SAMPLE_ID
    _record_attrition(
        attrition, dataset, step="Expression data availability",
        n_before=len(clinical_df), n_after=len(common_samples),
        reason="Identified samples with matching RNA-seq gene expression data.",
    )
    return aligned_expr, clinical_df



def process_iatlas_dataset(
    dataset: DatasetConfig,
    raw_dir: Path,
    processed_dir: Path,
) -> list[AttritionRecord]:
    """Process an iAtlas/cBioPortal immunotherapy cohort."""
    print(f"Cleaning {dataset.cohort_name} ({dataset.study_id})...")
    attrition: list[AttritionRecord] = []
    clinical_df = _process_iatlas_clinical_stage(raw_dir, dataset, attrition)
    expression_df = _load_iatlas_expression(raw_dir / dataset.expression_file, dataset)
    expression_df, clinical_df = _align_and_record_expression(
        expression_df, clinical_df, dataset, attrition
    )
    mutation_df = _process_mutations(raw_dir, clinical_df, dataset)
    bundle = CleanedDataBundle(clinical_df, expression_df, mutation_df, attrition)
    return _finalise_dataset(dataset, bundle, raw_dir, processed_dir)


def _resolve_entrez_cache_path(processed_dir: Path) -> Path:
    """Resolve Entrez-to-symbol cache path, preferring config/ over processed_dir/."""
    config_cache = CONFIG_PATH.parent / _ENTREZ_CACHE_FILENAME
    if config_cache.exists():
        return config_cache
    return processed_dir / _ENTREZ_CACHE_FILENAME


def process_tcga_dataset(
    dataset: DatasetConfig,
    raw_dir: Path,
    processed_dir: Path,
) -> list[AttritionRecord]:
    """Process the TCGA-SKCM dataset."""
    print(f"Cleaning {dataset.cohort_name} ({dataset.study_id})...")
    attrition: list[AttritionRecord] = []
    clinical_df = _process_tcga_clinical_stage(raw_dir, dataset, attrition)
    cache_path = _resolve_entrez_cache_path(processed_dir)
    expression_df = _load_tcga_expression(
        raw_dir / dataset.expression_file, dataset, cache_path
    )
    expression_df, clinical_df = _align_and_record_expression(
        expression_df, clinical_df, dataset, attrition
    )
    mutation_df = _process_mutations(raw_dir, clinical_df, dataset)
    bundle = CleanedDataBundle(clinical_df, expression_df, mutation_df, attrition)
    return _finalise_dataset(dataset, bundle, raw_dir, processed_dir)



def process_dataset(
    dataset: DatasetConfig,
) -> list[AttritionRecord]:
    """Process one configured dataset using its processing strategy."""
    raw_dir = RAW_DIR / dataset.raw_directory
    processed_dir = PROCESSED_DIR / dataset.processed_directory

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {display_path(raw_dir)}")

    if dataset.processing_strategy == _STRATEGY_IATLAS:
        return process_iatlas_dataset(dataset, raw_dir, processed_dir)

    if dataset.processing_strategy == _STRATEGY_TCGA:
        return process_tcga_dataset(dataset, raw_dir, processed_dir)

    raise ValueError(f"Unsupported processing strategy: {dataset.processing_strategy!r}")


# ============================================================================
# Output handling
# ============================================================================


def _finalise_dataset(
    dataset: DatasetConfig,
    bundle: CleanedDataBundle,
    raw_dir: Path,
    processed_dir: Path,
) -> list[AttritionRecord]:
    """Write outputs, print a summary, run sanity checks, and return records."""
    _write_processed_outputs(
        processed_dir, bundle.clinical_df, bundle.expression_df, bundle.mutation_df,
    )
    print(f"  {dataset.cohort_name}: Cleaned {len(bundle.clinical_df):,} samples.")
    _run_sanity_checks(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        dataset=dataset,
        bundle=bundle,
    )
    return bundle.attrition_records


def _write_processed_outputs(
    processed_dir: Path,
    clinical_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    mutation_df: pd.DataFrame,
) -> None:
    """Write cleaned clinical, expression, and mutation datasets."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    clinical_output = processed_dir / _CLINICAL_OUTPUT_FILENAME
    expression_output = processed_dir / _EXPRESSION_OUTPUT_FILENAME
    mutation_output = processed_dir / _MUTATIONS_OUTPUT_FILENAME

    clinical_df.reset_index().to_csv(clinical_output, index=False)
    expression_df.to_csv(expression_output)
    mutation_df.to_csv(mutation_output)

    print("  Wrote processed outputs:")
    print(f"    {display_path(clinical_output)}")
    print(f"    {display_path(expression_output)}")
    print(f"    {display_path(mutation_output)}")


def _write_attrition_output(
    processed_dir: Path,
    attrition_records: list[AttritionRecord],
) -> None:
    """Write attrition tracking table for a processed dataset."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    attrition_output = processed_dir / _ATTRITION_OUTPUT_FILENAME

    attrition_df = pd.DataFrame([r.to_dict() for r in attrition_records])
    attrition_df.to_csv(attrition_output, index=False)

    print("  Wrote attrition output:")
    print(f"    {display_path(attrition_output)}")
    print("-------------------------------------\n")


# ============================================================================
# Post-processing sanity checks
# ============================================================================


def _validate_raw_maf_columns(
    df_raw: pd.DataFrame,
) -> bool:
    """Return True if df_raw contains required MAF columns; print warning otherwise."""
    required_cols = {_COL_HUGO_SYMBOL, _COL_VARIANT_CLASSIFICATION}
    if required_cols.issubset(df_raw.columns):
        return True
    missing = required_cols - set(df_raw.columns)
    print(
        "  [WARNING] Raw mutation file is missing required columns: "
        + ", ".join(sorted(missing))
    )
    return False


def _extract_raw_non_synonymous_genes(
    mutation_path: Path,
) -> set[str] | None:
    """Read raw MAF and extract set of unique non-synonymous gene symbols."""
    try:
        df_raw = pd.read_csv(
            mutation_path, sep="\t", comment="#", low_memory=False,
        )
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        print(f"  [WARNING] Could not read raw mutation file: {error}")
        return None

    if not _validate_raw_maf_columns(df_raw):
        return None

    df_non_silent = df_raw[
        df_raw[_COL_VARIANT_CLASSIFICATION].isin(NON_SILENT_VARIANT_CLASSIFICATIONS)
    ]
    return set(
        df_non_silent[_COL_HUGO_SYMBOL].dropna().astype(str).str.strip()
    )


def _print_missing_genes_warning(missing: set[str]) -> None:
    """Print a warning when raw genes are absent from the cleaned matrix."""
    print(
        "  [WARNING] Some raw non-synonymous mutation genes "
        "were not retained in the cleaned matrix."
    )
    print(
        "    Example missing genes: "
        + ", ".join(sorted(missing)[:_MAX_EXAMPLE_GENES])
    )


def _print_extra_genes_warning(extra: set[str]) -> None:
    """Print a warning when the cleaned matrix contains genes absent from raw data."""
    print(
        "  [WARNING] Cleaned mutation matrix contains "
        "genes not found in the raw non-synonymous mutation set."
    )
    print(
        "    Example extra genes: "
        + ", ".join(sorted(extra)[:_MAX_EXAMPLE_GENES])
    )


def _report_gene_comparison(
    raw_genes: set[str],
    cleaned_genes: set[str],
) -> None:
    """Print detailed comparison metrics between raw and cleaned mutation gene sets."""
    retained = raw_genes & cleaned_genes
    missing = raw_genes - cleaned_genes
    extra = cleaned_genes - raw_genes

    print(f"    Raw non-synonymous genes: {len(raw_genes):,}")
    print(f"    Cleaned mutation genes:   {len(cleaned_genes):,}")
    print(f"    Genes retained:           {len(retained):,}")
    print(f"    Genes missing from clean: {len(missing):,}")
    print(f"    Extra cleaned genes:      {len(extra):,}")

    if missing:
        _print_missing_genes_warning(missing)
    elif raw_genes == cleaned_genes:
        print(
            "  [PASS] All raw non-synonymous mutation genes "
            "are represented in the cleaned matrix."
        )

    if extra:
        _print_extra_genes_warning(extra)


def _check_raw_mutation_gene_count(
    raw_dir: Path,
    dataset: DatasetConfig,
    mutation_df: pd.DataFrame,
) -> None:
    """
    Compare unique non-synonymous genes in raw mutation file with cleaned matrix.
    """
    mutation_path = raw_dir / dataset.mutation_file

    if not mutation_path.exists():
        print(
            "  [INFO] Raw mutation file not found. "
            "Skipping raw-versus-cleaned mutation comparison."
        )
        return

    print("\n  Mutation gene-count comparison:")

    raw_genes = _extract_raw_non_synonymous_genes(mutation_path)

    if raw_genes is None:
        return

    cleaned_genes = set(
        mutation_df.columns.astype(str).str.strip()
    )

    _report_gene_comparison(raw_genes, cleaned_genes)

def _check_output_files_exist(
    required_files: dict[str, Path],
) -> bool:
    """
    Verify all expected output files were written.

    Returns:
        ``True`` if all files exist; ``False`` otherwise (prints failure).
    """
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
        return False

    return True


def _check_clinical_integrity(
    clinical_df: pd.DataFrame,
) -> None:
    """Check clinical output for duplicate sample IDs and fully-empty columns."""
    if _COL_SAMPLE_ID not in clinical_df.columns:
        print("  [FAIL] Clinical data has no SAMPLE_ID column.")
        return

    duplicate_ids = clinical_df[_COL_SAMPLE_ID].duplicated().sum()

    if duplicate_ids:
        print(
            "  [WARNING] Duplicate clinical sample IDs: "
            f"{duplicate_ids:,}"
        )
    else:
        print("  [PASS] Clinical sample IDs are unique.")

    empty_columns = (
        clinical_df.columns[clinical_df.isna().all()].tolist()
    )

    if empty_columns:
        print("  [WARNING] Completely empty clinical columns:")
        for column in empty_columns:
            print(f"    - {column}")
    else:
        print("  [PASS] No completely empty clinical columns.")


def _check_expression_integrity(
    expression_df: pd.DataFrame,
) -> None:
    """Check expression matrix for duplicate sample IDs, NaN values, and gene count."""
    duplicate_ids = expression_df.index.duplicated().sum()
    if duplicate_ids:
        print(f"  [WARNING] Duplicate expression sample IDs: {duplicate_ids:,}")
    else:
        print("  [PASS] Expression sample IDs are unique.")

    nan_count = expression_df.isna().sum().sum()
    if nan_count:
        print(f"  [WARNING] Expression matrix contains {nan_count:,} NaN values.")
    else:
        print("  [PASS] Expression matrix contains no NaN values.")

    n_genes = len(expression_df.columns)
    if n_genes < _EXPR_GENE_COUNT_WARN:
        print(f"  [WARNING] Expression matrix contains only {n_genes:,} genes.")
    elif n_genes < _EXPR_GENE_COUNT_INFO:
        print(f"  [INFO] Expression matrix contains {n_genes:,} genes.")
    else:
        print(f"  [PASS] Expression matrix contains {n_genes:,} genes.")


def _check_expression_value_range(
    expression_df: pd.DataFrame,
) -> None:
    """Check that expression values fall within the expected log2(TPM+1) range."""
    expression_values = (
        expression_df.select_dtypes(include="number").to_numpy()
    )

    if not expression_values.size:
        return

    finite_values = expression_values[np.isfinite(expression_values)]

    if not finite_values.size:
        return

    print("\n  Expression value range:")
    print(f"    Minimum: {finite_values.min():.4f}")
    print(f"    Maximum: {finite_values.max():.4f}")

    if finite_values.min() < 0:
        print("  [WARNING] Negative expression values detected.")

    if finite_values.max() > _EXPR_VALUE_CEILING:
        print("  [WARNING] Extremely large expression values detected.")


def _check_mutation_integrity(
    mutation_df: pd.DataFrame,
) -> None:
    """Check that the mutation matrix is non-empty."""
    if mutation_df.empty:
        print("  [WARNING] Mutation matrix is empty.")
    else:
        print(
            "  [PASS] Mutation matrix contains "
            f"{len(mutation_df.columns):,} genes."
        )


def _check_sample_overlap(
    clinical_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    mutation_df: pd.DataFrame,
) -> None:
    """Report sample identifier overlap across the three output matrices."""
    if _COL_SAMPLE_ID not in clinical_df.columns:
        return

    clinical_ids = set(clinical_df[_COL_SAMPLE_ID].astype(str))
    expression_ids = set(expression_df.index.astype(str))
    mutation_ids = set(mutation_df.index.astype(str))

    print("\n  Sample identifier overlap:")
    print(f"    Clinical <-> Expression: {len(clinical_ids & expression_ids):,}")
    print(f"    Clinical <-> Mutation:   {len(clinical_ids & mutation_ids):,}")


def _print_dataset_dimensions(
    clinical_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    mutation_df: pd.DataFrame,
) -> None:
    """Print sample and feature dimensions of cleaned datasets."""
    print("\n  Dataset dimensions:")
    n_clin, c_clin = len(clinical_df), len(clinical_df.columns)
    n_expr, c_expr = len(expression_df), len(expression_df.columns)
    n_mut, c_mut = len(mutation_df), len(mutation_df.columns)
    print(f"    Clinical:   {n_clin:,} samples \u00d7 {c_clin:,} columns")
    print(f"    Expression: {n_expr:,} samples \u00d7 {c_expr:,} genes")
    print(f"    Mutations:  {n_mut:,} samples \u00d7 {c_mut:,} genes")


def _check_tcga_gene_count_plausibility(
    expression_df: pd.DataFrame,
) -> None:
    """Verify that TCGA expression matrix contains expected gene scale."""
    n_genes = len(expression_df.columns)
    if n_genes < _EXPR_GENE_COUNT_WARN:
        print(f"\n  [WARNING] TCGA expression matrix has only {n_genes:,} genes.")
        print("  Expected a substantially larger gene expression matrix.")
    else:
        print(f"\n  [PASS] TCGA expression gene count appears plausible: {n_genes:,}.")


def _evaluate_dataset_integrities(
    raw_dir: Path,
    dataset: DatasetConfig,
    clinical_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    mutation_df: pd.DataFrame,
) -> None:
    """Execute integrity assertion checks against in-memory cleaned DataFrames."""
    _print_dataset_dimensions(clinical_df, expression_df, mutation_df)
    _check_clinical_integrity(clinical_df)
    _check_expression_integrity(expression_df)
    _check_expression_value_range(expression_df)
    _check_mutation_integrity(mutation_df)
    _check_raw_mutation_gene_count(raw_dir, dataset, mutation_df)
    _check_sample_overlap(clinical_df, expression_df, mutation_df)

    if dataset.processing_strategy == _STRATEGY_TCGA:
        _check_tcga_gene_count_plausibility(expression_df)


def _run_sanity_checks(
    raw_dir: Path,
    processed_dir: Path,
    dataset: DatasetConfig,
    bundle: CleanedDataBundle,
) -> None:
    """Run post-processing sanity checks using in-memory bundle."""
    print("\n  Running post-processing sanity checks...")

    required_files = {
        "clinical": processed_dir / _CLINICAL_OUTPUT_FILENAME,
        "expression": processed_dir / _EXPRESSION_OUTPUT_FILENAME,
        "mutation": processed_dir / _MUTATIONS_OUTPUT_FILENAME,
    }

    if _check_output_files_exist(required_files):
        # Use in-memory DataFrames — avoids re-reading multi-hundred MB expression
        # and mutation CSVs that were just written to disk.
        clinical_df = bundle.clinical_df.reset_index()
        _evaluate_dataset_integrities(
            raw_dir, dataset,
            clinical_df, bundle.expression_df, bundle.mutation_df,
        )
        print("\n  Sanity checks complete.")

# ============================================================================
# Main workflow
# ============================================================================


def _run_cleaning_pipeline(
    datasets: list[DatasetConfig],
) -> int:
    """
    Execute data cleaning strategy for each configured dataset.

    Returns:
        Number of successfully cleaned datasets.
    """
    success_count = 0

    for dataset in datasets:
        try:
            attrition_records = process_dataset(dataset)
            processed_dir = PROCESSED_DIR / dataset.processed_directory
            _write_attrition_output(processed_dir, attrition_records)
            success_count += 1

        except Exception as error:
            print(f"\n[ERROR] Dataset '{dataset.cohort_name}' failed: {error}")
            print(traceback.format_exc())

    return success_count


def main() -> None:
    """
    Run the configured data-cleaning workflow.
    """
    print(_BANNER_LINE)
    print("Data Cleaning Pipeline: Transforming Raw Data")
    print(f"{_BANNER_LINE}\n")

    datasets = load_dataset_config(CONFIG_PATH)

    print(
        f"Loaded {len(datasets)} dataset configuration(s) "
        f"from {display_path(CONFIG_PATH)}.\n"
    )

    success_count = _run_cleaning_pipeline(datasets)

    if success_count > 0:
        try:
            print("\n  Updating config/data_dictionary.json...")
            from scripts.pillar_1_cohort_preprocessing.generate_data_dictionary import (
                OUTPUT_JSON,
                build_full_data_dictionary,
            )
            dictionary = build_full_data_dictionary(PROCESSED_DIR)
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                import json
                json.dump(dictionary, f, indent=2)
            print(f"  Data dictionary updated ({len(dictionary['columns']):,} features mapped).")
        except Exception as err:
            print(f"  [WARNING] Could not update data dictionary: {err}")

    print(f"\n{_BANNER_LINE}")
    print(
        f"Workflow completed: "
        f"{success_count}/{len(datasets)} datasets succeeded."
    )
    print(_BANNER_LINE)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

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