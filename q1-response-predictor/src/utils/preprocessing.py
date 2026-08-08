"""Reusable data-cleaning and preprocessing utilities.

This module contains shared transformations used by the melanoma response
prediction pipeline, including:

- sample and patient identifier standardisation;
- survival endpoint normalisation;
- clinical dataframe cleaning;
- RNA-seq dataframe cleaning;
- expression/clinical sample alignment;
- Entrez Gene ID to HGNC symbol mapping.

The functions are intentionally dataset-agnostic where possible. Dataset-specific
decisions, such as cohort-specific column mappings or treatment filtering, should
remain in the relevant pipeline scripts.
"""

from __future__ import annotations

import json
import time
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

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
# Constants
# ---------------------------------------------------------------------------
from src.config.constants import NON_SILENT_VARIANT_CLASSIFICATIONS

SURVIVAL_ENDPOINTS = (
    "OS",
    "PFS",
    "DSS",
)

IDENTIFIER_COLUMNS = {
    "PATIENT_ID",
    "SAMPLE_ID",
}

MYGENE_QUERY_URL = "https://mygene.info/v3/query"
MYGENE_CHUNK_SIZE = 1_000
MYGENE_TIMEOUT_SECONDS = 30
MYGENE_RETRIES = 3


# ---------------------------------------------------------------------------
# Identifier cleaning
# ---------------------------------------------------------------------------

def standardise_sample_id(sample_id: Any) -> str:
    """Standardise a sample identifier.

    TCGA barcodes are normalised to the first 15 characters, corresponding
    to the patient and sample portion of the barcode:

        TCGA-XX-YYYY-01

    Other identifiers are cleaned conservatively without truncation.

    Args:
        sample_id:
            Raw sample identifier.

    Returns:
        A standardised identifier, or an empty string for missing values.
    """
    if sample_id is None or pd.isna(sample_id):
        return ""

    value = str(sample_id).strip().replace(".", "-").upper()

    if value.startswith("TCGA-"):
        return value[:15]

    return value


def standardise_patient_id(patient_id: Any) -> str:
    """Standardise a patient identifier.

    Unlike sample identifiers, patient identifiers are not truncated because
    non-TCGA cohorts may use arbitrary identifier formats.

    Args:
        patient_id:
            Raw patient identifier.

    Returns:
        A standardised identifier, or an empty string for missing values.
    """
    if patient_id is None or pd.isna(patient_id):
        return ""

    return str(patient_id).strip().upper()


# ---------------------------------------------------------------------------
# Survival endpoint cleaning
# ---------------------------------------------------------------------------

def parse_survival_status(value: Any) -> Optional[float]:
    """Convert a clinical survival status value to a binary event indicator.

    Returns:

        ``1.0``
            Event occurred, such as death or progression.

        ``0.0``
            Event was censored or the patient was alive.

        ``None``
            Status could not be interpreted.

    Args:
        value:
            Raw clinical survival status.

    Returns:
        Binary event status or ``None``.
    """
    if value is None or pd.isna(value):
        return None

    status = str(value).strip().upper()

    # cBioPortal commonly represents statuses as "1:DECEASED".
    if status.startswith("1:") or status == "1":
        return 1.0

    if status.startswith("0:") or status == "0":
        return 0.0

    event_terms = (
        "DECEASED",
        "DEAD WITH TUMOUR",
        "RECURRED",
        "PROGRESSION",
    )

    censored_terms = (
        "LIVING",
        "ALIVE OR DEAD TUMOUR FREE",
        "DISEASEFREE",
        "DISEASE FREE",
        "CENSORED",
    )

    if any(term in status for term in event_terms):
        return 1.0

    if any(term in status for term in censored_terms):
        return 0.0

    return None


def clean_survival_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise available survival endpoints in a clinical dataframe.

    For each recognised endpoint, the corresponding ``*_STATUS`` column is
    converted to a binary event indicator and the ``*_MONTHS`` column is
    converted to numeric values.

    Missing endpoint columns are ignored.

    Args:
        df:
            Clinical dataframe.

    Returns:
        Dataframe with cleaned survival fields.
    """
    cleaned = df.copy()

    for endpoint in SURVIVAL_ENDPOINTS:
        status_column = f"{endpoint}_STATUS"
        time_column = f"{endpoint}_MONTHS"

        if status_column in cleaned.columns:
            cleaned[status_column] = cleaned[status_column].apply(
                parse_survival_status
            )

        if time_column in cleaned.columns:
            cleaned[time_column] = pd.to_numeric(
                cleaned[time_column],
                errors="coerce",
            )

    return cleaned


# ---------------------------------------------------------------------------
# Clinical data cleaning
# ---------------------------------------------------------------------------
def clean_clinical_df(
    raw_df: pd.DataFrame
) -> pd.DataFrame:
    """Clean and standardise a clinical dataframe.

    The function:

    1. copies the input dataframe;
    2. standardises available sample and patient identifiers;
    3. removes rows without a valid sample identifier when ``SAMPLE_ID``
       exists;
    4. removes duplicate samples;
    5. normalises survival endpoints;
    6. removes known administrative metadata;
    7. optionally removes constant columns.

    Args:
        raw_df:
            Raw clinical dataframe.

    Returns:
        Cleaned clinical dataframe.

    Raises:
        TypeError:
            If ``raw_df`` is not a pandas dataframe.
    """
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError(
            f"Expected pandas.DataFrame, got {type(raw_df).__name__}"
        )

    df = raw_df.copy()

    if "SAMPLE_ID" in df.columns:
        df["SAMPLE_ID"] = df["SAMPLE_ID"].map(
            standardise_sample_id
        )

        df = df.loc[df["SAMPLE_ID"].ne("")]

        # Keep the first occurrence deterministically.
        df = df.drop_duplicates(
            subset=["SAMPLE_ID"],
            keep="first",
        )

    if "PATIENT_ID" in df.columns:
        df["PATIENT_ID"] = df["PATIENT_ID"].map(
            standardise_patient_id
        )

    df = clean_survival_endpoints(df)

    return df.reset_index(drop=True)

def drop_empty_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Drop columns containing no usable values."""

    cleaned = df.copy()

    cleaned = cleaned.replace(
        r"^\s*$",
        np.nan,
        regex=True,
    )

    return cleaned.dropna(
        axis=1,
        how="all",
    )

# ---------------------------------------------------------------------------
# Mutation cleaning
# ---------------------------------------------------------------------------
def parse_maf_mutations(
    maf_path: Path,
    sample_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Create a binary non-synonymous mutation matrix from a MAF file.

    Args:
        maf_path:
            Path to the MAF file.
        sample_ids:
            Optional sample IDs to retain.

    Returns:
        Binary mutation matrix indexed by sample ID, with gene symbols as
        columns. Returns an empty DataFrame if the file does not exist or
        contains no qualifying mutations.
    """
    if not maf_path.exists():
        return pd.DataFrame()

    df_mut = pd.read_csv(
        maf_path,
        sep="\t",
        comment="#",
        low_memory=False,
    )

    required_columns = {
        "Tumor_Sample_Barcode",
        "Hugo_Symbol",
        "Variant_Classification",
    }

    missing_columns = required_columns.difference(df_mut.columns)

    if missing_columns:
        raise ValueError(
            f"MAF file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df_mut["Tumor_Sample_Barcode"] = (
        df_mut["Tumor_Sample_Barcode"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if sample_ids is not None:
        normalised_sample_ids = {
            str(sample_id).strip().upper()
            for sample_id in sample_ids
        }

        df_mut = df_mut[
            df_mut["Tumor_Sample_Barcode"].isin(normalised_sample_ids)
        ]

    df_mut = df_mut[
        df_mut["Variant_Classification"].isin(
            NON_SILENT_VARIANT_CLASSIFICATIONS
        )
    ]

    aa_col = next((c for c in ["HGVSp_Short", "HGVSp", "Protein_Change"] if c in df_mut.columns), None)
    if aa_col:
        is_braf_v600 = (df_mut["Hugo_Symbol"] == "BRAF") & df_mut[aa_col].astype(str).str.contains("V600", case=False, na=False)
        v600_rows = df_mut[is_braf_v600].copy()
        if not v600_rows.empty:
            v600_rows["Hugo_Symbol"] = "BRAF_V600"
            df_mut = pd.concat([df_mut, v600_rows], ignore_index=True)

    mutation_matrix = (
        df_mut.assign(mutation=1)
        .pivot_table(
            index="Tumor_Sample_Barcode",
            columns="Hugo_Symbol",
            values="mutation",
            aggfunc="max",
            fill_value=0,
        )
        .astype("int8")
    )

    mutation_matrix.index.name = None
    mutation_matrix.columns.name = None

    return mutation_matrix

# ---------------------------------------------------------------------------
# RNA-seq cleaning
# ---------------------------------------------------------------------------

def clean_rnaseq_df(
    rnaseq_df: pd.DataFrame,
    *,
    drop_missing_genes: bool = True,
    coerce_numeric: bool = True,
) -> pd.DataFrame:
    """Clean a sample-by-gene RNA-seq dataframe.

    Args:
        rnaseq_df:
            Expression dataframe containing a ``SAMPLE_ID`` column.

        drop_missing_genes:
            Remove gene columns containing one or more missing values.

        coerce_numeric:
            Convert expression columns to numeric values, coercing invalid
            values to ``NaN``.

    Returns:
        Cleaned RNA-seq dataframe.

    Raises:
        TypeError:
            If ``rnaseq_df`` is not a pandas dataframe.
    """
    if not isinstance(rnaseq_df, pd.DataFrame):
        raise TypeError(
            f"Expected pandas.DataFrame, got {type(rnaseq_df).__name__}"
        )

    if rnaseq_df.empty:
        return rnaseq_df.copy()

    if "SAMPLE_ID" not in rnaseq_df.columns:
        raise ValueError(
            "RNA-seq dataframe must contain a 'SAMPLE_ID' column."
        )

    df = rnaseq_df.copy()

    df["SAMPLE_ID"] = df["SAMPLE_ID"].map(
        standardise_sample_id
    )

    df = df.loc[df["SAMPLE_ID"].ne("")]

    df = df.drop_duplicates(
        subset=["SAMPLE_ID"],
        keep="first",
    )

    gene_columns = [
        column
        for column in df.columns
        if column != "SAMPLE_ID"
    ]

    if coerce_numeric and gene_columns:
        df[gene_columns] = df[gene_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )

    if drop_missing_genes and gene_columns:
        valid_gene_columns = [
            column
            for column in gene_columns
            if not df[column].isna().any()
        ]

        df = df[
            ["SAMPLE_ID", *valid_gene_columns]
        ]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Expression / clinical alignment
# ---------------------------------------------------------------------------

def align_expression_and_clinical(
    expression_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align expression and clinical data on their shared sample IDs.

    Both input dataframes must use sample identifiers as their index.

    The returned dataframes have identical row order and identical indices.

    Args:
        expression_df:
            Expression dataframe with samples as rows.

        clinical_df:
            Clinical dataframe indexed by sample ID.

    Returns:
        Tuple of aligned expression and clinical dataframes.

    Raises:
        ValueError:
            If either dataframe has duplicate sample identifiers.
    """
    if expression_df.index.has_duplicates:
        raise ValueError(
            "Expression dataframe contains duplicate sample IDs."
        )

    if clinical_df.index.has_duplicates:
        raise ValueError(
            "Clinical dataframe contains duplicate sample IDs."
        )

    common_samples = expression_df.index.intersection(
        clinical_df.index
    )

    aligned_expression = expression_df.loc[common_samples].copy()
    aligned_clinical = clinical_df.loc[common_samples].copy()

    aligned_expression.index.name = "SAMPLE_ID"
    aligned_clinical.index.name = "SAMPLE_ID"

    return aligned_expression, aligned_clinical



def read_cbioportal_table(
    path: Path,
    required_column: str,
) -> pd.DataFrame:
    """Read a cBioPortal tab-delimited file.

    Detects the header row dynamically rather than assuming a fixed number
    of metadata/comment lines.

    Args:
        path:
            Path to the cBioPortal tab-delimited file.

        required_column:
            Column expected to appear in the header.

    Returns:
        Parsed DataFrame.

    Raises:
        FileNotFoundError:
            If the file does not exist.

        ValueError:
            If the required column cannot be found in the detected header.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"cBioPortal file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            columns = line.rstrip("\n\r").split("\t")

            if required_column in columns:
                return pd.read_csv(
                    path,
                    sep="\t",
                    skiprows=line_number,
                )

    raise ValueError(
        f"Could not detect a header containing "
        f"'{required_column}' in {path}."
    )

# ---------------------------------------------------------------------------
# Entrez Gene ID mapping
# ---------------------------------------------------------------------------

def _normalise_entrez_ids(
    entrez_ids: Iterable[Any],
) -> list[str]:
    """Return unique, non-empty Entrez IDs in stable order."""
    normalised_ids: list[str] = []

    for value in entrez_ids:
        if value is None or pd.isna(value):
            continue

        value_str = str(value).strip()

        if not value_str:
            continue

        if value_str not in normalised_ids:
            normalised_ids.append(value_str)

    return normalised_ids


def _load_mapping_cache(
    cache_path: Path,
) -> dict[str, str]:
    """Load an Entrez-to-symbol mapping cache."""
    if not cache_path.exists():
        return {}

    try:
        with cache_path.open(
            "r",
            encoding="utf-8",
        ) as cache_file:
            data = json.load(cache_file)

    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"[WARNING] Could not load gene mapping cache "
            f"{cache_path}: {exc}"
        )
        return {}

    if not isinstance(data, dict):
        print(
            f"[WARNING] Gene mapping cache has unexpected format: "
            f"{cache_path}"
        )
        return {}

    return {
        str(key): str(value)
        for key, value in data.items()
        if value
    }


def _save_mapping_cache(
    mapping: Mapping[str, str],
    cache_path: Path,
) -> None:
    """Save an Entrez-to-symbol mapping cache atomically."""
    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = cache_path.with_suffix(
        f"{cache_path.suffix}.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as cache_file:
        json.dump(
            dict(mapping),
            cache_file,
            indent=2,
            sort_keys=True,
        )

    temporary_path.replace(cache_path)


def map_entrez_to_symbols(
    entrez_ids: Iterable[Any],
    cache_path: Optional[Path] = None,
    *,
    chunk_size: int = MYGENE_CHUNK_SIZE,
    timeout: int = MYGENE_TIMEOUT_SECONDS,
    retries: int = MYGENE_RETRIES,
) -> dict[str, str]:
    """Map Entrez Gene IDs to HGNC symbols using MyGene.info.

    Existing mappings are loaded from the optional local cache. Only IDs not
    already present in the cache are queried remotely.

    Args:
        entrez_ids:
            Iterable of Entrez Gene IDs.

        cache_path:
            Optional JSON cache path.

        chunk_size:
            Number of IDs per API request.

        timeout:
            HTTP request timeout in seconds.

        retries:
            Number of attempts for each API request.

    Returns:
        Mapping from Entrez Gene ID to HGNC symbol.
    """
    ids = _normalise_entrez_ids(entrez_ids)

    if not ids:
        return {}

    cache: dict[str, str] = {}

    if cache_path is not None:
        cache = _load_mapping_cache(Path(cache_path))

    missing_ids = [
        entrez_id
        for entrez_id in ids
        if entrez_id not in cache
    ]

    if not missing_ids:
        return {
            entrez_id: cache[entrez_id]
            for entrez_id in ids
            if entrez_id in cache
        }

    print(
        f"Mapping {len(missing_ids):,} Entrez IDs to HGNC symbols "
        f"via MyGene.info..."
    )

    for start in range(
        0,
        len(missing_ids),
        chunk_size,
    ):
        chunk = missing_ids[
            start:start + chunk_size
        ]

        query_data = urllib.parse.urlencode(
            {
                "q": ",".join(chunk),
                "scopes": "entrezgene",
                "fields": "symbol",
                "species": "human",
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            MYGENE_QUERY_URL,
            data=query_data,
            headers={
                "Content-Type": (
                    "application/x-www-form-urlencoded"
                )
            },
            method="POST",
        )

        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=timeout,
                ) as response:
                    payload = json.loads(
                        response.read().decode("utf-8")
                    )

                for item in payload:
                    query = item.get("query")
                    symbol = item.get("symbol")

                    if query and symbol:
                        cache[str(query)] = str(symbol)
                    elif query:
                        cache[str(query)] = str(query)

                for entrez_id in chunk:
                    if entrez_id not in cache:
                        cache[entrez_id] = entrez_id

                break

            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
            ) as exc:
                if attempt == retries:
                    print(
                        f"[WARNING] Failed to map Entrez ID chunk "
                        f"{start:,}-{start + len(chunk):,}: {exc}"
                    )
                    break

                wait_seconds = 2 ** (attempt - 1)

                print(
                    f"[WARNING] MyGene.info request failed "
                    f"(attempt {attempt}/{retries}): {exc}. "
                    f"Retrying in {wait_seconds}s..."
                )

                time.sleep(wait_seconds)

    if cache_path is not None:
        _save_mapping_cache(
            cache,
            Path(cache_path),
        )

    return {
        entrez_id: cache[entrez_id]
        for entrez_id in ids
        if entrez_id in cache
    }