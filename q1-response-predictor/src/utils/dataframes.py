"""Small reusable helpers for working with pandas DataFrames."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pandas as pd


# ---------------------------------------------------------------------------
# Identifier column candidates
# ---------------------------------------------------------------------------

DEFAULT_ID_CANDIDATES: tuple[str, ...] = (
    "PATIENT_ID",
    "patient_id",
    "SAMPLE_ID",
    "sample_id",
)

PATIENT_ID_CANDIDATES: tuple[str, ...] = (
    "PATIENT_ID",
    "patient_id",
)

SAMPLE_ID_CANDIDATES: tuple[str, ...] = (
    "SAMPLE_ID",
    "sample_id",
)


# ---------------------------------------------------------------------------
# Identifier discovery
# ---------------------------------------------------------------------------

def find_id_column(
    df: pd.DataFrame,
    candidates: Sequence[str] = DEFAULT_ID_CANDIDATES,
    *,
    case_sensitive: bool = False,
) -> str:
    """Return the first matching identifier column in a dataframe.

    Candidate order determines precedence. For example, the default order
    prefers ``PATIENT_ID`` over ``SAMPLE_ID``.

    Matching is case-insensitive by default, but the actual column name from
    the dataframe is returned.

    Args:
        df:
            DataFrame whose columns should be searched.

        candidates:
            Candidate identifier column names, in preferred order.

        case_sensitive:
            Whether matching should respect column-name case.

    Returns:
        The actual matching column name from ``df``.

    Raises:
        TypeError:
            If ``df`` is not a pandas DataFrame.

        ValueError:
            If no candidate column is found or no candidates are supplied.

    Examples:
        >>> find_id_column(df)
        'PATIENT_ID'

        >>> find_id_column(df, SAMPLE_ID_CANDIDATES)
        'sample_id'
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected pandas.DataFrame, got {type(df).__name__}."
        )

    if not candidates:
        raise ValueError(
            "At least one candidate identifier column must be provided."
        )

    if case_sensitive:
        dataframe_columns = set(df.columns)

        for candidate in candidates:
            if candidate in dataframe_columns:
                return candidate

    else:
        column_lookup = {
            str(column).casefold(): column
            for column in df.columns
        }

        for candidate in candidates:
            match = column_lookup.get(str(candidate).casefold())

            if match is not None:
                return match

    raise ValueError(
        "No recognised identifier column found. "
        f"Looked for: {list(candidates)}. "
        f"Available columns: {list(df.columns)}"
    )


def find_patient_id_column(
    df: pd.DataFrame,
    *,
    case_sensitive: bool = False,
) -> str:
    """Return the patient identifier column from a dataframe."""
    return find_id_column(
        df,
        PATIENT_ID_CANDIDATES,
        case_sensitive=case_sensitive,
    )


def find_sample_id_column(
    df: pd.DataFrame,
    *,
    case_sensitive: bool = False,
) -> str:
    """Return the sample identifier column from a dataframe."""
    return find_id_column(
        df,
        SAMPLE_ID_CANDIDATES,
        case_sensitive=case_sensitive,
    )


def find_preferred_id_column(
    df: pd.DataFrame,
    preferred: Literal["patient", "sample"] = "patient",
    *,
    case_sensitive: bool = False,
) -> str:
    """Find a preferred patient or sample identifier column.

    Args:
        df:
            DataFrame whose columns should be searched.

        preferred:
            Identifier type to prioritise.

        case_sensitive:
            Whether matching should respect column-name case.

    Returns:
        The matching identifier column name.

    Raises:
        ValueError:
            If the preferred identifier is not present.
    """
    if preferred == "patient":
        candidates = PATIENT_ID_CANDIDATES

    elif preferred == "sample":
        candidates = SAMPLE_ID_CANDIDATES

    else:
        raise ValueError(
            f"Unsupported identifier preference: {preferred!r}. "
            "Expected 'patient' or 'sample'."
        )

    return find_id_column(
        df,
        candidates,
        case_sensitive=case_sensitive,
    )