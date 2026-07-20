"""Small reusable DataFrame helpers."""

import pandas as pd

DEFAULT_ID_CANDIDATES = ["PATIENT_ID", "patient_id", "SAMPLE_ID", "sample_id"]


def find_id_column(df: pd.DataFrame, candidates=DEFAULT_ID_CANDIDATES) -> str:
    """
    Return whichever recognized patient/sample id column is present in `df`.
    Raises ValueError if none of the candidates are found, rather than
    silently picking the wrong column.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No recognizable id column found (looked for {candidates}) in: {list(df.columns)}")