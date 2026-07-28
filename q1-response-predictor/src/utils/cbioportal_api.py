"""Client utilities for the cBioPortal REST API.

Provides:
- resilient GET and POST requests with retry handling
- study and molecular-profile discovery
- clinical and molecular data retrieval
- conversion of cBioPortal API records into pandas DataFrames

Dataset-specific configuration, including study IDs, should be defined
outside this module.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------

CBIOPORTAL_BASE_URL = "https://www.cbioportal.org/api"

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_POST_TIMEOUT_SECONDS = 120
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_RETRY_DELAY_SECONDS = 2.0

TRANSIENT_STATUS_CODES = {
    429,  # Too Many Requests
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
}

DEFAULT_PAGE_SIZE = 100_000
MOLECULAR_DATA_CHUNK_SIZE = 100


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


class CbioPortalClient:
    """Small resilient client for the cBioPortal REST API.

    A persistent requests session is used so that connections can be reused
    across requests.
    """

    def __init__(
        self,
        base_url: str = CBIOPORTAL_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay: float = DEFAULT_RETRY_DELAY_SECONDS,
    ) -> None:
        """Initialises the API client.

        Args:
            base_url: Base URL of the cBioPortal API.
            timeout: Default request timeout in seconds.
            max_attempts: Maximum number of attempts for transient failures.
            retry_delay: Base delay between retry attempts.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.retry_delay = retry_delay

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
            }
        )

    def _build_url(
        self,
        endpoint: str,
    ) -> str:
        """Builds a complete API URL from an endpoint."""
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[Any] = None,
        timeout: Optional[int] = None,
    ) -> Any:
        """Executes a resilient HTTP request.

        Retries connection errors, timeouts, and transient HTTP responses.
        Non-transient HTTP errors are raised immediately.

        Args:
            method: HTTP method, such as ``GET`` or ``POST``.
            endpoint: API endpoint.
            params: Optional query parameters.
            json_body: Optional JSON request body.
            timeout: Optional request-specific timeout.

        Returns:
            Decoded JSON response.

        Raises:
            requests.RequestException: If the request fails.
            RuntimeError: If all retry attempts fail.
        """
        url = self._build_url(endpoint)
        request_timeout = timeout or self.timeout
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_body,
                    timeout=request_timeout,
                )

                response.raise_for_status()
                return response.json()

            except (
                requests.ConnectionError,
                requests.Timeout,
            ) as error:
                last_exception = error

            except requests.HTTPError as error:
                response = error.response

                if (
                    response is None
                    or response.status_code not in TRANSIENT_STATUS_CODES
                ):
                    raise

                last_exception = error

            if attempt < self.max_attempts:
                delay = self.retry_delay * (2 ** (attempt - 1))

                print(
                    f"cBioPortal request failed "
                    f"(attempt {attempt}/{self.max_attempts}). "
                    f"Retrying in {delay:.1f} seconds..."
                )

                time.sleep(delay)

        raise RuntimeError(
            f"cBioPortal request failed after "
            f"{self.max_attempts} attempts: "
            f"{method} {url}"
        ) from last_exception

    def get(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Any:
        """Executes a GET request."""
        return self._request(
            "GET",
            endpoint,
            params=params,
            timeout=timeout,
        )

    def post(
        self,
        endpoint: str,
        json_body: Optional[Any] = None,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Any:
        """Executes a POST request."""
        return self._request(
            "POST",
            endpoint,
            params=params,
            json_body=json_body,
            timeout=timeout,
        )


# ---------------------------------------------------------------------------
# Study and molecular-profile utilities
# ---------------------------------------------------------------------------


def get_study(
    client: CbioPortalClient,
    study_id: str,
) -> Optional[dict[str, Any]]:
    """Returns metadata for a study.

    Args:
        client: cBioPortal API client.
        study_id: cBioPortal study identifier.

    Returns:
        Study metadata, or ``None`` if the study does not exist.
    """
    try:
        return client.get(f"/studies/{study_id}")

    except requests.HTTPError as error:
        if (
            error.response is not None
            and error.response.status_code == 404
        ):
            return None

        raise


def get_molecular_profiles(
    client: CbioPortalClient,
    study_id: str,
) -> list[dict[str, Any]]:
    """Returns molecular profiles available for a study."""
    return client.get(
        f"/studies/{study_id}/molecular-profiles"
    )


def resolve_profile_id(
    client: CbioPortalClient,
    study_id: str,
    data_type: str,
) -> Optional[str]:
    """Finds the best molecular profile for a data type.

    Currently supports:
    - ``rnaseq``
    - ``cna``

    Args:
        client: cBioPortal API client.
        study_id: cBioPortal study identifier.
        data_type: Canonical data type.

    Returns:
        Molecular profile ID, or ``None`` if no suitable profile exists.
    """
    profile_types = {
        "rnaseq": {"MRNA_EXPRESSION"},
        "cna": {"COPY_NUMBER_ALTERATION"},
    }

    preferred_profiles = {
        "rnaseq": ("rna_seq_v2_mrna",),
        "cna": ("gistic",),
    }

    valid_types = profile_types.get(data_type, set())

    if not valid_types:
        raise ValueError(
            f"Unsupported molecular data type: {data_type}"
        )

    profiles = get_molecular_profiles(
        client,
        study_id,
    )

    candidates = [
        profile
        for profile in profiles
        if profile.get("molecularAlterationType")
        in valid_types
    ]

    if not candidates:
        return None

    for preferred in preferred_profiles.get(
        data_type,
        (),
    ):
        for candidate in candidates:
            profile_id = candidate.get(
                "molecularProfileId",
                "",
            )

            if (
                data_type == "rnaseq"
                and "zscore" in profile_id.lower()
            ):
                continue

            if preferred.lower() in profile_id.lower():
                return profile_id

    if data_type == "rnaseq":
        non_zscore = [
            candidate
            for candidate in candidates
            if "zscore"
            not in candidate.get(
                "molecularProfileId",
                "",
            ).lower()
        ]

        if non_zscore:
            return non_zscore[0][
                "molecularProfileId"
            ]

    return candidates[0]["molecularProfileId"]


# ---------------------------------------------------------------------------
# Data retrieval
# ---------------------------------------------------------------------------


def get_sample_ids(
    client: CbioPortalClient,
    study_id: str,
) -> list[str]:
    """Returns all sample IDs for a study."""
    samples = client.get(
        f"/studies/{study_id}/samples",
        params={
            "projection": "ID",
            "pageSize": DEFAULT_PAGE_SIZE,
        },
    )

    return [
        sample["sampleId"]
        for sample in samples
        if "sampleId" in sample
    ]


def get_clinical_data(
    client: CbioPortalClient,
    study_id: str,
    clinical_data_type: str = "SAMPLE",
) -> list[dict[str, Any]]:
    """Fetches sample- or patient-level clinical data.

    Args:
        client: cBioPortal API client.
        study_id: cBioPortal study identifier.
        clinical_data_type: ``SAMPLE`` or ``PATIENT``.

    Returns:
        Clinical data records.
    """
    clinical_data_type = clinical_data_type.upper()

    if clinical_data_type not in {
        "SAMPLE",
        "PATIENT",
    }:
        raise ValueError(
            "clinical_data_type must be 'SAMPLE' "
            "or 'PATIENT'."
        )

    return client.get(
        f"/studies/{study_id}/clinical-data",
        params={
            "clinicalDataType": clinical_data_type,
            "projection": "DETAILED",
            "pageSize": DEFAULT_PAGE_SIZE,
            "pageNumber": 0,
        },
    )


def get_molecular_data(
    client: CbioPortalClient,
    molecular_profile_id: str,
    sample_ids: list[str],
    chunk_size: int = MOLECULAR_DATA_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    """Fetches molecular data in batches.

    Args:
        client: cBioPortal API client.
        molecular_profile_id: Molecular profile identifier.
        sample_ids: Sample IDs to retrieve.
        chunk_size: Number of samples per API request.

    Returns:
        Molecular data records.
    """
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if not sample_ids:
        return []

    all_records: list[dict[str, Any]] = []

    total_chunks = (
        len(sample_ids) + chunk_size - 1
    ) // chunk_size

    for chunk_number, start in enumerate(
        range(
            0,
            len(sample_ids),
            chunk_size,
        ),
        start=1,
    ):
        sample_chunk = sample_ids[
            start:start + chunk_size
        ]

        records = client.post(
            f"/molecular-profiles/{molecular_profile_id}"
            "/molecular-data/fetch",
            json_body={
                "sampleIds": sample_chunk,
            },
            params={
                "projection": "SUMMARY",
            },
            timeout=DEFAULT_POST_TIMEOUT_SECONDS,
        )

        all_records.extend(records)

        print(
            f"  Fetched molecular data chunk "
            f"{chunk_number} of {total_chunks}..."
        )

    return all_records


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------


def build_clinical_df(
    sample_records: list[dict[str, Any]],
    patient_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Converts long-format clinical records into a wide DataFrame.

    Sample-level clinical data are used as the primary table. Patient-level
    data are merged onto the sample-level records using ``PATIENT_ID``.
    """
    if not sample_records:
        raise ValueError(
            "No sample-level clinical data returned from API."
        )

    sample_df = pd.DataFrame(
        sample_records
    )

    required_columns = {
        "sampleId",
        "clinicalAttributeId",
        "value",
    }

    missing_columns = (
        required_columns
        - set(sample_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Sample clinical data are missing "
            f"required columns: {missing_columns}"
        )

    sample_wide = (
        sample_df
        .pivot_table(
            index="sampleId",
            columns="clinicalAttributeId",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    sample_wide.columns.name = None

    sample_wide = sample_wide.rename(
        columns={
            "sampleId": "SAMPLE_ID",
        }
    )

    if {
        "sampleId",
        "patientId",
    }.issubset(sample_df.columns):
        patient_map = (
            sample_df[
                [
                    "sampleId",
                    "patientId",
                ]
            ]
            .drop_duplicates()
            .rename(
                columns={
                    "sampleId": "SAMPLE_ID",
                    "patientId": "PATIENT_ID",
                }
            )
        )

        sample_wide = sample_wide.merge(
            patient_map,
            on="SAMPLE_ID",
            how="left",
        )

    if patient_records:
        patient_df = pd.DataFrame(
            patient_records
        )

        patient_wide = (
            patient_df
            .pivot_table(
                index="patientId",
                columns="clinicalAttributeId",
                values="value",
                aggfunc="first",
            )
            .reset_index()
        )

        patient_wide.columns.name = None

        patient_wide = patient_wide.rename(
            columns={
                "patientId": "PATIENT_ID",
            }
        )

        if "PATIENT_ID" in sample_wide.columns:
            sample_wide = sample_wide.merge(
                patient_wide,
                on="PATIENT_ID",
                how="left",
                suffixes=(
                    "",
                    "_PATIENT",
                ),
            )

    return sample_wide


def build_molecular_df(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Converts molecular API records into a wide DataFrame.

    Returns:
        DataFrame with samples as rows and genes as columns.
    """
    if not records:
        return pd.DataFrame(
            columns=["SAMPLE_ID"]
        )

    rows: list[dict[str, Any]] = []

    for record in records:
        gene = record.get(
            "gene",
            {},
        )

        gene_symbol = (
            gene.get("hugoGeneSymbol")
            or str(
                record.get(
                    "entrezGeneId",
                    "",
                )
            )
        )

        rows.append(
            {
                "SAMPLE_ID": record["sampleId"],
                "gene": gene_symbol,
                "value": record.get("value"),
            }
        )

    long_df = pd.DataFrame(rows)

    long_df["value"] = pd.to_numeric(
        long_df["value"],
        errors="coerce",
    )

    wide_df = (
        long_df
        .pivot_table(
            index="SAMPLE_ID",
            columns="gene",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    wide_df.columns.name = None

    return wide_df