import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Any, Optional

# cBioPortal REST API v2 client setup
BASE_URL = "https://www.cbioportal.org/api"
_SESSION = requests.Session()
_SESSION.headers.update({"Accept": "application/json"})

def _get(
    endpoint: str,
    params: Optional[dict] = None,
    retries: int = 5,
    timeout: int = 60,
) -> Any:
    """
    Make a GET request to the cBioPortal API with retry logic.
    """
    _TRANSIENT_STATUS = {429, 502, 503, 504}
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    last_exc = None

    for attempt in range(retries):
        try:
            response = _SESSION.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in _TRANSIENT_STATUS:
                last_exc = exc
            else:
                raise
        if attempt < retries - 1:
            time.sleep(2 ** attempt)

    raise last_exc

def _post(
    endpoint: str,
    json_body: Any = None,
    params: Optional[dict] = None,
    retries: int = 5,
    timeout: int = 120,
) -> Any:
    """
    Make a POST request to the cBioPortal API with retry logic.
    """
    _TRANSIENT_STATUS = {429, 502, 503, 504}
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    last_exc = None

    for attempt in range(retries):
        try:
            response = _SESSION.post(url, json=json_body, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in _TRANSIENT_STATUS:
                last_exc = exc
            else:
                raise
        if attempt < retries - 1:
            time.sleep(2 ** attempt)

    raise last_exc

def get_study(study_id: str) -> Optional[dict]:
    """
    Return metadata for a specific study, or None if the study ID is not found.
    """
    try:
        return _get(f"/studies/{study_id}")
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return None
        raise

def get_molecular_profiles(study_id: str) -> list[dict]:
    """
    Return a list of molecular profiles (data types) available for a study.
    """
    return _get(f"/studies/{study_id}/molecular-profiles")

def resolve_profile_id(study_id: str, data_type: str) -> Optional[str]:
    """
    Find the best molecular profile ID for a given canonical data type.
    """
    profiles = get_molecular_profiles(study_id)
    alt_types = {"rnaseq": ["MRNA_EXPRESSION"], "cna": ["COPY_NUMBER_ALTERATION"]}.get(data_type, [])

    candidates = [p for p in profiles if p.get("molecularAlterationType", "") in alt_types]
    if not candidates:
        return None

    # Preferred profiles
    prefs = {"rnaseq": ["rna_seq_v2_mrna"], "cna": ["gistic"]}.get(data_type, [])
    for pref in prefs:
        for c in candidates:
            pid = c["molecularProfileId"]
            if data_type == "rnaseq" and "zscores" in pid.lower():
                continue
            if pref.lower() in pid.lower():
                return pid

    if data_type == "rnaseq":
        non_zscore = [c for c in candidates if "zscore" not in c["molecularProfileId"].lower()]
        if non_zscore:
            return non_zscore[0]["molecularProfileId"]

    return candidates[0]["molecularProfileId"]

def get_sample_ids(study_id: str) -> list[str]:
    """Return all sample IDs for a study."""
    samples = _get(f"/studies/{study_id}/samples", params={"projection": "ID", "pageSize": 10_000})
    return [s["sampleId"] for s in samples]

def get_clinical_data(study_id: str, clinical_data_type: str = "SAMPLE") -> list[dict]:
    """
    Fetch clinical data (SAMPLE or PATIENT level) for a study.
    """
    return _get(
        f"/studies/{study_id}/clinical-data",
        params={
            "clinicalDataType": clinical_data_type,
            "projection": "DETAILED",
            "pageSize": 100_000,
            "pageNumber": 0,
        },
    )

def get_molecular_data(molecular_profile_id: str, sample_ids: list[str]) -> list[dict]:
    """
    Fetch molecular data (expression) for a given profile.
    """
    all_records = []
    chunk_size = 100
    total_samples = len(sample_ids)

    for i in range(0, total_samples, chunk_size):
        chunk = sample_ids[i:i + chunk_size]
        body = {"sampleIds": chunk}
        records = _post(
            f"/molecular-profiles/{molecular_profile_id}/molecular-data/fetch",
            json_body=body,
            params={"projection": "SUMMARY"},
        )
        all_records.extend(records)
        print(f"  Fetched molecular data chunk {i // chunk_size + 1} of {-(-total_samples // chunk_size)}...")

    return all_records

# Cleaners and builders (from loaders.py)
def standardise_sample_id(sample_id: Any) -> str:
    """
    Standardise TCGA sample barcodes to a uniform 15-character hyphenated format.
    """
    if pd.isna(sample_id) or sample_id is None:
        return ""
    s = str(sample_id).strip().replace(".", "-")
    if s.upper().startswith("TCGA-"):
        s = s.upper()
        if len(s) > 15:
            return s[:15]
    return s

def parse_survival_status(val: Any) -> Optional[float]:
    """
    Map clinical survival status to a binary 0.0 (alive) or 1.0 (deceased).
    """
    if pd.isna(val) or val is None:
        return None
    val_str = str(val).strip().upper()
    if val_str.startswith("1:") or val_str == "1":
        return 1.0
    if val_str.startswith("0:") or val_str == "0":
        return 0.0
    if any(word in val_str for word in ["DECEASED", "DEAD WITH TUMOR", "RECURRED", "PROGRESSION"]):
        return 1.0
    if any(word in val_str for word in ["LIVING", "ALIVE OR DEAD TUMOR FREE", "DISEASEFREE", "CENSORED"]):
        return 0.0
    return None

def build_clinical_df(sample_records: List[dict], patient_records: List[dict]) -> pd.DataFrame:
    """
    Pivot long-format clinical API records into a wide table.
    """
    if not sample_records:
        raise ValueError("No sample-level clinical data returned from API")

    sample_df = pd.DataFrame(sample_records)
    sample_wide = sample_df.pivot_table(
        index="sampleId",
        columns="clinicalAttributeId",
        values="value",
        aggfunc="first",
    ).reset_index()
    sample_wide.columns.name = None
    sample_wide = sample_wide.rename(columns={"sampleId": "SAMPLE_ID"})

    if "patientId" in sample_df.columns:
        patient_map = (
            sample_df[["sampleId", "patientId"]]
            .drop_duplicates()
            .rename(columns={"sampleId": "SAMPLE_ID", "patientId": "PATIENT_ID"})
        )
        sample_wide = sample_wide.merge(patient_map, on="SAMPLE_ID", how="left")

    if patient_records:
        patient_df = pd.DataFrame(patient_records)
        patient_wide = patient_df.pivot_table(
            index="patientId",
            columns="clinicalAttributeId",
            values="value",
            aggfunc="first",
        ).reset_index()
        patient_wide.columns.name = None
        patient_wide = patient_wide.rename(columns={"patientId": "PATIENT_ID"})

        if "PATIENT_ID" in sample_wide.columns:
            sample_wide = sample_wide.merge(
                patient_wide,
                on="PATIENT_ID",
                how="left",
                suffixes=("", "_PATIENT"),
            )

    return sample_wide

def clean_clinical_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply deduplication and survival data cleaning."""
    df = raw_df.copy()
    df["SAMPLE_ID"] = df["SAMPLE_ID"].apply(standardise_sample_id)
    df = df[df["SAMPLE_ID"] != ""]
    df = df.drop_duplicates(subset=["SAMPLE_ID"])
    if "PATIENT_ID" in df.columns:
        df = df.drop_duplicates(subset=["PATIENT_ID"])
    if "OS_STATUS" in df.columns and "OS_MONTHS" in df.columns:
        df["OS_STATUS"] = df["OS_STATUS"].apply(parse_survival_status)
        df["OS_MONTHS"] = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
        invalid_mask = (
            df["OS_MONTHS"].isna() |
            (df["OS_MONTHS"] <= 0) |
            df["OS_STATUS"].isna()
        )
        df = df[~invalid_mask]
    if "PFS_STATUS" in df.columns and "PFS_MONTHS" in df.columns:
        df["PFS_STATUS"] = df["PFS_STATUS"].apply(parse_survival_status)
        df["PFS_MONTHS"] = pd.to_numeric(df["PFS_MONTHS"], errors="coerce")
    if "DSS_STATUS" in df.columns and "DSS_MONTHS" in df.columns:
        df["DSS_STATUS"] = df["DSS_STATUS"].apply(parse_survival_status)
        df["DSS_MONTHS"] = pd.to_numeric(df["DSS_MONTHS"], errors="coerce")
    # ---- New admin‑column filter ------------------------------------------------
    def _is_admin(col: str) -> bool:
        # Remove columns that are duplicates or cBioPortal administrative metadata
        admin_suffixes = ("_PATIENT", "_SAMPLE")
        if col.endswith(admin_suffixes):
            return True
        # Common admin columns that are not used in modeling
        unwanted = {"BIRTH_YEAR", "CANCER_TYPE_DETAILED", "PROTOCOL_SUBMIT_DATE"}
        if col in unwanted:
            return True
        # Keep identifier columns (only one copy)
        if col in {"PATIENT_ID", "SAMPLE_ID"}:
            # If there are multiple versions we will keep the base name only
            return False
        return False
    # Drop admin columns identified above
    cols_to_drop = [c for c in df.columns if _is_admin(c)]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    return df

def build_molecular_df(records: List[dict]) -> pd.DataFrame:
    """
    Pivot molecular data API records into a wide samples x genes matrix.
    """
    if not records:
        return pd.DataFrame(columns=["SAMPLE_ID"])

    rows = []
    for r in records:
        gene_symbol = r.get("gene", {}).get("hugoGeneSymbol") or str(r.get("entrezGeneId", ""))
        rows.append({
            "SAMPLE_ID": r["sampleId"],
            "gene": gene_symbol,
            "value": r.get("value"),
        })

    long_df = pd.DataFrame(rows)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    wide_df = long_df.pivot_table(
        index="SAMPLE_ID",
        columns="gene",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide_df.columns.name = None

    return wide_df

def clean_rnaseq_df(rnaseq_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate RNA-seq sample records and remove genes with missing values.
    """
    if rnaseq_df.empty or "SAMPLE_ID" not in rnaseq_df.columns:
        return rnaseq_df

    df = rnaseq_df.copy()
    df["SAMPLE_ID"] = df["SAMPLE_ID"].apply(standardise_sample_id)
    df = df[df["SAMPLE_ID"] != ""]

    df["PATIENT_ID"] = df["SAMPLE_ID"].apply(
        lambda x: "-".join(x.split("-")[:3]) if isinstance(x, str) and x.startswith("TCGA-") else x
    )
    df = df.drop_duplicates(subset=["PATIENT_ID"])
    df = df.drop(columns=["PATIENT_ID"])

    gene_cols = [c for c in df.columns if c != "SAMPLE_ID"]
    missing_counts = df[gene_cols].isna().sum()
    cols_with_nans = missing_counts[missing_counts > 0].index.tolist()
    if cols_with_nans:
        df = df.drop(columns=cols_with_nans)

    return df

def download_raw_tcga_skcm(output_dir: Path, study_id: str = "skcm_tcga_pan_can_atlas_2018"):
    """
    Downloads raw TCGA-SKCM clinical and RNA-seq data from cBioPortal REST API.

    @param Path output_dir The base data directory (e.g. DATA_DIR).
    @param str study_id The cBioPortal study ID (default is skcm_tcga_pan_can_atlas_2018).
    @return None
    """
    raw_dir = output_dir / "raw" / study_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching sample list for {study_id}...")
    sample_ids = get_sample_ids(study_id)
    print(f"Found {len(sample_ids)} samples.")

    print("Fetching clinical sample and patient data...")
    sample_clin = get_clinical_data(study_id, "SAMPLE")
    patient_clin = get_clinical_data(study_id, "PATIENT")
    raw_clin_df = build_clinical_df(sample_clin, patient_clin)

    raw_clin_df.to_csv(raw_dir / "clinical.csv", index=False)
    print(f"Saved raw clinical records: {len(raw_clin_df)} samples.")

    print("Fetching RNA-seq expression data (this may take a few minutes)...")
    profile_id = resolve_profile_id(study_id, "rnaseq")
    if not profile_id:
        raise ValueError("Could not resolve RNA-seq profile ID.")
    
    records = get_molecular_data(profile_id, sample_ids)
    raw_rnaseq_df = build_molecular_df(records)

    raw_rnaseq_df.to_csv(raw_dir / "rnaseq.csv", index=False)
    print(f"Saved raw RNA-seq records: {len(raw_rnaseq_df)} samples.")
