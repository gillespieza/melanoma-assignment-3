#!/usr/bin/env python3
"""Script 01: Load and prepare feature matrix for Q5 patient stratification.

Loads preprocessed clinical, expression, and genomic data from data/processed/merged/immunotherapy/,
computes immune signatures, calculates Macrophage STV (M1/M2 ratio) and cell deconvolution,
merges patient features into a unified feature matrix, and exports it to data/processed/q5/feature_matrix.csv.
Generates baseline biomarker boxplot visualisations for presentation.
"""

import contextlib
from pathlib import Path
import sys
from typing import Tuple
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SUBPROJECT_ROOT = SCRIPT_DIR.parent
for parent in [SCRIPT_DIR] + list(SCRIPT_DIR.parents):
    if (parent / "data").is_dir() and (parent / "src").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

if str(SUBPROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_ROOT / "src"))

q1_root = SCRIPT_DIR.parent.parent / "q1-response-predictor"
if q1_root.exists() and str(q1_root) not in sys.path:
    sys.path.insert(0, str(q1_root))

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from deconvolution import (
    compute_cell_deconvolution,
    compute_macrophage_stv,
)
from phenotyping import plot_baseline_signature_boxplots
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, PROCESSED_DIR, PROJECT_ROOT, rel_path

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "01_load_and_prepare.log"

INPUT_DIR = PROCESSED_DIR / "merged" / "immunotherapy"
OUTPUT_DIR = PROCESSED_DIR / "q5"
STV_PATH = DATA_DIR / "config" / "m1_m2_stv.csv"


def load_processed_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate preprocessed multi-modal clinical, transcriptomic, and genomic datasets.

    Imports harmonised patient cohorts (Liu 2019, Riaz 2017, Hugo 2016, TCGA-SKCM) from
    data/processed/merged/immunotherapy/. 
    
    Performs path verification, indexes expression matrices by SAMPLE_ID for efficient joining, 
    deduplicates genomic mutation features, and handles missing files via defensive error handling.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - df_clin: Patient clinical metadata, overall survival, and binary response labels (N=326).
            - df_expr: Transposed gene expression matrix (samples as rows, 19,757 log2(TPM+1) genes as columns).
            - df_genomic: Driver mutation binary indicators (BRAF, NRAS, NF1, Triple-WT) and TMB counts.

    Raises:
        FileNotFoundError: If essential clinical or expression merged files are missing.
    """
    # ---------------------------------------------------------------------------
    # Step 1: Define canonical file paths to preprocessed cohort matrices
    # ---------------------------------------------------------------------------
    clin_file = INPUT_DIR / "clin_merged.csv"
    expr_file = INPUT_DIR / "expr_merged.csv"
    genomic_file = INPUT_DIR / "merged_genomic.csv"

    # Defensive check: ensure critical prerequisite files exist before proceeding
    if not clin_file.exists():
        raise FileNotFoundError(f"Missing clinical data file at {rel_path(clin_file)}. Ensure preprocessing pipeline has run.")
    if not expr_file.exists():
        raise FileNotFoundError(f"Missing expression data file at {rel_path(expr_file)}. Ensure preprocessing pipeline has run.")

    # ---------------------------------------------------------------------------
    # Step 2: Load merged clinical metadata and response labels
    # ---------------------------------------------------------------------------
    df_clin = pd.read_csv(clin_file)
    print(f"Loaded clinical dataset: {len(df_clin)} patients from {rel_path(clin_file)}")

    # ---------------------------------------------------------------------------
    # Step 3: Load RNA-seq expression matrix and index by sample identifier
    # ---------------------------------------------------------------------------
    df_expr = pd.read_csv(expr_file)
    if "SAMPLE_ID" in df_expr.columns:
        # Set SAMPLE_ID as index so rows correspond to individual patient samples
        df_expr = df_expr.set_index("SAMPLE_ID")
    print(f"Loaded expression matrix: {df_expr.shape[0]} samples x {df_expr.shape[1]} genes")

    # ---------------------------------------------------------------------------
    # Step 4: Load genomic driver mutation indicators and TMB (if available)
    # ---------------------------------------------------------------------------
    if genomic_file.exists():
        df_genomic = pd.read_csv(genomic_file)
        # Deduplicate columns by stripping pandas duplicate suffixes (.1, .2) and keeping first occurrence
        df_genomic = df_genomic.loc[:, ~df_genomic.columns.str.replace(r"\.\d+$", "", regex=True).duplicated(keep="first")]
        print(f"Loaded genomic dataset: {len(df_genomic)} rows from {rel_path(genomic_file)}")
    else:
        # Fallback to empty DataFrame if genomic MAF file is absent
        df_genomic = pd.DataFrame()
        print("Genomic file not found; using empty DataFrame fallback.")

    return df_clin, df_expr, df_genomic


def extract_immune_signatures(df_expr: pd.DataFrame) -> pd.DataFrame:
    """Compute core transcriptomic immune signatures from bulk RNA-seq expression profiles.

    First attempts to import and run the shared project-wide signature extraction module
    `src.signatures` (from Question 1). If unavailable, provides a robust, self-contained fallback
    that calculates published immune proxies:
    - **PD-L1 Expression**: Single-gene marker `CD274` (or `PDL1`).
    - **Tumour Inflammation Signature (TIS)**: Ayers et al. 18-gene IFN-gamma responsive microenvironment score.
    - **Cytolytic Index (CYT)**: Rooney et al. Geometric/arithmetic mean of granzyme A (`GZMA`) and perforin 1 (`PRF1`).
    - **Interferon-Gamma Response (IFN_gamma)**: Expression average across canonical STAT1/CXCL9/CXCL10 signaling genes.
    - **CD8+ T-Cell Density**: Average expression of lineage markers `CD8A` and `CD8B`.

    Args:
        df_expr (pd.DataFrame): Normalized gene expression matrix (samples as rows, genes as columns).

    Returns:
        pd.DataFrame: Calculated immune signature scores indexed by sample identifier.
    """
    # ---------------------------------------------------------------------------
    # Step 1: Attempt import of central Question 1 signature extraction module
    # ---------------------------------------------------------------------------
    q1_root = PROJECT_ROOT / "q1-response-predictor"
    q1_src_path = q1_root / "src"

    # Add q1 root so `from src.config.constants import ...` inside signatures.py resolves
    if str(q1_root) not in sys.path:
        sys.path.insert(0, str(q1_root))
    if str(q1_src_path) not in sys.path:
        sys.path.insert(0, str(q1_src_path))

    try:
        from signatures import extract_all_signatures
        df_sig = extract_all_signatures(df_expr)
        print("Extracted core immune signatures using shared Q1 signature module.")
    except Exception as err:
        # Fallback executed if Q1 module is unimportable or raises an exception
        print(f"Q1 signature module fallback (Reason: {err}). Computing signatures locally.")
        df_sig = pd.DataFrame(index=df_expr.index)

        # ---------------------------------------------------------------------------
        # Step 2: Fallback PD-L1 single-gene biomarker extraction
        # ---------------------------------------------------------------------------
        if "CD274" in df_expr.columns:
            df_sig["PD_L1"] = df_expr["CD274"]
        elif "PDL1" in df_expr.columns:
            df_sig["PD_L1"] = df_expr["PDL1"]

        # ---------------------------------------------------------------------------
        # Step 3: Fallback Tumour Inflammation Signature (TIS) proxy calculation
        # ---------------------------------------------------------------------------
        tis_markers = ["CD274", "PDCD1", "STAT1", "HLA-DRA", "CXCL9", "CXCL10", "IDO1"]
        tis_found = [g for g in tis_markers if g in df_expr.columns]
        if tis_found:
            df_sig["TIS"] = df_expr[tis_found].mean(axis=1)

        # ---------------------------------------------------------------------------
        # Step 4: Fallback Cytolytic Index (CYT) calculation (PRF1 + GZMA mean)
        # ---------------------------------------------------------------------------
        cyt_found = [g for g in ["PRF1", "GZMA"] if g in df_expr.columns]
        if cyt_found:
            df_sig["CYT"] = df_expr[cyt_found].mean(axis=1)

        # ---------------------------------------------------------------------------
        # Step 5: Fallback IFN-gamma pathway activation proxy calculation
        # ---------------------------------------------------------------------------
        ifng_found = [g for g in ["IFNG", "STAT1", "IDO1", "CXCL9", "CXCL10"] if g in df_expr.columns]
        if ifng_found:
            df_sig["IFN_gamma"] = df_expr[ifng_found].mean(axis=1)

        # ---------------------------------------------------------------------------
        # Step 6: Fallback CD8 T-cell lineage abundance proxy calculation
        # ---------------------------------------------------------------------------
        cd8_found = [g for g in ["CD8A", "CD8B"] if g in df_expr.columns]
        if cd8_found:
            df_sig["CD8_Tcell"] = df_expr[cd8_found].mean(axis=1)

    return df_sig


def safe_save_csv(df: pd.DataFrame, out_file: Path) -> None:
    """Safely saves DataFrame to CSV, handling potential Windows/Dropbox file locking."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        try:
            out_file.unlink()
        except Exception:
            pass

    try:
        df.to_csv(out_file, index=False)
    except PermissionError:
        tmp_file = out_file.with_suffix(".tmp.csv")
        df.to_csv(tmp_file, index=False)
        try:
            tmp_file.replace(out_file)
        except Exception:
            print(f"Warning: Could not overwrite {out_file.name} directly. Saved to {tmp_file.name}")


def main() -> None:
    """Main execution function for dataset loading and preparation."""
    print(f"Starting Phase 1 Feature Matrix Preparation (Project root: {rel_path(PROJECT_ROOT)})")

    # 1. Load preprocessed datasets
    df_clin, df_expr, df_genomic = load_processed_datasets()

    # 2. Extract core immune signatures
    df_sig = extract_immune_signatures(df_expr)

    # 3. Calculate Macrophage STV and M1/M2 ratio
    print("Calculating Macrophage STV and M1/M2 ratio...")
    m1_score, m2_score, m1_m2_ratio, net_stv = compute_macrophage_stv(df_expr, STV_PATH)
    df_sig["M1_score"] = m1_score
    df_sig["M2_score"] = m2_score
    df_sig["M1_M2_Ratio"] = m1_m2_ratio
    df_sig["Macrophage_STV_Score"] = net_stv

    # 4. Perform transcriptomic cell deconvolution
    print("Performing transcriptomic cell-type deconvolution...")
    df_deconv = compute_cell_deconvolution(df_expr)

    # 5. Merge all feature matrices
    print("Merging clinical, signature, Macrophage STV, and cell deconvolution features...")
    df_master = df_clin.copy()

    # Set index for merging on SAMPLE_ID
    if "SAMPLE_ID" in df_master.columns:
        df_master = df_master.set_index("SAMPLE_ID")

    # Join immune signatures & deconvolution features
    df_master = df_master.join(df_sig, how="left")
    df_master = df_master.join(df_deconv, how="left")

    # Join genomic features if present
    if not df_genomic.empty and "SAMPLE_ID" in df_genomic.columns:
        genomic_cols = [c for c in df_genomic.columns if c not in df_master.columns and c != "SAMPLE_ID"]
        df_genomic_idx = df_genomic.set_index("SAMPLE_ID")[genomic_cols]
        df_master = df_master.join(df_genomic_idx, how="left")

    # Reset index to restore SAMPLE_ID column
    df_master = df_master.reset_index()

    # 6. Save consolidated feature matrix
    out_file = OUTPUT_DIR / "feature_matrix.csv"
    safe_save_csv(df_master, out_file)

    # 7. Generate 300 DPI publication boxplots figure for presentation
    plot_file = SUBPROJECT_ROOT / "plots" / "phenotypes" / "baseline_signature_boxplots.png"
    plot_baseline_signature_boxplots(df_master, plot_file)

    print("=" * 80)
    print("FEATURE MATRIX PREPARATION & PLOTTING COMPLETE")
    print(f"Output Matrix: {rel_path(out_file)}")
    print(f"Output Figure: {rel_path(plot_file)}")
    print(f"Total Patients: {len(df_master)}")
    print(f"Total Features: {df_master.shape[1]}")
    print("=" * 80)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
