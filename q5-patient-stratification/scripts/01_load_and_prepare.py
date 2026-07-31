#!/usr/bin/env python3
"""Script 01: Load and prepare feature matrix for Q5 patient stratification.

Loads preprocessed clinical, expression, and genomic data from data/processed/merged/,
computes immune signatures, calculates Macrophage STV (M1/M2 ratio) and cell deconvolution,
and exports two unified feature matrices to data/processed/q5/:

  - feature_matrix.csv       (ICI-treated patients only, N≈326) → Phases 2, 5, 6
  - feature_matrix_full.csv  (all melanoma patients, N≈699)      → Phases 3, 4, 7, Script 08
"""

import contextlib
from pathlib import Path
import sys
from typing import Tuple
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
from q5_constants import IMMUNE_SIGNATURE_MARKERS
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, PROCESSED_DIR, PROJECT_ROOT, rel_path

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "01_load_and_prepare.log"

# ICI-treated only (N≈326) — Phases 2, 5, 6 require RESPONSE_BINARY labels
INPUT_DIR_ICI = PROCESSED_DIR / "merged" / "immunotherapy"
# Full melanoma cohort (N≈699) — Phases 3, 4, 7 need no response label
INPUT_DIR_FULL = PROCESSED_DIR / "merged" / "full"

OUTPUT_DIR = PROCESSED_DIR / "q5"
STV_PATH = DATA_DIR / "config" / "m1_m2_stv.csv"


def load_processed_datasets(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate preprocessed multi-modal clinical, transcriptomic, and genomic datasets.

    Args:
        input_dir: Directory containing clin_merged.csv, expr_merged.csv, and merged_genomic.csv.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - df_clin:    Patient clinical metadata, OS, and response labels.
            - df_expr:    Transposed expression matrix (samples as rows, genes as columns).
            - df_genomic: Driver mutation indicators (BRAF, NRAS, NF1, Triple-WT) and TMB.

    Raises:
        FileNotFoundError: If essential clinical or expression files are missing from input_dir.
    """
    clin_file    = input_dir / "clin_merged.csv"
    expr_file    = input_dir / "expr_merged.csv"
    genomic_file = input_dir / "merged_genomic.csv"

    if not clin_file.exists():
        raise FileNotFoundError(
            f"Missing clinical data file at {rel_path(clin_file)}. "
            "Ensure preprocessing pipeline has run."
        )
    if not expr_file.exists():
        raise FileNotFoundError(
            f"Missing expression data file at {rel_path(expr_file)}. "
            "Ensure preprocessing pipeline has run."
        )

    df_clin = pd.read_csv(clin_file)
    print(f"  Loaded clinical dataset:  {len(df_clin)} patients from {rel_path(clin_file)}")

    df_expr = pd.read_csv(expr_file)
    if "SAMPLE_ID" in df_expr.columns:
        df_expr = df_expr.set_index("SAMPLE_ID")
    print(f"  Loaded expression matrix: {df_expr.shape[0]} samples x {df_expr.shape[1]} genes")

    if genomic_file.exists():
        df_genomic = pd.read_csv(genomic_file)
        # Deduplicate columns by stripping pandas duplicate suffixes (.1, .2)
        df_genomic = df_genomic.loc[
            :, ~df_genomic.columns.str.replace(r"\.\d+$", "", regex=True).duplicated(keep="first")
        ]
        print(f"  Loaded genomic dataset:   {len(df_genomic)} rows from {rel_path(genomic_file)}")
    else:
        df_genomic = pd.DataFrame()
        print("  Genomic file not found; using empty DataFrame fallback.")

    return df_clin, df_expr, df_genomic


def extract_immune_signatures(df_expr: pd.DataFrame) -> pd.DataFrame:
    """Compute core transcriptomic immune signatures from bulk RNA-seq expression profiles.

    Attempts to use the shared Q1 signature extraction module; falls back to local
    proxy calculations if unavailable.

    Signatures computed:
    - **PD-L1 Expression**: Single-gene marker `CD274`.
    - **Tumour Inflammation Signature (TIS)**: Ayers et al. 18-gene IFN-gamma score.
    - **Cytolytic Index (CYT)**: Rooney et al. mean of `GZMA` and `PRF1`.
    - **IFN-gamma Response**: Average across `STAT1`, `CXCL9`, `CXCL10` signalling genes.
    - **CD8+ T-Cell Density**: Average of `CD8A` and `CD8B`.

    Args:
        df_expr: Normalised gene expression matrix (samples x genes).

    Returns:
        pd.DataFrame: Immune signature scores indexed by SAMPLE_ID.
    """
    q1_root_path = PROJECT_ROOT / "q1-response-predictor"
    q1_src_path  = q1_root_path / "src"

    if str(q1_root_path) not in sys.path:
        sys.path.insert(0, str(q1_root_path))
    if str(q1_src_path) not in sys.path:
        sys.path.insert(0, str(q1_src_path))

    try:
        from signatures import extract_all_signatures
        df_sig = extract_all_signatures(df_expr)
        print("  Extracted immune signatures via shared Q1 module.")
    except (ImportError, ModuleNotFoundError, AttributeError) as err:
        print(f"  Q1 module fallback (reason: {err}). Computing signatures locally.")
        df_sig = pd.DataFrame(index=df_expr.index)

        if "CD274" in df_expr.columns:
            df_sig["PD_L1"] = df_expr["CD274"]
        elif "PDL1" in df_expr.columns:
            df_sig["PD_L1"] = df_expr["PDL1"]

        for sig_name, markers in IMMUNE_SIGNATURE_MARKERS.items():
            found = [g for g in markers if g in df_expr.columns]
            if found:
                df_sig[sig_name] = df_expr[found].mean(axis=1)

    return df_sig


def safe_save_csv(df: pd.DataFrame, out_file: Path) -> None:
    """Safely save a DataFrame to CSV, handling potential Windows/Dropbox file locking."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        try:
            out_file.unlink()
        except (PermissionError, OSError):
            pass
    try:
        df.to_csv(out_file, index=False)
    except (PermissionError, OSError):
        tmp_file = out_file.with_suffix(".tmp.csv")
        df.to_csv(tmp_file, index=False)
        try:
            tmp_file.replace(out_file)
        except (PermissionError, OSError):
            print(f"Warning: could not overwrite {out_file.name} directly. Saved to {tmp_file.name}")


def build_feature_matrix(input_dir: Path) -> pd.DataFrame:
    """Build the full multi-modal patient feature matrix from a given input directory.

    Orchestrates dataset loading, immune signature extraction, Macrophage STV calculation,
    transcriptomic deconvolution, and feature merging into a single per-patient DataFrame.

    Args:
        input_dir: Path to a merged cohort directory containing clin_merged.csv,
                   expr_merged.csv, and merged_genomic.csv.

    Returns:
        pd.DataFrame: Unified feature matrix (clinical + immune signatures + STV +
                      deconvolution + genomic mutations) for all patients in input_dir.
    """
    print(f"\n--- Building feature matrix from {rel_path(input_dir)} ---")
    df_clin, df_expr, df_genomic = load_processed_datasets(input_dir)

    df_sig = extract_immune_signatures(df_expr)

    print("  Calculating Macrophage STV and M1/M2 ratio...")
    m1_score, m2_score, m1_m2_ratio, net_stv = compute_macrophage_stv(df_expr, STV_PATH)
    df_sig["M1_score"]             = m1_score
    df_sig["M2_score"]             = m2_score
    df_sig["M1_M2_Ratio"]          = m1_m2_ratio
    df_sig["Macrophage_STV_Score"] = net_stv

    print("  Performing transcriptomic cell-type deconvolution...")
    df_deconv = compute_cell_deconvolution(df_expr)

    print("  Merging clinical, signature, STV, and deconvolution features...")
    df_master = df_clin.copy()
    if "SAMPLE_ID" in df_master.columns:
        df_master = df_master.set_index("SAMPLE_ID")

    df_master = df_master.join(df_sig,   how="left")
    df_master = df_master.join(df_deconv, how="left")

    if not df_genomic.empty and "SAMPLE_ID" in df_genomic.columns:
        genomic_cols = [c for c in df_genomic.columns if c not in df_master.columns and c != "SAMPLE_ID"]
        df_genomic_idx = df_genomic.set_index("SAMPLE_ID")[genomic_cols]
        df_master = df_master.join(df_genomic_idx, how="left")

    df_master = df_master.reset_index()
    print(f"  Result: {len(df_master)} patients x {df_master.shape[1]} features")
    return df_master


def main() -> None:
    """Main execution function for feature matrix preparation."""
    print(f"Starting Phase 1 Feature Matrix Preparation (Project root: {rel_path(PROJECT_ROOT)})")

    # ------------------------------------------------------------------
    # Matrix 1: ICI-treated only (N≈326) — Phases 2, 5, 6
    # These phases train response prediction models and run DCA;
    # they require RESPONSE_BINARY labels available only for ICI cohorts.
    # ------------------------------------------------------------------
    df_ici  = build_feature_matrix(INPUT_DIR_ICI)
    out_ici = OUTPUT_DIR / "feature_matrix.csv"
    safe_save_csv(df_ici, out_ici)

    plot_file_ici = SUBPROJECT_ROOT / "plots" / "phenotypes" / "baseline_response_violins.png"
    plot_baseline_signature_boxplots(df_ici, plot_file_ici)

    # ------------------------------------------------------------------
    # Matrix 2: Full cohort (N≈699) — Phases 3, 4, 7, Script 08
    # Includes all melanoma patients regardless of treatment type.
    # RESPONSE_BINARY is NaN for the 373 non-ICI TCGA-SKCM patients.
    # ------------------------------------------------------------------
    df_full  = build_feature_matrix(INPUT_DIR_FULL)
    out_full = OUTPUT_DIR / "feature_matrix_full.csv"
    safe_save_csv(df_full, out_full)

    print("=" * 80)
    print("FEATURE MATRIX PREPARATION & PLOTTING COMPLETE")
    print(f"ICI-only matrix  (Phases 2/5/6): {rel_path(out_ici)}  "
          f"[{len(df_ici)} patients, {df_ici.shape[1]} features]")
    print(f"Full matrix      (Phases 3/4/7): {rel_path(out_full)} "
          f"[{len(df_full)} patients, {df_full.shape[1]} features]")
    print(f"Violin plot (ICI cohort)        : {rel_path(plot_file_ici)}")
    print("=" * 80)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
