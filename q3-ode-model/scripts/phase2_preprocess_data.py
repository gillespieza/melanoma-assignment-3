#!/usr/bin/env python3
"""Phase 2: Data Pre-processing (TCGA-SKCM Melanoma)

Reads downloaded TCGA Skin Cutaneous Melanoma data and produces a single merged CSV
that the BRAF/MEK/ERK ODE simulation (Phase 3) can consume.

Steps:
  A. Parse clinical survival data      -> OS_MONTHS, SURV_STATUS
  B. Extract 12 pathway genes from RSEM -> normalise each gene by its mean
  B.1 Checkpoint signatures            -> IMPRES, PD_L1
  C. Determine BRAF & NRAS status       -> V600E/K/R/K601E = "mutant"
  D. Merge everything on SAMPLE_ID     -> melanoma_params_full.csv
"""

import contextlib
from pathlib import Path
import sys
from typing import Set, Tuple
import warnings
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

from src.utils.paths import DATA_DIR, PROJECT_ROOT, RAW_DIR, SUBPROJECT_ROOT, rel_path

Q1_SRC = PROJECT_ROOT / "q1-response-predictor" / "src"
if str(Q1_SRC) not in sys.path:
    sys.path.insert(0, str(Q1_SRC))

# Handle Q1 internal package alias for src.config.constants
try:
    import config.constants as q1_config_constants
    import sys
    sys.modules["src.config"] = sys.modules.get("src.config", q1_config_constants)
    sys.modules["src.config.constants"] = q1_config_constants
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from src.biology_constants import (
    BRAF_ACTIVATING,
    MODEL_GENES,
    NRAS_ACTIVATING,
    SIGNATURE_GENES,
)
from src.utils.logging import TeeStream

# Import Q1 checkpoint signature helpers if available
try:
    from signatures import compute_impres, compute_pd_l1  # type: ignore # noqa: E402
    HAVE_SIGNATURES = True
except (ImportError, ModuleNotFoundError) as err:
    warnings.warn(f"Could not import Q1 signatures.py ({err}); IMPRES/PD_L1 columns will fallback to NaN.")
    HAVE_SIGNATURES = False

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "q3_phase2_preprocess.log"
TCGA_DIR = RAW_DIR / "skcm_tcga_pan_can_atlas_2018"
OUT_DIR = SUBPROJECT_ROOT / "data"

CLINICAL_PATIENT_FILE = TCGA_DIR / "data_clinical_patient.txt"
CLINICAL_SAMPLE_FILE = TCGA_DIR / "data_clinical_sample.txt"
MRNA_FILE = TCGA_DIR / "data_mrna_seq_v2_rsem.txt"
MUTATION_FILE = TCGA_DIR / "data_mutations.txt"
OUTPUT_FILE = OUT_DIR / "melanoma_params_full.csv"


# ---------------------------------------------------------------------------
# Processing Helpers
# ---------------------------------------------------------------------------

def load_clinical_survival(patient_file: Path, sample_file: Path) -> pd.DataFrame:
    """Parse and clean clinical patient and sample survival metadata."""
    print("\n[Step A] Loading clinical survival data...")
    if not patient_file.exists() or not sample_file.exists():
        raise FileNotFoundError(f"Missing clinical data files in {rel_path(TCGA_DIR)}")

    clin_patient = pd.read_csv(patient_file, sep="\t", comment="#", low_memory=False)
    clin_patient = clin_patient[["PATIENT_ID", "OS_MONTHS", "OS_STATUS"]].copy()

    clin_sample = pd.read_csv(sample_file, sep="\t", comment="#", low_memory=False)
    clin_sample = clin_sample[["PATIENT_ID", "SAMPLE_ID"]].copy()

    clinical = clin_sample.merge(clin_patient, on="PATIENT_ID", how="left")
    clinical["OS_MONTHS"] = pd.to_numeric(clinical["OS_MONTHS"], errors="coerce")
    clinical = clinical.dropna(subset=["OS_STATUS", "OS_MONTHS"])
    clinical = clinical[clinical["OS_STATUS"].astype(str).str.strip() != ""]

    mask_keep = (
        (clinical["OS_MONTHS"] > 1) & (clinical["OS_STATUS"] == "0:LIVING")
    ) | (clinical["OS_STATUS"] == "1:DECEASED")
    clinical = clinical[mask_keep].copy()
    clinical["SURV_STATUS"] = (clinical["OS_STATUS"] == "1:DECEASED").astype(int)

    print(f"  {len(clinical)} samples with valid survival data")
    print(f"  Events (deceased): {clinical['SURV_STATUS'].sum()} | Censored (living): {(clinical['SURV_STATUS'] == 0).sum()}")
    return clinical


def extract_checkpoint_signatures(mrna: pd.DataFrame) -> pd.DataFrame:
    """Extract raw RSEM expression for signature genes and compute IMPRES / PD-L1 scores."""
    sig_genes_present = [g for g in SIGNATURE_GENES if g in set(mrna["GENE"])]
    sig_genes_missing = [g for g in SIGNATURE_GENES if g not in set(mrna["GENE"])]

    if sig_genes_missing:
        print(f"  NOTE: signature genes missing from RSEM: {sig_genes_missing}")

    sig_expr_raw = (
        mrna[mrna["GENE"].isin(sig_genes_present)]
        .groupby("GENE").mean(numeric_only=True).T
    )
    sig_expr_raw.index.name = "SAMPLE_ID"

    if HAVE_SIGNATURES and len(sig_genes_present) > 0:
        checkpoint_sig = pd.DataFrame({
            "IMPRES": compute_impres(sig_expr_raw),
            "PD_L1": compute_pd_l1(sig_expr_raw),
        })
    else:
        checkpoint_sig = pd.DataFrame({"IMPRES": np.nan, "PD_L1": np.nan}, index=sig_expr_raw.index)

    checkpoint_sig.index.name = "SAMPLE_ID"
    checkpoint_sig = checkpoint_sig.reset_index()
    print(f"  Checkpoint signatures computed for {checkpoint_sig.shape[0]} samples")
    return checkpoint_sig


def extract_model_expression(mrna_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract and mean-normalise 12 model genes, compute CYT score, and extract checkpoint signatures."""
    print("\n[Step B] Extracting 12 pathway genes from RSEM file...")
    if not mrna_file.exists():
        raise FileNotFoundError(f"mRNA seq RSEM file not found at {rel_path(mrna_file)}")

    mrna = pd.read_csv(mrna_file, sep="\t", low_memory=False)
    mrna = mrna.drop(columns=["Entrez_Gene_Id"])
    mrna = mrna.rename(columns={"Hugo_Symbol": "GENE"})

    missing = [g for g in MODEL_GENES if g not in set(mrna["GENE"])]
    if missing:
        raise KeyError(f"Required model genes not found in RSEM file: {missing}")

    checkpoint_sig = extract_checkpoint_signatures(mrna)

    expr = mrna[mrna["GENE"].isin(MODEL_GENES)].groupby("GENE").mean(numeric_only=True)
    expr = expr.loc[MODEL_GENES].T
    expr.index.name = "SAMPLE_ID"
    expr = expr.reset_index()

    print("  Normalising each gene by its mean across all patients...")
    for gene in MODEL_GENES:
        expr[gene] = expr[gene].clip(lower=0)
        col_mean = expr[gene].mean()
        if col_mean > 0:
            expr[gene] = expr[gene] / col_mean
        else:
            print(f"  WARNING: mean of {gene} is 0 - skipping normalisation")

    expr["CYT"] = (expr["GZMA"] + expr["PRF1"]) / 2.0
    print(f"  Expression matrix: {expr.shape[0]} samples x {len(MODEL_GENES)} genes (+ CYT)")
    return expr, checkpoint_sig


def determine_mutation_statuses(mutation_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse mutation MAF file for activating BRAF and NRAS variants."""
    print("\n[Step C] Determining BRAF and NRAS mutation status...")
    if not mutation_file.exists():
        raise FileNotFoundError(f"Mutations file not found at {rel_path(mutation_file)}")

    mut = pd.read_csv(
        mutation_file, sep="\t", comment="#", low_memory=False,
        usecols=["Hugo_Symbol", "HGVSp_Short", "Tumor_Sample_Barcode"],
    )

    # BRAF Status
    braf = mut[mut["Hugo_Symbol"] == "BRAF"].copy()
    braf["is_activating"] = braf["HGVSp_Short"].isin(BRAF_ACTIVATING)

    braf_status = (
        braf.groupby("Tumor_Sample_Barcode")["is_activating"].max()
        .rename("BRAF_MUT").reset_index()
        .rename(columns={"Tumor_Sample_Barcode": "SAMPLE_ID"})
    )
    braf_status["BRAF_MUT"] = braf_status["BRAF_MUT"].astype(int)

    variant_map = (
        braf[braf["is_activating"]]
        .groupby("Tumor_Sample_Barcode")["HGVSp_Short"].first()
        .rename("BRAF_VARIANT").reset_index()
        .rename(columns={"Tumor_Sample_Barcode": "SAMPLE_ID"})
    )

    # NRAS Status
    nras = mut[mut["Hugo_Symbol"] == "NRAS"].copy()
    nras["is_activating"] = nras["HGVSp_Short"].isin(NRAS_ACTIVATING)
    nras_status = (
        nras.groupby("Tumor_Sample_Barcode")["is_activating"].max()
        .rename("NRAS_MUT").reset_index()
        .rename(columns={"Tumor_Sample_Barcode": "SAMPLE_ID"})
    )
    nras_status["NRAS_MUT"] = nras_status["NRAS_MUT"].astype(int)

    print(f"  BRAF-mutated samples: {braf_status['BRAF_MUT'].sum()}")
    print(f"  NRAS-mutated samples: {nras_status['NRAS_MUT'].sum()}")
    return braf_status, variant_map, nras_status


def merge_phase2_data(
    clinical: pd.DataFrame,
    expr: pd.DataFrame,
    braf_status: pd.DataFrame,
    variant_map: pd.DataFrame,
    nras_status: pd.DataFrame,
    checkpoint_sig: pd.DataFrame,
) -> pd.DataFrame:
    """Merge survival, gene expression, mutation, and signature datasets."""
    print("\n[Step D] Merging datasets on SAMPLE_ID...")
    data = clinical.merge(expr, on="SAMPLE_ID", how="inner")
    data = data.merge(braf_status, on="SAMPLE_ID", how="left")
    data = data.merge(variant_map, on="SAMPLE_ID", how="left")
    data = data.merge(nras_status, on="SAMPLE_ID", how="left")
    data = data.merge(checkpoint_sig, on="SAMPLE_ID", how="left")

    data["BRAF_MUT"] = data["BRAF_MUT"].fillna(0).astype(int)
    data["NRAS_MUT"] = data["NRAS_MUT"].fillna(0).astype(int)
    data["BRAF_VARIANT"] = data["BRAF_VARIANT"].fillna("WT")
    data["MAPK_DRIVEN"] = ((data["BRAF_MUT"] == 1) | (data["NRAS_MUT"] == 1)).astype(int)

    data = data.dropna(subset=MODEL_GENES)
    n_braf = int(data["BRAF_MUT"].sum())
    n_nras = int(((data["NRAS_MUT"] == 1) & (data["BRAF_MUT"] == 0)).sum())
    n_wt = int((data["MAPK_DRIVEN"] == 0).sum())

    print(f"  Final merged dataset: {len(data)} patients")
    print(f"    BRAF-mutant: {n_braf} | NRAS-mutant: {n_nras} | MAPK-quiet WT: {n_wt}")
    return data


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Main execution entry point for Q3 Phase 2 Data Preprocessing."""
    print("=" * 60)
    print("  Phase 2: Data Pre-processing (TCGA-SKCM Melanoma)")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clinical = load_clinical_survival(CLINICAL_PATIENT_FILE, CLINICAL_SAMPLE_FILE)
    expr, checkpoint_sig = extract_model_expression(MRNA_FILE)
    braf_status, variant_map, nras_status = determine_mutation_statuses(MUTATION_FILE)

    data = merge_phase2_data(clinical, expr, braf_status, variant_map, nras_status, checkpoint_sig)

    data.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved merged dataset to: {rel_path(OUTPUT_FILE)}")
    print(f"  Shape: {data.shape[0]} rows x {data.shape[1]} columns")

    print("\n" + "=" * 60)
    print("  Phase 2 COMPLETE - Ready for Phase 3 (ODE Simulation)")
    print("=" * 60)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
