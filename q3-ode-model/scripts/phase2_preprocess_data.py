"""
Phase 2: Data Pre-processing (TCGA-SKCM Melanoma)
=================================================
Reads the already-downloaded TCGA Skin Cutaneous Melanoma data and produces a
single merged CSV that the BRAF/MEK/ERK ODE simulation (Phase 3) can read.

Steps:
  A. Parse clinical survival data      → OS_MONTHS, SURV_STATUS
  B. Extract 10 pathway genes from RSEM → normalise each gene by its mean
  C. Determine BRAF mutation status    → V600E/K/R/K601E = "mutant"
  D. Merge everything on SAMPLE_ID     → melanoma_params_full.csv

Usage:
    python phase2_preprocess_data.py

Output:
    data/melanoma_params_full.csv
"""

import os
import sys
import pandas as pd
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSIGN_DIR = os.path.dirname(BASE_DIR)
TCGA_DIR   = os.path.join(ASSIGN_DIR, "q1-response-predictor", "data", "raw",
                          "skcm_tcga_pan_can_atlas_2018")
OUT_DIR    = os.path.join(BASE_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

CLINICAL_PATIENT_FILE = os.path.join(TCGA_DIR, "data_clinical_patient.txt")
CLINICAL_SAMPLE_FILE  = os.path.join(TCGA_DIR, "data_clinical_sample.txt")
MRNA_FILE             = os.path.join(TCGA_DIR, "data_mrna_seq_v2_rsem.txt")
MUTATION_FILE         = os.path.join(TCGA_DIR, "data_mutations.txt")
OUTPUT_FILE           = os.path.join(OUT_DIR, "melanoma_params_full.csv")

# The 10 genes the BRAF/MEK/ERK/Tumour/Immune ODE needs:
#   Signalling cascade
MODEL_GENES = [
    "BRAF",     # RAF kinase (V600E = constitutively active)
    "MAP2K1",   # MEK1
    "MAP2K2",   # MEK2
    "MAPK1",    # ERK2
    "MAPK3",    # ERK1
    "CDKN2A",   # p16 tumour suppressor
    "MKI67",    # proliferation marker
    "CD8A",     # cytotoxic T-cell infiltration
    "PRF1",     # perforin (immune effector)
    "GZMA",     # granzyme A (immune effector)
]

# Activating BRAF mutations that make the tumour vemurafenib-sensitive.
# (Class I V600 mutations + the closely-related K601E.)
BRAF_ACTIVATING = {"p.V600E", "p.V600K", "p.V600R", "p.V600D", "p.K601E"}

# Activating NRAS mutations. These also switch the MAPK pathway ON (high pERK),
# but are NOT vemurafenib-sensitive — they help set the baseline oncogenic
# drive without conferring BRAF-inhibitor response.
NRAS_ACTIVATING = {"p.Q61R", "p.Q61K", "p.Q61L", "p.Q61H",
                   "p.G12D", "p.G12R", "p.G12C", "p.G13R", "p.G13D"}

print("=" * 60)
print("  Phase 2: Data Pre-processing (TCGA-SKCM Melanoma)")
print("=" * 60)

# ─── Step A: Clinical survival data ──────────────────────────────────────────
print("\n[Step A] Loading clinical survival data...")

clin_patient = pd.read_csv(CLINICAL_PATIENT_FILE, sep="\t", comment="#", low_memory=False)
clin_patient = clin_patient[["PATIENT_ID", "OS_MONTHS", "OS_STATUS"]].copy()

clin_sample = pd.read_csv(CLINICAL_SAMPLE_FILE, sep="\t", comment="#", low_memory=False)
clin_sample = clin_sample[["PATIENT_ID", "SAMPLE_ID"]].copy()

clinical = clin_sample.merge(clin_patient, on="PATIENT_ID", how="left")

# Numeric survival time; drop rows with no survival info
clinical["OS_MONTHS"] = pd.to_numeric(clinical["OS_MONTHS"], errors="coerce")
clinical = clinical.dropna(subset=["OS_STATUS", "OS_MONTHS"])
clinical = clinical[clinical["OS_STATUS"].astype(str).str.strip() != ""]

# Drop living patients with essentially no follow-up (<=1 month, uninformative)
mask_keep = (
    (clinical["OS_MONTHS"] > 1) & (clinical["OS_STATUS"] == "0:LIVING")
) | (clinical["OS_STATUS"] == "1:DECEASED")
clinical = clinical[mask_keep].copy()

# Binary event: 1 = deceased
clinical["SURV_STATUS"] = (clinical["OS_STATUS"] == "1:DECEASED").astype(int)

print(f"  {len(clinical)} samples with valid survival data")
print(f"  Events (deceased): {clinical['SURV_STATUS'].sum()}  |  "
      f"Censored (living): {(clinical['SURV_STATUS'] == 0).sum()}")

# ─── Step B: Expression — extract 10 pathway genes ───────────────────────────
print("\n[Step B] Extracting 10 pathway genes from RSEM file (~60 MB)...")

mrna = pd.read_csv(MRNA_FILE, sep="\t", low_memory=False)
# Column 1 = Hugo_Symbol, column 2 = Entrez_Gene_Id, rest = samples
mrna = mrna.drop(columns=["Entrez_Gene_Id"])
mrna = mrna.rename(columns={"Hugo_Symbol": "GENE"})

missing = [g for g in MODEL_GENES if g not in set(mrna["GENE"])]
if missing:
    print(f"  ERROR: genes not found in RSEM file: {missing}")
    sys.exit(1)

# Some Hugo symbols appear more than once — collapse duplicates by mean
expr = mrna[mrna["GENE"].isin(MODEL_GENES)].groupby("GENE").mean(numeric_only=True)
expr = expr.loc[MODEL_GENES]                      # fix gene order
expr = expr.T                                     # rows = samples, cols = genes
expr.index.name = "SAMPLE_ID"
expr = expr.reset_index()

# RSEM cannot be negative; clip, then normalise each gene by its mean (=> mean 1)
print("  Normalising each gene by its mean across all patients (mean -> 1.0)...")
for gene in MODEL_GENES:
    expr[gene] = expr[gene].clip(lower=0)
    col_mean = expr[gene].mean()
    if col_mean > 0:
        expr[gene] = expr[gene] / col_mean
    else:
        print(f"  WARNING: mean of {gene} is 0 — skipping normalisation")

print(f"  Expression matrix: {expr.shape[0]} samples × {len(MODEL_GENES)} genes")

# ─── Step C: BRAF mutation status ────────────────────────────────────────────
print("\n[Step C] Determining BRAF mutation status from mutations file (~550 MB)...")

# Only read the columns we need to keep memory low
mut = pd.read_csv(
    MUTATION_FILE, sep="\t", comment="#", low_memory=False,
    usecols=["Hugo_Symbol", "HGVSp_Short", "Tumor_Sample_Barcode"],
)
braf = mut[mut["Hugo_Symbol"] == "BRAF"].copy()
braf["is_activating"] = braf["HGVSp_Short"].isin(BRAF_ACTIVATING)

# Per sample: mutant if it carries any activating BRAF variant
braf_status = (
    braf.groupby("Tumor_Sample_Barcode")["is_activating"].max()
    .rename("BRAF_MUT").reset_index()
    .rename(columns={"Tumor_Sample_Barcode": "SAMPLE_ID"})
)
braf_status["BRAF_MUT"] = braf_status["BRAF_MUT"].astype(int)

# Record the specific activating variant (first one) for reference
variant_map = (
    braf[braf["is_activating"]]
    .groupby("Tumor_Sample_Barcode")["HGVSp_Short"].first()
    .rename("BRAF_VARIANT").reset_index()
    .rename(columns={"Tumor_Sample_Barcode": "SAMPLE_ID"})
)

n_v600e = (braf["HGVSp_Short"] == "p.V600E").sum()
n_v600k = (braf["HGVSp_Short"] == "p.V600K").sum()
print(f"  BRAF-mutated samples (activating): {braf_status['BRAF_MUT'].sum()}")
print(f"  (V600E={n_v600e}, V600K={n_v600k}, plus V600R/D/K601E)")

# NRAS status: sets baseline MAPK drive but confers NO vemurafenib response
nras = mut[mut["Hugo_Symbol"] == "NRAS"].copy()
nras["is_activating"] = nras["HGVSp_Short"].isin(NRAS_ACTIVATING)
nras_status = (
    nras.groupby("Tumor_Sample_Barcode")["is_activating"].max()
    .rename("NRAS_MUT").reset_index()
    .rename(columns={"Tumor_Sample_Barcode": "SAMPLE_ID"})
)
nras_status["NRAS_MUT"] = nras_status["NRAS_MUT"].astype(int)
print(f"  NRAS-mutated samples (activating): {nras_status['NRAS_MUT'].sum()}")

# ─── Step D: Merge everything ────────────────────────────────────────────────
print("\n[Step D] Merging survival + expression + BRAF status on SAMPLE_ID...")

data = clinical.merge(expr, on="SAMPLE_ID", how="inner")
data = data.merge(braf_status, on="SAMPLE_ID", how="left")
data = data.merge(variant_map, on="SAMPLE_ID", how="left")
data = data.merge(nras_status, on="SAMPLE_ID", how="left")

# Samples with no mutation record = wild-type for our purposes
data["BRAF_MUT"] = data["BRAF_MUT"].fillna(0).astype(int)
data["NRAS_MUT"] = data["NRAS_MUT"].fillna(0).astype(int)
data["BRAF_VARIANT"] = data["BRAF_VARIANT"].fillna("WT")

# MAPK_DRIVEN = pathway is constitutively ON (BRAF or NRAS activating mutation).
# Drives baseline pERK. Vemurafenib response is governed separately by BRAF_MUT.
data["MAPK_DRIVEN"] = ((data["BRAF_MUT"] == 1) | (data["NRAS_MUT"] == 1)).astype(int)

data = data.dropna(subset=MODEL_GENES)

n_braf = int(data["BRAF_MUT"].sum())
n_nras = int(((data["NRAS_MUT"] == 1) & (data["BRAF_MUT"] == 0)).sum())
n_wt = int((data["MAPK_DRIVEN"] == 0).sum())
print(f"  Final merged dataset: {len(data)} patients")
print(f"    BRAF-mutant: {n_braf}  |  NRAS-mutant (BRAF-WT): {n_nras}  |  "
      f"MAPK-quiet WT: {n_wt}")

# ─── Save ─────────────────────────────────────────────────────────────────────
data.to_csv(OUTPUT_FILE, index=False)
print(f"\n  Saved: {OUTPUT_FILE}")
print(f"  Shape: {data.shape[0]} rows × {data.shape[1]} columns")

print("\n" + "=" * 60)
print("  Phase 2 COMPLETE — Ready for Phase 3 (ODE Simulation)")
print("=" * 60)
