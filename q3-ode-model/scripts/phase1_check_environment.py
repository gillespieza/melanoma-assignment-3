"""
Phase 1: Environment Check
==========================
Checks that all required Python libraries are installed, that the TCGA-SKCM
(melanoma) data files are reachable, and reports the number of CPU cores
available for the parallel ODE simulation in Phase 3.

Run this FIRST to confirm the environment is ready.

Usage:
    python phase1_check_environment.py
"""

import os
import sys
import multiprocessing

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # q3-ode-model
ASSIGN_DIR = os.path.dirname(BASE_DIR)                                     # melanoma-assignment-3
TCGA_DIR   = os.path.join(ASSIGN_DIR, "q1-response-predictor", "data", "raw",
                          "skcm_tcga_pan_can_atlas_2018")

REQUIRED_FILES = [
    ("data_clinical_patient.txt", "Survival (OS_MONTHS, OS_STATUS)"),
    ("data_clinical_sample.txt",  "SAMPLE_ID ↔ PATIENT_ID mapping"),
    ("data_mrna_seq_v2_rsem.txt", "RSEM gene expression (20k genes)"),
    ("data_mutations.txt",        "BRAF V600E/K mutation status"),
    ("data_rppa.txt",             "RPPA phospho-protein (pERK validation)"),
]

print("=" * 60)
print("  Melanoma BRAF/MEK/ERK ODE Project — Environment Check")
print("=" * 60)
print(f"\nPython version: {sys.version}\n")

required = [
    ("numpy",           "Array maths — backbone of ODE solver"),
    ("pandas",          "Data tables — loads TCGA files"),
    ("scipy.integrate", "ODE solver (odeint)"),
    ("lifelines",       "Kaplan-Meier & Cox PH survival analysis"),
    ("matplotlib",      "Plotting KM curves and correlations"),
    ("sklearn",         "ML baseline (Phase 6)"),
]

all_ok = True
for module, description in required:
    try:
        __import__(module, fromlist=[""])
        top = module.split(".")[0]
        version = getattr(__import__(top), "__version__", "version unknown")
        print(f"  [OK]   {module:<18} v{version:<10}  — {description}")
    except ImportError:
        print(f"  [FAIL] {module:<18} NOT INSTALLED  — {description}")
        all_ok = False

print()
try:
    from scipy.integrate import odeint  # noqa: F401
    print("  [OK]   scipy.integrate.odeint confirmed available")
except ImportError:
    print("  [FAIL] scipy.integrate.odeint NOT available")
    all_ok = False

# ─── Data files ───────────────────────────────────────────────────────────────
print(f"\nTCGA-SKCM data directory:\n  {TCGA_DIR}\n")
for fname, desc in REQUIRED_FILES:
    path = os.path.join(TCGA_DIR, fname)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1e6
        print(f"  [OK]   {fname:<28} {size_mb:8.1f} MB  — {desc}")
    else:
        print(f"  [FAIL] {fname:<28} MISSING        — {desc}")
        all_ok = False

# ─── CPU cores ────────────────────────────────────────────────────────────────
n_cores = multiprocessing.cpu_count()
print(f"\n  CPU cores available: {n_cores}")
print(f"  (Phase 3 will use up to {n_cores} parallel workers for ~443 patients)\n")

print("=" * 60)
if all_ok:
    print("  ALL CHECKS PASSED — Ready for Phase 2")
else:
    print("  SOME CHECKS FAILED — fix the items marked [FAIL] above")
print("=" * 60)
sys.exit(0 if all_ok else 1)
