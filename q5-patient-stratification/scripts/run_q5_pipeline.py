#!/usr/bin/env python3
"""Master pipeline orchestrator for Q5 Patient Stratification.

Executes scripts 01 through 07 in sequence and logs overall pipeline status and timing.
"""

import contextlib
import importlib
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SUBPROJECT_ROOT = SCRIPT_DIR.parent
for parent in [SCRIPT_DIR] + list(SCRIPT_DIR.parents):
    if (parent / "src").is_dir() and (parent / "data").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from src.styles import set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import PROJECT_ROOT, rel_path

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "q5_pipeline.log"

SCRIPTS = [
    ("01_load_and_prepare", "01_load_and_prepare"),
    ("02_feature_analysis", "02_feature_analysis"),
    ("03_cluster_patients", "03_cluster_patients"),
    ("04_phenotype_characterisation", "04_phenotype_characterisation"),
    ("05_subgroup_models", "05_subgroup_models"),
    ("06_clinical_utility", "06_clinical_utility"),
    ("07_treatability_scoring", "07_treatability_scoring"),
]


def main() -> None:
    """Orchestrate end-to-end Q5 pipeline execution."""
    print("=" * 80)
    print("STARTING Q5 PATIENT STRATIFICATION PIPELINE")
    print(f"Project Root: {rel_path(PROJECT_ROOT)}")
    print(f"Subproject Root: {rel_path(SUBPROJECT_ROOT)}")
    print("=" * 80)

    for script_label, mod_name in SCRIPTS:
        print(f"\n--- Executing Step: {script_label} ---")
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "main"):
                mod.main()
            print(f"[OK] Step {script_label} completed successfully.")
        except Exception as err:
            print(f"[FAIL] Step {script_label} failed with error: {err}")
            raise err

    print("\n" + "=" * 80)
    print("Q5 PIPELINE EXECUTED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
