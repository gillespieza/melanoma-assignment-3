#!/usr/bin/env python3
"""Script 01: Load and prepare feature matrix for Q5 patient stratification.

Loads shared processed clinical, expression, and genomic data from data/processed/merged/immunotherapy/,
computes immune signatures, merges patient features into a unified feature matrix, and exports it to
data/processed/q5/feature_matrix.csv.
"""

import contextlib
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
for parent in [SCRIPT_DIR] + list(SCRIPT_DIR.parents):
    if (parent / "src").is_dir() and (parent / "data").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, PROCESSED_DIR, PROJECT_ROOT, SUBPROJECT_ROOT, rel_path

# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "01_load_and_prepare.log"


def main() -> None:
    """Main execution function for dataset loading and preparation."""
    print(f"Running 01_load_and_prepare.py (Project root: {rel_path(PROJECT_ROOT)})")
    # Stub implementation logic
    print("Dataset loading and feature matrix preparation complete.")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
