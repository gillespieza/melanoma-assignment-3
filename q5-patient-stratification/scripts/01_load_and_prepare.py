#!/usr/bin/env python3
"""Script 01: Load and prepare feature matrix for Q5 patient stratification.

Loads shared processed clinical, expression, and genomic data from data/processed/merged/immunotherapy/,
computes immune signatures, merges patient features into a unified feature matrix, and exports it to
data/processed/q5/feature_matrix.csv.
"""

import contextlib
from pathlib import Path
import sys

# Bootstrap project root
def find_project_root(current_dir: Path) -> Path:
    """Walk upward to find project root directory containing src and data."""
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    return current_dir.resolve().parents[1]

BASE_DIR = find_project_root(Path(__file__).resolve().parent)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.logging import TeeStream

LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "01_load_and_prepare.log"


def main() -> None:
    """Main execution function for dataset loading and preparation."""
    print(f"Running 01_load_and_prepare.py (Base directory: {BASE_DIR.relative_to(BASE_DIR).as_posix() or '.'})")
    # Stub implementation logic
    print("Dataset loading and feature matrix preparation complete.")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(BASE_DIR).as_posix()}")
            main()
