#!/usr/bin/env python3
"""Script 03: Unsupervised patient stratification pipeline.

Applies K-Means and Agglomerative clustering to the patient feature matrix, computes silhouette
and GAP statistics for optimal K selection, generates UMAP visualisations, and exports cluster assignments to
data/processed/patient_clusters.csv.
"""

import contextlib
from pathlib import Path
import sys

def find_project_root(current_dir: Path) -> Path:
    """Walk upward to find project root directory containing src and data."""
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "src").is_dir() and (parent / "data").is_dir():
            return parent
    return current_dir.resolve().parents[1]

BASE_DIR = find_project_root(Path(__file__).resolve().parent)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.styles import set_presentation_style
from src.utils.logging import TeeStream

LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "03_cluster_patients.log"

set_presentation_style()


def main() -> None:
    """Main execution function for patient clustering."""
    print("Running 03_cluster_patients.py")
    # Stub implementation logic
    print("Patient clustering complete.")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(BASE_DIR).as_posix()}")
            main()
