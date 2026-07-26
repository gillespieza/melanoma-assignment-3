#!/usr/bin/env python3
"""Script 03: Unsupervised patient stratification pipeline.

Applies K-Means and Agglomerative clustering to the patient feature matrix, computes silhouette
and GAP statistics for optimal K selection, generates UMAP visualisations, and exports cluster assignments to
data/processed/patient_clusters.csv.
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

from src.styles import set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import PROJECT_ROOT, SUBPROJECT_ROOT, rel_path

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "03_cluster_patients.log"

set_presentation_style()


def main() -> None:
    """Main execution function for patient clustering."""
    print(f"Running 03_cluster_patients.py (Project root: {rel_path(PROJECT_ROOT)})")
    # Stub implementation logic
    print("Patient clustering complete.")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
