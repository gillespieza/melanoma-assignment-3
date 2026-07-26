#!/usr/bin/env python3
"""Script 05: Train subgroup-specific predictive models.

Evaluates whether training separate predictive models within discovered patient phenotypes improves performance
compared to applying the global Q1 predictive model across all subgroups.
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
LOG_PATH = LOG_DIR / "05_subgroup_models.log"

set_presentation_style()


def main() -> None:
    """Main execution function for subgroup predictive modelling."""
    print("Running 05_subgroup_models.py")
    # Stub implementation logic
    print("Subgroup model evaluation complete.")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(BASE_DIR).as_posix()}")
            main()
