"""Script execution and subprocess helper utilities for the project."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

from src.utils.paths import rel_path


def run_companion_scripts(
    companion_scripts: Sequence[tuple[str, Path]],
    base_dir: Path | None = None,
) -> None:
    """Executes companion plot/analysis scripts in subprocesses.

    Args:
        companion_scripts: Sequence of (label, script_path) tuples to execute.
        base_dir: Optional base directory for formatting relative log paths.
    """
    print("\nRunning companion plot scripts...")
    for label, script_path in companion_scripts:
        if base_dir is not None:
            try:
                path_display = script_path.relative_to(base_dir).as_posix()
            except ValueError:
                path_display = rel_path(script_path)
        else:
            path_display = rel_path(script_path)

        print(f"   Running {label}: {path_display}")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            check=False,
        )
        if result.returncode != 0:
            print(
                f"   [WARNING] {label} exited with code {result.returncode}. "
                "Check its log for details."
            )
        else:
            print(f"   [OK] {label} completed successfully.")
