"""Shared I/O utility helpers for safe file writing across all pipeline scripts.

Provides platform-safe CSV serialisation that handles Windows / Dropbox
file-locking edge cases without crashing the pipeline.
"""

from pathlib import Path

import pandas as pd


def safe_save_csv(df: pd.DataFrame, out_file: Path) -> None:
    """Safely save a DataFrame to CSV, handling potential Windows/Dropbox file locking.

    Attempts to unlink then write the file directly.  Falls back to writing a
    ``.tmp.csv`` file and replacing atomically if the primary path is locked.

    Args:
        df: DataFrame to serialise.
        out_file: Destination file path. Parent directories are created automatically.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        try:
            out_file.unlink()
        except (PermissionError, OSError):
            pass
    try:
        df.to_csv(out_file, index=False)
    except (PermissionError, OSError):
        tmp_file = out_file.with_suffix(".tmp.csv")
        df.to_csv(tmp_file, index=False)
        try:
            tmp_file.replace(out_file)
        except (PermissionError, OSError):
            print(
                f"Warning: could not overwrite {out_file.name} directly. "
                f"Saved to {tmp_file.name}"
            )
