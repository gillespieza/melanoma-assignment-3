"""Shared I/O utility helpers for safe file writing, downloading, and
archive extraction across all pipeline scripts.

Provides:
- Platform-safe CSV serialisation that handles Windows / Dropbox file-locking
  edge cases without crashing the pipeline.
- URL download helper with progress output.
- tar.gz archive extraction helper.
"""

import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd


from src.utils.paths import rel_path as display_path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DOWNLOAD_CHUNK_BYTES = 8_192


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_file(url: str, dest: Path) -> None:
    """Stream a URL to a local file atomically, printing download progress.

    Downloads to a temporary file (``.tmp`` suffix) first and atomically
    replaces the target destination upon complete transfer. If the download
    fails or is interrupted, the temporary file is removed so partial
    downloads are never left on disk.

    Args:
        url:
            Full URL of the resource to download.
        dest:
            Destination file path. Parent directories are created automatically.

    Raises:
        urllib.error.URLError:
            If the URL cannot be reached or the request fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".tmp")

    print(f"Downloading {url}")
    print(f"  → {display_path(dest)}")

    try:
        with urllib.request.urlopen(url) as response:
            total_bytes = response.headers.get("Content-Length")
            if total_bytes:
                required_bytes = int(total_bytes)
                free_bytes = shutil.disk_usage(dest.parent).free
                if free_bytes < required_bytes:
                    raise OSError(
                        f"Insufficient disk space for download. "
                        f"Required: {required_bytes / 1_048_576:.1f} MB, "
                        f"Free: {free_bytes / 1_048_576:.1f} MB"
                    )
                total_mb = f"{required_bytes / 1_048_576:.1f} MB"
            else:
                total_mb = "unknown size"

            print(f"  File size: {total_mb}")

            downloaded = 0
            with tmp_dest.open("wb") as out_file:
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)

        tmp_dest.replace(dest)

        print(
            f"  Download complete: {downloaded / 1_048_576:.1f} MB written."
        )

    finally:
        if tmp_dest.exists():
            try:
                tmp_dest.unlink()
            except (PermissionError, OSError):
                pass



# ---------------------------------------------------------------------------
# Archive extraction helpers
# ---------------------------------------------------------------------------


def extract_tar_gz(tar_path: Path, extract_to: Path) -> None:
    """Extract a .tar.gz archive to a target directory.

    Args:
        tar_path:
            Path to the ``.tar.gz`` file to extract.
        extract_to:
            Directory into which the archive contents are extracted.
            Created automatically if it does not exist.

    Raises:
        FileNotFoundError:
            If ``tar_path`` does not exist.
        tarfile.TarError:
            If the archive is malformed or cannot be extracted.
    """
    if not tar_path.exists():
        raise FileNotFoundError(
            f"Archive not found: {display_path(tar_path)}"
        )

    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {tar_path.name} → {display_path(extract_to)}")

    with tarfile.open(tar_path, "r:gz") as archive:
        if sys.version_info >= (3, 12):
            archive.extractall(path=extract_to, filter="data")
        else:
            archive.extractall(path=extract_to)

    print("  Extraction complete.")


