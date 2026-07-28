"""Generic file, download, archive, and filesystem I/O utilities."""

import gzip
import shutil
import tarfile
import time
from pathlib import Path
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Default operation settings
# ---------------------------------------------------------------------------

DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 120
DEFAULT_FILE_OPERATION_MAX_ATTEMPTS = 10
DEFAULT_FILE_OPERATION_RETRY_DELAY_SECONDS = 2.0
DEFAULT_STREAM_CHUNK_SIZE = 8192


# ---------------------------------------------------------------------------
# Download utilities
# ---------------------------------------------------------------------------


def download_file(
    url: str,
    dest_path: Path,
    timeout: int = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
) -> None:
    """Downloads a file from a remote URL using streaming.

    Args:
        url: Source URL of the file to download.
        dest_path: Destination path on the local filesystem.
        timeout: HTTP request timeout in seconds.

    Raises:
        requests.RequestException: If the HTTP request fails.
        OSError: If the destination cannot be written.
    """
    print(f"Downloading {url} to {dest_path}...")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(
        url,
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()

        with dest_path.open("wb") as output_file:
            for chunk in response.iter_content(
                chunk_size=DEFAULT_STREAM_CHUNK_SIZE,
            ):
                if chunk:
                    output_file.write(chunk)

    print("Download complete.")


def download_if_missing(
    url: str,
    dest_path: Path,
) -> None:
    """Downloads a file if it is missing or empty.

    Args:
        url: Source URL of the file.
        dest_path: Destination path for the downloaded file.

    Raises:
        requests.RequestException: If the HTTP request fails.
        OSError: If file I/O fails.
    """
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(
            f"{dest_path.name} already exists and is non-empty."
        )
        return

    download_file(url, dest_path)


def download_and_decompress_gzip(
    url: str,
    output_path: Path,
    gzip_path: Optional[Path] = None,
) -> None:
    """Downloads and decompresses a gzip file.

    If ``gzip_path`` is provided, the compressed file is saved before being
    decompressed. Otherwise, the compressed response is decompressed directly
    from the HTTP stream.

    Args:
        url: URL of the gzip file.
        output_path: Destination path for the decompressed data.
        gzip_path: Optional path at which to save the compressed file.

    Raises:
        requests.RequestException: If the HTTP request fails.
        OSError: If file I/O fails.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if gzip_path is not None:
        download_file(url, gzip_path)

        with gzip.open(gzip_path, "rb") as input_file:
            with output_path.open("wb") as output_file:
                shutil.copyfileobj(
                    input_file,
                    output_file,
                )

        return

    with requests.get(
        url,
        stream=True,
        timeout=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    ) as response:
        response.raise_for_status()

        with gzip.GzipFile(
            fileobj=response.raw,
        ) as input_file:
            with output_path.open("wb") as output_file:
                shutil.copyfileobj(
                    input_file,
                    output_file,
                )


# ---------------------------------------------------------------------------
# Archive utilities
# ---------------------------------------------------------------------------


def extract_tar_gz(
    tar_path: Path,
    extract_to: Path,
) -> None:
    """Extracts a gzip-compressed tar archive safely.

    Args:
        tar_path: Path to the compressed tar archive.
        extract_to: Directory where the archive should be extracted.

    Raises:
        tarfile.TarError: If the archive cannot be read.
        ValueError: If the archive contains an unsafe path.
    """
    print(
        f"Extracting {tar_path} to {extract_to}..."
    )

    extract_to.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tarfile.open(
        tar_path,
        "r:gz",
    ) as tar_ref:
        _validate_tar_members(
            tar_ref,
            extract_to,
        )

        tar_ref.extractall(extract_to)

    print("Extraction complete.")


def _validate_tar_members(
    tar_ref: tarfile.TarFile,
    extract_to: Path,
) -> None:
    """Prevents archive members escaping the extraction directory."""
    extraction_root = extract_to.resolve()

    for member in tar_ref.getmembers():
        member_path = (
            extract_to / member.name
        ).resolve()

        try:
            member_path.relative_to(
                extraction_root
            )

        except ValueError as error:
            raise ValueError(
                f"Unsafe archive member path detected: "
                f"{member.name}"
            ) from error


# ---------------------------------------------------------------------------
# Filesystem utilities
# ---------------------------------------------------------------------------


def remove_path_with_retries(
    path: Path,
    max_attempts: int = DEFAULT_FILE_OPERATION_MAX_ATTEMPTS,
    delay_seconds: float = DEFAULT_FILE_OPERATION_RETRY_DELAY_SECONDS,
) -> None:
    """Removes a file or directory with retry handling.

    Retrying is useful when another process temporarily has a file or
    directory open, particularly on Windows.

    Args:
        path: File or directory to remove.
        max_attempts: Maximum number of removal attempts.
        delay_seconds: Delay between attempts in seconds.

    Raises:
        OSError: If the path cannot be removed after all attempts.
    """
    if not path.exists():
        return

    last_error: Optional[OSError] = None

    for attempt in range(1, max_attempts + 1):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

            return

        except OSError as error:
            last_error = error

            if attempt < max_attempts:
                print(
                    f"Unable to remove '{path}'. "
                    f"Retrying in {delay_seconds:.1f} seconds "
                    f"({attempt}/{max_attempts})..."
                )

                time.sleep(delay_seconds)

    raise OSError(
        f"Unable to remove path after "
        f"{max_attempts} attempts: {path}"
    ) from last_error


def copy_directory_with_retries(
    source: Path,
    destination: Path,
    max_attempts: int = DEFAULT_FILE_OPERATION_MAX_ATTEMPTS,
    delay_seconds: float = DEFAULT_FILE_OPERATION_RETRY_DELAY_SECONDS,
) -> None:
    """Copies a directory with retry handling.

    Args:
        source: Source directory.
        destination: Destination directory.
        max_attempts: Maximum number of copy attempts.
        delay_seconds: Delay between attempts in seconds.

    Raises:
        OSError: If the directory cannot be copied after all attempts.
    """
    last_error: Optional[OSError] = None

    for attempt in range(1, max_attempts + 1):
        try:
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            shutil.copytree(
                source,
                destination,
                dirs_exist_ok=True,
            )

            return

        except OSError as error:
            last_error = error

            if attempt < max_attempts:
                print(
                    f"Unable to copy '{source}'. "
                    f"Retrying in {delay_seconds:.1f} seconds "
                    f"({attempt}/{max_attempts})..."
                )

                time.sleep(delay_seconds)

    raise OSError(
        f"Unable to copy directory after "
        f"{max_attempts} attempts: {source}"
    ) from last_error