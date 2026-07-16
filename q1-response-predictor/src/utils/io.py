import gzip
import requests
import tarfile
from pathlib import Path

def download_file(url: str, dest_path: Path) -> None:
    """
    Downloads a file from a remote URL to the local filesystem using streaming.

    @param str url The source URL of the file to download.
    @param Path dest_path The destination path on the local disk.
    @return None
    @throws requests.RequestException if the HTTP request fails.
    """
    print(f"Downloading {url} to {dest_path}...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")


def extract_tar_gz(tar_path: Path, extract_to: Path) -> None:
    """
    Extracts all contents of a gzip-compressed tar archive.

    @param Path tar_path The path to the compressed tar archive.
    @param Path extract_to The directory where contents should be extracted.
    @return None
    @throws tarfile.TarError if extraction fails.
    """
    print(f"Extracting {tar_path} to {extract_to}...")
    with tarfile.open(tar_path, "r:gz") as tar_ref:
        tar_ref.extractall(extract_to)
    print("Extraction complete.")


def download_and_decompress_gzip(url: str, output_path: Path, gzip_path: Path = None) -> None:
    """
    Downloads a gzip file from a URL, decompresses it, and writes the contents to a destination file.

    @param str url The URL of the gzip file.
    @param Path output_path The destination path to write the decompressed data.
    @param Path gzip_path Optional path to save the raw gzip file.
    @return None
    @throws requests.RequestException if the HTTP request fails.
    @throws OSError if any file I/O operations fail.
    """
    import shutil
    if gzip_path:
        download_file(url, gzip_path)
        with gzip.open(gzip_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(output_path, 'wb') as f_out:
            with gzip.GzipFile(fileobj=response.raw) as f_in:
                f_out.write(f_in.read())


def download_if_missing(url: str, dest_path: Path) -> None:
    """
    Downloads a file from a URL and saves it to a destination path if it is missing or empty.

    @param str url The URL of the file to download.
    @param Path dest_path The target file path to save the downloaded content.
    @return None
    @throws requests.RequestException if the HTTP request fails.
    @throws OSError if any file I/O operations fail.
    """
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"{dest_path.name} already exists and is non-empty.")
        return
        
    download_file(url, dest_path)
