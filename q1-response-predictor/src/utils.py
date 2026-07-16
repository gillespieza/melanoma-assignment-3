import json
import gzip
import urllib.request
import urllib.error
import requests
import tarfile
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict

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


def parse_geo_metadata(matrix_file_path: Path) -> pd.DataFrame:
    """
    Parses a GEO series matrix file to extract sample metadata characteristics.

    @param Path matrix_file_path The file path of the GEO series matrix.
    @return pd.DataFrame A DataFrame of metadata characteristics indexed by GSM ID.
    """
    gsm_ids = []
    titles = []
    characteristics = {}

    with open(matrix_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('"ID_REF"'):
                break
            if not line.startswith('!'):
                continue
            
            parts = line.strip().split('\t')
            key = parts[0]
            values = [v.strip('"') for v in parts[1:]]
            
            if key == "!Sample_geo_accession":
                gsm_ids = values
            elif key == "!Sample_title":
                titles = values
            elif key == "!Sample_characteristics_ch1":
                for idx, val in enumerate(values):
                    if not val:
                        continue
                    if ':' in val:
                        char_key, char_val = val.split(':', 1)
                        char_key = char_key.strip().lower()
                        char_val = char_val.strip()
                        if char_key not in characteristics:
                            characteristics[char_key] = [None] * len(values)
                        characteristics[char_key][idx] = char_val
                    else:
                        if "char_unnamed" not in characteristics:
                            characteristics["char_unnamed"] = [None] * len(values)
                        characteristics["char_unnamed"][idx] = val

    df_meta = pd.DataFrame(index=gsm_ids)
    df_meta['title'] = titles
    for k, v in characteristics.items():
        if len(v) == len(df_meta):
            df_meta[k] = v
            
    return df_meta


def get_gene_symbol_mapping(entrez_ids: List[str], chunk_size: int = 1000) -> Dict[str, str]:
    """
    Translates Entrez ID identifiers to standard Hugo Symbols using the MyGene.info REST API.

    @param List[str] entrez_ids List of Entrez IDs to map.
    @param int chunk_size Number of Entrez IDs to send per query batch (default is 1000).
    @return Dict[str, str] Dictionary mapping Entrez ID strings to Hugo Symbol strings.
    """
    entrez_ids = [str(x) for x in entrez_ids]
    mapping: Dict[str, str] = {}
    for i in range(0, len(entrez_ids), chunk_size):
        chunk = entrez_ids[i:i+chunk_size]
        url = 'https://mygene.info/v3/query'
        q_str = ','.join(chunk)
        data = f'q={q_str}&scopes=entrezgene&fields=symbol&species=human'.encode('utf-8')
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                res = json.loads(response.read().decode('utf-8'))
                for item in res:
                    q = item.get('query')
                    sym = item.get('symbol')
                    if q and sym:
                        mapping[q] = sym
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f"  [WARNING] Failed to fetch gene mapping batch (chunk {i // chunk_size + 1}): {e}")
            
    return mapping


def align_expression_and_clinical(df_expr: pd.DataFrame, df_clin: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns sample IDs between the expression matrix and clinical DataFrame.

    @param pd.DataFrame df_expr Expression DataFrame (samples as rows).
    @param pd.DataFrame df_clin Clinical DataFrame (samples as index).
    @return Tuple[pd.DataFrame, pd.DataFrame] Aligned expression and clinical DataFrames.
    """
    common_samples = df_expr.index.intersection(df_clin.index)
    df_expr = df_expr.loc[common_samples]
    df_clin = df_clin.loc[common_samples]
    return df_expr, df_clin
