"""
Merge Datasets: Harmonising Clinical, Expression, and Genomic Features Across Melanoma Cohorts.

Intersects common gene features across TCGA-SKCM, Liu 2019, Hugo 2016, and Riaz 2017,
independently Z-score standardises expression matrices to prevent cross-cohort data leakage,
harmonises clinical response metadata, and exports two unified merged datasets to data/processed/merged/:
1. Full Merged Cohort (All 4 datasets, all patients)
2. Immunotherapy-Only Merged Cohort (Trial cohorts + TCGA immunotherapy-treated sub-cohort)
"""

import contextlib
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

# Bootstrap project root resolution for top-level import
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.utils.logging import TeeStream
from src.utils.paths import find_project_root

# Base directories
DATA_DIR = find_project_root(Path(__file__).resolve()) / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MERGED_DIR = PROCESSED_DIR / "merged"
LOG_DIR = find_project_root(Path(__file__).resolve()) / "logs"
LOG_PATH = LOG_DIR / "merge_datasets.log"

FULL_DIR = MERGED_DIR / "full"
IMMUNO_DIR = MERGED_DIR / "immunotherapy"


def load_tcga(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads TCGA-SKCM cleaned expression and clinical data, harmonising column names.

    Args:
        processed_dir: Base processed data directory.

    Returns:
        Tuple of (expression DataFrame, harmonised clinical DataFrame).
    """
    print("Loading TCGA-SKCM...")
    tcga_dir = processed_dir / "skcm_tcga_pan_can_atlas_2018"
    df_expr = pd.read_csv(tcga_dir / "expr_cleaned.csv", index_col="SAMPLE_ID")
    df_clin = pd.read_csv(tcga_dir / "clin_cleaned.csv")

    common_ids = list(df_expr.index.intersection(df_clin["SAMPLE_ID"]))
    df_expr_log = df_expr.loc[common_ids]
    df_clin = df_clin.set_index("SAMPLE_ID").loc[common_ids]

    df_clin_harm = pd.DataFrame(index=df_clin.index)
    df_clin_harm["PATIENT_ID"] = df_clin["PATIENT_ID"]
    df_clin_harm["COHORT"] = "TCGA"
    df_clin_harm["OS_MONTHS"] = df_clin["OS_MONTHS"]
    df_clin_harm["OS_STATUS"] = df_clin["OS_STATUS"].astype(float)
    df_clin_harm["RESPONSE"] = np.nan
    df_clin_harm["RESPONSE_BINARY"] = np.nan
    df_clin_harm["AGE"] = df_clin["AGE"]
    df_clin_harm["RACE"] = df_clin["RACE"]
    df_clin_harm["SEX"] = df_clin["SEX"].map({"Male": "Male", "Female": "Female"}).fillna("N/A")
    df_clin_harm["SPECIMEN_TYPE"] = df_clin["SAMPLE_TYPE"].map({"Primary": "Primary", "Metastatic": "Metastatic"}).fillna("N/A")

    if "TX_TYPE_IMMUNOTHERAPY" in df_clin.columns:
        df_clin_harm["IMMUNOTHERAPY"] = df_clin["TX_TYPE_IMMUNOTHERAPY"].astype(int)
    else:
        df_clin_harm["IMMUNOTHERAPY"] = 0

    return df_expr_log, df_clin_harm


def load_liu(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads Liu 2019 cleaned expression and clinical data, harmonising column names.

    Args:
        processed_dir: Base processed data directory.

    Returns:
        Tuple of (expression DataFrame, harmonised clinical DataFrame).
    """
    print("Loading Liu 2019...")
    liu_dir = processed_dir / "liu_2019"
    df_expr = pd.read_csv(liu_dir / "expr_cleaned.csv", index_col=0)
    df_clin = pd.read_csv(liu_dir / "clin_cleaned.csv", index_col="SAMPLE_ID")

    common_ids = list(set(df_expr.index) & set(df_clin.index))
    df_expr = df_expr.loc[common_ids]
    df_clin = df_clin.loc[common_ids]

    df_clin_harm = pd.DataFrame(index=df_clin.index)
    df_clin_harm["PATIENT_ID"] = df_clin["PATIENT_ID"]
    df_clin_harm["COHORT"] = "Liu_2019"
    df_clin_harm["OS_MONTHS"] = df_clin["OS_MONTHS"]
    df_clin_harm["OS_STATUS"] = df_clin["OS_STATUS"].map({"1:DECEASED": 1.0, "0:LIVING": 0.0})
    df_clin_harm["RESPONSE"] = df_clin["RESPONSE"]
    df_clin_harm["RESPONSE_BINARY"] = df_clin["RESPONSE_BINARY"].astype(float)
    df_clin_harm["AGE"] = np.nan
    df_clin_harm["RACE"] = np.nan
    df_clin_harm["SEX"] = df_clin["SEX"].map({"Male": "Male", "Female": "Female"}).fillna("N/A")

    if "BIOPSY_SITE" in df_clin.columns:
        spec_map = lambda s: "Primary" if "primary" in str(s).lower() else ("Metastatic" if pd.notna(s) and str(s).strip() != "" else "N/A")
        df_clin_harm["SPECIMEN_TYPE"] = df_clin["BIOPSY_SITE"].map(spec_map)
    else:
        df_clin_harm["SPECIMEN_TYPE"] = "N/A"

    df_clin_harm["IMMUNOTHERAPY"] = 1
    return df_expr, df_clin_harm


def load_hugo(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads Hugo 2016 cleaned expression and clinical data, harmonising column names.

    Args:
        processed_dir: Base processed data directory.

    Returns:
        Tuple of (expression DataFrame, harmonised clinical DataFrame).
    """
    print("Loading Hugo 2016...")
    hugo_dir = processed_dir / "hugo_2016"
    df_expr = pd.read_csv(hugo_dir / "expr_cleaned.csv", index_col=0)
    df_clin = pd.read_csv(hugo_dir / "clin_cleaned.csv", index_col="SAMPLE_ID")

    common_ids = list(set(df_expr.index) & set(df_clin.index))
    df_expr = df_expr.loc[common_ids]
    df_clin = df_clin.loc[common_ids]

    df_clin_harm = pd.DataFrame(index=df_clin.index)
    df_clin_harm["PATIENT_ID"] = df_clin["PATIENT_ID"]
    df_clin_harm["COHORT"] = "Hugo_2016"
    df_clin_harm["OS_MONTHS"] = df_clin["OS_MONTHS"]
    df_clin_harm["OS_STATUS"] = df_clin["OS_STATUS"].astype(float)
    df_clin_harm["RESPONSE"] = df_clin["RESPONSE"]
    df_clin_harm["RESPONSE_BINARY"] = df_clin["RESPONSE_BINARY"].astype(float)
    df_clin_harm["AGE"] = df_clin["AGE"].astype(float)
    df_clin_harm["RACE"] = np.nan
    df_clin_harm["SEX"] = df_clin["SEX"].map({"Male": "Male", "Female": "Female", "M": "Male", "F": "Female"}).fillna("N/A")
    df_clin_harm["SPECIMEN_TYPE"] = "Metastatic"
    df_clin_harm["IMMUNOTHERAPY"] = 1
    return df_expr, df_clin_harm


def load_riaz(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads Riaz 2017 cleaned expression and clinical data, harmonising column names.

    Args:
        processed_dir: Base processed data directory.

    Returns:
        Tuple of (expression DataFrame, harmonised clinical DataFrame).
    """
    print("Loading Riaz 2017...")
    riaz_dir = processed_dir / "riaz_2017"
    df_expr = pd.read_csv(riaz_dir / "expr_cleaned.csv", index_col=0)
    df_clin = pd.read_csv(riaz_dir / "clin_cleaned.csv", index_col="SAMPLE_ID")

    common_ids = list(set(df_expr.index) & set(df_clin.index))
    df_expr = df_expr.loc[common_ids]
    df_clin = df_clin.loc[common_ids]

    df_clin_harm = pd.DataFrame(index=df_clin.index)
    df_clin_harm["PATIENT_ID"] = df_clin["PATIENT_ID"]
    df_clin_harm["COHORT"] = "Riaz_2017"
    df_clin_harm["OS_MONTHS"] = df_clin["OS_MONTHS"]
    df_clin_harm["OS_STATUS"] = df_clin["OS_STATUS"].astype(float)
    df_clin_harm["RESPONSE"] = df_clin["RESPONSE"]
    df_clin_harm["RESPONSE_BINARY"] = df_clin["RESPONSE_BINARY"].astype(float)
    df_clin_harm["AGE"] = df_clin["AGE"].astype(float)
    df_clin_harm["RACE"] = df_clin["RACE"]
    df_clin_harm["SEX"] = df_clin["SEX"].map({"Male": "Male", "Female": "Female"}).fillna("N/A")
    df_clin_harm["SPECIMEN_TYPE"] = "Metastatic"
    df_clin_harm["IMMUNOTHERAPY"] = 1
    return df_expr, df_clin_harm


def zscore_expression(df: pd.DataFrame) -> pd.DataFrame:
    """Standardises expression matrix columns individually (Z-score scaling).

    Args:
        df: Expression DataFrame.

    Returns:
        Z-score scaled DataFrame.
    """
    means = df.mean(axis=0)
    stds = df.std(axis=0).replace(0, 1.0).fillna(1.0)
    return (df - means) / stds


def parse_tcga_mutations(raw_dir: Path, sample_ids: List[str]) -> pd.DataFrame:
    """Parses raw TCGA MAF file to extract binary status for key driver mutations (BRAF, NRAS, NF1).

    Args:
        raw_dir: Directory containing TCGA raw files.
        sample_ids: List of TCGA sample IDs to restrict matrix to.

    Returns:
        Binary driver mutation DataFrame.
    """
    mut_path = raw_dir / "data_mutations.txt"
    if not mut_path.exists():
        return pd.DataFrame(0, index=sample_ids, columns=["mut_BRAF", "mut_NRAS", "mut_NF1"])

    chunks = []
    for chunk in pd.read_csv(
        mut_path,
        sep="\t",
        comment="#",
        low_memory=False,
        usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"],
        chunksize=100000,
    ):
        filtered = chunk[chunk["Hugo_Symbol"].isin(["BRAF", "NRAS", "NF1"])]
        chunks.append(filtered)

    df_mut = pd.concat(chunks, ignore_index=True)

    non_syn = [
        "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
        "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
        "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site",
    ]
    df_mut = df_mut[df_mut["Variant_Classification"].isin(non_syn)]

    df_mut["SAMPLE_ID"] = df_mut["Tumor_Sample_Barcode"].apply(lambda x: x[:15] if isinstance(x, str) else "").str.upper()
    df_mut = df_mut[df_mut["SAMPLE_ID"].isin(sample_ids)]

    pivoted = df_mut.groupby(["SAMPLE_ID", "Hugo_Symbol"]).size().unstack(fill_value=0)
    for gene in ["BRAF", "NRAS", "NF1"]:
        if gene not in pivoted.columns:
            pivoted[gene] = 0

    pivoted = (pivoted[["BRAF", "NRAS", "NF1"]] > 0).astype(int)
    pivoted.columns = ["mut_BRAF", "mut_NRAS", "mut_NF1"]
    pivoted = pivoted.reindex(sample_ids, fill_value=0)
    return pivoted


def build_merged_genomic(df_clin_merged: pd.DataFrame, output_dir: Path) -> None:
    """Builds and exports unified merged_genomic.csv containing mutation and neoantigen features.

    Args:
        df_clin_merged: Merged clinical DataFrame.
        output_dir: Target output directory.
    """
    raw_tcga_dir = DATA_DIR / "raw" / "skcm_tcga_pan_can_atlas_2018"
    print(f"Building merged genomic features file for {output_dir.relative_to(BASE_DIR).as_posix()} cohort...")

    cohort_data = {}

    for c_name, c_dir in [
        ("Liu_2019", PROCESSED_DIR / "liu_2019"),
        ("Hugo_2016", PROCESSED_DIR / "hugo_2016"),
        ("Riaz_2017", PROCESSED_DIR / "riaz_2017"),
    ]:
        df_c_clin = pd.read_csv(c_dir / "clin_cleaned.csv", index_col="SAMPLE_ID")
        df_c_mut = pd.read_csv(c_dir / "mutations_cleaned.csv", index_col="SAMPLE_ID")
        df_c_mut = df_c_mut[["BRAF", "NRAS", "NF1"]].rename(
            columns={"BRAF": "mut_BRAF", "NRAS": "mut_NRAS", "NF1": "mut_NF1"}
        )
        cohort_data[c_name] = df_c_clin.join(df_c_mut, how="left")

    df_tcga_clin = pd.read_csv(PROCESSED_DIR / "skcm_tcga_pan_can_atlas_2018" / "clin_cleaned.csv", index_col="SAMPLE_ID")
    df_tcga_mut = parse_tcga_mutations(raw_tcga_dir, df_tcga_clin.index.tolist())
    cohort_data["TCGA"] = df_tcga_clin.join(df_tcga_mut, how="left")

    genomic_rows = []
    samples_df = df_clin_merged if "SAMPLE_ID" in df_clin_merged.columns else df_clin_merged.reset_index()

    for _, row in samples_df.iterrows():
        sample_id = row["SAMPLE_ID"]
        cohort = row["COHORT"]
        patient_id = row["PATIENT_ID"]

        feat_dict = {
            "PATIENT_ID": patient_id,
            "SAMPLE_ID": sample_id,
            "COHORT": cohort,
            "TMB_NONSYNONYMOUS": np.nan,
            "mut_BRAF": 0,
            "mut_NRAS": 0,
            "mut_NF1": 0,
            "SNV_NEOANTIGEN": np.nan,
            "INDEL_NEOANTIGEN": np.nan,
            "FUSION_NEOANTIGEN": np.nan,
            "SPLICE_NEOANTIGEN": np.nan,
            "CTA_SELF_NEOANTIGEN": np.nan,
        }

        if cohort in cohort_data:
            df_cohort = cohort_data[cohort]
            if sample_id in df_cohort.index:
                cohort_row = df_cohort.loc[sample_id]
                if isinstance(cohort_row, pd.DataFrame):
                    cohort_row = cohort_row.iloc[0]

                if "TMB_NONSYNONYMOUS" in cohort_row:
                    feat_dict["TMB_NONSYNONYMOUS"] = cohort_row["TMB_NONSYNONYMOUS"]

                for m_gene in ["mut_BRAF", "mut_NRAS", "mut_NF1"]:
                    if m_gene in cohort_row:
                        feat_dict[m_gene] = int(cohort_row[m_gene]) if pd.notna(cohort_row[m_gene]) else 0

                for neo_feat in ["SNV_NEOANTIGEN", "INDEL_NEOANTIGEN", "FUSION_NEOANTIGEN", "SPLICE_NEOANTIGEN", "CTA_SELF_NEOANTIGEN"]:
                    if neo_feat in cohort_row:
                        feat_dict[neo_feat] = cohort_row[neo_feat]

        genomic_rows.append(feat_dict)

    df_genomic = pd.DataFrame(genomic_rows)
    cols_order = [
        "PATIENT_ID", "SAMPLE_ID", "COHORT",
        "TMB_NONSYNONYMOUS",
        "mut_BRAF", "mut_NRAS", "mut_NF1",
        "SNV_NEOANTIGEN", "INDEL_NEOANTIGEN", "FUSION_NEOANTIGEN", "SPLICE_NEOANTIGEN", "CTA_SELF_NEOANTIGEN",
    ]
    df_genomic = df_genomic[cols_order]

    out_path = output_dir / "merged_genomic.csv"
    df_genomic.to_csv(out_path, index=False)
    print(f"Saved merged genomic features to: {out_path.relative_to(BASE_DIR).as_posix()} (shape: {df_genomic.shape})")


def batch_correct_and_save(
    df_expr_merged: pd.DataFrame, df_clin_merged: pd.DataFrame, output_dir: Path, label: str = ""
) -> None:
    """Saves pre-standardised expression matrix and clinical metadata to output directory.

    Args:
        df_expr_merged: Merged expression DataFrame.
        df_clin_merged: Merged clinical DataFrame.
        output_dir: Destination output directory.
        label: Cohort merge label for logging.
    """
    prefix = f"[{label}] " if label else ""
    assert (df_expr_merged.index == df_clin_merged.index).all(), "Inconsistent sample indices!"
    print(f"{prefix}Total cohort size: {len(df_clin_merged)} samples.")

    output_dir.mkdir(exist_ok=True, parents=True)
    expr_out_path = output_dir / "expr_merged.csv"
    clin_out_path = output_dir / "clin_merged.csv"

    df_expr_merged.to_csv(expr_out_path)

    df_clin_merged = df_clin_merged.reset_index()
    if "PATIENT_ID" in df_clin_merged.columns:
        cols = ["PATIENT_ID"] + [c for c in df_clin_merged.columns if c != "PATIENT_ID"]
        df_clin_merged = df_clin_merged[cols]
    df_clin_merged.to_csv(clin_out_path, index=False)

    print(f"{prefix}Saved expression matrix to: {expr_out_path.relative_to(BASE_DIR).as_posix()} (shape: {df_expr_merged.shape})")
    print(f"{prefix}Saved clinical metadata to:  {clin_out_path.relative_to(BASE_DIR).as_posix()} (shape: {df_clin_merged.shape})")

    build_merged_genomic(df_clin_merged, output_dir)


def main() -> None:
    """Orchestrates dataset loading, feature intersection, Z-score standardisation, and export."""
    print("==================================================")
    print("Merging Datasets: Harmonising Clinical & Expression Features")
    print("==================================================\n")

    FULL_DIR.mkdir(exist_ok=True, parents=True)
    IMMUNO_DIR.mkdir(exist_ok=True, parents=True)

    expr_tcga, clin_tcga = load_tcga(PROCESSED_DIR)
    expr_liu, clin_liu = load_liu(PROCESSED_DIR)
    expr_hugo, clin_hugo = load_hugo(PROCESSED_DIR)
    expr_riaz, clin_riaz = load_riaz(PROCESSED_DIR)

    print("\nAligning gene features...")
    common_genes = list(
        set(expr_tcga.columns) & set(expr_liu.columns) & set(expr_hugo.columns) & set(expr_riaz.columns)
    )
    common_genes.sort()
    print(f"Number of common genes intersected: {len(common_genes)}")

    expr_tcga = expr_tcga[common_genes]
    expr_liu = expr_liu[common_genes]
    expr_hugo = expr_hugo[common_genes]
    expr_riaz = expr_riaz[common_genes]

    print("Standardising expression datasets individually (Z-score)...")
    expr_tcga_scaled = zscore_expression(expr_tcga)
    expr_liu_scaled = zscore_expression(expr_liu)
    expr_hugo_scaled = zscore_expression(expr_hugo)
    expr_riaz_scaled = zscore_expression(expr_riaz)

    print("\n" + "=" * 60)
    print("Building FULL merged cohort (all 4 datasets)")
    print("=" * 60)

    df_expr_full = pd.concat([expr_tcga_scaled, expr_liu_scaled, expr_hugo_scaled, expr_riaz_scaled], axis=0)
    df_clin_full = pd.concat([clin_tcga, clin_liu, clin_hugo, clin_riaz], axis=0)
    batch_correct_and_save(df_expr_full, df_clin_full, FULL_DIR, label="Full")

    print("\n" + "=" * 60)
    print("Building IMMUNOTHERAPY-ONLY merged cohort")
    print("=" * 60)

    tcga_immuno_mask = clin_tcga["IMMUNOTHERAPY"] == 1
    expr_tcga_immuno_scaled = expr_tcga_scaled.loc[tcga_immuno_mask]
    clin_tcga_immuno = clin_tcga.loc[tcga_immuno_mask]
    print(f"  TCGA immunotherapy patients: {len(clin_tcga_immuno)} / {len(clin_tcga)}")

    df_expr_immuno = pd.concat([expr_tcga_immuno_scaled, expr_liu_scaled, expr_hugo_scaled, expr_riaz_scaled], axis=0)
    df_clin_immuno = pd.concat([clin_tcga_immuno, clin_liu, clin_hugo, clin_riaz], axis=0)
    batch_correct_and_save(df_expr_immuno, df_clin_immuno, IMMUNO_DIR, label="Immunotherapy")

    print("\n" + "=" * 60)
    print("Merged Cohort Generation Completed Successfully!")
    print("=" * 60)
    print(f"  Full merge:          {len(df_clin_full)} samples  -> {FULL_DIR.relative_to(BASE_DIR).as_posix()}")
    print(f"  Immunotherapy merge: {len(df_clin_immuno)} samples -> {IMMUNO_DIR.relative_to(BASE_DIR).as_posix()}")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(BASE_DIR).as_posix()}")
            main()
