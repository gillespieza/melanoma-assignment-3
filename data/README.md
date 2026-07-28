# Data Directory Structure & File Explanations

> [!NOTE]
> **NB**: Most raw data files are not included in git due to file size limitations. Run `python scripts/download_data.py` (or `python q1-response-predictor/scripts/download_data.py`) on your own system to download and unpack the raw data files yourself.

This directory houses the raw data downloads and processed datasets for the Melanoma Immunotherapy Response Prediction models (incorporating **TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, and **Riaz 2017** cohorts).

---

## 📁 Directory Hierarchy

```
data/
├── README.md                      # This file (explaining structures)
├── raw/                           # Raw cBioPortal / study downloads
│   ├── skcm_tcga_pan_can_atlas_2018/
│   ├── liu_2019/
│   ├── hugo_2016/
│   └── riaz_2017/
└── processed/                     # Cleaned, standardized, and merged datasets
    ├── skcm_tcga_pan_can_atlas_2018/
    │   ├── clin_cleaned.csv
    │   ├── expr_cleaned.csv
    │   └── entrez_to_symbol_cache.json
    ├── liu_2019/
    │   ├── clin_cleaned.csv
    │   ├── expr_cleaned.csv
    │   └── mutations_cleaned.csv
    ├── hugo_2016/
    │   ├── clin_cleaned.csv
    │   ├── expr_cleaned.csv
    │   └── mutations_cleaned.csv
    ├── riaz_2017/
    │   ├── clin_cleaned.csv
    │   ├── expr_cleaned.csv
    │   └── mutations_cleaned.csv
    └── merged/                    # Multi-cohort unified & batch-corrected variants
        ├── full/                  # Full cohort (all patients, N=699)
        │   ├── clin_merged.csv
        │   ├── expr_merged.csv
        │   └── merged_genomic.csv
        └── immunotherapy/         # Immunotherapy-treated patients only (N=326)
            ├── clin_merged.csv
            ├── expr_merged.csv
            └── merged_genomic.csv
```

---

## 📥 1. Raw Datasets (`data/raw/`)
Contains unmodified files retrieved from public databases (such as cBioPortal and literature supplementary material):
*   **`data_expression_median.txt` / `data_mrna_seq_v2_rsem.txt`**: mRNA abundance estimates.
*   **`data_clinical_patient.txt` / `data_clinical_sample.txt`**: Raw demographic, clinicopathological, and survival outcome annotations.
*   **`data_mutations.txt`**: Mutation annotation files (MAF format) listing individual somatic variants (e.g. SNVs, indels, missense/nonsense details).

---

## ⚙️ 2. Processed Datasets (`data/processed/`)
Contains individual study directories where the raw files have been filtered, aligned, and cleaned for statistical modeling:

### Key Files in Cohorts:
1.  **`clin_cleaned.csv`**:
    *   *Description*: Cleaned patient-level clinical variables.
    *   *Key Fields*: Patient ID, sample ID, survival outcomes (`OS_MONTHS`, `OS_STATUS`, `PFS_STATUS`), immunotherapy RECIST response (`RESPONSE`: string, `RESPONSE_BINARY`: 1 for CR/PR, 0 for SD/PD), gender, age, and race.
2.  **`expr_cleaned.csv`**:
    *   *Description*: Log-transformed and gene-symbol aligned expression profiles.
    *   *Format*: Wide table where **rows represent patient samples** and **columns represent genes** (Hugo symbols). Entries are $\log_2(\text{normalized_counts} + 1)$.
3.  **`mutations_cleaned.csv`**:
    *   *Description*: Wide-format binary somatic mutation matrix.
    *   *Format*: **Rows represent patient samples** and **columns represent genes**.
    *   *Values*: `1` (patient has a non-silent somatic mutation in that gene), `0` (gene is wild-type).
4.  **`entrez_to_symbol_cache.json` (TCGA only)**:
    *   *Description*: Mapping dictionary linking TCGA Entrez Gene IDs to standardized Hugo Symbols.

---

## 🔀 3. Merged Cohorts (`data/processed/merged/`)
Contains two unified multi-study cohort variants where technical batch effects in gene expression have been corrected using **ComBat** (`pycombat`).

### A. Full Merge (`merged/full/`) — All Patients ($N=699$)
Combines all samples across the four cohorts regardless of treatment background.
*   **`expr_merged.csv`** *(Shape: 699 rows $\times$ 19,682 columns)*: Log2 expression values for 19,682 intersected genes.
*   **`clin_merged.csv`** *(Shape: 699 rows $\times$ 12 columns)*: Harmonized patient/sample registry tracking identifiers, cohort origins, basic demographics (age, sex, race), clinical outcomes (OS months/status, response/response binary), specimen site types (primary/metastatic), and immunotherapy treatment labels.
*   **`merged_genomic.csv`** *(Shape: 699 rows $\times$ 12 columns)*: Dedicated genomic feature sheet. Separating TMB, driver mutations, and neoantigen loads from the clinical sheet.

### B. Immunotherapy Merge (`merged/immunotherapy/`) — treated Patients ($N=326$)
Includes all patients from the three clinical trials (Liu, Hugo, Riaz) and the subset of TCGA-SKCM patients ($N=70$) documented to have received adjuvant immunotherapy.
*   **`expr_merged.csv`** *(Shape: 326 rows $\times$ 19,639 columns)*: Log2 expression values for 19,639 intersected genes.
*   **`clin_merged.csv`** *(Shape: 326 rows $\times$ 12 columns)*: Harmonized clinical demographics and RECIST outcomes.
*   **`merged_genomic.csv`** *(Shape: 326 rows $\times$ 12 columns)*: Dedicated genomic features.

---

## 🔬 4. Genomic Features Format (`merged_genomic.csv`)
This file is generated alongside clinical records to keep genomic and tumor load features separate from demographics and outcomes. It contains:
*   `PATIENT_ID` & `SAMPLE_ID`: Identifiers.
*   `COHORT`: Origin study.
*   `TMB_NONSYNONYMOUS`: Non-synonymous tumor mutational burden.
*   `mut_BRAF`, `mut_NRAS`, `mut_NF1`: Binary driver somatic mutation indicators (`1` for mutated, `0` for WT).
*   `SNV_NEOANTIGEN`, `INDEL_NEOANTIGEN`, `FUSION_NEOANTIGEN`, `SPLICE_NEOANTIGEN`, `CTA_SELF_NEOANTIGEN`: Quantified neoantigen loads.
