# Data Acquisition and Cleaning Pipeline

This report documents the workflow and operations implemented in `download_data.py` and `clean_data.py` to fetch, extract, and clean datasets for the melanoma immunotherapy response predictor.

---

## 1. Data Acquisition (`download_data.py`)

The data acquisition script standardizes the raw input files by downloading official, curated datasets from the **cBioPortal DataHub**. It processes four cohorts:
1. **Liu 2019** (`mel_dfci_2019`)
2. **Hugo 2016** (`mel_iatlas_hugo_ucla_2016`)
3. **Riaz 2017** (`mel_iatlas_riaz_nivolumab_2017`)
4. **TCGA-SKCM** (`skcm_tcga_pan_can_atlas_2018`)

### Workflow Details
* **Directory Initialisation**: Creates the necessary target directories under `data/raw/{study_name}`.
* **Targeted Downloads**: Fetches tarballs (`.tar.gz`) directly from `https://datahub.assets.cbioportal.org`.
* **Redownload Prevention**: Checks for the existence of `data_clinical_patient.txt` in the destination directory to skip files that have already been retrieved.
* **Extraction & Clean-up**: Extracts the archive contents to a temporary folder, moves the files into the final destination folder, deletes any empty parent directories, and deletes the temporary `.tar.gz` archive to save disk space.

---

## 2. Data Cleaning (`clean_data.py`)

The data cleaning pipeline transforms raw inputs into normalized expression matrices and clinical metadata dataframes suitable for modeling. 

### Core Parsers and Helpers
* `parse_cbioportal_expression(expr_file_path)`: Consolidates common cBioPortal expression parsing steps:
  * Reads the expression file (TPM or RSEM values).
  * Excludes missing Hugo gene symbols.
  * Averages duplicate symbols.
  * Drops the administrative `Entrez_Gene_Id` column.
* `parse_maf_mutations(raw_dir, sample_ids)`: Parses mutation data from the MAF format file (`data_mutations.txt`):
  * Filters for non-silent (non-synonymous) variants in `BRAF`, `NRAS`, and `NF1` genes.
  * Generates binary indicator variables (`mut_BRAF`, `mut_NRAS`, `mut_NF1` set to `1` if mutated, `0` otherwise) to join with clinical metadata.

### Cohort-Specific Transformations

#### A. Liu 2019 (`clean_liu_2019`)
* Merges patient and sample clinical sheets.
* Maps best response (`BR`) to binary target variable `response` (CR/PR $\rightarrow 1$, PD $\rightarrow 0$).
* Log2-transforms TPM expression matrix ($\log_2(\text{TPM} + 1)$) and aligns samples.

#### B. Hugo 2016 (`clean_hugo_2016`)
* Merges sample and patient clinical sheets.
* Maps response outcomes to binary target `response`.
* Maps sex/gender and age columns, and parses overall survival months (`os_months`) and status (`os_status`).
* Appends driver mutations derived from MAF files.
* Log2-transforms expression data and aligns samples.

#### C. Riaz 2017 (`clean_riaz_2017`)
* Filters cohort to pre-treatment baseline biopsies only (restricting `SAMPLE_ID` to those ending in `_pre`).
* Maps response outcomes to binary target `response`.
* Resolves Riaz's mismatch where the MAF mutations are identified by `PATIENT_ID` rather than `SAMPLE_ID`, ensuring correct patient-to-sample linkage.
* Restricts expression matrix columns to pre-treatment samples, transposes, log-transforms, and aligns samples.

#### D. TCGA-SKCM (`clean_tcga_skcm`)
* Merges sample and patient sheets and calls custom TCGA clean helpers (`clean_clinical_df`, `clean_rnaseq_df`).
* **Timeline Treatment Integration**: Checks for the existence of `data_timeline_treatment.txt` to parse and aggregate prior clinical treatments (e.g. pivoting key agents like Ipilimumab, Nivolumab, Pembrolizumab, Vemurafenib, Dabrafenib, Trametinib, and Interferon into binary flag columns).
* Filters out redundant and administrative clinical columns.

---

## 3. Outputs Generated

For each cohort, `clean_data.py` writes output to `data/processed/{cohort_name}/`:
* **Expression Matrix**: Log2-transformed expression values with genes as columns and samples as rows (`expr_cleaned.csv` or `rnaseq_cleaned.csv`).
* **Clinical Metadata**: Standardised patient/sample data with mapped response, mutations, and demographics (`clin_cleaned.csv` or `clinical_cleaned.csv`).
