# Data Acquisition, Cleaning, and Merging Pipeline

This report documents the workflow and operations implemented in `download_data.py`, `clean_data.py`, and `src/merge_datasets.py` to fetch, extract, clean, and merge datasets for the melanoma immunotherapy response predictor.

---

## Codebase Architecture & Helper Modules

To maintain a clean and modular architecture, helper functions are organised under the `src/utils/` package directory:

* **`src/utils/io.py`**: Handles low-level file I/O operations including:
  * File downloading with streaming (`download_file`).
  * Gzip extraction and decompression (`extract_tar_gz`, `download_and_decompress_gzip`).
  * Conditional download checks (`download_if_missing`).
* **`src/utils/preprocessing.py`**: Coordinates data cleaning, normalisation, and sample alignment, including:
  * TCGA sample ID standardization (`standardise_sample_id`).
  * Survival status mapping (`parse_survival_status`).
  * Clinical metadata filtering (`clean_clinical_df`).
  * RNA-seq sample/gene filtering (`clean_rnaseq_df`).
  * Sample alignment between clinical and expression matrices (`align_expression_and_clinical`).
* **`src/utils/cbioportal_api.py`**: Wraps the cBioPortal REST API v2 client and provides functions to programmatically query study metadata, molecular profiles, sample lists, and download raw datasets (`download_raw_tcga_skcm`).

---


## 1. Data Acquisition (`scripts/download_data.py`)

The data acquisition script standardizes the raw input files by downloading official, curated datasets from the **cBioPortal DataHub**. It processes four cohorts:
1. **Liu 2019** (`mel_iatlas_liu_2019`)
2. **Hugo 2016** (`mel_iatlas_hugo_ucla_2016`)
3. **Riaz 2017** (`mel_iatlas_riaz_nivolumab_2017`)
4. **TCGA-SKCM** (`skcm_tcga_pan_can_atlas_2018`)

### Workflow Details
* **Directory Initialisation**: Creates the necessary target directories under `data/raw/{study_name}`.
* **Targeted Downloads**: Fetches tarballs (`.tar.gz`) directly from `https://datahub.assets.cbioportal.org`.
* **Redownload Prevention**: Checks for the existence of `data_clinical_patient.txt` in the destination directory to skip files that have already been retrieved.
* **Extraction & Clean-up**: Extracts the archive contents to a temporary folder, moves the files into the final destination folder, deletes any empty parent directories, and deletes the temporary `.tar.gz` archive to save disk space.

---

## 2. Data Cleaning (`scripts/clean_data.py`)

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
* Maps response (`RESPONSE`) to binary target variable `response` (CR/PR → 1, PD → 0) and standardises clinical metadata (patient_id, os_months, os_status, age, gender).
* Log2‑transforms TPM expression matrix (log2(TPM + 1)) and aligns samples.

#### B. Hugo 2016 (`clean_hugo_2016`)
* Merges sample and patient clinical sheets.
* Maps response outcomes to binary target `response`.
* Maps sex/gender and age columns, and parses overall survival months (`os_months`) and status (`os_status`).
* Appends driver mutations derived from MAF files.
* Log2-transforms expression data and aligns samples.

#### C. Riaz 2017 (`clean_riaz_2017`)
* Retains all samples (both pre and on-treatment) because `baseline_only=False` is set in the data cleaning script.
* Maps response outcomes to binary target `response_binary`.
* Resolves Riaz's mismatch by joining patient-level mutations with sample-level IDs in the clean mutations matrix.
* Transposes, log-transforms, and aligns all 107 samples.

#### D. TCGA-SKCM (`clean_tcga_skcm`)
* Merges sample and patient sheets and calls custom TCGA clean helpers (`clean_clinical_df`, `clean_rnaseq_df`).
* **Timeline Treatment Integration**: Checks for the existence of `data_timeline_treatment.txt` to parse and aggregate prior clinical treatments (e.g. pivoting key agents like Ipilimumab, Nivolumab, Pembrolizumab, Vemurafenib, Dabrafenib, Trametinib, and Interferon into binary flag columns).
* Filters out redundant and administrative clinical columns.

### Cohort Attrition (Samples Lost at Each Step)

The following table summarizes the number of samples/patients retained and lost at each phase of the cleaning pipeline:

| Cohort        | Step | Starting N | Action / Filter                                                          | Lost | Retained N |
|:------------- |:---- |:---------- |:------------------------------------------------------------------------ |:---- |:---------- |
| **Liu 2019**  | 1    | 122        | Raw patient-sample merge                                                 | -    | 122        |
|               | 2    | 122        | Response filter (keep CR/PR/PD; drop SD/MR/NaN)                          | 18   | 104        |
|               | 3    | 104        | Expression matrix sample alignment                                       | 0    | 104        |
| **Hugo 2016** | 1    | 27         | Raw patient-sample merge                                                 | -    | 27         |
|               | 2    | 27         | Response filter (keep CR/PR/PD; drop SD/MR/NaN)                          | 0    | 27         |
|               | 3    | 27         | Expression matrix sample alignment                                       | 0    | 27         |
| **Riaz 2017** | 1    | 107        | Raw patient-sample merge                                                 | -    | 107        |
|               | 2    | 107        | Response filter (keep CR/PR/PD; drop SD/MR/NaN)                          | 43   | 64         |
|               | 3    | 64         | Expression matrix sample alignment                                       | 0    | 64         |
| **TCGA-SKCM** | 1    | 448        | Raw patient-sample merge                                                 | -    | 448        |
|               | 2    | 448        | Cleaned intermediate clinical file size                                  | 0    | 448        |
|               | 3    | 448        | Patient deduplication in model analysis                                  | 6    | 442        |
|               | 4    | 442        | Survival validation cohort (drop invalid/missing OS)                     | 15   | 427        |

---

## 3. Dataset Merging (`scripts/merge_datasets.py`)

After cleaning, the merge script combines all four cohorts into unified expression and clinical matrices. It produces **two** merged variants, each batch-corrected independently:

### 3a. Full Merge (`data/processed/merged/full/`)
* Concatenates **all** samples from TCGA-SKCM, Liu 2019, Hugo 2016, and Riaz 2017.
* Includes all treatment backgrounds (immunotherapy, chemotherapy, radiation, targeted therapy, and untreated TCGA patients).
* Suitable for pan-cohort analyses such as survival modelling and general biomarker discovery.

### 3b. Immunotherapy-Only Merge (`data/processed/merged/immunotherapy/`)
* Includes **all** samples from the three immunotherapy trial cohorts (Liu 2019, Hugo 2016, Riaz 2017), which exclusively enrolled patients on anti-PD-1 therapy.
* Additionally includes the subset of TCGA-SKCM patients flagged as having received immunotherapy (`TX_TYPE_IMMUNOTHERAPY == 1`).
* Suitable for immunotherapy-specific response prediction and immune biomarker analyses.

### Shared Merging Workflow
Both variants follow the same processing steps:

1. **Gene Feature Alignment**: Intersects gene symbols across all four cleaned expression matrices, retaining only genes present in every cohort.
2. **Concatenation**: Row-wise concatenation of the selected expression and harmonised clinical dataframes.
3. **Zero-Variance Gene Removal**: Drops any genes with zero variance across the merged cohort to prevent division-by-zero errors during batch correction.
4. **pyCombat Batch Correction**: Applies parametric empirical Bayes batch-effect correction (ComBat) using the cohort label as the batch variable. The batch map is built dynamically from the cohorts present in each merge variant.
5. **Output**: Saves `expr_merged.csv` (batch-corrected expression) and `clin_merged.csv` (harmonised clinical metadata) to the respective output directory.

### Harmonised Clinical Columns
Each sample in the merged clinical metadata includes the following standardised fields:

| Column | Description |
| :--- | :--- |
| `patient_id` | Original patient identifier |
| `cohort` | Source dataset (TCGA, Liu_2019, Hugo_2016, Riaz_2017) |
| `os_months` | Overall survival in months |
| `os_status` | Overall survival status (1 = deceased, 0 = living) |
| `response` | Binary immunotherapy response (1 = CR/PR, 0 = PD; NaN for TCGA non-trial patients) |
| `age` | Age at diagnosis (where available) |
| `sex` | Sex (Male/Female/N/A) |
| `specimen_type` | Biopsy type (Primary/Metastatic/N/A) |
| `immunotherapy` | Whether the patient received immunotherapy (1 = yes, 0 = no) |

---

## 4. Clinical & Genomic Characterisation Pipeline

Once the clean datasets are generated, the characterisation scripts analyze clinical and genomic variables across trials (Liu, Hugo, Riaz) and the TCGA reference cohort.

### 4.1. Clinical Characterisation
*   **`scripts/clinical_analysis/run_clinical_analysis.py`**: Reads processed clinical data and generates stacked bar charts showing percentage response rates (CR/PR vs. PD) across studies, saved to `plots/clinical/response_distribution.png` and `plots/clinical/km_os_grid.png`.
*   **`scripts/exploratory_plots/run_response_distribution.py`**: Generates cohort-level response breakdown bar charts.
*   **`scripts/exploratory_plots/run_waffle_chart.py`**: Draws waffle charts representing absolute sample sizes and response status (1 block = 1 patient), saved to `plots/clinical/response_waffle_chart.png`.
*   **`scripts/exploratory_plots/run_response_km_curves.py`**: Evaluates overall survival stratified by response (Responder vs. Non-Responder) in trials, generating Kaplan-Meier curves and Log-Rank tests saved to `plots/clinical/km_os_by_response.png`.
*   **`scripts/exploratory_plots/run_forest_plot.py`**: Fits univariate logistic regression models for response across demographics and driver mutations. Generates a standardized forest plot (grey/red/blue color scheme) saved to `plots/clinical/forest_plot_odds_ratios.png`.
*   **`scripts/clinical_analysis/run_clinical_feature_selection.py`**: Performs univariate Cox hazards modelling and Random Forest Gini feature selection on clinical phenotypes.

### 4.2. Genomic Characterisation
*   **`scripts/biomarkers/run_genomic_characterisation.py`**: Evaluates baseline genomic properties of TCGA and trials:
    *   Generates a comparison of driver mutations (*BRAF*, *NRAS*, *NF1*, and Triple-WT) saved to `plots/genomic/mutation_frequencies.png`.
    *   Plots pre-treatment TMB distributions (trial boxplots by response, TCGA log-normal histogram) saved to `plots/genomic/tmb_distribution.png`.
    *   Plots a Spearman correlation matrix of somatic mutation and neoantigen loads in Liu 2019 saved to `plots/genomic/biomarker_correlation_heatmap.png`.
    *   Generates Kaplan-Meier curves for TCGA overall survival by driver mutation subtype and TMB median-split saved to `plots/genomic/km_genomic_features.png`.
*   **`scripts/biomarkers/run_merged_comut_plot.py`**: Aggregates clinical records and somatic mutations across the three trial studies to generate a pooled, sorted oncoplot ($N=195$) showing driver/resistance gene states aligned with TMB, Response, Cohort source, and Sex. Saved to `plots/genomic/comut_landscape_merged.png`.
*   **`scripts/biomarkers/run_extended_biomarkers.py`**: Evaluates advanced genomic biomarkers:
    *   Correlates total predicted neoantigens with TMB in the pooled trials, generating a regression plot saved to `plots/biomarkers/extended_neoantigen_tmb.png`.
    *   Correlates copy-number alterations (Aneuploidy Score in TCGA) and TMB against 5 continuous transcriptomic immune signatures in TCGA and pooled trials, saving the correlation matrix heatmap to `plots/biomarkers/extended_immune_correlations.png`.
    *   Computes Kaplan-Meier survival curves in TCGA stratified by Aneuploidy Score, saved to `plots/biomarkers/extended_aneuploidy_survival.png`.
    *   Trains cross-validated classifiers (Logistic Regression, Random Forest) on the pooled trial cohort ($N=195$) to evaluate the predictive benefit of signatures, driver mutations, TMB, and pathway mutations.

---

## 5. Outputs Generated

The pipeline outputs processed data, figures, and reports to their respective directories:

### Data Outputs (`data/processed/{study_name}/`)
*   **`expr_cleaned.csv`**: Normalized and log2-transformed expression values (genes as columns, samples as rows).
*   **`clin_cleaned.csv`**: Cleaned, standardized clinical metadata (patient demographic fields, survival timeline, response status, and driver mutation flags).

### Merged Data Outputs (`data/processed/merged/`)
*   **`full/expr_merged.csv`**: Batch-corrected expression matrix for the full merged cohort (all 4 datasets).
*   **`full/clin_merged.csv`**: Harmonised clinical metadata for the full merged cohort.
*   **`full/merged_genomic.csv`**: Unified genomic features (TMB, driver mutations, neoantigens) for the full merged cohort.
*   **`immunotherapy/expr_merged.csv`**: Batch-corrected expression matrix for immunotherapy-treated patients only.
*   **`immunotherapy/clin_merged.csv`**: Harmonised clinical metadata for immunotherapy-treated patients only.
*   **`immunotherapy/merged_genomic.csv`**: Unified genomic features for immunotherapy-treated patients only.

### Visualisation Outputs (`plots/`)
*   **Clinical Characterisation**: Waffle charts, response rates, survival by response, and standardized univariate forest plots in `plots/clinical/`.
*   **Genomic Characterisation**: Mutation landscapes, TMB distributions, correlation heatmaps, merged Co-Mutation oncoplots, and TCGA survival curves in `plots/genomic/` and `plots/biomarkers/`.
*   **Model Performance**: ROC and PR curves in `plots/models/`.

### Reporting Outputs (`reports/`)
*   **`README.md`**: Consolidated sitemap and 4-pillar documentation guide for the reports suite.
*   **`pillar-1-cohorts-and-preprocessing/`**:
    *   `cohort_characteristics_clinical.md`: Baseline report detailing clinical patient demographics, treatment histories, response distributions, survival curves, and forest plots.
    *   `cohort_characteristics_genomic.md`: Baseline report detailing driver mutations, pathway mutations, TMB, neoantigens, immune signature correlations, Aneuploidy overall survival curves, and the merged CoMut oncoplot.
    *   `batch_correction_report.md`: Evaluation report of technical batch effects (uncorrected PCA vs Z-score scaling vs ComBat) and cross-validation data leakage prevention.
*   **`pillar-2-clinical-subtyping/`**:
    *   `clinical_phenotyping_and_feature_selection.md`: Consolidated report detailing non-expression clinical feature importance (RF & Cox PH) and Ward's hierarchical subtype clustering.
*   **`pillar-3-transcriptomic-signatures/`**:
    *   `curated_signatures_report.md`: Comprehensive report detailing gene signature implementation (IFN-$\gamma$, TIS, CYT, IMPRES, CD8, TCGA OS), signature scaling, biomarker orthogonality, and 5-fold CV multimodal response predictor evaluation.
*   **`pillar-4-out-of-cohort-benchmarks/`**:
    *   `transcriptomic_feature_selection_results.md`: Benchmark report detailing response-based `SelectKBest` feature selection ($k=20, 100, 200$) vs. signatures across LOCO cross-validation, TCGA Cox 20-gene survival signature derivation, and InterPro domain annotations.


