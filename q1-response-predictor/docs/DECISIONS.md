---
title: "Analytical & Scientific Decisions Log (Q1 Response Predictor)"
aliases:
  - Analytical Decisions
  - Q1 Decisions
  - Scientific Decisions
tags:
  - decisions
  - immunotherapy
  - melanoma
  - methodology
  - q1-response-predictor
created: 2026-09-15 12:58
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-09-15 14:08
---

> [!summary]- Contents
> ```table-of-contents
> style: nestedList  # nestedList, nestedOrderedList, inlineFirstLevel
> hideWhenEmpty: true # Hide TOC if no headings are found
> ```

# Analytical & Scientific Decisions Log

> [!NOTE]  
> **Purpose**: This living record documents core analytical, statistical, and biological decisions made across the Q1 Response Predictor pipeline. It records the scientific rationale, alternatives considered, and downstream methodological implications for cross-cohort harmonisation and machine learning modelling.

## Locked Analytical & Scientific Decisions

| ID         | Title                                                       |   Status   |    Date    | Primary Rationale                                                                                                                                              |
|:---------- |:----------------------------------------------------------- |:----------:|:----------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ASD-01** | Clinical Response Harmonisation (CR/PR vs PD; SD Censored)  | **Locked** | 2026-09-15 | Resolves irRC vs RECIST 1.1 discrepancies, ensures clean phenotypic contrast for ML, and prevents biomarker signal dilution from heterogeneous Stable Disease. |
| **ASD-02** | Pre-Treatment Baseline Restriction (No On-Treatment Samples) | **Locked** | 2026-09-15 | Prevents pharmacodynamic leakage from on-treatment biopsies; ensures models predict response from pre-infusion biology only, matching prospective clinical use. |

## ASD-01: Clinical Response Harmonisation & Consolidation

### Context & Problem Statement

Supervised machine learning models in the Q1 pipeline require a consistent binary target (`RESPONSE_BINARY`) to train classifiers predicting anti-PD-1 response. However, the constituent clinical trial cohorts utilised divergent clinical response evaluation criteria:
- **Liu 2019**, **Riaz 2017**, and **Gide 2019** assessed radiographic response using **RECIST 1.1** criteria, recording Complete Response (CR), Partial Response (PR), Stable Disease (SD), and Progressive Disease (PD).
- **Hugo 2016** evaluated tumour burden using **immune-related response criteria (irRC)** / modified RECIST (mRECIST), classifying patients strictly into Complete Response (CR), Partial Response (PR), or Progressive Disease (PD), with zero patients categorised as Stable Disease.

To train generalised models under Leave-One-Cohort-Out (LOCO) cross-validation, the pipeline required an explicit decision on how to consolidate these categories into a unified binary response variable.

### Options Considered

1. **Objective Response Rate (ORR Conventional Consolidation: CR/PR = 1, SD/PD = 0)**:
   - _Description_: Group all Stable Disease patients together with Progressive Disease as non-responders.
   - _Drawbacks_: Conflates indolent tumours or prolonged disease stabilization with aggressive progression. Immunotherapy frequently produces sustained clinical benefit without meeting strict partial response shrinkage thresholds; treating these patients as non-responders introduces significant label noise into biomarker discovery.

2. **Durable Clinical Benefit (DCB Consolidation: CR/PR + SD $\ge 6$ months = 1, PD + short SD = 0)**:
   - _Description_: Incorporate Stable Disease patients into the responder class if progression-free survival exceeds a 6-month threshold.
   - _Drawbacks_: Requires uniform, high-granularity progression-free survival (PFS) tracking that is not identically available across all public datasets. Furthermore, Hugo 2016 lacked an explicit SD cohort, which would distort comparative distributions across LOCO validation folds.

3. **Bipolar / Extreme Contrast Consolidation (CR/PR = 1, PD = 0, SD = `NaN` / Censored)**:
   - _Description_: Restrict binary classification labels to unequivocal phenotypic extremes. Complete and Partial Responders are coded as `1.0` (Responders), Progressive Disease is coded as `0.0` (Non-Responders), while Stable Disease and Mixed Response cases are mapped to `NaN` and excluded from binary classification routines.

### Decision

**Adopt Option 3 (Bipolar / Extreme Contrast Consolidation)**.

In [`src/biology_constants.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/src/biology_constants.py#L9-L15) and [`q1-response-predictor/config/data_dictionary.json`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/config/data_dictionary.json#L57-L71), the canonical mapping is defined as:

```python
RECIST_RESPONSE_MAP: Dict[str, float] = {
    "Complete Response": 1.0,
    "Partial Response": 1.0,
    "Progressive Disease": 0.0,
    "Stable Disease": np.nan,
    "Mixed Response": np.nan,
}
```

Downstream machine learning pipelines ([`q1_infer.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/q1_infer.py#L14), [`run_clinical_feature_selection.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-2-clinical-subtyping/run_clinical_feature_selection.py#L376)) filter samples using:

```python
df_evaluable = df.loc[df["RESPONSE_BINARY"].notna()].copy()
```

### Scientific & Analytical Rationale

> [!INSIGHT]  
> **Key Takeaway**: Censoring intermediate Stable Disease isolates the high-confidence molecular signatures of biological sensitivity versus biological resistance, preventing intermediate transcriptional noise from degrading classifier generalisability across independent trial sites.

1. **Mitigating Criteria Incompatibilities**:  
   Hugo 2016 categorised patients by irRC without an SD class. Excluding SD harmonises the target phenotype across cohorts (Hugo, Liu, Riaz, Gide) so that LOCO test performance evaluates true biological concordance rather than artifacts of divergent restaging protocols.

2. **Maximising Molecular Signal-to-Noise Ratio**:  
   Transcriptomic signatures of response (such as IFN-$\gamma$, TIS, and CYT) reflect cytotoxic T-cell infiltration and interferon pathway activation. Stable Disease exhibits intermediate and highly heterogeneous immune microenvironments; grouping SD with either class attenuates the separation boundaries during feature selection and model tuning.

3. **Retention for Survival & Phenotype Stratification**:  
   Censoring SD applies strictly to **binary classification models**. Patients with Stable Disease remain fully preserved in:
   - Overall Survival analyses (Kaplan-Meier survival curves and Cox proportional hazards screening in Pillar 1 and Pillar 4), using `OS_MONTHS` and `OS_STATUS`.
   - Unsupervised clinical and multi-omic clustering (GMM stratification in Pillar 2 / Q1.1), using categorical `RESPONSE` labels to examine their intermediate clinical course.

### Cohort Sample Attrition & Evaluable Numbers

Live sample counts derived from `data/processed/*/clin_cleaned.csv`:

| Cohort | Response Criteria | Complete Response (CR) | Partial Response (PR) | Progressive Disease (PD) | Stable Disease (SD; Excluded) | Binary Evaluable Cohort ($N$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Hugo 2016** | irRC / mRECIST | 4 | 10 | 13 | 0 | **27** |
| **Liu 2019** | RECIST 1.1 | 16 | 32 | 56 | 18 | **104** |
| **Riaz 2017** | RECIST 1.1 | 6 | 14 | 44 | 34 (+9 unannotated) | **64** |
| **Gide 2019** | RECIST 1.1 | 17 | 32 | 29 | 13 | **78** |

## ASD-02: Pre-Treatment Baseline Restriction

### Context & Problem Statement

Two of the four core immunotherapy cohorts — **Riaz 2017** (CheckMate-038) and **Gide 2019** — include longitudinal biopsies collected from the same patients at two timepoints:

- **Pre-treatment** (`SAMPLE_TREATMENT == "Pre"`): tumour biopsy collected prior to the first dose of anti-PD-1 therapy.
- **On-treatment** (`SAMPLE_TREATMENT == "On"`): follow-up biopsy collected during treatment, typically at Week 4 (Riaz) or a comparable early on-therapy interval (Gide).

Prior to this decision, both timepoints were admitted into the cleaned processed datasets (`clin_cleaned.csv`, `expr_cleaned.csv`) with `baseline_only: false` in `config/datasets.yaml`. As a result, the merged immunotherapy training set contained 347 samples from only 288 unique patients — with 41 patients contributing two biopsies each (27 in Riaz, 14 in Gide).

### Options Considered

1. **Retain all samples (baseline_only: false)**:
   - _Description_: Include both pre- and on-treatment biopsies in the training and cross-validation data.
   - _Drawbacks_: On-treatment biopsies reflect post-infusion pharmacodynamic remodelling (reactive IFN-γ upregulation, CD8+ T-cell infiltration, neoantigen depletion) rather than the pre-treatment tumour microenvironment. Including them in training teaches models to detect treatment response rather than predict it from baseline biology. Additionally, treating paired samples from the same patient as independent observations violates the i.i.d. assumption and inflates the effective training-set representation of patients with paired biopsies, biasing cohort-level Z-score standardisation.

2. **Restrict to pre-treatment samples (baseline_only: true)**:
   - _Description_: Enforce `SAMPLE_TREATMENT == "Pre"` at the data-cleaning stage, so only one biopsy per patient enters the pipeline — the pre-infusion baseline sample.
   - _Consequence_: Each row in the training and evaluation data represents exactly one unique patient. Cohort-level Z-score standardisation reflects the true pre-treatment expression distribution.

3. **Restrict at the model-training stage only (post-hoc filter)**:
   - _Description_: Keep all samples in the processed data but filter `SAMPLE_TREATMENT == "Pre"` inside individual training scripts.
   - _Drawbacks_: Fragile — requires every downstream script (feature selection, clinical analysis, clustering, survival analysis) to independently apply the filter; any omission silently re-introduces leakage. Already partially implemented in [`run_transcriptomic_feature_selection.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-4-out-of-cohort-benchmarks/run_transcriptomic_feature_selection.py#L326-L327) but absent from the primary LOCO pipeline and merge step, confirming this approach is inconsistently applied.

### Decision

**Adopt Option 2 (`baseline_only: true` across all cohorts in `datasets.yaml`)**.

Enforced at the earliest possible stage — the data-cleaning script [`clean_data.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/clean_data.py) — so all downstream datasets (`clin_cleaned.csv`, `expr_cleaned.csv`, `mutations_cleaned.csv`, merged files, LOCO folds) are structurally guaranteed to contain only pre-treatment observations.

The filtering logic in `clean_data.py` uses the `SAMPLE_TREATMENT` metadata column where it exists, falling back to the `_PRE` sample ID suffix for cohorts without that column:

```python
if dataset.baseline_only:
    if "SAMPLE_TREATMENT" in df.columns:
        df = df[df["SAMPLE_TREATMENT"].str.strip().str.upper() == "PRE"]
    else:
        df = df[df["SAMPLE_ID"].str.endswith("_PRE")]
```

The `datasets.yaml` flag is set to `true` across all cohorts (Liu 2019, Hugo 2016, Riaz 2017, Gide 2019, and TCGA GDC 2025). Because Liu 2019 consists entirely of pre-treatment biopsies ($N=122$), all samples pass the filter, whereas on-treatment samples are pruned from Riaz 2017 ($N=56$), Gide 2019 ($N=18$), and Hugo 2016 ($N=1$).

### Scientific & Analytical Rationale

> [!INSIGHT]  
> **Key Takeaway**: Restricting training data to pre-treatment baseline biopsies ensures that the Q1 pipeline models what a clinician actually needs — a decision-support prediction made *before* therapy begins, from biology that is measurable at the point of treatment selection.

1. **Preventing Pharmacodynamic Feature Leakage**:  
   On-treatment biopsies are mechanistically downstream of the intervention. Within 2–4 weeks of anti-PD-1 initiation, responding tumours undergo extensive immune microenvironmental remodelling: CD8+ T-cell infiltration increases, IFN-γ signalling is upregulated, M1/M2 macrophage ratios shift, and neoantigen-bearing clones are depleted. Signatures computed from on-treatment samples reflect the pharmacological effect of the drug, not baseline patient biology. A model trained on mixed pre/on-treatment data would learn to detect treatment response rather than predict it prospectively, and would fail catastrophically in deployment where only pre-treatment tissue is available.

2. **Restoring Patient Independence (i.i.d. Assumption)**:  
   Standard machine learning models assume training observations are independent and identically distributed. Paired pre- and on-treatment biopsies from the same patient violate this assumption — the two samples share patient-level genetic background, immune constitution, prior treatment history, and outcome label. With 41 such paired patients across Riaz and Gide, the effective training set was structurally biased towards over-representing these individuals. Under LOCO cross-validation, the risk is amplified: if a patient's pre-treatment biopsy appears in the training fold and their on-treatment biopsy in the test fold (or vice versa), the fold boundary does not provide true held-out evaluation.

3. **Correcting Cohort Z-Score Standardisation**:  
   Cohort-level Z-score normalisation computes $\mu$ and $\sigma$ per gene across all samples in a cohort. On-treatment samples typically exhibit elevated IFN-γ, TIS, CYT, and CD8 T-cell signature scores relative to pre-treatment baselines in responding patients. Their inclusion inflates the cohort mean and variance, suppressing the normalised scores of pre-treatment responders and compressing the separation boundary that the classifier must learn. Restricting to pre-treatment samples ensures the reference distribution reflects true baseline biology.

4. **Alignment with Prospective Clinical Utility**:  
   The clinical question posed by the Q1 predictor is: *"Given a patient's pre-treatment tumour biopsy, will they respond to anti-PD-1 therapy?"* This mirrors the actual clinical decision context — treatment selection occurs before infusion, using baseline tumour profiling. A predictor trained on on-treatment data cannot be deployed prospectively, as the on-treatment biopsy would not yet exist at the point of decision.

### Implementation Details

The fix was applied in two locations within [`clean_data.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/clean_data.py):

- **Clinical filter** (`_harmonise_clinical_data`): checks `SAMPLE_TREATMENT` column first (robust for all cohorts); falls back to `_PRE` suffix matching when the column is absent.
- **Expression filter** (`_process_raw_expression_matrix`): filters on `_PRE`-suffixed column names before transposing; includes a guard that skips filtering when no `_PRE` columns exist (safe for Liu 2019, Hugo 2016 non-longitudinal samples).

### Cohort Impact

| Cohort | `baseline_only` | Total Cleaned Samples (Before) | Pre-Treatment Samples Retained | On-Treatment Samples Removed | Evaluable Binary Responders (Before → After) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Liu 2019** | `true` | 122 | 122 | 0 | 104 → **104** |
| **Hugo 2016** | `true` | 27 | 26 | 1 | 27 → **26** |
| **Riaz 2017** | `true` | 107 | 51 | 56 | 64 → **33** |
| **Gide 2019** | `true` | 91 | 73 | 18 | 78 → **62** |
| **Merged Immunotherapy** | — | **347** | **272** | **75** | 273 → **225** |
