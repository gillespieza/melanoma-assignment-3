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
updated: 2026-09-15 16:30
---

> [!summary]+ Contents
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
| **ASD-03** | Complete Exclusion of Van Allen 2015 (Pure CTLA-4 Blockade)  | **Locked** | 2026-09-15 | Removes pure ipilimumab (aCTLA-4) cohort from anti-PD-1 ML pipelines; prevents drug-class confounding and mechanism-specific biomarker signal distortion. |

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

Downstream machine learning pipelines ([`q1_infer.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/q1_infer.py#L14), [`run_clinical_feature_selection.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar_2_clinical_subtyping/run_clinical_feature_selection.py#L376)) filter samples using:

```python
df_evaluable = df.loc[df["RESPONSE_BINARY"].notna()].copy()
```

### Hugo 2016 Application and Prediction Semantics

Hugo et al. (2016) evaluated anti-PD-1 treatment response using immune-related response criteria (irRC) / modified RECIST (mRECIST), rather than standard RECIST 1.1. The retained pre-treatment RNA-seq cohort contains 26 evaluable patients: 4 Complete Responses, 10 Partial Responses, and 12 Progressive Disease cases. There are no Stable Disease cases in this modelling subset; one of the 27 pre-treatment RNA samples was excluded because it lacked an evaluable binary endpoint.

Accordingly, the Hugo labels are applied as follows:

| Hugo `RESPONSE` | `RESPONSE_BINARY` | Classification meaning |
| :--- | :---: | :--- |
| `Complete Response` | `1.0` | Responder |
| `Partial Response` | `1.0` | Responder |
| `Progressive Disease` | `0.0` | Non-responder |

This is an **objective response versus progression** endpoint, not a durable clinical-benefit endpoint. Stable Disease and Mixed Response remain `NaN` wherever present in other cohorts and are excluded only from binary classification; their raw clinical labels remain available for survival and phenotype analyses.

The binary ground truth is exported as `actual_response` in `dashboard/public/q1_predictions.csv` and is sourced from `RESPONSE_BINARY`. Model predictions are separate: `prob_ensemble` is the mean ensemble probability of response, and `pred_label` is assigned as `1` when `prob_ensemble >= 0.5`, otherwise `0`:

```python
df_pred["pred_label"] = (df_pred["prob_ensemble"] >= 0.5).astype(int)
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
| **Hugo 2016** | irRC / mRECIST | 4 | 10 | 12 | 0 | **26** |
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
   - _Drawbacks_: Fragile — requires every downstream script (feature selection, clinical analysis, clustering, survival analysis) to independently apply the filter; any omission silently re-introduces leakage. Already partially implemented in [`run_transcriptomic_feature_selection.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar_4_out_of_cohort_benchmarks/run_transcriptomic_feature_selection.py#L326-L327) but absent from the primary LOCO pipeline and merge step, confirming this approach is inconsistently applied.

### Decision

**Adopt Option 2 (`baseline_only: true` across all cohorts in `datasets.yaml`)**.

Enforced at the earliest possible stage — the data-cleaning script [`clean_data.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar_1_cohort_preprocessing/clean_data.py) — so all downstream datasets (`clin_cleaned.csv`, `expr_cleaned.csv`, `mutations_cleaned.csv`, merged files, LOCO folds) are structurally guaranteed to contain only pre-treatment observations.

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
> **Key Takeaway**: Restricting training data to pre-treatment baseline biopsies ensures that the Q1 pipeline models what a clinician actually needs — a decision-support prediction made _before_ therapy begins, from biology that is measurable at the point of treatment selection.

1. **Preventing Pharmacodynamic Feature Leakage**:  
   On-treatment biopsies are mechanistically downstream of the intervention. Within 2–4 weeks of anti-PD-1 initiation, responding tumours undergo extensive immune microenvironmental remodelling: CD8+ T-cell infiltration increases, IFN-γ signalling is upregulated, M1/M2 macrophage ratios shift, and neoantigen-bearing clones are depleted. Signatures computed from on-treatment samples reflect the pharmacological effect of the drug, not baseline patient biology. A model trained on mixed pre/on-treatment data would learn to detect treatment response rather than predict it prospectively, and would fail catastrophically in deployment where only pre-treatment tissue is available.

2. **Restoring Patient Independence (i.i.d. Assumption)**:  
   Standard machine learning models assume training observations are independent and identically distributed. Paired pre- and on-treatment biopsies from the same patient violate this assumption — the two samples share patient-level genetic background, immune constitution, prior treatment history, and outcome label. With 41 such paired patients across Riaz and Gide, the effective training set was structurally biased towards over-representing these individuals. Under LOCO cross-validation, the risk is amplified: if a patient's pre-treatment biopsy appears in the training fold and their on-treatment biopsy in the test fold (or vice versa), the fold boundary does not provide true held-out evaluation.

3. **Correcting Cohort Z-Score Standardisation**:  
   Cohort-level Z-score normalisation computes $\mu$ and $\sigma$ per gene across all samples in a cohort. On-treatment samples typically exhibit elevated IFN-γ, TIS, CYT, and CD8 T-cell signature scores relative to pre-treatment baselines in responding patients. Their inclusion inflates the cohort mean and variance, suppressing the normalised scores of pre-treatment responders and compressing the separation boundary that the classifier must learn. Restricting to pre-treatment samples ensures the reference distribution reflects true baseline biology.

4. **Alignment with Prospective Clinical Utility**:  
   The clinical question posed by the Q1 predictor is: _"Given a patient's pre-treatment tumour biopsy, will they respond to anti-PD-1 therapy?"_ This mirrors the actual clinical decision context — treatment selection occurs before infusion, using baseline tumour profiling. A predictor trained on on-treatment data cannot be deployed prospectively, as the on-treatment biopsy would not yet exist at the point of decision.

### Implementation Details

The fix was applied in two locations within [`clean_data.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar_1_cohort_preprocessing/clean_data.py):

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

## ASD-03: Complete Exclusion of Van Allen 2015 from Active Pipelines

### Context & Problem Statement

Van Allen et al. (2015) published whole-exome and transcriptomic profiles from $N = 110$ metastatic melanoma patients treated with **ipilimumab** (monoclonal antibody targeting CTLA-4). The cohort was initially retrieved into `data/raw/van_allen_2015/` as part of cross-cohort immune landscape comparisons.

However, the primary therapeutic target of the Q1 response predictor is **anti-PD-1 / anti-PD-L1 checkpoint blockade** (nivolumab, pembrolizumab), evaluated across Liu 2019, Hugo 2016, Riaz 2017, and Gide 2019. Prior CTLA-4 exposure in patients receiving subsequent anti-PD-1 represents an important clinical confounder (as observed in Liu 2019 where $40/104$ evaluable patients had prior anti-CTLA-4 therapy, and Riaz 2017 where $15/33$ had prior ipilimumab). In contrast, Van Allen 2015 represents an entirely different pharmacological class: **100% of patients received ipilimumab as primary therapy**, not anti-PD-1.

### Options Considered

1. **Include Van Allen 2015 in Merged Immunotherapy ML Training**:
   - _Description_: Pool Van Allen 2015 with anti-PD-1 cohorts to expand training sample size ($+110$ patients).
   - _Drawbacks_: Conflates CTLA-4 response biology with PD-1 response biology. CTLA-4 blockade acts primarily during T-cell priming in lymphoid tissues and promotes regulatory T-cell (Treg) depletion via ADCC, whereas PD-1 blockade acts in peripheral tumour tissue to restore exhausted effector T-cells. Predictive transcriptomic signatures diverge significantly: signatures predictive of anti-PD-1 benefit (e.g. `TIS`, `CYT`, `CD8_Tcell`) demonstrate weaker or distinct association patterns under CTLA-4 blockade. Training an anti-PD-1 classifier on ipilimumab response labels introduces drug-class confounding that cannot be resolved through statistical adjustment.

2. **Retain as an Active Secondary / Validation Cohort in `datasets.yaml`**:
   - _Description_: Keep Van Allen 2015 active with `merge_enabled: false` for exploratory comparisons.
   - _Drawbacks_: Increases maintenance overhead, risks unintended inclusion by scripts scanning `config/datasets.yaml`, and provides questionable validation utility for models specifically calibrated for anti-PD-1 clinical decision support.

3. **Complete Exclusion from Active Pipelines on This Branch**:
   - _Description_: Remove or keep omitted from `config/datasets.yaml`, ensure no pipeline scripts load or merge the cohort, and document permanent exclusion as a binding project rule.
   - _Consequence_: Raw and processed data files remain preserved in `data/raw/van_allen_2015/` and `data/processed/van_allen_2015/` for archival reference, but zero active pipeline code loads the cohort.

### Decision

**Adopt Option 3 (Complete Exclusion from Active Pipelines on This Branch)**.

- **`config/datasets.yaml`**: Confirmed omitted from active entries. Only anti-PD-1 trial cohorts (Liu 2019, Hugo 2016, Riaz 2017, Gide 2019) and reference TCGA GDC 2025 are listed.
- **`merge_datasets.py`**: Confirmed no references or ingestion paths for `van_allen_2015`.
- **Project Governance**: Codified as **Rule 21** in `.agents/AGENTS.md`, ensuring all future agent sessions enforce the exclusion.

### Scientific & Analytical Rationale

> [!INSIGHT]  
> **Key Takeaway**: Anti-PD-1 and anti-CTLA-4 therapies target fundamentally distinct phases of the cancer-immunity cycle. Pooling a pure ipilimumab cohort with anti-PD-1 cohorts conflates the biological mechanisms of response, undermining model interpretability and prospective clinical utility.

1. **Divergent Pharmacological Mechanisms**:  
   CTLA-4 is an early-checkpoint receptor expressed on naive and memory T-cells that dampens initial priming by antigen-presenting cells in draining lymph nodes. PD-1 is an inhibitory receptor induced on antigen-experienced, exhausted effector T-cells within the peripheral tumour microenvironment. Response to ipilimumab is strongly driven by baseline neoantigen load and host immune repertoire diversity, whereas anti-PD-1 response depends critically on pre-existing intratumoural cytotoxic infiltration (`CD8A`, `CYT`) and IFN-$\gamma$-mediated PD-L1 expression.

2. **Target Phenotype Purity for Machine Learning**:  
   The Q1 prediction task is defined as: _"Predict clinical response (CR/PR vs PD) to anti-PD-1/anti-PD-L1 checkpoint blockade from pre-treatment tumour profiling."_ Admitting a cohort where response was governed by anti-CTLA-4 pharmacology contaminates the target label `RESPONSE_BINARY`, diluting the feature importance of true PD-1 predictive biomarkers.

3. **Distinction Between Confounder Adjustment and Drug-Class Conflation**:  
   Prior ipilimumab exposure in anti-PD-1-treated patients (e.g. Liu 2019, Riaz 2017) represents an **effect modifier / clinical covariate** of PD-1 response, properly addressed via feature annotation or stratified reporting. In contrast, evaluating primary response to ipilimumab is an entirely different clinical question that belongs to a separate investigational arm.
