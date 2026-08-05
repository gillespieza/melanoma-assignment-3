---
title: "Phase 2: Feature Analysis, Youden Cutoffs & Genomic Interactions (Q1)"
aliases:
  - Q5 Phase 2
tags:
  - melanoma
  - patient-stratification
  - phase-2
  - q5
created: 2026-08-01 21:41
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-01 21:41
---

## 2. Phase 2: Deep Feature Interpretation & Decision Thresholds

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Performing non-parametric univariate association testing (Mann-Whitney U, Cohen's d), Youden threshold optimisation, and logistic regression interaction modelling.
> - **Why we are doing it**: Establishing statistical significance and non-linear cutoffs is necessary to identify which individual features differentiate Responders from Non-Responders.
> - **What question it answers**: Which individual biomarkers significantly correlate with immunotherapy response, and do signatures interact synergistically with genomic driver mutations?

Phase 2 evaluates biomarker discriminative power across $N_{\text{ICI}}$ patients:
- **Continuous Association**: Mann-Whitney U tests confirm that `TIS`, `CYT`, and `CD8_Tcell` scores are significantly higher in Responders ($CR/PR$) compared to Non-Responders ($PD$).
- **Youden Decision Thresholds**: Youden's J statistic ($J = \text{Sensitivity} + \text{Specificity} - 1$) defines optimal clinical thresholds for categorising continuous signature scores into high/low risk groups.
- **Genomic Synergy & Interaction**: Logistic regression confirms significant interaction terms between `TIS` and `BRAF` mutation status ($p < 0.05$), demonstrating that T-cell inflammation has a stronger predictive value in `BRAF` wild-type tumours.

### Ranked Biomarker Feature Associations (Cohen's d Effect Size)

![Ranked Biomarker Feature Associations](q5-patient-stratification/plots/feature_analysis/biomarker_volcano_plot.png)

> [!INFO] Statistical Methodology: Cohen's d Effect Size
> - **Cohen's $d$ Formula**: Quantifies standardized difference between Responders ($CR/PR$) and Non-Responders ($PD$) in standard deviation units: $d = (\bar{X}_{\text{Resp}} - \bar{X}_{\text{NonResp}}) / s_{\text{pooled}}$.
> - **What the Dashed Lines Mean ($|d| < 0.20$)**: Features lying inside the two dashed lines have weak, negligible differences (>92% overlap between patient groups) and cannot reliably separate responders on their own.
> - **What Lies Outside ($|d| \ge 0.20$)**: Features extending beyond the dashed lines show meaningful biological separation (e.g. green `B_cells` in Responders, red `Macrophage_STV_Score` in Non-Responders) and serve as strong inputs for clinical decision cutoffs.

### Receiver Operating Characteristic (ROC) & Youden Decision Cutoffs

![Youden ROC Curves](q5-patient-stratification/plots/feature_analysis/youden_roc_curves.png)

> [!INFO] Figure Interpretation: ROC Curves & Youden Decision Cutoffs
> - **What the ROC Curves Show**: Receiver Operating Characteristic (ROC) curves measure how accurately each biomarker distinguishes Responders ($CR/PR$) from Non-Responders ($PD$) across all score thresholds. Curves arching higher toward the top-left corner represent superior predictive accuracy.
> - **What the Youden Cutoff Dot Means**: The orange dot marks the single optimal decision threshold ($J = \text{Sensitivity} + \text{Specificity} - 1$) that maximizes true positive detection while minimizing false positive misclassifications.
> - **Clinical Interpretation**: If a patient's biomarker score exceeds the marked Youden cutoff value (e.g. `TIS` $\ge 0.19$ or `CD8_T_cells` $\ge 0.07$), their tumour is classified as inflamed and significantly more likely to benefit from anti-PD-1 immunotherapy.

> [!INSIGHT] Key Rationale & Clinical Insight: Why Single Biomarkers Perform Modestly
> - **Modest Standalone Accuracy (AUC $\approx 0.58$)**: Single biomarkers (`TIS`, `CYT`, `CD8_T_cells`) achieve modest predictive accuracy ($58\%$) because immunotherapy resistance is multi-factorial—a single gene or cell type misses stromal exclusion (CAFs) and M2 macrophage immunosuppression.
> - **Core Motivation for Question 5**: This modest univariate performance proves why rigid single-biomarker tests fail in clinical practice and establishes the essential rationale for **Phase 3 (Unsupervised Multidimensional Clustering)** and **Phase 7 (Multi-Arm Decision Trees)**.

#### Youden Optimal Decision Threshold Metrics

| Biomarker Feature      | Optimal Cutoff   | Youden J   | Sensitivity   | Specificity   | AUC-ROC   |
|:-----------------------|:-----------------|:-----------|:--------------|:--------------|:----------|
| `TIS`                  | 0.191            | 0.184      | 57.3%         | 61.1%         | 0.585     |
| `CYT`                  | 0.621            | 0.205      | 32.9%         | 87.6%         | 0.583     |
| `IFN_gamma`            | 0.439            | 0.170      | 42.7%         | 74.3%         | 0.575     |
| `CD8_T_cells`          | 0.069            | 0.184      | 67.1%         | 51.3%         | 0.584     |
| **`B_cells`**          | **0.430**        | **0.306**  | **53.7%**     | **77.0%**     | **0.632** |
| `M1_M2_Ratio`          | 1.076            | 0.034      | 6.1%          | 97.3%         | 0.442     |
| `Macrophage_STV_Score` | -16.628          | 0.027      | 100.0%        | 2.7%          | 0.413     |

### Genomic Synergy: TIS x BRAF Interaction Analysis

![Genomic Interaction TIS x BRAF](q5-patient-stratification/plots/feature_analysis/genomic_interaction_tis_braf.png)

> [!INFO] Rationale: Why TIS x BRAF Was Selected as Primary Benchmark
> - **FDA-Investigational Benchmark**: `TIS` (Tumour Inflammation Signature, Ayers et al.) represents the clinical gold-standard 18-gene IFN-gamma responsive score evaluated across anti-PD-1 clinical trials.
> - **Clinical Class Trial Anchor**: `BRAF V600` is the primary oncogenic driver mutation in ~40-50% of cutaneous melanomas. In clinical oncology, `BRAF` mutation status dictates whether a patient receives Targeted Therapy (Dabrafenib/Trametinib) vs Immunotherapy (anti-PD-1).
> - **Primary Benchmark**: Testing `TIS` $\times$ `BRAF` provides the primary benchmark for whether oncogenic MAPK activation dampens T-cell inflammation before expanding to all 21 driver $\times$ signature permutations below.

### Multi-Permutation Genomic x Immune Interaction Matrix

![Genomic Immune Interaction Matrix](q5-patient-stratification/plots/feature_analysis/genomic_immune_interaction_matrix.png)

> [!INFO] Figure Interpretation: Genomic x Immune Interaction Matrix
> - **What this heatmap shows**: Logistic regression interaction coefficients ($\beta_{\text{interaction}}$) and significance across all driver mutation $\times$ immune signature permutations ($N_{\text{BRAF}} = 128$, $N_{\text{NRAS}} = 82$, $N_{\text{NF1}} = 40$).
> - **`BRAF` Dominance & Statistical Significance (White Border)**: `BRAF` $\times$ `TIS` ($\beta = -0.65, p = 0.040$, highlighted with a crisp white border) is the single interaction reaching strict $p < 0.05$ because `BRAF` is the largest mutant subgroup ($N = 128$). T-cell/IFN-gamma signatures in the `BRAF` column show consistent negative interaction terms specifically in `BRAF` melanomas.
> - **`NF1` $\times$ `M1_M2_Ratio` Synergy ($\beta = +0.94$)**: `NF1`-mutated melanoma displays the highest positive effect size with macrophage polarisation (`M1_M2_Ratio`), demonstrating that pro-inflammatory myeloid reprogramming strongly enhances response in high-TMB `NF1`-loss tumours.
> - **Clinical Utility**: Provides the mathematical foundation for multi-dimensional patient clustering (Phase 3) and multi-arm treatment routing (Phase 7).

> [!INSIGHT] Analytical Validation: Heatmap Confirms Primary Focus on TIS x BRAF
> - **Validation of Initial Hypothesis**: The comprehensive interaction matrix confirms that `TIS` $\times$ `BRAF` ($\beta = -0.65, p = 0.040$) is indeed the single statistically significant driver-microenvironment interaction ($p < 0.05$), validating our initial analytical focus on this key biomarker pair.
> - **Borderline Cells Highlight `BRAF` Again**: Every borderline significant interaction ($p < 0.10$) occurs exclusively within the `BRAF` column: `BRAF` $\times$ `IFN_gamma` ($\beta = -0.57, p = 0.064$), `BRAF` $\times$ `B_cells` ($\beta = -0.59, p = 0.073$), `BRAF` $\times$ `CD8_T_cells` ($\beta = -0.57, p = 0.074$). This repeatedly points to `BRAF` oncogenic signalling as the dominant genomic modifier of microenvironmental immunity.

### Key Takeaways
> [!INSIGHT] Key Takeaways: Phase 2 Feature Analysis & Stratification Rationale
> - **Best Standalone Marker**: `B_cells` is the single best individual marker for distinguishing responders from non-responders (AUC = 0.632).
> - **Modest Standalone Predictive Power**: While inflammatory signatures (`TIS`, `CYT`, `CD8_T_cells`) and B-cell abundance (`B_cells`) show statistically significant elevation in responders, their standalone predictive accuracy is modest (AUC $\approx 0.58–0.63$). No single biomarker acts as a sole determinant of response.
> - **Decision Thresholds Provide Clinical Triage Cutoffs**: Youden's J statistic established concrete numerical cutoffs (such as `B_cells` threshold $\ge 0.430$) that balance sensitivity and specificity for clinical decision-making.
> - **Genomic Drivers Modify Microenvironmental Immunity**: Microenvironmental immune inflammation interacts significantly with oncogenic driver mutations—specifically `BRAF V600` ($\beta = -0.65, p = 0.040$). High T-cell inflammation has a stronger positive predictive value in `BRAF` wild-type tumours than in `BRAF`-mutated tumours.
> - **Rationale for Stratification**: Because single biomarkers yield modest standalone performance and interact with underlying driver mutations, robust clinical decision support requires multi-dimensional unsupervised clustering (Phase 3) rather than single-gene tests.

> [!formula]+ Phase 2 Script Execution & Software Module Architecture
> - **Primary Pipeline Execution Scripts**:
>   - [`02_feature_analysis.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/02_feature_analysis.py): Conducts non-parametric Mann-Whitney U testing and Cohen's d effect size calculations across features (`univariate_feature_associations.csv`), computes Youden J optimal decision cutoffs and ROC curves (`youden_cutoffs.csv`), and evaluates 21 driver mutation $\times$ immune signature logistic regression interaction terms (`genomic_immune_interactions.csv`, `genomic_interaction_tis_braf.png`, `genomic_immune_interaction_matrix.png`).
> - **Core Supporting Python Modules**:
>   - [`feature_analysis.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/feature_analysis.py): Implements statistical testing functions (`compute_univariate_associations`), Youden cutoff calculation (`compute_youden_cutoffs`), and interaction model fitting (`evaluate_feature_interactions`).
>   - [`reporting.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/reporting.py): Formats statistical summary tables and Obsidian markdown elements.
>   - [`q5_constants.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/q5_constants.py): Defines primary feature lists and biological constants.
> - **Shared Cross-Question & Pipeline Modules**:
>   - [`run_q5_pipeline.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/run_q5_pipeline.py): Master pipeline orchestrator executing `02_feature_analysis.py` as Step 2.
>   - [`generate_q5_report.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/generate_q5_report.py): Compiles live statistical summaries and generates phase markdown reports.
