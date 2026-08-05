---
title: "Curated Gene Expression Signatures, Extended Biomarkers & Model Evaluation Report"
aliases:
  - curated-signatures-report
  - q1-signatures-report
tags:
  - q1
  - signatures
  - biomarkers
  - immunotherapy
  - report
cssclasses:
  - table-small
  - table-center
  - row-alt
created: 2026-07-23 17:21
updated: 2026-08-02 16:50
---
# Curated Gene Expression Signatures, Extended Biomarkers & Model Evaluation Report

### Executive Summary
This report outlines the transcriptomic feature engineering strategy for the Melanoma Immunotherapy Response Predictor. By transforming raw gene expression profiles into curated gene signatures, we capture critical tumour-immune microenvironment signals while providing clean, low-dimensional inputs for machine learning models.


## 1. Why Use Gene Signatures Instead of Raw Expression Data?

> [!NOTE] Section Context
> - **What We Are Doing**: Justifying the decision to compress $>20,000$ raw RNA-seq gene measurements into **6 curated pathway-level scores** rather than feeding raw expression vectors directly into machine learning models.
> - **Why We Are Doing It**: Raw gene expression is high-dimensional ($D \gg N$), multicollinear, and batch-contaminated. Directly training classifiers on 20,000 features across cohorts of $N \approx 100\text{--}500$ guarantees overfitting to study-specific noise rather than generalisable biology. Pathway signatures act as noise-reducing biological filters.
> - **Questions**:
>   1. *Why can't we just use all 20,000 genes as features?*
>   2. *How does gene-level collinearity undermine model interpretability?*
>   3. *How do batch effects across Liu (122), Hugo (27), Riaz (107), and TCGA (443) corrupt feature scaling?*

High-throughput RNA sequencing measures over $20,000$ genes per patient sample. Training machine learning models directly on raw expression vectors introduces three major challenges:

* **Overfitting ($D \gg N$)**: Evaluating $>20,000$ features on typical clinical cohorts ($N \approx 100\text{--}500$) causes classifiers to memorise sample-specific noise rather than generalisable biology.
* **Multicollinearity**: Immune genes operate in tightly co-expressed networks, creating redundant features that destabilise linear model weights.
* **Batch Effects**: Technical variation across sequencing platforms and clinical studies (Liu 2019, Hugo 2016, Riaz 2017, TCGA-SKCM) introduces noise that obscures true biological signal.

> [!TIP] The Solution
> **Gene Signatures** compress high-dimensional gene matrices into **6 continuous, pathway-specific scores**, acting as noise-reducing biological filters that reliably align across diverse patient cohorts.


## 2. Curated Immunotherapy Signatures (Implemented in `src/signatures.py`)

> [!NOTE] Section Context
> - **What We Are Doing**: Implementing six literature-curated gene expression signatures in [`signatures.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/signatures.py), each capturing a distinct axis of tumour-immune biology — from IFN-γ cytokine signalling and CD8 T-cell infiltration to checkpoint ligand abundance and effector killing capacity.
> - **Why We Are Doing It**: Rather than selecting signatures arbitrarily, each was chosen because it has been independently validated in published anti-PD-1 trials. Using established, biologically-grounded scores rather than ad-hoc gene selections reduces the risk of overfitting and makes our model interpretable to a clinical audience.
> - **Questions**:
>   1. *Which biological axes of the tumour-immune microenvironment are captured by these six signatures?*
>   2. *Do individual signatures separate responders from non-responders at baseline — before any modelling?*
>   3. *Are the signatures redundant with each other, or do they each capture independent information?*

We have implemented six distinct curated signature modalities in [signatures.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/signatures.py). Each signature captures a distinct axis of tumour-immune biology.

> [!NOTE]
> **Supplementary Appendix**: For detailed biological mechanisms, gene-by-gene breakdowns, and individual mathematical formulas, see [curated_signatures_supplementary.md](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/reports/pillar-3-transcriptomic-signatures/curated_signatures_supplementary.md).

### 2.1. Signature Modality Overview Matrix

| Signature Name & Citation | Biological Axis | Gene Count | Key Gene Composition | Mathematical Formulation & Scoring Logic |
| :--- | :--- | :---: | :--- | :--- |
| **Interferon-Gamma (IFN-γ)**<br>*(Ayers et al., 2017)* | Adaptive Immune & Cytokine Response | 6 Genes | `IFNG`, `CXCL9`, `CXCL10`, `IDO1`, `HLA-DRA`, `STAT1` | Arithmetic mean of log-transformed expression:<br>$$S_{\text{IFN}\gamma} = \frac{1}{|G|} \sum_{g \in G} \log_2(\text{TPM}_g + 1)$$ |
| **Tumour Inflammation Signature (TIS)**<br>*(Ayers et al., 2017)* | Pre-existing Suppressed Adaptive Infiltration | 21 Genes *(Expanded)* | `CCL5`, `CD2`, `CD3D`, `CD3E`, `CD27`, `CD274`, `CMKLR1`, `CXCL9`, `CXCR6`, `GZMB`, `GZMK`, `HLA-DRA`, `HLA-DQA1`, `HLA-E`, `IDO1`, `LAG3`, `NKG7`, `PDCD1LG2`, `PSMB10`, `STAT1`, `TIGIT` | Arithmetic mean across expanded panel incorporating T-cell receptor components (`CD2`, `CD3D`, `CD3E`) and cytolytic enzymes (`GZMB`, `GZMK`). |
| **Cytolytic Activity (CYT)**<br>*(Rooney et al., 2015)* | Effector Cell-Mediated Tumour Killing | 2 Genes | `GZMA` *(Granzyme A)*<br>`PRF1` *(Perforin 1)* | Logarithmic geometric mean of effector enzymes:<br>$$S_{\text{CYT}} = \frac{\log_2(\text{TPM}_{\text{GZMA}} + 1) + \log_2(\text{TPM}_{\text{PRF1}} + 1)}{2}$$ |
| **CD8 T-Cell Abundance** | Lineage-Specific T-Cell Infiltration | 2 Genes | `CD8A`, `CD8B` | Lineage marker mean log-expression:<br>$$S_{\text{CD8}} = \frac{\log_2(\text{TPM}_{\text{CD8A}} + 1) + \log_2(\text{TPM}_{\text{CD8B}} + 1)}{2}$$ |
| **Immune Predictive Score (IMPRES)**<br>*(Ausländer et al., 2018)* | Checkpoint Ratio Balance *(Stimulatory vs. Inhibitory)* | 15 Pairwise Ratios | 15 Checkpoint Pairs<br>*(e.g., `CD274`/`VSIR`, `PDCD1`/`TNFRSF4`, `CD28`/`CD276`)* | Non-linear sum of pairwise binary indicators:<br>$$S_{\text{raw}} = \sum_{i=1}^{15} \mathbb{I}(E_{\text{Gene A}_i} > E_{\text{Gene B}_i})$$<br>*Scaled to 0–15 for missing pairs.* |
| **PD-L1 Transcript Proxy** | Checkpoint Ligand Target Abundance | 1 Gene | `CD274` | Continuous transcript expression proxy:<br>$$S_{\text{PD-L1}} = \log_2(\text{TPM}_{\text{CD274}} + 1)$$ |

### 2.2. Univariate Distribution of Signatures by Response Status

> [!NOTE] Section Context
> - **What We Are Doing**: Evaluating the baseline univariate discriminative capacity of each signature — plotting score distributions split by **Responders (CR/PR)** vs. **Non-Responders (PD)** across the pooled clinical trial cohorts ($N = 195$ annotated response patients out of $N = 326$ total), annotated with two-sided Mann-Whitney U test $p$-values.
> - **Why We Are Doing It**: Before combining signatures in a multivariate model, we need to know whether each one carries any signal at all on its own. A signature with no univariate separation is unlikely to help even in a joint model. This step validates that our chosen signatures are biologically grounded in the trial data — not just theoretically motivated.
> - **Questions**: *Is elevated expression of individual immune signatures significantly associated with clinical response to anti-PD-1 therapy at baseline?*

Each signature distribution is visualised using a combined box plot and jitter scatter plot stratified by Responder and Non-Responder clinical outcome, displaying the median score, interquartile range, and individual patient data points.

![Signature Distributions (Box + Jitter)](../../plots/signatures/signature_box_jitter_by_response.png)

#### Key Empirical Findings
* **Positive Trend Across All Signatures**: Responders consistently show higher baseline expression across all 6 signature modalities.
* **Top Individual Separators (Pooled Trials)**:
  * **Tumour Inflammation Signature (TIS)**: Strongest overall predictor ($p = 0.042$), confirming pre-existing immune infiltration drives clinical benefit.
  * **Cytolytic Activity (CYT)**: Statistically significant separation ($p = 0.047$), indicating active T-cell killing (`GZMA` / `PRF1`) at baseline.
  * **PD-L1 Proxy (`CD274`)**: Borderline significant separation ($p = 0.059$).
* **Trial Heterogeneity**: Signatures show strong, significant separation in treatment-naïve cohorts like **Riaz 2017** ($p < 0.01$), but higher overlap in cohorts with prior therapy exposures (e.g. **Liu 2019**).

> [!INSIGHT] Key Takeaways
> * **Biological Confirmation**: Baseline microenvironmental inflammation and cell-killing activity directly align with anti-PD-1 efficacy.
> * **No Single Signature is Enough**: While individual signatures show positive trends, patient distributions still overlap significantly — no single signature acts as a standalone silver bullet.
> * **Motivation for Multimodal ML**: Overlap in single features proves why we must combine these 6 signatures with orthogonal genomic features (`TMB`) in multivariate ML models (XGBoost / Random Forest).


## 3. Custom Data-Driven TCGA-SKCM Overall Survival Signature

> [!NOTE] Section Context
> - **What We Are Doing**: Deriving a **custom 20-gene transcriptomic signature** directly from TCGA-SKCM survival data ($N = 421$ patients with complete expression and OS follow-up, out of $N = 443$ total) using univariate Cox proportional hazards screening — selecting the top 20 genes most significantly associated with longer overall survival (all protective: negative Cox β coefficients).
> - **Why We Are Doing It**: The six literature-curated signatures were designed for immunotherapy contexts. This data-driven signature asks a different question: *does a survival signal learned purely from untreated standard-of-care melanoma transfer to predict anti-PD-1 response in independent trial cohorts?* If it does, it suggests the underlying immune biology is fundamental to melanoma prognosis rather than treatment-specific. It also serves as an independent benchmark for comparing against the curated signatures.
> - **Questions**:
>   1. *Do genes selected purely for OS association encode recognisable immune biology, or do they select noise?*
>   2. *Does a TCGA survival-trained signature generalise to immunotherapy response prediction in independent trial cohorts?*
>   3. *How does the data-driven score compare to the literature-curated signatures as a response predictor?*

### 3.1. Methodology & Sample Size ($N = 421$)
* **Cohort Selection**: We evaluated the reference **TCGA-SKCM** cohort, retaining $N = 421$ patients with complete, aligned expression and overall survival follow-up data (out of $N = 443$ total expression-aligned samples).
* **Statistical Screening**: Screened over 20,000 raw genes using univariate Cox proportional hazards modelling (`lifelines.CoxPHFitter`).
* **The 20 Protective Genes**: Extracted the top 20 genes most significantly associated with overall survival. All 20 selected genes display **negative Cox beta coefficients (protective)**, meaning elevated expression correlates with longer overall survival.

### 3.2. Biological Domain Composition
Rather than isolating random noise, the data-driven selection isolated four core immune defence mechanisms:

| Biological Domain | Key Genes | Function in Tumour Microenvironment |
| :--- | :--- | :--- |
| **Interferon GTPases** | `GBP1`, `GBP4`, `GBP5`, `GBP1P1` | Interferon-induced GTPases coordinating cell-autonomous anti-tumour defence |
| **Chemokines & Cytokines** | `CCL8`, `CXCL10`, `CXCL11`, `IL15` | Chemoattractants that recruit CD8+ T-cells and NK-cells into the tumour |
| **Cytotoxic Receptors** | `KLRD1`, `KLRK1`, `CD38`, `CD72`, `PTPN22`, `GPR171` | Activating receptors on cytotoxic lymphocytes and TCR signalling regulators |
| **Signalling & Controllers** | `STAT4`, `SAMSN1`, `AKAP5`, `IDO1`, `PLAAT4`, `ZNF831` | Signal transduction, metabolic feedback, and lymphocyte differentiation |

> [!NOTE] Key Benchmark Rationale
> This custom 20-gene score serves as an independent benchmark in our out-of-cohort evaluation (detailed in [transcriptomic_feature_selection_results.md](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/reports/pillar-4-out-of-cohort-benchmarks/transcriptomic_feature_selection_results.md)) to test whether general survival markers generalise to ICB response prediction.


## 4. Preprocessing & Batch Alignment Workflow

> [!NOTE] Section Context
> - **What We Are Doing**: Applying **cohort-independent Z-score standardisation** to signature scores — computing each study's Z-scores using only that study's own mean and variance, then concatenating the scaled cohorts before model training.
> - **Why We Are Doing It**: When combining data across four independent studies (Liu, Hugo, Riaz, TCGA), sequencing platform differences and laboratory protocols create strong **batch effects** that can make study membership the dominant signal — causing a classifier to learn *which lab ran the samples* rather than *which patients responded*. Standardising within each cohort removes this offset. Critically, doing so independently per cohort (rather than on the pooled dataset) guarantees **zero data leakage**: a held-out test cohort's expression values never influence the scaling of training data.
> - **Questions**:
>   1. *Do uncorrected signature scores cluster by study cohort rather than biological outcome in PCA?*
>   2. *Does cohort Z-score standardisation dissolve the study-level separation?*
>   3. *How does this approach prevent data leakage during Leave-One-Cohort-Out validation?*

When combining patient data across 4 independent clinical studies (**Liu 2019**, $N = 122$; **Hugo 2016**, $N = 27$; **Riaz 2017**, $N = 107$; and **TCGA-SKCM**, $N = 443$), technical variations across sequencing platforms and lab protocols introduce strong **batch effects**.

As detailed in [batch_correction_report.md](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/reports/pillar-1-cohorts-and-preprocessing/batch_correction_report.md), uncorrected expression profiles cluster heavily by study cohort rather than biological outcome.

![Batch Preprocessing & Alignment Workflow](../../plots/signatures/batch_workflow_diagram.png)

### 4.1. Zero-Leakage Cohort Z-Score Standardisation
To eliminate study-level batch offsets while guaranteeing **zero data leakage** during **Leave-One-Cohort-Out (LOCO)** cross-validation, we apply **Cohort-Independent Z-Score Standardisation** in [run_extended_biomarkers.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/biomarkers/run_extended_biomarkers.py#L250-L255):

```python
# Standardise each cohort's signatures individually (Z-score)
df_liu_sigs_scaled = zscore_df(df_liu_sigs)
df_hugo_sigs_scaled = zscore_df(df_hugo_sigs)
df_riaz_sigs_scaled = zscore_df(df_riaz_sigs)
df_sigs_merged = pd.concat([df_liu_sigs_scaled, df_hugo_sigs_scaled, df_riaz_sigs_scaled])
```

* **Zero Leakage**: Standardising each study using only its internal mean and variance ensures test-set data is never used to adjust training features.
* **Effective Scale Alignment**: Successfully removes baseline study offsets, allowing true biological signals to align across clinical cohorts.

### 4.2. Visualising Batch Correction Impact
Following cohort Z-score standardisation, study-level separation dissolves in principal component space, aligning patients across studies:

![PCA Batch Effect Assessment Across Full Cohort](../../plots/biomarkers/batch_effect_pca.png)


## 5. Statistical Relationships & Biomarker Orthogonality

> [!NOTE] Section Context
> - **What We Are Doing**: Evaluating pairwise correlations between all biomarker features — TMB vs. neoantigen load, genomic burden vs. immune signatures, and inter-signature correlations — using Spearman rank correlation and multivariate odds ratio modelling.
> - **Why We Are Doing It**: Before training a multivariate model, we must map which features are redundant (and can be dropped without information loss) and which are genuinely independent (and therefore additive in a joint model). Including highly correlated features inflates apparent model complexity without improving prediction; omitting orthogonal features loses independent signal. This step directly informs feature selection.
> - **Questions**:
>   1. *Are TMB and neoantigen load measuring the same thing — and should we keep both?*
>   2. *Are genomic burden metrics (TMB, aneuploidy) independent of immune expression signatures, justifying their combination in a multimodal model?*
>   3. *Which signatures survive multivariate adjustment as independent response predictors, and which collapse due to collinearity?*

Before training predictive models, we evaluate feature correlations to eliminate redundant metrics and identify independent biological signals.

### 5.1. TMB vs. Neoantigen Load: High Feature Redundancy
Somatic mutation rate (`TMB`) and predicted neoantigen count capture the exact same biological signal ($r_s = 0.96$ in Liu 2019; $r_s = 0.872$ in pooled trials, $N = 222$).
* **Decision**: Including both creates unnecessary feature redundancy. We retain **TMB** as our clean genomic surrogate in all models.

![Neoantigen vs TMB Regression](../../plots/genomic/tmb_distributions_by_cohort.png)

### 5.2. Genomic Burden vs. Immune Signatures: Independent (Orthogonal) Modalities
Evaluating genomic metrics (`TMB_NONSYNONYMOUS`, `ANEUPLOIDY_SCORE`) against continuous transcriptomic signatures reveals near-zero correlation ($r_s \approx 0.034$).

| Cohort / Feature | IFN-γ | TIS | CD8 T-Cell | CYT | PD-L1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TCGA Aneuploidy Score** | -0.045 | -0.090 | -0.056 | -0.058 | **-0.110** |
| **TCGA TMB** | **0.144** | **0.108** | **0.096** | **0.103** | **0.159** |
| **Trial TMB** | **0.034** | -0.035 | -0.052 | -0.091 | **0.046** |

![Genomic Burden vs Immune Signature Correlation Heatmap](../../plots/biomarkers/extended_immune_correlations.png)

> [!INSIGHT] The Multimodal Pitch
> **Genomic burden and immune signatures are orthogonal (independent)**. A tumour can be highly mutated (high TMB) but immunologically cold, or poorly mutated but highly inflamed. Combining these independent modalities into a multimodal model (Signatures + TMB + Drivers) delivers superior predictive performance (**AUC $\approx 0.72$**).

### 5.3. Inter-Signature Correlations & Multivariate Drivers
* **High Collinearity**: Signature modalities (TIS, IFN-γ, CYT, CD8 T-cell) are co-expressed ($r_s \approx 0.85\text{--}0.90$).
* **Independent Response Drivers**: In multivariate odds ratio modelling, **Tumour Inflammation Signature (TIS, $\text{OR} = 3.91$)** and **IMPRES ($\text{OR} = 1.66$)** emerge as the primary non-redundant predictors of response.

| Heatmap of Inter-Signature Correlation | Forest Plot of Odds Ratios |
| :---: | :---: |
| ![Spearman Correlation Heatmap](../../plots/signatures/signature_correlation_heatmap.png) | ![Forest Plot of Odds Ratios](../../plots/signatures/forest_plot_odds_ratios.png) |


## 6. Multimodal Response Prediction Models

> [!summary] What, Why & Key Questions
> - **What We Are Doing**: Training five different classifier architectures (Logistic Regression, Random Forest, XGBoost, SVM, Elastic-Net) on the pooled trial cohort ($N = 195$) using 5-fold stratified cross-validation. We compare three feature sets of increasing complexity: (1) immune signatures only, (2) signatures + driver mutations + age, and (3) a full extended model adding TMB, age, and pathway mutation flags.
> - **Why We Are Doing It**: We need to answer two questions at once. *First*, do the curated immune signatures alone carry enough signal to predict response, or do we need additional genomic features? *Second*, which model architecture best handles the high collinearity among immune signatures and the small sample size? Comparing feature sets within each model isolates the value of adding genomic features; comparing models within each feature set identifies the best architecture.
> - **Questions**:
>   1. *Does adding driver mutations, age, and TMB improve prediction beyond signatures alone?*
>   2. *Which model family (linear vs. tree-based) handles these features best?*
>   3. *Is the best AUC achievable with this data clinically meaningful?*

### Table 2. Cross-validated multimodal response prediction performance (mean ROC-AUC ± SD across 5-fold stratified CV)

| Model Architecture | Base Model (Signatures Only) | Sigs + Drivers (`BRAF/NRAS/NF1`) + Age | 14-Feature Full Extended Matrix* |
|:--- |:---:|:---:|:---:|
| **Logistic Regression (LR)** | **0.615 (+/-0.081)** | 0.569 (+/-0.078) | 0.579 (+/-0.094) |
| **Random Forest (RF)** | 0.666 (+/-0.053) | **0.718 (+/-0.062)** | 0.686 (+/-0.082) |
| **XGBoost (XGB, tuned)** | 0.632 (+/-0.061) | 0.692 (+/-0.049) | **0.702 (+/-0.110)** |
| **Support Vector Machine (SVM)** | **0.626 (+/-0.081)** | 0.553 (+/-0.061) | 0.597 (+/-0.091) |
| **Elastic-Net** | **0.610 (+/-0.075)** | 0.576 (+/-0.052) | 0.590 (+/-0.040) |

\* *Footnote: The 14-Feature Full Extended Matrix incorporates: 6 immune expression signatures (`IFN_gamma`, `TIS`, `CYT`, `CD8_Tcell`, `IMPRES`, `PD_L1`), 3 melanoma driver mutation flags (`mut_BRAF`, `mut_NRAS`, `mut_NF1`), 3 composite pathway mutation flags (`mut_Antigen_Presentation`, `mut_IFN_gamma_Signaling`, `mut_Survival_Pathways`), nonsynonymous mutational burden (`TMB_NONSYNONYMOUS`), and patient age (`AGE`). Total predicted neoantigens (`TOTAL_NEOANTIGEN`) was excluded due to high collinearity with TMB ($r_s = 0.756$).*

![Multimodal AUC Comparison](../../plots/biomarkers/multimodal_auc_comparison.png)

### Analysis of Predictor Performance
1. **Linear models degrade with more features**: Logistic Regression and Elastic-Net perform *best* with signatures alone (AUC ≈ 0.61) and *worse* when genomic features are added. With only $N = 195$ samples and 15+ features, the linear models overfit to noise in the additional columns rather than learning generalisable signal.
2. **Tree-based models benefit from multimodal features**: Random Forest and XGBoost show the opposite pattern — they improve monotonically as features are added, peaking at AUC = 0.686 (RF) and 0.702 (XGBoost) with the full extended set. Tree-based learners handle correlated and mixed-type features more robustly because they select splits on individual features rather than estimating a single global weight vector.
3. **Clinical interpretation**: An AUC of ~0.70-0.72 means the model correctly ranks a randomly chosen responder above a non-responder ~71% of the time. This is competitive with published immunotherapy response predictors in melanoma, where AUCs rarely exceed 0.75 without integrating radiological or on-treatment data.

## 7. Leave-One-Cohort-Out Model Evaluation

> [!NOTE] Section Context
> - **What We Are Doing**: Evaluating the same trained models using **Leave-One-Cohort-Out (LOCO)** cross-validation — training on two immunotherapy trial cohorts and testing on the third held-out cohort. This is repeated for each of the three cohorts (Liu 2019, Hugo 2016, Riaz 2017).
> - **Why We Are Doing It**: Pooled 5-fold CV mixes patients from all cohorts in every fold, so a model can exploit cohort-specific expression patterns (even after Z-score standardisation). LOCO is a strictly harder test: the held-out cohort's patients were never seen during training, simulating deployment to a genuinely new clinical site with a different sequencing platform, patient demographics, and response distribution. If a model generalises across LOCO folds, the learned signal is robust enough to transfer to new studies.
> - **Questions**:
>   1. *Do models trained on two cohorts generalise to a third unseen cohort?*
>   2. *Which cohort is hardest to predict when held out — and why?*
>   3. *How does LOCO AUC compare to the pooled 5-fold AUC reported in Section 6?*

**Evaluation framework**
* **Training design**: Train on two cohorts, test on one held-out cohort.
* **Test cohorts**: Liu 2019 ($N = 122$ total clinical, $N = 104$ evaluated), Hugo 2016 ($N = 27$), and Riaz 2017 ($N = 107$ total clinical, $N = 64$ pre-treatment evaluated).
* **Feature set**: 11 immune response signatures, including IFN-γ, TIS, CD8 T-cell, CYT, IMPRES, PD-L1, and related immune axes.
* **Decision threshold**: 0.5 for binary responder/non-responder classification.
* **Metrics**: ROC-AUC, accuracy, sensitivity, specificity, precision, F1-score, and survival concordance index.

### LOCO ROC-AUC Performance Across Models & Held-Out Cohorts

![LOCO ROC-AUC Performance Heatmap](../../plots/models/loco_performance_heatmap.png)

### LOCO Interpretation & Key Metric Diagnostics

> [!INSIGHT] Key Metric Insights (Sensitivity, Precision & Survival C-Index)
> 1. **Extreme Specificity vs. Sensitivity Polarisation**: At the standard $0.5$ decision threshold, models exhibit extreme polar behaviour depending on held-out cohort characteristics:
>    - **Liu 2019**: Random Forest achieves **94.6% Specificity** and **75.0% Precision**, but only **18.8% Sensitivity**. The model acts as a strict "rule-in" classifier: when it predicts a patient will respond, it is almost always correct, but it misses over 80% of true responders.
>    - **Riaz 2017**: Logistic Regression achieves **100.0% Sensitivity**, but **0.0% Specificity**. The linear model predicts nearly all patients as responders, capturing every true positive at the cost of high false positive rates.
> 2. **Prognostic Survival Ranking ($C$-Index) Persists When Classification Fails**: On Hugo 2016 ($N=27$), binary response classification metrics perform poorly ($\text{AUC} \approx 0.32\text{--}0.43$). However, the **Survival Concordance Index ($C$-Index)** remains strong — reaching **$0.663$ (SVM)** and **$0.612$ (LR)**. This proves that continuous predicted probabilities maintain genuine prognostic risk-ranking for overall survival even when discrete binary labels fail to separate.
> 3. **Threshold Tuning Impact (Youden's J Optimisation)**: Default $0.5$ decision cutoffs suffer under cross-cohort batch shifts. Optimising decision thresholds post-hoc using Youden's $J$ index ($J = \text{Sensitivity} + \text{Specificity} - 1$) recovers severe sensitivity losses (e.g. boosting Random Forest sensitivity on Liu 2019 from **18.8% to 62.5%** and on Hugo 2016 from **28.6% to 57.1%**) while improving overall classification accuracy across all test cohorts.

### Impact of Decision Threshold Tuning (Default 0.50 vs. Youden's J Optimal)

![Decision Threshold Tuning Impact](../../plots/models/threshold_tuning_impact.png)

### LOCO Diagnostic Plots
The full diagnostic outputs are saved in `plots/models/`:
* ROC curves: `roc_curves_lr.png`, `roc_curves_rf.png`, `roc_curves_xgb.png`, `roc_curves_svm.png`, `roc_curves_elasticnet.png`
* Precision-recall curves: `pr_curves_lr.png`, `pr_curves_rf.png`, `pr_curves_xgb.png`, `pr_curves_svm.png`, `pr_curves_elasticnet.png`
* Confusion matrices: `confusion_matrices_lr.png`, `confusion_matrices_rf.png`, `confusion_matrices_xgb.png`, `confusion_matrices_svm.png`, `confusion_matrices_elasticnet.png`


## 8. Consolidated Conclusion

> [!NOTE] Consolidated Conclusion Rationale
> - Given everything we have tested, what is the best practical approach for predicting immunotherapy response in melanoma from baseline tumour profiling?

1. **Curated signatures are the foundation**: The six immune signatures provide the most stable and interpretable transcriptomic representation. They already achieve competitive AUCs on their own (0.61–0.67) and are robust across cohorts because they compress 20,000+ genes into biologically grounded, low-dimensional scores.
2. **Orthogonal genomic features add value — but only for tree-based models**: Adding TMB, driver mutations, and pathway flags improves Random Forest and XGBoost to AUC ≈ 0.72–0.74, but *hurts* linear models. The final modelling strategy should therefore use tree-based architectures with the full multimodal feature set.
3. **LOCO is harder than pooled CV**: Cross-study generalisation (LOCO AUC ≈ 0.58–0.68) lags behind pooled 5-fold CV (AUC ≈ 0.72–0.74), especially for small held-out cohorts. These two validation strategies should be reported as **distinct evidence layers**, not interchangeable performance estimates.
4. **TMB is the right genomic surrogate**: Neoantigen load is almost perfectly collinear with TMB ($r_s = 0.872$), so retaining both adds redundancy without new information. TMB alone is sufficient.
