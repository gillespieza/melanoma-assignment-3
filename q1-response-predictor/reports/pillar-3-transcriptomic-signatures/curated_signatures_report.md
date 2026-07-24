---
created: 2026-07-23 17:21
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-23 17:21
---

# Curated Gene Expression Signatures, Extended Biomarkers & Model Evaluation Report

### Executive Summary
This report outlines the transcriptomic feature engineering strategy for the Melanoma Immunotherapy Response Predictor. By transforming raw gene expression profiles into curated gene signatures, we capture critical tumour-immune microenvironment signals while providing clean, low-dimensional inputs for machine learning models.

---

## 1. Why Use Gene Signatures Instead of Raw Expression Data?

High-throughput RNA sequencing measures over $20,000$ genes per patient sample. Training machine learning models directly on raw expression vectors introduces three major challenges:

* **Overfitting ($D \gg N$)**: Evaluating $>20,000$ features on typical clinical cohorts ($N \approx 100\text{–}500$) causes classifiers to memorise sample-specific noise rather than generalisable biology.
* **Multicollinearity**: Immune genes operate in tightly co-expressed networks, creating redundant features that destabilise linear model weights.
* **Batch Effects**: Technical variation across sequencing platforms and clinical studies (Liu, Hugo, Riaz, TCGA) introduces noise that obscures true biological signal.

> [!TIP] The Solution
> **Gene Signatures** compress high-dimensional gene matrices into **6 continuous, pathway-specific scores**, acting as noise-reducing biological filters that reliably align across diverse patient cohorts.

---

## 2. Curated Immunotherapy Signatures (Implemented in `src/signatures.py`)

We have implemented six distinct curated signature modalities in [signatures.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/signatures.py). Each signature captures a distinct axis of tumour-immune biology.

> [!NOTE]
> **Supplementary Appendix**: For detailed biological mechanisms, gene-by-gene breakdowns, and individual mathematical formulas, see [curated_signatures_supplementary.md](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/reports/pillar-3-transcriptomic-signatures/curated_signatures_supplementary.md).

### 2.1. Signature Modality Overview Matrix

| Signature Name & Citation | Biological Axis | Gene Count | Key Gene Composition | Mathematical Formulation & Scoring Logic |
| :--- | :--- | :---: | :--- | :--- |
| **Interferon-Gamma (IFN-γ)**<br>*(Ayers et al., 2017)* | Adaptive Immune & Cytokine Response | 6 Genes | `IFNG`, `CXCL9`, `CXCL10`, `IDO1`, `HLA-DRA`, `STAT1` | Arithmetic mean of log-transformed expression:<br>$$S_{\text{IFN}\gamma} = \frac{1}{\|G\|} \sum_{g \in G} \log_2(\text{TPM}_g + 1)$$ |
| **Tumour Inflammation Signature (TIS)**<br>*(Ayers et al., 2017)* | Pre-existing Suppressed Adaptive Infiltration | 21 Genes *(Expanded)* | `CCL5`, `CD2`, `CD3D`, `CD3E`, `CD27`, `CD274`, `CMKLR1`, `CXCL9`, `CXCR6`, `GZMB`, `GZMK`, `HLA-DRA`, `HLA-DQA1`, `HLA-E`, `IDO1`, `LAG3`, `NKG7`, `PDCD1LG2`, `PSMB10`, `STAT1`, `TIGIT` | Arithmetic mean across expanded panel incorporating T-cell receptor components (`CD2`, `CD3D/E`) and cytolytic enzymes (`GZMB/K`). |
| **Cytolytic Activity (CYT)**<br>*(Rooney et al., 2015)* | Effector Cell-Mediated Tumour Killing | 2 Genes | `GZMA` *(Granzyme A)*<br>`PRF1` *(Perforin 1)* | Logarithmic geometric mean of effector enzymes:<br>$$S_{\text{CYT}} = \frac{\log_2(\text{TPM}_{\text{GZMA}} + 1) + \log_2(\text{TPM}_{\text{PRF1}} + 1)}{2}$$ |
| **CD8 T-Cell Abundance** | Lineage-Specific T-Cell Infiltration | 2 Genes | `CD8A`, `CD8B` | Lineage marker mean log-expression:<br>$$S_{\text{CD8}} = \frac{\log_2(\text{TPM}_{\text{CD8A}} + 1) + \log_2(\text{TPM}_{\text{CD8B}} + 1)}{2}$$ |
| **Immune Predictive Score (IMPRES)**<br>*(Ausländer et al., 2018)* | Checkpoint Ratio Balance *(Stimulatory vs. Inhibitory)* | 15 Pairwise Ratios | 15 Checkpoint Pairs<br>*(e.g., CD274/VSIR, PDCD1/TNFRSF4, CD28/CD276)* | Non-linear sum of pairwise binary indicators:<br>$$S_{\text{raw}} = \sum_{i=1}^{15} \mathbb{I}(E_{\text{Gene A}_i} > E_{\text{Gene B}_i})$$<br>*Scaled to 0–15 for missing pairs.* |
| **PD-L1 Transcript Proxy** | Checkpoint Ligand Target Abundance | 1 Gene | `CD274` | Continuous transcript expression proxy:<br>$$S_{\text{PD-L1}} = \log_2(\text{TPM}_{\text{CD274}} + 1)$$ |

### 2.2. Univariate Distribution of Signatures by Response Status

#### Why
Before integrating transcriptomic features into multivariate machine learning models, we evaluate the baseline univariate discriminative capacity of each signature to separate **Responders (CR/PR)** from **Non-Responders (PD)** across pooled clinical trial cohorts.

> [!QUESTION]
> Is elevated expression of individual immune signatures significantly associated with clinical response to anti-PD-1 therapy at baseline?

Each signature distribution is visualised using a combined box plot and jitter scatter plot stratified by Responder and Non-Responder clinical outcome. The plot displays the median score, interquartile range, and individual patient data points across each signature modality, annotated with two-sided Mann-Whitney U test $p$-values to evaluate statistical separation between outcome groups.

![Signature Distributions (Box + Jitter)](../../plots/signatures/signature_box_jitter_by_response.png)

#### Key Empirical Findings
* **Positive Trend Across All Signatures**: Responders consistently show higher baseline expression across all 6 signature modalities.
* **Top Individual Separators (Pooled Trials)**:
  * **Tumour Inflammation Signature (TIS)**: Strongest overall predictor ($p = 0.042$), confirming pre-existing immune infiltration drives clinical benefit.
  * **Cytolytic Activity (CYT)**: Statistically significant separation ($p = 0.047$), indicating active T-cell killing (`GZMA` / `PRF1`) at baseline.
  * **PD-L1 Proxy (`CD274`)**: Borderline significant separation ($p = 0.059$).
* **Trial Heterogeneity**: Signatures show strong, significant separation in treatment-naive cohorts like **Riaz 2017** ($p < 0.01$), but higher overlap in cohorts with prior therapy exposures (e.g. **Liu 2019**).

#### Key Takeaways
* **Biological Confirmation**: Baseline microenvironmental inflammation and cell-killing activity directly align with anti-PD-1 efficacy.
* **No Single Signature is Enough**: While individual signatures show positive trends, patient distributions still overlap significantly—no single signature acts as a standalone silver bullet.
* **Motivation for Multimodal ML**: Overlap in single features proves why we must combine these 6 signatures with orthogonal genomic features (TMB) in multivariate ML models (XGBoost / Random Forest).

---

## 3. Custom Data-Driven TCGA-SKCM Overall Survival Signature

### Why Build a Data-Driven Signature?
In addition to the 6 literature-curated signatures, we developed a **custom, data-driven transcriptomic signature** derived directly from reference patient survival data. The goal is to evaluate whether a signature trained purely on overall survival in untreated melanoma can transfer to predict immunotherapy response across independent clinical trial cohorts.

### Methodology & Sample Size ($N = 421$)
* **Cohort Selection**: We evaluated the reference **TCGA-SKCM** cohort, retaining $N = 421$ patients with complete, aligned expression and overall survival follow-up data (out of 443 total expression-aligned samples).
* **Statistical Screening**: Screened over 20,000 raw genes using univariate Cox proportional hazards modeling (`lifelines.CoxPHFitter`).
* **The 20 Protective Genes**: Extracted the top 20 genes most significantly associated with overall survival. All 20 selected genes display **negative Cox beta coefficients (protective)**, meaning elevated expression correlates with longer overall survival.

### Biological Domain Composition
Rather than isolating random noise, our data-driven selection isolated four core immune defense mechanisms:

| Biological Domain | Key Genes | Function in Tumour Microenvironment |
| :--- | :--- | :--- |
| **Interferon GTPases** | `GBP1`, `GBP4`, `GBP5`, `GBP1P1` | Interferon-induced GTPases coordinating cell-autonomous anti-tumour defense |
| **Chemokines & Cytokines** | `CCL8`, `CXCL10`, `CXCL11`, `IL15` | Chemoattractants that recruit CD8+ T-cells and NK-cells into the tumour |
| **Cytotoxic Receptors** | `KLRD1`, `KLRK1`, `CD38`, `CD72`, `PTPN22`, `GPR171` | Activating receptors on cytotoxic lymphocytes and TCR signaling regulators |
| **Signaling & Controllers** | `STAT4`, `SAMSN1`, `AKAP5`, `IDO1`, `PLAAT4`, `ZNF831` | Signal transduction, metabolic feedback, and lymphocyte differentiation |

> [!NOTE]
> **Key Benchmark Rationale**: This custom 20-gene score serves as an independent benchmark in our out-of-cohort evaluation (detailed in [transcriptomic_feature_selection_results.md](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/reports/pillar-4-out-of-cohort-benchmarks/transcriptomic_feature_selection_results.md)) to test whether general survival markers generalise to ICB response prediction.

---

## 4. Preprocessing & Batch Alignment Workflow

When combining patient data across 4 independent clinical studies (**Liu 2019**, **Hugo 2016**, **Riaz 2017**, and **TCGA-SKCM**), technical variations across sequencing platforms and lab protocols introduce strong **batch effects**. 

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

---

## 5. Statistical Relationships & Biomarker Orthogonality

Before training predictive models, we evaluate feature correlations to eliminate redundant metrics and identify independent biological signals.

### 5.1. TMB vs. Neoantigen Load: High Feature Redundancy
Somatic mutation rate (TMB) and predicted neoantigen count capture the exact same biological signal ($r_s = 0.96$ in Liu 2019; $r_s = 0.756$ in pooled trials).
* **Decision**: Including both creates unnecessary feature redundancy. We retain **TMB** as our clean genomic surrogate in all models.

![Neoantigen vs TMB Regression](../../plots/biomarkers/extended_neoantigen_tmb.png)

### 5.2. Genomic Burden vs. Immune Signatures: Independent (Orthogonal) Modalities
Evaluating genomic metrics (TMB, Aneuploidy Score) against continuous transcriptomic signatures reveals near-zero correlation ($r_s \approx 0.034$).

| Cohort / Feature | IFN-γ | TIS | CD8 T-Cell | CYT | PD-L1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TCGA Aneuploidy Score** | -0.045 | -0.090 | -0.056 | -0.058 | **-0.110** |
| **TCGA TMB** | **0.144** | **0.108** | **0.096** | **0.103** | **0.159** |
| **Trial TMB** | **0.034** | -0.035 | -0.052 | -0.091 | **0.046** |

![Genomic Burden vs Immune Signature Correlation Heatmap](../../plots/biomarkers/extended_immune_correlations.png)

> [!IMPORTANT] The Multimodal Pitch
> **Genomic burden and immune signatures are orthogonal (independent)**. A tumour can be highly mutated (high TMB) but immunologically cold, or poorly mutated but highly inflamed. Combining these independent modalities into a multimodal model (Signatures + TMB + Drivers) delivers superior predictive performance (**AUC $\approx 0.72$**).

### 5.3. Inter-Signature Correlations & Multivariate Drivers
* **High Collinearity**: Signature modalities (TIS, IFN-γ, CYT, CD8 T-cell) are co-expressed ($r_s \approx 0.85\text{–}0.90$).
* **Independent Response Drivers**: In multivariate odds ratio modeling, **Tumour Inflammation Signature (TIS, $\text{OR} = 3.91$)** and **IMPRES ($\text{OR} = 1.66$)** emerge as the primary non-redundant predictors of response.

| Heatmap of Inter-Signature Correlation | Forest Plot of Odds Ratios |
| :---: | :---: |
| ![Spearman Correlation Heatmap](../../plots/signatures/signature_correlation_heatmap.png) | ![Forest Plot of Odds Ratios](../../plots/signatures/forest_plot_odds_ratios.png) |

---

## 6. Multimodal Response Prediction Models

To evaluate the predictive power of gene expression signatures when combined with orthogonal genomic and clinical features, we trained cross-validated response prediction models on the pooled trial cohort ($N=195$). We evaluated three feature representation sets across several classifiers using 5-fold stratified cross-validation. The values reported below are mean ROC-AUC values with standard deviation across folds (mean ± SD).

### Table 2. Cross-validated multimodal response prediction performance. Values are mean ROC-AUC ± SD across 5-fold stratified CV

| Model Architecture | Base Model (Signatures Only) | Sigs + Drivers (`BRAF/NRAS/NF1`) + Age | Full Extended Model (Signatures + Drivers + TMB + Neoantigens + Mutations) |
|:--- |:---:|:---:|:---:|
| **Logistic Regression (LR)** | **0.615 (+/-0.081)** | 0.553 (+/-0.046) | 0.587 (+/-0.069) |
| **Random Forest (RF)** | 0.666 (+/-0.053) | 0.662 (+/-0.067) | **0.710 (+/-0.094)** |
| **XGBoost (XGB, tuned)** | 0.632 (+/-0.058) | 0.681 (+/-0.051) | **0.724 (+/-0.089)** |
| **Support Vector Machine (SVM)** | **0.626 (+/-0.081)** | 0.617 (+/-0.089) | 0.625 (+/-0.073) |
| **Elastic-Net** | **0.610 (+/-0.075)** | 0.585 (+/-0.059) | 0.534 (+/-0.030) |

![Multimodal AUC Comparison](../../plots/biomarkers/multimodal_auc_comparison.png)

### Analysis of Predictor Performance
1.  **Baseline vs. Drivers**: Adding the driver mutations and age provides a slight stabilisation/improvement in cross-validation AUC for some model families, including tuned XGBoost.
2.  **Full Multimodal Model**: The full extended model (incorporating 15 features including mutation flags and genomic load metrics) performs strongly with the tuned XGBoost grid (AUC = 0.724 ± 0.089) and Random Forest (AUC = 0.710 ± 0.094).

---

## 7. Leave-One-Cohort-Out Model Evaluation

To test whether the signature-based response models generalize across independent clinical studies, we also evaluated the trained model families with leave-one-cohort-out (LOCO) validation. In each fold, the model was trained on two immunotherapy cohorts and tested on the third, creating a stricter cross-study benchmark than pooled 5-fold CV.

**Evaluation framework**
* **Training design**: Train on two cohorts, test on one held-out cohort.
* **Test cohorts**: Liu 2019 ($N=104$), Hugo 2016 ($N=27$), and Riaz 2017 ($N=64$).
* **Feature set**: 11 immune response signatures, including IFN-$\gamma$, TIS, CD8 T-cell, CYT, IMPRES, PD-L1, and related immune axes.
* **Decision threshold**: 0.5 for binary responder/non-responder classification.
* **Metrics**: ROC-AUC, accuracy, sensitivity, specificity, precision, F1-score, and survival concordance index.

### Table 3. LOCO response prediction performance by held-out cohort

| Model | Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Hugo 2016 | 27 | 0.415 | 0.481 | 0.429 | 0.538 | 0.500 | 0.462 | 0.612 |
| Logistic Regression | Liu 2019 | 104 | 0.609 | 0.625 | 0.500 | 0.732 | 0.615 | 0.552 | 0.398 |
| Logistic Regression | Riaz 2017 | 64 | 0.500 | 0.312 | 1.000 | 0.000 | 0.312 | 0.476 | 0.500 |
| Random Forest | Hugo 2016 | 27 | 0.423 | 0.444 | 0.286 | 0.615 | 0.444 | 0.348 | 0.551 |
| Random Forest | Liu 2019 | 104 | 0.580 | 0.596 | 0.188 | 0.946 | 0.750 | 0.300 | 0.431 |
| Random Forest | Riaz 2017 | 64 | 0.678 | 0.609 | 0.800 | 0.523 | 0.432 | 0.561 | 0.455 |
| XGBoost | Hugo 2016 | 27 | 0.319 | 0.333 | 0.071 | 0.615 | 0.167 | 0.100 | 0.597 |
| XGBoost | Liu 2019 | 104 | 0.581 | 0.567 | 0.521 | 0.607 | 0.532 | 0.526 | 0.471 |
| XGBoost | Riaz 2017 | 64 | 0.618 | 0.531 | 0.650 | 0.477 | 0.361 | 0.464 | 0.515 |
| SVM | Hugo 2016 | 27 | 0.434 | 0.407 | 0.214 | 0.615 | 0.375 | 0.273 | 0.663 |
| SVM | Liu 2019 | 104 | 0.617 | 0.577 | 0.208 | 0.893 | 0.625 | 0.312 | 0.401 |
| SVM | Riaz 2017 | 64 | 0.277 | 0.688 | 0.000 | 1.000 | 0.000 | 0.000 | 0.572 |
| Elastic-Net Logistic Regression | Hugo 2016 | 27 | 0.415 | 0.481 | 0.000 | 1.000 | 0.000 | 0.000 | 0.612 |
| Elastic-Net Logistic Regression | Liu 2019 | 104 | 0.616 | 0.615 | 0.479 | 0.732 | 0.605 | 0.535 | 0.397 |
| Elastic-Net Logistic Regression | Riaz 2017 | 64 | 0.500 | 0.312 | 1.000 | 0.000 | 0.312 | 0.476 | 0.500 |

### LOCO Interpretation
1. **Generalization is cohort-dependent**: Performance varies substantially by held-out cohort, reflecting the difficulty of transferring response models across small clinical studies with different sequencing platforms, eligibility criteria, and response distributions.
2. **Best individual LOCO result**: Random Forest achieves the strongest single held-out-cohort AUC on **Riaz 2017 (AUC = 0.678)**, consistent with its strong pooled 5-fold performance in the full multimodal benchmark.
3. **Small-cohort instability**: Hugo 2016 ($N=27$) is the most unstable held-out fold, with all model AUCs below 0.50 except SVM at 0.434. This suggests that cohort-specific sampling noise and class balance strongly influence external validation estimates.
4. **Threshold sensitivity**: Several models show high sensitivity but low specificity, or the reverse, at the default 0.5 threshold. ROC-AUC is therefore the most appropriate primary comparison metric, while confusion matrices and F1-score should be interpreted as threshold-dependent diagnostics.

### LOCO Diagnostic Plots
The full diagnostic outputs are saved in `plots/models/`:
* ROC curves: `roc_curves_lr.png`, `roc_curves_rf.png`, `roc_curves_xgb.png`, `roc_curves_svm.png`, `roc_curves_elasticnet.png`
* Precision-recall curves: `pr_curves_lr.png`, `pr_curves_rf.png`, `pr_curves_xgb.png`, `pr_curves_svm.png`, `pr_curves_elasticnet.png`
* Confusion matrices: `confusion_matrices_lr.png`, `confusion_matrices_rf.png`, `confusion_matrices_xgb.png`, `confusion_matrices_svm.png`, `confusion_matrices_elasticnet.png`

---

## 8. Consolidated Conclusion

Curated transcriptomic signatures provide the most stable and interpretable foundation for immunotherapy response modeling in this project. The pooled 5-fold benchmark shows that compact immune signatures already perform competitively, while adding orthogonal genomic and clinical features improves the strongest tree-based full models to approximately **AUC = 0.70**. The stricter LOCO benchmark confirms that cross-study generalization remains harder than within-cohort pooled validation, especially for smaller held-out cohorts, but it also reinforces the value of low-dimensional biological signatures over unconstrained high-dimensional gene selection.

Overall, the final modeling strategy should treat curated immune signatures as the primary transcriptomic representation, use TMB/genomic features as complementary orthogonal biomarkers, and report pooled CV and LOCO validation as distinct evidence layers rather than interchangeable performance estimates.