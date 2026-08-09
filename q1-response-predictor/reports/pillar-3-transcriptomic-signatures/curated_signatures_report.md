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
updated: 2026-08-07 15:26
---
# Curated Gene Expression Signatures, Extended Biomarkers & Model Evaluation Report

### Executive Summary
This report outlines the transcriptomic feature engineering strategy for the Melanoma Immunotherapy Response Predictor. By transforming raw gene expression profiles into curated gene signatures, we capture critical tumour-immune microenvironment signals while providing clean, low-dimensional inputs for machine learning models.


## 1. Why Use Gene Signatures Instead of Raw Expression Data?

> [!NOTE] Section Context
> - **What**: Justifying the decision to compress $>20,000$ raw RNA-seq gene measurements into **6 curated pathway-level scores** rather than feeding raw expression vectors directly into machine learning models.
> - **Why**: Raw gene expression is high-dimensional ($D \gg N$), multicollinear, and batch-contaminated. Directly training classifiers on 20,000 features across cohorts of $N \approx 100\text{--}500$ guarantees overfitting to study-specific noise rather than generalisable biology. Pathway signatures act as noise-reducing biological filters.
> - **Questions**:
>   1. *Why can't we just use all 20,000 genes as features?*
>   2. *How does gene-level collinearity undermine model interpretability?*
>   3. *How do batch effects across Liu (122), Hugo (27), and Riaz (107) corrupt feature scaling?*

High-throughput RNA sequencing measures over $20,000$ genes per patient sample. Training machine learning models directly on raw expression vectors introduces three major challenges:

* **Overfitting ($D \gg N$)**: Evaluating $>20,000$ features on typical clinical cohorts ($N \approx 100\text{--}500$) causes classifiers to memorise sample-specific noise rather than generalisable biology.
* **Multicollinearity**: Immune genes operate in tightly co-expressed networks, creating redundant features that destabilise linear model weights.
* **Batch Effects**: Technical variation across sequencing platforms and clinical studies (Liu 2019, Hugo 2016, Riaz 2017) introduces noise that obscures true biological signal.

> [!TIP] The Solution
> **Gene Signatures** compress high-dimensional gene matrices into **6 continuous, pathway-specific scores**, acting as noise-reducing biological filters that reliably align across diverse patient cohorts.


## 2. Curated Immunotherapy Signatures (Implemented in `src/signatures.py`)

> [!NOTE] Section Context
> - **What**: Implementing six literature-curated gene expression signatures in [`signatures.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/signatures.py), each capturing a distinct axis of tumour-immune biology — from IFN-γ cytokine signalling and CD8 T-cell infiltration to checkpoint ligand abundance and effector killing capacity.
> - **Why**: Rather than selecting signatures arbitrarily, each was chosen because it has been independently validated in published anti-PD-1 trials. Using established, biologically-grounded scores rather than ad-hoc gene selections reduces the risk of overfitting and makes our model interpretable to a clinical audience.
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
| **Interferon-Gamma (IFN-γ)**<br>*(Ayers et al., 2017)* | Adaptive Immune & Cytokine Response | 6 Genes | `IFNG`, `CXCL9`, `CXCL10`, `IDO1`, `HLA-DRA`, `STAT1` | Arithmetic mean of log-transformed expression:<br>$S_{\text{IFN-}\gamma} = \frac{1}{\|G\|} \sum_{g \in G} \log_2(\text{TPM}_g + 1)$ |
| **Tumour Inflammation Signature (TIS)**<br>*(Ayers et al., 2017)* | Pre-existing Suppressed Adaptive Infiltration | 21 Genes *(Expanded)* | `CCL5`, `CD2`, `CD3D`, `CD3E`, `CD27`, `CD274`, `CMKLR1`, `CXCL9`, `CXCR6`, `GZMB`, `GZMK`, `HLA-DRA`, `HLA-DQA1`, `HLA-E`, `IDO1`, `LAG3`, `NKG7`, `PDCD1LG2`, `PSMB10`, `STAT1`, `TIGIT` | Arithmetic mean across expanded panel incorporating T-cell receptor components (`CD2`, `CD3D`, `CD3E`) and cytolytic enzymes (`GZMB`, `GZMK`). |
| **Cytolytic Activity (CYT)**<br>*(Rooney et al., 2015)* | Effector Cell-Mediated Tumour Killing | 2 Genes | `GZMA` *(Granzyme A)*<br>`PRF1` *(Perforin 1)* | Logarithmic geometric mean of effector enzymes:<br>$$S_{\text{CYT}} = \frac{\log_2(\text{TPM}_{\text{GZMA}} + 1) + \log_2(\text{TPM}_{\text{PRF1}} + 1)}{2}$$ |
| **CD8 T-Cell Abundance** | Lineage-Specific T-Cell Infiltration | 2 Genes | `CD8A`, `CD8B` | Lineage marker mean log-expression:<br>$$S_{\text{CD8}} = \frac{\log_2(\text{TPM}_{\text{CD8A}} + 1) + \log_2(\text{TPM}_{\text{CD8B}} + 1)}{2}$$ |
| **Immune Predictive Score (IMPRES)**<br>*(Ausländer et al., 2018)* | Checkpoint Ratio Balance *(Stimulatory vs. Inhibitory)* | 15 Pairwise Ratios | 15 Checkpoint Pairs<br>*(e.g., `CD274`/`VSIR`, `PDCD1`/`TNFRSF4`, `CD28`/`CD276`)* | Non-linear sum of pairwise binary indicators:<br>$$S_{\text{raw}} = \sum_{i=1}^{15} \mathbb{I}(E_{\text{Gene A}_i} > E_{\text{Gene B}_i})$$<br>*Scaled to 0–15 for missing pairs.* |
| **PD-L1 Transcript Proxy** | Checkpoint Ligand Target Abundance | 1 Gene | `CD274` | Continuous transcript expression proxy:<br>$$S_{\text{PD-L1}} = \log_2(\text{TPM}_{\text{CD274}} + 1)$$ |

### 2.2. Univariate Distribution of Signatures by Response Status

> [!NOTE] Section Context
> - **What**: Evaluating the baseline univariate discriminative capacity of each signature — plotting score distributions split by **Responders (CR/PR)** vs. **Non-Responders (PD)** across the pooled clinical trial cohorts ($N = 195$ response-annotated patients out of $N = 256$ total across three cohorts: Liu 2019 $n = 104/122$, Hugo 2016 $n = 27/27$, Riaz 2017 $n = 64/107$), annotated with two-sided Mann-Whitney U test $p$-values.
> - **Why**: Before combining signatures in a multivariate model, we need to know whether each one carries any signal at all on its own. A signature with no univariate separation is unlikely to help even in a joint model. This step validates that our chosen signatures are biologically grounded in the trial data — not just theoretically motivated.
> - **Questions**: *Is elevated expression of individual immune signatures significantly associated with clinical response to anti-PD-1 therapy at baseline?*


Each signature distribution is visualised using a **raincloud plot** — combining a half-violin KDE (showing the full distribution shape), a compact IQR box with whiskers, and a jittered strip of individual patient data points — stratified by Responder (CR/PR) and Non-Responder (PD) clinical outcome. All 12 features in our multimodal matrix (6 transcriptomic signatures, Macrophage STV score, M1/M2 ratio, 3 driver mutation flags, and TMB) are Z-score normalised to a shared axis for direct visual comparison.

![Immune Signature Distributions — Raincloud Plot](../../plots/signatures/signature_raincloud_by_response.png)

For a comprehensive view contrasting pooled single-variable effect magnitude against out-of-cohort generalisation, the **combined forest plot** below directly displays Cohen's d (standardised mean difference, Responder − Non-Responder; 95% bootstrap CI) alongside out-of-cohort discriminative ability (AUROC across held-out clinical trials: Liu 2019, Hugo 2016, Riaz 2017) across the six curated signatures in report-table order.

![Pooled Effect Size vs. Out-of-Cohort Generalisation (2x1 Grid)](../../plots/signatures/combined_forest_plots.png)

#### Key Empirical Findings
* **Positive Trend Across All Signatures**: Responders consistently show higher baseline scores across all 6 signature modalities (all Cohen's d > 0).
* **IMPRES is the strongest univariate separator** (d = +0.37, p = 0.002) — the only signature whose confidence interval excludes zero, confirming that checkpoint-ratio balance carries independent predictive signal.
* **IFN-γ and TIS show moderate positive trends** (d ≈ +0.16–0.17, p ≈ 0.05–0.09) that fall just short of the α = 0.05 threshold — consistent with genuine but noisy signal in these pooled heterogeneous cohorts.
* **CYT, CD8 T-cell, and PD-L1** show small positive effects (d ≈ 0.04–0.15) with wide confidence intervals spanning zero — individually insufficient as standalone discriminators.

> [!INSIGHT] Key Takeaways
> * **Biological Confirmation**: Baseline checkpoint-ratio balance (IMPRES) and microenvironmental inflammation (IFN-γ / TIS) directly align with anti-PD-1 efficacy.
> * **No Single Signature is Enough**: While all individual signatures trend positive, only IMPRES crosses the significance threshold — patient distributions still overlap significantly.
> * **Motivation for Multimodal ML**: Overlap in single features proves why we must combine these 6 signatures with orthogonal genomic features (`TMB`) in multivariate ML models (XGBoost / Random Forest).



## 3. Statistical Relationships & Biomarker Orthogonality

> [!NOTE] Section Context
> - **What**: Evaluating pairwise correlations between all biomarker features — TMB vs. neoantigen load, genomic burden vs. immune signatures, and inter-signature correlations — using Spearman rank correlation and multivariate odds ratio modelling.
> - **Why**: Before training a multivariate model, we must map which features are redundant (and can be dropped without information loss) and which are genuinely independent (and therefore additive in a joint model). Including highly correlated features inflates apparent model complexity without improving prediction; omitting orthogonal features loses independent signal. This step directly informs feature selection.
> - **Questions**:
>   1. *Are TMB and neoantigen load measuring the same thing — and should we keep both?*
>   2. *Are genomic burden metrics (TMB, aneuploidy) independent of immune expression signatures, justifying their combination in a multimodal model?*
>   3. *Which signatures survive multivariate adjustment as independent response predictors, and which collapse due to collinearity?*

Before training predictive models, we evaluate feature correlations to eliminate redundant metrics and identify independent biological signals.

### 3.1. TMB vs. Neoantigen Load: High Feature Redundancy
Somatic mutation rate (`TMB`) and predicted neoantigen count capture the exact same biological signal ($r_s = 0.96$ in Liu 2019; $r_s = 0.872$ in pooled trials, $N = 222$).
* **Decision**: Including both creates unnecessary feature redundancy. We retain **TMB** as our clean genomic surrogate in all models.

![Neoantigen vs TMB Regression](../../plots/genomic/tmb_distributions_by_cohort.png)

### 3.2. Genomic Burden vs. Immune Signatures: Independent (Orthogonal) Modalities

> [!NOTE] Analysis Scope
> - **What**: Computing Spearman rank correlations between nonsynonymous mutational burden (`TMB_NONSYNONYMOUS`) and all six curated transcriptomic immune signatures.
> - **Cohort**: Pooled ICI trial cohort ($N = 418$: Liu 2019, Hugo 2016, Riaz 2017, TCGA GDC 2025, Gide 2019, Van Allen 2015).
> - **Why**: Establishing whether genomic mutational burden and transcriptomic immune activity are independent axes of variation within the ICI-treated population — a prerequisite for justifying a multimodal (genomic + transcriptomic) model.

Spearman rank correlation between nonsynonymous TMB and the six curated immune signatures in the pooled ICI trial cohort ($N = 418$) reveals near-complete biological orthogonality across all signature axes ($|r_s| \leq 0.088$, all $p > 0.16$):

| Feature | IFN-γ | TIS | CD8 T-Cell | CYT | IMPRES | PD-L1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trial TMB** ($r_s$) | 0.088 | 0.031 | 0.013 | −0.002 | 0.039 | 0.081 |
| *p*-value | 0.155 | 0.612 | 0.837 | 0.978 | 0.525 | 0.194 |

![Nonsynonymous TMB vs. Curated Immune Signatures — ICI Trial Cohort (N=418)](../../plots/biomarkers/extended_immune_correlations.png)

> [!INSIGHT] The Multimodal Pitch
> **Genomic burden (TMB) and transcriptomic immune signatures are orthogonal, independent axes of variation** within the ICI-treated melanoma population. No meaningful linear or rank-order relationship exists between the number of nonsynonymous somatic mutations a tumour carries and its inflammatory transcriptomic state ($|r_s| \leq 0.088$, all $p > 0.16$). A tumour can be hypermutated but immunologically cold, or nearly diploid yet profoundly inflamed. This orthogonality is precisely what makes a multimodal model (Signatures + TMB + Drivers) theoretically justified and, as shown in Section 4, empirically superior to any single modality alone.

### 3.3. Inter-Signature Correlations & Multivariate Drivers
* **High Collinearity**: Signature modalities (TIS, IFN-γ, CYT, CD8 T-cell) are strongly co-expressed ($r_s \approx 0.85\text{--}0.90$), reflecting their shared biological basis in cytotoxic lymphocyte infiltration.
* **Independent Response Drivers**: In multivariate logistic regression restricted to the six curated signatures (Z-scored, $N = 195$), **Tumour Inflammation Signature (TIS)** and **IMPRES** emerge as the primary non-redundant predictors of response — both showing OR > 1, consistent with their univariate effect sizes. Signatures sharing the same biological axis (IFN-γ, CYT, CD8 T-cell) show attenuated or reversed coefficients due to multicollinearity; interpretation should focus on the joint model's overall discriminative performance rather than individual ORs.

![Spearman Correlation Heatmap](../../plots/signatures/signature_correlation_heatmap.png)

> [!INSIGHT] Key Insights — Signature Structure & Multivariate Drivers
> 1. **High collinearity within the cytotoxic axis**: IFN-γ, TIS, CYT, and CD8 T-cell co-vary so tightly ($r_s \approx 0.85$–$0.90$) that they effectively measure a single latent dimension — cytotoxic lymphocyte infiltration. Including all four in a linear model inflates variance and produces unreliable individual coefficients; the relevant quantity is the axis itself, not any one signature.
> 2. **TIS and IMPRES are the non-redundant predictors**: In multivariate regression, **TIS** and **IMPRES** are the only two signatures that retain independent predictive signal. This is biologically coherent: TIS captures the cytotoxic infiltration axis, while IMPRES encodes a mechanistically distinct immune checkpoint resistance score derived from ligand–receptor interaction ratios — it is genuinely orthogonal to the infiltration axis.
> 3. **Practical consequence for feature selection**: Rather than entering all six signatures as raw features (which would introduce severe multicollinearity), the 12-feature multimodal model uses them as a structured block. Tree-based models (RF, XGBoost) handle this gracefully through implicit feature selection; linear models (LR, Elastic-Net) benefit from the L1/L2 penalty forcing coefficient shrinkage on redundant predictors.
> 4. **Interaction with genomic features**: Because TMB is orthogonal to all six signatures (Section 3.2), adding it to the model introduces genuinely new information on the genomic axis — explaining why XGBoost AUROC jumps from 0.618 (signatures only) to 0.692 when TMB is included, without requiring any adjustment for correlated input features.

## 4. Multimodal Response Prediction Models

> [!summary] What, Why & Key Questions
> - **What**: Training five classifiers on pooled trials ($N = 195$) using 5-fold stratified CV across six feature permutation tiers of the 12 final features.
> - **Why**: Evaluating whether adding TMB, driver mutations, M1/M2 ratio, Macrophage STV, or age/pathways improves upon signatures alone and identifying the best model architecture.
> - **Questions**: Does adding drivers/TMB/macrophage features improve AUROC? Which model family performs best?

![Multimodal AUROC Heatmap](../../plots/biomarkers/multimodal_auc_heatmap.png)

### Analysis of Predictor Performance
1. **Linear models degrade with features**: LR and Elastic-Net perform best with signatures alone (AUROC ≈ 0.60) and show lower performance as features increase ($N = 195$).
2. **Tree-based models benefit from feature permutations**: RF peaks at AUROC = 0.695 on the 12-Feature Final Model, while XGBoost reaches AUROC = 0.699 on the 12-Feature Final Model (and 0.692 on Sigs + TMB).
3. **Clinical interpretation**: AUROC of ~0.70–0.72 correctly ranks responder above non-responder ~71% of time, competitive with published IO response predictors.

## 5. Leave-One-Cohort-Out Model Evaluation

> [!NOTE] Section Context
> - **What**: Evaluating the same trained models using **Leave-One-Cohort-Out (LOCO)** cross-validation — training on two immunotherapy trial cohorts and testing on the third held-out cohort. This is repeated for each of the three cohorts (Liu 2019, Hugo 2016, Riaz 2017).
> - **Why**: Pooled 5-fold CV mixes patients from all cohorts in every fold, so a model can exploit cohort-specific expression patterns (even after Z-score standardisation). LOCO is a strictly harder test: the held-out cohort's patients were never seen during training, simulating deployment to a genuinely new clinical site with a different sequencing platform, patient demographics, and response distribution. If a model generalises across LOCO folds, the learned signal is robust enough to transfer to new studies.
> - **Questions**:
>   1. *Do models trained on two cohorts generalise to a third unseen cohort?*
>   2. *Which cohort is hardest to predict when held out — and why?*
>   3. *How does LOCO AUC compare to the pooled 5-fold AUC reported in Section 4?*

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


## 6. Consolidated Conclusion

> [!NOTE] Consolidated Conclusion Rationale
> - Given everything we have tested, what is the best practical approach for predicting immunotherapy response in melanoma from baseline tumour profiling?

1. **Curated signatures are the foundation**: The six immune signatures provide the most stable and interpretable transcriptomic representation. They already achieve competitive AUCs on their own (0.61–0.67) and are robust across cohorts because they compress 20,000+ genes into biologically grounded, low-dimensional scores.
2. **Orthogonal genomic features add value — but only for tree-based models**: Adding TMB, driver mutations, and pathway flags improves Random Forest and XGBoost to AUC ≈ 0.72–0.74, but *hurts* linear models. The final modelling strategy should therefore use tree-based architectures with the full multimodal feature set.
3. **LOCO is harder than pooled CV**: Cross-study generalisation (LOCO AUC ≈ 0.58–0.68) lags behind pooled 5-fold CV (AUC ≈ 0.72–0.74), especially for small held-out cohorts. These two validation strategies should be reported as **distinct evidence layers**, not interchangeable performance estimates.
4. **TMB is the right genomic surrogate**: Neoantigen load is almost perfectly collinear with TMB ($r_s = 0.872$), so retaining both adds redundancy without new information. TMB alone is sufficient.

---

> [!formula]+ Pillar 3 Script Execution & Software Module Architecture
>
> - [`train_multimodal_predictor.py`](../../scripts/pillar-3-transcriptomic-signatures/train_multimodal_predictor.py): Trains cross-validated machine learning classifiers (LR, RF, XGB, SVM, Elastic-Net) across feature set permutation tiers, generates AUROC comparison heatmaps, and updates Section 5 of `curated_signatures_report.md`.
> - [`run_extended_biomarkers.py`](../../scripts/pillar-3-transcriptomic-signatures/run_extended_biomarkers.py): Evaluates neoantigen load vs TMB, TMB-immune signature Spearman correlations, TCGA aneuploidy and TMB survival stratification, and somatic pathway mutation frequencies.
> - [`run_pipeline.py`](../../scripts/run_pipeline.py): Master pipeline orchestrator executing data preprocessing, biomarker evaluation, and multimodal predictor training in sequence.
> - [`signatures.py`](../../../src/signatures.py): Computes the six curated immune signatures (IFN-γ, TIS, CYT, IMPRES, CD8 T-cell, TCGA 20-gene OS) from normalised gene expression matrices.
> - [`models.py`](../../../src/models.py): Provides `get_model()` — the single entry point for tuned, calibrated classifier instances — and `run_loco_cv()` for LOCO cross-validation.
> - [`evaluation.py`](../../../src/evaluation.py): Implements AUROC, AUC-PR, concordance index, and Youden-optimal threshold metrics for model benchmarking.
> - [`styles.py`](../../../src/styles.py): Central definition of Okabe-Ito colour palettes.
