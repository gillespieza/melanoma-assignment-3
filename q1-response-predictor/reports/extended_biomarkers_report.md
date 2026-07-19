# Extended Biomarkers: Multimodal Predictive Modeling

This report documents the training and evaluation of response prediction models on the pooled immunotherapy trial cohort ($N=195$), comparing signature models, driver-mutation models, and a full extended clinical-genomic model.

### Model Performance (5-Fold Stratified Cross-Validation on Pooled Trial Cohort):
| Model | Base Model (Sigs only) | Sigs + Drivers (`BRAF/NRAS/NF1`) + Sex | Full Extended Model (Sigs + Drivers + TMB + CNA + Mutations) |
|---|---|---|---|
| **Logistic Regression (LR)** | 0.615 (±0.081) | 0.600 (±0.043) | **0.560 (±0.094)** |
| **Random Forest (RF)** | 0.666 (±0.053) | 0.661 (±0.072) | **0.700 (±0.102)** |
| **XGBoost (XGB, tuned)** | 0.632 (±0.061) | 0.668 (±0.095) | **0.699 (±0.123)** |
| **Support Vector Machine (SVM)** | 0.627 (±0.090) | 0.666 (±0.055) | **0.577 (±0.090)** |
| **Elastic-Net** | 0.606 (±0.070) | 0.613 (±0.053) | **0.537 (±0.067)** |

![Multimodal AUC Comparison](../plots/biomarkers/multimodal_auc_comparison.png)

### Analysis of Predictor Performance:
1.  **Baseline vs. Drivers**: Adding the driver mutations and sex provides a slight stabilization/improvement in cross-validation AUC for some model families, including tuned XGBoost.
2.  **Full Multimodal Model**: The full extended model (incorporating 15 features including mutation flags and genomic load metrics) performs strongly with tree-based learners. Tuned XGBoost remains competitive with Random Forest and favors shallow trees, moderate learning rates, and row/feature subsampling.
