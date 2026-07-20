# Extended Biomarkers: Multimodal Predictive Modeling

This report documents the training and evaluation of response prediction models on the pooled immunotherapy trial cohort ($N=195$), comparing signature models, driver-mutation models, and a full extended clinical-genomic model.

### Model Performance (5-Fold Stratified Cross-Validation on Pooled Trial Cohort):
| Model | Base Model (Sigs only) | Sigs + Drivers (`BRAF/NRAS/NF1`) + Age | Full Extended Model (Sigs + Drivers + TMB + Neoantigens + Mutations) |
|---|---|---|---|
| **Logistic Regression (LR)** | **0.615 (+/-0.081)** | 0.553 (+/-0.046) | 0.587 (+/-0.069) |
| **Random Forest (RF)** | 0.666 (+/-0.053) | 0.662 (+/-0.067) | **0.710 (+/-0.094)** |
| **XGBoost (XGB, tuned)** | 0.632 (+/-0.058) | 0.681 (+/-0.051) | **0.724 (+/-0.089)** |
| **Support Vector Machine (SVM)** | **0.626 (+/-0.081)** | 0.617 (+/-0.089) | 0.625 (+/-0.073) |
| **Elastic-Net** | **0.610 (+/-0.075)** | 0.585 (+/-0.059) | 0.534 (+/-0.030) |

![Multimodal AUC Comparison](../plots/biomarkers/multimodal_auc_comparison.png)

### Analysis of Predictor Performance:
1.  **Baseline vs. Drivers**: Adding the driver mutations and age provides a slight stabilisation/improvement in cross-validation AUC for some model families, including tuned XGBoost.
2.  **Full Multimodal Model**: The full extended model (incorporating 15 features including mutation flags and genomic load metrics) performs strongly with the tuned XGBoost grid, which favours shallow trees, moderate learning rates, and row/feature subsampling, and remains competitive with Random Forest.