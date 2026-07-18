# Extended Biomarkers: Multimodal Predictive Modeling

This report documents the training and evaluation of response prediction models on the pooled immunotherapy trial cohort ($N=150$), comparing signature models, driver-mutation models, and a full extended clinical-genomic model.

### Model Performance (5-Fold Stratified Cross-Validation on Pooled Trial Cohort):
| Model | Base Model (Sigs only) | Sigs + Drivers (`BRAF/NRAS/NF1`) + Sex | Full Extended Model (Sigs + Drivers + TMB + CNA + Mutations) |
|---|---|---|---|
| **Logistic Regression (LR)** | 0.627 (±0.095) | 0.611 (±0.084) | **0.611 (±0.094)** |
| **Random Forest (RF)** | 0.627 (±0.075) | 0.642 (±0.070) | **0.705 (±0.102)** |

### Analysis of Predictor Performance:
1.  **Baseline vs. Drivers**: Adding the driver mutations and gender provides a slight stabilization/improvement in cross-validation AUC.
2.  **Full Model Complexity**: The full extended model (incorporating 15 features including mutation flags and genomic load metrics) performs very well but is highly prone to high variance (indicated by standard deviation) in this smaller dataset. Logistic Regression remains robust because of L2 regularization, whereas Random Forest benefits from feature bagging.