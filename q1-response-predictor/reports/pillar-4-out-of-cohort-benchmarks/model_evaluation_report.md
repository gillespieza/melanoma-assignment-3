---
title: "Model Evaluation Report: Leave-One-Cohort-Out (LOCO) Cross-Validation"
aliases:
  - LOCO Model Evaluation Report
  - Q1 Response Predictor Evaluation
tags:
  - q1
  - model-evaluation
  - loco-cv
  - immunotherapy-response
  - calibration
created: 2026-08-09 17:06
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-09 17:06
---

# Model Evaluation Report: Leave-One-Cohort-Out (LOCO) Cross-Validation

> [!summary] What, Why & Key Questions
> - **What**: Tested 5 machine learning models (LR, RF, XGBoost, SVM, ElasticNet) predicting anti-PD-1 / immunotherapy response.
> - **Why**: Testing on held-out hospital trial cohorts (LOCO) evaluates real-world generalisability.
> **Key Questions**: Which model generalises best? Does multimodal integration improve accuracy?

## Overview & Methodology

1. **Evaluation Framework (LOCO)**: Trained on N-1 trial cohorts, tested on 1 unseen cohort.
2. **Test Cohorts**: Active trial cohorts loaded dynamically from `config/datasets.yaml` with cohort-independent Z-scoring.
3. **Features**: Transcriptomic immune signatures plus somatic driver mutations (`mut_BRAF`, `mut_NRAS`, `mut_NF1`).

## Cross-Model AUC Summary

| Model | Gide 2019 | Hugo 2016 | Liu 2019 | Riaz 2017 | TCGA GDC 2025 | Van Allen 2015 | **Mean AUC** |
|:---|---::---::---::---::---::---:---:|
| **LR** | 0.522 | 0.511 | 0.512 | 0.743 | 0.422 | 0.599 | **0.551** |
| **RF** | 0.374 | 0.489 | 0.555 | 0.657 | 0.414 | 0.527 | **0.503** |
| **XGB** | 0.498 | 0.412 | 0.570 | 0.591 | 0.482 | 0.489 | **0.507** |
| **SVM** | 0.626 | 0.308 | 0.571 | 0.343 | 0.420 | 0.566 | **0.472** |
| **ElasticNet** | 0.728 | 0.451 | 0.539 | 0.740 | 0.401 | 0.643 | **0.584** |

> [!INSIGHT] Best Generalising Model: ElasticNet
> **ElasticNet Logistic Regression** achieves the highest mean cross-cohort AUC of **0.584** across held-out test cohorts.

## Logistic Regression (L1-Penalised)

### Performance Metrics (Default Threshold = 0.5)

| Test Cohort | N | AUC | ECE | Brier | Accuracy | Sens | Spec | Prec | F1 | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gide 2019 | 78 | 0.522 | 0.235 | 0.294 | 0.410 | 0.224 | 0.724 | 0.579 | 0.324 | 0.481 |
| Hugo 2016 | 27 | 0.511 | 0.076 | 0.251 | 0.444 | 0.429 | 0.462 | 0.462 | 0.444 | 0.536 |
| Liu 2019 | 104 | 0.512 | 0.201 | 0.316 | 0.558 | 0.271 | 0.804 | 0.542 | 0.361 | 0.462 |
| Riaz 2017 | 64 | 0.743 | 0.187 | 0.244 | 0.625 | 0.750 | 0.568 | 0.441 | 0.556 | 0.311 |
| TCGA GDC 2025 | 52 | 0.422 | 0.081 | 0.255 | 0.442 | 0.385 | 0.500 | 0.435 | 0.408 | 0.441 |
| Van Allen 2015 | 33 | 0.599 | 0.287 | 0.246 | 0.576 | 0.714 | 0.538 | 0.294 | 0.417 | 0.361 |

#### Visualisations & Diagnostics

![Confusion Matrices](../../plots/models/confusion_matrices_lr.png)

![Calibration Curves](../../plots/models/calibration_curves_lr.png)

![ROC Curves](../../plots/models/roc_curves_lr.png)

![PR Curves](../../plots/models/pr_curves_lr.png)

## Random Forest Classifier

### Performance Metrics (Default Threshold = 0.5)

| Test Cohort | N | AUC | ECE | Brier | Accuracy | Sens | Spec | Prec | F1 | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gide 2019 | 78 | 0.374 | 0.204 | 0.280 | 0.372 | 0.000 | 1.000 | 0.000 | 0.000 | 0.558 |
| Hugo 2016 | 27 | 0.489 | 0.067 | 0.254 | 0.481 | 0.000 | 1.000 | 0.000 | 0.000 | 0.520 |
| Liu 2019 | 104 | 0.555 | 0.070 | 0.248 | 0.596 | 0.229 | 0.911 | 0.688 | 0.344 | 0.429 |
| Riaz 2017 | 64 | 0.657 | 0.205 | 0.244 | 0.719 | 0.500 | 0.818 | 0.556 | 0.526 | 0.411 |
| TCGA GDC 2025 | 52 | 0.414 | 0.124 | 0.271 | 0.462 | 0.115 | 0.808 | 0.375 | 0.176 | 0.483 |
| Van Allen 2015 | 33 | 0.527 | 0.270 | 0.240 | 0.788 | 0.000 | 1.000 | 0.000 | 0.000 | 0.452 |

#### Visualisations & Diagnostics

![Confusion Matrices](../../plots/models/confusion_matrices_rf.png)

![Calibration Curves](../../plots/models/calibration_curves_rf.png)

![ROC Curves](../../plots/models/roc_curves_rf.png)

![PR Curves](../../plots/models/pr_curves_rf.png)

## XGBoost Gradient Boosting

### Performance Metrics (Default Threshold = 0.5)

| Test Cohort | N | AUC | ECE | Brier | Accuracy | Sens | Spec | Prec | F1 | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gide 2019 | 78 | 0.498 | 0.215 | 0.280 | 0.372 | 0.000 | 1.000 | 0.000 | 0.000 | 0.497 |
| Hugo 2016 | 27 | 0.412 | 0.151 | 0.263 | 0.407 | 0.071 | 0.769 | 0.250 | 0.111 | 0.592 |
| Liu 2019 | 104 | 0.570 | 0.090 | 0.252 | 0.538 | 0.250 | 0.786 | 0.500 | 0.333 | 0.443 |
| Riaz 2017 | 64 | 0.591 | 0.175 | 0.243 | 0.641 | 0.450 | 0.727 | 0.429 | 0.439 | 0.404 |
| TCGA GDC 2025 | 52 | 0.482 | 0.061 | 0.256 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 0.515 |
| Van Allen 2015 | 33 | 0.489 | 0.271 | 0.241 | 0.788 | 0.000 | 1.000 | 0.000 | 0.000 | 0.523 |

#### Visualisations & Diagnostics

![Confusion Matrices](../../plots/models/confusion_matrices_xgb.png)

![Calibration Curves](../../plots/models/calibration_curves_xgb.png)

![ROC Curves](../../plots/models/roc_curves_xgb.png)

![PR Curves](../../plots/models/pr_curves_xgb.png)

## Support Vector Machine (SVM)

### Performance Metrics (Default Threshold = 0.5)

| Test Cohort | N | AUC | ECE | Brier | Accuracy | Sens | Spec | Prec | F1 | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gide 2019 | 78 | 0.626 | 0.372 | 0.372 | 0.628 | 1.000 | 0.000 | 0.628 | 0.772 | 0.446 |
| Hugo 2016 | 27 | 0.308 | 0.198 | 0.283 | 0.370 | 0.143 | 0.615 | 0.286 | 0.190 | 0.668 |
| Liu 2019 | 104 | 0.571 | 0.052 | 0.246 | 0.558 | 0.104 | 0.946 | 0.625 | 0.179 | 0.453 |
| Riaz 2017 | 64 | 0.343 | 0.177 | 0.249 | 0.562 | 0.000 | 0.818 | 0.000 | 0.000 | 0.647 |
| TCGA GDC 2025 | 52 | 0.420 | 0.211 | 0.287 | 0.462 | 0.385 | 0.538 | 0.455 | 0.417 | 0.446 |
| Van Allen 2015 | 33 | 0.566 | 0.286 | 0.246 | 0.606 | 0.714 | 0.577 | 0.312 | 0.435 | 0.373 |

#### Visualisations & Diagnostics

![Confusion Matrices](../../plots/models/confusion_matrices_svm.png)

![Calibration Curves](../../plots/models/calibration_curves_svm.png)

![ROC Curves](../../plots/models/roc_curves_svm.png)

![PR Curves](../../plots/models/pr_curves_svm.png)

## ElasticNet Logistic Regression

### Performance Metrics (Default Threshold = 0.5)

| Test Cohort | N | AUC | ECE | Brier | Accuracy | Sens | Spec | Prec | F1 | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gide 2019 | 78 | 0.728 | 0.218 | 0.269 | 0.372 | 0.000 | 1.000 | 0.000 | 0.000 | 0.366 |
| Hugo 2016 | 27 | 0.451 | 0.108 | 0.260 | 0.444 | 0.071 | 0.846 | 0.333 | 0.118 | 0.602 |
| Liu 2019 | 104 | 0.539 | 0.129 | 0.277 | 0.548 | 0.292 | 0.768 | 0.519 | 0.373 | 0.457 |
| Riaz 2017 | 64 | 0.740 | 0.182 | 0.238 | 0.703 | 0.650 | 0.727 | 0.520 | 0.578 | 0.327 |
| TCGA GDC 2025 | 52 | 0.401 | 0.103 | 0.266 | 0.462 | 0.192 | 0.731 | 0.417 | 0.263 | 0.455 |
| Van Allen 2015 | 33 | 0.643 | 0.271 | 0.234 | 0.515 | 0.429 | 0.538 | 0.200 | 0.273 | 0.334 |

#### Visualisations & Diagnostics

![Confusion Matrices](../../plots/models/confusion_matrices_elasticnet.png)

![Calibration Curves](../../plots/models/calibration_curves_elasticnet.png)

![ROC Curves](../../plots/models/roc_curves_elasticnet.png)

![PR Curves](../../plots/models/pr_curves_elasticnet.png)

## Multimodal Integration: Immune Signatures + Driver Mutations

### Logistic Regression (L1-Penalised) (Multimodal)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gide 2019 | 78 | 0.522 | 0.410 | 0.224 | 0.724 | 0.579 | 0.324 |
| Hugo 2016 | 27 | 0.511 | 0.444 | 0.429 | 0.462 | 0.462 | 0.444 |
| Liu 2019 | 104 | 0.512 | 0.558 | 0.271 | 0.804 | 0.542 | 0.361 |
| Riaz 2017 | 64 | 0.743 | 0.625 | 0.750 | 0.568 | 0.441 | 0.556 |
| TCGA GDC 2025 | 52 | 0.422 | 0.442 | 0.385 | 0.500 | 0.435 | 0.408 |
| Van Allen 2015 | 33 | 0.599 | 0.576 | 0.714 | 0.538 | 0.294 | 0.417 |

![Multimodal ROC](../../plots/models/roc_curves_combined_lr.png)

### Random Forest Classifier (Multimodal)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gide 2019 | 78 | 0.374 | 0.372 | 0.000 | 1.000 | 0.000 | 0.000 |
| Hugo 2016 | 27 | 0.489 | 0.481 | 0.000 | 1.000 | 0.000 | 0.000 |
| Liu 2019 | 104 | 0.555 | 0.596 | 0.229 | 0.911 | 0.688 | 0.344 |
| Riaz 2017 | 64 | 0.657 | 0.719 | 0.500 | 0.818 | 0.556 | 0.526 |
| TCGA GDC 2025 | 52 | 0.414 | 0.462 | 0.115 | 0.808 | 0.375 | 0.176 |
| Van Allen 2015 | 33 | 0.527 | 0.788 | 0.000 | 1.000 | 0.000 | 0.000 |

![Multimodal ROC](../../plots/models/roc_curves_combined_rf.png)

### XGBoost Gradient Boosting (Multimodal)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gide 2019 | 78 | 0.498 | 0.372 | 0.000 | 1.000 | 0.000 | 0.000 |
| Hugo 2016 | 27 | 0.412 | 0.407 | 0.071 | 0.769 | 0.250 | 0.111 |
| Liu 2019 | 104 | 0.570 | 0.538 | 0.250 | 0.786 | 0.500 | 0.333 |
| Riaz 2017 | 64 | 0.591 | 0.641 | 0.450 | 0.727 | 0.429 | 0.439 |
| TCGA GDC 2025 | 52 | 0.482 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 |
| Van Allen 2015 | 33 | 0.489 | 0.788 | 0.000 | 1.000 | 0.000 | 0.000 |

![Multimodal ROC](../../plots/models/roc_curves_combined_xgb.png)

### Support Vector Machine (SVM) (Multimodal)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gide 2019 | 78 | 0.626 | 0.628 | 1.000 | 0.000 | 0.628 | 0.772 |
| Hugo 2016 | 27 | 0.308 | 0.370 | 0.143 | 0.615 | 0.286 | 0.190 |
| Liu 2019 | 104 | 0.571 | 0.558 | 0.104 | 0.946 | 0.625 | 0.179 |
| Riaz 2017 | 64 | 0.343 | 0.562 | 0.000 | 0.818 | 0.000 | 0.000 |
| TCGA GDC 2025 | 52 | 0.420 | 0.462 | 0.385 | 0.538 | 0.455 | 0.417 |
| Van Allen 2015 | 33 | 0.566 | 0.606 | 0.714 | 0.577 | 0.312 | 0.435 |

![Multimodal ROC](../../plots/models/roc_curves_combined_svm.png)

### ElasticNet Logistic Regression (Multimodal)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gide 2019 | 78 | 0.728 | 0.372 | 0.000 | 1.000 | 0.000 | 0.000 |
| Hugo 2016 | 27 | 0.451 | 0.444 | 0.071 | 0.846 | 0.333 | 0.118 |
| Liu 2019 | 104 | 0.539 | 0.548 | 0.292 | 0.768 | 0.519 | 0.373 |
| Riaz 2017 | 64 | 0.740 | 0.703 | 0.650 | 0.727 | 0.520 | 0.578 |
| TCGA GDC 2025 | 52 | 0.401 | 0.462 | 0.192 | 0.731 | 0.417 | 0.263 |
| Van Allen 2015 | 33 | 0.643 | 0.515 | 0.429 | 0.538 | 0.200 | 0.273 |

![Multimodal ROC](../../plots/models/roc_curves_combined_elasticnet.png)

## Downstream Overall Survival Stratification

| Cohort | Model Selected | Best LOCO AUC | Log-Rank p-value | Significant? |
|:---|:---:|:---:|:---:|:---:|
| Liu 2019 | SVM | 0.571 | 3.233e-01 | No |
| Hugo 2016 | LR | 0.511 | 4.755e-01 | No |
| Riaz 2017 | LR | 0.743 | 6.863e-03 | Yes |
| TCGA GDC 2025 | XGB | 0.482 | 6.914e-01 | No |
| Gide 2019 | ELASTICNET | 0.728 | 3.668e-02 | Yes |
| Van Allen 2015 | ELASTICNET | 0.643 | 2.267e-01 | No |
| TCGA GDC 2025 (IT Subcohort) | LR | N/A (external) | 1.585e-01 | No |

### Kaplan-Meier Survival Curves

![Kaplan-Meier 2x2 Grid](../../plots/models/survival_2x2_grid.png)

