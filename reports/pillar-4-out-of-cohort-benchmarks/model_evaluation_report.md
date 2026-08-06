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
created: 2026-08-06 18:29
cssclasses:
  - table-small
  - table-center
  - row-alt
updated: 2026-08-06 18:29
---

# Model Evaluation Report: Leave-One-Cohort-Out (LOCO) Cross-Validation

> [!summary] What, Why & Key Questions
> **What**: Tested 5 machine learning models (LR, RF, XGBoost, SVM, ElasticNet) predicting anti-PD-1 response.
> **Why**: Testing on held-out hospital trial cohorts (LOCO) evaluates real-world generalisability.
> **Key Questions**: Which model generalises best? Does multimodal integration improve accuracy?

## Overview & Methodology

1. **Evaluation Framework (LOCO)**: Trained on 2 trial cohorts, tested on 1 unseen cohort.
2. **Test Cohorts**: Liu 2019, Hugo 2016, Riaz 2017 with cohort-independent Z-scoring.
3. **Features**: 8 transcriptomic signatures plus somatic driver mutations (`mut_BRAF`, `mut_NRAS`, `mut_NF1`).

## Cross-Model AUC Summary

| Model | Hugo 2016 | Liu 2019 | Riaz 2017 | **Mean AUC** |
|:---|---::---::---:---:|
| **LR** | 0.505 | 0.569 | 0.498 | **0.524** |
| **RF** | 0.407 | 0.569 | 0.618 | **0.531** |
| **XGB** | 0.269 | 0.595 | 0.606 | **0.490** |
| **SVM** | 0.516 | 0.554 | 0.458 | **0.509** |
| **ElasticNet** | 0.415 | 0.558 | 0.500 | **0.491** |

> [!INSIGHT] Best Generalising Model: RF
> **Random Forest Classifier** achieves the highest mean cross-cohort AUC of **0.531** across held-out test cohorts.

## Logistic Regression (L1-Penalised)

### Performance Metrics (Default Threshold = 0.5)

| Test Cohort | N | AUC | ECE | Brier | Accuracy | Sens | Spec | Prec | F1 | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.505 | 0.027 | 0.250 | 0.481 | 0.357 | 0.615 | 0.500 | 0.417 | 0.551 |
| Liu 2019 | 104 | 0.569 | 0.232 | 0.307 | 0.577 | 0.375 | 0.750 | 0.562 | 0.450 | 0.412 |
| Riaz 2017 | 64 | 0.498 | 0.191 | 0.252 | 0.500 | 0.350 | 0.568 | 0.269 | 0.304 | 0.493 |

#### Visualisations & Diagnostics

![Confusion Matrices](../../plots/models/confusion_matrices_lr.png)

![Calibration Curves](../../plots/models/calibration_curves_lr.png)

![ROC Curves](../../plots/models/roc_curves_lr.png)

![PR Curves](../../plots/models/pr_curves_lr.png)

## Random Forest Classifier

### Performance Metrics (Default Threshold = 0.5)

| Test Cohort | N | AUC | ECE | Brier | Accuracy | Sens | Spec | Prec | F1 | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.407 | 0.253 | 0.314 | 0.407 | 0.143 | 0.692 | 0.333 | 0.200 | 0.531 |
| Liu 2019 | 104 | 0.569 | 0.102 | 0.256 | 0.548 | 0.062 | 0.964 | 0.600 | 0.113 | 0.451 |
| Riaz 2017 | 64 | 0.618 | 0.194 | 0.248 | 0.547 | 0.650 | 0.500 | 0.371 | 0.473 | 0.481 |

#### Visualisations & Diagnostics

![Confusion Matrices](../../plots/models/confusion_matrices_rf.png)

![Calibration Curves](../../plots/models/calibration_curves_rf.png)

![ROC Curves](../../plots/models/roc_curves_rf.png)

![PR Curves](../../plots/models/pr_curves_rf.png)

## XGBoost Gradient Boosting

### Performance Metrics (Default Threshold = 0.5)

| Test Cohort | N | AUC | ECE | Brier | Accuracy | Sens | Spec | Prec | F1 | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.269 | 0.332 | 0.321 | 0.333 | 0.071 | 0.615 | 0.167 | 0.100 | 0.607 |
| Liu 2019 | 104 | 0.595 | 0.098 | 0.256 | 0.538 | 0.000 | 1.000 | 0.000 | 0.000 | 0.457 |
| Riaz 2017 | 64 | 0.606 | 0.202 | 0.250 | 0.531 | 0.700 | 0.455 | 0.368 | 0.483 | 0.524 |

#### Visualisations & Diagnostics

![Confusion Matrices](../../plots/models/confusion_matrices_xgb.png)

![Calibration Curves](../../plots/models/calibration_curves_xgb.png)

![ROC Curves](../../plots/models/roc_curves_xgb.png)

![PR Curves](../../plots/models/pr_curves_xgb.png)

## Support Vector Machine (SVM)

### Performance Metrics (Default Threshold = 0.5)

| Test Cohort | N | AUC | ECE | Brier | Accuracy | Sens | Spec | Prec | F1 | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.516 | 0.175 | 0.291 | 0.481 | 0.143 | 0.846 | 0.500 | 0.222 | 0.582 |
| Liu 2019 | 104 | 0.554 | 0.088 | 0.256 | 0.538 | 0.000 | 1.000 | 0.000 | 0.000 | 0.421 |
| Riaz 2017 | 64 | 0.458 | 0.200 | 0.246 | 0.578 | 0.050 | 0.818 | 0.111 | 0.069 | 0.522 |

#### Visualisations & Diagnostics

![Confusion Matrices](../../plots/models/confusion_matrices_svm.png)

![Calibration Curves](../../plots/models/calibration_curves_svm.png)

![ROC Curves](../../plots/models/roc_curves_svm.png)

![PR Curves](../../plots/models/pr_curves_svm.png)

## ElasticNet Logistic Regression

### Performance Metrics (Default Threshold = 0.5)

| Test Cohort | N | AUC | ECE | Brier | Accuracy | Sens | Spec | Prec | F1 | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.415 | 0.118 | 0.265 | 0.481 | 0.000 | 1.000 | 0.000 | 0.000 | 0.612 |
| Liu 2019 | 104 | 0.558 | 0.257 | 0.309 | 0.577 | 0.333 | 0.786 | 0.571 | 0.421 | 0.416 |
| Riaz 2017 | 64 | 0.500 | 0.195 | 0.253 | 0.312 | 1.000 | 0.000 | 0.312 | 0.476 | 0.500 |

#### Visualisations & Diagnostics

![Confusion Matrices](../../plots/models/confusion_matrices_elasticnet.png)

![Calibration Curves](../../plots/models/calibration_curves_elasticnet.png)

![ROC Curves](../../plots/models/roc_curves_elasticnet.png)

![PR Curves](../../plots/models/pr_curves_elasticnet.png)

## Multimodal Integration: Immune Signatures + Driver Mutations

### Logistic Regression (L1-Penalised) (Multimodal)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.505 | 0.481 | 0.357 | 0.615 | 0.500 | 0.417 |
| Liu 2019 | 104 | 0.569 | 0.577 | 0.375 | 0.750 | 0.562 | 0.450 |
| Riaz 2017 | 64 | 0.498 | 0.500 | 0.350 | 0.568 | 0.269 | 0.304 |

![Multimodal ROC](../../plots/models/roc_curves_combined_lr.png)

### Random Forest Classifier (Multimodal)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.407 | 0.407 | 0.143 | 0.692 | 0.333 | 0.200 |
| Liu 2019 | 104 | 0.569 | 0.548 | 0.062 | 0.964 | 0.600 | 0.113 |
| Riaz 2017 | 64 | 0.618 | 0.547 | 0.650 | 0.500 | 0.371 | 0.473 |

![Multimodal ROC](../../plots/models/roc_curves_combined_rf.png)

### XGBoost Gradient Boosting (Multimodal)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.269 | 0.333 | 0.071 | 0.615 | 0.167 | 0.100 |
| Liu 2019 | 104 | 0.595 | 0.538 | 0.000 | 1.000 | 0.000 | 0.000 |
| Riaz 2017 | 64 | 0.606 | 0.531 | 0.700 | 0.455 | 0.368 | 0.483 |

![Multimodal ROC](../../plots/models/roc_curves_combined_xgb.png)

### Support Vector Machine (SVM) (Multimodal)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.516 | 0.481 | 0.143 | 0.846 | 0.500 | 0.222 |
| Liu 2019 | 104 | 0.554 | 0.538 | 0.000 | 1.000 | 0.000 | 0.000 |
| Riaz 2017 | 64 | 0.458 | 0.578 | 0.050 | 0.818 | 0.111 | 0.069 |

![Multimodal ROC](../../plots/models/roc_curves_combined_svm.png)

### ElasticNet Logistic Regression (Multimodal)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.415 | 0.481 | 0.000 | 1.000 | 0.000 | 0.000 |
| Liu 2019 | 104 | 0.558 | 0.577 | 0.333 | 0.786 | 0.571 | 0.421 |
| Riaz 2017 | 64 | 0.500 | 0.312 | 1.000 | 0.000 | 0.312 | 0.476 |

![Multimodal ROC](../../plots/models/roc_curves_combined_elasticnet.png)

## Downstream Overall Survival Stratification

| Cohort | Model Selected | Best LOCO AUC | Log-Rank p-value | Significant? |
|:---|:---:|:---:|:---:|:---:|
| Hugo 2016 | SVM | 0.516 | 6.838e-01 | No |
| Liu 2019 | XGB | 0.595 | 7.720e-01 | No |
| Riaz 2017 | RF | 0.618 | 7.154e-01 | No |
| TCGA-SKCM | LR | N/A (external) | 6.987e-04 | Yes |

### Kaplan-Meier Survival Curves

![Kaplan-Meier 2x2 Grid](../../plots/models/survival_2x2_grid.png)

