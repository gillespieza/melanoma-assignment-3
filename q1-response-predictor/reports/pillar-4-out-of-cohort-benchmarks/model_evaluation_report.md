---
title:
aliases: 
tags: 
created: 2026-07-20 13:20
cssclasses: table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-20 14:04
---

# Model Evaluation Report: LOCO Cross-Cohort Validation
## Overview
This report documents the comprehensive evaluation of all trained models (Logistic Regression, Random Forest, XGBoost, SVM, ElasticNet) using Leave-One-Cohort-Out (LOCO) cross-validation on three independent melanoma immunotherapy cohorts.  
**Evaluation Framework:**
- **Cross-validation**: Leave-One-Cohort-Out (LOCO) — train on 2 cohorts, test on 1
- **Test cohorts**: Liu 2019 (N=104), Hugo 2016 (N=27), Riaz 2017 (N=64)
- **Features**: 11 immune response signatures (IFN-gamma, TIS, CD8 T-cell, CYT, IMPRES, PD-L1, etc.)
- **Decision threshold**: 0.5 (default) and Youden's J optimal (per-fold)
- **Metrics**: AUC-ROC, Accuracy, Sensitivity, Specificity, Precision, F1-score, C-index

---

## Logistic Regression (L1-penalized, GridSearchCV)

### Performance Metrics (Threshold = 0.5)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.415 | 0.481 | 0.429 | 0.538 | 0.500 | 0.462 | 0.612 |
| Liu 2019 | 104 | 0.609 | 0.625 | 0.500 | 0.732 | 0.615 | 0.552 | 0.398 |
| Riaz 2017 | 64 | 0.500 | 0.312 | 1.000 | 0.000 | 0.312 | 0.476 | 0.500 |

### Performance Metrics (Youden's J Optimal Threshold)

> Youden's J statistic ($J = \text{sensitivity} + \text{specificity} - 1$) identifies the threshold that maximises the sum of sensitivity and specificity. This is an **optimistic** estimate because the threshold is selected on the same data it is evaluated on; in production, the threshold should be fixed from a training set.

| Test Cohort | N | AUC | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.415 | 0.588 | 0.519 | 0.071 | 1.000 | 1.000 | 0.133 |
| Liu 2019 | 104 | 0.609 | 0.466 | 0.625 | 0.562 | 0.679 | 0.600 | 0.581 |
| Riaz 2017 | 64 | 0.500 | inf | 0.688 | 0.000 | 1.000 | 0.000 | 0.000 |

### Visualizations & Diagnostics

#### Confusion Matrices (Threshold = 0.5)
![Logistic Regression (L1-penalized, GridSearchCV) Confusion Matrices](../plots/models/confusion_matrices_lr.png)

```
Hugo 2016 (N=27):
                Predicted
              Non-Resp  Resp
Actual Non-Resp    7    6   
Actual Resp        8    6   

Liu 2019 (N=104):
                Predicted
              Non-Resp  Resp
Actual Non-Resp    41   15  
Actual Resp        24   24  

Riaz 2017 (N=64):
                Predicted
              Non-Resp  Resp
Actual Non-Resp    0    44  
Actual Resp        0    20  

```

#### ROC & Precision-Recall Curves
![Logistic Regression (L1-penalized, GridSearchCV) ROC Curves](../plots/models/roc_curves_lr.png)  
![Logistic Regression (L1-penalized, GridSearchCV) PR Curves](../plots/models/pr_curves_lr.png)

## Random Forest (GridSearchCV: n_estimators in [50,100,200], max_depth in [3,5,10,None], min_samples_leaf in [1,2,4])

### Performance Metrics (Threshold = 0.5)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.423 | 0.444 | 0.286 | 0.615 | 0.444 | 0.348 | 0.551 |
| Liu 2019 | 104 | 0.580 | 0.596 | 0.188 | 0.946 | 0.750 | 0.300 | 0.431 |
| Riaz 2017 | 64 | 0.678 | 0.609 | 0.800 | 0.523 | 0.432 | 0.561 | 0.455 |

### Performance Metrics (Youden's J Optimal Threshold)

> Youden's J statistic ($J = \text{sensitivity} + \text{specificity} - 1$) identifies the threshold that maximises the sum of sensitivity and specificity. This is an **optimistic** estimate because the threshold is selected on the same data it is evaluated on; in production, the threshold should be fixed from a training set.

| Test Cohort | N | AUC | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.423 | 0.319 | 0.593 | 0.714 | 0.462 | 0.588 | 0.645 |
| Liu 2019 | 104 | 0.580 | 0.426 | 0.635 | 0.417 | 0.821 | 0.667 | 0.513 |
| Riaz 2017 | 64 | 0.678 | 0.515 | 0.641 | 0.800 | 0.568 | 0.457 | 0.582 |

### Visualizations & Diagnostics

#### Confusion Matrices (Threshold = 0.5)
![Random Forest (GridSearchCV: n_estimators in [50,100,200], max_depth in [3,5,10,None], min_samples_leaf in [1,2,4]) Confusion Matrices](../plots/models/confusion_matrices_rf.png)

```
Hugo 2016 (N=27):
                Predicted
              Non-Resp  Resp
Actual Non-Resp    8    5   
Actual Resp        10   4   

Liu 2019 (N=104):
                Predicted
              Non-Resp  Resp
Actual Non-Resp    53   3   
Actual Resp        39   9   

Riaz 2017 (N=64):
                Predicted
              Non-Resp  Resp
Actual Non-Resp    23   21  
Actual Resp        4    16  

```

#### ROC & Precision-Recall Curves
![Random Forest (GridSearchCV: n_estimators in [50,100,200], max_depth in [3,5,10,None], min_samples_leaf in [1,2,4]) ROC Curves](../plots/models/roc_curves_rf.png)  
![Random Forest (GridSearchCV: n_estimators in [50,100,200], max_depth in [3,5,10,None], min_samples_leaf in [1,2,4]) PR Curves](../plots/models/pr_curves_rf.png)

## XGBoost (GridSearchCV: n_estimators in [50,100,150], max_depth in [3,5,7], learning_rate in [0.01,0.05,0.1,0.2])

### Performance Metrics (Threshold = 0.5)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.319 | 0.333 | 0.071 | 0.615 | 0.167 | 0.100 | 0.597 |
| Liu 2019 | 104 | 0.581 | 0.567 | 0.521 | 0.607 | 0.532 | 0.526 | 0.471 |
| Riaz 2017 | 64 | 0.618 | 0.531 | 0.650 | 0.477 | 0.361 | 0.464 | 0.515 |

### Performance Metrics (Youden's J Optimal Threshold)

> Youden's J statistic ($J = \text{sensitivity} + \text{specificity} - 1$) identifies the threshold that maximises the sum of sensitivity and specificity. This is an **optimistic** estimate because the threshold is selected on the same data it is evaluated on; in production, the threshold should be fixed from a training set.

| Test Cohort | N | AUC | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.319 | 0.007 | 0.519 | 0.929 | 0.077 | 0.520 | 0.667 |
| Liu 2019 | 104 | 0.581 | 0.387 | 0.596 | 0.688 | 0.518 | 0.550 | 0.611 |
| Riaz 2017 | 64 | 0.618 | 0.705 | 0.750 | 0.300 | 0.955 | 0.750 | 0.429 |

### Visualizations & Diagnostics

#### Confusion Matrices (Threshold = 0.5)
![XGBoost (GridSearchCV: n_estimators in [50,100,150], max_depth in [3,5,7], learning_rate in [0.01,0.05,0.1,0.2]) Confusion Matrices](../plots/models/confusion_matrices_xgb.png)

```
Hugo 2016 (N=27):
                Predicted
              Non-Resp  Resp
Actual Non-Resp    8    5   
Actual Resp        13   1   

Liu 2019 (N=104):
                Predicted
              Non-Resp  Resp
Actual Non-Resp    34   22  
Actual Resp        23   25  

Riaz 2017 (N=64):
                Predicted
              Non-Resp  Resp
Actual Non-Resp    21   23  
Actual Resp        7    13  

```

#### ROC & Precision-Recall Curves
![XGBoost (GridSearchCV: n_estimators in [50,100,150], max_depth in [3,5,7], learning_rate in [0.01,0.05,0.1,0.2]) ROC Curves](../plots/models/roc_curves_xgb.png)  
![XGBoost (GridSearchCV: n_estimators in [50,100,150], max_depth in [3,5,7], learning_rate in [0.01,0.05,0.1,0.2]) PR Curves](../plots/models/pr_curves_xgb.png)

## Support Vector Machine (GridSearchCV: C in [0.01,0.1,1.0,10.0], kernel in [linear,rbf])

### Performance Metrics (Threshold = 0.5)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.434 | 0.444 | 0.214 | 0.692 | 0.429 | 0.286 | 0.663 |
| Liu 2019 | 104 | 0.657 | 0.538 | 0.000 | 1.000 | 0.000 | 0.000 | 0.429 |
| Riaz 2017 | 64 | 0.717 | 0.719 | 0.500 | 0.818 | 0.556 | 0.526 | 0.428 |

### Performance Metrics (Youden's J Optimal Threshold)

> Youden's J statistic ($J = \text{sensitivity} + \text{specificity} - 1$) identifies the threshold that maximises the sum of sensitivity and specificity. This is an **optimistic** estimate because the threshold is selected on the same data it is evaluated on; in production, the threshold should be fixed from a training set.

| Test Cohort | N | AUC | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.434 | 0.201 | 0.556 | 0.929 | 0.154 | 0.542 | 0.684 |
| Liu 2019 | 104 | 0.657 | 0.370 | 0.663 | 0.583 | 0.732 | 0.651 | 0.615 |
| Riaz 2017 | 64 | 0.717 | 0.483 | 0.734 | 0.650 | 0.773 | 0.565 | 0.605 |

### Visualizations & Diagnostics

#### Confusion Matrices (Threshold = 0.5)
![Support Vector Machine (GridSearchCV: C in [0.01,0.1,1.0,10.0], kernel in [linear,rbf]) Confusion Matrices](../plots/models/confusion_matrices_svm.png)

```
Hugo 2016 (N=27):
                Predicted
              Non-Resp  Resp
Actual Non-Resp    9    4   
Actual Resp        11   3   

Liu 2019 (N=104):
                Predicted
              Non-Resp  Resp
Actual Non-Resp    56   0   
Actual Resp        48   0   

Riaz 2017 (N=64):
                Predicted
              Non-Resp  Resp
Actual Non-Resp    36   8   
Actual Resp        10   10  

```

#### ROC & Precision-Recall Curves
![Support Vector Machine (GridSearchCV: C in [0.01,0.1,1.0,10.0], kernel in [linear,rbf]) ROC Curves](../plots/models/roc_curves_svm.png)  
![Support Vector Machine (GridSearchCV: C in [0.01,0.1,1.0,10.0], kernel in [linear,rbf]) PR Curves](../plots/models/pr_curves_svm.png)

## ElasticNet Logistic Regression (GridSearchCV: C in [0.001,0.01,0.1,1.0,10.0], l1_ratio in [0.1,0.3,0.5,0.7,0.9])

### Performance Metrics (Threshold = 0.5)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.415 | 0.481 | 0.000 | 1.000 | 0.000 | 0.000 | 0.612 |
| Liu 2019 | 104 | 0.616 | 0.615 | 0.458 | 0.750 | 0.611 | 0.524 | 0.398 |
| Riaz 2017 | 64 | 0.500 | 0.312 | 1.000 | 0.000 | 0.312 | 0.476 | 0.500 |

### Performance Metrics (Youden's J Optimal Threshold)

> Youden's J statistic ($J = \text{sensitivity} + \text{specificity} - 1$) identifies the threshold that maximises the sum of sensitivity and specificity. This is an **optimistic** estimate because the threshold is selected on the same data it is evaluated on; in production, the threshold should be fixed from a training set.

| Test Cohort | N | AUC | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.415 | 0.443 | 0.519 | 0.071 | 1.000 | 1.000 | 0.133 |
| Liu 2019 | 104 | 0.616 | 0.448 | 0.625 | 0.583 | 0.661 | 0.596 | 0.589 |
| Riaz 2017 | 64 | 0.500 | inf | 0.688 | 0.000 | 1.000 | 0.000 | 0.000 |

### Visualizations & Diagnostics

#### Confusion Matrices (Threshold = 0.5)
![ElasticNet Logistic Regression (GridSearchCV: C in [0.001,0.01,0.1,1.0,10.0], l1_ratio in [0.1,0.3,0.5,0.7,0.9]) Confusion Matrices](../plots/models/confusion_matrices_elasticnet.png)

```
Hugo 2016 (N=27):
                Predicted
              Non-Resp  Resp
Actual Non-Resp    13   0   
Actual Resp        14   0   

Liu 2019 (N=104):
                Predicted
              Non-Resp  Resp
Actual Non-Resp    42   14  
Actual Resp        26   22  

Riaz 2017 (N=64):
                Predicted
              Non-Resp  Resp
Actual Non-Resp    0    44  
Actual Resp        0    20  

```

#### ROC & Precision-Recall Curves
![[roc_curves_elasticnet.png]]

ElasticNet Logistic Regression (GridSearchCV: C in [0.001,0.01,0.1,1.0,10.0], l1_ratio in [0.1,0.3,0.5,0.7,0.9]) ROC Curves

![[pr_curves_elasticnet.png]]

ElasticNet Logistic Regression (GridSearchCV: C in [0.001,0.01,0.1,1.0,10.0], l1_ratio in [0.1,0.3,0.5,0.7,0.9]) PR Curves

## Summary & Interpretation

### Key Metrics Explained
- **Sensitivity (Recall)**: TP / (TP + FN) — Proportion of actual responders correctly identified
- **Specificity**: TN / (TN + FP) — Proportion of actual non-responders correctly identified
- **Precision**: TP / (TP + FP) — Proportion of predicted responders who are actually responders
- **Accuracy**: (TP + TN) / Total — Overall correctness across both classes
- **F1-Score**: Harmonic mean of Precision and Recall — Balances both metrics
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve — Robustness to threshold selection
- **C-Index (Concordance Index)**: Evaluates how well predicted response probabilities rank patients by survival. 0.5 = random, 1.0 = perfect. Accounts for censoring in survival data.

### Visualizations
- **ROC Curves** (`roc_curves_*.png`): Trade-off between True Positive Rate and False Positive Rate
- **PR Curves** (`pr_curves_*.png`): Precision-Recall trade-off, especially relevant for class imbalance
- **Confusion Matrices** (`confusion_matrices_*.png`): Cell-level breakdown of predictions per cohort

