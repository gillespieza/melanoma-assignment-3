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
![[confusion_matrices_lr.png]]

_Figure: Confusion matrices for Logistic Regression (L1-penalized, GridSearchCV) at the default 0.5 decision threshold, per LOCO test cohort._

#### ROC & Precision-Recall Curves
![[roc_curves_lr.png]]

_Figure: ROC curves for Logistic Regression (L1-penalized, GridSearchCV) across LOCO test cohorts. Diagonal dashed line indicates chance-level performance (AUC = 0.5)._

![[pr_curves_lr.png]]

_Figure: Precision-Recall curves for Logistic Regression (L1-penalized, GridSearchCV). Particularly informative under class imbalance._

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
![[confusion_matrices_rf.png]]

_Figure: Confusion matrices for Random Forest (GridSearchCV: n_estimators in [50,100,200], max_depth in [3,5,10,None], min_samples_leaf in [1,2,4]) at the default 0.5 decision threshold, per LOCO test cohort._

#### ROC & Precision-Recall Curves
![[roc_curves_rf.png]]

_Figure: ROC curves for Random Forest (GridSearchCV: n_estimators in [50,100,200], max_depth in [3,5,10,None], min_samples_leaf in [1,2,4]) across LOCO test cohorts. Diagonal dashed line indicates chance-level performance (AUC = 0.5)._

![[pr_curves_rf.png]]

_Figure: Precision-Recall curves for Random Forest (GridSearchCV: n_estimators in [50,100,200], max_depth in [3,5,10,None], min_samples_leaf in [1,2,4]). Particularly informative under class imbalance._

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
![[confusion_matrices_xgb.png]]

_Figure: Confusion matrices for XGBoost (GridSearchCV: n_estimators in [50,100,150], max_depth in [3,5,7], learning_rate in [0.01,0.05,0.1,0.2]) at the default 0.5 decision threshold, per LOCO test cohort._

#### ROC & Precision-Recall Curves
![[roc_curves_xgb.png]]

_Figure: ROC curves for XGBoost (GridSearchCV: n_estimators in [50,100,150], max_depth in [3,5,7], learning_rate in [0.01,0.05,0.1,0.2]) across LOCO test cohorts. Diagonal dashed line indicates chance-level performance (AUC = 0.5)._

![[pr_curves_xgb.png]]

_Figure: Precision-Recall curves for XGBoost (GridSearchCV: n_estimators in [50,100,150], max_depth in [3,5,7], learning_rate in [0.01,0.05,0.1,0.2]). Particularly informative under class imbalance._

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
![[confusion_matrices_svm.png]]

_Figure: Confusion matrices for Support Vector Machine (GridSearchCV: C in [0.01,0.1,1.0,10.0], kernel in [linear,rbf]) at the default 0.5 decision threshold, per LOCO test cohort._

#### ROC & Precision-Recall Curves
![[roc_curves_svm.png]]

_Figure: ROC curves for Support Vector Machine (GridSearchCV: C in [0.01,0.1,1.0,10.0], kernel in [linear,rbf]) across LOCO test cohorts. Diagonal dashed line indicates chance-level performance (AUC = 0.5)._

![[pr_curves_svm.png]]

_Figure: Precision-Recall curves for Support Vector Machine (GridSearchCV: C in [0.01,0.1,1.0,10.0], kernel in [linear,rbf]). Particularly informative under class imbalance._

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
![[confusion_matrices_elasticnet.png]]

_Figure: Confusion matrices for ElasticNet Logistic Regression (GridSearchCV: C in [0.001,0.01,0.1,1.0,10.0], l1_ratio in [0.1,0.3,0.5,0.7,0.9]) at the default 0.5 decision threshold, per LOCO test cohort._

#### ROC & Precision-Recall Curves
![[roc_curves_elasticnet.png]]

_Figure: ROC curves for ElasticNet Logistic Regression (GridSearchCV: C in [0.001,0.01,0.1,1.0,10.0], l1_ratio in [0.1,0.3,0.5,0.7,0.9]) across LOCO test cohorts. Diagonal dashed line indicates chance-level performance (AUC = 0.5)._

![[pr_curves_elasticnet.png]]

_Figure: Precision-Recall curves for ElasticNet Logistic Regression (GridSearchCV: C in [0.001,0.01,0.1,1.0,10.0], l1_ratio in [0.1,0.3,0.5,0.7,0.9]). Particularly informative under class imbalance._

---

## Combined Features: Immune Signatures + Driver Mutations

This section benchmarks models trained on the 11 immune signatures plus 3 binary driver-mutation features (BRAF, NRAS, NF1) using the same 3-cohort LOCO framework. Adding genomic features tests whether mutation status provides complementary predictive signal beyond transcriptomic signatures alone.

### Logistic Regression (L1-penalized, GridSearchCV)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.415 | 0.481 | 0.429 | 0.538 | 0.500 | 0.462 |
| Liu 2019 | 104 | 0.551 | 0.538 | 0.417 | 0.643 | 0.500 | 0.455 |
| Riaz 2017 | 64 | 0.500 | 0.312 | 1.000 | 0.000 | 0.312 | 0.476 |

![[roc_curves_combined_lr.png]]

_Figure: ROC curves for Logistic Regression (L1-penalized, GridSearchCV) with combined immune signature + driver mutation features._

### Random Forest (GridSearchCV: n_estimators in [50,100,200], max_depth in [3,5,10,None], min_samples_leaf in [1,2,4])

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.423 | 0.407 | 0.214 | 0.615 | 0.375 | 0.273 |
| Liu 2019 | 104 | 0.574 | 0.596 | 0.271 | 0.875 | 0.650 | 0.382 |
| Riaz 2017 | 64 | 0.702 | 0.672 | 0.700 | 0.659 | 0.483 | 0.571 |

![[roc_curves_combined_rf.png]]

_Figure: ROC curves for Random Forest (GridSearchCV: n_estimators in [50,100,200], max_depth in [3,5,10,None], min_samples_leaf in [1,2,4]) with combined immune signature + driver mutation features._

### XGBoost (GridSearchCV: n_estimators in [50,100,150], max_depth in [3,5,7], learning_rate in [0.01,0.05,0.1,0.2])

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.374 | 0.370 | 0.143 | 0.615 | 0.286 | 0.190 |
| Liu 2019 | 104 | 0.565 | 0.548 | 0.354 | 0.714 | 0.515 | 0.420 |
| Riaz 2017 | 64 | 0.645 | 0.547 | 0.650 | 0.500 | 0.371 | 0.473 |

![[roc_curves_combined_xgb.png]]

_Figure: ROC curves for XGBoost (GridSearchCV: n_estimators in [50,100,150], max_depth in [3,5,7], learning_rate in [0.01,0.05,0.1,0.2]) with combined immune signature + driver mutation features._

### Support Vector Machine (GridSearchCV: C in [0.01,0.1,1.0,10.0], kernel in [linear,rbf])

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.478 | 0.444 | 0.214 | 0.692 | 0.429 | 0.286 |
| Liu 2019 | 104 | 0.537 | 0.529 | 0.188 | 0.821 | 0.474 | 0.269 |
| Riaz 2017 | 64 | 0.511 | 0.516 | 0.500 | 0.523 | 0.323 | 0.392 |

![[roc_curves_combined_svm.png]]

_Figure: ROC curves for Support Vector Machine (GridSearchCV: C in [0.01,0.1,1.0,10.0], kernel in [linear,rbf]) with combined immune signature + driver mutation features._

### ElasticNet Logistic Regression (GridSearchCV: C in [0.001,0.01,0.1,1.0,10.0], l1_ratio in [0.1,0.3,0.5,0.7,0.9])

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.415 | 0.481 | 0.000 | 1.000 | 0.000 | 0.000 |
| Liu 2019 | 104 | 0.549 | 0.538 | 0.417 | 0.643 | 0.500 | 0.455 |
| Riaz 2017 | 64 | 0.500 | 0.312 | 1.000 | 0.000 | 0.312 | 0.476 |

![[roc_curves_combined_elasticnet.png]]

_Figure: ROC curves for ElasticNet Logistic Regression (GridSearchCV: C in [0.001,0.01,0.1,1.0,10.0], l1_ratio in [0.1,0.3,0.5,0.7,0.9]) with combined immune signature + driver mutation features._

---

## Survival Analysis (Log-Rank Test)

Kaplan-Meier survival curves stratify patients into high and low predicted-response-probability groups using the best-performing LOCO model per cohort (selected by AUC, excluding models with degenerate predictions). A log-rank test assesses whether the two groups have significantly different overall survival.

### Table: Survival Stratification Summary

| Cohort | Model Used | LOCO AUC | Log-Rank p-value | Significant (p < 0.05)? |
|:---|:---:|:---:|:---:|:---:|
| Hugo 2016 | SVM | 0.434 | 1.107e-01 | No |
| Liu 2019 | SVM | 0.657 | 9.729e-01 | No |
| Riaz 2017 | SVM | 0.717 | 6.545e-02 | No |
| TCGA-SKCM | LR | N/A (external) | 9.798e-01 | No |

### Kaplan-Meier Curves

#### Hugo 2016

![[survival_hugo_svm.png]]

_Figure: KM survival curves for Hugo 2016 stratified by SVM predicted response probability (log-rank p = 1.107e-01)._

#### Liu 2019

![[survival_liu_svm.png]]

_Figure: KM survival curves for Liu 2019 stratified by SVM predicted response probability (log-rank p = 9.729e-01)._

#### Riaz 2017

![[survival_riaz_svm.png]]

_Figure: KM survival curves for Riaz 2017 stratified by SVM predicted response probability (log-rank p = 6.545e-02)._

#### TCGA-SKCM

![[survival_tcga_lr.png]]

_Figure: KM survival curves for TCGA-SKCM stratified by LR predicted response probability (log-rank p = 9.798e-01)._

---

## Summary & Interpretation

### Key Metrics Explained:
- **Sensitivity (Recall)**: TP / (TP + FN) — Proportion of actual responders correctly identified
- **Specificity**: TN / (TN + FP) — Proportion of actual non-responders correctly identified
- **Precision**: TP / (TP + FP) — Proportion of predicted responders who are actually responders
- **Accuracy**: (TP + TN) / Total — Overall correctness across both classes
- **F1-Score**: Harmonic mean of Precision and Recall — Balances both metrics
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve — Robustness to threshold selection
- **C-Index (Concordance Index)**: Evaluates how well predicted response probabilities rank patients by survival. 0.5 = random, 1.0 = perfect. Accounts for censoring in survival data.

### Visualizations:
- **ROC Curves** (`roc_curves_*.png`): Trade-off between True Positive Rate and False Positive Rate
- **PR Curves** (`pr_curves_*.png`): Precision-Recall trade-off, especially relevant for class imbalance
- **Confusion Matrices** (`confusion_matrices_*.png`): Cell-level breakdown of predictions per cohort

