# Model Evaluation Report: LOCO Cross-Cohort Validation
## Overview
This report documents the comprehensive evaluation of all trained models (Logistic Regression, Random Forest, XGBoost, SVM, ElasticNet) using Leave-One-Cohort-Out (LOCO) cross-validation on three independent melanoma immunotherapy cohorts.
**Evaluation Framework:**
- **Cross-validation**: Leave-One-Cohort-Out (LOCO) — train on 2 cohorts, test on 1
- **Test cohorts**: Liu 2019 (N=104), Hugo 2016 (N=27), Riaz 2017 (N=64)
- **Features**: 11 immune response signatures (IFN-gamma, TIS, CD8 T-cell, CYT, IMPRES, PD-L1, etc.)
- **Decision threshold**: 0.5 (standard for binary classification)
- **Metrics**: AUC-ROC, Accuracy, Sensitivity, Specificity, Precision, F1-score, C-index

---

## Logistic Regression (L1-penalized, GridSearchCV)

### Performance Metrics (Threshold = 0.5)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.415 | 0.481 | 0.429 | 0.538 | 0.500 | 0.462 | 0.612 |
| Liu 2019 | 104 | 0.609 | 0.625 | 0.500 | 0.732 | 0.615 | 0.552 | 0.398 |
| Riaz 2017 | 64 | 0.500 | 0.312 | 1.000 | 0.000 | 0.312 | 0.476 | 0.500 |

### Confusion Matrices (Threshold = 0.5)

**Hugo 2016** (N=27):
```
                Predicted
              Non-Resp  Resp
Actual Non-Resp    7    6   
Actual Resp        8    6   
```

**Liu 2019** (N=104):
```
                Predicted
              Non-Resp  Resp
Actual Non-Resp    41   15  
Actual Resp        24   24  
```

**Riaz 2017** (N=64):
```
                Predicted
              Non-Resp  Resp
Actual Non-Resp    0    44  
Actual Resp        0    20  
```

### Precision-Recall Curve Summaries

**Hugo 2016**: Average Precision = 0.539
**Liu 2019**: Average Precision = 0.611
**Riaz 2017**: Average Precision = 0.312

---

## Random Forest (GridSearchCV: n_estimators in [50,100,200], max_depth in [3,5,10,None], min_samples_leaf in [1,2,4])

### Performance Metrics (Threshold = 0.5)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.423 | 0.444 | 0.286 | 0.615 | 0.444 | 0.348 | 0.551 |
| Liu 2019 | 104 | 0.580 | 0.596 | 0.188 | 0.946 | 0.750 | 0.300 | 0.431 |
| Riaz 2017 | 64 | 0.678 | 0.609 | 0.800 | 0.523 | 0.432 | 0.561 | 0.455 |

### Confusion Matrices (Threshold = 0.5)

**Hugo 2016** (N=27):
```
                Predicted
              Non-Resp  Resp
Actual Non-Resp    8    5   
Actual Resp        10   4   
```

**Liu 2019** (N=104):
```
                Predicted
              Non-Resp  Resp
Actual Non-Resp    53   3   
Actual Resp        39   9   
```

**Riaz 2017** (N=64):
```
                Predicted
              Non-Resp  Resp
Actual Non-Resp    23   21  
Actual Resp        4    16  
```

### Precision-Recall Curve Summaries

**Hugo 2016**: Average Precision = 0.484
**Liu 2019**: Average Precision = 0.611
**Riaz 2017**: Average Precision = 0.544

---

## XGBoost (GridSearchCV: n_estimators in [50,100,150], max_depth in [3,5,7], learning_rate in [0.01,0.05,0.1,0.2])

### Performance Metrics (Threshold = 0.5)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.319 | 0.333 | 0.071 | 0.615 | 0.167 | 0.100 | 0.597 |
| Liu 2019 | 104 | 0.581 | 0.567 | 0.521 | 0.607 | 0.532 | 0.526 | 0.471 |
| Riaz 2017 | 64 | 0.618 | 0.531 | 0.650 | 0.477 | 0.361 | 0.464 | 0.515 |

### Confusion Matrices (Threshold = 0.5)

**Hugo 2016** (N=27):
```
                Predicted
              Non-Resp  Resp
Actual Non-Resp    8    5   
Actual Resp        13   1   
```

**Liu 2019** (N=104):
```
                Predicted
              Non-Resp  Resp
Actual Non-Resp    34   22  
Actual Resp        23   25  
```

**Riaz 2017** (N=64):
```
                Predicted
              Non-Resp  Resp
Actual Non-Resp    21   23  
Actual Resp        7    13  
```

### Precision-Recall Curve Summaries

**Hugo 2016**: Average Precision = 0.426
**Liu 2019**: Average Precision = 0.592
**Riaz 2017**: Average Precision = 0.478

---

## Support Vector Machine (GridSearchCV: C in [0.01,0.1,1.0,10.0], kernel in [linear,rbf])

### Performance Metrics (Threshold = 0.5)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.434 | 0.407 | 0.214 | 0.615 | 0.375 | 0.273 | 0.663 |
| Liu 2019 | 104 | 0.617 | 0.577 | 0.208 | 0.893 | 0.625 | 0.312 | 0.401 |
| Riaz 2017 | 64 | 0.277 | 0.688 | 0.000 | 1.000 | 0.000 | 0.000 | 0.572 |

### Confusion Matrices (Threshold = 0.5)

**Hugo 2016** (N=27):
```
                Predicted
              Non-Resp  Resp
Actual Non-Resp    8    5   
Actual Resp        11   3   
```

**Liu 2019** (N=104):
```
                Predicted
              Non-Resp  Resp
Actual Non-Resp    50   6   
Actual Resp        38   10  
```

**Riaz 2017** (N=64):
```
                Predicted
              Non-Resp  Resp
Actual Non-Resp    44   0   
Actual Resp        20   0   
```

### Precision-Recall Curve Summaries

**Hugo 2016**: Average Precision = 0.522
**Liu 2019**: Average Precision = 0.609
**Riaz 2017**: Average Precision = 0.272

---

## ElasticNet Logistic Regression (GridSearchCV: C in [0.001,0.01,0.1,1.0,10.0], l1_ratio in [0.1,0.3,0.5,0.7,0.9])

### Performance Metrics (Threshold = 0.5)

| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Hugo 2016 | 27 | 0.415 | 0.481 | 0.000 | 1.000 | 0.000 | 0.000 | 0.612 |
| Liu 2019 | 104 | 0.616 | 0.615 | 0.479 | 0.732 | 0.605 | 0.535 | 0.397 |
| Riaz 2017 | 64 | 0.500 | 0.312 | 1.000 | 0.000 | 0.312 | 0.476 | 0.500 |

### Confusion Matrices (Threshold = 0.5)

**Hugo 2016** (N=27):
```
                Predicted
              Non-Resp  Resp
Actual Non-Resp    13   0   
Actual Resp        14   0   
```

**Liu 2019** (N=104):
```
                Predicted
              Non-Resp  Resp
Actual Non-Resp    41   15  
Actual Resp        25   23  
```

**Riaz 2017** (N=64):
```
                Predicted
              Non-Resp  Resp
Actual Non-Resp    0    44  
Actual Resp        0    20  
```

### Precision-Recall Curve Summaries

**Hugo 2016**: Average Precision = 0.539
**Liu 2019**: Average Precision = 0.607
**Riaz 2017**: Average Precision = 0.312

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

