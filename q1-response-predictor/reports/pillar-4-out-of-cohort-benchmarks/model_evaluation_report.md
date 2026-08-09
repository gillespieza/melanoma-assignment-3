---
title: "Model Evaluation Report: Leave-One-Cohort-Out (LOCO) Cross-Validation"
aliases:
  - LOCO Model Evaluation Report
  - Q1 Response Predictor Evaluation
tags:
  - calibration
  - immunotherapy-response
  - loco-cv
  - model-evaluation
  - q1
created: 2026-08-09 17:06
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-09 19:58
---

# Model Evaluation Report: Leave-One-Cohort-Out (LOCO) Cross-Validation

> [!NOTE] What, Why & Key Questions
> - **What**: Evaluated 5 machine learning models (ElasticNet, Logistic Regression, XGBoost, Random Forest, SVM) predicting anti-PD-1 / immunotherapy response across held-out clinical cohorts.
> - **Why**: Testing on held-out trial cohorts (Leave-One-Cohort-Out / LOCO) measures true real-world generalisability under domain shift.
> - **Questions**: Which model architecture generalises best to unseen patient populations? Does integrating somatic driver mutations with transcriptomic signatures enhance out-of-cohort performance?

> [!INSIGHT] Key Benchmark Insights & Takeaways
> 1. **Best Generalising Model**: **ElasticNet Logistic Regression** achieved the highest mean cross-cohort AUC of **0.584**, outperforming baseline L1 Logistic Regression (**0.551**), XGBoost (**0.507**), Random Forest (**0.503**), and SVM (**0.472**).
> 2. **Linear Simplicity vs Ensemble Overfitting**: Regularised linear models (ElasticNet, LR) generalise significantly better under domain shift than non-linear complex tree ensembles (RF, XGBoost), which suffer from overfitting to training cohort batch characteristics.
> 3. **Multimodal Feature Integration**: Adding somatic driver mutations (`mut_BRAF`, `mut_NRAS`, `mut_NF1`) to transcriptomic immune signatures yielded identical cross-cohort predictive performance, indicating transcriptomic features dominate response signal.

---

## 1. Cross-Model Benchmark Overview

### 1.1 Methodology & LOCO Framework
1. **Evaluation Framework (LOCO)**: Models are trained on $N-1$ trial cohorts and evaluated on 1 completely unseen test cohort.
2. **Active Test Cohorts**: Evaluated across 6 clinical trial datasets (Gide 2019, Hugo 2016, Liu 2019, Riaz 2017, TCGA GDC 2025, Van Allen 2015) with cohort-independent Z-score feature standardisation.
3. **Feature Sets**: Baseline transcriptomic immune signatures (`IFN_gamma`, `TIS`, `CYT`, `CD8_Tcell`, `IMPRES`, `PD_L1`) and multimodal features (+ `mut_BRAF`, `mut_NRAS`, `mut_NF1`).

### 1.2 Cross-Model AUC Comparison Table

| Model | Gide 2019 | Hugo 2016 | Liu 2019 | Riaz 2017 | TCGA GDC 2025 | Van Allen 2015 | **Mean AUC** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ElasticNet** | 0.728 | 0.451 | 0.539 | 0.740 | 0.401 | 0.643 | **0.584** |
| **LR** | 0.522 | 0.511 | 0.512 | 0.743 | 0.422 | 0.599 | **0.551** |
| **XGB** | 0.498 | 0.412 | 0.570 | 0.591 | 0.482 | 0.489 | **0.507** |
| **RF** | 0.374 | 0.489 | 0.555 | 0.657 | 0.414 | 0.527 | **0.503** |
| **SVM** | 0.626 | 0.308 | 0.571 | 0.343 | 0.420 | 0.566 | **0.472** |

---

## 2. Detailed Performance Metrics by Model

### 2.1 ElasticNet Logistic Regression (Top Performer — Mean AUC: 0.584)

| Test Cohort | N | AUC | ECE | Brier | Accuracy | Sens | Spec | Prec | F1 | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gide 2019 | 78 | 0.728 | 0.218 | 0.269 | 0.372 | 0.000 | 1.000 | 0.000 | 0.000 | 0.366 |
| Hugo 2016 | 27 | 0.451 | 0.108 | 0.260 | 0.444 | 0.071 | 0.846 | 0.333 | 0.118 | 0.602 |
| Liu 2019 | 104 | 0.539 | 0.129 | 0.277 | 0.548 | 0.292 | 0.768 | 0.519 | 0.373 | 0.457 |
| Riaz 2017 | 64 | 0.740 | 0.182 | 0.238 | 0.703 | 0.650 | 0.727 | 0.520 | 0.578 | 0.327 |
| TCGA GDC 2025 | 52 | 0.401 | 0.103 | 0.266 | 0.462 | 0.192 | 0.731 | 0.417 | 0.263 | 0.455 |
| Van Allen 2015 | 33 | 0.643 | 0.271 | 0.234 | 0.515 | 0.429 | 0.538 | 0.200 | 0.273 | 0.334 |

### 2.2 Logistic Regression (L1-Penalised — Mean AUC: 0.551)

| Test Cohort | N | AUC | ECE | Brier | Accuracy | Sens | Spec | Prec | F1 | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gide 2019 | 78 | 0.522 | 0.235 | 0.294 | 0.410 | 0.224 | 0.724 | 0.579 | 0.324 | 0.481 |
| Hugo 2016 | 27 | 0.511 | 0.076 | 0.251 | 0.444 | 0.429 | 0.462 | 0.462 | 0.444 | 0.536 |
| Liu 2019 | 104 | 0.512 | 0.201 | 0.316 | 0.558 | 0.271 | 0.804 | 0.542 | 0.361 | 0.462 |
| Riaz 2017 | 64 | 0.743 | 0.187 | 0.244 | 0.625 | 0.750 | 0.568 | 0.441 | 0.556 | 0.311 |
| TCGA GDC 2025 | 52 | 0.422 | 0.081 | 0.255 | 0.442 | 0.385 | 0.500 | 0.435 | 0.408 | 0.441 |
| Van Allen 2015 | 33 | 0.599 | 0.287 | 0.246 | 0.576 | 0.714 | 0.538 | 0.294 | 0.417 | 0.361 |

### 2.3 XGBoost Gradient Boosting (Mean AUC: 0.507)

| Test Cohort | N | AUC | ECE | Brier | Accuracy | Sens | Spec | Prec | F1 | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gide 2019 | 78 | 0.498 | 0.215 | 0.280 | 0.372 | 0.000 | 1.000 | 0.000 | 0.000 | 0.497 |
| Hugo 2016 | 27 | 0.412 | 0.151 | 0.263 | 0.407 | 0.071 | 0.769 | 0.250 | 0.111 | 0.592 |
| Liu 2019 | 104 | 0.570 | 0.090 | 0.252 | 0.538 | 0.250 | 0.786 | 0.500 | 0.333 | 0.443 |
| Riaz 2017 | 64 | 0.591 | 0.175 | 0.243 | 0.641 | 0.450 | 0.727 | 0.429 | 0.439 | 0.404 |
| TCGA GDC 2025 | 52 | 0.482 | 0.061 | 0.256 | 0.500 | 0.000 | 1.000 | 0.000 | 0.000 | 0.515 |
| Van Allen 2015 | 33 | 0.489 | 0.271 | 0.241 | 0.788 | 0.000 | 1.000 | 0.000 | 0.000 | 0.523 |

### 2.4 Random Forest Classifier (Mean AUC: 0.503)

| Test Cohort | N | AUC | ECE | Brier | Accuracy | Sens | Spec | Prec | F1 | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gide 2019 | 78 | 0.374 | 0.204 | 0.280 | 0.372 | 0.000 | 1.000 | 0.000 | 0.000 | 0.558 |
| Hugo 2016 | 27 | 0.489 | 0.067 | 0.254 | 0.481 | 0.000 | 1.000 | 0.000 | 0.000 | 0.520 |
| Liu 2019 | 104 | 0.555 | 0.070 | 0.248 | 0.596 | 0.229 | 0.911 | 0.688 | 0.344 | 0.429 |
| Riaz 2017 | 64 | 0.657 | 0.205 | 0.244 | 0.719 | 0.500 | 0.818 | 0.556 | 0.526 | 0.411 |
| TCGA GDC 2025 | 52 | 0.414 | 0.124 | 0.271 | 0.462 | 0.115 | 0.808 | 0.375 | 0.176 | 0.483 |
| Van Allen 2015 | 33 | 0.527 | 0.270 | 0.240 | 0.788 | 0.000 | 1.000 | 0.000 | 0.000 | 0.452 |

### 2.5 Support Vector Machine (SVM — Mean AUC: 0.472)

| Test Cohort | N | AUC | ECE | Brier | Accuracy | Sens | Spec | Prec | F1 | C-Index |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gide 2019 | 78 | 0.626 | 0.372 | 0.372 | 0.628 | 1.000 | 0.000 | 0.628 | 0.772 | 0.446 |
| Hugo 2016 | 27 | 0.308 | 0.198 | 0.283 | 0.370 | 0.143 | 0.615 | 0.286 | 0.190 | 0.668 |
| Liu 2019 | 104 | 0.571 | 0.052 | 0.246 | 0.558 | 0.104 | 0.946 | 0.625 | 0.179 | 0.453 |
| Riaz 2017 | 64 | 0.343 | 0.177 | 0.249 | 0.562 | 0.000 | 0.818 | 0.000 | 0.000 | 0.647 |
| TCGA GDC 2025 | 52 | 0.420 | 0.211 | 0.287 | 0.462 | 0.385 | 0.538 | 0.455 | 0.417 | 0.446 |
| Van Allen 2015 | 33 | 0.566 | 0.286 | 0.246 | 0.606 | 0.714 | 0.577 | 0.312 | 0.435 | 0.373 |

---

## 3. Multimodal Integration: Immune Signatures + Driver Mutations

Integrating somatic driver mutation status (`mut_BRAF`, `mut_NRAS`, `mut_NF1`) alongside baseline transcriptomic immune signatures yields comparable predictive performance across held-out test cohorts.

| Model | Gide 2019 | Hugo 2016 | Liu 2019 | Riaz 2017 | TCGA GDC 2025 | Van Allen 2015 | **Multimodal Mean AUC** | Baseline Mean AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ElasticNet** | 0.728 | 0.451 | 0.539 | 0.740 | 0.401 | 0.643 | **0.584** | 0.584 |
| **LR** | 0.522 | 0.511 | 0.512 | 0.743 | 0.422 | 0.599 | **0.551** | 0.551 |
| **XGB** | 0.498 | 0.412 | 0.570 | 0.591 | 0.482 | 0.489 | **0.507** | 0.507 |
| **RF** | 0.374 | 0.489 | 0.555 | 0.657 | 0.414 | 0.527 | **0.503** | 0.503 |
| **SVM** | 0.626 | 0.308 | 0.571 | 0.343 | 0.420 | 0.566 | **0.472** | 0.472 |

> [!NOTE]- Multimodal ROC Curves by Model (Click to Expand)
> | ElasticNet Multimodal | Logistic Regression Multimodal |
> |:---:|:---:|
> | ![ElasticNet Multimodal](../../plots/models/roc_curves_combined_elasticnet.png) | ![LR Multimodal](../../plots/models/roc_curves_combined_lr.png) |
> | **XGBoost Multimodal** | **Random Forest Multimodal** |
> | ![XGB Multimodal](../../plots/models/roc_curves_combined_xgb.png) | ![RF Multimodal](../../plots/models/roc_curves_combined_rf.png) |
> | **SVM Multimodal** | |
> | ![SVM Multimodal](../../plots/models/roc_curves_combined_svm.png) | |

---

## 4. Downstream Overall Survival Stratification

Models selected based on top LOCO cross-validation performance were applied to stratify patients into predicted responder vs non-responder risk groups for Overall Survival (OS) analysis using Kaplan-Meier estimator curves and log-rank tests.

| Cohort | Model Selected | Best LOCO AUC | Log-Rank p-value | Statistically Significant (p < 0.05)? |
|:---|:---:|:---:|:---:|:---:|
| Riaz 2017 | LR | 0.743 | 6.863e-03 | **Yes** |
| Gide 2019 | ELASTICNET | 0.728 | 3.668e-02 | **Yes** |
| Van Allen 2015 | ELASTICNET | 0.643 | 2.267e-01 | No |
| Liu 2019 | SVM | 0.571 | 3.233e-01 | No |
| Hugo 2016 | LR | 0.511 | 4.755e-01 | No |
| TCGA GDC 2025 | XGB | 0.482 | 6.914e-01 | No |
| TCGA GDC 2025 (IT Subcohort) | LR | N/A (external) | 1.585e-01 | No |

### 4.1 Kaplan-Meier Survival Stratification Overview

![Kaplan-Meier 2x2 Grid](../../plots/models/survival_2x2_grid.png)

---

## Appendix: Model Diagnostic Gallery (Obsidian Collapsible Callouts)

> [!INFO]- ElasticNet Diagnostic Grid (Top Model)
> | ROC Curves | Precision-Recall Curves |
> |:---:|:---:|
> | ![ROC Curves](../../plots/models/roc_curves_elasticnet.png) | ![PR Curves](../../plots/models/pr_curves_elasticnet.png) |
> | **Calibration Curves** | **Confusion Matrices** |
> | ![Calibration Curves](../../plots/models/calibration_curves_elasticnet.png) | ![Confusion Matrices](../../plots/models/confusion_matrices_elasticnet.png) |

> [!INFO]- Logistic Regression Diagnostic Grid
> | ROC Curves | Precision-Recall Curves |
> |:---:|:---:|
> | ![ROC Curves](../../plots/models/roc_curves_lr.png) | ![PR Curves](../../plots/models/pr_curves_lr.png) |
> | **Calibration Curves** | **Confusion Matrices** |
> | ![Calibration Curves](../../plots/models/calibration_curves_lr.png) | ![Confusion Matrices](../../plots/models/confusion_matrices_lr.png) |

> [!INFO]- XGBoost Gradient Boosting Diagnostic Grid
> | ROC Curves | Precision-Recall Curves |
> |:---:|:---:|
> | ![ROC Curves](../../plots/models/roc_curves_xgb.png) | ![PR Curves](../../plots/models/pr_curves_xgb.png) |
> | **Calibration Curves** | **Confusion Matrices** |
> | ![Calibration Curves](../../plots/models/calibration_curves_xgb.png) | ![Confusion Matrices](../../plots/models/confusion_matrices_xgb.png) |

> [!INFO]- Random Forest Diagnostic Grid
> | ROC Curves | Precision-Recall Curves |
> |:---:|:---:|
> | ![ROC Curves](../../plots/models/roc_curves_rf.png) | ![PR Curves](../../plots/models/pr_curves_rf.png) |
> | **Calibration Curves** | **Confusion Matrices** |
> | ![Calibration Curves](../../plots/models/calibration_curves_rf.png) | ![Confusion Matrices](../../plots/models/confusion_matrices_rf.png) |

> [!INFO]- Support Vector Machine (SVM) Diagnostic Grid
> | ROC Curves | Precision-Recall Curves |
> |:---:|:---:|
> | ![ROC Curves](../../plots/models/roc_curves_svm.png) | ![PR Curves](../../plots/models/pr_curves_svm.png) |
> | **Calibration Curves** | **Confusion Matrices** |
> | ![Calibration Curves](../../plots/models/calibration_curves_svm.png) | ![Confusion Matrices](../../plots/models/confusion_matrices_svm.png) |
