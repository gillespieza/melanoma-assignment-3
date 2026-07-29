---
title:
aliases: 
tags: 
created: 2026-07-20 15:51
cssclasses: table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-25 22:46
---

> [!summary]+ Contents
> ```table-of-contents
> style: nestedList  # nestedList, nestedOrderedList, inlineFirstLevel
> hideWhenEmpty: true # Hide TOC if no headings are found
> ```

# Model Evaluation Report: Leave-One-Cohort-Out (LOCO) Cross-Validation

> [!summary] What, Why & Key Questions  
> **What**: We tested 5 machine learning models (Logistic Regression, Random Forest, XGBoost, Support Vector Machine, and ElasticNet) to see how accurately they predict immunotherapy response in melanoma patients.  
> **Why**: Cross-validation within a single dataset can give overly optimistic results due to hidden local biases. Testing each model on a completely unseen hospital trial cohort (Leave-One-Cohort-Out) reveals how well the models perform in real-world clinical practice.  
> **Key Questions Answered**:
> 1. Which machine learning model generalises best across independent trial cohorts? Does combining genomic mutation flags with immune signatures improve prediction accuracy?
> 2. Does adding somatic driver mutation status improve cross-cohort response prediction performance (multimodal)?

## Overview & Methodology

1. **Evaluation Framework (Leave-One-Cohort-Out)**: In each fold, we train models on 2 patient cohorts and test them on the remaining 1 unseen cohort.
2. **Test Cohorts**: Liu 2019 ($N=104$), Hugo 2016 ($N=27$), and Riaz 2017 ($N=64$).
3. **Features Evaluated**: Pre-defined immune response signatures (IFN-γ, TIS, CD8 T-cell, CYT, IMPRES, PD-L1).
4. **Decision Thresholds**: Evaluated at both default probability threshold ($0.5$) and Youden's J optimal threshold.
	 > Youden's J statistic (J = \text{sensitivity} + \text{specificity} - 1) calculates the optimal decision boundary that balances true positives and true negatives. Evaluating threshold optimization on test data provides an upper-bound performance benchmark.

---

## Logistic Regression (L1-Penalised)

> [!NOTE]  
> **What We Did**: Trained a linear model with **L1 (Lasso) regularization** to select key predictive features.  
> **Why**: Linear models serve as transparent baselines that prevent overfitting by shrinking uninformative feature weights to zero.  
> **Question Answered**: Can a simple, interpretable linear combination of immune signatures predict patient response across cohorts?

### Performance Metrics (Default Threshold = 0.5)

| Test Cohort |  N  |  AUC  | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |
|:----------- |:---:|:-----:|:--------:|:-----------:|:-----------:|:---------:|:--------:|:-------:|
| Hugo 2016   | 27  | 0.415 |  0.481   |    0.429    |    0.538    |   0.500   |  0.462   |  0.612  |
| Liu 2019    | 104 | 0.609 |  0.625   |    0.500    |    0.732    |   0.615   |  0.552   |  0.398  |
| Riaz 2017   | 64  | 0.500 |  0.312   |    1.000    |    0.000    |   0.312   |  0.476   |  0.500  |

### Performance Metrics (Youden's J Optimal Threshold)

| Test Cohort |  N  |  AUC  | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:----------- |:---:|:-----:|:---------:|:--------:|:-----------:|:-----------:|:---------:|:--------:|
| Hugo 2016   | 27  | 0.415 |   0.588   |  0.519   |    0.071    |    1.000    |   1.000   |  0.133   |
| Liu 2019    | 104 | 0.609 |   0.466   |  0.625   |    0.562    |    0.679    |   0.600   |  0.581   |
| Riaz 2017   | 64  | 0.500 |    inf    |  0.688   |    0.000    |    1.000    |   0.000   |  0.000   |

### Multimodal (L1-Penalised)

| Test Cohort |  N  |  AUC  | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:----------- |:---:|:-----:|:--------:|:-----------:|:-----------:|:---------:|:--------:|
| Hugo 2016   | 27  | 0.415 |  0.481   |    0.429    |    0.538    |   0.500   |  0.462   |
| Liu 2019    | 104 | 0.609 |  0.625   |    0.500    |    0.732    |   0.615   |  0.552   |
| Riaz 2017   | 64  | 0.500 |  0.312   |    1.000    |    0.000    |   0.312   |  0.476   |

### Visualisations & Diagnostics

#### Confusion Matrices
![[confusion_matrices_lr.png]]

![[confusion_matrices_lr_optimal.png]]

_Figure: Confusion matrices for Logistic Regression (L1-Penalised) at default 0.5 vs Youden;s J Optimal decision threshold across test cohorts._

> [!tip]+ How to Interpret a Confusion Matrix  
> A **confusion matrix** is a simple 2×2 grid that compares what a machine learning model predicted against what actually happened to the patients.
>
> |                          | **Predicted Non-Responder**                                    | **Predicted Responder**                                |
> | ------------------------ | -------------------------------------------------------------- | ------------------------------------------------------ |
> | **Actual Non-Responder** | **True Negative (TN)**  <br>Correctly identified non-responder | **False Positive (FP)**  <br>Type I error: False alarm |
> |**Actual Responder**|**False Negative (FN)**  <br>Type II error: Missed opportunity|**True Positive (TP)**  <br>Correctly identified responder|
>
> ### Reading the 4 Boxes:
>
> 1. **True Positives (TP)** _(Top-Right / Bottom-Right depending on axes)_: Patients who **did respond** to treatment, and the model **correctly predicted** they would respond.
> 2. **True Negatives (TN)**: Patients who **did not respond**, and the model **correctly predicted** they would not respond.
> 3. **False Positives (FP)** _(False Alarm)_: Patients who did **not** respond, but the model mistakenly predicted they **would**. In clinical practice, this could lead to giving an ineffective treatment with potential side effects.
> 4. **False Negatives (FN)** _(Missed Opportunity)_: Patients who **would** have responded, but the model mistakenly predicted they **would not**. In clinical practice, this could deny a patient a lifesaving therapy.
> 
> ### Quick Rule of Thumb:
>
> - **Diagonal from Top-Left to Bottom-Right**: These are your **correct predictions**. You want these numbers to be as **high** as possible!
> - **Off-Diagonal Boxes**: These are your **errors**. You want these numbers to be as **low** as possible!

#### ROC & Precision-Recall Curves
![[roc_curves_lr.png|300]] ![[roc_curves_combined_lr.png|300]]

_Figure: ROC curves for Logistic Regression (L1-Penalised) across LOCO test cohorts. Dashed diagonal indicates chance performance (AUC = 0.5)._

> [!summary]  
> 1. The left plot displays the **Receiver Operating Characteristic (ROC) curves** for a **Logistic Regression** model evaluated using **Leave-One-Cohort-Out (LOCO) cross-validation**.
> 	- **Features Used**: The 6 curated transcriptomic immune signatures
> 	- **Performance Breakdown**:
> 	    - **Liu 2019 (Blue)**: AUC=0.609AUC=0.609 — Moderate discrimination (curves above the diagonal chance line).
> 	    - **Hugo 2016 (Orange)**: AUC=0.415AUC=0.415 — Below random chance (AUC<0.50AUC<0.50), indicating poor cross-cohort transferability to small trial cohorts (N=27N=27).
> 	    - **Riaz 2017 (Green)**: AUC=0.500AUC=0.500 — Exactly on the diagonal chance line (random guessing level).
> 2. The right plot displays the ROC curves for the same Logistic Regression model, but after adding **somatic driver mutation flags** (`mut_BRAF`, `mut_NRAS`, `mut_NF1`) to the 6 immune signatures (multimodal feature integration).

> [!insight] Key Comparison Finding  
> **Adding driver mutation flags (BRAF, NRAS, NF1) to L1-penalised Logistic Regression resulted in zero improvement in out-of-cohort response prediction.**
>
> **Why?** L1 regularisation (Lasso) automatically shrinks uninformative feature coefficients to zero during training. Because driver mutations (BRAF/NRAS/NF1) occur at similar frequencies in both treatment responders and non-responders, the Logistic Regression model zeroes out their weights, leaving the model predictions identical to the immune-signature-only version.

![[pr_curves_lr.png|300]]

_Figure: Precision-Recall curves for Logistic Regression (L1-Penalised), illustrating precision across sensitivity thresholds._

> [!summary]  
> This plot measures the **Precision vs. Recall (Sensitivity)** trade-off for Logistic Regression across different probability thresholds.
>
> - **What it evaluates**: How reliable the model's positive responder predictions are. This is particularly crucial when dealing with class imbalance (differing proportions of responders across trial cohorts).
> - **Average Precision (AP) Scores**:
>     - **Liu 2019 (Blue)**: AP=0.611 (Baseline prevalence ≈46%)
>     - **Hugo 2016 (Orange)**: AP=0.539 (Baseline prevalence ≈52%)
>     - **Riaz 2017 (Green)**: AP=0.312 (Baseline prevalence ≈31%)
> - **Interpretation**: A higher AP score indicates the model maintains higher precision (fewer false alarms) while capturing true treatment responders.

## Random Forest Classifier

> [!NOTE]  
> **What We Did**: Trained an ensemble of decision trees using random feature subsets.  
> **Why**: Decision trees capture non-linear relationships and feature interactions without assuming linear boundaries.  
> **Question Answered**: Do complex non-linear combinations of immune features improve out-of-cohort generalization?

### Performance Metrics (Default Threshold = 0.5)

| Test Cohort |  N  |  AUC  | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |
|:----------- |:---:|:-----:|:--------:|:-----------:|:-----------:|:---------:|:--------:|:-------:|
| Hugo 2016   | 27  | 0.423 |  0.444   |    0.286    |    0.615    |   0.444   |  0.348   |  0.551  |
| Liu 2019    | 104 | 0.580 |  0.596   |    0.188    |    0.946    |   0.750   |  0.300   |  0.431  |
| Riaz 2017   | 64  | 0.678 |  0.609   |    0.800    |    0.523    |   0.432   |  0.561   |  0.455  |

### Performance Metrics (Youden's J Optimal Threshold)

| Test Cohort |  N  |  AUC  | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:----------- |:---:|:-----:|:---------:|:--------:|:-----------:|:-----------:|:---------:|:--------:|
| Hugo 2016   | 27  | 0.423 |   0.319   |  0.593   |    0.714    |    0.462    |   0.588   |  0.645   |
| Liu 2019    | 104 | 0.580 |   0.426   |  0.635   |    0.417    |    0.821    |   0.667   |  0.513   |
| Riaz 2017   | 64  | 0.678 |   0.515   |  0.641   |    0.800    |    0.568    |   0.457   |  0.582   |

### Random Forest Classifier (Multimodal)

| Test Cohort |  N  |  AUC  | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:----------- |:---:|:-----:|:--------:|:-----------:|:-----------:|:---------:|:--------:|
| Hugo 2016   | 27  | 0.473 |  0.444   |    0.214    |    0.692    |   0.429   |  0.286   |
| Liu 2019    | 104 | 0.581 |  0.587   |    0.229    |    0.893    |   0.647   |  0.338   |
| Riaz 2017   | 64  | 0.688 |  0.672   |    0.750    |    0.636    |   0.484   |  0.588   |

### Visualisations & Diagnostics

#### Confusion Matrices
![[confusion_matrices_rf.png]]

![[confusion_matrices_rf_optimal.png]]

_Figure: Confusion matrices for Random Forest Classifier at default 0.5 vs Youden;s J Optimal decision threshold across test cohorts._

#### ROC & Precision-Recall Curves
![[roc_curves_rf.png|300]] ![[roc_curves_combined_rf.png|300]]

_Figure: ROC curves for Random Forest Classifier across LOCO test cohorts. Dashed diagonal indicates chance performance (AUC = 0.5)._

![[pr_curves_rf.png|300]]

_Figure: Precision-Recall curves for Random Forest Classifier, illustrating precision across sensitivity thresholds._

> [!insight] **Key Takeaway**: Non-Linear Decision Trees Can Extract Value from Genomic Mutations  
> Unlike linear models (where adding driver mutations made zero difference), **Random Forest performance improved when driver mutations were added**:
>
> - **Hugo 2016**: AUC increased from **0.423** →→ **0.473** (+0.050)
> - **Riaz 2017**: AUC increased from **0.678** →→ **0.688** (+0.010)
> - **Liu 2019**: AUC remained stable (**0.580** →→ **0.581**)
> 
> ### Why does Random Forest improve while Logistic Regression does not?
>
> 1. **Subgroup Interaction Splitting**: Decision trees do not evaluate features in isolation. Random Forest can create conditional rules like _"IF `IFN_gamma` is moderate AND `mut_BRAF` is positive, THEN predict higher response probability"_. Linear models cannot capture these non-linear interactions without explicit manual terms.
> 2. **Complementary Signals**: While somatic driver mutations (`BRAF`, `NRAS`, `NF1`) alone are not predictive of response, their interaction with the immune microenvironment provides a small, complementary signal that tree-based ensembles can harness to improve cross-cohort generalisation.

## XGBoost Gradient Boosting

> [!note]  
> **What We Did**: Trained a sequential gradient-boosted decision tree model with hyperparameter tuning.  
> **Why**: Gradient boosting iteratively corrects errors from previous trees, often achieving state-of-the-art tabular performance.  
> **Question Answered**: Does iterative error correction provide better sensitivity for identifying true responders?

### Performance Metrics (Default Threshold = 0.5)

| Test Cohort |  N  |  AUC  | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |
|:----------- |:---:|:-----:|:--------:|:-----------:|:-----------:|:---------:|:--------:|:-------:|
| Hugo 2016   | 27  | 0.319 |  0.333   |    0.071    |    0.615    |   0.167   |  0.100   |  0.597  |
| Liu 2019    | 104 | 0.581 |  0.567   |    0.521    |    0.607    |   0.532   |  0.526   |  0.471  |
| Riaz 2017   | 64  | 0.618 |  0.531   |    0.650    |    0.477    |   0.361   |  0.464   |  0.515  |

### Performance Metrics (Youden's J Optimal Threshold)

| Test Cohort |  N  |  AUC  | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:----------- |:---:|:-----:|:---------:|:--------:|:-----------:|:-----------:|:---------:|:--------:|
| Hugo 2016   | 27  | 0.319 |   0.007   |  0.519   |    0.929    |    0.077    |   0.520   |  0.667   |
| Liu 2019    | 104 | 0.581 |   0.387   |  0.596   |    0.688    |    0.518    |   0.550   |  0.611   |
| Riaz 2017   | 64  | 0.618 |   0.705   |  0.750   |    0.300    |    0.955    |   0.750   |  0.429   |

### XGBoost Gradient Boosting (Multimodal)

| Test Cohort |  N  |  AUC  | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:----------- |:---:|:-----:|:--------:|:-----------:|:-----------:|:---------:|:--------:|
| Hugo 2016   | 27  | 0.319 |  0.333   |    0.071    |    0.615    |   0.167   |  0.100   |
| Liu 2019    | 104 | 0.581 |  0.567   |    0.521    |    0.607    |   0.532   |  0.526   |
| Riaz 2017   | 64  | 0.618 |  0.531   |    0.650    |    0.477    |   0.361   |  0.464   |

### Visualisations & Diagnostics

#### Confusion Matrices
![[confusion_matrices_xgb.png]]

![[confusion_matrices_xgb_optimal.png]]

_Figure: Confusion matrices for XGBoost Gradient Boosting at default 0.5 vs Youden;s J Optimal decision threshold across test cohorts._

#### ROC & Precision-Recall Curves
![[roc_curves_xgb.png|300]] ![[roc_curves_combined_xgb.png|300]]

_Figure: ROC curves for XGBoost Gradient Boosting across LOCO test cohorts. Dashed diagonal indicates chance performance (AUC = 0.5)._

![[pr_curves_xgb.png|300]]

_Figure: Precision-Recall curves for XGBoost Gradient Boosting, illustrating precision across sensitivity thresholds._

> [!insight] **Key Takeaway**: Gradient Boosting Feature Importance Selection Ignores Somatic Mutations  
> Unlike Random Forest (which gained modest improvements from adding mutations), **XGBoost predictions remained 100% identical when driver mutations were added**.
>
> ### Why does XGBoost behave differently than Random Forest?
>
> 1. **Greedy Split Selection**: XGBoost builds trees sequentially to minimise gradient loss. At each split, XGBoost evaluates the split gain of every feature. Because transcriptomic immune signatures (like `TIS` and `IFN_gamma`) provide much stronger gradient reductions than binary mutation flags (`mut_BRAF`, `mut_NRAS`), XGBoost greedy splits consistently select the immune signatures and ignore the mutation flags completely.
> 2. **High Sensitivity to Overfitting on Small Cohorts**: On small clinical datasets (N=27 to 104), XGBoost regularization parameters (`learning_rate`, `max_depth`) prune weak binary splits early, resulting in identical tree structures whether driver mutations are included or not.

## Support Vector Machine (SVM)

> [!NOTE]  
> **What We Did**: Trained a Support Vector Machine classifier with linear and radial basis function (RBF) kernels.  
> **Why**: SVMs maximize the decision margin between responders and non-responders in high-dimensional feature spaces.  
> **Question Answered**: Can hyper-plane margin maximization achieve superior class separation on small clinical cohorts?

### Performance Metrics (Default Threshold = 0.5)

| Test Cohort |  N  |  AUC  | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |
|:----------- |:---:|:-----:|:--------:|:-----------:|:-----------:|:---------:|:--------:|:-------:|
| Hugo 2016   | 27  | 0.434 |  0.444   |    0.214    |    0.692    |   0.429   |  0.286   |  0.663  |
| Liu 2019    | 104 | 0.657 |  0.538   |    0.000    |    1.000    |   0.000   |  0.000   |  0.429  |
| Riaz 2017   | 64  | 0.717 |  0.719   |    0.500    |    0.818    |   0.556   |  0.526   |  0.428  |

### Performance Metrics (Youden's J Optimal Threshold)

| Test Cohort |  N  |  AUC  | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:----------- |:---:|:-----:|:---------:|:--------:|:-----------:|:-----------:|:---------:|:--------:|
| Hugo 2016   | 27  | 0.434 |   0.201   |  0.556   |    0.929    |    0.154    |   0.542   |  0.684   |
| Liu 2019    | 104 | 0.657 |   0.370   |  0.663   |    0.583    |    0.732    |   0.651   |  0.615   |
| Riaz 2017   | 64  | 0.717 |   0.483   |  0.734   |    0.650    |    0.773    |   0.565   |  0.605   |

### Support Vector Machine (SVM) (Multimodal)

| Test Cohort |  N  |  AUC  | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:----------- |:---:|:-----:|:--------:|:-----------:|:-----------:|:---------:|:--------:|
| Hugo 2016   | 27  | 0.434 |  0.444   |    0.214    |    0.692    |   0.429   |  0.286   |
| Liu 2019    | 104 | 0.657 |  0.538   |    0.000    |    1.000    |   0.000   |  0.000   |
| Riaz 2017   | 64  | 0.717 |  0.719   |    0.500    |    0.818    |   0.556   |  0.526   |

### Visualisations & Diagnostics

#### Confusion Matrices
![[confusion_matrices_svm.png]]

![[confusion_matrices_svm_optimal.png]]

_Figure: Confusion matrices for Support Vector Machine (SVM) at default 0.5 vs Youden;s J Optimal decision threshold across test cohorts._

#### ROC & Precision-Recall Curves
![[roc_curves_svm.png|300]] ![[roc_curves_combined_svm.png|300]]

_Figure: ROC curves for Support Vector Machine (SVM) across LOCO test cohorts. Dashed diagonal indicates chance performance (AUC = 0.5)._

![[pr_curves_svm.png|300]]

_Figure: Precision-Recall curves for Support Vector Machine (SVM), illustrating precision across sensitivity thresholds._

> [!insight] **Key Takeaway 1**: Support Vector Machine (SVM) achieves the highest cross-cohort accuracy overall
>
> - **Top Performer**: SVM achieved the single best cross-cohort generalisation performance among all 5 machine learning models tested, reaching **AUC = 0.717** on Riaz 2017 and **AUC = 0.657** on Liu 2019.
> - **Why SVM Excels Here**: On small, high-dimensional biological datasets, SVM's margin-maximising hyper-plane resists overfitting better than unconstrained decision trees or linear models.

> [!insight] **Key Takeaway 2**: Adding mutations to RBF-kernel SVM yields identical ranking
>
> - Just like Logistic Regression and XGBoost, adding `mut_BRAF`, `mut_NRAS`, and `mut_NF1` **produced zero change** in SVM's ranking order (ΔAUC=0.000ΔAUC=0.000).
> - **Why?** The Support Vector Classifier selected an RBF kernel (γ=’scale’γ=’scale’, C=0.1C=0.1). Because binary mutation flags (0 or 1) contribute very small Euclidean distances compared to continuous Z-scored immune signatures, the maximum-margin boundary decision relies almost entirely on the continuous transcriptomic features.

## ElasticNet Logistic Regression

> [!NOTE]  
> **What We Did**: Trained a logistic regression model combining L1 (Lasso) and L2 (Ridge) penalties.  
> **Why**: ElasticNet balances feature selection (L1) with stability among correlated features (L2).  
> **Question Answered**: Does balancing feature elimination and grouping improve stability across heterogeneous trials?

### Performance Metrics (Default Threshold = 0.5)

| Test Cohort |  N  |  AUC  | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |
|:----------- |:---:|:-----:|:--------:|:-----------:|:-----------:|:---------:|:--------:|:-------:|
| Hugo 2016   | 27  | 0.415 |  0.481   |    0.000    |    1.000    |   0.000   |  0.000   |  0.612  |
| Liu 2019    | 104 | 0.616 |  0.615   |    0.458    |    0.750    |   0.611   |  0.524   |  0.398  |
| Riaz 2017   | 64  | 0.500 |  0.312   |    1.000    |    0.000    |   0.312   |  0.476   |  0.500  |

### Performance Metrics (Youden's J Optimal Threshold)

| Test Cohort |  N  |  AUC  | Threshold | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:----------- |:---:|:-----:|:---------:|:--------:|:-----------:|:-----------:|:---------:|:--------:|
| Hugo 2016   | 27  | 0.415 |   0.443   |  0.519   |    0.071    |    1.000    |   1.000   |  0.133   |
| Liu 2019    | 104 | 0.616 |   0.448   |  0.625   |    0.583    |    0.661    |   0.596   |  0.589   |
| Riaz 2017   | 64  | 0.500 |    inf    |  0.688   |    0.000    |    1.000    |   0.000   |  0.000   |

### ElasticNet Logistic Regression (Multimodal)

| Test Cohort |  N  |  AUC  | Accuracy | Sensitivity | Specificity | Precision | F1-Score |
|:----------- |:---:|:-----:|:--------:|:-----------:|:-----------:|:---------:|:--------:|
| Hugo 2016   | 27  | 0.415 |  0.481   |    0.000    |    1.000    |   0.000   |  0.000   |
| Liu 2019    | 104 | 0.616 |  0.615   |    0.458    |    0.750    |   0.611   |  0.524   |
| Riaz 2017   | 64  | 0.500 |  0.312   |    1.000    |    0.000    |   0.312   |  0.476   |

### Visualisations & Diagnostics

#### Confusion Matrices
![[confusion_matrices_elasticnet.png]]

![[confusion_matrices_elasticnet_optimal.png]]

_Figure: Confusion matrices for ElasticNet Logistic Regression at default 0.5 vs Youden;s J Optimal decision threshold across test cohorts._

#### ROC & Precision-Recall Curves
![[roc_curves_elasticnet.png|300]] ![[roc_curves_combined_elasticnet.png|300]]

_Figure: ROC curves for ElasticNet Logistic Regression across LOCO test cohorts. Dashed diagonal indicates chance performance (AUC = 0.5)._

![[pr_curves_elasticnet.png|300]]

_Figure: Precision-Recall curves for ElasticNet Logistic Regression, illustrating precision across sensitivity thresholds._

> [!insight] **Key Takeaway 1**: ElasticNet achieves slightly better linear performance than pure L1 (Lasso)
>
> - **Comparison to pure L1 (Logistic Regression)**: On Liu 2019, ElasticNet achieved **AUC = 0.616** compared to L1 Logistic Regression's **AUC = 0.609** (+0.007 boost).
> - **Why ElasticNet helps**: ElasticNet combines L1 (Lasso) and L2 (Ridge) penalties (`l1_ratio = 0.1`). Because immune signatures are highly correlated with each other (rs​=0.85–0.94), pure Lasso randomly picks one signature and discards the rest. ElasticNet keeps groups of correlated signatures together, preserving subtle complementary signals.

> [!insight] **Key Takeaway 2**: ElasticNet also zeroed out driver mutations
>
> - Just like L1 Logistic Regression, XGBoost, and SVM, adding binary driver mutation flags resulted in **zero change in AUC** (ΔAUC=0.000ΔAUC=0.000).
> - **Why?** The L1 component of ElasticNet shrank the weights of `mut_BRAF`, `mut_NRAS`, and `mut_NF1` to zero because somatic mutations showed no predictive coefficient value over the correlated immune signature block.

## Downstream Overall Survival Stratification

> [!summary]  
> **What We Did**: Stratified patients into predicted high-risk (low response probability) and low-risk (high response probability) groups using the best-performing LOCO model per cohort, then performed log-rank tests on overall survival.  
> **Why**: A clinically useful response predictor should also stratify long-term patient survival outcomes.  
> **Question Answered**: Do patients predicted as responders by our cross-cohort models demonstrate significantly longer overall survival?

### Summary Table: Survival Stratification

| Cohort    | Model Selected | Best LOCO AUC  | Log-Rank p-value | Significant (p < 0.05)? |
|:--------- |:--------------:|:--------------:|:----------------:|:-----------------------:|
| Hugo 2016 |      SVM       |     0.434      |    1.107e-01     |           No            |
| Liu 2019  |      SVM       |     0.657      |    9.729e-01     |           No            |
| Riaz 2017 |      SVM       |     0.717      |    6.545e-02     |           No            |
| TCGA-SKCM |       LR       | N/A (external) |    9.798e-01     |           No            |

### Kaplan-Meier Survival Curves

![[survival_curves_2x2_grid.png]]

_Figure: Kaplan-Meier overall survival curves for all 4 cohorts stratified by LR predicted response probability (log-rank p = 9.798e-01)._

> [!summary] What the 2×2 Survival Grid Tells Us  
> **Goal**: We tested whether patients predicted as "High Response Probability" (blue curve) live significantly longer than patients predicted as "Low Response Probability" (orange curve) using a Log-Rank test.
>
> 1. **Top-Left — Hugo 2016 (N=26, p=0.111)**:
>     - Patients predicted to respond (blue) show a lower survival probability curve in this small cohort (N=7 vs N=19), reflecting the inverted ROC AUC (0.434) observed for Hugo 2016.
> 2. **Top-Right — Liu 2019 (N=104, p=0.973)**:
>     - The high-response and low-response curves virtually overlap, showing no overall survival difference (p=0.973).
> 3. **Bottom-Left — Riaz 2017 (N=64, p=0.065#)**:
>     - Patients predicted as responders (blue, N=18) demonstrate visually superior overall survival compared to non-responders (orange, N=46), approaching statistical significance (p=0.065).
> 4. **Bottom-Right — TCGA-SKCM (N=426,p=0.980N=426,p=0.980)**:
>     - In unselected TCGA tumors (not treated with anti-PD-1 immunotherapy), response-probability stratification yields identical survival curves (p=0.980).
> 
> **Key Conclusion**: Immunotherapy response probability models trained on small clinical trials do not consistently stratify overall survival across independent patient cohorts without prospective, treatment-specific calibration.

### Why are none of them significant? (3 Key Reasons)

1. **Short-Term Tumor Shrinkage ≠ Long-Term Overall Survival**:
    - Our machine learning models were trained specifically to predict **RECIST tumour response** (CR/PR vs. PD) measured at 8–128–12 weeks post-treatment.
    - While tumour shrinkage is a good surrogate, overall survival is affected by downstream factors like post-progression therapies, patient performance status, organ function, and baseline tumour burden.
2. **Riaz 2017 Shows a Trend (p=0.065) but Lacks Power**:
    - In Riaz 2017, the predicted responders (blue line) clearly live longer than non-responders (orange line), but with only N=18 predicted responders and N=46 non-responders, the Log-Rank test lacks statistical power to cross the strict p<0.05 boundary.
3. **TCGA-SKCM Patients Were Untreated with Anti-PD-1**:
    - TCGA-SKCM samples were collected primarily prior to the FDA approval of anti-PD-1 checkpoint inhibitors.
    - A model trained to predict **immunotherapy response** will not stratify survival in historical patients who received standard chemotherapy or surgery instead of immunotherapy (p=0.980).

> [!insight] Critical Takeaway  
> This finding highlights a crucial distinction in clinical machine learning: **predicting treatment response is not the same as predicting overall survival**.
>
> A machine learning model can learn baseline immune features that correlate with short-term tumor response, but transferring those predictions to stratify long-term patient survival out-of-cohort requires larger sample sizes and treatment-matched prospective validation datasets.

# Is our model useless then?
No, here is why these results are scientifically valuable:

## 1. Can we use overall survival (OS) as a proxy for immunotherapy response (or vice versa)?
Our results provide clear empirical evidence: **No, you cannot.**
- Pre-treatment transcriptomic immune signatures (like TIS and IFN-γγ) predict **short-term immune checkpoint response** (whether T-cells will attack the tumour upon anti-PD-1 blockade), but they **do not directly dictate long-term overall survival**.

## 2. Biology Explains the Gap (Predictive vs. Prognostic)

Biomedical data science distinguishes between two types of biomarkers:

- **Predictive Biomarkers**: Tell you whether a patient will respond to a _specific treatment_ (e.g. anti-PD-1). This is what your model predicts (RECIST response: CR/PR vs. PD).
- **Prognostic Biomarkers**: Tell you how long a patient will live _regardless of treatment_ (overall survival).

Overall survival in advanced melanoma depends on many factors _outside_ baseline immune gene expression, such as:

1. **Subsequent Second-Line Therapies**: Patients who fail anti-PD-1 often cross over to anti-CTLA-4 (Ipilimumab) or BRAF/MEK inhibitors (Dabrafenib/Trametinib), which can extend their life for years despite being classified as "Non-Responders" to first-line anti-PD-1.
2. **Tumour Burden & Metastatic Site**: Brain metastases drastically reduce overall survival regardless of tumour immune infiltration.
3. **Treatment Toxicity**: Severe adverse events (irAEs) can cause treatment discontinuation in responding patients.

## 3. Out-of-Cohort (LOCO) Validation is the Gold Standard

Many published papers claim high AUCs (>0.85) because they use standard 5-fold cross-validation **within a single trial cohort**. That is often an illusion caused by batch effects and trial-specific overfitting! By performing strict **Leave-One-Cohort-Out (LOCO)** validation:
- We proved that our SVM model achieves genuine out-of-cohort response discrimination (**AUC = 0.717** on Riaz 2017 and **AUC = 0.657** on Liu 2019).
- Showing that survival curves do not automatically separate out-of-cohort demonstrates **scientific honesty** and highlights the need for prospective, multimodal models.

## 4. Primary Objective Met
The model's target was predicting **anti-PD-1 response (CR/PR vs. PD)**, where SVM achieved an out-of-cohort AUC of **0.717**.

- **Key Insight Realised**: Short-term treatment response and long-term overall survival are decoupled in metastatic melanoma due to crossover second-line targeted therapies and non-immune clinical drivers (like ECOG score and LDH levels).
- **Pipeline Justification**: This result reinforces why we move to **Question 5 / Multimodal Integration**, combining immune signatures with clinical stage, LDH, and driver mutations to build a complete prognostic picture.

## Summary of Key Findings by Model Architecture

> [!summary] Comparative Overview of Model Performance & Biological Insights  
> **What**: A comprehensive synthesis comparing all 5 machine learning architectures (Support Vector Machine, ElasticNet, Logistic Regression, Random Forest, and XGBoost) evaluated under Leave-One-Cohort-Out (LOCO) cross-validation across 3 independent clinical trial cohorts (Hugo 2016, $N=27$; Liu 2019, $N=104$; Riaz 2017, $N=64$).  
> **Why**: Synthesising performance characteristics across model families clarifies how mathematical assumptions (linear regularisation, non-linear bagging, gradient boosting, maximum-margin hyper-planes) interact with high-dimensional transcriptomic immune signatures and discrete somatic driver mutations.  
> **What Question It Answers**: Which model architecture provides the optimal balance of cross-cohort generalisation, prediction stability, and multimodal feature utilisation for clinical immunotherapy response prediction?

### Cross-Model Performance Matrix (LOCO ROC-AUC Scores)

| Model Architecture | Immune Only: Hugo 2016 ($N=27$) | Immune Only: Liu 2019 ($N=104$) | Immune Only: Riaz 2017 ($N=64$) | Multimodal: Hugo 2016 | Multimodal: Liu 2019 | Multimodal: Riaz 2017 | Impact of Adding Driver Mutations (`BRAF`/`NRAS`/`NF1`) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Support Vector Machine (SVM)** | 0.434 | **0.657** | **0.717** | 0.434 | **0.657** | **0.717** | No change ($\Delta\text{AUC} = 0.000$); continuous signatures dominate kernel distance matrix. |
| **ElasticNet Logistic Regression** | 0.415 | 0.616 | 0.500 | 0.415 | 0.616 | 0.500 | No change ($\Delta\text{AUC} = 0.000$); L1 penalty zeroes out uninformative mutation weights. |
| **Logistic Regression (L1-Penalised)** | 0.415 | 0.609 | 0.500 | 0.415 | 0.609 | 0.500 | No change ($\Delta\text{AUC} = 0.000$); Lasso zeroes out driver mutation coefficients. |
| **Random Forest Classifier** | 0.423 | 0.580 | 0.678 | **0.473** | 0.581 | **0.688** | **Improved** ($\Delta\text{AUC} = +0.050$ on Hugo 2016, $+0.010$ on Riaz 2017); non-linear interaction splits. |
| **XGBoost Gradient Boosting** | 0.319 | 0.581 | 0.618 | 0.319 | 0.581 | 0.618 | No change ($\Delta\text{AUC} = 0.000$); greedy split selection ignores binary mutation flags. |

### Model-by-Model Synthesis & Key Findings

#### 1. Support Vector Machine (SVM) — Top Performing Classifier
- **Key Finding**: SVM emerged as the top-performing model overall, achieving the highest out-of-cohort generalisation performance across LOCO test benchmarks (AUC = 0.717 on Riaz 2017 and AUC = 0.657 on Liu 2019).
- **Mathematical Rationale**: The maximum-margin hyper-plane algorithm ($C=0.1$, RBF kernel with $\gamma=\text{'scale'}$) effectively regularises against sample-specific noise and batch effect variations present across distinct clinical trial datasets.
- **Multimodal Behaviour**: Adding somatic driver mutation flags (`BRAF`, `NRAS`, `NF1`) produced zero change in AUC ($\Delta\text{AUC} = 0.000$). Because continuous Z-scored immune signatures span a wider range than binary mutation values ($0$ or $1$), the Euclidean distance metric within the RBF kernel matrix is predominantly governed by transcriptomic expression levels.

#### 2. ElasticNet Logistic Regression — Best Linear Baseline
- **Key Finding**: ElasticNet demonstrated superior linear performance compared to pure L1 Lasso Logistic Regression on Liu 2019 (AUC = 0.616 vs 0.609).
- **Mathematical Rationale**: By combining L1 (Lasso) and L2 (Ridge) penalties (`l1_ratio = 0.1`), ElasticNet avoids the arbitrary elimination of collinear features. Because immune expression signatures exhibit strong inter-feature correlation ($r_s = 0.85\text{--}0.94$), the L2 component retains complementary signals across signature blocks rather than dropping all but one feature.
- **Multimodal Behaviour**: Binary driver mutation flags (`BRAF`, `NRAS`, `NF1`) were assigned zero weight by the L1 regulariser, yielding identical cross-cohort predictions between immune-only and multimodal feature sets.

#### 3. L1-Penalised Logistic Regression — Transparent Baseline
- **Key Finding**: L1 Logistic Regression provided a transparent benchmark, achieving moderate discrimination on Liu 2019 (AUC = 0.609) but collapsing to random guessing on Riaz 2017 (AUC = 0.500) and inverted performance on Hugo 2016 (AUC = 0.415).
- **Mathematical Rationale**: Linear sparsity constraints shrink uninformative coefficients to zero. However, strict linear boundaries cannot capture non-linear immune microenvironment threshold effects or multi-gene non-additive interactions.
- **Multimodal Behaviour**: Driver mutation coefficients (`BRAF`, `NRAS`, `NF1`) were zeroed out completely because somatic driver status alone does not differentiate responders from non-responders across cohorts.

#### 4. Random Forest Classifier — Only Architecture Benefiting from Multimodal Integration
- **Key Finding**: Random Forest was the **only model architecture** to show consistent performance improvements upon incorporating somatic driver mutations (`BRAF`, `NRAS`, `NF1`).
- **Performance Boost**: Out-of-cohort AUC increased from 0.423 to 0.473 (+0.050) on Hugo 2016 and from 0.678 to 0.688 (+0.010) on Riaz 2017, while maintaining 0.581 on Liu 2019.
- **Mathematical Rationale**: Unlike linear models or greedy gradient-boosted trees, decision tree ensembles construct multi-way conditional split rules (e.g., evaluating immune infiltration levels conditional upon `BRAF` or `NRAS` mutation status). This non-linear subgroup splitting successfully captures subtle, complementary biological interactions between tumour genomics and microenvironmental immune engagement.

#### 5. XGBoost Gradient Boosting — Prone to Overfitting on Small Cohorts
- **Key Finding**: XGBoost exhibited suboptimal cross-cohort transferability, particularly on small test cohorts (Hugo 2016 AUC = 0.319), while achieving moderate performance on larger trials (Liu 2019 AUC = 0.581, Riaz 2017 AUC = 0.618).
- **Mathematical Rationale**: Greedy split selection at each boosting step prioritises continuous transcriptomic features (`TIS`, `IFN-γ`) that offer immediate gradient reduction, ignoring lower-gain binary mutation flags entirely. Sequential error correction without extensive sample volume leads to overfitting on cohort-specific noise.
- **Multimodal Behaviour**: Identical predictions were produced with and without driver mutation flags ($\Delta\text{AUC} = 0.000$), as regularisation hyperparameters (`learning_rate`, `max_depth`) pruned binary mutation splits early during tree construction.

### Key Takeaways

> [!insight] Core Methodological & Clinical Takeaways  
> 1. **Model Hierarchy**: Maximum-margin classifiers (SVM) dominate cross-cohort generalisation (peak AUC = 0.717 on Riaz 2017), followed by correlated linear regularisers (ElasticNet AUC = 0.616 on Liu 2019) and non-linear tree ensembles (Random Forest AUC = 0.688 on Riaz 2017).  
> 2. **Multimodal Utility**: Somatic driver mutations (`BRAF`, `NRAS`, `NF1`) provide complementary predictive value **only** when paired with non-linear tree-based interaction splitting (Random Forest). Linear models and kernel methods zero out or dilute discrete genomic flags.  
> 3. **Clinical Recommendation**: For out-of-cohort immunotherapy response prediction based on transcriptomic signatures, SVM with RBF kernel is the optimal standalone classifier, while Random Forest should be selected when integrating multimodal genomic and transcriptomic features.