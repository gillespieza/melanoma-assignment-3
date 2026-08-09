---
title: "Model C-Index Evaluation Table"  
aliases: Model C-Index Summary
tags:
  - c-index
  - model-evaluation
  - q1
created: 2026-08-04 10:41  
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-09 20:33
---

 **C-Index (Concordance Index)** breakdown for each model architecture evaluated under Leave-One-Cohort-Out (LOCO) cross-validation in `q1-response-predictor`:

# Cross-Model Concordance Index (C-Index) Summary

| Model Architecture                 | Hugo 2016 (N=27N=27) | Liu 2019 (N=104N=104) | Riaz 2017 (N=64N=64) | **Mean C-Index** |
| ---------------------------------- | -------------------- | --------------------- | -------------------- | ---------------- |
| **XGBoost Gradient Boosting**      | 0.607                | 0.457                 | 0.524                | **0.529**        |
| **ElasticNet Logistic Regression** | 0.612                | 0.416                 | 0.500                | **0.509**        |
| **Support Vector Machine (SVM)**   | 0.582                | 0.421                 | 0.522                | **0.508**        |
|**Logistic Regression (L1-Penalised)**|0.612|0.411|0.500|**0.508**|
|**Random Forest Classifier**|0.531|0.451|0.481|**0.488**|