---
title:
aliases: 
tags: 
created: 2026-08-03 19:42
updated: 2026-08-03 20:14
---

# 1
<span style="font-size: 23px">This chapter asks a simple but important question: can we use machine learning to predict which melanoma patients benefit from which treatment, and do those predictions reflect the underlying biology of treatment response?​</span>

# 2
<span style="font-size: 23px">The reason this matters becomes clear in the clinic. Anti-PD-1 checkpoint inhibitors can be life-changing for some patients, but nearly 60% of metastatic melanoma patients don't respond.</span>

<span style="font-size: 23px">Non-responders have a median overall survival of just 9.2 months. Meanwhile, responders have dramatically improved survival, with a 95% reduction in mortality risk and a 77% gap in 2-year survival compared with non-responders.</span>

<span style="font-size: 23px">Predicting response before treatment could help patients avoid ineffective therapies and ensure they receive the most appropriate treatment earlier.</span>

# 3
<span style="font-size: 23px">We compared two approaches: letting the data select features automatically, versus using biologically informed features.</span>

<span style="font-size: 23px">Even after batch correction and fold-wise selection, SelectKBest identified individual genes that were statistically predictive but had no clear melanoma or immune relevance.</span>

<span style="font-size: 23px">Instead, we reduced the data to a 12-feature biological profile combining immune signatures, tumour mutational burden, macrophage activity, and key melanoma driver mutations.</span>

<span style="font-size: 23px">These curated features improved performance for SVM, Random Forest, and XGBoost, with SVM achieving the best cross validation AUC. But, regularised linear models performed similarly with both approaches, suggesting that biological knowledge is most valuable when capturing complex non-linear patterns.</span>

# 4
<span style="font-size: 23px">To assess real-world deployment, we tested our 5 models using Leave-One-Cohort-Out (LOCO) validation, where an entire clinical trial cohort is held out during training. Unlike standard 5-fold cross-validation, LOCO tests how well a model transfers to a completely new clinical setting.</span>​

​<span style="font-size: 23px">As expected, performance decreased, reflecting differences between clinical centres and patient populations.​</span>

<span style="font-size: 23px"> SVM achieved the strongest out-of-cohort result on the Riaz data (AUC 0.685) while ElasticNet showed the most consistent performance across unseen cohorts</span>​

<span style="font-size: 23px">Small cohorts such as Hugo introduced a lot of variability​</span>

# 5  
<span style="font-size: 23px">This table brings together the key benchmarking results.. Looking beyond predictive performance alone, we considered both accuracy and clinical interpretability.  </span>​

<span style="font-size: 23px">SVM achieved the strongest predictive performance, benefiting from its ability to capture complex patterns in the feature space. But, ElasticNet provided the best balance between performance, interpretability, and stability across unseen cohorts, making it the more practical candidate for clinical translation and ethics committee approvals.</span>​

<span style="font-size: 23px">I’ll now hand over to Gift to discuss her drug-response analysis.</span>​

 ​.​

 ​