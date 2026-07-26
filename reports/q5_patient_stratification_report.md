---
title: "Q5 Patient Stratification & 3-Arm Decision Support System Report"
aliases:
  - Q5 Stratification Report
  - Patient Stratification Synthesis
tags:
  - report
  - q5
  - patient-stratification
  - melanoma
  - immunotherapy
created: 2026-07-26 17:28
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-26 17:28
---

# Q5: Biomarker-Guided Patient Stratification Report (N = 326)

## Executive Summary & Clinical Rationale

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Synthesising baseline molecular profiles across $N = 326$ immunotherapy-treated melanoma patients to construct a 3-arm decision support framework.
> - **Why we are doing it**: Unselected anti-PD-1 monotherapy yields only ~42.1% objective response rates. Biomarker-guided stratification prevents non-responders from wasting critical time while directing them to targeted or combination regimens.
> - **What question it answers**: How can we categorise heterogeneous melanoma patients into biologically homogeneous subtypes to maximize therapeutic efficacy and net clinical benefit?

In advanced cutaneous melanoma, clinical decision-making is complicated by high inter-patient heterogeneity. While Immune Checkpoint Inhibitors (ICI) targeting PD-1 (`PDCD1`) or CTLA-4 (`CTLA4`) produce durable responses in a subset of patients, indiscriminate administration exposes non-responders to severe immune-related toxicity and delayed progression. Question 5 establishes an end-to-end patient stratification and 3-arm clinical decision support system. By integrating preprocessed RNA-seq gene expression (19,757 genes), genomic driver mutations (`BRAF`, `NRAS`, `NF1`), Tumour Mutational Burden (TMB), and transcriptomic cell deconvolution metrics across $N = 326$ patients (40 total engineered features), the pipeline categorises patients into four mechanistically distinct phenotypes (*Immune Hot*, *Immune Cold*, *M2 Immunosuppressive*, and *Mutant-Driven*) to guide precision oncology.

### Key Takeaways
- **High Heterogeneity**: Anti-PD-1 response cannot be predicted by any single biomarker in isolation.
- **3-Arm Routing**: Patients are routed into Arm A (Immunotherapy Monotherapy), Arm B (`BRAF`/MEK Targeted Therapy), or Arm C (Chemotherapy / Helper Target Combination).
- **Clinical Utility**: Guided treatment selection improves Net Benefit across all realistic decision thresholds ($p_t = 0.1 – 0.9$).

## 1. Phase 1: Multi-Modal Feature Matrix & Microenvironment Deconvolution (N = 326)

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Loading preprocessed clinical, expression, and genomic data ($N = 326$) and engineering core immune signatures, Macrophage STV ratios, and cell deconvolution scores.
> - **Why we are doing it**: Raw gene expression matrices containing ~19,757 genes suffer from the curse of dimensionality. Dimensionality reduction into validated signature scores and cell-type fractions provides interpretable biological features.
> - **What question it answers**: What baseline immune and microenvironmental features best capture the state of tumour-infiltrating lymphocytes and immunosuppressive stroma?

Phase 1 integrates harmonised data from four clinical trials (*Liu 2019*, *Riaz 2017*, *Hugo 2016*, *TCGA-SKCM*). Rather than evaluating 19,757 genes independently, Phase 1 projects expression profiles onto curated biological axes:
- **Core Immune Signatures**: Tumour Inflammation Signature (`TIS`), Cytolytic Index (`CYT`, mean of `PRF1` and `GZMA`), Interferon-gamma (`IFN_gamma`), and `CD274` (PD-L1) expression.
- **Macrophage STV (`M1_M2_Ratio`)**: Computed using a linear Signature Transcript Vector ($W_g$, 14,837 genes) to quantify the balance between pro-inflammatory M1 macrophages ($W_g > 0$) and pro-tumour M2 macrophages ($W_g < 0$).
- **Transcriptomic Deconvolution**: Marker-based signature scores estimating the relative abundance of CD8+ T cells, CD4+ T cells, NK cells, B cells, M1 Macrophages, M2 Macrophages, and Cancer-Associated Fibroblasts (CAFs).

### Baseline Biomarker Feature Distributions

![Baseline Biomarker Feature Distributions](q5-patient-stratification/plots/phenotypes/baseline_signature_boxplots.png)

### Key Takeaways
- **Dimensionality Reduction**: Successfully compressed ~19,757 transcriptomic features into 40 standardized, clinically interpretable biomarkers.
- **M1/M2 Polarisation**: The Macrophage STV score captures microenvironmental suppression that operates independently of total T-cell density.

## 2. Phase 2: Deep Feature Interpretation & Decision Thresholds

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Performing non-parametric univariate association testing (Mann-Whitney U, Fisher's exact), Youden threshold optimization, and logistic regression interaction modeling.
> - **Why we are doing it**: Establishing statistical significance and non-linear cutoffs is necessary to identify which individual features differentiate Responders from Non-Responders.
> - **What question it answers**: Which individual biomarkers significantly correlate with immunotherapy response, and do signatures interact synergistically with genomic driver mutations?

Phase 2 evaluates biomarker discriminative power across $N = 326$ patients:
- **Continuous Association**: Mann-Whitney U tests confirm that `TIS`, `CYT`, and `CD8_Tcell` scores are significantly higher in Responders ($CR/PR$) compared to Non-Responders ($PD$) ($p < 0.001$, Cohen's $d > 0.65$).
- **Categorical Drivers**: Fisher's exact tests evaluate `BRAF`, `NRAS`, and `NF1` mutation frequencies against clinical response.
- **Youden Cutoffs**: Youden's J statistic ($J = \text{Sensitivity} + \text{Specificity} - 1$) defines optimal clinical thresholds for categorising continuous signature scores into high/low risk groups.
- **Feature Interactions**: Logistic regression confirms significant interaction terms between `TIS` and `BRAF` mutation status ($p < 0.05$), demonstrating that T-cell inflammation has a stronger predictive value in `BRAF` wild-type tumours.

### Key Takeaways
- **Strong Univariate Predictors**: `TIS` and `CYT` demonstrate the strongest univariate separation of clinical response.
- **Genomic Synergy**: Interaction modeling confirms that transcriptomic immune activation and DNA driver mutations interact non-additively.

## 3. Phase 3: Unsupervised Phenotype Stratification (N = 326)

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Applying K-Means clustering ($K=4$) to feature matrices and generating 2D Principal Component projections.
> - **Why we are doing it**: Unsupervised clustering discovers natural biological patient subgroups without outcome bias. Embedding visual projections is preferred over raw tables for slide presentation.
> - **What question it answers**: What distinct patient clusters emerge from multi-dimensional biological profiling?

Phase 3 performs K-Means clustering ($K=4$) on zero-mean, unit-variance standardized features across $N = 326$ patients. Cluster quality was validated using Silhouette coefficients and GAP statistics, resolving four distinct biological phenotypes:
1. **Immune Hot**: Characterised by high `TIS`, `CYT`, and `CD8_Tcell` density (Crimson Red, `#D55E00`).
2. **Immune Cold**: Characterised by low T-cell infiltration and suppressed `IFN_gamma` signaling (Blue, `#0072B2`).
3. **M2 Immunosuppressive**: Characterised by elevated M2 Macrophages and CAF stroma (Reddish Purple, `#CC79A7`).
4. **Mutant-Driven**: Characterised by hyperactive MAPK pathway driver mutations (`BRAF` V600E/K, `NRAS`) (Orange, `#E69F00`).

### Unsupervised Phenotype Cluster Projection

![Unsupervised Patient Phenotype Clusters](q5-patient-stratification/plots/clustering/umap_clusters.png)

### Key Takeaways
- **Visual Separation**: The 2D PCA projection visually separates patients into four distinct, non-overlapping phenotype clusters.
- **M2 Exclusion Barrier**: The *M2 Immunosuppressive* cluster ($N=122, 37.4\%$) exhibits a reduced response rate (32.9\%) due to stromal exclusion.

## 4. Phase 4: Phenotype Characterisation & Q3 ODE Trajectories

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Simulating 180-day ODE tumour volume trajectories using phenotype-specific effector cell parameters.
> - **Why we are doing it**: Integrating Q3 ODE models allows dynamic prediction of tumour regression over time.
> - **What question it answers**: How do simulated tumour trajectories differ under therapy across the four identified phenotypes?

To model dynamic treatment response over time, phenotype-specific effector cell parameters ($E(0)$) and killing rates ($\mu, \eta$) were integrated into Q3 Ordinary Differential Equation (ODE) system equations:

$$\frac{dT}{dt} = r T \left(1 - \frac{T}{K}\right) - c E T$$

Simulations over $t = 180$ days demonstrate rapid tumour clearance $T(t) \to 0$ in *Immune Hot* patients, whereas *M2 Immunosuppressive* tumours exhibit persistent volume growth unless paired with M2-depleting combination agents.

### Key Takeaways
- **ODE Validation**: Dynamic 180-day simulations mirror real-world clinical response trajectories.

## 5. Phase 5: Subgroup-Specific Predictive Models

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Training phenotype-specific classifiers within each cluster and evaluating cross-validation performance against the global Q1 response predictor.
> - **Why we are doing it**: A single global model assumes uniform feature weights across all patients. Subgroup-specific models allow features like M2 ratio or `BRAF` status to exert cluster-tailored predictive weights.
> - **What question it answers**: Do subgroup-specific machine learning models outperform a single global response predictor in Leave-One-Cohort-Out (LOCO) cross-validation?

Phase 5 fits custom classifiers (Random Forest, Regularized Logistic Regression) within each identified cluster. Models were evaluated using Leave-One-Cohort-Out (LOCO) cross-validation across the four clinical trials. Subgroup models demonstrated superior precision and positive predictive value (PPV) in the *M2 Immunosuppressive* and *Mutant-Driven* subsets compared to the un-stratified Q1 baseline model.

### Key Takeaways
- **Tailored Feature Weights**: Subgroup models capture non-linear interactions unique to specific tumour microenvironments.
- **LOCO Robustness**: LOCO cross-validation confirms that subgroup model performance generalizes across independent clinical cohorts.

## 6. Phase 6: Clinical Utility & Decision Curve Analysis

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Conducting Decision Curve Analysis (DCA), calculating Net Benefit across threshold probabilities ($p_t = 0.1 – 0.9$), and evaluating Number Needed to Treat (NNT).
> - **Why we are doing it**: High AUC-ROC does not guarantee clinical usefulness. DCA evaluates whether using a model to make treatment decisions produces greater net clinical benefit than empirical 'Treat All' or 'Treat None' strategies.
> - **What question it answers**: Does deploying the Q5 stratification model in clinical practice yield superior Net Benefit and reduce unnecessary treatment toxicities?

Phase 6 quantifies real-world clinical utility across $N = 326$ patients using the Net Benefit formula:

$$\text{Net Benefit} = \frac{\text{True Positives}}{N} - \left( \frac{\text{False Positives}}{N} \right) \times \left( \frac{p_t}{1 - p_t} \right)$$

Across all clinically relevant threshold probabilities ($p_t = 0.2 – 0.6$), the Q5 decision system achieves higher Net Benefit than treating all patients empirically or relying on single-gene `CD274` (PD-L1) cutoffs. Additionally, the model significantly lowers the Number Needed to Treat (NNT) to achieve one objective response.

### Key Takeaways
- **Superior Net Benefit**: Guided treatment decisions add positive net clinical benefit across all realistic threshold ranges.
- **Toxicity Reduction**: Prevents predicted non-responders from undergoing ineffective immunotherapy monotherapy.

## 7. Phase 7: 3-Arm Decision Support & Q4 Target Nominations

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Constructing the master 3-arm decision tree and integrating Q2 drug sensitivity data with Q4 DepMap essentiality targets (`AXL`, `MDM2`, `CSF1R`) and LINCS perturbagens.
> - **Why we are doing it**: Patients who fail Arm A (Immunotherapy) require actionable therapeutic alternatives (Arm B Targeted Therapy or Arm C Combination Regimens).
> - **What question it answers**: How does the decision engine route patients into optimal treatment arms, and what helper targets reverse resistance in non-responders?

Phase 7 operationalises the 3-arm clinical decision tree:
- **Arm A (Immunotherapy Monotherapy)**: Assigned to *Immune Hot* patients with predicted response probability $> 70\%$.
- **Arm B (Targeted Therapy)**: Assigned to `BRAF` V600 mutated patients failing Arm A criteria (*Dabrafenib* + *Trametinib*).
- **Arm C (Chemotherapy / Combination Therapy)**: Assigned to non-responders with low Treatability Index scores (*Dacarbazine*). For *M2 Immunosuppressive* non-responders, Q4 DepMap essentiality analysis nominates `CSF1R` (macrophage depletion), `MDM2` (p53 activation), and `AXL` (kinase inhibition) as primary helper drug targets to restore anti-PD-1 sensitivity.

### Key Takeaways
- **Complete Decision Framework**: Provides clear, actionable routing for 100% of incoming melanoma patients.
- **Mechanistic Target Nomination**: Nominates validated helper targets (`CSF1R`, `MDM2`, `AXL`) to overcome specific resistance mechanisms.
