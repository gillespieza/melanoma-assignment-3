# TCGA Pan-Cancer Derived Prognostic Signature Report

We performed transcriptomic feature selection on the **TCGA-SKCM** cohort ($N = 428$ aligned samples with survival data) to build a custom overall survival signature, and subsequently validated it on three independent clinical trial cohorts.

## 1. Top 20 Prognostic Genes in TCGA-SKCM
The 20 genes most significantly associated with overall survival in univariate Cox regression are visualised below. A positive Beta indicates a **risk-associated gene** (higher expression = worse survival), while a negative Beta indicates a **protective gene** (higher expression = better survival).

### Hazard Ratio Forest Plot (Top 20 Genes)
The forest plot below visualises the Hazard Ratios (HR) and their 95% confidence intervals for the top 20 most significant prognostic transcripts. Protective genes (HR < 1.0) are shown in blue, and risk-associated genes (HR > 1.0) are shown in red:

![Prognostic Gene Forest Plot](../../plots/feature_selection/transcriptomic_forest_plot.png)

## 2. Kaplan-Meier Survival Curve on TCGA
We partitioned TCGA-SKCM patients into High-Risk and Low-Risk groups using the median value of the signature score. The log-rank test indicates an extremely significant separation in survival curves:

*   **Log-Rank p-value**: **1.57e-08**

![KM Curve of TCGA Survival](../../plots/feature_selection/km_pancancer_signature.png)

## 3. Validation on Immunotherapy Clinical Trial Cohorts
We evaluated the custom 20-gene prognostic signature on three cohorts receiving anti-PD-1 or combination immunotherapies to see if the overall survival signature translates into predicting immunotherapy response.

| Cohort | N | Aligned Signature Genes | Response ROC AUC | Mann-Whitney U p-value | Mean Risk (Responders) | Mean Risk (Non-Responders) |
|---|---|---|---|---|---|---|
| Liu 2019 | 104 | 20/20 | **0.527** | 6.36e-01 | -18.067 | -18.224 |
| Hugo 2016 | 26 | 20/20 | **0.417** | 4.87e-01 | -18.842 | -19.102 |
| Riaz 2017 | 33 | 20/20 | **0.674** | 1.22e-01 | -19.225 | -18.652 |

### Validation Visualisations
#### ROC Curves Predicting Response
![ROC Curves for Response](../../plots/feature_selection/pancancer_signature_trial_validation.png)

#### Signature Risk Score Stratified by Responders vs. Non-Responders
![Signature Violin Plots](../../plots/feature_selection/pancancer_signature_violins.png)

## 4. Part 3: Synthesis & Pipeline Recommendation

> [!summary] What, Why & Key Questions
> **What**: We evaluated whether a custom 20-gene overall survival score built from TCGA-SKCM data should be added to our final prediction model alongside our 6 established immune signatures.
> **Why**: We must ensure every feature added to our machine learning model provides unique, genuine predictive value rather than repeating information or adding random noise.
> **Key Finding**: We decided **not** to include the TCGA survival score in our final model because it measures the exact same underlying immune signal as our existing signatures and fails to predict treatment response in real-world patient trials.

1. **Why Purely Data-Driven Gene Selection Fails**:
   Selecting genes purely by statistical algorithms (`SelectKBest`) picks up trial-specific noise (like nerve or stomach genes) that do not generalise to new patients. In contrast, using established, biologically curated immune signatures (like IFN-γ and TIS) condenses over 20,000 raw genes into 6 meaningful, reliable biological scores ($D=6$).

2. **Why We Do Not Need the TCGA Survival Score**:
   - **Biological Overlap**: Even though the 20 genes in the TCGA survival score are different names from the genes in our 6 immune signatures (0/20 literal gene overlap), they perform the exact same job. They are all helper genes turned on by interferon to activate the immune system.
   - **Strong Correlation**: Measuring the TCGA score against our immune signatures gives near-identical results across all patient datasets ($r_s = -0.83 \text{ to } -0.94$ with TIS and IFN-γ). Adding it would simply count the same biological signal twice.
   - **Poor Response Prediction**: When tested on actual immunotherapy patient trials, the TCGA survival score could not reliably tell responders from non-responders in any cohort (**Liu 2019**: $\text{AUC} = 0.527, p = 0.636$; **Hugo 2016**: $\text{AUC} = 0.417, p = 0.487$; **Riaz 2017 Pre-treatment**: $\text{AUC} = 0.674, p = 0.122$). General patient survival in TCGA does not equal immunotherapy success.

> [!insight] Final Pipeline Architecture Decision
> Because the TCGA survival score is biologically redundant and fails to predict treatment response out-of-cohort, we **exclude** it from our final machine learning pipeline. We use only the **6 curated functional immune signatures** (IFN-γ, TIS, CYT, CD8 T-cell, IMPRES, PD-L1) for transcriptomic features ($D=6$). Combined with our clinical and genomic variables, this builds our final 14-feature multimodal matrix ($D=14$).
