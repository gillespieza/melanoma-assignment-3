---
title: Genomic Characteristics of Data Cohorts
aliases: 
tags:
  - comut
  - driver-mutations
  - genomics
  - melanoma
  - neoantigens
  - tmb
created: 2026-07-24 13:19
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-24 15:23
---

# Genomic Characteristics of Data Cohorts

This report presents a comparative analysis of genomic features across the four melanoma study cohorts:
* **TCGA-SKCM**: Baseline genomic reference population ($N = 428$).
* **Liu 2019**: Anti-PD-1 clinical trial cohort ($N = 122$).
* **Hugo 2016**: Anti-PD-1 clinical trial cohort ($N = 27$).
* **Riaz 2017**: Anti-PD-1 clinical trial cohort ($N = 107$).

## 1. Mutation Landscape Comparison

> [!summary] What, Why & Key Questions  
> - **What We Are Doing**: Comparing the mutation frequencies of key melanoma driver genes (`BRAF`, `NRAS`, `NF1`, and Triple-WT) and core immune pathways across all four study cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, and **Riaz 2017**).  
> - **Why We Are Doing It**: To confirm that our clinical trial datasets reflect real-world melanoma epidemiology and to assess whether baseline pre-treatment mutations in antigen presentation (_B2M_) or IFN-$\gamma$ signaling (_JAK1/2_) drive primary resistance.  
> - **Questions**:
> 	1. _Are our clinical trial cohorts representative of standard melanoma epidemiology? _
> 	2. _Do trial patients harbour baseline mutations in antigen presentation or interferon signaling pathways prior to therapy?_

The distribution of driver mutations (`BRAF`, `NRAS`, `NF1`) and extended biological pathway mutation frequencies across all cohorts are shown below.

![[genomic_driver_frequencies.png]]

![Extended Pathway Mutation Frequencies](../../plots/genomic/extended_pathway_mutation_frequencies.png)

### Key Observations
- **Matches Real-World Melanoma Genetics**: The reference TCGA-SKCM cohort matches expected real-world melanoma genetics (`BRAF`: **52.1%**, `NRAS`: **28.0%**, `NF1`: **17.1%**, Triple-WT: **15.2%**).
- **Trial Cohorts are Biologically Representative**: The immunotherapy trial datasets (Liu 2019, Hugo 2016, Riaz 2017) closely align with baseline TCGA frequencies, confirming that trial patients are representative of general melanoma populations.
- **Driver Mutations Are Mutually Exclusive**: Tumours with `BRAF` mutations almost never carry co-occurring `NRAS` mutations, confirming that these drivers act through independent, non-overlapping growth pathways.
- **Immune Evasion Mutations Are Absent Before Therapy**: Pre-treatment mutations in antigen presentation (_B2M_, _TAP1/2_) and interferon signaling (_JAK1/2_) are virtually absent (<2%) prior to treatment. This confirms that genetic loss of antigen presentation is an **acquired resistance mechanism** that develops _during_ therapy, rather than a common baseline cause of initial treatment failure.

## 2. Tumour Mutational Burden (TMB) & Neoantigen Load

> [!summary] What, Why & Key Questions  
> - **What We Are Doing**: Analysing the distribution of Tumour Mutational Burden (TMB) across immunotherapy response groups (Responders vs. Non-Responders) and evaluating the correlation between TMB and predicted neoantigen load ($N = 256$).  
> - **Why We Are Doing It**: Somatic mutations generate novel peptide antigens (neoantigens) that trigger T-cell recognition. We test whether TMB correlates with treatment response and whether TMB can serve as a proxy for mutational neoantigen load in downstream modeling.  
> - **Questions**:
> 	1. _Do responders exhibit higher baseline TMB than non-responders? _
> 	2. _Is total TMB collinear with predicted neoantigen count?_

Tumour Mutational Burden (TMB) and predicted Neoantigen Load are key genomic measures of tumour immunogenicity. Below, we present the TMB distribution by response (left panel) alongside the correlation scatter plot illustrating Neoantigen Collinearity with TMB in the pooled trial cohorts ($N = 256$, right panel).

![TMB Distributions](../../plots/genomic/tmb_distributions_by_cohort.png)

### Key Observations
- **Responders Have Higher Baseline TMB**: Across all three clinical trial cohorts, patients who responded to immunotherapy (CR/PR) had higher pre-treatment TMB levels than non-responders (PD).
- **TMB Strongly Predicts Neoantigen Count ($r = 0.756$)**: TMB and predicted neoantigen load show a strong linear relationship. Tumours with more mutations produce more neoantigens, confirming that TMB is a reliable surrogate marker for mutational neoantigen burden.
- **Redundancy for Machine Learning**: Because TMB and neoantigen load measure the same biological axis, predictive models should not include both features simultaneously to avoid collinearity and model coefficient instability.

## 3. Continuous Biomarker Correlation

> [!summary] What, Why & Key Questions  
> - **What We Are Doing**: Computing Spearman rank correlations between continuous genomic features (TMB, neoantigen subtypes, aneuploidy) and transcriptomic immune signatures across trial ($N = 256$) and reference ($N = 428$) cohorts.  
> - **Why We Are Doing It**: To identify feature redundancy before model training and evaluate whether genomic mutational burden and transcriptomic immune infiltration capture independent biological axes.  
> - **Questions**:
> 	1. _Are TMB and neoantigen subtypes redundant?_
> 	2. _Is mutational burden correlated with transcriptomic T-cell infiltration, or are they orthogonal biomarkers?_

### 3.1. Biomarker Correlation in Pooled Trials
A Spearman rank correlation matrix mapping the relationships between continuous genomic features (somatic mutation and neoantigen subtypes) across the **Pooled Trials** cohort ($N = 256$) is presented below.

![Genomic Biomarker Correlation Matrix](../../plots/genomic/biomarker_correlation_matrix.png)

### Key Observations
- **TMB & SNV Neoantigens Are Highly Redundant ($r_s = 0.87$)**: Total TMB and single-nucleotide variant (SNV) neoantigens show a very strong positive correlation ($r_s = 0.87$). Including both in a machine learning model adds minimal new information and can destabilise model coefficients.
- **Distinct Neoantigen Subtypes**: Indel neoantigens ($r_s = 0.44$ with TMB) and cancer-testis self-antigens ($r_s = 0.33$ with TMB) show weaker correlations, capturing distinct immunogenic signals.

### 3.2. Genomic Burden vs. Immune Infiltration
To evaluate how tumour genomic features affect the microenvironment, we evaluated how copy-number burden (**Aneuploidy Score**, available exclusively in TCGA-SKCM, $N = 428$) and mutational burden (**TMB**, evaluated in both TCGA-SKCM and pooled trials, $N = 256$) correlate with continuous transcriptomic immune signatures.

_Data Availability Note: Aneuploidy Score was measured via SNP arrays/WGS in TCGA-SKCM, but is unavailable in the three clinical trial cohorts because their sequencing protocols focused on exome/targeted SNV mutational profiling rather than genome-wide copy-number alterations._

![Genomic Burden vs Immune Heatmap](../../plots/biomarkers/extended_immune_correlations.png)

### Key Observations
- **TMB & Immune Infiltration Are Independent (Orthogonal)**: TMB shows near-zero correlation ($r \approx -0.09$ to $0.16$) with transcriptomic immune signatures (like IFN-$\gamma$ or TIS). A tumour can be highly mutated (high TMB) but immunologically "cold" (uninflamed), or low-TMB but "hot" (highly inflamed). This proves that TMB and immune inflammation capture **two independent biological axes**, meaning predictive models should combine both modalities to maximize accuracy.
- **Aneuploidy Score Is a Weak Indicator**: Chromosomal instability (Aneuploidy Score, TCGA-SKCM) shows weak negative correlations ($r \approx -0.05$ to $-0.11$) with immune signatures, demonstrating it is a poor standalone predictor of immune exclusion in melanoma.

## 4. TCGA Survival Stratification by Genomic Features

> [!summary] What, Why & Key Questions  
> - **What We Are Doing**: Stratifying Kaplan-Meier overall survival in the baseline **TCGA-SKCM** reference cohort ($N = 428$) by driver mutation status (`BRAF`, `NRAS`, `NF1`), TMB median split, and chromosomal Aneuploidy Score.  
> - **Why We Are Doing It**: To determine whether baseline genomic mutations and copy-number alterations act as general prognostic survival markers in untreated/standard-of-care melanoma.  
> - **Questions**:
> 	1. _Do driver mutations or TMB predict baseline overall survival in general melanoma populations, or are their predictive effects limited specifically to immunotherapy treatment response?_

Overall Survival (OS) in the reference **TCGA-SKCM** survival cohort ($N = 428$ with complete survival follow-up out of $N = 443$ total cleaned samples) is stratified below. The top panel shows stratification by driver mutation subtype and TMB status ($N = 428$). The bottom panel shows overall survival stratified by chromosomal instability (Aneuploidy Score median split, $N = 417$ with complete copy-number data).

![[tcga_survival_by_mutation.png]]

![[extended_aneuploidy_survival.png]]

### Key Observations
- **Driver Mutations Do Not Predict Baseline Survival ($p = 0.0519$)**: Overall survival in standard melanoma patients does not differ significantly between `BRAF`, `NRAS`, and `NF1` mutant genotypes. Driver mutations guide targeted drug selection, but do not dictate baseline patient survival under standard care.
- **TMB Is Predictive, Not Prognostic ($p = 0.4483$)**: High TMB does not improve overall survival in general melanoma patients. While TMB predicts response specifically under immunotherapy, it has no general prognostic survival benefit on its own.

## 5. Co-Mutation Landscape (Oncoplot)

> [!summary] What, Why & Key Questions  
> - **What We Are Doing**: Constructing a multi-track co-mutation oncoplot across **$N = 195$** immunotherapy trial patients with binary response labels (CR/PR or PD, excluding Stable Disease), mapping somatic mutations in driver and resistance genes alongside patient TMB, response status, trial cohort, and sex tracks.  
> - **Why We Are Doing It**: To visualize patient-level co-occurrence and mutual exclusivity patterns across driver mutations and resistance pathways simultaneously.  
> - **Questions**:
> 	1. _Are `BRAF` and `NRAS` driver mutations strictly mutually exclusive in trial patients?_
> 	2. _Are responders enriched in specific driver mutation subtypes or clinical tracks?_

The complete co-mutation (oncoplot) landscape for patients with binary response labels (CR/PR or PD) across all three clinical trial cohorts ($N = 195$; excludes $N = 61$ Stable Disease patients without a binary response classification) is presented below.

![Co-Mutation Landscape (Merged Trials)](../../plots/genomic/comut_landscape_merged.png)

### Key Observations
- **MAPK Driver Mutual Exclusivity**: `BRAF` and `NRAS` mutations almost never co-occur in the same patient, confirming they represent separate, mutually exclusive driver pathways.
- **Driver Mutations Do Not Dictate Response**: Treatment responders (CR/PR) are distributed evenly across all driver genotypes (`BRAF`, `NRAS`, `NF1`, and Triple-WT). This confirms visually that driver mutation status alone cannot be used to predict immunotherapy outcome.
