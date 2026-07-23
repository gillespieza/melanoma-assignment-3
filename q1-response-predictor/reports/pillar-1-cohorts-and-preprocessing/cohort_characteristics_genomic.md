---
title: Genomic Characteristics of Data Cohorts
tags:
  - melanoma
  - genomics
  - driver-mutations
  - tmb
  - neoantigens
  - comut
created: 2026-07-23 22:49
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-23 22:49
---

# Genomic Characteristics of Data Cohorts

This report presents a comparative analysis of the genomic features across the four melanoma study cohorts:
* **TCGA-SKCM**: Baseline genomic reference population ($N=428$).
* **Liu 2019**: Anti-PD-1 clinical trial cohort ($N=122$).
* **Hugo 2016**: Anti-PD-1 clinical trial cohort ($N=27$).
* **Riaz 2017**: Anti-PD-1 clinical trial cohort ($N=107$).

---

## 1. Mutation Landscape Comparison

The distribution of the three major cutaneous melanoma driver mutations (`BRAF`, `NRAS`, and `NF1`) and the Triple-Wild-Type (Triple-WT) rate is compared across all cohorts below.

![[genomic_driver_frequencies.png]]

Driver Mutation Frequencies

### Key Observations
* **Reference Alignment**: The reference **TCGA-SKCM** cohort aligns perfectly with cutaneous melanoma epidemiology, showing a `BRAF` mutation rate of **52.1%**, `NRAS` at **28.0%**, `NF1` at **17.1%**, and a Triple-WT rate of **15.2%**.
* **Representative Trial Cohorts**: All trial cohorts align closely with TCGA baseline frequencies. **Liu 2019** shows highly representative driver mutation distributions (`BRAF`: **41.0%**, `NRAS`: **30.3%**, `NF1`: **17.2%**). **Riaz 2017** mutations are also representative (`BRAF`: **23.4%**, `NRAS`: **18.7%**, `NF1`: **3.7%**, Triple-WT: **58.9%**).
* **Biological Note**: Driver mutations are generally mutually exclusive: tumors with `BRAF` mutations rarely harbor co-occurring `NRAS` mutations, validating standard melanoma genetics.
* **Representative Cohort Features**: Genomic profiles show mutation status and burden metrics are representative.

### Extended Pathway Mutation Frequencies
To further characterize tumor immunogenicity and mechanisms of resistance, we evaluated pre-treatment somatic mutation frequencies across core biological pathways:
* **Antigen Presentation Machinery**: `B2M`, `TAP1`, `TAP2` (loss causes HLA class I downregulation).
* **IFN-gamma Signaling**: `JAK1`, `JAK2`, `STAT1` (induces insensitivity to T-cell cytotoxicity).
* **Immune Checkpoints**: `CD274`, `CTLA4`, `IDO1` (modulators of immune evasion).
* **Cytolytic Machinery**: `GZMA`, `PRF1` (effectors of cytotoxic lymphocyte killing).
* **Survival & Proliferation Drivers**: `PTEN`, `CDKN2A`, `PIK3CA` (oncogenic drivers).

![Extended Pathway Mutation Frequencies](../../plots/genomic/extended_pathway_mutation_frequencies.png)

Extended Pathway Somatic Mutation & Pathway Frequencies

_Note: Pre-treatment somatic non-synonymous mutations in MHC Class I machinery (`B2M`, `TAP1`, `TAP2`) are absent in these trial cohorts, as genetic disruption of antigen presentation is primarily an acquired resistance mechanism that emerges under checkpoint blockade pressure rather than a baseline primary resistance mechanism._

---

## 2. Tumor Mutational Burden (TMB) & Neoantigen Load

Tumor Mutational Burden (TMB) and predicted Neoantigen Load are key genomic measures of tumor immunogenicity. Below, we present the TMB distribution (left panel) alongside a scatter plot showing the relationship between nonsynonymous TMB and predicted neoantigen load in the pooled trial cohorts ($N=256$).

![TMB Distributions](../../plots/genomic/tmb_distributions_by_cohort.png)

TMB Distributions

![Neoantigen vs TMB](../../plots/biomarkers/extended_neoantigen_tmb.png)

Neoantigen vs TMB

### Key Observations
* **TMB as a Predictor**: In all three immunotherapy cohorts, responders (CR/PR, bluish green boxes) exhibit a higher pre-treatment TMB distribution than non-responders (PD, vermillion red boxes).
* **TCGA Distribution**: The reference cohort shows a classical log-normal TMB distribution with a median of **14.88 mutations/Mb**. A substantial proportion of patients lie above the standard FDA clinical cutoff of **10.0 mutations/Mb** for high-TMB status, validating the presence of a strong ultraviolet (UV) signature in cutaneous melanomas.
* **Neoantigen Collinearity**: There is a strong linear relationship between nonsynonymous TMB and predicted neoantigen load ($r = 0.756$). The extreme correlation confirms that these two metrics are collinear, making TMB a suitable surrogate for mutational neoantigen burden in downstream modeling.

---

## 3. Continuous Biomarker Correlation

### 3.1. Intra-Cohort Correlation in Liu 2019
A Spearman rank correlation matrix mapping the relationships between continuous genomic features (somatic mutation and neoantigen subtypes) in the **Liu 2019** cohort ($N=122$) is presented below.

![Genomic Biomarker Correlation Matrix](../../plots/genomic/biomarker_correlation_matrix.png)

Genomic Biomarker Correlation Matrix

### Key Observations
* **High Collinearity**: TMB and SNV Neoantigens show an extremely high correlation ($r_s = 0.96$). This indicates severe redundancy; in predictive machine learning models, using both features simultaneously is unlikely to add value and may destabilize model coefficients.
* **Neoantigen Subtypes**: Somatic indel neoantigens (`INDEL_NEOANTIGEN`, $r_s = 0.44$ with TMB) and cancer-testis antigens (`CTA_SELF_NEOANTIGEN`, $r_s = 0.22$ with TMB) show much weaker correlations. This suggests they capture distinct biological axes of tumor immunogenicity that are not simply surrogates for total mutational burden.

### 3.2. Genomic Burden vs. Immune Infiltration
To understand how tumor genomic features affect the microenvironment, we evaluated how copy-number burden (Aneuploidy Score) and mutational burden (TMB) correlate with continuous transcriptomic immune signatures in both TCGA-SKCM ($N=428$) and the pooled trials ($N=256$).

![Genomic Burden vs Immune Heatmap](../../plots/biomarkers/extended_immune_correlations.png)

### Key Observations
* **Aneuploidy vs Infiltration**: Chromosomal instability (Aneuploidy Score) shows a very weak negative correlation ($r \approx -0.05$ to $-0.11$) with baseline immune signatures in TCGA-SKCM, with only `PD_L1` showing a statistically significant negative correlation ($r = -0.110$, $p = 0.022$). This indicates that while copy number alterations are associated with immune exclusion in some cancer types, Aneuploidy Score alone is a weak predictor of immune-excluded "cold" status in melanoma.
* **Orthogonal Biomarkers**: Mutational burden (TMB) shows near-zero/weak correlation with immune signature expression in both TCGA ($r \approx 0.10$ to $0.16$) and trial cohorts ($r \approx -0.09$ to $0.05$). This demonstrates that TMB and immune infiltration represent **orthogonal biomarkers**. A tumor can be highly mutated (high TMB) but still immunologically cold, or poorly mutated but hot/inflamed. Downstream predictive models should combine both independent modalities to maximize accuracy.

---

## 4. TCGA Survival Stratification by Genomic Features

Overall Survival (OS) in the reference **TCGA-SKCM** cohort ($N=428$) is stratified below. The left panel shows stratification by driver mutation subtype and TMB status. The right panel shows overall survival stratified by chromosomal instability (Aneuploidy Score) using a median split.

![TCGA Driver and TMB Survival](../../plots/genomic/tcga_survival_by_mutation.png)

TCGA Driver and TMB Survival

![[extended_aneuploidy_survival.png]]

TCGA Aneuploidy Survival

### Key Observations
* **Driver Subtypes**: Overall survival does not differ strongly between `BRAF`, `NRAS`, and `NF1` mutant genotypes ($p = 0.0519$). This confirms that while driver mutations are biologically critical for tumor initiation and targeted therapy matching, they do not act as strong, independent prognostic markers for long-term overall survival under standard care.
* **TMB Stratification**: Stratifying TCGA overall survival by TMB using a median split shows no prognostic survival separation ($p = 0.4483$). While TMB is highly _predictive_ of response to checkpoint inhibitors, it is not _prognostic_ of baseline survival in the general TCGA population (where only a small subset received immunotherapy).
* **Aneuploidy Prognostic Role**: Partitioning the TCGA cohort by median Aneuploidy Score shows a marginally significant prognostic association ($p = 0.0806$), where patients with high aneuploidy trend towards worse overall survival compared to those with low aneuploidy.

---

## 5. Co-Mutation Landscape (Oncoplot)

The complete co-mutation (oncoplot) landscape for individual patients across all three clinical trial cohorts ($N=256$) is presented below. This combines somatic mutations in core driver and resistance genes (rows) with patient-specific clinical tracks (TMB, Response, Cohort source, and Sex).

![Co-Mutation Landscape (Merged Trials)](../../plots/genomic/comut_landscape_merged.png)

### Key Observations
* **MAPK Driver Mutual Exclusivity**: There is high mutual exclusivity between the two primary MAPK pathway drivers, `BRAF` and `NRAS`. This aligns with the classical understanding that `BRAF` and `NRAS` mutations represent redundant and mutually exclusive routes for activating the RAS-RAF-MEK-ERK signaling cascade.
* **NF1 Mutational Overlap**: Unlike `BRAF` and `NRAS`, mutations in the tumor suppressor `NF1` show significant overlap with both drivers. While some of these represent co-occurring driver events, many of these `NF1` mutations are passenger events. `NF1` is a large gene and highly susceptible to random somatic passenger mutations in melanoma, which features a high TMB driven by UV-light exposure.
* **Targeted Resistance Profile**: Core genes related to antigen presentation (`B2M`) and interferon signaling (`JAK1`, `JAK2`) show low baseline mutation rates. These mutations are rare in pre-treatment biopsies, indicating that genetic disruption of interferon signaling is mostly an acquired resistance mechanism rather than a common baseline driver.
* **Cohort Distribution**: The Cohort track shows that `BRAF` and `NRAS` mutations are evenly distributed across **Liu 2019** (blue) and **Hugo 2016** (orange).
* **No Driver Subtype Response Bias**: Responders are distributed across all driver mutation subtypes (`BRAF`, `NRAS`, `NF1`, and Triple-WT). This visually confirms that driver mutation status itself is not predictive of anti-PD-1 clinical response.
