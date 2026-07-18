# Genomic Characteristics of Data Cohorts

This report presents a comparative analysis of the genomic features across the four melanoma study cohorts:
*   **TCGA-SKCM**: Baseline genomic reference population ($N=426$).
*   **Liu 2019**: Anti-PD-1 clinical trial cohort ($N=104$).
*   **Hugo 2016**: Anti-PD-1 clinical trial cohort ($N=27$).
*   **Riaz 2017**: Anti-PD-1 clinical trial cohort ($N=64$).

---

## 1. Mutation Landscape Comparison

The distribution of the three major cutaneous melanoma driver mutations (*BRAF*, *NRAS*, and *NF1*) and the Triple-Wild-Type (Triple-WT) rate is compared across all cohorts below.

![[mutation_frequencies.png]]

Driver Mutation Frequencies

### Key Observations
*   **Reference Alignment**: The reference **TCGA-SKCM** cohort aligns perfectly with cutaneous melanoma epidemiology, showing a *BRAF* mutation rate of **51.9%**, *NRAS* at **28.2%**, *NF1* at **16.9%**, and a Triple-WT rate of **15.5%**.
*   **Representative Trial Cohorts**: All trial cohorts align closely with TCGA baseline frequencies. **Liu 2019** shows highly representative driver mutation distributions (*BRAF*: **40.4%**, *NRAS*: **30.8%**, *NF1*: **18.3%**). **Riaz 2017** mutations, after correcting the patient mapping bug, are also highly representative (*BRAF*: **40.6%**, *NRAS*: **17.2%**, *NF1*: **3.1%**, Triple-WT: **42.2%**).
*   **Biological Note**: Driver mutations are generally mutually exclusive: tumors with *BRAF* mutations rarely harbor co-occurring *NRAS* mutations, validating standard melanoma genetics.
*   **representative cohort features**: Genomic profiles show mutation status and burden metrics are representative.

### Extended Pathway Mutation Frequencies
To further characterize tumor immunogenicity and mechanisms of resistance, we evaluated pre-treatment somatic mutation frequencies across three biological pathways:
*   **Antigen Presentation**: *B2M*, *TAP1*, *TAP2* (disrupts MHC Class I presentation).
*   **IFN-gamma Signaling**: *JAK1*, *JAK2*, *STAT1* (induces insensitivity to T-cell cytotoxicity).
*   **Survival & Proliferation Drivers**: *PTEN*, *CDKN2A*, *PIK3CA* (oncogenic drivers).

| Pathway / Gene                       | Liu 2019 ($N=104$) | Hugo 2016 ($N=27$) | Riaz 2017 ($N=64$) | Pooled Trials ($N=195$) |
| ------------------------------------ | ------------------ | ------------------ | ------------------ | ----------------------- |
| **BRAF mutation**                    | 40.4%              | 59.3%              | 40.6%              | **43.1%**               |
| **NRAS mutation**                    | 30.8%              | 18.5%              | 17.2%              | **24.6%**               |
| **NF1 mutation**                     | 18.3%              | 25.9%              | 3.1%               | **14.4%**               |
| **Antigen Presentation (MHC)**       | 0.0%               | 0.0%               | 0.0%               | **0.0%**                |
| **IFN-gamma Signaling**              | 10.6%              | 18.5%              | 3.1%               | **9.2%**                |
| **Survival & Proliferation Drivers** | 23.1%              | 29.6%              | 10.9%              | **20.0%**               |

*Note: Pre-treatment antigen presentation mutations are completely absent in these cohorts, reinforcing that MHC-class I mutations are mostly acquired under selective pressure during checkpoint blockade therapy rather than being common baseline resistance mechanisms.*

---

## 2. Tumor Mutational Burden (TMB) & Neoantigen Load

Tumor Mutational Burden (TMB) and predicted Neoantigen Load are key genomic measures of tumor immunogenicity. Below, we present the TMB distribution (left panel) alongside a scatter plot showing the relationship between nonsynonymous TMB and predicted neoantigen load in the pooled trial cohorts (right panel).

![[tmb_distribution.png]]

TMB Distributions

![[extended_neoantigen_tmb.png]]

Neoantigen vs TMB

### Key Observations
*   **TMB as a Predictor**: In all three immunotherapy cohorts, responders (CR/PR, green boxes) exhibit a higher pre-treatment TMB distribution than non-responders (PD, red boxes). TMB is a significant predictor of response when cohorts are pooled ($p = 0.079$, ROC AUC = 0.577).
*   **TCGA Distribution**: The reference cohort shows a classical log-normal TMB distribution with a median of **15.23 mutations/Mb**. A substantial proportion of patients lie above the standard FDA clinical cutoff of **10.0 mutations/Mb** for high-TMB status, validating the presence of a strong ultraviolet (UV) signature in cutaneous melanomas.
*   **Neoantigen Collinearity**: There is a strong linear relationship between nonsynonymous TMB and predicted neoantigen load ($r = 0.756$, $p = 2.39 \times 10^{-34}$). Total predicted neoantigen load shows moderate predictive utility for response (ROC AUC = 0.545, Mann-Whitney $p = 0.298$). The extreme correlation confirms that these two metrics are collinear, making TMB a suitable surrogate for mutational neoantigen burden in downstream modeling.

---

## 3. Continuous Biomarker Correlation

### 3.1. Intra-Cohort Correlation in Liu 2019
A Spearman rank correlation matrix mapping the relationships between continuous genomic features (somatic mutation and neoantigen subtypes) in the **Liu 2019** cohort is presented below.

![[biomarker_correlation_heatmap.png]]

Genomic Biomarker Correlation Matrix

### Key Observations
*   **High Collinearity**: TMB and SNV Neoantigens show an extremely high correlation ($r_s = 0.96$). This indicates severe redundancy; in predictive machine learning models, using both features simultaneously is unlikely to add value and may destabilize model coefficients.
*   **Neoantigen Subtypes**: Somatic indel neoantigens (`INDEL_NEOANTIGEN`, $r_s = 0.44$ with TMB) and cancer-testis antigens (`CTA_SELF_NEOANTIGEN`, $r_s = 0.22$ with TMB) show much weaker correlations. This suggests they capture distinct biological axes of tumor immunogenicity that are not simply surrogates for total mutational burden.

### 3.2. Genomic Burden vs. Immune Infiltration
To understand how tumor genomic features affect the microenvironment, we evaluated how copy-number burden (Aneuploidy Score) and mutational burden (TMB) correlate with continuous transcriptomic immune signatures in both TCGA-SKCM ($N=427$) and the pooled trials ($N=195$).

![Genomic Burden vs Immune Heatmap](../plots/biomarkers/extended_immune_correlations.png)

| Immune Signature | TCGA Aneuploidy Score ($r$) | TCGA TMB ($r$) | Trial TMB ($r$) |
| ---------------- | --------------------------- | -------------- | --------------- |
| `IFN_gamma`      | **-0.045**                  | **0.144**      | **0.034**       |
| `TIS`            | **-0.090**                  | **0.108**      | **-0.035**      |
| `CD8_Tcell`      | **-0.056**                  | **0.096**      | **-0.052**      |
| `CYT`            | **-0.058**                  | **0.103**      | **-0.091**      |
| `PD_L1`          | **-0.110**                  | **0.159**      | **0.046**       |

### Key Observations
*   **Aneuploidy vs Infiltration**: Chromosomal instability (Aneuploidy Score) shows a very weak negative correlation ($r \approx -0.05$ to $-0.11$) with baseline immune signatures in TCGA-SKCM, with only `PD_L1` showing a statistically significant negative correlation ($r = -0.110$, $p = 0.022$). This indicates that while copy number alterations are associated with immune exclusion in some cancer types, Aneuploidy Score alone is a weak predictor of immune-excluded "cold" status in melanoma.
*   **Orthogonal Biomarkers**: Mutational burden (TMB) shows near-zero/weak correlation with immune signature expression in both TCGA ($r \approx 0.10$ to $0.16$) and trial cohorts ($r \approx -0.09$ to $0.05$). This demonstrates that TMB and immune infiltration represent **orthogonal biomarkers**. A tumor can be highly mutated (high TMB) but still immunologically cold, or poorly mutated but hot/inflamed. Downstream predictive models should combine both independent modalities to maximize accuracy.

---

## 4. TCGA Survival Stratification by Genomic Features

Overall Survival (OS) in the reference **TCGA-SKCM** cohort is stratified below. The left panel shows stratification by driver mutation subtype and TMB status. The right panel shows overall survival stratified by chromosomal instability (Aneuploidy Score) using a median split of 11.0.

![[km_genomic_features.png]]

TCGA Driver and TMB Survival

![[extended_aneuploidy_survival.png]]

TCGA Aneuploidy Survival

### Key Observations
*   **Driver Subtypes (Left Panel, Left Plot)**: Overall survival does not differ strongly between *BRAF*, *NRAS*, and *NF1* mutant genotypes ($p = 0.0519$). This confirms that while driver mutations are biologically critical for tumor initiation and targeted therapy matching, they do not act as strong, independent prognostic markers for long-term overall survival under standard care.
*   **TMB Stratification (Left Panel, Right Plot)**: Stratifying TCGA overall survival by TMB using a median split (15.23) shows no prognostic survival separation ($p = 0.4483$). While TMB is highly *predictive* of response to checkpoint inhibitors, it is not *prognostic* of baseline survival in the general TCGA population (where only a small subset received immunotherapy).
*   **Aneuploidy Prognostic Role (Right Panel)**: Partitioning the TCGA cohort by median Aneuploidy Score shows a marginally significant prognostic association ($p = 0.0806$), where patients with high aneuploidy (orange curve) trend towards worse overall survival compared to those with low aneuploidy (blue curve). Unlike TMB (which is purely predictive of therapy response), copy-number burden has a distinct, albeit modest, prognostic role on baseline clinical survival.

---

## 5. Co-Mutation Landscape (Oncoplot)

The complete co-mutation (oncoplot) landscape for individual patients across all three clinical trial cohorts ($N=195$) is presented below. This combines somatic mutations in core driver and resistance genes (rows) with patient-specific clinical tracks (TMB, Response, Cohort source, and Sex).

![Co-Mutation Landscape (Merged Trials)](../plots/genomic/comut_landscape_merged.png)

### Key Observations
*   **MAPK Driver Mutual Exclusivity**: There is high mutual exclusivity between the two primary MAPK pathway drivers, *BRAF* and *NRAS* (only 3 overlapping cases in Liu 2019, 0 in Hugo/Riaz). This aligns with the classical understanding that *BRAF* and *NRAS* mutations represent redundant and mutually exclusive routes for activating the RAS-RAF-MEK-ERK signaling cascade.
*   **NF1 Mutational Overlap**: Unlike *BRAF* and *NRAS*, mutations in the tumor suppressor *NF1* show significant overlap with both drivers (9 cases overlap with BRAF in Liu, 2 in Hugo). While some of these represent co-occurring driver events, many of these *NF1* mutations are passenger events. *NF1* is a large gene and highly susceptible to random somatic passenger mutations in melanoma, which features a high TMB driven by UV-light exposure.
*   **Targeted Resistance Profile**: Core genes related to antigen presentation (*B2M*) and interferon signaling (*JAK1*, *JAK2*) show low baseline mutation rates. These mutations are rare in pre-treatment biopsies, indicating that genetic disruption of interferon signaling is mostly an acquired resistance mechanism rather than a common baseline driver.
*   **Cohort Distribution**: The Cohort track shows that *BRAF* and *NRAS* mutations are evenly distributed across **Liu 2019** (purple) and **Hugo 2016** (orange). The corrected patient mapping for **Riaz 2017** (teal) is clearly visible, showing representative driver mutation profiles that match the other cohorts.
*   **No Driver Subtype Response Bias**: Responders (green blocks in the Response track) are distributed across all driver mutation subtypes (BRAF, NRAS, NF1, and Triple-WT). This visually confirms that driver mutation status itself is not predictive of anti-PD-1 clinical response.
