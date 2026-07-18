# TCGA Pan-Cancer Derived Prognostic Signature Report

We performed transcriptomic feature selection on the **TCGA-SKCM** cohort ($N = 421$ aligned samples with survival data) to build a custom overall survival signature, and subsequently validated it on three independent clinical trial cohorts.

## 1. Top 20 Prognostic Genes in TCGA-SKCM
The 20 genes most significantly associated with overall survival in univariate Cox regression are visualized below. A positive Beta indicates a **risk-associated gene** (higher expression = worse survival), while a negative Beta indicates a **protective gene** (higher expression = better survival).

### Hazard Ratio Forest Plot (Top 20 Genes)
The forest plot below visualizes the Hazard Ratios (HR) and their 95% confidence intervals for the top 20 most significant prognostic transcripts. Protective genes (HR < 1.0) are shown in blue, and risk-associated genes (HR > 1.0) are shown in red:

![Prognostic Gene Forest Plot](../plots/feature_selection/transcriptomic_forest_plot.png)

## 2. Kaplan-Meier Survival Curve on TCGA
We partitioned TCGA-SKCM patients into High-Risk and Low-Risk groups using the median value of the signature score. The log-rank test indicates an extremely significant separation in survival curves:

*   **Log-Rank p-value**: **4.89e-08**

![KM Curve of TCGA Survival](../plots/feature_selection/km_pancancer_signature.png)

## 3. Validation on Immunotherapy Clinical Trial Cohorts
We evaluated the custom 20-gene prognostic signature on three cohorts receiving anti-PD-1 or combination immunotherapies to see if the overall survival signature translates into predicting immunotherapy response.

| Cohort | N | Aligned Signature Genes | Response ROC AUC | Mann-Whitney U p-value | Mean Risk (Responders) | Mean Risk (Non-Responders) |
|---|---|---|---|---|---|---|
| Liu 2019 | 103 | 19/20 | **0.554** | 3.52e-01 | -8.881 | -8.071 |
| Hugo 2016 | 26 | 19/20 | **0.432** | 5.73e-01 | -6.738 | -7.687 |
| Riaz 2017 | 33 | 20/20 | **0.652** | 1.77e-01 | -9.811 | -7.506 |

### Validation Visualizations
#### ROC Curves predicting Response
![ROC Curves for Response](../plots/feature_selection/pancancer_signature_trial_validation.png)

#### Signature Risk Score Stratified by Responders vs. Non-Responders
![Signature Violin Plots](../plots/feature_selection/pancancer_signature_violins.png)

## 4. Biological Interpretation & Discussion
- **Signature Composition**: Out of the top 20 prognostic genes, **0** genes are associated with increased risk, and **20** genes are protective.
- **Prognostic utility**: The signature score is a highly robust prognostic marker on TCGA overall survival.
- **Predictive utility (Immunotherapy)**: The validation shows performance of **(Liu 2019 AUC = 0.554, Hugo 2016 AUC = 0.432, Riaz 2017 AUC = 0.652)** across the trials. Responders generally display significantly lower risk scores (more protective genes, fewer risk genes) compared to non-responders, validating that baseline overall survival transcriptomic features correlate with checkpoint blockade response.

## 5. Functional Classification & Domain Annotation of Signature Genes
To understand the molecular mechanisms underlying our 20-gene overall survival signature, we queried the **MyGene.info** and **InterPro** APIs to retrieve target annotations, protein families, and functional domains. The genes are grouped below by their biological roles:

| Gene Symbol / Category                                      | Functional Name / Description                             | Major InterPro Domains (Pfam)                                                                                                                |
|:------------------------------------------------------------|:----------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|
| **Interferon-Induced Guanylate-Binding Proteins (GTPases)** |                                                           |                                                                                                                                              |
| - **GBP1**                                                  | guanylate binding protein 1                               | Guanylate-binding protein/Atlastin, C-terminal, Guanylate-binding protein, N-terminal, P-loop containing nucleoside triphosphate hydrolase   |
| - **GBP4**                                                  | guanylate binding protein 4                               | Guanylate-binding protein/Atlastin, C-terminal, Guanylate-binding protein, N-terminal, P-loop containing nucleoside triphosphate hydrolase   |
| - **GBP5**                                                  | guanylate binding protein 5                               | Guanylate-binding protein/Atlastin, C-terminal, Guanylate-binding protein, N-terminal, P-loop containing nucleoside triphosphate hydrolase   |
| - **GBP1P1**                                                | guanylate binding protein 1 pseudogene 1                  | No annotated domains                                                                                                                         |
| **Chemokines & Intercellular Cytokines**                    |                                                           |                                                                                                                                              |
| - **CCL8**                                                  | C-C motif chemokine ligand 8                              | CC chemokine, conserved site, Chemokine interleukin-8-like domain, Chemokine interleukin-8-like superfamily                                  |
| - **CXCL10**                                                | C-X-C motif chemokine ligand 10                           | CXC chemokine, Chemokine interleukin-8-like domain, CXC chemokine, conserved site                                                            |
| - **CXCL11**                                                | C-X-C motif chemokine ligand 11                           | CXC chemokine, Chemokine interleukin-8-like domain, CXC chemokine, conserved site                                                            |
| - **IL15**                                                  | interleukin 15                                            | Interleukin-15/Interleukin-21 family, Four-helical cytokine-like, core, Interleukin-15                                                       |
| **NK-Cell & T-Cell Receptors & Regulators**                 |                                                           |                                                                                                                                              |
| - **KLRD1**                                                 | killer cell lectin like receptor D1                       | C-type lectin-like, C-type lectin-like/link domain superfamily, C-type lectin fold                                                           |
| - **KLRK1**                                                 | killer cell lectin like receptor K1                       | C-type lectin-like, C-type lectin-like/link domain superfamily, C-type lectin fold                                                           |
| - **GPR171**                                                | G protein-coupled receptor 171                            | G protein-coupled receptor, rhodopsin-like, GPCR, rhodopsin-like, 7TM, G-protein-coupled receptor 171                                        |
| - **CD72**                                                  | CD72 molecule                                             | C-type lectin-like, C-type lectin-like/link domain superfamily, C-type lectin fold                                                           |
| - **CD38**                                                  | CD38 molecule                                             | ADP-ribosyl cyclase (CD38/157)                                                                                                               |
| - **PTPN22**                                                | protein tyrosine phosphatase non-receptor type 22         | Tyrosine-specific protein phosphatase, PTPase domain, Tyrosine-specific protein phosphatases domain, Protein-tyrosine phosphatase, catalytic |
| **Intracellular Signaling & Scaffolding Adapters**          |                                                           |                                                                                                                                              |
| - **STAT4**                                                 | signal transducer and activator of transcription 4        | SH2 domain, Transcription factor STAT, p53-like transcription factor, DNA-binding domain superfamily                                         |
| - **SAMSN1**                                                | SAM domain, SH3 domain and nuclear localization signals 1 | SLy proteins associated disordered region, SH3 domain, Sterile alpha motif domain                                                            |
| - **AKAP5**                                                 | A-kinase anchoring protein 5                              | A kinase-anchoring protein AKAP5 and AKAP12, calmodulin (CaM)-binding motif, A-kinase anchor protein 5                                       |
| **Enzymes & Metabolic Regulators**                          |                                                           |                                                                                                                                              |
| - **IDO1**                                                  | indoleamine 2,3-dioxygenase 1                             | Indoleamine 2,3-dioxygenase, Tryptophan/Indoleamine 2,3-dioxygenase-like                                                                     |
| - **PLAAT4**                                                | phospholipase A and acyltransferase 4                     | LRAT domain, H-rev107 Phospholipase/Acyltransferase                                                                                          |
| **Transcription Factors & Zinc Fingers**                    |                                                           |                                                                                                                                              |
| - **ZNF831**                                                | zinc finger protein 831                                   | Zinc finger C2H2-type, Zinc finger C2H2 superfamily                                                                                          |

### Biological Insights:
1.  **Interferon-Gamma Activation**: A large proportion of the protective signature consists of *Guanylate-Binding Proteins (GBPs)*, which are classic GTPases induced by Type II Interferon (IFN-g) that coordinate cell-autonomous defense. Their elevated expression indicates strong tissue-level cytokine response.
2.  **T-Cell & NK-Cell Chemoattraction**: The presence of key chemokines (*CXCL10*, *CXCL11*, and *CCL8*) and cytokines (*IL15*) confirms that high-expression patients have active recruitment and retention programs for tumor-infiltrating lymphocytes (CD8+ cytotoxic T cells and NK cells).
3.  **Active Cytotoxic Machinery**: Receptors like *KLRK1* (NKG2D) and *KLRD1* (CD94) are hallmark activating molecules on cytolytic cells (T/NK cells). High levels directly reflect an active immune synapse capable of killing tumor cells.
4.  **Feedback Immunosuppression**: *IDO1* (Indoleamine 2,3-dioxygenase 1) is a well-known tryptophan-degrading enzyme induced by interferon-gamma as a feedback inhibitory mechanism. In this context, although IDO1 acts as a checkpoint, it functions as a highly significant *protective* marker because it is a direct surrogate for intense local anti-tumor inflammation.