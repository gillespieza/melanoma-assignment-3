---
title: "Supplementary Appendix: Curated Immunotherapy Signature Descriptions"
aliases:
  - curated-signatures-supplementary
  - q1-signatures-appendix
tags:
  - q1
  - signatures
  - biomarkers
  - supplementary
  - immunotherapy
cssclasses:
  - table-small
  - table-center
  - row-alt
created: 2026-07-24 14:28
updated: 2026-08-02 16:50
---
# Supplementary Appendix: Curated Immunotherapy Signature Descriptions

This supplementary document provides the full, detailed biological rationale, gene compositions, implementation specifics, and mathematical calculations for the eight curated transcriptomic gene signatures evaluated in the Melanoma Immunotherapy Response Predictor pipeline.

*Main Report Reference*: [curated_signatures_report.md](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1_response_predictor/reports/pillar_3_transcriptomic_signatures/curated_signatures_report.md)
*Python Implementation*: [signatures.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1_response_predictor/src/signatures.py)


## 0. Mathematical Collapsing Strategies & Preprocessing

> [!NOTE] Section Context
> - **What**: Describing the mathematical collapsing functions and preprocessing steps used to convert raw RNA-seq expression profiles into continuous signature inputs for predictive modeling.
> - **Why**: Raw expression profiles ($D > 20,000$) suffer from extreme high-dimensionality, multicollinearity, and technical batch variation. Collapsing genes into validated pathway scores acts as a biological noise filter.
> - **Questions**:
>   1. *What mathematical collapsing strategies are implemented in `src/signatures.py`?*
>   2. *How do non-linear rank-based ratios (e.g. IMPRES) differ from arithmetic log-means?*
>   3. *How is zero-leakage cohort-independent Z-score standardisation guaranteed across studies?*

To transform high-dimensional, multicollinear gene expression matrices into compact, interpretable predictors, our codebase ([signatures.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1_response_predictor/src/signatures.py)) employs **four distinct mathematical collapsing strategies** based on the biological intent and literature definition of each signature modality.

### 0.1. Overview of Collapsing Modalities

| Collapsing Strategy | Signatures Applied To | Mathematical Logic | Primary Advantage |
| :--- | :--- | :--- | :--- |
| **1. Mean Log-Expression** | IFN-γ, TIS, CYT, CD8 T-Cell, Macrophage STV | Arithmetic average across $\log_2(\text{TPM} + 1)$ gene values | Smooths gene-level measurement noise and stabilises co-expressed pathway signals |
| **2. Pairwise Binary Comparison** | IMPRES | Sum of 15 boolean ratio indicators ($\mathbb{I}[E_{\text{Gene A}} > E_{\text{Gene B}}]$) | Non-linear, scale-free checkpoint balance assessment resistant to normalisation offsets |
| **3. Direct Target Gene Selection** | PD-L1 Proxy | Single gene continuous transcript value ($\log_2[\text{TPM}_{\text{CD274}} + 1]$) | Direct molecular proxy for targeted checkpoint ligand burden |
| **4. Differential Polarisation Ratio** | M1/M2 Macrophage Ratio | Difference of arithmetic means ($S_{\text{M1}} - S_{\text{M2}}$) | Quantifies functional macrophage phenotype balance (pro-inflammatory vs. immunosuppressive) |

### 0.2. Strategy 1: Mean Log-Expression (Arithmetic Average)
Applied to multi-gene co-expression modules (**IFN-γ**, **TIS**, **CYT**, **CD8 T-Cell**, **Macrophage STV**):

$$S = \frac{1}{|G_{\text{found}}|} \sum_{g \in G_{\text{found}}} E_g$$

where $E_g = \log_2(\text{TPM}_g + 1)$ represents the log-transformed expression level of gene $g$. 

* **Robustness & Partial Coverage**: If a cohort lacks 1 or 2 non-critical signature genes due to filtering or platform differences, the calculation dynamically divides by $|G_{\text{found}}|$ (the number of successfully matched genes) rather than returning an invalid `NaN`.

### 0.3. Strategy 2: Pairwise Binary Comparison Sum (Non-Linear Checkpoint Ratio Balance)
Applied to **IMPRES** *(Ausländer et al., 2018)*:

$$S_{\text{raw}} = \sum_{i=1}^{15} \mathbb{I}\left(E_{\text{Gene A}_i} > E_{\text{Gene B}_i}\right)$$

$$S_{\text{final}} = S_{\text{raw}} \times \left(\frac{15}{N_{\text{valid\_pairs}}}\right)$$

where $\mathbb{I}(\cdot)$ is an indicator function evaluating to $1$ if Gene A expression exceeds Gene B expression, and $0$ otherwise.

* **Scale-Free Property**: Because it relies entirely on relative within-sample ranks ($A > B$), IMPRES is inherently invariant to monotonic global scaling or monotonic batch shifts across datasets.

### 0.4. Strategy 3: Direct Target Gene Proxy
Applied to **PD-L1 Proxy (`CD274`)**:

$$S_{\text{PD-L1}} = \log_2(\text{TPM}_{\text{CD274}} + 1)$$

* Uses automated alias matching (`CD274`, `PD-L1`, `PDL1`) to resolve target column names across heterogeneous study annotations.

### 0.5. Strategy 4: Differential Phenotype Polarisation Ratio
Applied to **M1/M2 Macrophage Polarisation Ratio**:

$$S_{\text{M1/M2}} = \bar{E}_{\text{M1}} - \bar{E}_{\text{M2}}$$

where $\bar{E}_{\text{M1}}$ and $\bar{E}_{\text{M2}}$ are the arithmetic log-means of pro-inflammatory M1 and immunosuppressive M2 macrophage gene panels, respectively. Positive values denote pro-inflammatory M1 dominance, while negative values indicate immunosuppressive M2 macrophage polarisation.

### 0.6. Zero-Leakage Cohort-Independent Z-Score Standardisation
After computing raw continuous signature scores $S_{i, k}$ for patient $i$ in cohort $k$, features are standardised **within each individual study cohort** prior to merging:

$$Z_{i, k} = \frac{S_{i, k} - \mu_{k}}{\sigma_{k}}$$

where $\mu_k$ and $\sigma_k$ represent the internal mean and standard deviation of study cohort $k$. This step removes sequencing depth and platform scale differences while preserving zero data leakage during cross-validation.


## 1. Interferon-Gamma (IFN-γ) 6-Gene Signature

> [!NOTE] Section Context
> - **What**: Detailing the biological rationale, gene panel, and formula for the 6-gene IFN-γ cytokine response signature.
> - **Why**: IFN-γ signaling is the central axis of adaptive anti-tumour immunity and MHC up-regulation.
> - **Questions**: *How does IFN-γ expression capture baseline immune priming?*

* **Source**: Ayers et al., 2017 (*Journal of Clinical Investigation*)
* **Biological Significance**: IFN-γ is the primary cytokine secreted by activated cytotoxic `CD8`+ T-cells, NK-cells, and antigen-presenting cells (APCs). It coordinates the adaptive anti-tumour immune response, upregulates MHC class I/II molecules, and recruits effector immune cells to the tumour microenvironment.
* **Gene Composition**: 6 genes
  1. `IFNG` (Interferon Gamma) - Primary effector cytokine.
  2. `CXCL9` (C-X-C motif chemokine ligand 9) - T-cell chemoattractant.
  3. `CXCL10` (C-X-C motif chemokine ligand 10) - T-cell chemoattractant.
  4. `IDO1` (Indoleamine 2,3-dioxygenase 1) - Tryptophan-degrading feedback inhibitor.
  5. `HLA-DRA` (Major histocompatibility complex, class II, DR alpha) - Antigen presentation.
  6. `STAT1` (Signal transducer and activator of transcription 1) - IFN-γ transcription factor.
* **Mathematical Calculation**: Arithmetic mean of the $\log_2(\text{TPM} + 1)$ expression values of detected genes:
  $$S_{\text{IFN}\gamma} = \frac{1}{|G|} \sum_{g \in G} E_g$$
  where $E_g$ is the normalised log-expression of gene $g$.


## 2. Tumour Inflammation Signature (TIS)

> [!NOTE] Section Context
> - **What**: Outlining the 21-gene expanded Tumour Inflammation Signature (TIS) implementation.
> - **Why**: TIS measures pre-existing suppressed adaptive immune infiltration across antigen presentation, T-cell receptors, and checkpoint axes.
> - **Questions**: *How does our expanded 21-gene implementation enhance the standard 18-gene NanoString panel?*

* **Source**: Ayers et al., 2017 (*Journal of Clinical Investigation*)
* **Biological Significance**: A clinical-grade signature developed to identify tumours with a pre-existing, suppressed adaptive immune response. It measures antigen presentation, T-cell abundance, chemokines, and checkpoint inhibition.
* **Gene Composition (21 Genes in Code)**:
  `CCL5`, `CD2`, `CD3D`, `CD3E`, `CD27`, `CD274`, `CMKLR1`, `CXCL9`, `CXCR6`, `GZMB`, `GZMK`, `HLA-DRA`, `HLA-DQA1`, `HLA-E`, `IDO1`, `LAG3`, `NKG7`, `PDCD1LG2`, `PSMB10`, `STAT1`, `TIGIT`.

  > [!NOTE] Implementation Rationale
  > The clinical NanoString TIS panel typically consists of 18 genes. Our implementation in [signatures.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1_response_predictor/src/signatures.py#L41-L53) expands this set to 21 genes by incorporating critical T-cell receptor components (`CD2`, `CD3D`, `CD3E`) and cytolytic enzymes (`GZMB`, `GZMK`), providing a broader capture of T-cell biology.
* **Mathematical Calculation**: Arithmetic mean of the $\log_2(\text{TPM} + 1)$ expression values of detected genes across the expanded panel.


## 3. Cytolytic Activity (CYT) Score

> [!NOTE] Section Context
> - **What**: Describing the 2-gene Cytolytic Activity (CYT) metric (`GZMA`, `PRF1`).
> - **Why**: CYT measures active physical cell killing mediated by perforin and granzymes.
> - **Questions**: *Does CYT capture effector function independently of total lymphocyte infiltration?*

* **Source**: Rooney et al., 2015 (*Cell*)
* **Biological Significance**: Directly measures the effector cytolytic function of local `CD8`+ cytotoxic T-lymphocytes and NK-cells. A high CYT score indicates active physical tumour cell killing by perforin-mediated entry of granzymes.
* **Gene Composition**: 2 genes
  1. `GZMA` (Granzyme A) - Protease that induces caspase-independent apoptosis.
  2. `PRF1` (Perforin 1) - Pore-forming protein that facilitates granzyme entry.
* **Mathematical Calculation**: Arithmetic mean of the log-transformed expression values of the two genes:
  $$S_{\text{CYT}} = \frac{E_{\text{GZMA}} + E_{\text{PRF1}}}{2}$$
  *(Equivalent to the logarithm of the geometric mean on the linear TPM scale).*


## 4. CD8 T-Cell Abundance Signature

> [!NOTE] Section Context
> - **What**: Describing the lineage-specific `CD8` T-cell abundance marker (`CD8A`, `CD8B`).
> - **Why**: `CD8`+ T-cell infiltration is the primary cellular target of anti-PD-1 therapy.
> - **Questions**: *How accurately do `CD8A` and `CD8B` reflect cytotoxic T-cell density?*

* **Biological Significance**: Serves as a direct lineage-specific marker for the presence of `CD8`+ cytotoxic T-cells within the tumour microenvironment. `CD8`+ infiltration is the primary target and driver of anti-PD-1 clinical efficacy.
* **Gene Composition**: 2 genes
  1. `CD8A` (CD8 cell surface glycoprotein antibody, alpha chain)
  2. `CD8B` (CD8 cell surface glycoprotein antibody, beta chain)
* **Mathematical Calculation**: Arithmetic mean of the log-transformed expression of `CD8A` and `CD8B`:
  $$S_{\text{CD8}} = \frac{E_{\text{CD8A}} + E_{\text{CD8B}}}{2}$$


## 5. Immune Predictive Score (IMPRES)

> [!NOTE] Section Context
> - **What**: Detailing the 15 logical pairwise ratios of checkpoint molecules comprising IMPRES.
> - **Why**: IMPRES evaluates the relative balance of stimulatory vs. inhibitory immune checkpoint signals.
> - **Questions**: *Why is a rank-based ratio score more resistant to batch offsets than absolute expression levels?*

* **Source**: Ausländer et al., 2018 (*Nature Medicine*)
* **Biological Significance**: Developed specifically for cutaneous melanoma. Rather than using raw expression values, IMPRES evaluates the relative balance of inhibitory and stimulatory checkpoint molecules. It is built on 15 logical pairwise relationships: a high score indicates that stimulatory signals dominate, suggesting a higher likelihood of response to immune checkpoint blockade.
* **Logical Pairs (Gene A, Gene B)**:
  1. `("CD274", "VSIR")` (`CD274` / PD-L1 vs. `VSIR` / VISTA)
  2. `("CD28", "CD276")` (`CD28` vs. `CD276` / B7-H3)
  3. `("CD86", "TNFRSF4")` (`CD86` vs. `TNFRSF4` / OX40)
  4. `("CD86", "CD200")` (`CD86` vs. `CD200`)
  5. `("CTLA4", "TNFRSF4")` (`CTLA4` vs. `TNFRSF4` / OX40)
  6. `("PDCD1", "TNFRSF4")` (`PDCD1` / PD-1 vs. `TNFRSF4` / OX40)
  7. `("CD80", "TNFSF9")` (`CD80` vs. `TNFSF9` / 4-1BBL)
  8. `("CD86", "HAVCR2")` (`CD86` vs. `HAVCR2` / TIM-3)
  9. `("CD28", "CD86")` (`CD28` vs. `CD86`)
  10. `("CD27", "PDCD1")` (`CD27` vs. `PDCD1` / PD-1)
  11. `("CD40", "CD274")` (`CD40` vs. `CD274` / PD-L1)
  12. `("CD40", "CD80")` (`CD40` vs. `CD80`)
  13. `("CD40", "CD28")` (`CD40` vs. `CD28`)
  14. `("CD40", "PDCD1")` (`CD40` vs. `PDCD1` / PD-1)
  15. `("TNFRSF14", "CD86")` (`TNFRSF14` / HVEM vs. `CD86`)

* **Mathematical Formula**:
  $$S_{\text{raw}} = \sum_{i=1}^{15} \mathbb{I}(E_{\text{Gene A}_i} > E_{\text{Gene B}_i})$$
  To account for missing genes, the score is scaled back to a range of 0–15:
  $$S_{\text{IMPRES}} = S_{\text{raw}} \times \left( \frac{15}{\text{number of valid pairs evaluated}} \right)$$


## 6. PD-L1 Transcript Proxy

> [!NOTE] Section Context
> - **What**: Describing `CD274` mRNA expression as a continuous proxy for PD-L1 protein ligand expression.
> - **Why**: `CD274` transcript level represents the primary molecular target of anti-PD-1/PD-L1 antibody therapy.
> - **Questions**: *How reliably does mRNA expression substitute for IHC staining in transcriptomic pipelines?*

* **Biological Significance**: Directly evaluates the transcript level of `CD274` (encoding PD-L1). While PD-L1 is usually measured by immunohistochemistry (IHC), mRNA expression acts as a clean continuous molecular proxy for checkpoint burden.
* **Gene**: `CD274`
* **Mathematical Calculation**: Standard $\log_2(\text{TPM} + 1)$ expression of `CD274`.


## 7. Macrophage STV Spatial Barrier Score (`Macrophage_STV_Score`)

> [!NOTE] Section Context
> - **What**: Describing the 5-gene M2 Macrophage Signature Transcript Vector (STV) Spatial Barrier Score.
> - **Why**: M2-polarized tumor-associated macrophages (TAMs) create a physical and immunosuppressive stromal barrier that restricts cytotoxic T-cell infiltration.
> - **Questions**: *How does M2 macrophage infiltration mediate resistance to anti-PD-1 monotherapy?*

* **Source**: Derived from M2 macrophage signature vectors in `data/config/m1_m2_stv.csv`.
* **Biological Significance**: Measures the density of immunosuppressive M2-polarized macrophages (`CD163`+, `MSR1`+, `MRC1`+, `CSF1R`+, `TGFB1`+). M2 TAMs secrete pro-fibrotic factors (such as TGF-β) and deplete essential amino acids, forming a physical stromal barrier that excludes effector `CD8`+ T-cells.
* **Gene Composition**: 5 M2 marker genes
  1. `CD163` (Scavenger receptor cysteine-rich type 1 protein) - M2 lineage marker.
  2. `MSR1` (Macrophage scavenger receptor 1) - Endocytic scavenger receptor.
  3. `MRC1` (Mannose receptor C-type 1 / CD206) - Endocytic receptor.
  4. `CSF1R` (Colony stimulating factor 1 receptor) - Macrophage survival/differentiation receptor.
  5. `TGFB1` (Transforming growth factor beta 1) - Immunosuppressive & pro-fibrotic cytokine.
* **Mathematical Calculation**: Arithmetic mean of log-transformed expression of detected M2 genes:
  $$S_{\text{Macrophage\_STV}} = \frac{1}{|G_{\text{M2}}|} \sum_{g \in G_{\text{M2}}} E_g$$


## 8. M1/M2 Macrophage Polarisation Ratio (`M1_M2_Ratio`)

> [!NOTE] Section Context
> - **What**: Outlining the 10-gene differential M1/M2 macrophage polarisation ratio (`M1_M2_Ratio`).
> - **Why**: Distinguishes pro-inflammatory anti-tumour (M1) macrophages from immunosuppressive pro-tumour (M2) macrophages in the microenvironment.
> - **Questions**: *Does the balance of M1 vs. M2 macrophages provide additive predictive value over total macrophage abundance?*

* **Biological Significance**: Captures the functional polarisation balance of tumour-associated macrophages. High M1/M2 ratios indicate a pro-inflammatory microenvironment favorable to adaptive immune activation, whereas low ratios highlight immunosuppressive stromal exclusion.
* **Gene Composition**: 10 genes (5 M1 markers vs. 5 M2 markers)
  * **M1 Panel**: `NOS2`, `TNF`, `IL1B`, `CD68`, `FCGR3A`
  * **M2 Panel**: `CD163`, `MSR1`, `MRC1`, `CSF1R`, `TGFB1`
* **Mathematical Calculation**: Difference of arithmetic means between M1 and M2 log-expression vectors:
  $$S_{\text{M1/M2}} = \bar{E}_{\text{M1}} - \bar{E}_{\text{M2}}$$
