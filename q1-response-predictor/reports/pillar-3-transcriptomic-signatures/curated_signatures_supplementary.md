---
created: 2026-07-24 14:28
cssclasses:
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-24 14:28
---

# Supplementary Appendix: Curated Immunotherapy Signature Descriptions

This supplementary document provides the full, detailed biological rationale, gene compositions, implementation specifics, and mathematical calculations for the six curated transcriptomic gene signatures evaluated in the Melanoma Immunotherapy Response Predictor pipeline.

*Main Report Reference*: [curated_signatures_report.md](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/reports/pillar-3-transcriptomic-signatures/curated_signatures_report.md)  
*Python Implementation*: [signatures.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/signatures.py)


## 0. Mathematical Collapsing Strategies & Preprocessing

To transform high-dimensional, multicollinear gene expression matrices into compact, interpretable predictors, our codebase ([signatures.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/signatures.py)) employs **three distinct mathematical collapsing strategies** based on the biological intent and literature definition of each signature modality.

### 0.1. Overview of Collapsing Modalities

| Collapsing Strategy | Signatures Applied To | Mathematical Logic | Primary Advantage |
| :--- | :--- | :--- | :--- |
| **1. Mean Log-Expression** | IFN-γ, TIS, CYT, CD8 T-Cell | Arithmetic average across $\log_2(\text{TPM} + 1)$ gene values | Smooths gene-level measurement noise and stabilizes co-expressed pathway signals |
| **2. Pairwise Binary Comparison** | IMPRES | Sum of 15 boolean ratio indicators ($\mathbb{I}[E_{\text{Gene A}} > E_{\text{Gene B}}]$) | Non-linear, scale-free checkpoint balance assessment resistant to normalisation offsets |
| **3. Direct Target Gene Selection** | PD-L1 Proxy | Single gene continuous transcript value ($\log_2[\text{TPM}_{\text{CD274}} + 1]$) | Direct molecular proxy for targeted checkpoint ligand burden |

### 0.2. Strategy 1: Mean Log-Expression (Arithmetic Average)
Applied to multi-gene co-expression modules (**IFN-γ**, **TIS**, **CYT**, **CD8 T-Cell**):

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

### 0.5. Zero-Leakage Cohort-Independent Z-Score Standardisation
After computing raw continuous signature scores $S_{i, k}$ for patient $i$ in cohort $k$, features are standardized **within each individual study cohort** prior to merging:

$$Z_{i, k} = \frac{S_{i, k} - \mu_{k}}{\sigma_{k}}$$

where $\mu_k$ and $\sigma_k$ represent the internal mean and standard deviation of study cohort $k$. This step removes sequencing depth and platform scale differences while preserving zero data leakage during cross-validation.


## 1. Interferon-Gamma (IFN-γ) 6-Gene Signature

* **Source**: Ayers et al., 2017 (*Journal of Clinical Investigation*)
* **Biological Significance**: IFN-γ is the primary cytokine secreted by activated cytotoxic T-cells, NK-cells, and antigen-presenting cells (APCs). It coordinates the adaptive anti-tumour immune response, upregulates MHC class I/II molecules, and recruits effector immune cells to the tumour microenvironment.
* **Gene Composition**: 6 genes
  1. `IFNG` (Interferon Gamma) - Primary effector cytokine.
  2. `CXCL9` (C-X-C motif chemokine ligand 9) - T-cell chemoattractant.
  3. `CXCL10` (C-X-C motif chemokine ligand 10) - T-cell chemoattractant.
  4. `IDO1` (Indoleamine 2,3-dioxygenase 1) - Tryptophan-degrading feedback inhibitor.
  5. `HLA-DRA` (Major histocompatibility complex, class II, DR alpha) - Antigen presentation.
  6. `STAT1` (Signal transducer and activator of transcription 1) - IFN-γ transcription factor.
* **Mathematical Calculation**: Arithmetic mean of the $\log_2(\text{TPM} + 1)$ expression values of the detected genes:
  \[S_{\text{IFN}\gamma} = \frac{1}{|G|} \sum_{g \in G} E_g\]
  where $E_g$ is the normalised log-expression of gene $g$.


## 2. Tumour Inflammation Signature (TIS)

* **Source**: Ayers et al., 2017 (*Journal of Clinical Investigation*)
* **Biological Significance**: A clinical-grade signature developed to identify tumours with a pre-existing, suppressed adaptive immune response. It measures antigen presentation, T-cell abundance, chemokines, and checkpoint inhibition.
* **Gene Composition (21 Genes in Code)**:
  `CCL5`, `CD2`, `CD3D`, `CD3E`, `CD27`, `CD274`, `CMKLR1`, `CXCL9`, `CXCR6`, `GZMB`, `GZMK`, `HLA-DRA`, `HLA-DQA1`, `HLA-E`, `IDO1`, `LAG3`, `NKG7`, `PDCD1LG2`, `PSMB10`, `STAT1`, `TIGIT`.

  > [!NOTE]
  > **Implementation Characteristic**: The clinical NanoString TIS panel typically consists of 18 genes. Our implementation in [signatures.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/signatures.py#L60-L75) expands this set to 21 genes by incorporating critical T-cell receptor components (`CD2`, `CD3D`, `CD3E`) and cytolytic enzymes (`GZMB`, `GZMK`), providing a broader capture of T-cell biology.
* **Mathematical Calculation**: Arithmetic mean of the $\log_2(\text{TPM} + 1)$ expression values of the detected genes across the expanded panel.


## 3. Cytolytic Activity (CYT) Score

* **Source**: Rooney et al., 2015 (*Cell*)
* **Biological Significance**: Directly measures the effector cytolytic function of local CD8+ cytotoxic T-lymphocytes and NK-cells. A high CYT score indicates active, physical tumour cell killing by perforin-mediated entry of granzymes.
* **Gene Composition**: 2 genes
  1. `GZMA` (Granzyme A) - Protease that induces caspase-independent apoptosis.
  2. `PRF1` (Perforin 1) - Pore-forming protein that facilitates granzyme entry.
* **Mathematical Calculation**: Arithmetic mean of the log-transformed expression values of the two genes:
  \[S_{\text{CYT}} = \frac{E_{\text{GZMA}} + E_{\text{PRF1}}}{2}\]
  *(Equivalent to the logarithm of the geometric mean on the linear TPM scale).*


## 4. CD8 T-Cell Abundance Signature

* **Biological Significance**: Serves as a direct lineage-specific marker for the presence of CD8+ cytotoxic T-cells within the tumour microenvironment. CD8+ infiltration is the primary target and driver of anti-PD-1 clinical efficacy.
* **Gene Composition**: 2 genes
  1. `CD8A` (CD8 cell surface glycoprotein antibody, alpha chain)
  2. `CD8B` (CD8 cell surface glycoprotein antibody, beta chain)
* **Mathematical Calculation**: Arithmetic mean of the log-transformed expression of `CD8A` and `CD8B`:
  \[S_{\text{CD8}} = \frac{E_{\text{CD8A}} + E_{\text{CD8B}}}{2}\]


## 5. Immune Predictive Score (IMPRES)

* **Source**: Ausländer et al., 2018 (*Nature Medicine*)
* **Biological Significance**: Developed specifically for cutaneous melanoma. Rather than using raw expression values, IMPRES evaluates the relative balance of inhibitory and stimulatory checkpoint molecules. It is built on 15 logical pairwise relationships: a high score indicates that stimulatory signals dominate, suggesting a higher likelihood of response to immune checkpoint blockade.
* **Logical Pairs (Gene A, Gene B)**:
  1. `("CD274", "VSIR")` (PD-L1 vs VISTA)
  2. `("CD28", "CD276")` (CD28 vs B7-H3)
  3. `("CD86", "TNFRSF4")` (CD86 vs OX40)
  4. `("CD86", "CD200")` (CD86 vs CD200)
  5. `("CTLA4", "TNFRSF4")` (`CTLA4` vs OX40)
  6. `("PDCD1", "TNFRSF4")` (PD-1 vs OX40)
  7. `("CD80", "TNFSF9")` (CD80 vs 4-1BBL)
  8. `("CD86", "HAVCR2")` (CD86 vs TIM-3)
  9. `("CD28", "CD86")` (CD28 vs CD86)
  10. `("CD27", "PDCD1")` (CD27 vs PD-1)
  11. `("CD40", "CD274")` (CD40 vs PD-L1)
  12. `("CD40", "CD80")` (CD40 vs CD80)
  13. `("CD40", "CD28")` (CD40 vs CD28)
  14. `("CD40", "PDCD1")` (CD40 vs PD-1)
  15. `("TNFRSF14", "CD86")` (HVEM vs CD86)

  \[S_{\text{raw}} = \sum_{i=1}^{15} \mathbb{I}(E_{\text{Gene A}_i} > E_{\text{Gene B}_i})\]
  To account for missing genes, the score is scaled back to a range of 0–15:
  \[S_{\text{IMPRES}} = S_{\text{raw}} \times \left( \frac{15}{\text{number of valid pairs evaluated}} \right)\]


## 6. PD-L1 Transcript Proxy

* **Biological Significance**: Directly evaluates the transcript level of `CD274` (encoding PD-L1). While PD-L1 is usually measured by immunohistochemistry (IHC), mRNA expression acts as a clean continuous molecular proxy for checkpoint burden.
* **Gene**: `CD274`
* **Mathematical Calculation**: Standard $\log_2(\text{TPM} + 1)$ expression of `CD274`.
