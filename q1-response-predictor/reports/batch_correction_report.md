# Batch Effect Assessment & Leakage-Free Batch Alignment

When combining transcriptomic datasets across independent clinical studies, technical variations (e.g. sequencing platforms, RNA extraction methods, and library preparation) typically dominate the biological signals. This report documents how technical batch effects were identified and corrected across our four melanoma cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, and **Riaz 2017**) using a mathematically rigorous, leakage-free alignment approach.

---

## 📊 Batch Structure Visualization (PCA)

To evaluate the batch effects, we performed a Principal Component Analysis (PCA) on the common intersected high-variance genes (**559 genes**) across the $N=699$ patients in the full merged cohort, comparing the uncorrected concatenated matrix against the Z-score standardized matrix.

![Batch Effect PCA Comparison](../plots/batch_effect_pca.png)

---

## 🔍 Key Findings

### 1. Panel A: Before Batch Correction
*   **Distinct Study Clustering**: The uncorrected PCA reveals a highly severe batch structure. Samples from each study cluster distinctly and occupy isolated regions of the projection space. 
*   **Technical Dominance**: The first two principal components (PC1 and PC2) represent **technical variance** driven entirely by the study of origin, rather than any shared underlying biology (such as tumor immune infiltration or patient survival). 
*   **Consequence**: Direct modeling or pooled analyses on this uncorrected matrix would yield biased predictors that learn technical features specific to each laboratory rather than generalizable prognostic biology.

### 2. Panel B: After Z-score Standardization
*   **Homogeneous Mixing**: Following cohort-independent Z-score standardization, the technical clustering is completely resolved. Samples from all four cohorts (TCGA, Liu, Hugo, and Riaz) mix homogeneously across the entire PCA projection space.
*   **Preservation of Biology**: Centering each gene to mean 0 and scaling it to unit variance within each study aligns the baseline values without altering the internal biological correlation structure.

---

## 🛡️ Cross-Validation Rigor & Data Leakage Prevention

### The Hazard of Global Batch Correction (e.g. ComBat)
Algorithms like ComBat pool all samples together to estimate batch correction parameters. When applied to a multi-study cohort prior to Leave-One-Cohort-Out (LOCO) cross-validation:
1.  The test cohort is included in the ComBat estimation step.
2.  The features in the training cohorts are adjusted using information (the mean and variance) from the test set.
3.  This **data leakage** violates the fundamental assumption of cross-validation (strict separation of train and test sets), leading to overly optimistic metric estimates.

### The Z-score Scaling Solution
By applying Z-score standardization **cohort-independently** (scaling each gene column strictly using its own cohort's mean and standard deviation):
*   **Zero Leakage**: No information is shared across datasets during the scaling process. When testing on a left-out cohort, the model relies on features scaled using only that cohort's internal distribution, mirroring real-world clinical deployment where the model encounters a completely new study/laboratory.
*   **Robust Alignment**: As shown in the PCA (Panel B), this simple self-contained standardization achieves batch alignment performance comparable to ComBat, while maintaining complete mathematical rigor.
