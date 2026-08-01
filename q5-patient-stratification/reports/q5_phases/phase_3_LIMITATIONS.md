---
title: "Phase 3 Limitations: Unsupervised Patient Stratification"
aliases: Q5 Phase 3 Limitations
tags:
  - limitations
  - melanoma
  - patient-stratification
  - phase-3
  - q5
created: 2026-08-01 14:06
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-01 14:09
---

# Phase 3 Limitations & Future Directions: Unsupervised Patient Stratification

> [!NOTE] Scope & Purpose of This Document  
> This document evaluates Phase 3 of the Q5 Patient Stratification pipeline in its **current state** across four domains: code quality, statistical methodology, biological assumptions, and computational/data constraints. All metrics are sourced directly from live pipeline outputs (`patient_clusters.csv`, `gmm_posterior_probabilities.csv`, `mahalanobis_spectral_metrics.csv`, `03_cluster_patients.log`) and `PROJECT_MAP.md`. Improvements are ranked by scientific impact.

---

# 1. Code Quality

> [!NOTE] What Is Being Evaluated  
> Structural and maintainability issues in the Phase 3 source modules (`clustering.py`, `phenotyping.py`, `q5_constants.py`) as they exist today — not historical issues that have been resolved.

## 1.1 Stub Implementations in `phenotyping.py`

Per `PROJECT_MAP.md` (Known Technical Debt), `plot_radar_chart()` and `plot_cluster_heatmap()` in `phenotyping.py` (lines 155–163) are `pass` stubs. These functions are registered and exported by the module but produce no output. Any calling code that depends on them silently succeeds without generating the intended artefact, making the failure invisible in pipeline logs. This is a silent no-op code smell.

**Impact**: Moderate. No figures are lost because the pipeline does not currently call these functions in its primary execution path — but the dead interface creates a false impression of feature completeness.

**Fix (Priority 2)**: Implement `plot_radar_chart()` using a `matplotlib.patches.FancyArrowPatch` polar axis to visualise per-phenotype median feature profiles; implement `plot_cluster_heatmap()` as a seaborn `clustermap` of the 9 clustering features scaled per-feature. Both should call `save_fig()` from `src.utils.plotting`.

## 1.2 Duplicate `get_phenotype_color()` in `src/styles.py`

Per `PROJECT_MAP.md` (L296), `src/styles.py` contains two definitions of `get_phenotype_color()` at lines 53 and 178. Python silently uses the second definition; the first is dead code that will never execute. Any future change made to the first definition will have no effect, creating a maintenance hazard.

**Impact**: Low (currently working correctly because the second definition is used), but risks introducing silent divergence if the file is edited.

**Fix (Priority 3)**: Remove the first definition (L53) and retain the second (L178). Add a brief inline comment explaining why a single definition exists.

## 1.3 `clustering.py` Module Name Mismatch

`PROJECT_MAP.md` lists `clustering.py` as exporting `run_kmeans()`, but Phase 3 now uses `run_gmm()` internally. The legacy `run_kmeans()` export is preserved for backwards compatibility (`gmm_model.pkl` is also saved as `kmeans_model.pkl` per the log output). This naming inconsistency will mislead future contributors into believing the pipeline runs K-Means when it runs GMM.

**Impact**: Low (functionality correct), but creates documentation drift and violates the principle of least surprise.

**Fix (Priority 3)**: Remove the `kmeans_model.pkl` legacy fallback write and deprecate `run_kmeans()` with a `DeprecationWarning`. Update `PROJECT_MAP.md` module table accordingly.

---

# 2. Statistical Weaknesses

> [!WARNING] Current Statistical Limitations  
> The following issues represent genuine statistical threats to the validity of the Phase 3 clustering solution, not implementation errors. They are limitations of the current methodological choices that future work should address.

## 2.1 Near-Degenerate Posterior Probability Distributions

The live `gmm_posterior_probabilities.csv` reveals a critical statistical pathology: **698 of 699 patients (99.9%) have a maximum posterior probability ≥ 0.99**, with a median max-posterior of exactly 1.0. Only 1 patient (0.1%) falls below the 0.70 ambiguity threshold; zero patients are genuinely uncertain (<0.50).

> [!WARNING] This Pattern Indicates Hard, Not Soft, Clustering  
> A well-specified GMM with full covariance matrices on a 9-dimensional continuous feature space should produce a meaningful posterior uncertainty distribution — particularly for patients near cluster boundaries. A posterior distribution concentrated at 0 or 1 is characteristic of **degenerate GMM fitting**, where the model collapses to near-deterministic assignments. This defeats the stated scientific rationale for choosing GMM over K-Means (soft, probabilistic assignments).

**Root Cause (Likely)**: The 9 clustering features include 3 binary mutation indicators (`mut_BRAF`, `mut_NRAS`, `mut_NF1`). When binary features are mixed with continuous immune scores (TIS, CYT, etc.) in a GMM with full covariance, the mutation binary split can dominate the likelihood function and drive clusters to hard boundaries. The mutation features are effectively operating as a hard partition key, overriding the continuous uncertainty signal from the immune features.

**Evidence from Metrics**:
- Standard Scaled GMM Silhouette: **0.1906** (poor — suggests cluster overlap in the continuous immune space)
- Mahalanobis-Transformed GMM Silhouette: **0.1951** (marginal improvement)
- Spectral Manifold Silhouette: **0.2063** (best, still moderate)
- Log-Likelihood: **+6.50** (standard scaling) vs **+4.38** (Mahalanobis) — neither indicates a well-separated generative model

All three Silhouette scores fall below the commonly accepted threshold of 0.25–0.30 for well-separated clusters in a biological feature space.

**Fix (Priority 1)**: Pre-process the binary mutation indicators separately from the continuous immune scores. Either: (a) embed mutations as a stratification variable _after_ continuous-feature GMM fitting; or (b) use a mixed-data model (e.g., latent class analysis or a variational autoencoder latent space) that treats binary and continuous features with their appropriate likelihoods.

## 2.2 $K=4$ Selection Lacks Rigorous Statistical Justification

The consensus clustering Delta Area $\Delta(4) = 0.1499$ supports $K=4$ clusters, but the Delta Area plot (`consensus_delta_area.png`) has not been supplemented with formal model-selection criteria applied directly to the GMM:

- **BIC for the full-cohort GMM** is reported as **−7,656.2** (standard scaling) and **−4,682.2** (Mahalanobis) in `mahalanobis_spectral_metrics.csv`. No BIC curve across $K \in [2, 8]$ has been computed and compared for the GMM itself — only the consensus clustering bootstrap ensemble was used for $K$ selection.
- The ICI-only cohort evaluation (N=326) reports **AIC = 4,501,941.2 and BIC = 4,502,770.5** — values many orders of magnitude larger than the full-cohort BIC, suggesting the ICI-only scaling is not comparable (likely raw log-likelihood ×N rather than normalised).

**Fix (Priority 1)**: Fit GMM models for $K \in [2, 8]$ on the full cohort and plot BIC against $K$. The minimum BIC value identifies the statistically optimal number of Gaussian components. Cross-validate this against the existing consensus clustering result.

## 2.3 Immune Cold Cluster is Severely Underpowered

The _Immune Cold_ phenotype contains only **$N=22$ patients (3.1%)** of the full cohort ($N=699$). Of these, only 15 were ICI-treated. A GMM component fitted on 22 observations across a 9-dimensional feature space is statistically fragile:

- The full covariance matrix $\boldsymbol{\Sigma}_k \in \mathbb{R}^{9 \times 9}$ requires estimating $9 \times 10 / 2 = 45$ unique covariance parameters from 22 observations — fewer observations than free parameters.
- This guarantees that the _Immune Cold_ covariance matrix is either near-singular or regularised (added noise diagonal), making the posterior probabilities computed for this component unreliable.

**Fix (Priority 2)**: Apply a minimum-component regularisation: enforce a minimum expected cluster size of $\geq 30$ observations before fitting the full covariance. For $N<30$ components, constrain to a diagonal (tied) covariance. Document the regularisation decision in the script.

## 2.4 No Permutation Test for Cluster Stability

The 1,000-bootstrap consensus clustering in `09_run_consensus_clustering.py` demonstrates that $K=4$ clusters are reproducible across sub-samples, but this does not test whether the clustering is non-random. No permutation test (random label shuffle → compare Silhouette to observed Silhouette) has been applied to confirm that the observed Silhouette score of 0.19–0.21 exceeds a null distribution.

**Fix (Priority 2)**: Run 100 permutations of the feature matrix (row-shuffle each feature independently to break structure) and compute Silhouette scores. If observed Silhouette exceeds the 95th percentile of permuted Silhouettes, the clustering is confirmed as non-random.

---

# 3. Biological Assumptions

> [!WARNING] Key Biological Assumptions Under-Validated in Phase 3  
> These are assumptions embedded in the current feature set and phenotype definitions that are scientifically reasonable but not directly validated against external biological references in Phase 3.

## 3.1 TIS, CYT, and Deconvolution Scores Conflate Distinct Biological Processes

The 9 clustering features include TIS, CYT, `CD8_T_cells`, `M1_Macrophages`, `M2_Macrophages`, and CAFs — all derived from RNA-seq gene expression. At least two of these pairs are expected to be **biologically correlated by construction**:

- `TIS` (Tumour Inflammation Score) and `CYT` (Cytolytic Activity) both measure cytotoxic immune activity and share core genes (`PRF1`, `GZMA`, `GZMB`). Including both may disproportionately weight the cytotoxic immune axis in the GMM — which is confirmed by the empirical PC1 loadings (TIS: +0.46, CYT: +0.45, `CD8_T_cells`: +0.45 — three near-identical loadings).
- The PC1 axis therefore primarily captures one biological construct (cytotoxic immunity) measured three times, not three independent biological dimensions.

**Fix (Priority 2)**: Conduct a Variance Inflation Factor (VIF) analysis on the 9 clustering features. If VIF > 5 for TIS, CYT, or CD8_T_cells against each other, consider retaining only the most mechanistically independent representative (e.g., CYT alone, as it is directly tied to cytolytic killing capacity per Rooney et al. 2015).

## 3.2 Binary Mutation Encoding Does Not Capture Variant Allele Frequency or Co-Mutation Complexity

`mut_BRAF`, `mut_NRAS`, and `mut_NF1` are encoded as binary indicators (0/1). This discards:

- **Variant Allele Frequency (VAF)**: A patient with `BRAF V600E` VAF of 0.85 (clonal, dominant) is biologically distinct from one with VAF 0.12 (subclonal). Both are encoded identically.
- **Co-mutation patterns**: The _Immune Cold_ cluster is 72.7% `BRAF`-mutated and 77.3% `NRAS`-mutated simultaneously. `BRAF`/`NRAS` co-mutation is biologically unusual and may represent data quality artefacts from multi-region sampling rather than a genuine co-driver biology. Phase 3 does not validate this.
- **`NF1` loss-of-function specificity**: `mut_NF1` is a binary indicator, but `NF1` mutations are heterogeneous (missense vs truncating vs deletion). Only truncating/splicing mutations reliably cause loss-of-function consistent with RAS hyperactivation.

**Fix (Priority 2)**: Replace binary mutation indicators with continuous VAF scores where available. For `NF1`, restrict the positive indicator to truncating/splicing/large-deletion variants using the variant classification field in `mutations_cleaned.csv`.

## 3.3 TCGA-SKCM Reference Cohort Dominates Cluster Geometry (63.4% of Patients)

TCGA-SKCM contributes **443 of 699 patients (63.4%)** to the full-cohort clustering. The three ICI-treated clinical trial cohorts (Liu 2019, Hugo 2016, Riaz 2017) contribute only 256 patients combined. The GMM phenotype geometry is therefore primarily shaped by untreated TCGA-SKCM patients, whose tumour microenvironments may differ systematically from anti-PD-1-eligible trial patients:

- TCGA-SKCM includes resected metastatic samples; ICI trials preferentially enrol patients with measurable metastatic disease
- TCGA RNA-seq uses RSEM/FPKM normalisation; iAtlas data uses a different harmonisation pipeline — batch effects between sources are handled by per-cohort Z-score normalisation in Phase 1, but residual technical variation may inflate between-cohort distances relative to within-phenotype biological variation

**Fix (Priority 1)**: Perform a sensitivity analysis: refit the GMM on ICI-trial patients only ($N=256$), and compare phenotype assignments to the full-cohort GMM. If $>85\%$ of patients receive the same phenotype label, the full-cohort geometry is validated. If concordance falls below 85%, the TCGA-SKCM dominance is distorting phenotype boundaries relevant to therapeutic routing.

## 3.4 Spatial Proxy Features Are Estimated, Not Measured

`CAFs` (Cancer-Associated Fibroblasts) and `M2_Macrophages` are deconvolution estimates derived from marker gene expression — not directly measured from histology or flow cytometry. Deconvolution from bulk RNA-seq:

- Assumes a fixed reference signature matrix for each cell type
- Cannot distinguish cells present but transcriptionally silenced from cells absent
- Is confounded by tumour purity (high tumour cell fraction dilutes immune signal)

The biological interpretation that _Immunosuppressive M2-High_ patients have "high M2 macrophages and CAF-mediated stromal exclusion" rests entirely on transcriptomic proxies with no direct cellular validation.

**Fix (Priority 3)**: Cross-reference the M2-High phenotype against available TIMER2.0 or CIBERSORT immune deconvolution estimates in TCGA-SKCM (publicly available) and report the Spearman correlation between Phase 3 deconvolution scores and TIMER2.0 estimates as an external validation metric.

---

# 4. Computational & Data Constraints

> [!NOTE] Infrastructure and Data Constraints in Current Phase 3  
> These are limitations imposed by available data, computational resources, and pipeline design decisions — not methodological errors.

## 4.1 GMM Fitting on 699 Patients with 9 Features: Underdetermined for $K=4$ Full Covariance

A GMM with $K=4$ and full covariance matrices on a $d=9$ dimensional space requires estimating:

$$K \cdot \left[\mu_d + \frac{d(d+1)}{2} + 1 \right] = 4 \cdot [9 + 45 + 1] = 220 \text{ free parameters}$$

With $N=699$ observations, the ratio of observations to parameters is approximately 3.2:1 — technically feasible but at the low end of what is considered statistically adequate ($\geq 5$:1 is a common rule of thumb). The _Immune Cold_ component ($N=22$) is severely below this threshold individually (see §2.3 above).

**Fix (Priority 2)**: Consider a **tied covariance** GMM (all $K$ components share the same $\boldsymbol{\Sigma}$), which reduces the parameter count to $\frac{d(d+1)}{2} + K \cdot (d + 1) = 45 + 40 = 85$ parameters — a 3:1 ratio improvement — as a robustness check.

## 4.2 t-SNE Non-Reproducibility

t-SNE with `perplexity=50` is stochastic. The `tsne_clusters.png` figure will differ across runs even with a fixed `random_state` if the scikit-learn version changes, because the Barnes-Hut approximation implementation is version-sensitive. This means the t-SNE manifold in `phase_3.md` is not scientifically reproducible across environments.

**Fix (Priority 2)**: Replace t-SNE with **UMAP** (`umap-learn` package), which is also non-linear but is deterministic given fixed `random_state` and `n_neighbors`, converges faster, and preserves both local and global structure. Log the UMAP version alongside the figure.

## 4.3 No Held-Out Validation Cohort for Phenotype Stability

Phase 3 fits GMM phenotypes on all 699 available patients. There is no independent external cohort on which to validate that the four discovered phenotypes replicate. The ICI-only sub-cohort evaluation in the log (`Silhouette=0.2387 on N=326`) applies the same fitted model — this is internal validation, not independent replication.

**Fix (Priority 1)**: Identify a published independent melanoma RNA-seq cohort (e.g., Gide et al. 2019, Nathanson et al. 2017) with mutation and bulk transcriptomic data available, apply the fitted `gmm_model.pkl` to project patients into the four phenotypes, and report phenotype prevalence and biomarker profiles. Concordance with the discovery cohort distributions would constitute genuine external validation.

## 4.4 Batch Effects Between TCGA-SKCM and iAtlas Cohorts Are Partially Mitigated But Not Formally Tested

Phase 1 applies per-cohort Z-score normalisation to harmonise TIS/CYT/deconvolution scores across iAtlas (Liu, Hugo, Riaz) and TCGA-SKCM. However, no formal batch correction (e.g., ComBat, Harmony) has been applied, and no UMAP or PCA coloured by cohort (rather than phenotype) has been generated to visualise residual batch structure.

**Fix (Priority 2)**: Generate a PCA projection coloured by `COHORT` rather than phenotype label. If cohorts form visually distinct regions in the same PCA space used for clustering, batch effects are inflating between-cohort variance and may be artificially driving cluster boundaries. Apply ComBat normalisation and re-evaluate clustering stability.

---

# 5. Key Takeaways & Prioritised Improvements

> [!INSIGHT] Phase 3 Limitations: Key Insights
> - **Most Critical (Priority 1)**: The GMM posterior probabilities are effectively degenerate — 99.9% of patients have max-posterior ≥ 0.99 — indicating the binary mutation features are driving near-hard assignments and eliminating the probabilistic uncertainty that justifies GMM over K-Means. This should be addressed before interpreting soft-assignment downstream results.
> - **Statistically Fragile**: _Immune Cold_ ($N=22$) has fewer patients than GMM free parameters for its covariance matrix ($d=9 \Rightarrow 45$ covariance parameters), making its component estimate unreliable.
> - **Cohort Imbalance**: TCGA-SKCM contributes 63.4% of all patients ($N=443$); the phenotype geometry is anchored primarily in untreated TCGA biology, not ICI-trial biology. A sensitivity analysis (ICI-only GMM) is the highest-priority validation step.
> - **Biological Redundancy in Features**: TIS, CYT, and `CD8_T_cells` load near-identically on PC1 (+0.46, +0.45, +0.45), suggesting the immune activation axis is triple-counted. VIF analysis should confirm whether feature reduction improves cluster separation.
> - **t-SNE Non-Reproducibility**: Stochastic across environments — replace with UMAP for deterministic and scientifically reproducible manifold projection.

## Prioritised Fix Schedule

| Priority | Issue | Effort | Expected Impact |
|:--------:|:------|:------:|:----------------|
| **1** | Separate binary mutation features from continuous GMM fit | Medium | Restores genuine soft-assignment uncertainty; meaningful posterior probabilities |
| **1** | GMM BIC curve across $K \in [2,8]$ for formal model selection | Low | Confirms or challenges $K=4$; publishable justification |
| **1** | ICI-only GMM sensitivity analysis ($N=256$) | Low | Validates that TCGA-SKCM dominance is not distorting phenotypes |
| **1** | External cohort validation (apply `gmm_model.pkl` to Gide/Nathanson) | High | Only route to genuine independent replication |
| **2** | VIF analysis on 9 clustering features; drop redundant features | Low | Reduces PC1 triple-counting; improves Silhouette |
| **2** | VAF-weighted or allele-specific `NF1`/`BRAF`/`NRAS` encoding | Medium | Biologically more faithful mutation representation |
| **2** | Permutation test for Silhouette significance | Low | Confirms clustering is non-random vs null |
| **2** | Regularised (diagonal/tied) covariance for small components | Low | Stabilises _Immune Cold_ covariance estimate |
| **2** | Replace t-SNE with UMAP (deterministic, reproducible) | Low | Reproducible manifold; better global structure preservation |
| **2** | PCA coloured by cohort to assess batch structure | Low | Diagnoses TCGA-SKCM batch contamination of cluster geometry |
| **3** | Implement `plot_radar_chart()` and `plot_cluster_heatmap()` stubs | Medium | Completes intended phenotype visualisation suite |
| **3** | Remove duplicate `get_phenotype_color()` from `src/styles.py` L53 | Low | Eliminates silent dead code |
| **3** | Deprecate `run_kmeans()` / `kmeans_model.pkl` legacy fallback | Low | Removes naming confusion between K-Means and GMM |
