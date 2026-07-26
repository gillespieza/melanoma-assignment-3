# Q5: Student Biological & Statistical Study Guide 📚

A comprehensive, student-focused reference explaining the **biology**, **statistics**, **machine learning**, and **dataset logic** behind Question 5.

---

## Table of Contents
1. [The Clinical Problem & Project Goal](#1-the-clinical-problem--project-goal)
2. [Biological Concepts Explained](#2-biological-concepts-explained)
   - [Immune Checkpoints & T-Cell Infiltration](#immune-checkpoints--t-cell-infiltration)
   - [Macrophage Polarisation (M1 vs. M2) & Macrophage STV](#macrophage-polarisation-m1-vs-m2--macrophage-stv)
   - [Mechanisms of Immunotherapy Resistance](#mechanisms-of-immunotherapy-resistance)
3. [Statistical & Machine Learning Concepts Explained](#3-statistical--machine-learning-concepts-explained)
   - [Univariate Association Testing & Youden Cutoffs](#univariate-association-testing--youden-cutoffs)
   - [Logistic Regression Interaction Terms](#logistic-regression-interaction-terms)
   - [Unsupervised Clustering (K-Means, Ward, Silhouette, GAP, UMAP)](#unsupervised-clustering)
   - [Decision Curve Analysis (DCA), NNT & PPV](#decision-curve-analysis-dca-nnt--ppv)
4. [Dataset Strategy: `merged/immunotherapy` vs. `merged/full`](#4-dataset-strategy-mergedimmunotherapy-vs-mergedfull)
5. [The 3-Arm Decision Tree Architecture](#5-the-3-arm-decision-tree-architecture)

---

## 1. The Clinical Problem & Project Goal

In cutaneous melanoma, patients diagnosed with advanced (Stage III/IV) disease have three main treatment avenues:

1. **Arm A — Immunotherapy**: Immune Checkpoint Inhibitors (ICI) such as Anti-PD-1 (*Nivolumab*, *Pembrolizumab*) or Anti-CTLA-4 (*Ipilimumab*). These drugs release the molecular "brakes" on cytotoxic CD8+ T cells so they can destroy tumour cells.
2. **Arm B — Targeted Therapy**: Small-molecule kinase inhibitors (BRAF inhibitor *Dabrafenib* + MEK inhibitor *Trametinib*). These selectively block hyperactive MAPK pathway signaling in tumours harboring a **`BRAF` V600E/K mutation**.
3. **Arm C — Chemotherapy & Combination Therapy**: Cytotoxic chemotherapy (*Dacarbazine*) or combination regimens (e.g., M2-macrophage depletion / epigenetic priming combined with anti-PD-1).

### The Challenge
If immunotherapy is given indiscriminately to all patients, only $\sim 35–40\%$ respond. For non-responders, time is wasted while the tumour progresses, and patients experience severe immune-related adverse events.

### The Q5 Objective
Question 5 builds a **biomarker-guided decision support system**. By inspecting a patient's baseline molecular profile (gene expression signatures, M1/M2 ratio, driver mutations like `BRAF`, `NRAS`, `NF1`, and mutational burden), the pipeline determines whether the patient will respond to Immunotherapy (Arm A), should switch to Targeted Therapy (Arm B), or requires Combination/Chemotherapy (Arm C).

---

## 2. Biological Concepts Explained

### Immune Checkpoints & T-Cell Infiltration
* **CD8+ Cytotoxic T Cells**: The primary immune effector cells capable of killing cancer cells.
* **PD-1 (`PDCD1`) & PD-L1 (`CD274`)**: PD-1 is a receptor expressed on T cells. Tumours express PD-L1 (`CD274`), which binds to PD-1 (`PDCD1`) and sends an inhibitory "don't kill me" signal to the T cell. Anti-PD-1 antibodies block this binding, restoring T-cell cytotoxicity.
* **Tumour Inflammation Signature (TIS)**: An 18-gene expression signature measuring pre-existing adaptive immune suppression and active IFN-$\gamma$ signaling in the tumour microenvironment.
* **Cytolytic Index (CYT)**: Defined as the geometric mean of Perforin 1 (`PRF1`) and Granzyme A (`GZMA`) transcript levels, quantifying active cytotoxic T-cell and NK-cell killing.

### Macrophage Polarisation (M1 vs. M2) & Macrophage STV
Tumour-Associated Macrophages (TAMs) exist along a functional spectrum:

* **M1 Macrophages (Pro-inflammatory / Antitumour)**: Produce pro-inflammatory cytokines (`TNF`, `IL12B`, `CXCL10`) and assist T cells in tumour destruction.
* **M2 Macrophages (Anti-inflammatory / Pro-tumour)**: Produce immunosuppressive factors (`IL10`, `TGFB1`, `ARG1`), stimulate angiogenesis, and recruit regulatory T cells (Tregs), blocking anti-PD-1 efficacy.

#### The Macrophage STV Matrix ([m1_m2_stv.csv](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/data/config/m1_m2_stv.csv))
The project utilizes a linear **Signature Transcript Vector (STV)** containing 14,837 gene weights ($W_g$). For patient $i$ with $\log_2$-transformed expression $E_{i,g}$:

$$\text{STV Score}_i = \sum_{g=1}^{14837} W_g \cdot E_{i,g}$$

* Positive weights correspond to M1-associated features; negative weights correspond to M2-associated features.
* The normalized score yields the **M1/M2 Ratio** ($M1 / [M1 + M2]$). High ratios reflect an inflamed, M1-dominant microenvironment; low ratios indicate M2-mediated immune exclusion.

### Mechanisms of Immunotherapy Resistance
Non-response to checkpoint blockade typically stems from four biological barriers:
1. **Antigen Presentation Defect**: Loss-of-function mutations or silencing in $\beta_2$-microglobulin (`B2M`) or transporter genes (`TAP1`, `TAP2`, `HLA-A`, `HLA-B`, `HLA-C`) prevent HLA class I molecules from displaying tumour neoantigens on the cell surface.
2. **IFN-$\gamma$ Pathway Resistance**: Loss of `JAK1`, `JAK2`, `STAT1`, or `STAT3` prevents tumour cells from responding to interferon-gamma, making them insensitive to T-cell killing.
3. **Immunosuppressive Stroma**: High expression of `IDO1`, `TGFB1`, or `ARG1` creates a metabolic barrier that depletes essential amino acids (tryptophan, arginine) needed for T-cell survival.
4. **Copy Number Alteration (CNA) Burden**: High chromosomal instability / aneuploidy physically disrupts T-cell infiltration into the tumour core.

---

## 3. Statistical & Machine Learning Concepts Explained

### Univariate Association Testing & Youden Cutoffs
To evaluate individual biomarkers ($X$) against binary response ($Y \in \{0, 1\}$):

* **Mann-Whitney U Test (Continuous Features)**: A non-parametric test assessing whether the distribution of signature scores (e.g. TIS, CYT) differs significantly between Responders and Non-Responders without assuming normal distributions.
* **Cohen's $d$ (Effect Size)**: Quantifies the standardized mean difference between groups:
  $$d = \frac{\bar{X}_1 - \bar{X}_0}{s_{\text{pooled}}}$$
  Values of $d > 0.8$ indicate large biological effect sizes.
* **Fisher's Exact Test (Categorical Features)**: Used for contingency tables (e.g., `BRAF` mutation status vs. Response) to calculate exact $p$-values, Odds Ratios (OR), and 95% confidence intervals.
* **Youden's J Statistic (Optimal Threshold)**: Identifies the numeric cutoff $c^*$ on a continuous feature that maximizes diagnostic accuracy:
  $$J(c) = \text{Sensitivity}(c) + \text{Specificity}(c) - 1$$
  The cutoff $c^*$ corresponding to $\max J(c)$ is used to discretize continuous signatures into clinical high/low categories.

### Logistic Regression Interaction Terms
To test whether two features act synergistically:

$$\text{logit}(P(Y=1)) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 (X_1 \cdot X_2)$$

A statistically significant interaction coefficient ($\beta_3 \ne 0, p < 0.05$) indicates that the predictive effect of feature $X_1$ (e.g., TIS score) depends on the status of feature $X_2$ (e.g., `BRAF` mutation status).

### Unsupervised Clustering

#### Feature Normalization (`StandardScaler`)
Before clustering, continuous features are standardized to zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$). This prevents features with large raw scales (e.g. TMB) from dominating distance calculations over bounded signature scores.

#### K-Means vs. Hierarchical Agglomerative (Ward) Clustering
* **K-Means**: Iteratively assigns patients to $K$ clusters by minimizing total within-cluster variance (inertia).
* **Agglomerative (Ward Linkage)**: A bottom-up hierarchical approach that merges pairs of clusters to minimize the increase in total within-cluster variance at each step.

#### Cluster Validation Metrics
* **Silhouette Coefficient ($s$)**: For patient $i$, measures how similar $i$ is to its own cluster ($a_i$) compared to the nearest neighboring cluster ($b_i$):
  $$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}, \quad s_i \in [-1, 1]$$
  Mean silhouette score $>0.5$ indicates strong cluster structure.
* **GAP Statistic**: Compares the total within-cluster variation $W_k$ against expected null values $W_k^*$ generated by Monte Carlo sampling from a uniform distribution. The optimal $K$ maximizes $\text{Gap}(K) = E^*[\log(W_k)] - \log(W_k)$.
* **UMAP (Uniform Manifold Approximation and Projection)**: A non-linear dimensionality reduction technique that projects high-dimensional signature space down to 2D for visual inspection while preserving local and global data topology.

### Decision Curve Analysis (DCA), NNT & PPV
Evaluating a model using AUC-ROC alone is insufficient for clinical adoption because AUC does not account for the real-world clinical costs of false positives vs. false negatives.

#### Net Benefit Formula
For a decision threshold probability $p_t \in [0, 1]$ (the probability threshold at which a doctor recommends treatment):

$$\text{Net Benefit} = \frac{\text{True Positives}}{N} - \left( \frac{\text{False Positives}}{N} \right) \times \left( \frac{p_t}{1 - p_t} \right)$$

* **Interpretation**: If a model's DCA curve lies above both the "Treat All" and "Treat None" strategies across relevant thresholds ($p_t = 0.2 - 0.6$), using the model to guide treatment decisions adds net clinical benefit.

#### Clinical Utility Metrics
* **Number Needed to Treat (NNT)**: The number of patients who must be treated according to the model to achieve one additional positive response:
  $$\text{NNT} = \frac{1}{\text{Absolute Risk Reduction}}$$
* **Positive Predictive Value (PPV)**: The probability that a patient predicted as "Responder" actually achieves clinical response ($CR/PR$).

---

## 4. Dataset Strategy: `merged/immunotherapy` vs. `merged/full`

Understanding why we use two merged datasets resolves common confusion:

```
┌──────────────────────────────────────────────────────────────────────────┐
│              merged/immunotherapy (N = 326 Patients)                     │
│  • Cohorts: Liu 2019 (122), Riaz 2017 (107), TCGA-IO (70), Hugo 2016 (27) │
│  • Property: Every patient received anti-PD-1 / anti-CTLA-4 immunotherapy  │
│  • Labels: 100% annotated with ground-truth Response (y ∈ {0, 1})         │
│  • Purpose: Model Training, Phenotype Discovery, & Validation            │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  merged/full (N = 699 Patients)                          │
│  • Cohorts: TCGA-SKCM Full (443), Liu (122), Riaz (107), Hugo (27)        │
│  • Property: Includes 373 unselected TCGA primary/surgical patients       │
│  • Labels: Response labels missing (NaN) for non-immunotherapy patients  │
│  • Purpose: Real-world 3-Arm Treatment Selection Simulation              │
└──────────────────────────────────────────────────────────────────────────┘
```

1. **Why Train on `merged/immunotherapy` ($N=326$)**:
   To fit machine learning classifiers ($\hat{y} = f(X)$) or identify response-associated phenotypes, we must have ground-truth response labels ($y$). `merged/immunotherapy` is the only dataset where 100% of patients have verified immunotherapy outcomes.

2. **Why Deploy on `merged/full` ($N=699$)**:
   When a random new patient arrives at the clinic, we do not know if they will respond to immunotherapy. `merged/full` represents the general, unselected melanoma population where Q5's 3-arm decision tree assigns patients to Immunotherapy, Targeted Therapy, or Chemotherapy.

---

## 5. The 3-Arm Decision Tree Architecture

The master decision engine routes patients through three sequential evaluation gates:

```
                  ┌────────────────────────────────────────┐
                  │          RANDOM NEW PATIENT            │
                  │   (Tumour Gene Expression & DNA)       │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
             [Gate 1] Is Immunotherapy Chance High? (>70% prob)
               ├── YES ──► 🟢 ARM A: Immunotherapy Monotherapy (Anti-PD-1)
               └── NO  ──► Proceed to Gate 2
                                      │
                                      ▼
             [Gate 2] Is the `BRAF` Gene Broken? (`BRAF` V600 Mutation)
               ├── YES ──► 🔵 ARM B: Targeted Therapy (Dabrafenib + Trametinib)
               └── NO  ──► Proceed to Gate 3
                                      │
                                      ▼
             [Gate 3] Treatability Index & Q4 Target Nomination
               ├── High Treatability (Reversible Barrier) ──► 🟡 COMBINATION (Helper Drug + IO)
               └── Low Treatability (High Resistance)     ──► 🔴 ARM C: Chemotherapy (Dacarbazine) / Trial
```

### Summary of Script Mapping

* **[scripts/01_load_and_prepare.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/01_load_and_prepare.py)**: Loads data, extracts signatures, applies Macrophage STV, computes deconvolution.
* **[scripts/02_feature_analysis.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/02_feature_analysis.py)**: Runs MW-U, Fisher's, Youden cutoffs, and interaction tests.
* **[scripts/03_cluster_patients.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/03_cluster_patients.py)**: Performs K-Means/Ward scaling, silhouette/GAP evaluation, and UMAP projection.
* **[scripts/04_phenotype_characterisation.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/04_phenotype_characterisation.py)**: Annotates phenotypes (*Immune Hot*, *Cold*, *M2-High*, *Mutant-Driven*), KM survival, and Q3 ODE trajectories.
* **[scripts/05_subgroup_models.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/05_subgroup_models.py)**: Trains subgroup-specific models using LOCO cross-validation.
* **[scripts/06_clinical_utility.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/06_clinical_utility.py)**: Computes DCA net benefit curves, NNT, and PPV.
* **[scripts/07_treatability_scoring.py](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/07_treatability_scoring.py)**: Scores treatability index, maps Q4 DepMap/LINCS drug targets, and constructs the 3-arm decision tree.
