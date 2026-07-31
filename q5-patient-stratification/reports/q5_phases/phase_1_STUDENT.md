---
title: "Phase 1: Feature Engineering & Baseline Signature Distribution"
aliases:
  - Phase 1 Reference
  - Q5 Phase 1 Guide
tags:
  - melanoma
  - patient-stratification
  - phase-1
  - q5
created: 2026-07-31 11:00
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-31 12:22
---

# Phase 1: Feature Engineering & Baseline Signature Distribution 📊

A clear, intuitive reference explaining **Phase 1** of Question 5: How we take thousands of raw tumour genes and turn them into neat, easy-to-understand immune scorecards.

## 1. The High-Dimensional RNA-Seq Challenge & Feature Engineering 🧬

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Loading multi-modal clinical, transcriptomic, and genomic datasets to engineer 17 biologically interpretable features across dual cohort feature matrices ($N_{\text{ICI}} = 326$ and $N_{\text{Full}} = 699$).
> - **Why we are doing it**: Raw bulk RNA-seq gene expression matrices contain ~19,757 genes per sample. High-dimensional feature spaces ($p \gg n$) cause severe overfitting, collinearity, and loss of biological interpretability. Projecting expression profiles onto validated immune signature scores and cell deconvolution fractions compresses noise into actionable clinical signals.
> - **What question it answers**: How do we clean, harmonise, and condense thousands of complex tumour genes into structured biological feature matrices that downstream response prediction models and patient clustering algorithms can learn from?

### The Dimensionality Reduction Strategy
- **The $p \gg n$ Problem**: Raw sequencing measures ~19,757 protein-coding genes per tumour sample across $N=699$ patients. Feeding raw gene matrices directly into machine learning algorithms creates severe curse-of-dimensionality issues and redundant feature weights.
- **The Biological Solution**: Instead of evaluating unconstrained individual genes, we project gene expression onto **17 biologically validated axes** (6 immune signatures, 4 macrophage polarisation scores, 7 cell deconvolution fractions) and **9 genomic/TMB markers**.
- **Translational Relevance**: Clinicians cannot interpret arbitrary weights across thousands of genes, but they can easily interpret established biomarkers like Tumour Inflammation Signature (`TIS`), Cytolytic Index (`CYT`), or `CD274` (`PD-L1`) expression.

## 2. Dual Feature Matrix Dataflow Architecture 🏫

To support distinct machine learning requirements across downstream analytical phases, Phase 1 constructs two separate feature matrices:

```
┌──────────────────────────────────────────────────────────┐
│      MATRIX 1: feature_matrix.csv (N_ICI = 326)          │
│            "THE RESPONSE-SUPERVISED COHORT"              │
├──────────────────────────────────────────────────────────┤
│ • Immunotherapy-treated patients (Liu, Riaz, Hugo, TCGA) │
│ • 100% annotated with RECIST response labels             │
│ • Powers: Phase 2 (Youden Cutoffs), Phase 5 (Predictive  │
│   Modelling), and Phase 6 (Decision Curve Analysis)      │
└──────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│    MATRIX 2: feature_matrix_full.csv (N_Full = 699)      │
│            "THE UNSELECTED MELANOMA POPULATION"          │
├──────────────────────────────────────────────────────────┤
│ • All melanoma patients (adds untreated TCGA-SKCM, N=443)│
│ • Captures total population-level biological diversity   │
│ • Powers: Phase 3 (Clustering), Phase 4 (Phenotypes),    │
│   and Phase 7 (3-Arm Decision Engine & Treatability)     │
└──────────────────────────────────────────────────────────┘
```

### Why Two Datasets Are Necessary
1. **Supervised Model Training ($N_{\text{ICI}} = 326$)**:
   - To fit predictive classifiers or determine diagnostic sensitivity/specificity cutoffs, algorithms require ground-truth treatment outcome labels (`Responder` vs `Non-Responder`).
   - `feature_matrix.csv` strictly isolates patients who received anti-PD-1 or anti-CTLA-4 immunotherapy across 4 trial cohorts (_Liu 2019_, _Riaz 2017_, _Hugo 2016_, and ICI-treated _TCGA-SKCM_).
2. **Unsupervised Stratification & Real-World Simulation ($N_{\text{Full}} = 699$)**:
   - In clinical practice, new primary or surgical patients arrive before receiving systemic therapy.
   - `feature_matrix_full.csv` merges trial cohorts with the complete reference cohort (_TCGA-SKCM_, $N=443$), capturing population-wide biological spectrums without biasing unsupervised clustering toward pre-treated trial populations.

## 3. Microenvironment Deconvolution: Cops, Soldiers & Barriers 🚓

Tumour tissues are heterogeneous microenvironments composed of malignant cells, infiltrating immune cells, and dense stromal barriers. We can understand the microenvironment by looking at four key biological roles:

- 👮 **M1 Macrophages ("Good Cops" — Antitumour Inflammation)**:
  - Express pro-inflammatory markers (`TNF`, `IL12B`, `CXCL10`, `NOS2`, `IRF5`). They act like vigilant police officers that actively swallow tumour debris and present antigens, sounding the alarm to recruit cytotoxic T cells.
- 🦹 **M2 Macrophages ("Double Agents" — Immunosuppressive Shielding)**:
  - Express anti-inflammatory markers (`CD163`, `MRC1`, `MSR1`, `TGFB1`, `ARG1`). Rather than attacking cancer, they act like corrupt double agents that secretively release immunosuppressive cytokines, promote tumour blood vessel growth (angiogenesis), and disarm infiltrating T cells.
- 🪖 **CD8+ Cytotoxic T Cells ("T-Cell Soldiers" — Frontline Effector Cells)**:
  - Measured by `CD8A`, `CD8B`, `CD3D`, and `CD3E`. These are the elite frontline soldiers capable of destroying malignant cells. Anti-PD-1 (`CD274`) immunotherapy works by unbinding checkpoint breaks on these soldiers so they can release cytolytic weapons (Perforin `PRF1` and Granzyme A `GZMA`).
- 🛡️ **Cancer-Associated Fibroblasts (`CAFs`, "Scar Tissue Physical Barriers")**:
  - Express stromal markers (`FAP`, `PDGFRB`, `COL1A1`, `ACTA2`). They deposit dense collagen scar tissue around the tumour, creating a physical fortress wall that prevents T-cell soldiers from penetrating into the tumour core (immune exclusion).

### How Transcriptomic Cell Deconvolution Was Determined
Think of bulk tumour RNA sequencing like taking a complex fruit smoothie and figuring out how much apple, banana, and strawberry went into it!

Phase 1 determines cell type abundances using two complementary approaches in `deconvolution.py`:

1. **The Average Marker Scorecard Method (`compute_cell_deconvolution`)**:
   - For each cell type (e.g. CD8+ T cells or CAFs), we select a panel of 3–5 specific marker genes that only that cell type produces in high amounts.
   - We calculate each patient's score by taking the average expression level across those marker genes.
   - **Key Marker Panels**: `CD8_T_cells` (`CD8A`, `CD8B`), `NK_cells` (`NCAM1`), `B_cells` (`CD19`), `M1_Macrophages` (`TNF`), `M2_Macrophages` (`CD163`), and `CAFs` (`FAP`, `PDGFRB`, `COL1A1`).

2. **The Macrophage Polarisation Scorecard (`compute_macrophage_stv`)**:
   - Rather than checking only 3 markers, we compare patient expression against a 14,837-gene scorecard (`m1_m2_stv.csv`).
   - Genes with positive weights build the **M1 Score** ("Good Cop"), while negative weights build the **M2 Score** ("Double Agent").
   - Combining these produces the **M1/M2 Ratio** (ranging from 0.0 to 1.0), showing whether the tumour microenvironment is dominated by antitumour inflammation or immunosuppressive stromal resistance.
   - **High M1/M2 Ratio**: M1-dominated, inflamed microenvironment highly responsive to anti-PD-1 monotherapy.
   - **Low M1/M2 Ratio**: M2-dominated stromal suppression. Even if total T-cell density is moderate, M2 dominance disarms T-cell soldiers, signaling a requirement for combination helper therapies.

## 4. Complete Feature Inventory Breakdown 📋

Phase 1 condenses ~19,757 genes into 17 transcriptomic features and 9 genomic/TMB features:

| Feature Category | Features Included | Primary Biological Signalling / Function |
| :--- | :--- | :--- |
| **Core Immune Signatures (6)** | `TIS`, `CYT`, `IFN_gamma`, `CD8_Tcell`, `IMPRES`, `PD_L1` (`CD274`) | Quantifies active IFN-$\gamma$ signalling, cytotoxic killing capability (`PRF1`, `GZMA`), and checkpoint target expression. |
| **Macrophage STV (4)** | `M1_score`, `M2_score`, `M1_M2_Ratio`, `Macrophage_STV_Score` | Resolves functional macrophage polarisation using a 14,837-gene linear vector. |
| **Cell Deconvolution (7)** | `CD8_T_cells`, `CD4_T_cells`, `NK_cells`, `B_cells`, `M1_Macrophages`, `M2_Macrophages`, `CAFs` | Marker-based relative cell abundance estimates distinguishing immune infiltrates from stromal fibroblasts. |
| **Genomic & Neoantigens (9)** | `mut_BRAF`, `mut_NRAS`, `mut_NF1`, `TMB_NONSYNONYMOUS`, Neoantigen counts | Identifies actionable oncogenic drivers (MAPK pathway activation) and mutational neoantigen load. |

## 5. Visualising Baseline Feature Distributions 📈

![Baseline Biomarker Feature Distributions](q5-patient-stratification/plots/phenotypes/baseline_response_violins.png)

> [!INFO] Baseline Biomarker Observations
> - **Immune Hot Characteristics**: Immunotherapy Responders (CR/PR) exhibit significantly higher baseline levels of `TIS`, `CYT`, `IFN_gamma`, and `CD8_Tcell` than Non-Responders (PD).
> - **Stromal Exclusion Mechanisms**: Non-responders frequently present with elevated `CAFs` (scar tissue walls) or low `M1_M2_Ratio` ("Double Agent" dominance), demonstrating that high stromal density or M2 macrophage polarization suppresses immune efficacy even in tumours with modest T-cell presence.

> [!IMPORTANT] Key Takeaways
> - **Feature Compression**: Transformed raw high-dimensional transcriptomics (~19,757 genes) into 17 interpretable biological signatures across 26 total clinical, genomic, and microenvironmental features.
> - **Dual Cohort Architecture**: Standardized `feature_matrix.csv` ($N_{\text{ICI}} = 326$) for response prediction models and `feature_matrix_full.csv` ($N_{\text{Full}} = 699$) for unsupervised stratification and 3-arm decision tree routing.
> - **Microenvironment Profiling**: Captured stromal immune exclusion via the 14,837-gene Macrophage STV score, providing a quantitative basis for identifying combination-therapy candidates.

> [!WARNING] Key Phase 1 Limitations to Keep in Mind
> - **Relative Scores vs True Percentages**: Deconvolution uses mean marker log-expression (`compute_cell_deconvolution`), producing relative arbitrary scores rather than true constrained $0–100\%$ cell percentage proportions (like CIBERSORTx).
> - **Spatial Blindness**: Bulk RNA-seq blends the whole biopsy together. It cannot tell if T-cell soldiers are physically inside the tumour nest (_Immune Hot_) or trapped outside in the scar tissue wall (_Immune Excluded_).
> - **Indirect DNA Proxies**: Because trial cohorts lack Copy Number Alteration (CNA) GISTIC files, Phase 1 relies on Tumour Mutational Burden (`TMB_NONSYNONYMOUS`) and single-gene transcript levels (`PTEN`, `CDKN2A`) as indirect proxies for chromosomal instability.
> - **1D Macrophage Axis**: The `M1_M2_Ratio` models macrophages as a single "Good Cop vs Double Agent" spectrum, whereas real macrophages exist in multi-dimensional states (e.g., M2a, M2c, lipid-laden).
> - **Static Snapshot**: Pre-treatment biopsies cannot capture early dynamic immune changes occurring 2–4 weeks after starting anti-PD-1 (`CD274`) treatment.
