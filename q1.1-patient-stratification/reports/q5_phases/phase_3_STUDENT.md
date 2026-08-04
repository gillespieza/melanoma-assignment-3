---
title: "Phase 3: Discovering Patient Subtypes in Melanoma — Unsupervised Tumour Microenvironment Stratification"
aliases:
  - Q5 Phase 3 Student Explainer
  - Phase 3 Patient Clustering Overview
tags:
  - melanoma
  - patient-stratification
  - phase-3
  - q5
  - clustering
  - tumour-microenvironment
created: 2026-08-01 16:08
cssclasses:
  - table-small
  - table-center
  - row-alt
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-01 16:08
---

# Phase 3: Discovering Patient Subtypes in Melanoma

> [!NOTE] What This Phase Does — and Why It Matters
> - **What is being done**: Phase 3 applies unsupervised machine learning to group 699 melanoma patients into biologically distinct tumour microenvironment (TME) subtypes — without using treatment outcome labels.
> - **Why we are doing it**: Before we can recommend the right therapy for any patient, we need to know what kind of tumour microenvironment they have. A patient with an immune-cold, T-cell-excluded tumour needs a completely different therapeutic strategy than one with a highly inflamed, immune-hot tumour.
> - **What question it answers**: Can we cluster patients based purely on their immune and stromal gene expression profiles, and do those clusters reveal clinically meaningful biological archetypes?

---

## Phase Overview: Grouping Patients by Their Tumour's Immune Landscape

Imagine the tumour microenvironment as a battlefield. Some tumours are teeming with immune soldiers (CD8+ T-cells, M1 macrophages); others have built up defensive fortifications (immunosuppressive M2 macrophages and cancer-associated fibroblasts) that repel the immune system; and some have genetically hijacked their own growth programme entirely, making the battlefield a secondary concern.

Phase 3 asks a deceptively simple question: **if we listen only to what a tumour's genes are saying about its immune landscape, can we detect these distinct battlefield archetypes?**

The approach uses a **Two-Stage Patient Stratification** strategy:

- **Stage 1 — Probabilistic Immune Clustering**: A Gaussian Mixture Model (GMM) clusters patients across six core immune and stromal features: Tumour Inflammation Signature (TIS), Cytolytic Activity (CYT), CD8+ T-cell abundance, M1 macrophages, M2 macrophages, and cancer-associated fibroblasts (CAFs). GMM is chosen over K-Means because it assigns each patient a *probability* of belonging to each subtype — recognising that biology is continuous, not binary. This stage produces three biologically interpretable immune clusters.

- **Stage 2 — Deterministic Genomic Refinement**: One cluster contains a subset of patients whose tumours are almost completely driven by `NF1` loss-of-function mutations. These patients share a distinct biological programme — RAS pathway hyperactivation, very high mutational burden, and a therapeutic response profile unlike any immune-driven subtype. A targeted post-hoc split extracts them as a fourth independent phenotype.

---

## Upstream Integration: What Phase 3 Inherits

> [!NOTE] Building on Phases 1 & 2
> - **Phase 1 (Data Loading & Feature Engineering)**: Phase 3 consumes the 33-feature multi-modal matrix constructed in Phase 1. Critically, it uses only the six continuous immune and stromal features in Stage 1, reserving mutation binary flags for Stage 2. The Phase 1 pipeline already computed TIS, CYT, macrophage polarisation scores, CAF estimates, and spatial proximity proxies — all of which feed directly into the GMM.
> - **Phase 2 (Feature Analysis & Biomarker Credibility)**: Phase 2 established which features best separate immunotherapy responders from non-responders via Mann-Whitney U tests, Fisher's exact tests, and Youden cutoff optimisation. This gave biological confidence that TIS and CYT — the top Phase 2 discriminators — are the right anchors for immune-based subtyping in Phase 3.
> - **Q1 Predictive Models**: The Q1 logistic regression and random forest models trained on TIS, CYT, and TMB provide external validation context: patients assigned to `Immune Hot` by Phase 3 should, in principle, score highest on Q1's probability-of-response scale — a key downstream cross-validation check.

---

## Clinical Relevance: Why Patient Subtyping is the Foundation of Personalised Oncology

> [!NOTE] From Population Averages to Individual Decisions
> - **The clinical problem**: Anti-PD-1 immunotherapy works brilliantly for some melanoma patients but provides zero benefit — and real toxicity — for others. Across the pooled trial cohorts in this study, overall response rates hover around 40–50%. Giving every patient the same drug based on a population average is equivalent to prescribing the same prescription to everyone who walks into a pharmacy.
> - **What patient subtyping unlocks**: By identifying biologically distinct microenvironment subtypes, we can route patients to therapies matched to their tumour's specific vulnerabilities. An inflamed `Immune Hot` tumour should respond to PD-1 blockade. A stromal-excluded `Immunosuppressive M2-High` tumour may first need stromal remodelling agents. A `Mutant-Driven` tumour with extreme mutational burden is a candidate for ICI plus MEK inhibition targeting the hyperactive RAS pathway.
> - **The Q5 master goal**: Phase 3 is the centrepiece of the Q5 synthesis pipeline. Its four phenotype labels propagate forward into Phase 4 (ODE tumour-immune trajectory modelling), Phase 5 (subgroup-specific predictive models), Phase 6 (clinical utility and net benefit analysis), and Phase 7 (treatability scoring and drug nomination).

---

## Key Phase Results: The Four Discovered Phenotypes

The Two-Stage Stratification identified four biologically coherent tumour microenvironment archetypes across 699 melanoma patients:

| Phenotype | N (%) | Biological Signature | `BRAF`+ | `NRAS`+ | `NF1`+ | Median TMB | High-TMB (≥10 mut/Mb) |
|---|---|---|---|---|---|---|---|
| **Immune Hot** | 341 (48.8%) | High TIS, high CYT, CD8+ T-cell infiltrated, inflamed | 49.6% | 25.2% | 12.6% | ~13 mut/Mb | 58.9% |
| **Immunosuppressive M2-High** | 256 (36.6%) | High M2 macrophages & CAF stromal exclusion, low TIS | 44.1% | 29.3% | 0.0% | ~11 mut/Mb | 52.5% |
| **Mutant-Driven** | 57 (8.2%) | 100% `NF1` loss, RAS hyperactivation, very high mutational burden | 29.8% | 31.6% | 100.0% | ~41 mut/Mb | 87.7% |
| **Immune Cold** | 45 (6.4%) | T-cell desert, low TIS, low CYT, minimal infiltration | 53.3% | 17.8% | 13.3% | ~9 mut/Mb | 48.8% |

> [!INSIGHT] Key Biological Insights from Phase 3
> - **Immune inflammation is multi-driver**: The `Immune Hot` subtype shows robust T-cell infiltration across `BRAF`+ (49.6%), `NRAS`+ (25.2%), and `NF1`+ (12.6%) patients. This disproves the assumption that strong anti-tumour immunity is limited to any single driver mutation genotype — inflammatory TME signalling arises regardless of the upstream oncogenic hit.
> - **`NF1` loss defines a genomically extreme subtype**: The `Mutant-Driven` group ($N=57$, $8.2\%$) carries a median TMB of approximately 41 mutations per megabase — more than three times higher than the `Immune Hot` group (~13 mut/Mb) and five times higher than the `Immune Cold` group (~9 mut/Mb). This extreme hypermutation load makes `Mutant-Driven` patients particularly likely to generate tumour-specific neo-antigens recognisable by the immune system, supporting checkpoint blockade plus MEK inhibitor combination strategies.
> - **Stromal exclusion, not immune dysfunction, drives M2-High resistance**: The `Immunosuppressive M2-High` cluster ($N=256$, $36.6\%$) is not T-cell deficient by nature — it is T-cell *excluded* by an M2 macrophage and CAF-mediated stromal barrier. This distinction is clinically important: the therapeutic target is the stromal compartment, not T-cell priming.
> - **`Immune Cold` is a rare but clinically challenging subtype**: At $N=45$ ($6.4\%$), this group lacks both immune infiltration (low TIS, low CYT) and the genomic hypermutation that makes `Mutant-Driven` patients immunologically visible. Existing checkpoint blockade strategies are unlikely to benefit this subtype without prior immune priming interventions.
> - **Soft probabilistic boundaries reflect biological reality**: GMM assigns each patient a probability of belonging to each phenotype rather than a hard binary label. Approximately 15–18% of patients sit near phenotype boundaries with assignment confidence below 70%, reflecting genuine biological continuity between subtypes — not classification error.

---

## Limitations & Future Directions

> [!WARNING] Current Methodological Boundaries
> The following limitations are derived from the Phase 3 methodological audit and bound the generalisability of these findings.

**The clusters are not perfectly separable.** An independent benchmarking experiment compared GMM against Spectral Manifold clustering (a graph-based method that captures curved, non-linear cluster shapes). Spectral clustering achieved better geometric separation across all three internal quality metrics. This tells us that the true structure of patient TME space is not perfectly described by elliptical Gaussian blobs — there are curved, ribbon-like transitions between archetypes that GMM partially misses.

**The `Mutant-Driven` split is a heuristic, not a probabilistic model.** Stage 2 is a deterministic rule: if a patient is `NF1`-positive and sits within the M2-High cluster, they are reclassified as `Mutant-Driven`. While biologically well-motivated, this does not produce a smooth probability estimate for `NF1` involvement — it is either zero or one. A fully Bayesian model would integrate continuous `NF1` mutation allele fraction with immune feature density for a more nuanced assignment.

**Spatial and immune cell estimates come from bulk tissue.** The macrophage polarisation scores, CAF abundances, and spatial proximity indicators are all inferred from bulk RNA-seq signal averaged across millions of cells in a tissue biopsy. They cannot resolve cell-cell contact networks, intratumoural spatial heterogeneity, or rare cell subpopulations that single-cell or spatial proteomics technologies can detect directly.

**The smallest phenotypes carry the highest uncertainty.** With $N=45$ (`Immune Cold`) and $N=57$ (`Mutant-Driven`) patients, subgroup-specific analyses in later pipeline phases have limited statistical power. Validation in independent, larger cohorts is essential before clinical translation.

> [!TIP] Priority Future Directions
> 1. **Validate phenotypes in an independent external cohort** — the most critical next step for clinical translation confidence.
> 2. **Explore Variational Bayesian or Graph Neural Network clustering** to better capture non-Gaussian manifold structure in TME feature space.
> 3. **Integrate `NF1` allele frequency continuously** into the probabilistic model rather than as a binary post-hoc split.
> 4. **Validate spatial proxy indicators** against CODEX or IMC multiplexed spatial proteomics data, where available, to confirm that transcriptomic distance ratios reflect true tissue architecture.
