---
title: "Phase 4: Phenotype Characterisation & Dynamic Tumour-Immune Trajectories"
aliases:
  - Phase 4 Explainer
  - Q5 Phase 4 Overview
tags:
  - melanoma
  - patient-stratification
  - phase-4
  - q5
  - ode-modelling
  - phenotyping
  - immunotherapy
created: 2026-07-31 15:38
cssclasses:
  - row-alt
  - table-center
  - table-small
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-31 15:38
---

# Phase 4: Phenotype Characterisation & Dynamic Tumour-Immune Trajectories 🧬

> [!INFO] What Is This Phase About?
> Phase 4 answers a deceptively simple question: **"Now that we have four patient clusters, what do they actually mean biologically?"**
>
> It takes the raw cluster assignments from Phase 3 (groups of patients numbered 0–3) and gives each one a clinically meaningful identity — a *phenotype* — by examining the immune, genomic, and cellular profile of every patient in that group. It then goes further by simulating how each phenotype is likely to behave under immunotherapy over 180 days.

---

## 1. Phase Overview: From Clusters to Clinical Phenotypes

Imagine running a clustering algorithm on a room full of melanoma patients. You get four groups. But a number like "Cluster 2" has no clinical meaning on its own — it tells an oncologist nothing about what treatment to prescribe.

Phase 4 solves this by asking: **what does the biology look like inside each cluster?** It does this in three steps:

1. **Profile each cluster** — compute the average levels of immune signals (`TIS`, `CYT`), immune cell types (CD8+ T cells, M1/M2 macrophages, cancer-associated fibroblasts), and driver mutation frequencies (`BRAF`, `NRAS`, `NF1`) for every cluster.
2. **Assign biological phenotype labels** — using these profiles, assign each cluster a meaningful clinical identity based on hallmark immune signatures. The four discovered phenotypes are described in Section 3 below.
3. **Simulate and compare** — run patient-group–specific mathematical simulations of tumour growth and immune response over 180 days, and compare overall survival curves across phenotypes.

The key output is a clinically interpretable stratification framework: 699 patients distributed across four biologically distinct subtypes, each with a predicted trajectory under anti-PD-1 immunotherapy.

---

## 2. Upstream Integration: How Phase 4 Builds on Prior Questions

> [!NOTE] Where Do the Inputs Come From?
> Phase 4 does not work in isolation. It sits at the intersection of all five research questions and acts as the first point where multi-source signals are translated into a coherent clinical narrative.

Phase 4 draws on outputs from three upstream questions:

### From Q1: Immunotherapy Response Prediction
Q1 built predictive models (Logistic Regression and Random Forest) trained on immune signatures — `TIS` (Tumour Inflammation Score), `CYT` (Cytolytic Activity), `IFN_gamma`, and genomic mutation flags. Phase 4 uses these same biomarkers as the *profile features* that define and distinguish the four phenotypes. The ranked biomarker importance from Q1 directly informs which signals are most clinically meaningful when interpreting each phenotype.

### From Q3: ODE Tumour-Immune Dynamics
Q3 developed a mathematical model of how tumours grow and how the immune system fights back under therapy. It described tumour burden over time as a function of two interacting forces: tumour proliferation rate and the efficiency with which effector immune cells (CD8+ T cells) destroy tumour cells.

Phase 4 borrows this mathematical framework and applies it *per phenotype*. Instead of running one simulation for the entire cohort, it runs four separate simulations — one for each phenotype — using rate constants tuned to reflect that phenotype's biological characteristics. An *Immune Hot* phenotype gets a high immune kill rate; an *Immunosuppressive M2-High* phenotype gets a low one. The resulting trajectories show visually divergent outcomes over 180 days.

### From Q2 and Q4 (indirect)
Q2 predicted drug sensitivity scores (viability AUC) per patient for agents including Dabrafenib, Trametinib, and Temozolomide. Q4 identified synthetic lethal drug targets from CRISPR functional genomics screens. These outputs feed into Phase 7 (Treatability Scoring), but the *biological context* they provide — which molecular subtypes are BRAF-dependent, which have high tumour mutational burden — is established in Phase 4 through the phenotype profiling step.

---

## 3. The Four Discovered Phenotypes

> [!NOTE] What Are Phenotypes?
> In oncology, a **phenotype** describes the observable biological and clinical characteristics of a tumour. Classifying patients by phenotype is essential because tumours that look identical under a microscope can behave very differently — responding or failing to respond to the same treatment — due to differences in their immune microenvironment and genomic drivers.

The 699-patient cohort (Liu 2019, Hugo 2016, Riaz 2017, and TCGA-SKCM reference) stratified into four phenotypes with the following characteristics:

| Phenotype | N | % | Hallmark Biology |
| :--- | :---: | :---: | :--- |
| **Immune Hot** | 304 | 43.5% | High `TIS`, high cytolytic activity (`CYT`), abundant CD8+ T cells, favourable M1/M2 macrophage ratio. The tumour microenvironment is actively inflamed and pro-immunogenic. |
| **Immunosuppressive M2-High** | 308 | 44.1% | Depleted T-cell infiltration, dominant M2 ("alternatively activated") macrophages, high Cancer-Associated Fibroblast (CAF) density. The immune system is present but actively suppressed. |
| **Immune Cold** | 22 | 3.1% | Very low `TIS`, near-absent immune infiltration. The tumour microenvironment is an "immune desert" — T cells simply are not recruited to the tumour site. |
| **Mutant-Driven** | 65 | 9.3% | 100% `NF1` mutation rate, elevated tumour mutational burden (TMB). Tumour growth is driven by a specific genomic lesion rather than immune evasion. |

These phenotypes align with established clinical frameworks in the melanoma immunotherapy literature: the "T-cell–inflamed" (hot) versus "non-T-cell–inflamed" (cold/excluded) dichotomy, with additional resolution into stromal-exclusion (M2-High) and genomic-driver (Mutant-Driven) subtypes.

---

## 4. The ODE Simulations: Watching Tumours Respond in Silico

The mathematical simulations in Phase 4 are a direct application of the Q3 ODE framework. Rather than following any individual patient, they model the *average expected trajectory* for a patient in each phenotype.

The underlying biological logic is straightforward: tumour burden over time is a balance between **tumour growth** (governed by intrinsic proliferation rate and carrying capacity) and **immune-mediated clearance** (governed by the rate at which effector T cells kill tumour cells). Anti-PD-1 therapy acts by "releasing the brakes" on T-cell activity — but its effect depends entirely on how many T cells are present and how well they can penetrate the tumour.

This produces four distinct 180-day trajectory curves:

- **Immune Hot**: Steep tumour clearance — high effector cell density and killing efficiency drive rapid regression.
- **Mutant-Driven**: Moderate regression — high TMB generates neoantigens recognised by T cells, but the T-cell infiltrate is less dominant than in the Hot phenotype.
- **Immune Cold**: Minimal response — few effector cells, tumour grows largely unchecked.
- **Immunosuppressive M2-High**: Persistent or growing tumour burden — M2 macrophages and CAFs physically and chemically exclude T cells, rendering checkpoint blockade ineffective.

These simulations serve as a bridge: they translate static biomarker profiles into a dynamic temporal prediction, grounding the patient stratification in mechanistic biology.

---

## 5. Why This Matters for Patient Stratification (Q5)

> [!IMPORTANT] The Clinical Significance of Phase 4
> Phase 4 is the point at which the Q5 framework shifts from a data-science exercise into a clinical decision-support tool. Without phenotype labels, the clusters are anonymous. With them, each patient is assigned to an actionable biological category that carries explicit implications for treatment.

The clinical relevance of Phase 4 operates at three levels:

**1. Treatment Triage**
Knowing a patient is *Immune Hot* strongly supports anti-PD-1 monotherapy as a first-line strategy. Knowing a patient is *Immunosuppressive M2-High* flags them for combination regimens targeting stromal remodelling or myeloid reprogramming — areas where anti-PD-1 alone has historically underperformed.

**2. Survival Stratification**
The Kaplan–Meier analysis performed in Phase 4 directly tests whether the four phenotypes separate out in terms of overall survival. Phenotype-specific survival curves — if significantly divergent by log-rank test — provide empirical justification for treating these groups as clinically distinct rather than statistically arbitrary.

**3. Informing Downstream Phases**
Phases 5, 6, and 7 all depend on the phenotype labels established here. Phase 5 trains a separate predictive model for each phenotype subgroup. Phase 6 measures clinical utility (net benefit, number-needed-to-treat) per phenotype. Phase 7 assigns patients to treatment arms — immunotherapy, targeted therapy, or combination — largely on the basis of their Phase 4 phenotype and the corresponding ODE-predicted trajectory.

---

## 6. Key Empirical Results at a Glance

| Output | Description |
| :--- | :--- |
| `baseline_signature_boxplots.png` | Violin plots showing `TIS`, `CYT`, CD8+ T cells, M1/M2 macrophages, and CAF levels stratified across the four phenotypes |
| `ode_trajectories.png` | 180-day simulated tumour burden trajectories per phenotype under anti-PD-1 therapy |
| `km_survival_by_phenotype.png` | Kaplan–Meier overall survival curves, log-rank tested, stratified by phenotype |
| `phenotype_characterisation.csv` | Per-phenotype summary statistics: mean biomarker profiles and mutation frequencies |

The four phenotypes show clearly divergent microenvironmental profiles in the violin plots, confirming that the GMM clustering (Phase 3) captured biologically meaningful variation rather than statistical noise. The ODE trajectories show that the *Immune Hot* and *Mutant-Driven* subtypes are predicted to respond to checkpoint blockade, while *Immunosuppressive M2-High* and *Immune Cold* tumours are not — a finding consistent with published clinical response rates in equivalent molecular subtypes.

---

## 7. Limitations & Future Directions

> [!NOTE] What Are the Key Caveats?
> Every analytical model involves simplifying assumptions. Being transparent about what Phase 4 does *not* capture is essential for interpreting results honestly and identifying where future work is needed.

The following limitations are drawn directly from the [Phase 4 Methodological Audit](phase_4_LIMITATIONS.md):

**Statistical Caveats**

- The *Immune Cold* (N=22, 3.1%) and *Mutant-Driven* (N=65, 9.3%) clusters are small relative to the two dominant groups. This reduces the statistical power of survival comparisons for these rare subtypes and increases the risk of missing real differences (Type II error).
- Each phenotype is described by its *mean* biomarker values. This collapses the natural variation within a cluster — a highly heterogeneous group might show an average similar to a homogeneous one, hiding important within-group diversity.
- The Kaplan–Meier survival analysis is unadjusted — it does not control for age, tumour stage, prior therapy, or LDH levels. Apparent survival differences between phenotypes may partly reflect these clinical confounders rather than phenotype biology alone.

**Biological Caveats**

- The ODE simulations treat every patient *within* a phenotype identically. All 304 *Immune Hot* patients run the same mathematical simulation with the same rate constants. In reality, individual patients differ in their immune cell kinetics, drug pharmacokinetics, and tumour growth rates.
- The 2-state ODE model does not explicitly account for CAF-mediated physical T-cell exclusion, M1-to-M2 macrophage polarisation switching, or genomic antigen presentation loss (mutations in `B2M`, `TAP1`, or the `HLA` loci). These are clinically important mechanisms for immunotherapy resistance that are currently modelled indirectly — as reduced kill rates — rather than as explicit biological variables.
- Kinetic parameters are fixed for the full 180-day window. Real tumours evolve: they develop resistance through clonal selection, and immune exhaustion markers such as `HAVCR2` (TIM-3) and `LAG3` emerge dynamically over the treatment course.

**Highest-Priority Future Improvements**

1. **Patient-specific ODE parameters** — personalise each simulation using individual patient transcriptomic and genomic data rather than cluster-wide averages.
2. **Extended ODE system** — add explicit model compartments for CAF density, M2 macrophage suppression, and antigen presentation intactness.
3. **Multivariable Cox regression** — replace unadjusted Kaplan–Meier with Cox proportional hazards models that control for key clinical covariates.
4. **Bootstrap confidence intervals** — report uncertainty ranges around cluster biomarker means to better communicate intra-cluster heterogeneity.

---

## 8. Key Takeaways 🎓

- Phase 4 converts anonymous cluster numbers into four biologically interpretable patient phenotypes — *Immune Hot*, *Immunosuppressive M2-High*, *Immune Cold*, and *Mutant-Driven* — each with distinct immune microenvironmental and genomic characteristics.
- The phenotype labels are derived directly from the Q1 biomarker framework (`TIS`, `CYT`, cell deconvolution) and are validated against Q3 ODE-predicted treatment trajectories.
- The ODE simulations show that the *Immune Hot* and *Mutant-Driven* subtypes are predicted to respond to anti-PD-1 therapy, while *Immunosuppressive M2-High* and *Immune Cold* tumours are not — consistent with published clinical evidence.
- Phase 4 phenotypes propagate forward through all remaining phases: they govern subgroup modelling (Phase 5), clinical utility assessment (Phase 6), and treatment arm assignment (Phase 7).
- Key simplifications — cluster-averaged ODE parameters, unadjusted survival analysis, and a 2-state tumour-immune system — highlight clear directions for future model refinement.
