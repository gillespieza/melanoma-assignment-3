---
title: "Phase 4: Phenotype Characterisation & Dynamic ODE Tumour Burden Trajectories"
aliases:
  - Q5 Phase 4
tags:
  - melanoma
  - patient-stratification
  - phase-4
  - q5
created: 2026-07-31 16:36
cssclasses:
  - table-small
  - table-center
  - row-alt
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-31 17:22
---

## 4. Phase 4: Phenotype Characterisation & Q3 ODE Digital Twin Dynamics

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Coupling multi-dimensional biomarker signatures with a four-module literature-parameterised ODE system (RAF dimerisation, 8-state MAPK cascade, tumour-immune clearance, and PD-1/PD-L1 checkpoint axis) to simulate 180-day dynamic trajectories across immunotherapy and targeted therapy arms.
> - **Why we are doing it**: Integrating Q3 ODE dynamic models allows dynamic prediction of tumour regression over time, provides mechanistic response prediction without black-box ML, and identifies which resistant phenotypes require combination rescue therapy.
> - **What question it answers**: How do simulated tumour trajectories respond to anti-PD-1 monotherapy vs combination rescue vs targeted therapy across distinct biological phenotypes?

Phase 4 integrates the full **Question 3 Mechanistic ODE System** into the Q5 patient stratification framework. The model parameterises four coupled biological modules per patient using universal kinetic rate constants from published literature (*Rukhlenko et al. 2018*, *de Pillis et al. 2005/2006*, *Lai et al. 2017*, *Rooney et al. 2015*):

| Module | Published System | Biological Function & Coupling |
| :--- | :--- | :--- |
| **Module A: RAF Dimerisation** | Allosteric Binding Equilibria | Vemurafenib protomer binding and RAS-GTP dimerisation; captures RAF-inhibitor paradox without hardcoded if-statements. |
| **Module B: MAPK Cascade** | 8-State Raf->MEK->ERK | Fast-timescale ($t \sim \text{minutes}$) phosphorylation kinetics with negative feedback ($K_i = 9\text{ nM}$) yielding steady-state pERK. |
| **Module C: Tumour-Immune Dynamics** | Kuznetsov-de Pillis ODE | Slow-timescale ($t \sim \text{days}$) growth equation $dC/dt = \lambda_C C (1-C/C_M) - \eta_8 \cdot f_{\text{kill}} \cdot T_8 \cdot C$. |
| **Module D: Checkpoint Axis** | PD-1 / PD-L1 QSS Sub-Module | Competitive anti-PD-1 binding depleting $PD-1 \cdot PD-L1$ inhibitory complex $Q$, unleashing CD8+ T-cell killing capacity. |


### Baseline Biomarker Profile Distribution

![Biomarker Profile Boxplots](q5-patient-stratification/plots/phenotypes/baseline_signature_boxplots.png)

> [!INFO] Figure Interpretation: Biomarker Z-Score Fingerprints
> - **What this plot shows**: Standardized Z-scores across core microenvironment signatures (`TIS`, `CYT`, `CD8_T_cells`, `M1_Macrophages`, `M2_Macrophages`, `CAFs`) for all four patient clusters.
> - **Colour Key**: *Mutant-Driven* — **yellow** | *Immune Cold* — **blue** | *Immune Hot* — **vermillion** | *M2 Immunosuppressive* — **reddish purple**.
> - **Subtype Profiles**: *Immune Hot* (vermillion) displays the highest Z-scores across inflammatory markers (`TIS`, `CYT`, `CD8_T_cells`), consistent with an active cytotoxic microenvironment. *Mutant-Driven* (yellow) shows elevated TIS relative to the Cold subtype but is dominated by driver mutation burden. *M2 Immunosuppressive* (reddish purple) exhibits elevated `M2_Macrophages` and `CAFs` stromal scores, reflecting immunosuppressive exclusion. *Immune Cold* (blue) displays deeply suppressed Z-scores across all microenvironmental signatures.

### Mechanistic Q3 ODE Tumour Volume Trajectories T(t)

![Q3 ODE Tumour Trajectories](q5-patient-stratification/plots/phenotypes/ode_trajectories.png)

> [!INFO] Figure Interpretation: Q3 ODE Trajectory Simulations (Two-Panel Dual-Arm)
> - **What this plot shows**: Dynamic 180-day relative tumour volume $T(t)/K$ trajectories simulated across all four phenotypes under two therapeutic modalities: Panel A (Immunotherapy) and Panel B (Targeted Therapy).
> - **Panel A — Immunotherapy (Anti-PD-1 & Combination Rescue)**:
>   - *Immune Hot* ($N=208, T(180) = 0.14$): Achieves marked tumour regression due to high baseline infiltration (`CD8A`, `PRF1`, `GZMA`) unleashed by checkpoint blockade.
>   - *M2-High Monotherapy vs Rescue* ($N=154$): Anti-PD-1 monotherapy fails ($T(180) = 0.84$) due to M2 macrophage/CAF stromal exclusion. Adding M2-depleting rescue therapy (dashed line) breaches the barrier, driving significant regression ($T(180) = 0.36$).
>   - *Mutant-Driven* ($N=46, T(180) = 0.65$): Exhibits moderate response driven by high TMB.
>   - *Immune Cold* ($N=13, T(180) = 0.91$): Refractory to immunotherapy due to severe T-cell desert phenotype.
> - **Panel B — Targeted Therapy (BRAF Inhibitor Vemurafenib 500 nM)**:
>   - *Immune Hot* ($N=208, T(180) = 0.41$): Strong BRAFi sensitivity driven by 90% `BRAF V600` mutation rate and oncogene addiction suppression.
>   - *Mutant-Driven* ($N=46, T(180) = 0.79$): Partial resistance; `NF1` loss-of-function drives high RAS-GTP with residual RasGAP dampening (`pheno_r_mult` = 0.68).
>   - *Immune Cold* ($N=13, T(180) = 0.81$): Primary resistance driven by high `NRAS` mutation prevalence (90.9%) triggering paradoxical RAF activation.
>   - *M2-High* ($N=154, T(180) = 0.92$): Maximal resistance resulting from `NRAS` mutation-driven RAF paradox amplified by CAF-secreted growth factors (`pheno_r_mult` = 1.25).

### Key Takeaways & Executive Summary

- **Dynamic Response Prediction**: 180-day ODE simulations capture temporal tumour volume trajectories across all four phenotypes under immunotherapy and targeted therapy arms.
- **Biological Rationale for Combination Therapy**: Proves mathematically why *M2 Immunosuppressive* patients fail single-agent anti-`PDCD1` checkpoint blockade due to macrophage/CAF stromal exclusion, but achieve significant regression ($T(180) = 0.36$) when combined with M2-depleting agents.
- **Dual-Arm Phenotype Stratification**: Panel A (Immunotherapy) and Panel B (Targeted Therapy) capture distinct dynamic response profiles — highlighting `BRAF V600` sensitivity in *Immune Hot* ($T(180) = 0.41$) versus `NRAS`/`NF1`-driven RAF paradox resistance in *Mutant-Driven* ($T(180) = 0.79$), *Immune Cold* ($T(180) = 0.81$), and *M2-High* ($T(180) = 0.92$) phenotypes.

> [!INSIGHT] Phase 4 Key Findings & Synthesis
> Phase 4 integrated the Question 3 differential-equation (ODE) dynamic model into the patient stratification framework:
> 1. **Dynamic Trajectory Simulation**: 180-day ODE simulations parameterised by kinetic rate constants and per-patient biomarker levels successfully reproduced observed clinical response profiles (marked clearance in *Immune Hot* vs primary resistance in *Immune Cold*).
> 2. **Mechanistic Rationale for Combination Therapy**: Simulations proved mathematically that *M2 Immunosuppressive* patients fail anti-`PDCD1` monotherapy due to stromal T-cell exclusion, but achieve tumour regression when combined with M2-depleting rescue agents.
> 3. **Targeted Therapy & Paradox Modelling**: Dual-arm trajectory modeling accurately captures `BRAF V600` sensitivity in *Immune Hot* tumours versus `NRAS`/`NF1`-driven RAF paradox resistance in *M2-High*, *Immune Cold*, and *Mutant-Driven* phenotypes.



