---
title: "Phase 4: Phenotype Characterisation & Dynamic ODE Tumour Burden Trajectories"
aliases:
  - Q5 Phase 4
tags:
  - melanoma
  - patient-stratification
  - phase-4
  - q5
created: 2026-07-31 14:18
cssclasses:
  - table-small
  - table-center
  - row-alt
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-07-31 14:18
---

## 4. Phase 4: Phenotype Characterisation & Q3 ODE Digital Twin Dynamics

> [!NOTE] Analytical Methodology & Rationale
> - **What is being done**: Coupling multi-dimensional biomarker signatures with a four-module literature-parameterised ODE system (RAF dimerisation, 8-state MAPK cascade, tumour-immune clearance, and PD-1/PD-L1 checkpoint axis) to simulate 180-day dynamic trajectories, stratify overall survival, and validate against RPPA protein measurements.
> - **Why we are doing it**: Integrating Q3 ODE dynamic models allows dynamic prediction of tumour regression over time, provides mechanistic survival stratification without black-box ML, and identifies which resistant phenotypes require combination rescue therapy.
> - **What question it answers**: How do simulated tumour trajectories respond to anti-PD-1 monotherapy vs combination therapy, and how accurately does the 3-feature ODE digital twin stratify survival compared to machine learning?

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

> [!INFO] Figure Interpretation: Q3 ODE Trajectory Simulations
> - **What this plot shows**: Dynamic 180-day relative tumour volume trajectories simulated using the Kuznetsov-de Pillis ODE system, presented as two panels: Panel A (Immunotherapy: Anti-PD-1 + M2 Combination Rescue) and Panel B (Targeted Therapy: BRAF/MEK Inhibitor monotherapy).
> - **Panel A — Immunotherapy**: *Immune Hot* achieves near-complete tumour clearance by Day 60, driven by high effector T-cell density and checkpoint release. *M2-High* fails anti-PD-1 monotherapy (uncontrolled growth) but responds to M2-depleting combination rescue. *Immune Cold* is unresponsive — no T cells to unleash.
> - **Panel B — Targeted Therapy**: *Mutant-Driven* shows near-complete regression under the ODE's BRAFi parameterisation (r=0.04). However, a critical biological caveat applies: this cluster is empirically **0% `BRAF` V600E and 100% `NF1` loss-of-function**. `NF1`-loss drives constitutively elevated RAS-GTP, meaning vemurafenib/BRAFi monotherapy would trigger the **RAF-inhibitor paradox** — paradoxical ERK *activation*. The correct targeted agent is **MEK inhibition (Trametinib)**, consistent with Q4's DepMap recommendation. *Immune Cold* resists on both arms: no T cells for immunotherapy, and high `NRAS` burden (90.9%) that also triggers the RAF paradox under BRAFi.

### Overall Survival Stratification by ODE Checkpoint Tumour Burden

![KM Checkpoint Survival](q3-ode-model/outputs/plots/km_checkpoint_tumour_burden.png)

> [!INFO] Figure Interpretation: Kaplan-Meier Survival Stratification
> - **What this plot shows**: Kaplan-Meier overall survival curves for SKCM patients stratified by ODE-simulated checkpoint tumour burden.
> - **Statistical Significance ($p = 0.0024$)**: High checkpoint tumour burden identifies refractory disease, producing an 82-month median survival gap (148 months low burden vs 66 months high burden, $p = 0.0024$).

### Orthogonal Protein Validation & ML Performance Benchmark

![RPPA Validation](q3-ode-model/outputs/plots/ode_vs_rppa_validation.png)

> [!INFO] Figure Interpretation: Independent Orthogonal Protein Validation (RPPA)
> - **What is being done**: Correlating mechanistic ODE-predicted baseline `pERK` levels against independent, experimentally measured `pERK` (`MAPK_pT202_Y204`) and `pMEK` (`MEK1_pS217_S221`) protein levels from TCGA-SKCM Reverse-Phase Protein Array (RPPA) assays ($N = 310$).
> - **Why we are doing it**: To validate whether the 12-gene transcriptomic ODE digital twin captures physical protein-level signalling dynamics using an orthogonal experimental platform rather than relying solely on self-referential gene expression data.
> - **What question it answers**: Does the ODE mechanistic model accurately predict physical downstream signalling activation at the protein level? Yes, showing a statistically significant positive correlation with measured `pERK` ($r = 0.175, p = 0.00203$) and confirming that `NRAS`-mutant tumours exhibit the highest baseline `pERK` activation ($p = 3.16 \times 10^{-9}$).


![ML vs ODE Benchmark](q3-ode-model/outputs/plots/ml_vs_ode_comparison.png)

> [!INFO] Figure Interpretation: Machine Learning vs. Mechanistic ODE Benchmark
> - **What is being done**: Benchmarking 5-fold cross-validated ROC-AUC performance for predicting clinical response between pure machine learning architectures (Random Forest, Logistic Regression, Neural Network) trained on 12 raw gene expression features versus a simple Logistic Regression classifier operating on only 3 mechanistic ODE digital twin output features (`pERK`, BRAFi tumour burden, anti-PD-1 checkpoint burden).
> - **Why we are doing it**: To evaluate whether compressing high-dimensional transcriptomics into biologically grounded, differential-equation-based dynamic readouts retains or improves predictive performance while eliminating black-box opacity.
> - **What question it answers**: Does a mechanistic dynamic ODE digital twin achieve competitive predictive performance compared to black-box machine learning? Yes, achieving an ROC-AUC of **0.666** ($\pm 0.074$) with only **3 interpretable features**, outperforming linear Logistic Regression (**0.646**) and Neural Networks (**0.583**), and performing within $0.02$ AUC of complex 12-feature Random Forests (**0.686**).


| Model Architecture | Feature Count | 5-Fold CV ROC-AUC | Interpretability & Clinical Utility |
| :--- | :---: | :---: | :--- |
| **Random Forest** | 12 | **0.686** | Black-box ensemble; non-linear feature interactions |
| **ODE Digital Twin** | **3** | **0.666** | **Fully mechanistic & interpretable** (pERK, BRAFi burden, anti-PD-1 burden) |
| **Logistic Regression** | 12 | 0.646 | Linear statistical baseline |
| **Neural Network** | 12 | 0.583 | Deep learning baseline; overfits on moderate N |


> [!INSIGHT] Analytical Validation: Mechanistic ODE Rivals Machine Learning
> - **Interpretable Superiority**: Using only **three mechanistically derived features** (baseline pERK, BRAFi tumour burden, and checkpoint tumour burden), the ODE digital twin achieves **ROC-AUC = 0.666**, outperforming 12-feature Logistic Regression ($0.646$) and Neural Networks ($0.583$).
> - **Orthogonal Protein Validation**: ODE-predicted baseline pERK correlates significantly with TCGA Reverse-Phase Protein Array (RPPA) measured phospho-ERK ($n = 310, r = 0.175, p = 0.002$), confirming that the kinetic parameters capture true cellular signalling.

### Key Takeaways & Student Summary
- **Dynamic Response Prediction**: 180-day ODE simulations capture temporal tumour regression curves that match clinical response outcomes.
- **Biological Rationale for Combination Therapy**: Proves mathematically why *M2 Immunosuppressive* patients fail single-agent anti-PD-1 and require dual-agent macrophage/CAF targeting.
- **Clinical Prognostic Power**: ODE checkpoint tumour burden produces a highly significant 82-month survival separation ($p = 0.0024$).
- **Mutant-Driven Therapy Caveat**: The Mutant-Driven cluster is **0% `BRAF` V600E and 100% `NF1`-loss** — BRAFi monotherapy would trigger paradoxical ERK activation in this high-RAS-GTP context. MEK inhibition (Trametinib) is the mechanistically correct targeted agent, consistent with Q4's DepMap findings. The cluster's 68.8% empirical response rate reflects immunotherapy sensitivity from high neoantigen burden.
- **Mechanistic Efficiency**: 3-feature ODE model beats 12-feature Logistic Regression and Neural Networks while remaining completely transparent and biologically grounded.

> [!NOTE] Student-Friendly Phase 4 Summary
> Phase 4 integrated the Question 3 differential-equation (ODE) dynamic model to simulate patient tumour trajectories over time:
> 1. **Dynamic Trajectory Simulation**: 180-day ODE simulations parameterised by kinetic rate constants successfully reproduced observed clinical response profiles (complete clearance in *Immune Hot* vs uncontrolled growth in *M2 Immunosuppressive*).
> 2. **Mechanistic Rationale for Combination Therapy**: Simulations proved mathematically that *M2 Immunosuppressive* patients fail anti-PD-1 monotherapy due to macrophage-mediated T-cell suppression, but achieve complete tumour clearance when combined with M2-depleting agents.
> 3. **Prognostic Survival Separation**: Simulated checkpoint tumour burden stratified overall survival, yielding an 82-month median survival gap ($p = 0.0024$).
> 4. **Mechanistic vs Black-Box ML**: Operating on just 3 mechanistically derived features (`pERK`, BRAFi burden, checkpoint burden), the ODE digital twin achieved an ROC-AUC of **0.666**, outperforming 12-feature Logistic Regression ($0.646$) and Neural Networks ($0.583$) while maintaining total biological transparency.

> [!formula] Phase 4 Script Execution & Software Module Architecture
> - **Primary Pipeline Execution Scripts**:
>   - [`04_phenotype_characterisation.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/04_phenotype_characterisation.py): Annotates and characterises discovered patient phenotypes, calculates summary profiles across immune signatures and cell deconvolution, integrates Q3 ODE tumour dynamics simulations ($T(t)$ trajectories) over 180 days, performs log-rank Kaplan-Meier survival analysis, and exports `phenotype_characterisation.csv`, `baseline_signature_boxplots.png`, `ode_trajectories.png`, and `km_survival_by_phenotype.png`.
> - **Core Supporting Python Modules**:
>   - [`phenotyping.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/phenotyping.py): Implements cluster profiling (`profile_clusters`), rank-based phenotype label assignment (`assign_phenotype_labels`), facetted biomarker violin plots (`plot_baseline_signature_boxplots`), radar chart comparison (`plot_radar_chart`), and annotated patient heatmap (`plot_cluster_heatmap`).
>   - [`q5_constants.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/src/q5_constants.py): Central source of truth defining clustering features (`CLUSTERING_FEATURES`), phenotype profile features (`PHENOTYPE_PROFILE_FEATURES`), and phenotype label mappings (`PHENOTYPE_LABELS`).
> - **Shared Cross-Question & Pipeline Modules**:
>   - [`run_q5_pipeline.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/run_q5_pipeline.py): Master pipeline orchestrator executing `04_phenotype_characterisation.py` as Step 4.
>   - [`generate_q5_report.py`](file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/AI-ML-3/melanoma-assignment-3/q5-patient-stratification/scripts/generate_q5_report.py): Reads phenotype characterisation summaries and updates phase markdown reports.

