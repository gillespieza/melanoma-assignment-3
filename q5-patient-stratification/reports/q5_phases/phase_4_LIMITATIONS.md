---
title: "Phase 4 Methodological & System Limitations Audit"
aliases:
  - Q5 Phase 4 Limitations
  - Phase 4 Limitations
tags:
  - melanoma
  - patient-stratification
  - phase-4
  - limitations
  - q5
created: 2026-08-01 17:28
cssclasses:
  - table-small
  - table-center
  - row-alt
obsidianEditingMode: preview
obsidianUIMode: source
updated: 2026-08-01 17:28
---

## 4. Phase 4: Methodological & System Limitations Audit

> [!NOTE] Analytical Audit Overview
> - **What is being done**: Systematic critical evaluation of current Phase 4 code-quality architecture, statistical vulnerabilities, biological model boundaries, and computational/data constraints.
> - **Why we are doing it**: To establish explicit methodological boundaries, prevent clinical over-interpretation of simplified ODE dynamics, and define actionable engineering priorities for future iterations.
> - **What question it answers**: What structural, statistical, and biological limitations restrict the clinical validity of Phase 4 patient stratification, and how can they be mitigated?

Phase 4 couples multi-dimensional patient stratification ($N = 699$ across four biological phenotypes: *Immunosuppressive M2-High* $N=256$, *Immune Cold* $N=45$, *Immune Hot* $N=341$, *Mutant-Driven* $N=57$) with a 4-module literature-parameterised Ordinary Differential Equation (ODE) digital twin system. While this framework connects static microenvironmental signatures to temporal treatment response, several critical system limitations exist in its current state.


## Code-Quality & Architectural Evaluation

> [!WARNING] Code-Quality & System Architecture Audit
> - **What is being done**: Evaluation of module encapsulation, type safety, constant isolation, and automated test coverage in the current Phase 4 implementation.
> - **Why we are doing it**: Software design flaws introduce maintainability bottlenecks and elevate the risk of silent numerical divergence across analysis runs.
> - **What question it answers**: Where are the current software architecture and pipeline modularity most vulnerable to software regression?

### Unit Testing Coverage Deficit
While Phase 4 scripts feature robust runtime logging (`logs/04_phenotype_characterisation.log`) and error handling, the script relies on end-to-end execution testing rather than automated unit test suites (e.g. `pytest`). Mathematical functions—such as the Kuznetsov ODE derivative calculation (`_kuznetsov_ode`), parameter derivation helpers (`_derive_q3_patient_params`), and trajectory array caching—lack isolated unit test contracts with boundary-value test assertions.

### Single-File Orchestration Scope
`04_phenotype_characterisation.py` currently handles cluster profiling, baseline boxplot rendering, ODE numerical integration, trajectory plot generation, JSON summary serialization, and Kaplan-Meier survival curve fitting within a single 729-line file. While internal functions are modular, separating plotting routines from numerical integration pipelines into distinct sub-modules would improve isolated testability.

### Module Dependency Boundary
The script relies on cross-subproject imports from `q3-ode-model/scripts/phase3_ode_simulation.py` by appending `PROJECT_ROOT / "q3-ode-model" / "scripts"` directly to `sys.path`. This dynamic `sys.path` manipulation introduces implicit environment coupling between subprojects rather than importing from a formal shared sub-package.


## Statistical Weaknesses

> [!WARNING] Statistical & Subgroup Power Audit
> - **What is being done**: Critical assessment of sample size distribution, univariable survival testing, and cross-cohort validation boundaries.
> - **Why we are doing it**: Statistical asymmetries and unadjusted log-rank tests can generate overconfident prognostic claims for underpowered patient subgroups.
> - **What question it answers**: Where are the statistical findings most vulnerable to sampling bias or unmeasured clinical confounding?

### Subgroup Power Asymmetry across Phenotype Clusters
The dataset exhibits extreme cluster size variation:
- *Immune Hot*: $N = 341$ (48.8%)
- *Immunosuppressive M2-High*: $N = 256$ (36.6%)
- *Mutant-Driven*: $N = 57$ (8.2%)
- *Immune Cold*: $N = 45$ (6.4%)

The *Immune Cold* phenotype represents a minor subset ($N = 45$). When calculating empirical response rates or mean trajectory endpoints, small subgroup sample sizes yield wider confidence intervals, increasing susceptibility to sampling variation compared to the dominant *Immune Hot* cohort.

### Univariable Survival Log-Rank Testing
Kaplan-Meier survival stratification across biological phenotypes relies on unadjusted multivariate log-rank tests ($p < 0.001$, $N = 677$ patients with OS data). Unadjusted log-rank tests do not control for key clinical confounders, including patient age, prior lines of systemic therapy, disease stage (Stage III vs IV), or baseline LDH levels. Furthermore, because clustering features (`TIS`, `CYT`, `CD8_T_cells`) correlate with known prognostic factors, survival separation reflects combined prognostic and predictive effects rather than pure treatment response stratification.

### Single-Event Overall Survival Censoring
Overall survival (OS) is evaluated as a single composite endpoint without competing risks analysis (e.g. cancer-specific mortality vs non-cancer death) or progression-free survival (PFS) benchmarking. In clinical trials, short-term ODE trajectories (180 days) align more directly with objective response rate (ORR) and PFS than with long-term overall survival ($> 24$ months).

### Absence of External Non-TCGA Survival Validation
While TCGA-SKCM provides long-term overall survival metadata ($N = 421$), external immunotherapy cohorts (Liu 2019, Hugo 2016, Riaz 2017) have shorter follow-up times or incomplete OS tracking. As a result, Kaplan-Meier phenotype survival curves primarily reflect TCGA-SKCM baseline demographics.


## Biological Assumptions & Model Boundaries

> [!WARNING] Biological & Pharmacological Assumptions Audit
> - **What is being done**: Evaluation of biophysical simplifications, spatial abstractions, and fixed pharmacology parameters in the ODE digital twin.
> - **Why we are doing it**: Mathematical ODE models abstract complex physiological processes into simplified rate equations; explicit documentation of these assumptions prevents over-interpretation.
> - **What question it answers**: Which physiological mechanisms are omitted or simplified in the current Kuznetsov ODE formulation?

### 2-State ODE System Abstraction
The ODE digital twin utilizes a 2-state Kuznetsov-de Pillis formulation modeling relative tumour volume ($T$) and cytotoxic effector cell density ($E$). While effective for population-level dynamic trajectory simulations, it omits explicit differential equations for:
- Cancer-Associated Fibroblasts (CAFs) and physical extracellular matrix (ECM) barriers.
- Immunosuppressive M2 macrophage populations (represented implicitly via lower killing coefficients $c$ and effector scale factors).
- Regulatory T-cells ($T_{\text{reg}}$) and myeloid-derived suppressor cells (MDSCs).

### Phenomenological Effector Density Parameterisation
Per-phenotype initial effector density ($E_0$) and proliferation rate ($p_{\text{rate}}$) are derived using structured phenotype scale factors (`_EFFECTOR_PARAMS` lookup dictionary: Immune Cold $E_0 \text{ scale} = 0.20, p_{\text{rate}} = 0.003$; Immune Hot $E_0 \text{ scale} = 1.00, p_{\text{rate}} = 0.140$). While grounded in baseline infiltration signatures (`CD8A`, `PRF1`, `GZMA`), these phenotype scale factors are calibrated phenomenologically to reproduce clinical clearance rates rather than measured kinetic rate constants.

### 1D Volume Approximation vs 3D Spatial Architecture
The ODE model assumes a well-mixed 1D differential equation system, ignoring spatial heterogeneity within the tumour microenvironment. It cannot model spatial T-cell exclusion (margin-infiltrated vs desert architecture) or localized cytokine gradients.

### Fixed Trough Pharmacokinetics
Drug concentrations are modeled as static trough values ($500\text{ nM}$ Vemurafenib for Targeted therapy, $250\text{ nM}$ anti-PD-1 for Immunotherapy) over the full 180-day simulation. Real-world oral dosing and intravenous infusions induce fluctuating peak-trough pharmacokinetic profiles and patient-specific clearance rates.

### Simplified Combination Rescue Multiplier
The combination rescue arm (*M2 Immunosuppressive* under anti-PD-1 + M2 depletion) applies a fixed scalar boost (`_M2_RESCUE_KILL_BOOST = 1.3`) to Module D checkpoint killing ($f_{\text{kill}}$). This parameterises macrophage depletion as an empirical synergy factor rather than modeling explicit macrophage-T cell cross-talk kinetics.


## Computational & Data Constraints

> [!WARNING] Computational & Data Flow Constraints Audit
> - **What is being done**: Audit of numerical integration scalability, multi-cohort data coverage, and cross-module output dependencies.
> - **Why we are doing it**: Technical data bottlenecks limit real-time pipeline execution and prospective clinical deployment.
> - **What question it answers**: What computational constraints restrict ODE simulation scale and cross-cohort data integration?

### Serial ODE Numerical Integration Scalability
Per-patient ODE trajectories are evaluated sequentially using SciPy's `solve_ivp` RK45 adaptive integrator. While execution takes $< 10$ seconds for $N = 699$ patients, scaling to large-scale biobanks ($N > 10,000$) will require vectorised ODE solvers or GPU-accelerated numerical integration (e.g. `torchdiffeq` or JAX).

### Absence of Full 2D Dosing Grid Simulations
The current pipeline runs two parallel single-agent dose sweeps (Immunotherapy anti-PD-1 and Targeted BRAFi Vemurafenib). A full 2D drug concentration grid (anti-PD-1 dose $\times$ BRAFi dose) is not evaluated per patient in the current pipeline run, constraining combination therapy exploration to fixed single-dose regimens.

### Partial Orthogonal Protein Data Coverage
TCGA Reverse-Phase Protein Array (RPPA) validation ($r = 0.175, p = 2.03 \times 10^{-3}$) is available for a subset of $N = 310$ patients. The remaining multi-cohort clinical trial datasets (Liu, Hugo, Riaz) lack RPPA protein measurements, restricting protein-level validation of predicted `pERK` to the TCGA subset.

### Cohort Label Availability Mismatch
While $N = 699$ patients are stratified into biological phenotypes, clinical response labels (`RESPONSE_BINARY`) are available for $N = 326$ ICI-treated trial patients, while survival metadata ($N = 677$) is heavily centered on TCGA-SKCM. This creates a data mismatch where dynamic response validation and survival analysis operate on partially overlapping patient subsets.


## Actionable Fixes & Roadmap

> [!WARNING] Prioritised Engineering & Methodological Roadmap
> - **What is being done**: Defining an actionable, prioritised engineering roadmap to resolve identified code, statistical, biological, and computational limitations.
> - **Why we are doing it**: Clear prioritization ensures high-impact methodological improvements are targeted in subsequent pipeline refactoring cycles.
> - **What question it answers**: What concrete technical enhancements should be prioritized in future iterations?

| Priority | Targeted Limitation | Proposed Technical Enhancement | Expected Scientific & System Impact |
| :--- | :--- | :--- | :--- |
| **P1** (Highest) | Unit Test Deficit | Implement `pytest` suite for `_kuznetsov_ode`, parameter derivation, and trajectory serialization functions | Ensures mathematical correctness and prevents numerical regression |
| **P1** | 2-State Abstraction | Expand Kuznetsov ODE to a 3-state system ($T, E, M_2$) incorporating explicit macrophage-mediated inhibition | Mechanistic simulation of stromal exclusion without empirical scale multipliers |
| **P2** | Dynamic PK Curves | Replace static drug doses with 1-compartment pharmacokinetic decay models $C(t) = C_0 e^{-k_e t}$ | Captures peak-trough drug fluctuation and patient clearance variability |
| **P2** | Unadjusted Log-Rank | Implement multivariate Cox Proportional Hazards regression adjusting for age, stage, and cohort | Isolates independent prognostic utility of phenotype clusters |
| **P3** | Serial Integration | Vectorize ODE integrations via JAX or PyTorch GPU differential equation solvers | Enables instant scaling to $N > 10,000$ patient cohorts |
| **P3** | Single-Dose Limit | Compute full 2D concentration surfaces ($\text{Anti-PD-1} \times \text{BRAFi}$) for optimal combination dosing | Identifies synergistic therapeutic windows per patient phenotype |


## Key Takeaways

> [!WARNING] Critical Limitations Takeaways
> 1. **Subgroup Imbalance**: *Immune Cold* ($N = 45, 6.4\%$) and *Mutant-Driven* ($N = 57, 8.2\%$) subsets are relatively small compared to *Immune Hot* ($N = 341, 48.8\%$), warranting cautious interpretation of subgroup statistics.
> 2. **Unadjusted Prognostic Evaluation**: Survival stratification ($p < 0.001$, $N = 677$) uses univariable log-rank testing; multivariate Cox modeling is required to control for clinical covariates (age, stage, prior therapy).
> 3. **2-State Biophysical Simplification**: The ODE digital twin models tumour-immune kinetics ($T, E$) accurately but abstracts M2 macrophage exclusion into parameter scale factors rather than explicit differential equations.
> 4. **Fixed Trough Dosing**: Drug exposure is modeled at static trough concentrations ($500\text{ nM}$ BRAFi, $250\text{ nM}$ pembrolizumab), omitting pharmacokinetic clearance dynamics.
> 5. **Clear Development Roadmap**: Unit test suite creation (P1), 3-state ODE expansion (P1), and multivariate Cox survival controls (P2) represent the highest-priority engineering improvements.
