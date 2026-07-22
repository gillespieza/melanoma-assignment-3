# Q3 Module C Refactor

**Status:** Core items implemented and run on TCGA-SKCM (n=421) — see §4 for results.
**Supersedes:** the raw notes in `q3-refactor-proposal.txt` (kept as source material).
**Scope:** Module C only (`scripts/phase3_ode_simulation.py`, `steady_tumour()` / `cancer_rhs()`). Modules A (RAF-dimer paradox) and B (MAPK cascade) are unaffected.

---

## 1. The gap this closes

The current Module C treats immunotherapy as a **fixed baseline**, not a drug:

```239:264:melanoma-assignment-3/q3-ode-model/scripts/phase3_ode_simulation.py
def cancer_rhs(C, t, prolif, T8):
    ...
    dC = (LF["lambdaC"] * prolif * C * (1.0 - C / LF["CM"])
          - LF["eta8"] * T8 * C
          - LF["dC"] * C)
    return [dC]

def steady_tumour(pERK, pERK_ref, infil):
    prolif = np.clip(pERK / max(pERK_ref, 1e-9), PERK_PROLIF_MIN, PERK_PROLIF_CAP)
    T8 = T8_SCALE * max(infil, 0.0)          # CD8 effector density from infiltration
    ...
```

`T8` is computed once from measured CD8A/PRF1/GZMA expression and never changes. There is no dose axis, no PD-1/PD-L1 checkpoint state, and no way to ask "what does giving this patient anti-PD-1 actually do?" Meanwhile, only the BRAF-inhibitor arm (Module A→B) has a real dose-response.

T**he two papers already cited as the source of Module C's constants** — ref [2] (Lai et al. 2017, *BMC Syst Biol*) and ref [5] (Rukhlenko et al. 2018, *Cell Systems*) — **contain exactly the missing mechanism** (a checkpoint axis and a mechanistic drug term). The current code only implemented the growth/death/killing skeleton of those papers and left the checkpoint dynamics out. 

---

## 2. Literature grounding


| Source (already in `Q3_ODE_report.md` refs, or newly proposed)                                                                   | What it establishes                                                                                                                                                                                                                                       | What's missing from current code                                                                                                                                         |
| -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Lai et al. 2017**, *BMC Syst Biol* 11:70 — [ref 2]                                                                             | Full melanoma tumour–immune model incl. an **anti-PD-1 concentration term** that depletes active PD-1 ($-\mu_{PA}AP$), plus a **dendritic-cell / IL-12 feedback** where tumour killing reduces necrotic-debris signalling and "starves" T-cell activation | The checkpoint term ($P,L,Q,A$) and the DC/IL-12 antagonism loop                                                                                                         |
| **Rukhlenko et al. 2018**, *Cell Systems* 7:161 — [ref 5]                                                                        | Apparent drug affinity depends on the dynamic mix of RAF conformational/phosphorylation **states**, not a single lumped rate; motivates state-based (not bulk-count) modelling                                                                            | Currently RAF dimer states are handled correctly in Module A, but Module C still uses one bulk $C$/$T8$ number instead of tracking effector vs. suppressed immune states |
| **Poulikakos et al. 2010**, *Nature* 464:427 — [ref 4]                                                                           | RAF-inhibitor resistance frequently arises from **secondary lesions acquired during treatment** (RAS reactivation, RAF isoform overexpression)                                                                                                            | Module A/B correctly model the *paradox at baseline*; nothing yet models resistance *evolving over the treatment course*                                                 |
| **Kholodenko 2000**, *Eur J Biochem* 267:1583 — [ref 1]                                                                          | The MAPK cascade's negative-feedback model this project already uses; separately, the *temporal pattern* (transient vs. sustained) of ERK activation — not just its magnitude — determines cell fate                                                      | Module B already implements the feedback correctly; the *temporal* aspect only matters if Module C becomes longitudinal (see §4, deferred)                               |
| **Auslander et al. 2018**, *Nat Med* — IMPRES (already implemented in `q1-response-predictor/src/signatures.py::compute_impres`) | A 15 checkpoint-gene-pair score that predicts checkpoint-blockade response from baseline expression                                                                                                                                                       | Not yet reused in Q3 — would be the natural per-patient initializer for the $P$/$L$ checkpoint state                                                                     |
| **Rooney et al. 2015** — CYT score (already implemented in `q1-response-predictor/src/signatures.py::compute_cyt`)               | Cytolytic activity (mean of GZMA, PRF1) sets a biological ceiling on how fast T-cells can actually kill, independent of how "unleashed" they are                                                                                                          | Current `eta8` (killing rate) is a fixed universal constant; not scaled per-patient by cytolytic capacity                                                                |


**Takeaway:** every piece of the proposal traces to a source already cited or already implemented elsewhere in this project (Q1's `signatures.py`). This is a rigor-preserving extension.

---

## 3. What we will actually change (scoped)

The raw proposal (`q3-refactor-proposal.txt`) sketches a full 9-state tumour-microenvironment model ($C, T_1, T_8, T_r, M, P, L, Q, R$) with periodic dosing and a 60-day longitudinal integration. That is biologically faithful to Lai et al., but it is too large a scope for the time remaining — it would require new gene inputs we don't currently extract (PDCD1, CD274, IL12A/B, FOXP3, MDSC markers), a new validation strategy (TCGA is cross-sectional, not longitudinal, so there's no direct "outcome at day 60" to check against), and a large increase in unvalidated free parameters.

We are scoping this down to a **Core** change that captures the essential biology and is fully implementable/validatable with data we already have.

### Core (recommended to implement)

1. **Checkpoint axis + diallable anti-PD-1 dose, at steady state (not longitudinal).**
  Add a minimal 3-state sub-module — PD-1 ($P$), PD-L1 ($L$), complex ($Q$) — solved to steady state (same QSS pattern already used for Modules A/B/C), with drug $A$ depleting $P$ via $-\mu_{PA}AP$ (Lai et al.). Effector killing becomes $\eta_8 \cdot T_8 \cdot f(Q) \cdot C$ instead of a fixed $\eta_8 \cdot T_8 \cdot C$, where $f(Q)$ is the fraction of effector cells *not* checkpoint-suppressed.
  - **Per-patient initializer for $P,L$:** reuse Q1's `compute_impres()` / `compute_pd_l1()` directly — no new signature code needed, just call the existing module against the same TCGA-SKCM expression matrix Q3 already reads.
  - **New dose axis:** mirror the existing `DOSE_UNITS` pattern used for `BRAFi_dose`, add an `antiPD1_dose` grid. This gives Q3 a real dose-response for the immunotherapy arm for the first time.
2. **CYT-scaled killing rate.**
  Replace the single universal `eta8` with `eta8_i = eta8_base * CYT_i / CYT_mean`, using Rooney's CYT score (already computed in Q1, or trivially recomputed from the GZMA/PRF1 columns Q3 already pulls in `phase2_preprocess_data.py`). This is a small, low-risk, high-value change: it makes the "cytolytic ceiling" argument (§4 of the raw proposal) real without adding new states.
3. **Re-run phases 4–6 with the new checkpoint-dose axis** exactly as currently done for the BRAF-inhibitor dose: Cox/KM stratification by "optimal anti-PD-1 dose," and an updated ML-vs-ODE comparison that can now include an immunotherapy-response feature.

### Stretch (only if Core lands with time to spare)

1. **Antagonistic pathway (DC/IL-12 starvation).** A single extra coupling term where a drop in $C$ reduces a lumped "activation signal" that dampens effector expansion — captures Lai et al.'s "kill too fast, starve the immune engine" finding without building the full DC/IL-12/MDSC state machine. This is the piece that would mechanistically justify *sequential vs. simultaneous* dosing for Q5.
2. **Two-timepoint (not fully longitudinal) resistance check.** Rather than a full 60-day integration, compare steady-state pERK at baseline vs. after simulated chronic BRAFi exposure (bump RAS-GTP input to represent acquired resistance from Poulikakos et al.) to show *qualitatively* why initial BRAF-inhibitor responders can still need a fallback to immunotherapy — without needing longitudinal TCGA data we don't have.

---

## 4. Results — the Core implementation, run on TCGA-SKCM (n=421)

**Status: implemented and run.** All three Core items landed (`phase2_preprocess_data.py`, `phase3_ode_simulation.py`, plus Phase 4/6 extended to consume the new checkpoint-dose axis). Full logs: `docs/refactor_run_logs.txt`. All figures/tables below are in `results/refactor_results/`.

**4.1 Mechanism sanity check (Phase 3).** The checkpoint axis produces a differentiated, biologically correct dose-response: anti-PD-1 shrinks modelled tumour burden **33.4%** in high-PD-L1 patients vs. only **2.0%** in low-PD-L1 patients (median `CD274` split). See `results/refactor_results/plots/checkpoint_dose_response.png` for the dose-response curve, and raw values in `results/refactor_results/results/checkpoint_tumour_simulations.csv`.

**4.2 Survival stratification (Phase 4) — the checkpoint readout is the strongest of the three.**


| Readout                                      | Log-rank *p* | High vs. Low median OS | Figure                                  |
| -------------------------------------------- | ------------ | ---------------------- | --------------------------------------- |
| pERK                                         | 0.0245       | 68.1 vs 103.1 mo       | `plots/km_pERK.png`                     |
| tumour burden (BRAFi arm)                    | 0.0269       | 68.1 vs 105.0 mo       | `plots/km_tumour_burden.png`            |
| **checkpoint tumour burden (anti-PD-1 arm)** | **0.0024**   | **65.9 vs 148.2 mo**   | `plots/km_checkpoint_tumour_burden.png` |


All three panels combined: refactor_results/plots`/km_ode_stratified.pdf`; Cox p-value-vs-dose scan for all three: refactor_results/plots`/cox_dose_scan.png`. Full numbers: `results/refactor_results/survival_summary.txt`. The new immunotherapy-arm readout has the smallest p-value *and* the widest OS gap of any readout Q3 has produced — patients whose modelled tumour persists despite simulated checkpoint blockade have the worst prognosis in the cohort.

**4.3 ODE vs. ML (Phase 6) — adding the checkpoint feature recovers and beats the pre-refactor baseline.**


| Digital twin                                   | Features                                  | AUC       |
| ---------------------------------------------- | ----------------------------------------- | --------- |
| Pre-refactor baseline                          | 2 (pERK, BRAFi tumour burden)             | 0.652     |
| After Module C/D change, before Phase 6 update | 2 (unchanged features, new Module C math) | 0.608     |
| **After adding `ode_checkpoint_maxdose`**      | **3**                                     | **0.666** |


The 3-feature twin now beats Logistic Regression (0.646) and the Neural Net (0.583), trailing only Random Forest (0.686, 12 raw features). Figure: `plots/ml_vs_ode_comparison.png`; table: `results/ml_vs_ode_comparison.csv`.

**4.4 Regression check — Modules A/B are untouched, as intended.** pERK survival stratification (p=0.0245) and the RPPA validation (Pearson r=0.175, p=2.0e-3) are numerically identical to the pre-refactor run (`plots/ode_vs_rppa_validation.png`, `plots/ode_vs_rppa_subtype.png`, `results/rppa_validation_summary.txt`) — confirming the checkpoint axis only changed Module C, not the signalling modules.

**What it means:** the immunotherapy arm of Q5's decision tree now has the same mechanistic backing as the targeted-therapy arm, and it turned out to be the *more* prognostic of the two — a stronger Q3↔Q5 connection than originally expected, not just a symmetric one.

---

## 5. Why this matters for the project narrative

- **Closes the Q3↔Q5 seam.** Q3 currently only mechanistically explains the *targeted-therapy* branch of Q5 decision tree (BRAF status → BRAFi vs. chemo). The Core refactor gives Q3 a real, dose-responsive mechanism for the *immunotherapy* branch too — "predicted responder → first-line checkpoint immunotherapy" now has a simulatable biological reason (low $Q$ / high IMPRES → drug unleashes effector killing), not just a Q1 classifier label.
- **Upgrades Phase 6's headline result.** The existing ODE-vs-ML comparison (2 ODE features, AUC 0.652, beats the neural net) can gain an immunotherapy-response feature, potentially the model's most clinically relevant output.
- **Costs almost nothing in new data engineering.** IMPRES and CYT are already implemented and validated in `q1-response-predictor`; this is a matter of importing and calling that module against data Q3 already loads, not sourcing new cohorts.

## 6. Honest caveats to state alongside this (matching the model's existing tone)

- The checkpoint-binding constants ($\mu_{PA}$, $K_d$ for $P$–$L$ binding) will be literature-order-of-magnitude values, exactly like Module A's RAF-dimer constants already are — not separately fit.
- TCGA-SKCM has no anti-PD-1 dosing or treatment-response labels, so the checkpoint-dose axis can be validated the same way the BRAFi axis was: **indirectly**, via survival stratification (Phase 4 pattern) and consistency with IMPRES/CYT rather than a direct drug-response readout.

