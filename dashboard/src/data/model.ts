// ---------------------------------------------------------------------------
// REAL model output from the q3 ODE digital twin (TCGA-SKCM, n=421).
// These arrays are group-averaged tumour-burden dose-response sweeps taken
// directly from:
//   q3-ode-model/outputs/results/tumour_burden_simulations.csv   (BRAF inhibitor)
//   q3-ode-model/outputs/results/checkpoint_tumour_simulations.csv (anti-PD-1)
// Dose axis (drug fraction of max): 0.01 .. 1.00 in 10 steps.
// ---------------------------------------------------------------------------

export const DOSE_AXIS = [0.01, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 1.0];

export interface DoseResponse {
  brafi: number[]; // tumour burden vs BRAF-inhibitor dose
  antipd1: number[]; // tumour burden vs anti-PD-1 dose
  n: number;
}

// Keyed by clinical archetype. Values are the exact group means we computed.
export const DOSE_RESPONSE: Record<string, DoseResponse> = {
  // BRAF-mutant / PD-L1-low  → targeted therapy works, immunotherapy does not.
  braf_mut_pdl1_low: {
    n: 84,
    brafi: [0.4465, 0.3858, 0.3531, 0.332, 0.3116, 0.2905, 0.2753, 0.2619, 0.2472, 0.2317],
    antipd1: [0.4427, 0.4349, 0.4336, 0.433, 0.4326, 0.4324, 0.4323, 0.4322, 0.4321, 0.432],
  },
  // BRAF-WT / PD-L1-high → immunotherapy works, targeted does nothing (RAF paradox).
  braf_wt_pdl1_high: {
    n: 103,
    brafi: [0.4024, 0.425, 0.4249, 0.4235, 0.4215, 0.4194, 0.4172, 0.4152, 0.4131, 0.4109],
    antipd1: [0.3862, 0.3017, 0.2789, 0.2666, 0.2591, 0.2535, 0.248, 0.2439, 0.2411, 0.2394],
  },
  // BRAF-mutant / PD-L1-high → both lanes respond (combination / sequencing case).
  braf_mut_pdl1_high: {
    n: 108,
    brafi: [0.3461, 0.292, 0.2644, 0.2496, 0.2433, 0.2371, 0.2282, 0.2192, 0.211, 0.2043],
    antipd1: [0.3343, 0.2861, 0.268, 0.2579, 0.2522, 0.2488, 0.2468, 0.245, 0.2428, 0.2402],
  },
};

// Real Kaplan-Meier readouts from q3 survival_summary.txt --------------------
// Checkpoint (anti-PD-1) tumour burden was the strongest prognostic signal.
export const KM_FACTS = {
  checkpoint: {
    highMedianOs: 65.9,
    lowMedianOs: 148.2,
    logRankP: 0.0024,
    label: "Checkpoint tumour burden (anti-PD-1 sweep, Module D)",
  },
  brafi: {
    highMedianOs: 68.1,
    lowMedianOs: 105.0,
    logRankP: 0.0269,
    label: "Modelled tumour burden (BRAFi sweep)",
  },
  perk: {
    highMedianOs: 68.1,
    lowMedianOs: 103.1,
    logRankP: 0.0245,
    label: "pERK (MAPK output)",
  },
  digitalTwinAuc: 0.666,
  brafiPerkSuppression: 0.49, // vemurafenib suppresses pERK ~49% in V600E
};

/** The four ODE modules named during the loading sequence. */
export const ODE_MODULES = [
  { id: "raf", label: "RAF-dimer + drug binding", note: "RAF paradox emerges from equilibria" },
  { id: "mapk", label: "MAPK cascade → pERK", note: "published constants, negative feedback" },
  { id: "tumour", label: "Tumour–immune dynamics", note: "killing scaled by CYT score" },
  { id: "checkpoint", label: "PD-1 / PD-L1 checkpoint axis", note: "anti-PD-1 unleashes CD8 killing" },
] as const;
