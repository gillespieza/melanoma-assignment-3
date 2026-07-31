import type { BrafCall, CohortPatient, NrasCall } from "../data/cohort";
import { DOSE_RESPONSE } from "../data/model";
import { PDL1_HIGH } from "./scoring";

// What-if editing.
//
// A cohort patient's ODE curves belong to their REAL molecular profile. The
// moment a clinician changes BRAF status or PD-L1, those curves no longer
// describe the hypothetical patient on screen — so we swap in the closest real
// subgroup-average sweep instead of silently keeping curves that no longer apply.
//
// The UI badges any edited patient as hypothetical. Nothing here ever writes
// back to the cohort.

export interface WhatIfEdits {
  braf?: BrafCall;
  nras?: NrasCall;
  pdl1Pct?: number;
  stageBand?: string;
  ldh?: "Normal" | "Elevated" | "High";
}

export const NO_EDITS: WhatIfEdits = {};

export function hasEdits(edits: WhatIfEdits): boolean {
  return Object.keys(edits).length > 0;
}

/** Which fields actually differ from the patient's real record. */
export function changedFields(patient: CohortPatient, edits: WhatIfEdits): string[] {
  const out: string[] = [];
  if (edits.braf !== undefined && edits.braf !== patient.braf) out.push("BRAF");
  if (edits.nras !== undefined && edits.nras !== patient.nras) out.push("NRAS");
  if (edits.pdl1Pct !== undefined && edits.pdl1Pct !== patient.pdl1Pct) out.push("PD-L1");
  if (edits.stageBand !== undefined && edits.stageBand !== patient.stageBand) out.push("Stage");
  // TCGA records no LDH, so any value at all is an addition to the real record.
  if (edits.ldh !== undefined) out.push("LDH");
  return out;
}

/** The subgroup whose real averaged sweep best matches a hypothetical profile. */
function subgroupFor(braf: BrafCall, pdl1Pct: number): keyof typeof DOSE_RESPONSE {
  if (braf === "WT") return "braf_wt_pdl1_high";
  return pdl1Pct >= PDL1_HIGH ? "braf_mut_pdl1_high" : "braf_mut_pdl1_low";
}

function reductionOf(curve: number[]): number {
  const best = Math.min(...curve);
  return Math.max(0, Math.min(1, 1 - best / curve[0]));
}

/** Curve rescaled so baseline = 100, matching the cohort.json convention. */
function normalise(curve: number[]): number[] {
  return curve.map((v) => Math.round((v / curve[0]) * 10000) / 100);
}

/**
 * Apply edits to a patient. If the molecular profile driving the ODE changed,
 * the twin curves are replaced with the matching subgroup averages and the
 * result is marked `synthetic` so the UI can say so.
 */
export function applyWhatIf(
  patient: CohortPatient,
  edits: WhatIfEdits
): { patient: CohortPatient; synthetic: boolean; changed: string[] } {
  const changed = changedFields(patient, edits);
  if (!changed.length) return { patient, synthetic: false, changed: [] };

  const braf = edits.braf ?? patient.braf;
  const nras = edits.nras ?? patient.nras;
  const pdl1Pct = edits.pdl1Pct ?? patient.pdl1Pct;
  const stageBand = edits.stageBand ?? patient.stageBand;

  // Only a change that alters the ODE's inputs invalidates the real curves.
  const curvesInvalid = changed.includes("BRAF") || changed.includes("PD-L1");

  const next: CohortPatient = {
    ...patient,
    braf,
    nras,
    pdl1Pct,
    stageBand,
    stage: stageBand === patient.stageBand ? patient.stage : stageBand,
    ldhOverride: edits.ldh,
  };

  if (!curvesInvalid) {
    return { patient: next, synthetic: false, changed };
  }

  const dr = DOSE_RESPONSE[subgroupFor(braf, pdl1Pct)];
  const brafiReduction = reductionOf(dr.brafi);
  const antipd1Reduction = reductionOf(dr.antipd1);

  return {
    patient: {
      ...next,
      brafiCurve: normalise(dr.brafi),
      antipd1Curve: normalise(dr.antipd1),
      brafiBaseline: dr.brafi[0],
      antipd1Baseline: dr.antipd1[0],
      brafiInformative: true,
      antipd1Informative: true,
      brafiReduction,
      antipd1Reduction,
      // Percentiles no longer meaningful against the real cohort ranking, so
      // approximate from the reduction itself for the agreement comparison.
      brafiReductionPct: Math.round(Math.min(1, brafiReduction / 0.6) * 100),
      antipd1ReductionPct: Math.round(Math.min(1, antipd1Reduction / 0.6) * 100),
      brafiOptimalDose: null,
      antipd1OptimalDose: null,
    },
    synthetic: true,
    changed,
  };
}
