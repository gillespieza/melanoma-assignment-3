import type { PatientInput, TriageResult, DecisionNode } from "../data/types";
import { DOSE_RESPONSE } from "../data/model";
import {
  ARMS,
  PDL1_HIGH,
  buildRankedOptions,
  clamp,
  scoreArms,
  type ScoringContext,
} from "./scoring";

// ---------------------------------------------------------------------------
// The archetype path: takes the hand-editable PatientInput from the intake
// panel and runs it through the shared scoring core (see ./scoring.ts).
//
// Archetypes have no real Q1 row, so their `signature` field is the clinician's
// own estimate of the expression-response percentile. Cohort patients take the
// same slot from real model output — see ./integrationEngine.ts.
// ---------------------------------------------------------------------------

export { ARMS, PDL1_HIGH };

/** Pick the closest real ODE dose-response subgroup for this patient. */
export function subgroupKey(p: PatientInput): keyof typeof DOSE_RESPONSE {
  const brafMut = p.braf !== "WT";
  const pdl1High = p.pdl1 >= PDL1_HIGH;
  if (brafMut && pdl1High) return "braf_mut_pdl1_high";
  if (brafMut && !pdl1High) return "braf_mut_pdl1_low";
  return "braf_wt_pdl1_high"; // WT patients keyed to the immuno-responsive curve
}

function reduction(curve: number[]): number {
  const base = curve[0];
  const best = Math.min(...curve);
  return clamp((1 - best / base) * 100, 0, 100) / 100;
}

export function triage(p: PatientInput): TriageResult {
  const dr = DOSE_RESPONSE[subgroupKey(p)];

  const ctx: ScoringContext = {
    brafMut: p.braf !== "WT",
    pdl1: p.pdl1,
    signature: p.signature,
    highLdh: p.ldh !== "Normal",
    ecog: p.ecog,
    age: p.age,
    immunoReduction: reduction(dr.antipd1),
    targetedReduction: reduction(dr.brafi),
  };

  const scores = scoreArms(ctx);
  const options = buildRankedOptions(ctx, scores);
  const primary = options.find((o) => o.tier === "primary") ?? options[0];

  const path = buildPath(p, ctx.brafMut, p.pdl1 >= PDL1_HIGH, ctx.highLdh, primary.arm.label);
  const headline = `${primary.arm.label} · ${primary.confidence}% model confidence · predicted median OS ${primary.medianOsMonths} mo`;

  return { options, path, headline };
}

function buildPath(
  p: PatientInput,
  brafMut: boolean,
  pdl1High: boolean,
  highLdh: boolean,
  recommendation: string
): DecisionNode[] {
  return [
    {
      id: "stage",
      label: `Stage ${p.stage}`,
      detail: highLdh ? `LDH ${p.ldh.toLowerCase()} — rapid-control pressure` : "LDH normal",
    },
    {
      id: "braf",
      label: brafMut ? `BRAF ${p.braf} (mutant)` : "BRAF wild-type",
      detail: brafMut ? "Targeted therapy is on the table" : "No BRAF/MEK option",
    },
    {
      id: "pdl1",
      label: `PD-L1 ${p.pdl1}% · signature ${p.signature}`,
      detail: pdl1High ? "Immuno-favourable biology" : "Immuno-cold biology",
    },
    {
      id: "rec",
      label: recommendation,
      detail: "Proposed — pending consultant sign-off",
    },
  ];
}
