// Domain types for the Melanoma Digital Twin decision-support cockpit.
// These mirror the clinical + molecular variables a real melanoma MDT
// (multidisciplinary team) uses to choose systemic therapy.

export type BrafStatus = "V600E" | "V600K" | "WT";
export type NrasStatus = "Mutant" | "WT";
export type Stage = "IIIB" | "IIIC" | "IIID" | "IVa (M1a)" | "IVb (M1b)" | "IVc (M1c)";
export type LdhBand = "Normal" | "Elevated" | "High (>2x ULN)";
export type Ecog = 0 | 1 | 2;

/** The editable patient state the clinician manipulates on the intake panel. */
export interface PatientInput {
  id: string;
  displayName: string;
  age: number;
  sex: "M" | "F";
  ecog: Ecog;
  stage: Stage;
  ldh: LdhBand;
  braf: BrafStatus;
  nras: NrasStatus;
  /** Tumour PD-L1 by IHC, percent tumour proportion score. */
  pdl1: number;
  /** Q1 expression-response signature (IMPRES-style), 0-100 percentile. */
  signature: number;
  /** Short clinical vignette for the demo. */
  vignette: string;
}

export type TherapyKey = "immuno" | "targeted" | "combo";

export interface TherapyArm {
  key: TherapyKey;
  label: string;
  /** e.g. "Nivolumab + Ipilimumab (anti-PD-1 / anti-CTLA-4)" */
  regimen: string;
  mechanism: string;
}

export interface RankedOption {
  arm: TherapyArm;
  /** 0-100 model confidence for THIS patient. */
  confidence: number;
  /** Predicted median overall survival, months. */
  medianOsMonths: number;
  /** Predicted 12-month tumour-burden reduction, fraction 0-1. */
  burdenReduction: number;
  /** False when the underlying ODE simulation could not produce a usable
   *  trajectory for this arm — burdenReduction is not meaningful and should
   *  be displayed as "Could not compute", not as a real 0%. */
  burdenInformative?: boolean;
  /** One-line clinical rationale. */
  rationale: string;
  /** Guideline / evidence citation shown on the card. */
  evidence: string;
  /** Practical caution the consultant should weigh. */
  caution: string;
  tier: "primary" | "alternative" | "not-recommended";
  /** True if this therapy arm is hard-blocked due to biological contraindication (e.g. BRAF-WT). */
  hardBlocked?: boolean;
  /** Explicit clinical contraindication message when hard-blocked. */
  contraindication?: string;
}

export interface DecisionNode {
  id: string;
  label: string;
  detail: string;
}

export interface TriageResult {
  options: RankedOption[];
  path: DecisionNode[];
  /** KM-style survival series per arm for the chart. */
  headline: string;
}
