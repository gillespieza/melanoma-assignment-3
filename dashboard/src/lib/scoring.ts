import type { RankedOption, TherapyArm, TherapyKey } from "../data/types";
import { KM_FACTS } from "../data/model";

// Shared scoring core, used by integrationEngine.integrate() for every patient
// — real or edited in the what-if explorer. Keeping the maths in one place
// means a hypothetical profile and a real patient with the same values always
// produce the same recommendation.
//
// Decision support only: ranked, evidence-anchored options; the consultant
// confirms the plan.

export const PDL1_HIGH = 25; // TPS % / percentile threshold, kept consistent app-wide

export const ARMS: Record<TherapyKey, TherapyArm> = {
  immuno: {
    key: "immuno",
    label: "Immunotherapy",
    regimen: "Nivolumab + Ipilimumab (anti-PD-1 / anti-CTLA-4)",
    mechanism: "Releases immune checkpoints so cytotoxic T-cells attack the tumour.",
  },
  targeted: {
    key: "targeted",
    label: "Targeted therapy",
    regimen: "Encorafenib + Binimetinib (BRAF / MEK inhibition)",
    mechanism: "Blocks the constitutively active BRAF-V600 → MEK → ERK growth signal.",
  },
  combo: {
    key: "combo",
    label: "Combination / sequencing",
    regimen: "Checkpoint induction → BRAF/MEK on progression (or triplet)",
    mechanism: "Durable immune control with targeted therapy held in reserve for rapid rescue.",
  },
};

export function clamp(n: number, lo = 0, hi = 100) {
  return Math.max(lo, Math.min(hi, n));
}

/** Everything the scoring math needs, however the caller obtained it. */
export interface ScoringContext {
  brafMut: boolean;
  /** PD-L1 as a 0-100 display value (IHC TPS for archetypes, cohort percentile for TCGA). */
  pdl1: number;
  /**
   * Expression-response signature, 0-100. This is the Q1 slot: the real Q1
   * ensemble P(response) x 100 when the patient has been scored, otherwise the
   * measured biomarker composite.
   */
  signature: number;
  /** Rapid-control pressure (elevated LDH, or stage IV where LDH is unavailable). */
  highLdh: boolean;
  ecog: number;
  age: number;
  /** Q3 ODE anti-PD-1 burden reduction, 0-1. */
  immunoReduction: number;
  /** Q3 ODE BRAFi burden reduction, 0-1. */
  targetedReduction: number;
}

export interface ArmScores {
  immuno: number;
  targeted: number;
  combo: number;
  comboReduction: number;
}

/** The confidence math. Deliberately transparent and inspectable. */
export function scoreArms(ctx: ScoringContext): ArmScores {
  const { brafMut, pdl1, signature, highLdh, ecog, age } = ctx;
  const pdl1High = pdl1 >= PDL1_HIGH;
  const immunoReduction = ctx.immunoReduction;
  const targetedReduction = ctx.targetedReduction;

  const comboReduction =
    clamp((1 - (1 - immunoReduction) * (1 - targetedReduction * 0.7)) * 100) / 100;

  // Immunotherapy: favoured by PD-L1, expression signature, WT status, good PS.
  const immuno =
    30 +
    (pdl1 / 100) * 34 +
    (signature / 100) * 22 +
    (brafMut ? 0 : 8) -
    (highLdh ? 12 : 0) -
    ecog * 6 +
    immunoReduction * 20;

  // Targeted: only if BRAF-mutant. Favoured for rapid control (high LDH/bulky).
  const targeted = brafMut
    ? 34 + targetedReduction * 42 + (highLdh ? 16 : 0) + (pdl1High ? -6 : 8) - ecog * 3
    : 0;

  // Combination / sequencing: needs BRAF-mutant AND an immuno-responsive tumour.
  const combo =
    brafMut && (pdl1High || signature >= 50)
      ? 30 + comboReduction * 34 + (pdl1High ? 12 : 0) - ecog * 6 - (age >= 75 ? 10 : 0)
      : brafMut && highLdh
        ? 44
        : 0;

  return {
    immuno: clamp(immuno),
    targeted: clamp(targeted),
    combo: clamp(combo),
    comboReduction,
  };
}

/** Predicted median OS, anchored to the real Q3 Kaplan-Meier medians. */
export function osFor(key: TherapyKey, reductionFrac: number): number {
  if (key === "immuno") return Math.round(KM_FACTS.checkpoint.lowMedianOs * (0.5 + 0.5 * reductionFrac));
  if (key === "targeted") return Math.round(KM_FACTS.brafi.lowMedianOs * (0.55 + 0.45 * reductionFrac));
  return Math.round(KM_FACTS.checkpoint.lowMedianOs * (0.6 + 0.45 * reductionFrac));
}

const pct = (frac: number) => `${Math.round(frac * 100)}%`;

/**
 * Builds the three option cards, ranks them by confidence and assigns tiers.
 * The top eligible arm becomes `primary`; ineligible arms stay `not-recommended`.
 */
export function buildRankedOptions(ctx: ScoringContext, scores: ArmScores): RankedOption[] {
  const { brafMut, pdl1, highLdh, immunoReduction, targetedReduction } = ctx;
  const pdl1High = pdl1 >= PDL1_HIGH;

  const draft: RankedOption[] = [
    {
      arm: ARMS.immuno,
      confidence: Math.round(scores.immuno),
      medianOsMonths: osFor("immuno", immunoReduction),
      burdenReduction: immunoReduction,
      rationale: pdl1High
        ? "High PD-L1 and a strong response signature predict durable checkpoint benefit."
        : brafMut
          ? `Low PD-L1: modelled checkpoint response is weak (~${pct(immunoReduction)} burden reduction).`
          : "No targetable driver; checkpoint blockade is the standard first-line lane.",
      evidence: "DREAMseq: 72% vs 52% alive at 2y (immuno-first). NCCN Cat 1 first-line.",
      caution: highLdh
        ? "Slower to act — risky if disease is rapidly progressing."
        : "Immune-related toxicity; ~40% grade 3-4 with the ipi/nivo doublet.",
      tier: "alternative",
    },
    {
      arm: ARMS.targeted,
      confidence: Math.round(scores.targeted),
      medianOsMonths: brafMut ? osFor("targeted", targetedReduction) : 0,
      burdenReduction: brafMut ? targetedReduction : 0,
      rationale: !brafMut
        ? "No BRAF V600 mutation — BRAF/MEK inhibitors have no target (RAF paradox risk)."
        : highLdh
          ? "Fast, deep response — appropriate for rapid control of bulky/symptomatic disease."
          : "Effective but responses are often not durable; resistance emerges.",
      evidence: "COLUMBUS: encorafenib+binimetinib median OS 33.6 mo in BRAF-V600.",
      caution: brafMut
        ? "Acquired resistance typically within 9-12 months; plan the next line early."
        : "Contraindicated without a BRAF V600 mutation.",
      tier: brafMut ? "alternative" : "not-recommended",
      hardBlocked: !brafMut,
      contraindication: !brafMut
        ? "Contraindicated: Patient is BRAF Wild-Type (lacks BRAF V600 hotspot). BRAF inhibitors cause paradoxical MAPK activation."
        : undefined,
    },
    {
      arm: ARMS.combo,
      confidence: Math.round(scores.combo),
      medianOsMonths: scores.combo > 0 ? osFor("combo", scores.comboReduction) : 0,
      burdenReduction: scores.combo > 0 ? scores.comboReduction : 0,
      rationale:
        scores.combo > 0
          ? "Both lanes active: checkpoint induction with targeted therapy reserved for rescue."
          : "Reserved for BRAF-mutant tumours with an immuno-responsive profile.",
      evidence: "SECOMBIT: sandwich/sequencing improves 3y OS vs targeted-first.",
      caution: "Higher cumulative toxicity and monitoring burden; MDT discussion advised.",
      tier: scores.combo > 0 ? "alternative" : "not-recommended",
    },
  ];

  const ranked = [...draft].sort((a, b) => b.confidence - a.confidence);
  let primaryAssigned = false;
  for (const opt of ranked) {
    if (opt.tier === "not-recommended") continue;
    if (!primaryAssigned && opt.confidence > 0) {
      opt.tier = "primary";
      primaryAssigned = true;
    } else {
      opt.tier = "alternative";
    }
  }
  return ranked;
}
