import type { DecisionNode, RankedOption, TherapyKey } from "../data/types";
import type { CohortPatient } from "../data/cohort";
import { biomarkerComposite, pdl1Band } from "../data/cohort";
import { PDL1_HIGH, buildRankedOptions, scoreArms, type ScoringContext } from "./scoring";

// Q5 — the integration engine.
//
// This is the synthesis step: it takes each method's independent read-out for a
// real TCGA patient and combines them into one ranked recommendation, while
// keeping the individual method positions visible so a consultant can see WHERE
// the methods agree and where they split.
//
//   Q1  statistical  gene-expression ML predictor  → P(response)
//   Q3  mechanistic  ODE digital twin              → simulated burden reduction
//   Q4  heuristic    resistance flags              → what to hold in reserve
//
// The honest bit: Q1 has not scored the TCGA cohort (see meta.q1Note), so the
// statistical side currently reads from the measured checkpoint biomarkers
// instead. `statistical.source` says which, every time, and the UI surfaces it.
// The moment real per-patient Q1 lands in cohort.json this switches over with no
// other code change.

export type EvidenceSource = "q1" | "biomarker" | "ode";

export interface MethodPosition {
  /** 0-1, where 1 = maximally favourable to checkpoint immunotherapy. */
  value: number;
  label: string;
  detail: string;
  source: EvidenceSource;
}

export type AgreementStatus = "concordant" | "partial" | "discordant" | "unavailable";

export interface AgreementResult {
  status: AgreementStatus;
  /** 1 - |statistical - mechanistic|, or null when one side is missing. */
  concordance: number | null;
  statistical: MethodPosition;
  mechanistic: MethodPosition | null;
  headline: string;
  detail: string;
}

export interface IntegratedResult {
  options: RankedOption[];
  path: DecisionNode[];
  headline: string;
  primaryKey: TherapyKey;
  agreement: AgreementResult;
  /** What fed the immunotherapy score, for the "show your working" panel. */
  evidence: MethodPosition[];
  context: ScoringContext;
}

const FAVOURABLE = 0.5;

/**
 * Rapid-control pressure. TCGA has no LDH, so stage IV stands in for it — and we
 * say so in the decision path rather than implying a lab value we don't have.
 * The what-if explorer can supply an explicit LDH, which then takes precedence.
 */
function hasRapidControlPressure(p: CohortPatient): boolean {
  if (p.ldhOverride) return p.ldhOverride !== "Normal";
  return p.stageBand === "IV";
}

/** The statistical (data-driven) position on immunotherapy. */
function statisticalPosition(p: CohortPatient): MethodPosition {
  if (p.q1) {
    return {
      value: p.q1.pResponse,
      label: `Q1 ML predictor · P(response) ${(p.q1.pResponse * 100).toFixed(0)}%`,
      detail: "Ensemble of five models over six gene-expression signatures.",
      source: "q1",
    };
  }
  const composite = biomarkerComposite(p);
  return {
    value: composite / 100,
    label: `Checkpoint biomarker composite · ${composite}th percentile`,
    detail:
      "Measured PD-L1 (CD274), PD-1 (PDCD1) and tumour mutational burden, ranked across the " +
      "cohort. Stands in for Q1 while the ML models cannot score TCGA expression.",
    source: "biomarker",
  };
}

/** The mechanistic (ODE) position on immunotherapy. Null when the twin is mute. */
function mechanisticPosition(p: CohortPatient): MethodPosition | null {
  if (!p.antipd1Informative || p.antipd1ReductionPct === null) return null;
  return {
    value: p.antipd1ReductionPct / 100,
    label: `Q3 digital twin · ${(p.antipd1Reduction * 100).toFixed(0)}% burden reduction`,
    detail: `Simulated anti-PD-1 dose sweep; ${p.antipd1ReductionPct}th percentile response across the cohort.`,
    source: "ode",
  };
}

function assessAgreement(
  statistical: MethodPosition,
  mechanistic: MethodPosition | null
): AgreementResult {
  if (!mechanistic) {
    return {
      status: "unavailable",
      concordance: null,
      statistical,
      mechanistic: null,
      headline: "Single method only",
      detail:
        "The ODE twin settled at a numerically-zero tumour compartment for this patient, so it " +
        "cannot corroborate the statistical read-out. Treat the recommendation as lower confidence.",
    };
  }

  const concordance = 1 - Math.abs(statistical.value - mechanistic.value);
  const statFavours = statistical.value >= FAVOURABLE;
  const mechFavours = mechanistic.value >= FAVOURABLE;
  const bothFavour = statFavours && mechFavours;

  if (statFavours !== mechFavours) {
    return {
      status: "discordant",
      concordance,
      statistical,
      mechanistic,
      headline: "Methods split — consultant review",
      detail: statFavours
        ? "The data-driven read-out favours immunotherapy but the mechanistic twin does not. " +
          "The biology and the statistics disagree; this patient warrants MDT discussion."
        : "The mechanistic twin favours immunotherapy but the data-driven read-out does not. " +
          "Consider whether the expression profile is under-calling an immune-responsive tumour.",
    };
  }

  if (concordance >= 0.7) {
    return {
      status: "concordant",
      concordance,
      statistical,
      mechanistic,
      headline: bothFavour
        ? "ML + ODE agree — immunotherapy favoured"
        : "ML + ODE agree — immunotherapy unfavoured",
      detail: bothFavour
        ? "Both the statistical and the mechanistic method independently predict checkpoint benefit."
        : "Both methods independently predict poor checkpoint benefit; weight the alternative lanes.",
    };
  }

  return {
    status: "partial",
    concordance,
    statistical,
    mechanistic,
    headline: "Same direction, differing magnitude",
    detail:
      "Both methods point the same way but disagree on how strongly. The recommendation holds, " +
      "with moderate rather than high confidence.",
  };
}

// ---------------------------------------------------------------------------
// Multi-step clinical reasoning chain
// ---------------------------------------------------------------------------

/**
 * Builds an inspectable, step-by-step clinical decision path explaining the
 * exact signals driving the recommendation for a given arm.
 *
 * Each string in the returned array is a concise reasoning step rendered as a
 * bullet on the primary recommendation card. The chain is generated from live
 * patient data at call time — no values are hardcoded.
 *
 * Steps tracked:
 *   1. MAPK driver mutation status (BRAF V600 / NRAS / Triple-WT)
 *   2. PD-L1 expression band (high / intermediate / low)
 *   3. Q1 ML response percentile (high / intermediate / low)
 *   4. Rapid-control pressure (LDH elevation or stage IV)
 *   5. Arm-specific clinical rationale conclusion
 */
function buildReasoningChain(p: CohortPatient, ctx: ScoringContext, armKey: TherapyKey): string[] {
  const steps: string[] = [];

  // Step 1: MAPK driver mutation status
  if (p.nras === "Mutant") {
    steps.push("NRAS-mutant — MAPK driver active via RAS; BRAF/MEK inhibitors not indicated");
  } else if (p.braf !== "WT") {
    steps.push(`BRAF ${p.braf} mutant — MAPK pathway constitutively active; targeted therapy on the table`);
  } else {
    steps.push("BRAF wild-type, NRAS wild-type — no targetable MAPK hotspot; checkpoint blockade is the standard first-line lane");
  }

  // Step 2: PD-L1 expression band
  if (p.pdl1Pct >= 75) {
    steps.push(`High PD-L1 (${p.pdl1Pct}th percentile) — strongly immuno-favourable biology`);
  } else if (p.pdl1Pct >= PDL1_HIGH) {
    steps.push(`Intermediate PD-L1 (${p.pdl1Pct}th percentile) — checkpoint benefit likely`);
  } else {
    steps.push(`Low PD-L1 (${p.pdl1Pct}th percentile) — immune-cold profile; checkpoint response attenuated`);
  }

  // Step 3: Q1 ML response signature
  if (p.q1) {
    // Use the true cohort-relative percentile rank when available (computed by
    // build_cohort.mjs); fall back to pResponse × 100 for legacy cohort.json files.
    const pct = p.q1.pResponsePct ?? Math.round(p.q1.pResponse * 100);
    if (pct >= 75) steps.push(`Q1 ML predictor: high response percentile (${pct}th) — strong statistical evidence for checkpoint benefit`);
    else if (pct >= 40) steps.push(`Q1 ML predictor: intermediate response percentile (${pct}th) — moderate checkpoint benefit signal`);
    else steps.push(`Q1 ML predictor: low response percentile (${pct}th) — statistical model does not favour checkpoint monotherapy`);
  } else {
    // Biomarker composite fallback
    const sig = ctx.signature;
    if (sig >= 75) steps.push(`Checkpoint biomarker composite: ${sig}th percentile — strong immuno-favourable signal`);
    else if (sig >= 40) steps.push(`Checkpoint biomarker composite: ${sig}th percentile — moderate immuno signal`);
    else steps.push(`Checkpoint biomarker composite: ${sig}th percentile — weak immuno signal`);
  }

  // Step 4: Rapid-control pressure
  if (ctx.highLdh) {
    steps.push(
      p.ldhOverride
        ? `LDH ${p.ldhOverride.toLowerCase()} — rapid tumour-control pressure; faster-acting regimens preferred`
        : "Stage IV disease (LDH not recorded in TCGA) — rapid-control pressure inferred from stage"
    );
  }

  // Step 5: Arm-specific conclusion
  if (armKey === "targeted") {
    steps.push("→ MAPK pathway is the primary actionable oncogenic driver; BRAF/MEK inhibition is the recommended lane");
  } else if (armKey === "combo") {
    steps.push("→ Combination / microenvironmental reversal strategy indicated — checkpoint monotherapy alone insufficient");
  } else {
    steps.push("→ High tumour immunogenicity and inflamed microenvironment — checkpoint blockade is the primary recommendation");
  }

  return steps;
}

export function integrate(p: CohortPatient): IntegratedResult {
  const statistical = statisticalPosition(p);
  const mechanistic = mechanisticPosition(p);
  const agreement = assessAgreement(statistical, mechanistic);

  // The Q1 slot in the shared scoring core takes the statistical position —
  // real Q1 output when it exists, the measured biomarker composite otherwise.
  const ctx: ScoringContext = {
    brafMut: p.braf !== "WT",
    pdl1: p.pdl1Pct,
    signature: Math.round(statistical.value * 100),
    highLdh: hasRapidControlPressure(p),
    ecog: 0, // performance status is not recorded in TCGA
    age: p.age ?? 60,
    immunoReduction: p.antipd1Informative ? p.antipd1Reduction : 0,
    targetedReduction: p.brafiInformative ? p.brafiReduction : 0,
    immunoInformative: p.antipd1Informative,
    targetedInformative: p.brafiInformative,
  };

  const scores = scoreArms(ctx);
  const options = buildRankedOptions(ctx, scores);

  // Hard-block safeguard: Arm B (targeted) is hard-blocked for BRAF Wild-Type
  // patients by `buildRankedOptions` (score → 0, hardBlocked: true). If the
  // top-ranked arm is still somehow flagged as blocked, fall back automatically
  // to the next eligible, non-contraindicated arm so no blocked arm is ever
  // surfaced as the primary recommendation.
  const candidateOptions = [...options];
  let primary = candidateOptions.find((o) => o.tier === "primary") ?? candidateOptions[0];
  if (primary.hardBlocked) {
    const fallback = candidateOptions.find((o) => !o.hardBlocked && o.confidence > 0);
    if (fallback) {
      // Demote blocked arm and promote fallback
      primary.tier = "not-recommended";
      fallback.tier = "primary";
      primary = fallback;
    }
  }

  // Attach an inspectable reasoning chain to whichever arm becomes primary
  const reasoningChain = buildReasoningChain(p, ctx, primary.arm.key);
  const finalOptions = candidateOptions.map((o) =>
    o.arm.key === primary.arm.key ? { ...o, reasoningChain } : o
  );

  const evidence: MethodPosition[] = [statistical];
  if (mechanistic) evidence.push(mechanistic);

  return {
    options: finalOptions,
    path: buildPath(p, ctx, agreement, primary.arm.label),
    headline: `${primary.arm.label} · ${primary.confidence}% model confidence · predicted median OS ${primary.medianOsMonths} mo`,
    primaryKey: primary.arm.key,
    agreement,
    evidence,
    context: ctx,
  };
}

function buildPath(
  p: CohortPatient,
  ctx: ScoringContext,
  agreement: AgreementResult,
  recommendation: string
): DecisionNode[] {
  const band = pdl1Band(p.pdl1Pct);
  return [
    {
      id: "stage",
      label: p.stage === "Unknown" ? "Stage not recorded" : `Stage ${p.stage}`,
      detail: p.ldhOverride
        ? `LDH ${p.ldhOverride.toLowerCase()}${ctx.highLdh ? " — rapid-control pressure" : ""}`
        : ctx.highLdh
          ? "Stage IV — rapid-control pressure (LDH not recorded in TCGA)"
          : "No rapid-control pressure recorded",
    },
    {
      id: "braf",
      label: ctx.brafMut ? "BRAF V600 (mutant)" : "BRAF wild-type",
      detail: ctx.brafMut
        ? "Targeted therapy is on the table"
        : p.nras === "Mutant"
          ? "NRAS-mutant — no BRAF/MEK option, MEK-directed salvage only"
          : "No BRAF/MEK option",
    },
    {
      id: "pdl1",
      label: `PD-L1 ${band.toLowerCase()} · ${p.pdl1Pct}th percentile`,
      detail: p.pdl1Pct >= PDL1_HIGH ? "Immuno-favourable biology" : "Immuno-cold biology",
    },
    {
      id: "agreement",
      label: agreement.headline,
      detail:
        agreement.concordance === null
          ? "Only one method could score this patient"
          : `Methods concordance ${(agreement.concordance * 100).toFixed(0)}%`,
    },
    {
      id: "rec",
      label: recommendation,
      detail: "Proposed — pending consultant sign-off",
    },
  ];
}
