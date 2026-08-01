import type { DecisionNode, RankedOption, TherapyKey } from "../data/types";
import type { CohortPatient } from "../data/cohort";
import { biomarkerComposite, pdl1Band } from "../data/cohort";
import { PDL1_HIGH, buildRankedOptions, scoreArms, type ScoringContext } from "./scoring";

// Q5 – the integration engine.
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
 * Rapid-control pressure. TCGA has no LDH, so stage IV stands in for it – and we
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
      headline: "Methods split – consultant review",
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
        ? "ML + ODE agree – immunotherapy favoured"
        : "ML + ODE agree – immunotherapy unfavoured",
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

export function integrate(p: CohortPatient): IntegratedResult {
  const statistical = statisticalPosition(p);
  const mechanistic = mechanisticPosition(p);
  const agreement = assessAgreement(statistical, mechanistic);

  const ctx: ScoringContext = {
    brafMut: p.braf !== "WT",
    pdl1: p.pdl1Pct,
    signature: Math.round(statistical.value * 100),
    highLdh: hasRapidControlPressure(p),
    ecog: 0,
    age: p.age ?? 60,
    immunoReduction: p.antipd1Informative ? p.antipd1Reduction : 0,
    targetedReduction: p.brafiInformative ? p.brafiReduction : 0,
  };

  let options: RankedOption[];
  let primaryKey: TherapyKey;

  if (p.q5) {
    const q5ArmMap: Record<"A" | "B" | "C", TherapyKey> = {
      A: "immuno",
      B: "targeted",
      C: "combo",
    };
    primaryKey = q5ArmMap[p.q5.treatmentArm] ?? "immuno";

    const baseOptions = buildRankedOptions(ctx, scoreArms(ctx));
    
    // Override the primary option with Q5's explicit recommendation details
    options = baseOptions.map((opt) => {
      if (opt.arm.key === primaryKey) {
        const confidence = p.q5!.treatabilityIndex !== null 
          ? Math.round(p.q5!.treatabilityIndex) 
          : opt.confidence;
        return {
          ...opt,
          confidence,
          tier: "primary" as const,
          rationale: p.q5!.recommendedTherapy || opt.rationale,
          evidence: `Q5 Stratification: ${p.q5!.shortLabel} (${p.q5!.confidenceBand} Confidence)`,
          caution: p.q5!.q4NominatedTarget 
            ? `Q4 Target nominated: ${p.q5!.q4NominatedTarget}` 
            : opt.caution,
        };
      } else {
        return {
          ...opt,
          tier: opt.tier === "primary" ? ("alternative" as const) : opt.tier,
        };
      }
    });
    // Ensure primary option is first
    options.sort((a, b) => (a.tier === "primary" ? -1 : b.tier === "primary" ? 1 : 0));
  } else {
    const scores = scoreArms(ctx);
    options = buildRankedOptions(ctx, scores);
    const primary = options.find((o) => o.tier === "primary") ?? options[0];
    primaryKey = primary.arm.key;
  }

  const primary = options.find((o) => o.tier === "primary") ?? options[0];
  const evidence: MethodPosition[] = [statistical];
  if (mechanistic) evidence.push(mechanistic);

  return {
    options,
    path: buildPath(p, ctx, agreement, primary.arm.label),
    headline: `${primary.arm.label} · TI score ${primary.confidence}/100 · predicted median OS ${primary.medianOsMonths} mo`,
    primaryKey,
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
  const nodes: DecisionNode[] = [
    {
      id: "stage",
      label: p.stage === "Unknown" ? "Stage not recorded" : `Stage ${p.stage}`,
      detail: p.ldhOverride
        ? `LDH ${p.ldhOverride.toLowerCase()}${ctx.highLdh ? " – rapid-control pressure" : ""}`
        : ctx.highLdh
          ? "Stage IV – rapid-control pressure (LDH not recorded in TCGA)"
          : "No rapid-control pressure recorded",
    },
    {
      id: "braf",
      label: ctx.brafMut ? "BRAF V600 (mutant)" : "BRAF wild-type",
      detail: ctx.brafMut
        ? "Targeted therapy is on the table"
        : p.nras === "Mutant"
          ? "NRAS-mutant – no BRAF/MEK option, MEK-directed salvage only"
          : "No BRAF/MEK option",
    },
  ];

  if (p.q5) {
    nodes.push({
      id: "phenotype",
      label: `Q5 · ${p.q5.shortLabel}`,
      detail: `Treatability Index ${p.q5.treatabilityIndex !== null ? p.q5.treatabilityIndex.toFixed(0) : 'N/A'}/100 · ${p.q5.confidenceBand} confidence`,
    });
  } else {
    const band = pdl1Band(p.pdl1Pct);
    nodes.push({
      id: "pdl1",
      label: `PD-L1 ${band.toLowerCase()} · ${p.pdl1Pct}th percentile`,
      detail: p.pdl1Pct >= PDL1_HIGH ? "Immuno-favourable biology" : "Immuno-cold biology",
    });
  }

  nodes.push(
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
      detail: "Proposed – pending consultant sign-off",
    }
  );

  return nodes;
}
