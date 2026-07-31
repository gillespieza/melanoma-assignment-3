import type { RankedOption } from "../data/types";

// Builds the 12-month "Tumour Burden Forecast" and the survival curves.
//
// The ENDPOINTS (how much each therapy shrinks the tumour) are anchored to the
// q3 ODE dose-response output. The month-by-month DYNAMICS encode well-known
// clinical behaviour: BRAF/MEK gives a fast, deep nadir then resistance rebound;
// checkpoint immunotherapy is slower but durable; combination/sequencing holds
// the deepest durable response. Baseline (no systemic therapy) grows.

export interface ForecastPoint {
  month: number;
  baseline: number;
  immuno: number;
  targeted: number | null;
  combo: number | null;
}

const MONTHS = Array.from({ length: 13 }, (_, i) => i); // 0..12

function finalBurden(reductionFrac: number): number {
  // Start index = 100 (% of diagnosis burden). Final = 100 * (1 - reduction).
  return 100 * (1 - reductionFrac);
}

function immunoTrajectory(reductionFrac: number): number[] {
  const end = finalBurden(reductionFrac);
  // Slow onset (pseudo-progression possible early), then steady durable decline.
  return MONTHS.map((m) => {
    const bump = m <= 1 ? 4 * m : 0; // mild early rise
    const decay = 1 - Math.exp(-m / 3.2);
    return Math.max(end, 100 + bump - (100 - end) * decay);
  });
}

function targetedTrajectory(reductionFrac: number): number[] {
  const nadir = finalBurden(reductionFrac);
  // Fast deep nadir by ~month 3, then partial resistance rebound.
  return MONTHS.map((m) => {
    const drop = 1 - Math.exp(-m / 1.1);
    const nadirVal = 100 - (100 - nadir) * drop;
    const rebound = m > 3 ? (m - 3) * (reductionFrac > 0.1 ? 3.4 : 0.4) : 0;
    return Math.min(102, nadirVal + rebound);
  });
}

function comboTrajectory(reductionFrac: number): number[] {
  const end = finalBurden(reductionFrac);
  // Fast initial control (targeted) handing off to durable immune plateau.
  return MONTHS.map((m) => {
    const drop = 1 - Math.exp(-m / 1.6);
    return Math.max(end, 100 - (100 - end) * drop);
  });
}

function baselineTrajectory(): number[] {
  // Untreated melanoma: exponential-ish growth capped at 200% for the axis.
  return MONTHS.map((m) => Math.min(200, 100 * Math.exp(m / 16)));
}

export interface Forecast {
  data: ForecastPoint[];
  hasTargeted: boolean;
  hasCombo: boolean;
}

/**
 * Builds the forecast from ranked options alone, so it stays decoupled from
 * however those options were produced.
 */
export function forecastFromOptions(options: RankedOption[]): Forecast {
  const opt = (key: string) => options.find((o) => o.arm.key === key)!;
  const immunoOpt = opt("immuno");
  const targetedOpt = opt("targeted");
  const comboOpt = opt("combo");

  const hasTargeted = targetedOpt.tier !== "not-recommended";
  const hasCombo = comboOpt.confidence > 0;

  const immuno = immunoTrajectory(immunoOpt.burdenReduction);
  const targeted = hasTargeted ? targetedTrajectory(targetedOpt.burdenReduction) : null;
  const combo = hasCombo ? comboTrajectory(comboOpt.burdenReduction) : null;
  const baseline = baselineTrajectory();

  const data: ForecastPoint[] = MONTHS.map((m) => ({
    month: m,
    baseline: round(baseline[m]),
    immuno: round(immuno[m]),
    targeted: targeted ? round(targeted[m]) : null,
    combo: combo ? round(combo[m]) : null,
  }));

  return { data, hasTargeted, hasCombo };
}

// ---- Survival (KM-style step curves) --------------------------------------

export interface SurvivalPoint {
  month: number;
  recommended: number;
  /** Null when the patient has no second eligible lane (e.g. BRAF wild-type,
   *  where targeted and combination are both off the table). */
  alternative: number | null;
}

/** Weibull-ish survival curve reaching S=0.5 at the given median. */
function survivalCurve(medianMonths: number, horizon = 120): number[] {
  const lambda = medianMonths / Math.pow(Math.log(2), 1 / 1.3);
  const pts: number[] = [];
  for (let m = 0; m <= horizon; m += 3) {
    pts.push(Math.exp(-Math.pow(m / lambda, 1.3)) * 100);
  }
  return pts;
}

export interface Survival {
  points: SurvivalPoint[];
  recMedian: number;
  /** Null when there is no distinct second lane — better than drawing a
   *  duplicate curve and implying a choice the patient does not have. */
  altMedian: number | null;
  altLabel: string | null;
}

export function survivalFromOptions(options: RankedOption[]): Survival {
  const eligible = options.filter((o) => o.tier !== "not-recommended");
  const rec = eligible[0];
  const alt = eligible[1] ?? null;
  const recCurve = survivalCurve(rec.medianOsMonths);
  const altCurve = alt ? survivalCurve(alt.medianOsMonths) : null;
  const points: SurvivalPoint[] = recCurve.map((v, i) => ({
    month: i * 3,
    recommended: round(v),
    alternative: altCurve ? round(altCurve[i]) : null,
  }));
  return {
    points,
    recMedian: rec.medianOsMonths,
    altMedian: alt ? alt.medianOsMonths : null,
    altLabel: alt ? alt.arm.label : null,
  };
}

function round(n: number): number {
  return Math.round(n * 10) / 10;
}
