#!/usr/bin/env node
// ---------------------------------------------------------------------------
// build_cohort.mjs — generates dashboard/public/cohort.json, the single file
// the OncoTwin app consumes.
//
// Sources (all real project output, nothing synthesised):
//   q3-ode-model/outputs/results/tumour_burden_simulations.csv      BRAFi sweep
//   q3-ode-model/outputs/results/checkpoint_tumour_simulations.csv  anti-PD-1 sweep
//   data/processed/skcm_tcga_pan_can_atlas_2018/clin_cleaned.csv    TCGA clinical
//   dashboard/public/q1_predictions.csv                             Q1 ML output
//
// Idempotent: re-run it any time. If q1_predictions.csv gains TCGA-SKCM rows,
// the per-patient `q1` block fills in with zero front-end changes.
//
//   node scripts/build_cohort.mjs
// ---------------------------------------------------------------------------

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DASHBOARD = resolve(HERE, "..");
const REPO = resolve(DASHBOARD, "..");

const PATHS = {
  tumour: resolve(REPO, "q3-ode-model/outputs/results/tumour_burden_simulations.csv"),
  checkpoint: resolve(REPO, "q3-ode-model/outputs/results/checkpoint_tumour_simulations.csv"),
  clinical: resolve(REPO, "data/processed/skcm_tcga_pan_can_atlas_2018/clin_cleaned.csv"),
  q1: resolve(DASHBOARD, "public/q1_predictions.csv"),
  out: resolve(DASHBOARD, "public/cohort.json"),
};

const DOSE_AXIS = [0.01, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 1.0];
const BRAFI_COLS = DOSE_AXIS.map((d) => `BRAFi_${d.toFixed(3)}`);
const ANTIPD1_COLS = DOSE_AXIS.map((d) => `antiPD1_${d.toFixed(3)}`);
const Q1_MODELS = ["lr", "rf", "xgb", "svm", "enet"];
const Q1_FEATURES = ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "IMPRES", "PD_L1"];

// ---- CSV -------------------------------------------------------------------

/** Quote-aware CSV parse → array of row objects. */
function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else quoted = false;
      } else field += c;
      continue;
    }
    if (c === '"') quoted = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
    else if (c !== "\r") field += c;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  const header = rows.shift();
  return rows
    .filter((r) => r.length > 1)
    .map((r) => Object.fromEntries(header.map((h, i) => [h, r[i] ?? ""])));
}

function readCsv(path) {
  return parseCsv(readFileSync(path, "utf8"));
}

/** "" / "NA" / "NaN" → null, else Number. */
function num(v) {
  if (v === undefined || v === null) return null;
  const s = String(v).trim();
  if (!s || s === "NA" || s === "NaN" || s === "nan" || s === "None") return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

const round = (n, dp = 4) => (n === null ? null : Math.round(n * 10 ** dp) / 10 ** dp);

// ---- stats -----------------------------------------------------------------

/** Percentile rank (0-100) of each value within `values`. */
function percentileRanks(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return values.map((v) => {
    // fraction of the cohort strictly below, plus half the ties → stable midranks
    let below = 0;
    let ties = 0;
    for (const s of sorted) {
      if (s < v) below++;
      else if (s === v) ties++;
    }
    return Math.round(((below + ties / 2) / sorted.length) * 100);
  });
}

/** AUC via the Mann-Whitney U identity, with midrank handling for ties. */
function auc(labels, scores) {
  const pairs = labels
    .map((y, i) => ({ y, s: scores[i] }))
    .filter((p) => p.y !== null && p.s !== null);
  const pos = pairs.filter((p) => p.y === 1).length;
  const neg = pairs.length - pos;
  if (!pos || !neg) return null;
  const order = [...pairs].sort((a, b) => a.s - b.s);
  const ranks = new Array(order.length);
  for (let i = 0; i < order.length; ) {
    let j = i;
    while (j + 1 < order.length && order[j + 1].s === order[i].s) j++;
    const mid = (i + j) / 2 + 1; // 1-based midrank
    for (let k = i; k <= j; k++) ranks[k] = mid;
    i = j + 1;
  }
  const rankSumPos = order.reduce((acc, p, i) => acc + (p.y === 1 ? ranks[i] : 0), 0);
  return (rankSumPos - (pos * (pos + 1)) / 2) / (pos * neg);
}

/** Confusion matrix + derived rates at a probability threshold. */
function classify(labels, scores, threshold = 0.5) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  labels.forEach((y, i) => {
    if (y === null || scores[i] === null) return;
    const pred = scores[i] >= threshold ? 1 : 0;
    if (y === 1 && pred === 1) tp++;
    else if (y === 0 && pred === 1) fp++;
    else if (y === 0 && pred === 0) tn++;
    else fn++;
  });
  const n = tp + fp + tn + fn;
  const safe = (a, b) => (b ? a / b : null);
  return {
    threshold,
    n, tp, fp, tn, fn,
    accuracy: safe(tp + tn, n),
    sensitivity: safe(tp, tp + fn),
    specificity: safe(tn, tn + fp),
    ppv: safe(tp, tp + fp),
    npv: safe(tn, tn + fn),
  };
}

/**
 * Threshold that maximises Youden's J. The models were calibrated on a cohort
 * with a much higher response rate than these trials, so 0.5 is a poor operating
 * point — we report both and let the panel show the honest comparison.
 */
function bestThreshold(labels, scores) {
  const cand = [...new Set(scores.filter((s) => s !== null))].sort((a, b) => a - b);
  let best = { threshold: 0.5, youden: -Infinity };
  for (const t of cand) {
    const c = classify(labels, scores, t);
    if (c.sensitivity === null || c.specificity === null) continue;
    const youden = c.sensitivity + c.specificity - 1;
    if (youden > best.youden) best = { threshold: t, youden };
  }
  return best.threshold;
}

// ---- clinical normalisation -------------------------------------------------

function normaliseStage(raw) {
  if (!raw) return "Unknown";
  const s = raw.replace(/^STAGE\s*/i, "").trim();
  if (!s || /^\[?not/i.test(s)) return "Unknown";
  if (/NOS/i.test(s)) return s.replace(/\s*\(NOS\)/i, "").trim();
  return s;
}

/** Coarse stage band for filtering/sorting. */
function stageBand(stage) {
  if (stage === "Unknown") return "Unknown";
  if (stage.startsWith("IV")) return "IV";
  if (stage.startsWith("III")) return "III";
  if (stage.startsWith("II")) return "II";
  if (stage.startsWith("I")) return "I";
  if (stage.startsWith("0")) return "0";
  return "Unknown";
}

// ---- Q3 curve derivations ---------------------------------------------------

// About 21% of the cohort's ODE solutions settle at a numerically-zero tumour
// compartment (baseline < 1e-3). For those the twin has no established burden to
// shrink, so a "reduction" is undefined — NOT zero. Conflating the two would tell
// a clinician the tumour is drug-refractory when the model simply has nothing to
// say. We flag them instead and let the engine and UI down-weight Q3 accordingly.
const BURDEN_FLOOR = 1e-3;

const isInformative = (curve) => Number.isFinite(curve[0]) && curve[0] >= BURDEN_FLOOR;

/** Fractional burden reduction achievable across the dose sweep. */
function reduction(curve) {
  if (!isInformative(curve)) return 0;
  const best = Math.min(...curve);
  return Math.max(0, Math.min(1, 1 - best / curve[0]));
}

/** Dose at which the curve first comes within 2% of its minimum. */
function optimalDose(curve) {
  if (!isInformative(curve)) return null;
  const best = Math.min(...curve);
  const span = curve[0] - best;
  for (let i = 0; i < curve.length; i++) {
    if (span <= 0 || curve[i] <= best + 0.02 * span) return DOSE_AXIS[i];
  }
  return DOSE_AXIS[DOSE_AXIS.length - 1];
}

/**
 * Curve rescaled so baseline = 100 (% of untreated burden) — the form every
 * chart wants, and scale-free so patients stay comparable. Null when the twin
 * has no meaningful baseline.
 */
function normalisedCurve(curve) {
  if (!isInformative(curve)) return null;
  return curve.map((v) => round((v / curve[0]) * 100, 2));
}

// ---- Q4 heuristics ----------------------------------------------------------
// Explicitly heuristic — derived from Q3 residual burden + driver status, NOT a
// fitted resistance model. The UI must label them as such.

function deriveQ4(p) {
  const flags = [];
  const odeUsable = p.brafiInformative || p.antipd1Informative;
  const bestReduction = Math.max(
    p.brafiInformative ? p.brafiReduction : 0,
    p.antipd1Informative ? p.antipd1Reduction : 0
  );

  if (p.nras === "Mutant") {
    flags.push({
      label: "NRAS-driven MAPK reactivation",
      detail: "NRAS mutation sustains MEK/ERK signalling downstream of BRAF blockade.",
    });
  }
  if (p.braf !== "WT" && p.mapkDriven) {
    flags.push({
      label: "BRAFi escape via MAPK rebound",
      detail: "MAPK-driven tumour: expect ERK reactivation and loss of BRAFi control.",
    });
  }
  if (p.pdl1Pct < 25) {
    flags.push({
      label: "Immune-cold (low PD-L1)",
      detail: "Bottom-quartile CD274 expression; primary checkpoint resistance is likely.",
    });
  }
  if (p.braf === "WT" && p.nras === "WT") {
    flags.push({
      label: "Triple-WT lineage dependency",
      detail: "No MAPK driver — SOX10/MITF lineage survival is the tractable axis.",
    });
  }
  if (!odeUsable) {
    flags.push({
      label: "Twin below model resolution",
      detail:
        "The ODE settles at a numerically-zero tumour compartment for this patient, so the " +
        "simulated arms carry no information — resistance risk is not assessable from Q3.",
    });
  } else if (bestReduction < 0.1) {
    flags.push({
      label: "Refractory in silico",
      detail: "Neither simulated arm clears meaningful burden across the full dose sweep.",
    });
  }

  const resistanceRisk = !odeUsable
    ? "unknown"
    : bestReduction < 0.15 ? "high" : bestReduction < 0.4 ? "moderate" : "low";

  // Reserve (salvage) targets, ordered by how well they fit this biology.
  const reserve = [];
  if (p.braf !== "WT") reserve.push("MEK inhibitor re-challenge after drug holiday");
  if (p.nras === "Mutant") reserve.push("MEK + CDK4/6 inhibition (NRAS-mutant salvage)");
  reserve.push("SOX10 / MITF lineage-switch targeting");
  if (p.pdl1Pct >= 50) reserve.push("LAG-3 blockade (relatlimab + nivolumab)");
  if (p.tmb !== null && p.tmb >= 20) reserve.push("High TMB — retry checkpoint blockade");

  return { resistanceRisk, flags, reserve };
}

// ---- Q1 ---------------------------------------------------------------------

function buildQ1(rows) {
  if (!rows.length) return { byPatient: new Map(), validation: null, nTcga: 0 };

  const parsed = rows.map((r) => ({
    sampleId: r.SAMPLE_ID,
    patientId: r.PATIENT_ID,
    cohort: r.cohort,
    features: Object.fromEntries(Q1_FEATURES.map((f) => [f, num(r[f])])),
    perModel: Object.fromEntries(Q1_MODELS.map((m) => [m, num(r[`prob_${m}`])])),
    pResponse: num(r.prob_ensemble),
    predLabel: num(r.pred_label),
    actual: num(r.actual_response),
    osMonths: num(r.os_months),
  }));

  // --- per-patient map, TCGA rows only (they join to the Q3 twin cohort) ---
  const byPatient = new Map();
  for (const r of parsed) {
    if (!/tcga/i.test(r.cohort)) continue;
    byPatient.set(r.patientId, {
      pResponse: round(r.pResponse),
      perModel: Object.fromEntries(Q1_MODELS.map((m) => [m, round(r.perModel[m])])),
      features: Object.fromEntries(Q1_FEATURES.map((f) => [f, round(r.features[f])])),
      cohort: r.cohort,
    });
  }

  // --- cohort-level validation on the labelled ICI trials ------------------
  const labelled = parsed.filter((r) => r.actual === 0 || r.actual === 1);
  let validation = null;
  if (labelled.length) {
    const y = labelled.map((r) => r.actual);
    const p = labelled.map((r) => r.pResponse);
    const tuned = bestThreshold(y, p);
    validation = {
      n: labelled.length,
      responders: y.filter((v) => v === 1).length,
      auc: round(auc(y, p), 4),
      atHalf: classify(y, p, 0.5),
      atTuned: classify(y, p, tuned),
      perModel: Q1_MODELS.map((m) => ({
        model: m,
        auc: round(auc(y, labelled.map((r) => r.perModel[m])), 4),
      })).filter((d) => d.auc !== null),
      byCohort: [...new Set(labelled.map((r) => r.cohort))].map((c) => {
        const g = labelled.filter((r) => r.cohort === c);
        return {
          cohort: c,
          n: g.length,
          responders: g.filter((r) => r.actual === 1).length,
          auc: round(auc(g.map((r) => r.actual), g.map((r) => r.pResponse)), 4),
          meanProb: round(g.reduce((a, r) => a + r.pResponse, 0) / g.length, 4),
        };
      }).sort((a, b) => b.n - a.n),
      // Response-probability distribution, split by the true outcome — this is
      // what makes "the model genuinely separates responders" visible.
      distribution: buildDistribution(labelled),
      roc: buildRoc(y, p),
    };
  }

  return { byPatient, validation, nTcga: byPatient.size };
}

function buildDistribution(labelled, bins = 10) {
  const out = [];
  for (let i = 0; i < bins; i++) {
    const lo = i / bins;
    const hi = (i + 1) / bins;
    const inBin = (r) => r.pResponse >= lo && (i === bins - 1 ? r.pResponse <= hi : r.pResponse < hi);
    out.push({
      bin: `${Math.round(lo * 100)}-${Math.round(hi * 100)}%`,
      lo: round(lo, 2),
      responders: labelled.filter((r) => r.actual === 1 && inBin(r)).length,
      nonResponders: labelled.filter((r) => r.actual === 0 && inBin(r)).length,
    });
  }
  return out;
}

function buildRoc(y, p) {
  const thresholds = [...new Set(p)].sort((a, b) => b - a);
  const pts = [{ fpr: 0, tpr: 0 }];
  for (const t of thresholds) {
    const c = classify(y, p, t);
    pts.push({ fpr: round(1 - c.specificity, 4), tpr: round(c.sensitivity, 4) });
  }
  pts.push({ fpr: 1, tpr: 1 });
  return pts;
}

// ---- main -------------------------------------------------------------------

function main() {
  for (const key of ["tumour", "checkpoint", "clinical"]) {
    if (!existsSync(PATHS[key])) {
      console.error(`ERROR: required input missing — ${PATHS[key]}`);
      process.exit(1);
    }
  }

  const tumour = readCsv(PATHS.tumour);
  const checkpoint = readCsv(PATHS.checkpoint);
  const clinical = readCsv(PATHS.clinical);

  const checkpointBySample = new Map(checkpoint.map((r) => [r.SAMPLE_ID, r]));
  const clinByPatient = new Map(clinical.map((r) => [r.PATIENT_ID, r]));

  const q1Csv = existsSync(PATHS.q1) ? readCsv(PATHS.q1) : [];
  const { byPatient: q1ByPatient, validation: q1Validation, nTcga } = buildQ1(q1Csv);
  if (!q1Csv.length) console.warn("  note: public/q1_predictions.csv absent — q1 will be null");

  // Pass 1 — assemble what we can per patient.
  const draft = [];
  let missingCheckpoint = 0;
  for (const t of tumour) {
    const c = checkpointBySample.get(t.SAMPLE_ID);
    if (!c) { missingCheckpoint++; continue; }
    const clin = clinByPatient.get(t.PATIENT_ID) ?? {};

    const brafiCurve = BRAFI_COLS.map((k) => num(t[k]) ?? 0);
    const antipd1Curve = ANTIPD1_COLS.map((k) => num(c[k]) ?? 0);
    const stage = normaliseStage(clin.AJCC_PATHOLOGIC_TUMOR_STAGE);

    draft.push({
      id: t.PATIENT_ID,
      sampleId: t.SAMPLE_ID,
      braf: num(t.BRAF_MUT) === 1 ? "V600E" : "WT",
      nras: num(t.NRAS_MUT) === 1 ? "Mutant" : "WT",
      mapkDriven: num(t.MAPK_DRIVEN) === 1,
      age: num(clin.AGE),
      sex: clin.SEX === "Male" ? "M" : clin.SEX === "Female" ? "F" : null,
      stage,
      stageBand: stageBand(stage),
      tmb: round(num(clin.TMB_NONSYNONYMOUS), 2),
      osMonths: round(num(clin.OS_MONTHS), 1),
      osEvent: num(clin.OS_STATUS),
      pdl1Expr: round(num(c.CD274) ?? 0),
      pdcd1Expr: round(num(c.PDCD1) ?? 0),
      // Curves as % of each patient's own untreated baseline (baseline = 100).
      brafiCurve: normalisedCurve(brafiCurve),
      antipd1Curve: normalisedCurve(antipd1Curve),
      brafiBaseline: round(brafiCurve[0], 6),
      antipd1Baseline: round(antipd1Curve[0], 6),
      brafiInformative: isInformative(brafiCurve),
      antipd1Informative: isInformative(antipd1Curve),
      brafiReduction: round(reduction(brafiCurve)),
      antipd1Reduction: round(reduction(antipd1Curve)),
      brafiOptimalDose: optimalDose(brafiCurve),
      antipd1OptimalDose: optimalDose(antipd1Curve),
    });
  }
  if (missingCheckpoint) {
    console.warn(`  note: ${missingCheckpoint} samples had no checkpoint simulation row — dropped`);
  }

  // Pass 2 — cohort-relative percentiles, then the derived blocks.
  const pdl1Ranks = percentileRanks(draft.map((p) => p.pdl1Expr));
  const pdcd1Ranks = percentileRanks(draft.map((p) => p.pdcd1Expr));
  const tmbValues = draft.map((p) => p.tmb).filter((v) => v !== null);
  const tmbRanksBySorted = percentileRanks(tmbValues);
  const tmbRank = new Map(tmbValues.map((v, i) => [v, tmbRanksBySorted[i]]));

  // Reduction percentiles are ranked among INFORMATIVE twins only, so a patient
  // the ODE couldn't model doesn't drag the scale. These put the mechanistic
  // read-out on the same 0-100 footing as the statistical one, which is what
  // makes the Q1-vs-Q3 agreement comparison meaningful rather than arbitrary.
  const rankWithin = (predicate, value) => {
    const pool = draft.filter(predicate).map(value);
    const ranks = percentileRanks(pool);
    const lookup = new Map(pool.map((v, i) => [v, ranks[i]]));
    return (p) => (predicate(p) ? lookup.get(value(p)) ?? null : null);
  };
  const brafiRedPct = rankWithin((p) => p.brafiInformative, (p) => p.brafiReduction);
  const antipd1RedPct = rankWithin((p) => p.antipd1Informative, (p) => p.antipd1Reduction);

  const patients = draft.map((p, i) => {
    const withPct = {
      ...p,
      pdl1Pct: pdl1Ranks[i],
      pdcd1Pct: pdcd1Ranks[i],
      tmbPct: p.tmb === null ? null : tmbRank.get(p.tmb),
      brafiReductionPct: brafiRedPct(p),
      antipd1ReductionPct: antipd1RedPct(p),
    };
    return {
      ...withPct,
      q1: q1ByPatient.get(p.id) ?? null,
      q4: deriveQ4(withPct),
    };
  });

  const out = {
    meta: {
      generated: new Date().toISOString(),
      source: "TCGA-SKCM PanCancer Atlas 2018 · Q3 ODE digital twin · Q1 expression predictor",
      nQ3: patients.length,
      nQ1Tcga: nTcga,
      doseAxis: DOSE_AXIS,
      q1Validation,
      q1Note:
        nTcga > 0
          ? "Per-patient Q1 probabilities available for the TCGA twin cohort."
          : "Q1 models scored the labelled ICI trial cohorts only. The TCGA twin cohort is not " +
            "yet scorable: its cleaned expression matrix lacks the IMPRES co-stimulatory gene " +
            "partners and is on a different normalisation scale from the training cohorts. " +
            "Q1 therefore appears as cohort-level validation, not a per-patient probability.",
    },
    patients,
  };

  writeFileSync(PATHS.out, JSON.stringify(out));
  report(out);
}

function report(out) {
  const { patients, meta } = out;
  const pct = (n) => `${((n / patients.length) * 100).toFixed(1)}%`;
  const mean = (f) => (patients.reduce((a, p) => a + f(p), 0) / patients.length).toFixed(3);

  console.log(`\nWrote ${PATHS.out}`);
  console.log(`  patients          ${patients.length}`);
  console.log(`  BRAF-mutant       ${patients.filter((p) => p.braf !== "WT").length} (${pct(patients.filter((p) => p.braf !== "WT").length)})`);
  console.log(`  NRAS-mutant       ${patients.filter((p) => p.nras === "Mutant").length}`);
  console.log(`  mean brafiRed.    ${mean((p) => p.brafiReduction)}`);
  console.log(`  mean antipd1Red.  ${mean((p) => p.antipd1Reduction)}`);
  console.log(`  informative twin  brafi=${patients.filter((p) => p.brafiInformative).length}  antipd1=${patients.filter((p) => p.antipd1Informative).length}  (of ${patients.length})`);
  console.log(`  q1 per-patient    ${meta.nQ1Tcga}`);
  if (meta.q1Validation) {
    const v = meta.q1Validation;
    console.log(`  q1 validation     n=${v.n} responders=${v.responders} AUC=${v.auc}`);
    for (const c of v.byCohort) console.log(`     ${c.cohort.padEnd(12)} n=${String(c.n).padStart(3)} AUC=${c.auc}`);
  }

  console.log("\n  Archetype sanity (mean reductions, informative twins only):");
  const groups = [
    ["BRAF-mut / PD-L1-low ", (p) => p.braf !== "WT" && p.pdl1Pct < 50],
    ["BRAF-WT  / PD-L1-high", (p) => p.braf === "WT" && p.pdl1Pct >= 50],
    ["BRAF-mut / PD-L1-high", (p) => p.braf !== "WT" && p.pdl1Pct >= 50],
  ];
  for (const [label, fn] of groups) {
    const g = patients.filter((p) => fn(p) && p.brafiInformative && p.antipd1Informative);
    if (!g.length) continue;
    const m = (f) => (g.reduce((a, p) => a + f(p), 0) / g.length).toFixed(3);
    console.log(`     ${label}  n=${String(g.length).padStart(3)}  brafi=${m((p) => p.brafiReduction)}  antipd1=${m((p) => p.antipd1Reduction)}`);
  }
}

main();
