#!/usr/bin/env node
// ---------------------------------------------------------------------------
// build_cohort.mjs – generates dashboard/public/cohort.json, the single file
// the OncoTwin app consumes.
//
// Sources (all real project output, nothing synthesised):
//   q3-ode-model/outputs/results/tumour_burden_simulations.csv      BRAFi sweep
//   q3-ode-model/outputs/results/checkpoint_tumour_simulations.csv  anti-PD-1 sweep
//   data/processed/skcm_tcga_pan_can_atlas_2018/clin_cleaned.csv    TCGA clinical
//   data/raw/skcm_tcga_pan_can_atlas_2018/data_timeline_treatment.txt  treatment
//   dashboard/public/q1_predictions.csv                             Q1 ML output
//   dashboard/public/q1_tcga_scores.csv                             Q1 TCGA scores
//   data/processed/q5/treatability_scores.csv                       Q5 per-patient
//   data/processed/q5/phenotype_characterisation.csv                Q5 phenotype stats
//   data/processed/q5/ode_trajectory_summary.json                   Q5 ODE endpoints
//   data/processed/q5/subgroup_models_evaluation.csv                Q5 subgroup AUC
//
// Idempotent: re-run any time. Run via:
//   node scripts/build_cohort.mjs
// or automatically as part of:
//   npm run build  (via prebuild hook)
// ---------------------------------------------------------------------------

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const DASHBOARD = resolve(HERE, "..");
const REPO = resolve(DASHBOARD, "..");

// ---------------------------------------------------------------------------
// Cohort scope switch – set to false to include all Q5 cohorts (Option B).
// When true only TCGA-SKCM rows from treatability_scores.csv are joined onto
// the Q3 twin cohort.  All downstream types already support null ODE fields
// so switching to false is a single-line change + UI cohort-selector addition.
// ---------------------------------------------------------------------------
const TCGA_ONLY = true;

const PATHS = {
  tumour:            resolve(REPO, "q3-ode-model/outputs/results/tumour_burden_simulations.csv"),
  checkpoint:        resolve(REPO, "q3-ode-model/outputs/results/checkpoint_tumour_simulations.csv"),
  clinical:          resolve(REPO, "data/processed/skcm_tcga_pan_can_atlas_2018/clin_cleaned.csv"),
  treatmentTimeline: resolve(REPO, "data/raw/skcm_tcga_pan_can_atlas_2018/data_timeline_treatment.txt"),
  q1:                resolve(DASHBOARD, "public/q1_predictions.csv"),
  q1Tcga:            resolve(DASHBOARD, "public/q1_tcga_scores.csv"),
  // Q5 outputs
  q5Scores:          resolve(REPO, "data/processed/q5/treatability_scores.csv"),
  q5Pheno:           resolve(REPO, "data/processed/q5/phenotype_characterisation.csv"),
  q5Ode:             resolve(REPO, "data/processed/q5/ode_trajectory_summary.json"),
  q5Auc:             resolve(REPO, "data/processed/q5/subgroup_models_evaluation.csv"),
  out:               resolve(DASHBOARD, "public/cohort.json"),
};

const DOSE_AXIS    = [0.01, 0.12, 0.23, 0.34, 0.45, 0.56, 0.67, 0.78, 0.89, 1.0];
const BRAFI_COLS   = DOSE_AXIS.map((d) => `BRAFi_${d.toFixed(3)}`);
const ANTIPD1_COLS = DOSE_AXIS.map((d) => `antiPD1_${d.toFixed(3)}`);
const Q1_MODELS    = ["lr", "rf", "xgb", "svm", "enet"];
const Q1_FEATURES  = ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "IMPRES", "PD_L1"];

// ---------------------------------------------------------------------------
// CSV / TSV parsing
// ---------------------------------------------------------------------------

/** Quote-aware CSV parse → array of row objects. */
function parseCsv(text) {
  const rows = [];
  let row = [], field = "", quoted = false;
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

function readCsv(path) { return parseCsv(readFileSync(path, "utf8")); }

function readTsv(path) {
  const lines = readFileSync(path, "utf8").split("\n").filter((l) => l.length);
  const header = lines[0].split("\t");
  return lines.slice(1).map((line) => {
    const cells = line.split("\t");
    return Object.fromEntries(header.map((h, i) => [h, (cells[i] ?? "").trim()]));
  });
}

// ---------------------------------------------------------------------------
// Number helpers
// ---------------------------------------------------------------------------

function num(v) {
  if (v === undefined || v === null) return null;
  const s = String(v).trim();
  if (!s || s === "NA" || s === "NaN" || s === "nan" || s === "None") return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

const round = (n, dp = 4) => (n === null ? null : Math.round(n * 10 ** dp) / 10 ** dp);

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

function percentileRanks(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return values.map((v) => {
    let below = 0, ties = 0;
    for (const s of sorted) {
      if (s < v) below++;
      else if (s === v) ties++;
    }
    return Math.round(((below + ties / 2) / sorted.length) * 100);
  });
}

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
    const mid = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[k] = mid;
    i = j + 1;
  }
  const rankSumPos = order.reduce((acc, p, i) => acc + (p.y === 1 ? ranks[i] : 0), 0);
  return (rankSumPos - (pos * (pos + 1)) / 2) / (pos * neg);
}

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
    threshold, n, tp, fp, tn, fn,
    accuracy: safe(tp + tn, n),
    sensitivity: safe(tp, tp + fn),
    specificity: safe(tn, tn + fp),
    ppv: safe(tp, tp + fp),
    npv: safe(tn, tn + fn),
  };
}

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

function buildDistribution(labelled, bins = 10) {
  const out = [];
  for (let i = 0; i < bins; i++) {
    const lo = i / bins, hi = (i + 1) / bins;
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

// ---------------------------------------------------------------------------
// Clinical normalisation
// ---------------------------------------------------------------------------

function normaliseStage(raw) {
  if (!raw) return "Unknown";
  const s = raw.replace(/^STAGE\s*/i, "").trim();
  if (!s || /^\[?not/i.test(s)) return "Unknown";
  if (/NOS/i.test(s)) return s.replace(/\s*\(NOS\)/i, "").trim();
  return s;
}

function stageBand(stage) {
  if (stage === "Unknown") return "Unknown";
  if (stage.startsWith("IV"))  return "IV";
  if (stage.startsWith("III")) return "III";
  if (stage.startsWith("II"))  return "II";
  if (stage.startsWith("I"))   return "I";
  if (stage.startsWith("0"))   return "0";
  return "Unknown";
}

// ---------------------------------------------------------------------------
// Q3 curve helpers
// ---------------------------------------------------------------------------

const BURDEN_FLOOR = 1e-3;
const isInformative = (curve) => Number.isFinite(curve[0]) && curve[0] >= BURDEN_FLOOR;

function reduction(curve) {
  if (!isInformative(curve)) return 0;
  const best = Math.min(...curve);
  return Math.max(0, Math.min(1, 1 - best / curve[0]));
}

function optimalDose(curve) {
  if (!isInformative(curve)) return null;
  const best = Math.min(...curve);
  const span = curve[0] - best;
  for (let i = 0; i < curve.length; i++) {
    if (span <= 0 || curve[i] <= best + 0.02 * span) return DOSE_AXIS[i];
  }
  return DOSE_AXIS[DOSE_AXIS.length - 1];
}

function normalisedCurve(curve) {
  if (!isInformative(curve)) return null;
  return curve.map((v) => round((v / curve[0]) * 100, 2));
}

// ---------------------------------------------------------------------------
// Treatment history
// ---------------------------------------------------------------------------

const CHECKPOINT_AGENTS = new Set(["Ipilimumab", "Pembrolizumab", "Nivolumab"]);
const TARGETED_AGENTS   = new Set(["Vemurafenib", "Dabrafenib", "Trametinib"]);
const CHEMO_AGENTS      = new Set(["Dacarbazine", "Temozolomide"]);

function emptyTreatment() {
  return { recorded: false, lines: [], checkpointInhibitor: false, targetedTherapy: false, chemotherapy: false, radiation: false };
}

function buildTreatmentByPatient(rows) {
  const byPatient = new Map();
  for (const r of rows) {
    const pid = r.PATIENT_ID, type = r.TREATMENT_TYPE?.trim();
    if (!pid || !type) continue;
    if (!byPatient.has(pid)) byPatient.set(pid, { ...emptyTreatment(), recorded: true });
    const entry = byPatient.get(pid);
    let line = entry.lines.find((l) => l.type === type);
    if (!line) { line = { type, agents: [] }; entry.lines.push(line); }
    const agent = r.AGENT?.trim();
    if (agent && !line.agents.includes(agent)) line.agents.push(agent);
    if (agent && CHECKPOINT_AGENTS.has(agent)) entry.checkpointInhibitor = true;
    if (agent && TARGETED_AGENTS.has(agent))   entry.targetedTherapy = true;
    if (agent && CHEMO_AGENTS.has(agent))       entry.chemotherapy = true;
    if (type === "Radiation Therapy")           entry.radiation = true;
  }
  return byPatient;
}

// ---------------------------------------------------------------------------
// Q4 heuristics (explicitly labelled as heuristic – not a fitted model)
// ---------------------------------------------------------------------------

function deriveQ4(p) {
  const flags = [];
  const odeUsable = p.brafiInformative || p.antipd1Informative;
  const bestReduction = Math.max(
    p.brafiInformative   ? p.brafiReduction   : 0,
    p.antipd1Informative ? p.antipd1Reduction : 0
  );

  if (p.nras === "Mutant")
    flags.push({ label: "NRAS-driven MAPK reactivation", detail: "NRAS mutation sustains MEK/ERK signalling downstream of BRAF blockade." });
  if (p.braf !== "WT" && p.mapkDriven)
    flags.push({ label: "BRAFi escape via MAPK rebound", detail: "MAPK-driven tumour: expect ERK reactivation and loss of BRAFi control." });
  if (p.pdl1Pct < 25)
    flags.push({ label: "Immune-cold (low PD-L1)", detail: "Bottom-quartile CD274 expression; primary checkpoint resistance is likely." });
  if (p.braf === "WT" && p.nras === "WT")
    flags.push({ label: "Triple-WT lineage dependency", detail: "No MAPK driver – SOX10/MITF lineage survival is the tractable axis." });
  if (!odeUsable)
    flags.push({ label: "Twin below model resolution", detail: "The ODE settles at a numerically-zero tumour compartment for this patient, so the simulated arms carry no information – resistance risk is not assessable from Q3." });
  else if (bestReduction < 0.1)
    flags.push({ label: "Refractory in silico", detail: "Neither simulated arm clears meaningful burden across the full dose sweep." });

  const resistanceRisk = !odeUsable ? "unknown"
    : bestReduction < 0.15 ? "high"
    : bestReduction < 0.4  ? "moderate"
    : "low";

  const reserve = [];
  if (p.braf !== "WT")   reserve.push("MEK inhibitor re-challenge after drug holiday");
  if (p.nras === "Mutant") reserve.push("MEK + CDK4/6 inhibition (NRAS-mutant salvage)");
  reserve.push("SOX10 / MITF lineage-switch targeting");
  if (p.pdl1Pct >= 50)   reserve.push("LAG-3 blockade (relatlimab + nivolumab)");
  if (p.tmb !== null && p.tmb >= 20) reserve.push("High TMB – retry checkpoint blockade");

  return { resistanceRisk, flags, reserve };
}

// ---------------------------------------------------------------------------
// Q1 helpers
// ---------------------------------------------------------------------------

function buildTcgaScores(rows) {
  const grouped = new Map();
  for (const r of rows) {
    if (!/tcga/i.test(r.COHORT ?? "")) continue;
    const score = num(r.PREDICTED_RESPONSE_SCORE);
    if (score === null) continue;
    if (!grouped.has(r.PATIENT_ID)) grouped.set(r.PATIENT_ID, []);
    grouped.get(r.PATIENT_ID).push(score);
  }
  const out = new Map();
  for (const [id, scores] of grouped) {
    out.set(id, {
      pResponse: round(scores.reduce((a, b) => a + b, 0) / scores.length),
      perModel: { lr: null, rf: null, xgb: null, svm: null, enet: null },
      features: Object.fromEntries(Q1_FEATURES.map((f) => [f, null])),
      cohort: "TCGA-SKCM",
      detail: "ensemble-only",
    });
  }
  return out;
}

function buildQ1(rows) {
  if (!rows.length) return { byPatient: new Map(), validation: null, nTcga: 0 };

  const parsed = rows.map((r) => ({
    sampleId: r.SAMPLE_ID, patientId: r.PATIENT_ID, cohort: r.cohort,
    features: Object.fromEntries(Q1_FEATURES.map((f) => [f, num(r[f])])),
    perModel: Object.fromEntries(Q1_MODELS.map((m) => [m, num(r[`prob_${m}`])])),
    pResponse: num(r.prob_ensemble),
    predLabel: num(r.pred_label),
    actual: num(r.actual_response),
    osMonths: num(r.os_months),
  }));

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
          cohort: c, n: g.length,
          responders: g.filter((r) => r.actual === 1).length,
          auc: round(auc(g.map((r) => r.actual), g.map((r) => r.pResponse)), 4),
          meanProb: round(g.reduce((a, r) => a + r.pResponse, 0) / g.length, 4),
        };
      }).sort((a, b) => b.n - a.n),
      distribution: buildDistribution(labelled),
      roc: buildRoc(y, p),
    };
  }
  return { byPatient, validation, nTcga: byPatient.size };
}

// ---------------------------------------------------------------------------
// Q5 helpers
// ---------------------------------------------------------------------------

/**
 * Parse the short phenotype label (the text before the first parenthesis),
 * e.g. "Immune Hot (High TIS & CYT, ...)" → "Immune Hot".
 * Normalises "Immunosuppressive M2-High" → "M2-High" for display brevity.
 */
function parseShortLabel(full) {
  if (!full) return "Unknown";
  const before = full.split("(")[0].trim();
  // Normalise the M2 label to the short form used in ode_trajectory_summary.json
  if (before.toLowerCase().includes("m2-high") || before.toLowerCase().includes("immunosuppressive")) {
    return "M2-High";
  }
  return before;
}

/**
 * Map Q5 Treatment_Arm string ("Arm A: Immunotherapy" etc.) to the
 * compact key used in the type system.
 */
function parseArm(armStr) {
  if (!armStr) return "A";
  if (armStr.includes("Arm B")) return "B";
  if (armStr.includes("Arm C")) return "C";
  return "A";
}

/** Build the per-patient Q5 block from a treatability_scores.csv row. */
function buildQ5Block(r) {
  return {
    label:       r.Phenotype_Label ?? "",
    shortLabel:  parseShortLabel(r.Phenotype_Label),
    clusterId:   num(r.Cluster_ID) ?? -1,
    probabilities: {
      immuneHot:    round(num(r.P_Immune_Hot)               ?? 0, 4),
      immuneCold:   round(num(r.P_Immune_Cold)              ?? 0, 4),
      m2High:       round(num(r.P_Immunosuppressive_M2_High) ?? 0, 4),
      mutantDriven: round(num(r.P_Mutant_Driven)            ?? 0, 4),
    },
    treatabilityIndex:     round(num(r.Treatability_Index)             ?? null, 2),
    dabrafenibSensitivity: round(num(r.Dabrafenib_Sensitivity_Index)   ?? null, 2),
    treatmentArm:          parseArm(r.Treatment_Arm),
    recommendedTherapy:    r.Recommended_Therapy ?? "",
    confidenceBand:        r.Confidence_Band ?? "Low",
    q4NominatedTarget:     r.Q4_Nominated_Target ?? "",
  };
}

/** Build map of PATIENT_ID → Q5 block, filtered to TCGA-SKCM when TCGA_ONLY. */
function buildQ5ByPatient(rows) {
  const out = new Map();
  for (const r of rows) {
    if (TCGA_ONLY && !/tcga/i.test(r.COHORT ?? "")) continue;
    if (!r.PATIENT_ID) continue;
    out.set(r.PATIENT_ID, buildQ5Block(r));
  }
  return out;
}

// Canonical cluster-ID → short label mapping (mirrors src/styles.py PHENOTYPE_PALETTE keys)
const CLUSTER_LABELS = {
  0: "Immune Hot",
  1: "Immune Cold",
  2: "M2-High",
  3: "Mutant-Driven",
};

/** Parse phenotype_characterisation.csv into the q5PhenotypeStats array. */
function buildPhenotypeStats(rows) {
  return rows.map((r) => {
    const clusterId = num(r.Cluster_ID) ?? -1;
    // phenotype_characterisation.csv has no Phenotype_Label column – derive it
    const label = r.Phenotype_Label ?? r.Phenotype ?? CLUSTER_LABELS[clusterId] ?? `Cluster ${clusterId}`;
    return {
      clusterId,
      label,
      n:            num(r.Patient_Count) ?? 0,
      cohortPct:    round(num(r.Cohort_Percentage) ?? 0, 2),
      responseRate: round(num(r.Response_Rate)     ?? null, 4),
      tis:          round(num(r.TIS)               ?? null, 4),
      cyt:          round(num(r.CYT)               ?? null, 4),
    };
  });
}


/** Parse subgroup_models_evaluation.csv into q5SubgroupAuc array. */
function buildSubgroupAuc(rows) {
  // One entry per phenotype (the subgroup-specific model row, not baseline)
  return rows
    .filter((r) => r.Model_Scope === "Subgroup Specific" || r.Model_Scope === "Subgroup Ensemble")
    .map((r) => ({
      phenotype: r.Phenotype ?? "",
      auc:       round(num(r.ROC_AUC), 4),
      n:         num(r.N) ?? 0,
    }));
}

/** Parse subgroup_models_evaluation.csv into full q5SubgroupEvaluation array. */
function buildSubgroupEvaluation(rows) {
  return rows.map((r) => ({
    clusterId:    num(r.Cluster_ID) ?? -1,
    phenotype:    r.Phenotype ?? "",
    modelScope:   r.Model_Scope ?? "",
    n:            num(r.N) ?? 0,
    responders:   num(r.Responders) ?? 0,
    responseRate: round(num(r.Response_Rate), 2),
    rocAuc:       round(num(r.ROC_AUC), 4),
    prAuc:        round(num(r.PR_AUC), 4),
    precision:    round(num(r.Precision), 4),
    recall:       round(num(r.Recall), 4),
    f1Score:      round(num(r.F1_Score), 4),
    accuracy:     round(num(r.Accuracy), 4),
    brierScore:   round(num(r.Brier_Score), 4),
  }));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main() {
  // Required inputs
  for (const key of ["tumour", "checkpoint", "clinical"]) {
    if (!existsSync(PATHS[key])) {
      console.error(`ERROR: required input missing – ${PATHS[key]}`);
      process.exit(1);
    }
  }

  const tumour     = readCsv(PATHS.tumour);
  const checkpoint = readCsv(PATHS.checkpoint);
  const clinical   = readCsv(PATHS.clinical);

  const checkpointBySample = new Map(checkpoint.map((r) => [r.SAMPLE_ID, r]));
  const clinByPatient      = new Map(clinical.map((r) => [r.PATIENT_ID, r]));

  const treatmentByPatient = existsSync(PATHS.treatmentTimeline)
    ? buildTreatmentByPatient(readTsv(PATHS.treatmentTimeline))
    : new Map();
  if (!treatmentByPatient.size)
    console.warn("  note: data_timeline_treatment.txt absent – treatment history unrecorded for all");

  const q1Csv = existsSync(PATHS.q1) ? readCsv(PATHS.q1) : [];
  if (!q1Csv.length) console.warn("  note: public/q1_predictions.csv absent – q1 will be null");
  const { byPatient: q1FromInfer, validation: q1Validation } = buildQ1(q1Csv);

  const tcgaScores  = existsSync(PATHS.q1Tcga) ? buildTcgaScores(readCsv(PATHS.q1Tcga)) : new Map();
  if (!tcgaScores.size) console.warn("  note: public/q1_tcga_scores.csv absent – per-patient Q1 unavailable");
  const q1ByPatient = new Map([...tcgaScores, ...q1FromInfer]);
  const nTcga       = q1ByPatient.size;

  // Q5 inputs (non-fatal if missing – dashboard degrades gracefully)
  const q5ByPatient = existsSync(PATHS.q5Scores)
    ? buildQ5ByPatient(readCsv(PATHS.q5Scores))
    : new Map();
  if (!q5ByPatient.size) console.warn("  note: data/processed/q5/treatability_scores.csv absent – q5 will be null for all patients");

  const q5PhenotypeStats = existsSync(PATHS.q5Pheno)
    ? buildPhenotypeStats(readCsv(PATHS.q5Pheno))
    : [];

  const q5OdeTrajectory = existsSync(PATHS.q5Ode)
    ? JSON.parse(readFileSync(PATHS.q5Ode, "utf8"))
    : {};

  const subgroupModelsCsv = existsSync(PATHS.q5Auc) ? readCsv(PATHS.q5Auc) : [];

  const q5SubgroupAuc = subgroupModelsCsv.length
    ? buildSubgroupAuc(subgroupModelsCsv)
    : [];

  // Pass 1 – assemble per-patient draft
  const draft = [];
  let missingCheckpoint = 0;
  for (const t of tumour) {
    const c    = checkpointBySample.get(t.SAMPLE_ID);
    if (!c) { missingCheckpoint++; continue; }
    const clin = clinByPatient.get(t.PATIENT_ID) ?? {};

    const brafiCurve   = BRAFI_COLS.map((k)   => num(t[k]) ?? 0);
    const antipd1Curve = ANTIPD1_COLS.map((k)  => num(c[k]) ?? 0);
    const stage        = normaliseStage(clin.AJCC_PATHOLOGIC_TUMOR_STAGE);

    draft.push({
      id:       t.PATIENT_ID,
      sampleId: t.SAMPLE_ID,
      // cohort field carried from day one for Option B extensibility
      cohort:   "TCGA-SKCM",
      braf:     num(t.BRAF_MUT) === 1 ? "V600E" : "WT",
      nras:     num(t.NRAS_MUT) === 1 ? "Mutant" : "WT",
      mapkDriven: num(t.MAPK_DRIVEN) === 1,
      age:   num(clin.AGE),
      sex:   clin.SEX === "Male" ? "M" : clin.SEX === "Female" ? "F" : null,
      stage, stageBand: stageBand(stage),
      tmb:      round(num(clin.TMB_NONSYNONYMOUS), 2),
      osMonths: round(num(clin.OS_MONTHS), 1),
      osEvent:  num(clin.OS_STATUS),
      pdl1Expr: round(num(c.CD274)  ?? 0),
      pdcd1Expr: round(num(c.PDCD1) ?? 0),
      brafiCurve:       normalisedCurve(brafiCurve),
      antipd1Curve:     normalisedCurve(antipd1Curve),
      brafiBaseline:    round(brafiCurve[0],   6),
      antipd1Baseline:  round(antipd1Curve[0], 6),
      brafiInformative:   isInformative(brafiCurve),
      antipd1Informative: isInformative(antipd1Curve),
      brafiReduction:     round(reduction(brafiCurve)),
      antipd1Reduction:   round(reduction(antipd1Curve)),
      brafiOptimalDose:   optimalDose(brafiCurve),
      antipd1OptimalDose: optimalDose(antipd1Curve),
      treatment: treatmentByPatient.get(t.PATIENT_ID) ?? emptyTreatment(),
    });
  }
  if (missingCheckpoint)
    console.warn(`  note: ${missingCheckpoint} samples had no checkpoint simulation row – dropped`);

  // Pass 2 – cohort-relative percentiles + derived blocks
  const pdl1Ranks   = percentileRanks(draft.map((p) => p.pdl1Expr));
  const pdcd1Ranks  = percentileRanks(draft.map((p) => p.pdcd1Expr));
  const tmbValues   = draft.map((p) => p.tmb).filter((v) => v !== null);
  const tmbRankMap  = new Map(tmbValues.map((v, i) => [v, percentileRanks(tmbValues)[i]]));

  const rankWithin = (pred, val) => {
    const pool  = draft.filter(pred).map(val);
    const ranks = percentileRanks(pool);
    const lkp   = new Map(pool.map((v, i) => [v, ranks[i]]));
    return (p) => (pred(p) ? lkp.get(val(p)) ?? null : null);
  };
  const brafiRedPct   = rankWithin((p) => p.brafiInformative,   (p) => p.brafiReduction);
  const antipd1RedPct = rankWithin((p) => p.antipd1Informative, (p) => p.antipd1Reduction);

  // Compute Q1 pResponse cohort-relative percentile ranks across patients with Q1 data
  const q1PrespPool = draft.map((p) => q1ByPatient.get(p.id)?.pResponse ?? null);
  const validQ1Indices = q1PrespPool.map((v, i) => (v !== null ? i : null)).filter((i) => i !== null);
  const q1Values = validQ1Indices.map((i) => q1PrespPool[i]);
  const q1Percentiles = percentileRanks(q1Values);
  const q1RankLkp = new Map(validQ1Indices.map((idx, i) => [draft[idx].id, q1Percentiles[i]]));

  const patients = draft.map((p, i) => {
    const withPct = {
      ...p,
      pdl1Pct:            pdl1Ranks[i],
      pdcd1Pct:           pdcd1Ranks[i],
      tmbPct:             p.tmb === null ? null : tmbRankMap.get(p.tmb),
      brafiReductionPct:  brafiRedPct(p),
      antipd1ReductionPct: antipd1RedPct(p),
    };
    const rawQ1 = q1ByPatient.get(p.id) ?? null;
    const q1 = rawQ1 ? { ...rawQ1, pResponsePct: q1RankLkp.get(p.id) ?? null } : null;
    return {
      ...withPct,
      q1,
      q4: deriveQ4(withPct),
      q5: q5ByPatient.get(p.id) ?? null,
    };
  });

  const nQ5Scored = patients.filter((p) => p.q5 !== null).length;
  const nHighConf = patients.filter((p) => p.q5?.confidenceBand === "High").length;

  const out = {
    meta: {
      generated: new Date().toISOString(),
      source: TCGA_ONLY
        ? "TCGA-SKCM PanCancer Atlas 2018 · Q3 ODE digital twin · Q1 expression predictor · Q5 patient stratification (TCGA-SKCM only)"
        : "All cohorts: TCGA-SKCM · Liu 2019 · Hugo 2016 · Riaz 2017 · Q3/Q1/Q5 methods",
      tcgaOnly: TCGA_ONLY,
      nQ3: patients.length,
      nQ1Tcga: nTcga,
      nQ5Scored,
      nHighConf,
      doseAxis: DOSE_AXIS,
      q1Validation,
      q1Note: nTcga > 0
        ? "Per-patient Q1 response scores available for the TCGA twin cohort."
        : "Q1 models scored the labelled ICI trial cohorts only; Q1 appears as cohort-level validation.",
      q5PhenotypeStats,
      q5OdeTrajectory,
      q5SubgroupAuc,
      q5SubgroupEvaluation: buildSubgroupEvaluation(subgroupModelsCsv),
    },
    patients,
  };

  writeFileSync(PATHS.out, JSON.stringify(out));
  report(out);
}

// ---------------------------------------------------------------------------
// Console report
// ---------------------------------------------------------------------------

function report(out) {
  const { patients, meta } = out;
  const pct  = (n) => `${((n / patients.length) * 100).toFixed(1)}%`;
  const mean = (f) => (patients.reduce((a, p) => a + f(p), 0) / patients.length).toFixed(3);

  console.log(`\nWrote ${PATHS.out}`);
  console.log(`  patients          ${patients.length}`);
  console.log(`  Q5-scored         ${meta.nQ5Scored} (${pct(meta.nQ5Scored)})`);
  console.log(`  High confidence   ${meta.nHighConf}`);
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

  if (meta.q5PhenotypeStats.length) {
    console.log("\n  Q5 phenotype breakdown:");
    for (const s of meta.q5PhenotypeStats) {
      const label = (s.label.split("(")[0].trim()).padEnd(30);
      console.log(`     ${label}  N=${String(s.n).padStart(3)}  (${(s.cohortPct ?? 0).toFixed(1)}%)  resp=${(s.responseRate * 100).toFixed(1)}%`);
    }
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
