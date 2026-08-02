import { BrainCircuit, FlaskConical, Hourglass } from "lucide-react";
import type { CohortPatient, Q1Validation } from "../data/cohort";
import { Panel, Pill } from "./ui";

// Q1 · Gene-expression response predictor (ML).
//
// Two states, both designed:
//   patient.q1 present  → P(response) gauge, per-model bars, signature scores
//   patient.q1 == null  → "awaiting inference" state explaining exactly what is
//                          missing and what unblocks it. NOT an error, and never
//                          a fabricated number (locked decision D4).

const MODEL_LABELS: Record<string, string> = {
  lr: "Logistic regression",
  rf: "Random forest",
  xgb: "XGBoost",
  svm: "Support vector machine",
  enet: "Elastic net",
};

const SIGNATURE_LABELS: Record<string, string> = {
  IFN_gamma: "IFN-γ (6-gene)",
  TIS: "Tumour inflammation (TIS)",
  CYT: "Cytolytic activity",
  CD8_Tcell: "CD8 T-cell",
  IMPRES: "IMPRES",
  PD_L1: "PD-L1 expression",
  Macrophage_STV_Score: "Macrophage STV Barrier",
  M1_M2_Ratio: "M1/M2 Polarization Ratio",
  mut_BRAF: "BRAF V600 Mutation",
  mut_NRAS: "NRAS Driver Mutation",
  mut_NF1: "NF1 Loss-of-Function",
  TMB_NONSYNONYMOUS: "Tumour Mutation Burden (TMB)",
};

function ordinal(n: number): string {
  if (n % 100 >= 11 && n % 100 <= 13) return "th";
  return ["th", "st", "nd", "rd"][n % 10] ?? "th";
}

/** Radial P(response) gauge with cohort-relative percentile rank pill. */
function Gauge({ value, rank }: { value: number; rank?: number | null }) {
  const pct = Math.round(value * 100);
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const dash = (value * circumference) / 2;

  return (
    <div className="flex flex-col items-center">
      <div className="relative flex h-[120px] w-[150px] shrink-0 items-end justify-center">
        <svg viewBox="0 0 140 78" className="absolute inset-0 h-full w-full">
          <path
            d="M 18 70 A 52 52 0 0 1 122 70"
            fill="none"
            stroke="#e6ebf1"
            strokeWidth={12}
            strokeLinecap="round"
          />
          <path
            d="M 18 70 A 52 52 0 0 1 122 70"
            fill="none"
            stroke={pct >= 50 ? "#CC79A7" : "#5b6b7c"}
            strokeWidth={12}
            strokeLinecap="round"
            strokeDasharray={`${dash} ${circumference}`}
          />
        </svg>
        <div className="relative pb-1 text-center">
          <div className="tabular text-[30px] font-extrabold leading-none text-clinical-ink">
            {pct}
            <span className="text-[15px]">%</span>
          </div>
          <div className="text-[10px] font-bold uppercase tracking-wide text-clinical-muted">
            P(response)
          </div>
        </div>
      </div>
      {rank !== null && rank !== undefined && (
        <div className="mt-1 text-center">
          <Pill tone="okabe-purple">
            {rank}{ordinal(rank)} percentile
          </Pill>
        </div>
      )}
    </div>
  );
}

function Bar({ label, value }: { label: string; value: number | null }) {
  if (value === null) return null;
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2.5">
      <span className="w-[150px] shrink-0 text-[11.5px] font-semibold text-clinical-muted">
        {label}
      </span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-clinical-bg">
        <div
          className="h-full rounded-full bg-clinical-bluedark transition-[width] duration-500 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="tabular w-9 text-right text-[12px] font-bold text-clinical-ink">{pct}%</span>
    </div>
  );
}

function AwaitingState({ note, validation }: { note: string; validation: Q1Validation | null }) {
  return (
    <div className="rounded-xl border border-dashed border-clinical-border bg-clinical-bg/60 p-5">
      <div className="flex items-start gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white text-clinical-muted shadow-card">
          <Hourglass size={18} />
        </div>
        <div className="min-w-0">
          <h3 className="text-[14px] font-extrabold text-clinical-ink">
            Per-patient inference pending
          </h3>
          <p className="mt-1.5 max-w-2xl text-[12.5px] leading-relaxed text-clinical-muted">{note}</p>

          {validation && (
            <p className="mt-3 text-[12.5px] leading-relaxed text-clinical-ink">
              The models themselves <span className="font-bold">are</span> validated – see the
              cohort-level accuracy panel below, computed on{" "}
              <span className="tabular font-bold">{validation.n}</span> held-out trial patients with
              real response outcomes.
            </p>
          )}

          <div className="mt-3.5 rounded-lg border border-clinical-border bg-white px-3.5 py-2.5">
            <div className="text-[10px] font-bold uppercase tracking-wide text-clinical-muted">
              To populate this lane
            </div>
            <code className="mt-1 block text-[11.5px] font-semibold text-clinical-ink">
              python q1_infer.py
            </code>
            <div className="mt-1 text-[11px] leading-snug text-clinical-muted">
              once the TCGA expression matrix carries the full signature gene panel on the training
              cohorts' normalisation scale. The lane fills in with no further code changes.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Q1Lane({
  patient,
  note,
  validation,
  cohortRank,
}: {
  patient: CohortPatient;
  note: string;
  validation: Q1Validation | null;
  /** Percentile of this patient's P(response) across the cohort, 0-100. */
  cohortRank?: number | null;
}) {
  const q1 = patient.q1;
  const rank = q1?.pResponsePct ?? cohortRank;
  const hasPerModel = q1 ? Object.values(q1.perModel).some((v) => v !== null) : false;
  const hasFeatures = q1 ? Object.values(q1.features).some((v) => v !== null) : false;
  const rankLabel =
    rank === null || rank === undefined
      ? "against the cohort"
      : `in the ${rank}${ordinal(rank)} percentile`;

  return (
    <Panel
      title="Q1 · Gene-Expression Response Predictor"
      subtitle="Five-model ensemble over 12 multimodal features (immune signatures, macrophage barrier, driver mutations & TMB) → P(response to checkpoint blockade)"
      icon={<BrainCircuit size={16} />}
      right={
        q1 ? (
          <Pill tone="okabe-purple">Scored</Pill>
        ) : (
          <Pill tone="neutral">
            <Hourglass size={11} /> Awaiting inference
          </Pill>
        )
      }
    >
      {!q1 ? (
        <AwaitingState note={note} validation={validation} />
      ) : (
        <div className="space-y-5">
          <div className="flex flex-col items-center gap-5 sm:flex-row sm:items-center">
            <Gauge value={q1.pResponse} rank={rank} />
            <div className="min-w-0 flex-1 space-y-2">
              {hasPerModel ? (
                <>
                  <div className="text-[10.5px] font-bold uppercase tracking-wide text-clinical-muted">
                    Per-model probability
                  </div>
                  {Object.entries(q1.perModel).map(([key, value]) => (
                    <Bar key={key} label={MODEL_LABELS[key] ?? key} value={value} />
                  ))}
                </>
              ) : (
                <div className="rounded-xl border border-clinical-border bg-clinical-bg px-4 py-3">
                  <div className="text-[12.5px] font-bold text-clinical-ink">
                    {q1.pResponse >= 0.5
                      ? "Predicted to respond to checkpoint blockade"
                      : "Predicted not to respond to checkpoint blockade"}
                  </div>
                  <p className="mt-1 text-[11.5px] leading-snug text-clinical-muted">
                    Ranked {rankLabel} against the rest of the cohort. The Q1 workstream supplied a
                    single ensemble score per patient, so the per-model and signature breakdowns are
                    not available for this patient.
                  </p>
                </div>
              )}
            </div>
          </div>

          {hasFeatures && (
            <div>
              <div className="mb-2 flex items-center gap-1.5 text-[10.5px] font-bold uppercase tracking-wide text-clinical-muted">
                <FlaskConical size={12} /> Input signature scores
              </div>
              <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3">
                {Object.entries(q1.features).map(([key, value]) => (
                  <div
                    key={key}
                    className="rounded-lg border border-clinical-border bg-clinical-bg px-3 py-2"
                  >
                    <div className="text-[10px] font-semibold uppercase tracking-wide text-clinical-muted">
                      {SIGNATURE_LABELS[key] ?? key}
                    </div>
                    <div className="tabular text-[15px] font-extrabold text-clinical-ink">
                      {value === null ? "–" : value.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <p className="text-[11.5px] leading-snug text-clinical-muted">
            Cohort of origin: {q1.cohort}. Produced by the trained Q1 models from this
            patient&apos;s gene-expression profile – a real prediction, not a derived score.
          </p>
        </div>
      )}
    </Panel>
  );
}
