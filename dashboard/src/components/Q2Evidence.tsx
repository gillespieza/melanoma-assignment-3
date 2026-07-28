import { FlaskConical, CheckCircle2, Hourglass } from "lucide-react";
import { Panel, Pill } from "./ui";

// ---------------------------------------------------------------------------
// Q2 · Experimental validation — cohort-level evidence, never per-patient.
//
// The cell-line arm of Q2 belongs to a separate workstream and its results are
// not in this repository, so it is shown as pending rather than invented. What
// IS shown is the orthogonal experimental validation that exists here: the RPPA
// proteomic check on the Q3 twin, and the ML-vs-ODE benchmark.
// ---------------------------------------------------------------------------

interface Evidence {
  status: "established" | "pending";
  claim: string;
  detail: string;
  stat?: string;
}

const EVIDENCE: Evidence[] = [
  {
    status: "established",
    claim: "ODE-predicted pERK tracks measured protein levels",
    detail:
      "Simulated baseline pERK compared against reverse-phase protein array (RPPA) measurements in " +
      "the same TCGA patients — an independent, wet-lab readout the model never saw.",
    stat: "n = 310 · Pearson r = 0.175 · p = 0.002",
  },
  {
    status: "established",
    claim: "Model and protein data agree that NRAS-mutant tumours have the highest pERK",
    detail:
      "The ODE ranks NRAS > BRAF > wild-type for MAPK output; the measured proteomics rank them the " +
      "same way. The mechanism the model encodes is the mechanism the assay sees.",
    stat: "Mann-Whitney p = 3.2 × 10⁻⁹",
  },
  {
    status: "established",
    claim: "Three ODE outputs rival twelve raw clinical features",
    detail:
      "The mechanistic twin reaches comparable discrimination to a random forest trained on four " +
      "times as many inputs — the biology is doing real work, not curve-fitting.",
    stat: "ODE AUC 0.666 ± 0.074 vs RF 0.686 ± 0.046",
  },
  // TO WIRE IN when the cell-line data arrives: flip this to
  // status: "established" and fill `stat` with the real correlation
  // (n, coefficient, p-value). See docs/07_SESSION_LOG.md §4 — it covers both
  // the quick headline-stat route and the fuller per-cell-line CSV route.
  // Do NOT invent a statistic; the pending card is the honest fallback.
  {
    status: "pending",
    claim: "Cell-line drug-response validation of the Q1 signature",
    detail:
      "The Q2 workstream applies the Q1 expression signature to melanoma cell lines with known drug " +
      "sensitivity. Those results sit outside this repository and are not wired in here.",
  },
];

export default function Q2Evidence() {
  return (
    <Panel
      title="Q2 · Experimental Validation"
      subtitle="Does the biology hold up against measurements the models never saw?"
      icon={<FlaskConical size={16} />}
      right={<Pill tone="neutral">Cohort-level evidence</Pill>}
    >
      <div className="grid gap-2.5 lg:grid-cols-2">
        {EVIDENCE.map((e) => {
          const done = e.status === "established";
          return (
            <div
              key={e.claim}
              className={
                "rounded-xl border p-3.5 " +
                (done
                  ? "border-clinical-border bg-white"
                  : "border-dashed border-clinical-border bg-clinical-bg/60")
              }
            >
              <div className="flex items-start gap-2.5">
                <span
                  className={
                    "mt-0.5 shrink-0 " + (done ? "text-clinical-green" : "text-clinical-muted")
                  }
                >
                  {done ? <CheckCircle2 size={15} /> : <Hourglass size={15} />}
                </span>
                <div className="min-w-0">
                  <h3
                    className={
                      "text-[12.5px] font-bold leading-snug " +
                      (done ? "text-clinical-ink" : "text-clinical-muted")
                    }
                  >
                    {e.claim}
                  </h3>
                  <p className="mt-1 text-[11.5px] leading-snug text-clinical-muted">{e.detail}</p>
                  {e.stat && (
                    <div className="tabular mt-1.5 inline-block rounded-md border border-clinical-border bg-clinical-bg px-2 py-0.5 text-[11px] font-bold text-clinical-ink">
                      {e.stat}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}
