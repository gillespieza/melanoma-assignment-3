import { ShieldAlert, TriangleAlert, Shield, PackageOpen } from "lucide-react";
import type { CohortPatient, ResistanceRisk } from "../data/cohort";
import { Panel, Pill } from "./ui";

// ---------------------------------------------------------------------------
// Q4 · Resistance & salvage targets.
//
// These flags are HEURISTIC — derived from the Q3 residual burden plus driver
// status, not from a fitted resistance model. The panel says so, plainly and
// permanently, because a clinician must know which numbers carry model weight.
// ---------------------------------------------------------------------------

const RISK: Record<ResistanceRisk, { label: string; tone: "green" | "amber" | "rose" | "neutral"; blurb: string }> = {
  low: {
    label: "Low",
    tone: "green",
    blurb: "At least one simulated arm clears substantial burden across the dose sweep.",
  },
  moderate: {
    label: "Moderate",
    tone: "amber",
    blurb: "Partial simulated response — plan the next line before progression.",
  },
  high: {
    label: "High",
    tone: "rose",
    blurb: "Little simulated benefit in either arm; escalate early and hold salvage options ready.",
  },
  unknown: {
    label: "Not assessable",
    tone: "neutral",
    blurb: "The ODE twin could not score this patient, so resistance risk cannot be derived from Q3.",
  },
};

export default function Q4Resistance({ patient }: { patient: CohortPatient }) {
  const { q4 } = patient;
  const risk = RISK[q4.resistanceRisk];

  return (
    <Panel
      title="Q4 · Resistance & Salvage Targets"
      subtitle="What is likely to fail, and what to hold in reserve"
      icon={<ShieldAlert size={16} />}
      right={<Pill tone="neutral">Heuristic</Pill>}
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div>
          <div className="flex items-center gap-2.5 rounded-xl border border-clinical-border bg-clinical-bg px-3.5 py-3">
            <Shield size={18} className="shrink-0 text-clinical-muted" />
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-[10.5px] font-bold uppercase tracking-wide text-clinical-muted">
                  Resistance risk
                </span>
                <Pill tone={risk.tone}>{risk.label}</Pill>
              </div>
              <p className="mt-1 text-[11.5px] leading-snug text-clinical-muted">{risk.blurb}</p>
            </div>
          </div>

          <div className="mt-3 space-y-2">
            <div className="text-[10.5px] font-bold uppercase tracking-wide text-clinical-muted">
              Escape mechanisms flagged
            </div>
            {q4.flags.length === 0 ? (
              <p className="rounded-lg border border-clinical-border bg-white px-3 py-2.5 text-[12px] text-clinical-muted">
                No specific escape mechanism flagged for this profile.
              </p>
            ) : (
              q4.flags.map((f) => (
                <div
                  key={f.label}
                  className="flex items-start gap-2 rounded-lg border border-clinical-border bg-white px-3 py-2.5"
                >
                  <TriangleAlert size={13} className="mt-0.5 shrink-0 text-amber-600" />
                  <div className="min-w-0">
                    <div className="text-[12px] font-bold text-clinical-ink">{f.label}</div>
                    <p className="mt-0.5 text-[11.5px] leading-snug text-clinical-muted">{f.detail}</p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div>
          <div className="mb-2 flex items-center gap-1.5 text-[10.5px] font-bold uppercase tracking-wide text-clinical-muted">
            <PackageOpen size={12} /> Hold in reserve
          </div>
          <ol className="space-y-2">
            {q4.reserve.map((r, i) => (
              <li
                key={r}
                className="flex items-start gap-2.5 rounded-lg border border-clinical-border bg-white px-3 py-2.5"
              >
                <span className="tabular mt-px flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-clinical-blue/10 text-[11px] font-extrabold text-clinical-bluedark">
                  {i + 1}
                </span>
                <span className="text-[12px] font-semibold leading-snug text-clinical-ink">{r}</span>
              </li>
            ))}
          </ol>
        </div>
      </div>

      <p className="mt-4 border-t border-clinical-border pt-3 text-[11.5px] leading-relaxed text-clinical-muted">
        <span className="font-bold text-clinical-ink">Provenance:</span> these flags are rule-based,
        derived from the Q3 residual burden together with BRAF/NRAS/MAPK driver status and PD-L1
        percentile. They are a structured prompt for the MDT, not a fitted resistance model, and they
        carry less evidential weight than the Q1 and Q3 read-outs above.
      </p>
    </Panel>
  );
}
