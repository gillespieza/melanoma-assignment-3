import { RotateCcw, FlaskConical } from "lucide-react";
import type { CohortPatient } from "../data/cohort";
import type { WhatIfEdits } from "../lib/whatIf";

// The what-if editor: one compact row, not a form.
//
// Change a molecular field and every lane below re-runs immediately. This is
// the live "what if this patient had been BRAF wild-type?" moment.

const STAGES = ["I", "II", "III", "IV"];

function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
  modified,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  modified: boolean;
}) {
  return (
    <div>
      <div
        className={
          "mb-1 text-[10px] font-bold uppercase tracking-wide " +
          (modified ? "text-amber-700" : "text-clinical-muted")
        }
      >
        {label}
        {modified && " ·  edited"}
      </div>
      <div className="flex rounded-lg border border-clinical-border bg-white p-0.5">
        {options.map((o) => (
          <button
            key={o.value}
            onClick={() => onChange(o.value)}
            className={
              "rounded-md px-2.5 py-1 text-[12px] font-bold transition " +
              (value === o.value
                ? "bg-okabe-purple-dark text-white"
                : "text-clinical-muted hover:bg-clinical-bg hover:text-clinical-ink")
            }
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function WhatIfBar({
  patient,
  edits,
  onChange,
  onReset,
  changed,
}: {
  patient: CohortPatient;
  edits: WhatIfEdits;
  onChange: (e: WhatIfEdits) => void;
  onReset: () => void;
  changed: string[];
}) {
  const braf = edits.braf ?? patient.braf;
  const nras = edits.nras ?? patient.nras;
  const pdl1 = edits.pdl1Pct ?? patient.pdl1Pct;
  const stage = edits.stageBand ?? patient.stageBand;
  // TCGA has no LDH, so "Normal" is the neutral starting point, not a record.
  const ldh = edits.ldh ?? "Normal";

  const set = (patch: WhatIfEdits) => onChange({ ...edits, ...patch });

  return (
    <div
      className={
        "rounded-2xl border p-4 shadow-card transition " +
        (changed.length ? "border-amber-300 bg-amber-50/50" : "border-clinical-border bg-white")
      }
    >
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wide text-clinical-muted">
          <FlaskConical size={13} /> What-if explorer
        </span>
        <span className="text-[12px] text-clinical-muted">
          Change a field – every method below re-runs instantly.
        </span>
        {changed.length > 0 && (
          <button
            onClick={onReset}
            className="ml-auto flex items-center gap-1.5 rounded-lg border border-amber-400 bg-white px-2.5 py-1.5 text-[11.5px] font-bold text-amber-700 transition hover:bg-amber-100"
          >
            <RotateCcw size={12} /> Restore real patient
          </button>
        )}
      </div>

      <div className="flex flex-wrap items-end gap-x-5 gap-y-3">
        <Segmented
          label="BRAF"
          value={braf}
          modified={changed.includes("BRAF")}
          options={[
            { value: "V600E" as const, label: "V600" },
            { value: "WT" as const, label: "WT" },
          ]}
          onChange={(v) => set({ braf: v })}
        />
        <Segmented
          label="NRAS"
          value={nras}
          modified={changed.includes("NRAS")}
          options={[
            { value: "Mutant" as const, label: "Mutant" },
            { value: "WT" as const, label: "WT" },
          ]}
          onChange={(v) => set({ nras: v })}
        />
        <Segmented
          label="Stage"
          value={stage}
          modified={changed.includes("Stage")}
          options={STAGES.map((s) => ({ value: s, label: s }))}
          onChange={(v) => set({ stageBand: v })}
        />
        <Segmented
          label="LDH"
          value={ldh}
          modified={changed.includes("LDH")}
          options={[
            { value: "Normal" as const, label: "Normal" },
            { value: "Elevated" as const, label: "Elevated" },
            { value: "High" as const, label: "High" },
          ]}
          onChange={(v) => set({ ldh: v })}
        />

        <div className="min-w-[190px] flex-1">
          <div
            className={
              "mb-1 flex items-baseline justify-between text-[10px] font-bold uppercase tracking-wide " +
              (changed.includes("PD-L1") ? "text-amber-700" : "text-clinical-muted")
            }
          >
            <span>PD-L1 {changed.includes("PD-L1") && "·  edited"}</span>
            <span className="tabular text-[12px] text-clinical-ink">{pdl1}th pct</span>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            value={pdl1}
            onChange={(e) => set({ pdl1Pct: Number(e.target.value) })}
            className="w-full accent-okabe-purple-dark"
            aria-label="PD-L1 percentile"
          />
          <div className="mt-0.5 flex justify-between text-[10px] text-clinical-muted">
            <span>Cold</span>
            <span>Threshold 25</span>
            <span>Hot</span>
          </div>
        </div>
      </div>
    </div>
  );
}
