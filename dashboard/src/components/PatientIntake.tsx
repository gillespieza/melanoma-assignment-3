import type { PatientInput, BrafStatus, LdhBand, Stage, Ecog } from "../data/types";
import { PATIENTS } from "../data/patients";
import { Panel, Pill } from "./ui";
import { UserRound, Dna, FlaskConical, Play } from "lucide-react";

const STAGES: Stage[] = ["IIIB", "IIIC", "IIID", "IVa (M1a)", "IVb (M1b)", "IVc (M1c)"];
const LDHS: LdhBand[] = ["Normal", "Elevated", "High (>2x ULN)"];
const BRAFS: BrafStatus[] = ["V600E", "V600K", "WT"];

function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: T[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1 rounded-lg bg-clinical-bg p-1">
      {options.map((o) => (
        <button
          key={o}
          onClick={() => onChange(o)}
          className={
            "rounded-md px-2.5 py-1 text-[12px] font-semibold transition " +
            (value === o
              ? "bg-white text-clinical-ink shadow-sm"
              : "text-clinical-muted hover:text-clinical-ink")
          }
        >
          {o}
        </button>
      ))}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1.5 text-[11px] font-bold uppercase tracking-wide text-clinical-muted">
        {label}
      </div>
      {children}
    </div>
  );
}

export default function PatientIntake({
  patient,
  onChange,
  onRun,
  running,
}: {
  patient: PatientInput;
  onChange: (p: PatientInput) => void;
  onRun: () => void;
  running: boolean;
}) {
  const set = (patch: Partial<PatientInput>) => onChange({ ...patient, ...patch });

  return (
    <Panel
      title="Patient Intake & Molecular Workup"
      subtitle="Select a case or edit any field — the twin re-simulates live."
      icon={<UserRound size={16} />}
    >
      {/* Case selector */}
      <div className="mb-4 grid grid-cols-3 gap-2">
        {PATIENTS.map((pt) => (
          <button
            key={pt.id}
            onClick={() => onChange(pt)}
            className={
              "rounded-xl border px-3 py-2 text-left transition " +
              (patient.id === pt.id
                ? "border-clinical-tealdark bg-clinical-teal/5 shadow-sm"
                : "border-clinical-border hover:border-clinical-teal/40")
            }
          >
            <div className="text-[12px] font-extrabold text-clinical-ink">Patient {pt.id}</div>
            <div className="mt-0.5 text-[10.5px] leading-snug text-clinical-muted">
              {pt.braf === "WT" ? "BRAF WT" : "BRAF mut"} · PD-L1 {pt.pdl1}%
            </div>
          </button>
        ))}
      </div>

      <p className="mb-4 rounded-lg border border-clinical-border bg-clinical-bg px-3 py-2 text-[11.5px] italic leading-snug text-clinical-muted">
        {patient.vignette}
      </p>

      <div className="space-y-4">
        <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-wide text-clinical-tealdark">
          <UserRound size={13} /> Clinical
        </div>
        <div className="grid grid-cols-2 gap-3">
          <Field label="Age">
            <input
              type="number"
              value={patient.age}
              onChange={(e) => set({ age: Number(e.target.value) })}
              className="w-full rounded-lg border border-clinical-border bg-white px-3 py-1.5 text-[13px] font-semibold text-clinical-ink outline-none focus:border-clinical-teal"
            />
          </Field>
          <Field label="ECOG PS">
            <Segmented
              value={String(patient.ecog) as "0" | "1" | "2"}
              options={["0", "1", "2"]}
              onChange={(v) => set({ ecog: Number(v) as Ecog })}
            />
          </Field>
        </div>
        <Field label="AJCC Stage">
          <Segmented value={patient.stage} options={STAGES} onChange={(v) => set({ stage: v })} />
        </Field>
        <Field label="Serum LDH">
          <Segmented value={patient.ldh} options={LDHS} onChange={(v) => set({ ldh: v })} />
        </Field>

        <div className="flex items-center gap-2 pt-1 text-[11px] font-bold uppercase tracking-wide text-clinical-tealdark">
          <Dna size={13} /> Molecular
        </div>
        <Field label="BRAF status">
          <Segmented value={patient.braf} options={BRAFS} onChange={(v) => set({ braf: v })} />
        </Field>

        <Field label={`PD-L1 (TPS) — ${patient.pdl1}%`}>
          <input
            type="range"
            min={0}
            max={100}
            value={patient.pdl1}
            onChange={(e) => set({ pdl1: Number(e.target.value) })}
            className="w-full accent-clinical-tealdark"
          />
          <div className="mt-0.5 flex justify-between text-[10px] text-clinical-muted">
            <span>Cold (0%)</span>
            <span>Threshold 25%</span>
            <span>Hot (100%)</span>
          </div>
        </Field>

        <Field label={`Q1 response signature — ${patient.signature} / 100`}>
          <input
            type="range"
            min={0}
            max={100}
            value={patient.signature}
            onChange={(e) => set({ signature: Number(e.target.value) })}
            className="w-full accent-clinical-blue"
          />
          <div className="mt-1 flex items-center gap-1.5 text-[10.5px] text-clinical-muted">
            <FlaskConical size={12} /> IMPRES-style expression score (our project&apos;s value-add)
          </div>
        </Field>
      </div>

      <button
        onClick={onRun}
        disabled={running}
        className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-clinical-tealdark py-3 text-[14px] font-bold text-white shadow-lift transition hover:bg-clinical-teal disabled:opacity-60"
      >
        <Play size={16} strokeWidth={2.6} />
        {running ? "Simulating…" : "Run Digital-Twin Simulation"}
      </button>

      <div className="mt-3 flex flex-wrap gap-1.5">
        <Pill tone="teal">TCGA-SKCM n=421</Pill>
        <Pill tone="blue">Mechanistic ODE · 4 modules</Pill>
        <Pill>Literature-parameterised</Pill>
      </div>
    </Panel>
  );
}
