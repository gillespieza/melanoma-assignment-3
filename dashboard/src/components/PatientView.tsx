import { useMemo, useState } from "react";
import { ArrowLeft, ChevronLeft, ChevronRight, Layers, Sparkles } from "lucide-react";
import type { CohortMeta, CohortPatient } from "../data/cohort";
import { integrate } from "../lib/integrationEngine";
import { forecastFromOptions, survivalFromOptions } from "../lib/forecast";
import PatientPassport from "./PatientPassport";
import Q1Lane from "./Q1Lane";
import Q2Evidence from "./Q2Evidence";
import DoseResponseChart from "./DoseResponseChart";
import TumourForecastChart from "./TumourForecastChart";
import SurvivalChart from "./SurvivalChart";
import Q4Resistance from "./Q4Resistance";
import AgreementBadge from "./AgreementBadge";
import RecommendationPanel from "./RecommendationPanel";
import DecisionTree from "./DecisionTree";
import ConsultantSignoff from "./ConsultantSignoff";
import { Panel } from "./ui";

// ---------------------------------------------------------------------------
// The full patient story: passport → Q1 → Q2 → Q3 → Q4 → Q5, in that order, so
// the recommendation arrives only after every method has shown its working.
// ---------------------------------------------------------------------------

function LaneDivider({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 pt-1">
      <span className="text-[10.5px] font-bold uppercase tracking-[0.08em] text-clinical-muted">
        {label}
      </span>
      <span className="h-px flex-1 bg-clinical-border" />
    </div>
  );
}

export default function PatientView({
  patient,
  meta,
  onBack,
  onStep,
  position,
}: {
  patient: CohortPatient;
  meta: CohortMeta;
  onBack: () => void;
  onStep: (delta: number) => void;
  position: { index: number; total: number };
}) {
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  const result = useMemo(() => integrate(patient), [patient]);
  const forecast = useMemo(() => forecastFromOptions(result.options), [result]);
  const survival = useMemo(() => survivalFromOptions(result.options), [result]);

  const selectedOption = result.options.find((o) => o.arm.key === selectedKey) ?? null;

  return (
    <div key={patient.id} className="animate-fade-up space-y-4">
      {/* --- context bar --- */}
      <div className="flex flex-wrap items-center gap-2.5">
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 rounded-lg border border-clinical-border bg-white px-3 py-1.5 text-[12.5px] font-bold text-clinical-ink transition hover:border-clinical-tealdark hover:text-clinical-tealdark"
        >
          <ArrowLeft size={14} /> Back to cohort
        </button>

        <div className="ml-auto flex items-center gap-1.5">
          <span className="tabular text-[11.5px] font-semibold text-clinical-muted">
            Patient {position.index + 1} of {position.total}
          </span>
          <button
            onClick={() => onStep(-1)}
            aria-label="Previous patient"
            className="rounded-lg border border-clinical-border bg-white p-1.5 text-clinical-ink transition hover:border-clinical-tealdark hover:text-clinical-tealdark"
          >
            <ChevronLeft size={15} />
          </button>
          <button
            onClick={() => onStep(1)}
            aria-label="Next patient"
            className="rounded-lg border border-clinical-border bg-white p-1.5 text-clinical-ink transition hover:border-clinical-tealdark hover:text-clinical-tealdark"
          >
            <ChevronRight size={15} />
          </button>
        </div>
      </div>

      <PatientPassport patient={patient} />

      <LaneDivider label="Method 1 · Statistical" />
      <Q1Lane patient={patient} note={meta.q1Note} validation={meta.q1Validation} />

      <LaneDivider label="Method 2 · Experimental" />
      <Q2Evidence />

      <LaneDivider label="Method 3 · Mechanistic" />
      <DoseResponseChart patient={patient} doseAxis={meta.doseAxis} />
      <div className="grid gap-4 xl:grid-cols-[1.35fr_1fr]">
        <TumourForecastChart forecast={forecast} />
        <SurvivalChart survival={survival} />
      </div>

      <LaneDivider label="Method 4 · Resistance" />
      <Q4Resistance patient={patient} />

      <LaneDivider label="Q5 · Integrated Recommendation" />

      <Panel
        title="Methods Agreement"
        subtitle="Do the independent methods converge on this patient?"
        icon={<Layers size={16} />}
      >
        <AgreementBadge agreement={result.agreement} />
      </Panel>

      <div className="flex items-center gap-3 rounded-2xl border border-clinical-tealdark/25 bg-gradient-to-r from-clinical-teal/10 to-clinical-blue/5 px-5 py-3.5 shadow-card">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-clinical-tealdark text-white">
          <Sparkles size={17} />
        </div>
        <div>
          <div className="text-[10.5px] font-bold uppercase tracking-wide text-clinical-tealdark">
            Integrated recommendation
          </div>
          <div className="text-[14.5px] font-extrabold tracking-tight text-clinical-ink">
            {result.headline}
          </div>
        </div>
      </div>

      <RecommendationPanel
        options={result.options}
        selectedKey={selectedKey}
        onSelect={setSelectedKey}
      />
      <DecisionTree path={result.path} />
      <ConsultantSignoff selected={selectedOption} />
    </div>
  );
}
