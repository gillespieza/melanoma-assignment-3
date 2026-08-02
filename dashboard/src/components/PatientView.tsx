import { useMemo, useState } from "react";
import {
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  Play,
  Sparkles,
  BrainCircuit,
  FlaskConical,
  Waves,
  GitBranch,
  TriangleAlert,
} from "lucide-react";
import type { CohortMeta, CohortPatient, Q1Validation } from "../data/cohort";
import { integrate } from "../lib/integrationEngine";
import { forecastFromOptions, survivalFromOptions } from "../lib/forecast";
import { applyWhatIf, changedFields, NO_EDITS, type WhatIfEdits } from "../lib/whatIf";
import PatientPassport from "./PatientPassport";
import WhatIfBar from "./WhatIfBar";
import SimulationOverlay from "./SimulationOverlay";
import Q1Lane from "./Q1Lane";
import Q2Evidence from "./Q2Evidence";
import DoseResponseChart from "./DoseResponseChart";
import TumourForecastChart from "./TumourForecastChart";
import SurvivalChart from "./SurvivalChart";
import AgreementBadge from "./AgreementBadge";
import RecommendationPanel from "./RecommendationPanel";
import DecisionTree from "./DecisionTree";
import ConsultantSignoff from "./ConsultantSignoff";
import { Panel } from "./ui";

// The patient workbench.
//
// Deliberately kept to a short vertical stack: passport → what-if → the answer
// → method detail behind tabs. The full multi-method story is all still here,
// but only one method is on screen at a time so the page never reads as a wall
// of panels.

type TabKey = "q1" | "q2" | "q3" | "path";

const TABS: { key: TabKey; label: string; icon: typeof BrainCircuit }[] = [
  { key: "q1", label: "Q1 · ML predictor", icon: BrainCircuit },
  { key: "q2", label: "Q2 · Validation", icon: FlaskConical },
  { key: "q3", label: "Q3 · Digital twin", icon: Waves },
  { key: "path", label: "Decision path", icon: GitBranch },
];

export default function PatientView({
  patient: realPatient,
  meta,
  q1Rank,
  onBack,
  onStep,
  position,
}: {
  patient: CohortPatient;
  meta: CohortMeta;
  q1Rank: number | null;
  onBack: () => void;
  onStep: (delta: number) => void;
  position: { index: number; total: number };
}) {
  const [edits, setEdits] = useState<WhatIfEdits>(NO_EDITS);
  const [running, setRunning] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [tab, setTab] = useState<TabKey>("q3");

  // Reset the whole workbench when a different patient is opened.
  const [lastId, setLastId] = useState(realPatient.id);
  if (lastId !== realPatient.id) {
    setLastId(realPatient.id);
    setEdits(NO_EDITS);
    setHasRun(false);
    setSelectedKey(null);
    setTab("q3");
  }

  const changed = changedFields(realPatient, edits);
  const { patient, synthetic } = useMemo(
    () => applyWhatIf(realPatient, edits),
    [realPatient, edits]
  );

  const result = useMemo(() => integrate(patient), [patient]);
  const forecast = useMemo(() => forecastFromOptions(result.options), [result]);
  const survival = useMemo(() => survivalFromOptions(result.options), [result]);
  const selectedOption = result.options.find((o) => o.arm.key === selectedKey) ?? null;

  const handleEdits = (next: WhatIfEdits) => {
    setEdits(next);
    setSelectedKey(null); // any edit invalidates a prior sign-off
  };

  return (
    <div key={realPatient.id} className="animate-fade-up space-y-4">
      <SimulationOverlay
        show={running}
        onDone={() => {
          setRunning(false);
          setHasRun(true);
        }}
      />

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

      <PatientPassport patient={patient} modified={changed} />

      <WhatIfBar
        patient={realPatient}
        edits={edits}
        onChange={handleEdits}
        onReset={() => handleEdits(NO_EDITS)}
        changed={changed}
      />

      {!hasRun ? (
        <RunGate patient={realPatient} onRun={() => setRunning(true)} />
      ) : (
        <>
          {synthetic && <SyntheticNotice changed={changed} />}

          {/* --- the answer --- */}
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

          <AgreementBadge agreement={result.agreement} />

          <RecommendationPanel
            options={result.options}
            selectedKey={selectedKey}
            onSelect={setSelectedKey}
          />

          {/* --- method detail, one at a time --- */}
          <div>
            <div className="flex flex-wrap gap-1.5 border-b border-clinical-border pb-2">
              {TABS.map(({ key, label, icon: Icon }) => (
                <button
                  key={key}
                  onClick={() => setTab(key)}
                  className={
                    "flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12.5px] font-bold transition " +
                    (tab === key
                      ? "bg-clinical-tealdark text-white shadow-card"
                      : "text-clinical-muted hover:bg-clinical-bg hover:text-clinical-ink")
                  }
                >
                  <Icon size={13} /> {label}
                </button>
              ))}
            </div>

            <div className="mt-4 space-y-4">
              {tab === "q1" && (
                <Q1Lane
                  patient={patient}
                  note={meta.q1Note}
                  validation={meta.q1Validation as Q1Validation | null}
                  cohortRank={q1Rank}
                />
              )}
              {tab === "q2" && <Q2Evidence />}
              {tab === "q3" && (
                <>
                  <DoseResponseChart patient={patient} doseAxis={meta.doseAxis} />
                  <div className="grid gap-4 xl:grid-cols-[1.35fr_1fr]">
                    <TumourForecastChart forecast={forecast} />
                    <SurvivalChart survival={survival} />
                  </div>
                </>
              )}
              {tab === "path" && <DecisionTree path={result.path} />}
            </div>
          </div>

          <ConsultantSignoff selected={selectedOption} />
        </>
      )}
    </div>
  );
}


function RunGate({ patient, onRun }: { patient: CohortPatient; onRun: () => void }) {
  return (
    <Panel className="text-center">
      <div className="mx-auto max-w-lg py-6">
        <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-clinical-teal/10 text-clinical-tealdark">
          <Waves size={26} />
        </div>
        <h2 className="text-[17px] font-extrabold text-clinical-ink">
          Ready to run the digital twin
        </h2>
        <p className="mx-auto mt-2 text-[13px] leading-relaxed text-clinical-muted">
          {patient.id} · BRAF {patient.braf === "WT" ? "wild-type" : "V600"} · PD-L1{" "}
          {patient.pdl1Pct}th percentile. Run the twin to score this patient across all four
          methods and produce the integrated recommendation.
        </p>
        <button
          onClick={onRun}
          className="mx-auto mt-5 flex items-center gap-2 rounded-xl bg-clinical-tealdark px-6 py-3 text-[13.5px] font-bold text-white shadow-lift transition hover:brightness-110"
        >
          <Play size={16} /> Run Digital-Twin Simulation
        </button>
      </div>
    </Panel>
  );
}

function SyntheticNotice({ changed }: { changed: string[] }) {
  return (
    <div className="flex items-start gap-2.5 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3">
      <TriangleAlert size={15} className="mt-0.5 shrink-0 text-amber-600" />
      <p className="text-[12.5px] leading-snug text-clinical-ink">
        <span className="font-bold">Hypothetical patient.</span> You changed{" "}
        {changed.join(" and ")}, so this is no longer the real record. The tumour curves shown are
        the cohort-average sweep for the edited profile, not this patient&apos;s own simulation.
      </p>
    </div>
  );
}
