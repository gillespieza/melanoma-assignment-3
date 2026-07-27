import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import Header from "./components/Header";
import PatientIntake from "./components/PatientIntake";
import SimulationOverlay from "./components/SimulationOverlay";
import RecommendationPanel from "./components/RecommendationPanel";
import TumourForecastChart from "./components/TumourForecastChart";
import SurvivalChart from "./components/SurvivalChart";
import DecisionTree from "./components/DecisionTree";
import ConsultantSignoff from "./components/ConsultantSignoff";
import { Panel } from "./components/ui";
import { PATIENTS } from "./data/patients";
import type { PatientInput } from "./data/types";
import { triage } from "./lib/decisionEngine";
import { Sparkles } from "lucide-react";

export default function App() {
  const [patient, setPatient] = useState<PatientInput>(PATIENTS[0]);
  const [running, setRunning] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  // Live re-computation — this is the "fully interactive branching" behaviour.
  const result = useMemo(() => triage(patient), [patient]);
  const selectedOption =
    result.options.find((o) => o.arm.key === selectedKey) ?? null;

  const handleChange = (p: PatientInput) => {
    setPatient(p);
    setSelectedKey(null); // any edit invalidates the prior sign-off
  };

  const handleRun = () => {
    setRunning(true);
  };

  return (
    <div className="min-h-screen bg-clinical-bg">
      <Header />

      <SimulationOverlay
        show={running}
        onDone={() => {
          setRunning(false);
          setHasRun(true);
        }}
      />

      <main className="mx-auto max-w-[1400px] px-6 py-6">
        <div className="grid gap-5 lg:grid-cols-[380px_1fr]">
          {/* Left: intake */}
          <div className="lg:sticky lg:top-[76px] lg:self-start">
            <PatientIntake
              patient={patient}
              onChange={handleChange}
              onRun={handleRun}
              running={running}
            />
          </div>

          {/* Right: results */}
          <div className="space-y-5">
            {!hasRun ? (
              <EmptyState />
            ) : (
              <motion.div
                key={patient.id + patient.braf + patient.pdl1 + patient.ldh}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.35 }}
                className="space-y-5"
              >
                <HeadlineBanner headline={result.headline} />
                <RecommendationPanel
                  options={result.options}
                  selectedKey={selectedKey}
                  onSelect={setSelectedKey}
                />
                <div className="grid gap-5 xl:grid-cols-[1.35fr_1fr]">
                  <TumourForecastChart patient={patient} />
                  <SurvivalChart patient={patient} />
                </div>
                <DecisionTree path={result.path} />
                <ConsultantSignoff selected={selectedOption} />
              </motion.div>
            )}
          </div>
        </div>

        <footer className="mt-8 border-t border-clinical-border pt-4 text-center text-[11px] text-clinical-muted">
          OncoTwin™ — research demonstrator for UCD AI in Personalised Medicine (Q5). Not a
          medical device. Outputs are model projections, not clinical directives.
        </footer>
      </main>
    </div>
  );
}

function HeadlineBanner({ headline }: { headline: string }) {
  return (
    <div className="flex items-center gap-3 rounded-2xl border border-clinical-tealdark/25 bg-gradient-to-r from-clinical-teal/10 to-clinical-blue/5 px-5 py-3.5 shadow-card">
      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-clinical-tealdark text-white">
        <Sparkles size={17} />
      </div>
      <div>
        <div className="text-[10.5px] font-bold uppercase tracking-wide text-clinical-tealdark">
          Digital-twin recommendation
        </div>
        <div className="text-[14.5px] font-extrabold tracking-tight text-clinical-ink">
          {headline}
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <Panel className="flex min-h-[420px] items-center justify-center text-center">
      <div className="max-w-md">
        <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-clinical-teal/10 text-clinical-tealdark">
          <Sparkles size={26} />
        </div>
        <h2 className="text-[17px] font-extrabold text-clinical-ink">
          Ready to simulate
        </h2>
        <p className="mx-auto mt-2 max-w-sm text-[13px] leading-relaxed text-clinical-muted">
          Choose a patient or set their clinical and molecular profile on the left, then run the
          ODE digital twin. You&apos;ll get ranked treatment options, a 12-month tumour-burden
          forecast, a survival projection, and the decision pathway — ready for consultant sign-off.
        </p>
      </div>
    </Panel>
  );
}
