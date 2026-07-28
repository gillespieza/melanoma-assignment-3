import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, Loader2, Sparkles } from "lucide-react";
import Header, { type ViewKey } from "./components/Header";
import PatientIntake from "./components/PatientIntake";
import SimulationOverlay from "./components/SimulationOverlay";
import RecommendationPanel from "./components/RecommendationPanel";
import TumourForecastChart from "./components/TumourForecastChart";
import SurvivalChart from "./components/SurvivalChart";
import DecisionTree from "./components/DecisionTree";
import ConsultantSignoff from "./components/ConsultantSignoff";
import CohortTable, { buildRows } from "./components/CohortTable";
import FeaturedCards from "./components/FeaturedCards";
import PatientView from "./components/PatientView";
import Q1ValidationPanel from "./components/Q1ValidationPanel";
import { Panel } from "./components/ui";
import { PATIENTS } from "./data/patients";
import type { PatientInput } from "./data/types";
import { loadCohort, type Cohort } from "./data/cohort";
import { triage } from "./lib/decisionEngine";
import { forecastFromOptions, survivalFromOptions } from "./lib/forecast";

/**
 * Minimal hash routing: `#/cohort`, `#/archetypes`, `#/patient/TCGA-XX-XXXX`.
 * No router dependency and no storage APIs — it just makes a patient view
 * linkable and survivable across a refresh, which matters during a live demo.
 */
function readHash(): { view: ViewKey; id: string | null } {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/");
  if (parts[0] === "patient" && parts[1]) return { view: "patient", id: decodeURIComponent(parts[1]) };
  if (parts[0] === "archetypes") return { view: "archetypes", id: null };
  return { view: "cohort", id: null };
}

export default function App() {
  const [cohort, setCohort] = useState<Cohort | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const initial = readHash();
  const [view, setView] = useState<ViewKey>(initial.view);
  const [selectedId, setSelectedId] = useState<string | null>(initial.id);

  // Keep the URL in step with the view, and respond to back/forward.
  useEffect(() => {
    const target =
      view === "patient" && selectedId
        ? `#/patient/${encodeURIComponent(selectedId)}`
        : `#/${view}`;
    if (window.location.hash !== target) window.history.replaceState(null, "", target);
  }, [view, selectedId]);

  useEffect(() => {
    const onHashChange = () => {
      const next = readHash();
      setView(next.view);
      if (next.id) setSelectedId(next.id);
    };
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    let cancelled = false;
    loadCohort()
      .then((c) => {
        if (cancelled) return;
        setCohort(c);
        console.info(`OncoTwin: loaded ${c.patients.length} cohort patients (${c.meta.nQ1Tcga} with Q1)`);
      })
      .catch((e: unknown) => {
        if (!cancelled) setLoadError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const rows = useMemo(() => (cohort ? buildRows(cohort.patients) : []), [cohort]);

  /** Percentile of each patient's Q1 P(response) across the cohort. */
  const q1Ranks = useMemo(() => {
    const map = new Map<string, number>();
    if (!cohort) return map;
    const scored = cohort.patients.filter((p) => p.q1);
    const sorted = scored.map((p) => p.q1!.pResponse).sort((a, b) => a - b);
    for (const p of scored) {
      const v = p.q1!.pResponse;
      const below = sorted.filter((s) => s < v).length;
      const ties = sorted.filter((s) => s === v).length;
      map.set(p.id, Math.round(((below + ties / 2) / sorted.length) * 100));
    }
    return map;
  }, [cohort]);

  const selectedIndex = cohort ? cohort.patients.findIndex((p) => p.id === selectedId) : -1;
  const selectedPatient = selectedIndex >= 0 ? cohort!.patients[selectedIndex] : null;

  const openPatient = (id: string) => {
    setSelectedId(id);
    setView("patient");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const stepPatient = (delta: number) => {
    if (!cohort || selectedIndex < 0) return;
    const next = (selectedIndex + delta + cohort.patients.length) % cohort.patients.length;
    setSelectedId(cohort.patients[next].id);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="min-h-screen bg-clinical-bg">
      <Header view={view} onChange={setView} patientEnabled={selectedPatient !== null} />

      <main className="mx-auto max-w-[1400px] px-6 py-6">
        {view === "archetypes" ? (
          <ArchetypeWorkbench />
        ) : loadError ? (
          <LoadErrorState message={loadError} />
        ) : !cohort ? (
          <LoadingState />
        ) : view === "patient" && selectedPatient ? (
          <PatientView
            patient={selectedPatient}
            meta={cohort.meta}
            q1Rank={q1Ranks.get(selectedPatient.id) ?? null}
            onBack={() => setView("cohort")}
            onStep={stepPatient}
            position={{ index: selectedIndex, total: cohort.patients.length }}
          />
        ) : (
          <div className="space-y-4">
            <CohortHeadline cohort={cohort} rows={rows} />
            <FeaturedCards
              rows={rows}
              onOpen={openPatient}
              onOpenArchetypes={() => setView("archetypes")}
            />
            <CohortTable rows={rows} onOpen={openPatient} />
            {cohort.meta.q1Validation && (
              <Q1ValidationPanel validation={cohort.meta.q1Validation} />
            )}
          </div>
        )}

        <footer className="mt-8 border-t border-clinical-border pt-4 text-center text-[11px] text-clinical-muted">
          OncoTwin™ — research demonstrator for UCD AI in Personalised Medicine (Q5). Not a
          medical device. Outputs are model projections, not clinical directives.
        </footer>
      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Cohort landing headline
// ---------------------------------------------------------------------------

function CohortHeadline({
  cohort,
  rows,
}: {
  cohort: Cohort;
  rows: ReturnType<typeof buildRows>;
}) {
  const discordant = rows.filter((r) => r.agreement === "discordant").length;
  const concordant = rows.filter((r) => r.agreement === "concordant").length;

  const tiles = [
    { label: "Patients", value: cohort.patients.length, tone: "text-clinical-ink" },
    { label: "BRAF V600", value: cohort.patients.filter((p) => p.braf !== "WT").length, tone: "text-clinical-bluedark" },
    { label: "Methods concordant", value: concordant, tone: "text-green-700" },
    { label: "Methods split", value: discordant, tone: "text-amber-700" },
  ];

  return (
    <div className="rounded-2xl border border-clinical-border bg-white px-5 py-4 shadow-card">
      <div className="flex flex-wrap items-start gap-4">
        <div className="min-w-0 flex-1">
          <h1 className="text-[17px] font-extrabold tracking-tight text-clinical-ink">
            Melanoma cohort · multi-method triage
          </h1>
          <p className="mt-1 max-w-3xl text-[12.5px] leading-relaxed text-clinical-muted">
            Every patient below is a real TCGA-SKCM case with a full ODE digital twin. Each is scored
            independently by the statistical and mechanistic methods, then integrated into one ranked
            recommendation. The cases where the methods <span className="font-bold">disagree</span> are
            the ones worth a consultant&apos;s attention.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
          {tiles.map((t) => (
            <div
              key={t.label}
              className="rounded-xl border border-clinical-border bg-clinical-bg px-3.5 py-2.5"
            >
              <div className="text-[10px] font-bold uppercase tracking-wide text-clinical-muted">
                {t.label}
              </div>
              <div className={"tabular text-[22px] font-extrabold leading-tight " + t.tone}>
                {t.value}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Archetype workbench — the original v1 experience: three editable patients whose
// molecular fields re-run the logic live. Kept for the scripted demo.
// ---------------------------------------------------------------------------

function ArchetypeWorkbench() {
  const [patient, setPatient] = useState<PatientInput>(PATIENTS[0]);
  const [running, setRunning] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  const result = useMemo(() => triage(patient), [patient]);
  const forecast = useMemo(() => forecastFromOptions(result.options), [result]);
  const survival = useMemo(() => survivalFromOptions(result.options), [result]);
  const selectedOption = result.options.find((o) => o.arm.key === selectedKey) ?? null;

  const handleChange = (p: PatientInput) => {
    setPatient(p);
    setSelectedKey(null); // any edit invalidates the prior sign-off
  };

  return (
    <>
      <SimulationOverlay
        show={running}
        onDone={() => {
          setRunning(false);
          setHasRun(true);
        }}
      />

      <div className="grid gap-5 lg:grid-cols-[380px_1fr]">
        <div className="lg:sticky lg:top-[76px] lg:self-start">
          <PatientIntake
            patient={patient}
            onChange={handleChange}
            onRun={() => setRunning(true)}
            running={running}
          />
        </div>

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
                <TumourForecastChart forecast={forecast} />
                <SurvivalChart survival={survival} />
              </div>
              <DecisionTree path={result.path} />
              <ConsultantSignoff selected={selectedOption} />
            </motion.div>
          )}
        </div>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// States
// ---------------------------------------------------------------------------

function LoadingState() {
  return (
    <Panel className="flex min-h-[420px] items-center justify-center text-center">
      <div className="max-w-md">
        <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-clinical-teal/10 text-clinical-tealdark">
          <Loader2 size={26} className="animate-spin" />
        </div>
        <h2 className="text-[17px] font-extrabold text-clinical-ink">Loading cohort</h2>
        <p className="mx-auto mt-2 max-w-sm text-[13px] leading-relaxed text-clinical-muted">
          Reading the digital-twin simulations for the TCGA-SKCM cohort.
        </p>
      </div>
    </Panel>
  );
}

function LoadErrorState({ message }: { message: string }) {
  return (
    <Panel className="flex min-h-[420px] items-center justify-center text-center">
      <div className="max-w-lg">
        <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-amber-50 text-amber-600">
          <AlertTriangle size={26} />
        </div>
        <h2 className="text-[17px] font-extrabold text-clinical-ink">Cohort data unavailable</h2>
        <p className="mx-auto mt-2 max-w-md text-[13px] leading-relaxed text-clinical-muted">
          {message}
        </p>
        <code className="mt-3 inline-block rounded-lg border border-clinical-border bg-clinical-bg px-3 py-1.5 text-[12px] font-semibold text-clinical-ink">
          node scripts/build_cohort.mjs
        </code>
      </div>
    </Panel>
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
        <h2 className="text-[17px] font-extrabold text-clinical-ink">Ready to simulate</h2>
        <p className="mx-auto mt-2 max-w-sm text-[13px] leading-relaxed text-clinical-muted">
          Choose a patient or set their clinical and molecular profile on the left, then run the
          ODE digital twin. You&apos;ll get ranked treatment options, a 12-month tumour-burden
          forecast, a survival projection, and the decision pathway — ready for consultant sign-off.
        </p>
      </div>
    </Panel>
  );
}
