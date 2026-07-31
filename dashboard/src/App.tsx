import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, Loader2 } from "lucide-react";
import Header, { type ViewKey } from "./components/Header";
import CohortTable, { buildRows } from "./components/CohortTable";
import FeaturedCards from "./components/FeaturedCards";
import MethodsStrip from "./components/MethodsStrip";
import PatientView from "./components/PatientView";
import Q1ValidationPanel from "./components/Q1ValidationPanel";
import UserGuide from "./components/UserGuide";
import { Panel } from "./components/ui";
import { loadCohort, type Cohort } from "./data/cohort";

/**
 * Minimal hash routing: `#/cohort`, `#/patient/TCGA-XX-XXXX`, `#/guide`.
 * No router dependency and no storage APIs — it just makes a patient view
 * linkable and survivable across a refresh, which matters during a live demo.
 */
function readHash(): { view: ViewKey; id: string | null } {
  const parts = window.location.hash.replace(/^#\/?/, "").split("/");
  if (parts[0] === "patient" && parts[1]) return { view: "patient", id: decodeURIComponent(parts[1]) };
  if (parts[0] === "guide") return { view: "guide", id: null };
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
        console.info(`Loaded ${c.patients.length} cohort patients (${c.meta.nQ1Tcga} with Q1)`);
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

  const goHome = () => {
    setView("cohort");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

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
      <Header
        view={view}
        onChange={setView}
        onHome={goHome}
        patientEnabled={selectedPatient !== null}
      />

      <main className="mx-auto max-w-[1400px] px-4 py-5 sm:px-6 sm:py-6">
        {loadError ? (
          <LoadErrorState message={loadError} />
        ) : !cohort ? (
          <LoadingState />
        ) : view === "guide" ? (
          <UserGuide />
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
            <MethodsStrip />
            <FeaturedCards rows={rows} onOpen={openPatient} />
            <CohortTable rows={rows} onOpen={openPatient} />
            {cohort.meta.q1Validation && (
              <Q1ValidationPanel validation={cohort.meta.q1Validation} />
            )}
          </div>
        )}

        <footer className="mt-8 border-t border-clinical-border pt-4 text-center text-[11px] text-clinical-muted">
          Melanoma Digital Twin · UCD AI in Personalised Medicine · research demonstrator, not a
          medical device
        </footer>
      </main>
    </div>
  );
}

// Cohort landing headline

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
    <div className="rounded-2xl border border-clinical-border bg-white px-4 py-4 shadow-card sm:px-5">
      {/* Stacked by default; the tiles only sit alongside the text once there is
          genuinely room for both, otherwise they overflow on narrow screens. */}
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start">
        <div className="min-w-0 xl:flex-1">
          <h1 className="text-[16px] font-extrabold tracking-tight text-clinical-ink sm:text-[17px]">
            Melanoma cohort · multi-method triage
          </h1>
          <p className="mt-1 max-w-3xl text-[12.5px] leading-relaxed text-clinical-muted">
            Every patient below is a real TCGA-SKCM case with a full ODE digital twin. Each is scored
            independently by the statistical and mechanistic methods, then integrated into one ranked
            recommendation. The cases where the methods <span className="font-bold">disagree</span> are
            the ones worth a consultant&apos;s attention.
          </p>
        </div>
        <div className="grid shrink-0 grid-cols-2 gap-2.5 sm:grid-cols-4">
          {tiles.map((t) => (
            <div
              key={t.label}
              className="min-w-0 rounded-xl border border-clinical-border bg-clinical-bg px-3 py-2.5 sm:px-3.5"
            >
              <div className="truncate text-[10px] font-bold uppercase tracking-wide text-clinical-muted">
                {t.label}
              </div>
              <div className={"tabular text-[20px] font-extrabold leading-tight sm:text-[22px] " + t.tone}>
                {t.value}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// Archetype workbench — the original v1 experience: three editable patients whose
// molecular fields re-run the logic live. Kept for the scripted demo.


// States

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


