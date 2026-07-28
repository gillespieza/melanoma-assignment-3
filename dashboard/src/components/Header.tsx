import { Activity, Users, User } from "lucide-react";

export type ViewKey = "cohort" | "patient";

const TABS: { key: ViewKey; label: string; icon: typeof Users }[] = [
  { key: "cohort", label: "Cohort", icon: Users },
  { key: "patient", label: "Patient", icon: User },
];

export default function Header({
  view,
  onChange,
  onHome,
  patientEnabled,
}: {
  view: ViewKey;
  onChange: (v: ViewKey) => void;
  onHome: () => void;
  patientEnabled: boolean;
}) {
  return (
    <header className="sticky top-0 z-30 border-b border-clinical-border bg-white/85 backdrop-blur">
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-3 gap-y-2 px-4 py-3 sm:px-6">
        {/* The mark and title are the way home from anywhere. */}
        <button
          onClick={onHome}
          aria-label="Back to cohort"
          className="flex items-center gap-3 rounded-lg text-left transition hover:opacity-80"
        >
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-clinical-tealdark text-white shadow-card">
            <Activity size={20} strokeWidth={2.4} />
          </span>
          <span className="leading-tight">
            <span className="block text-[15px] font-extrabold tracking-tight text-clinical-ink">
              Melanoma Digital Twin
            </span>
            <span className="hidden text-[11px] font-medium text-clinical-muted sm:block">
              Multi-method treatment decision support
            </span>
          </span>
        </button>

        <nav className="order-3 flex w-full items-center gap-1 sm:order-none sm:ml-4 sm:w-auto" aria-label="View">
          {TABS.map(({ key, label, icon: Icon }) => {
            const disabled = key === "patient" && !patientEnabled;
            const active = view === key;
            return (
              <button
                key={key}
                onClick={() => !disabled && onChange(key)}
                disabled={disabled}
                title={disabled ? "Select a patient from the cohort first" : undefined}
                className={
                  "flex flex-1 items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-[13px] font-semibold transition sm:flex-none " +
                  (active
                    ? "bg-clinical-bg text-clinical-ink"
                    : disabled
                      ? "cursor-not-allowed text-clinical-border"
                      : "text-clinical-muted hover:bg-clinical-bg/70 hover:text-clinical-ink")
                }
              >
                <Icon size={14} /> {label}
              </button>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-2">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-clinical-blue/10 text-[12px] font-bold text-clinical-bluedark">
            AG
          </div>
          <div className="hidden leading-tight lg:block">
            <div className="text-[12px] font-bold text-clinical-ink">Dr. Aoife Gallagher</div>
            <div className="text-[10px] text-clinical-muted">Consultant Oncologist</div>
          </div>
        </div>
      </div>
    </header>
  );
}
