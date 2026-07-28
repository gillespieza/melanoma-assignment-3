import { Activity, Users, User } from "lucide-react";

export type ViewKey = "cohort" | "patient";

const TABS: { key: ViewKey; label: string; icon: typeof Users }[] = [
  { key: "cohort", label: "Cohort", icon: Users },
  { key: "patient", label: "Patient", icon: User },
];

export default function Header({
  view,
  onChange,
  patientEnabled,
}: {
  view: ViewKey;
  onChange: (v: ViewKey) => void;
  patientEnabled: boolean;
}) {
  return (
    <header className="sticky top-0 z-30 border-b border-clinical-border bg-white/85 backdrop-blur">
      <div className="mx-auto flex max-w-[1400px] items-center gap-3 px-6 py-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-clinical-tealdark text-white shadow-card">
          <Activity size={20} strokeWidth={2.4} />
        </div>
        <div className="leading-tight">
          <div className="text-[15px] font-extrabold tracking-tight text-clinical-ink">
            OncoTwin<span className="text-clinical-teal">™</span>
          </div>
          <div className="text-[11px] font-medium text-clinical-muted">
            Melanoma Digital-Twin Decision Support
          </div>
        </div>

        <nav className="ml-6 hidden items-center gap-1 md:flex" aria-label="View">
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
                  "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-[13px] font-semibold transition " +
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

        <div className="ml-auto flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-clinical-blue/10 text-[12px] font-bold text-clinical-bluedark">
              AG
            </div>
            <div className="hidden leading-tight sm:block">
              <div className="text-[12px] font-bold text-clinical-ink">Dr. Aoife Gallagher</div>
              <div className="text-[10px] text-clinical-muted">Consultant Oncologist</div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
