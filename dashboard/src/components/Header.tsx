import { Activity, ShieldCheck } from "lucide-react";

export default function Header() {
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

        <nav className="ml-6 hidden items-center gap-1 md:flex">
          {["Triage", "Cohort", "Trials", "Audit"].map((t, i) => (
            <span
              key={t}
              className={
                "rounded-md px-3 py-1.5 text-[13px] font-semibold " +
                (i === 0
                  ? "bg-clinical-bg text-clinical-ink"
                  : "text-clinical-muted hover:text-clinical-ink")
              }
            >
              {t}
            </span>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-4">
          <div className="hidden items-center gap-1.5 rounded-full border border-clinical-border bg-clinical-bg px-3 py-1 text-[11px] font-semibold text-clinical-tealdark sm:flex">
            <ShieldCheck size={13} /> Decision support · not a directive
          </div>
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
