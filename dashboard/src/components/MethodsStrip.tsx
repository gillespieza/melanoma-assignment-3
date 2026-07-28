import { BrainCircuit, FlaskConical, Waves, ShieldAlert, Layers, ChevronRight } from "lucide-react";

// ---------------------------------------------------------------------------
// The five methods, in one line.
//
// Answers "what is this project?" without costing a click. Q5 is marked as the
// destination because that is what this dashboard actually is — the point where
// the other four converge.
// ---------------------------------------------------------------------------

const METHODS = [
  {
    q: "Q1",
    name: "ML predictor",
    detail: "Gene-expression signatures → P(response)",
    icon: BrainCircuit,
  },
  {
    q: "Q2",
    name: "Validation",
    detail: "Cell lines + proteomics vs the models",
    icon: FlaskConical,
  },
  {
    q: "Q3",
    name: "ODE digital twin",
    detail: "Mechanistic tumour–immune simulation",
    icon: Waves,
  },
  {
    q: "Q4",
    name: "Resistance",
    detail: "Escape routes and salvage targets",
    icon: ShieldAlert,
  },
  {
    q: "Q5",
    name: "Integration",
    detail: "All four methods → one recommendation",
    icon: Layers,
    isDestination: true,
  },
];

export default function MethodsStrip() {
  return (
    <div className="flex flex-wrap items-stretch gap-1.5">
      {METHODS.map((m, i) => {
        const { icon: Icon } = m;
        return (
          <div key={m.q} className="flex flex-1 items-stretch gap-1.5">
            <div
              className={
                "flex-1 rounded-xl border px-3 py-2.5 " +
                (m.isDestination
                  ? "border-clinical-tealdark/35 bg-clinical-teal/[0.07]"
                  : "border-clinical-border bg-white")
              }
            >
              <div className="flex items-center gap-1.5">
                <Icon
                  size={13}
                  className={m.isDestination ? "text-clinical-tealdark" : "text-clinical-muted"}
                />
                <span
                  className={
                    "text-[10px] font-extrabold uppercase tracking-wide " +
                    (m.isDestination ? "text-clinical-tealdark" : "text-clinical-muted")
                  }
                >
                  {m.q}
                </span>
                {m.isDestination && (
                  <span className="ml-auto rounded-full bg-clinical-tealdark px-1.5 py-px text-[9px] font-bold uppercase tracking-wide text-white">
                    You are here
                  </span>
                )}
              </div>
              <div className="mt-1 text-[12.5px] font-bold leading-tight text-clinical-ink">
                {m.name}
              </div>
              <div className="mt-0.5 text-[11px] leading-snug text-clinical-muted">{m.detail}</div>
            </div>

            {i < METHODS.length - 1 && (
              <div className="hidden shrink-0 items-center xl:flex">
                <ChevronRight size={14} className="text-clinical-border" />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
