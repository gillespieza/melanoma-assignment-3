import { BrainCircuit, FlaskConical, Waves, ShieldAlert, Layers, ChevronRight } from "lucide-react";

// The five methods, in one line. Q5 is marked as the destination — the point
// where the other four converge into one recommendation.

const METHODS = [
  {
    name: "ML predictor",
    detail: "Gene-expression signatures → P(response)",
    icon: BrainCircuit,
  },
  {
    name: "Validation",
    detail: "Cell lines + proteomics vs the models",
    icon: FlaskConical,
  },
  {
    name: "ODE digital twin",
    detail: "Mechanistic tumour–immune simulation",
    icon: Waves,
  },
  {
    name: "Resistance",
    detail: "Escape routes and salvage targets",
    icon: ShieldAlert,
  },
  {
    name: "Integration",
    detail: "All four methods → one recommendation",
    icon: Layers,
    isDestination: true,
  },
];

export default function MethodsStrip() {
  return (
    // One per row on phones, widening to all five in a line on large screens.
    <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
      {METHODS.map((m, i) => {
        const { icon: Icon } = m;
        return (
          <div key={m.name} className="flex items-stretch gap-1.5">
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
                  size={14}
                  className={
                    "shrink-0 " +
                    (m.isDestination ? "text-clinical-tealdark" : "text-clinical-muted")
                  }
                />
                <span className="text-[12.5px] font-bold leading-tight text-clinical-ink">
                  {m.name}
                </span>
                {m.isDestination && (
                  <span className="ml-auto shrink-0 whitespace-nowrap rounded-full bg-clinical-tealdark px-1.5 py-px text-[9px] font-bold uppercase tracking-wide text-white">
                    You are here
                  </span>
                )}
              </div>
              <div className="mt-1 text-[11px] leading-snug text-clinical-muted">{m.detail}</div>
            </div>

            {/* The connectors only make sense when the tiles are actually in a row. */}
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
