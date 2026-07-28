import { BrainCircuit, Waves, ArrowRight, CircleAlert, CircleCheck, CircleMinus } from "lucide-react";
import type { AgreementResult, MethodPosition } from "../lib/integrationEngine";

// ---------------------------------------------------------------------------
// The methods-agreement signal — the centrepiece of the Q5 lane.
//
// Two independent methods scored this patient. This makes their positions, and
// whether they converge, readable at a glance from the back of a lecture theatre.
// ---------------------------------------------------------------------------

const TONES = {
  concordant: {
    ring: "border-clinical-green/40 bg-green-50/60",
    chip: "bg-clinical-green text-white",
    accent: "text-green-700",
    bar: "bg-clinical-green",
    Icon: CircleCheck,
  },
  partial: {
    ring: "border-clinical-blue/35 bg-blue-50/50",
    chip: "bg-clinical-bluedark text-white",
    accent: "text-clinical-bluedark",
    bar: "bg-clinical-blue",
    Icon: CircleMinus,
  },
  discordant: {
    ring: "border-amber-300 bg-amber-50/70",
    chip: "bg-amber-600 text-white",
    accent: "text-amber-700",
    bar: "bg-amber-500",
    Icon: CircleAlert,
  },
  unavailable: {
    ring: "border-clinical-border bg-clinical-bg",
    chip: "bg-clinical-muted text-white",
    accent: "text-clinical-muted",
    bar: "bg-clinical-muted",
    Icon: CircleMinus,
  },
} as const;

function MethodCard({
  position,
  icon,
  kind,
}: {
  position: MethodPosition | null;
  icon: React.ReactNode;
  kind: "Statistical" | "Mechanistic";
}) {
  if (!position) {
    return (
      <div className="flex-1 rounded-xl border border-dashed border-clinical-border bg-white/60 p-3.5">
        <div className="flex items-center gap-1.5 text-[10.5px] font-bold uppercase tracking-wide text-clinical-muted">
          {icon} {kind}
        </div>
        <div className="mt-2 text-[13px] font-semibold text-clinical-muted">Not assessable</div>
        <p className="mt-1 text-[11.5px] leading-snug text-clinical-muted">
          This method could not score the patient.
        </p>
      </div>
    );
  }

  const favours = position.value >= 0.5;
  return (
    <div className="flex-1 rounded-xl border border-clinical-border bg-white p-3.5 shadow-card">
      <div className="flex items-center gap-1.5 text-[10.5px] font-bold uppercase tracking-wide text-clinical-muted">
        {icon} {kind}
      </div>
      <div className="mt-2 text-[13px] font-bold leading-tight text-clinical-ink">
        {position.label}
      </div>

      <div className="mt-2.5 flex items-center gap-2">
        <div className="h-2 flex-1 overflow-hidden rounded-full bg-clinical-bg">
          <div
            className={
              "h-full rounded-full transition-[width] duration-500 ease-out " +
              (favours ? "bg-clinical-tealdark" : "bg-clinical-muted")
            }
            style={{ width: `${Math.round(position.value * 100)}%` }}
          />
        </div>
        <span className="tabular w-9 text-right text-[12px] font-extrabold text-clinical-ink">
          {Math.round(position.value * 100)}
        </span>
      </div>

      <div
        className={
          "mt-2 text-[11px] font-bold uppercase tracking-wide " +
          (favours ? "text-clinical-tealdark" : "text-clinical-muted")
        }
      >
        {favours ? "Favours immunotherapy" : "Does not favour immunotherapy"}
      </div>
      <p className="mt-1.5 text-[11.5px] leading-snug text-clinical-muted">{position.detail}</p>
    </div>
  );
}

export default function AgreementBadge({ agreement }: { agreement: AgreementResult }) {
  const tone = TONES[agreement.status];
  const { Icon } = tone;

  return (
    <div className={"animate-fade-up rounded-2xl border p-4 shadow-card " + tone.ring}>
      <div className="flex flex-wrap items-center gap-3">
        <span
          className={
            "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[11.5px] font-bold " +
            tone.chip
          }
        >
          <Icon size={13} /> {agreement.headline}
        </span>

        {agreement.concordance !== null && (
          <div className="flex items-center gap-2">
            <span className="text-[10.5px] font-bold uppercase tracking-wide text-clinical-muted">
              Concordance
            </span>
            <span className={"tabular text-[20px] font-extrabold leading-none " + tone.accent}>
              {Math.round(agreement.concordance * 100)}
              <span className="text-[12px]">%</span>
            </span>
          </div>
        )}
      </div>

      <p className="mt-2.5 max-w-3xl text-[12.5px] leading-relaxed text-clinical-ink">
        {agreement.detail}
      </p>

      {/* The converging-methods visual */}
      <div className="mt-3.5 flex flex-col items-stretch gap-2.5 sm:flex-row sm:items-center">
        <MethodCard
          position={agreement.statistical}
          kind="Statistical"
          icon={<BrainCircuit size={13} className="text-clinical-bluedark" />}
        />
        <div className="flex shrink-0 items-center justify-center sm:flex-col">
          <ArrowRight size={18} className={tone.accent + " rotate-90 sm:rotate-0"} />
        </div>
        <MethodCard
          position={agreement.mechanistic}
          kind="Mechanistic"
          icon={<Waves size={13} className="text-clinical-tealdark" />}
        />
      </div>

      {agreement.statistical.source === "biomarker" && (
        <p className="mt-3 rounded-lg border border-clinical-border bg-white/70 px-3 py-2 text-[11.5px] leading-snug text-clinical-muted">
          <span className="font-bold text-clinical-ink">Provenance:</span> the statistical side is a
          measured biomarker composite, not the Q1 ML model — the Q1 models have not scored the TCGA
          cohort. This panel switches to real Q1 probabilities automatically once they are available.
        </p>
      )}
    </div>
  );
}
