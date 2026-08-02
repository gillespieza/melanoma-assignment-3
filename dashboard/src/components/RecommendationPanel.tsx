import type { RankedOption } from "../data/types";
import { Panel, Pill } from "./ui";
import { Stethoscope, TrendingUp, ShieldAlert, BookOpen, Check, Ban, Sparkles } from "lucide-react";
import { TreatabilityHelpPopover } from "./TreatabilityHelpPopover";

function tierPill(opt: RankedOption) {
  if (opt.hardBlocked) return <Pill tone="rose"><Ban size={11} className="inline mr-1" />Contraindicated</Pill>;
  if (opt.tier === "primary") return <Pill tone="okabe-purple">Primary recommendation</Pill>;
  if (opt.tier === "alternative") return <Pill tone="blue">Alternative</Pill>;
  return <Pill tone="rose">Not recommended</Pill>;
}

function ConfidenceBar({ value, muted }: { value: number; muted?: boolean }) {
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-clinical-bg">
      <div
        className={"h-full rounded-full " + (muted ? "bg-clinical-border" : "bg-okabe-purple-dark")}
        style={{ width: `${value}%` }}
      />
    </div>
  );
}

function OptionCard({
  opt,
  selected,
  onSelect,
}: {
  opt: RankedOption;
  selected: boolean;
  onSelect: () => void;
}) {
  const isNo = opt.tier === "not-recommended" || opt.hardBlocked;
  return (
    <div
      className={
        "rounded-2xl border p-4 transition " +
        (opt.tier === "primary"
          ? "border-okabe-purple-dark bg-okabe-purple/[0.04] shadow-card"
          : isNo
            ? "border-red-200/80 bg-red-50/30 opacity-85"
            : "border-clinical-border bg-white")
      }
    >
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex items-center gap-2">
            {tierPill(opt)}
            {selected && (
              <Pill tone="green">
                <Check size={11} /> Confirmed plan
              </Pill>
            )}
          </div>
          <h3 className="text-[15px] font-extrabold tracking-tight text-clinical-ink">
            {opt.arm.label}
          </h3>
          <div className="text-[12px] font-semibold text-clinical-muted">{opt.arm.regimen}</div>
        </div>
        <div className="text-right shrink-0">
          <div className="flex items-center justify-end gap-1 text-[10.5px] font-semibold uppercase tracking-wide text-clinical-muted">
            <span>Confidence</span>
          </div>
          <div
            className={
              "tabular text-[26px] font-extrabold leading-none " +
              (isNo ? "text-clinical-muted" : "text-okabe-purple-dark")
            }
          >
            {opt.confidence}
            <span className="text-[13px]">/100</span>
          </div>
          {opt.tiScore != null && (
            <div className="mt-1 text-[10px] text-clinical-muted text-right leading-tight">
              TI&nbsp;{opt.tiScore.toFixed(0)}/100
              <TreatabilityHelpPopover />
            </div>
          )}
        </div>
      </div>

      <div className="mt-2.5">
        <ConfidenceBar value={opt.confidence} muted={isNo} />
      </div>

      {!isNo && (
        <div className="mt-3 grid grid-cols-2 gap-2">
          <div className="rounded-lg border border-clinical-border bg-white px-3 py-2">
            <div className="flex items-center gap-1 text-[10px] font-semibold uppercase text-clinical-muted">
              <TrendingUp size={11} /> Median OS
            </div>
            <div className="tabular text-[16px] font-extrabold text-clinical-ink">
              {opt.medianOsMonths} <span className="text-[11px] font-semibold">mo</span>
            </div>
          </div>
          <div className="rounded-lg border border-clinical-border bg-white px-3 py-2">
            <div className="flex items-center gap-1 text-[10px] font-semibold uppercase text-clinical-muted">
              <Stethoscope size={11} /> 12-mo burden ↓
            </div>
            <div className="tabular text-[16px] font-extrabold text-clinical-ink">
              {Math.round(opt.burdenReduction * 100)}%
            </div>
          </div>
        </div>
      )}

      <p className="mt-3 text-[12.5px] leading-snug text-clinical-ink">{opt.rationale}</p>

      {opt.reasoningChain && opt.reasoningChain.length > 0 && (
        <div className="mt-2.5 rounded-lg border border-okabe-purple/30 bg-okabe-purple/[0.06] p-2.5 space-y-1 text-[11.5px] leading-snug">
          <div className="font-bold text-[10px] uppercase tracking-wide text-okabe-purple-dark flex items-center gap-1">
            <Sparkles size={11} /> Clinical Reasoning Summary:
          </div>
          {opt.reasoningChain.map((step, idx) => (
            <div
              key={idx}
              className={
                step.startsWith("→")
                  ? "font-semibold text-okabe-purple-dark pl-1.5"
                  : "text-clinical-ink font-medium"
              }
            >
              {step}
            </div>
          ))}
        </div>
      )}

      {opt.contraindication && (
        <div className="mt-2.5 flex items-start gap-1.5 rounded-lg border border-red-200 bg-red-50 p-2.5 text-[11.5px] font-medium leading-snug text-red-800">
          <Ban size={14} className="mt-0.5 shrink-0 text-red-600" />
          <span>{opt.contraindication}</span>
        </div>
      )}

      <div className="mt-2.5 space-y-1.5">
        <div className="flex items-start gap-1.5 text-[11.5px] text-clinical-muted">
          <BookOpen size={13} className="mt-0.5 shrink-0 text-clinical-bluedark" />
          <span>{opt.evidence}</span>
        </div>
        <div className="flex items-start gap-1.5 text-[11.5px] text-clinical-muted">
          <ShieldAlert size={13} className="mt-0.5 shrink-0 text-amber-600" />
          <span>{opt.caution}</span>
        </div>
      </div>

      {!isNo ? (
        <button
          onClick={onSelect}
          className={
            "mt-3.5 flex w-full items-center justify-center gap-1.5 rounded-lg py-2 text-[12.5px] font-bold transition " +
            (selected
              ? "bg-clinical-green text-white"
              : "border border-okabe-purple-dark text-okabe-purple-dark hover:bg-okabe-purple/10")
          }
        >
          {selected ? (
            <>
              <Check size={14} /> Selected by consultant
            </>
          ) : (
            "Select & confirm this plan"
          )}
        </button>
      ) : (
        <div className="mt-3.5 flex items-center justify-center gap-1.5 rounded-lg border border-clinical-border py-2 text-[12px] font-semibold text-clinical-muted">
          <Ban size={13} /> Ineligible for this patient
        </div>
      )}
    </div>
  );
}

export default function RecommendationPanel({
  options,
  selectedKey,
  onSelect,
}: {
  options: RankedOption[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
}) {
  return (
    <Panel
      title="Ranked Treatment Options"
      icon={<Stethoscope size={16} />}
    >
      <div className="grid gap-3 lg:grid-cols-3">
        {options.map((opt) => (
          <OptionCard
            key={opt.arm.key}
            opt={opt}
            selected={selectedKey === opt.arm.key}
            onSelect={() => onSelect(opt.arm.key)}
          />
        ))}
      </div>
    </Panel>
  );
}
