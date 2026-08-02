import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LabelList,
  ReferenceLine,
} from "recharts";
import { Target, Gauge } from "lucide-react";
import type { Q1Validation } from "../data/cohort";
import { Panel, Pill, Stat } from "./ui";

// Q1 model accuracy — measured, not claimed.
//
// Computed at build time from prob_ensemble vs the REAL response labels of the
// held-out ICI trial cohorts (Liu 2019, Riaz 2017, Hugo 2016). This is the
// evidence that the ML lane is genuinely predictive.
//
// Palette: responders #0ea5a4 / non-responders #d97706 — validated for CVD
// separation; the sub-3:1 contrast on the teal is relieved by direct labels.

const RESPONDER = "#0ea5a4";
const NON_RESPONDER = "#d97706";
const INK = "#5b6b7c";
const GRID = "#eef2f6";

const TOOLTIP_STYLE = {
  borderRadius: 12,
  border: "1px solid #e6ebf1",
  fontSize: 12,
  boxShadow: "0 8px 24px rgba(16,32,51,0.10)",
} as const;

function aucTone(auc: number | null) {
  if (auc === null) return "neutral" as const;
  if (auc >= 0.7) return "green" as const;
  if (auc >= 0.6) return "teal" as const;
  return "amber" as const;
}

function ConfusionGrid({
  title,
  stats,
}: {
  title: string;
  stats: Q1Validation["atHalf"];
}) {
  const cell = (label: string, value: number, tone: string) => (
    <div className={"rounded-lg border px-3 py-2 " + tone}>
      <div className="text-[9.5px] font-bold uppercase tracking-wide opacity-80">{label}</div>
      <div className="tabular text-[18px] font-extrabold leading-tight">{value}</div>
    </div>
  );
  const rate = (v: number | null) => (v === null ? "—" : `${(v * 100).toFixed(0)}%`);

  return (
    <div className="rounded-xl border border-clinical-border bg-white p-3.5">
      <div className="flex items-baseline justify-between">
        <div className="text-[11.5px] font-bold text-clinical-ink">{title}</div>
        <div className="tabular text-[11px] font-semibold text-clinical-muted">
          threshold {stats.threshold.toFixed(2)}
        </div>
      </div>
      <div className="mt-2.5 grid grid-cols-2 gap-2">
        {cell("True responder", stats.tp, "border-clinical-teal/30 bg-clinical-teal/10 text-clinical-tealdark")}
        {cell("False positive", stats.fp, "border-amber-200 bg-amber-50 text-amber-700")}
        {cell("False negative", stats.fn, "border-amber-200 bg-amber-50 text-amber-700")}
        {cell("True non-responder", stats.tn, "border-clinical-border bg-clinical-bg text-clinical-ink")}
      </div>
      <div className="mt-2.5 grid grid-cols-3 gap-2 text-center">
        {[
          ["Accuracy", stats.accuracy],
          ["Sensitivity", stats.sensitivity],
          ["Specificity", stats.specificity],
        ].map(([label, value]) => (
          <div key={label as string}>
            <div className="text-[9.5px] font-bold uppercase tracking-wide text-clinical-muted">
              {label as string}
            </div>
            <div className="tabular text-[14px] font-extrabold text-clinical-ink">
              {rate(value as number | null)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Q1ValidationPanel({ validation }: { validation: Q1Validation }) {
  return (
    <Panel
      title="Q1 · Model Accuracy on Held-Out Trials"
      subtitle="Measured against real response outcomes in the ICI trial cohorts — not a claim, a result"
      icon={<Target size={16} />}
      right={<Pill tone={aucTone(validation.auc)}>AUC {validation.auc?.toFixed(3) ?? "—"}</Pill>}
    >
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
        {/* ---- left: headline numbers + per-cohort table ---- */}
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-2.5">
            <Stat label="Patients" value={validation.n} tone="ink" />
            <Stat label="Responders" value={validation.responders} tone="teal" />
            <Stat
              label="AUC"
              value={validation.auc?.toFixed(3) ?? "—"}
              tone={validation.auc && validation.auc >= 0.7 ? "teal" : "ink"}
            />
          </div>

          {/* Per-cohort breakdown — also serves as the chart's table view. */}
          <div className="overflow-hidden rounded-xl border border-clinical-border">
            <table className="w-full text-left">
              <thead className="bg-clinical-bg">
                <tr className="text-[10px] font-bold uppercase tracking-wide text-clinical-muted">
                  <th className="px-3 py-2">Cohort</th>
                  <th className="px-3 py-2 text-right">n</th>
                  <th className="px-3 py-2 text-right">Responders</th>
                  <th className="px-3 py-2 text-right">AUC</th>
                </tr>
              </thead>
              <tbody>
                {validation.byCohort.map((c) => (
                  <tr key={c.cohort} className="border-t border-clinical-border">
                    <td className="px-3 py-2 text-[12.5px] font-bold text-clinical-ink">{c.cohort}</td>
                    <td className="tabular px-3 py-2 text-right text-[12.5px] text-clinical-ink">{c.n}</td>
                    <td className="tabular px-3 py-2 text-right text-[12.5px] text-clinical-ink">
                      {c.responders}
                    </td>
                    <td className="px-3 py-2 text-right">
                      <span
                        className={
                          "tabular text-[12.5px] font-extrabold " +
                          (c.auc && c.auc >= 0.7
                            ? "text-clinical-tealdark"
                            : c.auc && c.auc >= 0.6
                              ? "text-clinical-ink"
                              : "text-amber-700")
                        }
                      >
                        {c.auc?.toFixed(3) ?? "—"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="grid gap-2.5 sm:grid-cols-2">
            <ConfusionGrid title="At the tuned threshold" stats={validation.atTuned} />
            <ConfusionGrid title="At the default 0.50" stats={validation.atHalf} />
          </div>
        </div>

        {/* ---- right: the two charts ---- */}
        <div className="space-y-4">
          <div className="rounded-xl border border-clinical-border bg-white p-3.5">
            <div className="mb-1 flex items-center gap-1.5 text-[11.5px] font-bold text-clinical-ink">
              <Gauge size={13} className="text-clinical-tealdark" /> Predicted probability vs true
              outcome
            </div>
            <p className="mb-2 text-[11px] leading-snug text-clinical-muted">
              Responders should sit to the right of non-responders.
            </p>
            <div className="h-[196px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={validation.distribution}
                  margin={{ top: 12, right: 8, left: -14, bottom: 2 }}
                  barCategoryGap="18%"
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
                  <XAxis
                    dataKey="bin"
                    tick={{ fontSize: 9.5, fill: INK }}
                    tickLine={false}
                    axisLine={{ stroke: "#e6ebf1" }}
                    interval={1}
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: INK }}
                    tickLine={false}
                    axisLine={false}
                    width={38}
                    allowDecimals={false}
                  />
                  <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ fill: "rgba(15,32,51,0.04)" }} />
                  <Legend wrapperStyle={{ fontSize: 11, paddingTop: 4 }} iconType="circle" iconSize={8} />
                  <Bar isAnimationActive={false} dataKey="responders" name="Responders" fill={RESPONDER} radius={[4, 4, 0, 0]}>
                    <LabelList
                      dataKey="responders"
                      position="top"
                      formatter={(v: number) => (v > 0 ? v : "")}
                      style={{ fontSize: 9.5, fill: INK, fontWeight: 700 }}
                    />
                  </Bar>
                  <Bar isAnimationActive={false}
                    dataKey="nonResponders"
                    name="Non-responders"
                    fill={NON_RESPONDER}
                    radius={[4, 4, 0, 0]}
                  >
                    <LabelList
                      dataKey="nonResponders"
                      position="top"
                      formatter={(v: number) => (v > 0 ? v : "")}
                      style={{ fontSize: 9.5, fill: INK, fontWeight: 700 }}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-xl border border-clinical-border bg-white p-3.5">
            <div className="mb-2 text-[11.5px] font-bold text-clinical-ink">
              ROC curve · ensemble model
            </div>
            <div className="h-[176px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={validation.roc} margin={{ top: 6, right: 10, left: -16, bottom: 2 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
                  <XAxis
                    dataKey="fpr"
                    type="number"
                    domain={[0, 1]}
                    tick={{ fontSize: 10.5, fill: INK }}
                    tickLine={false}
                    axisLine={{ stroke: "#e6ebf1" }}
                    tickFormatter={(v) => v.toFixed(1)}
                    label={{
                      value: "False-positive rate",
                      position: "insideBottom",
                      offset: -1,
                      style: { fontSize: 10.5, fill: INK },
                    }}
                  />
                  <YAxis
                    dataKey="tpr"
                    type="number"
                    domain={[0, 1]}
                    tick={{ fontSize: 10.5, fill: INK }}
                    tickLine={false}
                    axisLine={false}
                    width={42}
                    tickFormatter={(v) => v.toFixed(1)}
                  />
                  <ReferenceLine
                    segment={[
                      { x: 0, y: 0 },
                      { x: 1, y: 1 },
                    ]}
                    stroke="#cbd5e1"
                    strokeDasharray="4 4"
                  />
                  <Tooltip
                    contentStyle={TOOLTIP_STYLE}
                    formatter={(v: number) => (v as number).toFixed(3)}
                    labelFormatter={(v) => `FPR ${Number(v).toFixed(3)}`}
                  />
                  <Line isAnimationActive={false}
                    type="monotone"
                    dataKey="tpr"
                    name="Sensitivity"
                    stroke="#0f766e"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-1.5 text-[11px] leading-snug text-clinical-muted">
              Dashed diagonal = chance. Area under this curve is {validation.auc?.toFixed(3) ?? "—"}.
            </p>
          </div>
        </div>
      </div>

      <p className="mt-4 border-t border-clinical-border pt-3 text-[11.5px] leading-relaxed text-clinical-muted">
        <span className="font-bold text-clinical-ink">Reading this honestly:</span> discrimination
        varies by cohort — strong in Riaz 2017 (AUC{" "}
        {validation.byCohort.find((c) => c.cohort.startsWith("Riaz"))?.auc?.toFixed(3) ?? "—"}),
        moderate in Liu 2019 (AUC{" "}
        {validation.byCohort.find((c) => c.cohort.startsWith("Liu"))?.auc?.toFixed(3) ?? "—"}), and
        around chance in the small Hugo 2016 set (n=
        {validation.byCohort.find((c) => c.cohort.startsWith("Hugo"))?.n ?? "—"}). The raw probabilities
        cluster tightly around ~0.45 — so the tuned threshold ({validation.atTuned.threshold.toFixed(3)})
        matters more than the default 0.50 cut-off. The relative ranking carries the signal; the absolute
        number should be interpreted in cohort-relative terms.
      </p>
    </Panel>
  );
}
