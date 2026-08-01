import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import type { Survival } from "../lib/forecast";
import { KM_FACTS } from "../data/model";
import { Panel, Stat } from "./ui";
import { HeartPulse } from "lucide-react";

export default function SurvivalChart({ survival }: { survival: Survival }) {
  const { points, recMedian, altMedian, altLabel } = survival;
  const hasAlt = altMedian !== null;

  return (
    <Panel
      title="Predicted Overall Survival"
      subtitle={
        hasAlt
          ? "Kaplan–Meier projection · recommended vs next-best lane"
          : "Kaplan–Meier projection · only one lane is eligible for this patient"
      }
      icon={<HeartPulse size={16} />}
    >
      <div className="mb-3 grid grid-cols-2 gap-2.5">
        <Stat label="Median OS · recommended" value={recMedian} unit="mo" tone="okabe-purple" />
        {hasAlt ? (
          <Stat
            label={`Median OS · ${altLabel ?? "alternative"}`}
            value={altMedian}
            unit="mo"
            tone="blue"
          />
        ) : (
          <div className="rounded-xl border border-dashed border-clinical-border bg-clinical-bg px-3.5 py-2.5">
            <div className="text-[10.5px] font-semibold uppercase tracking-wide text-clinical-muted">
              Alternative lane
            </div>
            <div className="text-[13px] font-bold leading-tight text-clinical-muted">
              None eligible
            </div>
          </div>
        )}
      </div>

      <div className="h-[210px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={points} margin={{ top: 6, right: 10, left: -10, bottom: 4 }}>
            <defs>
              <linearGradient id="recFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#CC79A7" stopOpacity={0.28} />
                <stop offset="100%" stopColor="#CC79A7" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#eef2f6" />
            <XAxis
              dataKey="month"
              tick={{ fontSize: 11, fill: "#5b6b7c" }}
              tickLine={false}
              axisLine={{ stroke: "#e6ebf1" }}
              label={{
                value: "Months",
                position: "insideBottom",
                offset: -2,
                style: { fontSize: 11, fill: "#5b6b7c" },
              }}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fontSize: 11, fill: "#5b6b7c" }}
              tickLine={false}
              axisLine={false}
              width={48}
              tickFormatter={(v) => `${v}%`}
            />
            <ReferenceLine y={50} stroke="#cbd5e1" strokeDasharray="4 4" />
            <Tooltip
              contentStyle={{
                borderRadius: 12,
                border: "1px solid #e6ebf1",
                fontSize: 12,
              }}
              formatter={(v: number) => `${v}% alive`}
              labelFormatter={(m) => `Month ${m}`}
            />
            <Area isAnimationActive={false}
              type="monotone"
              dataKey="recommended"
              name="Recommended"
              stroke="#CC79A7"
              strokeWidth={3}
              fill="url(#recFill)"
            />
            {hasAlt && (
              <Area
                isAnimationActive={false}
                type="monotone"
                dataKey="alternative"
                name={altLabel ?? "Alternative"}
                stroke="#2563eb"
                strokeWidth={2}
                strokeDasharray="5 4"
                fill="transparent"
                connectNulls
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <p className="mt-1 text-[11px] leading-snug text-clinical-muted">
        Calibrated to the q3 checkpoint KM signal (median OS {KM_FACTS.checkpoint.lowMedianOs} vs{" "}
        {KM_FACTS.checkpoint.highMedianOs} mo; log-rank p={KM_FACTS.checkpoint.logRankP}). Digital-twin
        discrimination AUC {KM_FACTS.digitalTwinAuc}.
      </p>
    </Panel>
  );
}
