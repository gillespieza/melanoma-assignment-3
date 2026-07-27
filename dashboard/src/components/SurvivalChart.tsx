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
import type { PatientInput } from "../data/types";
import { buildSurvival } from "../lib/forecast";
import { KM_FACTS } from "../data/model";
import { Panel, Stat } from "./ui";
import { HeartPulse } from "lucide-react";

export default function SurvivalChart({ patient }: { patient: PatientInput }) {
  const { points, recMedian, altMedian } = buildSurvival(patient);

  return (
    <Panel
      title="Predicted Overall Survival"
      subtitle="Kaplan–Meier projection · recommended vs next-best lane"
      icon={<HeartPulse size={16} />}
    >
      <div className="mb-3 grid grid-cols-2 gap-2.5">
        <Stat label="Median OS · recommended" value={recMedian} unit="mo" tone="teal" />
        <Stat label="Median OS · alternative" value={altMedian} unit="mo" tone="blue" />
      </div>

      <div className="h-[210px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={points} margin={{ top: 6, right: 10, left: -10, bottom: 4 }}>
            <defs>
              <linearGradient id="recFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#0f766e" stopOpacity={0.28} />
                <stop offset="100%" stopColor="#0f766e" stopOpacity={0.02} />
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
              width={40}
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
            <Area
              type="monotone"
              dataKey="recommended"
              name="Recommended"
              stroke="#0f766e"
              strokeWidth={3}
              fill="url(#recFill)"
            />
            <Area
              type="monotone"
              dataKey="alternative"
              name="Alternative"
              stroke="#2563eb"
              strokeWidth={2}
              strokeDasharray="5 4"
              fill="transparent"
            />
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
