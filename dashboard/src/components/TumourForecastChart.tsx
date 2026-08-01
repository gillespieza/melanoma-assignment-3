import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Legend,
} from "recharts";
import type { Forecast } from "../lib/forecast";
import { Panel, Pill } from "./ui";
import { LineChart as LineIcon } from "lucide-react";

export default function TumourForecastChart({ forecast }: { forecast: Forecast }) {
  const { data, hasTargeted, hasCombo } = forecast;

  return (
    <Panel
      title="Tumour Burden Forecast"
      subtitle="Projected tumour burden (% of baseline) over 12 months, per therapy lane"
      icon={<LineIcon size={16} />}
      right={<Pill tone="okabe-purple">Anchored to ODE dose-response</Pill>}
    >
      <div className="h-[290px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, left: -8, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#eef2f6" />
            <XAxis
              dataKey="month"
              tick={{ fontSize: 11, fill: "#5b6b7c" }}
              tickLine={false}
              axisLine={{ stroke: "#e6ebf1" }}
              label={{
                value: "Months on treatment",
                position: "insideBottom",
                offset: -2,
                style: { fontSize: 11, fill: "#5b6b7c" },
              }}
            />
            <YAxis
              domain={[0, 200]}
              tick={{ fontSize: 11, fill: "#5b6b7c" }}
              tickLine={false}
              axisLine={false}
              width={44}
              label={{
                value: "% baseline burden",
                angle: -90,
                position: "insideLeft",
                offset: 16,
                style: { fontSize: 11, fill: "#5b6b7c", textAnchor: "middle" },
              }}
            />
            <ReferenceLine y={100} stroke="#cbd5e1" strokeDasharray="4 4" />
            <Tooltip
              contentStyle={{
                borderRadius: 12,
                border: "1px solid #e6ebf1",
                fontSize: 12,
                boxShadow: "0 8px 24px rgba(16,32,51,0.10)",
              }}
              formatter={(v: number) => `${v}%`}
              labelFormatter={(m) => `Month ${m}`}
            />
            <Legend wrapperStyle={{ fontSize: 12, paddingTop: 6 }} />
            <Line isAnimationActive={false}
              type="monotone"
              dataKey="baseline"
              name="No systemic therapy"
              stroke="#94a3b8"
              strokeDasharray="5 4"
              strokeWidth={2}
              dot={false}
            />
            <Line isAnimationActive={false}
              type="monotone"
              dataKey="immuno"
              name="Immunotherapy (anti-PD-1)"
              stroke="#CC79A7"
              strokeWidth={3}
              dot={false}
            />
            {hasTargeted && (
              <Line isAnimationActive={false}
                type="monotone"
                dataKey="targeted"
                name="Targeted (BRAF/MEK)"
                stroke="#2563eb"
                strokeWidth={3}
                dot={false}
                connectNulls
              />
            )}
            {hasCombo && (
              <Line isAnimationActive={false}
                type="monotone"
                dataKey="combo"
                name="Combination / sequencing"
                stroke="#d97706"
                strokeWidth={2.5}
                strokeDasharray="6 3"
                dot={false}
                connectNulls
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-2 text-[11px] leading-snug text-clinical-muted">
        Endpoints anchored to the q3 ODE tumour-burden sweep; month-by-month dynamics encode the
        known pattern of fast BRAF/MEK nadir with resistance rebound vs slower, durable checkpoint
        response.
      </p>
    </Panel>
  );
}
