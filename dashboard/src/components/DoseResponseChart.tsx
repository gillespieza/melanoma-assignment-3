import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";
import { Waves, CircleAlert } from "lucide-react";
import type { CohortPatient } from "../data/cohort";
import { Panel, Pill, Stat } from "./ui";

// Q3 lane · the patient's OWN simulated dose-response sweep.
//
// This is raw digital-twin output – the tumour burden the ODE predicts at each
// drug dose, as a percentage of that patient's untreated baseline. Arm colours
// follow the locked design system: targeted = blue, immunotherapy = okabe-purple.

const INK = "#5b6b7c";

export default function DoseResponseChart({
  patient,
  doseAxis,
}: {
  patient: CohortPatient;
  doseAxis: number[];
}) {
  const anyInformative = patient.brafiInformative || patient.antipd1Informative;

  const data = doseAxis.map((dose, i) => ({
    dose,
    doseLabel: `${Math.round(dose * 100)}%`,
    brafi: patient.brafiCurve ? patient.brafiCurve[i] : null,
    antipd1: patient.antipd1Curve ? patient.antipd1Curve[i] : null,
  }));

  return (
    <Panel
      title="Q3 · ODE Digital Twin – Dose Response"
      subtitle="Simulated tumour burden across the drug-dose sweep, for this patient specifically"
      icon={<Waves size={16} />}
      right={<Pill tone="okabe-purple">Raw model output</Pill>}
    >
      {!anyInformative ? (
        <div className="rounded-xl border border-dashed border-clinical-border bg-clinical-bg/60 p-5">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white text-amber-600 shadow-card">
              <CircleAlert size={18} />
            </div>
            <div>
              <h3 className="text-[14px] font-extrabold text-clinical-ink">
                Twin below model resolution
              </h3>
              <p className="mt-1.5 max-w-2xl text-[12.5px] leading-relaxed text-clinical-muted">
                For this patient the ODE settles at a numerically-zero tumour compartment, so the
                simulated arms carry no usable signal. This is <span className="font-bold">not</span>{" "}
                a prediction of drug resistance – the model simply has nothing to say here, and the
                recommendation falls back to the statistical and molecular evidence alone.
              </p>
              <p className="mt-2 text-[11.5px] text-clinical-muted">
                {patient.brafiInformative || patient.antipd1Informative
                  ? ""
                  : `Baseline burden: BRAFi arm ${patient.brafiBaseline.toExponential(2)}, anti-PD-1 arm ${patient.antipd1Baseline.toExponential(2)} (floor 1e-3).`}
              </p>
            </div>
          </div>
        </div>
      ) : (
        <>
          <div className="mb-3 grid grid-cols-2 gap-2.5 sm:grid-cols-4">
            <Stat
              label="BRAFi reduction"
              value={patient.brafiInformative ? `${Math.round(patient.brafiReduction * 100)}` : "–"}
              unit={patient.brafiInformative ? "%" : undefined}
              tone="blue"
            />
            <Stat
              label="Anti-PD-1 reduction"
              value={
                patient.antipd1Informative ? `${Math.round(patient.antipd1Reduction * 100)}` : "–"
              }
              unit={patient.antipd1Informative ? "%" : undefined}
              tone="okabe-purple"
            />
            <Stat
              label="Optimal BRAFi dose"
              value={patient.brafiOptimalDose !== null ? patient.brafiOptimalDose.toFixed(2) : "–"}
            />
            <Stat
              label="Optimal anti-PD-1"
              value={
                patient.antipd1OptimalDose !== null ? patient.antipd1OptimalDose.toFixed(2) : "–"
              }
            />
          </div>

          <div className="h-[262px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ top: 8, right: 12, left: 18, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eef2f6" />
                <XAxis
                  dataKey="doseLabel"
                  tick={{ fontSize: 11, fill: INK }}
                  tickLine={false}
                  axisLine={{ stroke: "#e6ebf1" }}
                  label={{
                    value: "Drug dose (% of maximum)",
                    position: "insideBottom",
                    offset: -2,
                    style: { fontSize: 11, fill: INK },
                  }}
                />
                <YAxis
                  domain={[0, "auto"]}
                  tick={{ fontSize: 11, fill: INK }}
                  tickLine={false}
                  axisLine={false}
                  width={64}
                  tickFormatter={(v) => `${v}%`}
                  label={{
                    value: "% of untreated baseline",
                    angle: -90,
                    position: "insideLeft",
                    offset: 6,
                    style: { fontSize: 11, fill: INK, textAnchor: "middle" },
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
                  formatter={(v: number) => `${v.toFixed(1)}% of baseline`}
                  labelFormatter={(d) => `Dose ${d}`}
                />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 6 }} iconType="line" />
                {patient.brafiInformative && (
                  <Line isAnimationActive={false}
                    type="monotone"
                    dataKey="brafi"
                    name="BRAF inhibitor"
                    stroke="#2563eb"
                    strokeWidth={3}
                    dot={{ r: 3, strokeWidth: 0, fill: "#2563eb" }}
                    connectNulls
                  />
                )}
                {patient.antipd1Informative && (
                  <Line isAnimationActive={false}
                    type="monotone"
                    dataKey="antipd1"
                    name="Anti-PD-1"
                    stroke="#CC79A7"
                    strokeWidth={3}
                    dot={{ r: 3, strokeWidth: 0, fill: "#CC79A7" }}
                    connectNulls
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <p className="mt-2 text-[11.5px] leading-snug text-clinical-muted">
            {patient.braf === "WT" && patient.brafiInformative
              ? "Note the flat – or rising – BRAF-inhibitor curve: this is the RAF paradox, which the ODE reproduces mechanistically in BRAF wild-type tumours."
              : "Each curve is this patient's own simulation, normalised to their untreated baseline so the two arms are directly comparable."}
          </p>
        </>
      )}
    </Panel>
  );
}
