import { Layers, Target, Activity, ShieldAlert, Award, Compass, HelpCircle } from "lucide-react";
import type { CohortMeta, CohortPatient } from "../data/cohort";
import { getPhenotypeColor, getArmColor } from "../data/palette";
import { getPhenotypeDescription } from "./PatientPassport";
import { Panel, Pill } from "./ui";

interface Props {
  patient: CohortPatient;
  meta: CohortMeta;
}

export default function Q5PhenotypePanel({ patient, meta }: Props) {
  const q5 = patient.q5;

  if (!q5) {
    return (
      <Panel
        title="Q5 Patient Stratification"
        subtitle="Two-Stage GMM & Treatability Index"
        icon={<Layers size={16} />}
      >
        <div className="py-8 text-center text-clinical-muted">
          No Q5 stratification data available for this patient.
        </div>
      </Panel>
    );
  }

  const phenoColor = getPhenotypeColor(q5.shortLabel);
  const armColor = getArmColor(q5.treatmentArm);
  const odeEndpoints = meta.q5OdeTrajectory?.[q5.shortLabel];

  const probs = [
    { key: "Immune Hot", val: q5.probabilities.immuneHot, color: getPhenotypeColor("Immune Hot") },
    { key: "Immune Cold", val: q5.probabilities.immuneCold, color: getPhenotypeColor("Immune Cold") },
    { key: "M2-High", val: q5.probabilities.m2High, color: getPhenotypeColor("M2-High") },
    { key: "Mutant-Driven", val: q5.probabilities.mutantDriven, color: getPhenotypeColor("Mutant-Driven") },
  ];

  return (
    <div className="space-y-4">
      {/* Top Banner: Stratification Summary */}
      <Panel
        title="Q5 Patient Stratification Engine"
        subtitle="Two-Stage GMM Phenotyping & Treatability Index"
        icon={<Layers size={16} />}
        right={
          <Pill tone={q5.confidenceBand === "High" ? "green" : q5.confidenceBand === "Moderate" ? "amber" : "neutral"}>
            <Award size={11} /> {q5.confidenceBand} Confidence
          </Pill>
        }
      >
        <div className="grid gap-4 md:grid-cols-3">
          {/* Assigned Phenotype Tile */}
          <div className="rounded-xl border border-clinical-border bg-clinical-bg p-4 flex flex-col justify-between">
            <div>
              <div className="text-[10px] font-bold uppercase tracking-wide text-clinical-muted flex items-center justify-between">
                <span className="flex items-center gap-1">
                  <Compass size={12} /> Assigned Phenotype
                </span>
                <div className="group relative cursor-help">
                  <HelpCircle size={11} className="text-clinical-muted group-hover:text-okabe-purple transition" />
                  <div className="pointer-events-none absolute left-0 top-full z-30 mt-1 hidden w-72 rounded-xl border border-clinical-border bg-white p-3 text-left text-[11px] font-normal normal-case leading-snug text-clinical-muted shadow-lift group-hover:block">
                    <div className="font-bold text-clinical-ink">{q5.shortLabel} Phenotype</div>
                    <div className="mt-1 text-[11px] text-clinical-muted">{getPhenotypeDescription(q5.shortLabel)}</div>
                  </div>
                </div>
              </div>
              <div className="mt-2 flex items-center gap-2">
                <span
                  className="h-3.5 w-3.5 rounded-full shrink-0"
                  style={{ backgroundColor: phenoColor }}
                />
                <span className="text-[17px] font-extrabold tracking-tight text-clinical-ink">
                  {q5.shortLabel}
                </span>
              </div>
              <p className="mt-1 text-[11.5px] leading-snug text-clinical-muted">
                {q5.label}
              </p>
            </div>
            <div className="mt-3 text-[11px] font-bold text-clinical-muted">
              Subgroup Cluster #{q5.clusterId >= 0 ? q5.clusterId : "N/A"}
            </div>
          </div>

          {/* Treatability Index Tile */}
          <div className="rounded-xl border border-clinical-border bg-clinical-bg p-4 flex flex-col justify-between">
            <div>
              <div className="text-[10px] font-bold uppercase tracking-wide text-clinical-muted flex items-center gap-1">
                <Activity size={12} /> Treatability Index
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-[28px] font-extrabold leading-none text-clinical-ink">
                  {q5.treatabilityIndex !== null ? q5.treatabilityIndex.toFixed(0) : "N/A"}
                </span>
                <span className="text-[12px] font-bold text-clinical-muted">/ 100</span>
              </div>
              <p className="mt-1 text-[11.5px] leading-snug text-clinical-muted">
                Quantile-scaled composite of antigen presentation, IFN-γ, and microenvironment barriers.
              </p>
            </div>
            {q5.dabrafenibSensitivity !== null && (
              <div className="mt-2 text-[11px] font-bold text-clinical-muted">
                Q2 Dabrafenib Sensitivity: {q5.dabrafenibSensitivity.toFixed(1)}/100
              </div>
            )}
          </div>

          {/* Recommended Arm & Therapy */}
          <div className="rounded-xl border border-clinical-border bg-clinical-bg p-4 flex flex-col justify-between">
            <div>
              <div className="text-[10px] font-bold uppercase tracking-wide text-clinical-muted flex items-center gap-1">
                <Target size={12} /> Primary Treatment Arm
              </div>
              <div className="mt-2 flex items-center gap-2">
                <span
                  className="rounded px-2 py-0.5 text-[12px] font-extrabold text-white"
                  style={{ backgroundColor: armColor }}
                >
                  Arm {q5.treatmentArm}
                </span>
              </div>
              <p className="mt-1.5 text-[12.5px] font-bold leading-snug text-clinical-ink">
                {q5.recommendedTherapy || "Standard of care protocol"}
              </p>
            </div>
            {q5.q4NominatedTarget && q5.q4NominatedTarget !== "N/A (Arm A Candidate)" && (
              <div className="mt-2 text-[11px] font-bold text-clinical-rose flex items-center gap-1">
                <ShieldAlert size={12} /> Target: {q5.q4NominatedTarget}
              </div>
            )}
          </div>
        </div>
      </Panel>

      {/* GMM Soft Membership Probabilities */}
      <Panel
        title="GMM Cluster Membership Probabilities"
        subtitle="Soft-weighted phenotype assignment across the 4 subgroups"
      >
        <div className="space-y-3">
          {/* Stacked bar */}
          <div className="flex h-3.5 w-full overflow-hidden rounded-full border border-clinical-border bg-clinical-bg">
            {probs.map((p) =>
              p.val > 0 ? (
                <div
                  key={p.key}
                  style={{ width: `${(p.val * 100).toFixed(1)}%`, backgroundColor: p.color }}
                  title={`${p.key}: ${(p.val * 100).toFixed(1)}%`}
                />
              ) : null
            )}
          </div>

          {/* Legend Grid */}
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 pt-1">
            {probs.map((p) => (
              <div key={p.key} className="flex items-center justify-between rounded-lg border border-clinical-border bg-clinical-bg px-3 py-2">
                <div className="flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ backgroundColor: p.color }} />
                  <span className="text-[11.5px] font-bold text-clinical-ink">{p.key}</span>
                </div>
                <span className="text-[12px] font-bold text-clinical-muted">
                  {(p.val * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </Panel>

      {/* Phenotype ODE Trajectory Summary (if available) */}
      {odeEndpoints && (
        <Panel
          title={`Phase 4 ODE Dynamic Trajectories – ${q5.shortLabel}`}
          subtitle="Modelled endpoint tumour burden across therapy arms for this phenotype"
        >
          <div className="grid grid-cols-3 gap-3 text-center">
            <div className="rounded-xl border border-clinical-border bg-clinical-bg p-3">
              <div className="text-[10px] font-bold uppercase tracking-wide text-clinical-muted">
                Anti-PD-1 Monotherapy
              </div>
              <div className="mt-1 text-[20px] font-extrabold text-okabe-purple-dark">
                {(odeEndpoints.immuno_mono * 100).toFixed(1)}%
              </div>
              <div className="text-[10.5px] text-clinical-muted mt-0.5">Residual burden</div>
            </div>
            <div className="rounded-xl border border-clinical-border bg-clinical-bg p-3">
              <div className="text-[10px] font-bold uppercase tracking-wide text-clinical-muted">
                Anti-PD-1 + Rescue Combo
              </div>
              <div className="mt-1 text-[20px] font-extrabold text-green-700">
                {(odeEndpoints.immuno_rescue * 100).toFixed(1)}%
              </div>
              <div className="text-[10.5px] text-clinical-muted mt-0.5">Residual burden</div>
            </div>
            <div className="rounded-xl border border-clinical-border bg-clinical-bg p-3">
              <div className="text-[10px] font-bold uppercase tracking-wide text-clinical-muted">
                Targeted BRAFi (500nM)
              </div>
              <div className="mt-1 text-[20px] font-extrabold text-amber-700">
                {(odeEndpoints.targeted * 100).toFixed(1)}%
              </div>
              <div className="text-[10.5px] text-clinical-muted mt-0.5">Residual burden</div>
            </div>
          </div>
        </Panel>
      )}
    </div>
  );
}
