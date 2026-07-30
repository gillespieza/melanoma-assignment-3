import { IdCard, Dna, Microscope, Syringe } from "lucide-react";
import type { CohortPatient } from "../data/cohort";
import { pdl1Band } from "../data/cohort";
import { Panel, Pill } from "./ui";

// ---------------------------------------------------------------------------
// Patient passport — who this patient is, in the two registers a melanoma MDT
// actually uses: demographics/staging and molecular profile.
// Every value here is real TCGA-SKCM data.
// ---------------------------------------------------------------------------

function Field({ label, value, muted }: { label: string; value: string; muted?: boolean }) {
  return (
    <div>
      <div className="text-[10px] font-bold uppercase tracking-wide text-clinical-muted">{label}</div>
      <div
        className={
          "tabular text-[13.5px] font-bold leading-tight " +
          (muted ? "text-clinical-muted" : "text-clinical-ink")
        }
      >
        {value}
      </div>
    </div>
  );
}

const CHECKPOINT_AGENTS = new Set(["Ipilimumab", "Pembrolizumab", "Nivolumab"]);
const TARGETED_AGENTS = new Set(["Vemurafenib", "Dabrafenib", "Trametinib"]);

function agentTone(agent: string): "teal" | "blue" | "neutral" {
  if (CHECKPOINT_AGENTS.has(agent)) return "teal";
  if (TARGETED_AGENTS.has(agent)) return "blue";
  return "neutral";
}

export default function PatientPassport({
  patient,
  modified = [],
}: {
  patient: CohortPatient;
  /** Fields the clinician has changed in the what-if explorer. */
  modified?: string[];
}) {
  const band = pdl1Band(patient.pdl1Pct);
  const brafMut = patient.braf !== "WT";

  return (
    <Panel
      title={patient.id}
      subtitle={
        modified.length
          ? `Hypothetical variant — ${modified.join(", ")} edited`
          : `TCGA-SKCM · sample ${patient.sampleId}`
      }
      icon={<IdCard size={16} />}
      right={
        <div className="flex flex-wrap items-center justify-end gap-1.5">
          {modified.length > 0 && <Pill tone="amber">Modified</Pill>}
          <Pill tone={brafMut ? "blue" : "neutral"}>
            <Dna size={11} /> BRAF {brafMut ? "V600" : "WT"}
          </Pill>
          <Pill tone={patient.nras === "Mutant" ? "amber" : "neutral"}>
            NRAS {patient.nras === "Mutant" ? "mut" : "WT"}
          </Pill>
          <Pill tone={band === "High" ? "teal" : band === "Intermediate" ? "neutral" : "rose"}>
            <Microscope size={11} /> PD-L1 {band.toLowerCase()}
          </Pill>
        </div>
      }
    >
      <div className="grid grid-cols-2 gap-x-4 gap-y-3.5 sm:grid-cols-4">
        <Field label="Age" value={patient.age !== null ? `${patient.age} y` : "Not recorded"} muted={patient.age === null} />
        <Field
          label="Sex"
          value={patient.sex === "M" ? "Male" : patient.sex === "F" ? "Female" : "Not recorded"}
          muted={patient.sex === null}
        />
        <Field label="Stage" value={patient.stage === "Unknown" ? "Not recorded" : patient.stage} muted={patient.stage === "Unknown"} />
        <Field
          label="Follow-up"
          value={
            patient.osMonths !== null
              ? `${patient.osMonths.toFixed(0)} mo ${patient.osEvent === 1 ? "(deceased)" : "(censored)"}`
              : "Not recorded"
          }
          muted={patient.osMonths === null}
        />

        <Field label="PD-L1 (CD274)" value={`${patient.pdl1Pct}th pct`} />
        <Field label="PD-1 (PDCD1)" value={`${patient.pdcd1Pct}th pct`} />
        <Field
          label="TMB"
          value={patient.tmb !== null ? `${patient.tmb.toFixed(1)} /Mb` : "Not recorded"}
          muted={patient.tmb === null}
        />
        <Field label="MAPK driven" value={patient.mapkDriven ? "Yes" : "No"} />
      </div>

      <div className="mt-4 border-t border-clinical-border pt-3.5">
        <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wide text-clinical-muted">
          <Syringe size={11} /> Treatment received
        </div>
        {patient.treatment.recorded ? (
          <>
            <div className="mt-1.5 flex flex-col gap-2">
              {patient.treatment.lines.map((line) => (
                <div key={line.type}>
                  <div className="text-[11.5px] font-bold text-clinical-ink">{line.type}</div>
                  {line.agents.length ? (
                    <div className="mt-1 flex flex-wrap gap-1.5">
                      {line.agents.map((a) => (
                        <Pill key={a} tone={agentTone(a)}>
                          {a}
                        </Pill>
                      ))}
                    </div>
                  ) : (
                    <div className="mt-1 text-[11px] text-clinical-muted">Agent not specified</div>
                  )}
                </div>
              ))}
            </div>
            {(patient.treatment.checkpointInhibitor || patient.treatment.targetedTherapy) && (
              <p className="mt-2 text-[11.5px] leading-snug text-clinical-muted">
                This patient actually received{" "}
                {patient.treatment.targetedTherapy && patient.treatment.checkpointInhibitor
                  ? "both a BRAF/MEK inhibitor and a checkpoint inhibitor"
                  : patient.treatment.targetedTherapy
                    ? "a BRAF/MEK inhibitor"
                    : "a checkpoint inhibitor"}{" "}
                — directly comparable to the {patient.treatment.targetedTherapy && patient.treatment.checkpointInhibitor
                  ? "BRAFi and anti-PD-1 arms"
                  : patient.treatment.targetedTherapy
                    ? "BRAFi arm"
                    : "anti-PD-1 arm"}{" "}
                the Q3 twin simulates below.
              </p>
            )}
            {!patient.treatment.checkpointInhibitor &&
              !patient.treatment.targetedTherapy &&
              (patient.treatment.chemotherapy || patient.treatment.radiation) && (
                <p className="mt-2 text-[11.5px] leading-snug text-clinical-muted">
                  Chemotherapy and radiotherapy are not modelled by any method here — no ODE arm exists
                  for either. In melanoma today both are largely palliative or adjuvant, not
                  survival-directed the way checkpoint or BRAF/MEK blockade is, so this is shown as
                  historical record only, not a comparator for the twin.
                </p>
              )}
          </>
        ) : (
          <div className="tabular mt-1 text-[13.5px] font-bold leading-tight text-clinical-muted">
            Not recorded
          </div>
        )}
      </div>

      <p className="mt-4 border-t border-clinical-border pt-3 text-[11.5px] leading-snug text-clinical-muted">
        TCGA does not record LDH or ECOG performance status. Where the decision logic would normally
        weigh rapid-control pressure, stage IV stands in — shown explicitly in the decision path.
      </p>
    </Panel>
  );
}
