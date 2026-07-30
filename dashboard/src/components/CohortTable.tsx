import { useMemo, useState } from "react";
import { Search, Users, ArrowUpDown, ChevronRight, X } from "lucide-react";
import type { CohortPatient } from "../data/cohort";
import { pdl1Band } from "../data/cohort";
import { integrate, type AgreementStatus } from "../lib/integrationEngine";
import { Panel, Pill } from "./ui";

// ---------------------------------------------------------------------------
// The cohort browser — all 421 real TCGA-SKCM patients, searchable, filterable
// and sortable, with the Q5 recommendation and methods-agreement resolved for
// every row so the clinician can scan for the interesting cases (the discordant
// ones) rather than clicking through blindly.
// ---------------------------------------------------------------------------

export interface CohortRow {
  patient: CohortPatient;
  recommendation: string;
  recommendationKey: string;
  confidence: number;
  agreement: AgreementStatus;
  statistical: number;
}

/** Resolve the Q5 output for every patient once, then reuse it. */
export function buildRows(patients: CohortPatient[]): CohortRow[] {
  return patients.map((patient) => {
    const result = integrate(patient);
    const primary = result.options.find((o) => o.tier === "primary") ?? result.options[0];
    return {
      patient,
      recommendation: primary.arm.label,
      recommendationKey: primary.arm.key,
      confidence: primary.confidence,
      agreement: result.agreement.status,
      statistical: result.agreement.statistical.value,
    };
  });
}

const AGREEMENT_PILL: Record<AgreementStatus, { label: string; tone: "green" | "blue" | "amber" | "neutral" }> = {
  concordant: { label: "Concordant", tone: "green" },
  partial: { label: "Partial", tone: "blue" },
  discordant: { label: "Discordant", tone: "amber" },
  unavailable: { label: "Single method", tone: "neutral" },
};

const REC_TONE: Record<string, "teal" | "blue" | "amber"> = {
  immuno: "teal",
  targeted: "blue",
  combo: "amber",
};

type SortKey = "id" | "statistical" | "antipd1" | "brafi" | "confidence" | "recommendation";

const SORTS: { key: SortKey; label: string }[] = [
  { key: "id", label: "Patient ID" },
  { key: "statistical", label: "Response evidence" },
  { key: "antipd1", label: "Anti-PD-1 reduction" },
  { key: "brafi", label: "BRAFi reduction" },
  { key: "confidence", label: "Confidence" },
  { key: "recommendation", label: "Recommendation" },
];

function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="flex items-center gap-1.5">
      <span className="text-[10.5px] font-bold uppercase tracking-wide text-clinical-muted">
        {label}
      </span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-lg border border-clinical-border bg-white px-2.5 py-1.5 text-[12px] font-semibold text-clinical-ink outline-none transition hover:border-clinical-teal focus:border-clinical-tealdark focus:ring-2 focus:ring-clinical-teal/20"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function CohortTable({
  rows,
  onOpen,
}: {
  rows: CohortRow[];
  onOpen: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [braf, setBraf] = useState("all");
  const [pdl1, setPdl1] = useState("all");
  const [agreement, setAgreement] = useState("all");
  const [treatment, setTreatment] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("id");
  const [descending, setDescending] = useState(false);
  const [limit, setLimit] = useState(50);

  const filtered = useMemo(() => {
    const q = query.trim().toUpperCase();
    const out = rows.filter(({ patient, agreement: a }) => {
      if (q && !patient.id.toUpperCase().includes(q)) return false;
      if (braf !== "all" && (braf === "mut" ? patient.braf === "WT" : patient.braf !== "WT")) return false;
      if (pdl1 !== "all" && pdl1Band(patient.pdl1Pct) !== pdl1) return false;
      if (agreement !== "all" && a !== agreement) return false;
      if (treatment === "recorded" && !patient.treatment.recorded) return false;
      if (treatment === "unrecorded" && patient.treatment.recorded) return false;
      return true;
    });

    const value = (r: CohortRow): number | string => {
      switch (sortKey) {
        case "statistical": return r.statistical;
        case "antipd1": return r.patient.antipd1Reduction;
        case "brafi": return r.patient.brafiReduction;
        case "confidence": return r.confidence;
        case "recommendation": return r.recommendation;
        default: return r.patient.id;
      }
    };

    return [...out].sort((a, b) => {
      const av = value(a);
      const bv = value(b);
      const cmp = typeof av === "string" ? av.localeCompare(bv as string) : (av as number) - (bv as number);
      return descending ? -cmp : cmp;
    });
  }, [rows, query, braf, pdl1, agreement, treatment, sortKey, descending]);

  const visible = filtered.slice(0, limit);
  const hasFilters =
    query !== "" || braf !== "all" || pdl1 !== "all" || agreement !== "all" || treatment !== "all";

  const reset = () => {
    setQuery("");
    setBraf("all");
    setPdl1("all");
    setAgreement("all");
    setTreatment("all");
  };

  return (
    <Panel
      title="Patient Cohort"
      subtitle={`${rows.length} TCGA-SKCM patients with a full digital twin`}
      icon={<Users size={16} />}
      right={
        <Pill tone={filtered.length === rows.length ? "neutral" : "teal"}>
          {filtered.length} shown
        </Pill>
      }
    >
      {/* --- controls, one row above the table --- */}
      <div className="mb-3.5 flex flex-wrap items-center gap-2.5">
        <div className="relative min-w-[190px] flex-1">
          <Search
            size={14}
            className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-clinical-muted"
          />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search patient ID…"
            className="w-full rounded-lg border border-clinical-border bg-white py-1.5 pl-8 pr-3 text-[12.5px] font-semibold text-clinical-ink outline-none transition placeholder:font-normal placeholder:text-clinical-muted hover:border-clinical-teal focus:border-clinical-tealdark focus:ring-2 focus:ring-clinical-teal/20"
          />
        </div>

        <Select
          label="BRAF"
          value={braf}
          onChange={setBraf}
          options={[
            { value: "all", label: "All" },
            { value: "mut", label: "V600 mutant" },
            { value: "wt", label: "Wild-type" },
          ]}
        />
        <Select
          label="PD-L1"
          value={pdl1}
          onChange={setPdl1}
          options={[
            { value: "all", label: "All" },
            { value: "High", label: "High" },
            { value: "Intermediate", label: "Intermediate" },
            { value: "Low", label: "Low" },
          ]}
        />
        <Select
          label="Agreement"
          value={agreement}
          onChange={setAgreement}
          options={[
            { value: "all", label: "All" },
            { value: "concordant", label: "Concordant" },
            { value: "partial", label: "Partial" },
            { value: "discordant", label: "Discordant" },
            { value: "unavailable", label: "Single method" },
          ]}
        />
        <Select
          label="Treatment"
          value={treatment}
          onChange={setTreatment}
          options={[
            { value: "all", label: "All" },
            { value: "recorded", label: "Recorded" },
            { value: "unrecorded", label: "Not recorded" },
          ]}
        />
        <Select
          label="Sort"
          value={sortKey}
          onChange={(v) => setSortKey(v as SortKey)}
          options={SORTS.map((s) => ({ value: s.key, label: s.label }))}
        />
        <button
          onClick={() => setDescending((d) => !d)}
          title={descending ? "Descending" : "Ascending"}
          className="flex items-center gap-1 rounded-lg border border-clinical-border bg-white px-2.5 py-1.5 text-[11.5px] font-bold text-clinical-ink transition hover:border-clinical-teal hover:text-clinical-tealdark"
        >
          <ArrowUpDown size={13} /> {descending ? "Desc" : "Asc"}
        </button>
        {hasFilters && (
          <button
            onClick={reset}
            className="flex items-center gap-1 rounded-lg border border-clinical-border bg-white px-2.5 py-1.5 text-[11.5px] font-bold text-clinical-muted transition hover:border-clinical-rose hover:text-clinical-rose"
          >
            <X size={13} /> Clear
          </button>
        )}
      </div>

      {/* --- table --- */}
      {filtered.length === 0 ? (
        <div className="rounded-xl border border-dashed border-clinical-border bg-clinical-bg/60 px-4 py-10 text-center">
          <p className="text-[13px] font-bold text-clinical-ink">No patients match these filters</p>
          <p className="mt-1 text-[12px] text-clinical-muted">
            Try clearing the search or widening the molecular filters.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px] border-collapse text-left">
            <thead>
              <tr className="border-b border-clinical-border text-[10px] font-bold uppercase tracking-wide text-clinical-muted">
                <th className="py-2 pr-3">Patient</th>
                <th className="py-2 pr-3">BRAF</th>
                <th className="py-2 pr-3">NRAS</th>
                <th className="py-2 pr-3">PD-L1</th>
                <th className="py-2 pr-3 text-right">Response evidence</th>
                <th className="py-2 pr-3 text-right">Anti-PD-1 ↓</th>
                <th className="py-2 pr-3 text-right">BRAFi ↓</th>
                <th className="py-2 pr-3">Q5 recommendation</th>
                <th className="py-2 pr-3">Agreement</th>
                <th className="py-2" />
              </tr>
            </thead>
            <tbody>
              {visible.map(({ patient, recommendation, recommendationKey, confidence, agreement: a, statistical }) => {
                const band = pdl1Band(patient.pdl1Pct);
                const pill = AGREEMENT_PILL[a];
                return (
                  <tr
                    key={patient.id}
                    onClick={() => onOpen(patient.id)}
                    className="cursor-pointer border-b border-clinical-border transition hover:bg-clinical-teal/[0.05]"
                  >
                    <td className="py-2.5 pr-3 text-[12.5px] font-bold text-clinical-ink">
                      {patient.id}
                    </td>
                    <td className="py-2.5 pr-3">
                      <span
                        className={
                          "text-[12px] font-bold " +
                          (patient.braf === "WT" ? "text-clinical-muted" : "text-clinical-bluedark")
                        }
                      >
                        {patient.braf === "WT" ? "WT" : "V600"}
                      </span>
                    </td>
                    <td className="py-2.5 pr-3">
                      <span
                        className={
                          "text-[12px] font-bold " +
                          (patient.nras === "WT" ? "text-clinical-muted" : "text-amber-700")
                        }
                      >
                        {patient.nras === "WT" ? "WT" : "mut"}
                      </span>
                    </td>
                    <td className="py-2.5 pr-3">
                      <span className="tabular text-[12px] font-semibold text-clinical-ink">
                        {band}{" "}
                        <span className="text-[11px] font-normal text-clinical-muted">
                          ({patient.pdl1Pct})
                        </span>
                      </span>
                    </td>
                    <td className="tabular py-2.5 pr-3 text-right text-[12.5px] font-bold text-clinical-ink">
                      {Math.round(statistical * 100)}
                    </td>
                    <td className="tabular py-2.5 pr-3 text-right text-[12.5px] text-clinical-ink">
                      {patient.antipd1Informative ? `${Math.round(patient.antipd1Reduction * 100)}%` : "—"}
                    </td>
                    <td className="tabular py-2.5 pr-3 text-right text-[12.5px] text-clinical-ink">
                      {patient.brafiInformative ? `${Math.round(patient.brafiReduction * 100)}%` : "—"}
                    </td>
                    <td className="py-2.5 pr-3">
                      <div className="flex items-center gap-1.5">
                        <Pill tone={REC_TONE[recommendationKey] ?? "neutral"}>{recommendation}</Pill>
                        <span className="tabular text-[11px] font-bold text-clinical-muted">
                          {confidence}%
                        </span>
                      </div>
                    </td>
                    <td className="py-2.5 pr-3">
                      <Pill tone={pill.tone}>{pill.label}</Pill>
                    </td>
                    <td className="py-2.5 text-right">
                      <ChevronRight size={15} className="text-clinical-muted" />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {filtered.length > visible.length && (
        <div className="mt-3.5 text-center">
          <button
            onClick={() => setLimit((l) => l + 100)}
            className="rounded-lg border border-clinical-tealdark px-4 py-2 text-[12.5px] font-bold text-clinical-tealdark transition hover:bg-clinical-teal/10"
          >
            Show more — {filtered.length - visible.length} remaining
          </button>
        </div>
      )}
    </Panel>
  );
}
