import { useMemo, useState } from "react";
import { Search, Users, ArrowUpDown, ChevronRight, X, HelpCircle } from "lucide-react";
import type { CohortPatient } from "../data/cohort";
import { getPhenotypeColor } from "../data/palette";
import { integrate, type AgreementStatus } from "../lib/integrationEngine";
import { Panel, Pill } from "./ui";

// The cohort browser – all 421 real TCGA-SKCM patients, searchable, filterable
// and sortable, with the Q5 recommendation and methods-agreement resolved for
// every row so the clinician can scan for the interesting cases (the discordant
// ones) rather than clicking through blindly.

export interface CohortRow {
  patient: CohortPatient;
  cohort: string;
  phenotype: string;
  treatabilityIndex: number | null;
  confidenceBand: "Low" | "Moderate" | "High" | "N/A";
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
    const q5 = patient.q5;
    return {
      patient,
      cohort: patient.cohort ?? "TCGA-SKCM",
      phenotype: q5?.shortLabel ?? "Unknown",
      treatabilityIndex: q5?.treatabilityIndex ?? null,
      confidenceBand: q5?.confidenceBand ?? "N/A",
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

const REC_TONE: Record<string, "okabe-purple" | "blue" | "amber"> = {
  immuno: "okabe-purple",
  targeted: "blue",
  combo: "amber",
};

type SortKey = "id" | "phenotype" | "treatability" | "antipd1" | "brafi" | "confidence" | "recommendation" | "recconfidence";

const SORTS: { key: SortKey; label: string }[] = [
  { key: "id", label: "Patient ID" },
  { key: "phenotype", label: "Phenotype" },
  { key: "treatability", label: "Treatability Index" },
  { key: "antipd1", label: "ICI tumour reduction" },
  { key: "brafi", label: "BRAFi tumour reduction" },
  { key: "confidence", label: "Phenotype confidence" },
  { key: "recommendation", label: "Recommendation" },
  { key: "recconfidence", label: "Recommendation confidence" },
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
        className="rounded-lg border border-clinical-border bg-white px-2.5 py-1.5 text-[12px] font-semibold text-clinical-ink outline-none transition hover:border-okabe-purple focus:border-okabe-purple-dark focus:ring-2 focus:ring-okabe-purple/20"
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
  const [phenotypeFilter, setPhenotypeFilter] = useState("all");
  const [agreement, setAgreement] = useState("all");
  const [treatment, setTreatment] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("id");
  const [descending, setDescending] = useState(false);
  const [limit, setLimit] = useState(50);

  const filtered = useMemo(() => {
    const q = query.trim().toUpperCase();
    const out = rows.filter(({ patient, phenotype, agreement: a }) => {
      if (q && !patient.id.toUpperCase().includes(q)) return false;
      if (phenotypeFilter !== "all" && phenotype.toLowerCase() !== phenotypeFilter.toLowerCase()) return false;
      if (agreement !== "all" && a !== agreement) return false;
      if (treatment === "recorded" && !patient.treatment.recorded) return false;
      if (treatment === "unrecorded" && patient.treatment.recorded) return false;
      return true;
    });

    const value = (r: CohortRow): number | string => {
      switch (sortKey) {
        case "phenotype": return r.phenotype;
        case "treatability": return r.treatabilityIndex ?? -1;
        case "recconfidence": return r.confidence;
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
  }, [rows, query, phenotypeFilter, agreement, treatment, sortKey, descending]);

  const visible = filtered.slice(0, limit);
  const hasFilters =
    query !== "" || phenotypeFilter !== "all" || agreement !== "all" || treatment !== "all";

  const reset = () => {
    setQuery("");
    setPhenotypeFilter("all");
    setAgreement("all");
    setTreatment("all");
  };

  return (
    <Panel
      title="Patient Cohort"
      subtitle={`${rows.length} TCGA-SKCM patients with full Q5 phenotyping & digital twin`}
      icon={<Users size={16} />}
      right={
        <Pill tone={filtered.length === rows.length ? "neutral" : "okabe-purple"}>
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
            className="w-full rounded-lg border border-clinical-border bg-white py-1.5 pl-8 pr-3 text-[12.5px] font-semibold text-clinical-ink outline-none transition placeholder:font-normal placeholder:text-clinical-muted hover:border-okabe-purple focus:border-okabe-purple-dark focus:ring-2 focus:ring-okabe-purple/20"
          />
        </div>

        <Select
          label="Phenotype"
          value={phenotypeFilter}
          onChange={setPhenotypeFilter}
          options={[
            { value: "all", label: "All Phenotypes" },
            { value: "Immune Hot", label: "Immune Hot" },
            { value: "Immune Cold", label: "Immune Cold" },
            { value: "M2-High", label: "M2-High" },
            { value: "Mutant-Driven", label: "Mutant-Driven" },
          ]}
        />
        <Select
          label="Agreement"
          value={agreement}
          onChange={setAgreement}
          options={[
            { value: "all", label: "All Agreements" },
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
            { value: "all", label: "All Treatments" },
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
          className="flex items-center gap-1 rounded-lg border border-clinical-border bg-white px-2.5 py-1.5 text-[11.5px] font-bold text-clinical-ink transition hover:border-okabe-purple hover:text-okabe-purple-dark"
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
            Try clearing the search or widening the phenotype filters.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px] border-collapse text-left">
            <thead>
              <tr className="border-b border-clinical-border text-[10px] font-bold uppercase tracking-wide text-clinical-muted">
                <th className="py-2 pr-3">Patient</th>
                <th className="py-2 pr-3">Q5 Phenotype</th>
                <th className="py-2 pr-3">
                  <div className="group relative inline-flex items-center gap-1 cursor-help">
                    <span>Phenotype confidence</span>
                    <HelpCircle size={11} className="text-clinical-muted group-hover:text-okabe-purple transition" />
                    <div className="pointer-events-none absolute left-0 top-full z-30 mt-1.5 hidden w-72 rounded-xl border border-clinical-border bg-white p-3 text-left shadow-lift group-hover:block">
                      <div className="text-[11px] font-bold text-clinical-ink">Phenotype Assignment Confidence</div>
                      <div className="mt-1 text-[11px] font-normal normal-case leading-snug text-clinical-muted">
                        How confidently the Q5 GMM places this patient in their assigned phenotype cluster. Derived from the GMM posterior probability across all four subgroups — a High patient sits clearly inside one cluster; Low means they fall near a boundary between clusters.
                      </div>
                    </div>
                  </div>
                </th>
                <th className="py-2 pr-3 text-right">
                  <div className="group relative inline-flex items-center justify-end gap-1 cursor-help">
                    <span>TI (%)</span>
                    <HelpCircle size={11} className="text-clinical-muted group-hover:text-okabe-purple transition" />
                    <div className="pointer-events-none absolute right-0 top-full z-30 mt-1.5 hidden w-64 rounded-xl border border-clinical-border bg-white p-3 text-left shadow-lift group-hover:block">
                      <div className="text-[11px] font-bold text-clinical-ink">Treatability Index (0–100)</div>
                      <div className="mt-1 text-[11px] font-normal normal-case leading-snug text-clinical-muted">
                        Quantifies overall therapeutic tractability by combining antigen presentation, IFN-γ signaling, and microenvironmental barriers.
                      </div>
                    </div>
                  </div>
                </th>

                <th className="py-2 pr-3 text-right">
                  <div className="group relative inline-flex items-center justify-end gap-1 cursor-help">
                    <span>ICI tumour ↓</span>
                    <HelpCircle size={11} className="text-clinical-muted group-hover:text-okabe-purple transition" />
                    <div className="pointer-events-none absolute right-0 top-full z-30 mt-1.5 hidden w-64 rounded-xl border border-clinical-border bg-white p-3 text-left shadow-lift group-hover:block">
                      <div className="text-[11px] font-bold text-clinical-ink">ICI Tumour Burden Reduction</div>
                      <div className="mt-1 text-[11px] font-normal normal-case leading-snug text-clinical-muted">
                        Projected percentage tumour burden reduction under Anti-PD-1 immunotherapy (ICI), as simulated by the Q3 ODE digital-twin model. A dash (–) means the simulation was uninformative for this patient.
                      </div>
                    </div>
                  </div>
                </th>
                <th className="py-2 pr-3 text-right">
                  <div className="group relative inline-flex items-center justify-end gap-1 cursor-help">
                    <span>BRAFi ↓</span>
                    <HelpCircle size={11} className="text-clinical-muted group-hover:text-okabe-purple transition" />
                    <div className="pointer-events-none absolute right-0 top-full z-30 mt-1.5 hidden w-64 rounded-xl border border-clinical-border bg-white p-3 text-left shadow-lift group-hover:block">
                      <div className="text-[11px] font-bold text-clinical-ink">BRAFi Tumour Burden Reduction</div>
                      <div className="mt-1 text-[11px] font-normal normal-case leading-snug text-clinical-muted">
                        Projected percentage tumour burden reduction under BRAF inhibitor therapy (e.g. Dabrafenib + Trametinib), as simulated by the Q3 ODE digital-twin model. A dash (–) means the simulation was uninformative for this patient.
                      </div>
                    </div>
                  </div>
                </th>
                <th className="py-2 pr-3">Q5 recommendation</th>
                <th className="py-2 pr-3 text-right">
                  <div className="group relative inline-flex items-center justify-end gap-1 cursor-help">
                    <span>Rec. confidence</span>
                    <HelpCircle size={11} className="text-clinical-muted group-hover:text-okabe-purple transition" />
                    <div className="pointer-events-none absolute right-0 top-full z-30 mt-1.5 hidden w-72 rounded-xl border border-clinical-border bg-white p-3 text-left shadow-lift group-hover:block">
                      <div className="text-[11px] font-bold text-clinical-ink">Recommendation Confidence</div>
                      <div className="mt-1 text-[11px] font-normal normal-case leading-snug text-clinical-muted">
                        A 0–100 heuristic score ranking how strongly the integration engine favours the recommended arm over the alternatives. Inputs: PD-L1 percentile, IFN-γ signature, BRAF status, LDH, ECOG, and Q3 ODE tumour reduction. Not a probability of clinical response — use Phenotype confidence for the model-grounded certainty estimate.
                      </div>
                    </div>
                  </div>
                </th>
                <th className="py-2 pr-3">Agreement</th>
                <th className="py-2" />
              </tr>
            </thead>
            <tbody>
              {visible.map(({ patient, phenotype, treatabilityIndex, confidenceBand, recommendation, recommendationKey, confidence, agreement: a }) => {
                const phenoColor = getPhenotypeColor(phenotype);
                const pill = AGREEMENT_PILL[a];
                return (
                  <tr
                    key={patient.id}
                    onClick={() => onOpen(patient.id)}
                    className="cursor-pointer border-b border-clinical-border transition hover:bg-okabe-purple/[0.05]"
                  >
                    <td className="py-2.5 pr-3 text-[12.5px] font-bold text-clinical-ink">
                      {patient.id}
                    </td>
                    <td className="py-2.5 pr-3">
                      <div className="flex items-center gap-1.5">
                        <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ backgroundColor: phenoColor }} />
                        <span className="text-[12px] font-bold text-clinical-ink">
                          {phenotype}
                        </span>
                      </div>
                    </td>
                    <td className="py-2.5 pr-3">
                      <Pill tone={confidenceBand === "High" ? "green" : confidenceBand === "Moderate" ? "amber" : "neutral"}>
                        {confidenceBand}
                      </Pill>
                    </td>
                    <td className="tabular py-2.5 pr-3 text-right text-[12.5px] font-bold text-clinical-ink">
                      {treatabilityIndex !== null ? treatabilityIndex.toFixed(0) : "–"}
                    </td>

                    <td className="tabular py-2.5 pr-3 text-right text-[12.5px] text-clinical-ink">
                      {patient.antipd1Informative ? `${Math.round(patient.antipd1Reduction * 100)}%` : "–"}
                    </td>
                    <td className="tabular py-2.5 pr-3 text-right text-[12.5px] text-clinical-ink">
                      {patient.brafiInformative ? `${Math.round(patient.brafiReduction * 100)}%` : "–"}
                    </td>
                    <td className="py-2.5 pr-3">
                      <Pill tone={REC_TONE[recommendationKey] ?? "neutral"}>{recommendation}</Pill>
                    </td>
                    <td className="tabular py-2.5 pr-3 text-right text-[12.5px] font-bold text-clinical-ink">
                      {confidence}%
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
            className="rounded-lg border border-okabe-purple-dark px-4 py-2 text-[12.5px] font-bold text-okabe-purple-dark transition hover:bg-okabe-purple/10"
          >
            Show more – {filtered.length - visible.length} remaining
          </button>
        </div>
      )}
    </Panel>
  );
}
