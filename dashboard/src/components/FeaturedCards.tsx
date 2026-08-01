import { Sparkles, ArrowRight } from "lucide-react";
import type { CohortRow } from "./CohortTable";
import { getPhenotypeColor } from "../data/palette";
import { Panel, Pill } from "./ui";

interface PhenotypeCardDef {
  key: string;
  shortLabel: string;
  story: string;
  expect: string;
  match: (r: CohortRow) => boolean;
  rank: (r: CohortRow) => number;
}

const PHENOTYPE_CARDS: PhenotypeCardDef[] = [
  {
    key: "hot",
    shortLabel: "Immune Hot",
    story: "Inflamed microenvironment – primary checkpoint blockade candidate.",
    expect: "Immunotherapy",
    match: (r) => r.patient.q5?.shortLabel === "Immune Hot",
    rank: (r) => r.patient.q5?.treatabilityIndex ?? 0,
  },
  {
    key: "cold",
    shortLabel: "Immune Cold",
    story: "T-cell desert – dual M2-depleting agent + checkpoint combination needed.",
    expect: "Combination rescue arm",
    match: (r) => r.patient.q5?.shortLabel === "Immune Cold",
    rank: (r) => r.patient.q5?.treatabilityIndex ?? 0,
  },
  {
    key: "m2",
    shortLabel: "M2-High",
    story: "Stromal exclusion by M2 macrophages – targeted BRAF/MEK or stromal remodelling.",
    expect: "Targeted therapy",
    match: (r) => r.patient.q5?.shortLabel === "M2-High" || r.patient.q5?.shortLabel === "Immunosuppressive M2-High",
    rank: (r) => r.patient.q5?.dabrafenibSensitivity ?? r.patient.q5?.treatabilityIndex ?? 0,
  },
  {
    key: "nf1",
    shortLabel: "Mutant-Driven",
    story: "100% NF1 loss-of-function, RAS hyperactivation, high TMB – ICI + MEK adjunct.",
    expect: "ICI + MEK adjunct",
    match: (r) => r.patient.q5?.shortLabel === "Mutant-Driven",
    rank: (r) => r.patient.q5?.treatabilityIndex ?? 0,
  },
];

/** Pick the most representative real patient for each phenotype. */
export function pickFeatured(rows: CohortRow[]): { cardDef: PhenotypeCardDef; row: CohortRow }[] {
  const used = new Set<string>();
  const out: { cardDef: PhenotypeCardDef; row: CohortRow }[] = [];
  for (const cardDef of PHENOTYPE_CARDS) {
    const best = rows
      .filter((r) => cardDef.match(r) && !used.has(r.patient.id))
      .sort((a, b) => cardDef.rank(b) - cardDef.rank(a))[0];
    if (best) {
      used.add(best.patient.id);
      out.push({ cardDef, row: best });
    }
  }
  return out;
}

export default function FeaturedCards({
  rows,
  onOpen,
}: {
  rows: CohortRow[];
  onOpen: (id: string) => void;
}) {
  const featured = pickFeatured(rows);
  if (!featured.length) return null;

  return (
    <Panel
      title="Subgroup Stratification Archetypes"
      subtitle="Representative TCGA-SKCM cases for each of the four Q5 phenotypes"
      icon={<Sparkles size={16} />}
    >
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {featured.map(({ cardDef, row }, i) => {
          const phenoColor = getPhenotypeColor(cardDef.shortLabel);
          const q5 = row.patient.q5;
          return (
            <button
              key={cardDef.key}
              onClick={() => onOpen(row.patient.id)}
              style={{ animationDelay: `${i * 60}ms` }}
              className="animate-fade-up group flex flex-col justify-between rounded-2xl border border-clinical-border bg-white p-4 text-left shadow-card transition hover:-translate-y-0.5 hover:border-okabe-purple-dark hover:shadow-lift"
            >
              <div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <span
                      className="h-3 w-3 rounded-full shrink-0"
                      style={{ backgroundColor: phenoColor }}
                    />
                    <span className="text-[13px] font-extrabold tracking-tight text-clinical-ink">
                      {cardDef.shortLabel}
                    </span>
                  </div>
                  <span className="tabular text-[11px] font-bold text-clinical-muted">
                    {row.patient.id}
                  </span>
                </div>

                <p className="mt-2 text-[11.5px] leading-snug text-clinical-muted">
                  {cardDef.story}
                </p>
              </div>

              <div className="mt-3.5">
                <div className="grid grid-cols-2 gap-2">
                  <div className="rounded-lg border border-clinical-border bg-clinical-bg px-2.5 py-1.5">
                    <div className="text-[9px] font-bold uppercase tracking-wide text-clinical-muted">
                      Treatability
                    </div>
                    <div className="tabular text-[13.5px] font-extrabold text-clinical-ink">
                      {q5?.treatabilityIndex !== null && q5?.treatabilityIndex !== undefined
                        ? q5.treatabilityIndex.toFixed(0)
                        : "–"}
                    </div>
                  </div>
                  <div className="rounded-lg border border-clinical-border bg-clinical-bg px-2.5 py-1.5">
                    <div className="text-[9px] font-bold uppercase tracking-wide text-clinical-muted">
                      Confidence
                    </div>
                    <div
                      className={
                        "tabular text-[13.5px] font-extrabold " +
                        (q5?.confidenceBand === "High"
                          ? "text-green-700"
                          : q5?.confidenceBand === "Moderate"
                            ? "text-amber-700"
                            : "text-clinical-muted")
                      }
                    >
                      {q5?.confidenceBand ?? "–"}
                    </div>
                  </div>
                </div>

                <div className="mt-3 flex items-center justify-between pt-1">
                  <Pill tone="neutral">{cardDef.expect}</Pill>
                  <span className="flex items-center gap-1 text-[11.5px] font-bold text-okabe-purple-dark shrink-0">
                    Open <ArrowRight size={13} className="transition group-hover:translate-x-0.5" />
                  </span>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </Panel>
  );
}
