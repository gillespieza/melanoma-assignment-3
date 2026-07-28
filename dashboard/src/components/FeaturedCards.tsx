import { Sparkles, ArrowRight, SlidersHorizontal } from "lucide-react";
import type { CohortRow } from "./CohortTable";
import { Panel, Pill } from "./ui";

// ---------------------------------------------------------------------------
// Featured archetypes for the scripted live demo.
//
// These are NOT the hardcoded v1 patients — they are three REAL cohort patients
// chosen at render time because they cleanly exhibit the three clinical
// archetypes. That way the demo story and the browsable cohort are the same data.
// ---------------------------------------------------------------------------

interface Archetype {
  key: string;
  title: string;
  story: string;
  expect: string;
  match: (r: CohortRow) => boolean;
  rank: (r: CohortRow) => number;
}

const ARCHETYPES: Archetype[] = [
  {
    key: "A",
    title: "BRAF-mutant · PD-L1 low",
    story: "A targetable driver in an immune-cold tumour — the case for going after the pathway.",
    expect: "Expect targeted therapy to lead",
    match: (r) =>
      r.patient.braf !== "WT" && r.patient.pdl1Pct < 33 && r.patient.brafiInformative,
    rank: (r) => r.patient.brafiReduction - r.patient.antipd1Reduction,
  },
  {
    key: "B",
    title: "BRAF wild-type · PD-L1 high",
    story: "No targetable driver, but hot immune biology — and the RAF paradox on display.",
    expect: "Expect immunotherapy to lead",
    match: (r) =>
      r.patient.braf === "WT" && r.patient.pdl1Pct >= 66 && r.patient.antipd1Informative,
    rank: (r) => r.patient.antipd1Reduction,
  },
  {
    key: "C",
    title: "BRAF-mutant · PD-L1 high",
    story: "Both lanes open. This is the sequencing debate that DREAMseq and SECOMBIT exist to settle.",
    expect: "Expect a close call between lanes",
    match: (r) =>
      r.patient.braf !== "WT" &&
      r.patient.pdl1Pct >= 66 &&
      r.patient.brafiInformative &&
      r.patient.antipd1Informative,
    rank: (r) => Math.min(r.patient.brafiReduction, r.patient.antipd1Reduction),
  },
];

/** Pick the most representative real patient for each archetype. */
export function pickFeatured(rows: CohortRow[]): { archetype: Archetype; row: CohortRow }[] {
  const used = new Set<string>();
  const out: { archetype: Archetype; row: CohortRow }[] = [];
  for (const archetype of ARCHETYPES) {
    const best = rows
      .filter((r) => archetype.match(r) && !used.has(r.patient.id))
      .sort((a, b) => archetype.rank(b) - archetype.rank(a))[0];
    if (best) {
      used.add(best.patient.id);
      out.push({ archetype, row: best });
    }
  }
  return out;
}

export default function FeaturedCards({
  rows,
  onOpen,
  onOpenArchetypes,
}: {
  rows: CohortRow[];
  onOpen: (id: string) => void;
  onOpenArchetypes: () => void;
}) {
  const featured = pickFeatured(rows);
  if (!featured.length) return null;

  return (
    <Panel
      title="Featured Cases"
      subtitle="Real cohort patients that cleanly show each clinical archetype"
      icon={<Sparkles size={16} />}
      right={
        <button
          onClick={onOpenArchetypes}
          className="flex items-center gap-1.5 rounded-lg border border-clinical-border px-2.5 py-1.5 text-[11.5px] font-bold text-clinical-ink transition hover:border-clinical-teal hover:text-clinical-tealdark"
        >
          <SlidersHorizontal size={13} /> Editable archetypes
        </button>
      }
    >
      <div className="grid gap-3 lg:grid-cols-3">
        {featured.map(({ archetype, row }, i) => (
          <button
            key={archetype.key}
            onClick={() => onOpen(row.patient.id)}
            style={{ animationDelay: `${i * 60}ms` }}
            className="animate-fade-up group rounded-2xl border border-clinical-border bg-white p-4 text-left shadow-card transition hover:-translate-y-0.5 hover:border-clinical-tealdark hover:shadow-lift"
          >
            <div className="flex items-center justify-between">
              <span className="flex h-6 w-6 items-center justify-center rounded-lg bg-clinical-teal/10 text-[12px] font-extrabold text-clinical-tealdark">
                {archetype.key}
              </span>
              <span className="tabular text-[11.5px] font-bold text-clinical-muted">
                {row.patient.id}
              </span>
            </div>

            <h3 className="mt-2.5 text-[14px] font-extrabold tracking-tight text-clinical-ink">
              {archetype.title}
            </h3>
            <p className="mt-1 text-[12px] leading-snug text-clinical-muted">{archetype.story}</p>

            <div className="mt-3 grid grid-cols-2 gap-2">
              <div className="rounded-lg border border-clinical-border bg-clinical-bg px-2.5 py-1.5">
                <div className="text-[9.5px] font-bold uppercase tracking-wide text-clinical-muted">
                  Anti-PD-1 ↓
                </div>
                <div className="tabular text-[14px] font-extrabold text-clinical-tealdark">
                  {Math.round(row.patient.antipd1Reduction * 100)}%
                </div>
              </div>
              <div className="rounded-lg border border-clinical-border bg-clinical-bg px-2.5 py-1.5">
                <div className="text-[9.5px] font-bold uppercase tracking-wide text-clinical-muted">
                  BRAFi ↓
                </div>
                <div className="tabular text-[14px] font-extrabold text-clinical-bluedark">
                  {Math.round(row.patient.brafiReduction * 100)}%
                </div>
              </div>
            </div>

            <div className="mt-3 flex items-center justify-between">
              <Pill tone="neutral">{archetype.expect}</Pill>
              <span className="flex items-center gap-1 text-[12px] font-bold text-clinical-tealdark">
                Open <ArrowRight size={13} className="transition group-hover:translate-x-0.5" />
              </span>
            </div>
          </button>
        ))}
      </div>
    </Panel>
  );
}
