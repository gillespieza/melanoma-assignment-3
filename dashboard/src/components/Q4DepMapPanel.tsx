import { Dna, Target, Beaker, CircleSlash } from "lucide-react";
import { Panel, Pill } from "./ui";

// Q4 · DepMap target nomination.
//
// This is cohort-level, not per-patient: a CRISPR co-essentiality screen run
// once across melanoma cell lines (Q4_dep_map/), not a score computed for the
// patient currently on screen. It replaces the earlier per-patient "Q4 ·
// Resistance" tab, which was a Q3-derived heuristic that predated this real
// analysis and was never wired to it.
//
// Numbers below are pulled from Hena's real analysis output, committed at
// Q4_dep_map/outputs_v2/results/ (main SOX10 dependency screen) and
// Q4_dep_map/outputs_proxy/results/ (co-dependency druggable-proxy search) —
// see sox10_summary.txt and sox10_proxy_candidates_ranked.csv respectively.
//
// One honesty caveat worth keeping visible: the "druggable" categorisation
// on the three candidates below (target class / mechanism) is Hena's own
// domain judgement — the formal audit file meant to verify each one against
// ChEMBL/Open Targets (top10_druggability_audit_template.csv) is still an
// empty template. These are plausible candidates, not confirmed drug targets.

const CANDIDATES = [
  {
    gene: "LCMT1",
    note: "Druggable PP2A regulator",
  },
  {
    gene: "AMD1",
    note: "Druggable metabolic enzyme",
  },
  {
    gene: "PGM3",
    note: "Druggable metabolic enzyme",
  },
];

const CONTROLS = ["BRAF", "MAPK1"];

export default function Q4DepMapPanel() {
  return (
    <Panel
      title="Q4 · DepMap Target Nomination"
      subtitle="CRISPR co-dependency screen across melanoma cell lines — cohort-level, not a per-patient score"
      icon={<Dna size={16} />}
      right={<Pill tone="neutral">Cohort-level</Pill>}
    >
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div>
          <div className="flex items-start gap-2.5 rounded-xl border border-clinical-border bg-clinical-bg px-3.5 py-3">
            <CircleSlash size={18} className="mt-0.5 shrink-0 text-clinical-muted" />
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-[12px] font-bold text-clinical-ink">SOX10</span>
                <Pill tone="blue">Confirmed lineage dependency</Pill>
              </div>
              <p className="mt-1 text-[11.5px] leading-snug text-clinical-muted">
                84% of melanoma cell lines are CRISPR-dependent on SOX10 vs. 2.8% of other-cancer
                lines (Hedges&apos; g = -4.49, FDR = 1.2×10⁻²⁰) — one of the largest, most selective
                dependencies in the screen. Expression tracks with essentiality (Spearman ρ = -0.338,
                p = 0.0032; 1000-iteration bootstrap 95% CI for selectivity: [-1.34, -1.04], well clear
                of zero). But SOX10 is a transcription factor, not tractable with current small-molecule
                drugs, so the screen looked for genes that <span className="font-semibold">co-depend</span>{" "}
                with it instead.
              </p>
            </div>
          </div>

          <div className="mt-3 space-y-2">
            <div className="text-[10.5px] font-bold uppercase tracking-wide text-clinical-muted">
              Method validation
            </div>
            <div className="flex items-start gap-2 rounded-lg border border-clinical-border bg-white px-3 py-2.5">
              <Beaker size={13} className="mt-0.5 shrink-0 text-clinical-bluedark" />
              <p className="min-w-0 text-[11.5px] leading-snug text-clinical-muted">
                The same co-dependency ranking correctly recovers{" "}
                <span className="font-semibold text-clinical-ink">{CONTROLS.join(" and ")}</span> —
                established melanoma drug targets — as top-scoring positive controls, which is
                evidence the ranking approach itself is sound.
              </p>
            </div>
          </div>
        </div>

        <div>
          <div className="mb-2 flex items-center gap-1.5 text-[10.5px] font-bold uppercase tracking-wide text-clinical-muted">
            <Target size={12} /> Candidate druggable proxies
          </div>
          <ol className="space-y-2">
            {CANDIDATES.map((c, i) => (
              <li
                key={c.gene}
                className="flex items-start gap-2.5 rounded-lg border border-clinical-border bg-white px-3 py-2.5"
              >
                <span className="tabular mt-px flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-green-50 text-[11px] font-extrabold text-green-700">
                  {i + 1}
                </span>
                <div className="min-w-0">
                  <span className="text-[12px] font-bold text-clinical-ink">{c.gene}</span>
                  <p className="mt-0.5 text-[11.5px] leading-snug text-clinical-muted">{c.note}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </div>

      <p className="mt-4 border-t border-clinical-border pt-3 text-[11.5px] leading-relaxed text-clinical-muted">
        <span className="font-bold text-clinical-ink">Provenance:</span> from the real DepMap
        CRISPR co-essentiality analysis in{" "}
        <code className="text-[11px]">Q4_dep_map/outputs_v2/</code> and{" "}
        <code className="text-[11px]">Q4_dep_map/outputs_proxy/</code>, not a fitted model and not
        specific to any patient in this cohort. The dependency statistics above are real and
        bootstrap-validated; the "druggable" categorisation on the candidate list is expert
        judgement, not yet backed by a completed ChEMBL/Open Targets audit — these are
        hypothesis-generating candidates from a correlational screen, pending both mechanistic
        validation and a formal druggability check before they inform an actual treatment decision.
      </p>
    </Panel>
  );
}
