import type { DecisionNode } from "../data/types";
import { Panel } from "./ui";
import { GitBranch, ChevronRight, CircleDot } from "lucide-react";
import { motion } from "framer-motion";

export default function DecisionTree({ path }: { path: DecisionNode[] }) {
  return (
    <Panel
      title="Q5 Treatment Decision Pathway"
      subtitle="The route this patient took through the triage logic"
      icon={<GitBranch size={16} />}
    >
      <div className="flex flex-col gap-3 md:flex-row md:items-stretch">
        {path.map((node, i) => {
          const isFinal = i === path.length - 1;
          return (
            <div key={node.id} className="flex flex-1 items-stretch gap-3">
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.08 }}
                className={
                  "flex-1 rounded-xl border p-3.5 " +
                  (isFinal
                    ? "border-clinical-tealdark bg-clinical-tealdark text-white shadow-lift"
                    : "border-clinical-border bg-clinical-bg")
                }
              >
                <div className="flex items-center gap-1.5">
                  <CircleDot
                    size={13}
                    className={isFinal ? "text-white" : "text-clinical-tealdark"}
                  />
                  <span
                    className={
                      "text-[10px] font-bold uppercase tracking-wide " +
                      (isFinal ? "text-white/80" : "text-clinical-muted")
                    }
                  >
                    {isFinal ? "Recommendation" : `Step ${i + 1}`}
                  </span>
                </div>
                <div
                  className={
                    "mt-1 text-[13.5px] font-extrabold leading-tight " +
                    (isFinal ? "text-white" : "text-clinical-ink")
                  }
                >
                  {node.label}
                </div>
                <div
                  className={
                    "mt-0.5 text-[11px] leading-snug " +
                    (isFinal ? "text-white/80" : "text-clinical-muted")
                  }
                >
                  {node.detail}
                </div>
              </motion.div>
              {!isFinal && (
                <div className="hidden items-center md:flex">
                  <ChevronRight size={18} className="text-clinical-border" />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </Panel>
  );
}
