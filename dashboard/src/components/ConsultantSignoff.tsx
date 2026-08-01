import { useState } from "react";
import type { RankedOption } from "../data/types";
import { ShieldCheck, PenLine, Check } from "lucide-react";

export default function ConsultantSignoff({
  selected,
}: {
  selected: RankedOption | null;
}) {
  const [signed, setSigned] = useState(false);

  return (
    <div className="rounded-2xl border border-clinical-border bg-white shadow-card">
      <div className="flex flex-col items-start gap-4 p-5 md:flex-row md:items-center">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-clinical-blue/10 text-clinical-bluedark">
            <ShieldCheck size={20} />
          </div>
          <div className="leading-tight">
            <div className="text-[13px] font-extrabold text-clinical-ink">
              Consultant sign-off
            </div>
            <div className="text-[11.5px] text-clinical-muted">
              The model advises; the consultant decides and is accountable for the plan.
            </div>
          </div>
        </div>

        <div className="flex-1 rounded-xl border border-clinical-border bg-clinical-bg px-4 py-2.5 text-[12.5px]">
          {selected ? (
            <span className="text-clinical-ink">
              Confirmed plan:{" "}
              <span className="font-extrabold text-okabe-purple-dark">{selected.arm.label}</span> –{" "}
              {selected.arm.regimen}
            </span>
          ) : (
            <span className="text-clinical-muted">
              Select a treatment lane above to stage it for sign-off.
            </span>
          )}
        </div>

        <button
          disabled={!selected}
          onClick={() => setSigned(true)}
          className={
            "flex items-center gap-2 rounded-xl px-5 py-2.5 text-[13px] font-bold transition " +
            (!selected
              ? "cursor-not-allowed bg-clinical-bg text-clinical-muted"
              : signed
                ? "bg-clinical-green text-white"
                : "bg-clinical-ink text-white hover:bg-clinical-bluedark")
          }
        >
          {signed ? (
            <>
              <Check size={16} /> Signed · Dr. Aoife Gallagher
            </>
          ) : (
            <>
              <PenLine size={16} /> Sign & finalise plan
            </>
          )}
        </button>
      </div>
    </div>
  );
}
