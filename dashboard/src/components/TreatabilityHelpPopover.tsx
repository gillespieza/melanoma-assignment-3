import { useState, useRef, useEffect } from "react";
import { HelpCircle, X, Sparkles, Activity } from "lucide-react";

export function TreatabilityHelpPopover() {
  const [isOpen, setIsOpen] = useState(false);
  const popoverRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (popoverRef.current && !popoverRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  return (
    <div className="relative inline-flex items-center normal-case font-normal" ref={popoverRef}>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        className="inline-flex items-center justify-center text-clinical-muted hover:text-okabe-purple-dark transition p-0.5 rounded-full hover:bg-clinical-bg focus:outline-none focus:ring-1 focus:ring-okabe-purple/30 normal-case"
        title="Explain Treatability Index (TI) Score"
        aria-label="Explain Treatability Index Score"
      >
        <HelpCircle size={13} className="shrink-0" />
      </button>

      {isOpen && (
        <div className="absolute right-0 top-6 z-[100] w-80 sm:w-96 rounded-xl border border-clinical-border bg-white p-4 shadow-xl text-left text-clinical-ink normal-case font-normal animate-in fade-in zoom-in-95 duration-150">
          <div className="flex items-center justify-between border-b border-clinical-border pb-2 mb-2.5">
            <div className="flex items-center gap-1.5 font-extrabold text-[13px] text-okabe-purple-dark">
              <Sparkles size={14} />
              <span>Treatability Index (TI Score)</span>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="text-clinical-muted hover:text-clinical-ink p-1 rounded-md hover:bg-clinical-bg"
            >
              <X size={14} />
            </button>
          </div>

          <p className="text-[11.5px] leading-relaxed text-clinical-ink">
            The <strong>Treatability Index (TI, 0–100)</strong> quantifies a patient&apos;s overall tumor-microenvironmental susceptibility to checkpoint blockade monotherapy versus targeted/combination therapy.
          </p>

          <div className="mt-3 space-y-2 text-[11px]">

            <div className="rounded-lg border border-clinical-border bg-clinical-bg p-2.5 space-y-1">
              <div className="font-bold text-[10.5px] text-clinical-ink">Clinical Interpretation:</div>
              <div className="flex items-start gap-1 text-[10.5px] text-clinical-muted">
                <span className="font-bold text-emerald-700">TI ≥ 50:</span>
                <span>Favourable immune microenvironment; <strong>checkpoint monotherapy (Arm A)</strong> indicated.</span>
              </div>
              <div className="flex items-start gap-1 text-[10.5px] text-clinical-muted">
                <span className="font-bold text-amber-700">TI &lt; 50:</span>
                <span>Stromal exclusion / unprimed; <strong>targeted therapy (Arm B)</strong> or <strong>microenvironmental combination (Arm C)</strong> indicated.</span>
              </div>
            </div>

            <div className="font-bold uppercase tracking-wide text-clinical-muted flex items-center gap-1 text-[10px]">
              <Activity size={12} /> 4-Dimensional Empirical Component Weights
            </div>

            <div className="grid grid-cols-2 gap-1.5">
              <div className="rounded-lg bg-emerald-50/60 border border-emerald-200/60 p-2">
                <span className="font-bold text-emerald-900">1. Antigen Presentation</span>
                <p className="text-[10px] text-emerald-800 mt-0.5">MHC-I &amp; B2M machinery (+weight)</p>
              </div>
              <div className="rounded-lg bg-blue-50/60 border border-blue-200/60 p-2">
                <span className="font-bold text-blue-900">2. IFN-γ Signalling</span>
                <p className="text-[10px] text-blue-800 mt-0.5">Inflammatory signalling (+weight)</p>
              </div>
              <div className="rounded-lg bg-purple-50/60 border border-purple-200/60 p-2">
                <span className="font-bold text-purple-900">3. Effector Cytotoxicity</span>
                <p className="text-[10px] text-purple-800 mt-0.5">CD8 T-cell &amp; GZMA/PRF1 (+weight)</p>
              </div>
              <div className="rounded-lg bg-rose-50/60 border border-rose-200/60 p-2">
                <span className="font-bold text-rose-900">4. M2 TAM Barrier</span>
                <p className="text-[10px] text-rose-800 mt-0.5">Macrophage exclusion (-penalty)</p>
              </div>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}
