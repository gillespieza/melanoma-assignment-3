import { useState, useRef, useEffect } from "react";
import { HelpCircle, X, ShieldCheck, CheckCircle2, AlertCircle, AlertTriangle } from "lucide-react";

export function ConfidenceHelpPopover({ size = 11 }: { size?: number }) {
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
        title="Explain Confidence Band"
        aria-label="Explain Confidence Band"
      >
        <HelpCircle size={size} className="shrink-0" />
      </button>

      {isOpen && (
        <div className="absolute right-0 top-6 z-[100] w-72 sm:w-80 rounded-xl border border-clinical-border bg-white p-3.5 shadow-xl text-left text-clinical-ink normal-case font-normal animate-in fade-in zoom-in-95 duration-150">
          <div className="flex items-center justify-between border-b border-clinical-border pb-2 mb-2.5">
            <div className="flex items-center gap-1.5 font-extrabold text-[12.5px] text-okabe-purple-dark">
              <ShieldCheck size={14} />
              <span>Confidence Band Classification</span>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setIsOpen(false);
              }}
              className="text-clinical-muted hover:text-clinical-ink p-1 rounded-md hover:bg-clinical-bg"
            >
              <X size={13} />
            </button>
          </div>

          <p className="text-[11px] leading-relaxed text-clinical-ink">
            Confidence reflects the degree of multi-method agreement (Q1 ML predictor + Q3 ODE digital twin) and biomarker support.
          </p>

          <div className="mt-2.5 space-y-2 text-[10.5px]">
            <div className="rounded-lg bg-emerald-50/70 border border-emerald-200/70 p-2 space-y-0.5">
              <div className="font-bold text-emerald-900 flex items-center gap-1">
                <CheckCircle2 size={12} className="text-emerald-600" /> High Confidence
              </div>
              <p className="text-emerald-800 text-[10px] leading-snug">
                ML predictors and ODE digital twin strongly agree; robust biomarker evidence across all channels.
              </p>
            </div>

            <div className="rounded-lg bg-amber-50/70 border border-amber-200/70 p-2 space-y-0.5">
              <div className="font-bold text-amber-900 flex items-center gap-1">
                <AlertTriangle size={12} className="text-amber-600" /> Moderate Confidence
              </div>
              <p className="text-amber-800 text-[10px] leading-snug">
                Same recommendation direction between methods, but varying strength or moderate biomarker signals.
              </p>
            </div>

            <div className="rounded-lg bg-rose-50/70 border border-rose-200/70 p-2 space-y-0.5">
              <div className="font-bold text-rose-900 flex items-center gap-1">
                <AlertCircle size={12} className="text-rose-600" /> Low / Disagree Confidence
              </div>
              <p className="text-rose-800 text-[10px] leading-snug">
                Methods split (ML vs ODE disagree) or single-method readout only; MDT consultant review advised.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
