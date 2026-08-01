import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ODE_MODULES } from "../data/model";
import { Check, Loader2, Activity } from "lucide-react";

export default function SimulationOverlay({
  show,
  onDone,
}: {
  show: boolean;
  onDone: () => void;
}) {
  const [step, setStep] = useState(0);

  useEffect(() => {
    if (!show) return;
    setStep(0);
    const per = 520;
    const timers = ODE_MODULES.map((_, i) =>
      setTimeout(() => setStep(i + 1), per * (i + 1))
    );
    const done = setTimeout(onDone, per * (ODE_MODULES.length + 1) + 250);
    return () => {
      timers.forEach(clearTimeout);
      clearTimeout(done);
    };
  }, [show, onDone]);

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-clinical-ink/40 backdrop-blur-sm"
        >
          <motion.div
            initial={{ scale: 0.96, y: 8 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.98, opacity: 0 }}
            className="w-[440px] max-w-[92vw] rounded-3xl border border-white/50 bg-white p-7 shadow-lift"
          >
            <div className="flex items-center gap-3">
              <div className="relative flex h-11 w-11 items-center justify-center rounded-xl bg-okabe-purple-dark text-white">
                <Activity size={22} />
                <motion.span
                  className="absolute inset-0 rounded-xl border-2 border-okabe-purple"
                  animate={{ scale: [1, 1.35], opacity: [0.7, 0] }}
                  transition={{ duration: 1.2, repeat: Infinity }}
                />
              </div>
              <div>
                <div className="text-[15px] font-extrabold text-clinical-ink">
                  Running ODE Digital-Twin Simulation
                </div>
                <div className="text-[12px] text-clinical-muted">
                  Solving fast signalling + slow tumour timescales…
                </div>
              </div>
            </div>

            <div className="mt-6 space-y-2.5">
              {ODE_MODULES.map((m, i) => {
                const state = step > i ? "done" : step === i ? "active" : "pending";
                return (
                  <div
                    key={m.id}
                    className={
                      "flex items-center gap-3 rounded-xl border px-3.5 py-2.5 transition " +
                      (state === "done"
                        ? "border-okabe-purple/30 bg-okabe-purple/5"
                        : state === "active"
                          ? "border-clinical-blue/40 bg-clinical-blue/5"
                          : "border-clinical-border bg-clinical-bg opacity-60")
                    }
                  >
                    <span className="flex h-6 w-6 items-center justify-center">
                      {state === "done" ? (
                        <span className="flex h-6 w-6 items-center justify-center rounded-full bg-okabe-purple-dark text-white">
                          <Check size={14} strokeWidth={3} />
                        </span>
                      ) : state === "active" ? (
                        <Loader2 size={18} className="animate-spin text-clinical-bluedark" />
                      ) : (
                        <span className="h-2 w-2 rounded-full bg-clinical-border" />
                      )}
                    </span>
                    <div className="leading-tight">
                      <div className="text-[13px] font-bold text-clinical-ink">
                        Module {String.fromCharCode(65 + i)} · {m.label}
                      </div>
                      <div className="text-[11px] text-clinical-muted">{m.note}</div>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="mt-5 h-1.5 w-full overflow-hidden rounded-full bg-clinical-bg">
              <motion.div
                className="h-full rounded-full bg-okabe-purple-dark"
                initial={{ width: "0%" }}
                animate={{ width: `${(step / ODE_MODULES.length) * 100}%` }}
                transition={{ ease: "easeOut" }}
              />
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
