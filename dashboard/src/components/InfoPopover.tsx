import { useState, useRef, useEffect, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { HelpCircle, X, Info } from "lucide-react";

// Generic "(i)" info button for any panel or card — click to reveal where a
// number comes from, how it's calculated, and what it does/doesn't mean.
// Pure UI shell (positioning, click-outside, portal rendering); no data or
// scoring logic lives here — content is passed in per call site, and every
// call site is responsible for keeping its own copy accurate.

export function InfoPopover({
  title,
  children,
  size = 13,
}: {
  title: string;
  children: ReactNode;
  size?: number;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const [coords, setCoords] = useState<{ top: number; right: number }>({ top: 0, right: 0 });

  const updateCoords = () => {
    if (buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect();
      setCoords({
        top: rect.bottom + 6,
        right: Math.max(16, window.innerWidth - rect.right),
      });
    }
  };

  useEffect(() => {
    if (isOpen) {
      updateCoords();
      window.addEventListener("scroll", updateCoords, true);
      window.addEventListener("resize", updateCoords);
    }
    return () => {
      window.removeEventListener("scroll", updateCoords, true);
      window.removeEventListener("resize", updateCoords);
    };
  }, [isOpen]);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        popoverRef.current &&
        !popoverRef.current.contains(event.target as Node) &&
        buttonRef.current &&
        !buttonRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    }
    if (isOpen) document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  return (
    <span className="relative inline-flex items-center normal-case font-normal align-middle">
      <button
        ref={buttonRef}
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          setIsOpen((v) => !v);
        }}
        className="inline-flex items-center justify-center text-clinical-muted hover:text-clinical-tealdark transition p-0.5 rounded-full hover:bg-clinical-bg focus:outline-none focus:ring-1 focus:ring-clinical-teal/30"
        title={`About: ${title}`}
        aria-label={`About: ${title}`}
      >
        <HelpCircle size={size} className="shrink-0" />
      </button>

      {isOpen &&
        createPortal(
          <div
            ref={popoverRef}
            style={{ position: "fixed", top: `${coords.top}px`, right: `${coords.right}px`, zIndex: 9999 }}
            className="w-80 sm:w-96 rounded-xl border border-clinical-border bg-white p-3.5 shadow-lift text-left text-clinical-ink normal-case font-normal"
          >
            <div className="flex items-center justify-between border-b border-clinical-border pb-2 mb-2.5">
              <div className="flex items-center gap-1.5 font-extrabold text-[12.5px] text-clinical-tealdark">
                <Info size={14} />
                <span>{title}</span>
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
            <div className="text-[11.5px] leading-relaxed text-clinical-ink space-y-2">{children}</div>
          </div>,
          document.body
        )}
    </span>
  );
}
