import type { ReactNode } from "react";

export function Panel({
  title,
  subtitle,
  icon,
  right,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  icon?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={
        "rounded-2xl border border-clinical-border bg-clinical-panel shadow-card " + className
      }
    >
      {title && (
        <div className="flex items-center gap-2.5 border-b border-clinical-border px-5 py-3.5">
          {icon && <span className="text-okabe-purple-dark">{icon}</span>}
          <div className="leading-tight">
            <h2 className="text-[13.5px] font-bold tracking-tight text-clinical-ink">{title}</h2>
            {subtitle && <p className="text-[11.5px] text-clinical-muted">{subtitle}</p>}
          </div>
          {right && <div className="ml-auto">{right}</div>}
        </div>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function Pill({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "teal" | "blue" | "amber" | "rose" | "green" | "okabe-purple";
}) {
  const map: Record<string, string> = {
    neutral: "bg-clinical-bg text-clinical-muted border-clinical-border",
    teal: "bg-okabe-purple/10 text-okabe-purple-dark border-okabe-purple/20",
    "okabe-purple": "bg-okabe-purple/10 text-okabe-purple-dark border-okabe-purple/20",
    blue: "bg-clinical-blue/10 text-clinical-bluedark border-clinical-blue/20",
    amber: "bg-amber-50 text-amber-700 border-amber-200",
    rose: "bg-rose-50 text-rose-700 border-rose-200",
    green: "bg-green-50 text-green-700 border-green-200",
  };
  return (
    <span
      className={
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-semibold whitespace-nowrap " +
        map[tone]
      }
    >
      {children}
    </span>
  );
}

export function Stat({
  label,
  value,
  unit,
  tone = "ink",
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  tone?: "ink" | "teal" | "blue" | "rose" | "okabe-purple";
}) {
  const color =
    tone === "teal" || tone === "okabe-purple"
      ? "text-okabe-purple-dark"
      : tone === "blue"
        ? "text-clinical-bluedark"
        : tone === "rose"
          ? "text-clinical-rose"
          : "text-clinical-ink";
  return (
    <div className="rounded-xl border border-clinical-border bg-clinical-bg px-3.5 py-2.5">
      <div className="text-[10.5px] font-semibold uppercase tracking-wide text-clinical-muted">
        {label}
      </div>
      <div className={"tabular text-[22px] font-extrabold leading-tight " + color}>
        {value}
        {unit && <span className="ml-1 text-[12px] font-semibold text-clinical-muted">{unit}</span>}
      </div>
    </div>
  );
}
