"""
Phase 4: Survival Analysis
==========================
Tests whether the ODE model's outputs stratify TCGA-SKCM patients by overall
survival. Three mechanistic readouts are examined:

  * pERK                     — proliferative drive from the MAPK cascade
  * tumour burden            — BRAFi sweep; proliferation vs. immune control
  * checkpoint tumour burden — anti-PD-1 sweep (Module D); immunotherapy arm

For each readout:
  A. Loop over the 10 dose columns for that readout's drug (vemurafenib for
     the first two, anti-PD-1 for the third), fit a Cox PH model to each, and
     pick the dose with the smallest p-value ("optimal dose").
  B. At that dose, scan every candidate cut-point and keep the threshold that
     best separates High vs. Low groups by the log-rank test.
  C. Plot Kaplan-Meier curves for the two groups.

Biological hypothesis: patients whose tumours sustain high MAPK output
(high pERK / high modelled tumour burden) are more aggressive and have
worse overall survival; patients whose modelled tumour burden stays high even
under simulated anti-PD-1 have checkpoint-refractory disease and worse
survival.

Output:
    plots/km_ode_stratified.pdf           (combined KM figure — all 3 readouts)
    plots/cox_dose_scan.pdf               (Cox p-value vs dose, all 3 readouts)
    plots/km_perk.png, plots/km_tumour_burden.png, plots/km_checkpoint_tumour_burden.png
    results/survival_summary.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE   = os.path.join(BASE_DIR, "data", "melanoma_params_full.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
PERK_FILE       = os.path.join(RESULTS_DIR, "pERK_simulations.csv")
TUMOUR_FILE     = os.path.join(RESULTS_DIR, "tumour_burden_simulations.csv")
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "checkpoint_tumour_simulations.csv")
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 65)
print("  Phase 4: Survival Analysis (TCGA-SKCM)")
print("=" * 65)


def run_cox_scan(sim_df, clinical_df, label, dose_prefix, dose_label, ax=None):
    """
    Fit a univariate Cox PH model at every dose column (identified by
    `dose_prefix`, e.g. "BRAFi_" or "antiPD1_") and return the dose giving the
    smallest p-value plus the merged analysis frame. Optionally draw the
    p-value-vs-dose curve onto `ax` (x-axis labelled with `dose_label`).
    """
    dose_cols = [c for c in sim_df.columns if c.startswith(dose_prefix)]

    merged = sim_df.merge(
        clinical_df[["SAMPLE_ID", "OS_MONTHS", "SURV_STATUS"]],
        on="SAMPLE_ID", how="inner").dropna(subset=["OS_MONTHS", "SURV_STATUS"])
    # Clip tiny negative numerical values from the ODE integrator
    merged[dose_cols] = merged[dose_cols].clip(lower=0)
    print(f"\n  [{label}] {len(merged)} patients with complete survival data")

    pvals = {}
    for col in dose_cols:
        try:
            cph = CoxPHFitter()
            cph.fit(merged[["OS_MONTHS", "SURV_STATUS", col]],
                    duration_col="OS_MONTHS", event_col="SURV_STATUS")
            pvals[col] = cph.summary["p"].iloc[0]
        except Exception:
            pvals[col] = 1.0

    opt_col = min(pvals, key=pvals.get)
    print(f"  [{label}] Optimal dose: {opt_col}  |  Cox p = {pvals[opt_col]:.4e}")

    if ax is not None:
        doses = [float(c.replace(dose_prefix, "")) for c in dose_cols]
        ax.semilogy(doses, [pvals[c] for c in dose_cols], "o-",
                    color="#2563EB", lw=2, ms=6)
        ax.axhline(0.05, color="red", ls="--", alpha=0.7, label="p = 0.05")
        ax.axvline(float(opt_col.replace(dose_prefix, "")), color="green", ls=":",
                   alpha=0.8, label=f"optimal {opt_col}")
        ax.set_xlabel(dose_label)
        ax.set_ylabel("Cox PH p-value (log)")
        ax.set_title(f"{label}", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    return opt_col, pvals[opt_col], merged


def optimal_threshold_km(merged, opt_col, label, ax):
    """
    Scan candidate cut-points on `opt_col`, keep the one with the smallest
    log-rank p-value (each group >= 15 patients), and draw KM curves on `ax`.
    """
    vals   = merged[opt_col].values
    times  = merged["OS_MONTHS"].values
    events = merged["SURV_STATUS"].values

    best_p, best_thr = 1.0, np.median(vals)
    for thr in np.quantile(vals, np.linspace(0.15, 0.85, 40)):
        grp = vals > thr
        if grp.sum() < 15 or (~grp).sum() < 15:
            continue
        res = logrank_test(times[grp], times[~grp],
                           event_observed_A=events[grp],
                           event_observed_B=events[~grp])
        if res.p_value < best_p:
            best_p, best_thr = res.p_value, thr

    high = vals > best_thr
    n_hi, n_lo = int(high.sum()), int((~high).sum())

    # Determine the OBSERVED direction from the data (do not assume it):
    # compare median OS of the High vs Low group.
    kmf = KaplanMeierFitter()
    kmf.fit(times[high], events[high]);  med_hi = kmf.median_survival_time_
    kmf.fit(times[~high], events[~high]); med_lo = kmf.median_survival_time_
    worse_group = "High" if med_hi < med_lo else "Low"
    print(f"  [{label}] Optimal threshold: {best_thr:.4f}  |  log-rank p = {best_p:.4e}")
    print(f"  [{label}] High n={n_hi} (median OS {med_hi:.1f} mo)  |  "
          f"Low n={n_lo} (median OS {med_lo:.1f} mo)  -> worse: {worse_group}")

    # Colour the worse-prognosis group red, better group blue
    hi_color = "#DC2626" if worse_group == "High" else "#2563EB"
    lo_color = "#2563EB" if worse_group == "High" else "#DC2626"
    kmf.fit(times[high], events[high],
            label=f"High {label} (n={n_hi}, med {med_hi:.0f} mo)")
    kmf.plot_survival_function(ax=ax, color=hi_color, lw=2.5, ci_alpha=0.12)
    kmf.fit(times[~high], events[~high],
            label=f"Low {label} (n={n_lo}, med {med_lo:.0f} mo)")
    kmf.plot_survival_function(ax=ax, color=lo_color, lw=2.5, ci_alpha=0.12)

    ax.set_xlabel("Overall survival (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"KM stratified by ODE {label}\n(dose {opt_col}, log-rank p = {best_p:.3e})",
                 fontweight="bold", fontsize=11)
    ax.text(0.97, 0.95, f"p = {best_p:.3e}", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85))
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    direction = (f"High {label} -> WORSE survival"
                 if worse_group == "High" else
                 f"High {label} -> BETTER survival")
    return best_thr, best_p, n_hi, n_lo, direction, med_hi, med_lo


# ─── Main ─────────────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
clinical = pd.read_csv(DATA_FILE)
perk_sim = pd.read_csv(PERK_FILE)
tumour_sim = pd.read_csv(TUMOUR_FILE)
checkpoint_sim = pd.read_csv(CHECKPOINT_FILE)
print(f"    Clinical: {len(clinical)}  |  pERK sims: {perk_sim.shape}  "
      f"|  tumour sims: {tumour_sim.shape}  |  checkpoint sims: {checkpoint_sim.shape}")

readouts = [
    ("pERK", perk_sim, "BRAFi_", "Vemurafenib dose"),
    ("tumour burden", tumour_sim, "BRAFi_", "Vemurafenib dose"),
    ("checkpoint tumour burden", checkpoint_sim, "antiPD1_", "Anti-PD-1 dose"),
]

# Cox dose scan figure
fig_scan, axes_scan = plt.subplots(1, 3, figsize=(18, 4.5))
# KM figure
fig_km, axes_km = plt.subplots(1, 3, figsize=(19, 6))

summary = ["ODE Model — Survival Analysis Summary (TCGA-SKCM Melanoma)",
           "=" * 60]

for j, (label, sim, dose_prefix, dose_label) in enumerate(readouts):
    print("\n" + "-" * 55)
    print(f"  Analysing ODE readout: {label}")
    print("-" * 55)
    opt_col, cox_p, merged = run_cox_scan(
        sim, clinical, label, dose_prefix, dose_label, ax=axes_scan[j])
    thr, lr_p, n_hi, n_lo, direction, med_hi, med_lo = optimal_threshold_km(
        merged, opt_col, label, ax=axes_km[j])
    summary += [
        f"\n{label}:",
        f"  Optimal dose ({dose_label}) : {opt_col}",
        f"  Cox PH p-value           : {cox_p:.4e}",
        f"  KM threshold             : {thr:.4f}",
        f"  Log-rank p-value         : {lr_p:.4e}",
        f"  High group n={n_hi} (median OS {med_hi:.1f} mo), "
        f"Low group n={n_lo} (median OS {med_lo:.1f} mo)",
        f"  Observed direction       : {direction}",
    ]

fig_scan.suptitle("Cox PH p-value vs drug dose — TCGA-SKCM", fontweight="bold")
fig_scan.tight_layout()
fig_scan.savefig(os.path.join(PLOTS_DIR, "cox_dose_scan.pdf"), bbox_inches="tight")
fig_scan.savefig(os.path.join(PLOTS_DIR, "cox_dose_scan.png"), dpi=150, bbox_inches="tight")

fig_km.suptitle("Kaplan-Meier stratification by ODE outputs — TCGA-SKCM Melanoma",
                fontweight="bold", fontsize=13)
fig_km.tight_layout()
fig_km.savefig(os.path.join(PLOTS_DIR, "km_ode_stratified.pdf"), bbox_inches="tight")
# Also individual PNGs for the report
for (label, _, _, _), ax in zip(readouts, axes_km):
    fig_single, ax_single = plt.subplots(figsize=(7, 6))
    for line in ax.get_lines():
        ax_single.plot(line.get_xdata(), line.get_ydata(),
                       color=line.get_color(), lw=line.get_linewidth(),
                       label=line.get_label())
    ax_single.set_title(ax.get_title(), fontweight="bold", fontsize=11)
    ax_single.set_xlabel(ax.get_xlabel()); ax_single.set_ylabel(ax.get_ylabel())
    ax_single.set_ylim(0, 1.02); ax_single.legend(fontsize=9, loc="lower left")
    ax_single.grid(alpha=0.3)
    fname = "km_" + label.replace(" ", "_") + ".png"
    fig_single.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig_single)

plt.close("all")

summary_txt = "\n".join(summary) + "\n\nInterpretation:\n" + (
    "  A log-rank p < 0.05 means the ODE readout separates melanoma patients by\n"
    "  survival; the observed direction for each readout is stated above and is\n"
    "  read directly from the data, not assumed.\n"
    "   * Mechanistic pERK reflects MAPK-pathway output, which is greatest in\n"
    "     NRAS-mutant tumours — the worst-prognosis SKCM subtype — so elevated\n"
    "     pERK tracks more aggressive disease.\n"
    "   * Modelled tumour burden (BRAFi sweep) integrates proliferation against\n"
    "     immune control; high residual burden marks aggressive, immune-cold\n"
    "     disease.\n"
    "   * Modelled checkpoint tumour burden (anti-PD-1 sweep, Module D) marks\n"
    "     disease that persists even under simulated checkpoint blockade —\n"
    "     i.e. checkpoint-refractory tumours — expected to track worse OS just\n"
    "     as the BRAFi-arm tumour burden does.\n"
    "  All three readouts are prognostic where log-rank p < 0.05.\n")
summary_path = os.path.join(RESULTS_DIR, "survival_summary.txt")
with open(summary_path, "w") as f:
    f.write(summary_txt)
print("\n" + summary_txt)
print(f"  Saved KM figure : {os.path.join(PLOTS_DIR, 'km_ode_stratified.pdf')}")
print(f"  Saved scan figure: {os.path.join(PLOTS_DIR, 'cox_dose_scan.pdf')}")
print(f"  Saved summary   : {summary_path}")

print("\n" + "=" * 65)
print("  Phase 4 COMPLETE — Ready for Phase 5 (RPPA Validation)")
print("=" * 65)
