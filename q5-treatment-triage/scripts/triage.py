"""
Q5: Treatment-selection triage for melanoma (TCGA-SKCM)
======================================================
Integrates the two response axes the group already modelled, plus mutation
status, into a single first-line treatment recommendation per patient:

  * Targeted-therapy axis (BRAF/MEK inhibitors)  <- Q3 ODE model
        Predicted vemurafenib pERK suppression. High only in BRAF-V600
        tumours; NRAS/wild-type show the RAF paradox (no benefit).
  * Immunotherapy axis (checkpoint blockade)     <- Q1 immune signatures
        Immune-infiltration score from CD8A / PRF1 / GZMA (cytotoxic-T proxy).
        In deployment this slot takes Q1's calibrated response predictor.
  * Eligibility gate                             <- mutation status
        BRAF-V600 vs NRAS vs triple-wild-type.

Decision logic (first-line, when multiple options are on the table):

  BRAF-V600 mutant
     immune-hot, indolent      -> IMMUNOTHERAPY FIRST (durable; targeted in reserve)
     immune-hot, high burden   -> TARGETED -> IMMUNOTHERAPY (rapid control, then switch)
     immune-cold               -> TARGETED FIRST (BRAF/MEK inhibitor)
  BRAF wild-type
     immune-hot                -> IMMUNOTHERAPY
     immune-cold (NRAS/triple) -> TRIAL / CHEMOTHERAPY fallback (hardest group)

Chemotherapy (dacarbazine/temozolomide) is reserved as a fallback throughout,
reflecting current melanoma practice.

NOTE: research-grade decision support, not a validated clinical device. Real
first-line choice also uses guideline criteria, LDH, stage, performance status,
brain-metastasis status, and patient preference.

Output:
    results/triage_assignments.csv     one row per patient + recommendation
    results/triage_summary.txt         cohort distribution + survival by group
    plots/triage_decision_matrix.png   the BRAF x immune decision grid
    plots/triage_km_by_recommendation.png  survival by recommended strategy
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSIGN_DIR = os.path.dirname(BASE_DIR)
Q3_DIR     = os.path.join(ASSIGN_DIR, "q3-ode-model")
RESULTS    = os.path.join(BASE_DIR, "results")
PLOTS      = os.path.join(BASE_DIR, "plots")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

PARAMS = os.path.join(Q3_DIR, "data", "melanoma_params_full.csv")
PERK   = os.path.join(Q3_DIR, "results", "pERK_simulations.csv")
TUMOUR = os.path.join(Q3_DIR, "results", "tumour_burden_simulations.csv")

# Recommendation labels
IO_FIRST   = "Immunotherapy first"
TGT_TO_IO  = "Targeted -> Immunotherapy"
TGT_FIRST  = "Targeted therapy first"
TRIAL_CHEMO = "Trial / Chemotherapy"

REC_COLORS = {
    IO_FIRST:    "#2563EB",
    TGT_TO_IO:   "#7C3AED",
    TGT_FIRST:   "#DC2626",
    TRIAL_CHEMO: "#6B7280",
}


def recommend(row):
    """Return (recommendation, one-line rationale) for one patient."""
    braf = row["braf_v600"]
    immune_hot = row["immune_hot"]
    high_burden = row["high_burden"]

    if braf:
        if immune_hot and not high_burden:
            return IO_FIRST, "BRAF-V600 + immune-hot, low burden: durable IO benefit, targeted in reserve"
        if immune_hot and high_burden:
            return TGT_TO_IO, "BRAF-V600 + immune-hot, high burden: targeted for rapid control, then IO"
        return TGT_FIRST, "BRAF-V600 + immune-cold: BRAF/MEK inhibitor gives the clearest predicted benefit"
    else:
        if immune_hot:
            return IO_FIRST, "BRAF-wild-type + immune-hot: checkpoint blockade (no targeted option)"
        return TRIAL_CHEMO, "BRAF-wild-type + immune-cold: hardest group; trial or chemo fallback"


def main():
    print("=" * 62)
    print("  Q5: Melanoma treatment-selection triage (TCGA-SKCM)")
    print("=" * 62)

    params = pd.read_csv(PARAMS)
    perk = pd.read_csv(PERK)
    tumour = pd.read_csv(TUMOUR)
    dose_cols = [c for c in perk.columns if c.startswith("BRAFi_")]

    df = params.copy()

    # ── Axis 1: targeted-therapy response (Q3 ODE) ────────────────────────────
    perk_low = perk[dose_cols[0]].clip(lower=1e-9)
    perk_high = perk[dose_cols[-1]].clip(lower=0)
    df["targeted_response"] = ((perk_low - perk_high) / perk_low).clip(lower=0).values
    df["tumour_burden"] = tumour[dose_cols[-1]].clip(lower=0).values

    # ── Axis 2: immunotherapy response proxy (Q1-style immune score) ──────────
    df["immune_score"] = df[["CD8A", "PRF1", "GZMA"]].mean(axis=1)

    # ── Eligibility + stratification thresholds (cohort-relative) ─────────────
    df["braf_v600"] = df["BRAF_MUT"] == 1
    imm_thr = df["immune_score"].median()
    burden_thr = df["tumour_burden"].median()
    df["immune_hot"] = df["immune_score"] >= imm_thr
    df["high_burden"] = df["tumour_burden"] >= burden_thr

    # ── Assign recommendations ────────────────────────────────────────────────
    recs = df.apply(recommend, axis=1)
    df["recommendation"] = [r[0] for r in recs]
    df["rationale"] = [r[1] for r in recs]

    # Molecular-immune subtype label (for the matrix / KM)
    df["subtype"] = np.where(df["braf_v600"], "BRAF",
                    np.where(df["NRAS_MUT"] == 1, "NRAS", "WT"))
    df["immune_label"] = np.where(df["immune_hot"], "immune-hot", "immune-cold")

    # ── Save per-patient triage table ─────────────────────────────────────────
    out_cols = ["PATIENT_ID", "SAMPLE_ID", "subtype", "BRAF_VARIANT", "NRAS_MUT",
                "immune_score", "immune_label", "targeted_response", "tumour_burden",
                "MKI67", "recommendation", "rationale", "OS_MONTHS", "SURV_STATUS"]
    df[out_cols].to_csv(os.path.join(RESULTS, "triage_assignments.csv"), index=False)

    # ── Cohort distribution ───────────────────────────────────────────────────
    print(f"\n[1] Cohort: {len(df)} patients")
    print(f"    thresholds: immune median={imm_thr:.3f}, burden median={burden_thr:.3f}")
    dist = df["recommendation"].value_counts()
    print("\n[2] First-line recommendation distribution:")
    for rec, n in dist.items():
        print(f"    {rec:<28} {n:4d}  ({n/len(df)*100:4.1f}%)")

    print("\n[3] Recommendation by molecular subtype:")
    print(pd.crosstab(df["subtype"], df["recommendation"]).to_string())

    # ── Survival by recommendation group (are the strata prognostic?) ─────────
    surv = df.dropna(subset=["OS_MONTHS", "SURV_STATUS"]).copy()
    lr = multivariate_logrank_test(surv["OS_MONTHS"], surv["recommendation"],
                                   surv["SURV_STATUS"])
    print(f"\n[4] Overall survival differs across recommendation groups: "
          f"log-rank p = {lr.p_value:.3e}")

    fig, ax = plt.subplots(figsize=(9, 6))
    kmf = KaplanMeierFitter()
    med_rows = []
    for rec in [IO_FIRST, TGT_TO_IO, TGT_FIRST, TRIAL_CHEMO]:
        sub = surv[surv["recommendation"] == rec]
        if len(sub) < 5:
            continue
        kmf.fit(sub["OS_MONTHS"], sub["SURV_STATUS"],
                label=f"{rec} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, color=REC_COLORS[rec], lw=2.3, ci_show=False)
        med_rows.append((rec, len(sub), kmf.median_survival_time_))
    ax.set_title(f"Overall survival by recommended first-line strategy\n"
                 f"TCGA-SKCM (log-rank p = {lr.p_value:.2e})",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Overall survival (months)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "triage_km_by_recommendation.png"),
                dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(PLOTS, "triage_km_by_recommendation.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    # ── Decision-matrix figure (BRAF status x immune status) ──────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    cats_x = ["BRAF-V600", "BRAF wild-type"]
    cats_y = ["immune-hot", "immune-cold"]
    # recommendation shown in each cell (burden splits BRAF/hot cell -> show dominant)
    cell_text = {
        ("BRAF-V600", "immune-hot"):  "Immunotherapy first\n(targeted if high burden)",
        ("BRAF-V600", "immune-cold"): "Targeted therapy first\n(BRAF/MEK inhibitor)",
        ("BRAF wild-type", "immune-hot"):  "Immunotherapy",
        ("BRAF wild-type", "immune-cold"): "Trial / Chemotherapy\n(hardest group)",
    }
    cell_color = {
        ("BRAF-V600", "immune-hot"):  "#2563EB",
        ("BRAF-V600", "immune-cold"): "#DC2626",
        ("BRAF wild-type", "immune-hot"):  "#2563EB",
        ("BRAF wild-type", "immune-cold"): "#6B7280",
    }
    for i, cy in enumerate(cats_y):
        for j, cx in enumerate(cats_x):
            # count patients in this cell
            if cx == "BRAF-V600":
                mask = df["braf_v600"]
            else:
                mask = ~df["braf_v600"]
            mask = mask & (df["immune_label"] == cy)
            n = int(mask.sum())
            ax2.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=cell_color[(cx, cy)],
                                        alpha=0.18, edgecolor="black", lw=1.5))
            ax2.text(j + 0.5, 1 - i + 0.62, cell_text[(cx, cy)], ha="center", va="center",
                     fontsize=11, fontweight="bold")
            ax2.text(j + 0.5, 1 - i + 0.22, f"n = {n} ({n/len(df)*100:.0f}%)",
                     ha="center", va="center", fontsize=10, color="#333")
    ax2.set_xlim(0, 2); ax2.set_ylim(0, 2)
    ax2.set_xticks([0.5, 1.5]); ax2.set_xticklabels(cats_x, fontsize=11)
    ax2.set_yticks([1.5, 0.5]); ax2.set_yticklabels(cats_y, fontsize=11)
    ax2.set_title("Melanoma first-line treatment triage matrix\n"
                  "BRAF status (targeted axis) x immune infiltration (IO axis)",
                  fontweight="bold", fontsize=12)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.tick_params(length=0)
    fig2.tight_layout()
    fig2.savefig(os.path.join(PLOTS, "triage_decision_matrix.png"),
                 dpi=150, bbox_inches="tight")
    fig2.savefig(os.path.join(PLOTS, "triage_decision_matrix.pdf"),
                 bbox_inches="tight")
    plt.close(fig2)

    # ── Text summary ──────────────────────────────────────────────────────────
    lines = ["Q5 Treatment-Selection Triage — TCGA-SKCM Melanoma", "=" * 55,
             f"Patients: {len(df)}", "",
             "First-line recommendation distribution:"]
    for rec, n in dist.items():
        lines.append(f"  {rec:<28} {n:4d}  ({n/len(df)*100:4.1f}%)")
    lines += ["", "Median OS by recommendation group:"]
    for rec, n, med in med_rows:
        lines.append(f"  {rec:<28} n={n:<4} median OS = {med:.1f} months")
    lines += ["", f"Log-rank across groups: p = {lr.p_value:.3e}",
              "", "Recommendation by subtype:", pd.crosstab(df['subtype'],
              df['recommendation']).to_string()]
    with open(os.path.join(RESULTS, "triage_summary.txt"), "w") as f:
        f.write("\n".join(lines))

    print("\n[5] Saved:")
    print(f"    {os.path.join(RESULTS, 'triage_assignments.csv')}")
    print(f"    {os.path.join(RESULTS, 'triage_summary.txt')}")
    print(f"    {os.path.join(PLOTS, 'triage_decision_matrix.png')}")
    print(f"    {os.path.join(PLOTS, 'triage_km_by_recommendation.png')}")
    print("\n" + "=" * 62)
    print("  Q5 triage complete.")
    print("=" * 62)


if __name__ == "__main__":
    main()
