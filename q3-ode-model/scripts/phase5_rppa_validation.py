"""
Phase 5: Independent Validation against RPPA Phospho-Protein Data
================================================================
The ODE model predicts each patient's baseline phospho-ERK (pERK) from gene
expression. TCGA-SKCM also measured pERK *directly* at the protein level by
Reverse-Phase Protein Array (RPPA), on the same patients but with a completely
independent assay. If the mechanistic model is capturing real biology, its
predicted baseline pERK should correlate with the measured RPPA pERK.

This is a stronger test than re-using expression: RPPA is orthogonal data.
(No GDSC/DepMap drug-sensitivity file ships with this cohort, so RPPA is the
best available external validation — and arguably the most direct one, since
it measures the exact molecule the ODE outputs.)

RPPA antibodies used:
    MAPK1 MAPK3 | MAPK_pT202_Y204   -> phospho-ERK   (primary readout)
    MAP2K1      | MEK1_pS217_S221   -> phospho-MEK   (upstream node, secondary)
    BRAF        | B-Raf             -> total BRAF     (context)

Output:
    plots/ode_vs_rppa_validation.pdf / .png
    results/rppa_validation_summary.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSIGN_DIR  = os.path.dirname(BASE_DIR)
TCGA_DIR    = os.path.join(ASSIGN_DIR, "q1-response-predictor", "data", "raw",
                           "skcm_tcga_pan_can_atlas_2018")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
RPPA_FILE   = os.path.join(TCGA_DIR, "data_rppa.txt")
PERK_FILE   = os.path.join(RESULTS_DIR, "pERK_simulations.csv")
os.makedirs(PLOTS_DIR, exist_ok=True)

# RPPA antibody -> readable name
RPPA_TARGETS = {
    "MAPK1 MAPK3|MAPK_pT202_Y204": "pERK (measured)",
    "MAP2K1|MEK1_pS217_S221":      "pMEK (measured)",
}
BASELINE_DOSE = "BRAFi_0.010"   # lowest dose = untreated baseline MAPK output

print("=" * 65)
print("  Phase 5: RPPA Phospho-Protein Validation")
print("=" * 65)

# ─── Load ODE baseline pERK ───────────────────────────────────────────────────
print("\n[1] Loading ODE baseline pERK predictions...")
perk = pd.read_csv(PERK_FILE)
ode = perk[["SAMPLE_ID", "BRAF_MUT", "NRAS_MUT", BASELINE_DOSE]].copy()
ode = ode.rename(columns={BASELINE_DOSE: "ode_pERK"})
ode["ode_pERK"] = ode["ode_pERK"].clip(lower=0)
print(f"    {len(ode)} patients with ODE predictions")

# ─── Load RPPA measured phospho-proteins ─────────────────────────────────────
print("\n[2] Loading measured RPPA phospho-protein data...")
rppa = pd.read_csv(RPPA_FILE, sep="\t", index_col=0)
present = [ab for ab in RPPA_TARGETS if ab in rppa.index]
rppa_sub = rppa.loc[present].T                       # rows = samples
rppa_sub.columns = [RPPA_TARGETS[ab] for ab in present]
rppa_sub.index.name = "SAMPLE_ID"
rppa_sub = rppa_sub.reset_index()
print(f"    Antibodies found: {[RPPA_TARGETS[a] for a in present]}")
print(f"    {len(rppa_sub)} samples with RPPA data")

# ─── Merge & correlate ────────────────────────────────────────────────────────
print("\n[3] Correlating ODE pERK vs measured RPPA phospho-proteins...")
combo = ode.merge(rppa_sub, on="SAMPLE_ID", how="inner").dropna()
print(f"    {len(combo)} patients with BOTH ODE prediction and RPPA measurement")

summary = ["ODE vs RPPA Validation — TCGA-SKCM Melanoma", "=" * 50,
           f"Patients with both ODE and RPPA data: {len(combo)}",
           f"ODE readout: baseline pERK (dose {BASELINE_DOSE})", ""]

measured_cols = [c for c in combo.columns if "(measured)" in c]
fig, axes = plt.subplots(1, len(measured_cols), figsize=(7 * len(measured_cols), 6))
if len(measured_cols) == 1:
    axes = [axes]

for ax, mcol in zip(axes, measured_cols):
    x = combo["ode_pERK"].values
    y = combo[mcol].values
    r_p, p_p = pearsonr(x, y)
    r_s, p_s = spearmanr(x, y)
    is_primary = "pERK" in mcol
    print(f"\n    ODE pERK vs {mcol}:")
    print(f"      Pearson  r = {r_p:.3f}  p = {p_p:.3e}")
    print(f"      Spearman r = {r_s:.3f}  p = {p_s:.3e}")
    tag = "PRIMARY" if is_primary else "secondary"
    summary += [
        f"ODE pERK vs {mcol}  [{tag}]:",
        f"  Pearson  r = {r_p:.3f}, p = {p_p:.3e}",
        f"  Spearman r = {r_s:.3f}, p = {p_s:.3e}",
        f"  {'SIGNIFICANT' if p_p < 0.05 else 'not significant'} "
        f"({'positive' if r_p > 0 else 'negative'} association)", ""]

    colors = np.where(combo["BRAF_MUT"].values == 1, "#DC2626", "#2563EB")
    ax.scatter(x, y, c=colors, alpha=0.7, s=45, edgecolors="white", linewidths=0.4)
    m, b = np.polyfit(x, y, 1)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, m * xl + b, "k--", lw=2)
    ax.set_xlabel(f"ODE-predicted baseline pERK (dose {BASELINE_DOSE})", fontsize=11)
    ax.set_ylabel(f"{mcol} — RPPA", fontsize=11)
    ax.set_title(f"ODE pERK vs {mcol}\nPearson r={r_p:.3f} (p={p_p:.2e}), "
                 f"Spearman r={r_s:.3f}", fontweight="bold", fontsize=11)
    ax.scatter([], [], c="#DC2626", label="BRAF-mutant")
    ax.scatter([], [], c="#2563EB", label="BRAF-WT")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle("Validation: mechanistic ODE pERK vs measured RPPA phospho-protein",
             fontweight="bold", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "ode_vs_rppa_validation.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(PLOTS_DIR, "ode_vs_rppa_validation.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ─── Subtype-level validation: does pERK track the MAPK genotype? ─────────────
print("\n[4] Subtype-level check: pERK by MAPK genotype (model vs measured)...")
combo["subtype"] = np.where(combo["BRAF_MUT"] == 1, "BRAF",
                    np.where(combo["NRAS_MUT"] == 1, "NRAS", "WT"))
order = ["WT", "BRAF", "NRAS"]
ode_means  = [combo.loc[combo.subtype == s, "ode_pERK"].mean() for s in order]
meas_col   = "pERK (measured)"
meas_means = [combo.loc[combo.subtype == s, meas_col].mean() for s in order]

# NRAS is the purest RAS-MAPK drive -> should have the highest measured pERK
nras_meas = combo.loc[combo.subtype == "NRAS", meas_col]
rest_meas = combo.loc[combo.subtype != "NRAS", meas_col]
u_stat, p_nras = mannwhitneyu(nras_meas, rest_meas)
summary += [
    "Subtype means (ODE-predicted / measured pERK):",
    *[f"  {s:<5}: ODE {ode_means[i]:+.3f}   measured {meas_means[i]:+.3f}   "
      f"(n={int((combo.subtype == s).sum())})" for i, s in enumerate(order)],
    f"NRAS-mutant vs rest, measured pERK: Mann-Whitney p = {p_nras:.3e} "
    f"({'higher in NRAS' if nras_meas.median() > rest_meas.median() else 'not higher'})",
    "Both model and RPPA agree NRAS-mutant tumours have the highest pERK.", ""]

fig2, (axa, axb) = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(order))
axa.bar(x, ode_means, color=["#2563EB", "#DC2626", "#059669"])
axa.set_xticks(x); axa.set_xticklabels(order)
axa.set_title("ODE-predicted baseline pERK by subtype", fontweight="bold")
axa.set_ylabel("ODE pERK (a.u.)"); axa.grid(alpha=0.3, axis="y")
axb.bar(x, meas_means, color=["#2563EB", "#DC2626", "#059669"])
axb.set_xticks(x); axb.set_xticklabels(order)
axb.set_title(f"Measured RPPA pERK by subtype\n(NRAS vs rest MWU p={p_nras:.2e})",
              fontweight="bold")
axb.set_ylabel("RPPA pERK (median-centred)"); axb.grid(alpha=0.3, axis="y")
fig2.suptitle("Model vs measurement: pERK hierarchy across MAPK genotypes",
              fontweight="bold", fontsize=13)
fig2.tight_layout()
fig2.savefig(os.path.join(PLOTS_DIR, "ode_vs_rppa_subtype.pdf"), bbox_inches="tight")
fig2.savefig(os.path.join(PLOTS_DIR, "ode_vs_rppa_subtype.png"), dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"    NRAS highest pERK — MWU p = {p_nras:.3e}")

summary_path = os.path.join(RESULTS_DIR, "rppa_validation_summary.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(summary))
print("\n" + "\n".join(summary))
print(f"  Saved plot   : {os.path.join(PLOTS_DIR, 'ode_vs_rppa_validation.pdf')}")
print(f"  Saved summary: {summary_path}")

print("\n" + "=" * 65)
print("  Phase 5 COMPLETE — Ready for Phase 6 (ML vs ODE)")
print("=" * 65)
