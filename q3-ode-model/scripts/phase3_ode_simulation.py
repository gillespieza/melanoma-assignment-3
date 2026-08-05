"""
Phase 3: Mechanistic Melanoma ODE Simulation
============================================
Every rate constant comes from an established published model. Kinetic
constants are identical for every patient; only INPUTS vary per patient —
protein abundances (from RSEM expression), mutation state (RAS-GTP level /
BRAF-V600E monomer), and drug dose.

Four coupled modules with an explicit fast/slow timescale split (a
quasi-steady-state approximation):

  MODULE A — RAF dimerisation + drug binding  (the RAF-inhibitor paradox)
    RAF signals as a dimer; RAS-GTP drives dimerisation; a RAF inhibitor
    (vemurafenib) binds a protomer, but (i) a singly-liganded dimer keeps its
    drug-free protomer catalytically active (transactivation), (ii) the second
    site binds with strong NEGATIVE cooperativity, and (iii) inhibitor binding
    PROMOTES dimerisation. Paradoxical ERK activation in RAS-mutant /
    BRAF-wild-type tumours therefore EMERGES from the binding equilibria — it
    is not hard-coded with an if-statement. A BRAF-V600E monomer signals
    RAS-independently and is directly inhibited by the drug (monotonic
    suppression). Output: effective active-RAF drive -> sets V1 of Module B.

  MODULE B — MAPK cascade with negative feedback  (fast, minutes)
    Eight-state Raf/MEK/ERK cascade (each un-, mono-, di-phosphorylated) with
    di-phospho-ERK feeding back to inhibit the top of the cascade (constant
    Ki = 9 nM, the same for every patient). Published kinetic constants.
    Readout: steady-state (time-averaged) di-phospho-ERK == pERK.

  MODULE C — Melanoma tumour / immune dynamics  (slow, days)
    Published melanoma tumour-immune model (rate constants in day^-1). The
    baseline cancer equation is

      dC/dt = lambdaC*C*(1 - C/CM) * [drug factor] - (eta8*T8)*C - dC*C

    Instead of a phenomenological drug factor, proliferation is scaled by the
    mechanistic pERK output of Module B (pERK / pERK_ref). Because that pERK
    carries the RAF paradox from Module A, the drug shrinks BRAF-V600E tumours
    but not NRAS/WT tumours. CD8 killing uses the patient's measured
    infiltration (CD8A/PRF1/GZMA) as the effector level, scaled by cytolytic
    capacity (Module D) and gated by checkpoint suppression (Module D). Solved
    on its own slow timescale with pERK as a fixed input (quasi-steady-state),
    so the fast cascade is not co-integrated with the slow tumour dynamics.
    Readout: steady-state tumour burden C (g/cm^3).

  MODULE D — Checkpoint axis + diallable anti-PD-1  (steady state)
    Minimal PD-1/PD-L1 binding equilibrium (Lai et al. 2017, BMC Syst Biol
    11:70 — the same paper Module C's tumour-immune constants come from).
    Anti-PD-1 drug (A) competitively depletes free PD-1 (P) before it can form
    the inhibitory complex (Q) with PD-L1 (L); Q suppresses CD8 killing.
    f_kill = 1 - Q/P_tot is the fraction of killing capacity NOT blocked by the
    checkpoint (1.0 = fully unleashed, 0.0 = fully suppressed). At drug = 0,
    f_kill reflects the patient's own baseline PD-1/PD-L1 burden — this is why
    Module C's killing term is no longer a fixed universal eta8*T8: it is
    gated by each patient's checkpoint state even before any drug is given.
    Also implements the CYT "speed limit" (Rooney et al. 2015): even a fully
    unleashed (f_kill=1) tumour can only be killed as fast as the patient's
    baseline GZMA/PRF1 cytolytic machinery allows.
    Readout: tumour burden under a vemurafenib-only sweep (checkpoint held at
    each patient's untreated baseline) AND under an anti-PD-1-only sweep
    (BRAFi held at zero) — the two arms of the Q5 treatment decision tree.

Per-patient INPUTS only (kinetics are fixed):
  * protein totals  : Raf<-BRAF, MEK<-mean(MAP2K1,MAP2K2), ERK<-mean(MAPK1,MAPK3)
  * RAS-GTP level   : high if NRAS-activating mutation, basal otherwise
  * BRAF-V600E pool : present iff BRAF-V600 activating mutation
  * infiltration    : mean(CD8A, PRF1, GZMA)
  * cytolytic level : CYT = mean(GZMA, PRF1)                    (Module D)
  * checkpoint pool : PD-1 <- PDCD1, PD-L1 <- CD274 expression  (Module D)
  * drug dose       : vemurafenib concentration (nM) AND/OR anti-PD-1 (nM)

Output:
    results/pERK_simulations.csv              pERK per patient/BRAFi-dose
    results/tumour_burden_simulations.csv     tumour burden per patient/BRAFi-dose
    results/checkpoint_tumour_simulations.csv tumour burden per patient/anti-PD-1-dose
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from multiprocessing import Pool, cpu_count

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE   = os.path.join(BASE_DIR, "data", "melanoma_params_full.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_PERK       = os.path.join(RESULTS_DIR, "pERK_simulations.csv")
OUT_TUMOUR     = os.path.join(RESULTS_DIR, "tumour_burden_simulations.csv")
OUT_CHECKPOINT = os.path.join(RESULTS_DIR, "checkpoint_tumour_simulations.csv")

# ─── Simulation grid ──────────────────────────────────────────────────────────
DOSE_UNITS     = np.linspace(0.01, 1.0, num=10)      # normalised dose knob
DRUG_MAX_NM    = 1000.0                              # u = 1.0  ->  1 uM vemurafenib
BRAFi_dose     = DOSE_UNITS                          # kept for column naming
DOSE_COL_NAMES = ["BRAFi_" + f"{v:.3f}" for v in DOSE_UNITS]

# Anti-PD-1 dose sweep (Module D). Kept as its own grid/output — a full
# BRAFi x anti-PD-1 combination grid is deliberately deferred
ANTIPD1_DOSE_UNITS = np.linspace(0.01, 1.0, num=10)  # normalised dose knob
ANTIPD1_MAX_NM     = 500.0                           # u = 1.0 -> 500 nM anti-PD-1
ANTIPD1_COL_NAMES  = ["antiPD1_" + f"{v:.3f}" for v in ANTIPD1_DOSE_UNITS]

# ══════════════════════════════════════════════════════════════════════════════
#  UNIVERSAL PARAMETERS (identical for every patient) — all based on the literature
# ══════════════════════════════════════════════════════════════════════════════

# ── Module B: MAPK cascade — published kinetic constants (nM, min) ────────────
# Feedback constant Ki is universal (identical for every patient).
KH = dict(
    V1=2.5, Ki=9.0, n1=1.0, K1=10.0,
    V2=0.25, K2=8.0,
    k3=0.025, K3=15.0,
    k4=0.025, K4=15.0,
    V5=0.75, K5=15.0,
    V6=0.75, K6=15.0,
    k7=0.025, K7=15.0,
    k8=0.025, K8=15.0,
    V9=0.5, K9=15.0,
    V10=0.5, K10=15.0,
)
RAF_TOTAL_0 = 100.0    # nM, total Raf   — scaled per patient by expression
MEK_TOTAL_0 = 300.0    # nM, total MEK   — scaled per patient by expression
ERK_TOTAL_0 = 300.0    # nM, total ERK   — scaled per patient by expression

# ── Module C: melanoma tumour-immune — published rate constants (per day) ─────
LF = dict(
    lambdaC=0.616,   # cancer-cell growth rate (day^-1)
    dC=0.17,         # cancer-cell death rate (day^-1)
    CM=0.8,          # carrying capacity (g/cm^3)
    eta8=46.0,       # CD8+ T-cell killing rate (day^-1 cm^3/g)
    dT8=0.18,        # CD8+ T-cell death rate (day^-1)  [context]
)
# Maps the dimensionless infiltration score (cohort mean ~1) onto the model's
# effector-density units so that eta8*T8 is commensurate with lambdaC. This is
# the single calibration constant of Module C (an input scaling, not a kinetic
# rate); all kinetic constants above are based on published values.
T8_SCALE = 0.008

# ── Module A: reduced RAF-dimer + drug module ─────────────────────────────────
# A minimal mechanistic reduction with the correct topology; the allosteric
# constants are literature-motivated order-of-magnitude values (vemurafenib
# biochemistry + negative cooperativity + drug-induced dimerisation).
KD_RAF   = 50.0    # nM, vemurafenib first-protomer affinity (biochemical ~30-100)
COOP     = 20.0    # negative cooperativity: 2nd-site Kd = COOP * KD_RAF
GAMMA    = 4.0     # drug-induced dimerisation strength (relief of autoinhibition)
KDIM     = 1.0     # dimerisation scaling per unit RAS-GTP
# RAS-GTP level by subtype (a per-patient INPUT, not a kinetic constant):
#   NRAS-mutant  -> high constitutive RAS-GTP (drives feedback-resistant dimers)
#   BRAF-V600E   -> LOW RAS-GTP: V600E signals RAS-independently and ERK feedback
#                   suppresses RAS, so V600E tumours are RAS-low & dimer-poor,
#                   hence directly drug-sensitive via the monomer.
#   wild-type    -> basal RAS-GTP.
RASGTP_NRAS  = 0.9
RASGTP_BASAL = 0.3
RASGTP_V600E = 0.05

# ── Coupling constants (Module B pERK -> Module C proliferation) ──────────────
PERK_PROLIF_CAP = 3.0    # max fold-change of tumour growth from ERK
PERK_PROLIF_MIN = 0.05   # floor (avoids zero growth)

# ── Module D: checkpoint axis (PD-1 / PD-L1 / complex) + anti-PD-1 ────────────
# Minimal steady-state binding model (Lai et al. 2017, BMC Syst Biol 11:70 —
# the same paper Module C's lambdaC/dC/CM/eta8/dT8 constants are taken from).
# Anti-PD-1 (drug A, nM) competitively occupies PD-1 (P) before it can bind
# PD-L1 (L) to form the inhibitory complex (Q). Like Module A's KD_RAF/COOP,
# these binding constants are literature-motivated order-of-magnitude values,
# not separately fit.
KD_PA        = 5.0     # nM, anti-PD-1 affinity for PD-1 (biochemical, ~nM range)
KD_PL        = 8.0     # nM, PD-1-PD-L1 complex affinity (published range)
PDCD1_TOTAL_0 = 50.0    # nM, total PD-1 pool at mean PDCD1 expression (=1.0)
PDL1_TOTAL_0  = 50.0    # nM, total PD-L1 pool at mean CD274 expression (=1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE A — RAF dimerisation + drug  ->  effective active-RAF drive
# ══════════════════════════════════════════════════════════════════════════════
def active_raf_signal(drug_nM, raf_total, rasgtp, v600e_frac):
    """
    Effective active-RAF kinase drive as a function of drug concentration.

    The RAF-inhibitor paradox emerges from binding equilibria, not a switch:
      - dimer occupancy states (0/1/2 drugs) via sequential binding with
        negative cooperativity (2nd site much weaker),
      - a singly-liganded dimer keeps ONE drug-free, transactivated protomer,
      - drug PROMOTES dimerisation (autoinhibition relief), factor (1 + GAMMA*p).
    A BRAF-V600E monomer signals RAS-independently and is inhibited directly.
    """
    a = drug_nM / KD_RAF               # first-site occupancy ratio
    b = drug_nM / (COOP * KD_RAF)      # second-site (negatively cooperative)
    p = drug_nM / (KD_RAF + drug_nM)   # protomer occupancy probability

    # Dimer occupancy partition function (states: 0, 1, 2 drugs bound)
    Z = 1.0 + 2.0 * a + a * b
    # Mean number of ACTIVE (drug-free) protomers per dimer:
    #   state0 -> 2 active,  state1 -> 1 active (transactivated),  state2 -> 0
    active_per_dimer = (2.0 * 1.0 + 1.0 * (2.0 * a)) / Z          # = (2 + 2a)/Z

    # Dimer abundance rises with RAS-GTP and is further promoted by drug
    dimer_amount = KDIM * rasgtp * raf_total * (1.0 + GAMMA * p)
    A_dimer = dimer_amount * active_per_dimer

    # BRAF-V600E active monomer, directly inhibited by drug (monotonic)
    A_mono = v600e_frac * raf_total * (KD_RAF / (KD_RAF + drug_nM))

    return A_mono + A_dimer


# Reference RAF drive (wild-type, basal RAS-GTP, mean expression, no drug) —
# used to normalise Module A output onto the published baseline V1 = 2.5.
_A_REF = active_raf_signal(0.0, RAF_TOTAL_0, RASGTP_BASAL, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE D — checkpoint axis (PD-1/PD-L1 binding equilibrium) + anti-PD-1
# ══════════════════════════════════════════════════════════════════════════════
def checkpoint_kill_factor(drug_nM, pdcd1_pool, pdl1_pool):
    """
    Fraction of CD8 killing capacity NOT blocked by the PD-1/PD-L1 checkpoint.

    P_free : PD-1 remaining after anti-PD-1 competitively occupies some of it.
    Q      : inhibitory PD-1/PD-L1 complex formed from the remaining free PD-1.
    f_kill = 1 - (Q / P_tot)  ->  1.0 = fully unleashed, 0.0 = fully suppressed.

    At drug_nM = 0 this still returns < 1 whenever the patient's own PD-L1
    pool is non-trivial relative to KD_PL — i.e. the checkpoint is already
    partly "on" from the tumour's baseline biology, exactly as in Lai et al.
    Increasing drug_nM depletes P_free, which depletes Q, which raises f_kill.
    """
    P_free = pdcd1_pool / (1.0 + drug_nM / KD_PA)
    Q = pdl1_pool * P_free / (KD_PL + P_free)
    Q = min(Q, pdcd1_pool)                       # Q cannot exceed total PD-1
    return 1.0 - (Q / max(pdcd1_pool, 1e-9))


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE B — MAPK cascade (fast)
# ══════════════════════════════════════════════════════════════════════════════
def mapk_cascade_rhs(y, t, V1_eff, RAF_T, MEK_T, ERK_T):
    """
    Eight-state MAPK cascade with published parameters.
    y = [Raf, RafP, MEK, MEKp, MEKpp, ERK, ERKp, ERKpp]
    Di-phospho-ERK (ERKpp) feeds back to inhibit Raf activation (v1).
    V1_eff is the effective Raf-activation Vmax set by Module A.
    """
    Raf, RafP, MEK, MEKp, MEKpp, ERK, ERKp, ERKpp = y
    P = KH

    # v1: Raf activation, inhibited by ERKpp (universal feedback, Ki=9, n=1)
    v1 = V1_eff * Raf / ((1.0 + (ERKpp / P["Ki"]) ** P["n1"]) * (P["K1"] + Raf))
    v2 = P["V2"] * RafP / (P["K2"] + RafP)

    v3 = P["k3"] * RafP * MEK  / (P["K3"] + MEK)
    v4 = P["k4"] * RafP * MEKp / (P["K4"] + MEKp)
    v5 = P["V5"] * MEKpp / (P["K5"] + MEKpp)
    v6 = P["V6"] * MEKp  / (P["K6"] + MEKp)

    v7 = P["k7"] * MEKpp * ERK  / (P["K7"] + ERK)
    v8 = P["k8"] * MEKpp * ERKp / (P["K8"] + ERKp)
    v9 = P["V9"] * ERKpp / (P["K9"] + ERKpp)
    v10 = P["V10"] * ERKp / (P["K10"] + ERKp)

    return [
        v2 - v1,               # Raf
        v1 - v2,               # RafP
        v6 - v3,               # MEK
        v3 + v5 - v4 - v6,     # MEKp
        v4 - v5,               # MEKpp
        v10 - v7,              # ERK
        v7 + v9 - v8 - v10,    # ERKp
        v8 - v9,               # ERKpp  (== pERK readout)
    ]


# Fast-module time grid (cascade native units, ~minutes). Long enough to reach
# the attractor; we time-average the tail to handle oscillatory cases.
FAST_T = np.linspace(0.0, 4000.0, 800)
FAST_TAIL = 200  # average ERKpp over the last 200 points


def steady_pERK(V1_eff, RAF_T, MEK_T, ERK_T):
    """Integrate the fast cascade and return time-averaged di-phospho-ERK."""
    y0 = [RAF_T, 0.0, MEK_T, 0.0, 0.0, ERK_T, 0.0, 0.0]
    try:
        sol = odeint(mapk_cascade_rhs, y0, FAST_T,
                     args=(V1_eff, RAF_T, MEK_T, ERK_T),
                     rtol=1e-6, atol=1e-8, mxstep=5000)
        return float(np.mean(sol[-FAST_TAIL:, 7]))
    except Exception:
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE C — melanoma tumour / immune (slow, QSS in pERK)
# ══════════════════════════════════════════════════════════════════════════════
SLOW_T = np.linspace(0.0, 200.0, 400)   # days


def cancer_rhs(C, t, prolif, T8, f_kill, eta8_i):
    """
    Reduced melanoma cancer-cell ODE (published constants). Proliferation is
    scaled by `prolif` (= pERK/pERK_ref from the mechanistic cascade), so the
    RAF paradox governs whether the drug actually suppresses proliferation.
    Killing is scaled by `f_kill` (Module D checkpoint gate: 0=fully blocked,
    1=fully unleashed) and by `eta8_i` (this patient's CYT-scaled killing
    ceiling), instead of the universal LF["eta8"].
    """
    C = max(C[0], 0.0)
    dC = (LF["lambdaC"] * prolif * C * (1.0 - C / LF["CM"])
          - eta8_i * f_kill * T8 * C
          - LF["dC"] * C)
    return [dC]


def steady_tumour(pERK, pERK_ref, infil, f_kill, cyt_level, cyt_ref):
    """
    Slow melanoma tumour steady state, using fast-module pERK as a fixed input.
    ERK drives proliferation; measured infiltration sets the CD8 effector
    density. Killing capacity is gated by the Module D checkpoint state
    (`f_kill`) and capped by this patient's cytolytic machinery relative to
    the cohort (`cyt_level / cyt_ref`, Rooney et al. 2015) — even a fully
    unleashed checkpoint (f_kill=1) cannot kill faster than CYT allows.
    """
    prolif = np.clip(pERK / max(pERK_ref, 1e-9), PERK_PROLIF_MIN, PERK_PROLIF_CAP)
    T8 = T8_SCALE * max(infil, 0.0)          # CD8 effector density from infiltration
    eta8_i = LF["eta8"] * max(cyt_level, 0.0) / max(cyt_ref, 1e-9)
    try:
        sol = odeint(cancer_rhs, [0.1 * LF["CM"]], SLOW_T,
                     args=(prolif, T8, f_kill, eta8_i), rtol=1e-6, atol=1e-9, mxstep=5000)
        return float(max(sol[-1, 0], 0.0))
    except Exception:
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
#  Per-patient driver
# ══════════════════════════════════════════════════════════════════════════════
# Cohort-reference pERK/CYT (wild-type baseline / cohort mean) are computed in
# __main__ and injected.
def simulate_patient(args):
    """
    Simulate one patient across the BRAFi dose sweep AND the anti-PD-1 dose
    sweep. Returns (i, pERK_row, tumour_row, checkpoint_row).
    """
    i_patient, row, pERK_ref, cyt_ref = args

    # Per-patient INPUTS (never kinetic constants):
    RAF_T = RAF_TOTAL_0 * float(row["BRAF"])
    MEK_T = MEK_TOTAL_0 * 0.5 * (float(row["MAP2K1"]) + float(row["MAP2K2"]))
    ERK_T = ERK_TOTAL_0 * 0.5 * (float(row["MAPK1"]) + float(row["MAPK3"]))

    braf_v600e = int(row["BRAF_MUT"]) == 1
    nras_mut   = int(row["NRAS_MUT"]) == 1
    if braf_v600e:
        rasgtp = RASGTP_V600E      # V600E is RAS-low (RAS-independent, feedback)
    elif nras_mut:
        rasgtp = RASGTP_NRAS       # NRAS drives high RAS-GTP
    else:
        rasgtp = RASGTP_BASAL      # wild-type basal
    v600e_frac = 1.0 if braf_v600e else 0.0
    raf_amount = RAF_TOTAL_0 * float(row["BRAF"])     # RAF pool for Module A
    infil      = (float(row["CD8A"]) + float(row["PRF1"]) + float(row["GZMA"])) / 3.0

    # Module D per-patient inputs: checkpoint pools and cytolytic level.
    pdcd1_pool = PDCD1_TOTAL_0 * float(row["PDCD1"])
    pdl1_pool  = PDL1_TOTAL_0 * float(row["CD274"])
    cyt_level  = float(row["CYT"])

    # ── BRAFi-only sweep: checkpoint held at this patient's untreated
    # baseline (no anti-PD-1) — isolates the targeted-therapy arm, as before,
    # but killing is now checkpoint-gated and CYT-scaled instead of universal.
    f_kill_baseline = checkpoint_kill_factor(0.0, pdcd1_pool, pdl1_pool)
    perk_row, tumour_row = [], []
    for u in DOSE_UNITS:
        drug_nM = u * DRUG_MAX_NM
        A = active_raf_signal(drug_nM, raf_amount, rasgtp, v600e_frac)
        V1_eff = KH["V1"] * A / _A_REF          # normalise onto published V1=2.5
        pERK = steady_pERK(V1_eff, RAF_T, MEK_T, ERK_T)
        tumour = steady_tumour(pERK, pERK_ref, infil, f_kill_baseline, cyt_level, cyt_ref)
        perk_row.append(pERK)
        tumour_row.append(tumour)

    # ── Anti-PD-1-only sweep: BRAFi held at zero (untreated pERK), checkpoint
    # dose scanned — isolates the immunotherapy arm of the Q5 decision tree.
    pERK_untreated = perk_row[0]
    checkpoint_row = []
    for u in ANTIPD1_DOSE_UNITS:
        checkpoint_dose_nM = u * ANTIPD1_MAX_NM
        f_kill = checkpoint_kill_factor(checkpoint_dose_nM, pdcd1_pool, pdl1_pool)
        checkpoint_row.append(
            steady_tumour(pERK_untreated, pERK_ref, infil, f_kill, cyt_level, cyt_ref))

    return i_patient, perk_row, tumour_row, checkpoint_row


def compute_reference_pERK():
    """
    Cohort reference pERK: wild-type inputs at mean expression and zero drug.
    Used to normalise the ERK->proliferation coupling in Module C.
    """
    A = active_raf_signal(0.0, RAF_TOTAL_0, RASGTP_BASAL, 0.0)
    V1_eff = KH["V1"] * A / _A_REF              # == V1 by construction
    return steady_pERK(V1_eff, RAF_TOTAL_0, MEK_TOTAL_0, ERK_TOTAL_0)


if __name__ == "__main__":
    print("=" * 68)
    print("  Phase 3: Mechanistic Melanoma ODE (signalling + tumour-immune)")
    print("=" * 68)

    print(f"\n[1] Loading patient data: {DATA_FILE}")
    patients = pd.read_csv(DATA_FILE)
    n_patients = len(patients)
    n_mut = int(patients["BRAF_MUT"].sum())
    print(f"    {n_patients} patients  ({n_mut} BRAF-mutant, {n_patients - n_mut} WT)")

    required_cols = ["PDCD1", "CD274", "CYT"]
    missing_cols = [c for c in required_cols if c not in patients.columns]
    if missing_cols:
        raise SystemExit(
            f"Missing checkpoint-axis columns {missing_cols} in {DATA_FILE}. "
            "Re-run phase2_preprocess_data.py (updated for Module D) first.")

    print("\n[2] Computing cohort-reference pERK (wild-type baseline) and CYT...")
    pERK_ref = compute_reference_pERK()
    cyt_ref = float(patients["CYT"].mean())
    print(f"    reference pERK (ERKpp) = {pERK_ref:.3f} nM  "
          f"(cascade at published V1={KH['V1']})")
    print(f"    reference CYT (cohort mean) = {cyt_ref:.3f}  (Rooney et al. 2015)")

    n_cores = min(cpu_count(), n_patients)
    print(f"\n[3] Simulating on {n_cores} CPU cores")
    print(f"    BRAFi dose u in [0.01, 1.0] -> vemurafenib 0..{DRUG_MAX_NM:.0f} nM")
    print(f"    anti-PD-1 dose u in [0.01, 1.0] -> 0..{ANTIPD1_MAX_NM:.0f} nM")
    print(f"    fast module: 8-ODE MAPK cascade (feedback Ki={KH['Ki']})")
    print(f"    slow module: melanoma tumour-immune (lambdaC={LF['lambdaC']}/day)")
    print(f"    checkpoint module: PD-1/PD-L1 binding (Kd_PA={KD_PA}, Kd_PL={KD_PL} nM)")
    print(f"    {n_patients} patients x ({len(DOSE_UNITS)} BRAFi + "
          f"{len(ANTIPD1_DOSE_UNITS)} anti-PD-1) doses = "
          f"{n_patients * (len(DOSE_UNITS) + len(ANTIPD1_DOSE_UNITS))} fast+slow solves")

    args_list = [(i, patients.iloc[i], pERK_ref, cyt_ref) for i in range(n_patients)]

    t0 = time.time()
    perk_res, tumour_res, checkpoint_res = {}, {}, {}
    with Pool(processes=n_cores) as pool:
        done = 0
        for i_patient, perk_row, tumour_row, checkpoint_row in pool.imap_unordered(
                simulate_patient, args_list, chunksize=4):
            perk_res[i_patient] = perk_row
            tumour_res[i_patient] = tumour_row
            checkpoint_res[i_patient] = checkpoint_row
            done += 1
            if done % 50 == 0 or done == n_patients:
                print(f"    Progress: {done}/{n_patients} "
                      f"({done / n_patients * 100:.0f}%) — {time.time() - t0:.0f}s")

    print(f"\n    Simulation complete in {time.time() - t0:.1f}s")

    print("\n[4] Assembling result tables...")
    perk_df = pd.DataFrame([perk_res[i] for i in range(n_patients)],
                           columns=DOSE_COL_NAMES)
    tumour_df = pd.DataFrame([tumour_res[i] for i in range(n_patients)],
                             columns=DOSE_COL_NAMES)
    checkpoint_df = pd.DataFrame([checkpoint_res[i] for i in range(n_patients)],
                                 columns=ANTIPD1_COL_NAMES)
    for df in (perk_df, tumour_df, checkpoint_df):
        df.insert(0, "SAMPLE_ID", patients["SAMPLE_ID"].values)
        df.insert(1, "PATIENT_ID", patients["PATIENT_ID"].values)
        df.insert(2, "BRAF_MUT", patients["BRAF_MUT"].values)
        df.insert(3, "NRAS_MUT", patients["NRAS_MUT"].values)
        df.insert(4, "MAPK_DRIVEN", patients["MAPK_DRIVEN"].values)
    checkpoint_df.insert(5, "CD274", patients["CD274"].values)
    checkpoint_df.insert(6, "PDCD1", patients["PDCD1"].values)

    perk_df.to_csv(OUT_PERK, index=False)
    tumour_df.to_csv(OUT_TUMOUR, index=False)
    checkpoint_df.to_csv(OUT_CHECKPOINT, index=False)
    print(f"    Saved: {OUT_PERK}  {perk_df.shape}")
    print(f"    Saved: {OUT_TUMOUR}  {tumour_df.shape}")
    print(f"    Saved: {OUT_CHECKPOINT}  {checkpoint_df.shape}")

    print("\n[5] Sanity check — mean pERK (low dose -> high dose):")
    sub_defs = [
        ("BRAF-V600E", perk_df[perk_df.BRAF_MUT == 1]),
        ("NRAS-mutant", perk_df[(perk_df.NRAS_MUT == 1) & (perk_df.BRAF_MUT == 0)]),
        ("MAPK-quiet WT", perk_df[perk_df.MAPK_DRIVEN == 0]),
    ]
    for label, sub in sub_defs:
        if len(sub) == 0:
            continue
        lo = sub[DOSE_COL_NAMES[0]].mean()
        hi = sub[DOSE_COL_NAMES[-1]].mean()
        peak = sub[DOSE_COL_NAMES].mean().max()
        trend = ("suppressed" if hi < 0.8 * lo else
                 "paradox/flat" if hi >= lo else "partial")
        print(f"    {label:<14} low={lo:7.2f}  high={hi:7.2f}  peak={peak:7.2f}  ({trend})")

    print("\n[6] Sanity check — checkpoint blockade shrinks tumour burden more in")
    print("    high-PD-L1 patients than low-PD-L1 patients (median CD274 split):")
    pdl1_med = checkpoint_df["CD274"].median()
    for label, sub in [("High PD-L1 (CD274 > median)", checkpoint_df[checkpoint_df.CD274 > pdl1_med]),
                       ("Low PD-L1 (CD274 <= median)", checkpoint_df[checkpoint_df.CD274 <= pdl1_med])]:
        if len(sub) == 0:
            continue
        lo = sub[ANTIPD1_COL_NAMES[0]].mean()
        hi = sub[ANTIPD1_COL_NAMES[-1]].mean()
        pct_drop = 100.0 * (lo - hi) / lo if lo > 0 else float("nan")
        print(f"    {label:<28} untreated={lo:7.4f}  max anti-PD-1={hi:7.4f}  "
              f"(burden drop {pct_drop:5.1f}%)")

    print("\n" + "=" * 68)
    print("  Phase 3 COMPLETE — Ready for Phase 4 (Survival Analysis)")
    print("=" * 68)
