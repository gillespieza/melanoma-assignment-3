"""
Q5 interactive predictor — recommend first-line therapy for NEW patients
========================================================================
Feed in a new dataset (one or more patients) and get a first-line treatment
recommendation per patient: immunotherapy, targeted therapy, targeted->IO, or
trial/chemotherapy.

Input CSV columns (one row per patient):
    patient_id            free-text ID
    BRAF_V600             1 if BRAF-V600 activating mutation, else 0
    NRAS                  1 if NRAS activating mutation, else 0
    BRAF, MAP2K1, MAP2K2, MAPK1, MAPK3, CD8A, PRF1, GZMA
                          raw expression (any consistent unit; RSEM/TPM/etc.)

Pipeline per patient:
  1. expression is normalised to the input dataset's own per-gene mean
     (so it matches the scale the models were built on, platform-independently)
  2. the BRAF/MEK/ERK ODE (Q3) is run -> predicted pERK suppression + tumour burden
  3. an immune-infiltration score is computed (CD8A/PRF1/GZMA)
  4. frozen reference thresholds route the patient on the triage grid

Usage:
    python predict_dataset.py INPUT.csv [OUTPUT.csv]
    python predict_dataset.py --demo          # runs on built-in example patients

NOTE: research-grade decision support, not a validated device. A single isolated
patient cannot be self-normalised — run patients as a batch, or against a
reference panel. Truly cross-platform datasets should be batch-harmonised first.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Reuse the Q3 ODE model
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSIGN_DIR = os.path.dirname(BASE_DIR)
Q3_SCRIPTS = os.path.join(ASSIGN_DIR, "q3-ode-model", "scripts")
sys.path.insert(0, Q3_SCRIPTS)
import phase3_ode_simulation as ode  # noqa: E402

REF_FILE = os.path.join(BASE_DIR, "reference", "reference.json")
EXPR_GENES = ["BRAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3", "CD8A", "PRF1", "GZMA"]

IO_FIRST    = "Immunotherapy first"
TGT_TO_IO   = "Targeted -> Immunotherapy"
TGT_FIRST   = "Targeted therapy first"
TRIAL_CHEMO = "Trial / Chemotherapy"


def recommend(braf_v600, immune_hot, high_burden):
    """Return (recommendation, rationale) for one patient's routed features."""
    if braf_v600:
        if immune_hot and not high_burden:
            return IO_FIRST, "BRAF-V600 + immune-hot, low burden: durable IO benefit, targeted in reserve"
        if immune_hot and high_burden:
            return TGT_TO_IO, "BRAF-V600 + immune-hot, high burden: targeted for rapid control, then IO"
        return TGT_FIRST, "BRAF-V600 + immune-cold: BRAF/MEK inhibitor gives the clearest predicted benefit"
    if immune_hot:
        return IO_FIRST, "BRAF-wild-type + immune-hot: checkpoint blockade (no targeted option)"
    return TRIAL_CHEMO, "BRAF-wild-type + immune-cold: hardest group; trial or chemo fallback"


def predict(df_raw, ref):
    """
    df_raw: DataFrame with patient_id, BRAF_V600, NRAS, and raw expression genes.
    Returns a DataFrame with recommendations and the driving features.
    """
    df = df_raw.copy()
    for g in EXPR_GENES:
        if g not in df.columns:
            raise ValueError(f"Missing required expression column: {g}")

    # 1. Normalise each gene to THIS dataset's own mean (-> mean 1, model scale)
    norm = df[EXPR_GENES].clip(lower=0)
    means = norm.mean(axis=0).replace(0, 1.0)
    norm = norm / means

    pERK_ref = ref["pERK_ref"]
    out_rows = []
    for i in range(len(df)):
        braf = int(df.iloc[i]["BRAF_V600"]) == 1
        nras = int(df.iloc[i]["NRAS"]) == 1
        row = {g: float(norm.iloc[i][g]) for g in EXPR_GENES}
        row["BRAF_MUT"] = 1 if braf else 0
        row["NRAS_MUT"] = 1 if nras else 0
        row["MAPK_DRIVEN"] = 1 if (braf or nras) else 0

        # 2. Run the Q3 ODE for this patient
        _, perk_row, tumour_row = ode.simulate_patient((i, row, pERK_ref))
        perk_lo, perk_hi = max(perk_row[0], 1e-9), max(perk_row[-1], 0.0)
        targeted_response = max((perk_lo - perk_hi) / perk_lo, 0.0)
        tumour_burden = max(tumour_row[-1], 0.0)

        # 3. Immune score
        immune_score = float(np.mean([row["CD8A"], row["PRF1"], row["GZMA"]]))

        # 4. Route on frozen thresholds
        immune_hot = immune_score >= ref["immune_threshold"]
        high_burden = tumour_burden >= ref["burden_threshold"]
        rec, rationale = recommend(braf, immune_hot, high_burden)

        out_rows.append({
            "patient_id": df.iloc[i].get("patient_id", f"patient_{i+1}"),
            "subtype": "BRAF" if braf else ("NRAS" if nras else "WT"),
            "immune_score": round(immune_score, 3),
            "immune_status": "hot" if immune_hot else "cold",
            "targeted_pERK_suppression": round(targeted_response, 3),
            "tumour_burden": round(tumour_burden, 3),
            "burden_status": "high" if high_burden else "low",
            "RECOMMENDATION": rec,
            "rationale": rationale,
        })
    return pd.DataFrame(out_rows)


def demo_dataframe():
    """A few illustrative patients spanning the decision grid."""
    # raw-ish expression values; relative levels are what matter after normalisation
    rows = [
        # id, BRAF_V600, NRAS, BRAF, MAP2K1, MAP2K2, MAPK1, MAPK3, CD8A, PRF1, GZMA
        ("P1_BRAF_hot",  1, 0, 500, 400, 300, 600, 500, 900, 800, 850),
        ("P2_BRAF_cold", 1, 0, 520, 420, 310, 610, 520,  60,  40,  50),
        ("P3_NRAS_hot",  0, 1, 300, 380, 290, 590, 480, 950, 900, 880),
        ("P4_WT_cold",   0, 0, 280, 360, 280, 520, 450,  40,  30,  35),
        ("P5_BRAF_hot_bulk", 1, 0, 900, 800, 700, 950, 900, 700, 650, 680),
    ]
    cols = ["patient_id", "BRAF_V600", "NRAS"] + EXPR_GENES
    return pd.DataFrame(rows, columns=cols)


def main():
    with open(REF_FILE) as f:
        ref = json.load(f)

    args = sys.argv[1:]
    if not args or args[0] == "--demo":
        print("Running on built-in demo patients (no input CSV given).\n")
        df_in = demo_dataframe()
        out_path = os.path.join(BASE_DIR, "results", "demo_predictions.csv")
    else:
        df_in = pd.read_csv(args[0])
        out_path = args[1] if len(args) > 1 else os.path.splitext(args[0])[0] + "_predictions.csv"

    result = predict(df_in, ref)
    result.to_csv(out_path, index=False)

    print("=" * 78)
    print("  First-line treatment recommendations")
    print("=" * 78)
    for _, r in result.iterrows():
        print(f"\n  {r['patient_id']}  [{r['subtype']}, immune-{r['immune_status']}, "
              f"burden-{r['burden_status']}]")
        print(f"    -> {r['RECOMMENDATION']}")
        print(f"       {r['rationale']}")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
