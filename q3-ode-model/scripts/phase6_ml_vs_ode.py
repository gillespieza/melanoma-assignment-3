"""
Phase 6: Mechanistic ODE vs Machine Learning
============================================
Asks whether the interpretable, 3-feature "ODE digital twin" competes with
black-box ML models trained on raw data for predicting melanoma survival.

Task: binary classification of 2-year overall survival in TCGA-SKCM
    y = 1  survived  > 24 months
    y = 0  died     <= 24 months
    (patients censored before 24 months are dropped — ambiguous outcome)

Models compared by 5-fold cross-validated ROC-AUC:
    1. ODE mechanistic : LogisticRegression on THREE ODE outputs only
                         (baseline pERK; tumour burden at max vemurafenib dose;
                          tumour burden at max anti-PD-1 dose, Module D)
    2. Neural network  : MLP on the 10 pathway genes (+ BRAF/NRAS flags)
    3. Random forest   : RF on the 10 pathway genes (+ BRAF/NRAS flags)
    4. Logistic (genes): linear baseline on the same raw features

The ML models see far more inputs (12 features) than the ODE (3). If the ODE
score keeps up, that is the point: a mechanistic model compresses the biology
into a few interpretable numbers — now covering both the targeted-therapy arm
(BRAFi) and the immunotherapy arm (anti-PD-1, Module D) of the Q5 decision.

Output:
    results/ml_vs_ode_comparison.csv
    plots/ml_vs_ode_comparison.pdf / .png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE   = os.path.join(BASE_DIR, "data", "melanoma_params_full.csv")
PERK_FILE       = os.path.join(BASE_DIR, "results", "pERK_simulations.csv")
TUMOUR_FILE     = os.path.join(BASE_DIR, "results", "tumour_burden_simulations.csv")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "results", "checkpoint_tumour_simulations.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_GENES = ["BRAF", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3",
               "CDKN2A", "MKI67", "CD8A", "PRF1", "GZMA"]

print("=" * 65)
print("  Phase 6: Mechanistic ODE vs Machine Learning (TCGA-SKCM)")
print("=" * 65)

# ─── Assemble data ────────────────────────────────────────────────────────────
df    = pd.read_csv(DATA_FILE)
perk  = pd.read_csv(PERK_FILE)
tum   = pd.read_csv(TUMOUR_FILE)
ckpt  = pd.read_csv(CHECKPOINT_FILE)

# ODE-derived features (the whole "digital twin" reduced to three numbers:
# baseline signalling, targeted-therapy response, immunotherapy response)
df = df.merge(perk[["SAMPLE_ID", "BRAFi_0.010"]], on="SAMPLE_ID")
df = df.merge(tum[["SAMPLE_ID", "BRAFi_1.000"]], on="SAMPLE_ID")
df = df.merge(ckpt[["SAMPLE_ID", "antiPD1_1.000"]], on="SAMPLE_ID")
df = df.rename(columns={"BRAFi_0.010": "ode_pERK_baseline",
                        "BRAFi_1.000": "ode_tumour_maxdose",
                        "antiPD1_1.000": "ode_checkpoint_maxdose"})
df["ode_pERK_baseline"] = df["ode_pERK_baseline"].clip(lower=0)
df["ode_tumour_maxdose"] = df["ode_tumour_maxdose"].clip(lower=0)
df["ode_checkpoint_maxdose"] = df["ode_checkpoint_maxdose"].clip(lower=0)

# 2-year survival target
df["is_dead"] = (df["OS_STATUS"] == "1:DECEASED").astype(int)
df["y"] = np.nan
df.loc[df["OS_MONTHS"] > 24, "y"] = 1
df.loc[(df["OS_MONTHS"] <= 24) & (df["is_dead"] == 1), "y"] = 0
clean = df.dropna(subset=["y"]).copy()
clean["y"] = clean["y"].astype(int)

print(f"\nPatients with unambiguous 2-year outcome: {len(clean)}")
print(f"  Survived > 2 yr: {int(clean['y'].sum())}  |  "
      f"Died <= 2 yr: {len(clean) - int(clean['y'].sum())}")

# Feature matrices
X_ode   = clean[["ode_pERK_baseline", "ode_tumour_maxdose", "ode_checkpoint_maxdose"]].values
gene_cols = MODEL_GENES + ["BRAF_MUT", "NRAS_MUT"]
X_genes = clean[gene_cols].values
y       = clean["y"].values

# ─── Models ───────────────────────────────────────────────────────────────────
models = {
    'ODE "Digital Twin"\n(3 ODE outputs)': (
        Pipeline([("sc", StandardScaler()),
                  ("lr", LogisticRegression(class_weight="balanced", max_iter=1000))]),
        X_ode),
    "Neural Net\n(12 raw features)": (
        Pipeline([("sc", StandardScaler()),
                  ("mlp", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000,
                                        alpha=0.01, random_state=42))]),
        X_genes),
    "Random Forest\n(12 raw features)": (
        Pipeline([("sc", StandardScaler()),
                  ("rf", RandomForestClassifier(n_estimators=300, max_depth=5,
                                                class_weight="balanced",
                                                random_state=42))]),
        X_genes),
    "Logistic Reg\n(12 raw features)": (
        Pipeline([("sc", StandardScaler()),
                  ("lr", LogisticRegression(class_weight="balanced", max_iter=1000))]),
        X_genes),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n[Running 5-fold cross-validation — ROC-AUC]")
rows = []
for name, (model, X) in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    rows.append({"model": name.replace("\n", " "),
                 "mean_auc": scores.mean(), "std_auc": scores.std(),
                 "n_features": X.shape[1]})
    print(f"  {name.splitlines()[0]:<22} ROC-AUC = {scores.mean():.3f} "
          f"± {scores.std():.3f}  ({X.shape[1]} features)")

results = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_DIR, "ml_vs_ode_comparison.csv")
results.to_csv(csv_path, index=False)

# ─── Plot ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from src.styles import set_presentation_style
set_presentation_style(dpi=300)

names = list(models.keys())
means = [r["mean_auc"] for r in rows]
stds  = [r["std_auc"] for r in rows]
colors = ["#0072B2", "#B0BEC5", "#37474F", "#78909C"]  # Okabe-Ito Blue for ODE, Slate/Grey for ML

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
bars = ax.bar(names, means, yerr=stds, capsize=6, color=colors,
              alpha=0.9, edgecolor="#1E293B", linewidth=1.2,
              error_kw={"elinewidth": 1.5, "ecolor": "#1E293B"})
ax.axhline(0.5, color="#D55E00", ls="--", lw=1.5, label="Random chance (0.5)")
ax.set_ylabel("ROC-AUC (5-fold CV)\nhigher is better", fontsize=12, fontweight="bold")
ax.set_title("Mechanistic ODE vs Machine Learning\nPredicting 2-Year Survival — "
             "TCGA-SKCM Melanoma", fontsize=13, fontweight="bold", pad=15)
ax.set_ylim(0.4, max(0.75, max(means) + 0.1))
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, m + 0.012, f"{m:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CCCCCC")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "ml_vs_ode_comparison.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(PLOTS_DIR, "ml_vs_ode_comparison.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n  Saved: {csv_path}")
print(f"  Saved: {os.path.join(PLOTS_DIR, 'ml_vs_ode_comparison.pdf')}")

best = results.loc[results["mean_auc"].idxmax(), "model"]
ode_auc = results.loc[results["model"].str.startswith("ODE"), "mean_auc"].iloc[0]
print(f"\n  Best model: {best}")
print(f"  ODE digital twin AUC = {ode_auc:.3f} using only 3 interpretable features")

print("\n" + "=" * 65)
print("  Phase 6 COMPLETE — Pipeline finished!")
print("=" * 65)
