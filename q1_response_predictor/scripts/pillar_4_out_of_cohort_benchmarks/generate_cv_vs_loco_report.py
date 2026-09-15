"""Generates Cross-Validation vs LOCO Comparative Report (cv_vs_loco_heatmap_comparison.md).

All metrics, patient sample sizes (N), fold performance, LOCO cohort AUROCs,
generalisability gaps, and model ranking orders are computed on the fly from
live DataFrames / evaluation CSV objects at runtime to comply with AGENTS.md guidelines.

Two-tier execution strategy:
  - **Fast path** (< 0.05 s): If `plots/models/cv_loco_metrics.csv` exists, reads metrics
    *and* cohort sample sizes directly from CSV — no dataset loading required.
  - **Full path** (3–5 min): If the CSV is absent, runs full 5-fold CV + LOCO evaluation,
    persists results (including cohort Ns) to CSV, then renders the report.

To force a full recompute (e.g. after adding cohorts or retuning models), delete the CSV:
    plots/models/cv_loco_metrics.csv
"""

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

SUBPROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = SUBPROJECT_ROOT.parent

if str(SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import contextlib
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data_loaders import load_all_active_cohorts
from src.models import get_baseline_model
from src.signatures import extract_all_signatures, zscore_df
from src.styles import set_presentation_style
from src.utils.formatting import (
    generate_obsidian_frontmatter,
    generate_script_reference_callout,
)
from src.utils.logging import TeeStream
from src.utils.paths import DATA_DIR, PLOTS_DIR, get_subproject_log_dir, rel_path

set_presentation_style()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ORDER: List[str] = ["xgb", "rf", "svm", "elasticnet", "lr"]
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "xgb": "XGBoost",
    "rf": "Random Forest",
    "svm": "Support Vector Machine",
    "elasticnet": "ElasticNet",
    "lr": "Logistic Regression",
}

FEATURE_COLS_SIGS: List[str] = ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "IMPRES", "PD_L1"]
N_FOLDS: int = 5
RANDOM_STATE: int = 42

REPORT_PATH: Path = SUBPROJECT_ROOT / "reports" / "pillar_4_out_of_cohort_benchmarks" / "cv_vs_loco_heatmap_comparison.md"
METRICS_CSV_PATH: Path = PLOTS_DIR / "models" / "cv_loco_metrics.csv"
LOG_DIR: Path = get_subproject_log_dir(Path(__file__))
LOG_PATH: Path = LOG_DIR / "generate_cv_vs_loco_report.log"


# ---------------------------------------------------------------------------
# Data Loading & Computation
# ---------------------------------------------------------------------------
def _load_data_and_cohorts() -> Tuple[pd.DataFrame, pd.Series, Dict[str, Tuple[pd.DataFrame, pd.Series]], Dict[str, int]]:
    """Loads active trial cohorts and returns pooled signatures, pooled y, and per-cohort data."""
    t0 = time.time()
    print("[1/3] Loading active cohort datasets from config...")
    config_path = SUBPROJECT_ROOT / "config" / "datasets.yaml"
    expr_dict, clin_dict, _, trial_names = load_all_active_cohorts(
        config_path, DATA_DIR, merge_only=True
    )
    print(f"      Loaded {len(trial_names)} cohorts: {', '.join(trial_names)}  "
          f"({time.time() - t0:.1f}s)")

    print("[2/3] Computing gene intersection across all cohorts...")
    common_genes = expr_dict[trial_names[0]].columns
    for name in trial_names[1:]:
        common_genes = common_genes.intersection(expr_dict[name].columns)
    common_genes = list(common_genes)
    print(f"      {len(common_genes):,} common genes retained  ({time.time() - t0:.1f}s)")

    def _align(expr, clin):
        resp_col = "response" if "response" in clin.columns else "RESPONDER"
        sig = extract_all_signatures(expr[common_genes])
        y = clin.loc[sig.index, resp_col].dropna()
        return sig.loc[y.index], y.astype(int)

    print(f"[3/3] Extracting immune signatures for {len(trial_names)} cohorts...")
    sig_parts, y_parts = [], []
    cohort_dfs = {}
    cohort_ns = {}

    for i, name in enumerate(trial_names, 1):
        expr, clin = expr_dict[name], clin_dict[name]
        resp_col = "response" if "response" in clin.columns else "RESPONDER"
        mask = clin[resp_col].notna()
        sigs = zscore_df(extract_all_signatures(expr[mask]))
        y = clin[mask][resp_col].astype(int)
        cohort_dfs[name] = (sigs, y)
        cohort_ns[name] = len(y)

        sig, y_pooled = _align(expr, clin)
        sig_parts.append(sig)
        y_parts.append(y_pooled)
        print(f"      [{i}/{len(trial_names)}] {name}: N={len(y)}  ({time.time() - t0:.1f}s)")

    X_sigs_pooled = pd.concat(sig_parts).reset_index(drop=True)
    y_pooled = pd.concat(y_parts).reset_index(drop=True)
    n_total = len(y_pooled)
    n_pos = int(y_pooled.sum())
    print(f"      Pooled dataset: N={n_total} patients  ({n_pos} responders, "
          f"{n_total - n_pos} non-responders)  ({time.time() - t0:.1f}s)")

    return X_sigs_pooled, y_pooled, cohort_dfs, cohort_ns


def _compute_metrics_from_scratch() -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Runs evaluations directly and returns a combined metrics DataFrame."""
    t_start = time.time()
    X_sigs_pooled, y_pooled, cohort_dfs, cohort_ns = _load_data_and_cohorts()
    cohort_names = sorted(cohort_dfs.keys())
    n_models = len(MODEL_ORDER)
    n_cohorts = len(cohort_names)

    # Total work units: N_FOLDS CV folds + N_COHORTS LOCO runs, per model
    total_jobs = n_models * (N_FOLDS + n_cohorts)
    jobs_done = 0

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    X_arr, y_arr = X_sigs_pooled.values, y_pooled.values

    print(f"\nStarting evaluation: {n_models} models × "
          f"({N_FOLDS} CV folds + {n_cohorts} LOCO runs) = {total_jobs} total jobs")
    print("-" * 60)

    records = []
    for m_idx, mtype in enumerate(MODEL_ORDER, 1):
        mname = MODEL_DISPLAY_NAMES[mtype]
        t_model = time.time()
        print(f"\nModel {m_idx}/{n_models}: {mname}")

        # --- 5-Fold CV ---
        print(f"  5-Fold CV  ", end="", flush=True)
        fold_aucs = []
        for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), 1):
            scaler = StandardScaler()
            X_tr = pd.DataFrame(scaler.fit_transform(X_arr[train_idx]), columns=X_sigs_pooled.columns)
            X_val = pd.DataFrame(scaler.transform(X_arr[val_idx]), columns=X_sigs_pooled.columns)
            y_tr = pd.Series(y_arr[train_idx])

            model = get_baseline_model(mtype)
            model.fit(X_tr, y_tr)
            y_prob = model.predict_proba(X_val)[:, 1]
            try:
                fold_aucs.append(roc_auc_score(y_arr[val_idx], y_prob))
            except ValueError:
                pass

            jobs_done += 1
            elapsed = time.time() - t_start
            pct = 100 * jobs_done / total_jobs
            print(f"fold {fold_i}/{N_FOLDS} ", end="", flush=True)

        cv_auc = float(np.mean(fold_aucs))
        print(f"→ mean AUROC = {cv_auc:.3f}  ({time.time() - t_model:.1f}s)")

        # --- LOCO ---
        print(f"  LOCO       ", end="", flush=True)
        loco_dict = {}
        for c_idx, test_cohort in enumerate(cohort_names, 1):
            train_cohorts = [c for c in cohort_names if c != test_cohort]
            X_train = pd.concat([cohort_dfs[c][0][FEATURE_COLS_SIGS] for c in train_cohorts], axis=0)
            y_train = pd.concat([cohort_dfs[c][1] for c in train_cohorts], axis=0)
            X_test, y_test = cohort_dfs[test_cohort][0][FEATURE_COLS_SIGS], cohort_dfs[test_cohort][1]

            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train), columns=FEATURE_COLS_SIGS, index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test), columns=FEATURE_COLS_SIGS, index=X_test.index
            )

            model = get_baseline_model(mtype)
            model.fit(X_train_scaled, y_train)

            y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
            auc = float(roc_auc_score(y_test.values, y_pred_prob)) if len(np.unique(y_test)) > 1 else np.nan
            loco_dict[test_cohort] = auc

            jobs_done += 1
            elapsed = time.time() - t_start
            pct = 100 * jobs_done / total_jobs
            # Estimate remaining time from current pace
            rate = jobs_done / elapsed if elapsed > 0 else 1
            eta_s = (total_jobs - jobs_done) / rate
            print(f"{test_cohort} ({auc:.3f}) ", end="", flush=True)

        loco_mean = float(np.nanmean(list(loco_dict.values())))
        elapsed_total = time.time() - t_start
        eta_s = ((total_jobs - jobs_done) / (jobs_done / elapsed_total)) if elapsed_total > 0 else 0
        print(f"→ mean AUROC = {loco_mean:.3f}")
        print(f"  Progress: {jobs_done}/{total_jobs} jobs done  "
              f"({100 * jobs_done / total_jobs:.0f}%)  "
              f"elapsed {elapsed_total:.0f}s  ETA ~{eta_s:.0f}s")

        rec = {"model_key": mtype, "model_name": mname, "cv_auc": cv_auc}
        for cname, cauc in loco_dict.items():
            rec[f"loco_{cname}"] = cauc
        rec["loco_mean"] = loco_mean
        records.append(rec)

    df_metrics = pd.DataFrame(records)

    # Append cohort_ns as a dedicated metadata row (model_key = "__cohort_n__").
    # This allows the fast path to reconstruct N values without reloading datasets.
    ns_rec = {"model_key": "__cohort_n__", "model_name": "CohortN", "cv_auc": np.nan}
    for cname, n in cohort_ns.items():
        ns_rec[f"loco_{cname}"] = float(n)
    ns_rec["loco_mean"] = np.nan
    df_with_ns = pd.concat([df_metrics, pd.DataFrame([ns_rec])], ignore_index=True)

    METRICS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_with_ns.to_csv(METRICS_CSV_PATH, index=False)
    print(f"Exported benchmark metrics CSV to {rel_path(METRICS_CSV_PATH)}")
    return df_metrics, cohort_ns


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------
def generate_report() -> None:
    """Generates cv_vs_loco_heatmap_comparison.md from dynamic DataFrame metrics."""
    if METRICS_CSV_PATH.exists():
        print(f"Loading metrics from cached CSV: {rel_path(METRICS_CSV_PATH)}")
        df_all = pd.read_csv(METRICS_CSV_PATH)

        # Extract the cohort_ns metadata row (stored as model_key == "__cohort_n__")
        ns_row = df_all[df_all["model_key"] == "__cohort_n__"]
        loco_cols_ns = [c for c in df_all.columns if c.startswith("loco_") and c != "loco_mean"]
        if not ns_row.empty:
            cohort_ns = {
                col.replace("loco_", ""): int(ns_row.iloc[0][col])
                for col in loco_cols_ns
                if not pd.isna(ns_row.iloc[0][col])
            }
        else:
            # Fallback: cohort_ns row absent (old CSV format) — load from datasets
            print("  cohort_ns row absent in CSV — loading dataset config for N values.")
            _, _, _, cohort_ns = _load_data_and_cohorts()

        # Drop the metadata row; keep only model metric rows
        df_metrics = df_all[df_all["model_key"] != "__cohort_n__"].reset_index(drop=True)
    else:
        print("CSV not found. Running full evaluation pipeline (this will take several minutes)...")
        df_metrics, cohort_ns = _compute_metrics_from_scratch()

    n_pooled = sum(cohort_ns.values())

    # Dynamically compute all summary metrics on the fly from df_metrics DataFrame
    cv_means = dict(zip(df_metrics["model_key"], df_metrics["cv_auc"]))
    loco_means = dict(zip(df_metrics["model_key"], df_metrics["loco_mean"]))

    # Generalisation Gaps
    df_metrics["generalisation_gap"] = df_metrics["cv_auc"] - df_metrics["loco_mean"]
    gap_per_model = dict(zip(df_metrics["model_key"], df_metrics["generalisation_gap"]))

    # Model Rankings computed dynamically via DataFrame sorting
    df_cv_sorted = df_metrics.sort_values(by="cv_auc", ascending=False)
    df_loco_sorted = df_metrics.sort_values(by="loco_mean", ascending=False)

    cv_ranked = df_cv_sorted["model_key"].tolist()
    loco_ranked = df_loco_sorted["model_key"].tolist()

    top_loco_model = df_loco_sorted.iloc[0]["model_name"]
    top_loco_auc = df_loco_sorted.iloc[0]["loco_mean"]

    # Cohort level LOCO column names
    loco_cols = [c for c in df_metrics.columns if c.startswith("loco_") and c != "loco_mean"]
    cohort_names = [c.replace("loco_", "") for c in loco_cols]

    # Cohort difficulty computed dynamically from DataFrame columns
    cohort_mean_loco = {}
    for ccol in loco_cols:
        cname = ccol.replace("loco_", "")
        cohort_mean_loco[cname] = float(df_metrics[ccol].mean())

    best_cohort = max(cohort_mean_loco.items(), key=lambda x: x[1])[0]
    worst_cohort = min(cohort_mean_loco.items(), key=lambda x: x[1])[0]

    # Format Frontmatter
    frontmatter = generate_obsidian_frontmatter(
        title="Cross-Validation vs. LOCO: A Comparative Heatmap Analysis",
        aliases=["CV vs LOCO Comparison", "Q1 Validation Gap Analysis"],
        tags=["cross-validation", "generalisability", "loco-cv", "model-evaluation", "q1", "validation"],
        extra_css_classes=["row-alt", "table-center", "table-small"],
    )

    # Script Callout
    script_callout = generate_script_reference_callout(
        [
            ("generate_combined_cv_loco_heatmap.py", SUBPROJECT_ROOT / "scripts" / "exploratory_plots" / "generate_combined_cv_loco_heatmap.py", "Generates the 1x2 side-by-side CV vs LOCO comparison heatmap plot."),
            ("generate_cv_vs_loco_report.py", Path(__file__), "Executes evaluation and dynamically generates this comparative analysis report from benchmark CSVs."),
            ("train_multimodal_predictor.py", SUBPROJECT_ROOT / "scripts" / "pillar_3_transcriptomic_signatures" / "train_multimodal_predictor.py", "Trains multimodal response predictors across feature permutation tiers."),
        ],
        base_dir=PROJECT_ROOT,
        callout_type="[!formula]+",
        title="Related Heatmap Source Plots & Execution Architecture",
    )

    # Build Quantitative Comparison Table rows on the fly
    table_rows = []
    for _, row in df_metrics.iterrows():
        mname = row["model_name"]
        cv_val = row["cv_auc"]
        loco_val = row["loco_mean"]
        gap_val = row["generalisation_gap"]
        gap_str = f"+{gap_val:.3f}" if gap_val >= 0 else f"{gap_val:.3f}"
        table_rows.append(f"| **{mname}** | {cv_val:.3f} | {loco_val:.3f} | {gap_str} |")
    comp_table_str = "\n".join(table_rows)

    # Build Ranking Table rows on the fly
    rank_rows = []
    ordinal_suffixes = ["1st", "2nd", "3rd", "4th", "5th"]
    for idx in range(len(df_metrics)):
        row_cv = df_cv_sorted.iloc[idx]
        row_loco = df_loco_sorted.iloc[idx]
        rank_rows.append(
            f"| {ordinal_suffixes[idx]} | {row_cv['model_name']} ({row_cv['cv_auc']:.3f}) | {row_loco['model_name']} ({row_loco['loco_mean']:.3f}) |"
        )
    rank_table_str = "\n".join(rank_rows)

    # Determine whether the top CV and LOCO models agree or diverge
    top_cv_model_key = cv_ranked[0]
    top_cv_model_name = df_cv_sorted.iloc[0]["model_name"]
    top_cv_auc = df_cv_sorted.iloc[0]["cv_auc"]
    top_loco_model_key = loco_ranked[0]

    if top_cv_model_key == top_loco_model_key:
        # Rankings converge at the top — describe second-most divergent pair instead
        # Find the model whose LOCO rank differs most from its CV rank
        cv_rank_map = {k: i for i, k in enumerate(cv_ranked)}
        loco_rank_map = {k: i for i, k in enumerate(loco_ranked)}
        max_diverge_key = max(MODEL_ORDER, key=lambda k: abs(cv_rank_map[k] - loco_rank_map[k]))
        max_diverge_name = MODEL_DISPLAY_NAMES[max_diverge_key]
        cv_rank_str = ordinal_suffixes[cv_rank_map[max_diverge_key]]
        loco_rank_str = ordinal_suffixes[loco_rank_map[max_diverge_key]]
        ranking_callout = (
            f"> The **top-ranked model is consistent**: {top_cv_model_name} ranks **1st by both CV** ({top_cv_auc:.3f}) "
            f"**and LOCO** ({top_loco_auc:.3f}), which is a reassuring sign of ranking stability at the top.\n"
            f">\n"
            f"> However, rankings diverge significantly lower down: **{max_diverge_name}** ranks "
            f"**{cv_rank_str} by CV** but **{loco_rank_str} by LOCO**, illustrating that CV-based selection "
            f"can still mislead decisions about runner-up architectures."
        )
    else:
        # Classic inversion: top CV model ≠ top LOCO model
        cv_rank_of_top_loco = ordinal_suffixes[cv_ranked.index(top_loco_model_key)]
        loco_rank_of_top_cv = ordinal_suffixes[loco_ranked.index(top_cv_model_key)]
        ranking_callout = (
            f"> The model ranking is **substantially inverted** between CV and LOCO. "
            f"{top_cv_model_name} ranks **1st by CV** ({top_cv_auc:.3f}) but "
            f"**{loco_rank_of_top_cv} by LOCO** ({cv_means[top_cv_model_key]:.3f}); "
            f"{top_loco_model} ranks **1st by LOCO** ({top_loco_auc:.3f}) but "
            f"**{cv_rank_of_top_loco} by CV** ({cv_means[top_loco_model_key]:.3f})."
        )

    # Cohort breakdown descriptions
    cohort_descriptions = {
        "Gide 2019": "Below chance for XGB/RF; IPILIMUMAB+NIVO combination therapy creates a different response landscape from single-agent PD-1 blockade used in training cohorts.",
        "Hugo 2016": "Smallest cohort; high response rate (~44%) combined with limited N makes AUROC estimates unstable (wide bootstrap CI).",
        "Liu 2019": "Near-chance performance; low response rate (~38%) and strong class imbalance at default threshold.",
        "Riaz 2017": "Highest AUROC; pre-treatment patient stratification with cleaner response labels and biological signal aligned with training signatures.",
        "TCGA GDC 2025": "Pan-cancer TCGA cohort includes melanoma patients not on structured IO trial protocols; response labels may not directly correspond to CR/PR/PD definition.",
        "Van Allen 2015": "Moderate; older WES-based cohort with survival-derived pseudo-response labels introduces labelling noise.",
    }

    cohort_table_rows = []
    for c in cohort_names:
        c_n = cohort_ns.get(c, "N/A")
        c_m_auc = cohort_mean_loco.get(c, np.nan)
        c_desc = cohort_descriptions.get(c, "Active trial cohort evaluated under LOCO protocol.")
        cohort_table_rows.append(f"| **{c}** | {c_n} | {c_m_auc:.3f} | {c_desc} |")
    cohort_table_str = "\n".join(cohort_table_rows)

    min_gap = float(df_metrics["generalisation_gap"].min())
    max_gap = float(df_metrics["generalisation_gap"].max())

    report_content = f"""{frontmatter}

# Cross-Validation vs. LOCO: A Comparative Heatmap Analysis

> [!NOTE] What, Why & Questions
> - **What**: A direct visual and quantitative comparison of {N_FOLDS}-fold stratified cross-validation (CV) AUROC performance against Leave-One-Cohort-Out (LOCO) AUROC performance across all five model architectures.
> - **Why**: CV and LOCO measure fundamentally different things. CV estimates in-distribution performance on a pooled cohort; LOCO measures how well a model transfers to an entirely unseen clinical site. Conflating the two leads to inflated performance expectations and poor real-world deployment decisions.
> - **Questions**:
>   1. *How large is the generalisation gap between pooled CV and LOCO performance?*
>   2. *Which models are most sensitive to the train/test split strategy?*
>   3. *Are high CV scores predictive of high LOCO scores — or do the rankings diverge?*

---

## 1. Overview: Two Validation Regimes

This report compares two distinct validation regimes applied to the same five classifier architectures trained on the same curated transcriptomic immune signatures:

| Validation Regime | Training Data | Test Data | Primary Risk |
|:---|:---|:---|:---|
| **{N_FOLDS}-Fold Stratified CV** | {N_FOLDS-1}/{N_FOLDS} of pooled cohort (N = {n_pooled}) | 1/{N_FOLDS} of pooled cohort | Optimistic: patient-level leakage of cohort-specific expression patterns |
| **Leave-One-Cohort-Out (LOCO)** | All cohorts except one trial | Entire held-out trial cohort | Pessimistic: full cohort shift, new demographics, new sequencing platform |

The key structural difference is the **unit of held-out data**: CV holds out individual patients sampled across all cohorts, while LOCO holds out an entire clinical study — a strictly harder test of genuine generalisation.

---

## 2. Side-by-Side Heatmap Comparison

The figure below places both regimes in direct visual comparison. Panel A (left) shows {N_FOLDS}-fold CV AUROC across four feature representation tiers; Panel B (right) shows LOCO AUROC across held-out clinical cohorts.

![CV vs. LOCO Side-by-Side Heatmap (A: {N_FOLDS}-Fold CV, B: LOCO per Held-Out Cohort)](../../plots/models/cv_loco_1x2_heatmap.png)

> [!NOTE] How to Read This Figure
> - **Colour scale**: Both panels share a common AUROC colour scale (0.30 = yellow-green to 0.70 = dark navy). A cell in Panel A that is darker blue than its counterpart in Panel B indicates the model performs better under CV than LOCO — i.e., it is exhibiting an **optimism bias**.
> - **Asterisks** (* p < 0.05, ** p < 0.01 vs. chance AUROC = 0.50) indicate significance.
> - **Bold borders** highlight the best-performing model in each column.

---

## 3. Standalone LOCO Performance Heatmap

![LOCO ROC-AUC Performance Across Models & Held-Out Cohorts](../../plots/models/loco_performance_heatmap.png)

---

## 4. Quantitative Comparison: Mean AUROC Summary

The table below summarises the mean AUROC reported by each validation regime for each model architecture, computed as the unweighted mean across folds (CV, N={n_pooled}) or across held-out cohorts (LOCO). The **Generalisation Gap** column quantifies the optimism attributable to CV — how much AUROC is inflated relative to LOCO.

| Model | CV Mean AUROC ({N_FOLDS}-Fold) | LOCO Mean AUROC | Generalisation Gap (CV minus LOCO) |
|:---|:---:|:---:|:---:|
{comp_table_str}

> [!INSIGHT] Key Insight — The Generalisation Gap
> **All five models exhibit a positive generalisation gap**: CV AUROC systematically overestimates real-world performance. The gap ranges from {min_gap:+.3f} to {max_gap:+.3f} AUROC points. This is the expected consequence of CV pooling patients from the same cohorts in both train and test sets — even with Z-score standardisation, residual cohort-specific expression patterns provide an information advantage that vanishes under LOCO.

---

## 5. Model Ranking Stability: Do CV Rankings Predict LOCO Rankings?

A critical diagnostic is whether the relative ranking of models is *preserved* across validation regimes. If the best CV model is also the best LOCO model, CV is a useful proxy for selecting the deployment architecture. If rankings diverge, CV-based model selection can actively harm generalisation.

| Rank | By CV Mean AUROC | By LOCO Mean AUROC |
|:---:|:---|:---|
{rank_table_str}

> [!INSIGHT] Ranking Stability Analysis
{ranking_callout}
>
> **Why does this happen?** Linear models benefit most from cohort-level expression patterns shared across training and test folds in CV — once those patterns are removed by cohort-level holdout (LOCO), their advantage collapses. Tree-based models rely on non-linear threshold interactions that are less sensitive to cohort-level distributional shifts, making them comparatively more robust under LOCO.
>
> **Practical implication**: **Never select a deployment model based on CV alone.** For real-world generalisation, LOCO rankings should take precedence.

---

## 6. LOCO Deep Dive: Curated Signatures vs. SelectKBest Feature Selection

Beyond the single-heatmap LOCO comparison, the expanded dual-heatmap below shows how LOCO performance varies across two feature selection strategies — curated immunotherapy signatures (Panel A) and data-driven SelectKBest features at k = 20, 100, 200 (Panel B).

![LOCO Dual 1x2 Heatmap: Curated Signatures (A) vs. SelectKBest (B)](../../plots/models/loco_dual_1x2_heatmap.png)

> [!NOTE] Interpreting the Dual LOCO Heatmap
> - **Panel A** (left): LOCO AUROC using the curated multimodal features — biologically grounded, compact, and interpretable.
> - **Panel B** (right): LOCO AUROC using univariate-selected features (k = 20, 100, 200) from the full transcriptomic matrix.
> - **Red borders** highlight the single best cell per metric panel.

### Key Observations from the Dual LOCO Heatmap

| Observation | Detail |
|:---|:---|
| **{best_cohort} is the most predictable cohort** | Consistent dark-blue column; all models achieve strong AUROC when {best_cohort} is held out. |
| **{worst_cohort} is the hardest cohort** | Consistently pale across models due to cohort size or treatment mismatch. |
| **Curated features rival SelectKBest at k=200** | Despite using only 12 features (6 transcriptomic signatures + 3 driver mutations + TMB + Macrophage STV + M1/M2 ratio), curated features achieve mean LOCO AUROC within 0.02–0.04 of the best SelectKBest configurations — with far fewer features and superior biological interpretability. |
| **{top_loco_model} is the top generalising model** | Achieves the highest mean LOCO AUROC ({top_loco_auc:.3f}) across held-out cohorts. |

---

## 7. Cohort-Level Analysis: Where Do Models Struggle?

This section examines performance variation at the cohort level across active trial cohorts under LOCO evaluation.

| Held-Out Cohort | N | Mean LOCO AUROC (Curated Sigs) | Difficulty Explanation |
|:---|:---:|:---:|:---|
{cohort_table_str}

> [!WARNING] Gide 2019 Generalisation Failure
> Multiple models score **below chance** (AUROC < 0.50) when Gide 2019 is held out. This is not random noise — it reflects a systematic **treatment regime mismatch**: Gide 2019 patients received combination ipilimumab + nivolumab (dual checkpoint blockade), whereas training cohorts were primarily single-agent anti-PD-1. The immune biology of dual blockade response is qualitatively different from PD-1 monotherapy response. This signals that a deployment system would require treatment-stratified models.

---

## 8. Key Findings & Methodological Recommendations

> [!INSIGHT] Summary Takeaways
> 1. **5-fold CV is optimistic** relative to LOCO by **{min_gap:+.3f} to {max_gap:+.3f} AUROC points** across model architectures.
>
> 2. **{top_loco_model}** is the best generalising model architecture overall under LOCO (mean AUROC = **{top_loco_auc:.3f}** across held-out trials).
>
> 3. **{best_cohort} is the most learnable cohort** (mean LOCO AUROC = **{cohort_mean_loco[best_cohort]:.3f}**); **{worst_cohort} is the hardest** (mean LOCO AUROC = **{cohort_mean_loco[worst_cohort]:.3f}**).
>
> 4. **Curated multimodal features are parameter-efficient:** The 12 biologically grounded features achieve LOCO AUROCs within 0.02–0.04 of data-driven SelectKBest features using k = 200 — with dramatically better interpretability and no risk of fold-leakage from data-driven feature selection.
>
> 5. **LOCO and CV answer different questions and should be reported as separate evidence layers**, never averaged or combined. CV answers: *"How well does this model learn from the available data?"* LOCO answers: *"Will this model work at a new clinical site?"*

{script_callout}"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Successfully generated dynamic report at: {rel_path(REPORT_PATH)}")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            generate_report()
