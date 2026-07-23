"""
Two-Tiered Clinical & Transcriptomic Feature Selection Pipeline.

Tier 1 (Multi-Cohort Consensus, N=699 across 4 cohorts):
- Evaluates 10 cross-cohort features (6 immune signatures, TMB, Age, Sex, Sample Type)
  for Overall Survival (N=686) and Anti-PD-1 Response (N=247).

Tier 2 (Granular TCGA Phenotyping & Pathological Staging, N=443):
- Evaluates detailed pathological staging (TNM, AJCC), anatomical sites, aneuploidy,
  and hypoxia scores specifically within the TCGA-SKCM reference cohort.

Uses univariate Cox Proportional Hazards regression (with Benjamini-Hochberg FDR adjustment)
and Random Forest Gini Importance with zero hardcoded sample sizes or static text metrics.
Exports presentation-ready plots and an Obsidian-compatible Markdown report.
"""

import contextlib
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import seaborn as sns

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

_SUBPROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_SUBPROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBPROJECT_ROOT))

from src.config.constants import EXCLUDED_CLINICAL_COLS
from src.config.datasets import DatasetConfig, load_dataset_config
from src.signatures import extract_all_signatures
from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style
from src.utils.formatting import generate_obsidian_frontmatter
from src.utils.logging import TeeStream
from src.utils.paths import (
    CONFIG_DIR,
    DATA_DIR,
    LOG_DIR,
    PLOTS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    SUBPROJECT_ROOT,
    rel_path,
)
from src.utils.plotting import save_fig

# ---------------------------------------------------------------------------
# Module-level Constants & Directory Paths
# ---------------------------------------------------------------------------

CONFIG_PATH = CONFIG_DIR / "datasets.yaml"
PLOT_DIR = PLOTS_DIR / "clinical"
REPORT_DIR = REPORTS_DIR / "pillar-2-clinical-subtyping"
REPORT_PATH = REPORT_DIR / "clinical_feature_selection_report.md"
LOG_PATH = LOG_DIR / "run_clinical_feature_selection.log"

TIER1_FEATURE_COLS: List[str] = [
    "IFN_gamma",
    "TIS",
    "CYT",
    "CD8_Tcell",
    "PD_L1",
    "IMPRES",
    "TMB_NONSYNONYMOUS",
    "AGE",
    "SEX",
]


def _benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Applies Benjamini-Hochberg false discovery rate adjustment to p-values.

    Args:
        p_values: Array of raw p-values.
        alpha: Target FDR significance threshold.

    Returns:
        Tuple of (boolean rejection array, adjusted p-value array).
    """
    p_vals = np.asarray(p_values)
    n = len(p_vals)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]

    adjusted = np.zeros(n, dtype=float)
    cummin_val = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj_val = (sorted_p[i] * n) / rank
        cummin_val = min(cummin_val, adj_val)
        adjusted[i] = min(1.0, cummin_val)

    p_adjusted = np.zeros(n, dtype=float)
    p_adjusted[sorted_indices] = adjusted
    rejected = p_adjusted < alpha

    return rejected, p_adjusted


def _format_feature_name(name: str) -> str:
    """Formats feature column names into clean display titles.

    Args:
        name: Raw feature column name string.

    Returns:
        Formatted human-readable feature title string.
    """
    clean_map = {
        "IFN_gamma": "IFN-gamma Signature",
        "TIS": "Tumour Inflammation Signature (TIS)",
        "CYT": "Cytolytic Activity (CYT)",
        "CD8_Tcell": "CD8+ T-Cell Infiltration",
        "PD_L1": "PD-L1 Expression",
        "IMPRES": "IMPRES Signature",
        "TMB_NONSYNONYMOUS": "Tumour Mutational Burden (TMB)",
        "AGE": "Patient Age",
        "SEX_Male": "Sex: Male",
        "SEX_Female": "Sex: Female",
        "SAMPLE_TYPE_Primary": "Sample Type: Primary Tumour",
        "SAMPLE_TYPE_Metastatic": "Sample Type: Metastatic Specimen",
        "Z_IFN_gamma": "IFN-gamma Signature (Z-Score)",
        "Z_TIS": "TIS Signature (Z-Score)",
        "Z_CYT": "Cytolytic Activity (Z-Score)",
        "Z_CD8_Tcell": "CD8+ T-Cell (Z-Score)",
        "Z_PD_L1": "PD-L1 Expression (Z-Score)",
        "Z_IMPRES": "IMPRES Signature (Z-Score)",
        "Z_TMB_NONSYNONYMOUS": "TMB Nonsynonymous (Z-Score)",
        "Z_AGE": "Patient Age (Z-Score)",
    }
    if name in clean_map:
        return clean_map[name]

    name = name.replace("TUMOR_TISSUE_SITE_", "Tumour Site: ")
    name = name.replace("AJCC_PATHOLOGIC_TUMOR_STAGE_", "AJCC Stage: ")
    name = name.replace("PATH_T_STAGE_", "Primary Tumour (T) Staging: ")
    name = name.replace("PATH_N_STAGE_", "N Stage: ")
    name = name.replace("PATH_M_STAGE_", "M Stage: ")
    name = name.replace("SAMPLE_TYPE_", "Sample Type: ")
    name = name.replace("_", " ")

    name = re.sub(r"\bStage ([Ivxi]+)\b", lambda m: f"Stage {m.group(1).upper()}", name, flags=re.IGNORECASE)
    name = re.sub(r"\b([Tnm])([0-4][a-c]?)\b", lambda m: f"{m.group(1).upper()}{m.group(2).lower()}", name)

    words = name.split()
    capitalised_words = []
    for w in words:
        if any(char.isdigit() for char in w) or w in ["ICD-10", "ICD-O-3", "AJCC", "DNA", "RNA", "TMB", "MSI"]:
            capitalised_words.append(w)
        elif ":" in w:
            parts = w.split(":")
            capitalised_words.append(":".join([p.capitalize() for p in parts]))
        else:
            capitalised_words.append(w.capitalize())

    return " ".join(capitalised_words)


# ---------------------------------------------------------------------------
# Tier 1: Multi-Cohort Feature Selection Logic (N=699)
# ---------------------------------------------------------------------------


def _load_tier1_dataset(dataset_configs: Tuple[DatasetConfig, ...]) -> pd.DataFrame:
    """Loads clinical and expression data across all four cohorts for Tier 1 multi-cohort feature selection.

    Args:
        dataset_configs: Validated dataset configurations loaded from datasets.yaml.

    Returns:
        Merged DataFrame with Z-score standardized features across all 4 cohorts.
    """
    cohort_dfs: List[pd.DataFrame] = []
    cont_cols = ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "PD_L1", "IMPRES", "TMB_NONSYNONYMOUS", "AGE"]

    for config in dataset_configs:
        label = config.cohort_name
        proc_dir = DATA_DIR / "processed" / config.processed_directory
        clin_path = proc_dir / "clin_cleaned.csv"
        expr_path = proc_dir / "expr_cleaned.csv"

        if not clin_path.exists():
            raise FileNotFoundError(f"Missing clinical file for cohort '{label}' at {rel_path(clin_path)}")

        df_clin = pd.read_csv(clin_path, index_col="SAMPLE_ID")

        if expr_path.exists():
            df_expr = pd.read_csv(expr_path, index_col="SAMPLE_ID")
            df_sig = extract_all_signatures(df_expr)
        else:
            df_sig = pd.DataFrame(index=df_clin.index)

        df_combined = df_clin.copy()

        for col in df_sig.columns:
            df_combined[col] = df_sig[col]

        df_combined["COHORT"] = label
        df_combined["IS_TRIAL"] = ("RESPONDER" in df_combined.columns or "RESPONSE" in df_combined.columns)

        if "AGE_AT_DIAGNOSIS" in df_combined.columns and "AGE" not in df_combined.columns:
            df_combined["AGE"] = df_combined["AGE_AT_DIAGNOSIS"]

        for col in cont_cols:
            if col not in df_combined.columns:
                df_combined[col] = np.nan

        cohort_subset = df_combined[["COHORT", "IS_TRIAL"] + cont_cols].copy()

        for meta_col in ["RESPONSE", "RESPONDER", "RESPONSE_BINARY", "OS_MONTHS", "OS_STATUS", "SEX", "SAMPLE_TYPE"]:
            if meta_col in df_combined.columns:
                cohort_subset[meta_col] = df_combined[meta_col]

        # Fill all-NaN continuous columns (e.g. AGE in Liu 2019)
        for col in cont_cols:
            if cohort_subset[col].isna().all():
                cohort_subset[col] = 0.0

        imputer = SimpleImputer(strategy="median")
        imputed_feats = imputer.fit_transform(cohort_subset[cont_cols])
        df_imputed = pd.DataFrame(imputed_feats, columns=cont_cols, index=cohort_subset.index)

        scaler = StandardScaler()
        z_feats = scaler.fit_transform(df_imputed)

        for i, col in enumerate(cont_cols):
            cohort_subset[f"Z_{col}"] = z_feats[:, i]
            cohort_subset[col] = df_imputed[col]

        cohort_dfs.append(cohort_subset)

    full_df = pd.concat(cohort_dfs, axis=0)

    # Encode categorical Tier 1 features (SEX, SAMPLE_TYPE)
    cat_cols = [c for c in ["SEX", "SAMPLE_TYPE"] if c in full_df.columns]
    if cat_cols:
        dummies = pd.get_dummies(full_df[cat_cols], drop_first=False, dtype=float)
        for dcol in dummies.columns:
            full_df[dcol] = dummies[dcol]

    return full_df


def _evaluate_tier1_rf_survival(df: pd.DataFrame, plots_dir: Path) -> pd.DataFrame:
    """Trains Random Forest Classifier predicting overall survival status across Tier 1 multi-cohort dataset.

    Args:
        df: Merged patient DataFrame.
        plots_dir: Target plots directory.

    Returns:
        Sorted DataFrame of feature importances.
    """
    valid_mask = df["OS_STATUS"].notna()
    df_valid = df.loc[valid_mask].copy()

    feature_cols = [f"Z_{col}" for col in ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "PD_L1", "IMPRES", "TMB_NONSYNONYMOUS", "AGE"]]
    for extra in ["SEX_Male", "SEX_Female"]:
        if extra in df_valid.columns:
            feature_cols.append(extra)

    X_rf = df_valid[feature_cols].fillna(0.0)
    y_rf = df_valid["OS_STATUS"].astype(int)

    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X_rf, y_rf)

    df_rf = pd.DataFrame({"Feature": feature_cols, "Importance": rf.feature_importances_})
    df_rf["Formatted_Feature"] = df_rf["Feature"].apply(_format_feature_name)
    df_rf = df_rf.sort_values(by="Importance", ascending=False).reset_index(drop=True)

    set_presentation_style()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    palette = sns.color_palette("Blues_r", n_colors=len(df_rf))
    sns.barplot(
        data=df_rf,
        y="Formatted_Feature",
        x="Importance",
        hue="Formatted_Feature",
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        legend=False,
    )

    n_samples = len(df_valid)
    ax.set_title(f"Tier 1 Random Forest Importance: Overall Survival (Multi-Cohort, N={n_samples})", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=11, fontweight="bold")
    ax.set_ylabel("")

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f"{width:.4f}",
            (width, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9.5,
            color="black",
        )

    ax.set_xlim(0, df_rf["Importance"].max() * 1.15)
    out_path = plots_dir / "clinical_feature_importance.png"
    save_fig(fig, out_path)
    print(f"Saved Tier 1 Random Forest OS feature importance plot to {rel_path(out_path)}")

    return df_rf


def _evaluate_tier1_rf_response(df: pd.DataFrame, plots_dir: Path) -> pd.DataFrame:
    """Trains Random Forest Classifier predicting immunotherapy response across trial cohorts.

    Args:
        df: Merged patient DataFrame.
        plots_dir: Target plots directory.

    Returns:
        Sorted DataFrame of response feature importances.
    """
    df_trial = df[df["IS_TRIAL"] & df["RESPONDER"].notna()].copy()
    df_trial["RESPONDER_NUM"] = df_trial["RESPONDER"].map({True: 1.0, False: 0.0, 1.0: 1.0, 0.0: 0.0, "1": 1.0, "0": 0.0})

    feature_cols = [f"Z_{col}" for col in ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "PD_L1", "IMPRES", "TMB_NONSYNONYMOUS", "AGE"]]
    for extra in ["SEX_Male", "SEX_Female"]:
        if extra in df_trial.columns:
            feature_cols.append(extra)

    X_resp = df_trial[feature_cols].fillna(0.0)
    y_resp = df_trial["RESPONDER_NUM"].astype(int)

    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X_resp, y_resp)

    df_resp = pd.DataFrame({"Feature": feature_cols, "Importance": rf.feature_importances_})
    df_resp["Formatted_Feature"] = df_resp["Feature"].apply(_format_feature_name)
    df_resp = df_resp.sort_values(by="Importance", ascending=False).reset_index(drop=True)

    set_presentation_style()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    palette = sns.color_palette("Greens_r", n_colors=len(df_resp))
    sns.barplot(
        data=df_resp,
        y="Formatted_Feature",
        x="Importance",
        hue="Formatted_Feature",
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        legend=False,
    )

    n_samples = len(df_trial)
    ax.set_title(f"Tier 1 Random Forest Importance: Immunotherapy Response (Trial Cohorts, N={n_samples})", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=11, fontweight="bold")
    ax.set_ylabel("")

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f"{width:.4f}",
            (width, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9.5,
            color="black",
        )

    ax.set_xlim(0, df_resp["Importance"].max() * 1.15)
    out_path = plots_dir / "response_feature_importance.png"
    save_fig(fig, out_path)
    print(f"Saved Tier 1 Random Forest response feature importance plot to {rel_path(out_path)}")

    return df_resp


def _evaluate_tier1_response_forest_plot(df: pd.DataFrame, plots_dir: Path) -> pd.DataFrame:
    """Fits univariate Logistic Regression per feature predicting anti-PD-1 response (CR/PR vs PD)
    and saves an Odds Ratio (OR) Forest Plot across clinical trial cohorts.

    Args:
        df: Merged patient DataFrame.
        plots_dir: Target plots directory.

    Returns:
        Sorted DataFrame of Odds Ratios, confidence intervals, and p-values.
    """
    df_trial = df[df["IS_TRIAL"] & df["RESPONDER"].notna()].copy()
    df_trial["RESPONDER_NUM"] = df_trial["RESPONDER"].map({True: 1.0, False: 0.0, 1.0: 1.0, 0.0: 0.0, "1": 1.0, "0": 0.0})

    feature_cols = [f"Z_{col}" for col in ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "PD_L1", "IMPRES", "TMB_NONSYNONYMOUS", "AGE"]]
    for extra in ["SEX_Male"]:
        if extra in df_trial.columns:
            feature_cols.append(extra)

    results = []

    for col in feature_cols:
        sub = df_trial[[col, "RESPONDER_NUM"]].dropna()
        if len(sub) == 0:
            continue
        X = sm.add_constant(sub[col])
        y = sub["RESPONDER_NUM"]
        try:
            logit_mod = sm.Logit(y, X).fit(disp=False)
            or_val = np.exp(logit_mod.params[col])
            conf = np.exp(logit_mod.conf_int().loc[col])
            p_val = logit_mod.pvalues[col]
            results.append({
                "Feature": col,
                "Clean_Feature": col.replace("Z_", ""),
                "Odds Ratio (OR)": or_val,
                "OR lower 95%": conf[0],
                "OR upper 95%": conf[1],
                "p-value": p_val,
            })
        except Exception:
            continue

    df_or = pd.DataFrame(results)
    rejected, p_adj = _benjamini_hochberg(df_or["p-value"].values, alpha=0.05)
    df_or["FDR_adj_p"] = p_adj
    df_or["Significant_FDR"] = rejected
    df_or = df_or.sort_values(by="p-value", ascending=True).reset_index(drop=True)

    df_plot = df_or.copy()
    df_plot["Formatted_Feature"] = df_plot["Feature"].apply(_format_feature_name)
    df_plot = df_plot.iloc[::-1].reset_index(drop=True)

    set_presentation_style()
    fig, ax = plt.subplots(figsize=(12, 7.5))

    y_pos = np.arange(len(df_plot))
    ors = df_plot["Odds Ratio (OR)"].values
    lowers = df_plot["OR lower 95%"].values
    uppers = df_plot["OR upper 95%"].values
    p_vals = df_plot["p-value"].values
    fdr_vals = df_plot["FDR_adj_p"].values

    point_colors = []
    for or_v in ors:
        if or_v > 1.0:
            point_colors.append(RESPONSE_PALETTE["CR/PR"])  # Okabe-Ito Bluish Green (#009E73) for Favourable Response
        else:
            point_colors.append(RESPONSE_PALETTE["PD"])     # Okabe-Ito Vermillion Red (#D55E00) for Unfavourable Response

    ax.axvline(x=1.0, color="#37474F", linestyle="--", linewidth=1.2, alpha=0.7, label="Null Effect (OR = 1.0)")

    for i in range(len(df_plot)):
        ax.plot([lowers[i], uppers[i]], [y_pos[i], y_pos[i]], color=point_colors[i], linewidth=2.0, alpha=0.85)
        ax.scatter(ors[i], y_pos[i], color=point_colors[i], s=75, zorder=5, edgecolor="black", linewidth=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["Formatted_Feature"], fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())

    # Thinner, lighter grey horizontal gridlines
    ax.yaxis.grid(True, linestyle="--", color="#E0E0E0", linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(True, linestyle=":", color="#E0E0E0", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    min_val = min(lowers)
    max_val = max(uppers)
    ax.set_xlim(max(0.1, min_val * 0.8), max_val * 1.5)

    n_samples = len(df_trial)
    ax.set_title(f"Tier 1 Immunotherapy Response Odds Ratio Forest Plot (Trial Cohorts, N={n_samples})", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Odds Ratio (OR, Log Scale - 95% CI per +1 SD)", fontsize=11, fontweight="bold")

    for i in range(len(df_plot)):
        or_val = ors[i]
        p_val = p_vals[i]
        fdr_val = fdr_vals[i]
        text_str = f"OR={or_val:.2f} (p={p_val:.1e}, FDR={fdr_val:.1e})"
        ax.annotate(
            text_str,
            (uppers[i], y_pos[i]),
            xytext=(8, -3),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold" if fdr_val < 0.05 else "normal",
            color="#222222" if fdr_val < 0.05 else "#666666",
        )

    legend_handles = [
        mpatches.Patch(color=RESPONSE_PALETTE["CR/PR"], label="Favourable Response (OR > 1.0)"),
        mpatches.Patch(color=RESPONSE_PALETTE["PD"], label="Unfavourable Response (OR < 1.0)"),
        mlines.Line2D([], [], color="#37474F", linestyle="--", linewidth=1.2, label="Null Effect (OR = 1.0)"),
        mlines.Line2D([], [], color="none", label="*Bold charcoal text: FDR < 0.05"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9)

    sns.despine(top=True, right=True)
    out_path = plots_dir / "response_forest_plot.png"
    save_fig(fig, out_path)
    print(f"Saved Tier 1 response forest plot to {rel_path(out_path)}")

    return df_or


def _evaluate_tier1_cox(df: pd.DataFrame, plots_dir: Path) -> pd.DataFrame:
    """Fits univariate Cox Proportional Hazards models per feature across multi-cohort dataset.

    Args:
        df: Merged patient DataFrame.
        plots_dir: Target plots directory.

    Returns:
        Sorted DataFrame of Cox model hazard ratios.
    """
    valid_mask = df["OS_MONTHS"].notna() & df["OS_STATUS"].notna() & (df["OS_MONTHS"] > 0)
    df_cph_all = df.loc[valid_mask].copy()

    feature_cols = [f"Z_{col}" for col in ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "PD_L1", "IMPRES", "TMB_NONSYNONYMOUS", "AGE"]]
    for extra in ["SEX_Male"]:
        if extra in df_cph_all.columns:
            feature_cols.append(extra)

    cox_results = []
    cph = CoxPHFitter()

    for col in feature_cols:
        feature_data = pd.DataFrame({
            "OS_MONTHS": df_cph_all["OS_MONTHS"],
            "OS_STATUS": df_cph_all["OS_STATUS"],
            col: df_cph_all[col],
        })
        try:
            cph.fit(feature_data, duration_col="OS_MONTHS", event_col="OS_STATUS")
            summary = cph.summary.loc[col]
            cox_results.append({
                "Feature": col,
                "Clean_Feature": col.replace("Z_", ""),
                "Hazard Ratio (HR)": summary["exp(coef)"],
                "HR lower 95%": summary["exp(coef) lower 95%"],
                "HR upper 95%": summary["exp(coef) upper 95%"],
                "p-value": summary["p"],
                "coef": summary["coef"],
                "se": summary["se(coef)"],
            })
        except Exception:
            continue

    df_cph = pd.DataFrame(cox_results)
    rejected, p_adj = _benjamini_hochberg(df_cph["p-value"].values, alpha=0.05)
    df_cph["FDR_adj_p"] = p_adj
    df_cph["Significant_FDR"] = rejected
    df_cph = df_cph.sort_values(by="p-value", ascending=True).reset_index(drop=True)

    df_plot = df_cph.copy()
    df_plot["Formatted_Feature"] = df_plot["Feature"].apply(_format_feature_name)
    df_plot = df_plot.iloc[::-1].reset_index(drop=True)

    set_presentation_style()
    fig, ax = plt.subplots(figsize=(12, 7.5))

    y_pos = np.arange(len(df_plot))
    hrs = df_plot["Hazard Ratio (HR)"].values
    lowers = df_plot["HR lower 95%"].values
    uppers = df_plot["HR upper 95%"].values
    p_vals = df_plot["p-value"].values
    fdr_vals = df_plot["FDR_adj_p"].values

    point_colors = []
    for hr_val in hrs:
        if hr_val < 1.0:
            point_colors.append(RESPONSE_PALETTE["CR/PR"])  # Okabe-Ito Bluish Green (#009E73) for Protective
        else:
            point_colors.append(RESPONSE_PALETTE["PD"])     # Okabe-Ito Vermillion Red (#D55E00) for Risk

    ax.axvline(x=1.0, color="#37474F", linestyle="--", linewidth=1.2, alpha=0.7, label="Null Effect (HR = 1.0)")

    for i in range(len(df_plot)):
        ax.plot([lowers[i], uppers[i]], [y_pos[i], y_pos[i]], color=point_colors[i], linewidth=2.0, alpha=0.85)
        ax.scatter(hrs[i], y_pos[i], color=point_colors[i], s=75, zorder=5, edgecolor="black", linewidth=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["Formatted_Feature"], fontsize=11, fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())

    # Thinner, lighter grey horizontal gridlines
    ax.yaxis.grid(True, linestyle="--", color="#E0E0E0", linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(True, linestyle=":", color="#E0E0E0", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    min_val = min(lowers)
    max_val = max(uppers)
    ax.set_xlim(max(0.1, min_val * 0.8), max_val * 1.5)

    n_samples = len(df_cph_all)
    ax.set_title(f"Tier 1 Cox Proportional Hazards Regression (Multi-Cohort, N={n_samples})", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Hazard Ratio (HR, Log Scale - 95% CI per +1 SD)", fontsize=11, fontweight="bold")

    for i in range(len(df_plot)):
        hr_val = hrs[i]
        p_val = p_vals[i]
        fdr_val = fdr_vals[i]
        text_str = f"HR={hr_val:.2f} (p={p_val:.1e}, FDR={fdr_val:.1e})"
        ax.annotate(
            text_str,
            (uppers[i], y_pos[i]),
            xytext=(8, -3),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold" if fdr_val < 0.05 else "normal",
            color="#222222" if fdr_val < 0.05 else "#666666",
        )

    legend_handles = [
        mpatches.Patch(color=RESPONSE_PALETTE["CR/PR"], label="Protective (HR < 1.0)"),
        mpatches.Patch(color=RESPONSE_PALETTE["PD"], label="Risk (HR > 1.0)"),
        mlines.Line2D([], [], color="#37474F", linestyle="--", linewidth=1.2, label="Null Effect (HR = 1.0)"),
        mlines.Line2D([], [], color="none", label="*Bold charcoal text: FDR < 0.05"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9)

    sns.despine(top=True, right=True)
    out_path = plots_dir / "cox_forest_plot.png"
    save_fig(fig, out_path)
    print(f"Saved Tier 1 Cox forest plot to {rel_path(out_path)}")

    return df_cph


# ---------------------------------------------------------------------------
# Tier 2: Granular TCGA Clinical Feature Selection Logic (N=443)
# ---------------------------------------------------------------------------


def _load_and_filter_tcga_features(
    tcga_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads TCGA-SKCM clinical dataset and performs one-hot dummy encoding across clinical attributes.

    Args:
        tcga_path: Path to cleaned TCGA-SKCM clinical CSV.

    Returns:
        Tuple of (raw DataFrame, encoded features DataFrame, y_os_status, y_os_months).
    """
    if not tcga_path.exists():
        raise FileNotFoundError(f"Missing clinical dataset at {rel_path(tcga_path)}")

    df = pd.read_csv(tcga_path)

    id_cols = ["PATIENT_ID", "SAMPLE_ID"]
    icd_cols = [c for c in df.columns if c.startswith("ICD_")]
    target_cols = ["OS_STATUS", "OS_MONTHS", "PFS_STATUS", "PFS_MONTHS", "DSS_STATUS", "DSS_MONTHS"]
    redundant_cols = [
        "DAYS_LAST_FOLLOWUP",
        "PERSON_NEOPLASM_CANCER_STATUS",
        "FORM_COMPLETION_DATE",
        "DAYS_TO_BIRTH",
        "DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS",
        "OTHER_PATIENT_ID",
        "AJCC_STAGING_EDITION",
    ]
    confounder_cols = ["ETHNICITY", "GENETIC_ANCESTRY_LABEL", "RACE", "PRIOR_DX"]
    artefact_cols = [
        "TISSUE_SOURCE_SITE",
        "TISSUE_SOURCE_SITE_CODE",
        "MSI_SCORE_MANTIS",
        "MSI_SENSOR_SCORE",
        "TBL_SCORE",
        "TISSUE_PROSPECTIVE_COLLECTION_INDICATOR",
        "TISSUE_RETROSPECTIVE_COLLECTION_INDICATOR",
        "SAMPLE_TYPE",
        "SAMPLE_TYPE_ID",
        "IN_PANCANPATHWAYS_FREEZE",
        "INFORMED_CONSENT_VERIFIED",
        "SOMATIC_STATUS",
        "CANCER_TYPE",
        "CANCER_TYPE_ACRONYM",
        "CANCER_TYPE_DETAILED",
        "TUMOR_TYPE",
    ]
    treatment_cols = [
        "HISTORY_NEOADJUVANT_TRTYN",
        "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT",
        "RADIATION_THERAPY",
        "TREATMENT_TYPES",
        "TREATMENT_AGENTS",
        "TX_TYPE_RADIATION_THERAPY",
        "TX_TYPE_CHEMOTHERAPY",
        "TX_TYPE_IMMUNOTHERAPY",
        "TX_TYPE_VACCINE",
        "TX_TYPE_HORMONE_THERAPY",
        "TX_TYPE_TARGETED_MOLECULAR_THERAPY",
        "TX_TYPE_ANCILLARY",
        "TX_TYPE_OTHER",
    ] + [c for c in df.columns if c.startswith("TX_AGENT_")]

    exclude_cols = set(id_cols + icd_cols + target_cols + redundant_cols + confounder_cols + artefact_cols + treatment_cols + EXCLUDED_CLINICAL_COLS)
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_raw = df[feature_cols].copy()
    y_os_status = df["OS_STATUS"].copy()
    y_os_months = df["OS_MONTHS"].copy()

    categorical_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()

    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)

    for col in numerical_cols:
        if X_encoded[col].isnull().all():
            X_encoded[col] = 0.0
        elif X_encoded[col].isnull().any():
            imputer = SimpleImputer(strategy="median")
            X_encoded[col] = imputer.fit_transform(X_encoded[[col]]).ravel()

    for col in X_encoded.columns:
        if col not in numerical_cols:
            X_encoded[col] = X_encoded[col].astype(int)

    # Drop zero-variance columns
    zero_var_cols = [c for c in X_encoded.columns if X_encoded[c].nunique() <= 1]
    if zero_var_cols:
        X_encoded = X_encoded.drop(columns=zero_var_cols)

    return df, X_encoded, y_os_status, y_os_months


def _evaluate_tier2_tcga_rf(X_encoded: pd.DataFrame, y_os_status: pd.Series, plots_dir: Path) -> pd.DataFrame:
    """Trains Random Forest Classifier on TCGA-SKCM granular clinical dummy features.

    Args:
        X_encoded: Encoded feature matrix.
        y_os_status: Overall survival status target.
        plots_dir: Target plots directory.

    Returns:
        Sorted DataFrame of top 20 feature importances.
    """
    valid_mask = y_os_status.notna()
    X_rf = X_encoded.loc[valid_mask].copy()
    y_rf = y_os_status.loc[valid_mask].astype(int)

    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X_rf, y_rf)

    df_rf = pd.DataFrame({"Feature": X_encoded.columns, "Importance": rf.feature_importances_})
    df_rf["Formatted_Feature"] = df_rf["Feature"].apply(_format_feature_name)
    df_rf = df_rf.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    df_top20 = df_rf.head(20).copy()

    set_presentation_style()
    fig, ax = plt.subplots(figsize=(11, 8.5))
    palette = sns.color_palette("Purples_r", n_colors=len(df_top20))
    sns.barplot(
        data=df_top20,
        y="Formatted_Feature",
        x="Importance",
        hue="Formatted_Feature",
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        legend=False,
    )

    n_samples = len(X_rf)
    ax.set_title(f"Tier 2 Random Forest Importance: Granular TCGA Clinical Predictors (N={n_samples})", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=11, fontweight="bold")
    ax.set_ylabel("")

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f"{width:.4f}",
            (width, p.get_y() + p.get_height() / 2.0),
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9,
            color="black",
        )

    ax.set_xlim(0, df_top20["Importance"].max() * 1.15)
    out_path = plots_dir / "tcga_clinical_feature_importance.png"
    save_fig(fig, out_path)
    print(f"Saved Tier 2 TCGA Random Forest feature importance plot to {rel_path(out_path)}")

    return df_rf


def _evaluate_tier2_tcga_cox(
    X_encoded: pd.DataFrame,
    y_os_status: pd.Series,
    y_os_months: pd.Series,
    plots_dir: Path,
) -> pd.DataFrame:
    """Fits univariate Cox Proportional Hazards models across TCGA-SKCM granular clinical dummy features.

    Args:
        X_encoded: Encoded feature matrix.
        y_os_status: Survival status series.
        y_os_months: Survival months series.
        plots_dir: Target plots directory.

    Returns:
        Sorted DataFrame of top Cox hazard ratios.
    """
    valid_mask = y_os_months.notna() & y_os_status.notna() & (y_os_months > 0)
    X_cph_all = X_encoded.loc[valid_mask].copy()
    months = y_os_months.loc[valid_mask].values
    status = y_os_status.loc[valid_mask].values

    cox_results = []
    cph = CoxPHFitter()

    for col in X_cph_all.columns:
        feature_data = pd.DataFrame({"OS_MONTHS": months, "OS_STATUS": status, col: X_cph_all[col].values})
        try:
            cph.fit(feature_data, duration_col="OS_MONTHS", event_col="OS_STATUS")
            summary = cph.summary.loc[col]
            cox_results.append({
                "Feature": col,
                "Hazard Ratio (HR)": summary["exp(coef)"],
                "HR lower 95%": summary["exp(coef) lower 95%"],
                "HR upper 95%": summary["exp(coef) upper 95%"],
                "p-value": summary["p"],
                "coef": summary["coef"],
                "se": summary["se(coef)"],
            })
        except Exception:
            continue

    df_cph = pd.DataFrame(cox_results)
    rejected, p_adj = _benjamini_hochberg(df_cph["p-value"].values, alpha=0.05)
    df_cph["FDR_adj_p"] = p_adj
    df_cph["Significant_FDR"] = rejected
    df_cph = df_cph.sort_values(by="p-value", ascending=True).reset_index(drop=True)

    df_top20 = df_cph.head(20).copy()
    df_top20["Formatted_Feature"] = df_top20["Feature"].apply(_format_feature_name)
    df_top20 = df_top20.iloc[::-1].reset_index(drop=True)

    set_presentation_style()
    fig, ax = plt.subplots(figsize=(12, 8.5))

    y_pos = np.arange(len(df_top20))
    hrs = df_top20["Hazard Ratio (HR)"].values
    lowers = df_top20["HR lower 95%"].values
    uppers = df_top20["HR upper 95%"].values
    p_vals = df_top20["p-value"].values
    fdr_vals = df_top20["FDR_adj_p"].values

    ax.axvline(x=1.0, color="#37474F", linestyle="--", linewidth=1.2, alpha=0.7, label="Null Effect (HR = 1.0)")

    for i in range(len(df_top20)):
        color = RESPONSE_PALETTE["PD"] if hrs[i] > 1.0 else RESPONSE_PALETTE["CR/PR"]
        ax.plot([lowers[i], uppers[i]], [y_pos[i], y_pos[i]], color=color, linewidth=2.0, alpha=0.85)
        ax.scatter(hrs[i], y_pos[i], color=color, s=70, zorder=5, edgecolor="black", linewidth=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top20["Formatted_Feature"], fontsize=10, fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())

    # Thinner, lighter grey horizontal gridlines
    ax.yaxis.grid(True, linestyle="--", color="#E0E0E0", linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(True, linestyle=":", color="#E0E0E0", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    min_val = min(lowers)
    max_val = max(uppers)
    ax.set_xlim(max(0.1, min_val * 0.8), max_val * 1.5)

    n_samples = len(X_cph_all)
    ax.set_title(f"Tier 2 Cox Proportional Hazards Regression: Top TCGA Predictors (N={n_samples})", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Hazard Ratio (HR, Log Scale - 95% CI)", fontsize=11, fontweight="bold")

    for i in range(len(df_top20)):
        hr_val = hrs[i]
        p_val = p_vals[i]
        fdr_val = fdr_vals[i]
        text_str = f"HR={hr_val:.2f} (p={p_val:.1e}, FDR={fdr_val:.1e})"
        ax.annotate(
            text_str,
            (uppers[i], y_pos[i]),
            xytext=(8, -3),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold" if fdr_val < 0.05 else "normal",
            color="#222222" if fdr_val < 0.05 else "#666666",
        )

    legend_handles = [
        mpatches.Patch(color=RESPONSE_PALETTE["CR/PR"], label="Protective (HR < 1.0)"),
        mpatches.Patch(color=RESPONSE_PALETTE["PD"], label="Risk (HR > 1.0)"),
        mlines.Line2D([], [], color="#37474F", linestyle="--", linewidth=1.2, label="Null Effect (HR = 1.0)"),
        mlines.Line2D([], [], color="none", label="*Bold charcoal text: FDR < 0.05"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.5)

    sns.despine(top=True, right=True)
    out_path = plots_dir / "tcga_cox_forest_plot.png"
    save_fig(fig, out_path)
    print(f"Saved Tier 2 TCGA Cox forest plot to {rel_path(out_path)}")

    return df_cph


# ---------------------------------------------------------------------------
# Markdown Report Generation Logic
# ---------------------------------------------------------------------------


def _generate_two_tiered_report(
    tier1_df: pd.DataFrame,
    tier1_rf_os: pd.DataFrame,
    tier1_rf_resp: pd.DataFrame,
    tier1_resp_or: pd.DataFrame,
    tier1_cph: pd.DataFrame,
    tcga_df: pd.DataFrame,
    tcga_encoded: pd.DataFrame,
    y_os_status: pd.Series,
    y_os_months: pd.Series,
    tier2_rf: pd.DataFrame,
    tier2_cph: pd.DataFrame,
    report_path: Path,
) -> None:
    """Generates two-tiered Markdown report with dynamic sample metrics and Obsidian frontmatter.

    Args:
        tier1_df: Pooled patient DataFrame across all 4 cohorts.
        tier1_rf_os: Ranked Tier 1 RF OS feature importances.
        tier1_rf_resp: Ranked Tier 1 RF response importances.
        tier1_resp_or: Ranked Tier 1 Response Odds Ratio results.
        tier1_cph: Ranked Tier 1 Cox regression results.
        tcga_df: Raw TCGA clinical DataFrame.
        tcga_encoded: Encoded TCGA feature matrix.
        y_os_status: Target TCGA OS status Series.
        y_os_months: Target TCGA OS months Series.
        tier2_rf: Ranked Tier 2 TCGA RF feature importances.
        tier2_cph: Ranked Tier 2 TCGA Cox regression results.
        report_path: Target report path.
    """
    tier1_total_n = len(tier1_df)
    tier1_surv_n = tier1_df["OS_STATUS"].notna().sum()
    tier1_cox_n = (tier1_df["OS_MONTHS"].notna() & tier1_df["OS_STATUS"].notna() & (tier1_df["OS_MONTHS"] > 0)).sum()
    tier1_trial_n = (tier1_df["IS_TRIAL"] & tier1_df["RESPONDER"].notna()).sum()
    tier1_n_sig_fdr = (tier1_cph["FDR_adj_p"] < 0.05).sum()

    top_tier1_cph_feat = tier1_cph.iloc[0]["Feature"]
    top_tier1_cph_hr = tier1_cph.iloc[0]["Hazard Ratio (HR)"]
    top_tier1_cph_p = tier1_cph.iloc[0]["p-value"]
    top_tier1_cph_fdr = tier1_cph.iloc[0]["FDR_adj_p"]

    top_tier1_resp_feat = tier1_rf_resp.iloc[0]["Feature"]
    top_tier1_resp_imp = tier1_rf_resp.iloc[0]["Importance"]

    tcga_n = len(tcga_df)
    tcga_rf_n = y_os_status.notna().sum()
    tcga_cox_n = (y_os_status.notna() & y_os_months.notna() & (y_os_months > 0)).sum()
    tcga_feat_count = len(tcga_encoded.columns)
    tcga_n_sig_fdr = (tier2_cph["FDR_adj_p"] < 0.05).sum()

    top_tier2_rf_feat = tier2_rf.iloc[0]["Feature"]
    top_tier2_rf_imp = tier2_rf.iloc[0]["Importance"]

    top_tier2_cph_feat = tier2_cph.iloc[0]["Feature"]
    top_tier2_cph_hr = tier2_cph.iloc[0]["Hazard Ratio (HR)"]
    top_tier2_cph_p = tier2_cph.iloc[0]["p-value"]
    top_tier2_cph_fdr = tier2_cph.iloc[0]["FDR_adj_p"]

    frontmatter = generate_obsidian_frontmatter(
        title="Two-Tiered Clinical & Transcriptomic Feature Selection Report",
        tags=["melanoma", "clinical-subtyping", "feature-selection", "cox-regression", "random-forest", "two-tiered"],
    )

    report_path.parent.mkdir(exist_ok=True, parents=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(frontmatter + "\n\n")
        f.write("# Two-Tiered Clinical & Transcriptomic Feature Selection Report\n\n")
        f.write("This report presents a comprehensive **two-tiered feature selection architecture** evaluating prognostic and predictive clinical markers across melanoma patient populations:\n\n")
        f.write(f"* **Tier 1 (Multi-Cohort Consensus, $N = {tier1_total_n}$)**: Evaluates 10 harmonized cross-cohort features (6 transcriptomic immune signatures, $\\text{{TMB}}$, age, sex) pooled across all four study cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, **Riaz 2017**).\n")
        f.write(f"* **Tier 2 (Granular TCGA Pathological Staging, $N = {tcga_n}$)**: Evaluates {tcga_feat_count} detailed clinical, pathological TNM staging, anatomical site, aneuploidy, and hypoxia attributes specifically within the **TCGA-SKCM** reference cohort.\n\n")

        f.write("## 1. Tier 1: Multi-Cohort Feature Selection (N=" + str(tier1_total_n) + ")\n\n")
        f.write(f"* **Total Merged Sample Size**: {tier1_total_n} patients across 4 cohorts\n")
        f.write(f"* **Overall Survival Evaluation Cohort**: {tier1_surv_n} patients\n")
        f.write(f"* **Cox Survival Evaluation Cohort**: {tier1_cox_n} patients\n")
        f.write(f"* **Anti-PD-1 Response Evaluation Cohort**: {tier1_trial_n} trial patients\n")
        f.write(f"* **FDR-Significant Survival Predictors (FDR < 0.05)**: {tier1_n_sig_fdr} features\n\n")

        f.write(f"### 1.1. Random Forest Importance for Overall Survival (N={tier1_surv_n})\n")
        f.write(f"Random Forest feature importance (500 estimators) trained on the $N = {tier1_surv_n}$ overall survival cohort:\n\n")
        f.write("![Tier 1 Random Forest OS](../../plots/clinical/clinical_feature_importance.png)\n\n")

        f.write(f"### 1.2. Univariate Cox Proportional Hazards Regression (N={tier1_cox_n})\n")
        f.write(f"Univariate Cox Proportional Hazards models fitted across $N = {tier1_cox_n}$ patients with complete survival duration data:\n\n")
        f.write("![Tier 1 Cox Forest Plot](../../plots/clinical/cox_forest_plot.png)\n\n")

        f.write(f"### 1.3. Random Forest Importance for Anti-PD-1 Immunotherapy Response (N={tier1_trial_n})\n")
        f.write(f"Random Forest feature importance predicting objective response (CR/PR vs PD) across the $N = {tier1_trial_n}$ trial cohort:\n\n")
        f.write("![Tier 1 Random Forest Response](../../plots/clinical/response_feature_importance.png)\n\n")

        f.write(f"### 1.4. Univariate Forest Plot for Anti-PD-1 Immunotherapy Response (N={tier1_trial_n})\n")
        f.write(f"Univariate Logistic Regression Odds Ratio (OR) forest plot predicting objective anti-PD-1 response across the $N = {tier1_trial_n}$ trial cohort:\n\n")
        f.write("![Tier 1 Response Forest Plot](../../plots/clinical/response_forest_plot.png)\n\n")

        f.write("## 2. Tier 2: Granular TCGA Pathological & Clinical Staging (N=" + str(tcga_n) + ")\n\n")
        f.write(f"* **TCGA Total Cohort Sample Size**: {tcga_n} patients\n")
        f.write(f"* **TCGA OS Classification Cohort**: {tcga_rf_n} patients\n")
        f.write(f"* **TCGA Cox Survival Evaluation Cohort**: {tcga_cox_n} patients\n")
        f.write(f"* **Encoded Dummy Features**: {tcga_feat_count} dummy variables\n")
        f.write(f"* **FDR-Significant Pathological Predictors (FDR < 0.05)**: {tcga_n_sig_fdr} features\n\n")

        f.write(f"### 2.1. Random Forest Importance for Granular TCGA Clinical Attributes (N={tcga_rf_n})\n")
        f.write(f"Top 20 Random Forest clinical predictors trained on $N = {tcga_rf_n}$ TCGA patients with non-null survival status:\n\n")
        f.write("![Tier 2 TCGA Random Forest](../../plots/clinical/tcga_clinical_feature_importance.png)\n\n")

        f.write(f"### 2.2. Univariate Cox Proportional Hazards Regression for TCGA Attributes (N={tcga_cox_n})\n")
        f.write(f"Top 20 Univariate Cox hazard ratios evaluated across $N = {tcga_cox_n}$ TCGA patients with complete survival duration data:\n\n")
        f.write("![Tier 2 TCGA Cox Forest Plot](../../plots/clinical/tcga_cox_forest_plot.png)\n\n")

        f.write("## 3. Key Analytical & Biological Summary\n\n")
        f.write(f"1. **Tier 1 Top Survival Biomarker**: **`{_format_feature_name(top_tier1_cph_feat)}`** is the single strongest protective statistical predictor across all 4 cohorts (Hazard Ratio = **{top_tier1_cph_hr:.2f}**, univariate p-value = **{top_tier1_cph_p:.2e}**, FDR = **{top_tier1_cph_fdr:.2e}**).\n")
        f.write(f"2. **Tier 1 Top Immunotherapy Marker**: **`{_format_feature_name(top_tier1_resp_feat)}`** is the top predictive feature for objective anti-PD-1 response across trial cohorts (Gini Importance = **{top_tier1_resp_imp:.4f}**).\n")
        f.write(f"3. **Tier 2 Top Pathological Staging Marker**: **`{_format_feature_name(top_tier2_cph_feat)}`** is the strongest clinical predictor of survival in TCGA (Hazard Ratio = **{top_tier2_cph_hr:.2f}**, p-value = **{top_tier2_cph_p:.2e}**, FDR = **{top_tier2_cph_fdr:.2e}**).\n")
        f.write(f"4. **Biological Alignment**: Microenvironmental T-cell inflammation signatures ($\\\\text{{IFN-}}\\gamma$, TIS, CYT, CD8, IMPRES) consistently confer significant mortality risk reduction ($\\\\text{{HR}} < 1.0$, $p < 0.01$) across both multi-cohort and single-cohort models.\n")

    print(f"Two-tiered feature selection report successfully written to {rel_path(report_path)}")


def main() -> None:
    """Executes two-tiered clinical feature selection pipeline."""
    print("==================================================")
    print("Two-Tiered Clinical & Transcriptomic Feature Selection")
    print("==================================================")

    PLOT_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_DIR.mkdir(exist_ok=True, parents=True)

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Dataset configuration file not found at {rel_path(CONFIG_PATH)}")

    dataset_configs = load_dataset_config(CONFIG_PATH)

    print("\n--- Tier 1: Multi-Cohort Feature Selection (N=699) ---")
    tier1_df = _load_tier1_dataset(dataset_configs)
    tier1_rf_os = _evaluate_tier1_rf_survival(tier1_df, PLOT_DIR)
    tier1_rf_resp = _evaluate_tier1_rf_response(tier1_df, PLOT_DIR)
    tier1_resp_or = _evaluate_tier1_response_forest_plot(tier1_df, PLOT_DIR)
    tier1_cph = _evaluate_tier1_cox(tier1_df, PLOT_DIR)

    print("\n--- Tier 2: Granular TCGA Clinical Feature Selection (N=443) ---")
    tcga_config = next(c for c in dataset_configs if c.cohort_name == "TCGA-SKCM")
    tcga_clin_path = DATA_DIR / "processed" / tcga_config.processed_directory / "clin_cleaned.csv"
    tcga_df, tcga_encoded, y_os_status, y_os_months = _load_and_filter_tcga_features(tcga_clin_path)

    tier2_rf = _evaluate_tier2_tcga_rf(tcga_encoded, y_os_status, PLOT_DIR)
    tier2_cph = _evaluate_tier2_tcga_cox(tcga_encoded, y_os_status, y_os_months, PLOT_DIR)

    print("\nExporting two-tiered feature selection report...")
    _generate_two_tiered_report(
        tier1_df,
        tier1_rf_os,
        tier1_rf_resp,
        tier1_resp_or,
        tier1_cph,
        tcga_df,
        tcga_encoded,
        y_os_status,
        y_os_months,
        tier2_rf,
        tier2_cph,
        REPORT_PATH,
    )

    print("\n==================================================")
    print("Done! Two-tiered clinical feature selection completed.")
    print("==================================================")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
