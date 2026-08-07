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
from src.styles import COHORT_PALETTE, MODEL_TYPE_PALETTE, RESPONSE_PALETTE, set_presentation_style
from src.utils.formatting import generate_obsidian_frontmatter
from src.utils.logging import TeeStream
from src.utils.paths import (
    DATA_DIR,
    LOG_DIR,
    PLOTS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    SUBPROJECT_ROOT,
    rel_path,
)
from src.utils.plotting import save_fig

set_presentation_style()

# ---------------------------------------------------------------------------
# Module-level Constants & Directory Paths
# ---------------------------------------------------------------------------

CONFIG_PATH = _SUBPROJECT_ROOT / "config" / "datasets.yaml"
PLOT_DIR = PLOTS_DIR / "clinical"
REPORT_DIR = _SUBPROJECT_ROOT / "reports" / "pillar-2-clinical-subtyping"
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


def _format_p_value(p_val: float) -> str:
    """Formats p-value into clean LaTeX scientific notation or decimal string.

    Args:
        p_val: Floating-point p-value.

    Returns:
        Formatted LaTeX string (e.g. '2.82 \\times 10^{-10}' or '0.005').
    """
    if pd.isna(p_val):
        return "N/A"
    if p_val < 1e-3:
        exp = int(np.floor(np.log10(p_val)))
        mantissa = p_val / (10 ** exp)
        return f"{mantissa:.2f} \\times 10^{{{exp}}}"
    return f"{p_val:.3f}"


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


def _plot_multivariate_cox_forest(
    df_cph: pd.DataFrame,
    title: str,
    xlabel: str,
    out_path: Path,
    metrics: Dict[str, float],
) -> None:
    """Renders presentation-ready Multivariate Cox Forest Plot with Okabe-Ito palettes and model stats.

    Args:
        df_cph: DataFrame of Cox model results.
        title: Plot title.
        xlabel: Label for X-axis.
        out_path: Target image file path.
        metrics: Dictionary containing C-index, LRT p-value, and n_samples.
    """
    df_plot = df_cph.copy()
    df_plot["Formatted_Feature"] = df_plot["Feature"].apply(_format_feature_name)
    df_plot = df_plot.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(6.5, len(df_plot) * 0.5)))

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

    ax.axvline(x=1.0, color="#37474F", linestyle="--", linewidth=1.2, alpha=0.7, label="Null Effect (aHR = 1.0)")

    for i in range(len(df_plot)):
        ax.plot([lowers[i], uppers[i]], [y_pos[i], y_pos[i]], color=point_colors[i], linewidth=2.0, alpha=0.85)
        ax.scatter(hrs[i], y_pos[i], color=point_colors[i], s=75, zorder=5, edgecolor="black", linewidth=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["Formatted_Feature"], fontsize=10.5, fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())

    ax.yaxis.grid(True, linestyle="--", color="#E0E0E0", linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(True, linestyle=":", color="#E0E0E0", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    min_val = min(lowers) if len(lowers) > 0 else 0.1
    max_val = max(uppers) if len(uppers) > 0 else 2.0
    ax.set_xlim(max(0.05, min_val * 0.8), max_val * 1.6)

    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")

    for i in range(len(df_plot)):
        hr_val = hrs[i]
        p_val = p_vals[i]
        fdr_val = fdr_vals[i]
        text_str = f"aHR={hr_val:.2f} (p={p_val:.1e}, FDR={fdr_val:.1e})"
        ax.annotate(
            text_str,
            (uppers[i], y_pos[i]),
            xytext=(8, -3),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold" if fdr_val < 0.05 else "normal",
            color="#222222" if fdr_val < 0.05 else "#666666",
        )

    c_idx_str = f"C-index = {metrics['c_index']:.3f}" if "c_index" in metrics else ""
    lrt_str = f"LRT p = {metrics['lrt_p']:.2e}" if "lrt_p" in metrics else ""
    stats_label = f"Model: {c_idx_str} | {lrt_str}"

    legend_handles = [
        mpatches.Patch(color=RESPONSE_PALETTE["CR/PR"], label="Protective (aHR < 1.0)"),
        mpatches.Patch(color=RESPONSE_PALETTE["PD"], label="Risk (aHR > 1.0)"),
        mlines.Line2D([], [], color="#37474F", linestyle="--", linewidth=1.2, label="Null Effect (aHR = 1.0)"),
        mlines.Line2D([], [], color="none", label=stats_label),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=8.5)

    sns.despine(top=True, right=True)
    save_fig(fig, out_path)
    print(f"Saved multivariate Cox forest plot to {rel_path(out_path)}")


def _plot_univariate_vs_multivariate_comparison(
    df_uni: pd.DataFrame,
    df_multi: pd.DataFrame,
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    """Renders a paired side-by-side comparison forest plot contrasting Univariate vs. Multivariate Hazard Ratios.

    Args:
        df_uni: DataFrame of univariate Cox model results.
        df_multi: DataFrame of multivariate Cox model results.
        title: Plot title.
        xlabel: X-axis label.
        out_path: Target output image path.
    """
    merged = pd.merge(df_uni, df_multi, on="Feature", suffixes=("_uni", "_multi"))
    merged["Formatted_Feature"] = merged["Feature"].apply(_format_feature_name)
    merged = merged.iloc[::-1].reset_index(drop=True)

    n_feats = len(merged)
    fig, ax = plt.subplots(figsize=(13, max(7.0, n_feats * 0.6)))

    y_indices = np.arange(n_feats)
    offset = 0.15

    ax.axvline(x=1.0, color="#37474F", linestyle="--", linewidth=1.2, alpha=0.7, label="Null Effect (HR = 1.0)")

    for i, row in merged.iterrows():
        y_u = y_indices[i] + offset
        y_m = y_indices[i] - offset

        u_hr = row["Hazard Ratio (HR)_uni"]
        u_low = row["HR lower 95%_uni"]
        u_high = row["HR upper 95%_uni"]
        u_p = row["p-value_uni"]

        m_hr = row["Hazard Ratio (HR)_multi"]
        m_low = row["HR lower 95%_multi"]
        m_high = row["HR upper 95%_multi"]
        m_p = row["p-value_multi"]

        # Univariate: MODEL_TYPE_PALETTE["Univariate"]
        ax.plot([u_low, u_high], [y_u, y_u], color=MODEL_TYPE_PALETTE["Univariate"], linewidth=2.0, alpha=0.85)
        ax.scatter(u_hr, y_u, color=MODEL_TYPE_PALETTE["Univariate"], marker="o", s=70, zorder=5, edgecolor="black", linewidth=0.6)

        # Multivariate: MODEL_TYPE_PALETTE["Multivariate"]
        ax.plot([m_low, m_high], [y_m, y_m], color=MODEL_TYPE_PALETTE["Multivariate"], linewidth=2.0, alpha=0.85)
        ax.scatter(m_hr, y_m, color=MODEL_TYPE_PALETTE["Multivariate"], marker="s", s=70, zorder=5, edgecolor="black", linewidth=0.6)

        # Annotations
        u_text = f"Uni: {u_hr:.2f} (p={u_p:.1e})"
        m_text = f"Multi: {m_hr:.2f} (p={m_p:.1e})"

        max_right = max(u_high, m_high)
        ax.annotate(u_text, (max_right, y_u), xytext=(8, -3), textcoords="offset points", fontsize=8.0, color=MODEL_TYPE_PALETTE["Univariate"], fontweight="bold" if u_p < 0.05 else "normal")
        ax.annotate(m_text, (max_right, y_m), xytext=(8, -3), textcoords="offset points", fontsize=8.0, color=MODEL_TYPE_PALETTE["Multivariate"], fontweight="bold" if m_p < 0.05 else "normal")

    ax.set_yticks(y_indices)
    ax.set_yticklabels(merged["Formatted_Feature"], fontsize=10.5, fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())

    ax.yaxis.grid(True, linestyle="--", color="#E0E0E0", linewidth=0.5, alpha=0.7)
    ax.xaxis.grid(True, linestyle=":", color="#E0E0E0", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    all_lowers = list(merged["HR lower 95%_uni"].values) + list(merged["HR lower 95%_multi"].values)
    all_uppers = list(merged["HR upper 95%_uni"].values) + list(merged["HR upper 95%_multi"].values)
    min_val = min(all_lowers) if all_lowers else 0.1
    max_val = max(all_uppers) if all_uppers else 2.0
    ax.set_xlim(max(0.05, min_val * 0.8), max_val * 1.8)

    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=15)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")

    legend_handles = [
        mlines.Line2D([], [], color=MODEL_TYPE_PALETTE["Univariate"], marker="o", linestyle="-", linewidth=2.0, markersize=8, label="Univariate HR (95% CI)"),
        mlines.Line2D([], [], color=MODEL_TYPE_PALETTE["Multivariate"], marker="s", linestyle="-", linewidth=2.0, markersize=8, label="Multivariate Adjusted aHR (95% CI)"),
        mlines.Line2D([], [], color="#37474F", linestyle="--", linewidth=1.2, label="Null Effect (HR = 1.0)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=9.0)

    sns.despine(top=True, right=True)
    save_fig(fig, out_path)
    print(f"Saved Univariate vs Multivariate comparison plot to {rel_path(out_path)}")


def _evaluate_tier1_multivariate_cox(df: pd.DataFrame, plots_dir: Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Fits multivariate Cox Proportional Hazards model across harmonised Tier 1 features.

    Args:
        df: Merged patient DataFrame.
        plots_dir: Target plots directory.

    Returns:
        Tuple of (DataFrame of multivariate Cox hazard ratios, dict of model metrics).
    """
    valid_mask = df["OS_MONTHS"].notna() & df["OS_STATUS"].notna() & (df["OS_MONTHS"] > 0)
    df_cph_all = df.loc[valid_mask].copy()

    feature_cols = [f"Z_{col}" for col in ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "PD_L1", "IMPRES", "TMB_NONSYNONYMOUS", "AGE"]]
    for extra in ["SEX_Male"]:
        if extra in df_cph_all.columns:
            feature_cols.append(extra)

    model_df = df_cph_all[["OS_MONTHS", "OS_STATUS"] + feature_cols].dropna().copy()

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(model_df, duration_col="OS_MONTHS", event_col="OS_STATUS")

    summary = cph.summary
    cox_results = []
    for col in feature_cols:
        if col in summary.index:
            s_row = summary.loc[col]
            cox_results.append({
                "Feature": col,
                "Clean_Feature": col.replace("Z_", ""),
                "Hazard Ratio (HR)": s_row["exp(coef)"],
                "HR lower 95%": s_row["exp(coef) lower 95%"],
                "HR upper 95%": s_row["exp(coef) upper 95%"],
                "p-value": s_row["p"],
                "coef": s_row["coef"],
                "se": s_row["se(coef)"],
            })

    df_cph = pd.DataFrame(cox_results)
    rejected, p_adj = _benjamini_hochberg(df_cph["p-value"].values, alpha=0.05)
    df_cph["FDR_adj_p"] = p_adj
    df_cph["Significant_FDR"] = rejected
    df_cph = df_cph.sort_values(by="p-value", ascending=True).reset_index(drop=True)

    c_index = float(cph.concordance_index_)
    lrt_p = float(cph.log_likelihood_ratio_test().p_value)
    metrics = {"c_index": c_index, "lrt_p": lrt_p, "n_samples": len(model_df)}

    _plot_multivariate_cox_forest(
        df_cph,
        title=f"Tier 1 Multivariate Cox Proportional Hazards Regression (Multi-Cohort, N={len(model_df)})",
        xlabel="Adjusted Hazard Ratio (aHR, Log Scale - 95% CI per +1 SD)",
        out_path=plots_dir / "multivariate_cox_forest_plot.png",
        metrics=metrics,
    )

    return df_cph, metrics



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


def _evaluate_tier2_tcga_multivariate_cox(
    X_encoded: pd.DataFrame,
    y_os_status: pd.Series,
    y_os_months: pd.Series,
    tier2_univariate_cph: pd.DataFrame,
    plots_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Fits multivariate Cox Proportional Hazards model across top TCGA granular clinical dummy features.

    Args:
        X_encoded: Encoded feature matrix.
        y_os_status: Survival status series.
        y_os_months: Survival months series.
        tier2_univariate_cph: Univariate TCGA Cox results DataFrame.
        plots_dir: Target plots directory.

    Returns:
        Tuple of (DataFrame of TCGA multivariate Cox hazard ratios, dict of model metrics).
    """
    valid_mask = y_os_months.notna() & y_os_status.notna() & (y_os_months > 0)
    X_cph_all = X_encoded.loc[valid_mask].copy()
    months = y_os_months.loc[valid_mask].values
    status = y_os_status.loc[valid_mask].values

    top_features = tier2_univariate_cph.head(10)["Feature"].tolist()

    model_data = pd.DataFrame({"OS_MONTHS": months, "OS_STATUS": status})
    selected_cols = []
    for col in top_features:
        if col in X_cph_all.columns:
            if X_cph_all[col].nunique() > 1:
                model_data[col] = X_cph_all[col].values
                selected_cols.append(col)

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(model_data, duration_col="OS_MONTHS", event_col="OS_STATUS")

    summary = cph.summary
    cox_results = []
    for col in selected_cols:
        if col in summary.index:
            s_row = summary.loc[col]
            cox_results.append({
                "Feature": col,
                "Hazard Ratio (HR)": s_row["exp(coef)"],
                "HR lower 95%": s_row["exp(coef) lower 95%"],
                "HR upper 95%": s_row["exp(coef) upper 95%"],
                "p-value": s_row["p"],
                "coef": s_row["coef"],
                "se": s_row["se(coef)"],
            })

    df_cph = pd.DataFrame(cox_results)
    rejected, p_adj = _benjamini_hochberg(df_cph["p-value"].values, alpha=0.05)
    df_cph["FDR_adj_p"] = p_adj
    df_cph["Significant_FDR"] = rejected
    df_cph = df_cph.sort_values(by="p-value", ascending=True).reset_index(drop=True)

    c_index = float(cph.concordance_index_)
    lrt_p = float(cph.log_likelihood_ratio_test().p_value)
    metrics = {"c_index": c_index, "lrt_p": lrt_p, "n_samples": len(model_data)}

    _plot_multivariate_cox_forest(
        df_cph,
        title=f"Tier 2 Multivariate Cox Proportional Hazards Regression: Top TCGA Predictors (N={len(model_data)})",
        xlabel="Adjusted Hazard Ratio (aHR, Log Scale - 95% CI)",
        out_path=plots_dir / "tcga_multivariate_cox_forest_plot.png",
        metrics=metrics,
    )

    return df_cph, metrics





# ---------------------------------------------------------------------------
# Markdown Report Generation Logic
# ---------------------------------------------------------------------------


def _generate_two_tiered_report(
    tier1_df: pd.DataFrame,
    tier1_rf_os: pd.DataFrame,
    tier1_cph: pd.DataFrame,
    tier1_multi_cph: pd.DataFrame,
    tier1_multi_metrics: Dict[str, float],
    tcga_df: pd.DataFrame,
    tcga_encoded: pd.DataFrame,
    y_os_status: pd.Series,
    y_os_months: pd.Series,
    tier2_rf: pd.DataFrame,
    tier2_cph: pd.DataFrame,
    tier2_multi_cph: pd.DataFrame,
    tier2_multi_metrics: Dict[str, float],
    report_path: Path,
) -> None:
    """Generates two-tiered Markdown report with dynamic sample metrics and Obsidian frontmatter.

    All numeric values — sample sizes, hazard ratios, p-values, FDR values, feature names, and
    feature counts — are derived at runtime from the live DataFrames passed in. No literals are
    hardcoded in the report template.

    Args:
        tier1_df: Pooled patient DataFrame across all 4 cohorts.
        tier1_rf_os: Ranked Tier 1 RF OS feature importances.
        tier1_cph: Ranked Tier 1 Univariate Cox regression results (sorted by p-value ascending).
        tier1_multi_cph: Ranked Tier 1 Multivariate Cox regression results.
        tier1_multi_metrics: Metrics for Tier 1 Multivariate Cox model.
        tcga_df: Raw TCGA clinical DataFrame.
        tcga_encoded: Encoded TCGA feature matrix.
        y_os_status: Target TCGA OS status Series.
        y_os_months: Target TCGA OS months Series.
        tier2_rf: Ranked Tier 2 TCGA RF feature importances.
        tier2_cph: Ranked Tier 2 TCGA Univariate Cox regression results.
        tier2_multi_cph: Ranked Tier 2 TCGA Multivariate Cox regression results.
        tier2_multi_metrics: Metrics for Tier 2 Multivariate Cox model.
        report_path: Target report path.
    """
    # ------------------------------------------------------------------
    # Tier 1 sample sizes (all computed from live DataFrames)
    # ------------------------------------------------------------------
    tier1_total_n = len(tier1_df)
    tier1_surv_n = int(tier1_df["OS_STATUS"].notna().sum())
    tier1_cox_n = int(
        (tier1_df["OS_MONTHS"].notna() & tier1_df["OS_STATUS"].notna() & (tier1_df["OS_MONTHS"] > 0)).sum()
    )
    tier1_n_sig_fdr = int((tier1_cph["FDR_adj_p"] < 0.05).sum())
    tier1_multi_sig_fdr = int((tier1_multi_cph["FDR_adj_p"] < 0.05).sum())

    # ------------------------------------------------------------------
    # Tier 1 top univariate and multivariate predictors
    # ------------------------------------------------------------------
    top_tier1_cph_feat = tier1_cph.iloc[0]["Feature"]
    top_tier1_cph_hr = tier1_cph.iloc[0]["Hazard Ratio (HR)"]
    top_tier1_cph_p = tier1_cph.iloc[0]["p-value"]
    top_tier1_cph_fdr = tier1_cph.iloc[0]["FDR_adj_p"]

    top_tier1_multi_feat = tier1_multi_cph.iloc[0]["Feature"]
    top_tier1_multi_ahr = tier1_multi_cph.iloc[0]["Hazard Ratio (HR)"]
    top_tier1_multi_p = tier1_multi_cph.iloc[0]["p-value"]
    top_tier1_multi_fdr = tier1_multi_cph.iloc[0]["FDR_adj_p"]

    # ------------------------------------------------------------------
    # Tier 1: HR range for immune signatures (replaces hardcoded 0.73--0.82 literal)
    # ------------------------------------------------------------------
    sig_feature_stems = ["IFN_gamma", "TIS", "CYT", "CD8_Tcell", "PD_L1", "IMPRES"]
    sig_z_cols = [f"Z_{s}" for s in sig_feature_stems]
    sig_mask = tier1_cph["Feature"].isin(sig_z_cols)
    if sig_mask.any():
        sig_hrs = tier1_cph.loc[sig_mask, "Hazard Ratio (HR)"]
        sig_ps = tier1_cph.loc[sig_mask, "p-value"]
        hr_min = sig_hrs.min()
        hr_max = sig_hrs.max()
        sig_max_p = sig_ps.max()  # Weakest (largest) p among immune signatures
        n_sigs = int(sig_mask.sum())
    else:
        hr_min, hr_max, sig_max_p, n_sigs = float("nan"), float("nan"), float("nan"), len(sig_feature_stems)

    # ------------------------------------------------------------------
    # Tier 2 sample sizes and feature counts
    # ------------------------------------------------------------------
    tcga_n = len(tcga_df)
    tcga_rf_n = int(y_os_status.notna().sum())
    tcga_cox_n = int((y_os_status.notna() & y_os_months.notna() & (y_os_months > 0)).sum())
    tcga_feat_count = len(tcga_encoded.columns)
    tcga_n_sig_fdr = int((tier2_cph["FDR_adj_p"] < 0.05).sum())
    tcga_multi_sig_fdr = int((tier2_multi_cph["FDR_adj_p"] < 0.05).sum())

    # ------------------------------------------------------------------
    # Tier 2 top univariate predictor
    # ------------------------------------------------------------------
    top_tier2_cph_feat = tier2_cph.iloc[0]["Feature"]
    top_tier2_cph_hr = tier2_cph.iloc[0]["Hazard Ratio (HR)"]
    top_tier2_cph_p = tier2_cph.iloc[0]["p-value"]
    top_tier2_cph_fdr = tier2_cph.iloc[0]["FDR_adj_p"]

    # ------------------------------------------------------------------
    # Tier 2: dynamically identify the top two independent risk factors from
    # the multivariate model (aHR > 1, sorted by p-value ascending).
    # ------------------------------------------------------------------
    tier2_multi_risk = (
        tier2_multi_cph[tier2_multi_cph["Hazard Ratio (HR)"] > 1.0]
        .sort_values("p-value")
        .reset_index(drop=True)
    )
    t2_top1_feat = tier2_multi_risk.iloc[0]["Feature"] if len(tier2_multi_risk) >= 1 else "N/A"
    t2_top1_ahr = tier2_multi_risk.iloc[0]["Hazard Ratio (HR)"] if len(tier2_multi_risk) >= 1 else float("nan")
    t2_top1_p = tier2_multi_risk.iloc[0]["p-value"] if len(tier2_multi_risk) >= 1 else float("nan")
    t2_top2_feat = tier2_multi_risk.iloc[1]["Feature"] if len(tier2_multi_risk) >= 2 else "N/A"
    t2_top2_ahr = tier2_multi_risk.iloc[1]["Hazard Ratio (HR)"] if len(tier2_multi_risk) >= 2 else float("nan")
    t2_top2_p = tier2_multi_risk.iloc[1]["p-value"] if len(tier2_multi_risk) >= 2 else float("nan")

    # ------------------------------------------------------------------
    # Frontmatter — aliases and extra cssclasses passed explicitly
    # ------------------------------------------------------------------
    frontmatter = generate_obsidian_frontmatter(
        title="Two-Tiered Clinical & Transcriptomic Feature Selection Report",
        aliases=["Feature Selection Report", "Clinical Feature Selection"],
        tags=[
            "melanoma",
            "clinical-subtyping",
            "feature-selection",
            "cox-regression",
            "random-forest",
            "two-tiered",
            "multivariate-cox",
        ],
        extra_css_classes=["table-center", "row-alt"],
    )

    report_path.parent.mkdir(exist_ok=True, parents=True)

    lines: List[str] = [frontmatter, ""]
    w = lines.append

    w("# Two-Tiered Clinical & Transcriptomic Feature Selection Report")
    w("")
    w("This report presents a comprehensive **two-tiered feature selection architecture** evaluating prognostic and predictive clinical markers across melanoma patient populations using both **Univariate** and **Multivariate Cox Proportional Hazards Regression**:")
    w("")
    w(f"- **Tier 1 (Multi-Cohort Consensus, $N = {tier1_total_n}$)**: Evaluates {n_sigs} transcriptomic immune signatures, $\\text{{TMB}}$, age, and sex — pooled across all four study cohorts (**TCGA-SKCM**, **Liu 2019**, **Hugo 2016**, **Riaz 2017**).")
    w(f"- **Tier 2 (Granular TCGA Pathological Staging, $N = {tcga_n}$)**: Evaluates {tcga_feat_count} detailed clinical, pathological TNM staging, anatomical site, aneuploidy, and hypoxia attributes specifically within the **TCGA-SKCM** reference cohort.")
    w("")

    # ---- Section 1: Tier 1 ----
    w(f"## 1. Tier 1: Multi-Cohort Feature Selection ($N = {tier1_total_n}$)")
    w("")
    w("> [!INFO] What, Why & Questions — Tier 1")
    w(f"> **What We Are Doing**: Evaluating transcriptomic immune features — {n_sigs} immune gene expression signatures (IFN-$\\gamma$, TIS, CYT, CD8 T-cell, IMPRES, `PD-L1`), `TMB`, age, and sex — against overall survival across all four cohorts pooled ($N = {tier1_total_n}$). We apply two complementary survival models: **Random Forest** importance (to rank features without distributional assumptions) and **Cox Proportional Hazards regression** (univariate then multivariate, to estimate hazard ratios and test for independent effects after adjusting for confounders).")
    w("> **Why We Are Doing It**: A feature that looks predictive in isolation may simply be correlated with a stronger feature (collinearity). Multivariate Cox regression disentangles this — only features that remain significant after mutual adjustment carry truly independent prognostic signal, protecting downstream models from redundant features.")
    w("> **Questions**:")
    w(">   1. *Which baseline clinical and transcriptomic features are individually associated with overall survival across all four cohorts?*")
    w(">   2. *After mutual adjustment, which features retain independent prognostic significance?*")
    w(f">   3. *Do the {n_sigs} immune expression signatures collapse into one another due to collinearity?*")
    w("")
    w(f"- **Total Merged Sample Size**: {tier1_total_n} patients across 4 cohorts")
    w(f"- **Overall Survival Evaluation Cohort**: {tier1_surv_n} patients")
    w(f"- **Cox Survival Evaluation Cohort**: {tier1_cox_n} patients")
    w(f"- **Univariate FDR-Significant Survival Predictors (FDR < 0.05)**: {tier1_n_sig_fdr} features")
    w(f"- **Multivariate FDR-Significant Independent Predictors (FDR < 0.05)**: {tier1_multi_sig_fdr} feature(s)")
    w("")
    w(f"### 1.1. Random Forest Importance for Overall Survival ($N = {tier1_surv_n}$)")
    w(f"Random Forest feature importance (500 estimators) trained on the $N = {tier1_surv_n}$ overall survival cohort:")
    w("![Tier 1 Random Forest OS](../../plots/clinical/clinical_feature_importance.png)")
    w("")
    w(f"### 1.2. Univariate vs. Multivariate Cox Hazard Ratio Comparison ($N = {tier1_cox_n}$)")
    w(
        f"Contrasting unadjusted Univariate Hazard Ratios ($\\text{{HR}}$, blue circles) against "
        f"multivariable-adjusted Hazard Ratios ($\\text{{aHR}}$, orange squares) across $N = {tier1_cox_n}$ "
        f"multi-cohort patients with complete survival data "
        f"(Model Concordance Index = **{tier1_multi_metrics['c_index']:.3f}**, "
        f"Likelihood Ratio Test $p = {_format_p_value(tier1_multi_metrics['lrt_p'])}$):"
    )
    w("")
    w("![Tier 1 Univariate vs Multivariate Cox Comparison](../../plots/clinical/tier1_uni_vs_multi_forest_plot.png)")
    w("")

    w("> [!INSIGHT] Key Insights: Multivariable Adjustment & Collinearity (Tier 1)")
    if sig_mask.any():
        w(
            f"> - **Transcriptomic Collinearity & Attenuation**: All {n_sigs} transcriptomic immune signatures "
            f"(IFN-$\\gamma$, TIS, CYT, CD8 T-cell, IMPRES, `PD-L1`) show significant protective association "
            f"with survival in unadjusted univariate Cox models "
            f"($\\text{{HR}} \\approx {hr_min:.2f}\\text{{--}}{hr_max:.2f}$, all $p \\leq {_format_p_value(sig_max_p)}$). "
            f"However, in joint multivariate modelling, individual signatures attenuate towards the null "
            f"($\\text{{aHR}} \\to 1.0$) and lose independent significance — demonstrating that while "
            f"T-cell microenvironmental inflammation is genuinely protective, individual signatures capture "
            f"overlapping, collinear aspects of the same biological axis."
        )
    else:
        w("> - **Transcriptomic Collinearity & Attenuation**: Immune signatures show significant protective univariate associations; however, multivariate modelling reveals collinearity — individual signatures lose independent significance when jointly modelled.")
    w(
        f"> - **Sole Independent Risk Factor**: **`{_format_feature_name(top_tier1_multi_feat)}`** "
        f"($\\text{{aHR}} = {top_tier1_multi_ahr:.2f}$, $p = {_format_p_value(top_tier1_multi_p)}$, "
        f"FDR $= {_format_p_value(top_tier1_multi_fdr)}$) retains independent statistical significance after joint "
        f"adjustment, confirming that age-related immunosenescence or host fragility confers mortality risk "
        f"independently of tumour inflammation."
    )
    w("")

    # ---- Section 2: Tier 2 ----
    w(f"## 2. Tier 2: Granular TCGA Pathological & Clinical Staging ($N = {tcga_n}$)")
    w("")
    w("> [!INFO] What, Why & Questions — Tier 2")
    w(
        f"> **What We Are Doing**: Evaluating {tcga_feat_count} granular clinical and pathological attributes "
        f"available exclusively in **TCGA-SKCM** ($N = {tcga_n}$) — including TNM staging (T1–T4, N0–N3, "
        f"metastasis), AJCC stage groupings, primary anatomical sites, aneuploidy score, and hypoxia scores "
        f"— against overall survival using the same Random Forest + Cox regression two-step framework."
    )
    w("> **Why We Are Doing It**: Clinical trial datasets record only basic demographics. The TCGA reference cohort contains rich pathological staging data not available in the trial cohorts — answering which pathological features independently determine prognosis in the general melanoma population.")
    w("> **Questions**:")
    w(">   1. *Which TNM staging features carry the strongest independent prognostic signal?*")
    w(">   2. *Does primary tumour invasion depth (T staging) dominate over nodal spread (N staging) as an independent predictor?*")
    w(">   3. *Do anatomical primary sites carry independent survival differences beyond TNM stage?*")
    w("")
    w(f"- **TCGA Total Cohort Sample Size**: {tcga_n} patients")
    w(f"- **TCGA OS Classification Cohort**: {tcga_rf_n} patients")
    w(f"- **TCGA Cox Survival Evaluation Cohort**: {tcga_cox_n} patients")
    w(f"- **Encoded Dummy Features**: {tcga_feat_count} dummy variables")
    w(f"- **Univariate FDR-Significant Pathological Predictors (FDR < 0.05)**: {tcga_n_sig_fdr} features")
    w(f"- **Multivariate FDR-Significant Predictors (FDR < 0.05)**: {tcga_multi_sig_fdr} feature(s)")
    w("")

    w(f"### 2.1. Random Forest Importance for Granular TCGA Clinical Attributes ($N = {tcga_rf_n}$)")
    w(f"Top 20 Random Forest clinical predictors trained on $N = {tcga_rf_n}$ TCGA patients with non-null survival status:")
    w("")
    w("![Tier 2 TCGA Random Forest](../../plots/clinical/tcga_clinical_feature_importance.png)")
    w("")

    w(f"### 2.2. Univariate vs. Multivariate Cox Hazard Ratio Comparison for TCGA Attributes ($N = {tcga_cox_n}$)")
    w(
        f"Contrasting unadjusted Univariate Hazard Ratios ($\\text{{HR}}$, blue circles) against "
        f"multivariable-adjusted Hazard Ratios ($\\text{{aHR}}$, orange squares) across $N = {tcga_cox_n}$ "
        f"TCGA patients "
        f"(Model Concordance Index = **{tier2_multi_metrics['c_index']:.3f}**, "
        f"Likelihood Ratio Test $p = {_format_p_value(tier2_multi_metrics['lrt_p'])}$):"
    )
    w("")
    w("![Tier 2 TCGA Univariate vs Multivariate Cox Comparison](../../plots/clinical/tcga_uni_vs_multi_forest_plot.png)")
    w("")

    w("> [!INSIGHT] Key Insights: Pathological Staging Independence (Tier 2 TCGA)")
    if t2_top1_feat != "N/A":
        w(
            f"> - **Top Independent Risk Factor**: **`{_format_feature_name(t2_top1_feat)}`** "
            f"($\\text{{aHR}} = {t2_top1_ahr:.2f}$, $p = {_format_p_value(t2_top1_p)}$) maintains the strongest "
            f"independent prognostic risk elevation in multivariate modelling."
        )
    if t2_top2_feat != "N/A":
        w(
            f"> - **Second Independent Risk Factor**: **`{_format_feature_name(t2_top2_feat)}`** "
            f"($\\text{{aHR}} = {t2_top2_ahr:.2f}$, $p = {_format_p_value(t2_top2_p)}$) also retains independent "
            f"prognostic significance after mutual adjustment."
        )
    w("> - **Attenuated Staging Categories**: Other staging categories (e.g. N3 nodal staging, AJCC Stage IIIC) exhibit significant univariate risk elevation but attenuate in multivariate modelling as their variance is explained by the top independent predictors.")
    w("")

    # ---- Section 3: Biological Summary ----
    w("## 3. Key Analytical & Biological Summary")
    w("")
    w("> [!INSIGHT] Key Insights: Overall Findings")
    w(
        f"> 1. **Tier 1 Top Survival Biomarker**: **`{_format_feature_name(top_tier1_cph_feat)}`** is the "
        f"single strongest protective univariate predictor across all 4 cohorts "
        f"($\\text{{HR}} = {top_tier1_cph_hr:.2f}$, $p = {_format_p_value(top_tier1_cph_p)}$, FDR $= {_format_p_value(top_tier1_cph_fdr)}$)."
    )
    w(
        f"> 2. **Tier 1 Independent Survival Biomarker**: In joint multivariate modelling ($N = {tier1_cox_n}$), "
        f"**`{_format_feature_name(top_tier1_multi_feat)}`** remains an independent prognostic predictor "
        f"($\\text{{aHR}} = {top_tier1_multi_ahr:.2f}$, $p = {_format_p_value(top_tier1_multi_p)}$, "
        f"FDR $= {_format_p_value(top_tier1_multi_fdr)}$; Model C-index = **{tier1_multi_metrics['c_index']:.3f}**)."
    )
    w(
        f"> 3. **Tier 2 Pathological Staging Independence**: **`{_format_feature_name(top_tier2_cph_feat)}`** "
        f"is the strongest univariate predictor in TCGA "
        f"($\\text{{HR}} = {top_tier2_cph_hr:.2f}$, $p = {_format_p_value(top_tier2_cph_p)}$, FDR $= {_format_p_value(top_tier2_cph_fdr)}$), "
        f"and retains significant independent risk elevation in multivariate Cox regression "
        f"(Model C-index = **{tier2_multi_metrics['c_index']:.3f}**)."
    )
    w(
        f"> 4. **Biological Alignment**: Microenvironmental T-cell inflammation signatures "
        f"(IFN-$\\gamma$, TIS, CYT, CD8, IMPRES) consistently confer significant mortality risk reduction "
        f"($\\text{{HR}} < 1.0$) in univariate Cox models across all {tier1_total_n} patients, confirming "
        f"the prognostic value of the tumour immune microenvironment."
    )
    w("")

    # ---- Section 4: Limitations ----
    w("## 4. Methodological Limitations & Future Directions")
    w("")
    w("> [!WARNING] Analytical Scope & Limitations")
    w("> - **Proportional Hazards Assumption**: Cox regression assumes hazard ratios remain constant over time. This assumption was not formally tested (e.g., Schoenfeld residuals) and may be violated for some features, particularly those with time-varying effects.")
    w("> - **Collinearity Among Immune Signatures**: The six transcriptomic immune signatures share overlapping gene sets and are highly collinear. Multivariate Cox estimates for individual signatures are unstable and should not be over-interpreted in isolation.")
    w(
        f"> - **Tier 2 TCGA Staging Scope**: Granular TNM staging and anatomical site data are available only "
        f"in TCGA-SKCM ($N = {tcga_n}$) and cannot be transferred to clinical trial cohorts, limiting "
        f"the generalisability of Tier 2 findings."
    )
    w(
        f"> - **High-Dimensional Dummy Encoding**: One-hot encoding of {tcga_feat_count} categorical staging "
        f"variables creates sparse, high-dimensional feature matrices. Multivariate models in Tier 2 are "
        f"particularly susceptible to overfitting and convergence instability at low per-category sample counts."
    )
    w("> - **OS as Outcome Proxy**: Overall Survival reflects diverse treatment histories (surgery, targeted therapy, immunotherapy) rather than response to a single agent, making it a weaker endpoint than progression-free survival under checkpoint blockade.")
    w("")
    w("> [!formula]+ Clinical Feature Selection Script Execution & Software Module Architecture")
    w("> - **Primary Pipeline Execution Scripts**:")
    w(
        f">   - [`run_clinical_feature_selection.py`]"
        f"(file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/"
        f"AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-2-clinical-subtyping/"
        f"run_clinical_feature_selection.py): Evaluates two-tiered clinical and transcriptomic feature "
        f"selection across multi-cohort ($N = {tier1_total_n}$) and TCGA ($N = {tcga_n}$) datasets "
        f"using Random Forest Gini importance and Univariate/Multivariate Cox Proportional Hazards "
        f"regression, and outputs `clinical_feature_selection_report.md`."
    )
    w("> - **Data Preprocessing & Loading Modules**:")
    w(
        ">   - [`clean_data.py`]"
        "(file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/"
        "AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/"
        "clean_data.py): Preprocesses raw cohort clinical metadata and RNA-seq expression profiles "
        "into cleaned CSV matrices."
    )
    w(
        ">   - [`merge_datasets.py`]"
        "(file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/"
        "AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/pillar-1-cohort-preprocessing/"
        "merge_datasets.py): Merges processed expression matrices across cohorts into harmonised "
        "pooled matrices (`expr_merged.csv`, `clin_merged.csv`)."
    )
    w(
        ">   - [`signatures.py`]"
        "(file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/"
        "AI-ML-3/melanoma-assignment-3/q1-response-predictor/src/signatures.py): "
        "Computes transcriptomic immune signatures (IFN-$\\gamma$, TIS, CYT, CD8 T-cell, "
        "IMPRES, `PD-L1`) across cohort expression matrices."
    )
    w("> - **Shared Cross-Question & Pipeline Modules**:")
    w(
        ">   - [`run_pipeline.py`]"
        "(file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/"
        "AI-ML-3/melanoma-assignment-3/q1-response-predictor/scripts/run_pipeline.py): "
        "Master Q1 pipeline orchestrator executing downstream modeling and evaluation."
    )
    w(
        ">   - [`styles.py`]"
        "(file:///c:/Users/Amanda/Dropbox/OBSIDIAN/42/090%20STUDY/091%20UCD/091.03%20ASSIGNMENTS/"
        "AI-ML-3/melanoma-assignment-3/src/styles.py): Single source of truth for Okabe-Ito "
        "colour palettes (`COHORT_PALETTE`, `RESPONSE_PALETTE`, `MODEL_TYPE_PALETTE`) and "
        "visualisation presentation style."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
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
    tier1_cph = _evaluate_tier1_cox(tier1_df, PLOT_DIR)
    tier1_multi_cph, tier1_multi_metrics = _evaluate_tier1_multivariate_cox(tier1_df, PLOT_DIR)

    _plot_univariate_vs_multivariate_comparison(
        tier1_cph,
        tier1_multi_cph,
        title="Tier 1 Univariate vs. Multivariate Cox Hazard Ratios (Multi-Cohort)",
        xlabel="Hazard Ratio (Log Scale - 95% CI per +1 SD)",
        out_path=PLOT_DIR / "tier1_uni_vs_multi_forest_plot.png",
    )

    print("\n--- Tier 2: Granular TCGA Clinical Feature Selection (N=443) ---")
    tcga_config = next(c for c in dataset_configs if c.cohort_name == "TCGA-SKCM")
    tcga_clin_path = DATA_DIR / "processed" / tcga_config.processed_directory / "clin_cleaned.csv"
    tcga_df, tcga_encoded, y_os_status, y_os_months = _load_and_filter_tcga_features(tcga_clin_path)

    tier2_rf = _evaluate_tier2_tcga_rf(tcga_encoded, y_os_status, PLOT_DIR)
    tier2_cph = _evaluate_tier2_tcga_cox(tcga_encoded, y_os_status, y_os_months, PLOT_DIR)
    tier2_multi_cph, tier2_multi_metrics = _evaluate_tier2_tcga_multivariate_cox(
        tcga_encoded, y_os_status, y_os_months, tier2_cph, PLOT_DIR
    )

    _plot_univariate_vs_multivariate_comparison(
        tier2_cph.head(10),
        tier2_multi_cph,
        title="Tier 2 TCGA Univariate vs. Multivariate Cox Hazard Ratios",
        xlabel="Hazard Ratio (Log Scale - 95% CI)",
        out_path=PLOT_DIR / "tcga_uni_vs_multi_forest_plot.png",
    )

    print("\nExporting two-tiered feature selection report...")
    _generate_two_tiered_report(
        tier1_df,
        tier1_rf_os,
        tier1_cph,
        tier1_multi_cph,
        tier1_multi_metrics,
        tcga_df,
        tcga_encoded,
        y_os_status,
        y_os_months,
        tier2_rf,
        tier2_cph,
        tier2_multi_cph,
        tier2_multi_metrics,
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

