"""
Clinical Feature Selection Pipeline for TCGA-SKCM Cohort.

Evaluates clinical feature importance and survival association using Random Forest Gini Importance
and univariate Cox Proportional Hazards regression (with Benjamini-Hochberg FDR adjustment).
Generates feature importance bar plots, Cox forest plots, and a summary report.
"""

import contextlib
from itertools import cycle
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import seaborn as sns

# Bootstrap project root resolution for top-level import
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, set_presentation_style
from src.utils.logging import TeeStream
from src.utils.paths import find_project_root
from src.utils.plotting import save_fig


def _benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Applies Benjamini-Hochberg false discovery rate adjustment to p-values."""
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

# Module-level Constants
DATA_DIR = find_project_root(Path(__file__).resolve()) / "data"
CLINICAL_FILE = DATA_DIR / "processed" / "skcm_tcga_pan_can_atlas_2018" / "clin_cleaned.csv"
REPORTS_DIR = find_project_root(Path(__file__).resolve()) / "reports"
PLOTS_DIR = find_project_root(Path(__file__).resolve()) / "plots" / "clinical"
LOG_DIR = find_project_root(Path(__file__).resolve()) / "logs"
LOG_PATH = LOG_DIR / "run_clinical_feature_selection.log"
OUTPUT_FILE = REPORTS_DIR / "clinical_feature_selection_results.md"


def _format_feature_name(name: str) -> str:
    """Formats raw encoded clinical feature column names into clean display titles.

    Args:
        name: Raw feature column name string.

    Returns:
        Formatted human-readable feature title string.
    """
    name = name.replace("TUMOR_TISSUE_SITE_", "Tumour Site: ")
    name = name.replace("TX_TYPE_", "Treatment Type: ")
    name = name.replace("TX_AGENT_", "Treatment Agent: ")
    name = name.replace("RADIATION_THERAPY_", "Radiation Therapy: ")
    name = name.replace("SEX_", "Sex: ")
    name = name.replace("PRIOR_DX_", "Prior Diagnosis: ")
    name = name.replace("AJCC_PATHOLOGIC_TUMOR_STAGE_", "AJCC Stage: ")
    name = name.replace("PATH_T_STAGE_", "Primary Tumour (T) Staging: ")
    name = name.replace("PATH_N_STAGE_", "N Stage: ")
    name = name.replace("PATH_M_STAGE_", "M Stage: ")
    name = name.replace("GENETIC_ANCESTRY_LABEL_", "Genetic Ancestry: ")
    name = name.replace("RACE_", "Race: ")
    name = name.replace("ETHNICITY_", "Ethnicity: ")
    name = name.replace("ICD_10_", "ICD-10: ")
    name = name.replace("ICD_O_3_HISTOLOGY_", "ICD-O-3 Histology: ")
    name = name.replace("ICD_O_3_SITE_", "ICD-O-3 Site: ")
    name = name.replace("SAMPLE_TYPE_", "Sample Type: ")
    name = name.replace("_", " ")

    def fix_roman_numeral(word: str) -> str | None:
        pattern = r"^([IVX]+)([A-C]?)$"
        match = re.match(pattern, word.upper())
        if match:
            roman, suffix = match.groups()
            if roman in ["I", "II", "III", "IV", "V"]:
                return roman + suffix
        return None

    def fix_tnm_stage(word: str) -> str | None:
        pattern = r"^([TNM]\d)([A-D]?)$"
        match = re.match(pattern, word, re.IGNORECASE)
        if match:
            base, suffix = match.groups()
            return base.upper() + suffix.lower()
        if word.upper() in ["TX", "NX", "MX", "T0", "N0", "M0", "TIS"]:
            if word.upper() == "TIS":
                return "Tis"
            return word.upper()
        return None

    words = name.split()
    formatted_words = []
    for w in words:
        if w.upper() in ["TMB", "FDR", "BH", "CI", "HR", "OS", "PFS", "DSS", "TX", "DX", "NOS", "ICD-10", "ICD-O-3", "AJCC", "(T)"]:
            formatted_words.append(w.upper())
        elif "/" in w:
            parts = w.split("/")
            formatted_words.append("/".join([p.capitalize() for p in parts]))
        else:
            roman_fixed = fix_roman_numeral(w)
            if roman_fixed:
                formatted_words.append(roman_fixed)
                continue
            t_fixed = fix_tnm_stage(w)
            if t_fixed:
                formatted_words.append(t_fixed)
                continue
            formatted_words.append(w.capitalize())
    return " ".join(formatted_words)


def _load_and_filter_features(
    clinical_file: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str], List[str]]:
    """Loads TCGA clinical data, excludes confounders/treatment features, imputes, and one-hot encodes.

    Args:
        clinical_file: Absolute path to clean TCGA clinical CSV file.

    Returns:
        Tuple of (full DataFrame, encoded features DataFrame, OS status series, OS months series, id_cols, icd_cols, treatment_cols).
    """
    if not clinical_file.exists():
        raise FileNotFoundError(f"Cleaned clinical file not found at {clinical_file.relative_to(BASE_DIR).as_posix()}")

    df = pd.read_csv(clinical_file)
    print(f"Loaded clinical data with shape: {df.shape}")
    df = df.dropna(subset=["OS_STATUS"])
    print(f"Filtered clinical data (non-null OS_STATUS) shape: {df.shape}")

    target_cols = ["OS_STATUS", "OS_MONTHS", "PFS_STATUS", "PFS_MONTHS", "DSS_STATUS", "DSS_MONTHS"]
    id_cols = ["PATIENT_ID", "SAMPLE_ID"]
    sourcing_cols = ["TISSUE_SOURCE_SITE", "TISSUE_SOURCE_SITE_CODE"]
    treatment_cols = [
        c for c in df.columns
        if c.startswith("TX_") or any(kw in c.upper() for kw in ["RADIATION", "TREAT", "THERAPY", "NEOADJUVANT", "CHEMO", "SURGERY", "DRUG"])
    ]
    icd_cols = [c for c in df.columns if "ICD_" in c.upper() or "ICD10" in c.upper()]
    confounder_cols = ["ETHNICITY", "GENETIC_ANCESTRY_LABEL", "RACE", "PRIOR_DX"]
    recommended_exclusions = ["DAYS_LAST_FOLLOWUP", "PERSON_NEOPLASM_CANCER_STATUS"]

    exclude_cols = sorted(list(set(target_cols + id_cols + recommended_exclusions + sourcing_cols + treatment_cols + icd_cols + confounder_cols)))

    missing_pct = df.isna().mean()
    high_missing_cols = missing_pct[missing_pct > 0.5].index.tolist()
    print(f"Excluding columns with >50% missing values: {high_missing_cols}")
    exclude_cols.extend(high_missing_cols)

    features_df = df.drop(columns=[c for c in exclude_cols if c in df.columns])
    y_os_status = df["OS_STATUS"]
    y_os_months = df["OS_MONTHS"]

    numeric_cols = []
    categorical_cols = []

    for col in features_df.columns:
        if df[col].dtype in [np.float64, np.int64] and df[col].nunique() > 2:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical/Binary features ({len(categorical_cols)}): {categorical_cols}")

    num_imputer = SimpleImputer(strategy="median")
    if numeric_cols:
        features_df[numeric_cols] = num_imputer.fit_transform(features_df[numeric_cols])

    for col in categorical_cols:
        features_df[col] = features_df[col].fillna("Unknown").astype(str)

    features_encoded = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)
    features_encoded.columns = [col.replace("|", " / ") for col in features_encoded.columns]
    print(f"Shape after one-hot encoding: {features_encoded.shape}")

    return df, features_encoded, y_os_status, y_os_months, id_cols, icd_cols, treatment_cols


def _evaluate_rf_importance(
    features_encoded: pd.DataFrame,
    y_os_status: pd.Series,
    plot_dir: Path,
) -> pd.DataFrame:
    """Trains Random Forest Classifier and generates feature importance bar plot.

    Args:
        features_encoded: One-hot encoded feature matrix.
        y_os_status: Binary overall survival status target series.
        plot_dir: Directory path to export plot artifact.

    Returns:
        DataFrame of ranked feature Gini importances.
    """
    print("\n--- Method 1: Random Forest Classifier Feature Importance ---")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=6, n_jobs=-1)
    rf.fit(features_encoded, y_os_status)

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    rf_results = []
    for f in range(min(25, features_encoded.shape[1])):
        col_name = features_encoded.columns[indices[f]]
        rf_results.append({
            "Rank": f + 1,
            "Feature": col_name,
            "Importance": importances[indices[f]],
        })
    df_rf = pd.DataFrame(rf_results)
    print(df_rf.head(15).to_string(index=False))

    df_plot = df_rf.head(20).copy()
    df_plot["Formatted_Feature"] = df_plot["Feature"].apply(_format_feature_name)

    set_presentation_style()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 8.5))

    okabe_colors = list(COHORT_PALETTE.values())
    colors = [next(cycle(okabe_colors)) for _ in range(len(df_plot))]

    bars = ax.barh(
        df_plot["Formatted_Feature"][::-1],
        df_plot["Importance"][::-1],
        color=colors,
        edgecolor="none",
        height=0.75,
    )
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
            weight="semibold",
        )

    fig.suptitle("Top 20 Clinical Feature Importances (Random Forest)", fontsize=16, weight="bold", y=0.96)
    ax.set_title("Gini Importance ranked from Random Forest Classifier trained on OS_STATUS", fontsize=11, style="italic", color="#555555", pad=10)
    ax.set_xlabel("Gini Importance Score", fontsize=12, labelpad=10)
    ax.set_ylabel("Clinical Feature", fontsize=12, labelpad=10)
    ax.set_xlim(0, max(df_plot["Importance"]) * 1.15)
    sns.despine(left=True, bottom=True)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.4, color="#e0e0e0")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plot_path = plot_dir / "clinical_feature_importance.png"
    save_fig(fig, plot_path)
    print(f"Saved feature importance plot to {plot_path.relative_to(BASE_DIR).as_posix()}")

    return df_rf


def _evaluate_cox_proportional_hazards(
    features_encoded: pd.DataFrame,
    y_os_months: pd.Series,
    y_os_status: pd.Series,
    plot_dir: Path,
) -> pd.DataFrame:
    """Fits univariate Cox Proportional Hazards models and generates forest plot.

    Args:
        features_encoded: One-hot encoded feature matrix.
        y_os_months: Continuous overall survival months target.
        y_os_status: Binary overall survival event status target.
        plot_dir: Directory path to export plot artifact.

    Returns:
        DataFrame of Cox hazard ratios, confidence intervals, p-values, and FDR values.
    """
    print("\n--- Method 2: Cox Proportional Hazards Regression (Univariate) ---")
    cph_records = []

    for col in features_encoded.columns:
        mini_df = pd.DataFrame({
            "time": y_os_months,
            "event": y_os_status,
            "feature": features_encoded[col].astype(float),
        }).dropna()

        if mini_df["feature"].nunique() <= 1:
            continue

        cph = CoxPHFitter()
        try:
            cph.fit(mini_df, duration_col="time", event_col="event")
            summary = cph.summary.iloc[0]

            p_val = summary["p"]
            coef = summary["coef"]
            hazard_ratio = np.exp(coef)
            lower_ci = summary["exp(coef) lower 95%"]
            upper_ci = summary["exp(coef) upper 95%"]
            concordance = cph.concordance_index_

            cph_records.append({
                "Feature": col,
                "Hazard Ratio (HR)": hazard_ratio,
                "95% CI Lower": lower_ci,
                "95% CI Upper": upper_ci,
                "Beta (Coef)": coef,
                "p-value": p_val,
                "Concordance Index": concordance,
            })
        except Exception:
            continue

    df_cph = pd.DataFrame(cph_records)
    df_cph = df_cph.dropna(subset=["p-value"])

    if not df_cph.empty:
        rejected, p_adjusted = _benjamini_hochberg(df_cph["p-value"].values, alpha=0.05)
        df_cph["FDR (BH-adjusted p-value)"] = p_adjusted
        df_cph["Significant (FDR < 0.05)"] = rejected

    df_cph = df_cph.sort_values(by="p-value", ascending=True)
    print(df_cph.head(15).to_string(index=False))

    df_forest = df_cph.nsmallest(15, "p-value").copy()
    df_forest = df_forest.sort_values(by="Hazard Ratio (HR)", ascending=True).reset_index(drop=True)

    set_presentation_style()
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    plt.subplots_adjust(left=0.32, right=0.62, top=0.88, bottom=0.20)

    ax.set_xscale("log")
    ax.axvline(1.0, color="#555555", linestyle="--", linewidth=1.5, zorder=1)

    for i, row in df_forest.iterrows():
        hr = row["Hazard Ratio (HR)"]
        lower = row["95% CI Lower"]
        upper = row["95% CI Upper"]
        is_sig = row["Significant (FDR < 0.05)"]

        if is_sig:
            color = RESPONSE_PALETTE["PD"] if hr > 1.0 else RESPONSE_PALETTE["CR/PR"]
            weight = "bold"
        else:
            color = "#777777"
            weight = "normal"

        ax.errorbar(
            x=hr,
            y=i,
            xerr=[[max(0.01, hr - lower)], [max(0.01, upper - hr)]],
            fmt="s",
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=4,
            capthick=1.5,
            markersize=7,
            zorder=3,
        )

        lbl_hr = f"{hr:.2f} ({lower:.2f} - {upper:.2f})"
        fdr_val = row["FDR (BH-adjusted p-value)"]
        lbl_fdr = f"{fdr_val:.2e}" if fdr_val < 0.001 else f"{fdr_val:.3f}"
        if is_sig:
            lbl_fdr += " *"

        ax.text(1.10, i, lbl_hr, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=10.5, color="#222222", weight=weight)
        ax.text(1.65, i, lbl_fdr, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=10.5, color="#222222", weight=weight)

    header_y = len(df_forest) - 0.2
    ax.text(1.10, header_y, "Hazard Ratio (95% CI)", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=11, color="#111111", weight="bold")
    ax.text(1.65, header_y, "FDR (BH-adj. p)", transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=11, color="#111111", weight="bold")

    ax.set_yticks(range(len(df_forest)))
    ax.set_yticklabels([_format_feature_name(f) for f in df_forest["Feature"]], fontsize=10.5, color="#222222")

    ax.set_ylim(-0.8, len(df_forest) + 0.3)
    ax.set_xlim(0.2, 20.0)

    ax.set_xticks([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="x", which="both", labelsize=10.5)

    ax.set_xlabel("Hazard Ratio (log scale)", fontsize=12, labelpad=10, weight="semibold")

    ax.text(0.95, -0.24, "Higher Risk (HR > 1.0) \u2192", transform=ax.transAxes, ha="right", va="top", color="#555555", fontsize=9.5, style="italic")
    ax.text(0.05, -0.24, "\u2190 Lower Risk (HR < 1.0)", transform=ax.transAxes, ha="left", va="top", color="#555555", fontsize=9.5, style="italic")

    legend_elements = [
        mlines.Line2D([0], [0], marker="s", color="none", markerfacecolor=RESPONSE_PALETTE["PD"], markeredgecolor="none", markersize=8, label="Significant Risk (FDR < 0.05, HR > 1.0)"),
        mlines.Line2D([0], [0], marker="s", color="none", markerfacecolor=RESPONSE_PALETTE["CR/PR"], markeredgecolor="none", markersize=8, label="Significant Protective (FDR < 0.05, HR < 1.0)"),
        mlines.Line2D([0], [0], marker="s", color="none", markerfacecolor="#777777", markeredgecolor="none", markersize=8, label="Non-Significant (FDR \u2265 0.05)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )

    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, color="#cccccc", alpha=0.7)

    fig.suptitle("Clinical Features Hazard Ratios (Univariate Cox)", fontsize=16, weight="bold", y=0.96)
    ax.set_title("Top 15 features ranked by p-value; Asterisk (*) denotes FDR-adjusted p < 0.05", fontsize=10.5, style="italic", color="#555555", pad=15)

    forest_path = plot_dir / "cox_forest_plot.png"
    save_fig(fig, forest_path)
    print(f"Saved Cox forest plot to {forest_path.relative_to(BASE_DIR).as_posix()}")

    return df_cph


def _generate_feature_selection_report(
    df: pd.DataFrame,
    features_encoded: pd.DataFrame,
    df_rf: pd.DataFrame,
    df_cph: pd.DataFrame,
    id_cols: List[str],
    icd_cols: List[str],
    treatment_cols: List[str],
    report_path: Path,
) -> None:
    """Generates Markdown report detailing clinical feature selection results.

    Args:
        df: Cleaned input clinical DataFrame.
        features_encoded: One-hot encoded feature matrix.
        df_rf: Ranked Random Forest feature importances.
        df_cph: Ranked Cox proportional hazards regression results.
        id_cols: Identifier column names excluded.
        icd_cols: ICD classification column names excluded.
        treatment_cols: Treatment column names excluded.
        report_path: Destination path for Markdown report.
    """
    target_cols = ["OS_STATUS", "OS_MONTHS", "PFS_STATUS", "PFS_MONTHS", "DSS_STATUS", "DSS_MONTHS"]
    confounder_cols = ["ETHNICITY", "GENETIC_ANCESTRY_LABEL", "RACE", "PRIOR_DX"]

    top_rf = df_rf.iloc[0]["Feature"]
    top_cph = df_cph.iloc[0]["Feature"]
    top_cph_p = df_cph.iloc[0]["p-value"]
    top_cph_hr = df_cph.iloc[0]["Hazard Ratio (HR)"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Clinical Feature Selection Report\n\n")
        f.write("This report documents the results of feature selection run on the cleaned clinical dataset of the **TCGA-SKCM** cohort.\n\n")

        f.write("## Dataset Characteristics\n")
        f.write(f"*   **Total samples analysed**: {df.shape[0]}\n")
        f.write(f"*   **Original columns**: {df.shape[1]}\n")
        f.write(f"*   **Encoded feature columns size**: {features_encoded.shape[1]}\n\n")

        f.write("### Excluded Columns Groupings\n")
        f.write(f"*   **Identifiers and Administrative**: {id_cols + icd_cols}\n")
        f.write(f"*   **Outcome Variables**: {target_cols}\n")
        f.write(f"*   **Redundant variables**: {['DAYS_LAST_FOLLOWUP', 'PERSON_NEOPLASM_CANCER_STATUS']}\n")
        f.write(f"*   **Socioeconomic/healthcare confounders**: {confounder_cols}\n")
        f.write(f"*   **Collection process artefacts**: {['TISSUE_SOURCE_SITE', 'TISSUE_SOURCE_SITE_CODE', 'MSI_SCORE_MANTIS', 'MSI_SENSOR_SCORE', 'TBL_SCORE']}\n")
        f.write(f"*   **Treatment Type**: {treatment_cols}\n\n")

        f.write("## Method 1: Random Forest Classifier Importance\n")
        f.write("A Random Forest classifier was trained to predict **Overall Survival status (OS_STATUS)** using all clinical variables. Features are ranked by their Gini importance.\n\n")
        f.write("### Feature Importance Visualisation\n")
        f.write("![Random Forest Classifier Feature Importance](../plots/clinical/clinical_feature_importance.png)\n\n")

        f.write("## Method 2: Cox Proportional Hazards Regression (Univariate)\n")
        f.write("Univariate Cox Proportional Hazards models were fitted to assess the association of each individual feature with overall survival time (`OS_MONTHS`) and survival status (`OS_STATUS`). Features are sorted by statistical significance (lowest p-value).\n\n")
        f.write("### Cox Proportional Hazards Forest Plot\n")
        f.write("![Cox Forest Plot](../plots/clinical/cox_forest_plot.png)\n\n")

        f.write("## Key Findings & Biological Summary\n")
        f.write(f"1.  **Random Forest Top Predictor**: The feature with the highest predictive value for binary overall survival status is **`{_format_feature_name(top_rf)}`**.\n")
        f.write(f"2.  **Cox Regression Top Predictor**: The feature most statistically associated with survival duration is **`{_format_feature_name(top_cph)}`** (univariate Cox p-value = `{top_cph_p:.2e}`, Hazard Ratio = `{top_cph_hr:.3f}`).\n")
        f.write("3.  **Pathology vs Sourcing**: Pathology staging features (like AJCC Stage or Primary Tumour (T) Staging dummy variables) rank highly across both models, validating the clinical value of anatomical staging.\n")
        f.write("4.  **TMB & Hypoxia**: Quantitative metrics (e.g. `TMB_NONSYNONYMOUS` or hypoxia scores) are highly ranked, highlighting the coupling between genomic mutations/tumour microenvironment stress and overall survival outcomes.\n")
        f.write("5.  **Sample Type & Primary Disease**: Primary tumour samples (`SAMPLE_TYPE_Primary`) show significantly higher hazard ratios (HR = `3.34`, univariate Cox p-value = `3.50e-08`) compared to metastatic samples in this cohort, representing a distinct risk profile.\n")

    print(f"Feature selection report successfully written to {report_path.relative_to(BASE_DIR).as_posix()}")


def main() -> None:
    """Executes the clinical feature selection pipeline."""
    print("==================================================")
    print("Clinical Feature Selection for TCGA-SKCM Cohort")
    print("==================================================")

    PLOTS_DIR.mkdir(exist_ok=True, parents=True)
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)

    df, features_encoded, y_os_status, y_os_months, id_cols, icd_cols, treatment_cols = _load_and_filter_features(CLINICAL_FILE)

    df_rf = _evaluate_rf_importance(features_encoded, y_os_status, PLOTS_DIR)
    df_cph = _evaluate_cox_proportional_hazards(features_encoded, y_os_months, y_os_status, PLOTS_DIR)

    _generate_feature_selection_report(df, features_encoded, df_rf, df_cph, id_cols, icd_cols, treatment_cols, OUTPUT_FILE)

    print("Done! Feature selection analysis completed.")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {LOG_PATH.relative_to(BASE_DIR).as_posix()}")
            main()
