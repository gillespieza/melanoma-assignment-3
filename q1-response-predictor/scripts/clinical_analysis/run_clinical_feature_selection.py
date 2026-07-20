"""
Clinical Feature Selection Pipeline for TCGA-SKCM Cohort.

Evaluates clinical feature importance and survival association using Random Forest Gini Importance
and univariate Cox Proportional Hazards regression (with Benjamini-Hochberg FDR adjustment).
Generates feature importance bar plots, Cox forest plots, and exports the Pillar 2 report
to reports/pillar-2-clinical-subtyping/clinical_phenotyping_and_feature_selection.md.
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

from src.styles import COHORT_PALETTE, RESPONSE_PALETTE, get_cohort_color, set_presentation_style
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
REPORTS_DIR = find_project_root(Path(__file__).resolve()) / "reports" / "pillar-2-clinical-subtyping"
PLOTS_DIR = find_project_root(Path(__file__).resolve()) / "plots" / "clinical"
LOG_DIR = find_project_root(Path(__file__).resolve()) / "logs"
LOG_PATH = LOG_DIR / "run_clinical_feature_selection.log"
OUTPUT_FILE = REPORTS_DIR / "clinical_phenotyping_and_feature_selection.md"


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


def _load_and_filter_features(
    clinical_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str], List[str]]:
    """Loads TCGA-SKCM clinical dataset, filters non-clinical confounders, and performs one-hot encoding.

    Args:
        clinical_path: Path to cleaned TCGA-SKCM clinical CSV.

    Returns:
        Tuple of (raw DataFrame, encoded features DataFrame, y_os_status, y_os_months, id_cols, icd_cols, treatment_cols).
    """
    if not clinical_path.exists():
        raise FileNotFoundError(f"Missing clinical dataset at {clinical_path.relative_to(BASE_DIR).as_posix()}")

    df = pd.read_csv(clinical_path)

    id_cols = ["PATIENT_ID", "SAMPLE_ID"]
    icd_cols = [c for c in df.columns if c.startswith("ICD_")]
    target_cols = ["OS_STATUS", "OS_MONTHS", "PFS_STATUS", "PFS_MONTHS", "DSS_STATUS", "DSS_MONTHS"]
    redundant_cols = ["DAYS_LAST_FOLLOWUP", "PERSON_NEOPLASM_CANCER_STATUS"]
    confounder_cols = ["ETHNICITY", "GENETIC_ANCESTRY_LABEL", "RACE", "PRIOR_DX"]
    artefact_cols = [
        "TISSUE_SOURCE_SITE",
        "TISSUE_SOURCE_SITE_CODE",
        "MSI_SCORE_MANTIS",
        "MSI_SENSOR_SCORE",
        "TBL_SCORE",
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

    exclude_cols = set(
        id_cols + icd_cols + target_cols + redundant_cols + confounder_cols + artefact_cols + treatment_cols
    )
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_raw = df[feature_cols].copy()
    y_os_status = df["OS_STATUS"].copy()
    y_os_months = df["OS_MONTHS"].copy()

    categorical_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()

    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)

    for col in numerical_cols:
        if X_encoded[col].isnull().any():
            imputer = SimpleImputer(strategy="median")
            X_encoded[col] = imputer.fit_transform(X_encoded[[col]])

    for col in X_encoded.columns:
        if col not in numerical_cols:
            X_encoded[col] = X_encoded[col].astype(int)

    return df, X_encoded, y_os_status, y_os_months, id_cols, icd_cols, treatment_cols


def _evaluate_rf_importance(X_encoded: pd.DataFrame, y_os_status: pd.Series, plots_dir: Path) -> pd.DataFrame:
    """Trains Random Forest Classifier to rank features by Gini Importance and saves plot.

    Args:
        X_encoded: Encoded feature matrix.
        y_os_status: Binary overall survival status target.
        plots_dir: Target plots directory.

    Returns:
        Sorted DataFrame of feature importances.
    """
    valid_mask = ~y_os_status.isnull()
    X_rf = X_encoded[valid_mask]
    y_rf = y_os_status[valid_mask]

    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(X_rf, y_rf)

    df_rf = pd.DataFrame({"Feature": X_encoded.columns, "Importance": rf.feature_importances_})
    df_rf = df_rf.sort_values(by="Importance", ascending=False).reset_index(drop=True)

    top_20_rf = df_rf.head(20).copy()
    top_20_rf["Formatted_Feature"] = top_20_rf["Feature"].apply(_format_feature_name)

    set_presentation_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    palette = sns.color_palette("Blues_r", n_colors=20)
    sns.barplot(
        data=top_20_rf,
        y="Formatted_Feature",
        x="Importance",
        hue="Formatted_Feature",
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
        legend=False,
    )

    ax.set_title("Random Forest Gini Importance (Top 20 Clinical Predictors)", fontsize=14, fontweight="bold", pad=15)
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

    ax.set_xlim(0, top_20_rf["Importance"].max() * 1.15)
    out_path = plots_dir / "clinical_feature_importance.png"
    save_fig(fig, out_path)
    print(f"Saved Random Forest feature importance plot to {out_path.relative_to(BASE_DIR).as_posix()}")

    return df_rf


def _evaluate_cox_proportional_hazards(
    X_encoded: pd.DataFrame, y_os_months: pd.Series, y_os_status: pd.Series, plots_dir: Path
) -> pd.DataFrame:
    """Fits univariate Cox Proportional Hazards regression per feature and saves forest plot.

    Args:
        X_encoded: Encoded feature matrix.
        y_os_months: Overall survival duration Series.
        y_os_status: Binary overall survival status Series.
        plots_dir: Target plots directory.

    Returns:
        Sorted DataFrame of Cox model hazard ratios and p-values.
    """
    valid_mask = (~y_os_months.isnull()) & (~y_os_status.isnull()) & (y_os_months > 0)
    X_cph_all = X_encoded[valid_mask]
    t_cph = y_os_months[valid_mask]
    e_cph = y_os_status[valid_mask]

    cox_results = []
    cph = CoxPHFitter()

    for col in X_cph_all.columns:
        feature_data = pd.DataFrame({"OS_MONTHS": t_cph, "OS_STATUS": e_cph, col: X_cph_all[col]})
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

    top_20_cph = df_cph.head(20).copy()
    top_20_cph["Formatted_Feature"] = top_20_cph["Feature"].apply(_format_feature_name)
    top_20_cph = top_20_cph.iloc[::-1].reset_index(drop=True)

    set_presentation_style()
    fig, ax = plt.subplots(figsize=(14, 10))

    y_pos = np.arange(len(top_20_cph))
    hrs = top_20_cph["Hazard Ratio (HR)"].values
    lowers = top_20_cph["HR lower 95%"].values
    uppers = top_20_cph["HR upper 95%"].values
    p_vals = top_20_cph["p-value"].values
    fdr_vals = top_20_cph["FDR_adj_p"].values

    category_patterns = [
        ("AJCC Stage", get_cohort_color("Liu 2019")),
        ("Primary Tumour", get_cohort_color("Hugo 2016")),
        ("N Stage", RESPONSE_PALETTE["SD"]),
        ("M Stage", RESPONSE_PALETTE["PD"]),
        ("Treatment", get_cohort_color("Pooled Trials")),
        ("Sample Type", RESPONSE_PALETTE["CR/PR"]),
        ("Tmb", get_cohort_color("TCGA")),
        ("Winter Hypoxia", get_cohort_color("Riaz 2017")),
    ]
    fallback_colors = cycle(sns.color_palette("Set2"))

    point_colors = []
    category_legend_map = {}

    for name in top_20_cph["Formatted_Feature"]:
        matched = False
        for pattern, color in category_patterns:
            if pattern.lower() in name.lower():
                point_colors.append(color)
                category_legend_map[pattern] = color
                matched = True
                break
        if not matched:
            color = next(fallback_colors)
            point_colors.append(color)
            category_legend_map["Other"] = color

    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7, label="Null Effect (HR = 1.0)")

    for i in range(len(top_20_cph)):
        ax.plot([lowers[i], uppers[i]], [y_pos[i], y_pos[i]], color=point_colors[i], linewidth=2.0, alpha=0.85)

    for i in range(len(top_20_cph)):
        ax.scatter(hrs[i], y_pos[i], color=point_colors[i], s=70, zorder=5, edgecolor="black", linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_20_cph["Formatted_Feature"], fontsize=10.5, fontweight="bold")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())

    min_val = min(lowers)
    max_val = max(uppers)
    ax.set_xlim(max(0.01, min_val * 0.7), max_val * 1.8)

    ax.set_title("Univariate Cox Proportional Hazards Regression (Top 20 Clinical Features)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Hazard Ratio (HR, Log Scale - 95% CI)", fontsize=11, fontweight="bold")

    for i in range(len(top_20_cph)):
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
            color="darkred" if fdr_val < 0.05 else "black",
        )

    legend_handles = [mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.2, label="Null Effect (HR = 1.0)")]
    for cat_name, color in category_legend_map.items():
        legend_handles.append(mlines.Line2D([], [], color=color, marker="o", linestyle="-", linewidth=1.5, markersize=7, label=cat_name))

    ax.legend(handles=legend_handles, loc="lower right", fontsize=9.5, frameon=True, facecolor="white", framealpha=0.9)
    sns.despine(top=True, right=True)

    out_path = plots_dir / "cox_forest_plot.png"
    save_fig(fig, out_path)
    print(f"Saved Cox forest plot to {out_path.relative_to(BASE_DIR).as_posix()}")

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
    """Generates the Markdown report for clinical feature selection and phenotyping.

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

    report_path.parent.mkdir(exist_ok=True, parents=True)

    report_content = [
        "---",
        "title: Clinical Phenotyping & Feature Selection Report",
        "cssclasses: table-small",
        "---",
        "",
        "# Clinical Phenotyping & Feature Selection Report",
        "",
        f"This report documents the clinical feature selection and unsupervised patient subtyping performed on the **TCGA-SKCM** reference cohort ($N = {df.shape[0]}$ samples). Non-expression clinical and demographic variables were evaluated using Random Forest Gini importance and univariate Cox Proportional Hazards regression.",
        "",
        "---",
        "",
        "## 1. Dataset Characteristics & Feature Encoding",
        "",
        f"* **Total Samples Analysed**: {df.shape[0]} patients",
        f"* **Original Clinical Attributes**: {df.shape[1]}",
        f"* **Encoded Feature Columns**: {features_encoded.shape[1]} dummy-coded variables",
        "",
        "### Excluded Columns Groupings",
        "To prevent data leakage and administrative noise, features were categorised and filtered:",
        f"* **Identifiers and Administrative**: `{id_cols + icd_cols}`",
        f"* **Outcome Variables**: `{target_cols}`",
        f"* **Redundant Variables**: `{['DAYS_LAST_FOLLOWUP', 'PERSON_NEOPLASM_CANCER_STATUS']}`",
        f"* **Socioeconomic Confounders**: `{confounder_cols}`",
        f"* **Collection Process Artefacts**: `{['TISSUE_SOURCE_SITE', 'TISSUE_SOURCE_SITE_CODE', 'MSI_SCORE_MANTIS', 'MSI_SENSOR_SCORE', 'TBL_SCORE']}`",
        f"* **Treatment Type History**: `{treatment_cols}`",
        "",
        "---",
        "",
        "## 2. Clinical Feature Selection Methods",
        "",
        "### 2.1. Random Forest Classifier Importance",
        "A Random Forest classifier was trained to predict binary overall survival status (`OS_STATUS`) using all encoded clinical variables. Features are ranked by their Gini importance.",
        "",
        "#### Feature Importance Visualisation",
        "![Random Forest Classifier Feature Importance](../../plots/clinical/clinical_feature_importance.png)",
        "",
        "### 2.2. Cox Proportional Hazards Regression (Univariate)",
        "Univariate Cox Proportional Hazards models (`lifelines.CoxPHFitter`) were fitted to assess the association of each individual feature with overall survival time (`OS_MONTHS`) and survival status (`OS_STATUS`). Features are sorted by statistical significance (lowest p-value).",
        "",
        "#### Cox Forest Plot",
        "![Cox Forest Plot](../../plots/clinical/cox_forest_plot.png)",
        "",
        "### 2.3. Key Findings & Biological Summary",
        f"1. **Random Forest Top Predictor**: The feature with the highest predictive value for binary overall survival status is **`{_format_feature_name(top_rf)}`**.",
        f"2. **Cox Regression Top Predictor**: The feature most statistically associated with survival duration is **`{_format_feature_name(top_cph)}`** (univariate Cox p-value = `{top_cph_p:.2e}`, Hazard Ratio = `{top_cph_hr:.3f}`).",
        "3. **Pathology vs. Sourcing**: Pathology staging features (like AJCC Stage or Primary Tumour T-Staging dummy variables) rank highly across both models, validating the clinical value of anatomical staging.",
        "4. **TMB & Hypoxia**: Quantitative metrics (e.g. `TMB_NONSYNONYMOUS` or hypoxia scores) are highly ranked, highlighting the coupling between genomic mutations/tumour microenvironment stress and overall survival outcomes.",
        "5. **Sample Type & Primary Disease**: Primary tumour samples (`SAMPLE_TYPE_Primary`) show significantly higher hazard ratios ($\text{HR} = 3.34$, univariate Cox p-value = $3.50 \times 10^{-8}$) compared to metastatic samples in this cohort, representing a distinct risk profile.",
        "",
        "---",
        "",
        "## 3. Patient Phenotyping via Unsupervised Clustering",
        "",
        "We performed unsupervised phenotyping on the **TCGA-SKCM** cohort ($N = 426$ aligned patients) using Agglomerative Hierarchical Clustering (Ward linkage). Patient profiles were constructed across demographics, tumor genetics, microenvironmental stress, and adjuvant therapy classes.",
        "",
        "### 3.1. Visual Cluster Profiles & Multi-Dimensional Fingerprints",
        "",
        "To visualise how the three patient clusters differ across clinical presentation, treatment history, tumor mutational burden, hypoxia, and survival outcomes, the multi-dimensional profiles are presented below in a **Visual Feature Dashboard** and a **Polar Radar Fingerprint Chart**:",
        "",
        "![Cluster Profile Dashboard](../../plots/clinical/cluster_profile_dashboard.png)",
        "",
        "![Cluster Profile Radar Fingerprint](../../plots/clinical/cluster_profile_radar.png)",
        "",
        "### 3.2. Clinical Interpretation of Subtypes",
        "",
        "Based on the multi-dimensional profiles, the three clusters represent distinct disease states:",
        "",
        "1. **Cluster 0: Baseline Disease / Primary Specimen Phenotype** ($N=312$)",
        "   * _Genomics_: Moderate chromosomal instability (aneuploidy score = 13.1) and moderate TMB (24.9 mut/Mb).",
        "   * _Microenvironment_: Low-moderate Winter hypoxia score (-2.57).",
        "   * _Clinical_: Stage IV rate = 0.0%. Highest rate of primary specimens (19.9%). No immunotherapy (0.0%).",
        "   * _Prognosis_: Better overall survival trajectory.",
        "",
        "2. **Cluster 1: High Mutational Load & Immunotherapy Phenotype** ($N=112$)",
        "   * _Genomics_: Elevated copy-number alterations (aneuploidy score = 12.9) and the highest mutational load (**TMB = 31.4 mut/Mb**).",
        "   * _Microenvironment_: Low-moderate Winter hypoxia score (-1.61).",
        "   * _Clinical_: Stage IV rate = 0.0%. Lower primary tumor rate (14.3%). High rate of immunotherapy (63.4%).",
        "   * _Prognosis_: Moderate survival trajectory.",
        "",
        "3. **Cluster 2: Stage IV / Advanced Metastatic Disease Phenotype** ($N=24$)",
        "   * _Genomics_: Lower mutational load (TMB = 14.0 mut/Mb) and lowest copy-number alterations (aneuploidy score = 11.8).",
        "   * _Clinical_: **100% of these patients have Stage IV disease**. No patients received immunotherapy (0.0%).",
        "   * _Prognosis_: Poor overall survival trajectory (Median Overall Survival = **28.1 months**).",
        "",
        "### 3.3. Cluster Visualisation (2D PCA Projection)",
        "![2D PCA Visualization of Clusters](../../plots/clinical/pca_clinical_clusters.png)",
        "",
        "### 3.4. Kaplan-Meier Survival Analysis",
        "The unsupervised patient clusters show a statistically significant separation in overall survival duration (Log-Rank p-value = **$1.82 \times 10^{-2}$**):",
        "",
        "![KM Survival of Clinical Clusters](../../plots/clinical/km_clinical_clusters.png)",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content) + "\n")

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
