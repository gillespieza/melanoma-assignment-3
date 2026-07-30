#!/usr/bin/env python3
"""Script 05: Train subgroup-specific predictive models.

Evaluates whether training separate predictive models within discovered patient phenotypes
improves response prediction performance compared to applying the global Q1 predictive
model across all subgroups.
"""

import contextlib
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Bootstrap project root resolution for top-level imports
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SUBPROJECT_ROOT = SCRIPT_DIR.parent
for parent in [SCRIPT_DIR] + list(SCRIPT_DIR.parents):
    if (parent / "src").is_dir() and (parent / "data").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

if str(SUBPROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Project Imports
# ---------------------------------------------------------------------------

from src.styles import (
    PHENOTYPE_PALETTE,
    RESPONSE_PALETTE,
    STRATEGY_PALETTE,
    set_presentation_style,
)
from src.utils.logging import TeeStream
from src.utils.paths import PROCESSED_DIR, PROJECT_ROOT, rel_path
from src.utils.plotting import save_fig


# ---------------------------------------------------------------------------
# Module-level Constants & Definitions
# ---------------------------------------------------------------------------

LOG_DIR = SUBPROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "05_subgroup_models.log"

INPUT_FILE = PROCESSED_DIR / "q5" / "patient_clusters.csv"
OUTPUT_DIR = PROCESSED_DIR / "q5"
MODELS_DIR = SUBPROJECT_ROOT / "models"
PLOTS_DIR = SUBPROJECT_ROOT / "plots" / "subgroup_models"

set_presentation_style()

PHENOTYPE_COLORS: Dict[int, str] = {
    0: PHENOTYPE_PALETTE["Immune Hot"],                 # Okabe-Ito Vermillion (Immune Hot)
    1: PHENOTYPE_PALETTE["Immune Cold"],                # Okabe-Ito Blue (Immune Cold)
    2: PHENOTYPE_PALETTE["Immunosuppressive M2-High"],  # Okabe-Ito Reddish Purple (M2 Immunosuppressive)
    3: PHENOTYPE_PALETTE["Mutant-Driven"],              # Okabe-Ito Orange (Mutant-Driven / NF1 Loss)
}

PHENOTYPE_SHORT_NAMES: Dict[int, str] = {
    0: "Immune Hot",
    1: "Immune Cold",
    2: "M2 Immunosuppressive",
    3: "Mutant-Driven",
}


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Identify available numerical and binary features for modelling."""
    candidate_features = [
        "IFN_gamma",
        "TIS",
        "CYT",
        "CD8_Tcell",
        "IMPRES",
        "PD_L1",
        "M1_M2_Ratio",
        "Macrophage_STV_Score",
        "CD8_T_cells",
        "CD4_T_cells",
        "NK_cells",
        "B_cells",
        "M1_Macrophages",
        "M2_Macrophages",
        "CAFs",
        "TMB_NONSYNONYMOUS",
        "mut_BRAF",
        "mut_NRAS",
        "mut_NF1",
    ]
    avail = [col for col in candidate_features if col in df.columns and df[col].notna().any()]
    return avail


def evaluate_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "ROC_AUC": auc,
        "PR_AUC": pr_auc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "Accuracy": acc,
        "Brier_Score": brier,
        "PPV": ppv,
        "NPV": npv,
    }


def train_and_eval_loco(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]], Dict[str, RandomForestClassifier]]:
    """Perform Leave-One-Cohort-Out CV for Global vs Subgroup models."""
    df_valid = df.dropna(subset=["RESPONSE_BINARY"]).copy()
    df_valid["RESPONSE_BINARY"] = df_valid["RESPONSE_BINARY"].astype(int)

    cohorts = df_valid["COHORT"].values
    clusters = df_valid["Cluster_ID"].values
    unique_clusters = sorted(df_valid["Cluster_ID"].unique())

    # Pre-impute and scale features globally per fold to prevent leakage
    X_raw = df_valid[feature_cols].values
    y_raw = df_valid["RESPONSE_BINARY"].values

    # Storage for out-of-fold probability predictions
    global_oof_prob = np.zeros(len(df_valid))
    subgroup_oof_prob = np.zeros(len(df_valid))

    logo = LeaveOneGroupOut()

    # 1. Global Model Out-Of-Fold predictions
    for train_idx, test_idx in logo.split(X_raw, y_raw, groups=cohorts):
        X_tr, y_tr = X_raw[train_idx], y_raw[train_idx]
        X_te = X_raw[test_idx]

        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_tr_proc = scaler.fit_transform(imp.fit_transform(X_tr))
        X_te_proc = scaler.transform(imp.transform(X_te))

        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        clf.fit(X_tr_proc, y_tr)
        global_oof_prob[test_idx] = clf.predict_proba(X_te_proc)[:, 1]

    # 2. Subgroup-Specific Model Out-Of-Fold predictions
    for cid in unique_clusters:
        c_mask = clusters == cid
        c_indices = np.where(c_mask)[0]
        X_c = X_raw[c_indices]
        y_c = y_raw[c_indices]
        groups_c = cohorts[c_indices]

        # Check if cohort groups allow LOCO within this subgroup
        unique_groups_c = np.unique(groups_c)
        if len(unique_groups_c) >= 2:
            logo_sub = LeaveOneGroupOut()
            for tr_sub, te_sub in logo_sub.split(X_c, y_c, groups=groups_c):
                real_tr = c_indices[tr_sub]
                real_te = c_indices[te_sub]

                if len(np.unique(y_c[tr_sub])) < 2:
                    subgroup_oof_prob[real_te] = global_oof_prob[real_te]
                    continue

                imp = SimpleImputer(strategy="median")
                scaler = StandardScaler()
                X_tr_p = scaler.fit_transform(imp.fit_transform(X_c[tr_sub]))
                X_te_p = scaler.transform(imp.transform(X_c[te_sub]))

                clf_sub = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
                clf_sub.fit(X_tr_p, y_c[tr_sub])
                subgroup_oof_prob[real_te] = clf_sub.predict_proba(X_te_p)[:, 1]
        else:
            # Fallback to Stratified K-Fold within small subgroup
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            for tr_sub, te_sub in skf.split(X_c, y_c):
                real_tr = c_indices[tr_sub]
                real_te = c_indices[te_sub]

                imp = SimpleImputer(strategy="median")
                scaler = StandardScaler()
                X_tr_p = scaler.fit_transform(imp.fit_transform(X_c[tr_sub]))
                X_te_p = scaler.transform(imp.transform(X_c[te_sub]))

                clf_sub = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
                clf_sub.fit(X_tr_p, y_c[tr_sub])
                subgroup_oof_prob[real_te] = clf_sub.predict_proba(X_te_p)[:, 1]

    df_valid["Global_OOF_Prob"] = global_oof_prob
    df_valid["Subgroup_OOF_Prob"] = subgroup_oof_prob

    # 3. Fit final production models on full datasets
    imp_final = SimpleImputer(strategy="median")
    scaler_final = StandardScaler()
    X_full_proc = scaler_final.fit_transform(imp_final.fit_transform(X_raw))

    global_final_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    global_final_model.fit(X_full_proc, y_raw)

    final_subgroup_models: Dict[str, RandomForestClassifier] = {"Global": global_final_model}

    results_list = []

    # Overall dataset metrics
    g_metrics_all = evaluate_predictions(df_valid["RESPONSE_BINARY"].values, df_valid["Global_OOF_Prob"].values)
    s_metrics_all = evaluate_predictions(df_valid["RESPONSE_BINARY"].values, df_valid["Subgroup_OOF_Prob"].values)

    results_list.append(
        {
            "Cluster_ID": -1,
            "Phenotype": "Overall Cohort",
            "Model_Scope": "Global Q1 Predictor",
            "N": len(df_valid),
            "Responders": int(df_valid["RESPONSE_BINARY"].sum()),
            "Response_Rate": df_valid["RESPONSE_BINARY"].mean() * 100,
            **g_metrics_all,
        }
    )
    results_list.append(
        {
            "Cluster_ID": -1,
            "Phenotype": "Overall Cohort",
            "Model_Scope": "Subgroup Ensemble",
            "N": len(df_valid),
            "Responders": int(df_valid["RESPONSE_BINARY"].sum()),
            "Response_Rate": df_valid["RESPONSE_BINARY"].mean() * 100,
            **s_metrics_all,
        }
    )

    roc_data: Dict[str, Dict[str, np.ndarray]] = {}

    # Per-cluster metrics
    for cid in unique_clusters:
        p_name = PHENOTYPE_SHORT_NAMES.get(cid, f"Cluster {cid}")
        c_sub = df_valid[df_valid["Cluster_ID"] == cid]
        y_sub = c_sub["RESPONSE_BINARY"].values
        n_sub = len(c_sub)
        n_resp = int(y_sub.sum())

        g_prob = c_sub["Global_OOF_Prob"].values
        s_prob = c_sub["Subgroup_OOF_Prob"].values

        g_met = evaluate_predictions(y_sub, g_prob)
        s_met = evaluate_predictions(y_sub, s_prob)

        results_list.append(
            {
                "Cluster_ID": cid,
                "Phenotype": p_name,
                "Model_Scope": "Global Q1 Predictor",
                "N": n_sub,
                "Responders": n_resp,
                "Response_Rate": y_sub.mean() * 100,
                **g_met,
            }
        )
        results_list.append(
            {
                "Cluster_ID": cid,
                "Phenotype": p_name,
                "Model_Scope": "Subgroup Specific",
                "N": n_sub,
                "Responders": n_resp,
                "Response_Rate": y_sub.mean() * 100,
                **s_met,
            }
        )

        # ROC Curve points
        fpr_g, tpr_g, _ = roc_curve(y_sub, g_prob)
        fpr_s, tpr_s, _ = roc_curve(y_sub, s_prob)
        roc_data[p_name] = {
            "fpr_global": fpr_g,
            "tpr_global": tpr_g,
            "fpr_subgroup": fpr_s,
            "tpr_subgroup": tpr_s,
        }

        # Train final subgroup model on all cluster samples
        X_sub_raw = c_sub[feature_cols].values
        imp_sub = SimpleImputer(strategy="median")
        scaler_sub = StandardScaler()
        X_sub_proc = scaler_sub.fit_transform(imp_sub.fit_transform(X_sub_raw))

        clf_final = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)
        clf_final.fit(X_sub_proc, y_sub)
        final_subgroup_models[p_name] = clf_final

    df_eval = pd.DataFrame(results_list)
    return df_eval, roc_data, final_subgroup_models


def plot_subgroup_roc_curves(
    roc_data: Dict[str, Dict[str, np.ndarray]], df_eval: pd.DataFrame, out_path: Path
) -> None:
    """Generate 2x2 multi-panel ROC curves comparing Global Q1 vs Subgroup Models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (p_name, r_dict) in enumerate(roc_data.items()):
        ax = axes[idx]
        g_row = df_eval[(df_eval["Phenotype"] == p_name) & (df_eval["Model_Scope"] == "Global Q1 Predictor")].iloc[0]
        s_row = df_eval[(df_eval["Phenotype"] == p_name) & (df_eval["Model_Scope"] == "Subgroup Specific")].iloc[0]

        n_pts = int(g_row["N"])
        g_auc = g_row["ROC_AUC"]
        s_auc = s_row["ROC_AUC"]

        ax.plot(
            r_dict["fpr_global"],
            r_dict["tpr_global"],
            color="#37474F",
            linestyle="--",
            linewidth=2,
            label=f"Global Q1 (AUC = {g_auc:.3f})",
        )
        phenotype_color = PHENOTYPE_PALETTE.get(p_name, RESPONSE_PALETTE["CR/PR"])
        ax.plot(
            r_dict["fpr_subgroup"],
            r_dict["tpr_subgroup"],
            color=phenotype_color,
            linestyle="-",
            linewidth=2.5,
            label=f"Subgroup Model (AUC = {s_auc:.3f})",
        )
        ax.plot([0, 1], [0, 1], color="#9E9E9E", linestyle=":", linewidth=1)

        ax.set_title(f"{p_name} (N = {n_pts})", fontsize=13, fontweight="bold")
        ax.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=11)
        ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=11)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=10)
        ax.grid(True, color="#E5E7EB", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    save_fig(fig, out_path, dpi=300)
    plt.close(fig)


def plot_performance_comparison(df_eval: pd.DataFrame, out_path: Path) -> None:
    """Generate comparative bar chart of metrics (ROC-AUC, Precision, Recall) across phenotypes."""
    df_plot = df_eval[df_eval["Phenotype"] != "Overall Cohort"].copy()

    metrics = ["ROC_AUC", "PR_AUC", "Precision", "Recall"]
    df_melt = df_plot.melt(
        id_vars=["Phenotype", "Model_Scope"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )
    df_melt["Metric"] = df_melt["Metric"].replace({"ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC"})

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = STRATEGY_PALETTE

    sns.barplot(
        data=df_melt,
        x="Phenotype",
        y="Score",
        hue="Model_Scope",
        palette=palette,
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_title("Subgroup Model vs Global Q1 Predictor Performance across Melanoma Phenotypes", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Biological Phenotype Subgroup", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cross-Validation Score (LOCO CV)", fontsize=11, fontweight="bold")
    ax.set_ylim([0.0, 1.05])
    ax.legend(title="Model Strategy", frameon=True, facecolor="white", edgecolor="#E5E7EB", fontsize=10)
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.5, alpha=0.6)

    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    save_fig(fig, out_path, dpi=300)
    plt.close(fig)


def plot_feature_importances(
    final_models: Dict[str, RandomForestClassifier], feature_cols: List[str], out_path: Path
) -> None:
    """Plot feature importance heatmaps comparing driver weights across phenotypes."""
    importance_dict = {}
    for p_name, clf in final_models.items():
        if hasattr(clf, "feature_importances_"):
            importance_dict[p_name] = clf.feature_importances_

    df_imp = pd.DataFrame(importance_dict, index=feature_cols)

    # Re-order features by highest average importance
    df_imp["Mean_Importance"] = df_imp.mean(axis=1)
    df_imp = df_imp.sort_values(by="Mean_Importance", ascending=False).drop(columns=["Mean_Importance"])
    top_df_imp = df_imp.head(12)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        top_df_imp,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={"label": "Random Forest Gini Importance"},
        ax=ax,
        linewidths=0.5,
        linecolor="#E5E7EB",
    )

    ax.set_title("Phenotype-Specific Feature Importance Profiles (Top 12 Features)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Melanoma Phenotype Subgroup", fontsize=11, fontweight="bold")
    ax.set_ylabel("Biomarker / Cell Signature Feature", fontsize=11, fontweight="bold")

    plt.tight_layout()
    save_fig(fig, out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Main execution entry point for Phase 5 Subgroup Predictive Modelling."""
    print("=" * 80)
    print(f"Starting Phase 5: Subgroup-Specific Predictive Modelling (Project root: {rel_path(PROJECT_ROOT)})")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing patient clusters file at {rel_path(INPUT_FILE)}. Run Phase 3 first.")

    df_clusters = pd.read_csv(INPUT_FILE)
    print(f"Loaded patient dataset: {len(df_clusters)} patients across {df_clusters['COHORT'].nunique()} cohorts")

    feature_cols = get_feature_columns(df_clusters)
    print(f"Selected {len(feature_cols)} biomarker & genomic features for subgroup modelling:")
    print(f"  * {', '.join(feature_cols)}")

    # 1. Evaluate LOCO CV & fit production models
    df_eval, roc_data, final_models = train_and_eval_loco(df_clusters, feature_cols)

    # 2. Save evaluation summary CSV
    out_eval_csv = OUTPUT_DIR / "subgroup_models_evaluation.csv"
    if out_eval_csv.exists():
        try:
            out_eval_csv.unlink()
        except Exception:
            pass
    df_eval.to_csv(out_eval_csv, index=False)
    print(f"\nSaved subgroup models evaluation summary to {rel_path(out_eval_csv)}")

    # 3. Serialise trained models to models/ directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in final_models.items():
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        m_file = MODELS_DIR / f"subgroup_model_{clean_name}.joblib"
        joblib.dump(model, m_file)
        print(f"  * Serialised {name} model -> {rel_path(m_file)}")

    # 4. Generate 300 DPI publication plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    roc_plot_file = PLOTS_DIR / "subgroup_roc_curves.png"
    comp_plot_file = PLOTS_DIR / "subgroup_performance_comparison.png"
    imp_plot_file = PLOTS_DIR / "subgroup_feature_importances.png"

    plot_subgroup_roc_curves(roc_data, df_eval, roc_plot_file)
    plot_performance_comparison(df_eval, comp_plot_file)
    plot_feature_importances(final_models, feature_cols, imp_plot_file)

    print("\n" + "=" * 80)
    print("PHASE 5 SUBGROUP MODELLING COMPLETE")
    print("=" * 80)
    print("Summary of Cross-Validation Performance (Global Q1 vs Subgroup Model):")
    for phenotype in df_eval["Phenotype"].unique():
        sub_df = df_eval[df_eval["Phenotype"] == phenotype]
        g_auc = sub_df[sub_df["Model_Scope"] == "Global Q1 Predictor"]["ROC_AUC"].values[0]
        s_auc = sub_df[sub_df["Model_Scope"] != "Global Q1 Predictor"]["ROC_AUC"].values[0]
        g_ppv = sub_df[sub_df["Model_Scope"] == "Global Q1 Predictor"]["PPV"].values[0]
        s_ppv = sub_df[sub_df["Model_Scope"] != "Global Q1 Predictor"]["PPV"].values[0]

        diff_auc = s_auc - g_auc if not np.isnan(s_auc) and not np.isnan(g_auc) else 0.0
        print(
            f"  * {phenotype:<25}: Global AUC = {g_auc:.3f} | Subgroup AUC = {s_auc:.3f} (Delta = {diff_auc:+.3f}) | Subgroup PPV = {s_ppv:.3f}"
        )

    print(f"\nGenerated Plots:")
    print(f"  1. {rel_path(roc_plot_file)}")
    print(f"  2. {rel_path(comp_plot_file)}")
    print(f"  3. {rel_path(imp_plot_file)}")
    print("=" * 80)


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log_file:
        stdout_tee = TeeStream(sys.stdout, log_file)
        stderr_tee = TeeStream(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            print(f"Logging console output to {rel_path(LOG_PATH)}")
            main()
