import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set matplotlib backend to Agg to avoid GUI errors
import matplotlib
matplotlib.use('Agg')

# Add project root to sys.path for importing src modules
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_liu_2019, load_hugo_2016, load_riaz_2017
from src.signatures import extract_all_signatures
from src.models import run_loco_cv
from src.evaluation import plot_roc_curves, plot_pr_curves, plot_confusion_matrices, calculate_extended_metrics, calculate_cindex, run_survival_analysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def zscore_df(df):
    """
    Standardize DataFrame columns individually (Z-score scaling).
    Avoids division by zero if std is zero.
    """
    means = df.mean(axis=0)
    stds = df.std(axis=0)
    stds = stds.replace(0, 1.0).fillna(1.0)
    return (df - means) / stds

# Paths
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots" / "models"
PLOT_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

def extract_driver_mutations(df_meta):
    """
    Returns BRAF, NRAS, NF1 mutation status from pre-cleaned clinical metadata.
    Columns mut_BRAF, mut_NRAS, mut_NF1 are pre-joined from mutations_cleaned.csv by the data loader.
    """
    df = df_meta.copy()
    for col in ['mut_BRAF', 'mut_NRAS', 'mut_NF1']:
        if col not in df.columns:
            df[col] = 0
    return df[['mut_BRAF', 'mut_NRAS', 'mut_NF1']]

def generate_model_evaluation_report(all_loco_results, output_dir):
    """
    Generates a comprehensive markdown report documenting model evaluation metrics.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    report_lines = [
        "# Model Evaluation Report: LOCO Cross-Cohort Validation\n",
        "## Overview\n",
        "This report documents the comprehensive evaluation of all trained models (Logistic Regression, Random Forest, XGBoost, SVM, ElasticNet) using Leave-One-Cohort-Out (LOCO) cross-validation on three independent melanoma immunotherapy cohorts.\n",
        "**Evaluation Framework:**\n",
        "- **Cross-validation**: Leave-One-Cohort-Out (LOCO) — train on 2 cohorts, test on 1\n",
        "- **Test cohorts**: Liu 2019 (N=104), Hugo 2016 (N=27), Riaz 2017 (N=64)\n",
        "- **Features**: 11 immune response signatures (IFN-gamma, TIS, CD8 T-cell, CYT, IMPRES, PD-L1, etc.)\n",
        "- **Decision threshold**: 0.5 (standard for binary classification)\n",
        "- **Metrics**: AUC-ROC, Accuracy, Sensitivity, Specificity, Precision, F1-score, C-index\n",
        "\n---\n\n"
    ]
    
    # Model summaries
    model_names = {
        'lr': 'Logistic Regression (L1-penalized, GridSearchCV)',
        'rf': 'Random Forest (GridSearchCV: n_estimators in [50,100,200], max_depth in [3,5,10,None], min_samples_leaf in [1,2,4])',
        'xgb': 'XGBoost (GridSearchCV: n_estimators in [50,100,150], max_depth in [3,5,7], learning_rate in [0.01,0.05,0.1,0.2])',
        'svm': 'Support Vector Machine (GridSearchCV: C in [0.01,0.1,1.0,10.0], kernel in [linear,rbf])',
        'elasticnet': 'ElasticNet Logistic Regression (GridSearchCV: C in [0.001,0.01,0.1,1.0,10.0], l1_ratio in [0.1,0.3,0.5,0.7,0.9])'
    }
    
    for model_key, loco_results in all_loco_results.items():
        report_lines.append(f"## {model_names.get(model_key, model_key.upper())}\n\n")
        
        # Create metrics table
        report_lines.append("### Performance Metrics (Threshold = 0.5)\n\n")
        report_lines.append("| Test Cohort | N | AUC | Accuracy | Sensitivity | Specificity | Precision | F1-Score | C-Index |\n")
        report_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        
        for cohort, res in sorted(loco_results.items()):
            if 'metrics_extended' in res:
                m = res['metrics_extended']
            else:
                m = calculate_extended_metrics(res['y_true'], res['y_pred_prob'])
            
            n_samples = len(res['y_true'])
            cindex_str = f"{m['cindex']:.3f}" if 'cindex' in m and not np.isnan(m.get('cindex', np.nan)) else "N/A"
            report_lines.append(
                f"| {cohort} | {n_samples} | {m['auc']:.3f} | {m['accuracy']:.3f} | {m['sensitivity']:.3f} | {m['specificity']:.3f} | {m['precision']:.3f} | {m['f1']:.3f} | {cindex_str} |\n"
            )
        
        # Confusion matrices
        report_lines.append("\n### Confusion Matrices (Threshold = 0.5)\n\n")
        for cohort, res in sorted(loco_results.items()):
            if 'metrics_extended' in res:
                m = res['metrics_extended']
            else:
                m = calculate_extended_metrics(res['y_true'], res['y_pred_prob'])
            
            report_lines.append(f"**{cohort}** (N={len(res['y_true'])}):\n")
            report_lines.append(f"```\n")
            report_lines.append(f"                Predicted\n")
            report_lines.append(f"              Non-Resp  Resp\n")
            report_lines.append(f"Actual Non-Resp    {m['tn']:<4} {m['fp']:<4}\n")
            report_lines.append(f"Actual Resp        {m['fn']:<4} {m['tp']:<4}\n")
            report_lines.append(f"```\n\n")
        
        # Precision-Recall summary
        report_lines.append("### Precision-Recall Curve Summaries\n\n")
        for cohort, res in sorted(loco_results.items()):
            y_true = res['y_true']
            y_pred_prob = res['y_pred_prob']
            
            if len(np.unique(y_true)) > 1:
                ap = average_precision_score(y_true, y_pred_prob)
                report_lines.append(f"**{cohort}**: Average Precision = {ap:.3f}\n")
            else:
                report_lines.append(f"**{cohort}**: Single class (skipped)\n")
        
        report_lines.append("\n---\n\n")
    
    report_lines.append("## Summary & Interpretation\n\n")
    report_lines.append("### Key Metrics Explained:\n")
    report_lines.append("- **Sensitivity (Recall)**: TP / (TP + FN) — Proportion of actual responders correctly identified\n")
    report_lines.append("- **Specificity**: TN / (TN + FP) — Proportion of actual non-responders correctly identified\n")
    report_lines.append("- **Precision**: TP / (TP + FP) — Proportion of predicted responders who are actually responders\n")
    report_lines.append("- **Accuracy**: (TP + TN) / Total — Overall correctness across both classes\n")
    report_lines.append("- **F1-Score**: Harmonic mean of Precision and Recall — Balances both metrics\n")
    report_lines.append("- **AUC-ROC**: Area under the Receiver Operating Characteristic curve — Robustness to threshold selection\n")
    report_lines.append("- **C-Index (Concordance Index)**: Evaluates how well predicted response probabilities rank patients by survival. 0.5 = random, 1.0 = perfect. Accounts for censoring in survival data.\n\n")
    report_lines.append("### Visualizations:\n")
    report_lines.append("- **ROC Curves** (`roc_curves_*.png`): Trade-off between True Positive Rate and False Positive Rate\n")
    report_lines.append("- **PR Curves** (`pr_curves_*.png`): Precision-Recall trade-off, especially relevant for class imbalance\n")
    report_lines.append("- **Confusion Matrices** (`confusion_matrices_*.png`): Cell-level breakdown of predictions per cohort\n\n")
    
    report_path = output_dir / "model_evaluation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print(f"\n✓ Model evaluation report saved to: {report_path}")
    return report_path

def main():
    print("==================================================")
    print("Phase 1: Loading Melanoma IO Cohorts...")
    print("==================================================")
    
    try:
        expr_liu, clin_liu = load_liu_2019(DATA_DIR)
        print(f"Loaded Liu 2019: {expr_liu.shape[0]} samples, {expr_liu.shape[1]} genes")
    except Exception as e:
        print(f"Error loading Liu 2019: {e}")
        return
        
    try:
        expr_hugo, clin_hugo = load_hugo_2016(DATA_DIR)
        print(f"Loaded Hugo 2016: {expr_hugo.shape[0]} samples, {expr_hugo.shape[1]} genes")
    except Exception as e:
        print(f"Error loading Hugo 2016: {e}")
        return
        
    try:
        expr_riaz, clin_riaz = load_riaz_2017(DATA_DIR)
        print(f"Loaded Riaz 2017: {expr_riaz.shape[0]} samples, {expr_riaz.shape[1]} genes")
    except Exception as e:
        print(f"Error loading Riaz 2017: {e}")
        return

    # Find intersection of genes
    common_genes = expr_liu.columns.intersection(expr_hugo.columns).intersection(expr_riaz.columns)
    print(f"Common genes across all 3 cohorts: {len(common_genes)}")
    
    # Filter expression matrices
    expr_liu = expr_liu[common_genes]
    expr_hugo = expr_hugo[common_genes]
    expr_riaz = expr_riaz[common_genes]

    print("\n==================================================")
    print("Phase 2: Computing Immune Signatures...")
    print("==================================================")
    
    sig_liu = extract_all_signatures(expr_liu)
    sig_hugo = extract_all_signatures(expr_hugo)
    sig_riaz = extract_all_signatures(expr_riaz)
    
    print(f"Liu signature shape: {sig_liu.shape}")
    print(f"Hugo signature shape: {sig_hugo.shape}")
    print(f"Riaz signature shape: {sig_riaz.shape}")

    # Align labels and filter out samples with missing response (NaN)
    non_nan_liu = clin_liu.loc[sig_liu.index, 'response'].dropna().index
    sig_liu = sig_liu.loc[non_nan_liu]
    y_liu = clin_liu.loc[non_nan_liu, 'response']
    
    non_nan_hugo = clin_hugo.loc[sig_hugo.index, 'response'].dropna().index
    sig_hugo = sig_hugo.loc[non_nan_hugo]
    y_hugo = clin_hugo.loc[non_nan_hugo, 'response']
    
    non_nan_riaz = clin_riaz.loc[sig_riaz.index, 'response'].dropna().index
    sig_riaz = sig_riaz.loc[non_nan_riaz]
    y_riaz = clin_riaz.loc[non_nan_riaz, 'response']

    print("\n==================================================")
    print("Phase 3: Cohort-Independent Standardization (Z-score)...")
    print("==================================================")
    
    # Scale each cohort individually to guarantee zero data leakage
    sig_corrected_liu = zscore_df(sig_liu)
    sig_corrected_hugo = zscore_df(sig_hugo)
    sig_corrected_riaz = zscore_df(sig_riaz)
    
    # Concatenate the standardized signatures for overall scaler fitting
    sig_corrected = pd.concat([sig_corrected_liu, sig_corrected_hugo, sig_corrected_riaz], axis=0)
    
    # Prepare clinical data for C-index calculation (extract survival info per cohort)
    def get_survival_data(clin_df, sig_index, os_time_col, os_status_col):
        """Extract survival times and events for aligned samples."""
        clin_aligned = clin_df.loc[sig_index].copy()
        os_time = pd.to_numeric(clin_aligned[os_time_col], errors='coerce').values
        os_status = pd.to_numeric(clin_aligned[os_status_col], errors='coerce').values
        return os_time, os_status
    
    # Liu 2019
    os_time_liu, os_status_liu = get_survival_data(clin_liu, sig_corrected_liu.index, 'OS_MONTHS', 'OS_STATUS')
    # Hugo 2016
    os_time_hugo, os_status_hugo = get_survival_data(clin_hugo, sig_corrected_hugo.index, 'os_months', 'os_status')
    # Riaz 2017
    os_time_riaz, os_status_riaz = get_survival_data(clin_riaz, sig_corrected_riaz.index, 'os_months', 'os_status')
    
    cohort_dfs = {
        'Liu 2019': (sig_corrected_liu, y_liu),
        'Hugo 2016': (sig_corrected_hugo, y_hugo),
        'Riaz 2017': (sig_corrected_riaz, y_riaz)
    }
    
    # Store survival data per cohort for C-index calculation
    cohort_survival = {
        'Liu 2019': {'os_time': os_time_liu, 'os_status': os_status_liu},
        'Hugo 2016': {'os_time': os_time_hugo, 'os_status': os_status_hugo},
        'Riaz 2017': {'os_time': os_time_riaz, 'os_status': os_status_riaz}
    }

    print("\n==================================================")
    print("Phase 4: LOCO Cross-Validation (Expression Signatures Only)...")
    print("==================================================")
    
    signature_cols = sig_corrected.columns.tolist()
    
    # Store all results for report generation
    all_loco_results = {}
    
    for model_type in ["lr", "rf", "xgb", "svm", "elasticnet"]:
        print(f"\nTraining and testing model: {model_type.upper()}")
        loco_results = run_loco_cv(cohort_dfs, signature_cols, model_type=model_type)
        all_loco_results[model_type] = loco_results
        
        # Print metrics table
        metrics_rows = []
        for cohort, res in loco_results.items():
            # Recalculate with extended metrics
            extended_metrics = calculate_extended_metrics(res['y_true'], res['y_pred_prob'])
            
            # Calculate C-index using survival data
            survival_data = cohort_survival[cohort]
            cindex = calculate_cindex(res['y_pred_prob'], survival_data['os_time'], survival_data['os_status'])
            extended_metrics['cindex'] = cindex
            
            res['metrics_extended'] = extended_metrics
            
            metrics_rows.append({
                'Test Cohort': cohort,
                'AUC': f"{extended_metrics['auc']:.3f}",
                'Accuracy': f"{extended_metrics['accuracy']:.3f}",
                'Sensitivity': f"{extended_metrics['sensitivity']:.3f}",
                'Specificity': f"{extended_metrics['specificity']:.3f}",
                'Precision': f"{extended_metrics['precision']:.3f}",
                'F1': f"{extended_metrics['f1']:.3f}",
                'C-Index': f"{extended_metrics['cindex']:.3f}" if not np.isnan(extended_metrics['cindex']) else "N/A"
            })
        print(pd.DataFrame(metrics_rows).to_string(index=False))
        
        # Save plots for the best performing model (e.g., Logistic Regression or XGBoost)
        # Let's save curves for all models
        plot_roc_curves(loco_results, model_type.upper(), PLOT_DIR / f"roc_curves_{model_type}.png")
        plot_pr_curves(loco_results, model_type.upper(), PLOT_DIR / f"pr_curves_{model_type}.png")
        plot_confusion_matrices(loco_results, model_type.upper(), PLOT_DIR / f"confusion_matrices_{model_type}.png")

    print("\n==================================================")
    print("Phase 5: Survival Analysis (Log-rank test)...")
    print("==================================================")
    
    def clean_os_status(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float)):
            if val in [1, 1.0]:
                return 1
            if val in [0, 0.0]:
                return 0
        if isinstance(val, str):
            val_upper = val.upper()
            if 'DECEASED' in val_upper or '1' in val_upper:
                return 1
            if 'LIVING' in val_upper or '0' in val_upper:
                return 0
        return np.nan

    # We will use Logistic Regression LOCO results for survival analysis
    loco_lr = run_loco_cv(cohort_dfs, signature_cols, model_type="lr")
    
    # 1. Hugo 2016 survival
    hugo_pred = loco_lr['Hugo 2016']['y_pred_prob']
    p_hugo = run_survival_analysis(
        clin_hugo.loc[sig_hugo.index], hugo_pred, 
        time_col='os_months', status_col='os_status', 
        save_path=PLOT_DIR / "survival_hugo_lr.png"
    )
    print(f"Hugo 2016 Overall Survival difference p-value: {p_hugo:.3e}" if p_hugo else "Hugo 2016: No survival data")

    # 2. Liu 2019 survival
    clin_liu_clean = clin_liu.loc[sig_liu.index].copy()
    clin_liu_clean['os_status_clean'] = clin_liu_clean['OS_STATUS'].apply(clean_os_status)
    liu_pred = loco_lr['Liu 2019']['y_pred_prob']
    p_liu = run_survival_analysis(
        clin_liu_clean, liu_pred,
        time_col='OS_MONTHS', status_col='os_status_clean',
        save_path=PLOT_DIR / "survival_liu_lr.png"
    )
    print(f"Liu 2019 Overall Survival difference p-value: {p_liu:.3e}" if p_liu else "Liu 2019: No survival data")

    # 3. Riaz 2017 survival
    riaz_pred = loco_lr['Riaz 2017']['y_pred_prob']
    p_riaz = run_survival_analysis(
        clin_riaz.loc[sig_riaz.index], riaz_pred, 
        time_col='os_months', status_col='os_status', 
        save_path=PLOT_DIR / "survival_riaz_lr.png"
    )
    print(f"Riaz 2017 Overall Survival difference p-value: {p_riaz:.3e}" if p_riaz else "Riaz 2017: No survival data")

    # 4. TCGA-SKCM survival validation
    try:
        print("Loading TCGA-SKCM data for survival validation...")
        tcga_dir = DATA_DIR / "processed" / "skcm_tcga_pan_can_atlas_2018"
        df_tcga_expr_raw = pd.read_csv(tcga_dir / "expr_cleaned.csv", index_col=0)
        df_tcga_clin = pd.read_csv(tcga_dir / "clin_cleaned.csv", index_col=0)
        
        print("Using pre-log-transformed, pre-filtered TCGA values...")
        df_tcga_expr_filtered = df_tcga_expr_raw
        
        # Keep common genes
        tcga_common_genes = df_tcga_expr_filtered.columns.intersection(common_genes)
        df_tcga_expr = df_tcga_expr_filtered[tcga_common_genes]
        df_tcga_expr = df_tcga_expr.reindex(columns=common_genes, fill_value=0)
        
        # Compute signatures (correctly log-transformed)
        sig_tcga = extract_all_signatures(df_tcga_expr)
        
        # Standardise sample IDs to 12-char patient IDs to align expression and clinical
        sig_tcga.index = sig_tcga.index.str.upper().str[:12]
        df_tcga_clin.index = df_tcga_clin.index.str.upper().str[:12]
        
        # Group by index and take first to deduplicate patients
        sig_tcga = sig_tcga.groupby(sig_tcga.index).first()
        df_tcga_clin = df_tcga_clin.groupby(df_tcga_clin.index).first()
        
        # Align patients
        common_tcga_patients = sig_tcga.index.intersection(df_tcga_clin.index)
        sig_tcga = sig_tcga.loc[common_tcga_patients]
        df_tcga_clin = df_tcga_clin.loc[common_tcga_patients]
        
        # Scale TCGA signatures using cohort-independent Z-score scaling
        sig_tcga_corrected = zscore_df(sig_tcga)
        
        # Train model on all clinical trials pooled
        X_train_full = sig_corrected
        y_train_full = pd.concat([y_liu, y_hugo, y_riaz], axis=0)
        
        # Scale features
        scaler_full = StandardScaler()
        X_train_full_scaled = scaler_full.fit_transform(X_train_full)
        sig_tcga_corrected_scaled = scaler_full.transform(sig_tcga_corrected)
        
        final_model = LogisticRegression(max_iter=1000, C=1.0)
        final_model.fit(X_train_full_scaled, y_train_full)
        
        # Predict on TCGA
        tcga_pred = final_model.predict_proba(sig_tcga_corrected_scaled)[:, 1]
        
        # Clean TCGA survival columns
        df_tcga_clin_clean = df_tcga_clin.copy()
        df_tcga_clin_clean['os_status_clean'] = df_tcga_clin_clean['OS_STATUS'].apply(clean_os_status)
        
        p_tcga = run_survival_analysis(
            df_tcga_clin_clean, tcga_pred,
            time_col='OS_MONTHS', status_col='os_status_clean',
            save_path=PLOT_DIR / "survival_tcga_lr.png"
        )
        print(f"TCGA-SKCM Overall Survival difference p-value: {p_tcga:.3e}" if p_tcga else "TCGA-SKCM: No survival data")
        
    except Exception as e:
        print(f"Skipping TCGA survival validation: {e}")


    print("\n==================================================")
    print("Phase 6: Combined Features Model (Liu + Hugo)...")
    print("==================================================")
    
    # Get mutation status
    mut_hugo = extract_driver_mutations(clin_hugo.loc[sig_hugo.index])
    mut_liu = extract_driver_mutations(clin_liu.loc[sig_liu.index])
    
    # Combine sig + mutations
    comb_liu = pd.concat([sig_corrected_liu, mut_liu], axis=1)
    comb_hugo = pd.concat([sig_corrected_hugo, mut_hugo], axis=1)
    
    cohort_dfs_comb = {
        'Liu 2019': (comb_liu, y_liu),
        'Hugo 2016': (comb_hugo, y_hugo)
    }
    
    comb_features = signature_cols + ['mut_BRAF', 'mut_NRAS', 'mut_NF1']
    
    print("\nTraining combined Expression + Mutation model (LOCO between Liu and Hugo):")
    for model_type in ["lr", "rf", "xgb", "svm", "elasticnet"]:
        print(f"\nCombined Model: {model_type.upper()}")
        loco_results_comb = run_loco_cv(cohort_dfs_comb, comb_features, model_type=model_type)
        metrics_rows = []
        for cohort, res in loco_results_comb.items():
            m = res['metrics']
            metrics_rows.append({
                'Test Cohort': cohort,
                'AUC': f"{m['auc']:.3f}",
                'Accuracy': f"{m['accuracy']:.3f}",
                'Precision': f"{m['precision']:.3f}",
                'Recall': f"{m['recall']:.3f}",
                'F1': f"{m['f1']:.3f}"
            })
        print(pd.DataFrame(metrics_rows).to_string(index=False))
        plot_roc_curves(loco_results_comb, f"COMBINED_{model_type.upper()}", PLOT_DIR / f"roc_curves_combined_{model_type}.png")

    # Phase 7: Save Best Models for Q5 Integration
    print("\n==================================================")
    print("Phase 7: Saving Best Models for Q5 Integration...")
    print("==================================================")
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    
    # Train final models on all pooled clinical trial data
    X_train_final = sig_corrected
    y_train_final = pd.concat([y_liu, y_hugo, y_riaz], axis=0)
    
    # Standard scale features
    scaler_final = StandardScaler()
    X_train_final_scaled = scaler_final.fit_transform(X_train_final)
    X_train_final_scaled = pd.DataFrame(X_train_final_scaled, columns=X_train_final.columns, index=X_train_final.index)
    
    rf_final = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf_final.fit(X_train_final_scaled, y_train_final)
    
    lr_final = LogisticRegression(max_iter=1000, C=1.0)
    lr_final.fit(X_train_final_scaled, y_train_final)
    
    svm_final = SVC(probability=True, random_state=42, C=1.0)
    svm_final.fit(X_train_final_scaled, y_train_final)
    
    elasticnet_final = LogisticRegression(solver='saga', l1_ratio=0.5, C=1.0, random_state=42, max_iter=2000)
    elasticnet_final.fit(X_train_final_scaled, y_train_final)
    
    with open(models_dir / "final_rf_model.pkl", "wb") as f:
        pickle.dump(rf_final, f)
    with open(models_dir / "final_lr_model.pkl", "wb") as f:
        pickle.dump(lr_final, f)
    with open(models_dir / "final_svm_model.pkl", "wb") as f:
        pickle.dump(svm_final, f)
    with open(models_dir / "final_elasticnet_model.pkl", "wb") as f:
        pickle.dump(elasticnet_final, f)
    with open(models_dir / "final_scaler.pkl", "wb") as f:
        pickle.dump(scaler_final, f)
        
    print(f"Saved final Random Forest, Logistic Regression, SVM, ElasticNet models and scaler to {models_dir}/")

    print("\n==================================================")
    print("Phase 7: Generating Model Evaluation Report...")
    print("==================================================")
    
    # Generate comprehensive evaluation report
    generate_model_evaluation_report(all_loco_results, REPORTS_DIR)

    print("\n==================================================")
    print("Done! All analysis runs completed successfully.")
    print("All evaluation plots saved to the 'plots/' directory.")
    print("All reports saved to the 'reports/' directory.")
    print("==================================================")

if __name__ == "__main__":
    main()
