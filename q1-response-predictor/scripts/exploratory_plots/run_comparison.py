import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from pycombat import Combat

# Add project root to sys.path for importing src modules
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_loaders import load_liu_2019, load_hugo_2016, load_riaz_2017
from src.signatures import extract_all_signatures

# Paths
DATA_DIR = BASE_DIR / "data"

def run_loco_signatures(cohort_dfs, model_type="rf"):
    """
    Runs LOCO CV using the 6 immune signatures.
    """
    cohorts = list(cohort_dfs.keys())
    results = {}
    
    for test_cohort in cohorts:
        train_cohorts = [c for c in cohorts if c != test_cohort]
        
        # Concatenate train cohorts
        X_train = pd.concat([cohort_dfs[c][0] for c in train_cohorts], axis=0)
        y_train = pd.concat([cohort_dfs[c][1] for c in train_cohorts], axis=0)
        
        X_test = cohort_dfs[test_cohort][0]
        y_test = cohort_dfs[test_cohort][1]
        
        # Train model
        if model_type == "lr":
            model = LogisticRegression(max_iter=1000, C=1.0)
        elif model_type == "rf":
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif model_type == "xgb":
            model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            
        model.fit(X_train, y_train)
        
        # Predict
        y_prob = model.predict_proba(X_test)[:, 1]
        
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = np.nan
            
        results[test_cohort] = auc
        
    return results

def run_loco_feature_selection(cohort_expr_dfs, cohort_y_dfs, k_features=20, model_type="rf"):
    """
    Runs LOCO CV with feature selection on raw expression data:
    1. Filter out low-variance genes on training cohorts.
    2. Select top k_features using ANOVA F-value on training cohorts.
    3. Train and test on selected genes.
    """
    cohorts = list(cohort_expr_dfs.keys())
    results = {}
    
    for test_cohort in cohorts:
        train_cohorts = [c for c in cohorts if c != test_cohort]
        
        # Concatenate train cohorts
        X_train_raw = pd.concat([cohort_expr_dfs[c] for c in train_cohorts], axis=0)
        y_train = pd.concat([cohort_y_dfs[c] for c in train_cohorts], axis=0)
        
        X_test_raw = cohort_expr_dfs[test_cohort]
        y_test = cohort_y_dfs[test_cohort]
        
        # 1. Remove low-variance genes (keep top 1000 most variable genes in train data)
        train_vars = X_train_raw.var(axis=0)
        top_var_genes = train_vars.nlargest(1000).index
        X_train_filtered = X_train_raw[top_var_genes]
        X_test_filtered = X_test_raw[top_var_genes]
        
        # 2. Select top K genes using ANOVA F-value (f_classif)
        selector = SelectKBest(score_func=f_classif, k=k_features)
        selector.fit(X_train_filtered, y_train)
        
        selected_genes = top_var_genes[selector.get_support()]
        
        X_train_sel = X_train_filtered[selected_genes]
        X_test_sel = X_test_filtered[selected_genes]
        
        # Train model
        if model_type == "lr":
            model = LogisticRegression(max_iter=1000, C=1.0)
        elif model_type == "rf":
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif model_type == "xgb":
            model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
            
        model.fit(X_train_sel, y_train)
        
        # Predict
        y_prob = model.predict_proba(X_test_sel)[:, 1]
        
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = np.nan
            
        results[test_cohort] = (auc, list(selected_genes))
        
    return results

def main():
    print("==================================================")
    print("Loading datasets for feature selection comparison...")
    print("==================================================")
    
    expr_liu, clin_liu = load_liu_2019(DATA_DIR)
    expr_hugo, clin_hugo = load_hugo_2016(DATA_DIR)
    expr_riaz, clin_riaz = load_riaz_2017(DATA_DIR)
    
    # Intersect genes
    common_genes = expr_liu.columns.intersection(expr_hugo.columns).intersection(expr_riaz.columns)
    print(f"Intersection genes: {len(common_genes)}")
    
    # Filter
    expr_liu = expr_liu[common_genes]
    expr_hugo = expr_hugo[common_genes]
    expr_riaz = expr_riaz[common_genes]
    
    y_liu = clin_liu.loc[expr_liu.index, 'response']
    y_hugo = clin_hugo.loc[expr_hugo.index, 'response']
    y_riaz = clin_riaz.loc[expr_riaz.index, 'response']
    
    # Concatenate all cohorts for batches
    batches = (['liu'] * len(expr_liu)) + (['hugo'] * len(expr_hugo)) + (['riaz'] * len(expr_riaz))
    
    print("\n==================================================")
    print("Correcting Batch Effects on raw genes for comparison (ComBat)...")
    print("==================================================")
    expr_all = pd.concat([expr_liu, expr_hugo, expr_riaz], axis=0)
    expr_corrected_arr = Combat().fit_transform(expr_all.values, batches)
    expr_corrected = pd.DataFrame(expr_corrected_arr, index=expr_all.index, columns=expr_all.columns)
    
    expr_corrected_liu = expr_corrected.iloc[:len(expr_liu)]
    expr_corrected_hugo = expr_corrected.iloc[len(expr_liu):len(expr_liu)+len(expr_hugo)]
    expr_corrected_riaz = expr_corrected.iloc[len(expr_liu)+len(expr_hugo):]
    
    cohort_expr_dfs = {
        'Liu 2019': expr_corrected_liu,
        'Hugo 2016': expr_corrected_hugo,
        'Riaz 2017': expr_corrected_riaz
    }
    cohort_y_dfs = {
        'Liu 2019': y_liu,
        'Hugo 2016': y_hugo,
        'Riaz 2017': y_riaz
    }
    
    print("\n==================================================")
    print("Computing signatures on raw data and correcting signatures...")
    print("==================================================")
    # Compute signatures first (Order A - stable and robust)
    sig_raw_liu = extract_all_signatures(expr_liu)
    sig_raw_hugo = extract_all_signatures(expr_hugo)
    sig_raw_riaz = extract_all_signatures(expr_riaz)
    
    sig_all = pd.concat([sig_raw_liu, sig_raw_hugo, sig_raw_riaz], axis=0)
    sig_corrected_arr = Combat().fit_transform(sig_all.values, batches)
    sig_corrected = pd.DataFrame(sig_corrected_arr, index=sig_all.index, columns=sig_all.columns)
    
    sig_corrected_liu = sig_corrected.iloc[:len(expr_liu)]
    sig_corrected_hugo = sig_corrected.iloc[len(expr_liu):len(expr_liu)+len(expr_hugo)]
    sig_corrected_riaz = sig_corrected.iloc[len(expr_liu)+len(expr_hugo):]
    
    cohort_sig_dfs = {
        'Liu 2019': (sig_corrected_liu, y_liu),
        'Hugo 2016': (sig_corrected_hugo, y_hugo),
        'Riaz 2017': (sig_corrected_riaz, y_riaz)
    }
    
    print("\n==================================================")
    print("Running LOCO Comparison (Signatures vs. Raw Selection)...")
    print("==================================================")
    
    models = ["lr", "rf", "xgb"]
    comparison_rows = []
    
    for m in models:
        print(f"\n--- Model: {m.upper()} ---")
        sig_aucs = run_loco_signatures(cohort_sig_dfs, model_type=m)
        raw_results = run_loco_feature_selection(cohort_expr_dfs, cohort_y_dfs, k_features=20, model_type=m)
        
        for cohort in cohort_expr_dfs.keys():
            sig_auc = sig_aucs[cohort]
            raw_auc, selected_genes = raw_results[cohort]
            comparison_rows.append({
                'Model': m.upper(),
                'Test Cohort': cohort,
                'Signature AUC': f"{sig_auc:.3f}" if not pd.isna(sig_auc) else "NaN",
                'Raw Selection AUC': f"{raw_auc:.3f}" if not pd.isna(raw_auc) else "NaN",
                'Top 5 Selected Genes': ", ".join(selected_genes[:5])
            })
            
    df_compare = pd.DataFrame(comparison_rows)
    print("\nComparison Results:")
    print(df_compare.to_string(index=False))
    
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True, parents=True)
    out_file = reports_dir / "feature_selection_comparison.md"
    df_compare.to_markdown(out_file, index=False)
    print(f"\nSaved comparison results to '{out_file}'")

if __name__ == "__main__":
    main()
