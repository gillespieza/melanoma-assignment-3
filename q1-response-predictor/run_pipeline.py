import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set matplotlib backend to Agg to avoid GUI errors
import matplotlib
matplotlib.use('Agg')

from src.data_loaders import load_liu_2019, load_hugo_2016, load_riaz_2017
from src.signatures import extract_all_signatures
from src.models import run_loco_cv
from src.evaluation import plot_roc_curves, plot_pr_curves, run_survival_analysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

def parse_hugo_mutations(df_meta):
    """
    Returns BRAF, NRAS, NF1 mutation status from Hugo 2016 clinical metadata.
    Columns mut_BRAF, mut_NRAS, mut_NF1 are pre-parsed by clean_data.py from the MAF file.
    """
    df = df_meta.copy()
    for col in ['mut_BRAF', 'mut_NRAS', 'mut_NF1']:
        if col not in df.columns:
            df[col] = 0
    return df[['mut_BRAF', 'mut_NRAS', 'mut_NF1']]

def parse_liu_mutations(data_dir, patient_ids):
    """
    Parses BRAF, NRAS, NF1 mutation status from Liu 2019 data_mutations.txt.
    """
    mut_file = Path(data_dir) / "raw/liu_2019/data_mutations.txt"
    clin_sample_file = Path(data_dir) / "raw/liu_2019/data_clinical_sample.txt"
    
    # cBioPortal mutations file
    df_mut = pd.read_csv(mut_file, sep="\t")
    
    # We need to map Tumor_Sample_Barcode to PATIENT_ID
    df_sample = pd.read_csv(clin_sample_file, sep="\t", skiprows=4)
    sample_to_patient = df_sample.set_index("SAMPLE_ID")["PATIENT_ID"].to_dict()
    
    # Map mutations to Patient ID
    df_mut['patient_id'] = df_mut['Tumor_Sample_Barcode'].map(sample_to_patient)
    
    # Filter for non-silent mutations
    non_silent = [
        "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
        "Missense_Mutation", "Nonsense_Mutation", "Splice_Site",
        "Translation_Start_Site", "Nonstop_Mutation"
    ]
    df_mut = df_mut[df_mut['Variant_Classification'].isin(non_silent)]
    
    # Pivot to sample x gene mutation matrix
    df_mut_wide = df_mut.pivot_table(
        index='patient_id', 
        columns='Hugo_Symbol', 
        values='Entrez_Gene_Id', 
        aggfunc='count'
    ).fillna(0).astype(int)
    
    # Reindex to match patient_ids
    df_mut_wide = df_mut_wide.reindex(patient_ids, fill_value=0)
    
    # Extract BRAF, NRAS, NF1
    df_out = pd.DataFrame(index=patient_ids)
    df_out['mut_BRAF'] = (df_mut_wide['BRAF'] > 0).astype(int) if 'BRAF' in df_mut_wide.columns else 0
    df_out['mut_NRAS'] = (df_mut_wide['NRAS'] > 0).astype(int) if 'NRAS' in df_mut_wide.columns else 0
    df_out['mut_NF1'] = (df_mut_wide['NF1'] > 0).astype(int) if 'NF1' in df_mut_wide.columns else 0
    
    return df_out

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

    # Align labels
    y_liu = clin_liu.loc[sig_liu.index, 'response']
    y_hugo = clin_hugo.loc[sig_hugo.index, 'response']
    y_riaz = clin_riaz.loc[sig_riaz.index, 'response']

    print("\n==================================================")
    print("Phase 3: Batch Effect Correction (ComBat)...")
    print("==================================================")
    
    from pycombat import Combat
    
    # Concatenate signature DataFrames
    sig_all = pd.concat([sig_liu, sig_hugo, sig_riaz], axis=0)
    batches = (['liu'] * len(sig_liu)) + (['hugo'] * len(sig_hugo)) + (['riaz'] * len(sig_riaz))
    
    # Run ComBat (expects samples as rows, signatures/features as columns)
    sig_corrected_arr = Combat().fit_transform(sig_all.values, batches)
    sig_corrected = pd.DataFrame(sig_corrected_arr, index=sig_all.index, columns=sig_all.columns)
    
    # Split back into individual cohorts
    sig_corrected_liu = sig_corrected.iloc[:len(sig_liu)]
    sig_corrected_hugo = sig_corrected.iloc[len(sig_liu):len(sig_liu)+len(sig_hugo)]
    sig_corrected_riaz = sig_corrected.iloc[len(sig_liu)+len(sig_hugo):]
    
    cohort_dfs = {
        'Liu 2019': (sig_corrected_liu, y_liu),
        'Hugo 2016': (sig_corrected_hugo, y_hugo),
        'Riaz 2017': (sig_corrected_riaz, y_riaz)
    }

    print("\n==================================================")
    print("Phase 4: LOCO Cross-Validation (Expression Signatures Only)...")
    print("==================================================")
    
    signature_cols = sig_corrected.columns.tolist()
    
    for model_type in ["lr", "rf", "xgb", "svm", "elasticnet"]:
        print(f"\nTraining and testing model: {model_type.upper()}")
        loco_results = run_loco_cv(cohort_dfs, signature_cols, model_type=model_type)
        
        # Print metrics table
        metrics_rows = []
        for cohort, res in loco_results.items():
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
        
        # Save plots for the best performing model (e.g., Logistic Regression or XGBoost)
        # Let's save curves for all models
        plot_roc_curves(loco_results, model_type.upper(), PLOT_DIR / f"roc_curves_{model_type}.png")
        plot_pr_curves(loco_results, model_type.upper(), PLOT_DIR / f"pr_curves_{model_type}.png")

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
        
        print("Log2-transforming and filtering genes by variance first...")
        df_tcga_log = np.log2(df_tcga_expr_raw + 1)
        variances = df_tcga_log.var()
        # Keep top 15% high-variance genes (approx 3000 genes)
        var_cutoff = variances.quantile(0.85)
        high_var_entrez = variances[variances >= var_cutoff].index.tolist()
        df_tcga_expr_filtered = df_tcga_log[high_var_entrez]
        print(f"Kept {len(high_var_entrez)} high-variance genes out of {df_tcga_expr_raw.shape[1]}")
        
        print("Mapping TCGA Entrez IDs to Hugo Symbols...")
        import urllib.request
        import json
        
        cache_file = tcga_dir / "entrez_to_symbol_cache.json"
        entrez_mapping = {}
        
        if cache_file.exists():
            print(f"Loading mapping cache from: {cache_file}")
            with open(cache_file, "r") as f:
                entrez_mapping = json.load(f)
        else:
            print("Mapping high-variance Entrez IDs via MyGene.info API...")
            chunk_size = 1000
            for i in range(0, len(high_var_entrez), chunk_size):
                chunk = [str(x) for x in high_var_entrez[i:i+chunk_size]]
                url = 'https://mygene.info/v3/query'
                q_str = ','.join(chunk)
                data = f'q={q_str}&scopes=entrezgene&fields=symbol&species=human'.encode('utf-8')
                req = urllib.request.Request(
                    url, 
                    data=data, 
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                )
                try:
                    with urllib.request.urlopen(req, timeout=30) as response:
                        res = json.loads(response.read().decode('utf-8'))
                        for item in res:
                            q = item.get('query')
                            sym = item.get('symbol')
                            if q and sym:
                                entrez_mapping[q] = sym
                except Exception as e:
                    print(f"Error mapping Entrez chunk {i}: {e}")
            
            # Save cache
            print(f"Saving mapping cache to: {cache_file}")
            with open(cache_file, "w") as f:
                json.dump(entrez_mapping, f)
                
        # Rename columns to symbols and take the average of duplicates (using non-deprecated groupby)
        mapped_columns = [entrez_mapping.get(str(col), str(col)) for col in df_tcga_expr_filtered.columns]
        df_tcga_expr_filtered.columns = mapped_columns
        df_tcga_expr_mapped = df_tcga_expr_filtered.T.groupby(level=0).mean().T
        
        # Keep common genes
        tcga_common_genes = df_tcga_expr_mapped.columns.intersection(common_genes)
        df_tcga_expr = df_tcga_expr_mapped[tcga_common_genes]
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
        
        # Batch-correct TCGA signatures pooled with Liu, Hugo, Riaz
        sig_all_tcga = pd.concat([sig_liu, sig_hugo, sig_riaz, sig_tcga], axis=0)
        batches_tcga = (['liu'] * len(sig_liu)) + (['hugo'] * len(sig_hugo)) + (['riaz'] * len(sig_riaz)) + (['tcga'] * len(sig_tcga))
        sig_tcga_corrected_arr = Combat().fit_transform(sig_all_tcga.values, batches_tcga)
        sig_tcga_corrected_df = pd.DataFrame(sig_tcga_corrected_arr, index=sig_all_tcga.index, columns=sig_all_tcga.columns)
        sig_tcga_corrected = sig_tcga_corrected_df.iloc[len(sig_liu)+len(sig_hugo)+len(sig_riaz):]
        
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
    mut_hugo = parse_hugo_mutations(clin_hugo.loc[sig_hugo.index])
    # Liu patient clinical has PATIENT_ID, we need to map SAMPLE_ID to PATIENT_ID
    # In Liu loader, we merged on PATIENT_ID but kept SAMPLE_ID as index of df_expr (sig_corrected_liu)
    # Let's get the patient IDs for sig_corrected_liu
    liu_patients = clin_liu.loc[sig_liu.index, 'PATIENT_ID']
    mut_liu = parse_liu_mutations(DATA_DIR, liu_patients)
    # Reset index to match SAMPLE_ID
    mut_liu.index = sig_liu.index
    
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
    
    elasticnet_final = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, random_state=42, max_iter=2000)
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
    print("Done! All analysis runs completed successfully.")
    print("All evaluation plots saved to the 'plots/' directory.")
    print("==================================================")

if __name__ == "__main__":
    main()
