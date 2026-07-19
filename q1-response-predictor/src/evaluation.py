import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

def plot_roc_curves(loco_results, model_name, save_path=None):
    """
    Plots ROC curves for all LOCO CV folds (one per test cohort).
    """
    plt.figure(figsize=(8, 6))
    
    for cohort, res in loco_results.items():
        y_true = res['y_true']
        y_pred_prob = res['y_pred_prob']
        
        # Check if we have both classes in y_true
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{cohort} (AUC = {roc_auc:.3f})', lw=2)
        else:
            print(f"Skipping ROC curve for {cohort} due to single class in true labels.")
            
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=2, label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'LOCO Cross-Cohort ROC Curves ({model_name})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_pr_curves(loco_results, model_name, save_path=None):
    """
    Plots Precision-Recall curves.
    """
    plt.figure(figsize=(8, 6))
    
    for cohort, res in loco_results.items():
        y_true = res['y_true']
        y_pred_prob = res['y_pred_prob']
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        avg_prec = average_precision_score(y_true, y_pred_prob)
        
        plt.plot(recall, precision, label=f'{cohort} (AP = {avg_prec:.3f})', lw=2)
            
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'LOCO Cross-Cohort PR Curves ({model_name})')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrices(loco_results, model_name, save_path=None):
    """
    Plots confusion matrices for all test cohorts in a grid.
    """
    n_cohorts = len(loco_results)
    fig, axes = plt.subplots(1, n_cohorts, figsize=(5*n_cohorts, 4))
    
    if n_cohorts == 1:
        axes = [axes]
    
    for idx, (cohort, res) in enumerate(loco_results.items()):
        y_true = res['y_true']
        y_pred_prob = res['y_pred_prob']
        y_pred_class = (y_pred_prob >= 0.5).astype(int)
        
        cm = confusion_matrix(y_true, y_pred_class)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
        axes[idx].set_title(f'{cohort}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_xticklabels(['Non-Resp', 'Responder'])
        axes[idx].set_yticklabels(['Non-Resp', 'Responder'])
    
    plt.suptitle(f'Confusion Matrices ({model_name})', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def calculate_extended_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    Calculates extended metrics including specificity at a fixed threshold.
    
    Returns:
        dict with keys: auc, accuracy, precision, recall (sensitivity), specificity, f1
    """
    y_pred_class = (y_pred_prob >= threshold).astype(int)
    
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    
    metrics = {
        'auc': roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else np.nan,
        'accuracy': accuracy_score(y_true, y_pred_class),
        'precision': precision_score(y_true, y_pred_class, zero_division=0),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1_score(y_true, y_pred_class, zero_division=0),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }
    return metrics

def calculate_cindex(y_pred_prob, duration, event_observed):
    """
    Calculates C-index (concordance index) for survival prediction.
    
    Parameters:
        y_pred_prob: predicted probabilities (higher = better predicted response/survival)
        duration: survival time in months
        event_observed: 1 if event occurred, 0 if censored
    
    Returns:
        float: C-index score (0.5 = random, 1.0 = perfect)
    """
    try:
        # Filter out samples with missing survival data
        valid_idx = (~np.isnan(duration)) & (~np.isnan(event_observed)) & (~np.isnan(y_pred_prob))
        
        if valid_idx.sum() < 2:
            return np.nan
        
        duration_valid = duration[valid_idx].astype(float)
        event_valid = event_observed[valid_idx].astype(bool)
        pred_valid = y_pred_prob[valid_idx].astype(float)
        
        # Calculate C-index
        cindex = concordance_index(duration_valid, -pred_valid, event_valid)
        
        return cindex
    except Exception as e:
        print(f"Warning: C-index calculation failed: {e}")
        return np.nan

def run_survival_analysis(df_clin, y_pred_prob, time_col='os_months', status_col='os_status', save_path=None):
    """
    Groups patients based on predicted response probability (high vs low) 
    and plots Kaplan-Meier survival curves.
    """
    df = df_clin.copy()
    
    # Check if time and status columns exist
    if time_col not in df.columns or status_col not in df.columns:
        print(f"Skipping survival analysis: columns {time_col} or {status_col} not found.")
        return
        
    df[time_col] = pd.to_numeric(df[time_col], errors='coerce').astype(float)
    df[status_col] = pd.to_numeric(df[status_col], errors='coerce').astype(float)
    
    # Add predicted probabilities
    df['y_pred_prob'] = y_pred_prob
    df = df.dropna(subset=[time_col, status_col, 'y_pred_prob'])
    
    if len(df) == 0:
        print("No samples with valid survival and prediction data.")
        return
        
    # Split into High Prob (predicted Responder) and Low Prob (predicted Non-responder)
    # Using 0.5 threshold or median of predictions
    threshold = 0.5
    high_prob = df[df['y_pred_prob'] >= threshold]
    low_prob = df[df['y_pred_prob'] < threshold]
    
    if len(high_prob) == 0 or len(low_prob) == 0:
        # Fallback to median split
        threshold = df['y_pred_prob'].median()
        high_prob = df[df['y_pred_prob'] >= threshold]
        low_prob = df[df['y_pred_prob'] < threshold]
        print(f"Using median probability threshold: {threshold:.3f}")
        
    if len(high_prob) == 0 or len(low_prob) == 0:
        print("Cannot split cohort into high and low probability groups (constant predictions). Skipping survival analysis.")
        return None
        
    # Fit KM curves
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    
    plt.figure(figsize=(8, 6))
    
    kmf_high.fit(high_prob[time_col], event_observed=high_prob[status_col], label=f'High predicted prob (N={len(high_prob)})')
    kmf_high.plot_survival_function(ci_show=True)
    
    kmf_low.fit(low_prob[time_col], event_observed=low_prob[status_col], label=f'Low predicted prob (N={len(low_prob)})')
    kmf_low.plot_survival_function(ci_show=True)
    
    # Calculate log-rank test
    results = logrank_test(
        high_prob[time_col], low_prob[time_col], 
        event_observed_A=high_prob[status_col], event_observed_B=low_prob[status_col]
    )
    p_value = results.p_value
    
    plt.title(f'Overall Survival by Predicted Response Probability (p = {p_value:.2e})')
    plt.xlabel('Survival Time (Months)')
    plt.ylabel('Survival Probability')
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
        
    return p_value
