from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

from src.styles import RESPONSE_PALETTE, set_presentation_style
from src.utils.plotting import save_fig

set_presentation_style()

def plot_roc_curves(loco_results, model_name, save_path=None):
    """
    Plots ROC curves for all LOCO CV folds (one per test cohort).
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    
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
    
    if save_path:
        save_fig(fig, save_path)
    else:
        plt.show()

def plot_pr_curves(loco_results, model_name, save_path=None):
    """
    Plots Precision-Recall curves.
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    
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
    
    if save_path:
        save_fig(fig, save_path)
    else:
        plt.show()

def plot_confusion_matrices(loco_results, model_name, save_path=None, use_optimal_threshold=False):
    """
    Plots confusion matrices for all test cohorts in a grid.
    """
    n_cohorts = len(loco_results)
    fig, axes = plt.subplots(1, n_cohorts, figsize=(12, 4.5))
    
    if n_cohorts == 1:
        axes = [axes]
    
    for idx, (cohort, res) in enumerate(loco_results.items()):
        y_true = res['y_true']
        y_pred_prob = res['y_pred_prob']
        if use_optimal_threshold:
            threshold = find_optimal_threshold(y_true, y_pred_prob)
            title_suffix = f" (t={threshold:.2f})"
        else:
            threshold = 0.5
            title_suffix = ""
            
        y_pred_class = (y_pred_prob >= threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred_class)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
        axes[idx].set_title(f'{cohort}{title_suffix}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_xticklabels(['Non-Resp', 'Responder'])
        axes[idx].set_yticklabels(['Non-Resp', 'Responder'])
    
    thresh_label = "Optimal Youden's J Threshold" if use_optimal_threshold else "Default Threshold 0.5"
    plt.suptitle(f'Confusion Matrices ({model_name}) - {thresh_label}', fontsize=14, y=1.02)
    
    if save_path:
        save_fig(fig, save_path)
        print(f"Saved confusion matrix plot to {save_path}")
    else:
        plt.show()

def find_optimal_threshold(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """Finds the decision threshold that maximises Youden's J statistic.

    Youden's J = sensitivity + specificity - 1, equivalent to the point
    on the ROC curve furthest from the chance diagonal.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_pred_prob: Predicted probabilities for the positive class.

    Returns:
        Optimal threshold (float). Falls back to 0.5 if both classes are
        not present in y_true.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    j_scores = tpr - fpr  # Youden's J = TPR - FPR = sensitivity + specificity - 1
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def compute_expected_calibration_error(y_true: np.ndarray, y_pred_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) for binary classification.
    ECE measures the difference between expected accuracy and actual empirical accuracy across confidence bins.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find predictions in bin
        if i == n_bins - 1:
            in_bin = (y_pred_prob >= bin_lower) & (y_pred_prob <= bin_upper)
        else:
            in_bin = (y_pred_prob >= bin_lower) & (y_pred_prob < bin_upper)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_pred_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def plot_calibration_curves(loco_results, model_name, save_path=None):
    """
    Plots calibration curves (reliability diagrams) for all LOCO CV folds.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    for cohort, res in loco_results.items():
        y_true = res['y_true']
        y_pred_prob = res['y_pred_prob']

        if len(np.unique(y_true)) > 1:
            prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=5)
            ece = compute_expected_calibration_error(y_true, y_pred_prob, n_bins=5)
            brier = brier_score_loss(y_true, y_pred_prob)
            plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f'{cohort} (ECE = {ece:.3f}, Brier = {brier:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Perfectly Calibrated')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (Empirical Response Rate)')
    plt.title(f'LOCO Calibration Curves / Reliability Diagrams ({model_name})')
    plt.legend(loc="upper left")

    if save_path:
        save_fig(fig, save_path)
    else:
        plt.show()


def calculate_extended_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    Calculates extended metrics including specificity, Brier score, and ECE at a given threshold.

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_pred_prob: Predicted probabilities for the positive class.
        threshold: Decision threshold for converting probabilities to
            class labels. Defaults to 0.5.

    Returns:
        dict with keys: auc, accuracy, precision, sensitivity, specificity,
        f1, brier_score, ece, tp, tn, fp, fn, threshold.
    """
    y_pred_class = (y_pred_prob >= threshold).astype(int)

    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()

    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    brier = brier_score_loss(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else np.nan
    ece = compute_expected_calibration_error(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else np.nan

    metrics = {
        'auc': roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else np.nan,
        'accuracy': accuracy_score(y_true, y_pred_class),
        'precision': precision_score(y_true, y_pred_class, zero_division=0),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1_score(y_true, y_pred_class, zero_division=0),
        'brier_score': brier,
        'ece': ece,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'threshold': threshold,
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
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
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
    plt.legend(loc="lower left")
    
    if save_path:
        save_fig(fig, save_path)
    else:
        plt.show()
        
    return p_value


def plot_survival_2x2_grid(cohort_survival_data, save_path=None):
    """
    Plots a 2x2 grid of Kaplan-Meier survival curves for the 4 cohorts (Hugo, Liu, Riaz, TCGA-SKCM).
    
    Args:
        cohort_survival_data: list of dicts with keys:
            'cohort', 'df_clin', 'y_pred_prob', 'time_col', 'status_col', 'model_name'
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, item in enumerate(cohort_survival_data):
        ax = axes[idx]
        cohort_name = item['cohort']
        df = item['df_clin'].copy()
        time_col = item['time_col']
        status_col = item['status_col']
        y_pred_prob = item['y_pred_prob']

        df[time_col] = pd.to_numeric(df[time_col], errors='coerce').astype(float)
        df[status_col] = pd.to_numeric(df[status_col], errors='coerce').astype(float)
        df['y_pred_prob'] = y_pred_prob
        df = df.dropna(subset=[time_col, status_col, 'y_pred_prob'])

        threshold = 0.5
        high_prob = df[df['y_pred_prob'] >= threshold]
        low_prob = df[df['y_pred_prob'] < threshold]

        if len(high_prob) == 0 or len(low_prob) == 0:
            threshold = df['y_pred_prob'].median()
            high_prob = df[df['y_pred_prob'] >= threshold]
            low_prob = df[df['y_pred_prob'] < threshold]

        if len(high_prob) > 0 and len(low_prob) > 0:
            kmf_high = KaplanMeierFitter()
            kmf_low = KaplanMeierFitter()

            kmf_high.fit(high_prob[time_col], event_observed=high_prob[status_col], label=f'High Prob (N={len(high_prob)})')
            kmf_high.plot_survival_function(ax=ax, ci_show=True, color=RESPONSE_PALETTE['CR/PR'], lw=2)

            kmf_low.fit(low_prob[time_col], event_observed=low_prob[status_col], label=f'Low Prob (N={len(low_prob)})')
            kmf_low.plot_survival_function(ax=ax, ci_show=True, color=RESPONSE_PALETTE['PD'], lw=2)

            results = logrank_test(
                high_prob[time_col], low_prob[time_col],
                event_observed_A=high_prob[status_col], event_observed_B=low_prob[status_col]
            )
            p_val = results.p_value
            ax.set_title(f"{cohort_name} ({item.get('model_name', 'Model').upper()}, p = {p_val:.3f})", fontsize=12, fontweight='bold')
        else:
            ax.set_title(f"{cohort_name} (Constant Predictions)", fontsize=12, fontweight='bold')

        ax.set_xlabel('Survival Time (Months)', fontweight='bold')
        ax.set_ylabel('Overall Survival Probability', fontweight='bold')
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", color="#E5E7EB", alpha=0.6, linewidth=0.8)
        ax.legend(loc="lower left")

    plt.suptitle("Overall Survival Stratification Across Cohorts by Model Predictions", fontsize=15, fontweight='bold', y=0.98)

    if save_path:
        save_path = Path(save_path)
        save_fig(fig, save_path)
        # Mirror to root or subproject directory if path contains plots/models
        if "q1-response-predictor" in str(save_path):
            alt_path = Path(str(save_path).replace("q1-response-predictor/", "").replace("q1-response-predictor\\", ""))
            if alt_path != save_path:
                alt_path.parent.mkdir(parents=True, exist_ok=True)
                save_fig(fig, alt_path)
        else:
            alt_path = save_path.parent.parent / "q1-response-predictor" / "plots" / "models" / save_path.name
            if alt_path != save_path:
                alt_path.parent.mkdir(parents=True, exist_ok=True)
                save_fig(fig, alt_path)

