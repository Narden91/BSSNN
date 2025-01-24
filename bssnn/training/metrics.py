import torch
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from scipy.stats import entropy


def calculate_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for BSSNN predictions.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        threshold: Classification threshold for binary predictions
        
    Returns:
        Dictionary containing various performance metrics
    """
    # Convert tensors to numpy for metric calculation
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Binary predictions using threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate basic classification metrics
    metrics = {
        'accuracy': calculate_accuracy(y_true, y_pred_binary),
        'precision': calculate_precision(y_true, y_pred_binary),
        'recall': calculate_recall(y_true, y_pred_binary),
        'f1_score': calculate_f1_score(y_true, y_pred_binary),
        'auc_roc': roc_auc_score(y_true, y_pred),
        'average_precision': average_precision_score(y_true, y_pred)
    }
    
    # Calculate Bayesian-specific metrics
    metrics.update({
        'calibration_error': calculate_calibration_error(y_true, y_pred),
        'expected_calibration_error': calculate_expected_calibration_error(y_true, y_pred),
        'predictive_entropy': calculate_predictive_entropy(y_pred)
    })
    
    # Calculate optimal threshold using PR curve
    metrics['optimal_threshold'] = find_optimal_threshold(y_true, y_pred)
    
    return metrics


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate classification accuracy."""
    return np.mean(y_true == y_pred)


def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate precision score."""
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives > 0 else 0.0


def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate recall score."""
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0.0


def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate F1 score."""
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def calculate_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> float:
    """Calculate calibration error using binning approach.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        n_bins: Number of bins for calibration calculation
        
    Returns:
        Calibration error score
    """
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_pred, bins) - 1
    
    bin_sums = np.bincount(binids, weights=y_pred, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    return np.mean(np.abs(prob_true - prob_pred))


def calculate_expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> float:
    """Calculate expected calibration error.
    
    This metric measures the expected difference between predicted probabilities
    and empirical frequencies, weighted by the number of samples in each bin.
    """
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)
    binids = np.digitize(y_pred, bins) - 1
    
    bin_sums = np.bincount(binids, weights=y_pred, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))
    
    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    
    weights = bin_total[nonzero] / np.sum(bin_total)
    return np.sum(np.abs(prob_true - prob_pred) * weights)


def calculate_predictive_entropy(y_pred: np.ndarray) -> float:
    """Calculate predictive entropy of the model's predictions.
    
    This metric quantifies the model's uncertainty in its predictions.
    Higher values indicate more uncertainty.
    """
    # Ensure predictions are properly bounded
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    # Calculate entropy for binary predictions
    return -np.mean(y_pred * np.log(y_pred) + (1 - y_pred) * np.log(1 - y_pred))


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 1.0
) -> float:
    """Find optimal classification threshold using F-beta score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        beta: Beta parameter for F-beta score calculation
        
    Returns:
        Optimal threshold value
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    # Calculate F-beta score for each threshold
    f_scores = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-7)
    
    # Find threshold that maximizes F-beta score
    optimal_idx = np.argmax(f_scores)
    return thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5


def calculate_uncertainty_metrics(
    predictions: np.ndarray,
    n_samples: int = 10
) -> Dict[str, float]:
    """Calculate uncertainty metrics for ensemble predictions.
    
    Args:
        predictions: Array of shape (n_samples, n_instances) containing
                   multiple predictions for each instance
        n_samples: Number of Monte Carlo samples used
        
    Returns:
        Dictionary containing uncertainty metrics
    """
    # Calculate mean and variance of predictions
    mean_pred = np.mean(predictions, axis=0)
    var_pred = np.var(predictions, axis=0)
    
    # Calculate epistemic and aleatoric uncertainty
    epistemic = np.mean(var_pred)  # Between-sample variance
    aleatoric = np.mean(mean_pred * (1 - mean_pred))  # Average predictive variance
    
    return {
        'epistemic_uncertainty': epistemic,
        'aleatoric_uncertainty': aleatoric,
        'total_uncertainty': epistemic + aleatoric
    }