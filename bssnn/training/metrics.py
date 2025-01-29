import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from scipy.stats import entropy


def ensure_numpy(tensor_or_array: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert input to numpy array with proper type handling."""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    return np.array(tensor_or_array)

def ensure_numpy_scalar(tensor_or_array: Union[torch.Tensor, np.ndarray]) -> float:
    """Convert input to a scalar float value with proper handling of multi-dimensional inputs."""
    if isinstance(tensor_or_array, torch.Tensor):
        tensor_or_array = tensor_or_array.detach().cpu().numpy()
    
    array = np.array(tensor_or_array)
    if array.size > 1:
        return float(np.mean(array))
    return float(array.item())

def calculate_metrics(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    additional_outputs: Optional[Dict] = None
) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics with enhanced type handling."""
    # Convert inputs to numpy arrays
    y_true_np = ensure_numpy(y_true)
    y_pred_np = ensure_numpy(y_pred)
    
    # Ensure arrays are floating point type before comparison
    y_pred_np = y_pred_np.astype(np.float64)
    threshold = 0.5
    y_pred_binary = np.where(y_pred_np >= threshold, 1, 0)
    
    # Calculate metrics
    metrics = {
        'accuracy': calculate_accuracy(y_true_np, y_pred_binary),
        'precision': calculate_precision(y_true_np, y_pred_binary),
        'recall': calculate_recall(y_true_np, y_pred_binary),
        'f1_score': calculate_f1_score(y_true_np, y_pred_binary),
        'auc_roc': roc_auc_score(y_true_np, y_pred_np),
        'average_precision': average_precision_score(y_true_np, y_pred_np)
    }
    
    # Handle additional outputs
    if additional_outputs:
        for key, value in additional_outputs.items():
            try:
                metrics[key] = ensure_numpy_scalar(value)
            except Exception as e:
                print(f"Warning: Could not convert {key} to scalar. Error: {str(e)}")
                metrics[key] = 0.0
    
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


def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray, handle_undefined: str = 'zero') -> float:
    """Calculate F1 score."""
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    if (precision + recall) == 0:
        if handle_undefined == 'nan':
            return np.nan
        return 0.0
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
    eps = np.finfo(y_pred.dtype).eps  
    y_pred = np.clip(y_pred, eps, 1 - eps)
    
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
    
    # Remove last elements to align with thresholds
    precision = precision[:-1]
    recall = recall[:-1]
    
    # Calculate F-scores with aligned arrays
    f_scores = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-7)
    optimal_idx = np.argmax(f_scores)
    return thresholds[optimal_idx]


