import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from scipy.stats import entropy


# def ensure_numpy(tensor_or_array: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
#     """Convert input to numpy array with proper type handling."""
#     if isinstance(tensor_or_array, torch.Tensor):
#         return tensor_or_array.detach().cpu().numpy()
#     return np.array(tensor_or_array)

# def ensure_numpy_scalar(tensor_or_array: Union[torch.Tensor, np.ndarray]) -> float:
#     """Convert input to a scalar float value with proper handling of multi-dimensional inputs."""
#     if isinstance(tensor_or_array, torch.Tensor):
#         tensor_or_array = tensor_or_array.detach().cpu().numpy()
    
#     array = np.array(tensor_or_array)
#     if array.size > 1:
#         return float(np.mean(array))
#     return float(array.item())


# In metrics.py

def calculate_predictive_entropy(y_pred: torch.Tensor) -> float:
    """Calculate predictive entropy for binary classification probabilities.
    
    This function is specifically designed to handle binary classification outputs
    where we have a tensor of shape [batch_size, 2] containing class probabilities.
    It computes the entropy in a numerically stable way while staying in PyTorch
    for as long as possible to avoid problematic tensor-numpy conversions.
    
    Args:
        y_pred: Probability tensor from softmax output, shape [batch_size, 2]
               Contains probabilities for both classes that sum to 1
        
    Returns:
        float: Average entropy across the batch
    """
    try:
        # Stay in PyTorch as long as possible
        eps = torch.finfo(y_pred.dtype).eps
        
        # Get probabilities for both classes
        p0 = y_pred[:, 0].clamp(eps, 1 - eps)
        p1 = y_pred[:, 1].clamp(eps, 1 - eps)
        
        # Calculate entropy using PyTorch operations
        entropy = -(p0 * torch.log(p0) + p1 * torch.log(p1))
        
        # Return mean entropy as a Python float
        return float(entropy.mean().item())
        
    except Exception as e:
        print(f"\nError in entropy calculation:")
        print(f"- Error type: {type(e).__name__}")
        print(f"- Message: {str(e)}")
        print(f"- Input tensor shape: {y_pred.shape}")
        print(f"- Input tensor dtype: {y_pred.dtype}")
        return 0.0

def process_model_outputs(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    additional_outputs: Optional[Dict] = None
) -> Dict[str, float]:
    """Process model outputs with robust type handling and error recovery."""
    try:
        # Ensure outputs and labels are proper tensors and on CPU
        outputs = outputs.detach().cpu()
        labels = labels.detach().cpu()
        
        # Convert to numpy arrays for metric calculation
        outputs_np = outputs.numpy()
        labels_np = labels.numpy()
        
        # Calculate base metrics with error handling
        try:
            metrics = calculate_metrics(labels_np, outputs_np)
        except Exception as e:
            print(f"Warning: Error in base metrics calculation: {str(e)}")
            # Provide fallback metrics
            metrics = {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'auc_roc': 0.0,
                'average_precision': 0.0
            }

        # Process additional outputs if available
        if additional_outputs:
            for key in ['consistency_loss', 'uncertainty', 'sum_constraint', 'non_negativity']:
                if key in additional_outputs:
                    value = additional_outputs[key]
                    metrics[key] = float(value) if isinstance(value, (torch.Tensor, np.ndarray)) else float(value)

        return metrics
    except Exception as e:
        print(f"Warning: Error in process_model_outputs: {str(e)}")
        # Return default metrics to prevent pipeline failure
        return {
            'accuracy': 0.0,
            'f1_score': 0.0,
            'auc_roc': 0.0,
            'average_precision': 0.0
        }


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate metrics with robust type handling and error recovery."""
    try:
        # Ensure arrays are the right type
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        
        # Create binary predictions
        y_pred_binary = (y_pred >= threshold).astype(np.float32)
        
        # Calculate metrics with error handling
        metrics = {}
        
        try:
            metrics['accuracy'] = float(calculate_accuracy(y_true, y_pred_binary))
        except:
            metrics['accuracy'] = 0.0
            
        try:
            metrics['precision'] = float(calculate_precision(y_true, y_pred_binary))
        except:
            metrics['precision'] = 0.0
            
        try:
            metrics['recall'] = float(calculate_recall(y_true, y_pred_binary))
        except:
            metrics['recall'] = 0.0
            
        try:
            metrics['f1_score'] = float(calculate_f1_score(y_true, y_pred_binary))
        except:
            metrics['f1_score'] = 0.0
            
        try:
            metrics['auc_roc'] = float(roc_auc_score(y_true, y_pred))
        except:
            metrics['auc_roc'] = 0.0
            
        try:
            metrics['average_precision'] = float(average_precision_score(y_true, y_pred))
        except:
            metrics['average_precision'] = 0.0
        
        return metrics
        
    except Exception as e:
        print(f"Error in calculate_metrics: {str(e)}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'auc_roc': 0.0,
            'average_precision': 0.0
        }


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


