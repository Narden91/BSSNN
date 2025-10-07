from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize
import torch


def ensure_numpy(arr: Union[np.ndarray, List]) -> np.ndarray:
    """Convert input to numpy array with proper type handling."""
    if isinstance(arr, np.ndarray):
        return arr
    return np.array(arr)


def calculate_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate metrics for binary classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted class labels
        y_prob: Predicted probabilities (optional)
        threshold: Classification threshold
    
    Returns:
        Dictionary of binary classification metrics
    """
    metrics = {
        'accuracy': float(np.mean(y_true == y_pred)),
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1_score': float(f1_score(y_true, y_pred))
    }
    
    if y_prob is not None:
        metrics.update({
            'auc_roc': float(roc_auc_score(y_true, y_prob)),
            'average_precision': float(average_precision_score(y_true, y_prob)),
            'brier_score': float(calculate_brier_score(y_true, y_prob)),
            'ece': float(calculate_ece(y_true, y_prob)),
            'mce': float(calculate_mce(y_true, y_prob))
        })
    
    return metrics


def calculate_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = None
) -> Dict[str, float]:
    """Calculate metrics for multiclass classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted class labels
        y_prob: Predicted probabilities (optional)
        num_classes: Number of classes
    
    Returns:
        Dictionary of multiclass classification metrics
    """
    if num_classes is None:
        num_classes = len(np.unique(y_true))
    
    metrics = {
        'accuracy': float(np.mean(y_true == y_pred)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro')),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro')),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted'))
    }
    
    if y_prob is not None:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        metrics.update({
            'macro_auc_roc': float(calculate_macro_auc_roc(y_true_bin, y_prob)),
            'macro_average_precision': float(calculate_macro_average_precision(y_true_bin, y_prob)),
            'multiclass_brier_score': float(calculate_multiclass_brier_score(y_true, y_prob)),
            'multiclass_ece': float(calculate_multiclass_ece(y_true, y_prob)),
            'multiclass_mce': float(calculate_multiclass_mce(y_true, y_prob))
        })
    
    return metrics


def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Brier score for binary classification."""
    return float(np.mean((y_prob - y_true) ** 2))


def calculate_multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate Brier score for multiclass classification."""
    y_true_bin = label_binarize(y_true, classes=range(y_prob.shape[1]))
    return float(np.mean(np.sum((y_prob - y_true_bin) ** 2, axis=1)))


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error for binary classification."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = np.logical_and(y_prob >= bin_lower, y_prob < bin_upper)
        if not np.any(in_bin):
            continue
        
        # Calculate accuracy and confidence in this bin
        accuracy_in_bin = np.mean(y_true[in_bin] == (y_prob[in_bin] >= 0.5))
        avg_confidence_in_bin = np.mean(y_prob[in_bin])
        
        # Add weighted absolute difference to ECE
        ece += np.sum(in_bin) * np.abs(accuracy_in_bin - avg_confidence_in_bin)
    
    return float(ece / len(y_prob))


def calculate_multiclass_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """Calculate Expected Calibration Error for multiclass classification."""
    predictions = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if not np.any(bin_mask):
            continue
        
        bin_accuracy = np.mean(accuracies[bin_mask])
        bin_confidence = np.mean(confidences[bin_mask])
        bin_samples = np.sum(bin_mask)
        
        ece += (bin_samples / total_samples) * np.abs(bin_accuracy - bin_confidence)
    
    return float(ece)


def calculate_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Maximum Calibration Error for binary classification."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    max_ce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(y_prob >= bin_lower, y_prob < bin_upper)
        if not np.any(in_bin):
            continue
        
        accuracy_in_bin = np.mean(y_true[in_bin] == (y_prob[in_bin] >= 0.5))
        avg_confidence_in_bin = np.mean(y_prob[in_bin])
        
        ce = np.abs(accuracy_in_bin - avg_confidence_in_bin)
        max_ce = max(max_ce, ce)
    
    return float(max_ce)


def calculate_multiclass_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """Calculate Maximum Calibration Error for multiclass classification."""
    predictions = np.argmax(y_prob, axis=1)
    confidences = np.max(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_ce = 0.0
    
    for i in range(n_bins):
        bin_mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if not np.any(bin_mask):
            continue
        
        bin_accuracy = np.mean(accuracies[bin_mask])
        bin_confidence = np.mean(confidences[bin_mask])
        ce = np.abs(bin_accuracy - bin_confidence)
        max_ce = max(max_ce, ce)
    
    return float(max_ce)


def calculate_macro_auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate macro-averaged ROC AUC for multiclass classification."""
    n_classes = y_prob.shape[1]
    auc_scores = []
    
    for i in range(n_classes):
        try:
            auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            auc_scores.append(auc)
        except ValueError:
            continue
    
    return float(np.mean(auc_scores)) if auc_scores else 0.0


def calculate_macro_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate macro-averaged Average Precision for multiclass classification."""
    n_classes = y_prob.shape[1]
    ap_scores = []
    
    for i in range(n_classes):
        try:
            ap = average_precision_score(y_true[:, i], y_prob[:, i])
            ap_scores.append(ap)
        except ValueError:
            continue
    
    return float(np.mean(ap_scores)) if ap_scores else 0.0


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 2
) -> Dict[str, float]:
    """Calculate comprehensive metrics for classification.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted class labels
        y_prob: Predicted probabilities (optional)
        num_classes: Number of classes
    
    Returns:
        Dictionary containing all relevant metrics
    """
    try:
        y_true = ensure_numpy(y_true)
        y_pred = ensure_numpy(y_pred)
        if y_prob is not None:
            y_prob = ensure_numpy(y_prob)
        
        if num_classes == 2:
            return calculate_binary_metrics(y_true, y_pred, y_prob)
        else:
            return calculate_multiclass_metrics(y_true, y_pred, y_prob, num_classes)
    
    except Exception as e:
        print(f"Error in calculate_metrics: {str(e)}")
        return create_default_metrics(num_classes == 2)


def process_model_outputs(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    additional_outputs: Optional[Dict] = None
) -> Dict[str, float]:
    """Process model outputs and calculate comprehensive metrics.
    
    Args:
        outputs: Model output logits
        labels: Ground truth labels
        additional_outputs: Additional model outputs including probabilities
        
    Returns:
        Dictionary containing calculated metrics
    """
    try:
        # Ensure outputs and labels are on CPU and convert to numpy
        outputs = outputs.detach().cpu()
        labels = labels.detach().cpu()
        
        # Get number of classes from outputs shape or additional info
        num_classes = (outputs.shape[1] if len(outputs.shape) > 1 
                      else 2)  # Default to binary if shape is ambiguous
        
        # Get probabilities and predictions
        if additional_outputs and 'probabilities' in additional_outputs:
            probs = additional_outputs['probabilities'].detach().cpu()
        else:
            if num_classes == 2:
                probs = torch.sigmoid(outputs)
            else:
                probs = torch.softmax(outputs, dim=1)
        
        # Convert to numpy arrays
        outputs_np = outputs.numpy()
        labels_np = labels.numpy()
        probs_np = probs.numpy()
        
        # Get predictions
        if num_classes == 2:
            preds_np = (probs_np >= 0.5).astype(np.float32)
            if len(preds_np.shape) > 1:
                preds_np = preds_np[:, 1]  # For binary case, use second class
                probs_np = probs_np[:, 1]
        else:
            preds_np = np.argmax(probs_np, axis=1)
        
        # Calculate metrics
        metrics = calculate_metrics(
            y_true=labels_np,
            y_pred=preds_np,
            y_prob=probs_np,
            num_classes=num_classes
        )
        
        # Add additional metrics if available
        if additional_outputs:
            for key in ['consistency_loss', 'uncertainty', 'entropy']:
                if key in additional_outputs:
                    value = additional_outputs[key]
                    if torch.is_tensor(value):
                        metrics[key] = float(value.detach().cpu().item())
                    else:
                        metrics[key] = float(value)
        
        return metrics
        
    except Exception as e:
        print(f"Error in process_model_outputs: {str(e)}")
        return create_default_metrics(num_classes == 2)

def create_default_metrics(is_binary: bool = True) -> Dict[str, float]:
    """Create default metrics dictionary with zero values."""
    if is_binary:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'auc_roc': 0.0,
            'average_precision': 0.0,
            'brier_score': 1.0,
            'ece': 1.0,
            'mce': 1.0
        }
    else:
        return {
            'accuracy': 0.0,
            'macro_precision': 0.0,
            'macro_recall': 0.0,
            'macro_f1': 0.0,
            'weighted_f1': 0.0,
            'macro_auc_roc': 0.0,
            'macro_average_precision': 0.0,
            'multiclass_brier_score': 1.0,
            'multiclass_ece': 1.0,
            'multiclass_mce': 1.0
        }