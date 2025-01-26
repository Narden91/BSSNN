from typing import Dict, Generator, Tuple, Optional, List
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from rich.console import Console

from bssnn.config.config import BSSNNConfig

from ..model.bssnn import BSSNN
from ..training.trainer import run_training
from ..visualization.visualization import CrossValidationProgress

console = Console()

class CrossValidator:
    """Handles cross-validation while preventing data leakage."""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """Initialize cross-validator.
        
        Args:
            n_splits: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def get_test_scaler(self):
        """Retrieve the scaler fit on the full training data."""
        return self.scaler
        
    def _scale_features(self, X_train: np.ndarray, X_val: np.ndarray, X_test: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Scale features using only training data statistics.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Optional test features
            
        Returns:
            Scaled feature arrays
        """
        # Fit scaler only on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        # Transform validation and test sets using training statistics
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test) if X_test is not None else None
        
        return X_train_scaled, X_val_scaled, X_test_scaled
        
    def create_folds(self, X: np.ndarray, y: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """Create stratified cross-validation folds.
        
        Args:
            X: Input features
            y: Target labels
            
        Yields:
            Tuples of (X_train, X_val, y_train, y_val) for each fold
        """
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for train_idx, val_idx in skf.split(X, y):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale features
            X_train_scaled, X_val_scaled, _ = self._scale_features(X_train, X_val, None)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            X_val_tensor = torch.FloatTensor(X_val_scaled)
            y_train_tensor = torch.FloatTensor(y_train)
            y_val_tensor = torch.FloatTensor(y_val)
            
            yield X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor

def run_cross_validation(
    config: 'BSSNNConfig',
    X: torch.Tensor,
    y: torch.Tensor,
    output_dir: Optional[Path] = None
) -> Tuple[Optional[BSSNN], Dict[str, Dict[str, float]]]:
    """Run cross-validation with proper data handling.
    
    Args:
        config: Model and training configuration
        X: Input features tensor
        y: Target labels tensor
        output_dir: Optional directory for saving results
        
    Returns:
        Tuple of (best model, cross-validation metrics)
    """
    # Convert tensors to numpy for sklearn compatibility
    X_np = X.numpy()
    y_np = y.numpy()
    
    # Initialize cross-validator
    cv = CrossValidator(
        n_splits=config.data.validation.n_folds,
        random_state=config.data.random_state
    )
    
    # Initialize metrics storage
    cv_metrics: List[Dict[str, float]] = []
    best_model = None
    best_val_score = float('-inf')
    
    # Initialize progress tracking
    progress = CrossValidationProgress(cv.n_splits)
    
    # Run cross-validation
    for fold, (X_train, X_val, y_train, y_val) in enumerate(cv.create_folds(X_np, y_np), 1):
        progress.start_fold(fold)
        
        # Initialize and train model
        model = BSSNN(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            dropout_rate=config.model.dropout_rate
        )
        
        # Train model
        trainer = run_training(
            config=config,
            model=model,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            fold=fold
        )
        
        # Evaluate on validation set
        _, val_metrics = trainer.evaluate(X_val, y_val)
        cv_metrics.append(val_metrics)
        
        # Update best model based on validation performance
        val_score = val_metrics['auc_roc']
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = model
    
    # Calculate and display average metrics
    if not cv_metrics:
        raise ValueError("No metrics were collected during cross-validation.")
    
    avg_metrics = calculate_cv_statistics(cv_metrics)
    progress.print_summary(avg_metrics)
    
    return best_model, avg_metrics, cv.get_test_scaler()


def calculate_cv_statistics(cv_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Calculate mean and standard deviation for metrics across folds.
    
    Args:
        cv_metrics: List of metric dictionaries from each fold
        
    Returns:
        Dictionary containing mean and std for each metric
    """
    avg_metrics = {}
    for metric in cv_metrics[0].keys():
        values = [m[metric] for m in cv_metrics]
        avg_metrics[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }
    return avg_metrics