from typing import Dict, Union
from torch import nn


class EarlyStopping:
    """Early stopping implementation to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, restore_best_weights: bool = True):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored value to qualify as an improvement
            restore_best_weights: Whether to restore model to best weights when stopped
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.best_weights = None
        self.counter = 0
        self.early_stop = False
        
    def get_loss_value(self, loss: Union[float, Dict[str, float]]) -> float:
        """Extract comparable loss value from different loss formats.
        
        Args:
            loss: Loss value as float or dictionary
            
        Returns:
            Float loss value for comparison
        """
        if isinstance(loss, dict):
            return loss.get('main_loss', 0.0) + loss.get('consistency_loss', 0.0)
        return float(loss)
    
    def __call__(self, model: nn.Module, current_loss: Union[float, Dict[str, float]]) -> bool:
        """Check if training should stop.
        
        Args:
            model: Current model instance
            current_loss: Current loss value or dictionary
            
        Returns:
            Boolean indicating whether to stop training
        """
        loss_value = self.get_loss_value(current_loss)
        
        if loss_value < self.best_loss - self.min_delta:
            self.best_loss = loss_value
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {
                    name: param.clone().detach()
                    for name, param in model.state_dict().items()
                }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        return False
    
    def reset(self):
        """Reset all tracking variables for reuse."""
        self.best_loss = float('inf')
        self.best_weights = None
        self.counter = 0
        self.early_stop = False