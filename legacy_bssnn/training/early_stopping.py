from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn


class EarlyStoppingMonitor:
    """Enhanced early stopping implementation that supports flexible metric monitoring.
    
    This implementation allows monitoring of any metric (not just loss) and supports both
    minimization and maximization scenarios. It also provides better state management
    and more detailed tracking of the stopping process.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min',
        restore_best: bool = True,
        metric_name: str = 'val_loss'
    ):
        """Initialize the early stopping monitor with enhanced configuration.
        
        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change in monitored metric to qualify as an improvement
            mode: Either 'min' or 'max' depending on whether the metric should be minimized or maximized
            restore_best: Whether to restore the model to its best weights when stopped
            metric_name: Name of the metric to monitor in the metrics dictionary
        """
        if mode not in ['min', 'max']:
            raise ValueError("Mode must be 'min' or 'max'")
            
        self.patience = patience
        self.min_delta = abs(min_delta)  # Ensure positive
        self.mode = mode
        self.restore_best = restore_best
        self.metric_name = metric_name
        
        # Initialize tracking variables
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
        
        # Create comparison function based on mode
        self.is_improvement = (
            lambda current, best: current < (best - self.min_delta)
            if mode == 'min'
            else current > (best + self.min_delta)
        )
    
    def _extract_metric(
        self,
        metrics: Union[float, Dict[str, Any]]
    ) -> float:
        """Extract the monitored metric value from the metrics object.
        
        Args:
            metrics: Either a float value or a dictionary containing metrics
            
        Returns:
            The metric value to monitor
            
        Raises:
            KeyError: If metric_name is not found in the metrics dictionary
            ValueError: If metrics value cannot be converted to float
        """
        if isinstance(metrics, dict):
            if self.metric_name not in metrics:
                raise KeyError(f"Metric '{self.metric_name}' not found in metrics")
            value = metrics[self.metric_name]
        else:
            value = metrics
            
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Could not convert metric value to float: {e}")
    
    def __call__(
        self,
        model: nn.Module,
        metrics: Union[float, Dict[str, Any]],
        epoch: int
    ) -> bool:
        """Check if training should be stopped based on monitored metric.
        
        Args:
            model: Current model instance
            metrics: Current epoch's metrics (float or dictionary)
            epoch: Current epoch number
            
        Returns:
            Boolean indicating whether to stop training
        """
        try:
            current = self._extract_metric(metrics)
        except (KeyError, ValueError) as e:
            print(f"Warning: Error extracting metric: {e}")
            return False
            
        if self.is_improvement(current, self.best_score):
            # We found a better score
            self.best_score = current
            self.best_epoch = epoch
            self.counter = 0
            
            if self.restore_best:
                # Save a copy of the model weights
                self.best_weights = {
                    name: param.clone().detach()
                    for name, param in model.state_dict().items()
                }
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                
                if self.restore_best and self.best_weights is not None:
                    # Restore the best weights
                    model.load_state_dict(self.best_weights)
                    
                # Report improvement history
                print(f"\nEarly stopping triggered:")
                print(f"Best {self.metric_name}: {self.best_score:.6f}")
                print(f"Best epoch: {self.best_epoch}")
                print(f"Episodes without improvement: {self.counter}")
                return True
                
        return False
    
    def reset(self):
        """Reset the monitor state to allow reuse."""
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0
        self.best_epoch = 0