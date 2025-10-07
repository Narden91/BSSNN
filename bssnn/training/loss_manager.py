# loss_manager.py
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


@dataclass
class LossWeights:
    """Configuration for loss component weights."""
    main_loss: float = 1.0
    kl_weight: float = 0.01
    consistency_weight: float = 0.1
    prior_strength: float = 0.5
    temperature_scale: float = 1.0


class LossManager:
    """Enhanced loss manager supporting both binary and multiclass scenarios."""
    
    def __init__(
        self,
        weights: LossWeights,
        num_classes: int = 2,
        criterion: Optional[nn.Module] = None
    ):
        """Initialize the loss manager.
        
        Args:
            weights: Configuration for loss component weights
            num_classes: Number of output classes
            criterion: Optional custom criterion
        """
        self.weights = weights
        self.num_classes = num_classes
        self.criterion = criterion or (
            nn.BCEWithLogitsLoss() if num_classes == 2
            else nn.CrossEntropyLoss()
        )
    
    def calculate_main_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the main classification loss.
        
        Args:
            logits: Model output logits
            targets: Target values
            
        Returns:
            Classification loss
        """
        if self.num_classes == 2:
            # Binary case: ensure proper shape
            logits = logits.view(-1)
            return self.criterion(logits, targets)
        else:
            # Multiclass case
            return self.criterion(logits, targets.long())
    
    def calculate_kl_divergence(
        self,
        probs: torch.Tensor
    ) -> torch.Tensor:
        """Calculate KL divergence from uniform prior for any number of classes.
        
        Args:
            probs: Predicted probabilities
            
        Returns:
            KL divergence loss
        """
        # Create uniform prior distribution
        prior = torch.full_like(
            probs,
            self.weights.prior_strength / self.num_classes
        )
        
        # Calculate KL divergence with numerical stability
        eps = torch.finfo(probs.dtype).eps
        kl_div = torch.sum(
            probs * (
                torch.log(probs + eps) -
                torch.log(prior + eps)
            ),
            dim=1
        )
        return kl_div.mean()
    
    def calculate_consistency_loss(
        self,
        probs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate probability consistency constraints for any number of classes.
        
        Args:
            probs: Predicted probabilities
            
        Returns:
            Tuple of (consistency loss, constraint metrics)
        """
        # Probability sum constraint
        prob_sum = probs.sum(dim=1)
        sum_constraint = torch.nn.functional.mse_loss(
            prob_sum,
            torch.ones_like(prob_sum)
        )
        
        # Non-negativity constraint
        non_negativity = torch.mean(
            torch.nn.functional.relu(-probs)
        )
        
        # Entropy regularization for multiclass
        if self.num_classes > 2:
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
            consistency_loss = sum_constraint + non_negativity + 0.1 * entropy
            
            return consistency_loss, {
                'sum_constraint_loss': float(sum_constraint.item()),
                'non_negativity_loss': float(non_negativity.item()),
                'entropy_loss': float(entropy.item())
            }
        
        # Binary case
        consistency_loss = sum_constraint + non_negativity
        return consistency_loss, {
            'sum_constraint_loss': float(sum_constraint.item()),
            'non_negativity_loss': float(non_negativity.item())
        }
    
    def compute_total_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        additional_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss combining all components.
        
        Args:
            logits: Model output logits
            targets: Target values
            additional_outputs: Additional model outputs
            
        Returns:
            Tuple of (total loss tensor, loss components dictionary)
        """
        # Calculate main loss with temperature scaling
        scaled_logits = logits / self.weights.temperature_scale
        main_loss = self.calculate_main_loss(scaled_logits, targets)
        loss_dict = {'main_loss': main_loss.item()}
        
        # Initialize total loss
        total_loss = self.weights.main_loss * main_loss
        
        if additional_outputs and 'probabilities' in additional_outputs:
            probs = additional_outputs['probabilities']
            
            # Add KL divergence
            kl_loss = self.calculate_kl_divergence(probs)
            loss_dict['kl_loss'] = kl_loss.item()
            total_loss += self.weights.kl_weight * kl_loss
            
            # Add consistency loss
            consistency_loss, consistency_info = self.calculate_consistency_loss(probs)
            loss_dict.update(consistency_info)
            loss_dict['consistency_loss'] = consistency_loss.item()
            total_loss += self.weights.consistency_weight * consistency_loss
        
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict


class EarlyStoppingMonitor:
    """Flexible early stopping implementation with customizable monitoring."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min',
        restore_best: bool = True,
        metric_name: str = 'val_loss'
    ):
        """Initialize early stopping monitor.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: Either 'min' or 'max' for metric optimization
            restore_best: Whether to restore best weights when stopped
            metric_name: Name of metric to monitor
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.metric_name = metric_name
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0
    
    def _is_improvement(self, current: float) -> bool:
        """Check if current score is an improvement."""
        if self.mode == 'min':
            return current < (self.best_score - self.min_delta)
        return current > (self.best_score + self.min_delta)
    
    def _get_metric_value(
        self,
        metrics: Union[float, Dict[str, Any]]
    ) -> float:
        """Extract metric value from metrics dictionary or scalar."""
        if isinstance(metrics, dict):
            return float(metrics[self.metric_name])
        return float(metrics)
    
    def __call__(
        self,
        model: nn.Module,
        metrics: Union[float, Dict[str, Any]],
        epoch: int
    ) -> bool:
        """Check if training should stop.
        
        Args:
            model: Current model instance
            metrics: Current metrics (scalar or dictionary)
            epoch: Current epoch number
            
        Returns:
            Boolean indicating whether to stop training
        """
        current = self._get_metric_value(metrics)
        
        if self._is_improvement(current):
            # Save best score and weights
            self.best_score = current
            self.counter = 0
            if self.restore_best:
                self.best_weights = {
                    name: param.clone().detach()
                    for name, param in model.state_dict().items()
                }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        return False
    
    def reset(self):
        """Reset monitor state."""
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.best_weights = None
        self.counter = 0
        self.stopped_epoch = 0