import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from ..model.bssnn import BSSNN
from .metrics import calculate_metrics


class BSSNNTrainer:
    """Trainer class for BSSNN models."""
    
    def __init__(
        self,
        model: BSSNN,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 0.001
    ):
        """Initialize the trainer.
        
        Args:
            model: BSSNN model instance
            criterion: Loss function (defaults to BCELoss if None)
            optimizer: Optimizer (defaults to Adam if None)
            lr: Learning rate for default optimizer
        """
        self.model = model
        self.criterion = criterion or nn.BCELoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=lr)
        
    def train_epoch(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor
    ) -> float:
        """Train for one epoch.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(X_train).squeeze()
        loss = self.criterion(outputs, y_train)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor
    ) -> Tuple[float, dict]:
        """Evaluate the model.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Tuple of (validation loss, metrics dictionary)
        """
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val).squeeze()
            val_loss = self.criterion(val_outputs, y_val)
            metrics = calculate_metrics(y_val, val_outputs)
        
        return val_loss.item(), metrics