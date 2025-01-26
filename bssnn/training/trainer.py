import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from pathlib import Path
from rich import print

from bssnn.config.config import BSSNNConfig
from bssnn.explainability.explainer import run_explanations
from bssnn.training.early_stopping import EarlyStopping
from bssnn.utils.data_loader import DataLoader
from ..model.bssnn import BSSNN
from .metrics import calculate_metrics
from ..visualization.visualization import TrainingProgress


class BSSNNTrainer:
    """Trainer class for BSSNN models."""
    
    def __init__(
        self,
        model: BSSNN,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4
    ):
        self.model = model
        self.criterion = criterion or nn.BCELoss()
        self.optimizer = optimizer or optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta
        )
        
    def train_epoch(self, X_train: torch.Tensor, y_train: torch.Tensor) -> float:
        """Train for one epoch.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Training loss for this epoch
        """
        try:
            self.model.train()
            self.optimizer.zero_grad()  # Reset gradients
            
            outputs = self.model(X_train).squeeze()
            loss = self.criterion(outputs, y_train)
            loss_value = loss.item()
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()  # Update weights
            
            # Clear unnecessary tensors
            del outputs
            torch.cuda.empty_cache()
            
            return loss_value
        except RuntimeError as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int,
        callback = None
    ):
        """Train the model with early stopping and progress updates."""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training step
            train_loss = self.train_epoch(X_train, y_train)
            
            # Validation step
            val_loss, metrics = self.evaluate(X_val, y_val)
            
            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # Call progress callback
            if callback:
                callback(epoch + 1, val_loss, metrics)
            
            # Check early stopping
            if self.early_stopping(self.model, val_loss):
                if callback:  # Make sure we update the progress one last time
                    callback(epoch + 1, val_loss, metrics)
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
    
    def evaluate(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor
    ) -> Tuple[float, dict]:
        """Evaluate the model."""
        self.model.eval()
        with torch.no_grad():
            # Get predictions
            val_outputs = self.model(X_val).squeeze()
            val_loss = self.criterion(val_outputs, y_val)
            
            # Calculate metrics
            metrics = calculate_metrics(y_val, val_outputs)
        
        return val_loss.item(), metrics


def run_training(
    config,
    model: BSSNN,
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    y_train: torch.Tensor,
    y_val: torch.Tensor,
    fold: Optional[int] = None,
    is_final: bool = False,
    silent: bool = False
) -> BSSNNTrainer:
    """Execute the training process with progress tracking.
    
    Args:
        config: Training configuration
        model: Initialized BSSNN model
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        fold: Optional fold number for cross-validation
        is_final: Whether this is the final model training
        silent: Whether to suppress progress output
    
    Returns:
        Trained model trainer instance
    """
    trainer = BSSNNTrainer(
        model=model,
        lr=config.training.learning_rate
    )
    
    # Create progress tracker with fold information
    progress = TrainingProgress(
        total_epochs=config.training.num_epochs,
        display_metrics=config.training.display_metrics,
        fold=fold
    )
    
    # Define progress callback
    def update_progress(epoch: int, loss: float, metrics: dict):
        progress.update(epoch, loss, metrics)
    
    # Train the model
    trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=config.training.num_epochs,
        callback=update_progress
    )
    
    # Complete progress
    _, final_metrics = trainer.evaluate(X_val, y_val)
    progress.complete(trainer.early_stopping.best_loss, final_metrics)
    
    return trainer


def run_final_model_training(config: BSSNNConfig, X, y, output_dir: Path):
    """Train final model on full dataset."""
    data_loader = DataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.get_cross_validation_splits(
        X, y, config.data, fold=1
    )
    
    final_model = BSSNN(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size
    )
    
    trainer = run_training(
        config, final_model,
        X_train, X_val,
        y_train, y_val,
        is_final=True,
        silent=False
    )
    
    # Run explanations if enabled
    if config.explainability.enabled:
        explanations_dir = output_dir / config.output.explanations_dir
        explanations_dir.mkdir(parents=True, exist_ok=True)
        run_explanations(
            config, 
            final_model, 
            X_train, 
            X_test, 
            save_dir=str(explanations_dir)
        )
    
    return final_model