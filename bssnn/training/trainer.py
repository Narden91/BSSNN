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
        """Train for one epoch with enhanced validation and debugging.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Training loss for this epoch
        """
        try:
            self.model.train()
            self.optimizer.zero_grad()
            
            # Validate input data
            assert torch.all((y_train >= 0) & (y_train <= 1)), "Labels must be between 0 and 1"
            
            # Forward pass
            model_output = self.model(X_train)
            
            # Handle both tuple output (new model) and tensor output (old model)
            if isinstance(model_output, tuple):
                outputs, additional_outputs = model_output
                consistency_loss = additional_outputs['consistency_loss']
            else:
                outputs = model_output
                consistency_loss = 0.0
            
            # Ensure outputs are properly bounded
            if not torch.all((outputs >= 0) & (outputs <= 1)):
                raise ValueError("Model outputs must be between 0 and 1")
            
            # Calculate loss with consistency regularization
            main_loss = self.criterion(outputs, y_train)
            total_loss = main_loss + 0.1 * consistency_loss  # Weight for consistency loss
            loss_value = total_loss.item()
            
            # Backpropagation
            total_loss.backward()
            self.optimizer.step()
            
            # Clear memory
            del outputs, model_output
            torch.cuda.empty_cache()
            
            return loss_value
            
        except Exception as e:
            print(f"\n[Debug] Error in train_epoch:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
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
        for epoch in range(epochs):
            # Training step
            train_loss = self.train_epoch(X_train, y_train)
            
            # Validation step
            val_loss, metrics = self.evaluate(X_val, y_val)
            
            # Call progress callback
            if callback:
                callback(epoch + 1, val_loss, metrics)
            
            # Early stopping check (no need to track best_val_loss)
            if self.early_stopping(self.model, val_loss):
                if callback:
                    callback(epoch + 1, val_loss, metrics)
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
    
    def evaluate(self, X_val: torch.Tensor, y_val: torch.Tensor) -> Tuple[float, dict]:
        """Evaluate the model."""
        self.model.eval()
        with torch.no_grad():
            # Get predictions
            model_output = self.model(X_val)
            
            # Handle both tuple output (new model) and tensor output (old model)
            if isinstance(model_output, tuple):
                val_outputs, additional_outputs = model_output
                consistency_loss = additional_outputs['consistency_loss']
            else:
                val_outputs = model_output
                consistency_loss = 0.0
            
            # Calculate main loss
            main_loss = self.criterion(val_outputs, y_val)
            total_loss = main_loss + 0.1 * consistency_loss
            
            # Calculate metrics
            metrics = calculate_metrics(y_val, val_outputs)
            
            # Add consistency loss to metrics if available
            if consistency_loss != 0.0:
                metrics['consistency_loss'] = consistency_loss.item()
        
        return total_loss.item(), metrics


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
    
    return final_model


def evaluate_on_test_set(model: BSSNN, X_test: torch.Tensor, y_test: torch.Tensor) -> Tuple[float, dict]:
    """Evaluate model performance on the test set."""
    trainer = BSSNNTrainer(model=model)
    test_loss, test_metrics = trainer.evaluate(X_test, y_test)
    return test_loss, test_metrics