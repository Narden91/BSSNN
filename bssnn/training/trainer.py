import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from pathlib import Path
from rich import print

from bssnn.config.config import BSSNNConfig
from bssnn.explainability.explainer import run_explanations
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
        lr: float = 0.001
    ):
        self.model = model
        self.criterion = criterion or nn.BCELoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=lr)
        
    def train_epoch(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor
    ) -> float:
        """Train for one epoch."""
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
        """Evaluate the model."""
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val).squeeze()
            val_loss = self.criterion(val_outputs, y_val)
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
    # Initialize trainer
    trainer = BSSNNTrainer(
        model=model,
        lr=config.training.learning_rate
    )
    
    # Create progress tracker
    progress = TrainingProgress(
        total_epochs=config.training.num_epochs,
        display_metrics=config.training.display_metrics
    )
    
    # Determine training phase
    phase = "Final Model" if is_final else f"Fold {fold}" if fold is not None else "Training"
    
    # Print training configuration if not in silent mode
    if not silent:
        print(
            f"\n[cyan]{phase} Configuration[/cyan]",
            f"\nEpochs: {config.training.num_epochs}",
            f"\nLearning Rate: {config.training.learning_rate}",
            f"\nBatch Size: {config.training.batch_size}"
        )
        print(f"\n[bold cyan]Starting {phase.lower()}...[/bold cyan]")
    
    for epoch in range(1, config.training.num_epochs + 1):
        # Training step
        train_loss = trainer.train_epoch(X_train, y_train)
        
        # Validation step
        val_loss, metrics = trainer.evaluate(X_val, y_val)
        
        # Update progress if not in silent mode
        if not silent:
            progress.update(epoch, val_loss, metrics)
    
    # Complete training with final metrics if not in silent mode
    if not silent:
        progress.complete(val_loss, metrics)
    
    # Save model if specified in config and this is the final model
    if config.output.save_model and is_final:
        save_path = Path(config.output.model_dir) / "model.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        if not silent:
            print(f"\n[green]Model saved to[/green] {save_path}")
    
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