import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from pathlib import Path
from rich import print
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
    y_val: torch.Tensor
) -> BSSNNTrainer:
    """Execute the training process with progress tracking.
    
    Args:
        config: Training configuration
        model: Initialized BSSNN model
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
    
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
    
    # Print training configuration
    print(
        "\n[cyan]Training Configuration[/cyan]",
        f"\nEpochs: {config.training.num_epochs}",
        f"\nLearning Rate: {config.training.learning_rate}",
        f"\nBatch Size: {config.training.batch_size}"
    )
    
    # Train model
    print("\n[bold cyan]Starting training...[/bold cyan]")
    
    for epoch in range(1, config.training.num_epochs + 1):
        # Training step
        train_loss = trainer.train_epoch(X_train, y_train)
        
        # Validation step
        val_loss, metrics = trainer.evaluate(X_val, y_val)
        
        # Update progress
        progress.update(epoch, val_loss, metrics)
    
    # Complete training
    progress.complete(val_loss, metrics)
    
    # Save model if specified in config
    if config.output.save_model:
        save_path = Path(config.output.model_dir) / "model.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"\n[green]Model saved to[/green] {save_path}")
    
    return trainer