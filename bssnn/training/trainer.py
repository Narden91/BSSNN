from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Dict, Tuple, Optional, Union
from pathlib import Path
from rich import print

from bssnn.config.config import BSSNNConfig
from bssnn.training.loss_manager import EarlyStoppingMonitor
from ..model.bssnn import BSSNN
from .metrics import process_model_outputs
from ..visualization.visualization import TrainingProgress



class BSSNNTrainer:
    """Trainer class for BSSNN models."""
    
    def __init__(
        self,
        model: BSSNN,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4,
        early_stopping_metric: str = 'val_loss',
        device: Optional[torch.device] = None,
        **kwargs  # Add this to handle additional config parameters
    ):
        """Initialize the trainer with enhanced configuration.
        
        Args:
            model: BSSNN model instance
            optimizer: Optional optimizer (defaults to Adam)
            lr: Learning rate for default optimizer
            weight_decay: L2 regularization factor
            early_stopping_patience: Number of epochs to wait before stopping
            early_stopping_min_delta: Minimum improvement for early stopping
            early_stopping_metric: Metric to monitor for early stopping
            device: Optional device for model and data
            **kwargs: Additional configuration parameters (ignored but allowed for compatibility)
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optimizer or optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Initialize criterion
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Initialize early stopping with enhanced monitor
        self.early_stopping = EarlyStoppingMonitor(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode='min',
            metric_name=early_stopping_metric
        )
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }
    
    def calculate_total_loss(
        self, 
        log_outputs: torch.Tensor,
        targets: torch.Tensor,
        additional_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate loss using log-space computations for numerical stability.
        
        Args:
            log_outputs: Model outputs in log-space (logits)
            targets: Target values
            additional_outputs: Additional model outputs including consistency metrics
            
        Returns:
            Tuple of (total loss tensor, loss dictionary)
        """
        # Main loss (BCEWithLogitsLoss)
        main_loss = self.criterion(log_outputs, targets)
        loss_dict = {'main_loss': main_loss.item()}
        
        if additional_outputs:
            # KL divergence with adaptive weighting
            probs = torch.softmax(additional_outputs['log_joint'], dim=1)
            kl_loss = self._calculate_kl_divergence(probs)
            loss_dict['kl_loss'] = kl_loss.item()
            
            # Consistency loss (already scaled)
            consistency_loss = additional_outputs.get('consistency_loss', 0.0)
            loss_dict['consistency_loss'] = consistency_loss
            
            # Total loss with separate scaling
            total_loss = (
                main_loss 
                + self.consistency_weight * consistency_loss 
                + self.kl_weight * kl_loss
            )
            loss_dict['total_loss'] = total_loss.item()
        else:
            total_loss = main_loss
            loss_dict['total_loss'] = main_loss.item()
            
        return total_loss, loss_dict
    
    def _prepare_batch(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare a batch for training or evaluation.
        
        Args:
            X: Input tensor
            y: Optional target tensor
            
        Returns:
            Tuple of (device-moved input tensor, optional device-moved target tensor)
        """
        X = X.to(self.device)
        if y is not None:
            y = y.to(self.device)
        return X, y
    
    def _calculate_kl_divergence(
        self, 
        probs: torch.Tensor  # Now takes probabilities directly
    ) -> torch.Tensor:
        """Calculate KL divergence between model predictions and uniform prior."""
        # Uniform prior (adjusted by prior_strength)
        prior = torch.full_like(probs, self.prior_strength * 0.5)  # 0.5 for binary
        prior = prior / prior.sum(dim=1, keepdim=True)  # Ensure normalization
        
        # KL(pred || prior)
        kl_div = torch.sum(
            probs * (torch.log(probs + 1e-10) - torch.log(prior + 1e-10)),
            dim=1
        )
        return kl_div.mean()
    
    def train_epoch(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor
    ) -> Dict[str, float]:
        """Train for one epoch with improved error handling and monitoring.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary containing training metrics
            
        Raises:
            RuntimeError: If training fails due to model or optimizer issues
        """
        self.model.train()
        epoch_metrics = {}
        
        try:
            # Prepare batch
            X_train, y_train = self._prepare_batch(X_train, y_train)
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, additional_outputs = self.model(X_train)
            outputs = outputs.view(-1)
            
            # Calculate loss
            loss = self.criterion(outputs, y_train)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Calculate training metrics
            with torch.no_grad():
                epoch_metrics = process_model_outputs(
                    outputs.detach(),
                    y_train,
                    additional_outputs
                )
                epoch_metrics['train_loss'] = loss.item()
            
            return epoch_metrics
            
        except RuntimeError as e:
            print(f"\nError during training epoch: {str(e)}")
            raise
    
    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        epochs: int,
        callback: Optional[Callable] = None):
        """Train the model with comprehensive monitoring and callbacks.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs to train
            callback: Optional callback for progress updates
        """
        best_val_loss = float('inf')
        
        try:
            for epoch in range(epochs):
                # Training step
                train_metrics = self.train_epoch(X_train, y_train)
                
                # Validation step
                val_loss, val_metrics = self.evaluate(X_val, y_val)
                
                # Update history
                self.history['train_loss'].append(train_metrics['train_loss'])
                self.history['val_loss'].append(val_loss)
                self.history['metrics'].append(val_metrics)
                
                # Update progress if callback provided
                if callback:
                    callback(epoch + 1, val_loss, val_metrics)
                
                # Early stopping check
                if self.early_stopping(self.model, val_metrics, epoch):
                    break
                
                # Track best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            raise
        
        print(f"\nTraining completed:")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Final epoch: {epoch + 1}")
    
    def evaluate(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model with comprehensive metrics.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Tuple of (validation loss, metrics dictionary)
        """
        self.model.eval()
        
        try:
            with torch.no_grad():
                # Prepare batch
                X_val, y_val = self._prepare_batch(X_val, y_val)
                
                # Forward pass
                outputs, additional_outputs = self.model(X_val)
                outputs = outputs.view(-1)
                
                # Calculate loss
                val_loss = self.criterion(outputs, y_val)
                
                # Calculate metrics
                metrics = process_model_outputs(
                    outputs,
                    y_val,
                    additional_outputs
                )
                metrics['val_loss'] = val_loss.item()
                
                return val_loss.item(), metrics
                
        except Exception as e:
            print(f"\nError during evaluation: {str(e)}")
            return float('inf'), {'error': str(e)}


def run_training(
    config: 'BSSNNConfig',
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
    
    This function creates and configures a trainer instance based on the provided
    configuration. It handles the training process and provides appropriate progress
    tracking and early stopping functionality.
    
    Args:
        config: Training configuration
        model: Initialized BSSNN model
        X_train: Training features
        X_val: Validation features
        y_train: Training targets
        y_val: Validation targets
        fold: Optional fold number for cross-validation
        is_final: Whether this is the final model training
        silent: Whether to suppress progress output
    
    Returns:
        Trained model trainer instance
    """
    # Create trainer with configuration parameters
    trainer = BSSNNTrainer(
        model=model,
        lr=config.training.learning_rate,
        weight_decay=config.model.weight_decay,
        early_stopping_patience=config.model.early_stopping_patience,
        early_stopping_min_delta=config.model.early_stopping_min_delta,
        early_stopping_metric='val_loss'  # We now specify the metric to monitor
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
    val_loss, final_metrics = trainer.evaluate(X_val, y_val)
    progress.complete(trainer.early_stopping.best_score, final_metrics)
    
    return trainer


def run_final_model_training(config: BSSNNConfig, X_train_val: torch.Tensor, y_train_val: torch.Tensor, output_dir: Path) -> BSSNN:
    """Train final model on the complete training+validation dataset.
    
    This function trains the final model on the entire train_val dataset without
    splitting it further. This is done after cross-validation has been used to
    validate the model architecture and hyperparameters.
    
    Args:
        config: Configuration object containing model and training parameters
        X_train_val: Complete training+validation features tensor
        y_train_val: Complete training+validation labels tensor
        output_dir: Directory for saving outputs
        
    Returns:
        Trained BSSNN model
    """
    # Initialize model with the same configuration used in cross-validation
    final_model = BSSNN(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        dropout_rate=config.model.dropout_rate
    )
    
    # For final training, we'll use the same data for both training and "validation"
    # This allows us to use the same training infrastructure while effectively
    # training on all data
    trainer = run_training(
        config=config,
        model=final_model,
        X_train=X_train_val,  # Use full train_val set for training
        X_val=X_train_val,    # Use same data for validation
        y_train=y_train_val,
        y_val=y_train_val,
        is_final=True,        # Flag indicating this is final training
        silent=False
    )
    
    return final_model


def evaluate_on_test_set(model: BSSNN, X_test: torch.Tensor, y_test: torch.Tensor) -> Tuple[float, dict]:
    """Evaluate model performance on the test set."""
    trainer = BSSNNTrainer(model=model)
    test_loss, test_metrics = trainer.evaluate(X_test, y_test)
    return test_loss, test_metrics