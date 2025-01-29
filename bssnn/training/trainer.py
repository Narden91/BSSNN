from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
from rich import print

from bssnn.config.config import BSSNNConfig
from bssnn.explainability.explainer import run_explanations
from bssnn.training.early_stopping import EarlyStopping
from bssnn.utils.data_loader import DataLoader
from ..model.bssnn import BSSNN
from .metrics import calculate_calibration_error, calculate_expected_calibration_error, calculate_metrics, calculate_predictive_entropy, find_optimal_threshold, process_model_outputs
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
        consistency_weight: float = 0.1 
    ):
        super().__init__()
        self.model = model
        # Replace BCELoss with BCEWithLogitsLoss for numerical stability
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optimizer or optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta
        )
        self.consistency_weight = consistency_weight
    
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
        # Calculate main loss using logits directly
        main_loss = self.criterion(log_outputs, targets)
        
        loss_dict = {'main_loss': main_loss.item()}
        
        if additional_outputs:
            # Add KL divergence loss for probability consistency
            if 'log_joint' in additional_outputs and 'log_conditional' in additional_outputs:
                kl_loss = self._calculate_kl_divergence(
                    additional_outputs['log_joint'],
                    additional_outputs['log_conditional']
                )
                loss_dict['kl_loss'] = kl_loss.item()
                
                # Add consistency loss if available
                if 'consistency_loss' in additional_outputs:
                    consistency_loss = additional_outputs['consistency_loss']
                    loss_dict['consistency_loss'] = consistency_loss
                    
                    # Combine all losses
                    total_loss = main_loss + self.consistency_weight * (consistency_loss + kl_loss)
                    loss_dict['total_loss'] = total_loss.item()
                else:
                    total_loss = main_loss + self.consistency_weight * kl_loss
                    loss_dict['total_loss'] = total_loss.item()
            else:
                total_loss = main_loss
                loss_dict['total_loss'] = main_loss.item()
        else:
            total_loss = main_loss
            loss_dict['total_loss'] = main_loss.item()
            
        return total_loss, loss_dict
    
    def _calculate_kl_divergence(
        self,
        log_joint: torch.Tensor,
        log_conditional: torch.Tensor
    ) -> torch.Tensor:
        """Calculate KL divergence between joint and conditional distributions.
        
        Args:
            log_joint: Log joint probabilities
            log_conditional: Log conditional probabilities
            
        Returns:
            KL divergence loss
        """
        # Convert from log-space to probability space
        joint_probs = torch.softmax(log_joint, dim=1)
        conditional_probs = torch.softmax(log_conditional, dim=1)
        
        # Calculate KL divergence
        kl_div = torch.sum(joint_probs * (
            torch.log(joint_probs + 1e-10) - torch.log(conditional_probs + 1e-10)
        ), dim=1)
        
        return kl_div.mean()
    
    def train_epoch(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor
    ) -> Dict[str, float]:
        """Train for one epoch with improved log-space handling.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary containing loss values
        """
        try:
            self.model.train()
            self.optimizer.zero_grad()
            
            # Get model outputs in log-space
            log_outputs, additional_outputs = self.model(X_train)
            
            # Calculate loss using log-space computations
            total_loss, loss_dict = self.calculate_total_loss(
                log_outputs,
                y_train,
                additional_outputs
            )
            
            # Backpropagation
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            return loss_dict
            
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
            
            # Early stopping check with proper loss handling
            if self.early_stopping(self.model, val_loss):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
    
    def evaluate(self, X_val: torch.Tensor, y_val: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model with enhanced error handling and type checking.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Tuple of (loss value, metrics dictionary)
        """
        self.model.eval()
        with torch.no_grad():
            try:
                # Get predictions and additional outputs
                outputs, additional_outputs = self.model(X_val)
                outputs = outputs.view(-1)  # Ensure correct shape
                
                # Calculate main loss
                main_loss = self.criterion(outputs, y_val)
                
                # Process all metrics
                metrics = process_model_outputs(outputs, y_val, additional_outputs)
                
                # Add calibration metrics
                metrics.update({
                    'calibration_error': calculate_calibration_error(
                        y_val.cpu().numpy(),
                        outputs.cpu().numpy()
                    ),
                    'expected_calibration_error': calculate_expected_calibration_error(
                        y_val.cpu().numpy(),
                        outputs.cpu().numpy()
                    ),
                    'predictive_entropy': calculate_predictive_entropy(
                        outputs.cpu().numpy()
                    ),
                    'optimal_threshold': find_optimal_threshold(
                        y_val.cpu().numpy(),
                        outputs.cpu().numpy()
                    )
                })
                
                # Add loss components
                loss_value = float(main_loss.item())
                metrics['main_loss'] = loss_value
                
                if additional_outputs and 'consistency_loss' in additional_outputs:
                    consistency_loss = float(additional_outputs['consistency_loss'])
                    # Scale down consistency loss to prevent it from dominating
                    consistency_loss = consistency_loss * self.consistency_weight
                    metrics.update({
                        'consistency_loss': consistency_loss,
                        'total_loss': loss_value + consistency_loss
                    })
                else:
                    metrics['total_loss'] = loss_value
                
                return metrics['total_loss'], metrics
                    
            except Exception as e:
                print(f"Error in evaluate: {str(e)}")
                return float('inf'), {'error': str(e)}


def run_training(
    config: BSSNNConfig,
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
        lr=config.training.learning_rate,
        weight_decay=config.model.weight_decay,
        early_stopping_patience=config.model.early_stopping_patience,
        early_stopping_min_delta=config.model.early_stopping_min_delta,
        consistency_weight=config.model.consistency_weight
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