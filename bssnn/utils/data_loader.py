from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple, Generator
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import torch
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from rich import print
from bssnn.config.config import BSSNNConfig, DataConfig


class DataLoader:
    """Data loading and preparation class with cross-validation support."""
    def split_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        test_size: float = 0.2,
        random_state: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split data into train+val and test sets.
        
        Args:
            X: Full feature tensor
            y: Full target tensor
            test_size: Fraction of data to reserve for testing
            random_state: Random seed for reproducibility
            
        Returns:
            (X_train_val, X_test, y_train_val, y_test)
        """
        # Convert tensors to numpy for splitting
        X_np = X.numpy()
        y_np = y.numpy()
        
        # Perform stratified split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_np, y_np,
            test_size=test_size,
            stratify=y_np if self.data_config.validation.stratify else None,
            random_state=random_state
        )
        
        # Convert back to tensors (no scaling here to avoid double-processing)
        return (
            torch.tensor(X_train_val, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_train_val, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )
    
    @staticmethod
    def load_and_prepare_data(config: 'DataConfig') -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Main entry point for data loading and preparation."""
        X, y, n_features = DataLoader._load_data(config)
        return X, y, n_features
    
    @staticmethod
    def get_cross_validation_splits(
        X: torch.Tensor,
        y: torch.Tensor,
        config: 'DataConfig',
        fold: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate cross-validation split for a specific fold.
        
        Args:
            X: Feature tensor
            y: Target tensor
            config: Data configuration
            fold: Current fold number (1-based)
            
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test) for the specified fold
        """
        # Convert tensors to numpy for sklearn operations
        X_np = X.numpy()
        y_np = y.numpy()
        
        # Create splitter
        if config.validation.stratify:
            cv = StratifiedKFold(
                n_splits=config.validation.n_folds,
                shuffle=True,
                random_state=config.random_state
            )
        else:
            cv = KFold(
                n_splits=config.validation.n_folds,
                shuffle=True,
                random_state=config.random_state
            )
        
        # Get indices for the specified fold
        splits = list(cv.split(X_np, y_np))
        train_idx, test_idx = splits[fold - 1]  # fold is 1-based
        
        # Get initial train and test sets for this fold
        X_train_fold = X_np[train_idx]
        y_train_fold = y_np[train_idx]
        X_test_fold = X_np[test_idx]
        y_test_fold = y_np[test_idx]
        
        # Further split training data into train and validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_fold,
            y_train_fold,
            test_size=config.validation.val_size,
            stratify=y_train_fold if config.validation.stratify else None,
            random_state=config.random_state
        )
        
        # Scale features using only training data statistics
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_final)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test_fold)
        
        # Convert back to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_final, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_fold, dtype=torch.float32)
        
        return (
            X_train_tensor, X_val_tensor, X_test_tensor,
            y_train_tensor, y_val_tensor, y_test_tensor
        )
    
    def prepare_data(self, config: 'BSSNNConfig') -> tuple:
        """Load data and store configuration."""
        self.data_config = config.data  # Track config for split_data
        X, y, n_features = self.load_and_prepare_data(config.data)
        config.model.adapt_to_data(n_features)
        return X, y

    @staticmethod
    def _load_data(config: 'DataConfig') -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Load data from file or generate synthetic data."""
        if config.input_path:
            return DataLoader._load_from_file(config)
        else:
            return DataLoader._generate_synthetic(config)
    
    @staticmethod
    def _load_from_file(config: 'DataConfig') -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Load data from file with automatic feature detection."""
        try:
            with pd.read_csv(config.input_path) as df:
                if not config.feature_columns:
                    if config.target_column:
                        config.feature_columns = [col for col in df.columns 
                                                if col != config.target_column]
                    else:
                        config.feature_columns = list(df.columns[:-1])
                        config.target_column = df.columns[-1]
                
                X = torch.tensor(df[config.feature_columns].values, dtype=torch.float32)
                y = torch.tensor(df[config.target_column].values, dtype=torch.float32)
                
                return X, y, X.shape[1]
        except Exception as e:
            print(f"Error loading data from {config.input_path}: {str(e)}")
            raise
    
    @staticmethod
    def _generate_synthetic(config: 'DataConfig') -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Generate synthetic data with adapted parameters."""
        config.adapt_synthetic_params()
        
        print(
            "\n[cyan]Generating synthetic data:[/cyan]",
            f"\nTotal features: {config.synthetic_features}",
            f"\nInformative features: {config.synthetic_informative}",
            f"\nRedundant features: {config.synthetic_redundant}"
        )
        
        X, y = make_classification(
            n_samples=config.synthetic_samples,
            n_features=config.synthetic_features,
            n_informative=config.synthetic_informative,
            n_redundant=config.synthetic_redundant,
            random_state=config.random_state
        )
        
        return torch.tensor(X, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32), \
               X.shape[1]
    
    @staticmethod
    def _prepare_data(
        X: torch.Tensor,
        y: torch.Tensor,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare and split data for training."""
        # Standardize features
        scaler = StandardScaler()
        X_scaled = torch.tensor(scaler.fit_transform(X), dtype=torch.float32)
        
        # Generate split indices
        n_samples = len(X)
        indices = torch.randperm(n_samples)
        split_idx = int(n_samples * (1 - test_size))
        
        # Split data
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        X_train = X_scaled[train_indices]
        X_val = X_scaled[val_indices]
        y_train = y[train_indices]
        y_val = y[val_indices]
        
        print(
            "\n[green]Data preparation completed:[/green]",
            f"\nTraining samples: {len(X_train)}",
            f"\nValidation samples: {len(X_val)}"
        )
        
        return X_train, X_val, y_train, y_val