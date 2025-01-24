from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import torch
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from rich import print
from bssnn.config.config import DataConfig


class DataLoader:
    """Data loading and preparation class that handles all data operations."""
    
    @staticmethod
    def load_and_prepare_data(config: 'DataConfig') -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Main entry point for data loading and preparation.
        
        Args:
            config: Data configuration object
            
        Returns:
            Tuple containing (features, targets, number of features)
        """
        # Load raw data
        X, y, n_features = DataLoader._load_data(config)
        
        # Prepare and split data
        data_splits = DataLoader._prepare_data(
            X, y,
            test_size=config.test_size,
            random_state=config.random_state
        )
        
        return data_splits, n_features
    
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
        df = pd.read_csv(config.input_path)
        
        # Auto-detect features if not specified
        if not config.feature_columns:
            if config.target_column:
                config.feature_columns = [col for col in df.columns 
                                        if col != config.target_column]
            else:
                config.feature_columns = list(df.columns[:-1])
                config.target_column = df.columns[-1]
                print("[cyan]Auto-selected target column:[/cyan]", config.target_column)
        
        print("\n[green]Features loaded:[/green]", ", ".join(config.feature_columns))
        
        X = torch.tensor(df[config.feature_columns].values, dtype=torch.float32)
        y = torch.tensor(df[config.target_column].values, dtype=torch.float32)
        
        return X, y, X.shape[1]
    
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