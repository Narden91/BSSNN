import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from typing import Dict, Any
from pathlib import Path
from rich import print

from bssnn.config.config import BSSNNConfig


class HandwritingDataAdapter:
    """Adapter for processing handwriting classification data."""
    
    def __init__(self, data_path: str, config: BSSNNConfig):
        """Initialize the handwriting data adapter.
        
        Args:
            data_path: Path to the CSV file containing handwriting data
            config: BSSNN configuration object
        """
        self.data_path = Path(data_path)
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        print("\n[Debug] HandwritingDataAdapter initialized with:")
        print(f"Data path: {self.data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """Load the handwriting dataset.
        
        Returns:
            DataFrame containing the loaded data
        
        Raises:
            RuntimeError: If data loading fails
        """
        try:
            df = pd.read_csv(self.data_path)
            print("\n[Debug] Data loading:")
            print(f"DataFrame shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            raise RuntimeError(f"Error loading data from {self.data_path}: {str(e)}")
        
    def _validate_and_get_columns(self, df: pd.DataFrame) -> tuple[str, list[str]]:
        """Validate and determine target and feature columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (target column name, list of feature column names)
            
        Raises:
            ValueError: If column validation fails
        """
        # Get target column
        target_col = self.config.data.target_column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
            
        # Get columns to exclude
        exclude_cols = set(self.config.data.exclude_columns or [])
        exclude_cols.add(target_col)  # Always exclude target column
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"\n[Debug] Column Selection:")
        print(f"Target column: {target_col}")
        print(f"Excluded columns: {exclude_cols}")
        print(f"Number of feature columns: {len(feature_cols)}")
        print(f"Feature columns: {feature_cols}")
        
        return target_col, feature_cols
    
    def preprocess_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess the handwriting data for BSSNN.
        
        Args:
            df: Input DataFrame containing raw data
            
        Returns:
            Tuple of (features array, labels array)
        """
        print("\n[Debug] Starting data preprocessing")
        
        # Validate and get columns
        target_col, feature_cols = self._validate_and_get_columns(df)
        
        # Store feature names for later use
        self.feature_names = feature_cols
        
        # Extract features and labels
        X = df[feature_cols].values
        y = df[target_col].values
        
        print("\n[Debug] Data shapes:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"y unique values: {np.unique(y)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X_scaled, y_encoded
    
    def prepare_data_for_bssnn(self) -> Dict[str, Any]:
        """Prepare data in format suitable for BSSNN model."""
        print("\n[Debug] Starting data preparation for BSSNN")
        
        # Load and preprocess data
        df = self.load_data()
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=self.config.data.random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=self.config.data.random_state
        )
        
        # Prepare return dictionary
        data = {
            'X_train': torch.FloatTensor(X_train),
            'X_val': torch.FloatTensor(X_val),
            'X_test': torch.FloatTensor(X_test),
            'y_train': torch.FloatTensor(y_train),
            'y_val': torch.FloatTensor(y_val),
            'y_test': torch.FloatTensor(y_test),
            'feature_names': self.feature_names,
            'class_names': self.label_encoder.classes_.tolist()
        }
        
        print("\n[Debug] Dataset preparation complete:")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Training samples: {len(data['X_train'])}")
        print(f"Validation samples: {len(data['X_val'])}")
        print(f"Test samples: {len(data['X_test'])}")
        
        return data