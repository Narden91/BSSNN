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
    
    def preprocess_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess the handwriting data for BSSNN.
        
        Args:
            df: Input DataFrame containing raw data
            
        Returns:
            Tuple of (features array, labels array)
        """
        print("\n[Debug] Starting data preprocessing")
        
        # Get target column
        target_col = 'Label' if 'Label' in df.columns else df.columns[-1]
        
        # Get columns to exclude
        exclude_cols = set(self.config.data.exclude_columns or [])
        
        # Generate feature columns, excluding target and specified columns
        feature_cols = [col for col in df.columns 
                       if col != target_col and col not in exclude_cols]
        
        print(f"\n[Debug] Column Selection:")
        print(f"Target column: {target_col}")
        print(f"Excluded columns: {exclude_cols}")
        print(f"Selected feature columns: {feature_cols}")
        
        print(f"\n[Debug] Target column: {target_col}")
        print(f"Number of features: {len(feature_cols)}")
        
        # Extract features and labels
        X = df[feature_cols].values
        y = df[target_col].values
        
        print("\n[Debug] Before preprocessing:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"y unique values: {np.unique(y)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print("\n[Debug] After preprocessing:")
        print(f"X_scaled shape: {X_scaled.shape}")
        print(f"X_scaled range: [{X_scaled.min()}, {X_scaled.max()}]")
        print(f"y_encoded unique values: {np.unique(y_encoded)}")
        
        return X_scaled, y_encoded
    
    def prepare_data_for_bssnn(self) -> Dict[str, Any]:
        """Prepare data in format suitable for BSSNN model.
        
        Returns:
            Dictionary containing:
                - X_train, X_val, X_test: Feature tensors
                - y_train, y_val, y_test: Label tensors
                - feature_names: List of feature names
                - class_names: List of class names
        """
        print("\n[Debug] Starting data preparation for BSSNN")
        
        # Load and preprocess data
        df = self.load_data()
        X, y = self.preprocess_data(df)
        
        # Split data into train, validation, and test sets
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=self.config.data.random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=0.2,
            stratify=y_temp,
            random_state=self.config.data.random_state
        )
        
        # Convert to tensors
        data = {
            'X_train': torch.FloatTensor(X_train),
            'X_val': torch.FloatTensor(X_val),
            'X_test': torch.FloatTensor(X_test),
            'y_train': torch.FloatTensor(y_train),
            'y_val': torch.FloatTensor(y_val),
            'y_test': torch.FloatTensor(y_test),
            'feature_names': [col for col in df.columns if col != 'Label'],
            'class_names': self.label_encoder.classes_.tolist()
        }
        
        print("\n[Debug] Final dataset statistics:")
        print(f"Training set size: {len(data['X_train'])}")
        print(f"Validation set size: {len(data['X_val'])}")
        print(f"Test set size: {len(data['X_test'])}")
        print(f"Number of features: {data['X_train'].shape[1]}")
        print(f"Number of classes: {len(data['class_names'])}")
        
        return data