# import torch
# from typing import Tuple
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


# def prepare_data(
#     X: torch.Tensor,
#     y: torch.Tensor,
#     test_size: float = 0.2,
#     random_state: int = 42
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Prepare data for training.
    
#     Args:
#         X: Feature tensor
#         y: Target tensor
#         test_size: Proportion of data to use for validation
#         random_state: Random seed for reproducibility
        
#     Returns:
#         Tuple of (X_train, X_val, y_train, y_val)
#     """
#     # Convert to numpy for sklearn operations
#     X_np = X.numpy()
#     y_np = y.numpy()
    
#     # Split data
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_np, y_np,
#         test_size=test_size,
#         random_state=random_state
#     )
    
#     # Scale features
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_val = scaler.transform(X_val)
    
#     # Convert back to tensors
#     X_train = torch.tensor(X_train, dtype=torch.float32)
#     X_val = torch.tensor(X_val, dtype=torch.float32)
#     y_train = torch.tensor(y_train, dtype=torch.float32)
#     y_val = torch.tensor(y_val, dtype=torch.float32)
    
#     return X_train, X_val, y_train, y_val