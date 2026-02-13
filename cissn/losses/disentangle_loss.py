import torch
import torch.nn as nn

class DisentanglementLoss(nn.Module):
    """
    Unsupervised loss to encourage disentangled state representations.
    
    Components:
    1. Covariance regularization (independence)
    2. Temporal consistency (slow vs fast dynamics)
    """
    
    def __init__(
        self,
        lambda_cov: float = 1.0,
        lambda_temporal: float = 0.5
    ):
        super().__init__()
        self.lambda_cov = lambda_cov
        self.lambda_temporal = lambda_temporal
        
    def covariance_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Penalize off-diagonal covariance (encourage independence).
        
        Args:
            states: (batch, seq_len, state_dim)
        """
        # Flatten batch and time
        batch_size, seq_len, state_dim = states.shape
        flat_states = states.reshape(-1, state_dim)  # (batch*seq_len, state_dim)
        
        # Center the states
        mean = flat_states.mean(dim=0, keepdim=True)
        centered = flat_states - mean
        
        # Compute covariance matrix
        n = flat_states.size(0)
        if n < 2:
            return torch.tensor(0.0, device=states.device)
            
        cov = torch.mm(centered.t(), centered) / (n - 1)
        
        # Penalize off-diagonal elements
        # Create mask for off-diagonal
        eye = torch.eye(state_dim, device=states.device)
        off_diag_cov = cov * (1 - eye)
        
        return torch.norm(off_diag_cov, p='fro') ** 2
    
    def temporal_consistency_loss(self, states: torch.Tensor) -> torch.Tensor:
        """
        Encourage appropriate temporal dynamics per dimension.
        
        - Level (dim 0): Variational constraint (should vary slowly)
        - Trend (dim 1): Smoothness constraint
        - Seasonal (dim 2): Periodic constraint (not strictly enforced here, relied on structure)
        - Residual (dim 3): Variance constraint (should be small)
        """
        if states.shape[1] < 2:
            return torch.tensor(0.0, device=states.device)
            
        # Diff across time
        diffs = states[:, 1:, :] - states[:, :-1, :]
        
        # Level: slow varying -> minimize diff
        level_loss = torch.mean(diffs[:, :, 0] ** 2)
        
        # Trend: smooth -> minimize second derivative (diff of diffs)
        if states.shape[1] > 2:
            diff2 = diffs[:, 1:, 1] - diffs[:, :-1, 1]
            trend_loss = torch.mean(diff2 ** 2)
        else:
            trend_loss = torch.mean(diffs[:, :, 1] ** 2)
            
        # Residual: minimize magnitude (sparsity)
        residual_loss = torch.mean(states[:, :, 3] ** 2)
        
        return level_loss + trend_loss + residual_loss

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute total disentanglement loss.
        
        Args:
            states: (batch, seq_len, state_dim)
        """
        l_cov = self.covariance_loss(states)
        l_temp = self.temporal_consistency_loss(states)
        
        return self.lambda_cov * l_cov + self.lambda_temporal * l_temp
