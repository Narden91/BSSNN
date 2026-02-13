import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Optional, Union, Tuple

class StateConditionalConformal:
    """
    State-Conditional Conformal Prediction (SCCP).
    
    Uses latent states to cluster time steps and compute adaptive prediction intervals.
    """
    
    def __init__(self, alpha: float = 0.1, n_clusters: int = 5):
        """
        Args:
            alpha: Significance level (coverage = 1 - alpha)
            n_clusters: Number of state clusters
        """
        self.alpha = alpha
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.quantiles = {}  # {cluster_id: quantile_value}
        self.calibrated = False
        
    def fit(self, states: Union[torch.Tensor, np.ndarray], residuals: Union[torch.Tensor, np.ndarray]):
        """
        Calibrate the conformal predictor.
        
        Args:
            states: (n_samples, state_dim) - Latent states from calibration set
            residuals: (n_samples,) - Absolute verification residuals |y - y_hat|
        """
        if isinstance(states, torch.Tensor):
            states = states.detach().cpu().numpy()
        if isinstance(residuals, torch.Tensor):
            residuals = residuals.detach().cpu().numpy()
            
        # 1. Cluster states
        self.kmeans.fit(states)
        cluster_labels = self.kmeans.predict(states)
        
        # 2. Compute per-cluster quantiles
        n = len(residuals)
        for k in range(self.n_clusters):
            # Get residuals for this cluster
            mask = (cluster_labels == k)
            cluster_residuals = residuals[mask]
            
            if len(cluster_residuals) > 0:
                # Finite sample correction
                n_k = len(cluster_residuals)
                q_level = np.ceil((n_k + 1) * (1 - self.alpha)) / n_k
                q_level = min(q_level, 1.0)
                
                # Compute quantile
                q_val = np.quantile(cluster_residuals, q_level, method='higher')
                self.quantiles[k] = q_val
            else:
                # Fallback to global quantile if cluster is empty (rare)
                self.quantiles[k] = np.quantile(residuals, 1 - self.alpha, method='higher')
                
        self.calibrated = True
        
    def predict(self, states: Union[torch.Tensor, np.ndarray], point_forecasts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate prediction intervals.
        
        Args:
            states: (n_samples, state_dim)
            point_forecasts: (n_samples, output_dim)
            
        Returns:
            lower_bound: (n_samples, output_dim)
            upper_bound: (n_samples, output_dim)
        """
        if not self.calibrated:
            raise RuntimeError("Conformal predictor not calibrated. Call fit() first.")
            
        if isinstance(states, torch.Tensor):
            states_np = states.detach().cpu().numpy()
        else:
            states_np = states
            
        # Assign clusters
        cluster_labels = self.kmeans.predict(states_np)
        
        # Retrieve quantiles
        q_values = np.array([self.quantiles[k] for k in cluster_labels])
        q_tensor = torch.tensor(q_values, device=point_forecasts.device, dtype=point_forecasts.dtype).unsqueeze(-1)
        
        lower = point_forecasts - q_tensor
        upper = point_forecasts + q_tensor
        
        return lower, upper
