from dataclasses import dataclass
from typing import List, Dict
import torch

@dataclass
class ExplanationResult:
    level_contribution: float
    trend_contribution: float
    seasonal_contribution: float
    residual_contribution: float
    bias: float
    total_prediction: float

class ForecastExplainer:
    """
    Explains forecasts by decomposing them into interpretable state components.
    """
    
    def __init__(self, forecast_head):
        self.forecast_head = forecast_head
        
    def explain(self, state: torch.Tensor) -> List[ExplanationResult]:
        """
        Explain a batch of forecasts.
        
        Args:
            state: (batch, state_dim)
            
        Returns:
            List of ExplanationResult objects
        """
        # Get contributions dictionary from head
        # Keys: 'level', 'trend', 'seasonal', 'residual', 'bias'
        contributions = self.forecast_head.get_contributions(state)
        
        batch_size = state.size(0)
        explanations = []
        
        for i in range(batch_size):
            res = ExplanationResult(
                level_contribution=float(contributions.get('level', torch.zeros(batch_size))[i].item()),
                trend_contribution=float(contributions.get('trend', torch.zeros(batch_size))[i].item()),
                seasonal_contribution=float(contributions.get('seasonal', torch.zeros(batch_size))[i].item()),
                residual_contribution=float(contributions.get('residual', torch.zeros(batch_size))[i].item()),
                bias=float(contributions['bias'].item()),
                total_prediction=float(contributions['total_linear'][i].item()) # Approx
            )
            explanations.append(res)
            
        return explanations
