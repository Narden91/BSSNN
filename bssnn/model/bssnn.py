import torch
import torch.nn as nn
import torch.nn.functional as F


class BSSNN(nn.Module):
    """Bayesian State-Space Neural Network (BSSNN) model.
    
    Implements a neural network that explicitly models joint and marginal probabilities
    for improved interpretability and probabilistic predictions.
    """
    
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float = 0.2):
        """Initialize the BSSNN model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            dropout_rate: Dropout probability (default: 0.2)
        """
        super(BSSNN, self).__init__()
        
        # Joint pathway: P(y, X)
        self.joint_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1)
        )
        
        # Marginal pathway: P(X)
        self.marginal_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Conditional probability P(y|X) as tensor of shape (batch_size, 1)
        """
        # Joint probability computation
        joint = self.joint_network(x)
        
        # Marginal probability computation
        marginal = self.marginal_network(x)
        
        # Bayesian division: P(y|X) = P(y, X) / P(X)
        conditional = joint - marginal  # Log-space division
        return self.sigmoid(conditional)