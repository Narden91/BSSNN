import torch
import torch.nn as nn


class BSSNN(nn.Module):
    """Bayesian State-Space Neural Network (BSSNN) model.
    
    Implements a neural network that explicitly models joint and marginal probabilities
    for improved interpretability and probabilistic predictions.
    """
    
    def __init__(self, input_size: int, hidden_size: int):
        """Initialize the BSSNN model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
        """
        super(BSSNN, self).__init__()
        
        # Joint pathway: P(y, X)
        self.fc1_joint = nn.Linear(input_size, hidden_size)
        self.relu_joint = nn.ReLU()
        self.fc2_joint = nn.Linear(hidden_size, 1)
        
        # Marginal pathway: P(X)
        self.fc1_marginal = nn.Linear(input_size, hidden_size)
        self.relu_marginal = nn.ReLU()
        self.fc2_marginal = nn.Linear(hidden_size, 1)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Conditional probability P(y|X) as tensor of shape (batch_size, 1)
        """
        # Joint probability computation
        joint = self.relu_joint(self.fc1_joint(x))
        joint = self.fc2_joint(joint)
        
        # Marginal probability computation
        marginal = self.relu_marginal(self.fc1_marginal(x))
        marginal = self.fc2_marginal(marginal)
        
        # Bayesian division: P(y|X) = P(y, X) / P(X)
        conditional = joint - marginal  # Log-space division
        return self.sigmoid(conditional)