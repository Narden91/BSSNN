import torch
import torch.nn as nn
import torch.nn.functional as F


class BSSNN(nn.Module):
    """Bayesian State-Space Neural Network (BSSNN) model.
    
    Implements a neural network that explicitly models joint and marginal probabilities
    for improved interpretability and probabilistic predictions.
    """
    
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float = 0.2):
        """Initialize the enhanced BSSNN model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            dropout_rate: Dropout probability
        """
        super(BSSNN, self).__init__()
        
        # Joint pathway for P(y,X)
        self.joint_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2)  # Output logits for both classes
        )
        
        # Marginal pathway for P(X)
        self.marginal_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
                
    def compute_consistency_loss(self, log_joint: torch.Tensor, log_marginal: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss between joint and marginal probabilities.
        
        Args:
            log_joint: Log joint probabilities for both classes
            log_marginal: Log marginal probability
            
        Returns:
            Consistency loss value
        """
        # Compute implied marginal from joint (log-sum-exp)
        implied_marginal = torch.logsumexp(log_joint, dim=1, keepdim=True)
        return F.mse_loss(implied_marginal, log_marginal)
    
    def compute_uncertainty(self, log_joint: torch.Tensor) -> torch.Tensor:
        """Compute prediction uncertainty using entropy.
        
        Args:
            log_joint: Log joint probabilities
            
        Returns:
            Uncertainty scores
        """
        probs = torch.softmax(log_joint, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        return entropy
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with consistency constraints and uncertainty.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Tuple of (conditional probabilities, additional outputs)
        """
        # Compute log probabilities
        log_joint = self.joint_network(x)
        log_marginal = self.marginal_network(x)
        
        # Compute consistency loss during training
        consistency_loss = self.compute_consistency_loss(log_joint, log_marginal)
        
        # Compute conditional probabilities in log-space
        log_conditional = log_joint - log_marginal
        
        # Convert to probability space with numerical stability
        probs = torch.softmax(log_conditional, dim=1)
        
        # Compute uncertainty
        uncertainty = self.compute_uncertainty(log_joint)
        
        # Return probability of positive class and additional outputs
        outputs = {
            'consistency_loss': consistency_loss,
            'uncertainty': uncertainty,
            'log_joint': log_joint,
            'log_marginal': log_marginal
        }
        
        return probs[:, 1], outputs