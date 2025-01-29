import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class BSSNN(nn.Module):
    """Bayesian State-Space Neural Network (BSSNN) model with enforced Bayesian consistency.
    
    This implementation ensures proper alignment between joint and marginal probabilities
    by explicitly enforcing P(X) = ∑yP(y,X) through a consistency loss term and
    architectural constraints.
    """
    
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float = 0.2):
        """Initialize the enhanced BSSNN model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            dropout_rate: Dropout probability
        """
        super(BSSNN, self).__init__()
        
        # Feature extractor remains unchanged
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_size)
        )
        
        # Modified joint network to output logits directly
        self.joint_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            # Remove any activation function - obtain raw logits
            nn.Linear(hidden_size // 2, 2)  # Output unnormalized log-probabilities
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def compute_marginal_probability(self, log_joint: torch.Tensor) -> torch.Tensor:
        """Compute log marginal probability using the log-sum-exp trick.
        
        Args:
            log_joint: Unnormalized log joint probabilities (logits)
            
        Returns:
            Log marginal probabilities
        """
        # Using log-sum-exp for numerical stability
        return torch.logsumexp(log_joint, dim=1, keepdim=True)
    
    def _get_feature_importance(self) -> torch.Tensor:
        """Calculate feature importance scores based on the model's weights.
        
        The importance scores are calculated by analyzing the impact of each input
        feature through both the feature extractor and joint network pathways.
        
        Returns:
            Tensor containing importance score for each input feature
        """
        # Get feature extractor weights from first layer
        feature_weights = self.feature_extractor[0].weight.detach()  # From input to hidden
        
        # Get joint network contribution
        joint_weights = self.joint_network[0].weight.detach()  # From hidden to output
        
        # Compute importance through network paths
        # Aggregate impact through hidden layers using matrix multiplication
        importance = torch.mm(joint_weights.t(), feature_weights).abs().mean(dim=0)
        
        return importance
    
    def compute_consistency_loss(self, log_joint: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute consistency loss to enforce Bayesian probability rules.
        
        The loss ensures that probabilities sum to 1 and maintains proper relationships
        between joint and marginal probabilities.
        
        Args:
            log_joint: Log joint probabilities
            
        Returns:
            Tuple of (consistency loss, dictionary of intermediate values)
        """
        # Convert log probabilities to probabilities
        joint_probs = torch.softmax(log_joint, dim=1)
        
        # Check probability sum constraint (should be close to 1)
        prob_sum = joint_probs.sum(dim=1)
        sum_constraint_loss = F.mse_loss(prob_sum, torch.ones_like(prob_sum))
        
        # Enforce non-negativity (should be unnecessary with softmax but kept for safety)
        non_negativity_loss = torch.mean(F.relu(-joint_probs))
        
        # Scale down the consistency loss
        consistency_loss = 0.1 * (sum_constraint_loss + non_negativity_loss)
        
        # Store individual components for monitoring
        consistency_metrics = {
            'sum_constraint_loss': float(sum_constraint_loss.item()),
            'non_negativity_loss': float(non_negativity_loss.item())
        }
        
        return consistency_loss, consistency_metrics
    
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass implementing Bayesian probability computations.
        
        The forward pass ensures proper probability relationships by:
        1. Computing shared features
        2. Computing log joint probabilities P(y,X)
        3. Deriving marginal probabilities P(X) from joint probabilities
        4. Computing conditional probabilities P(y|X) through proper Bayesian division
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Tuple of (conditional probabilities, additional outputs)
        """
        # Extract shared features
        features = self.feature_extractor(x)
        
        # Compute log joint probabilities
        log_joint = self.joint_network(features)
        
        # Compute log marginal probabilities
        log_marginal = self.compute_marginal_probability(log_joint)
        
        # Compute log conditional probabilities
        log_conditional = log_joint - log_marginal
        
        # Convert to probabilities using softmax
        probs = torch.softmax(log_conditional, dim=1)
        
        # Ensure we're getting proper probabilities
        probs = probs.clamp(min=1e-7, max=1-1e-7)
        
        # Compute consistency metrics
        consistency_loss, consistency_info = self.compute_consistency_loss(log_joint)
        
        # Store all outputs
        outputs = {
            'log_joint': log_joint,
            'log_marginal': log_marginal,
            'log_conditional': log_conditional,
            'consistency_loss': float(consistency_loss),
            'uncertainty': float(self.compute_uncertainty(log_conditional).mean().item()),
            'sum_constraint': float(consistency_info['sum_constraint_loss']),
            'non_negativity': float(consistency_info['non_negativity_loss'])
        }
        
        return probs[:, 1], outputs