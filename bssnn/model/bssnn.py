import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class BSSNN(nn.Module):
    """Bayesian State-Space Neural Network (BSSNN) with enhanced architecture for linear and nonlinear patterns.
    
    This implementation uses a hybrid architecture that combines:
    1. A linear pathway that preserves direct linear relationships
    2. A nonlinear pathway that captures complex patterns
    3. Skip connections to maintain gradient flow
    4. Gated mixing of linear and nonlinear components
    """
    
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float = 0.2):
        """Initialize the enhanced BSSNN model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            dropout_rate: Dropout probability
        """
        super(BSSNN, self).__init__()
        
        # Linear pathway
        self.linear_path = nn.Linear(input_size, hidden_size)
        
        # Add skip connection projection
        self.skip_projection = nn.Linear(input_size, hidden_size)
        
        # Nonlinear pathway with feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Gating mechanism for mixing linear and nonlinear features
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # Joint probability network
        self.joint_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 2)
        )
        
        # Skip connection scaling factor (learnable)
        self.skip_scale = nn.Parameter(torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using a combination of He and Xavier initialization.
        
        We use different initialization strategies for different parts of the network:
        - Xavier uniform for linear pathways to maintain variance
        - He initialization for nonlinear pathways to handle activation functions
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if getattr(module, 'bias', None) is not None:
                    nn.init.zeros_(module.bias)
                    
                # Use Xavier for linear path and He for nonlinear path
                if module is self.linear_path:
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    
    def _combine_features(self, x: torch.Tensor) -> torch.Tensor:
        """Combine linear and nonlinear features using gated mechanism.
        
        The key change is that we now project the skip connection to match
        the hidden dimension before adding it to the mixed features.
        """
        # Linear pathway
        linear_features = self.linear_path(x)
        
        # Nonlinear pathway
        nonlinear_features = self.feature_extractor(x)
        
        # Concatenate features for gating
        combined = torch.cat([linear_features, nonlinear_features], dim=1)
        
        # Compute mixing weights
        gate_weights = self.gate(combined)
        
        # Mix features with learned gating
        mixed_features = (gate_weights * linear_features + 
                         (1 - gate_weights) * nonlinear_features)
        
        # Project skip connection to match hidden dimension
        skip_connection = self.skip_projection(x)
        
        # Add skip connection with learnable scaling
        return mixed_features + self.skip_scale * skip_connection
    
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
        """Compute consistency loss to enforce Bayesian probability rules."""
        # Get probabilities from logits
        probs = torch.softmax(log_joint, dim=1)
        
        # Check probability sum constraint (should be close to 1)
        prob_sum = probs.sum(dim=1)
        sum_constraint_loss = F.mse_loss(prob_sum, torch.ones_like(prob_sum))
        
        # Non-negativity is ensured by softmax
        non_negativity_loss = torch.mean(F.relu(-probs))
        
        # Scale losses
        consistency_loss = 0.1 * (sum_constraint_loss + non_negativity_loss)
        
        return consistency_loss, {
            'sum_constraint_loss': sum_constraint_loss.item(),
            'non_negativity_loss': non_negativity_loss.item()
        }
    
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
        
        The forward pass ensures proper probability relationships while handling both
        linear and nonlinear patterns through:
        1. Feature extraction with parallel linear and nonlinear paths
        2. Gated feature combination
        3. Skip connection for preserving linear relationships
        4. Proper probability normalization
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Tuple of (conditional probabilities, additional outputs)
        """
        # Extract and combine features
        features = self._combine_features(x)
        
        # Compute log joint probabilities (logits)
        log_joint = self.joint_network(features)
        
        # Compute probabilities with numerical stability
        probs = torch.softmax(log_joint, dim=1)
        probs = probs.clamp(min=1e-7, max=1-1e-7)
        
        # Compute consistency metrics
        consistency_loss, consistency_info = self.compute_consistency_loss(log_joint)
        
        # Store outputs
        outputs = {
            'log_joint': log_joint,
            'consistency_loss': float(consistency_loss),
            'uncertainty': float(self.compute_uncertainty(log_joint).mean().item()),
            'sum_constraint': float(consistency_info['sum_constraint_loss']),
            'non_negativity': float(consistency_info['non_negativity_loss']),
            'gate_values': self.gate[0].weight.abs().mean().item()
        }
        
        return probs[:, 1], outputs