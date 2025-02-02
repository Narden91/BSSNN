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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
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
        """Initialize network weights with improved stability.
        
        Uses Kaiming initialization for ReLU layers and Xavier for linear layers.
        Initializes gates with small weights to start close to identity mapping.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.linear_path:
                    # Xavier for linear pathway
                    nn.init.xavier_uniform_(module.weight)
                else:
                    # He initialization for nonlinear pathway
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                    
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize gate layers with small weights
        for gate_layer in self.gate:
            if isinstance(gate_layer, nn.Linear):
                # Initialize close to 0.5 to start with equal mixing
                nn.init.constant_(gate_layer.weight, 0.0)
                if gate_layer.bias is not None:
                    nn.init.constant_(gate_layer.bias, 0.0)
        
        # Initialize skip connection scale with proper bounds
        with torch.no_grad():
            self.skip_scale.data.fill_(0.5)
                    
    def _combine_features(self, x: torch.Tensor) -> torch.Tensor:
        """Combine linear and nonlinear features with improved stability.
        
        Args:
            x: Input tensor
        
        Returns:
            Combined features tensor
        """
        # Linear pathway with normalization
        linear_features = self.linear_path(x)
        linear_features = F.layer_norm(linear_features, linear_features.shape[1:])
        
        # Nonlinear pathway
        nonlinear_features = self.feature_extractor(x)
        
        # Concatenate features for gating
        combined = torch.cat([linear_features, nonlinear_features], dim=1)
        
        # Compute mixing weights with gradient clipping
        gate_weights = self.gate(combined)
        gate_weights = torch.clamp(gate_weights, 0.1, 0.9)  # Prevent extreme values
        
        # Mix features with normalized gate weights
        mixed_features = (gate_weights * linear_features + 
                        (1 - gate_weights) * nonlinear_features)
        
        # Apply bounded skip connection
        skip_scale = torch.sigmoid(self.skip_scale)  # Bound between 0 and 1
        skip_connection = self.skip_projection(x)
        
        return mixed_features + skip_scale * skip_connection
    
    def compute_marginal_probability(self, log_joint: torch.Tensor) -> torch.Tensor:
        """Compute log marginal probability with enhanced numerical stability.
        
        This implementation uses the log-sum-exp trick with double precision and 
        careful handling of edge cases. It maintains numerical stability even with
        very small or large probabilities.
        
        Args:
            log_joint: Unnormalized log joint probabilities
            
        Returns:
            Log marginal probabilities with guaranteed numerical stability
        """
        # Convert to double precision for stability
        log_joint = log_joint.double()
        
        # Find maximum value for numerical stability
        max_val = torch.max(log_joint, dim=1, keepdim=True)[0]
        
        # Compute log-sum-exp with stable subtraction
        stable_exp = torch.exp(log_joint - max_val)
        marginal = max_val + torch.log(stable_exp.sum(dim=1, keepdim=True))
        
        # Handle potential infinities and NaNs
        marginal = torch.where(
            torch.isinf(marginal) | torch.isnan(marginal),
            max_val,
            marginal
        )
        
        return marginal.float()
    
    def _get_feature_importance(self) -> torch.Tensor:
        """Calculate comprehensive feature importance scores.
        
        This implementation considers:
        1. Direct linear pathway contributions
        2. Nonlinear pathway interactions
        3. Skip connection influence
        4. Gate utilization patterns
        5. Feature interaction effects
        
        Returns:
            Tensor containing comprehensive importance scores for each feature
        """
        with torch.no_grad():
            # Get linear pathway contributions
            linear_weights = torch.abs(self.linear_path.weight)
            linear_importance = linear_weights.mean(dim=0)
            
            # Get nonlinear pathway first layer weights
            nonlinear_weights = torch.abs(self.feature_extractor[0].weight)
            nonlinear_importance = nonlinear_weights.mean(dim=0)
            
            # Get skip connection influence
            skip_weights = torch.abs(self.skip_projection.weight)
            skip_importance = skip_weights.mean(dim=0)
            
            # Calculate gate utilization for each feature
            gate_weights = torch.abs(self.gate[0].weight)
            gate_importance = gate_weights[:self.hidden_size].mean(dim=0)
            
            # Combine importance scores with learned weights
            total_importance = (
                linear_importance +
                nonlinear_importance +
                self.skip_scale.sigmoid() * skip_importance +
                gate_importance
            )
            
            # Add second-order feature interaction terms
            feature_interactions = torch.mm(
                nonlinear_weights.t(),
                nonlinear_weights
            ).diagonal()
            
            # Normalize importance scores
            final_importance = (total_importance + 0.1 * feature_interactions)
            final_importance = final_importance / final_importance.sum()
            
            return final_importance
    
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
        
        # Compute log probabilities with stability checks
        log_joint = self.joint_network(features)
        
        # Apply log-space computations
        log_marginal = self.compute_marginal_probability(log_joint)
        log_conditional = log_joint - log_marginal
        
        # Convert to probabilities with stable exp
        probs = torch.exp(log_conditional)
        
        # Apply graduated clamping for extreme values
        eps = torch.finfo(probs.dtype).eps
        probs = torch.where(
            probs < eps,
            torch.full_like(probs, eps) + eps * torch.rand_like(probs),
            probs
        )
        probs = torch.where(
            probs > 1 - eps,
            torch.full_like(probs, 1 - eps) - eps * torch.rand_like(probs),
            probs
        )
        
        outputs = {
            'log_joint': log_joint,
            'log_marginal': log_marginal,
            'log_conditional': log_conditional
        }
        
        return probs[:, 1], outputs