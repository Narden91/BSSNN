from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateSpaceBSSNN(nn.Module):
    """Enhanced Bayesian State-Space Neural Network with explicit state dynamics."""
    
    def __init__(self, input_size: int, hidden_size: int, state_size: int = 32,
                 dropout_rate: float = 0.2, num_state_layers: int = 2):
        """Initialize the enhanced BSSNN model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            state_size: Dimension of state space
            dropout_rate: Dropout probability
            num_state_layers: Number of state transition layers
        """
        super(StateSpaceBSSNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_state_layers = num_state_layers
        
        # Joint pathway components
        self.fc1_joint = nn.Linear(input_size, hidden_size)
        self.state_transitions_joint = nn.ModuleList([
            StateTransitionLayer(hidden_size, state_size)
            for _ in range(num_state_layers)
        ])
        self.dropout_joint = nn.Dropout(dropout_rate)
        self.batch_norm_joint = nn.BatchNorm1d(hidden_size)
        self.fc2_joint = nn.Linear(hidden_size, 1)
        
        # Marginal pathway components
        self.fc1_marginal = nn.Linear(input_size, hidden_size)
        self.state_transitions_marginal = nn.ModuleList([
            StateTransitionLayer(hidden_size, state_size)
            for _ in range(num_state_layers)
        ])
        self.dropout_marginal = nn.Dropout(dropout_rate)
        self.batch_norm_marginal = nn.BatchNorm1d(hidden_size)
        self.fc2_marginal = nn.Linear(hidden_size, 1)
        
        # Final activation
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def _forward_pathway(self, x: torch.Tensor, 
                        transitions: nn.ModuleList,
                        fc1: nn.Linear,
                        dropout: nn.Dropout,
                        batch_norm: nn.BatchNorm1d,
                        fc2: nn.Linear) -> torch.Tensor:
        """Forward pass through a pathway with state transitions.
        
        Args:
            x: Input tensor
            transitions: List of state transition layers
            fc1: First fully connected layer
            dropout: Dropout layer
            batch_norm: Batch normalization layer
            fc2: Second fully connected layer
            
        Returns:
            Output tensor
        """
        # Initial projection
        h = fc1(x)
        h = batch_norm(h)
        h = F.relu(h)
        
        # Initialize state
        state = torch.zeros(x.size(0), self.state_size, device=x.device)
        
        # Process through state transition layers
        for transition in transitions:
            h, state = transition(h, state)
        
        # Final processing
        h = dropout(h)
        out = fc2(h)
        return out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing P(y|X) using state-space dynamics.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Conditional probability P(y|X) as tensor of shape (batch_size, 1)
        """
        # Compute joint pathway
        joint = self._forward_pathway(
            x,
            transitions=self.state_transitions_joint,
            fc1=self.fc1_joint,
            dropout=self.dropout_joint,
            batch_norm=self.batch_norm_joint,
            fc2=self.fc2_joint
        )
        
        # Compute marginal pathway
        marginal = self._forward_pathway(
            x,
            transitions=self.state_transitions_marginal,
            fc1=self.fc1_marginal,
            dropout=self.dropout_marginal,
            batch_norm=self.batch_norm_marginal,
            fc2=self.fc2_marginal
        )
        
        # Compute conditional probability
        conditional = joint - marginal
        return self.sigmoid(conditional)


class StateTransitionLayer(nn.Module):
    """State transition layer implementing SSM dynamics."""
    
    def __init__(self, hidden_size: int, state_size: int):
        """Initialize state transition layer.
        
        Args:
            hidden_size: Size of hidden representations
            state_size: Size of state vectors
        """
        super(StateTransitionLayer, self).__init__()
        
        # State update components
        self.state_update = nn.Linear(state_size + hidden_size, state_size)
        self.input_proj = nn.Linear(hidden_size, state_size)
        self.output_proj = nn.Linear(state_size, hidden_size)
        
        # Gating mechanism
        self.update_gate = nn.Linear(state_size + hidden_size, state_size)
        self.reset_gate = nn.Linear(state_size + hidden_size, state_size)
        
    def forward(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass implementing state-space dynamics.
        
        Args:
            x: Input tensor
            state: Previous state tensor
            
        Returns:
            Tuple of (output tensor, new state tensor)
        """
        # Combine input and state for gate computation
        combined = torch.cat([x, state], dim=1)
        
        # Update and reset gates
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        
        # Compute candidate state
        reset_state = reset * state
        state_input = torch.cat([x, reset_state], dim=1)
        candidate = torch.tanh(self.state_update(state_input))
        
        # Update state
        new_state = (1 - update) * state + update * candidate
        
        # Project state to output space
        output = self.output_proj(new_state)
        output = x + output  # Residual connection
        
        return output, new_state