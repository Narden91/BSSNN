import torch.nn as nn
from ..config.config import BSSNNConfig
from ..model.bssnn import BSSNN
from ..model.state_space_bssnn import StateSpaceBSSNN

def create_model(config: BSSNNConfig) -> nn.Module:
    """Create appropriate model based on configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Initialized model instance
    """
    if config.model.model_type.lower() == "state_space_bssnn":
        return StateSpaceBSSNN(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            state_size=config.model.state_size,
            dropout_rate=config.model.dropout_rate,
            num_state_layers=config.model.num_state_layers
        )
    else:
        return BSSNN(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            dropout_rate=config.model.dropout_rate
        )