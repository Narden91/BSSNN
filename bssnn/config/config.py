from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from rich import print 
import yaml


@dataclass
class ValidationConfig:
    """Configuration for validation strategy."""
    test_size: float = 0.2
    val_size: float = 0.2
    n_folds: int = 5
    stratify: bool = True

@dataclass
class ModelConfig:
    """Configuration for BSSNN model architecture.
    
    This class defines the model architecture parameters with comprehensive validation
    for each field. The configuration supports both linear and nonlinear components
    through carefully tuned hyperparameters.
    
    Attributes:
        input_size: Number of input features (can be None initially)
        hidden_size: Size of hidden layers
        state_size: Dimension of state space
        num_state_layers: Number of state transition layers
        dropout_rate: Dropout probability
        weight_decay: L2 regularization factor
        model_type: Type of BSSNN model ("bssnn" or "state_space_bssnn")
        early_stopping_patience: Number of epochs to wait before early stopping
        early_stopping_min_delta: Minimum change in validation loss for early stopping
        consistency_weight: Weight for the consistency loss term
        output_classes: Number of output classes
        kl_weight: Weight for the KL divergence loss term (default: 0.01)
            Controls the strength of the regularization towards the prior distribution.
            Smaller values allow more flexible learning, while larger values enforce
            stronger regularization.
        prior_strength: Strength of the uniform prior (default: 0.5)
            Controls how strongly the model is biased towards uniform predictions.
            Values closer to 1.0 enforce stronger uniformity, while values closer to
            0.0 allow more peaked distributions.
    """
    input_size: Optional[int]
    hidden_size: int
    state_size: int = 32
    num_state_layers: int = 2
    dropout_rate: float = 0.2
    weight_decay: float = 0.01
    model_type: str = "bssnn"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    consistency_weight: float = 0.1
    output_classes: int = 2
    kl_weight: float = 0.01
    prior_strength: float = 0.5

    def __post_init__(self):
        """Validate configuration parameters after initialization.
        
        This method ensures all parameters are within valid ranges and logically
        consistent with each other. It performs comprehensive validation of both
        architectural parameters and learning hyperparameters.
        
        Raises:
            ValueError: If any parameter is invalid or inconsistent
        """
        # Existing validations
        if self.input_size is not None and self.input_size <= 0:
            raise ValueError("input_size must be positive when specified")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.state_size <= 0:
            raise ValueError("state_size must be positive")
        if self.num_state_layers <= 0:
            raise ValueError("num_state_layers must be positive")
        if not (0 <= self.dropout_rate <= 1):
            raise ValueError("dropout_rate must be between 0 and 1")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.early_stopping_min_delta <= 0:
            raise ValueError("early_stopping_min_delta must be positive")
        if self.model_type not in ["bssnn", "state_space_bssnn"]:
            raise ValueError("model_type must be either 'bssnn' or 'state_space_bssnn'")            
        if self.kl_weight < 0:
            raise ValueError("kl_weight must be non-negative")
        if not (0 <= self.prior_strength <= 1):
            raise ValueError("prior_strength must be between 0 and 1")
        
        print("[bold green]Model configuration validated successfully[/bold green]")
    
    def adapt_to_data(self, n_features: int):
        """Adapt model configuration to data dimensions.
        
        Args:
            n_features: Number of input features from the data
            
        Raises:
            ValueError: If n_features is invalid
        """
        if n_features <= 0:
            raise ValueError("Number of features must be positive")
            
        self.input_size = n_features


@dataclass
class TrainingConfig:
    """Configuration for model training process."""
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    display_metrics: List[str] = field(default_factory=lambda: ['accuracy', 'f1_score', 'auc_roc', 'calibration_error'])


@dataclass
class DataConfig:
    """Configuration for data handling with enhanced column exclusion support."""
    input_path: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    target_column: Optional[str] = None
    exclude_columns: Optional[List[str]] = field(default_factory=list)  # New field for columns to exclude
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    random_state: int = 42
    
    # Synthetic data parameters
    synthetic_samples: int = 1000
    synthetic_features: Optional[int] = None
    synthetic_informative: Optional[int] = None
    synthetic_redundant: Optional[int] = None

    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Convert single exclude column to list if needed
        if isinstance(self.exclude_columns, str):
            self.exclude_columns = [self.exclude_columns]
        elif self.exclude_columns is None:
            self.exclude_columns = []
            
        # Ensure exclude_columns is always a list
        self.exclude_columns = list(self.exclude_columns)

    def adapt_synthetic_params(self, n_features: Optional[int] = None):
        """Adapt synthetic data parameters based on feature count."""
        if n_features is not None:
            self.synthetic_features = n_features
        elif self.synthetic_features is None:
            self.synthetic_features = 10  # Default if no other information available
            
        if self.synthetic_informative is None:
            self.synthetic_informative = max(1, self.synthetic_features // 2)
            
        if self.synthetic_redundant is None:
            self.synthetic_redundant = max(1, self.synthetic_features // 5)
            
        # Ensure parameters are consistent
        total_special_features = self.synthetic_informative + self.synthetic_redundant
        if total_special_features > self.synthetic_features:
            ratio = self.synthetic_features / total_special_features
            self.synthetic_informative = int(self.synthetic_informative * ratio)
            self.synthetic_redundant = int(self.synthetic_redundant * ratio)


@dataclass
class ExplainabilityConfig:
    """Configuration for model explainability."""
    enabled: bool = True
    background_samples: int = 100
    max_val_samples: int = 100
    feature_names: Optional[List[str]] = None


@dataclass
class OutputConfig:
    """Configuration for output handling."""
    base_dir: str = "results"
    model_dir: str = "models"
    explanations_dir: str = "explanations"
    logs_dir: str = "logs"
    save_model: bool = True


class BSSNNConfig:
    """Master configuration class for BSSNN model system.
    
    This class manages all configuration aspects of the BSSNN model including:
    - Model architecture parameters
    - Training hyperparameters
    - Data processing settings
    - Explainability options
    - Output and logging preferences
    
    The configuration can be loaded from a YAML file or created with default values.
    It also handles directory setup and configuration validation.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        data_config: DataConfig,
        explainability_config: ExplainabilityConfig,
        output_config: OutputConfig
    ):
        self.model = model_config
        self.training = training_config
        self.data = data_config
        self.explainability = explainability_config
        self.output = output_config
        self.model.kl_weight = 0.01  # Start with small KL weight
        self.model.prior_strength = 0.5  # Weak prior initially
        # Create output directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary output directories."""
        base_path = Path(self.output.base_dir)
        self.directories = {
            'model': 'models',
            'explanations': 'explanations', 
            'logs': 'logs'
        }
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'BSSNNConfig':
        """Create configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create validation config first if it exists
        data_config_dict = config_dict.get('data', {})
        if 'validation' in data_config_dict:
            data_config_dict['validation'] = ValidationConfig(**data_config_dict['validation'])
        
        # Create individual config objects
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**data_config_dict)
        explainability_config = ExplainabilityConfig(**config_dict.get('explainability', {}))
        output_config = OutputConfig(**config_dict.get('output', {}))
        
        return cls(
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            explainability_config=explainability_config,
            output_config=output_config
        )
    
    @classmethod
    def default(cls) -> 'BSSNNConfig':
        """Create default configuration."""
        return cls(
            model_config=ModelConfig(input_size=10),
            training_config=TrainingConfig(),
            data_config=DataConfig(),
            explainability_config=ExplainabilityConfig(),
            output_config=OutputConfig()
        )
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': {
                **{k: v for k, v in self.data.__dict__.items() if k != 'validation'},
                'validation': self.data.validation.__dict__
            },
            'explainability': self.explainability.__dict__,
            'output': self.output.__dict__
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
    def initialize_and_validate(self, config_path: str, output_dir: Path) -> 'BSSNNConfig':
        """Initialize and validate configuration."""
        self.output.base_dir = str(output_dir)
        
        # Validate critical configuration
        if self.output.save_model and not self.output.model_dir:
            raise ValueError("Model directory must be specified when save_model is True")
            
        return self

    def save_final_config(self, output_dir: Path):
        """Save final configuration to output directory."""
        try:
            config_save_path = output_dir / "final_config.yaml"
            self.save(str(config_save_path))
            return config_save_path
        except Exception as e:
            raise ValueError(f"Could not save configuration: {str(e)}")