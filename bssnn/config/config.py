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
    input_size: int
    hidden_size: int
    state_size: int = 32
    num_state_layers: int = 2
    dropout_rate: float = 0.2
    weight_decay: float = 0.01
    model_type: str = "bssnn"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    def __post_init__(self):
        if self.input_size <= 0 or self.hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive integers")

    def adapt_to_data(self, n_features: int):
        """Adapt model configuration to data dimensions."""
        self.input_size = n_features
        if self.hidden_size is None:
            self.hidden_size = max(64, min(256, 2 * n_features))
        if self.state_size is None:
            self.state_size = max(32, self.hidden_size // 2)
        print(f"Setting hidden size to {self.hidden_size} and state size to {self.state_size}")


@dataclass
class TrainingConfig:
    """Configuration for model training process."""
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    display_metrics: List[str] = field(default_factory=lambda: ['accuracy', 'f1_score', 'auc_roc', 'calibration_error'])


@dataclass
class DataConfig:
    """Configuration for data handling."""
    input_path: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    target_column: Optional[str] = None
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    random_state: int = 42
    
    # Synthetic data parameters (used if input_path is None)
    synthetic_samples: int = 1000
    synthetic_features: Optional[int] = None  # Will match model input_size if not specified
    synthetic_informative: Optional[int] = None  # Will be calculated as 50% of features
    synthetic_redundant: Optional[int] = None  # Will be calculated as 20% of features

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