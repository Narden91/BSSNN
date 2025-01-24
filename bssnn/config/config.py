import yaml
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for BSSNN model architecture."""
    input_size: int
    hidden_size: int = 64
    dropout_rate: float = 0.0


@dataclass
class TrainingConfig:
    """Configuration for model training process."""
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    display_metrics: List[str] = None

    def __post_init__(self):
        if self.display_metrics is None:
            self.display_metrics = ['accuracy', 'f1_score', 'auc_roc', 'calibration_error']


@dataclass
class DataConfig:
    """Configuration for data handling."""
    input_path: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    target_column: Optional[str] = None
    test_size: float = 0.2
    random_state: int = 42
    
    # Synthetic data parameters (used if input_path is None)
    synthetic_samples: int = 1000
    synthetic_features: int = 10
    synthetic_informative: int = 5
    synthetic_redundant: int = 2


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
    """Master configuration class for BSSNN system."""
    
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
        
        # Create all required directories
        directories = {
            'model': base_path / self.output.model_dir,
            'explanations': base_path / self.output.explanations_dir,
            'logs': base_path / self.output.logs_dir
        }
        
        for path in directories.values():
            path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'BSSNNConfig':
        """Create configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create individual config objects
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
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
            'data': self.data.__dict__,
            'explainability': self.explainability.__dict__,
            'output': self.output.__dict__
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)