import sys
sys.dont_write_bytecode = True

from pathlib import Path
from rich import print
from rich.console import Console
from typing import Dict, Any, Optional

import torch
from bssnn.config.config import BSSNNConfig
from bssnn.utils.file_utils import create_timestamped_dir
from bssnn.training.trainer import run_training, evaluate_on_test_set
from bssnn.training.cross_validation import run_cross_validation
from bssnn.explainability.explainer import run_explanations
from bssnn.training.model_factory import create_model
from bssnn.visualization.visualization import (
    print_test_metrics,
    save_metrics_to_csv,
    CrossValidationProgress
)

from adapter import HandwritingDataAdapter

console = Console()

def setup_experiment(config_path: str) -> tuple[BSSNNConfig, Path]:
    """Set up the experiment configuration and directory."""
    config = BSSNNConfig.from_yaml(config_path)
    config.data.validation.n_folds = 5  # Force 5-fold CV
    output_dir = create_timestamped_dir(config.output.base_dir)
    return config, output_dir

def run_cross_validation_experiment(
    config: BSSNNConfig,
    data: Dict[str, torch.Tensor],
    output_dir: Path
) -> tuple[Optional[Dict[str, Dict[str, float]]], Any]:
    """Run cross-validation experiment to validate model and find best hyperparameters."""
    console.print("\n[bold blue]Running Cross-validation[/bold blue]")
    
    # Combine training and validation sets for CV
    X_train_val = torch.cat([data['X_train'], data['X_val']], dim=0)
    y_train_val = torch.cat([data['y_train'], data['y_val']], dim=0)
    
    # Run cross-validation
    _, cv_metrics, final_scaler = run_cross_validation(
        config=config,
        X=X_train_val,
        y=y_train_val,
        output_dir=output_dir
    )
    
    # Save cross-validation metrics
    cv_metrics_path = output_dir / "cross_validation_metrics.csv"
    save_metrics_to_csv(cv_metrics, cv_metrics_path)
    
    return cv_metrics, final_scaler

def train_final_model(
    config: BSSNNConfig,
    data: Dict[str, torch.Tensor],
    scaler: Any
) -> torch.nn.Module:
    """Train final model using all training data.
    
    This crucial step trains the final model on the complete training dataset
    after cross-validation has validated the approach and hyperparameters.
    """
    console.print("\n[bold blue]Training Final Model on Complete Dataset[/bold blue]")
    
    # Scale features using the scaler from cross-validation
    X_train_full = torch.cat([data['X_train'], data['X_val']], dim=0)
    y_train_full = torch.cat([data['y_train'], data['y_val']], dim=0)
    
    X_train_scaled = torch.tensor(
        scaler.transform(X_train_full.numpy()),
        dtype=torch.float32
    )
    
    # Create and train final model
    final_model = create_model(config)
    
    # Train on full training set
    trainer = run_training(
        config=config,
        model=final_model,
        X_train=X_train_scaled,
        X_val=X_train_scaled,  # Use same data for validation to track training
        y_train=y_train_full,
        y_val=y_train_full,
        is_final=True
    )
    
    return trainer.model

def run_final_evaluation(
    model: torch.nn.Module,
    data: Dict[str, torch.Tensor],
    scaler: Any,
    output_dir: Path
) -> tuple[float, Dict[str, float]]:
    """Evaluate final model on held-out test set."""
    console.print("\n[bold blue]Evaluating Final Model on Test Set[/bold blue]")
    
    # Scale test data using the same scaler
    X_test_scaled = torch.tensor(
        scaler.transform(data['X_test'].numpy()),
        dtype=torch.float32
    )
    
    # Evaluate on test set
    test_loss, test_metrics = evaluate_on_test_set(
        model=model,
        X_test=X_test_scaled,
        y_test=data['y_test']
    )
    
    # Print and save metrics
    print_test_metrics(test_loss, test_metrics)
    metrics_path = output_dir / "test_metrics.csv"
    save_metrics_to_csv({'test_loss': test_loss, **test_metrics}, metrics_path)
    
    return test_loss, test_metrics

def run_handwriting_classification(config_path: str) -> int:
    """Run the complete handwriting classification experiment."""
    try:
        # Setup
        config, output_dir = setup_experiment(config_path)
        console.print("\n[bold blue]Starting Handwriting Classification[/bold blue]")
        
        # Load and prepare data
        data_adapter = HandwritingDataAdapter(
            data_path=config.data.input_path,
            config=config
        )
        data = data_adapter.prepare_data_for_bssnn()
        config.model.input_size = data['X_train'].shape[1]
        
        # Run cross-validation to validate approach
        cv_metrics, final_scaler = run_cross_validation_experiment(
            config=config,
            data=data,
            output_dir=output_dir
        )
        
        # Train final model on all training data
        final_model = train_final_model(
            config=config,
            data=data,
            scaler=final_scaler
        )
        
        # Evaluate on test set
        test_loss, test_metrics = run_final_evaluation(
            model=final_model,
            data=data,
            scaler=final_scaler,
            output_dir=output_dir
        )
        
        # Generate explanations if enabled
        if config.explainability.enabled:
            explanations_dir = output_dir / config.output.explanations_dir
            
            # Update explainability configuration with actual feature names
            config.explainability.feature_names = data['feature_names']
            # console.print(f"\nUsing feature names for explanations:")
            # for i, name in enumerate(data['feature_names']):
            #     console.print(f"  {i+1}. {name}")
            
            explanations = run_explanations(
                config=config,
                model=final_model,
                X_train=data['X_train'],
                X_val=data['X_test'],
                save_dir=str(explanations_dir)
            )
        
        # Save final model if configured
        if config.output.save_model:
            model_path = output_dir / config.output.model_dir / "final_model.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': final_model.state_dict(),
                'config': config,
                'cv_metrics': cv_metrics,
                'test_metrics': test_metrics
            }, model_path)
        
        console.print("\n[bold green]Classification experiment completed successfully![/bold green]")
        return 0
        
    except Exception as e:
        console.print(f"\n[bold red]Error during execution: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        console.print("[red]Please provide path to configuration file.[/red]")
        console.print("Usage: python handwriting_classification.py <config_path>")
        sys.exit(1)
        
    sys.exit(run_handwriting_classification(sys.argv[1]))