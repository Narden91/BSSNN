import sys
import torch

sys.dont_write_bytecode = True

from rich import print
from rich.console import Console
from bssnn.utils.file_utils import create_timestamped_dir
from bssnn.training.cross_validation import run_cross_validation
from bssnn.training.trainer import evaluate_on_test_set, run_final_model_training
from bssnn.training.metrics import calculate_metrics
from bssnn.explainability.explainer import run_explanations
from bssnn.utils.data_loader import DataLoader
from bssnn.config.config import BSSNNConfig
from bssnn.visualization.visualization import CrossValidationProgress, setup_rich_logging


console = Console()

def main(config_path: str):
    try:
        # Create output directory
        output_dir = create_timestamped_dir("results")
        config = BSSNNConfig.from_yaml(config_path)
        
        # Set up logging
        setup_rich_logging(output_dir / "experiment.log")
        console.print("\n[bold blue]Starting Classification Experiment[/bold blue]")
        
        # Load and prepare data
        data_loader = DataLoader()
        X_full, y_full = data_loader.prepare_data(config)  # Required to set data_config
        
        # Split into train_val and test sets
        X_train_val, X_test, y_train_val, y_test = data_loader.split_data(
            X_full, y_full,
            test_size=0.2,
            random_state=config.data.random_state
        )

        # Run cross-validation on train+val data only
        console.print("\n[bold blue]Running Cross-validation[/bold blue]")
        best_model, cv_metrics, final_scaler = run_cross_validation(config, X_train_val, y_train_val, output_dir)
        
        X_test_scaled = torch.tensor(
            final_scaler.transform(X_test.numpy()),  
            dtype=torch.float32
        )
        
        # Train final model on ALL train+val data (no validation split)
        console.print("\n[bold blue]Training Final Model[/bold blue]")
        final_model = run_final_model_training(config, X_train_val, y_train_val, output_dir)
        
        # Evaluate on test set
        test_loss, test_metrics = evaluate_on_test_set(final_model, X_test, y_test)
        console.print(f"\n[bold]Test Set Metrics:[/bold]\n{test_metrics}")
        
        # Generate explanations using TEST set
        if config.explainability.enabled and final_model is not None:
            console.print("\n[bold blue]Generating Model Explanations[/bold blue]")
            explanations_dir = output_dir / config.output.explanations_dir
            run_explanations(
                config=config,
                model=final_model,
                X_train=X_train_val,  # Training data (no test leakage)
                X_val=X_test,         # Explanations on test set
                save_dir=str(explanations_dir)
            )
        
        # Save the best model if requested
        if config.output.save_model and best_model is not None:
            model_path = output_dir / config.output.model_dir / 'best_model.pt'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model state and configuration
            torch.save({
                'model_state_dict': best_model.state_dict(),
                'config': config,
                'cv_metrics': cv_metrics
            }, model_path)
            
            console.print(f"\n[green]Model saved to:[/green] {model_path}")
        elif config.output.save_model:
            console.print("\n[yellow]No model to save as best_model is None.[/yellow]")
        
        # Save final configuration
        config.save_final_config(output_dir)
        
        console.print("\n[green]Experiment completed successfully![/green]")
        return best_model, cv_metrics
        
    except Exception as e:
        console.print(f"\n[red]Error during execution: {str(e)}[/red]")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        console.print("[red]Please provide the path to the configuration file.[/red]")
        console.print("Usage: python basic_classification.py <config_path>")
        sys.exit(1)
    
    main(sys.argv[1])