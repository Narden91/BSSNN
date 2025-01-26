import sys
import torch

sys.dont_write_bytecode = True

from rich import print
from rich.console import Console
from bssnn.utils.file_utils import create_timestamped_dir
from bssnn.training.cross_validation import run_cross_validation
from bssnn.training.trainer import run_final_model_training
from bssnn.training.metrics import calculate_metrics
from bssnn.explainability.explainer import run_explanations
from bssnn.utils.data_loader import DataLoader
from bssnn.config.config import BSSNNConfig
from bssnn.visualization.visualization import setup_rich_logging

console = Console()

def main(config_path: str):
    """Main execution function for BSSNN training pipeline."""
    try:
        # Create experiment directory
        output_dir = create_timestamped_dir("results")
        console.print(f"\n[cyan]Created results directory:[/cyan] {output_dir}")
        
        # Setup logging
        setup_rich_logging(str(output_dir / 'run.log'))
        
        # Initialize configuration
        config = BSSNNConfig.from_yaml(config_path)
        
        # Create subdirectories within experiment directory
        for subdir in config.directories.values():
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)
            
        config.initialize_and_validate(config_path, output_dir)
        
        # Prepare data
        data_loader = DataLoader()
        X, y = data_loader.prepare_data(config)
        
        # Run cross-validation first to get performance estimates
        cv_metrics, _ = run_cross_validation(config, X, y, output_dir)
        
        # Train final model on full dataset
        console.print("\n[bold cyan]Training Final Model[/bold cyan]")
        final_model = run_final_model_training(config, X, y, output_dir)
        
        # Get final holdout test set for evaluation
        splits = data_loader.get_cross_validation_splits(X, y, config.data, fold=1)
        _, _, X_test, _, _, y_test = splits
        
        # Evaluate final model performance
        final_model.eval()
        with torch.no_grad():
            test_preds = final_model(X_test).squeeze()
            test_metrics = calculate_metrics(y_test, test_preds)
        
        # Save final performance metrics
        metrics_path = output_dir / 'final_model_metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write("Final Model Performance on Test Set\n")
            f.write("="*40 + "\n\n")
            for metric, value in test_metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        console.print("\n[bold cyan]Final Model Performance:[/bold cyan]")
        for metric, value in test_metrics.items():
            console.print(f"{metric}: {value:.4f}")
        
        # Run explanations if enabled
        if config.explainability.enabled:
            explanations_dir = output_dir / 'explanations'
            run_explanations(config, final_model, X, X_test, save_dir=str(explanations_dir))
        
        # Save final model if requested
        if config.output.save_model:
            model_path = output_dir / 'models' / 'final_model.pt'
            torch.save(final_model.state_dict(), model_path)
            console.print(f"\n[green]Model saved to[/green] {model_path}")
        
        # Save configuration
        config_path = config.save_final_config(output_dir)
        console.print(f"\n[green]Configuration saved to[/green] {config_path}")
        console.print("\n[green]Execution completed successfully![/green]")
        
    except Exception as e:
        console.print(f"\n[red]Error during execution: {str(e)}[/red]")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        console.print("[red]Please provide the path to the configuration file.[/red]")
        console.print("Usage: python basic_classification.py <config_path>")
        sys.exit(1)
    
    main(sys.argv[1])