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
from bssnn.visualization.visualization import CrossValidationProgress, setup_rich_logging

console = Console()

def main(config_path: str):
    try:
        output_dir = create_timestamped_dir("results")
        config = BSSNNConfig.from_yaml(config_path)
        
        data_loader = DataLoader()
        X, y = data_loader.prepare_data(config)
        
        console.print("\n[bold blue]Running Cross-validation[/bold blue]")
        best_model, cv_metrics = run_cross_validation(config, X, y, output_dir)
        
        _, _, X_test, _, _, y_test = data_loader.get_cross_validation_splits(X, y, config.data, fold=1)
        
        console.print("\n[bold blue]Final Model Performance on Test Set[/bold blue]")
        best_model.eval()
        with torch.no_grad():
            test_preds = best_model(X_test).squeeze()
            test_metrics = calculate_metrics(y_test, test_preds)
            
        for metric, value in test_metrics.items():
            console.print(f"{metric}: {value:.4f}")
        
        if config.explainability.enabled:
            console.print("\n[bold blue]Generating Model Explanations[/bold blue]")
            explanations_dir = output_dir / config.output.explanations_dir
            run_explanations(config, best_model, X, X_test, save_dir=str(explanations_dir))
        
        if config.output.save_model:
            model_path = output_dir / config.output.model_dir / 'best_model.pt'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_model.state_dict(), model_path)
            console.print(f"\n[green]Model saved to:[/green] {model_path}")
        
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