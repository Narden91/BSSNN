import sys
sys.dont_write_bytecode = True

from rich import print
from pathlib import Path
from bssnn.model.bssnn import BSSNN
from bssnn.training.trainer import run_training
from bssnn.explainability.explainer import run_explanations
from bssnn.utils.data_loader import DataLoader
from bssnn.config.config import BSSNNConfig
from bssnn.visualization.visualization import setup_rich_logging


def main(config_path: str):
    """Main execution function."""
    # Set up logging
    setup_rich_logging()
    
    # Load configuration
    config = BSSNNConfig.from_yaml(config_path)
    print("\n[bold cyan]Loading configuration...[/bold cyan]")
    
    # Load and prepare data
    (X_train, X_val, y_train, y_val), n_features = DataLoader.load_and_prepare_data(
        config.data
    )
    
    # Update model configuration based on data
    config.model.adapt_to_data(n_features)
    
    # Initialize model
    model = BSSNN(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size
    )
    
    # Run training
    run_training(config, model, X_train, X_val, y_train, y_val)
    
    # Run explanations if enabled
    if config.explainability.enabled:
        explanations_dir = Path(config.output.base_dir) / config.output.explanations_dir
        explanations_dir.mkdir(parents=True, exist_ok=True)
        run_explanations(config, model, X_train, X_val, save_dir=str(explanations_dir))
    
    # Save final configuration
    config_save_path = Path(config.output.base_dir) / "final_config.yaml"
    config.save(str(config_save_path))
    print(f"\n[green]Configuration saved to[/green] {config_save_path}")
    
    print("\n[bold green]Processing completed successfully![/bold green]")
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("[red]Please provide the path to the configuration file.[/red]")
        sys.exit(1)
    
    main(sys.argv[1])