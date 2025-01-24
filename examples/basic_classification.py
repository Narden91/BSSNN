import sys
sys.dont_write_bytecode = True

from rich import print
from rich.console import Console
from bssnn.utils.file_utils import create_timestamped_dir
from bssnn.training.cross_validation import run_cross_validation
from bssnn.training.trainer import run_final_model_training
from bssnn.utils.data_loader import DataLoader
from bssnn.config.config import BSSNNConfig
from bssnn.visualization.visualization import setup_rich_logging

console = Console()

def main(config_path: str):
    """Main execution function for BSSNN training pipeline."""
    try:
        # Create output directory
        output_dir = create_timestamped_dir("results")
        console.print(f"\n[cyan]Created results directory:[/cyan] {output_dir}")
        
        # Setup logging
        setup_rich_logging(str(output_dir / 'run.log'))
        
        # Initialize configuration
        config = BSSNNConfig.from_yaml(config_path)
        config.initialize_and_validate(config_path, output_dir)
        
        # Prepare data
        data_loader = DataLoader()
        X, y = data_loader.prepare_data(config)
        
        # Run cross-validation
        cv_metrics, _ = run_cross_validation(config, X, y, output_dir)
        
        # Train final model if requested
        if config.output.save_model:
            final_model = run_final_model_training(config, X, y, output_dir)
        
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