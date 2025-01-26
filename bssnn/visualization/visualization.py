import csv
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler
from datetime import datetime
import logging
import sys
from typing import Any, Dict, List, Optional


def setup_rich_logging(log_path: Optional[str] = None):
    """Configure Rich logging handler with optional file output.
    
    Args:
        log_path: Optional path to save log file
    """
    # Configure console logging
    console_handler = RichHandler(
        rich_tracebacks=True,
        console=Console(force_terminal=True)
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # Add file handler if log_path is provided
    if log_path:
        # Create directory if it doesn't exist
        log_dir = Path(log_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_path}")


class TrainingProgress:
    """Enhanced training progress visualization using Rich."""
    
    def __init__(self, total_epochs: int, display_metrics: List[str], results_dir: str = "results"):
        """Initialize the training progress tracker."""
        self.total_epochs = total_epochs
        self.display_metrics = display_metrics[:3]  # Limit to first 3 metrics for display
        self.start_time = datetime.now()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._setup_progress()

    def _setup_progress(self):
        """Set up the progress display."""
        self.console = Console(force_terminal=True)
        
        # Define columns for progress display
        columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("Epoch: {task.fields[epoch]}/{task.fields[total_epochs]}"),
            TextColumn("Loss: {task.fields[loss]:.4f}")
        ]
        
        # Add columns for metrics
        for metric in self.display_metrics:
            columns.append(
                TextColumn(f"{metric}: {{task.fields[{metric}]:.4f}}")
            )
            
        columns.append(TimeElapsedColumn())
        
        self.progress = Progress(*columns, console=self.console)
        
        # Initialize fields
        fields = {
            'epoch': 0,
            'total_epochs': self.total_epochs,
            'loss': 0.0
        }
        for metric in self.display_metrics:
            fields[metric] = 0.0
            
        # Create task
        self.task_id = self.progress.add_task(
            description="Training:",
            total=self.total_epochs,
            **fields
        )
        
        self.progress.start()
    
    def update(self, epoch: int, loss: float, metrics: Dict[str, float]):
        """Update and display the training progress."""
        try:
            # Prepare update fields
            update_fields = {
                'epoch': epoch,
                'loss': loss
            }
            
            # Update metric fields
            for metric in self.display_metrics:
                if metric in metrics:
                    update_fields[metric] = metrics[metric]
            
            # Update progress
            self.progress.update(
                task_id=self.task_id,
                completed=epoch,  # This updates the progress bar
                **update_fields
            )
            
            # Force refresh display
            self.progress.refresh()
            
        except Exception as e:
            print(f"Error updating progress: {str(e)}")

    
    def _save_training_results(self, duration: datetime, final_loss: float, final_metrics: Dict[str, float]):
        """Save training results to text and CSV files."""
        # Save detailed summary as text
        summary_file = self.results_dir / "training_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*75 + "\n")
            f.write("Training Results Summary\n")
            f.write("="*75 + "\n\n")
            f.write(f"Training Duration: {duration}\n")
            f.write(f"Final Loss: {float(final_loss):.4f}\n\n")
            f.write("Final Metrics:\n")
            f.write("-"*35 + "\n")
            for metric, value in final_metrics.items():
                f.write(f"{metric}: {float(value):.4f}\n")
            f.write("\n" + "="*75 + "\n")
        
        # Save metrics to CSV
        metrics_file = self.results_dir / "training_metrics.csv"
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['loss', float(final_loss)])
            for metric, value in final_metrics.items():
                writer.writerow([metric, float(value)])
    
    def complete(self, final_loss: float, final_metrics: Dict[str, float]):
        """Complete the progress tracking and display final results."""
        try:
            # Update final state
            self.update(self.total_epochs, final_loss, final_metrics)
            
            # Stop progress display
            self.progress.stop()
            
            # Calculate duration
            duration = datetime.now() - self.start_time
            
            # Save results
            self._save_training_results(duration, final_loss, final_metrics)
            
            # Display final summary
            self.console.print("\n" + "="*75)
            self.console.print("Training Results:")
            self.console.print("-"*35)
            self.console.print(f"Total time: {duration}")
            self.console.print(f"Final loss: {float(final_loss):.4f}")
            self.console.print("-"*35)
            self.console.print("Final Metrics:")
            
            for metric, value in final_metrics.items():
                self.console.print(f"{metric}: {float(value):.4f}")
            
        except Exception as e:
            print(f"Error completing progress: {str(e)}")


def print_cv_header(n_folds: int, dataset_sizes: Dict[str, int], total_epochs: int):
    """Print initial cross-validation information."""
    console = Console()
    console.print("\n[bold blue]Cross-validation Configuration[/bold blue]")
    console.print("=" * 80)
    console.print(f"Number of folds: {n_folds}")
    console.print(f"Total epochs per fold: {total_epochs}\n")
    console.print("[bold]Dataset sizes:[/bold]")
    console.print(f"Training samples:   {dataset_sizes['train']}")
    console.print(f"Validation samples: {dataset_sizes['val']}")
    console.print(f"Test samples:       {dataset_sizes['test']}")
    console.print("=" * 80)


def print_fold_start(fold: int, total_folds: int):
    """Print minimal fold header."""
    console = Console()
    console.print(f"\n[bold cyan]Starting fold {fold}/{total_folds}[/bold cyan]")
    

class ExplainerProgress:
    """Progress visualization for model explanation process."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._progress = None
        
    def start(self, steps: List[str]):
        """Start progress tracking with given steps."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=Console(force_terminal=True),
            transient=True  # This makes the progress bar disappear when done
        )
        self._progress.start()
        self.task = self._progress.add_task("", total=len(steps))
        self.steps = steps
        self.current_step = 0
    
    def update(self, description: str = None):
        """Update progress with optional new description."""
        if self._progress:
            if description:
                self._progress.update(self.task, description=f"Processing: {description}")
            self._progress.update(self.task, advance=1)
            self.current_step += 1
    
    def stop(self):
        """Stop progress tracking."""
        if self._progress:
            self._progress.stop()
            self._progress = None
    
    def run_with_progress(self, steps, execute_step):
        """Run steps with progress tracking."""
        self.start(steps)
        for step in steps:
            execute_step(step)
            self.update(step)
        self.stop()