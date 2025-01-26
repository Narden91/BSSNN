import csv
from pathlib import Path
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler
from datetime import datetime
import logging
import sys
from typing import Any, Dict, List, Optional


console = Console()


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
    """Training progress visualization using Rich."""
    
    def __init__(self, total_epochs: int, display_metrics: List[str], fold: Optional[int] = None):
        """Initialize the training progress tracker."""
        self.total_epochs = total_epochs
        self.display_metrics = display_metrics[:3]  # Limit to first 3 metrics for display
        self.start_time = datetime.now()
        self.fold = fold
        self._setup_progress()

    def _setup_progress(self):
        """Set up the progress display with improved layout."""
        # Define columns for progress display
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("Loss: {task.fields[loss]:.4f}")
        ]
        
        # Add columns for metrics
        for metric in self.display_metrics:
            columns.append(
                TextColumn(f"{metric}: {{task.fields[{metric}]:.4f}}")
            )
            
        columns.append(TimeElapsedColumn())
        
        self.progress = Progress(*columns, console=console, transient=True)
        
        # Initialize fields
        fields = {
            'loss': 0.0
        }
        for metric in self.display_metrics:
            fields[metric] = 0.0
            
        # Create task with appropriate description
        desc = "Training" if self.fold is None else f"Fold {self.fold}"
        self.task_id = self.progress.add_task(
            description=desc,
            total=self.total_epochs,
            **fields
        )
        
        self.progress.start()
    
    def update(self, epoch: int, loss: float, metrics: Dict[str, float]):
        """Update progress display."""
        update_fields = {
            'loss': loss
        }
        
        # Update metric fields
        for metric in self.display_metrics:
            if metric in metrics:
                update_fields[metric] = metrics[metric]
        
        # Update progress
        self.progress.update(
            task_id=self.task_id,
            completed=epoch,
            **update_fields
        )

    
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
        """Complete progress tracking with final metrics."""
        self.progress.stop()
        
        if self.fold is not None:
            console.print(f"[green]✓ Completed fold {self.fold}[/green]")
        else:
            self._display_final_metrics(final_loss, final_metrics)
    
    def _display_final_metrics(self, final_loss: float, final_metrics: Dict[str, float]):
        """Display final metrics in a clean table format."""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        
        table.add_row("Loss", f"{final_loss:.4f}")
        for metric, value in final_metrics.items():
            table.add_row(metric, f"{value:.4f}")
        
        console.print("\n[bold]Final Results:[/bold]")
        console.print(table)
        

class CrossValidationProgress:
    """Progress tracking for cross-validation process."""
    
    def __init__(self, n_folds: int, dataset_sizes: Optional[Dict[str, int]] = None):
        """Initialize cross-validation progress tracking."""
        self.n_folds = n_folds
        if dataset_sizes:
            self._print_cv_header(dataset_sizes)
    
    def _print_cv_header(self, dataset_sizes: Dict[str, int]):
        """Print initial cross-validation information."""
        console.print("\n[bold blue]Cross-validation Configuration[/bold blue]")
        
        table = Table(show_header=False, box=None)
        table.add_row("Number of folds", str(self.n_folds))
        
        if dataset_sizes:
            if 'train' in dataset_sizes:
                table.add_row("Training samples", str(dataset_sizes['train']))
            if 'val' in dataset_sizes:
                table.add_row("Validation samples", str(dataset_sizes['val']))
            if 'test' in dataset_sizes:
                table.add_row("Test samples", str(dataset_sizes['test']))
        
        console.print(table)
        console.print()
    
    def start_fold(self, fold: int):
        """Signal start of a new fold."""
        console.print(f"\n[cyan]Starting fold {fold}/{self.n_folds}[/cyan]")
    
    def print_summary(self, metrics: Dict[str, Dict[str, float]], phase: str = "Cross-validation"):
        """Print metrics summary with phase information."""
        console.print(f"\n[bold blue]{phase} Results[/bold blue]")
        
        if isinstance(metrics.get(list(metrics.keys())[0]), dict):
            # Cross-validation results with mean/std
            table = Table(show_header=True, header_style="bold")
            table.add_column("Metric")
            table.add_column("Mean", justify="right")
            table.add_column("Std", justify="right")
            
            for metric, stats in metrics.items():
                table.add_row(
                    metric,
                    f"{stats['mean']:.4f}",
                    f"{stats['std']:.4f}"
                )
        else:
            # Single set of metrics
            table = Table(show_header=True, header_style="bold")
            table.add_column("Metric")
            table.add_column("Value", justify="right")
            
            for metric, value in metrics.items():
                table.add_row(
                    metric,
                    f"{value:.4f}"
                )
        
        console.print(table)


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