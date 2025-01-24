from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from rich.console import Console
from ..model.bssnn import BSSNN
from ..utils.data_loader import DataLoader
from ..config.config import BSSNNConfig
from ..training.trainer import run_training
from ..visualization.visualization import print_cv_header, print_fold_start

console = Console()

def save_fold_metrics(metrics: dict, fold: int, output_dir: Path) -> None:
    """Save metrics for a single fold to CSV.
    
    Args:
        metrics: Dictionary of metric values
        fold: Current fold number
        output_dir: Directory to save metrics
    """
    metrics_df = pd.DataFrame([metrics])
    metrics_df['fold'] = fold
    metrics_dir = output_dir / 'fold_metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_dir / f'fold_{fold}_metrics.csv', index=False)

def save_cv_summary(avg_metrics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    """Save cross-validation summary statistics.
    
    Args:
        avg_metrics: Dictionary of metric statistics
        output_dir: Directory to save summary
    """
    metrics_dir = output_dir / 'fold_metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_dir / 'cv_summary.txt', 'w') as f:
        f.write("Cross-validation Summary\n")
        f.write("=" * 50 + "\n\n")
        for metric, stats in avg_metrics.items():
            f.write(f"{metric:<25} Mean: {stats['mean']:>8.4f}    Std: {stats['std']:>8.4f}\n")

def print_cv_summary(avg_metrics: Dict[str, Dict[str, float]]) -> None:
    """Print cross-validation results summary with proper spacing.
    
    Args:
        avg_metrics: Dictionary of metric statistics
    """
    # Add spacing after the progress bar
    console.print("\n\n" + "=" * 80)  # Extra newline for spacing
    console.print("[bold green]Cross-validation Results (Test Set)[/bold green]")
    console.print("=" * 80)
    
    # Format metrics output
    metric_format = "{:<28} Mean: {:>8.4f}    Std: {:>8.4f}"
    for metric, stats in avg_metrics.items():
        console.print(metric_format.format(
            metric,
            stats['mean'],
            stats['std']
        ))
    # Add extra newline at the end for spacing
    console.print()

def calculate_cv_statistics(cv_metrics: list) -> Dict[str, Dict[str, float]]:
    """Calculate mean and standard deviation for all metrics across folds.
    
    Args:
        cv_metrics: List of metric dictionaries from each fold
        
    Returns:
        Dictionary containing mean and std for each metric
    """
    avg_metrics = {}
    for metric in cv_metrics[0].keys():
        values = [m[metric] for m in cv_metrics]
        avg_metrics[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }
    return avg_metrics

def run_cross_validation(
    config: BSSNNConfig,
    X: torch.Tensor,
    y: torch.Tensor,
    output_dir: Path
) -> Tuple[Dict[str, Dict[str, float]], Optional[BSSNN]]:
    """Run cross-validation training with proper train/val/test splits.
    
    Args:
        config: Training configuration
        X: Input features tensor
        y: Target labels tensor
        output_dir: Directory for saving results
        
    Returns:
        Tuple of (average metrics, final model if requested)
    """
    cv_metrics = []
    n_folds = config.data.validation.n_folds
    data_loader = DataLoader()
    final_model = None
    
    # Get sample splits for first fold to show dataset sizes
    splits = data_loader.get_cross_validation_splits(X, y, config.data, fold=1)
    X_train, X_val, X_test = splits[:3]
    
    # Print initial configuration
    print_cv_header(
        n_folds=n_folds,
        dataset_sizes={
            'train': len(X_train),
            'val': len(X_val),
            'test': len(X_test)
        },
        total_epochs=config.training.num_epochs
    )
    
    # Run cross-validation
    for fold in range(1, n_folds + 1):
        print_fold_start(fold, n_folds)
        
        splits = data_loader.get_cross_validation_splits(X, y, config.data, fold)
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        
        model = BSSNN(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size
        )
        
        trainer = run_training(
            config, model,
            X_train, X_val,
            y_train, y_val,
            fold=fold,
            silent=True
        )
        
        _, test_metrics = trainer.evaluate(X_test, y_test)
        cv_metrics.append(test_metrics)
        
        save_fold_metrics(test_metrics, fold, output_dir)
        
        # Add extra spacing after progress bar
        console.print("\n[green]✓ Completed fold {}/{}[/green]\n".format(fold, n_folds))
        
        if config.output.save_model and fold == n_folds:
            final_model = model
    
    # Calculate and save summary statistics
    avg_metrics = calculate_cv_statistics(cv_metrics)
    save_cv_summary(avg_metrics, output_dir)
    print_cv_summary(avg_metrics)
    
    return avg_metrics, final_model