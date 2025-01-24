import time
import sys
from typing import Dict, List, Optional
from datetime import datetime

class TrainingProgress:
    def __init__(self, total_epochs: int, display_metrics: List[str]):
        """Initialize the training progress tracker.
        
        Args:
            total_epochs: Total number of epochs
            display_metrics: List of metric names to display (max 4)
        """
        self.total_epochs = total_epochs
        self.display_metrics = display_metrics[:4]  # Limit to 4 metrics
        self.start_time = time.time()
        self.last_update = self.start_time
    
    def _format_time(self, seconds: float) -> str:
        """Format time in a human-readable way."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _create_progress_bar(self, current: int, width: int = 20) -> str:
        """Create the progress bar string."""
        filled = int(width * current / self.total_epochs)
        bar = '█' * filled + '░' * (width - filled)
        return bar
    
    def update(self, epoch: int, loss: float, metrics: Dict[str, float]):
        """Update and display the training progress.
        
        Args:
            epoch: Current epoch number
            loss: Current loss value
            metrics: Dictionary of current metrics
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        epoch_time = current_time - self.last_update
        self.last_update = current_time
        
        # Calculate estimated time remaining
        epochs_remaining = self.total_epochs - epoch
        time_per_epoch = elapsed / epoch
        eta = time_per_epoch * epochs_remaining
        
        # Create progress bar
        bar = self._create_progress_bar(epoch)
        
        # Format metrics string
        metrics_str = ' '.join(
            f"{name[:4]}:{metrics.get(name, 0):.3f}" 
            for name in self.display_metrics
        )
        
        # Create the progress line
        progress_line = (
            f"\rEpoch: [{epoch}/{self.total_epochs}] {bar} "
            f"Loss: {loss:.3f} {metrics_str} "
            f"| {self._format_time(elapsed)}<{self._format_time(eta)}"
        )
        
        # Print the progress
        sys.stdout.write(progress_line)
        sys.stdout.flush()
        
        # Add newline at the end of training
        if epoch == self.total_epochs:
            print()  # New line after completion
            self._print_final_summary(loss, metrics)
    
    def _print_final_summary(self, final_loss: float, final_metrics: Dict[str, float]):
        """Print the final training summary."""
        total_time = time.time() - self.start_time
        print("\nTraining completed!")
        print(f"Total time: {self._format_time(total_time)}")
        print(f"Final loss: {final_loss:.4f}")
        for metric in self.display_metrics:
            if metric in final_metrics:
                print(f"Final {metric}: {final_metrics[metric]:.4f}")