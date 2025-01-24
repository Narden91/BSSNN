import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import os
import contextlib
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from rich import print


class BSSNNExplainer:
    """Improved explainability module with full output suppression."""
    
    def __init__(self, model, feature_names: Optional[List[str]] = None):
        """Initialize the explainer.
        
        Args:
            model: BSSNN model instance
            feature_names: Optional list of feature names
        """
        self.model = model
        self.feature_names = feature_names or [
            f"Feature {i+1}" for i in range(model.fc1_joint.in_features)
        ]

    def explain(self, X_train: torch.Tensor, X_val: torch.Tensor,
                save_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Generate comprehensive explanations with robust output suppression.
        
        Args:
            X_train: Training data tensor
            X_val: Validation data tensor
            save_dir: Optional directory to save visualizations
            
        Returns:
            Dictionary containing explanation results
        """
        print("[bold cyan]Starting explanation process...[/bold cyan]")
        
        results = {}
        
        print("[bold cyan]1. Computing feature importance...[/bold cyan]")
        results['feature_importance'] = self._get_feature_importance()
        results['feature_names'] = self.feature_names.copy()
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            print("[bold cyan]2. Creating visualizations...[/bold cyan]")
            self._plot_feature_importance(results['feature_importance'], save_path)
            
            print("[bold cyan]3. Computing SHAP values...[/bold cyan]")
            results.update(self._compute_shap_values(X_train, X_val, save_path))
        
        print("[bold green]✓ Explanation process completed[/bold green]")
        return results

    def _get_feature_importance(self) -> np.ndarray:
        """Calculate feature importance from model weights.
        
        Returns:
            Array of feature importance scores
        """
        joint_weights = self.model.fc1_joint.weight.data.numpy()
        marginal_weights = self.model.fc1_marginal.weight.data.numpy()
        return np.abs(joint_weights).mean(axis=0) + np.abs(marginal_weights).mean(axis=0)

    def _plot_feature_importance(self, importance: np.ndarray, save_dir: Path):
        """Save feature importance visualization.
        
        Args:
            importance: Array of importance scores
            save_dir: Directory to save the plot
        """
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            plt.figure(figsize=(10, 6))
            indices = np.argsort(importance)
            plt.barh(range(len(indices)), importance[indices], 
                    tick_label=np.array(self.feature_names)[indices])
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig(save_dir/"feature_importance.png", bbox_inches='tight', dpi=300)
            plt.close()

    def _compute_shap_values(self, X_train: torch.Tensor, X_val: torch.Tensor,
                           save_dir: Path) -> Dict[str, np.ndarray]:
        """Compute SHAP values with comprehensive output suppression.
        
        Args:
            X_train: Training data tensor
            X_val: Validation data tensor
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary containing SHAP values and expected value
        """
        self.model.eval()
        
        # Suppress all output during SHAP computation
        with open(os.devnull, 'w') as fnull, \
             contextlib.redirect_stdout(fnull), \
             contextlib.redirect_stderr(fnull):
            
            def model_wrapper(x):
                with torch.no_grad():
                    x_tensor = torch.tensor(x, dtype=torch.float32)
                    joint = self.model.fc2_joint(self.model.relu_joint(
                        self.model.fc1_joint(x_tensor)))
                    marginal = self.model.fc2_marginal(self.model.relu_marginal(
                        self.model.fc1_marginal(x_tensor)))
                    return (joint - marginal).numpy().flatten()

            # Use subset of training data for background
            background = X_train[:100].numpy()
            explainer = shap.KernelExplainer(model_wrapper, background, silent=True)
            
            # Compute SHAP values for subset of validation data
            val_sample = X_val[:100].numpy()
            # Temporarily disable all Python warnings and logging
            import warnings
            import logging
            logging.disable(logging.CRITICAL)
            warnings.filterwarnings('ignore')
            
            # Redirect all possible outputs
            null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
            save_fds = [os.dup(1), os.dup(2)]
            os.dup2(null_fds[0], 1)
            os.dup2(null_fds[1], 2)
            
            try:
                shap_values = explainer.shap_values(val_sample, silent=True, nsamples=100)
            finally:
                # Restore file descriptors
                os.dup2(save_fds[0], 1)
                os.dup2(save_fds[1], 2)
                
                # Close all temporary file descriptors
                for fd in null_fds + save_fds:
                    os.close(fd)
                    
                # Re-enable logging and warnings
                logging.disable(logging.NOTSET)
                warnings.filterwarnings('default')
            
            # Create and save SHAP summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, val_sample, 
                            feature_names=self.feature_names, 
                            show=False,
                            plot_type="violin")
            plt.tight_layout()
            plt.savefig(save_dir/"shap_summary.png", bbox_inches='tight', dpi=300)
            plt.close()

            # Create and save SHAP bar plot
            plt.figure(figsize=(10, 6))
            shap_importance = np.abs(shap_values).mean(axis=0)
            idx = np.argsort(shap_importance)
            plt.barh(range(len(self.feature_names)), shap_importance[idx])
            plt.yticks(range(len(self.feature_names)), 
                      [self.feature_names[i] for i in idx])
            plt.xlabel('Mean |SHAP value|')
            plt.title('Feature Importance (SHAP)')
            plt.tight_layout()
            plt.savefig(save_dir/"shap_importance.png", bbox_inches='tight', dpi=300)
            plt.close()

        return {
            'shap_values': np.array(shap_values),
            'expected_value': explainer.expected_value
        }


def run_explanations(config, model, X_train: torch.Tensor,
                    X_val: torch.Tensor, save_dir: Optional[str] = None) -> Dict:
    """Public interface for running explanations with rich output.
    
    Args:
        config: Configuration object containing explainability settings
        model: BSSNN model instance
        X_train: Training data tensor
        X_val: Validation data tensor
        save_dir: Optional directory to save results
        
    Returns:
        Dictionary containing explanation results
    """
    print("\n[bold blue]==== Model Explanation ====[/bold blue]")
    
    feature_names = getattr(config.explainability, 'feature_names', None)
    explainer = BSSNNExplainer(model, feature_names)
    
    results = explainer.explain(X_train, X_val, save_dir)
    
    if save_dir:
        print(f"[bold green]Results saved to:[/bold green] {Path(save_dir).resolve()}")
    
    return results