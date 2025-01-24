import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
import os
import contextlib
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from rich import print
from ..model.bssnn import BSSNN

class BSSNNExplainer:
    """Explainability module with full output suppression."""
    
    def __init__(self, model: BSSNN, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names or [
            f"Feature {i+1}" for i in range(model.fc1_joint.in_features)
        ]

    def explain(self, X_train: torch.Tensor, X_val: torch.Tensor,
                save_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Generate explanations with robust output suppression."""
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
        """Calculate feature importance from model weights."""
        joint_weights = self.model.fc1_joint.weight.data.numpy()
        marginal_weights = self.model.fc1_marginal.weight.data.numpy()
        return np.abs(joint_weights).mean(axis=0) + np.abs(marginal_weights).mean(axis=0)

    def _plot_feature_importance(self, importance: np.ndarray, save_dir: Path):
        """Save feature importance visualization."""
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            plt.figure(figsize=(10, 6))
            indices = np.argsort(importance)
            plt.barh(range(len(indices)), importance[indices], 
                    tick_label=np.array(self.feature_names)[indices])
            plt.title("Feature Importance")
            plt.savefig(save_dir/"feature_importance.png", bbox_inches='tight')
            plt.close()

    def _compute_shap_values(self, X_train: torch.Tensor, X_val: torch.Tensor,
                           save_dir: Path) -> Dict[str, np.ndarray]:
        """Compute SHAP values with nuclear-grade output suppression."""
        self.model.eval()
        
        # Nuclear output suppression
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

            background = X_train[:100].numpy()
            explainer = shap.KernelExplainer(model_wrapper, background, silent=True)
            shap_values = explainer.shap_values(X_val[:100].numpy(), silent=True)
            
            # Plotting within suppression context
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_val[:100].numpy(), 
                            feature_names=self.feature_names, show=False)
            plt.savefig(save_dir/"shap_summary.png", bbox_inches='tight')
            plt.close()

        return {
            'shap_values': np.array(shap_values),
            'expected_value': explainer.expected_value
        }

def run_explanations(config, model: BSSNN, X_train: torch.Tensor,
                   X_val: torch.Tensor, save_dir: Optional[str] = None):
    """Public interface for explanations with rich output."""
    print("\n[bold blue]==== Model Explanation ====[/bold blue]")
    explainer = BSSNNExplainer(model, config.explainability.feature_names)
    results = explainer.explain(X_train, X_val, save_dir)
    if save_dir:
        print(f"[bold green]Results saved to:[/bold green] {Path(save_dir).resolve()}")
    return results