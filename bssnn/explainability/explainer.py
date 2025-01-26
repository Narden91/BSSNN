# explainer.py
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import contextlib
from rich import print
import os

class LogitModelWrapper(torch.nn.Module):
    """Wrapper to expose logits for SHAP explanations."""
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        
    def forward(self, x):
        if isinstance(self.original_model, torch.nn.Module):
            # Handle BSSNN model variants
            if hasattr(self.original_model, 'joint_network'):  # Original BSSNN
                joint = self.original_model.joint_network(x)
                marginal = self.original_model.marginal_network(x)
            else:  # StateSpaceBSSNN
                joint = self.original_model._forward_pathway(
                    x,
                    self.original_model.state_transitions_joint,
                    self.original_model.fc1_joint,
                    self.original_model.dropout_joint,
                    self.original_model.batch_norm_joint,
                    self.original_model.fc2_joint
                )
                marginal = self.original_model._forward_pathway(
                    x,
                    self.original_model.state_transitions_marginal,
                    self.original_model.fc1_marginal,
                    self.original_model.dropout_marginal,
                    self.original_model.batch_norm_marginal,
                    self.original_model.fc2_marginal
                )
            logits = joint - marginal
            return logits.squeeze(-1) 
        raise ValueError("Unsupported model type")

class BSSNNExplainer:
    """Enhanced explainer for BSSNN architectures with state-space support."""
    
    def __init__(self, model: torch.nn.Module, feature_names: Optional[List[str]] = None):
        self.model = model
        self.device = next(model.parameters()).device
        self.feature_names = feature_names or self._default_feature_names()
        
        # Validate feature names
        if len(self.feature_names) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {len(self.feature_names)}")

    @property
    def input_size(self) -> int:
        """Get model input size dynamically."""
        if hasattr(self.model, 'joint_network'):  # Original BSSNN
            return self.model.joint_network[0].in_features
        return self.model.fc1_joint.in_features  # StateSpaceBSSNN

    def _default_feature_names(self) -> List[str]:
        return [f"Feature {i+1}" for i in range(self.input_size)]

    def _get_feature_importance(self) -> np.ndarray:
        """Calculate combined feature importance from both pathways."""
        if hasattr(self.model, 'joint_network'):  # Original BSSNN
            joint_weights = self.model.joint_network[0].weight.detach().cpu().numpy()
            marginal_weights = self.model.marginal_network[0].weight.detach().cpu().numpy()
        else:  # StateSpaceBSSNN
            joint_weights = self.model.fc1_joint.weight.detach().cpu().numpy()
            marginal_weights = self.model.fc1_marginal.weight.detach().cpu().numpy()
            
        return np.abs(joint_weights).mean(axis=0) + np.abs(marginal_weights).mean(axis=0)

    def _compute_shap_values(self, X_train: torch.Tensor, X_val: torch.Tensor,
                        save_dir: Path) -> Dict[str, np.ndarray]:
        """Compute SHAP values with improved error handling and proper model wrapping.
        
        Args:
            X_train: Training data tensor
            X_val: Validation data tensor
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary containing SHAP results
        """
        self.model.eval()
        
        def model_wrapper(x):
            with torch.no_grad():
                # Convert numpy array to tensor, ensure it's on CPU
                x_tensor = torch.FloatTensor(x)
                output = self.model(x_tensor)
                # Return numpy array, ensuring correct shape for binary classification
                return output.numpy().reshape(-1)
        
        try:
            # Ensure data is in numpy format before passing to SHAP
            background = X_train[:100].numpy()
            val_sample = X_val[:100].numpy()
            
            # Disable SHAP verbosity
            import logging
            logging.getLogger('shap').setLevel(logging.WARNING)
            
            # Create explainer with minimal output
            explainer = shap.KernelExplainer(
                model_wrapper, 
                background,
                silent=True
            )
            
            # Compute SHAP values silently
            shap_values = explainer.shap_values(
                val_sample,
                silent=True,
                nsamples=100
            )
            
            # Handle both single and multi-output cases
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Create visualizations if directory exists
            if save_dir:
                plt.clf()  # Clear any existing plots
                shap.summary_plot(
                    shap_values, 
                    val_sample,
                    feature_names=self.feature_names,
                    show=False,
                    plot_type="violin"
                )
                plt.tight_layout()
                plt.savefig(save_dir/"shap_summary.png", bbox_inches='tight', dpi=300)
                plt.close()
                
                # Create feature importance plot
                plt.figure(figsize=(10, 6))
                shap_importance = np.abs(shap_values).mean(axis=0)
                plt.barh(self.feature_names, shap_importance)
                plt.title("SHAP Feature Importance")
                plt.tight_layout()
                plt.savefig(save_dir/"shap_importance.png", bbox_inches='tight', dpi=300)
                plt.close()
            
            return {
                'shap_values': shap_values,
                'expected_value': explainer.expected_value if isinstance(explainer.expected_value, float)
                                else explainer.expected_value[0]
            }
            
        except Exception as e:
            print(f"SHAP computation failed: {str(e)}")
            # Return basic feature importance as fallback
            importance = self._get_feature_importance()
            return {
                'feature_importance': importance,
                'shap_values': np.zeros((min(len(X_val), 100), X_val.shape[1])),
                'expected_value': 0.0
            }

    def _validate_shap_inputs(self, shap_values: np.ndarray, 
                            val_sample: torch.Tensor) -> bool:
        """Validate SHAP values dimensions."""
        if shap_values.ndim != 2 or shap_values.shape[1] != self.input_size:
            print(f"Invalid SHAP dimensions: {shap_values.shape}")
            return False
        return True

    def _plot_shap_summary(self, shap_values: np.ndarray, 
                         val_sample: torch.Tensor, save_dir: Path):
        """Enhanced SHAP summary plot with proper formatting."""
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            val_sample.cpu().numpy(),
            feature_names=self.feature_names,
            show=False,
            plot_size=None,
            max_display=15,
            plot_type="dot"
        )
        plt.title("SHAP Value Distribution", fontsize=14)
        plt.gcf().axes[-1].set_aspect(0.1)
        plt.gcf().axes[-1].set_box_aspect(10)
        plt.savefig(save_dir/"shap_summary.png", bbox_inches='tight', dpi=150)
        plt.close()

    def _plot_feature_importance_comparison(self, shap_values: np.ndarray, 
                                          save_dir: Path):
        """Compare direct and SHAP-based feature importance."""
        direct_importance = self._get_feature_importance()
        shap_importance = np.abs(shap_values).mean(0)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        indices = np.arange(len(self.feature_names))
        
        ax.barh(indices - 0.2, direct_importance, 0.4, 
               label='Direct Importance')
        ax.barh(indices + 0.2, shap_importance, 0.4, 
               label='SHAP Importance')
        
        ax.set_yticks(indices)
        ax.set_yticklabels(self.feature_names)
        ax.legend()
        plt.title("Feature Importance Comparison")
        plt.tight_layout()
        plt.savefig(save_dir/"importance_comparison.png", bbox_inches='tight', dpi=150)
        plt.close()

    def explain(self, X_train: torch.Tensor, X_val: torch.Tensor,
                save_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Generate comprehensive model explanations."""
        print("[bold cyan]Generating explanations...[/bold cyan]")
        results = {'feature_importance': self._get_feature_importance()}
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save feature importance
            np.save(save_path/"direct_importance.npy", results['feature_importance'])
            
            # Compute and save SHAP values
            shap_results = self._compute_shap_values(X_train, X_val, save_path)
            results.update(shap_results)
            
            if shap_results['shap_values'].size > 0:
                np.save(save_path/"shap_values.npy", shap_results['shap_values'])
        
        print("[bold green]✓ Explanation complete[/bold green]")
        return results

def run_explanations(config, model, X_train: torch.Tensor,
                    X_val: torch.Tensor, save_dir: Optional[str] = None) -> Dict:
    """Run model explanations with improved error handling.
    
    Args:
        config: Configuration object containing explainability settings
        model: BSSNN model instance
        X_train: Training data tensor
        X_val: Validation data tensor
        save_dir: Optional directory to save results
        
    Returns:
        Dictionary containing explanation results
    """
    print("\n[bold blue]Generating Model Explanations...[/bold blue]")
    
    try:
        feature_names = getattr(config.explainability, 'feature_names', None)
        explainer = BSSNNExplainer(model, feature_names)
        
        with torch.no_grad():  # Ensure no gradients are computed during explanation
            results = explainer.explain(X_train, X_val, save_dir)
        
        if save_dir:
            print(f"[bold green]Results saved to:[/bold green] {Path(save_dir).resolve()}")
        
        return results
        
    except Exception as e:
        print(f"[bold red]Error during explanation generation: {str(e)}[/bold red]")
        print("[yellow]Falling back to basic feature importance analysis...[/yellow]")
        
        # Return basic feature importance when full explanation fails
        return {
            'feature_importance': model._get_feature_importance(),
            'feature_names': feature_names or [f"Feature {i+1}" for i in range(X_train.shape[1])]
        }