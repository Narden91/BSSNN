import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import shap
from typing import List, Tuple, Dict, Optional
from ..model.bssnn import BSSNN

class BSSNNExplainer:
    """Explainability module for BSSNN models."""
    
    def __init__(self, model: BSSNN, feature_names: Optional[List[str]] = None):
        """Initialize the explainer.
        
        Args:
            model: Trained BSSNN model instance
            feature_names: Optional list of feature names for visualization
        """
        self.model = model
        self.feature_names = feature_names or [f"Feature {i+1}" for i in range(model.fc1_joint.in_features)]
        
    def get_feature_importance(self) -> np.ndarray:
        """Extract feature importance from model weights.
        
        Returns:
            Array of importance scores for each feature
        """
        joint_weights = self.model.fc1_joint.weight.data.numpy()
        marginal_weights = self.model.fc1_marginal.weight.data.numpy()
        
        # Average the absolute weights across hidden units
        joint_importance = np.mean(np.abs(joint_weights), axis=0)
        marginal_importance = np.mean(np.abs(marginal_weights), axis=0)
        
        # Combine joint and marginal importance
        return joint_importance + marginal_importance
    
    def create_dag_visualization(self, feature_importance: np.ndarray, save_path: Optional[str] = None):
        """Create and optionally save a DAG visualization of feature relationships.
        
        Args:
            feature_importance: Array of importance scores
            save_path: Optional path to save the visualization
        """
        G = nx.DiGraph()
        
        # Add nodes
        for feature in self.feature_names:
            G.add_node(feature)
        G.add_node("Target (y)")
        
        # Add edges with weights
        for feature, importance in zip(self.feature_names, feature_importance):
            G.add_edge(feature, "Target (y)", weight=float(importance))
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2000, 
                node_color="lightblue", font_size=10, font_weight="bold")
        
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {k: f"{v:.3f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def compute_shapley_values(
        self,
        X_train: torch.Tensor,
        X_val: torch.Tensor,
        background_samples: int = 100,
        max_val_samples: int = 100
    ) -> Tuple[np.ndarray, float]:
        """Compute SHAP values for model interpretation.
        
        Args:
            X_train: Training data for background distribution
            X_val: Validation data for explanation
            background_samples: Number of background samples for SHAP
            max_val_samples: Maximum number of validation samples to explain
            
        Returns:
            Tuple of (SHAP values array, expected value)
        """
        self.model.eval()
        
        def model_wrapper(x):
            with torch.no_grad():
                x_tensor = torch.tensor(x, dtype=torch.float32)
                joint = self.model.fc2_joint(
                    self.model.relu_joint(self.model.fc1_joint(x_tensor)))
                marginal = self.model.fc2_marginal(
                    self.model.relu_marginal(self.model.fc1_marginal(x_tensor)))
                logits = joint - marginal
                return logits.numpy().reshape(-1)
        
        # Prepare data
        background = shap.kmeans(X_train.numpy(), background_samples)
        val_sample = X_val[:max_val_samples].numpy()
        
        # Create explainer and compute values
        explainer = shap.KernelExplainer(model_wrapper, background)
        shap_values = explainer.shap_values(val_sample)
        
        # Ensure correct shape
        if len(np.array(shap_values).shape) == 3:
            shap_values = np.array(shap_values).squeeze(axis=-1)
            
        return shap_values, explainer.expected_value
    
    def plot_feature_importance(
        self,
        shap_values: np.ndarray,
        val_sample: np.ndarray,
        save_dir: Optional[str] = None
    ):
        """Create and save feature importance visualizations.
        
        Args:
            shap_values: Computed SHAP values
            val_sample: Validation samples used for SHAP
            save_dir: Optional directory to save visualizations
        """
        # Calculate mean absolute SHAP values
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        idx = np.argsort(feature_importance)
        pos = np.arange(len(self.feature_names))
        
        plt.barh(pos, feature_importance[idx])
        plt.yticks(pos, [self.feature_names[i] for i in idx])
        plt.xlabel('Mean |SHAP value|')
        plt.title('Feature Importance Based on SHAP Values')
        
        if save_dir:
            plt.savefig(f"{save_dir}/shap_feature_importance.png", 
                       dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            val_sample,
            feature_names=self.feature_names,
            show=False,
            plot_type="violin"
        )
        
        if save_dir:
            plt.savefig(f"{save_dir}/shap_summary_plot.png", 
                       dpi=300, bbox_inches='tight')
        plt.close()

    def explain(
        self,
        X_train: torch.Tensor,
        X_val: torch.Tensor,
        save_dir: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Generate comprehensive model explanations.
        
        Args:
            X_train: Training data
            X_val: Validation data
            save_dir: Optional directory to save visualizations
            
        Returns:
            Dictionary containing explanation results
        """
        # Get feature importance
        feature_importance = self.get_feature_importance()
        
        # Create DAG visualization
        if save_dir:
            self.create_dag_visualization(
                feature_importance,
                save_path=f"{save_dir}/feature_dag.png"
            )
        
        # Compute SHAP values
        shap_values, expected_value = self.compute_shapley_values(
            X_train, X_val
        )
        
        # Create SHAP visualizations
        if save_dir:
            self.plot_feature_importance(
                shap_values,
                X_val[:100].numpy(),
                save_dir=save_dir
            )
        
        return {
            'feature_importance': feature_importance,
            'shap_values': shap_values,
            'expected_value': expected_value
        }