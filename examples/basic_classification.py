import torch
import sys 
from pathlib import Path

sys.dont_write_bytecode = True

from sklearn.datasets import make_classification
from bssnn.model.bssnn import BSSNN
from bssnn.training.trainer import BSSNNTrainer
from bssnn.utils.data import prepare_data
from bssnn.utils.visualization import TrainingProgress
from bssnn.explainability.explainer import BSSNNExplainer


def main():
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    # Prepare data
    X_train, X_val, y_train, y_val = prepare_data(X, y)
    
    # Initialize model and trainer
    model = BSSNN(input_size=X.shape[1], hidden_size=64)
    trainer = BSSNNTrainer(model)
    
    # Define metrics to display during training
    display_metrics = ['accuracy', 'f1_score', 'auc_roc', 'calibration_error']
    
    # Initialize progress tracker
    num_epochs = 100
    progress = TrainingProgress(
        total_epochs=num_epochs,
        display_metrics=display_metrics
    )
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        # Training step
        train_loss = trainer.train_epoch(X_train, y_train)
        val_loss, metrics = trainer.evaluate(X_val, y_val)
        
        # Update progress display
        progress.update(epoch, val_loss, metrics)
    
    # Model Explainability
    # Create feature names for better visualization
    feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]
    
    # Initialize explainer
    explainer = BSSNNExplainer(model, feature_names=feature_names)
    
    # Create output directory for visualizations
    output_dir = Path("explanations")
    output_dir.mkdir(exist_ok=True)
    
    # Generate comprehensive explanations
    print("\nGenerating model explanations...")
    explanations = explainer.explain(
        X_train,
        X_val,
        save_dir=str(output_dir)
    )
    
    # Print feature importance scores
    print("\nFeature Importance Scores:")
    importance_scores = explanations['feature_importance']
    for feature, importance in zip(feature_names, importance_scores):
        print(f"{feature}: {importance:.3f}")
    
    print(f"\nExplanation visualizations saved to: {output_dir}")
    print("Generated files:")
    print("- feature_dag.png: Directed Acyclic Graph of feature relationships")
    print("- shap_feature_importance.png: SHAP-based feature importance")
    print("- shap_summary_plot.png: SHAP summary visualization")

if __name__ == "__main__":
    main()