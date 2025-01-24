# README.md

# Bayesian State-Space Neural Networks (BSSNN)

BSSNN is a framework that combines Bayesian probability theory with neural networks to create interpretable and probabilistic models. By explicitly modeling joint and marginal probabilities, BSSNN provides insights into the relationships between features and predictions while maintaining the powerful representation capabilities of neural networks.

## Core Features

The BSSNN framework offers several key capabilities that distinguish it from traditional neural networks. It provides explicit probability modeling through its dual-pathway architecture, allowing for both forward (Y|X) and reverse (X|Y) predictions. The framework includes comprehensive uncertainty quantification, making it particularly valuable for applications where understanding prediction confidence is crucial.

The implementation includes built-in visualization tools for model interpretation and a flexible architecture that can be adapted for various types of data and prediction tasks. 

## Installation

Install BSSNN using pip:

```bash
pip install bssnn
```

For development installation:

```bash
git clone https://github.com/Narden91/bssnn.git
cd bssnn
pip install -e .
```

## Quick Start

Here's a simple example of using BSSNN for binary classification:

```python
import torch
from bssnn.model import BSSNN
from bssnn.training import BSSNNTrainer
from bssnn.utils.data import prepare_data

# Initialize model
model = BSSNN(input_size=10, hidden_size=64)

# Create trainer
trainer = BSSNNTrainer(model)

# Train model
for epoch in range(num_epochs):
    loss = trainer.train_epoch(X_train, y_train)
    metrics = trainer.evaluate(X_val, y_val)
```

## Documentation

Detailed documentation is available in the docs directory:
- [API Reference](docs/api.md): Complete API documentation
- [Examples](docs/examples.md): Usage examples and tutorials
- [Improvements](docs/improvements.md): Planned enhancements

## Contributing

We welcome contributions to BSSNN. Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

<!-- ## Citation

If you use BSSNN in your research, please cite: -->

<!-- ```bibtex
@software{bssnn2024,
  title = {BSSNN: Bayesian State-Space Neural Networks},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/bssnn}
}
``` -->
