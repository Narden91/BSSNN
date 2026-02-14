# README.md

# Conformally Calibrated Interpretable State-Space Networks (CISSN)

CISSN (formerly BSSNN) is a framework that combines State-Space Models (SSMs) with neural networks to create interpretable and probabilistic time-series models. By explicitly modeling level, trend, seasonality, and residuals, CISSN provides deep insights into the drivers of predictions while maintaining the powerful representation capabilities of neural networks.

## Core Features

*   **Interpretability**: Decomposes time series into Level, Trend, Seasonal, and Residual components.
*   **Conformal Prediction**: Provides statistically valid prediction intervals using State-Conditional Conformal Prediction (SCCP).
*   **Dual-Pathway**: Supports both forward (Y|X) and reverse (X|Y) predictions (in development).
*   **Flexibility**: Built on PyTorch for easy extension and integration.

## Installation

Install CISSN using pip:

```bash
pip install cissn
```

For development installation:

```bash
git clone https://github.com/Narden91/bssnn.git
cd bssnn
pip install -e .
```

## Quick Start

Here's a simple example of using CISSN for forecasting:

```python
import torch
from cissn.models import DisentangledStateEncoder, ForecastHead
from cissn.conformal import StateConditionalConformal

# Initialize model
model = DisentangledStateEncoder(input_dim=10, state_dim=5)
head = ForecastHead(state_dim=5, horizon=5)

# Forward pass
x = torch.randn(1, 20, 10) # (batch, seq_len, input_dim)
state = model(x)
forecast = head(state)

print(f"Forecast: {forecast.shape}")
```

See `examples/demo_cissn.py` for a complete example including conformal calibration.

## Documentation

Detailed documentation is available in the docs directory:
- [API Reference](docs/api.md): Complete API documentation
- [Examples](docs/examples.md): Usage examples and tutorials

## Contributing

We welcome contributions to CISSN. Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
