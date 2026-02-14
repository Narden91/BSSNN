# Conformally Calibrated Interpretable State-Space Networks (CISSN)

CISSN (formerly BSSNN) is a framework that combines State-Space Models (SSMs) with neural networks to create interpretable and probabilistic time-series models. By explicitly modeling level, trend, seasonality, and residuals, CISSN provides deep insights into the drivers of predictions while maintaining the powerful representation capabilities of neural networks.

## Core Features

*   **Interpretability**: Decomposes time series into Level, Trend, Seasonal, and Residual components.
*   **Explainable Forecasting**: Built-in `ForecastExplainer` provides per-component contribution analysis for every prediction.
*   **Conformal Prediction**: Provides statistically valid prediction intervals using State-Conditional Conformal Prediction (SCCP).
*   **Structured Data Loading**: Robust `BaseETTDataset` implementation for consistent handling of ETT benchmarks (Hourly and Minute).
*   **Dual-Pathway**: Supports both forward (Y|X) and reverse (X|Y) predictions (in development).
*   **Flexibility**: Built on PyTorch for easy extension and integration.

## Installation

Install CISSN using pip:

```bash
uv pip install -e .
```

For development installation:

```bash
git clone https://github.com/Narden91/CISSN.git
cd CISSN
uv pip install -e .
```

## Quick Start

### Basic Usage

Here's a simple example of using CISSN for forecasting:

```python
import torch
from cissn.models import DisentangledStateEncoder, ForecastHead
from cissn.conformal import StateConditionalConformal
from cissn.explanations import ForecastExplainer

# Initialize model
model = DisentangledStateEncoder(input_dim=10, state_dim=5)
head = ForecastHead(state_dim=5, horizon=5)

# Forward pass
x = torch.randn(1, 20, 10) # (batch, seq_len, input_dim)
state = model(x)
forecast = head(state)

print(f"Forecast: {forecast.shape}")

# Explain Forecast
explainer = ForecastExplainer(head)
explanations = explainer.explain(state)
print(f"Level Contribution: {explanations[0].level_contribution}")
```

### Running the Demo

See the full workflow including conformal calibration and explanations:

```bash
uv run examples/demo_cissn.py
```

## Documentation

Detailed documentation is available in the `docs` and `architecture` directories.


## Contributing

We welcome contributions to CISSN. Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
