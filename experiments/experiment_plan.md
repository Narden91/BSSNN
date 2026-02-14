# Experimental Setup for CISSN ArXiv Publication

To ensure a fast and robust publication path, we must demonstrate that CISSN provides **state-of-the-art accuracy** OR **comparable accuracy with superior interpretability and uncertainty quantification**.

## 1. Datasets (The "Standard" Suite)
We will use the standard **Long-Term Time Series Forecasting (LTSF)** benchmarks. Using these ensures reviewers cannot complain about "cherry-picked" data.

| Dataset | Variables | Frequency | Horizon | Description |
| :--- | :--- | :--- | :--- | :--- |
| **ETT (h1, h2, m1, m2)** | 7 | Hourly/15min | 96, 192, 336, 720 | Transformer temperature data. The "Hello World" of LTSF. |
| **Electricity** | 321 | Hourly | 96, 192, 336, 720 | Hourly electricity consumption of 321 clients. High dim. |
| **Traffic** | 862 | Hourly | 96, 192, 336, 720 | Road occupancy rates. Very high dim, complex spatial limits. |
| **Weather** | 21 | 10min | 96, 192, 336, 720 | Local weather metrics. |

**Strategy:**
- Start with **ETTh1** and **ETTh2** for rapid development and tuning.
- Expand to **Electricity** and **Weather** for the paper.
- **Traffic** is optional if we need more evidence, but computationally expensive.

## 2. Baselines
We need to compare against three categories of models:

### A. SOTA Linear/Transformer Models (Accuracy Benchmarks)
- **iTransformer / PatchTST**: Current SOTA. CISSN might not beat them on pure MSE, but must be close.
- **DLinear**: Simple linear baseline. CISSN **MUST** beat this to be taken seriously.

### B. Probabilistic/State-Space Models (Direct Competitors)
- **DeepState (GluonTS)**: The classic RNN+SSM baseline.
- **DeepAR**: Standard probabilistic RNN.

### C. Interpretable Models
- **Prophet / NeuralProphet**: For visual comparison of trend/seasonality decomposition.

## 3. Evaluation Metrics
We will report metrics in two tables:

### Table 1: Deterministic Accuracy (Point Forecasting)
- **MSE (Mean Squared Error)**: Standard.
- **MAE (Mean Absolute Error)**: Standard.

### Table 2: Probabilistic Calibration & Interpretability (The "Selling Point")
- **CRPS (Continuous Ranked Probability Score)**: Measures distribution quality.
- **MSIS (Mean Scaled Interval Score)**: For interval width and coverage.
- **Coverage Error**: Abs(Target Coverage - Actual Coverage).
- **Disentanglement Score**: (Novel metric) Correlation between learned trend and ground-truth trend (using synthetic data).

## 4. Implementation Roadmap

### Phase 1: Data Pipeline (Days 1-2)
- [x] Implement `cissn.data.dataset.BaseETTDataset` to download and process ETT datasets.
- [ ] Create `DataLoaders` consistent with Autoformer/Informer standards (70/10/20 splot).

### Phase 2: Benchmarking Engine (Days 2-3)
- [ ] Create `experiments/bench_trainer.py`: A standardized trainer.
- [ ] Implement **Rolling Window Evaluation** (crucial for time series).
- [ ] Integrate **WandB** for logging.

### Phase 3: Ablation Studies (Day 4)
- **w/o Structure**: Replace structured SSM with standard GRU.
- **w/o Disentanglement Loss**: Train with only MSE.
- **w/o SCCP**: Use standard Conformal Prediction (EnbPI) instead of State-Conditional.

### Phase 4: Visualization & Paper (Day 5)
- [ ] Generate "Component Decomposition" plots (Level, Trend, Seasonal).
- [ ] Generate "Interval Width vs. State" plots (Show how uncertainty adapts to regimes).

## 5. Directory Structure
```
experiments/
├── datasets/           # Raw data
├── baselines/          # Adapter code for baselines
├── configs/            # YAML configs for each dataset
├── run_benchmark.py    # Main entry point
└── analysis.ipynb      # Visualization notebook
```
