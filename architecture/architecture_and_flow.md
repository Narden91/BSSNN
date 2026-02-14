# CISSN Architecture and Data Flow

## 1. System Overview

CISSN (Conformally Calibrated Interpretable State-Space Networks) is a hybrid architecture combining the interpretability of State-Space Models (SSMs) with the learning capacity of Neural Networks. It decomposes time-series data into clinically/physically meaningful components—Level, Trend, Seasonality, and Residuals—while quantifying uncertainty via State-Conditional Conformal Prediction (SCCP).

## 2. Core Architecture

The system consists of three main modules:
1.  **Disentangled State Encoder (`DisentangledStateEncoder`)**: Maps input sequences to a latent state space.
2.  **Forecast Head (`ForecastHead`)**: Projects latent states into future horizons.
3.  **State-Conditional Conformal Predictor (`StateConditionalConformal`)**: Calibrates prediction intervals based on the latent state.

### 2.1 State Space Structure
The latent state $s_t \in \mathbb{R}^5$ is explicitly structured:
-   **Level ($s_t^{(0)}$)**: The baseline value of the series. Slow-varying dynamics.
-   **Trend ($s_t^{(1)}$)**: The rate of change (slope). Smooth dynamics.
-   **Seasonal ($s_t^{(2)}, s_t^{(3)}$)**: A 2D component modeling periodic behavior via rotation.
-   **Residual ($s_t^{(4)}$)**: Fast-varying noise or innovation.

### 2.2 Transition Dynamics
The state evolves according to a structured transition equation:

$$
s_t = \mathbf{A} s_{t-1} + \mathbf{B}(x_t) + \text{NN}(s_{t-1}, x_t)
$$

Where $\mathbf{A}$ is a block-diagonal matrix enforcing component behavior:
$$
\mathbf{A} = \begin{bmatrix}
\alpha_L & 0 & 0 & 0 & 0 \\
0 & \alpha_T & 0 & 0 & 0 \\
0 & 0 & \gamma \cos(\omega) & -\gamma \sin(\omega) & 0 \\
0 & 0 & \gamma \sin(\omega) & \gamma \cos(\omega) & 0 \\
0 & 0 & 0 & 0 & \alpha_R
\end{bmatrix}
$$

-   $\alpha_L, \alpha_T, \alpha_R \in [0, 1]$ are damping factors.
-   $\omega$ is the learnable frequency.
-   $\gamma$ is the seasonal damping factor.
-   $\mathbf{B}(x_t)$ is the "innovation" extracted from input $x_t$ by a neural network.

## 3. Data Flow

### Step 1: Input Processing
-   **Input**: Batch of time series $X \in \mathbb{R}^{B \times T \times D_{in}}$.
-   **Projection**: $X$ is projected to hidden size $H$ via `Linear -> LayerNorm -> GELU`.

### Step 2: State Encoding (Recurrent Loop)
For each time step $t$:
1.  **Innovation Extraction**: Neural networks extract "innovations" (updates) for each state component from the projected input.
2.  **Linear Transition**: The previous state $s_{t-1}$ is updated using matrix $\mathbf{A}$.
3.  **Non-Linear Correction**: A small residual network adds detailed corrections to the linear update.
4.  **Output**: New state $s_t$.

### Step 3: Forecasting
-   The final state $s_T$ is passed to the `ForecastHead`.
-   **Linear Path**: Each component (Level, Trend, Seasonal, Residual) is linearly projected to the forecast horizon.
-   **Non-Linear Path**: A small MLP refines the sum.
-   **Aggregation**:
    $$ \hat{y} = \text{Level} + \text{Trend} + \text{Seasonal}_{sum} + \text{Residual} + \text{Bias} $$

### Step 4: Uncertainty Quantification (Training/Calibration)
1.  **Calibration Set**: A hold-out set is encoded into states $S_{cal}$ and residuals $R_{cal} = |y - \hat{y}|$.
2.  **Clustering**: K-Means clusters the state space $S_{cal}$ into $K$ regimes (e.g., "High Trend", "Stable", "Volatile").
3.  **Quantile Estimation**: For each cluster, the $1-\alpha$ quantile of residuals is computed ($q_k$).

### Step 5: Inference
1.  New input $x_{new}$ is encoded to $s_{new}$.
2.  Point forecast $\hat{y}_{new}$ is generated.
3.  $s_{new}$ is assigned to a cluster $k$.
4.  Interval is $\hat{y}_{new} \pm q_k$.

## 4. Key Improvements (From Recent Fixes)
-   **2D Seasonality**: The seasonal component was upgraded from a scalar to a 2D rotation system, allowing for the representation of true oscillating sine/cosine waves rather than just decaying exponentials.
-   **5D State**: Data flow now explicitly handles 5 dimensions throughout the encoder, loss, and forecast head.
