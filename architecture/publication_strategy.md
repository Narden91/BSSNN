# CISSN Publication Strategy

## 1. Scientific Positioning (The "Hook")

**Problem:** Existing time-series methods force a trade-off between **interpretability** (State-Space Models, ARIMA) and **performance** (Transformers, LSTMs). Pure methods often fail to quantify uncertainty reliably under distribution shift.

**Solution (CISSN):** A hybrid architecture that enforces *structural interpretability* (Level, Trend, Seasonal) using deep learning for the flexible transition dynamics, coupled with *Conformal Prediction* to guarantee uncertainty coverage regardless of model bias.

**Novelty Claim:**
> "We propose Conformally Calibrated Interpretable State-Space Networks (CISSN), a framework that unifies disentangled representation learning with state-conditional conformal prediction. Unlike standard hybrids, CISSN enforces valid physical dynamics (via structured transition matrices) on the latent space and uses this latent structure to adaptively calibrate uncertainty intervals."

## 2. Gap Analysis & Improvements

### A. Methodological Rigor
*   **Ablation Studies:** You must prove every component matters.
    *   *Experiment:* CISSN vs. CISSN (No Seasonal) vs. CISSN (No Conformal).
    *   *Experiment:* 2D Rotation Seasonal vs. 1D Scalar Seasonal (show the "flaw" fix in action).
*   **Baselines:** Compare against:
    *   *Statistical:* ARIMA, ETS.
    *   *DL:* LSTM, DeepAR, N-BEATS.
    *   *Hybrid:* DeepState (Amazon).

### B. Validation Strategy
*   **Datasets:** Use standard benchmarks (M4, M5, Electricity, Traffic) plus one "interpretation-heavy" dataset (e.g., medical vitals or economic indicators) where components have meaning.
*   **Metrics:**
    *   *Accuracy:* MAE, MSE, CRPS (Continuous Ranked Probability Score).
    *   *Uncertainty:* Coverage Error (ACE), Interval Width (MPIW).
    *   *Interpretability:* Visual inspection of decomposed components.

## 3. Critical Thinking Review (Self-Correction)

### Biases to Watch
*   **Selection Bias:** Don't cherry-pick datasets where CISSN wins. Include "failure cases" (e.g., chaotic series with no trend/seasonality) to show honesty.
*   **Complexity Bias:** Ensure the "Deep" part is actually needed. If a simple Kalman Filter works as well, acknowledge it.

### Mathematical Validity
*   **Seasonal Component:** We have fixed the 1D flaw. In the paper, explicitly derive the 2D rotation matrix update rule to demonstrate theoretical soundness:
    $$ \begin{bmatrix} s_{t+1}^{(1)} \\ s_{t+1}^{(2)} \end{bmatrix} = \gamma \begin{bmatrix} \cos \omega & -\sin \omega \\ \sin \omega & \cos \omega \end{bmatrix} \begin{bmatrix} s_t^{(1)} \\ s_t^{(2)} \end{bmatrix} $$

## 4. Brainstorming: Future Directions & Experiments

### "What If?" Scenarios
*   *What if the frequency $\omega$ changes over time?* -> Adaptive seasonality (future work).
*   *What if we have multiple seasonalities?* -> Add more 2D blocks to the $\mathbf{A}$ matrix (e.g., daily + weekly).
*   *What if we reverse it?* -> Use the disentangled state to generate synthetic data (counterfactuals).

### The "Killer App"
*   **Anomaly detection via Conformal Intervals:** If the observed value falls outside the conformal interval, flag it. Since the interval is state-conditional, it adapts to "volatile" vs "stable" periods, reducing false positives.

## 5. Roadmap to Submission
1.  **Refine Code:** (Done - Seasonality fixed, clean architecture).
2.  **Run Benchmarks:** Set up a `benchmark_runner.py` for M4/Electricity.
3.  **Draft Paper:**
    *   **Intro:** The Interpretability-Accuracy trade-off.
    *   **Method:** The math of $\mathbf{A}$ and Conformal Clustering.
    *   **Results:** Tables + Component Plots.
    *   **Discussion:** Why it works (Disentanglement leads to better calibration).
