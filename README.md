# Fractal Wavelet Tail-Risk Warning System

## Overview
This project implements a **Point-in-Time (PIT) Tail Risk Warning System** based on the **Fractal Markets Hypothesis (FMH)**. Unlike the Efficient Market Hypothesis (EMH), FMH suggests that market stability is derived from the diversity of investment horizons (fractal structure). A crash occurs when this structure breaks down and the market becomes dominated by a single (usually short-term) investment horizon.

## Methodology

### 1. Fractal Markets Hypothesis (FMH)
The core thesis is that markets are stable as long as long-term investors (e.g., pension funds) provide liquidity to short-term speculators. When a tail-risk event approaches, long-term investors often withdraw, leaving short-term high-frequency players to dominate. This "loss of horizon diversity" leads to liquidity voids and extreme volatility.

### 2. Point-in-Time (PIT) Constraints
To ensure the system is viable for real-world trading, the analysis is strictly **Point-in-Time**. For every day $t$ in the backtest:
*   We use a rolling window of $W = 256$ trading days.
*   The Continuous Wavelet Transform (CWT) is calculated only for the window $[t-W+1, t]$.
*   We extract only the power values corresponding to the final day $t$ (the last column of the CWT matrix).
*   **Result**: Zero look-ahead bias. The signals generated at time $t$ use only data available at time $t$.

### 3. Mathematical Foundations

#### Log-Returns
Daily returns are calculated as:
$$r_t = \ln(P_t) - \ln(P_{t-1})$$

#### Continuous Wavelet Transform (CWT)
We use the **Complex Morlet Wavelet** ($cmor 1.5-1.0$) to decompose the returns $r_t$ into time-frequency components. The wavelet power $P(s, t)$ at scale $s$ and time $t$ is calculated as the squared magnitude of the wavelet coefficients.

#### Short-Horizon Dominance Ratio ($R_t$)
We define the dominance ratio as the proportion of total power concentrated in short investment horizons (2 to 16 days):
$$R_t = \frac{\sum_{s \in [1, 16]} P(s, t)}{\sum_{s \in [1, 512]} P(s, t)}$$

#### Signal Generation (Z-Score)
To identify anomalies in the dominance ratio, we calculate a rolling 60-day Z-score:
$$Z_t = \frac{R_t - \mu_{R, 60}}{\sigma_{R, 60}}$$
A **WARNING** signal is triggered when $Z_t > 2.0$.

#### Total Power (Panic Measure)
Total aggregated power across all scales signifies market-wide energy:
$$P_{total, t} = \sum_{s=1}^{512} P(s, t)$$
While $R_t$ detects the *structural fragility*, $P_{total, t}$ detects the *actual panic* where all investment horizons show extreme volatility.

## System Architecture

*   `wavelet_risk_engine.py`: Core `FractalWaveletManager` class containing PIT logic and plotting.
*   `multi_market_runner.py`: Backtest orchestrator for global indices.
*   `daily_monitor.py`: Routine monitoring script for current market status.

## Interpretation
1.  **Red Shading (Price Chart)**: Active tail-risk warnings based on $Z_t > 2.0$.
2.  **Heatmap (Horizon Axis)**: Visualizes the concentration of energy. Hot colors (red) at low values (bottom) indicate short-term dominance.
3.  **Total Power Panel**: Spikes indicate broad market volatility spikes (panic).

---
*Based on the research of Edgar E. Peters and the Fractal Markets Hypothesis.*
