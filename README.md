# QuantJL

**Deep Reinforcement Learning for Algorithmic Trading. In Julia, from scratch.**

[![Julia](https://img.shields.io/badge/Julia-1.8+-blue.svg)](https://julialang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-orange.svg)]()
![alt text](plots/gbm_path_full.png)
## Introduction

QuantJL is a sophisticated algorithmic trading system that uses Deep Deterministic Policy Gradient (DDPG) reinforcement learning to optimize stock trading strategies. The system learns to make optimal trading decisions by analyzing minute-level intraday market data and technical indicators to maximize return on investment while managing risk.

### Problem Statement

Traditional trading strategies often rely on static rules or simple technical indicators that fail to adapt to changing market conditions. QuantJL addresses this by implementing a continuous control reinforcement learning agent that learns optimal trading policies through interaction with market data. The agent determines whether to **hold (0)** or **long (1)** positions based on the state of the market over the last 30 minutes of 10 features (V-score signals, technical indicators and time of day).

### Results
Unfortunately, the average reward from each episode seems to plateau at around -0.01 cents, which I believe to be largely due to transaction costs. This behavior is consistent with minute-level intraday environments in which transaction costs dominate per-timestep returns. Because the RL agent must output an action every single minute, even small position adjustments generate cumulative friction that overwhelms micro-profits.
![alt text](total_rewards_nice_graph.png)

However, the brownian motion indicator later discussed the **vscore formulation** part of this readme, shows promise, as the movement of an asset changes direction after a critical value. For example, the backtest (`test/gbm_backtest.jl`) takes a long position (all-in) when the vscore goes below -2.0 and sells (all-out) when it goes above 1.72. The following is one of many example backtests done on this strategy.
![alt text](backtests/SUIG.png)

**However**, the backtests enforce a **trade-cooldown of 30 minutes** while the reinforcement learning algorithm does not. 
This throttling drastically reduces transaction costs and prevents repeated oscillation around the threshold. Consequently:

* The strategy trades only when the signal is strong,
* avoids noise-driven whipsaws, and
* pays transaction costs orders of magnitude less frequently than the RL agent.

## Key Features

- **Deep Deterministic Policy Gradient (DDPG)** implementation for continuous action spaces
- **minute-level intraday trading** support with minute-level data processing
- **Custom technical indicators** including V-score signals, RSI, EMA, Bollinger %B, volume and time-of-day encodings (MACD and VWAP are implemented in `data.jl` but not used in the state)
- **Risk management** with volatility penalties and capital protection mechanisms
- **Experience replay** (10,000-transition FIFO buffer, uniform sampling)
- **Target networks** for stable training with soft updates
- **Ornstein-Uhlenbeck noise** for exploration during training
- **Comprehensive visualization** tools for monitoring training progress

## Architecture Overview

QuantJL implements a Deep Deterministic Policy Gradient (DDPG) algorithm, a model-free, off-policy actor-critic method for continuous control. The architecture consists of four neural networks working in tandem to learn optimal trading policies.

### Core Components

**Actor Network (π)**: A deterministic policy network that maps market states to trading actions. The network takes as input the last 30 minutes of 10 features plus the current market allocation and outputs a continuous action value between 0 and 1 (hold to long).

**Critic Network (Q)**: A Q-value network that evaluates state-action pairs. It takes both the market state and the action as input and outputs the expected return for that state-action combination.

**Target Networks**: Separate target networks (π_target, Q_target) for both actor and critic to provide stable learning targets and reduce training instability through soft updates.

**Experience Replay Buffer**: Stores past experiences (state, action, reward, next_state, done) with uniform sampling for efficient learning from diverse experiences.

### DDPG Algorithm Pseudocode

Based on OpenAI Spinning Up's DDPG implementation:

```python
# DDPG Algorithm Pseudocode
def ddpg_algorithm():
    # Initialize networks
    π, Q = initialize_networks()
    π_target, Q_target = copy_networks(π, Q)
    replay_buffer = ReplayBuffer()
    
    for episode in range(num_episodes):
        state = environment.reset()
        episode_reward = 0
        
        for t in range(max_timesteps):
            # Select action with exploration noise
            action = π(state) + noise()
            action = clip(action, action_low, action_high)
            
            # Execute action in environment
            next_state, reward, done = environment.step(action)
            
            # Store experience
            replay_buffer.store(state, action, reward, next_state, done)
            
            # Update networks
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample_batch(batch_size)
                
                # Update critic
                Q_target_values = Q_target(next_state, π_target(next_state))
                y = reward + gamma * (1 - done) * Q_target_values
                Q_loss = MSE(Q(state, action), y)
                Q.update(Q_loss)
                
                # Update actor
                π_loss = -Q(state, π(state)).mean()
                π.update(π_loss)
                
                # Soft update target networks
                soft_update(π_target, π, tau)
                soft_update(Q_target, Q, tau)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
```

### Mathematical Formulation

**Policy Gradient**: The actor network is updated using the deterministic policy gradient:

$$\nabla_{\theta_{\pi}} J \approx \mathbb{E}_{s \sim \rho^{\beta}} \left[ \nabla_{\theta_{\pi}} Q(s, a| \theta_Q) \big|_{a=\pi(s|\theta_{\pi})} \nabla_{\theta_{\pi}} \pi(s|\theta_{\pi}) \right]$$

**Q-Learning Update**: The critic network is updated using the Bellman equation:

$$y_i = r_i + \gamma Q'(s_{i+1}, \pi'(s_{i+1}|\theta_{\pi'})|\theta_{Q'})$$

$$L = \frac{1}{N} \sum_i (y_i - Q(s_i, a_i|\theta_Q))^2$$

**Soft Target Updates**: Target networks are updated using exponential moving averages:

$$\theta_{Q'} \leftarrow \tau \theta_Q + (1-\tau) \theta_{Q'}$$

$$\theta_{\pi'} \leftarrow \tau \theta_{\pi} + (1-\tau) \theta_{\pi'}$$

### State Space

The state representation combines multiple technical indicators:

$$s_t = [f^{(1)}_{t-29:t}, \dots, f^{(10)}_{t-29:t}, \text{alloc}_t] \in \mathbb{R}^{301}$$

Each of the 10 feature columns from `get_all_features` (`data.jl`) contributes its last 30 values (`LOOK_BACK_PERIOD = 30`), concatenated column by column, followed by the current market allocation (1 dimension): 10 × 30 + 1 = 301. The next state $s'$ is the same window shifted forward by one bar.

The 10 features, in order:
1. `buy_sig` = max(0, −1.5 − V)
2. `sell_sig` = max(0, V − 1.5)
3. `buy_sig_deriv`: first difference of `buy_sig`
4. `sell_sig_deriv`: first difference of `sell_sig`
5. `ema`: 14-period EMA of close (z-standardized over the day)
6. `volume`: log base 1000 of (volume + 1)
7. `rsi`: 14-period RSI (z-standardized over the day)
8. `bb_percentb`: Bollinger %B (20-period, 2σ)
9. `sin_feat` = sin(2π · minute / 420)
10. `cos_feat` = cos(2π · minute / 420)

Here *minute* is hour × 60 + minute of the bar's timestamp. Only `rsi` and `ema` are standardized.

### Action Space

Continuous action space representing position sizing. The actor's sigmoid output $a_t$ is perturbed by Ornstein-Uhlenbeck noise (θ = 0.15, μ = 0, σ = 0.2, dt = 1; σ decays each step as σ ← max(0.05, σ·e^{−0.00005})) and clipped to give the target allocation:

$$\text{alloc}_t = \text{clip}(a_t + \varepsilon_t, 0, 1)$$

Where:
- $a_t = 0$: Hold (no position)
- $a_t = 1$: Long position (full capital allocation)

### Reward Function

Each step (`test/quantposgbm.jl`) first charges a transaction cost on the allocation change, then applies the next bar's percent price change:

$$
\text{cost}_t = 0.0002 \cdot |\text{alloc}_t - \text{alloc}_{t-1}| \cdot \text{capital}
$$
$$
\text{marketReturn}_t = \text{alloc}_t \cdot \text{capital}_t \cdot \frac{\text{priceChange}_{t+1}}{100}
$$
$$
\text{rawReturn}_t = \text{marketReturn}_t - \text{cost}_t
$$

The reward fed to the agent incorporates multiple risk management components:

$$
r_t = \frac{\text{rawReturn}_t}{\text{prevCapital}_t} - \text{riskPenalty}_t - \text{capitalPenalty}_t
$$
$$
\text{riskPenalty}_t = 0.1 \cdot \text{std}(\text{last 20 returns})
$$
$$
\text{capitalPenalty}_t =
\begin{cases}
  0.2 \cdot \dfrac{1000 - \text{capital}_t}{1000}, & \text{if } \text{capital}_t < 1000 \\
  0, & \text{otherwise}
\end{cases}
$$

Negative rewards are scaled by 0.75. An episode (one random trading day, starting capital 1000) ends when capital falls below 950 or the day ends; the final step adds a terminal bonus of $15 \cdot (\text{capital} - 1000)/1000$. Episodes are compared to a buy-and-hold benchmark.

**Key Insight**: During initial experiments, when trading penalties were set too high, the actor network learned to avoid making any movements whatsoever, effectively learning a "do nothing" policy. This highlights the importance of carefully balancing risk penalties with trading incentives to encourage active learning while maintaining risk management.

### V-Score Mathematical Formulation

The V-score is a custom volatility indicator based on Monte Carlo simulation of Geometric Brownian Motion (GBM):

**GBM Process**:
For each time $t$, `vscore` in `gbm.jl` takes the last OBS prices, computes their simple returns, and sets $\mu$ = mean and $\sigma$ = std of those returns. It then simulates EPOCH paths of EXT steps from the current price $s_0$, with $\Delta t$ = 1 bar:

$$S_k = s_0 \exp\left(\left(\mu + \frac{\sigma^2}{2}\right)k + \sigma \sum_{j=1}^{k} Z_j\right), \quad k = 1,\dots,\text{EXT}$$

Where:
- $s_0$: Current stock price
- $\mu$, $\sigma$: Mean and standard deviation of the window's simple returns
- $Z_j \sim \mathcal{N}(0,1)$: Standard normal random variables

Note the drift in the code is $\mu + \frac{\sigma^2}{2}$ and there is no $\sqrt{\Delta t}$ factor, since each step is one bar.

**V-Score Calculation**: the raw score is the fraction of all simulated path points (every path, every step) above $s_0$:

$$v_t = \frac{\text{count}(S_{i,k} > s_0)}{\text{EPOCH} \cdot \text{EXT}}$$

The raw series is then z-scored over the **whole series** (not a rolling window):

$$V_t = \frac{v_t - \text{mean}(v)}{\text{std}(v)}$$

Defaults as called: feature pipeline OBS = 100 (passed as `LOOK_BACK_PERIOD`), EPOCH = 1000, EXT = 20; backtest OBS = 100, EPOCH = 5000, EXT = 20. The RL features use thresholds of ±1.5 (`buy_sig`, `sell_sig`), while the backtest buys below −2.0 and sells above 1.72.

### Key DDPG Aspects

**State Representation**: The state space combines 30 minutes of 10 market features (V-score signals, indicators, time of day) with the current market allocation, providing the agent with both market context and portfolio state information.

**Action Space**: Continuous actions allow for nuanced position sizing rather than discrete buy/sell decisions, enabling more sophisticated trading strategies.

**Reward Design**: The multi-component reward function balances profit maximization with risk management, preventing the agent from learning overly conservative or aggressive strategies.

**Exploration Strategy**: Ornstein-Uhlenbeck noise provides correlated exploration that maintains temporal consistency in action selection while encouraging exploration.

**Target Networks**: Soft updates to target networks provide stable learning targets, preventing the instability common in direct policy gradient methods.

**Experience Replay**: Uniform sampling from the replay buffer ensures the agent learns from diverse market conditions and prevents catastrophic forgetting.

### Architecture Diagram

```
┌─────────────────┐
│   Market Data   │  (OHLCV)
└─────────────────┘
         │
         ▼
┌─────────────────────────────┐
│     Feature Engineering     │
│  (Indicators + V-Scores)    │
└─────────────────────────────┘
         │
         ▼
      state s_t
         │
         ├─────────────────────────────┐
         ▼                             │
┌─────────────────┐                    │
│   Actor π(s_t)  │────────────────────┤  action a_t (0–1) + OU noise
└─────────────────┘                    │
         │                             │
         ▼                             │
┌─────────────────────────────┐        │
│        Environment          │◀───────┘
│  (apply a_t, get r_t, s′)   │
└─────────────────────────────┘
         │
         ▼
(Store transition)
(s_t, a_t, r_t, s′, d_t)
         │
         ▼
┌─────────────────────────────┐
│       Replay Buffer         │
└─────────────────────────────┘
         │         │
   sample▼         ▼sample
┌─────────────────┐   uses targets for y = r + γ(1–d)Q⁻(s′,π⁻(s′))
│  Critic Q(s,a)  │<───────────────────────────────────────────────┐
└─────────────────┘                                                │
    ▲     │ backprop MSE on (Q – y)                                │
    │     └────────────────────────────────────────────────────────┘
    │                                        targets
    │                          ┌───────────────────────────────────┐
    │                          │                                   │
    │                   ┌───────────────┐                    ┌───────────────┐
    │                   │  Target Q⁻    │◀── soft update ───▶│     Q         │
    │                   └───────────────┘                    └───────────────┘
    │
    │ policy gradient via ∇_a Q(s, a)|_{a=π(s)}
    │
    │     ┌───────────────────────────────────────────────────────────────┐
    │     │  g = (∂Q/∂a) at (s, π(s));  backprop through Actor with –g   │
    │     ▼                                                               │
┌─────────────────┐                                                       │
│   Actor π(s)    │<──────────────────────────────────────────────────────┘
└─────────────────┘
    ▲                                   targets
    │                   ┌───────────────┐                    ┌───────────────┐
    │                   │  Target π⁻    │◀── soft update ───▶│     π         │
    │                   └───────────────┘                    └───────────────┘
    │
    └── (next step uses updated π to act)

```

## Getting Started

### Prerequisites

- **Julia 1.8+** ([Download](https://julialang.org/downloads/))
- **Python 3.7+** (for data downloading)
- **Financial Modeling Prep API key** ([Get API key](https://financialmodelingprep.com/))
- **Git** for version control

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sentientplatypus/quantjl.git
   cd quantjl
   ```

2. **Install Julia dependencies**:
   ```julia
   using Pkg
   Pkg.add(["CSV", "DataFrames", "Statistics", "Dates", "Plots", "UnicodePlots", "Test", "Random", "StatsBase"])
   ```

3. **Install Python dependencies**:
   ```bash
   pip install pandas certifi
   ```

4. **Set up API key**:
   ```bash
   echo "your_api_key_here" > apikey
   ```

### Environment Variables

Create a `.env` file (optional):
```bash
FMP_API_KEY=your_api_key_here
DATA_DIR=./data
PLOTS_DIR=./plots
```

## Quickstart

### Download Market Data

```bash
# Download minute-level intraday data for Microsoft (MSFT) for the past 30 days
python download.py MSFT
```

### Run Training

```julia
# Start Julia REPL
julia

# Include the main training script
include("test/quantposgbm.jl")
```

### Minimal Example

```julia
using Random
include("quant.jl")
include("data.jl")

# Set random seed for reproducibility
Random.seed!(3)

# Create neural networks (state = 10 features × 30 lookback + allocation = 301)
nIndicators, LOOK_BACK_PERIOD = 10, 30
π_ = Net([Layer(nIndicators * LOOK_BACK_PERIOD + 1, 300, relu, relu′),
          Layer(300, 200, relu, relu′),
          Layer(200, 100, relu, relu′),
          Layer(100, 50, relu, relu′),
          Layer(50, 30, relu, relu′),
          Layer(30, 10, relu, relu′),
          Layer(10, 1, sigmoid, sigmoid′)], mse_loss, mse_loss′)

Q̂ = Net([Layer(nIndicators * LOOK_BACK_PERIOD + 2, 300, relu, relu′),
         Layer(300, 200, relu, relu′),
         Layer(200, 100, relu, relu′),
         Layer(100, 50, relu, relu′),
         Layer(50, 30, relu, relu′),
         Layer(30, 10, relu, relu′),
         Layer(10, 1, idty, idty′)], mse_loss, mse_loss′)

# Initialize DDPG agent
quant = Quant(π_, Q̂, 0.99, 0.005)

# Get market data
price_data = get_historical("MSFT")
vscores = get_historical_vscores("MSFT")   # OBS=100, EPOCH=1000, EXT=20

# Training loop (simplified)
for episode in 1:100
    # ... training logic ...
end
```

## Configuration

### Network Architecture

The default network configuration can be modified in the test files:

```julia
# Actor network: 301 → 300 → 200 → 100 → 50 → 30 → 10 → 1 (ReLU, sigmoid output)
# Critic network: 302 → 300 → 200 → 100 → 50 → 30 → 10 → 1 (ReLU, identity output)
# See test/quantposgbm.jl and the Minimal Example above for the full definitions.
```

### Training Parameters

Key hyperparameters can be adjusted:

```julia
γ = 0.99          # Discount factor
τ = 0.005         # Target network update rate
α_Q = 3e-4        # Critic learning rate
α_π = 5e-5        # Actor learning rate
λ = 1e-4          # Regularization parameter
batch_size = 256  # Training batch size
# called as train!(quant, 3e-4, 5e-5, 0.0001, 256)
```

## Data & Preprocessing

### Data Sources

- **Financial Modeling Prep API**: minute-level intraday (1-minute) and historical data
- **Supported tickers**: MSFT, AAPL, NVDA, PLTR, SPY, and more
- **Data format**: CSV files with OHLCV data and calculated percentage changes

### Data Directory Layout

```
data/
├── 2025-06-14/          # Date-based directories
│   ├── MSFT_day1.csv    # Daily minute-level intraday data
│   ├── MSFT_day2.csv
│   └── ...
├── MSFT.csv             # Historical data
├── AAPL.csv
└── merge.csv            # Combined historical data
```

### Preprocessing Commands

```bash
# Download historical data
python download.py MSFT AAPL NVDA

# Download minute-level intraday data for past 30 days
python download.py MSFT
```

### Technical Indicators

The system calculates multiple technical indicators:

- **V-Score signals**: `buy_sig`/`sell_sig` (±1.5 thresholds) and their first differences
- **RSI**: Relative Strength Index (14-period)
- **EMA**: Exponential Moving Average (14-period)
- **Bollinger Bands %B**: Bollinger Band position indicator (20-period, 2σ)
- **Volume**: log base 1000 of (volume + 1)
- **Time-of-day features**: sin/cos encoding with a 420-minute period

`data.jl` also implements MACD and VWAP, but they are commented out of `get_all_features` and are not part of the state.

## Training

### Single-GPU Training

```julia
# Run the main training script
include("test/quantposgbm.jl")
```

### Training Configuration

```julia
Random.seed!(3)
ticker = "ASST"
LOOK_BACK_PERIOD = 30     # Minutes of history per feature in the state
NUM_EPISODES = 200        # Training episodes (one random day each)
INITIAL_CAPITAL = 1000.0  # Starting capital
MIN_CAPITAL = 950.0       # Episode termination threshold
```



The system automatically saves:
- Training progress plots in `plots/` directory
- Capital distribution analysis
- Action trajectory visualizations

## Evaluation & Benchmarks

### Performance Metrics

The system tracks several key metrics:

- **Total Return**: Cumulative profit/loss over episodes
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Capital Preservation**: Ability to maintain initial capital

### Benchmark Comparison

Results are compared against:
- **Buy & Hold**: Keeping 100% of capital invested in the market at all times
- **Random Trading**: Random action selection with uniform distribution
- **Technical Indicators**: Traditional technical analysis strategies

### Reproducing Results

To reproduce the published results:

```julia
# Set random seed
Random.seed!(3)

# Use default configuration
include("test/quantposgbm.jl")

# Results will be saved to plots/ directory
```

## Monitoring & Experiment Tracking

### Visualization Tools

The system provides comprehensive visualization:

```julia
# Visualize neural network activations
include("test/visualize.jl")
visualize_net(net, input_vector)

# Plot training progress
plot(capitals, title="Capital over time")
plot(actions, title="Actions over time")
```

### Output Files

Training generates several visualization files:
- `plots/total_rewards.png`: Training reward progression
- `plots/capital_distribution/`: Episode-by-episode capital analysis
- `plots/better_rewards.png`: Reward function analysis

## Security, Privacy & Responsible AI

### Data Privacy

- **API Keys**: Stored locally in `apikey` file (excluded from version control)

### Performance Considerations

- **Training Time**: Full training can take several hours on CPU
- **Memory Requirements**: Requires sufficient RAM for replay buffer
- **Data Storage**: minute-level intraday data requires significant disk space

## Roadmap
- [ ] **Short Selling Support**: Extend action space to include short positions (-1)
- [ ] **Multi-Asset Training**: Implement portfolio-level optimization
- [ ] **Hyperparameter Optimization**: Automated hyperparameter tuning
- [ ] **GPU Support**: CUDA acceleration for faster training

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

**This software is for educational and research purposes only. It is not intended for live trading or investment advice. Trading financial instruments involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results.**

## Acknowledgments

- Financial Modeling Prep for market data API
- OpenAI Spinning Up DDPG page.

---

For questions, issues, or contributions, please open an issue or contact the maintainers.





