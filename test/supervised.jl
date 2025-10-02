# supervised_allocator.jl
# A simple supervised baseline (no RL):
# - Input: stacked feature window from data.jl (same lookback rolling window)
# - Target: next-step direction (up=1, down=0) from percent changes
# - Loss: MSE between sigmoid(output) and label
# - Allocation: use the network output directly as allocation in [0,1]
# - Plot: agent capital vs. buy&hold for a sample episode

using Random
using Statistics
using Plots

include("../nn.jl")
include("../data.jl")

# -------------------------
# Config
# -------------------------
const TICKER            = "MSFT"
const LOOK_BACK_PERIOD  = 30
const DAYS_TO_SAMPLE    = 30           # how many day-sessions to load
const NUM_EPOCHS        = 10            # sweep the data multiple times
const LR                = 1e-3         # learning rate
const WD                = 1e-4         # L2 weight decay
const TX_COST_RATE      = 0.0002       # same cost you used elsewhere
const INIT_CAPITAL      = 1000.0
const SEED              = 3

# -------------------------
# Utilities
# -------------------------

# Build a single input vector by stacking each indicator's LOOK_BACK_PERIOD window.
# This mirrors your RL state construction (minus capital).
function build_input_window(day_df, t::Int, L::Int)
    # Concatenate each column's window [t-L+1 : t]
    return vcat([day_df[!, col][t - L + 1:t] for col in names(day_df)]...)
end

# One pass over all days, online SGD.
function train_one_epoch!(net::Net, month_features, month_prices; lr::Float64=LR, wd::Float64=WD)
    total_loss = 0.0
    n_samples = 0

    for day in 1:length(month_prices)
        day_df     = month_features[day]
        day_change = month_prices[day]             # percent changes vector (post-lookback in data.jl)

        # Iterate time steps where (input at t) -> label from (t+1)
        for t in LOOK_BACK_PERIOD:(nrow(day_df)-1)
            x = build_input_window(day_df, t, LOOK_BACK_PERIOD)

            # label: up (1) if next %change > 0, else 0
            y = (day_change[t+1] > 0.0) ? [1.0] : [0.0]

            # Forward (net(x) already called inside step!)
            # We want the raw output AFTER sigmoid, so set the last layer to sigmoid.
            # Our Net already has per-layer activations; we'll build the model accordingly below.
            step!(net, x, y, lr, wd, 1.0, true)

            # Compute loss for logging (MSE)
            ŷ = net.output.a
            total_loss += sum((ŷ .- y).^2)
            n_samples  += 1
        end
    end

    return total_loss / max(n_samples, 1)
end

# Simulate one episode's capital using the trained net.
function simulate_episode(net::Net, day_df, day_change)
    capital_traj    = Float64[INIT_CAPITAL]
    bh_traj         = Float64[INIT_CAPITAL]
    alloc_traj      = Float64[]
    alloc           = 0.0
    capital         = INIT_CAPITAL
    share_price     = 100.0                  # virtual reference for buy&hold cap

    for t in LOOK_BACK_PERIOD:(nrow(day_df)-1)
        x = build_input_window(day_df, t, LOOK_BACK_PERIOD)
        ŷ = net(x)                           # forward
        a = clamp(ŷ[1], 0.0, 1.0)            # allocation in [0,1]
        push!(alloc_traj, a)

        # transaction cost for rebalancing
        Δa = a - alloc
        cost = TX_COST_RATE * abs(Δa) * capital
        capital -= cost
        alloc = a

        # apply market move with percent change at t+1
        r = day_change[t+1] / 100.0
        pnl = alloc * capital * r
        capital += pnl

        # benchmark buy & hold at 100% allocation
        share_price *= (1 + r)
        bh_capital = INIT_CAPITAL * (share_price / 100.0)

        push!(capital_traj, capital)
        push!(bh_traj, bh_capital)
    end

    return capital_traj, bh_traj, alloc_traj
end

# -------------------------
# Build data & model
# -------------------------
Random.seed!(SEED)
month_features, month_prices = get_month_features(TICKER, DAYS_TO_SAMPLE, LOOK_BACK_PERIOD)
@assert !isempty(month_features) "No features loaded—check your data files and get_month_features."

nIndicators = ncol(month_features[1])
input_dim   = nIndicators * LOOK_BACK_PERIOD

# "Normal net": input -> hidden -> ... -> 1 (sigmoid)
model = Net([
    Layer(input_dim, 256, relu,      relu′),
    Layer(256,      128,  relu,      relu′),
    Layer(128,      64,   relu,      relu′),
    Layer(64,       32,   relu,      relu′),
    Layer(32,       1,    sigmoid,   sigmoid′)
], mse_loss, mse_loss′)

# -------------------------
# Train
# -------------------------
println("Training supervised allocator on $(TICKER) with input_dim=$(input_dim)")
loss_hist = Float64[]
for epoch in 1:NUM_EPOCHS
    epoch_loss = train_one_epoch!(model, month_features, month_prices; lr=LR, wd=WD)
    push!(loss_hist, epoch_loss)
    println("Epoch $epoch  MSE: $(round(epoch_loss, digits=6))")
end

# -------------------------
# Evaluate on test set
# -------------------------

# Split: first 70% of days for training, rest for testing
n_days = length(month_prices)
n_train = Int(floor(0.7 * n_days))
train_days = 1:n_train
test_days  = (n_train+1):n_days

agent_final  = Float64[]
bh_final     = Float64[]

for day in test_days
    day_df     = month_features[day]
    day_change = month_prices[day]
    agent_cap, bh_cap, _ = simulate_episode(model, day_df, day_change)
    push!(agent_final, agent_cap[end])
    push!(bh_final,    bh_cap[end])
end

avg_agent = mean(agent_final)
avg_bh    = mean(bh_final)

println("=== Test Set Results on $(length(test_days)) days ===")
println("Average terminal capital (Agent):     $(round(avg_agent, digits=2))")
println("Average terminal capital (Buy&Hold):  $(round(avg_bh, digits=2))")
println("Relative outperformance:              $(round(avg_agent - avg_bh, digits=2))")
println("Agent / B&H ratio:                    $(round(avg_agent / avg_bh, digits=3))")

# Optional: plot distributions
histogram(agent_final, alpha=0.5, label="Agent", bins=15)
histogram!(bh_final, alpha=0.5, label="Buy & Hold", bins=15,
           title="Distribution of Final Capitals (Test Days)",
           xlabel="Terminal Capital", ylabel="Frequency")
savefig("plots/supervised/test_performance.png")
println("Saved plots/supervised/test_performance.png")
