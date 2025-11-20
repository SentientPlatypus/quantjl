using CSV
using DataFrames
using Statistics
using Dates
using Plots

include("../data.jl")

# --- your existing helpers (assumed already defined) ---
# get_historical_raw, get_historical_vscores, etc.

"""
    backtest_vscore_strategy(ticker::String;
                             buy_threshold=-2.0,
                             sell_threshold=2.0,
                             initial_capital=1000.0,
                             cooldown_minutes=30)

Backtest the simple vscore strategy:

- If vscore < buy_threshold and you're flat, BUY with all capital.
- If vscore > sell_threshold and you're long, SELL entire position.
- At most one trade every `cooldown_minutes`.

Produces a 3-panel plot:
1. Price
2. Vscore (+ thresholds)
3. Capital over time
"""
function backtest_vscore_strategy(ticker::String;
                                  buy_threshold::Float64 = -2.0,
                                  sell_threshold::Float64 = 2.0,
                                  initial_capital::Float64 = 1000.0,
                                  cooldown_minutes::Int = 30)

    # 1. Load price/time data
    df = get_historical_raw(ticker)  # your function; sorted by :date
    prices = Float64.(df.close)

    # Parse timestamps
    times = if eltype(df.date) <: AbstractString
        [DateTime(dt, dateformat"yyyy-mm-dd HH:MM:SS") for dt in df.date]
    else
        DateTime.(df.date)
    end

    # 2. Compute vscores (or load via your helper)
    vscores = get_historical_vscores(ticker)  # must align with prices
    n = min(length(prices), length(vscores))

    prices = prices[end-n+1:end]
    times  = times[end-n+1:end]
    vscores = vscores[end-n+1:end]

    # 3. Backtest loop
    capital = similar(prices, Float64)
    position = 0.0  # number of shares
    cash     = initial_capital
    last_trade_time = nothing :: Union{Nothing, DateTime}

    for i in 1:n
        price = prices[i]
        v     = vscores[i]

        # Trade only if vscore is not NaN
        if !isnan(v)
            can_trade = last_trade_time === nothing ||
                        (times[i] - last_trade_time) >= Minute(cooldown_minutes)

            if can_trade
                # SELL signal
                if v > sell_threshold && position > 0
                    cash += position * price
                    position = 0.0
                    last_trade_time = times[i]

                # BUY signal
                elseif v < buy_threshold && position == 0.0
                    position = cash / price
                    cash = 0.0
                    last_trade_time = times[i]
                end
            end
        end

        capital[i] = cash + position * price
    end

    # 4. Plots: price (top), vscore (middle), capital (bottom)
    p_price = plot(times, prices,
        xlabel = "Time",
        ylabel = "Price",
        title  = "$ticker Price",
        legend = false)

    p_vscore = plot(times, vscores,
        xlabel = "Time",
        ylabel = "VScore",
        title  = "VScore",
        legend = false)
    hline!(p_vscore, [buy_threshold], linestyle = :dash, label = "Buy thresh")
    hline!(p_vscore, [sell_threshold], linestyle = :dash, label = "Sell thresh")

    p_capital = plot(times, capital,
        xlabel = "Time",
        ylabel = "Capital (\$)",
        title  = "Portfolio Value",
        legend = false)

    plt = plot(p_price, p_vscore, p_capital,
               layout = @layout([a; b; c]),
               size = (900, 900))

    return plt, (times = times,
                 prices = prices,
                 vscores = vscores,
                 capital = capital)
end

# Convenience wrapper for ASST specifically
TICKER = "SPY"
backtest_asst() = backtest_vscore_strategy(TICKER)
plt, results = backtest_asst()
savefig(plt, "backtests/$(TICKER).png")