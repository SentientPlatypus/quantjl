using CSV
using DataFrames
using Statistics
using Dates
using Plots

include("../data.jl")

function backtest_vscore_strategy(ticker::String;
                                  buy_threshold::Float64 = -2.0,
                                  sell_threshold::Float64 = 1.72,
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

    prices  = prices[end-n+1:end]
    times   = times[end-n+1:end]
    vscores = vscores[end-n+1:end]

    # 3. Backtest loop (strategy)
    capital = similar(prices, Float64)
    position = 0.0  # number of shares
    cash     = initial_capital
    last_trade_time = nothing :: Union{Nothing, DateTime}

    # --- Benchmark: always fully invested from the start ---
    benchmark_shares   = initial_capital / prices[1]
    benchmark_capital  = similar(prices, Float64)

    for i in 1:n
        price = prices[i]
        v     = vscores[i]

        # Strategy logic
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

        # Benchmark value at this time
        benchmark_capital[i] = benchmark_shares * price
    end

    # 4. Plots: price (top), vscore (middle), capital vs benchmark (bottom)
    p_price = plot(times, prices,
        xlabel = "Time",
        ylabel = "Price",
        title  = "$ticker Price",
        legend = false)

    p_vscore = plot(times, vscores,
        xlabel = "Time",
        ylabel = "VScore",
        title  = "VScore",
        legend = true)
    hline!(p_vscore, [buy_threshold], linestyle = :dash, label = "Buy thresh")
    hline!(p_vscore, [sell_threshold], linestyle = :dash, label = "Sell thresh")

    p_capital = plot(times, capital,
        xlabel = "Time",
        ylabel = "Capital (\$)",
        title  = "Portfolio Value vs Benchmark",
        label  = "Strategy")
    plot!(p_capital, times, benchmark_capital,
        label = "Buy & Hold")

    plt = plot(p_price, p_vscore, p_capital,
               layout = @layout([a; b; c]),
               size = (900, 900))

    return plt, (times = times,
                 prices = prices,
                 vscores = vscores,
                 capital = capital,
                 benchmark_capital = benchmark_capital)
end

TICKER = "SPY"
backtest_asst() = backtest_vscore_strategy(TICKER)
plt, results = backtest_asst()
savefig(plt, "backtests/$(TICKER).png")
