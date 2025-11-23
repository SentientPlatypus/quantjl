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
    df = get_historical_raw(ticker)
    prices = Float64.(df.close)

    times = if eltype(df.date) <: AbstractString
        [DateTime(dt, dateformat"yyyy-mm-dd HH:MM:SS") for dt in df.date]
    else
        DateTime.(df.date)
    end

    # 2. Compute vscores
    vscores = get_historical_vscores(ticker)
    n = min(length(prices), length(vscores))

    prices  = prices[end-n+1:end]
    times   = times[end-n+1:end]
    vscores = vscores[end-n+1:end]

    # 3. Backtest
    capital = similar(prices, Float64)
    position = 0.0
    cash = initial_capital
    last_trade_time = nothing :: Union{Nothing, DateTime}

    benchmark_shares  = initial_capital / prices[1]
    benchmark_capital = similar(prices, Float64)

    # --- store trade events ---
    trade_times = DateTime[]
    trade_type  = Symbol[]    # :buy or :sell

    for i in 1:n
        price = prices[i]
        v     = vscores[i]

        if !isnan(v)
            can_trade = last_trade_time === nothing ||
                        (times[i] - last_trade_time) >= Minute(cooldown_minutes)

            if can_trade
                if v > sell_threshold && position > 0
                    # SELL
                    cash += position * price
                    position = 0.0
                    last_trade_time = times[i]

                    push!(trade_times, times[i])
                    push!(trade_type, :sell)

                elseif v < buy_threshold && position == 0.0
                    # BUY
                    position = cash / price
                    cash = 0.0
                    last_trade_time = times[i]

                    push!(trade_times, times[i])
                    push!(trade_type, :buy)
                end
            end
        end

        capital[i] = cash + position * price
        benchmark_capital[i] = benchmark_shares * price
    end

    # 4. Plots
    p_price = plot(times, prices, xlabel="Time", ylabel="Price",
                   title="$ticker Price", legend=false)

    p_vscore = plot(times, vscores, xlabel="Time", ylabel="VScore",
                    title="VScore", legend=true)
    hline!(p_vscore, [buy_threshold], linestyle=:dash, label="Buy thresh")
    hline!(p_vscore, [sell_threshold], linestyle=:dash, label="Sell thresh")

    p_capital = plot(times, capital, xlabel="Time", ylabel="Capital (\$)",
                     title="Portfolio Value vs Benchmark",
                     label="Strategy")
    plot!(p_capital, times, benchmark_capital, label="Buy & Hold")

    # --- add vertical trade lines to all plots ---
    for (t, typ) in zip(trade_times, trade_type)
        col = (typ == :buy) ? :green : :red
        plot!(p_price,  [t, t], [minimum(prices),  maximum(prices)],  color=col, lw=1, label=false)
        plot!(p_vscore, [t, t], [minimum(vscores), maximum(vscores)], color=col, lw=1, label=false)
        plot!(p_capital,[t, t], [minimum(capital), maximum(capital)], color=col, lw=1, label=false)
    end

    plt = plot(p_price, p_vscore, p_capital,
               layout = @layout([a; b; c]),
               size = (900, 900))

    return plt, (times=times, prices=prices, vscores=vscores,
                 capital=capital, benchmark_capital=benchmark_capital)
end

TICKER = "SUIG"
backtest_asst() = backtest_vscore_strategy(TICKER)
plt, results = backtest_asst()
savefig(plt, "backtests/$(TICKER).png")
